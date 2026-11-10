"""PoseBusters redock metrics for one or more run_inference.py output roots.

Every pose is busted against each deposited copy of the ground-truth ligand
(``<gt-dir>/<id>/<id>_ligand(s).sdf``) with the pose's own protein as the
conditioning receptor, keeping the lowest-RMSD copy. Receptor side-chain RMSD
is read from ``results.pt`` against the featurised input coordinates.

Selectors pick ONE pose per system and every metric is read off that same pose:
``top-1 (sample_0)``, ``best-of-N`` oracles, and — with ``--score-csv`` from
``scripts/vina_rescore.py`` or ``scripts/smina_rescore.py`` — ``min Vina`` /
``min Vinardo``, the rankings used for reported numbers.

    python scripts/eval_posebusters_redock.py \\
        --gt-dir /path/to/posebusters_benchmark_set \\
        --run phys=data/posebusters/prepared \\
        --score-csv data/posebusters/vina_real.csv \\
        --out data/posebusters/pb_metrics.csv

Per-pose PoseBusters rows are cached in ``<root>/posebusters_redock.csv`` and
recomputed when any pose file changes.
"""

import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from posebusters.modules import energy_ratio
from equiflex.inference.vina_scoring import extract_protein_pdb
from equiflex.prep.const import STD_AA_ATOMS

RDLogger.DisableLog('rdApp.*')

_BUSTER = None
_GT_INDEX = {}
ENERGY_CACHE = Path(__file__).resolve().parents[1] / 'data/posebusters/energy_cache'


def _persist_ensemble_energies():
    """Share PoseBusters' conformer-ensemble energies across poses, workers and runs.

    ``energy_ratio`` embeds and UFF-minimises an ensemble of the ligand's conformers
    for every pose, which is most of the evaluation time (tens of seconds for
    flexible ligands). The ensemble depends only on the InChI and the conformer
    count (ETKDG seed is fixed at 42), and all samples of a system, and every
    checkpoint evaluated on the benchmark, share their ligands, so one JSON file per
    (InChI, count) is computed once and reused.
    """
    compute = energy_ratio.get_energies.__wrapped__
    ENERGY_CACHE.mkdir(parents=True, exist_ok=True)

    def get_energies(inchi, n_confs=50, num_threads=0):
        path = ENERGY_CACHE / (hashlib.sha1(f'{inchi}|{n_confs}'.encode()).hexdigest() + '.json')
        if path.exists():
            return json.loads(path.read_text())
        energies = compute(inchi, n_confs, num_threads)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(energies))
        tmp.replace(path)
        return energies

    energy_ratio.get_energies = functools.cache(get_energies)


_persist_ensemble_energies()

POSE_KEYS = ['run', 'system', 'sample']
# Chemically equivalent side-chain atom pairs; each group is swapped as a unit
# (the two ring carbons of a pair flip together), the lower-error assignment wins.
SC_SWAPS = {'ARG': [('NH1', 'NH2')], 'ASP': [('OD1', 'OD2')], 'GLU': [('OE1', 'OE2')],
            'PHE': [('CD1', 'CD2'), ('CE1', 'CE2')], 'TYR': [('CD1', 'CD2'), ('CE1', 'CE2')]}
# score column -> selector label; the columns scripts/vina_rescore.py (vina_real)
# and scripts/smina_rescore.py (vinardo) write
SCORE_SELECTORS = {'vina_real': 'min Vina', 'vinardo': 'min Vinardo',
                   'genscore': 'max GenScore'}
# GenScore is a summed pair likelihood, so its best pose is the LARGEST value;
# Vina and Vinardo are affinities and keep the smallest. A GenScore column must
# come from a predicted-receptor rescore (scripts/score_pb_predicted_receptor.py
# or equiflex.inference.genscore_scoring): scoring it against the crystal
# receptor while Vina/Vinardo see the sampled one is not a fair comparison.
MAXIMISED_SCORES = frozenset({'genscore'})


def load_scores(paths):
    """Outer-join score tables on POSE_KEYS, keeping the SCORE_SELECTORS columns; None if no paths."""
    scores = None
    for path in paths:
        table = pd.read_csv(path)
        cols = [c for c in SCORE_SELECTORS if c in table.columns]
        if not cols:
            raise SystemExit(f'{path}: no score column among {list(SCORE_SELECTORS)}')
        table = table[POSE_KEYS + cols].drop_duplicates(POSE_KEYS)
        scores = table if scores is None else scores.merge(table, on=POSE_KEYS, how='outer')
    return scores


def _buster():
    global _BUSTER
    if _BUSTER is None:
        _BUSTER = PoseBusters('redock')
    return _BUSTER


def _single_thread_buster():
    """Pool initializer: build the buster, with RDKit's UFF ensemble on one thread.

    Each worker is already one of *workers* processes on this machine, so leaving
    energy_ratio's default thread count oversubscribes the cores. Per pose a single
    thread is 2-3x slower, but aggregate throughput over 10 workers measured ~1.7x
    better (3.3 vs 5.7 s/pose on a size-spanning sample of 44 TPD poses, 12 cores).
    Only this parameter is touched; every scientific check and threshold keeps its
    redock default.
    """
    for module in _buster().config['modules']:
        if module['function'] == 'energy_ratio':
            module.setdefault('parameters', {})['num_threads'] = 1


def _gt_ligand_path(system_id):
    gt_dir = _GT_INDEX.get(system_id)
    if gt_dir is None:
        return None
    for pattern in ('*_ligands.sdf', '*_ligand.sdf'):
        hits = sorted(p for p in gt_dir.glob(pattern) if 'start_conf' not in p.name)
        if hits:
            return hits[0]
    return None


def _pb_one(task):
    """One (system, sample) -> best-RMSD redock row across deposited GT copies."""
    system_id, sample_name, pred_lig, pred_cif, gt_sdf = task
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cond = tmp / 'protein.pdb'
                extract_protein_pdb(pred_cif, cond, setup_entities=True)
                copies = [m for m in Chem.SDMolSupplier(str(gt_sdf)) if m is not None]
                if not copies:
                    return None
                best = None
                for i, m in enumerate(copies):
                    p = tmp / f'gt_{i}.sdf'
                    w = Chem.SDWriter(str(p)); w.write(m); w.close()
                    df_one = _buster().bust(mol_pred=str(pred_lig), mol_true=str(p),
                                            mol_cond=str(cond), full_report=True)
                    if len(df_one) == 0:
                        continue
                    row = df_one.iloc[0].to_dict()
                    if best is None or (pd.notna(row.get('rmsd')) and
                                        (pd.isna(best.get('rmsd')) or row['rmsd'] < best['rmsd'])):
                        best = row
                if best is None:
                    return None
                best.update(system=system_id, sample=sample_name, n_gt_copies=len(copies))
                return best
        except Exception as exc:
            return {'system': system_id, 'sample': sample_name, 'pb_error': repr(exc)[:200]}


def _pb_system(tasks):
    """All samples of one system in one worker, so the ligand's ensemble is built once."""
    return [_pb_one(t) for t in tasks]


def build_pb_tasks(pred_root: Path):
    tasks = []
    for sys_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        gt_sdf = _gt_ligand_path(sys_dir.name)
        if gt_sdf is None:
            continue
        for sample_dir in sorted(sys_dir.glob('sample_*')):
            lig, cif = sample_dir / 'ligand.sdf', sample_dir / 'complex.cif'
            if lig.exists() and cif.exists():
                tasks.append((sys_dir.name, sample_dir.name, lig, cif, gt_sdf))
    return tasks


def _pose_fingerprint(tasks):
    """Cache key: predicted pose files plus the ground-truth ligand files they were busted against.

    ``--gt-dir`` is a per-invocation choice, so the GT identity must be part of the key or a
    run evaluated against one benchmark copy would be reported under another. Lists, not
    tuples, so the JSON round-trip through the cache file compares equal.
    """
    mt = [max(lig.stat().st_mtime, cif.stat().st_mtime) for _, _, lig, cif, _ in tasks]
    gt = [[str(g), g.stat().st_mtime, g.stat().st_size] for g in sorted({g for *_, g in tasks})]
    return {'n_poses': len(tasks), 'max_mtime': max(mt) if mt else 0.0, 'gt': gt}


def canonical_check_columns(task):
    """Checks PoseBusters itself reports (full_report adds vacuously-False diagnostics)."""
    _, _, pred_lig, pred_cif, gt_sdf = task
    with tempfile.TemporaryDirectory() as tmp:
        cond = Path(tmp) / 'protein.pdb'
        extract_protein_pdb(pred_cif, cond, setup_entities=True)
        out = _buster().bust(mol_pred=str(pred_lig), mol_true=str(gt_sdf), mol_cond=str(cond))
    return [c for c in out.columns if c not in ('file', 'molecule', 'position')]


def run_posebusters(pred_root: Path, workers: int, use_cache=True):
    cache = pred_root / 'posebusters_redock.csv'
    cache_meta = pred_root / 'posebusters_redock_checks.json'
    tasks = build_pb_tasks(pred_root)
    if not tasks:
        raise RuntimeError(f'no poses with a ground-truth ligand found under {pred_root}')
    fp = _pose_fingerprint(tasks)

    if use_cache and cache.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if meta.get('fingerprint') == fp:
            print(f'  {pred_root.name}: cached, reusing')
            return pd.read_csv(cache), meta['columns']
        print(f'  {pred_root.name}: cache stale — recomputing')

    cols = canonical_check_columns(tasks[0])
    print(f'  {pred_root.name}: {len(tasks)} poses, {len(cols)} checks, {workers} workers')
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'),
                             initializer=_single_thread_buster) as ex:
        by_system = {}
        for t in tasks:
            by_system.setdefault(t[0], []).append(t)
        futures = [ex.submit(_pb_system, group) for group in by_system.values()]
        with tqdm(total=len(tasks), desc=pred_root.name) as bar:
            for fut in as_completed(futures):
                batch = fut.result()
                rows.extend(r for r in batch if r is not None)
                bar.update(len(batch))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    cache_meta.write_text(json.dumps({'columns': cols, 'fingerprint': fp}))
    return out, cols


def _swap_groups(pf):
    """Per symmetric residue: (atom-offset array, swapped atom permutation) from residue_keys order."""
    groups, start = [], 0
    for key, n in zip(pf['residue_keys'], pf['n_atoms_per_residue']):
        names = STD_AA_ATOMS[key[1]]
        assert len(names) == n, f'{key}: {n} atoms vs {len(names)} in STD_AA_ATOMS'
        pairs = SC_SWAPS.get(key[1], [])
        if pairs:
            perm = np.arange(n)
            for a, b in pairs:
                perm[names.index(a)], perm[names.index(b)] = names.index(b), names.index(a)
            groups.append((start + np.arange(n), start + perm))
        start += n
    return groups


def sidechain_rmsd(pred_root: Path) -> pd.DataFrame:
    """Per-sample receptor RMSD split by backbone_mask, over input-present atoms.

    ``rmsd_sc`` keeps the input atom correspondence; ``rmsd_sc_sym`` lets each
    residue take the lower-squared-error assignment of its equivalent atoms (SC_SWAPS).
    """
    rows = []
    for system_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        pf_path = system_dir / 'system_data' / 'pocket_features.pt'
        res_path = system_dir / 'results.pt'
        if not (pf_path.exists() and res_path.exists()):
            continue
        pf = torch.load(pf_path, weights_only=False)
        res = torch.load(res_path, weights_only=False, map_location='cpu')

        gt = pf['coords']
        pred = res['rec_coords_final'] + res['backbone_com']
        bb = pf['backbone_mask'].bool()
        valid = ~pf['missing_mask'].bool()

        def _rmsd(mask):
            m = mask & valid
            if m.sum() == 0:
                return np.full(pred.shape[0], np.nan)
            diff = pred[:, m, :] - gt[m, :]
            return diff.pow(2).sum(-1).mean(-1).sqrt().numpy()

        r_all = _rmsd(torch.ones_like(bb))
        r_bb = _rmsd(bb)
        r_sc = _rmsd(~bb)

        m_sc = ((~bb) & valid).numpy()
        sq = (pred - gt).pow(2).sum(-1).numpy()
        for idx, perm in _swap_groups(pf):
            # swap only when both atoms of every exchanged pair are observed
            keep = m_sc[idx] & m_sc[perm]
            sq_sw = (pred[:, idx, :] - gt[perm, :]).pow(2).sum(-1).numpy()
            better = ((sq_sw * keep).sum(-1) < (sq[:, idx] * keep).sum(-1))[:, None] & keep[None, :]
            sq[:, idx] = np.where(better, sq_sw, sq[:, idx])
        r_sym = (np.sqrt((sq * m_sc).sum(-1) / m_sc.sum()) if m_sc.any()
                 else np.full(pred.shape[0], np.nan))
        for b in range(pred.shape[0]):
            rows.append({'system': system_dir.name, 'sample': f'sample_{b}',
                         'rmsd_all': float(r_all[b]), 'rmsd_bb': float(r_bb[b]),
                         'rmsd_sc': float(r_sc[b]), 'rmsd_sc_sym': float(r_sym[b])})
    # Named columns even when empty: a run from another docking method has no
    # featurisation to read, and a bare empty frame breaks the caller's merge.
    return pd.DataFrame(rows, columns=['system', 'sample', 'rmsd_all', 'rmsd_bb', 'rmsd_sc', 'rmsd_sc_sym'])


LIG_THR, SC_THR = 2.0, 1.0


def summarise(df: pd.DataFrame, check_cols, label: str) -> pd.DataFrame:
    df = df.copy()
    # PoseBusters' 'redock' config carries its own rmsd_<=_2A check. Leaving it in
    # the validity conjunction makes pb_valid == success by construction, so the
    # RMSD criterion is dropped here and applied separately.
    present = [c for c in check_cols if c in df.columns and not c.startswith('rmsd_')]
    df['pb_valid'] = df[present].fillna(False).astype(bool).all(axis=1)
    df['lig_ok'] = df['rmsd'] < LIG_THR
    df['sc_ok'] = df['rmsd_sc'] < SC_THR
    df['sc_sym_ok'] = df['rmsd_sc_sym'] < SC_THR

    selectors = {
        'top-1 (sample_0)': df[df['sample'] == 'sample_0'],
        'best-of-N (lig)': df.loc[df.groupby('system')['rmsd'].idxmin()],
        'best-of-N (sc)': df.loc[df.groupby('system')['rmsd_sc'].idxmin()],
    }
    for col, name in SCORE_SELECTORS.items():
        if col not in df.columns:
            continue
        # Poses whose receptor or ligand failed scoring have no score and cannot
        # be ranked; dropping them keeps idxmin from raising on an all-NaN system.
        pool = df.dropna(subset=[col])
        grouped = pool.groupby('system')[col]
        idx = grouped.idxmax() if col in MAXIMISED_SCORES else grouped.idxmin()
        selectors[name] = pool.loc[idx]
    out = []
    for name, sel in selectors.items():
        out.append({
            'run': label, 'selector': name, 'n_sys': sel['system'].nunique(),
            'lig_rmsd_med': sel['rmsd'].median(),
            'lig_rmsd_mean': sel['rmsd'].mean(),
            'lig<2A_%': 100 * sel['lig_ok'].mean(),
            'pb_valid_%': 100 * sel['pb_valid'].mean(),
            'success_%': 100 * (sel['lig_ok'] & sel['pb_valid']).mean(),
            'sc_rmsd_med': sel['rmsd_sc'].median(),
            'sc_rmsd_mean': sel['rmsd_sc'].mean(),
            'sc<1A_%': 100 * sel['sc_ok'].mean(),
            'sc_success_%': 100 * (sel['sc_ok'] & sel['pb_valid']).mean(),
            'scsym_rmsd_med': sel['rmsd_sc_sym'].median(),
            'scsym_rmsd_mean': sel['rmsd_sc_sym'].mean(),
            'scsym<1A_%': 100 * sel['sc_sym_ok'].mean(),
            'scsym_success_%': 100 * (sel['sc_sym_ok'] & sel['pb_valid']).mean(),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt-dir', type=Path, default=os.environ.get('EVAL_GT_DIR'),
                    help='benchmark root with <id>/<id>_ligand(s).sdf (default: $EVAL_GT_DIR)')
    ap.add_argument('--run', action='append', required=True,
                    help='label=path, repeatable')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('pb_metrics.csv'))
    ap.add_argument('--score-csv', '--vina-csv', dest='score_csv', type=Path,
                    action='append', default=[],
                    help='scripts/vina_rescore.py / scripts/smina_rescore.py output; adds the '
                         'min Vina / min Vinardo selector; repeatable')
    args = ap.parse_args()
    if args.gt_dir is None:
        ap.error('--gt-dir (or $EVAL_GT_DIR) is required')
    _GT_INDEX.update({p.name: p for p in Path(args.gt_dir).iterdir() if p.is_dir()})

    scores = load_scores(args.score_csv)

    summaries, per_pose = [], []
    for spec in args.run:
        label, path = spec.split('=', 1)
        root = Path(path)
        df_pb, cols = run_posebusters(root, args.workers, use_cache=not args.no_cache)
        if 'pb_error' in df_pb:
            n_err = df_pb['pb_error'].notna().sum()
            if n_err:
                print(f'  {label}: {n_err} pose(s) errored in PoseBusters')
            df_pb = df_pb[df_pb['pb_error'].isna()]
        df_sc = sidechain_rmsd(root)
        df = df_pb.merge(df_sc, on=['system', 'sample'], how='inner')
        df = df.assign(run=label)
        if scores is not None:
            df = df.merge(scores, on=POSE_KEYS, how='left')
            for col in SCORE_SELECTORS:
                n_miss = int(df[col].isna().sum()) if col in df.columns else 0
                if n_miss:
                    print(f'  {label}: {n_miss} pose(s) without a {col} score')
        print(f'  {label}: {len(df)} poses / {df["system"].nunique()} systems after merge')
        per_pose.append(df)
        summaries.append(summarise(df, cols, label))

    summary = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(per_pose, ignore_index=True).to_csv(args.out, index=False)
    pd.set_option('display.width', 200, 'display.max_columns', None)
    print('\n' + summary.round(3).to_string(index=False))
    print(f'\nper-pose table -> {args.out}')


if __name__ == '__main__':
    main()
