"""PoseBusters validity for the DiffBindFR CrossDock poses.

DiffBindFR's own eval reports L-RMSD only, so its CrossDock column is not
PB-gated while EquiFlex's joint column is. This busts DiffBindFR's selected
poses with the same harness EquiFlex's table came from -- PoseBusters 'redock',
``pb_valid`` as the conjunction of every non-RMSD check -- so the two methods
can be read on one definition, and so the result can be checked against the
PB-success rates the DiffBindFR paper reports.

Each pose is busted against *its own* predicted receptor (``prot_final.pdb``),
never the crystal one; see the rescorer receptor-frame confound.

    python scripts/pb_diffbindfr_crossdock.py --selector mdn_score --workers 11
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from crossdock_diffbindfr_summary import NEW_ROOT, OLD_ROOT, SEQSIM, subset_of_run

_BUSTER = None


def _init_worker():
    """One buster per process, RDKit held to a single thread."""
    global _BUSTER
    from posebusters import PoseBusters
    try:
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.*')
    except ImportError:
        pass
    _BUSTER = PoseBusters('redock')


def _bust(key, pred_lig, prot_pdb, gt_lig):
    out = _BUSTER.bust(mol_pred=pred_lig, mol_true=gt_lig, mol_cond=prot_pdb)
    row = {c: v for c, v in out.iloc[0].items() if c not in ('file', 'molecule', 'position')}
    row['key'] = key
    return row


def load_selected(selector):
    """-> one row per subset|system: the pose this selector keeps, with its file paths."""
    frames = []
    for root in (OLD_ROOT, NEW_ROOT):
        for csv in sorted(root.glob('*/results/results_ec.csv')):
            df = pd.read_csv(csv, usecols=['complex_name', 'l-rmsd', 'sample_id', 'docked_lig',
                                           'protein_pdb', 'crystal_ligand', selector])
            df['run'] = csv.parent.parent.name
            df['subset'] = subset_of_run(df.run.iloc[0])
            frames.append(df)
    poses = pd.concat(frames, ignore_index=True)
    poses['key'] = poses.subset + '|' + poses.complex_name
    # The smoke slice overlaps chunk 0; keep the first run that produced a system.
    poses = poses.sort_values(['key', 'run'])
    poses = poses[poses.run == poses.key.map(poses.groupby('key').run.first())]
    return poses.loc[poses.groupby('key')[selector].idxmax()].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selector', default='mdn_score', choices=['mdn_score', 'smina_score'],
                    help='mdn_score takes the max (DiffBindFR-MDN); smina_score is negated first')
    ap.add_argument('--workers', type=int, default=11)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', type=Path,
                    default=Path('crossdock/results_diffbindfr/pb_mdn_top1.csv'))
    args = ap.parse_args()

    sel = load_selected(args.selector)
    if args.limit:
        sel = sel.head(args.limit)
    print(f'{len(sel)} selected poses over {sel.subset.nunique()} subsets')

    missing = [c for c in ('docked_lig', 'protein_pdb', 'crystal_ligand')
               if not sel[c].map(lambda p: Path(p).exists()).all()]
    if missing:
        n = {c: int((~sel[c].map(lambda p: Path(p).exists())).sum()) for c in missing}
        print(f'WARNING missing files: {n}')

    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        futures = {ex.submit(_bust, r.key, r.docked_lig, r.protein_pdb, r.crystal_ligand): r.key
                   for r in sel.itertuples()}
        for f in tqdm(as_completed(futures), total=len(futures), smoothing=0.05):
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({'key': futures[f], 'pb_error': repr(e)})

    pb = pd.DataFrame(rows)
    checks = [c for c in pb.columns
              if c not in ('key', 'pb_error') and not c.startswith('rmsd')
              and pb[c].dropna().isin([True, False]).all() and pb[c].notna().any()]
    pb['pb_valid'] = pb[checks].fillna(False).astype(bool).all(axis=1)
    print(f'{len(checks)} validity checks: {checks}')

    out = sel[['key', 'subset', 'complex_name', 'l-rmsd', 'sample_id']].merge(pb, on='key')
    out['dbfr_lig_success'] = out['l-rmsd'] < 2
    out['dbfr_pb_success'] = out.dbfr_lig_success & out.pb_valid
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    seq = pd.read_csv(SEQSIM, sep='\t')
    seq['key'] = seq.subset + '|' + seq.system
    out = out.merge(seq[['key', 'identity_aln']], on='key', how='left')

    def table(df, by, observed=True):
        g = df.groupby(by, observed=observed)
        return pd.DataFrame({
            'n': g.size(),
            'pb_valid_%': 100 * g.pb_valid.mean(),
            'lig<2A_%': 100 * g.dbfr_lig_success.mean(),
            'pb_success_%': 100 * g.dbfr_pb_success.mean(),
        }).round(2)

    print(f'\nDiffBindFR-{args.selector.split("_")[0].upper()} top-1, by subset')
    print(table(out, 'subset').to_string())
    out['identity_band'] = pd.cut(out.identity_aln, [-.01, .3, .7, .9, 1.01],
                                  labels=['<30%', '30-70%', '70-90%', '>=90%'])
    print('\nby receptor identity to the v10 training split')
    print(table(out, 'identity_band').to_string())
    if 'pb_error' in out:
        n_err = int(out.pb_error.notna().sum())
        if n_err:
            print(f'\n{n_err} poses errored; counted as invalid')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
