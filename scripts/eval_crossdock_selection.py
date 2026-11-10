"""Compare pose-selection scores on a per-pose PoseBusters table (CrossDock layout).

The table is ``crossdock/evaluate_posebusters.py`` output: one row per
(system, sample) with ``subset``, ``ligand_rmsd``, ``physical_valid`` and
``pb_success``. Each ``--score-csv`` adds a selector that keeps the
lowest-scoring pose per system (ties by sample index); every metric is then
read off that one pose. ``sample_0`` and the oracle (min RMSD) are always
reported. Selectors are paired per system, so a sign test on the
PB-success flips between two selectors is also printed.

    python scripts/eval_crossdock_selection.py \\
        --pb-csv crossdock/pb_results.csv \\
        --score-csv crossdock/vina_real_all.csv \\
        --score-csv crossdock/vinardo_all.csv \\
        --skipped-json crossdock/inference_all.skipped.json \\
        --out crossdock/selection_report.csv
"""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

KEYS = ['system', 'sample']
# score column -> selector label; ``vina_affinity`` is the torch reimplementation
SCORE_LABELS = {'vina_affinity': 'min Vina (torch)', 'vina_real': 'min Vina',
                'vinardo': 'min Vinardo'}
RMSD_THR = 2.0


def load_scores(path: Path) -> pd.DataFrame:
    """Normalise a score table to (system, sample:int, <score columns>)."""
    table = pd.read_csv(path)
    if table['sample'].dtype == object:
        table['sample'] = table['sample'].str.removeprefix('sample_').astype(int)
    cols = [c for c in SCORE_LABELS if c in table.columns]
    if not cols:
        raise SystemExit(f'{path}: no known score column among {list(table.columns)}')
    return table[KEYS + cols].drop_duplicates(KEYS)


def select(df: pd.DataFrame, col: str) -> pd.DataFrame:
    pool = df.dropna(subset=[col])
    return pool.sort_values([col, 'sample']).drop_duplicates('system').set_index('system')


def summarise(sel: pd.DataFrame, label: str, n_total: int) -> dict:
    """Success rates over *n_total* systems: systems with no selected pose count as failures."""
    return {'selector': label, 'systems': n_total, 'selected': len(sel),
            'physical_valid_%': 100 * sel['physical_valid'].sum() / n_total,
            'rmsd<2A_%': 100 * (sel['ligand_rmsd'] < RMSD_THR).sum() / n_total,
            'pb_success_%': 100 * sel['pb_success'].sum() / n_total,
            'median_rmsd': sel['ligand_rmsd'].median()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pb-csv', type=Path, required=True)
    ap.add_argument('--score-csv', type=Path, action='append', default=[])
    ap.add_argument('--skipped-json', type=Path,
                    help='config-generation skips; counted as failures per subset')
    ap.add_argument('--out', type=Path, required=True, help='summary CSV')
    args = ap.parse_args()

    # score columns the PB table already carries (evaluate_posebusters.py merges
    # its own Vina run in as ``vina_affinity``) join the selectors too
    header = pd.read_csv(args.pb_csv, nrows=0).columns
    pb = pd.read_csv(args.pb_csv, low_memory=False,
                     usecols=['system', 'subset', 'sample', 'ligand_rmsd', 'physical_valid', 'pb_success']
                     + [c for c in SCORE_LABELS if c in header])
    pb['physical_valid'] = pb['physical_valid'].fillna(False).astype(bool)
    pb['pb_success'] = pb['pb_success'].fillna(False).astype(bool)
    for path in args.score_csv:
        pb = pb.merge(load_scores(path), on=KEYS, how='left', validate='one_to_one')
    score_cols = [c for c in SCORE_LABELS if c in pb.columns]
    for c in score_cols:
        n_miss = int(pb[c].isna().sum())
        if n_miss:
            print(f'{n_miss} poses without a {c} score')

    skipped = {}
    if args.skipped_json:
        seen = set(pb['system'])
        for rec in json.loads(args.skipped_json.read_text()):
            if rec['system'] not in seen:
                parts = rec['system'].split('__')
                subset = parts[1] if parts[0] == 'Ensemble' and len(parts) > 2 else parts[0]
                skipped[subset] = skipped.get(subset, 0) + 1

    selectors = {'sample_0': pb[pb['sample'] == 0].set_index('system'),
                 'oracle (min RMSD)': select(pb, 'ligand_rmsd')}
    selectors.update({SCORE_LABELS[c]: select(pb, c) for c in score_cols})

    n_sys = pb['system'].nunique() + sum(skipped.values())
    rows = [dict(summarise(sel, label, n_sys), subset='all') for label, sel in selectors.items()]
    for subset, group in pb.groupby('subset', sort=True):
        n = group['system'].nunique() + skipped.get(subset, 0)
        for label, sel in selectors.items():
            rows.append(dict(summarise(sel[sel['subset'] == subset], label, n), subset=subset))
    summary = pd.DataFrame(rows)[['subset', 'selector', 'systems', 'selected', 'physical_valid_%',
                                  'rmsd<2A_%', 'pb_success_%', 'median_rmsd']]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    pd.set_option('display.width', 200, 'display.max_columns', None)
    print(summary.round(2).to_string(index=False))

    print('\npaired PB-success flips (systems where exactly one selector succeeds)')
    for a, b in itertools.combinations([SCORE_LABELS[c] for c in score_cols], 2):
        sa, sb = selectors[a]['pb_success'], selectors[b]['pb_success']
        common = sa.index.intersection(sb.index)
        sa, sb = sa.reindex(common), sb.reindex(common)
        same = int((selectors[a].loc[common, 'sample'] == selectors[b].loc[common, 'sample']).sum())
        wins_a, wins_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
        p = binomtest(wins_a, wins_a + wins_b, 0.5).pvalue if wins_a + wins_b else 1.0
        print(f'  {a} vs {b}: same pose {same}/{len(common)} | {a} only {wins_a}, '
              f'{b} only {wins_b}, net {wins_a - wins_b:+d} | sign-test p = {p:.3g}')


if __name__ == '__main__':
    main()
