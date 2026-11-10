"""PoseBusters redock success of the single pose a docking score selects per system.

For each run the lowest-scoring pose per system (ties by sample index) is taken
from a ``scripts/vina_rescore.py`` table and busted fresh — no per-root cache —
with the harness of ``scripts/eval_posebusters_redock.py``. The denominator is
every system directory under the run root that has a ground-truth ligand, so a
system with no scored pose, a PoseBusters error or a missing RMSD counts as a
failure rather than dropping out.

    python scripts/eval_posebusters_selected.py \\
        --gt-dir /path/to/posebusters_benchmark_set \\
        --run phys=data/posebusters/prepared \\
        --scores data/posebusters/vinardo.csv --score-col vinardo \\
        --out-prefix data/posebusters/vinardo_selected

Writes ``<prefix>_poses.csv``, ``<prefix>_summary.csv`` and ``<prefix>_metadata.json``.
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import eval_posebusters_redock as evaluator

METRICS = ['pb_valid', 'lig_success', 'lig_success_pb', 'sc_success',
           'sc_success_pb', 'joint_success', 'joint_success_pb']


# Limits RDKit's ensemble optimization threads inside each worker. All scientific
# checks and thresholds retain the installed redock defaults.
initialize_buster = evaluator._single_thread_buster


def select_poses(scores: pd.DataFrame, run: str, score_col: str, systems) -> pd.DataFrame:
    """Lowest-*score_col* pose per system, reindexed to *systems* (unscored systems stay NaN)."""
    pool = scores[(scores.run == run) & np.isfinite(scores[score_col])].copy()
    pool['sample_number'] = pool['sample'].str.removeprefix('sample_').astype(int)
    selected = (pool.sort_values([score_col, 'sample_number'])
                .drop_duplicates('system').set_index('system')
                .reindex(systems).reset_index())
    selected['run'] = run
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt-dir', type=Path, default=os.environ.get('EVAL_GT_DIR'),
                    help='benchmark root with <id>/<id>_ligand(s).sdf (default: $EVAL_GT_DIR)')
    ap.add_argument('--run', action='append', required=True,
                    metavar='LABEL=ROOT', help='run_inference output_root; repeatable')
    ap.add_argument('--scores', type=Path, required=True,
                    help='scripts/vina_rescore.py output covering every --run label')
    ap.add_argument('--score-col', default='vina_real', choices=sorted(evaluator.SCORE_SELECTORS))
    ap.add_argument('--out-prefix', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if args.gt_dir is None:
        ap.error('--gt-dir (or $EVAL_GT_DIR) is required')
    evaluator._GT_INDEX.update({p.name: p for p in Path(args.gt_dir).iterdir() if p.is_dir()})

    scores = pd.read_csv(args.scores)
    assert not scores.duplicated(evaluator.POSE_KEYS).any()
    initialize_buster()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows, summaries = [], []
    check_columns = None
    for spec in args.run:
        run, path = spec.split('=', 1)
        root = Path(path)
        systems = sorted(p.name for p in root.iterdir()
                         if p.is_dir() and evaluator._gt_ligand_path(p.name) is not None)
        selected = select_poses(scores, run, args.score_col, systems)
        task_map = {(t[0], t[1]): t for t in evaluator.build_pb_tasks(root)}
        tasks = [task_map[(r.system, r.sample)] for r in selected.itertuples()
                 if (r.system, r.sample) in task_map]
        if check_columns is None:
            check_columns = evaluator.canonical_check_columns(tasks[0])
        rows = []
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('fork'),
                                 initializer=initialize_buster) as executor:
            futures = {executor.submit(evaluator._pb_one, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc=run):
                task = futures[future]
                result = future.result()
                rows.append(result if result is not None else
                            {'system': task[0], 'sample': task[1],
                             'pb_error': 'PoseBusters returned no result'})
        frame = selected.merge(pd.DataFrame(rows), on=['system', 'sample'],
                               how='left', validate='one_to_one')
        frame = frame.merge(evaluator.sidechain_rmsd(root),
                            on=['system', 'sample'], how='left', validate='one_to_one')
        checks = [c for c in check_columns if not c.startswith('rmsd_')]
        frame['pb_valid'] = frame.reindex(columns=checks).fillna(False).astype(bool).all(axis=1)
        frame['lig_success'] = frame['rmsd'] < evaluator.LIG_THR
        frame['sc_success'] = frame['rmsd_sc'] < evaluator.SC_THR
        frame['lig_success_pb'] = frame.lig_success & frame.pb_valid
        frame['sc_success_pb'] = frame.sc_success & frame.pb_valid
        frame['joint_success'] = frame.lig_success & frame.sc_success
        frame['joint_success_pb'] = frame.joint_success & frame.pb_valid
        summary = {'run': run, 'selector': args.score_col, 'systems': len(systems),
                   'scored_systems': int(frame[args.score_col].notna().sum()),
                   'pb_evaluated': len(rows),
                   'pb_errors': int(frame.get('pb_error', pd.Series(dtype=object)).notna().sum()),
                   'missing_lig_rmsd': int(frame.rmsd.isna().sum()),
                   'missing_sc_rmsd': int(frame.rmsd_sc.isna().sum())}
        for metric in METRICS:
            summary[metric + '_count'] = int(frame[metric].sum())
            summary[metric + '_pct'] = 100 * frame[metric].mean()
        summaries.append(summary)
        all_rows.append(frame)
        print(json.dumps(summary), flush=True)
        pd.concat(all_rows, ignore_index=True).to_csv(f'{args.out_prefix}_poses.csv', index=False)
        pd.DataFrame(summaries).to_csv(f'{args.out_prefix}_summary.csv', index=False)

    metadata = {'posebusters_version': version('posebusters'),
                'selection': f'minimum {args.score_col} per system; ties by sample index',
                'scores': str(args.scores), 'gt_dir': str(args.gt_dir),
                'runs': dict(spec.split('=', 1) for spec in args.run),
                'checks': check_columns,
                'validity_excludes': [c for c in check_columns if c.startswith('rmsd_')],
                'posebusters_config': evaluator._buster().config,
                'sidechain_rmsd': 'GT-present non-backbone pocket atoms, original correspondence, '
                                  'no alignment or symmetry correction',
                'denominator': 'every system directory with a ground-truth ligand; '
                               'missing values fail success'}
    Path(f'{args.out_prefix}_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
