"""Per-class (molecular glue / PROTAC) redock summary of a TPD benchmark run.

Reuses the harness of ``scripts/eval_posebusters_redock.py`` (cached per-pose
PoseBusters rows, receptor side-chain RMSD, score selectors) and splits every
selector by the ``class`` column of ``tpd/records.tsv``, so top-1, the
best-of-N oracles and the docking-score selectors are reported for glues,
PROTACs and all systems side by side. When records carry a ``covalent`` column
each class is also split into covalent / non-covalent ligands (``covalent``
column of the summary; ``all`` for the unsplit rows).

    python scripts/summarize_tpd.py --gt-dir tpd/raw --run tpd=tpd/prepared \\
        --score-csv tpd/vinardo.csv --records tpd/records.tsv \\
        --out tpd/tpd_summary.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd

import eval_posebusters_redock as evaluator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt-dir', type=Path, default=os.environ.get('EVAL_GT_DIR'),
                    help='benchmark root with <id>/<id>_ligand.sdf (default: $EVAL_GT_DIR)')
    ap.add_argument('--run', action='append', required=True, help='label=path, repeatable')
    ap.add_argument('--score-csv', type=Path, action='append', default=[],
                    help='scripts/vina_rescore.py or scripts/smina_rescore.py table; repeatable')
    ap.add_argument('--records', type=Path, default=Path('tpd/records.tsv'),
                    help='scripts/prepare_tpd.py metadata with the class column')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=Path('tpd/tpd_summary.csv'))
    args = ap.parse_args()
    if args.gt_dir is None:
        ap.error('--gt-dir (or $EVAL_GT_DIR) is required')
    evaluator._GT_INDEX.update({p.name: p for p in args.gt_dir.iterdir() if p.is_dir()})

    records = pd.read_csv(args.records, sep='\t')
    classes = records.set_index('record')['class']
    covalent = records.set_index('record')['covalent'].map({True: 'yes', False: 'no'}) if 'covalent' in records else None
    scores = evaluator.load_scores(args.score_csv)

    summaries = []
    for spec in args.run:
        label, path = spec.split('=', 1)
        root = Path(path)
        df_pb, cols = evaluator.run_posebusters(root, args.workers)
        if 'pb_error' in df_pb:
            df_pb = df_pb[df_pb['pb_error'].isna()]
        df = df_pb.merge(evaluator.sidechain_rmsd(root), on=['system', 'sample'], how='inner')
        df = df.assign(run=label)
        if scores is not None:
            df = df.merge(scores, on=evaluator.POSE_KEYS, how='left')
        df['class'] = df['system'].map(classes)
        df['covalent'] = df['system'].map(covalent) if covalent is not None else 'all'
        n_missing = int(df['class'].isna().sum())
        if n_missing:
            print(f'  {label}: {n_missing} pose(s) from systems absent in {args.records}')

        for cls, group in [('all', df), *df.groupby('class', sort=True)]:
            summaries.append(evaluator.summarise(group, cols, label).assign(**{'class': cls, 'covalent': 'all'}))
            if covalent is None:
                continue
            for flag, subset in group.groupby('covalent', sort=True):
                summaries.append(evaluator.summarise(subset, cols, label).assign(**{'class': cls, 'covalent': flag}))

    summary = pd.concat(summaries, ignore_index=True)
    keys = ['run', 'class', 'covalent', 'selector']
    summary = summary[keys + [c for c in summary.columns if c not in keys]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    pd.set_option('display.width', 220, 'display.max_columns', None)
    print(summary.round(2).to_string(index=False))
    print(f'\nsummary -> {args.out}')


if __name__ == '__main__':
    main()
