"""Rescore generated poses with the real AutoDock Vina binary (``--score_only``).

The affinities in each system's ``summary.md`` come from
``equiflex.inference.vina_scoring``, a torch reimplementation whose hydrogen
placement and atom typing differ from the binary's, so Vina-based pose
selection (the "min Vina" selector in ``scripts/eval_posebusters_redock.py``)
rescores every pose with the binary rather than reusing them.

The receptor is rebuilt from each pose's own ``complex.cif`` (protein residues
only), so the sampled side chains are what gets scored.

    python scripts/vina_rescore.py --run phys=data/posebusters/prepared \\
                                   --out data/posebusters/vina_real.csv

``vina`` and ``obabel`` are taken from ``$PATH`` unless ``--vina`` / ``--obabel``
(or ``$VINA_BIN`` / ``$OBABEL_BIN``) point elsewhere. Re-running with the same
``--out`` scores only poses not already in the CSV.
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from equiflex.inference.vina_scoring import vina_binary_score_pose


def score_pose(job):
    run, system, sample, complex_path, ligand_path, vina, obabel = job
    scored = vina_binary_score_pose(complex_path, ligand_path, vina=vina, obabel=obabel)
    return {'run': run, 'system': system, 'sample': sample,
            'vina_real': scored['affinity'], 'error': scored['error']}


def collect_jobs(runs, vina, obabel):
    jobs = []
    for run, root in runs.items():
        for sys_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            for sample_dir in sorted(sys_dir.glob('sample_*')):
                complex_path, ligand_path = sample_dir / 'complex.cif', sample_dir / 'ligand.sdf'
                if complex_path.exists() and ligand_path.exists():
                    jobs.append((run, sys_dir.name, sample_dir.name,
                                 complex_path, ligand_path, vina, obabel))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True,
                    metavar='LABEL=ROOT', help='run_inference output_root to score; repeatable')
    ap.add_argument('--out', type=Path, required=True, help='CSV with one row per pose')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--vina', default=os.environ.get('VINA_BIN', 'vina'),
                    help='AutoDock Vina executable (default: $VINA_BIN or "vina" on PATH)')
    ap.add_argument('--obabel', default=os.environ.get('OBABEL_BIN', 'obabel'),
                    help='Open Babel executable (default: $OBABEL_BIN or "obabel" on PATH)')
    args = ap.parse_args()

    for name, exe in (('vina', args.vina), ('obabel', args.obabel)):
        if shutil.which(exe) is None:
            ap.error(f'{name} executable not found: {exe}')

    runs = dict(r.split('=', 1) for r in args.run)
    jobs = collect_jobs(runs, args.vina, args.obabel)

    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(map(tuple, prev[['run', 'system', 'sample']].values))
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f'{len(done)} poses cached in {args.out}')
    print(f'scoring {len(jobs)} poses with {args.vina} on {args.workers} workers')

    rows = []
    if jobs:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(score_pose, j) for j in jobs]
            for f in tqdm(as_completed(futures), total=len(futures)):
                rows.append(f.result())

    df = pd.DataFrame(rows, columns=['run', 'system', 'sample', 'vina_real', 'error'])
    if args.out.exists():
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    bad = df[df.vina_real.isna()]
    print(f'wrote {args.out}: {len(df)} poses, {len(bad)} failed')
    if len(bad):
        print(bad.groupby('error').size().sort_values(ascending=False).head())


if __name__ == '__main__':
    main()
