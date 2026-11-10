"""Rescore every pose under run_inference.py output roots with smina (Vinardo by default).

Companion to ``scripts/vina_rescore.py``: the receptor is the protein part of each
pose's own ``complex.cif`` (the sampled side chains), the ligand its ``ligand.sdf``,
and smina scores the pose in place (``--score_only``). One row per pose with the
columns ``run, system, sample, <scoring>, error``, the table that
``scripts/eval_posebusters_redock.py --score-csv`` and
``scripts/eval_posebusters_selected.py --scores --score-col vinardo`` read.

    python scripts/smina_rescore.py --run tpd=tpd/prepared --out tpd/vinardo.csv

``smina`` comes from ``--smina`` / ``$SMINA_BIN`` / ``$PATH``. Re-running with the
same ``--out`` scores only poses not already in the CSV.
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from equiflex.inference.vina_scoring import extract_protein_pdb

AFFINITY = re.compile(r'^Affinity:\s*(-?\d+\.?\d*)', re.M)


def score_pose(job):
    run, system, sample, complex_path, ligand_path, smina, scoring = job
    row = {'run': run, 'system': system, 'sample': sample, scoring: None, 'error': None}
    with tempfile.TemporaryDirectory(prefix='smina_pose_') as tmp:
        protein = Path(tmp) / 'protein.pdb'
        if not extract_protein_pdb(complex_path, protein):
            row['error'] = 'no protein residues in complex'
            return row
        cmd = [smina, '--score_only', '-r', str(protein), '-l', str(ligand_path)]
        if scoring != 'vina':
            cmd += ['--scoring', scoring]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            row['error'] = 'timeout'
            return row
    match = AFFINITY.search(out.stdout)
    if match:
        row[scoring] = float(match.group(1))
    else:
        row['error'] = (out.stderr.strip().splitlines() or ['no Affinity line'])[-1][:200]
    return row


def collect_jobs(runs, smina, scoring):
    jobs = []
    for run, root in runs.items():
        for sys_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            for sample_dir in sorted(sys_dir.glob('sample_*')):
                complex_path, ligand_path = sample_dir / 'complex.cif', sample_dir / 'ligand.sdf'
                if complex_path.exists() and ligand_path.exists():
                    jobs.append((run, sys_dir.name, sample_dir.name,
                                 complex_path, ligand_path, smina, scoring))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True,
                    metavar='LABEL=ROOT', help='run_inference output_root to score; repeatable')
    ap.add_argument('--out', type=Path, required=True, help='CSV with one row per pose')
    ap.add_argument('--scoring', default='vinardo', help='smina scoring function (default: vinardo)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--smina', default=os.environ.get('SMINA_BIN', 'smina'),
                    help='smina executable (default: $SMINA_BIN or "smina" on PATH)')
    args = ap.parse_args()

    if shutil.which(args.smina) is None:
        ap.error(f'smina executable not found: {args.smina}')

    runs = dict(r.split('=', 1) for r in args.run)
    jobs = collect_jobs(runs, args.smina, args.scoring)

    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(map(tuple, prev[['run', 'system', 'sample']].values))
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f'{len(done)} poses cached in {args.out}')
    print(f'scoring {len(jobs)} poses with {args.smina} ({args.scoring}) on {args.workers} workers')

    rows = []
    if jobs:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(score_pose, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures)):
                rows.append(future.result())

    new = pd.DataFrame(rows, columns=['run', 'system', 'sample', args.scoring, 'error'])
    if args.out.exists():
        new = pd.concat([pd.read_csv(args.out), new], ignore_index=True)
    new = new.sort_values(['run', 'system', 'sample']).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.out, index=False)

    n_err = int(new['error'].notna().sum())
    print(f'{len(new)} poses -> {args.out}  ({n_err} errors)')
    if n_err:
        print(new.loc[new['error'].notna(), 'error'].value_counts().head(10).to_string())


if __name__ == '__main__':
    main()
