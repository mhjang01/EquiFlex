"""Symmetry-corrected ligand RMSD for every generated pose (no PoseBusters checks).

Uses PoseBusters' own ``check_rmsd``, so values match the ``rmsd`` column of
``eval_posebusters_redock.py``; the geometry checks, which dominate that script's
cost, are skipped. Per pose the minimum over deposited ground-truth copies is kept.

    python scripts/rmsd_all_poses.py --pred-root /mnt/drive/crossdock_v10_41k \
        --gt-dir /mnt/drive/crossdock_v10_41k_gt --out crossdock/results_v10_41k/rmsd_all.csv
"""

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger
from posebusters.modules.rmsd import check_rmsd

RDLogger.DisableLog('rdApp.*')


def gt_path(gt_dir: Path, system_id: str):
    d = gt_dir / system_id
    for pattern in ('*_ligands.sdf', '*_ligand.sdf'):
        hits = sorted(p for p in d.glob(pattern) if 'start_conf' not in p.name)
        if hits:
            return hits[0]
    return None


def one_system(task):
    system_id, sys_dir, gt_sdf = task
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        copies = [m for m in Chem.SDMolSupplier(str(gt_sdf)) if m is not None]
        rows = []
        for sample_dir in sorted(sys_dir.glob('sample_*'), key=lambda p: int(p.name.split('_')[1])):
            pred = sample_dir / 'ligand.sdf'
            if not pred.exists():
                continue
            mol = Chem.MolFromMolFile(str(pred), sanitize=True)
            best = None
            for true in copies:
                try:
                    r = check_rmsd(mol_pred=mol, mol_true=true)['results']['rmsd']
                except Exception:
                    continue
                if r is not None and (best is None or r < best):
                    best = r
            rows.append((system_id, int(sample_dir.name.split('_')[1]), best))
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred-root', type=Path, required=True)
    ap.add_argument('--gt-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N systems only')
    args = ap.parse_args()

    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {r['system'] for r in csv.DictReader(f)}

    tasks = []
    for sys_dir in sorted(p for p in args.pred_root.iterdir() if p.is_dir()):
        if sys_dir.name in done or sys_dir.name == 'logs':
            continue
        gt = gt_path(args.gt_dir, sys_dir.name)
        if gt is not None:
            tasks.append((sys_dir.name, sys_dir, gt))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f'{len(tasks)} systems to score ({len(done)} already in {args.out})', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a', newline='') as f:
        w = csv.writer(f)
        if not done:
            w.writerow(['system', 'sample', 'rmsd'])
        with ProcessPoolExecutor(args.workers) as pool:
            for i, rows in enumerate(pool.map(one_system, tasks, chunksize=4), 1):
                w.writerows(rows)
                if i % 200 == 0:
                    f.flush()
                    print(f'{i}/{len(tasks)}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
