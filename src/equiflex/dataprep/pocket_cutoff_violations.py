'''
Sanity check: does every stored pocket residue have >=1 atom within a given
distance of some ligand atom?

For each system:
    Load valid ligand atoms (concat all ligand coords).
    For each pocket residue: take its stored coords[i] where ~missing_masks[i].
    min_dist = min ||residue_atom - ligand_atom|| over all (i, j) valid pairs.
    Residue 'violates' at cutoff c if min_dist > c.

Reports per-cutoff:
    - n_systems with any violating residue
    - n violating residues total
    - fraction of pocket residues that violate
    - examples of worst offenders

Cutoffs scanned by default: 4.0, 5.0, 5.1, 6.0 A.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, load
from tqdm import tqdm

STORAGE_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')
SPLITS = ['train', 'val', 'test']
CUTOFFS = [4.0, 5.0, 5.1, 6.0]


def system_min_dists(system_dir):
    '''Return list of per-residue min distances to nearest ligand atom, plus sys id.'''
    lf = system_dir / 'ligands.pkl'
    pf = system_dir / 'pocket_residues.pkl'
    if not (lf.exists() and pf.exists()):
        return None
    try:
        ligs = load(lf)
        pocket = load(pf)
    except Exception:
        return None

    lig_xyz = np.concatenate([np.asarray(l['coords']) for l in ligs], axis=0)  # [L,3]

    per_res = []
    for chain_id, residues in pocket.items():
        for key, res in residues.items():
            coords = np.asarray(res['coords'])
            mask = np.asarray(res['missing_masks'])
            valid = coords[~mask]
            if len(valid) == 0:
                per_res.append((chain_id, key, np.inf))
                continue
            diff = valid[:, None, :] - lig_xyz[None, :, :]
            dist = np.linalg.norm(diff, axis=2).min()
            per_res.append((chain_id, key, float(dist)))
    return system_dir.name, per_res


def collect_split(split, n_jobs):
    sd = STORAGE_ROOT / split
    dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
    print(f'[{split}] {len(dirs)} dirs')
    out = Parallel(n_jobs=n_jobs)(
        delayed(system_min_dists)(d) for d in tqdm(dirs, desc=f'viol:{split}')
    )
    return [r for r in out if r is not None]


def summarize(records, cutoffs):
    '''records: list of (sys_id, [(chain, key, min_d), ...]).'''
    summary = {}
    all_dists = []
    for _, per_res in records:
        all_dists.extend(d for _, _, d in per_res)
    all_dists = np.array(all_dists)
    n_total_res = len(all_dists)
    n_systems = len(records)

    for c in cutoffs:
        n_viol_res = int((all_dists > c).sum())
        sys_with_viol = 0
        worst = []  # (sys_id, n_viol, max_min_d)
        for sys_id, per_res in records:
            min_d_arr = np.array([d for _, _, d in per_res])
            n_v = int((min_d_arr > c).sum())
            if n_v > 0:
                sys_with_viol += 1
                worst.append((sys_id, n_v, float(min_d_arr.max())))
        worst.sort(key=lambda r: (-r[1], -r[2]))
        summary[f'cutoff_{c}'] = {
            'n_violating_residues': n_viol_res,
            'frac_violating_residues': n_viol_res / n_total_res if n_total_res else 0.0,
            'n_systems_with_any_violation': sys_with_viol,
            'frac_systems_with_any_violation': sys_with_viol / n_systems if n_systems else 0.0,
            'worst_systems': [
                {'sys_id': sid, 'n_violating_residues': nv, 'max_min_dist_A': round(md, 2)}
                for sid, nv, md in worst[:10]
            ],
        }

    pcts = [50, 75, 90, 95, 99, 99.9]
    summary['per_residue_min_dist_A'] = {
        'n_residues': n_total_res,
        'n_systems': n_systems,
        'mean': float(all_dists[np.isfinite(all_dists)].mean()),
        'max': float(np.nan_to_num(all_dists, nan=0, posinf=0).max()),
        'percentiles': {f'p{p}': float(np.percentile(all_dists, p)) for p in pcts},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--splits', type=str, default='train,val,test')
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    summary = {}
    for split in splits:
        recs = collect_split(split, args.n_jobs)
        summary[split] = summarize(recs, CUTOFFS)

    out = STORAGE_ROOT / 'pocket_cutoff_violations.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')

    print('\n=== HEADLINE: residues with min_dist_to_lig > cutoff ===')
    print(f'{"split":<7}{"n_sys":<8}{"n_res":<10}',
          ''.join(f'{f"cut={c}A":<22}' for c in CUTOFFS))
    for split in splits:
        s = summary[split]
        row = f'{split:<7}{s["per_residue_min_dist_A"]["n_systems"]:<8}{s["per_residue_min_dist_A"]["n_residues"]:<10}'
        for c in CUTOFFS:
            d = s[f'cutoff_{c}']
            row += f'{d["n_violating_residues"]:>5}res {d["n_systems_with_any_violation"]:>5}sys '.ljust(22)
        print(row)

    print('\n=== per-residue min_dist percentiles (train) ===')
    print(summary['train']['per_residue_min_dist_A'])


if __name__ == '__main__':
    main()
