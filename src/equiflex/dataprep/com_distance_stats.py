'''
Backbone-COM vs ligand-COM distance stats for v2_plinder.

For each system:
    pocket_bb_com = mean of (N, CA, C, O) coords across all pocket residues
                    (first 4 atoms per residue; missing atoms masked out)
    ligand_com    = mean of all ligand-atom coords (all valid ligs concatenated)
    d             = || ligand_com - pocket_bb_com ||

Also reports n_pocket_residues and n_valid_bb_atoms.

Outputs JSON histogram + percentiles per split.
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


def system_stats(system_dir):
    lf = system_dir / 'ligands.pkl'
    pf = system_dir / 'pocket_residues.pkl'
    if not (lf.exists() and pf.exists()):
        return None
    try:
        ligs = load(lf)
        pocket = load(pf)
    except Exception:
        return None

    # Ligand COM
    lig_coords = np.concatenate([np.asarray(l['coords']) for l in ligs], axis=0)
    lig_com = lig_coords.mean(axis=0)

    # Pocket backbone COM: first 4 atoms per residue, masked
    bb_pts = []
    n_res = 0
    for chain_id, residues in pocket.items():
        for res in residues.values():
            n_res += 1
            coords = np.asarray(res['coords'])[:4]              # N,CA,C,O
            mask = np.asarray(res['missing_masks'])[:4]
            valid = coords[~mask]
            if len(valid):
                bb_pts.append(valid)

    if not bb_pts:
        return system_dir.name, None, None, n_res, 0
    bb = np.concatenate(bb_pts, axis=0)
    bb_com = bb.mean(axis=0)
    d = float(np.linalg.norm(lig_com - bb_com))
    return system_dir.name, d, float(np.linalg.norm(lig_coords.std(axis=0))), n_res, int(bb.shape[0])


def collect_split(split, n_jobs):
    sd = STORAGE_ROOT / split
    dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
    print(f'[{split}] {len(dirs)} dirs')
    out = Parallel(n_jobs=n_jobs)(
        delayed(system_stats)(d) for d in tqdm(dirs, desc=f'com:{split}')
    )
    return [r for r in out if r is not None and r[1] is not None]


def summarize(records):
    ds = np.array([r[1] for r in records])
    n_res = np.array([r[3] for r in records])
    bb_n = np.array([r[4] for r in records])

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    hist_edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 1e9]
    hist_counts, _ = np.histogram(ds, bins=hist_edges)
    hist = {f'{hist_edges[i]:g}-{hist_edges[i+1]:g}': int(c) for i, c in enumerate(hist_counts)}

    thresholds = [2, 3, 4, 5, 6, 8, 10, 15]
    above = {f'>={t}A': int((ds >= t).sum()) for t in thresholds}

    return {
        'n_systems': len(ds),
        'distance_A': {
            'mean': float(ds.mean()),
            'std': float(ds.std()),
            'min': float(ds.min()),
            'max': float(ds.max()),
            'percentiles': {f'p{p}': float(np.percentile(ds, p)) for p in pcts},
        },
        'hist_A': hist,
        'fraction_above': {k: v / len(ds) for k, v in above.items()},
        'count_above': above,
        'n_pocket_residues': {
            'mean': float(n_res.mean()),
            'p5': float(np.percentile(n_res, 5)),
            'p50': float(np.percentile(n_res, 50)),
            'p95': float(np.percentile(n_res, 95)),
            'min': int(n_res.min()),
            'max': int(n_res.max()),
            'le_2': int((n_res <= 2).sum()),
            'le_4': int((n_res <= 4).sum()),
            'le_6': int((n_res <= 6).sum()),
        },
        'n_bb_atoms': {
            'mean': float(bb_n.mean()),
            'min': int(bb_n.min()),
            'p5': float(np.percentile(bb_n, 5)),
        },
    }


def cross_tab(records):
    '''How does distance correlate with pocket-residue count?'''
    ds = np.array([r[1] for r in records])
    n_res = np.array([r[3] for r in records])
    bins = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 1_000)]
    out = {}
    for lo, hi in bins:
        mask = (n_res >= lo) & (n_res <= hi)
        if not mask.any():
            continue
        d_sub = ds[mask]
        out[f'pocket_res_{lo}-{hi}'] = {
            'n': int(mask.sum()),
            'd_median': float(np.median(d_sub)),
            'd_p95': float(np.percentile(d_sub, 95)),
            'frac_above_6A': float((d_sub >= 6).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--splits', type=str, default='train,val,test')
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    summary = {}
    examples = {}
    for split in splits:
        recs = collect_split(split, args.n_jobs)
        summary[split] = summarize(recs)
        summary[split]['by_pocket_size'] = cross_tab(recs)

        # Worst-offender examples (largest distance, smallest pocket)
        worst_d = sorted(recs, key=lambda r: -r[1])[:10]
        small_pocket = sorted(recs, key=lambda r: r[3])[:10]
        examples[split] = {
            'largest_distance': [
                {'sys_id': r[0], 'd_A': round(r[1], 2), 'n_pocket_res': r[3], 'n_bb_atoms': r[4]}
                for r in worst_d
            ],
            'smallest_pocket': [
                {'sys_id': r[0], 'd_A': round(r[1], 2), 'n_pocket_res': r[3], 'n_bb_atoms': r[4]}
                for r in small_pocket
            ],
        }
        summary[split]['examples'] = examples[split]

    out = STORAGE_ROOT / 'com_distance_stats.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')

    print('\n=== HEADLINE ===')
    print(f'{"split":<7}{"n":<8}{"mean":<8}{"med":<8}{"p95":<8}{"p99":<8}{"max":<8}{">=6A":<10}{">=10A":<10}{"<=2res":<10}')
    for split in splits:
        s = summary[split]
        d = s['distance_A']
        n = s['n_systems']
        p = d['percentiles']
        print(f'{split:<7}{n:<8}'
              f'{d["mean"]:<8.2f}{p["p50"]:<8.2f}{p["p95"]:<8.2f}{p["p99"]:<8.2f}{d["max"]:<8.1f}'
              f'{s["count_above"][">=6A"]:<10}{s["count_above"][">=10A"]:<10}'
              f'{s["n_pocket_residues"]["le_2"]:<10}')


if __name__ == '__main__':
    main()
