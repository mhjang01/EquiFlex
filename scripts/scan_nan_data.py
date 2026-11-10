"""
Fast coord-only scan for cdist-NaN-inducing systems.

Loads only pocket+ligand coord tensors (+ missing_mask). Checks:
  1. NaN/Inf in coords
  2. Min pairwise distance after combining valid rec + lig atoms (mirrors training _shared_step)

cdist backward NaN is triggered when distance = 0 (duplicate atoms) because
grad = (x - y) / dist. Anything < ~1e-4 Å is suspect.
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial


def scan_system(system_path: Path, min_dist_threshold: float = 1e-3):
    issues = {}
    try:
        p = torch.load(system_path / 'pocket_features.pt', weights_only=False, map_location='cpu')
        l = torch.load(system_path / 'ligand_features.pt', weights_only=False, map_location='cpu')
    except Exception as e:
        return {'system': system_path.name, 'error': f'load_failed: {e!s}'[:120]}

    rec_coords = p['coords']
    lig_coords = l['coords']
    missing_mask = p.get('missing_mask')
    if missing_mask is None:
        missing_mask = torch.zeros(rec_coords.shape[0], dtype=torch.bool)
    elif not torch.is_tensor(missing_mask):
        try:
            missing_mask = torch.as_tensor(missing_mask).bool()
        except Exception:
            missing_mask = torch.zeros(rec_coords.shape[0], dtype=torch.bool)

    if torch.isnan(rec_coords).any():
        issues['rec_coord_nan'] = int(torch.isnan(rec_coords).any(dim=-1).sum())
    if torch.isinf(rec_coords).any():
        issues['rec_coord_inf'] = int(torch.isinf(rec_coords).any(dim=-1).sum())
    if torch.isnan(lig_coords).any():
        issues['lig_coord_nan'] = int(torch.isnan(lig_coords).any(dim=-1).sum())
    if torch.isinf(lig_coords).any():
        issues['lig_coord_inf'] = int(torch.isinf(lig_coords).any(dim=-1).sum())

    # Skip pair-dist check if coords already broken
    if not issues:
        valid_rec = rec_coords[~missing_mask]
        if valid_rec.shape[0] > 0:
            all_coords = torch.cat([valid_rec, lig_coords], dim=0)
        else:
            all_coords = lig_coords
        if all_coords.shape[0] > 1:
            d = torch.cdist(all_coords, all_coords)
            d.fill_diagonal_(float('inf'))
            min_d, flat_idx = d.view(-1).min(dim=0)
            min_d = min_d.item()
            if min_d < min_dist_threshold:
                N = all_coords.shape[0]
                i = int(flat_idx.item() // N)
                j = int(flat_idx.item() % N)
                n_rec_valid = valid_rec.shape[0]
                tag_i = f'rec{i}' if i < n_rec_valid else f'lig{i - n_rec_valid}'
                tag_j = f'rec{j}' if j < n_rec_valid else f'lig{j - n_rec_valid}'
                issues['min_pair_dist'] = (round(min_d, 6), tag_i, tag_j)

    if not issues:
        return None
    issues['system'] = system_path.name
    return issues


def scan_dir(data_dir: Path, min_dist_threshold: float, workers: int, limit: int = None):
    systems = sorted([d for d in data_dir.iterdir()
                      if d.is_dir() and (d / 'ligand_features.pt').exists() and (d / 'pocket_features.pt').exists()])
    if limit:
        systems = systems[:limit]
    print(f'Scanning {len(systems)} systems in {data_dir} with {workers} workers')

    fn = partial(scan_system, min_dist_threshold=min_dist_threshold)
    bad = []
    if workers <= 1:
        for i, s in enumerate(systems):
            r = fn(s)
            if r:
                bad.append(r)
            if (i + 1) % 2000 == 0:
                print(f'  ... {i+1}/{len(systems)} ({len(bad)} flagged)', file=sys.stderr, flush=True)
    else:
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, systems, chunksize=64)):
                if r:
                    bad.append(r)
                if (i + 1) % 2000 == 0:
                    print(f'  ... {i+1}/{len(systems)} ({len(bad)} flagged)', file=sys.stderr, flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='featurised data root containing the split subdirectories')
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    ap.add_argument('--min-dist', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='nan_scan_report.txt')
    args = ap.parse_args()

    root = Path(args.data)
    all_bad = {}
    for split in args.splits:
        d = root / split
        if not d.is_dir():
            print(f'skip {d}: not a dir')
            continue
        bad = scan_dir(d, args.min_dist, args.workers, args.limit)
        all_bad[split] = bad
        print(f'\n=== {split}: {len(bad)} flagged systems ===')

    with open(args.out, 'w') as f:
        for split, bad in all_bad.items():
            f.write(f'\n=== Split: {split} ({len(bad)} flagged) ===\n')

            def severity(item):
                keys = set(item.keys()) - {'system'}
                fatal = sum(1 for k in keys if 'nan' in k or 'inf' in k or 'error' in k)
                min_d = item.get('min_pair_dist', (10.0,))[0] if 'min_pair_dist' in item else 10.0
                return (-fatal, min_d)
            bad_sorted = sorted(bad, key=severity)
            for item in bad_sorted:
                detail = {k: v for k, v in item.items() if k != 'system'}
                f.write(f'{item["system"]}\t{detail}\n')

    print(f'\nReport written to {args.out}')
    for split, bad in all_bad.items():
        nan_systems = [b for b in bad if any('nan' in k or 'inf' in k or 'error' in k for k in b)]
        close_systems = [b for b in bad if 'min_pair_dist' in b and not any('nan' in k or 'inf' in k for k in b)]
        print(f'\n--- {split}: {len(nan_systems)} NaN/Inf/error, {len(close_systems)} close-pair only ---')
        for item in nan_systems[:30]:
            detail = {k: v for k, v in item.items() if k != 'system'}
            print(f'  [FATAL] {item["system"]}: {detail}')
        # Show worst close-pair systems (lowest dist)
        close_sorted = sorted(close_systems, key=lambda x: x['min_pair_dist'][0])[:20]
        for item in close_sorted:
            detail = {k: v for k, v in item.items() if k != 'system'}
            print(f'  [close]  {item["system"]}: {detail}')


if __name__ == '__main__':
    main()
