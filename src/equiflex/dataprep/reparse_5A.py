'''
Re-parse Stage 1 (metadata extraction) into v2_plinder_5A with:
    - 5.0 Å pocket adjacency cutoff (was 4.0 Å)
    - Full label_asym_id as chain key (fixes assembly-collapse bug that put
      far-copy coords into "pocket" residues; see pocket_cutoff_violations.json).

Source: same parquet as run_pl50_pipeline.py. Output dir: v2_plinder_5A/.
Featurization is NOT run here; the existing v2_plinder/ featurize artifacts
are stale w.r.t. the new chain_id scheme and must be regenerated separately.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from equiflex.dataprep.metadata_extraction import parse_single_system

PLINDER_SYSTEM_DIR = Path('/mnt/drive/new_plinder/v2/systems/')
PARQUET_DIR = Path('/mnt/drive/plinder/2024-06/v2/splits/plinder-no-posebusters.parquet/')
STORAGE_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')

SPLITS = ['train', 'val', 'test']


def load_split_systems():
    df = pd.read_parquet(PARQUET_DIR)
    buckets = {s: [] for s in SPLITS}
    for sid, sp in zip(df['system_id'], df['split']):
        if sp in buckets:
            buckets[sp].append(sid)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--splits', type=str, default='train,val,test')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Do not re-process system dirs that already have pkls in v2_plinder_5A')
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    buckets = load_split_systems()
    print({s: len(buckets[s]) for s in SPLITS})

    summary = {}
    for split in splits:
        if split not in buckets:
            continue
        storage_dir = STORAGE_ROOT / split
        storage_dir.mkdir(parents=True, exist_ok=True)

        sys_dirs, missing = [], []
        for sid in buckets[split]:
            src = PLINDER_SYSTEM_DIR / sid
            if src.exists():
                sys_dirs.append(src)
            else:
                missing.append(sid)

        print(f'\n[{split}] {len(sys_dirs)} present, {len(missing)} missing source')
        Parallel(n_jobs=args.n_jobs)(
            delayed(parse_single_system)(sd, storage_dir, args.skip_existing)
            for sd in tqdm(sys_dirs, desc=f'reparse:{split}')
        )

        produced = sum(
            1 for d in storage_dir.iterdir()
            if d.is_dir() and (d / 'pocket_residues.pkl').exists()
        )
        summary[split] = {
            'requested': len(buckets[split]),
            'present_source': len(sys_dirs),
            'missing_source': len(missing),
            'produced_pkl': produced,
        }
        print(f'[{split}] produced pkl dirs: {produced}')

    (STORAGE_ROOT / 'reparse_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nSummary:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
