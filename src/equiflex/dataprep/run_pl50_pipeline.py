'''
PL50 pipeline runner: metadata extraction + featurization for train/val/test.

Stages:
1. parse_single_system per split (parallel CPU, skip_existing=True)
2. featurize_data.main per split (strict=True)
3. Aggregate failure summary across splits and steps
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
from equiflex.dataprep.featurize_data import main as featurize_main


PLINDER_SYSTEM_DIR = Path('/mnt/drive/new_plinder/v2/systems/')
PARQUET_DIR = Path('/mnt/drive/plinder/2024-06/v2/splits/plinder-no-posebusters.parquet/')
STORAGE_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')

SPLITS = ['train', 'val', 'test']


def load_split_systems(use_survivors: bool = False):
    if use_survivors:
        buckets = {}
        for s in SPLITS:
            fp = STORAGE_ROOT / f'{s}_survivors.txt'
            if not fp.exists():
                raise FileNotFoundError(f'Survivor file missing: {fp}')
            buckets[s] = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
        return buckets
    df = pd.read_parquet(PARQUET_DIR)
    buckets = {s: [] for s in SPLITS}
    for sid, sp in zip(df['system_id'], df['split']):
        if sp in buckets:
            buckets[sp].append(sid)
    return buckets


def run_metadata_extraction(buckets, n_jobs=-1, force_splits=()):
    print('\n' + '=' * 60)
    print('STAGE 1: Metadata extraction (parse_single_system)')
    print('=' * 60)
    missing_input = {s: [] for s in SPLITS}
    for split in SPLITS:
        sys_ids = buckets[split]
        storage_dir = STORAGE_ROOT / split
        storage_dir.mkdir(parents=True, exist_ok=True)
        sys_dirs = []
        for sid in sys_ids:
            src = PLINDER_SYSTEM_DIR / sid
            if not src.exists():
                missing_input[split].append(sid)
                continue
            sys_dirs.append(src)
        skip_existing = split not in force_splits
        mode = 'FORCE-REGEN' if not skip_existing else 'skip-existing'
        print(f'\n[{split}] {len(sys_dirs)}/{len(sys_ids)} system dirs available '
              f'({len(missing_input[split])} missing source) [{mode}]')
        Parallel(n_jobs=n_jobs)(
            delayed(parse_single_system)(sd, storage_dir, skip_existing)
            for sd in tqdm(sys_dirs, desc=f'metadata:{split}')
        )
    return missing_input


def run_featurization():
    print('\n' + '=' * 60)
    print('STAGE 2: Featurization (strict mode)')
    print('=' * 60)
    for split in SPLITS:
        data_dir = STORAGE_ROOT / split
        if not data_dir.exists():
            print(f'[{split}] missing dir, skip')
            continue
        print(f'\n[{split}] featurize_data.main on {data_dir}')
        featurize_main(
            data_dir=str(data_dir),
            n_jobs=-1,
            skip_existing=True,
            cleanup_invalid=True,
            run_esm3=True,
            run_unimol=True,
            unimol_size='1.1B',
            device='cuda',
            strict=True,
        )


def aggregate_failures(missing_input):
    print('\n' + '=' * 60)
    print('STAGE 3: Failure summary')
    print('=' * 60)
    summary = {'missing_source_dirs': {}, 'metadata_empty': {}, 'featurize': {}}

    # Source dirs not unzipped
    for split, ids in missing_input.items():
        summary['missing_source_dirs'][split] = {
            'count': len(ids),
            'examples': ids[:10],
        }

    # Metadata-stage: storage subdirs lacking pkl files
    for split in SPLITS:
        storage_dir = STORAGE_ROOT / split
        if not storage_dir.exists():
            summary['metadata_empty'][split] = {'count': 0, 'examples': []}
            continue
        empty = []
        for sub in storage_dir.iterdir():
            if not sub.is_dir():
                continue
            if not list(sub.glob('*.pkl')):
                empty.append(sub.name)
        summary['metadata_empty'][split] = {
            'count': len(empty),
            'examples': empty[:10],
        }

    # Featurize-stage error files
    for split in SPLITS:
        data_dir = STORAGE_ROOT / split
        split_errs = {}
        for fname in ['step1_errors.txt', 'step2_esm3_errors.txt', 'step3_unimol_errors.txt']:
            fp = data_dir / fname
            if not fp.exists():
                split_errs[fname] = {'count': 0, 'examples': []}
                continue
            lines = [l.rstrip() for l in fp.read_text().splitlines() if l.strip()]
            split_errs[fname] = {'count': len(lines), 'examples': lines[:10]}
        # featurize stats
        stats_fp = data_dir / 'featurize_stats.json'
        if stats_fp.exists():
            split_errs['featurize_stats'] = json.loads(stats_fp.read_text())
        summary['featurize'][split] = split_errs

    out = STORAGE_ROOT / 'pipeline_failure_summary.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')
    print(json.dumps({
        'missing_source_dirs': {s: summary['missing_source_dirs'][s]['count'] for s in SPLITS},
        'metadata_empty': {s: summary['metadata_empty'][s]['count'] for s in SPLITS},
        'featurize_step_errors': {
            s: {k: v['count'] if isinstance(v, dict) and 'count' in v else None
                for k, v in summary['featurize'][s].items()}
            for s in SPLITS
        },
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-metadata', action='store_true')
    parser.add_argument('--skip-featurize', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--force-splits', type=str, default='',
                        help='Comma-separated splits to regenerate metadata for (skip_existing=False)')
    parser.add_argument('--use-survivors', action='store_true',
                        help='Load system_ids from {split}_survivors.txt instead of parquet')
    args = parser.parse_args()

    force_splits = tuple(s.strip() for s in args.force_splits.split(',') if s.strip())

    buckets = load_split_systems(use_survivors=args.use_survivors)
    print({s: len(buckets[s]) for s in SPLITS})

    missing_input = {s: [] for s in SPLITS}
    if not args.skip_metadata:
        missing_input = run_metadata_extraction(buckets, n_jobs=args.n_jobs, force_splits=force_splits)

    if not args.skip_featurize:
        run_featurization()

    aggregate_failures(missing_input)


if __name__ == '__main__':
    main()
