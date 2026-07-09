'''
Featurization script for bsite_unique dataset.

Processes the extracted metadata from bsite_unique_extraction.py:
1. Graph featurization (atom/bond features, coordinates, masks)
2. ESM3 embedding extraction
3. UniMol embedding extraction

Usage:
    # Run all steps
    python exec/bsite_unique_featurize.py --data-dir ./bsite_data

    # Run specific steps
    python exec/bsite_unique_featurize.py --data-dir ./bsite_data --no-esm3
    python exec/bsite_unique_featurize.py --data-dir ./bsite_data --no-unimol

    # Skip existing and use strict mode (remove failures)
    python exec/bsite_unique_featurize.py --data-dir ./bsite_data --strict
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import List, Optional
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

# Import featurization functions from existing featurize_data.py
from equiflex.dataprep.featurize_data import (
    run_step1_graph_featurization,
    run_step2_esm3_extraction,
    run_step3_unimol_extraction,
    FeaturizationStats,
)


def main(
    data_dir: str = '/home/mhjang/workspace/flex/bsite_data',
    n_jobs: int = -1,
    skip_existing: bool = True,
    cleanup_invalid: bool = True,
    run_esm3: bool = True,
    run_unimol: bool = True,
    unimol_size: str = '1.1B',
    device: str = 'cuda',
    strict: bool = False
):
    '''
    Run featurization pipeline on bsite_unique extracted data.

    Args:
        data_dir: Directory containing extracted metadata (ligands.pkl, pocket_residues.pkl)
        n_jobs: Number of parallel jobs for graph featurization
        skip_existing: Skip if features already exist
        cleanup_invalid: Remove directories with invalid data
        run_esm3: Run ESM3 embedding extraction
        run_unimol: Run UniMol embedding extraction
        unimol_size: UniMol model size ('570m' or '1.1B')
        device: Device for GPU models
        strict: Remove directories that fail ESM3/UniMol extraction
    '''
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    print("=" * 60)
    print("BSITE_UNIQUE FEATURIZATION PIPELINE")
    print("=" * 60)
    print(f"Data directory: {data_path}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Skip existing: {skip_existing}")
    print(f"Cleanup invalid: {cleanup_invalid}")
    print(f"Run ESM3: {run_esm3}")
    print(f"Run UniMol: {run_unimol}")
    if strict:
        print("STRICT MODE: Directories failing extraction will be removed")
    print("=" * 60)

    # Step 1: Graph featurization
    global_stats, valid_dirs = run_step1_graph_featurization(
        data_path, n_jobs, cleanup_invalid
    )

    # Step 2: ESM3 extraction
    if run_esm3:
        valid_dirs = run_step2_esm3_extraction(
            data_path, valid_dirs, device, skip_existing, strict=strict
        )

    # Step 3: UniMol extraction
    if run_unimol:
        valid_dirs = run_step3_unimol_extraction(
            data_path, valid_dirs, unimol_size, skip_existing=skip_existing, strict=strict
        )

    # Final statistics
    global_stats.print_summary()
    print(f"Final valid directories: {len(valid_dirs)}")

    # Save statistics
    import json
    stats_summary = global_stats.summarize()
    stats_file = data_path / 'featurize_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Statistics saved to: {stats_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Featurize bsite_unique dataset')
    parser.add_argument('--data-dir', type=str, default='/home/mhjang/workspace/flex/bsite_data',
                        help='Directory containing extracted metadata')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for Step 1')
    parser.add_argument('--no-skip', action='store_true',
                        help='Reprocess even if features exist')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Do not remove invalid directories')
    parser.add_argument('--no-esm3', action='store_true',
                        help='Skip ESM3 extraction')
    parser.add_argument('--no-unimol', action='store_true',
                        help='Skip UniMol extraction')
    parser.add_argument('--unimol-size', type=str, default='1.1B',
                        choices=['570m', '1.1B'],
                        help='UniMol model size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for GPU models')
    parser.add_argument('--strict', action='store_true',
                        help='Remove directories that fail ESM3/UniMol extraction')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        n_jobs=args.n_jobs,
        skip_existing=not args.no_skip,
        cleanup_invalid=not args.no_cleanup,
        run_esm3=not args.no_esm3,
        run_unimol=not args.no_unimol,
        unimol_size=args.unimol_size,
        device=args.device,
        strict=args.strict,
    )
