"""
Create train/val/test split CSV for PLINDER data.

Generates a CSV file mapping system IDs to their splits, which can be used
by the data loader for consistent train/val/test separation.

Usage:
    python exec/lp_plinder/create_splits.py --data-dir ./plinder_data --output splits.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


REQUIRED_FILES = [
    'ligand_features.pt',
    'pocket_features.pt',
    'esm3_embeddings.pt',
    'unimol_embeddings.pt',
]


def is_valid_directory(system_dir: Path) -> bool:
    """Check if all required files exist."""
    return all((system_dir / f).exists() for f in REQUIRED_FILES)


def create_splits_from_plinder_dirs(
    plinder_dir: Path,
    output_path: Path,
    test_dir: Path = None,
):
    """
    Create splits CSV based on PLINDER's original train/val structure.
    
    The PLINDER data already has train/val splits, so we use those directly.
    Optionally includes an external test directory.
    """
    records = []
    
    for split in ['train', 'val']:
        split_dir = plinder_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found")
            continue
            
        systems = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"Found {len(systems)} systems in {split}")
        
        for system_dir in tqdm(systems, desc=f"Processing {split}"):
            records.append({
                'system_id': system_dir.name,
                'new_split': split,
                'source_path': str(system_dir),
            })
    
    # Add external test directory if provided
    if test_dir is not None and test_dir.exists():
        systems = [d for d in test_dir.iterdir() if d.is_dir()]
        print(f"Found {len(systems)} systems in test (external)")
        
        for system_dir in tqdm(systems, desc="Processing test"):
            records.append({
                'system_id': system_dir.name,
                'new_split': 'test',
                'source_path': str(system_dir),
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} records to {output_path}")
    print(f"  Train: {len(df[df['new_split'] == 'train'])}")
    print(f"  Val: {len(df[df['new_split'] == 'val'])}")
    print(f"  Test: {len(df[df['new_split'] == 'test'])}")
    
    return df


def create_splits_from_processed_dir(
    data_dir: Path,
    output_path: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Create splits CSV from a processed data directory.
    
    If the original split information is not available, randomly split
    the data into train/val/test.
    """
    import random
    random.seed(seed)
    
    # Find all valid directories
    all_dirs = [d for d in data_dir.iterdir() 
                if d.is_dir() and d.name != 'errors' and is_valid_directory(d)]
    
    print(f"Found {len(all_dirs)} valid directories")
    
    # Shuffle
    random.shuffle(all_dirs)
    
    # Split
    n_total = len(all_dirs)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    train_dirs = all_dirs[:n_train]
    val_dirs = all_dirs[n_train:n_train + n_val]
    test_dirs = all_dirs[n_train + n_val:]
    
    records = []
    for system_dir in train_dirs:
        records.append({'system_id': system_dir.name, 'new_split': 'train'})
    for system_dir in val_dirs:
        records.append({'system_id': system_dir.name, 'new_split': 'val'})
    for system_dir in test_dirs:
        records.append({'system_id': system_dir.name, 'new_split': 'test'})
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df)} records to {output_path}")
    print(f"  Train: {n_train}")
    print(f"  Val: {n_val}")
    print(f"  Test: {n_test}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Create splits CSV for PLINDER data')
    parser.add_argument('--plinder-dir', type=str, default=None,
                        help='Original PLINDER directory with train/val subdirs')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Processed data directory (if no plinder-dir)')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='External test data directory (already processed)')
    parser.add_argument('--output', type=str, default='plinder_splits.csv',
                        help='Output CSV path')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation ratio (only used with --data-dir)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test ratio (only used with --data-dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    test_dir = Path(args.test_dir) if args.test_dir else None
    
    if args.plinder_dir:
        # Use original PLINDER splits
        plinder_dir = Path(args.plinder_dir)
        if not plinder_dir.exists():
            raise FileNotFoundError(f"PLINDER directory not found: {plinder_dir}")
        create_splits_from_plinder_dirs(plinder_dir, output_path, test_dir)
    elif args.data_dir:
        # Random split from processed directory
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        create_splits_from_processed_dir(
            data_dir, output_path, 
            args.val_ratio, args.test_ratio, args.seed
        )
    else:
        raise ValueError("Must specify either --plinder-dir or --data-dir")


if __name__ == '__main__':
    main()
