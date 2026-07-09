"""
Create train/val/test splits from LP-PDBBind CSV.

Uses the 'new_split' column from the CSV which contains pre-defined splits.
Outputs split files in multiple formats (JSON, TXT, YAML).

Usage:
    python scripts/lp_pdbbind/create_splits.py \
        --csv scripts/lp_pdbbind/LP_PDBBind.csv \
        --output-dir ./pdbbind/splits
"""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict


def create_splits(csv_path: str, output_dir: str, random_seed: int = 42):
    """
    Create train/val/test splits from LP-PDBBind CSV.

    The CSV has a 'new_split' column with values: 'train', 'val', 'test', or empty.
    Empty entries are assigned to train set.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded {len(df)} entries from {csv_path}")

    # Get PDB IDs and splits
    splits = defaultdict(list)

    for pdb_id, row in df.iterrows():
        split = row.get('new_split', '')

        # Handle empty or NaN split values
        if pd.isna(split) or split == '' or split not in ['train', 'val', 'test']:
            split = 'train'  # Default to train

        splits[split].append(pdb_id)

    # Print statistics
    print("\nSplit statistics:")
    print(f"  Train: {len(splits['train'])} systems")
    print(f"  Val:   {len(splits['val'])} systems")
    print(f"  Test:  {len(splits['test'])} systems")
    print(f"  Total: {sum(len(v) for v in splits.values())} systems")

    # Create split data structure
    split_data = {
        'train': sorted(splits['train']),
        'val': sorted(splits['val']),
        'test': sorted(splits['test']),
        'metadata': {
            'source': str(csv_path),
            'total_systems': len(df),
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
        }
    }

    # Save as JSON
    json_path = output_path / 'splits.json'
    with open(json_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save as YAML
    yaml_path = output_path / 'splits.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(split_data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved YAML: {yaml_path}")

    # Save as TXT files (one PDB ID per line)
    for split_name in ['train', 'val', 'test']:
        txt_path = output_path / f'split_{split_name}.txt'
        with open(txt_path, 'w') as f:
            for pdb_id in sorted(splits[split_name]):
                f.write(f"{pdb_id}\n")
        print(f"Saved TXT:  {txt_path}")

    # Create combined file with split labels
    all_path = output_path / 'all_systems.tsv'
    with open(all_path, 'w') as f:
        f.write("pdb_id\tsplit\n")
        for split_name in ['train', 'val', 'test']:
            for pdb_id in sorted(splits[split_name]):
                f.write(f"{pdb_id}\t{split_name}\n")
    print(f"Saved TSV:  {all_path}")

    # Additional info: save systems with binding affinity data
    affinity_data = []
    for pdb_id, row in df.iterrows():
        if pd.notna(row.get('value')):
            affinity_data.append({
                'pdb_id': pdb_id,
                'split': row.get('new_split', 'train') if pd.notna(row.get('new_split')) else 'train',
                'affinity_type': row.get('kd/ki', ''),
                'pKd_pKi': row.get('value', None),
                'category': row.get('category', ''),
            })

    affinity_df = pd.DataFrame(affinity_data)
    affinity_path = output_path / 'affinity_data.csv'
    affinity_df.to_csv(affinity_path, index=False)
    print(f"Saved affinity data: {affinity_path}")

    return split_data


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits from LP-PDBBind CSV')
    parser.add_argument('--csv', type=str, default='scripts/lp_pdbbind/LP_PDBBind.csv',
                        help='Path to LP-PDBBind CSV file')
    parser.add_argument('--output-dir', type=str, default='./pdbbind/splits',
                        help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (for reproducibility)')
    args = parser.parse_args()

    create_splits(args.csv, args.output_dir, args.seed)


if __name__ == '__main__':
    main()
