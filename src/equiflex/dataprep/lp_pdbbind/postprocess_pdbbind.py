"""
Post-process PDBBind data.

1. Validates all required feature files exist
2. Excludes PDB IDs in Astex Diverse Set or PoseBusters benchmark
3. Generates statistics (ligand/pocket atom count histograms per split)
4. Moves invalid entries to error directory

Usage:
    # Dry run - show statistics and what would be moved
    python scripts/lp_pdbbind/postprocess_pdbbind.py --data-dir ./pdbbind_data --dry-run

    # Actually move invalid entries
    python scripts/lp_pdbbind/postprocess_pdbbind.py --data-dir ./pdbbind_data
"""

import argparse
import shutil
from pathlib import Path
from typing import Set, Tuple, List, Dict
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


REQUIRED_FILES = [
    'ligand_features.pt',
    'pocket_features.pt',
    'esm3_embeddings.pt',
    'unimol_embeddings.pt',
]

MAX_LIGAND_ATOMS = 100


def load_split_mapping(csv_path: Path) -> Dict[str, str]:
    """Load PDB ID to split mapping from LP_PDBBind CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    mapping = {}
    for pdb_id, row in df.iterrows():
        split = row.get('new_split', '')
        if pd.notna(split) and split:
            mapping[str(pdb_id).lower()] = split
    return mapping


def load_benchmark_pdb_ids(csv_path: Path) -> Set[str]:
    """Load PDB IDs from Astex Diverse Set and PoseBusters benchmark."""
    df = pd.read_csv(csv_path)
    benchmark_df = df[df['dataset'].isin(['astex', 'posebuster'])]
    pdb_ids = set(benchmark_df['pdb_id'].str.lower().unique())
    return pdb_ids


def check_directory(
    system_dir: Path,
    benchmark_pdb_ids: Set[str],
) -> Tuple[bool, str, Dict]:
    """
    Check if a directory is valid.

    Returns:
        (is_valid, reason, stats_dict)
    """
    pdb_id = system_dir.name.lower()
    stats = {'pdb_id': pdb_id, 'ligand_atoms': 0, 'pocket_atoms': 0}

    # Check for missing files
    missing_files = [f for f in REQUIRED_FILES if not (system_dir / f).exists()]
    if missing_files:
        return False, f"missing_files:{','.join(missing_files)}", stats

    # Check if in benchmark set
    if pdb_id in benchmark_pdb_ids:
        return False, "in_benchmark", stats

    # Load features and check atom counts
    try:
        ligand_features = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
        pocket_features = torch.load(system_dir / 'pocket_features.pt', weights_only=False)

        # Get atom counts
        if isinstance(ligand_features, dict):
            if 'atom_features' in ligand_features:
                ligand_atoms = ligand_features['atom_features'].shape[0]
            elif 'coords' in ligand_features:
                ligand_atoms = ligand_features['coords'].shape[0]
            else:
                for k, v in ligand_features.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        ligand_atoms = v.shape[0]
                        break
                else:
                    ligand_atoms = 0
        else:
            ligand_atoms = ligand_features.shape[0] if hasattr(ligand_features, 'shape') else 0

        if isinstance(pocket_features, dict):
            if 'atom_features' in pocket_features:
                pocket_atoms = pocket_features['atom_features'].shape[0]
            elif 'coords' in pocket_features:
                pocket_atoms = pocket_features['coords'].shape[0]
            else:
                for k, v in pocket_features.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        pocket_atoms = v.shape[0]
                        break
                else:
                    pocket_atoms = 0
        else:
            pocket_atoms = pocket_features.shape[0] if hasattr(pocket_features, 'shape') else 0

        stats['ligand_atoms'] = ligand_atoms
        stats['pocket_atoms'] = pocket_atoms

        # Check ligand atom count
        if ligand_atoms > MAX_LIGAND_ATOMS:
            return False, f"ligand_too_large:{ligand_atoms}", stats

        if ligand_atoms == 0:
            return False, "no_ligand_atoms", stats

        if pocket_atoms == 0:
            return False, "no_pocket_atoms", stats

        # Check for all-zero ref_coords (RDKit failed to generate conformer)
        # This causes CUDA errors in torch.cdist
        lig_ref_coords = ligand_features.get('ref_coords')
        if lig_ref_coords is not None:
            if isinstance(lig_ref_coords, np.ndarray):
                lig_ref_coords = torch.from_numpy(lig_ref_coords)
            if lig_ref_coords.abs().sum() == 0:
                return False, "ligand_zero_ref_coords", stats

        # Check pocket ref_coords for all-zero (same issue)
        pocket_ref_coords = pocket_features.get('ref_coords')
        if pocket_ref_coords is not None:
            if isinstance(pocket_ref_coords, np.ndarray):
                pocket_ref_coords = torch.from_numpy(pocket_ref_coords)
            if pocket_ref_coords.abs().sum() == 0:
                return False, "pocket_zero_ref_coords", stats

        # Check ESM embedding alignment with residue_uid_vec
        # This prevents scatter/gather index out of bounds errors
        esm_embeddings = torch.load(system_dir / 'esm3_embeddings.pt', weights_only=False)
        rec_uid_vec = pocket_features.get('residue_uid_vec')
        if rec_uid_vec is not None and esm_embeddings is not None:
            if isinstance(rec_uid_vec, np.ndarray):
                rec_uid_vec = torch.from_numpy(rec_uid_vec)
            n_esm_tokens = esm_embeddings.shape[0]
            max_uid = rec_uid_vec.max().item() if rec_uid_vec.numel() > 0 else -1
            if max_uid >= n_esm_tokens:
                return False, f"uid_out_of_bounds:max_uid={max_uid},n_esm={n_esm_tokens}", stats

        # Check ESM embedding dimension (should be 1536)
        if esm_embeddings is not None and esm_embeddings.shape[-1] != 1536:
            return False, f"esm_wrong_dim:{esm_embeddings.shape[-1]}", stats

        # Check UniMol embedding dimension (should be 1536)
        unimol_embeddings = torch.load(system_dir / 'unimol_embeddings.pt', weights_only=False)
        if unimol_embeddings is not None and unimol_embeddings.shape[-1] != 1536:
            return False, f"unimol_wrong_dim:{unimol_embeddings.shape[-1]}", stats

        # Check for missing atoms (coordinates at origin cause geometry issues)
        # missing_mask: True = missing, False = present
        pocket_missing_mask = pocket_features.get('missing_mask')
        if pocket_missing_mask is not None:
            if isinstance(pocket_missing_mask, np.ndarray):
                pocket_missing_mask = torch.from_numpy(pocket_missing_mask)
            n_missing = pocket_missing_mask.sum().item()
            if n_missing > 0:
                return False, f"pocket_missing_atoms:{n_missing}", stats

        # Also check for zero-coordinate atoms even without missing_mask
        pocket_coords = pocket_features.get('coords')
        if pocket_coords is not None:
            if isinstance(pocket_coords, np.ndarray):
                pocket_coords = torch.from_numpy(pocket_coords)
            zero_coord_mask = (pocket_coords.abs().sum(dim=-1) == 0)
            n_zero = zero_coord_mask.sum().item()
            if n_zero > 0:
                return False, f"pocket_zero_coords:{n_zero}", stats

        lig_coords = ligand_features.get('coords')
        if lig_coords is not None:
            if isinstance(lig_coords, np.ndarray):
                lig_coords = torch.from_numpy(lig_coords)
            zero_coord_mask = (lig_coords.abs().sum(dim=-1) == 0)
            n_zero = zero_coord_mask.sum().item()
            if n_zero > 0:
                return False, f"ligand_zero_coords:{n_zero}", stats

    except Exception as e:
        return False, f"load_error:{str(e)[:50]}", stats

    return True, "valid", stats


def plot_histograms_by_split(
    stats_list: List[Dict],
    split_mapping: Dict[str, str],
    output_dir: Path,
):
    """Plot histograms of ligand and pocket atom counts per split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group stats by split
    split_stats = {'train': [], 'val': [], 'test': [], 'unknown': []}
    for s in stats_list:
        if s['ligand_atoms'] > 0:
            split = split_mapping.get(s['pdb_id'], 'unknown')
            if split in split_stats:
                split_stats[split].append(s)
            else:
                split_stats['unknown'].append(s)

    # Plot combined histogram with all splits
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'train': 'blue', 'val': 'orange', 'test': 'green', 'unknown': 'gray'}

    # Top row: Ligand atoms
    ax1 = axes[0, 0]
    for split in ['train', 'val', 'test']:
        if split_stats[split]:
            atoms = [s['ligand_atoms'] for s in split_stats[split]]
            ax1.hist(atoms, bins=50, alpha=0.5, label=f'{split} (n={len(atoms)})', color=colors[split])
    ax1.axvline(x=MAX_LIGAND_ATOMS, color='r', linestyle='--', label=f'Max threshold ({MAX_LIGAND_ATOMS})')
    ax1.set_xlabel('Number of Ligand Atoms')
    ax1.set_ylabel('Count')
    ax1.set_title('Ligand Atom Count Distribution (All Splits)')
    ax1.legend()

    # Top right: Pocket atoms
    ax2 = axes[0, 1]
    for split in ['train', 'val', 'test']:
        if split_stats[split]:
            atoms = [s['pocket_atoms'] for s in split_stats[split]]
            ax2.hist(atoms, bins=50, alpha=0.5, label=f'{split} (n={len(atoms)})', color=colors[split])
    ax2.set_xlabel('Number of Pocket Atoms')
    ax2.set_ylabel('Count')
    ax2.set_title('Pocket Atom Count Distribution (All Splits)')
    ax2.legend()

    # Bottom row: Statistics table
    ax3 = axes[1, 0]
    ax3.axis('off')
    table_data = []
    for split in ['train', 'val', 'test']:
        if split_stats[split]:
            lig_atoms = [s['ligand_atoms'] for s in split_stats[split]]
            pock_atoms = [s['pocket_atoms'] for s in split_stats[split]]
            table_data.append([
                split,
                len(split_stats[split]),
                f'{np.mean(lig_atoms):.1f}',
                f'{np.median(lig_atoms):.1f}',
                f'{max(lig_atoms)}',
                f'{np.mean(pock_atoms):.1f}',
                f'{np.median(pock_atoms):.1f}',
                f'{max(pock_atoms)}',
            ])

    if table_data:
        table = ax3.table(
            cellText=table_data,
            colLabels=['Split', 'Count', 'Lig Mean', 'Lig Med', 'Lig Max', 'Pock Mean', 'Pock Med', 'Pock Max'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Statistics by Split', y=0.8)

    # Bottom right: Count by threshold
    ax4 = axes[1, 1]
    ax4.axis('off')

    threshold_data = []
    for threshold in [50, 75, 100, 125, 150]:
        row = [f'≤{threshold}']
        for split in ['train', 'val', 'test']:
            if split_stats[split]:
                count = sum(1 for s in split_stats[split] if s['ligand_atoms'] <= threshold)
                total = len(split_stats[split])
                row.append(f'{count} ({100*count/total:.1f}%)')
            else:
                row.append('-')
        threshold_data.append(row)

    if threshold_data:
        table2 = ax4.table(
            cellText=threshold_data,
            colLabels=['Ligand Atoms', 'Train', 'Val', 'Test'],
            loc='center',
            cellLoc='center',
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.5)
        ax4.set_title('Samples by Ligand Atom Threshold', y=0.8)

    plt.tight_layout()
    output_path = output_dir / 'atom_count_histograms_by_split.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")

    # Also plot individual histograms per split
    for split in ['train', 'val', 'test']:
        if not split_stats[split]:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        lig_atoms = [s['ligand_atoms'] for s in split_stats[split]]
        pock_atoms = [s['pocket_atoms'] for s in split_stats[split]]

        ax1 = axes[0]
        ax1.hist(lig_atoms, bins=50, edgecolor='black', alpha=0.7, color=colors[split])
        ax1.axvline(x=MAX_LIGAND_ATOMS, color='r', linestyle='--', label=f'Max threshold ({MAX_LIGAND_ATOMS})')
        ax1.set_xlabel('Number of Ligand Atoms')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Ligand Atom Count - {split.upper()} (n={len(lig_atoms)})')
        ax1.legend()
        ax1.text(0.95, 0.95, f'Mean: {np.mean(lig_atoms):.1f}\nMedian: {np.median(lig_atoms):.1f}\nMax: {max(lig_atoms)}',
                 transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2 = axes[1]
        ax2.hist(pock_atoms, bins=50, edgecolor='black', alpha=0.7, color=colors[split])
        ax2.set_xlabel('Number of Pocket Atoms')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Pocket Atom Count - {split.upper()} (n={len(pock_atoms)})')
        ax2.text(0.95, 0.95, f'Mean: {np.mean(pock_atoms):.1f}\nMedian: {np.median(pock_atoms):.1f}\nMax: {max(pock_atoms)}',
                 transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = output_dir / f'atom_count_histogram_{split}.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved {split} histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Post-process PDBBind data')
    parser.add_argument('--data-dir', type=str, default='./pdbbind_data',
                        help='Directory containing processed PDBBind data')
    parser.add_argument('--benchmark-csv', type=str,
                        default='prep/posebusters_paper_results.csv',
                        help='Path to PoseBusters benchmark CSV')
    parser.add_argument('--split-csv', type=str,
                        default='exec/lp_pdbbind/LP_PDBBind.csv',
                        help='Path to LP_PDBBind CSV with split info')
    parser.add_argument('--output-dir', type=str, default='./outputs/pdbbind_stats',
                        help='Directory to save histogram plots')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show statistics, do not move files')
    parser.add_argument('--max-ligand-atoms', type=int, default=100,
                        help='Maximum allowed ligand atoms')
    args = parser.parse_args()

    global MAX_LIGAND_ATOMS
    MAX_LIGAND_ATOMS = args.max_ligand_atoms

    data_dir = Path(args.data_dir)
    error_dir = data_dir / 'errors'
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load benchmark PDB IDs
    benchmark_csv = Path(args.benchmark_csv)
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_csv}")

    benchmark_pdb_ids = load_benchmark_pdb_ids(benchmark_csv)
    print(f"Loaded {len(benchmark_pdb_ids)} benchmark PDB IDs (Astex + PoseBusters)")

    # Load split mapping
    split_csv = Path(args.split_csv)
    if split_csv.exists():
        split_mapping = load_split_mapping(split_csv)
        print(f"Loaded split mapping for {len(split_mapping)} PDB IDs")
    else:
        print(f"Warning: Split CSV not found: {split_csv}")
        split_mapping = {}

    # Find all system directories
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'errors']
    print(f"Found {len(all_dirs)} directories to check")

    # Check each directory
    valid_dirs = []
    invalid_dirs = []
    all_stats = []

    for system_dir in tqdm(all_dirs, desc="Checking directories"):
        is_valid, reason, stats = check_directory(system_dir, benchmark_pdb_ids)
        all_stats.append(stats)

        if is_valid:
            valid_dirs.append(system_dir)
        else:
            invalid_dirs.append((system_dir, reason))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total directories: {len(all_dirs)}")
    print(f"Valid directories: {len(valid_dirs)}")
    print(f"Invalid directories: {len(invalid_dirs)}")

    # Breakdown of invalid reasons
    reason_counts = {}
    for _, reason in invalid_dirs:
        base_reason = reason.split(':')[0]
        reason_counts[base_reason] = reason_counts.get(base_reason, 0) + 1

    print("\nInvalid breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Plot histograms by split
    plot_histograms_by_split(all_stats, split_mapping, output_dir)

    # Show some examples
    print("\nExamples of invalid directories:")
    for system_dir, reason in invalid_dirs[:10]:
        print(f"  {system_dir.name}: {reason}")
    if len(invalid_dirs) > 10:
        print(f"  ... and {len(invalid_dirs) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files moved. Run without --dry-run to move invalid directories.")
        return

    # Move invalid directories
    print(f"\nMoving {len(invalid_dirs)} invalid directories to {error_dir}...")
    error_dir.mkdir(exist_ok=True)

    for system_dir, reason in tqdm(invalid_dirs, desc="Moving"):
        try:
            dest = error_dir / system_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(system_dir), str(dest))
        except Exception as e:
            print(f"Failed to move {system_dir}: {e}")

    print(f"\nDone. {len(valid_dirs)} valid directories remain in {data_dir}")


if __name__ == '__main__':
    main()
