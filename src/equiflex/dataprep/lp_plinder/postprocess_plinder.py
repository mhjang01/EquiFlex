"""
Post-process PLINDER data.

1. Validates all required feature files exist
2. Checks for data quality issues (zero coords, embedding misalignment, etc.)
3. Generates statistics (ligand/pocket atom count histograms)
4. Moves invalid entries to error directory

Usage:
    # Dry run - show statistics and what would be moved
    python exec/lp_plinder/postprocess_plinder.py --data-dir ./plinder_data --dry-run

    # Actually move invalid entries
    python exec/lp_plinder/postprocess_plinder.py --data-dir ./plinder_data
"""

import argparse
import shutil
from pathlib import Path
from typing import Set, Tuple, List, Dict
import torch
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


def check_directory(system_dir: Path) -> Tuple[bool, str, Dict]:
    """
    Check if a directory is valid.

    Returns:
        (is_valid, reason, stats_dict)
    """
    system_id = system_dir.name
    stats = {'system_id': system_id, 'ligand_atoms': 0, 'pocket_atoms': 0}

    # Check for missing files
    missing_files = [f for f in REQUIRED_FILES if not (system_dir / f).exists()]
    if missing_files:
        return False, f"missing_files:{','.join(missing_files)}", stats

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
        lig_ref_coords = ligand_features.get('ref_coords')
        if lig_ref_coords is not None:
            if isinstance(lig_ref_coords, np.ndarray):
                lig_ref_coords = torch.from_numpy(lig_ref_coords)
            if lig_ref_coords.abs().sum() == 0:
                return False, "ligand_zero_ref_coords", stats

        # Check pocket ref_coords
        pocket_ref_coords = pocket_features.get('ref_coords')
        if pocket_ref_coords is not None:
            if isinstance(pocket_ref_coords, np.ndarray):
                pocket_ref_coords = torch.from_numpy(pocket_ref_coords)
            if pocket_ref_coords.abs().sum() == 0:
                return False, "pocket_zero_ref_coords", stats

        # Check ESM embedding alignment with residue_uid_vec
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

        # Check for missing atoms (coordinates at origin)
        pocket_missing_mask = pocket_features.get('missing_mask')
        if pocket_missing_mask is not None:
            if isinstance(pocket_missing_mask, np.ndarray):
                pocket_missing_mask = torch.from_numpy(pocket_missing_mask)
            n_missing = pocket_missing_mask.sum().item()
            if n_missing > 0:
                return False, f"pocket_missing_atoms:{n_missing}", stats

        # Check for zero-coordinate atoms
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


def plot_histograms(
    stats_list: List[Dict],
    output_dir: Path,
):
    """Plot histograms of ligand and pocket atom counts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid stats
    valid_stats = [s for s in stats_list if s['ligand_atoms'] > 0]

    if not valid_stats:
        print("No valid stats to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lig_atoms = [s['ligand_atoms'] for s in valid_stats]
    pock_atoms = [s['pocket_atoms'] for s in valid_stats]

    # Ligand atoms histogram
    ax1 = axes[0]
    ax1.hist(lig_atoms, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax1.axvline(x=MAX_LIGAND_ATOMS, color='r', linestyle='--', label=f'Max threshold ({MAX_LIGAND_ATOMS})')
    ax1.set_xlabel('Number of Ligand Atoms')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Ligand Atom Count Distribution (n={len(valid_stats)})')
    ax1.legend()
    ax1.text(0.95, 0.95, f'Mean: {np.mean(lig_atoms):.1f}\nMedian: {np.median(lig_atoms):.1f}\nMax: {max(lig_atoms)}',
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Pocket atoms histogram
    ax2 = axes[1]
    ax2.hist(pock_atoms, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Number of Pocket Atoms')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Pocket Atom Count Distribution (n={len(valid_stats)})')
    ax2.text(0.95, 0.95, f'Mean: {np.mean(pock_atoms):.1f}\nMedian: {np.median(pock_atoms):.1f}\nMax: {max(pock_atoms)}',
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'atom_count_histograms.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Post-process PLINDER data')
    parser.add_argument('--data-dir', type=str, default='./plinder_data',
                        help='Directory containing processed PLINDER data')
    parser.add_argument('--output-dir', type=str, default='./outputs/plinder_stats',
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

    # Find all system directories
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'errors']
    print(f"Found {len(all_dirs)} directories to check")

    # Check each directory
    valid_dirs = []
    invalid_dirs = []
    all_stats = []

    for system_dir in tqdm(all_dirs, desc="Checking directories"):
        is_valid, reason, stats = check_directory(system_dir)
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

    # Plot histograms
    plot_histograms(all_stats, output_dir)

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
