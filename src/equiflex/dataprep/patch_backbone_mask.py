"""
Patch existing pocket_features.pt files to add backbone_mask tensor.

This script:
1. Loads pocket_residues.pkl to get atom names
2. Computes backbone_mask (1 for N, CA, C, O; 0 for sidechain)
3. Adds backbone_mask to pocket_features.pt

Usage:
    python exec/patch_backbone_mask.py --data-dir data
    python exec/patch_backbone_mask.py --data-dir data --dry-run  # Check without saving
"""

import argparse
from pathlib import Path
from joblib import load
import torch
from tqdm import tqdm

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


def patch_system(system_dir: Path, dry_run: bool = False) -> tuple:
    """
    Patch a single system's pocket_features.pt with backbone_mask.

    Returns:
        (system_name, success, message, n_backbone, n_total)
    """
    system_name = system_dir.name
    pocket_features_path = system_dir / 'pocket_features.pt'
    pocket_residues_path = system_dir / 'pocket_residues.pkl'

    if not pocket_features_path.exists():
        return system_name, False, "pocket_features.pt not found", 0, 0

    if not pocket_residues_path.exists():
        return system_name, False, "pocket_residues.pkl not found", 0, 0

    try:
        # Load existing features
        pocket_features = torch.load(pocket_features_path, weights_only=False)

        # Check if already patched
        if 'backbone_mask' in pocket_features:
            n_backbone = int(pocket_features['backbone_mask'].sum().item())
            n_total = len(pocket_features['backbone_mask'])
            return system_name, True, "already patched", n_backbone, n_total

        # Load pocket residues to get atom names
        pocket_residues = load(pocket_residues_path)

        # Collect atom names in the same order as process_pocket_residues
        atom_names = []
        for chain_id, residues in sorted(pocket_residues.items()):
            for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
                for atom in res_data['atoms']:
                    atom_names.append(atom['name'])

        # Verify length matches
        n_atoms = pocket_features['atom_features'].shape[0]
        if len(atom_names) != n_atoms:
            return system_name, False, f"atom count mismatch: {len(atom_names)} vs {n_atoms}", 0, 0

        # Create backbone mask
        backbone_mask = torch.tensor(
            [1.0 if name in BACKBONE_ATOMS else 0.0 for name in atom_names],
            dtype=torch.float32
        )

        n_backbone = int(backbone_mask.sum().item())
        n_total = len(backbone_mask)

        if not dry_run:
            pocket_features['backbone_mask'] = backbone_mask
            torch.save(pocket_features, pocket_features_path)

        return system_name, True, "patched", n_backbone, n_total

    except Exception as e:
        return system_name, False, f"error: {e}", 0, 0


def main():
    parser = argparse.ArgumentParser(description='Patch pocket_features.pt with backbone_mask')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dry-run', action='store_true', help='Check without saving')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    # Find all system directories with pocket_features.pt
    system_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d / 'pocket_features.pt').exists()
    ])

    print(f"Found {len(system_dirs)} systems to patch")
    if args.dry_run:
        print("DRY RUN - no files will be modified")

    success_count = 0
    already_patched = 0
    error_count = 0
    total_backbone = 0
    total_atoms = 0

    for system_dir in tqdm(system_dirs, desc="Patching"):
        name, success, msg, n_bb, n_total = patch_system(system_dir, args.dry_run)

        if success:
            if "already" in msg:
                already_patched += 1
            else:
                success_count += 1
            total_backbone += n_bb
            total_atoms += n_total
        else:
            error_count += 1
            if error_count <= 5:
                print(f"  Error: {name}: {msg}")

    print(f"\nResults:")
    print(f"  Patched: {success_count}")
    print(f"  Already patched: {already_patched}")
    print(f"  Errors: {error_count}")

    if total_atoms > 0:
        print(f"\nBackbone statistics:")
        print(f"  Total atoms: {total_atoms:,}")
        print(f"  Backbone atoms: {total_backbone:,}")
        print(f"  Backbone ratio: {total_backbone / total_atoms * 100:.1f}%")


if __name__ == '__main__':
    main()
