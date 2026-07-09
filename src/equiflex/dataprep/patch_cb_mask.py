"""
Patch existing pocket_features.pt files to add cb_mask and backbone_atom_indices.

Adds:
- cb_mask: [N_atoms] float, 1.0 for CB atoms, 0.0 otherwise
- backbone_atom_indices: dict with per-residue N/CA/C indices and GLY flag

Usage:
    python exec/patch_cb_mask.py --data-dir pdbbind_data
    python exec/patch_cb_mask.py --data-dir plinder_data
    python exec/patch_cb_mask.py --data-dir pdbbind_data --dry-run
"""

import argparse
from pathlib import Path
from joblib import load
import torch
from tqdm import tqdm

CB_ATOM_NAME = "CB"


def patch_system(system_dir: Path, dry_run: bool = False) -> tuple:
    system_name = system_dir.name
    pocket_features_path = system_dir / 'pocket_features.pt'
    pocket_residues_path = system_dir / 'pocket_residues.pkl'

    if not pocket_features_path.exists():
        return system_name, False, "pocket_features.pt not found", 0, 0

    if not pocket_residues_path.exists():
        return system_name, False, "pocket_residues.pkl not found", 0, 0

    try:
        pocket_features = torch.load(pocket_features_path, weights_only=False)

        if 'cb_mask' in pocket_features and 'backbone_atom_indices' in pocket_features:
            n_cb = int(pocket_features['cb_mask'].sum().item())
            n_total = len(pocket_features['cb_mask'])
            return system_name, True, "already patched", n_cb, n_total

        pocket_residues = load(pocket_residues_path)

        atom_names = []
        atom_offset = 0
        residue_n_idx = []
        residue_ca_idx = []
        residue_c_idx = []
        residue_is_gly = []

        for chain_id, residues in sorted(pocket_residues.items()):
            if chain_id in ('', '_'):
                continue
            for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
                res_atom_names = [atom['name'] for atom in res_data['atoms']]
                atom_names.extend(res_atom_names)

                residue_n_idx.append(atom_offset + res_atom_names.index('N') if 'N' in res_atom_names else -1)
                residue_ca_idx.append(atom_offset + res_atom_names.index('CA') if 'CA' in res_atom_names else -1)
                residue_c_idx.append(atom_offset + res_atom_names.index('C') if 'C' in res_atom_names else -1)
                residue_is_gly.append(res_data.get('aa_code', resname) == 'GLY')

                atom_offset += len(res_atom_names)

        n_atoms = pocket_features['atom_features'].shape[0]
        if len(atom_names) != n_atoms:
            return system_name, False, f"atom count mismatch: {len(atom_names)} vs {n_atoms}", 0, 0

        cb_mask = torch.tensor(
            [1.0 if name == CB_ATOM_NAME else 0.0 for name in atom_names],
            dtype=torch.float32,
        )

        backbone_atom_indices = {
            'n_idx': torch.tensor(residue_n_idx, dtype=torch.long),
            'ca_idx': torch.tensor(residue_ca_idx, dtype=torch.long),
            'c_idx': torch.tensor(residue_c_idx, dtype=torch.long),
            'is_gly': torch.tensor(residue_is_gly, dtype=torch.bool),
        }

        n_cb = int(cb_mask.sum().item())
        n_total = len(cb_mask)

        if not dry_run:
            pocket_features['cb_mask'] = cb_mask
            pocket_features['backbone_atom_indices'] = backbone_atom_indices
            torch.save(pocket_features, pocket_features_path)

        return system_name, True, "patched", n_cb, n_total

    except Exception as e:
        return system_name, False, f"error: {e}", 0, 0


def main():
    parser = argparse.ArgumentParser(description='Patch pocket_features.pt with cb_mask and backbone_atom_indices')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--dry-run', action='store_true', help='Check without saving')
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    system_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name != 'errors' and (d / 'pocket_features.pt').exists()
    ])

    print(f"Found {len(system_dirs)} systems to patch")
    if args.dry_run:
        print("DRY RUN - no files will be modified")

    success_count = 0
    already_patched = 0
    error_count = 0
    total_cb = 0
    total_atoms = 0

    for system_dir in tqdm(system_dirs, desc="Patching"):
        name, success, msg, n_cb, n_total = patch_system(system_dir, args.dry_run)

        if success:
            if "already" in msg:
                already_patched += 1
            else:
                success_count += 1
            total_cb += n_cb
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
        print(f"\nCB statistics:")
        print(f"  Total atoms: {total_atoms:,}")
        print(f"  CB atoms: {total_cb:,}")
        print(f"  CB ratio: {total_cb / total_atoms * 100:.1f}%")


if __name__ == '__main__':
    main()
