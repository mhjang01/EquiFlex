"""
Patch script to upgrade test_data feature files from 124-dim to 380-dim.

The old featurization was missing the 256-dim Boltz-style atom name encoding.
This script adds the encoding as a prefix to existing features.

Usage:
    python -m prep.patch_features --data-dir ./test_data
    python -m prep.patch_features --data-dir ./test_data --dry-run
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional

from .featurize import encode_atom_name_chars
from .const import STD_AA_ATOMS, NONSTD_AA_ATOMS


def get_ligand_atom_names_from_smiles(smiles: str) -> Optional[List[str]]:
    """
    Generate atom names from SMILES using canonical ranks.
    Names follow: {symbol}{canonical_rank+1}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    names = [
        f"{atom.GetSymbol().upper()}{canonical_ranks[atom.GetIdx()]+1}"
        for atom in mol.GetAtoms()
    ]
    return names


def get_pocket_atom_names(residue_keys: List[tuple], n_atoms_per_residue: List[int]) -> List[str]:
    """
    Get atom names for pocket residues from standard AA atom orders.
    residue_keys format: [(chain, resname, resnum, entity_id), ...]
    """
    atom_names = []

    for (chain, resname, resnum, entity_id), n_atoms in zip(residue_keys, n_atoms_per_residue):
        if resname in STD_AA_ATOMS:
            atom_order = STD_AA_ATOMS[resname]
        else:
            atom_order = NONSTD_AA_ATOMS.get('UNK', ['N', 'CA', 'C', 'O'])

        # Use first n_atoms from the atom order
        names = atom_order[:n_atoms]
        atom_names.extend(names)

    return atom_names


def add_name_encoding_to_features(
    features: torch.Tensor,
    atom_names: List[str]
) -> torch.Tensor:
    """
    Add 256-dim atom name encoding as prefix to existing 124-dim features.
    Returns 380-dim features.
    """
    n_atoms = features.shape[0]

    if len(atom_names) != n_atoms:
        raise ValueError(f"Atom count mismatch: {len(atom_names)} names vs {n_atoms} features")

    name_encodings = torch.stack([
        encode_atom_name_chars(name) for name in atom_names
    ])  # [n_atoms, 256]

    # Concatenate: [name_encoding (256), chemistry_features (124)] = 380
    new_features = torch.cat([name_encodings, features], dim=1)

    return new_features


def patch_ligand_features(system_dir: Path, dry_run: bool = False) -> tuple:
    """Patch ligand features for a single system."""
    features_path = system_dir / 'ligand_features.pt'

    if not features_path.exists():
        return False, "ligand_features.pt not found"

    data = torch.load(features_path, weights_only=False)

    # Check if already patched
    if data['atom_features'].shape[1] == 380:
        return True, "already 380-dim (skipped)"

    if data['atom_features'].shape[1] != 124:
        return False, f"unexpected dim: {data['atom_features'].shape[1]}"

    smiles_list = data['smiles_list']
    n_atoms_per_ligand = data['n_atoms_per_ligand']

    all_atom_names = []
    for smiles, n_atoms in zip(smiles_list, n_atoms_per_ligand):
        names = get_ligand_atom_names_from_smiles(smiles)
        if names is None:
            return False, f"failed to parse SMILES: {smiles}"
        if len(names) != n_atoms:
            return False, f"atom count mismatch for {smiles}: {len(names)} vs {n_atoms}"
        all_atom_names.extend(names)

    new_features = add_name_encoding_to_features(data['atom_features'], all_atom_names)

    if dry_run:
        return True, f"would patch: {data['atom_features'].shape[1]} -> {new_features.shape[1]}"

    data['atom_features'] = new_features
    torch.save(data, features_path)

    return True, f"patched: 124 -> {new_features.shape[1]}"


def patch_pocket_features(system_dir: Path, dry_run: bool = False) -> tuple:
    """Patch pocket features for a single system."""
    features_path = system_dir / 'pocket_features.pt'

    if not features_path.exists():
        return False, "pocket_features.pt not found"

    data = torch.load(features_path, weights_only=False)

    # Check if already patched
    if data['atom_features'].shape[1] == 380:
        return True, "already 380-dim (skipped)"

    if data['atom_features'].shape[1] != 124:
        return False, f"unexpected dim: {data['atom_features'].shape[1]}"

    residue_keys = data['residue_keys']
    n_atoms_per_residue = data['n_atoms_per_residue']

    atom_names = get_pocket_atom_names(residue_keys, n_atoms_per_residue)

    if len(atom_names) != data['atom_features'].shape[0]:
        return False, f"atom count mismatch: {len(atom_names)} vs {data['atom_features'].shape[0]}"

    new_features = add_name_encoding_to_features(data['atom_features'], atom_names)

    if dry_run:
        return True, f"would patch: {data['atom_features'].shape[1]} -> {new_features.shape[1]}"

    data['atom_features'] = new_features
    torch.save(data, features_path)

    return True, f"patched: 124 -> {new_features.shape[1]}"


def patch_data_directory(data_dir: Path, dry_run: bool = False):
    """Patch all systems in a data directory."""
    system_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    print(f"Found {len(system_dirs)} systems in {data_dir}")
    if dry_run:
        print("[DRY RUN] No files will be modified\n")

    ligand_results = {'success': 0, 'skipped': 0, 'failed': 0}
    pocket_results = {'success': 0, 'skipped': 0, 'failed': 0}

    for system_dir in tqdm(system_dirs, desc="Patching"):
        success, msg = patch_ligand_features(system_dir, dry_run)
        if success:
            if 'skipped' in msg:
                ligand_results['skipped'] += 1
            else:
                ligand_results['success'] += 1
        else:
            ligand_results['failed'] += 1
            print(f"  Ligand {system_dir.name}: {msg}")

        success, msg = patch_pocket_features(system_dir, dry_run)
        if success:
            if 'skipped' in msg:
                pocket_results['skipped'] += 1
            else:
                pocket_results['success'] += 1
        else:
            pocket_results['failed'] += 1
            print(f"  Pocket {system_dir.name}: {msg}")

    print(f"\nLigand features: {ligand_results}")
    print(f"Pocket features: {pocket_results}")


def main():
    parser = argparse.ArgumentParser(description='Patch feature files from 124-dim to 380-dim')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory to patch')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    patch_data_directory(data_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
