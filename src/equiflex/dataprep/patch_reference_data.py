'''
Patch script to add reference conformer coordinates and torsion indices to existing featurized data.

This patches:
1. ligand_features.pt:
   - 'ref_coords': RDKit-generated reference conformer
   - 'sp2_torsion_indices': planar torsion atom indices
   - 'torsion_ligand_idx': which ligand each torsion belongs to

2. pocket_features.pt:
   - 'ref_coords': from AA templates (falls back to ground truth if missing)
   - 'sp2_torsion_indices': planar torsion atom indices
   - 'torsion_residue_idx': which residue each torsion belongs to

The ligand reference conformer is generated using RDKit's ETKDGv3 conformer generator.
Torsion indices identify SP2-SP2 bonds that should remain planar.

Options:
    --move_errors: Move erroneous system directories to ./data/errors/
    --strict: Treat systems with failed ref_coords generation as errors (move to errors dir)
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import shutil
sys.path.insert(0, '/home/mhjang/workspace/flex')

import torch
import numpy as np
from joblib import load, Parallel, delayed
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from equiflex.prep.parse import get_sp2_torsion_indices, get_all_torsion_indices


def generate_ligand_ref_coords(smiles: str, num_atoms: int) -> Optional[np.ndarray]:
    '''
    Generate reference conformer coordinates for a ligand using RDKit.

    Args:
        smiles: SMILES string
        num_atoms: Expected number of heavy atoms

    Returns:
        coords: [num_atoms, 3] array or None if generation fails
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Verify atom count matches
        if mol.GetNumAtoms() != num_atoms:
            print(f"Atom count mismatch: SMILES has {mol.GetNumAtoms()}, expected {num_atoms}")
            return None

        # Add hydrogens for better conformer generation
        mol_h = Chem.AddHs(mol)

        # Generate conformer using ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # Reproducibility
        result = AllChem.EmbedMolecule(mol_h, params)

        if result == -1:
            # Fallback: try without ETKDGv3
            result = AllChem.EmbedMolecule(mol_h)
            if result == -1:
                return None

        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol_h)

        # Extract heavy atom coordinates (hydrogens come after heavy atoms)
        conf = mol_h.GetConformer(0)
        coords = np.zeros((num_atoms, 3), dtype=np.float32)
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

        return coords

    except Exception as e:
        print(f"Error generating conformer for {smiles}: {e}")
        return None


def compute_ligand_torsion_indices(
    atoms: List[Dict],
    bonds: List[Dict],
    n_atoms_per_ligand: List[int]
) -> Dict[str, torch.Tensor]:
    '''
    Compute torsion indices for all ligands.

    Returns dict with:
        - sp2_torsion_indices: [N_torsions, 4] - planar torsions (should be 0 or 180 deg)
        - all_torsion_indices: [N_torsions, 4] - all rotatable torsions
        - torsion_ligand_idx: [N_torsions] - which ligand each torsion belongs to
    '''
    all_sp2_torsions = []
    all_rotatable_torsions = []
    torsion_ligand_idx = []

    atom_offset = 0
    for lig_idx, n_atoms in enumerate(n_atoms_per_ligand):
        # Extract atoms and bonds for this ligand
        lig_atoms = atoms[atom_offset:atom_offset + n_atoms]

        # Filter bonds belonging to this ligand
        lig_bonds = []
        for bond in bonds:
            src, dst = bond['begin_atom_idx'], bond['end_atom_idx']
            if atom_offset <= src < atom_offset + n_atoms and atom_offset <= dst < atom_offset + n_atoms:
                # Adjust to local indices
                local_bond = bond.copy()
                local_bond['begin_atom_idx'] = src - atom_offset
                local_bond['end_atom_idx'] = dst - atom_offset
                lig_bonds.append(local_bond)

        # Compute torsion indices (local)
        sp2_torsions = get_sp2_torsion_indices(lig_atoms, lig_bonds)

        # Convert back to global indices
        if len(sp2_torsions) > 0:
            global_torsions = sp2_torsions + atom_offset
            all_sp2_torsions.append(global_torsions)
            torsion_ligand_idx.extend([lig_idx] * len(sp2_torsions))

        atom_offset += n_atoms

    result = {}
    if all_sp2_torsions:
        result['sp2_torsion_indices'] = torch.tensor(np.concatenate(all_sp2_torsions), dtype=torch.long)
        result['torsion_ligand_idx'] = torch.tensor(torsion_ligand_idx, dtype=torch.long)
    else:
        result['sp2_torsion_indices'] = torch.empty((0, 4), dtype=torch.long)
        result['torsion_ligand_idx'] = torch.empty((0,), dtype=torch.long)

    return result


def compute_pocket_ref_coords_and_torsions(
    pocket_features: Dict,
    pocket_residues_data: Dict
) -> Dict[str, torch.Tensor]:
    '''
    Compute reference coordinates and torsion indices for pocket residues.

    Reference coords come from AA templates (stored in res_data['ref_coords']).
    If not available, falls back to ground truth coords.
    '''
    all_ref_coords = []
    all_sp2_torsions = []
    torsion_residue_idx = []

    atom_offset = 0
    residue_idx = 0

    for chain_id, residues in sorted(pocket_residues_data.items()):
        for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
            atoms = res_data['atoms']
            bonds = res_data['bonds']
            n_atoms = len(atoms)

            # Reference coordinates from AA template
            if 'ref_coords' in res_data and res_data['ref_coords'] is not None:
                ref_coords = np.array(res_data['ref_coords'], dtype=np.float32)
            else:
                # Fallback: use ground truth coords from res_data
                ref_coords = np.array(res_data['coords'], dtype=np.float32)
            all_ref_coords.append(ref_coords)

            # Compute local torsion indices
            sp2_torsions = get_sp2_torsion_indices(atoms, bonds)

            if len(sp2_torsions) > 0:
                global_torsions = sp2_torsions + atom_offset
                all_sp2_torsions.append(global_torsions)
                torsion_residue_idx.extend([residue_idx] * len(sp2_torsions))

            atom_offset += n_atoms
            residue_idx += 1

    result = {}

    # Reference coordinates
    if all_ref_coords:
        result['ref_coords'] = torch.tensor(np.concatenate(all_ref_coords), dtype=torch.float32)

    # Torsion indices
    if all_sp2_torsions:
        result['sp2_torsion_indices'] = torch.tensor(np.concatenate(all_sp2_torsions), dtype=torch.long)
        result['torsion_residue_idx'] = torch.tensor(torsion_residue_idx, dtype=torch.long)
    else:
        result['sp2_torsion_indices'] = torch.empty((0, 4), dtype=torch.long)
        result['torsion_residue_idx'] = torch.empty((0,), dtype=torch.long)

    return result


def patch_single_system(system_dir: Path, overwrite: bool = False, strict: bool = False) -> Tuple[str, bool, str]:
    '''
    Patch a single system with reference coordinates and torsion indices.

    Returns:
        Tuple of (system_name, success, message)
        - success=False with "missing required files" → missing input files
        - success=False with "error: ..." → processing error
        - success=False with "ref_coords_failed" → ref coords generation failed (strict mode)
        - success=True with "patched" → successfully patched
        - success=True with "already patched" → no changes needed
    '''
    system_name = system_dir.name

    ligand_path = system_dir / 'ligand_features.pt'
    pocket_path = system_dir / 'pocket_features.pt'
    ligands_pkl = system_dir / 'ligands.pkl'
    pocket_pkl = system_dir / 'pocket_residues.pkl'

    if not all(p.exists() for p in [ligand_path, pocket_path, ligands_pkl]):
        return system_name, False, "missing required files"

    try:
        # Load existing data
        ligand_features = torch.load(ligand_path, weights_only=False)
        pocket_features = torch.load(pocket_path, weights_only=False)
        ligands = load(ligands_pkl)

        modified = False
        ref_coords_failed = False

        # === Patch ligand features ===
        if 'ref_coords' not in ligand_features or overwrite:
            # Generate reference conformers for each ligand
            all_ref_coords = []

            for lig_data in ligands:
                smiles = lig_data['smiles']
                n_atoms = len(lig_data['atoms'])

                ref_coords = generate_ligand_ref_coords(smiles, n_atoms)
                if ref_coords is None:
                    # Fallback: use zero coords (will need special handling during training)
                    ref_coords = np.zeros((n_atoms, 3), dtype=np.float32)
                    ref_coords_failed = True

                all_ref_coords.append(ref_coords)

            ligand_features['ref_coords'] = torch.tensor(
                np.concatenate(all_ref_coords), dtype=torch.float32
            )
            modified = True

            # In strict mode, treat ref_coords failures as errors
            if strict and ref_coords_failed:
                return system_name, False, "ref_coords_failed"

        if 'sp2_torsion_indices' not in ligand_features or overwrite:
            # Reconstruct atoms/bonds structure for torsion computation
            all_atoms = []
            all_bonds = []
            atom_offset = 0

            for lig_data in ligands:
                all_atoms.extend(lig_data['atoms'])
                for bond in lig_data['bonds']:
                    global_bond = bond.copy()
                    global_bond['begin_atom_idx'] += atom_offset
                    global_bond['end_atom_idx'] += atom_offset
                    all_bonds.append(global_bond)
                atom_offset += len(lig_data['atoms'])

            torsion_data = compute_ligand_torsion_indices(
                all_atoms, all_bonds, ligand_features['n_atoms_per_ligand']
            )
            ligand_features.update(torsion_data)
            modified = True

        # === Patch pocket features ===
        needs_pocket_patch = (
            'ref_coords' not in pocket_features or
            'sp2_torsion_indices' not in pocket_features or
            overwrite
        )
        if needs_pocket_patch:
            if pocket_pkl.exists():
                pocket_residues = load(pocket_pkl)
                pocket_data = compute_pocket_ref_coords_and_torsions(pocket_features, pocket_residues)
                pocket_features.update(pocket_data)
                modified = True

        # Save if modified
        if modified:
            torch.save(ligand_features, ligand_path)
            torch.save(pocket_features, pocket_path)
            return system_name, True, "patched"
        else:
            return system_name, True, "already patched"

    except Exception as e:
        return system_name, False, f"error: {e}"


def main(
    data_dir: str = '/home/mhjang/workspace/flex/data',
    n_jobs: int = 1,  # RDKit conformer gen can be memory-heavy, use fewer jobs
    overwrite: bool = False,
    move_errors: bool = False,
    strict: bool = False,
):
    data_path = Path(data_dir)
    errors_dir = data_path / 'errors'

    # Find all systems with ligand_features.pt
    system_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
    ])

    print(f"Found {len(system_dirs)} systems to patch")
    print(f"Overwrite mode: {overwrite}")
    print(f"Strict mode: {strict}")
    print(f"Move errors: {move_errors}")

    if n_jobs == 1:
        results = []
        for d in tqdm(system_dirs, desc="Patching"):
            results.append(patch_single_system(d, overwrite, strict))
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(patch_single_system)(d, overwrite, strict)
            for d in tqdm(system_dirs, desc="Patching")
        )

    # Summary
    success = sum(1 for _, s, m in results if s and 'patched' in m)
    skipped = sum(1 for _, s, m in results if s and 'already' in m)
    failed = sum(1 for _, s, _ in results if not s)

    print(f"\nComplete: Patched={success}, Already done={skipped}, Failed={failed}")

    # Collect failures
    failures = [(n, m) for n, s, m in results if not s]

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for name, msg in failures[:10]:
            print(f"  {name}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

        # Write error log
        with open(data_path / 'patch_errors.txt', 'w') as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")

        # Move erroneous systems if requested
        if move_errors:
            errors_dir.mkdir(parents=True, exist_ok=True)
            moved_count = 0

            for name, msg in failures:
                src_dir = data_path / name
                dst_dir = errors_dir / name

                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.move(str(src_dir), str(dst_dir))
                    moved_count += 1

            print(f"\nMoved {moved_count} erroneous systems to {errors_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Patch featurized data with reference conformers and torsion indices')
    parser.add_argument('--data_dir', type=str, default='/home/mhjang/workspace/flex/data')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default 1 due to RDKit memory usage)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing ref_coords and torsion indices')
    parser.add_argument('--move_errors', action='store_true', help='Move erroneous system directories to ./data/errors/')
    parser.add_argument('--strict', action='store_true', help='Treat systems with failed ref_coords generation as errors')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
        move_errors=args.move_errors,
        strict=args.strict,
    )
