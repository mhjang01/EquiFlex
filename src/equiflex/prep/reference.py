from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
import numpy as np
from pathlib import Path
from typing import Union
from .const import AA_URL, STD_AA_ATOMS, NONSTD_AA_ATOMS
import requests
from .parse import get_rdkit_mol, extract_atom_data, extract_bond_data, extract_coordinates, get_sp2_torsion_indices, get_chiral_atom_indices, compute_mol_context

#### Amino Acid Reference ####

def download_aa_reference_mol(aa_code: str, out_path: Path = Path('./.refs/')):
    '''
    Downloads the reference mol for a given amino acid
    '''
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'{aa_code}.sdf'
    
    url = AA_URL[aa_code]
    response = requests.get(url)
    response.raise_for_status()
    with open(out_file, 'wb') as f:
        f.write(response.content)

# Constraint matrix generation

def generate_constraint_matrix(mol: Chem.Mol, len_atoms: int):
    '''
    Generate a distance-based boundary matrix for a molecule.
    
    Constists of:
    Upper triangle: Upper bound of distance between atoms
    Lower triangle: Lower bound of distance between atoms
    '''
    mat = GetMoleculeBoundsMatrix(mol, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True, useMacrocycle14config=False)
    mat = mat[:len_atoms, :len_atoms]
    return mat

# One-shot reference generation

def generate_aa_reference_data(aa_code: str):
    
    aa_path = Path(f'./.refs/{aa_code}.sdf')
    if not aa_path.exists():
        download_aa_reference_mol(aa_code)
        
    if aa_code not in STD_AA_ATOMS:
        aa_code = 'UNK'
        aa_atoms = NONSTD_AA_ATOMS[aa_code]
    else:
        aa_atoms = STD_AA_ATOMS[aa_code]
    
    # Extract atom data
    len_atoms = len(aa_atoms)
    
    mol = get_rdkit_mol(aa_path)
    mat = generate_constraint_matrix(mol, len_atoms)
    # No canonical reordering needed for amino acids
    coords = extract_coordinates(mol)[:len_atoms, :] # OXT + Hydrogen atoms are skipped
    mol_ctx = compute_mol_context(mol)

    atoms, bonds = [], []

    for idx, atom in enumerate(mol.GetAtoms()):
        if idx >= len_atoms: # Skip OXT + Hydrogen atoms
            break
        atom_name = aa_atoms[idx]
        atom_data = extract_atom_data(atom, name=atom_name, mol_ctx=mol_ctx)
        assert atom_data['symbol'].upper() == atom_name[0].upper(), f'Mismatching atom symbol: {atom_data["symbol"]} != {atom_name[0]}' # Just in case
        atoms.append(atom_data)
        
    for bond in mol.GetBonds():
        bond_data = extract_bond_data(bond) # Directed parsing for data efficiency; Will be extended from triu to full matrix
        if bond_data['begin_atom_idx'] >= len_atoms or bond_data['end_atom_idx'] >= len_atoms:
            continue
        bonds.append(bond_data)
    
    sp2_torsion_indices = get_sp2_torsion_indices(atoms, bonds)
        
    ref_data = {
        'aa_code': aa_code,
        'atoms': atoms,
        'bonds': bonds,
        'ref_coords': coords,
        'constraint_matrix': mat,
        'sp2_torsion_indices': sp2_torsion_indices,
    }
    
    np.savez_compressed(aa_path.with_suffix('.npz'), **ref_data)

#### Ligand Reference ####

def extract_ligand_dist_info(mol: Chem.Mol):
    '''
    Generate a 3D conformer for a molecule using RDKit.
    Returns None if embedding fails.
    '''
    num_atoms = mol.GetNumAtoms()

    # Add Hs for better conformer generation
    mol = Chem.AddHs(mol)

    # Try embedding with ETKDGv3
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    # EmbedMolecule returns -1 on failure
    if result == -1:
        # Retry with random coordinates as fallback
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if result == -1:
            print(f'Failed to generate reference conformer')
            return None

    # Check if conformer was generated
    if mol.GetNumConformers() == 0:
        return None

    # Extract coordinates for heavy atoms only (first num_atoms)
    conf = mol.GetConformer(0)
    coords = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = np.array([pos.x, pos.y, pos.z])
        
    mat = generate_constraint_matrix(mol, num_atoms)
    
    return coords, mat

def generate_ligand_reference_data(mol_or_smiles):
    '''
    Generate reference conformer and constraint matrix for a ligand.

    Accepts either an RDKit Mol or a SMILES string. When Mol is supplied, all
    derived features (constraint_matrix, ref_coords, atoms, bonds, sp2_torsion)
    are indexed in the SAME atom order as the input mol — critical so that
    cm[i,j] aligns with the bond/coord pairs (i,j) downstream.

    Passing a SMILES is preserved for legacy callers but is unsafe when the
    caller already has SDF-parsed atoms/bonds/coords in a different order
    (canonical-SMILES round-trip permutes atom indices).

    Returns None if mol parsing or conformer generation fails.
    '''
    if isinstance(mol_or_smiles, str):
        mol = get_rdkit_mol(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return None

    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i]+1) for i in range(mol.GetNumAtoms())}

    mol_ctx = compute_mol_context(mol)
    atoms = [extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}', mol_ctx=mol_ctx) for atom in mol.GetAtoms()]
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]
    coords, mat = extract_ligand_dist_info(mol)
    sp2_torsion_indices = get_sp2_torsion_indices(atoms, bonds)
    chiral_atom_indices = get_chiral_atom_indices(mol)

    if coords is None:
        return None

    ref_data = {
        'atoms': atoms,
        'bonds': bonds,
        'ref_coords': coords,
        'constraint_matrix': mat,
        'sp2_torsion_indices': sp2_torsion_indices,
        'chiral_atom_indices': chiral_atom_indices,
        'mapping': mapping,
    }

    return ref_data

def main():
    for aa_code in STD_AA_ATOMS.keys():
        generate_aa_reference_data(aa_code)
    for aa_code in NONSTD_AA_ATOMS.keys():
        generate_aa_reference_data(aa_code)

if __name__ == '__main__':
    main()