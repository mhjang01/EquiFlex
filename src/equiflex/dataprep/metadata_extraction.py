from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')
from equiflex.prep.parse import *
from equiflex.prep.reference import generate_ligand_reference_data
from tqdm import tqdm
from joblib import Parallel, delayed, dump
import plinder
import numpy as np
import pandas as pd

def get_test_data_ids(test_data_path: Path):
    df = pd.read_csv(test_data_path)
    test_ids = df['pdb_id'].unique()
    lowercase_ids = set(test_id.lower() for test_id in test_ids)
    
    return list(lowercase_ids)
    

def get_plinder_systems():
    from plinder.core.scores import query_index
    columns=[
        "system_id",
        "entry_pdb_id"
    ]

    filters = [
        ("system_num_ligand_chains", ">", 0),
        ("system_num_protein_chains", ">", 0),
        ("system_num_pocket_residues", ">", 2),
        #("system_num_pocket_residues", "<", 40),
        ("system_proper_num_ligand_chains", ">", 0),
        ("system_proper_ligand_max_molecular_weight", "<", 700),
        ("ligand_num_heavy_atoms", "<", 100),
        ("ligand_is_proper", "==", True),
        ("entry_resolution", "<", 4.0),
        ("entry_validation_r", "<", 0.4),
        ("entry_validation_rfree", "<", 0.45),
        ("entry_validation_r_minus_rfree", "<", 0.075),
        ('ligand_num_unresolved_heavy_atoms', '<', 1)
    ]
    
    test_ids = get_test_data_ids(Path('/home/mhjang/workspace/flex/prep/posebusters_paper_results.csv'))
    
    index = query_index(columns=columns, filters=filters)
    index = index.drop_duplicates("system_id")
    index = index[~index['entry_pdb_id'].isin(test_ids)]
    
    system_ids = sorted(index['system_id'].tolist())
    
    print(f'Found {len(system_ids)} plinder systems')
    return system_ids


def add_ligand_reference_data(lig_data: Dict) -> Dict:
    '''
    Add reference conformer coordinates, sp2 torsion indices, and the
    distance-bound constraint matrix to a parsed ligand record.

    Uses the SDF-parsed RDKit Mol attached on `lig_data['_mol']` so that
    constraint_matrix, ref_coords, and sp2_torsion_indices are all indexed in
    the same SDF atom order as `lig_data['atoms']`, `bonds`, and `coords`.
    Falls back to SMILES re-parse only when `_mol` is missing.
    '''
    atoms = lig_data['atoms']
    bonds = lig_data['bonds']
    mol = lig_data.pop('_mol', None)

    if mol is None:
        ref_data = generate_ligand_reference_data(lig_data['smiles'])
    else:
        ref_data = generate_ligand_reference_data(mol)

    if ref_data is not None and ref_data['ref_coords'] is not None:
        lig_data['ref_coords'] = ref_data['ref_coords']
        lig_data['sp2_torsion_indices'] = ref_data['sp2_torsion_indices']
        lig_data['constraint_matrix'] = ref_data['constraint_matrix']
    else:
        n_atoms = len(atoms)
        lig_data['ref_coords'] = np.zeros((n_atoms, 3), dtype=np.float32)
        lig_data['sp2_torsion_indices'] = get_sp2_torsion_indices(atoms, bonds)
        lig_data['constraint_matrix'] = None

    return lig_data


def parse_single_system(system_dir: Path, storage_dir: Path, skip_existing: bool = False):
    system_name = system_dir.name
    storage_dir = storage_dir / system_name
    if skip_existing and storage_dir.exists():
        return
    storage_dir.mkdir(parents=True, exist_ok=True)

    lig_dir = system_dir / 'ligand_files'
    rec_file = system_dir / 'receptor.cif'

    try:
        lig_data_list = []
        for lig_file in lig_dir.glob('*.sdf'):
            lig_data = extract_ligand_data(lig_file)
            if lig_data is not None:
                # Add reference conformer and torsion indices
                lig_data = add_ligand_reference_data(lig_data)
                lig_data_list.append(lig_data)

        if len(lig_data_list) == 0:
            print(f'No valid ligands found in {system_dir}')
            storage_dir.rmdir()
            return

        valid_ligs = get_valid_ligands(lig_data_list)
        parsed_data = parse_cif(rec_file)
        pocket_data, sequence_data = get_pocket_residues(parsed_data, valid_ligs)
        pocket_residues = overwrite_coords_on_reference_data(pocket_data)

        dump(valid_ligs, storage_dir / 'ligands.pkl', compress=3)
        dump(pocket_residues, storage_dir / 'pocket_residues.pkl', compress=3)
        dump(sequence_data, storage_dir / 'sequence_data.pkl', compress=3)

    except Exception as e:
        print(f'Error parsing system {system_dir}: {e}')
        return

if __name__ == '__main__':
    plinder_system_dir = Path('/mnt/drive/plinder/2024-06/v2/systems')
    storage_dir = Path('/home/mhjang/workspace/flex/data')
    
    plinders_systems = get_plinder_systems()
    plinders_systems = [plinder_system_dir / i for i in plinders_systems]
    
    Parallel(n_jobs=-1)(delayed(parse_single_system)(system_dir, storage_dir) for system_dir in tqdm(plinders_systems))