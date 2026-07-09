from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

bsite_dir = Path('/mnt/drive/BsitePDB2')

def get_canonical_smiles(filename: str):
    assert filename.suffix == '.sdf', f'Wrong suffix: {filename.suffix}'
    suppl = Chem.SDMolSupplier(filename)
    for mol in suppl:
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return None

def bsite_chain_parse(filename: str):
    chains = filename.split('.')[0].split('_')[-1]
    
    return chains

def process_single_directory(subdir: Path):
    pdb_id = subdir.name
    
    protein_files, ligand_files = {}, {}
    
    for file in subdir.glob('*'):
        if 'protein_' in file.name and file.suffix == '.pdb':
            protein_files[bsite_chain_parse(file.name)] = file
        elif 'ligand_' in file.name and file.suffix == '.sdf':
            smiles = get_canonical_smiles(file)
            if smiles is not None:
                ligand_files[bsite_chain_parse(file.name)] = [smiles, file]
            else:
                continue
            
    chain_smi_map = {chain:smiles for chain, [smiles, _] in ligand_files.items()}
    
    # Sort mapping by chain order
    chain_ordered = {}
    for chain in sorted(chain_smi_map.keys()):
        chain_ordered[chain] = chain_smi_map[chain]
        
    # Find unqiue smiles
    unique_smiles = set()
    unique_chain_ordered = {}
    for chain, smiles in chain_ordered.items():
        if smiles not in unique_smiles:
            unique_smiles.add(smiles)
            unique_chain_ordered[chain] = smiles
            
    # Final mapping
    final_mapping = {}
    for chain, smiles in unique_chain_ordered.items():
        key = f'{pdb_id}_{chain}'
        final_mapping[key] = {'receptor': protein_files[chain], 'ligand': ligand_files[chain][1]}
        
        assert final_mapping[key]['receptor'].exists(), f'Receptor file does not exist: {final_mapping[key]["receptor"]}'
        assert final_mapping[key]['ligand'].exists(), f'Ligand file does not exist: {final_mapping[key]["ligand"]}'
        
    return final_mapping

import shutil
import os
from tqdm import tqdm

def run():
    output_dir = Path('/mnt/drive/bsite_unique')
    output_dir.mkdir(exist_ok=True)
    
    for subdir in tqdm(sorted(bsite_dir.glob('*'))):
        if subdir.is_dir():
            try:
                final_mapping = process_single_directory(subdir)
                for key, files in final_mapping.items():
                    output_subdir = output_dir / key
                    output_subdir.mkdir(exist_ok=True)
                    
                    shutil.copy(files['receptor'], output_subdir / 'protein.pdb')
                    shutil.copy(files['ligand'], output_subdir / 'ligand.sdf')
            except Exception as e:
                print(f'Error processing {subdir}: {e}')
                continue

if __name__ == '__main__':
    run()