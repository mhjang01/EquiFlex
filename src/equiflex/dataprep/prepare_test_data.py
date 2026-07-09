'''
Test set data preparation for Posebusters and Astex datasets.

Processes all systems from benchmark datasets into the same format as training data:
- ligands.pkl
- pocket_residues.pkl
- sequence_data.pkl
- ligand_features.pt
- pocket_features.pt
- esm3_embeddings.pt
- unimol_embeddings.pt

Subdirectory names are preserved from source datasets (e.g., 5SD5_HWI, 1GKC_NFH).

Usage:
    python prepare_test_data.py --dataset posebusters
    python prepare_test_data.py --dataset astex
    python prepare_test_data.py --dataset all
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import numpy as np
from joblib import dump, load

from equiflex.prep.parse import (
    parse_structure,
    extract_ligand_data,
    extract_coordinates,
    get_pocket_residues,
    overwrite_coords_on_reference_data,
    get_sp2_torsion_indices,
)
from equiflex.prep.featurize import (
    featurize_atoms,
    featurize_bonds,
    ESM3Extractor,
    UniMolExtractor,
)
from equiflex.prep.reference import generate_ligand_reference_data

# Backbone atoms for backbone mask
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
_CB_ATOM_NAME = 'CB'

# Dataset paths
POSEBUSTERS_PATH = Path('/mnt/drive/posebusters')
ASTEX_PATH = Path('/mnt/drive/astex_diverse_set')
TEST_DATA_PATH = Path('/home/mhjang/workspace/flex/test_data')


def discover_all_systems(dataset: str) -> List[Tuple[str, Path, Path]]:
    '''
    Discover all systems from a dataset directory.

    Returns list of (system_name, protein_path, ligand_path) tuples.
    system_name preserves the original subdirectory name (e.g., 5SD5_HWI).
    '''
    if dataset == 'astex':
        base_path = ASTEX_PATH
    else:
        base_path = POSEBUSTERS_PATH

    systems = []
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue

        protein_path = None
        ligand_path = None

        for f in subdir.glob('*'):
            if f.name.endswith('_protein.pdb'):
                protein_path = f
            elif f.name.endswith('_ligand.sdf'):
                ligand_path = f

        if protein_path and ligand_path:
            systems.append((subdir.name, protein_path, ligand_path))

    return systems


def process_ligands(ligands: List[Dict]) -> Dict:
    '''Process ligands into feature tensors.'''
    all_atom_features = []
    all_coords = []
    all_ref_coords = []
    all_constraint_matrices = []
    ligand_uid_vec = []
    smiles_list = []
    n_atoms_per_ligand = []
    all_sp2_torsions = []
    torsion_ligand_idx = []
    atom_offset = 0
    all_bonds_data = []

    for lig_idx, lig_data in enumerate(ligands):
        # Atom features (ligand: drop canonical-rank suffix, encode bare element
        # symbol so 256-dim ASCII block stays element-discriminative without
        # noisy rank bits — same convention as exec/featurize_data.py)
        atom_feats = []
        for atom in lig_data['atoms']:
            atom_for_feat = {**atom, 'name': atom['symbol']}
            feat_dict = featurize_atoms(atom_for_feat)
            atom_feats.append(feat_dict['features'])
        atom_feats = torch.stack(atom_feats, dim=0)
        all_atom_features.append(atom_feats)

        # Coordinates
        coords = torch.tensor(lig_data['coords'], dtype=torch.float32)
        all_coords.append(coords)

        # Reference coordinates
        n_atoms = len(lig_data['atoms'])
        if 'ref_coords' in lig_data and lig_data['ref_coords'] is not None:
            ref_coords = torch.tensor(lig_data['ref_coords'], dtype=torch.float32)
        else:
            ref_coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
        all_ref_coords.append(ref_coords)

        # UID vector
        ligand_uid_vec.extend([lig_idx] * n_atoms)

        # Bonds
        for bond in lig_data['bonds']:
            feat_dict = featurize_bonds(bond)
            global_i = feat_dict['begin_atom_idx'] + atom_offset
            global_j = feat_dict['end_atom_idx'] + atom_offset
            all_bonds_data.append((global_i, global_j, feat_dict['features']))

        # Torsion indices
        if 'sp2_torsion_indices' in lig_data and len(lig_data['sp2_torsion_indices']) > 0:
            local_torsions = lig_data['sp2_torsion_indices']
            if isinstance(local_torsions, np.ndarray):
                global_torsions = local_torsions + atom_offset
            else:
                global_torsions = np.array(local_torsions) + atom_offset
            all_sp2_torsions.append(global_torsions)
            torsion_ligand_idx.extend([lig_idx] * len(global_torsions))

        # Constraint matrix
        if 'constraint_matrix' in lig_data and lig_data['constraint_matrix'] is not None:
            constraint_matrix = torch.tensor(lig_data['constraint_matrix'], dtype=torch.float32)
            all_constraint_matrices.append(constraint_matrix)
        else:
            raise ValueError("Failed to generate ligand constraint matrix")

        smiles_list.append(lig_data['smiles'])
        n_atoms_per_ligand.append(n_atoms)
        atom_offset += n_atoms

    # Stack features
    atom_features = torch.cat(all_atom_features, dim=0)
    coords = torch.cat(all_coords, dim=0)
    ref_coords = torch.cat(all_ref_coords, dim=0)
    ligand_uid_vec = torch.tensor(ligand_uid_vec, dtype=torch.long)
    constraint_matrix = torch.cat(all_constraint_matrices, dim=0)

    # Build bond matrix
    total_atoms = atom_features.shape[0]
    bond_dim = 26
    bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
    for global_i, global_j, feats in all_bonds_data:
        bond_features[global_i, global_j] = feats
        bond_features[global_j, global_i] = feats

    # Stack torsions
    if all_sp2_torsions:
        sp2_torsion_indices = torch.tensor(np.concatenate(all_sp2_torsions), dtype=torch.long)
        torsion_ligand_idx = torch.tensor(torsion_ligand_idx, dtype=torch.long)
    else:
        sp2_torsion_indices = torch.empty((0, 4), dtype=torch.long)
        torsion_ligand_idx = torch.empty((0,), dtype=torch.long)

    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'coords': coords,
        'ref_coords': ref_coords,
        'ligand_uid_vec': ligand_uid_vec,
        'smiles_list': smiles_list,
        'n_atoms_per_ligand': n_atoms_per_ligand,
        'sp2_torsion_indices': sp2_torsion_indices,
        'torsion_ligand_idx': torsion_ligand_idx,
        'constraint_matrix': constraint_matrix,
    }


def process_pocket_residues(pocket_residues: Dict) -> Optional[Dict]:
    '''Process pocket residues into feature tensors.'''
    all_atom_features = []
    all_coords = []
    all_ref_coords = []
    all_missing_masks = []
    all_backbone_masks = []
    all_cb_masks = []
    residue_uid_vec = []
    residue_keys = []
    n_atoms_per_residue = []
    all_sp2_torsions = []
    torsion_residue_idx = []
    all_constraint_matrices = []
    atom_offset = 0
    all_bonds_data = []
    residue_idx = 0
    residue_n_idx: List[int] = []
    residue_ca_idx: List[int] = []
    residue_c_idx: List[int] = []
    residue_is_gly: List[bool] = []

    for chain_id, residues in sorted(pocket_residues.items()):
        for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
            # Atom features
            atom_feats = []
            for atom in res_data['atoms']:
                feat_dict = featurize_atoms(atom)
                atom_feats.append(feat_dict['features'])
            atom_feats = torch.stack(atom_feats, dim=0)
            all_atom_features.append(atom_feats)

            # Coordinates
            coords = torch.tensor(res_data['coords'], dtype=torch.float32)
            all_coords.append(coords)

            # Reference coordinates
            n_atoms = len(res_data['atoms'])
            if 'ref_coords' in res_data and res_data['ref_coords'] is not None:
                ref_coords = torch.tensor(res_data['ref_coords'], dtype=torch.float32)
            else:
                ref_coords = coords.clone()
            all_ref_coords.append(ref_coords)

            # Missing mask
            missing_mask = torch.tensor(res_data['missing_masks'], dtype=torch.bool)
            all_missing_masks.append(missing_mask)

            atom_names = [atom['name'] for atom in res_data['atoms']]

            # Backbone mask
            backbone_mask = torch.tensor(
                [1.0 if name in BACKBONE_ATOMS else 0.0 for name in atom_names],
                dtype=torch.float32
            )
            all_backbone_masks.append(backbone_mask)

            # CB mask: 1.0 for CB atom (sidechain root, fixed via N/CA/C frame)
            cb_mask = torch.tensor(
                [1.0 if name == _CB_ATOM_NAME else 0.0 for name in atom_names],
                dtype=torch.float32,
            )
            all_cb_masks.append(cb_mask)

            residue_n_idx.append(atom_offset + atom_names.index('N') if 'N' in atom_names else -1)
            residue_ca_idx.append(atom_offset + atom_names.index('CA') if 'CA' in atom_names else -1)
            residue_c_idx.append(atom_offset + atom_names.index('C') if 'C' in atom_names else -1)
            residue_is_gly.append(res_data.get('aa_code', resname) == 'GLY')

            # UID vector
            residue_uid_vec.extend([residue_idx] * n_atoms)

            # Bonds
            for bond in res_data['bonds']:
                feat_dict = featurize_bonds(bond)
                global_i = feat_dict['begin_atom_idx'] + atom_offset
                global_j = feat_dict['end_atom_idx'] + atom_offset
                all_bonds_data.append((global_i, global_j, feat_dict['features']))

            # Torsion indices
            if 'sp2_torsion_indices' in res_data and len(res_data['sp2_torsion_indices']) > 0:
                local_torsions = res_data['sp2_torsion_indices']
                if isinstance(local_torsions, np.ndarray):
                    global_torsions = local_torsions + atom_offset
                else:
                    global_torsions = np.array(local_torsions) + atom_offset
                all_sp2_torsions.append(global_torsions)
                torsion_residue_idx.extend([residue_idx] * len(global_torsions))
            else:
                atoms = res_data['atoms']
                bonds = res_data['bonds']
                local_torsions = get_sp2_torsion_indices(atoms, bonds)
                if len(local_torsions) > 0:
                    global_torsions = local_torsions + atom_offset
                    all_sp2_torsions.append(global_torsions)
                    torsion_residue_idx.extend([residue_idx] * len(global_torsions))

            # Constraint matrix
            if 'constraint_matrix' in res_data and res_data['constraint_matrix'] is not None:
                constraint_matrix = torch.tensor(res_data['constraint_matrix'], dtype=torch.float32)
                all_constraint_matrices.append(constraint_matrix)
            else:
                raise ValueError("Failed to generate pocket constraint matrix")

            residue_keys.append((chain_id, resname, resnum, entity_id))
            n_atoms_per_residue.append(n_atoms)
            atom_offset += n_atoms
            residue_idx += 1

    if not all_atom_features:
        return None

    # Stack features
    atom_features = torch.cat(all_atom_features, dim=0)
    coords = torch.cat(all_coords, dim=0)
    ref_coords = torch.cat(all_ref_coords, dim=0)
    missing_mask = torch.cat(all_missing_masks, dim=0)
    backbone_mask = torch.cat(all_backbone_masks, dim=0)
    cb_mask = torch.cat(all_cb_masks, dim=0)
    residue_uid_vec = torch.tensor(residue_uid_vec, dtype=torch.long)

    backbone_atom_indices = {
        'n_idx': torch.tensor(residue_n_idx, dtype=torch.long),
        'ca_idx': torch.tensor(residue_ca_idx, dtype=torch.long),
        'c_idx': torch.tensor(residue_c_idx, dtype=torch.long),
        'is_gly': torch.tensor(residue_is_gly, dtype=torch.bool),
    }

    # Build bond matrix
    total_atoms = atom_features.shape[0]
    bond_dim = 26
    bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
    for global_i, global_j, feats in all_bonds_data:
        bond_features[global_i, global_j] = feats
        bond_features[global_j, global_i] = feats

    # Build full constraint matrix (block diagonal)
    constraint_matrices = torch.block_diag(*all_constraint_matrices)

    # Stack torsions
    if all_sp2_torsions:
        sp2_torsion_indices = torch.tensor(np.concatenate(all_sp2_torsions), dtype=torch.long)
        torsion_residue_idx = torch.tensor(torsion_residue_idx, dtype=torch.long)
    else:
        sp2_torsion_indices = torch.empty((0, 4), dtype=torch.long)
        torsion_residue_idx = torch.empty((0,), dtype=torch.long)

    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'coords': coords,
        'ref_coords': ref_coords,
        'missing_mask': missing_mask,
        'backbone_mask': backbone_mask,
        'cb_mask': cb_mask,
        'backbone_atom_indices': backbone_atom_indices,
        'residue_uid_vec': residue_uid_vec,
        'residue_keys': residue_keys,
        'n_atoms_per_residue': n_atoms_per_residue,
        'sp2_torsion_indices': sp2_torsion_indices,
        'torsion_residue_idx': torsion_residue_idx,
        'constraint_matrix': constraint_matrices,
    }


def process_single_complex(
    system_name: str,
    protein_path: Path,
    ligand_path: Path,
    output_dir: Path,
) -> Tuple[bool, str]:
    '''
    Process a single protein-ligand complex.

    Args:
        system_name: Original subdirectory name (e.g., 5SD5_HWI)
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand SDF file
        output_dir: Output directory

    Returns:
        (success, message) tuple
    '''
    system_dir = output_dir / system_name
    system_dir.mkdir(parents=True, exist_ok=True)

    # Parse protein structure directly (PDB or CIF)
    # Note: Don't convert PDB to CIF - the converted CIF lacks proper sequence info
    protein_data = parse_structure(protein_path)
    if protein_data is None:
        return False, "Failed to parse protein structure"

    # Parse ligand
    ligand_data = extract_ligand_data(ligand_path)
    if ligand_data is None:
        return False, "Failed to parse ligand"

    # Get pocket residues using ligand coordinates
    # Returns: ({chain_id: [residue_list]}, {chain_id: sequence})
    pocket_data, sequence_data = get_pocket_residues(protein_data, [ligand_data])

    if not pocket_data or all(len(v) == 0 for v in pocket_data.values()):
        return False, "No pocket residues found"

    # overwrite_coords_on_reference_data takes {chain_id: [residue_list]} format
    # and returns {chain_id: {(resname, resnum, entity_id): residue_dict}} format
    pocket_residues = overwrite_coords_on_reference_data(pocket_data)

    # Generate reference data using the SDF-parsed Mol so ref_coords / cm /
    # sp2_torsion_indices are indexed in the same SDF atom order as
    # ligand_data['atoms'/'bonds'/'coords']. Falls back to SMILES re-parse only
    # if `_mol` is missing (legacy paths).
    mol = ligand_data.pop('_mol', None)
    ref_data = generate_ligand_reference_data(mol if mol is not None else ligand_data['smiles'])
    if ref_data is None:
        return False, "Failed to generate ligand reference data"
    ligand_data['ref_coords'] = ref_data['ref_coords']
    ligand_data['constraint_matrix'] = ref_data['constraint_matrix']
    ligand_data['sp2_torsion_indices'] = ref_data['sp2_torsion_indices']

    ligands = [ligand_data]

    # Save raw data (pkl files)
    dump(ligands, system_dir / 'ligands.pkl')
    dump(pocket_residues, system_dir / 'pocket_residues.pkl')
    dump(sequence_data, system_dir / 'sequence_data.pkl')

    # Process and save features (pt files)
    ligand_features = process_ligands(ligands)
    pocket_features = process_pocket_residues(pocket_residues)

    if pocket_features is None:
        return False, "Failed to process pocket residues"

    torch.save(ligand_features, system_dir / 'ligand_features.pt')
    torch.save(pocket_features, system_dir / 'pocket_features.pt')

    return True, "success"


def extract_esm3_embeddings(
    esm3_extractor: ESM3Extractor,
    system_dir: Path,
) -> bool:
    '''Extract ESM3 embeddings for a system.'''
    pocket_residues = load(system_dir / 'pocket_residues.pkl')
    sequence_data = load(system_dir / 'sequence_data.pkl')
    pocket_features = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    residue_keys = pocket_features['residue_keys']

    pocket_embeddings = esm3_extractor.extract_pocket_embeddings(
        sequence_data, pocket_residues, system_dir=system_dir
    )

    embeddings = []
    for chain_id, resname, resnum, entity_id in residue_keys:
        key = (chain_id, resnum, entity_id)
        if key in pocket_embeddings:
            embeddings.append(pocket_embeddings[key])
        else:
            embeddings.append(torch.zeros(1536))

    if not embeddings:
        return False

    esm3_embeddings = torch.stack(embeddings, dim=0)
    torch.save(esm3_embeddings, system_dir / 'esm3_embeddings.pt')
    return True


def extract_unimol_embeddings(
    unimol_extractor: UniMolExtractor,
    system_dir: Path,
) -> bool:
    '''Extract UniMol embeddings for a system.'''
    ligands = load(system_dir / 'ligands.pkl')
    ligand_features = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
    n_atoms_per_ligand = ligand_features['n_atoms_per_ligand']

    smiles_list = [lig['smiles'] for lig in ligands]
    atoms_list = [lig['atoms'] for lig in ligands]

    embeddings_list = unimol_extractor.extract_batch(smiles_list, atoms_list)

    all_embeddings = []
    for i, emb in enumerate(embeddings_list):
        if emb is None:
            embed_dim = 512
            all_embeddings.append(torch.zeros(n_atoms_per_ligand[i], embed_dim))
        else:
            all_embeddings.append(emb)

    if not all_embeddings:
        return False

    unimol_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(unimol_embeddings, system_dir / 'unimol_embeddings.pt')
    return True


def main(
    dataset: str = 'all',
    output_dir: str = None,
    skip_existing: bool = True,
    device: str = 'cuda',
):
    if output_dir is None:
        output_dir = TEST_DATA_PATH
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all systems from dataset directories
    systems_to_process = []
    if dataset in ['all', 'posebusters']:
        pb_systems = discover_all_systems('posebusters')
        systems_to_process.extend([(s, 'posebusters') for s in pb_systems])
        print(f"Found {len(pb_systems)} systems in Posebusters")
    if dataset in ['all', 'astex']:
        astex_systems = discover_all_systems('astex')
        systems_to_process.extend([(s, 'astex') for s in astex_systems])
        print(f"Found {len(astex_systems)} systems in Astex")

    print(f"Total systems to process: {len(systems_to_process)}")
    print(f"Output directory: {output_dir}")

    # Step 1: Parse and featurize
    print("\n" + "="*60)
    print("STEP 1: Parse and Featurize")
    print("="*60)

    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []
    valid_dirs = []

    for (system_name, protein_path, ligand_path), ds in tqdm(systems_to_process, desc="Parsing"):
        system_dir = output_dir / system_name

        # Skip if exists
        if skip_existing and (system_dir / 'pocket_features.pt').exists():
            skip_count += 1
            valid_dirs.append(system_dir)
            continue

        # Process - skip on any error
        try:
            success, msg = process_single_complex(
                system_name, protein_path, ligand_path, output_dir
            )
            if success:
                success_count += 1
                valid_dirs.append(system_dir)
            else:
                errors.append((system_name, msg))
                error_count += 1
                # Clean up failed directory
                if system_dir.exists():
                    shutil.rmtree(system_dir)
        except Exception as e:
            errors.append((system_name, str(e)))
            error_count += 1
            if system_dir.exists():
                shutil.rmtree(system_dir)

    print(f"Step 1 Complete: Success={success_count}, Skipped={skip_count}, Errors={error_count}")

    if errors:
        print(f"Errors ({len(errors)}):")
        for system_name, msg in errors[:10]:
            print(f"  {system_name}: {msg}")
        with open(output_dir / 'step1_errors.txt', 'w') as f:
            for system_name, msg in errors:
                f.write(f"{system_name}: {msg}\n")

    # Step 2: ESM3 extraction
    print("\n" + "="*60)
    print("STEP 2: ESM3 Embedding Extraction")
    print("="*60)

    print("Loading ESM3 model...")
    esm3_extractor = ESM3Extractor(device=device)

    esm_success = 0
    esm_skip = 0
    esm_error = 0
    esm_errors = []
    esm_valid_dirs = []

    for system_dir in tqdm(valid_dirs, desc="ESM3"):
        system_name = system_dir.name

        if skip_existing and (system_dir / 'esm3_embeddings.pt').exists():
            esm_skip += 1
            esm_valid_dirs.append(system_dir)
            continue

        try:
            if extract_esm3_embeddings(esm3_extractor, system_dir):
                esm_success += 1
                esm_valid_dirs.append(system_dir)
            else:
                esm_errors.append((system_name, "ESM3 extraction failed"))
                esm_error += 1
                shutil.rmtree(system_dir)
        except Exception as e:
            esm_errors.append((system_name, str(e)))
            esm_error += 1
            shutil.rmtree(system_dir)

    print(f"Step 2 Complete: Success={esm_success}, Skipped={esm_skip}, Errors={esm_error}")

    if esm_errors:
        with open(output_dir / 'step2_esm3_errors.txt', 'w') as f:
            for system_name, msg in esm_errors:
                f.write(f"{system_name}: {msg}\n")

    # Step 3: UniMol extraction
    print("\n" + "="*60)
    print("STEP 3: UniMol Embedding Extraction")
    print("="*60)

    print("Loading UniMol model...")
    unimol_extractor = UniMolExtractor(model_size='1.1B')

    unimol_success = 0
    unimol_skip = 0
    unimol_error = 0
    unimol_errors = []
    final_dirs = []

    for system_dir in tqdm(esm_valid_dirs, desc="UniMol"):
        system_name = system_dir.name

        if skip_existing and (system_dir / 'unimol_embeddings.pt').exists():
            unimol_skip += 1
            final_dirs.append(system_dir)
            continue

        try:
            if extract_unimol_embeddings(unimol_extractor, system_dir):
                unimol_success += 1
                final_dirs.append(system_dir)
            else:
                unimol_errors.append((system_name, "UniMol extraction failed"))
                unimol_error += 1
                shutil.rmtree(system_dir)
        except Exception as e:
            unimol_errors.append((system_name, str(e)))
            unimol_error += 1
            shutil.rmtree(system_dir)

    print(f"Step 3 Complete: Success={unimol_success}, Skipped={unimol_skip}, Errors={unimol_error}")

    if unimol_errors:
        with open(output_dir / 'step3_unimol_errors.txt', 'w') as f:
            for system_name, msg in unimol_errors:
                f.write(f"{system_name}: {msg}\n")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total systems requested: {len(systems_to_process)}")
    print(f"Successfully processed: {len(final_dirs)}")
    print(f"Output directory: {output_dir}")

    # Save processed list
    processed_systems = [d.name for d in final_dirs]
    with open(output_dir / 'processed_systems.json', 'w') as f:
        json.dump(processed_systems, f, indent=2)
    print(f"Processed systems saved to: {output_dir / 'processed_systems.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare test data for Posebusters and Astex')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'posebusters', 'astex'],
                        help='Which dataset to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: ./test_data)')
    parser.add_argument('--no_skip', action='store_true',
                        help='Reprocess even if exists')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for GPU models')

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        output_dir=args.output,
        skip_existing=not args.no_skip,
        device=args.device,
    )
