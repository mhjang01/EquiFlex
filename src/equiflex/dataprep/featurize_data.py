'''
Featurization script for pre-processed flex/data subdirectories.

Processes:
1. Atom features -> stacked tensor [N_atoms, F_atom]
2. Bond features -> full adjacency matrix [N_atoms, N_atoms, F_bond]
3. Residue uid_vec for scatter operations
4. Coordinates and masks
5. ESM3 embeddings for pocket residues
6. UniMol embeddings for ligands
7. Statistics collection (atom counts, missing atoms)
8. Cleanup of invalid directories

Pipeline:
- Step 1: Graph featurization (parallel CPU)
- Step 2: ESM3 extraction (sequential GPU, batched by chain)
- Step 3: UniMol extraction (batched GPU)
'''

import os
# Suppress tokenizers parallelism warning (must be set before importing transformers/tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import torch
import numpy as np
from joblib import load, dump, Parallel, delayed
from tqdm import tqdm
import shutil

from equiflex.prep.featurize import featurize_atoms, featurize_bonds, UniMolExtractor, ESM3Extractor
from equiflex.prep.parse import get_sp2_torsion_indices
from equiflex.prep.const import STD_AA_ATOMS, NONSTD_AA_ATOMS, AA_3_TO_1
from equiflex.prep.validation import (
    SIGNATURE_KEY, ligand_atom_signature, validate_ligand_records,
    unimol_cache_reason, save_unimol_cache,
)

# Backbone atoms that remain fixed during flow matching
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}
_CB_ATOM_NAME = 'CB'


# Statistics collection
class FeaturizationStats:
    '''Collects statistics during featurization.'''

    def __init__(self):
        self.ligand_atom_counts = []
        self.pocket_atom_counts = []
        self.missing_atom_counts = []
        self.total_missing_atoms = 0
        self.total_pocket_atoms = 0
        self.removed_dirs = []

    def add_ligand_stats(self, n_atoms: int):
        self.ligand_atom_counts.append(n_atoms)

    def add_pocket_stats(self, n_atoms: int, n_missing: int):
        self.pocket_atom_counts.append(n_atoms)
        self.missing_atom_counts.append(n_missing)
        self.total_pocket_atoms += n_atoms
        self.total_missing_atoms += n_missing

    def add_removed_dir(self, dir_name: str):
        self.removed_dirs.append(dir_name)

    def summarize(self) -> Dict:
        if not self.ligand_atom_counts:
            return {'error': 'no data collected'}

        lig_counts = np.array(self.ligand_atom_counts)
        pocket_counts = np.array(self.pocket_atom_counts)
        missing_counts = np.array(self.missing_atom_counts)

        return {
            'n_systems': len(self.ligand_atom_counts),
            'ligand_atoms': {
                'total': int(lig_counts.sum()),
                'mean': float(lig_counts.mean()),
                'std': float(lig_counts.std()),
                'min': int(lig_counts.min()),
                'max': int(lig_counts.max()),
                'median': float(np.median(lig_counts)),
            },
            'pocket_atoms': {
                'total': int(pocket_counts.sum()),
                'mean': float(pocket_counts.mean()),
                'std': float(pocket_counts.std()),
                'min': int(pocket_counts.min()),
                'max': int(pocket_counts.max()),
                'median': float(np.median(pocket_counts)),
            },
            'missing_atoms': {
                'total': int(missing_counts.sum()),
                'mean': float(missing_counts.mean()),
                'std': float(missing_counts.std()),
                'min': int(missing_counts.min()),
                'max': int(missing_counts.max()),
                'median': float(np.median(missing_counts)),
                'missing_ratio': float(self.total_missing_atoms / self.total_pocket_atoms) if self.total_pocket_atoms > 0 else 0,
            },
            'removed_dirs': {
                'count': len(self.removed_dirs),
                'names': self.removed_dirs[:20],  # First 20 for brevity
            }
        }

    def print_summary(self):
        stats = self.summarize()
        if 'error' in stats:
            print(f"Stats error: {stats['error']}")
            return

        print("\n" + "="*60)
        print("FEATURIZATION STATISTICS")
        print("="*60)

        print(f"\nTotal systems processed: {stats['n_systems']}")

        print(f"\n--- Ligand Atoms ---")
        lig = stats['ligand_atoms']
        print(f"  Total: {lig['total']:,}")
        print(f"  Mean ± Std: {lig['mean']:.1f} ± {lig['std']:.1f}")
        print(f"  Min / Median / Max: {lig['min']} / {lig['median']:.0f} / {lig['max']}")

        print(f"\n--- Pocket Atoms ---")
        pkt = stats['pocket_atoms']
        print(f"  Total: {pkt['total']:,}")
        print(f"  Mean ± Std: {pkt['mean']:.1f} ± {pkt['std']:.1f}")
        print(f"  Min / Median / Max: {pkt['min']} / {pkt['median']:.0f} / {pkt['max']}")

        print(f"\n--- Missing Atoms ---")
        miss = stats['missing_atoms']
        print(f"  Total: {miss['total']:,}")
        print(f"  Mean ± Std: {miss['mean']:.1f} ± {miss['std']:.1f}")
        print(f"  Min / Median / Max: {miss['min']} / {miss['median']:.0f} / {miss['max']}")
        print(f"  Missing Ratio: {miss['missing_ratio']*100:.2f}%")

        print(f"\n--- Removed Directories ---")
        rm = stats['removed_dirs']
        print(f"  Count: {rm['count']}")
        if rm['names']:
            print(f"  Examples: {rm['names'][:5]}")

        print("="*60 + "\n")


def featurize_ligand_atoms(ligand_data: Dict) -> torch.Tensor:
    '''Featurize all atoms in a ligand. Returns [N_atoms, F].

    Uses featurize_atoms directly (identical to the prep.process path used at
    inference). The atom-name ASCII block was removed from the feature schema,
    so ligand atoms no longer carry rank-suffix noise and train/inference
    featurization is consistent by construction.
    '''
    atom_features = []
    for atom in ligand_data['atoms']:
        feat_dict = featurize_atoms(atom)
        atom_features.append(feat_dict['features'])
    return torch.stack(atom_features, dim=0)


def featurize_ligand_bonds_full_matrix(ligand_data: Dict, bond_dim: int = 26) -> torch.Tensor:
    '''
    Convert directed bond list to full adjacency matrix [N, N, F].
    Bonds are symmetric, so we fill both (i,j) and (j,i).
    Diagonal and non-bonded pairs are zeros.
    '''
    n_atoms = len(ligand_data['atoms'])
    bond_matrix = torch.zeros(n_atoms, n_atoms, bond_dim)

    for bond in ligand_data['bonds']:
        feat_dict = featurize_bonds(bond)
        i, j = feat_dict['begin_atom_idx'], feat_dict['end_atom_idx']
        bond_matrix[i, j] = feat_dict['features']
        bond_matrix[j, i] = feat_dict['features']  # Symmetric

    return bond_matrix


def featurize_residue_atoms(residue_data: Dict) -> torch.Tensor:
    '''Featurize all atoms in a residue. Returns [N_atoms, F].'''
    atom_features = []
    for atom in residue_data['atoms']:
        feat_dict = featurize_atoms(atom)
        atom_features.append(feat_dict['features'])
    return torch.stack(atom_features, dim=0)


def featurize_residue_bonds_full_matrix(residue_data: Dict, bond_dim: int = 26) -> torch.Tensor:
    '''Convert residue bonds to full adjacency matrix [N, N, F].'''
    n_atoms = len(residue_data['atoms'])
    bond_matrix = torch.zeros(n_atoms, n_atoms, bond_dim)

    for bond in residue_data['bonds']:
        feat_dict = featurize_bonds(bond)
        i, j = feat_dict['begin_atom_idx'], feat_dict['end_atom_idx']
        bond_matrix[i, j] = feat_dict['features']
        bond_matrix[j, i] = feat_dict['features']

    return bond_matrix


def process_ligands(ligands: List[Dict]) -> Dict:
    '''
    Process all ligands in a system.

    Returns:
        Dict with:
            - atom_features: [total_atoms, F_atom]
            - bond_features: [total_atoms, total_atoms, F_bond]
            - coords: [total_atoms, 3] - ground truth coordinates
            - ref_coords: [total_atoms, 3] - reference conformer coordinates
            - ligand_uid_vec: [total_atoms] - which ligand each atom belongs to
            - smiles_list: List of SMILES strings
            - n_atoms_per_ligand: List of atom counts
            - sp2_torsion_indices: [N_torsions, 4] - planar torsion atom indices
            - torsion_ligand_idx: [N_torsions] - which ligand each torsion belongs to
    '''
    all_atom_features = []
    all_coords = []
    all_ref_coords = []
    ligand_uid_vec = []
    smiles_list = []
    n_atoms_per_ligand = []

    all_sp2_torsions = []
    torsion_ligand_idx = []
    all_constraint_matrices = []

    atom_offset = 0
    all_bonds_data = []  # Store (global_i, global_j, features) for later matrix construction

    for lig_idx, lig_data in enumerate(ligands):
        atom_feats = featurize_ligand_atoms(lig_data)
        all_atom_features.append(atom_feats)

        coords = torch.tensor(lig_data['coords'], dtype=torch.float32)
        all_coords.append(coords)

        n_atoms = len(lig_data['atoms'])
        if 'ref_coords' in lig_data and lig_data['ref_coords'] is not None:
            ref_coords = torch.tensor(lig_data['ref_coords'], dtype=torch.float32)
        else:
            raise ValueError("Failed to generate ligand reference data")
            
        all_ref_coords.append(ref_coords)

        # UID vector (which ligand this atom belongs to)
        ligand_uid_vec.extend([lig_idx] * n_atoms)

        # Collect bonds with global indices
        for bond in lig_data['bonds']:
            feat_dict = featurize_bonds(bond)
            global_i = feat_dict['begin_atom_idx'] + atom_offset
            global_j = feat_dict['end_atom_idx'] + atom_offset
            all_bonds_data.append((global_i, global_j, feat_dict['features']))

        # Torsion indices (convert to global indices)
        if 'sp2_torsion_indices' in lig_data and len(lig_data['sp2_torsion_indices']) > 0:
            local_torsions = lig_data['sp2_torsion_indices']
            if isinstance(local_torsions, np.ndarray):
                global_torsions = local_torsions + atom_offset
            else:
                global_torsions = np.array(local_torsions) + atom_offset
            all_sp2_torsions.append(global_torsions)
            torsion_ligand_idx.extend([lig_idx] * len(global_torsions))

        if 'constraint_matrix' in lig_data and lig_data['constraint_matrix'] is not None:
            constraint_matrix = torch.tensor(lig_data['constraint_matrix'], dtype=torch.float32)
            all_constraint_matrices.append(constraint_matrix)
        else:
            raise ValueError("Failed to generate ligand constraint matrix")
        
        smiles_list.append(lig_data['smiles'])
        n_atoms_per_ligand.append(n_atoms)
        atom_offset += n_atoms

    atom_features = torch.cat(all_atom_features, dim=0)
    coords = torch.cat(all_coords, dim=0)
    ref_coords = torch.cat(all_ref_coords, dim=0)
    ligand_uid_vec = torch.tensor(ligand_uid_vec, dtype=torch.long)
    total_atoms = atom_features.shape[0]

    # Build block-diagonal constraint matrix (per-ligand blocks). Single-atom
    # metal-ion ligands contribute a [1,1] block; organic ligands contribute
    # [N_i, N_i]. Direct cat on dim=0 fails when ligand sizes differ.
    constraint_matrix = torch.zeros(total_atoms, total_atoms, dtype=torch.float32)
    cm_offset = 0
    for cm in all_constraint_matrices:
        n_i = cm.shape[0]
        constraint_matrix[cm_offset:cm_offset + n_i, cm_offset:cm_offset + n_i] = cm
        cm_offset += n_i

    bond_dim = 26
    bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
    for global_i, global_j, feats in all_bonds_data:
        bond_features[global_i, global_j] = feats
        bond_features[global_j, global_i] = feats

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
        SIGNATURE_KEY: ligand_atom_signature(ligands),
        'sp2_torsion_indices': sp2_torsion_indices,
        'torsion_ligand_idx': torsion_ligand_idx,
        'constraint_matrix': constraint_matrix,
    }


def process_pocket_residues(pocket_residues: Dict) -> Dict:
    '''
    Process all pocket residues.

    Args:
        pocket_residues: {chain_id: {(resname, resnum, entity_id): residue_data}}

    Returns:
        Dict with:
            - atom_features: [total_atoms, F_atom]
            - bond_features: [total_atoms, total_atoms, F_bond]
            - coords: [total_atoms, 3] - ground truth coordinates
            - ref_coords: [total_atoms, 3] - reference conformer coordinates (from AA templates)
            - missing_mask: [total_atoms] - True for missing atoms
            - backbone_mask: [total_atoms] - 1.0 for backbone (N, CA, C, O), 0.0 for sidechain
            - residue_uid_vec: [total_atoms] - which residue each atom belongs to
            - residue_keys: List of (chain_id, resname, resnum, entity_id) tuples
            - n_atoms_per_residue: List of atom counts
            - sp2_torsion_indices: [N_torsions, 4] - planar torsion atom indices
            - torsion_residue_idx: [N_torsions] - which residue each torsion belongs to
    '''
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

    # Per-residue backbone atom indices (global) for CB placement / overlay
    residue_n_idx: List[int] = []
    residue_ca_idx: List[int] = []
    residue_c_idx: List[int] = []
    residue_is_gly: List[bool] = []

    # Sort residues for consistent ordering
    for chain_id, residues in sorted(pocket_residues.items()):
        for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
            # Atom features from reference data
            atom_feats = featurize_residue_atoms(res_data)
            all_atom_features.append(atom_feats)

            # Ground truth coordinates (from pocket, overwritten on reference)
            coords = torch.tensor(res_data['coords'], dtype=torch.float32)
            all_coords.append(coords)

            # Reference coordinates (from AA template)
            n_atoms = len(res_data['atoms'])
            if 'ref_coords' in res_data and res_data['ref_coords'] is not None:
                ref_coords = torch.tensor(res_data['ref_coords'], dtype=torch.float32)
            else:
                # Fallback: use ground truth coords as reference
                ref_coords = coords.clone()
            all_ref_coords.append(ref_coords)

            missing_mask = torch.tensor(res_data['missing_masks'], dtype=torch.bool)
            all_missing_masks.append(missing_mask)

            atom_names = [atom['name'] for atom in res_data['atoms']]

            # Backbone mask: 1.0 for backbone atoms (N, CA, C, O), 0.0 for sidechain
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

            # Per-residue backbone atom indices (global) for CB placement
            residue_n_idx.append(atom_offset + atom_names.index('N') if 'N' in atom_names else -1)
            residue_ca_idx.append(atom_offset + atom_names.index('CA') if 'CA' in atom_names else -1)
            residue_c_idx.append(atom_offset + atom_names.index('C') if 'C' in atom_names else -1)
            residue_is_gly.append(res_data.get('aa_code', resname) == 'GLY')

            residue_uid_vec.extend([residue_idx] * n_atoms)

            # Collect bonds with global indices
            for bond in res_data['bonds']:
                feat_dict = featurize_bonds(bond)
                global_i = feat_dict['begin_atom_idx'] + atom_offset
                global_j = feat_dict['end_atom_idx'] + atom_offset
                all_bonds_data.append((global_i, global_j, feat_dict['features']))

            # Torsion indices (convert to global indices)
            if 'sp2_torsion_indices' in res_data and len(res_data['sp2_torsion_indices']) > 0:
                local_torsions = res_data['sp2_torsion_indices']
                if isinstance(local_torsions, np.ndarray):
                    global_torsions = local_torsions + atom_offset
                else:
                    global_torsions = np.array(local_torsions) + atom_offset
                all_sp2_torsions.append(global_torsions)
                torsion_residue_idx.extend([residue_idx] * len(global_torsions))
            else:
                # Compute torsion indices on the fly if not present
                atoms = res_data['atoms']
                bonds = res_data['bonds']
                local_torsions = get_sp2_torsion_indices(atoms, bonds)
                if len(local_torsions) > 0:
                    global_torsions = local_torsions + atom_offset
                    all_sp2_torsions.append(global_torsions)
                    torsion_residue_idx.extend([residue_idx] * len(global_torsions))

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

    total_atoms = atom_features.shape[0]
    bond_dim = 26
    bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
    for global_i, global_j, feats in all_bonds_data:
        bond_features[global_i, global_j] = feats
        bond_features[global_j, global_i] = feats
    
    # Concatenate square matrices along the diagonal
    constraint_matrices = torch.block_diag(*all_constraint_matrices)

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


# ESM3 Extraction Functions
def extract_esm3_for_system(
    esm3_extractor: ESM3Extractor,
    pocket_residues: Dict,
    sequence_data: Dict[str, str],
    residue_keys: List[Tuple],
    system_dir: Optional[Path] = None,
) -> Optional[torch.Tensor]:
    '''
    Extract ESM3 embeddings for pocket residues.

    Args:
        esm3_extractor: ESM3Extractor instance
        pocket_residues: {chain_id: {(resname, resnum, entity_id): residue_data}}
        sequence_data: {chain_id: sequence_string}
        residue_keys: List of (chain_id, resname, resnum, entity_id) from pocket_features
        system_dir: Optional[Path] to the system directory
    Returns:
        Tensor [n_residues, 1536] ordered to match residue_keys
    '''
    pocket_embeddings = esm3_extractor.extract_pocket_embeddings(sequence_data, pocket_residues, system_dir=system_dir)

    # Reorder to match residue_keys
    embeddings = []
    for chain_id, resname, resnum, entity_id in residue_keys:
        key = (chain_id, resnum, entity_id)
        if key in pocket_embeddings:
            embeddings.append(pocket_embeddings[key])
        else:
            # Missing embedding - use zeros
            embeddings.append(torch.zeros(1536))

    if not embeddings:
        return None

    return torch.stack(embeddings, dim=0)


# UniMol Extraction Functions
def extract_unimol_for_system(
    unimol_extractor: UniMolExtractor,
    ligands: List[Dict],
    n_atoms_per_ligand: List[int],
) -> Optional[torch.Tensor]:
    '''
    Extract UniMol embeddings for ligands.

    Args:
        unimol_extractor: UniMolExtractor instance
        ligands: List of ligand dicts with 'smiles' and 'atoms' keys
        n_atoms_per_ligand: List of atom counts per ligand

    Returns:
        Tensor [total_atoms, embed_dim] concatenated for all ligands
    '''
    smiles_list = [lig['smiles'] for lig in ligands]
    atoms_list = [lig['atoms'] for lig in ligands]

    embeddings_list = unimol_extractor.extract_batch(smiles_list, atoms_list)

    all_embeddings = []
    for i, emb in enumerate(embeddings_list):
        if emb is None:
            embed_dim = 1536  # UniMol-1.1B atomic_repr dim
            all_embeddings.append(torch.zeros(n_atoms_per_ligand[i], embed_dim))
        else:
            all_embeddings.append(emb)

    if not all_embeddings:
        return None

    return torch.cat(all_embeddings, dim=0)


def featurize_single_system(system_dir: Path, cleanup_invalid: bool = True, skip_existing: bool = True) -> Tuple[str, bool, str, Optional[Dict]]:
    '''
    Featurize a single system directory.

    Expects:
        - ligands.pkl
        - pocket_residues.pkl
        - sequence_data.pkl

    Outputs:
        - ligand_features.pt
        - pocket_features.pt

    Returns:
        Tuple of (system_name, success, message, stats_dict)
        stats_dict contains: n_ligand_atoms, n_pocket_atoms, n_missing_atoms, removed
    '''
    system_name = system_dir.name
    stats = {'n_ligand_atoms': 0, 'n_pocket_atoms': 0, 'n_missing_atoms': 0, 'removed': False}

    ligands_path = system_dir / 'ligands.pkl'
    pocket_path = system_dir / 'pocket_residues.pkl'

    if not ligands_path.exists() or not pocket_path.exists():
        return system_name, False, "missing input files", stats

    try:
        ligands = load(ligands_path)
        pocket_residues = load(pocket_path)

        if len(ligands) < 1:
            if cleanup_invalid:
                shutil.rmtree(system_dir)
                stats['removed'] = True
                return system_name, False, "removed (no ligands)", stats
            else:
                return system_name, False, "no ligands", stats

        ligand_out = system_dir / 'ligand_features.pt'
        pocket_out = system_dir / 'pocket_features.pt'
        if skip_existing and ligand_out.exists() and pocket_out.exists():
            # Verify the current ordered ligand list, not just file existence.
            try:
                lig_data = torch.load(ligand_out, weights_only=False)
                pkt_data = torch.load(pocket_out, weights_only=False)
                validate_ligand_records(ligands, lig_data, system_name)
                stats['n_ligand_atoms'] = lig_data['atom_features'].shape[0]
                stats['n_pocket_atoms'] = pkt_data['atom_features'].shape[0]
                stats['n_missing_atoms'] = int(pkt_data['missing_mask'].sum().item())
                return system_name, True, "skipped (validated)", stats
            except Exception as exc:
                print(f'{system_name}: rebuilding graph features: {exc}')

        ligand_features = process_ligands(ligands)
        stats['n_ligand_atoms'] = ligand_features['atom_features'].shape[0]

        pocket_features = process_pocket_residues(pocket_residues)

        if pocket_features is None:
            if cleanup_invalid:
                shutil.rmtree(system_dir)
                stats['removed'] = True
                return system_name, False, "removed (no pocket residues)", stats
            else:
                return system_name, False, "no pocket residues", stats

        stats['n_pocket_atoms'] = pocket_features['atom_features'].shape[0]
        stats['n_missing_atoms'] = int(pocket_features['missing_mask'].sum().item())

        torch.save(ligand_features, ligand_out)
        torch.save(pocket_features, pocket_out)

        return system_name, True, "success", stats

    except Exception as e:
        return system_name, False, f"error: {e}", stats


def run_step1_graph_featurization(
    data_path: Path,
    n_jobs: int = -1,
    cleanup_invalid: bool = True,
    skip_existing: bool = True
) -> Tuple[FeaturizationStats, List[Path]]:
    '''Step 1: Graph featurization (parallel CPU).'''
    print("\n" + "="*60)
    print("STEP 1: Graph Featurization")
    print("="*60)

    system_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d / 'ligands.pkl').exists()
    ])[:]

    print(f"Found {len(system_dirs)} systems to process")
    if cleanup_invalid:
        print("Cleanup mode: will remove directories with no ligands or pocket residues")
    if not skip_existing:
        print("Force mode: will regenerate all feature files")

    results = Parallel(n_jobs=n_jobs)(
        delayed(featurize_single_system)(d, cleanup_invalid, skip_existing) for d in tqdm(system_dirs, desc="Graph features")
    )

    global_stats = FeaturizationStats()
    valid_dirs = []
    for name, success, msg, stats in results:
        if stats['removed']:
            global_stats.add_removed_dir(name)
        elif success or 'skipped' in msg:
            if stats['n_ligand_atoms'] > 0:
                global_stats.add_ligand_stats(stats['n_ligand_atoms'])
            if stats['n_pocket_atoms'] > 0:
                global_stats.add_pocket_stats(stats['n_pocket_atoms'], stats['n_missing_atoms'])
            valid_dirs.append(data_path / name)

    success = sum(1 for _, s, m, _ in results if s and 'success' in m)
    skipped = sum(1 for _, s, m, _ in results if s and 'skipped' in m)
    removed = sum(1 for _, _, m, _ in results if 'removed' in m)
    failed = sum(1 for _, s, m, _ in results if not s and 'removed' not in m)

    print(f"Step 1 Complete: Success={success}, Skipped={skipped}, Removed={removed}, Failed={failed}")

    failures = [(n, m) for n, s, m, _ in results if not s and 'removed' not in m]
    if failures:
        print(f"Failures: {len(failures)}")
        for name, msg in failures[:5]:
            print(f"  {name}: {msg}")
        with open(data_path / 'step1_errors.txt', 'w') as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")

    return global_stats, valid_dirs


def run_step2_esm3_extraction(
    data_path: Path,
    valid_dirs: List[Path],
    device: str = 'cuda',
    skip_existing: bool = True,
    strict: bool = False
) -> List[Path]:
    '''
    Step 2: ESM3 extraction (sequential GPU).

    Returns:
        List of directories that succeeded (for chaining to next step)
    '''
    print("\n" + "="*60)
    print("STEP 2: ESM3 Embedding Extraction")
    print("="*60)

    print("Loading ESM3 model...")
    esm3_extractor = ESM3Extractor(device=device)

    success_count = 0
    skip_count = 0
    error_count = 0
    removed_count = 0
    errors = []
    succeeded_dirs = []

    for system_dir in tqdm(valid_dirs, desc="ESM3 extraction"):
        system_name = system_dir.name
        esm3_out = system_dir / 'esm3_embeddings.pt'

        if skip_existing and esm3_out.exists():
            skip_count += 1
            succeeded_dirs.append(system_dir)
            continue

        try:
            pocket_residues = load(system_dir / 'pocket_residues.pkl')
            sequence_data = load(system_dir / 'sequence_data.pkl')
            pocket_features = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
            residue_keys = pocket_features['residue_keys']

            esm3_embeddings = extract_esm3_for_system(
                esm3_extractor, pocket_residues, sequence_data, residue_keys, system_dir=system_dir
            )

            if esm3_embeddings is not None:
                torch.save(esm3_embeddings, esm3_out)
                success_count += 1
                succeeded_dirs.append(system_dir)
            else:
                errors.append((system_name, "no ESM3 embeddings generated"))
                error_count += 1
                if strict:
                    shutil.rmtree(system_dir)
                    removed_count += 1

        except Exception as e:
            errors.append((system_name, str(e)))
            error_count += 1
            if strict:
                shutil.rmtree(system_dir)
                removed_count += 1

    print(f"Step 2 Complete: Success={success_count}, Skipped={skip_count}, Errors={error_count}", end="")
    if strict:
        print(f", Removed={removed_count}")
    else:
        print()

    if errors:
        with open(data_path / 'step2_esm3_errors.txt', 'w') as f:
            for name, msg in errors:
                f.write(f"{name}: {msg}\n")

    return succeeded_dirs


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ('out of memory' in msg) or ('cuda out of memory' in msg) or isinstance(exc, torch.cuda.OutOfMemoryError)


def _run_unimol_subbatch(unimol_extractor, sub_dirs, sub_smiles, sub_atoms,
                        sub_per_system_n_atoms, errors, pbar):
    '''
    Process one sub-batch. On CUDA OOM, split in half and recurse.
    On singleton OOM, give up on that one system (record error, no save).

    Side effects: writes unimol_embeddings.pt per successful system, updates
    errors list, advances pbar. Returns list of successful system_dirs.
    '''
    if not sub_dirs:
        return []

    try:
        embeddings_flat = unimol_extractor.extract_batch(sub_smiles, sub_atoms)
        expected_ligands = sum(len(counts) for counts in sub_per_system_n_atoms)
        if len(embeddings_flat) != expected_ligands:
            raise ValueError(f'Uni-Mol returned {len(embeddings_flat)} ligand tensors '
                             f'for {expected_ligands} ligand entities')
    except Exception as e:
        if _is_oom_error(e):
            torch.cuda.empty_cache()
            if len(sub_dirs) == 1:
                errors.append((sub_dirs[0].name, f"singleton OOM: {e}"))
                pbar.update(1)
                return []
            mid = len(sub_dirs) // 2
            # rebuild flat lists for each half by per-system slicing
            left_n = sub_per_system_n_atoms[:mid]
            right_n = sub_per_system_n_atoms[mid:]
            left_atom_total = sum(len(a) for a in left_n) if left_n else 0
            for x in left_n:  # left_n is list-of-list-of-int
                pass
            # Compute split point in flat arrays
            left_ligs = sum(len(n_list) for n_list in left_n)
            left_smiles = sub_smiles[:left_ligs]
            right_smiles = sub_smiles[left_ligs:]
            left_atoms = sub_atoms[:left_ligs]
            right_atoms = sub_atoms[left_ligs:]
            ok = []
            ok += _run_unimol_subbatch(unimol_extractor, sub_dirs[:mid], left_smiles, left_atoms,
                                        left_n, errors, pbar)
            ok += _run_unimol_subbatch(unimol_extractor, sub_dirs[mid:], right_smiles, right_atoms,
                                        right_n, errors, pbar)
            return ok
        # non-OOM error: mark all in sub-batch as errored; do NOT delete data
        for sd in sub_dirs:
            errors.append((sd.name, f"batch error: {e}"))
            pbar.update(1)
        return []

    # Distribute embeddings back to systems
    succeeded = []
    flat_idx = 0
    for sd, n_atoms_per_lig in zip(sub_dirs, sub_per_system_n_atoms):
        system_embeddings = embeddings_flat[flat_idx:flat_idx + len(n_atoms_per_lig)]
        flat_idx += len(n_atoms_per_lig)
        try:
            embeds = []
            for n, emb in zip(n_atoms_per_lig, system_embeddings):
                if emb is None:
                    embeds.append(torch.zeros(n, 1536))
                elif emb.ndim != 2 or emb.shape[0] != n:
                    raise ValueError(f'ligand entity expects {n} embedding rows; '
                                     f'got tensor shape {tuple(emb.shape)}')
                else:
                    embeds.append(emb)
            full = torch.cat(embeds, dim=0) if embeds else None
            if full is None:
                raise ValueError("no embeddings produced")
            ligand_features = torch.load(sd / 'ligand_features.pt', weights_only=False)
            ligands = load(sd / 'ligands.pkl')
            save_unimol_cache(sd, ligand_features, ligands, full)
            succeeded.append(sd)
        except Exception as e:
            errors.append((sd.name, str(e)))
        pbar.update(1)
    return succeeded


def run_step3_unimol_extraction(
    data_path: Path,
    valid_dirs: List[Path],
    model_size: str = '1.1B',
    batch_size: int = 16,
    skip_existing: bool = True,
    strict: bool = False
) -> List[Path]:
    '''
    Step 3: UniMol extraction (batched GPU).

    Step-3 strict mode is intentionally NON-destructive: failed UniMol systems
    keep their pocket/ligand/esm3 artifacts and are excluded from succeeded_dirs.
    The loader rejects incomplete/inconsistent data rather than guessing rows. This avoids the
    batch-OOM-wipe failure mode where 1 OOM destroyed N-1 innocent systems.

    On CUDA OOM, the offending batch is split in half and retried recursively
    down to size 1. Truly OOM-at-singleton systems are logged but kept.
    '''
    print("\n" + "="*60)
    print("STEP 3: UniMol Embedding Extraction")
    print("="*60)

    items_to_process = []
    skip_count = 0
    succeeded_dirs = []

    for system_dir in valid_dirs:
        unimol_out = system_dir / 'unimol_embeddings.pt'
        if skip_existing and unimol_out.exists():
            try:
                features = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
                ligands = load(system_dir / 'ligands.pkl')
                reason = unimol_cache_reason(system_dir, features, ligands)
            except Exception as exc:
                reason = str(exc)
            if reason is None:
                skip_count += 1
                succeeded_dirs.append(system_dir)
                continue
            print(f'{system_dir.name}: regenerating Uni-Mol cache: {reason}')
        items_to_process.append(system_dir)

    print(f"Items to process: {len(items_to_process)}, Skipped: {skip_count}, batch_size={batch_size}")

    if not items_to_process:
        return succeeded_dirs
    print(f"Loading UniMol model ({model_size})...")
    unimol_extractor = UniMolExtractor(model_size=model_size)

    success_count = 0
    error_count = 0
    errors = []

    pbar = tqdm(total=len(items_to_process), desc="UniMol extraction (batched)")
    for batch_start in range(0, len(items_to_process), batch_size):
        batch_dirs = items_to_process[batch_start:batch_start + batch_size]

        flat_smiles = []
        flat_atoms = []
        per_system_n_atoms = []
        loaded_dirs = []
        for system_dir in batch_dirs:
            try:
                ligands = load(system_dir / 'ligands.pkl')
                ligand_features = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
                n_atoms_per_ligand = list(ligand_features['n_atoms_per_ligand'])
                validate_ligand_records(ligands, ligand_features, system_dir.name)
            except Exception as e:
                errors.append((system_dir.name, f"load error: {e}"))
                error_count += 1
                pbar.update(1)
                continue
            loaded_dirs.append(system_dir)
            per_system_n_atoms.append(n_atoms_per_ligand)
            for lig in ligands:
                flat_smiles.append(lig['smiles'])
                flat_atoms.append(lig['atoms'])

        ok = _run_unimol_subbatch(
            unimol_extractor, loaded_dirs, flat_smiles, flat_atoms,
            per_system_n_atoms, errors, pbar,
        )
        success_count += len(ok)
        succeeded_dirs.extend(ok)

    pbar.close()
    error_count = len(errors)
    print(f"Step 3 Complete: Success={success_count}, Skipped={skip_count}, Errors={error_count} (no rmtree)")

    if errors:
        with open(data_path / 'step3_unimol_errors.txt', 'w') as f:
            for name, msg in errors:
                f.write(f"{name}: {msg}\n")

    return succeeded_dirs


def main(
    data_dir: str = 'data',
    n_jobs: int = -1,
    skip_existing: bool = True,
    cleanup_invalid: bool = True,
    run_esm3: bool = True,
    run_unimol: bool = True,
    unimol_size: str = '1.1B',
    device: str = 'cuda',
    strict: bool = False
):
    data_path = Path(data_dir)

    if strict:
        print("STRICT MODE: Directories failing ESM3/UniMol extraction will be removed")

    # Step 1: Graph featurization
    global_stats, valid_dirs = run_step1_graph_featurization(
        data_path, n_jobs, cleanup_invalid, skip_existing
    )

    # Step 2: ESM3 extraction
    if run_esm3:
        valid_dirs = run_step2_esm3_extraction(
            data_path, valid_dirs, device, skip_existing, strict=strict
        )

    # Step 3: UniMol extraction
    if run_unimol:
        valid_dirs = run_step3_unimol_extraction(
            data_path, valid_dirs, unimol_size, skip_existing=skip_existing, strict=strict
        )

    global_stats.print_summary()
    print(f"Final valid directories: {len(valid_dirs)}")

    import json
    stats_summary = global_stats.summarize()
    with open(data_path / 'featurize_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Statistics saved to: {data_path / 'featurize_stats.json'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root holding one parsed system directory per system')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs for Step 1')
    parser.add_argument('--no_skip', action='store_true', help='Reprocess even if features exist')
    parser.add_argument('--no_cleanup', action='store_true', help='Do not remove invalid directories')
    parser.add_argument('--no_esm3', action='store_true', help='Skip ESM3 extraction')
    parser.add_argument('--no_unimol', action='store_true', help='Skip UniMol extraction')
    parser.add_argument('--unimol_size', type=str, default='1.1B', choices=['570m', '1.1B'])
    parser.add_argument('--device', type=str, default='cuda', help='Device for GPU models')
    parser.add_argument('--strict', action='store_true',
                        help='Remove directories that fail ESM3/UniMol extraction')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        n_jobs=args.n_jobs,
        skip_existing=not args.no_skip,
        cleanup_invalid=not args.no_cleanup,
        run_esm3=not args.no_esm3,
        run_unimol=not args.no_unimol,
        unimol_size=args.unimol_size,
        device=args.device,
        strict=args.strict
    )
