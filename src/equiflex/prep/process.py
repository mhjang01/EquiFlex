"""Assembly of parsed + featurized data into tensor dicts.

Converts per-atom/bond feature dicts into stacked tensors suitable for
model input. Used by both data preparation scripts and the inference pipeline.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from .featurize import featurize_atoms, featurize_bonds
from .parse import get_sp2_torsion_indices
from .const import PROTEIN_BACKBONE_ATOM_NAMES
from .validation import SIGNATURE_KEY, ligand_atom_signature

_BACKBONE_ATOMS = set(PROTEIN_BACKBONE_ATOM_NAMES)
_CB_ATOM_NAME = "CB"


def process_ligands(ligands: List[Dict]) -> Dict:
    """Process parsed ligand dicts into stacked feature tensors."""
    all_atom_features = []
    all_coords = []
    all_ref_coords = []
    all_constraint_matrices = []
    ligand_uid_vec = []
    smiles_list = []
    n_atoms_per_ligand = []
    all_sp2_torsions = []
    all_chiral_atoms = []
    torsion_ligand_idx = []
    atom_offset = 0
    all_bonds_data = []

    for lig_idx, lig_data in enumerate(ligands):
        atom_feats = []
        for atom in lig_data['atoms']:
            feat_dict = featurize_atoms(atom)
            atom_feats.append(feat_dict['features'])
        atom_feats = torch.stack(atom_feats, dim=0)
        all_atom_features.append(atom_feats)

        coords = torch.tensor(lig_data['coords'], dtype=torch.float32)
        all_coords.append(coords)

        n_atoms = len(lig_data['atoms'])
        if 'ref_coords' in lig_data and lig_data['ref_coords'] is not None:
            ref_coords = torch.tensor(lig_data['ref_coords'], dtype=torch.float32)
        else:
            ref_coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
        all_ref_coords.append(ref_coords)

        ligand_uid_vec.extend([lig_idx] * n_atoms)

        for bond in lig_data['bonds']:
            feat_dict = featurize_bonds(bond)
            global_i = feat_dict['begin_atom_idx'] + atom_offset
            global_j = feat_dict['end_atom_idx'] + atom_offset
            all_bonds_data.append((global_i, global_j, feat_dict['features']))

        if 'sp2_torsion_indices' in lig_data and len(lig_data['sp2_torsion_indices']) > 0:
            local_torsions = lig_data['sp2_torsion_indices']
            if isinstance(local_torsions, np.ndarray):
                global_torsions = local_torsions + atom_offset
            else:
                global_torsions = np.array(local_torsions) + atom_offset
            all_sp2_torsions.append(global_torsions)
            torsion_ligand_idx.extend([lig_idx] * len(global_torsions))

        if 'chiral_atom_indices' in lig_data and len(lig_data['chiral_atom_indices']) > 0:
            local_chiral = np.asarray(lig_data['chiral_atom_indices']).copy()
            # cols 0:4 are atom indices; col 4 is the R/S flag and must not shift
            local_chiral[:, :4] += atom_offset
            all_chiral_atoms.append(local_chiral)

        if 'constraint_matrix' in lig_data and lig_data['constraint_matrix'] is not None:
            constraint_matrix = torch.tensor(lig_data['constraint_matrix'], dtype=torch.float32)
            all_constraint_matrices.append(constraint_matrix)
        else:
            raise ValueError("Missing ligand constraint matrix")

        smiles_list.append(lig_data['smiles'])
        n_atoms_per_ligand.append(n_atoms)
        atom_offset += n_atoms

    atom_features = torch.cat(all_atom_features, dim=0)
    coords = torch.cat(all_coords, dim=0)
    ref_coords = torch.cat(all_ref_coords, dim=0)
    ligand_uid_vec = torch.tensor(ligand_uid_vec, dtype=torch.long)

    total_atoms = atom_features.shape[0]
    bond_dim = 26

    # Block-diagonal constraint matrix across ligands (each is [n_i, n_i]).
    # cat(dim=0) is wrong for >1 ligand (e.g. ligand + ion) — shapes mismatch.
    constraint_matrix = torch.zeros(total_atoms, total_atoms, dtype=torch.float32)
    off = 0
    for cm in all_constraint_matrices:
        k = cm.shape[0]
        constraint_matrix[off:off + k, off:off + k] = cm
        off += k
    bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
    for global_i, global_j, feats in all_bonds_data:
        bond_features[global_i, global_j] = feats
        bond_features[global_j, global_i] = feats

    if all_chiral_atoms:
        chiral_atom_indices = torch.tensor(np.concatenate(all_chiral_atoms), dtype=torch.long)
    else:
        chiral_atom_indices = torch.empty((0, 5), dtype=torch.long)

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
        'chiral_atom_indices': chiral_atom_indices,
        'torsion_ligand_idx': torsion_ligand_idx,
        'constraint_matrix': constraint_matrix,
    }


def process_pocket_residues(pocket_residues: Dict) -> Optional[Dict]:
    """Process pocket residue dicts into stacked feature tensors.

    Input format: ``{chain_id: {(resname, resnum, entity_id): residue_dict}}``.
    """
    all_atom_features = []
    all_coords = []
    all_ref_coords = []
    all_missing_masks = []
    all_backbone_masks = []
    all_cb_masks = []
    residue_uid_vec = []
    residue_keys = []
    overlay_residue_keys: List[Tuple[str, str, int, str]] = []
    n_atoms_per_residue = []
    all_sp2_torsions = []
    torsion_residue_idx = []
    all_constraint_matrices = []
    atom_offset = 0
    all_bonds_data = []
    residue_idx = 0
    # Per-residue backbone atom indices (global, for CB placement)
    residue_n_idx = []   # index of N atom in flat array (-1 if missing)
    residue_ca_idx = []  # index of CA atom
    residue_c_idx = []   # index of C atom
    residue_is_gly = []  # whether residue is GLY (no CB)

    for chain_id, residues in sorted(pocket_residues.items()):
        for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
            atom_feats = []
            for atom in res_data['atoms']:
                feat_dict = featurize_atoms(atom)
                atom_feats.append(feat_dict['features'])
            atom_feats = torch.stack(atom_feats, dim=0)
            all_atom_features.append(atom_feats)

            coords = torch.tensor(res_data['coords'], dtype=torch.float32)
            all_coords.append(coords)

            n_atoms = len(res_data['atoms'])
            if 'ref_coords' in res_data and res_data['ref_coords'] is not None:
                ref_coords = torch.tensor(res_data['ref_coords'], dtype=torch.float32)
            else:
                ref_coords = coords.clone()
            all_ref_coords.append(ref_coords)

            missing_mask = torch.tensor(res_data['missing_masks'], dtype=torch.bool)
            all_missing_masks.append(missing_mask)

            backbone_mask = torch.tensor(
                [1.0 if atom['name'] in _BACKBONE_ATOMS else 0.0 for atom in res_data['atoms']],
                dtype=torch.float32,
            )
            all_backbone_masks.append(backbone_mask)

            cb_mask = torch.tensor(
                [1.0 if atom['name'] == _CB_ATOM_NAME else 0.0 for atom in res_data['atoms']],
                dtype=torch.float32,
            )
            all_cb_masks.append(cb_mask)

            # Per-residue backbone atom indices (global) for CB placement
            atom_names = [atom['name'] for atom in res_data['atoms']]
            residue_n_idx.append(atom_offset + atom_names.index('N') if 'N' in atom_names else -1)
            residue_ca_idx.append(atom_offset + atom_names.index('CA') if 'CA' in atom_names else -1)
            residue_c_idx.append(atom_offset + atom_names.index('C') if 'C' in atom_names else -1)
            residue_is_gly.append(res_data.get('aa_code', '') == 'GLY')

            residue_uid_vec.extend([residue_idx] * n_atoms)

            for bond in res_data['bonds']:
                feat_dict = featurize_bonds(bond)
                global_i = feat_dict['begin_atom_idx'] + atom_offset
                global_j = feat_dict['end_atom_idx'] + atom_offset
                all_bonds_data.append((global_i, global_j, feat_dict['features']))

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

            if 'constraint_matrix' in res_data and res_data['constraint_matrix'] is not None:
                constraint_matrix = torch.tensor(res_data['constraint_matrix'], dtype=torch.float32)
                all_constraint_matrices.append(constraint_matrix)
            else:
                raise ValueError("Missing pocket constraint matrix")

            residue_keys.append((chain_id, resname, resnum, entity_id))
            # res_data is often reference-template dict + coords (no ``resnum`` key);
            # use tuple key ``resnum`` and optional ``pdb_resnum`` from parse.
            struct_num = int(res_data.get('pdb_resnum', resnum))
            icode_raw = res_data.get('pdb_icode', '') or ''
            icode = icode_raw.strip() if isinstance(icode_raw, str) else str(icode_raw).strip()
            overlay_residue_keys.append(
                (str(chain_id).upper(), str(resname).upper(), struct_num, icode),
            )
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

    # Per-residue backbone atom indices for CB placement
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
        'overlay_residue_keys': overlay_residue_keys,
        'n_atoms_per_residue': n_atoms_per_residue,
        'sp2_torsion_indices': sp2_torsion_indices,
        'torsion_residue_idx': torsion_residue_idx,
        'constraint_matrix': constraint_matrices,
    }
