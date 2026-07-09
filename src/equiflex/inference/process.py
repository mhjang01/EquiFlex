'''
Inference data processing for Flex model.

Processes input PDB/CIF + reference ligand SDF + target SMILES into model-ready tensors.
Reference ligand defines the pocket location; target SMILES is the molecule to dock.

Usage:
    processor = InferenceProcessor(device='cuda')
    batch = processor.process(
        protein_path='protein.cif',
        ref_ligand_path='reference.sdf',
        target_smiles=['CCO', 'c1ccccc1'],
    )
    # batch is ready for model.forward()
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import numpy as np
import string
import random

from rdkit import Chem
from rdkit.Chem import AllChem

from equiflex.prep.parse import (
    parse_structure,
    extract_ligand_data,
    extract_atom_data,
    extract_bond_data,
    extract_coordinates,
    get_pocket_residues,
    overwrite_coords_on_reference_data,
)
from equiflex.prep.featurize import (
    featurize_atoms,
    featurize_bonds,
    ESM3Extractor,
    UniMolExtractor,
)
from equiflex.prep.const import PROTEIN_BACKBONE_ATOM_NAMES

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()


def generate_run_id(prefix: str = 'run') -> str:
    '''Generate a random run ID like run_abc123.'''
    chars = string.ascii_lowercase + string.digits
    suffix = ''.join(random.choices(chars, k=6))
    return f'{prefix}_{suffix}'


def get_default_output_dir(run_id: Optional[str] = None) -> Path:
    '''Get default output directory under inference/.tmp/{run_id}.'''
    if run_id is None:
        run_id = generate_run_id()
    return SCRIPT_DIR / '.tmp' / run_id


def extract_ligand_from_smiles(smiles: str, lig_idx: int = 0) -> Optional[Dict]:
    '''
    Create ligand data from SMILES string.
    Generates 3D conformer using RDKit.

    Args:
        smiles: SMILES string
        lig_idx: Ligand index for naming

    Returns:
        Ligand data dictionary compatible with pipeline
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
        return None

    # Add hydrogens and generate 3D conformer
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        # Try with random coordinates if ETKDG fails
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if result == -1:
            print(f"Failed to generate 3D conformer for: {smiles}")
            return None

    # Optimize geometry
    AllChem.MMFFOptimizeMolecule(mol)

    # Remove hydrogens for consistency with training data
    mol = Chem.RemoveHs(mol)

    # Extract atom data with canonical naming
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    atoms = []
    for atom in mol.GetAtoms():
        name = f"{atom.GetSymbol().upper()}{canonical_ranks[atom.GetIdx()]+1}"
        atom_data = extract_atom_data(atom, name)
        atoms.append(atom_data)

    # Extract bond data
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]

    # Extract coordinates
    coords = extract_coordinates(mol)

    return {
        'chain': f'L{lig_idx}',
        'smiles': Chem.MolToSmiles(mol, canonical=True),
        'atoms': atoms,
        'bonds': bonds,
        'coords': coords,
    }


class InferenceProcessor:
    '''
    Processes protein-ligand complex data for inference.

    Handles:
    1. Parsing protein (CIF) and reference ligand (SDF) files
    2. Extracting pocket residues using reference ligand location
    3. Generating 3D conformers for target SMILES
    4. Featurizing atoms and bonds
    5. Extracting ESM3 and UniMol embeddings
    6. Building model-ready batch tensors
    '''

    def __init__(
        self,
        device: str = 'cuda',
        esm_model: str = 'esm3_sm_open_v1',
        unimol_size: str = '1.1B',
        pocket_distance: float = 4.0,
        lazy_load_models: bool = True,
    ):
        self.device = device
        self.esm_model = esm_model
        self.unimol_size = unimol_size
        self.pocket_distance = pocket_distance
        self.lazy_load_models = lazy_load_models

        self._esm3_extractor = None
        self._unimol_extractor = None

    @property
    def esm3_extractor(self) -> ESM3Extractor:
        if self._esm3_extractor is None:
            self._esm3_extractor = ESM3Extractor(device=self.device)
        return self._esm3_extractor

    @property
    def unimol_extractor(self) -> UniMolExtractor:
        if self._unimol_extractor is None:
            self._unimol_extractor = UniMolExtractor(model_size=self.unimol_size)
        return self._unimol_extractor

    def process(
        self,
        protein_path: Union[str, Path],
        ref_ligand_path: Union[str, Path],
        target_smiles: Union[str, List[str]],
    ) -> Dict[str, torch.Tensor]:
        '''
        Process protein-ligand complex for inference.

        Args:
            protein_path: Path to protein CIF/PDB file
            ref_ligand_path: Path to reference ligand SDF (defines pocket location)
            target_smiles: SMILES string(s) for target ligand(s) to dock

        Returns:
            Dictionary of tensors (unbatched) ready for batching in inference loop
        '''
        protein_path = Path(protein_path)
        ref_ligand_path = Path(ref_ligand_path)
        if isinstance(target_smiles, str):
            target_smiles = [target_smiles]

        # Parse protein
        protein_data = parse_structure(protein_path)
        
        # Parse reference ligand to define pocket
        ref_ligand_data = extract_ligand_data(ref_ligand_path)
        if ref_ligand_data is None:
            raise ValueError(f"Failed to parse reference ligand: {ref_ligand_path}")

        # Extract pocket residues using reference ligand location
        pocket_data, sequence_data = get_pocket_residues(protein_data, [ref_ligand_data])
        
        print(sequence_data)

        # Overwrite pocket coords on reference templates
        pocket_residues = overwrite_coords_on_reference_data(pocket_data)

        # Generate target ligand data from SMILES
        target_ligand_list = [
            extract_ligand_from_smiles(smi, idx)
            for idx, smi in enumerate(target_smiles)
        ]
        target_ligand_list = [ld for ld in target_ligand_list if ld is not None]

        if not target_ligand_list:
            raise ValueError("No valid target ligands found")

        # Featurize
        ligand_features = self._process_ligands(target_ligand_list)
        pocket_features = self._process_pocket(pocket_residues)

        if pocket_features is None:
            raise ValueError("No pocket residues found")

        # Extract embeddings
        esm_embeddings = self._extract_esm3(
            pocket_residues, sequence_data, pocket_features['residue_keys']
        )
        unimol_embeddings = self._extract_unimol(
            target_ligand_list, ligand_features['n_atoms_per_ligand']
        )

        # Build output tensors (unbatched)
        data = self._build_data(
            ligand_features=ligand_features,
            pocket_features=pocket_features,
            esm_embeddings=esm_embeddings,
            unimol_embeddings=unimol_embeddings,
        )

        return data

    def _process_ligands(self, ligands: List[Dict]) -> Dict:
        '''Process ligands into feature tensors.'''
        all_atom_features = []
        all_coords = []
        all_ref_coords = []
        ligand_uid_vec = []
        n_atoms_per_ligand = []
        atom_offset = 0
        all_bonds_data = []

        for lig_idx, lig_data in enumerate(ligands):
            # Atom features
            atom_feats = []
            for atom in lig_data['atoms']:
                feat_dict = featurize_atoms(atom)
                atom_feats.append(feat_dict['features'])
            atom_feats = torch.stack(atom_feats, dim=0)
            all_atom_features.append(atom_feats)

            # Coordinates (ground truth = input coordinates for inference)
            coords = torch.tensor(lig_data['coords'], dtype=torch.float32)
            all_coords.append(coords)

            # Reference coordinates (use input coords as reference for inference)
            n_atoms = len(lig_data['atoms'])
            all_ref_coords.append(coords.clone())

            # UID vector
            ligand_uid_vec.extend([lig_idx] * n_atoms)

            # Bonds with global indices
            for bond in lig_data['bonds']:
                feat_dict = featurize_bonds(bond)
                global_i = feat_dict['begin_atom_idx'] + atom_offset
                global_j = feat_dict['end_atom_idx'] + atom_offset
                all_bonds_data.append((global_i, global_j, feat_dict['features']))

            n_atoms_per_ligand.append(n_atoms)
            atom_offset += n_atoms

        # Stack features
        atom_features = torch.cat(all_atom_features, dim=0)
        coords = torch.cat(all_coords, dim=0)
        ref_coords = torch.cat(all_ref_coords, dim=0)
        ligand_uid_vec = torch.tensor(ligand_uid_vec, dtype=torch.long)

        # Build bond matrix
        total_atoms = atom_features.shape[0]
        bond_dim = 26
        bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
        for global_i, global_j, feats in all_bonds_data:
            bond_features[global_i, global_j] = feats
            bond_features[global_j, global_i] = feats

        return {
            'atom_features': atom_features,
            'bond_features': bond_features,
            'coords': coords,
            'ref_coords': ref_coords,
            'ligand_uid_vec': ligand_uid_vec,
            'n_atoms_per_ligand': n_atoms_per_ligand,
        }

    def _process_pocket(self, pocket_residues: Dict) -> Optional[Dict]:
        '''Process pocket residues into feature tensors.'''
        all_atom_features = []
        all_coords = []
        all_ref_coords = []
        all_backbone_masks = []
        residue_uid_vec = []
        residue_keys = []
        n_atoms_per_residue = []
        atom_offset = 0
        all_bonds_data = []
        residue_idx = 0

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

                # Backbone mask
                backbone_mask = torch.tensor(
                    [1.0 if atom['name'] in PROTEIN_BACKBONE_ATOM_NAMES else 0.0 for atom in res_data['atoms']],
                    dtype=torch.float32
                )
                all_backbone_masks.append(backbone_mask)

                # UID vector
                residue_uid_vec.extend([residue_idx] * n_atoms)

                # Bonds with global indices
                for bond in res_data['bonds']:
                    feat_dict = featurize_bonds(bond)
                    global_i = feat_dict['begin_atom_idx'] + atom_offset
                    global_j = feat_dict['end_atom_idx'] + atom_offset
                    all_bonds_data.append((global_i, global_j, feat_dict['features']))

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
        backbone_mask = torch.cat(all_backbone_masks, dim=0)
        residue_uid_vec = torch.tensor(residue_uid_vec, dtype=torch.long)

        # Build bond matrix
        total_atoms = atom_features.shape[0]
        bond_dim = 26
        bond_features = torch.zeros(total_atoms, total_atoms, bond_dim)
        for global_i, global_j, feats in all_bonds_data:
            bond_features[global_i, global_j] = feats
            bond_features[global_j, global_i] = feats

        return {
            'atom_features': atom_features,
            'bond_features': bond_features,
            'coords': coords,
            'ref_coords': ref_coords,
            'backbone_mask': backbone_mask,
            'residue_uid_vec': residue_uid_vec,
            'residue_keys': residue_keys,
            'n_atoms_per_residue': n_atoms_per_residue,
        }

    def _extract_esm3(
        self,
        pocket_residues: Dict,
        sequence_data: Dict[str, str],
        residue_keys: List,
    ) -> torch.Tensor:
        '''Extract ESM3 embeddings for pocket residues.'''
        pocket_embeddings = self.esm3_extractor.extract_pocket_embeddings(
            sequence_data, pocket_residues
        )

        embeddings = []
        for chain_id, resname, resnum, entity_id in residue_keys:
            key = (chain_id, resnum, entity_id)
            if key in pocket_embeddings:
                embeddings.append(pocket_embeddings[key])
            else:
                embeddings.append(torch.zeros(1536))

        return torch.stack(embeddings, dim=0)

    def _extract_unimol(
        self,
        ligands: List[Dict],
        n_atoms_per_ligand: List[int],
    ) -> torch.Tensor:
        '''Extract UniMol embeddings for ligands.'''
        smiles_list = [lig['smiles'] for lig in ligands]
        atoms_list = [lig['atoms'] for lig in ligands]

        embeddings_list = self.unimol_extractor.extract_batch(smiles_list, atoms_list)

        all_embeddings = []
        for i, emb in enumerate(embeddings_list):
            if emb is None:
                embed_dim = 512
                all_embeddings.append(torch.zeros(n_atoms_per_ligand[i], embed_dim))
            else:
                all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

    def _build_data(
        self,
        ligand_features: Dict,
        pocket_features: Dict,
        esm_embeddings: torch.Tensor,
        unimol_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        '''Build unbatched tensors for inference. Batching handled in inference loop.'''
        N_rec = pocket_features['atom_features'].shape[0]
        N_lig = ligand_features['atom_features'].shape[0]
        N_rec_tokens = esm_embeddings.shape[0]

        # Compute backbone COM for centering
        backbone_mask = pocket_features['backbone_mask']
        rec_coords = pocket_features['coords']
        missing_mask = pocket_features.get('missing_mask')
        if missing_mask is None:
            missing_mask = torch.zeros(N_rec, dtype=torch.bool)
        elif not isinstance(missing_mask, torch.Tensor):
            missing_mask = torch.tensor(missing_mask, dtype=torch.bool)
        else:
            missing_mask = missing_mask.bool()

        valid_backbone_mask = backbone_mask * (~missing_mask).float()
        if valid_backbone_mask.sum() > 0:
            backbone_com = (
                (rec_coords * valid_backbone_mask.unsqueeze(-1)).sum(dim=0)
                / valid_backbone_mask.sum()
            )
        else:
            backbone_com = rec_coords.mean(dim=0)

        # Center coordinates by backbone COM
        rec_coords_centered = rec_coords - backbone_com
        lig_coords_centered = ligand_features['coords'] - backbone_com

        data = {
            # Receptor - unbatched [N_rec, ...]
            'rec_atom_feats': pocket_features['atom_features'],  # [N_rec, 124]
            'rec_atom_coords': rec_coords_centered,  # [N_rec, 3] - centered ground truth
            'rec_atom_ref_coords': pocket_features['ref_coords'],  # [N_rec, 3] - RDKit conformer (NOT centered, only for distograms)
            'rec_bond_feats': pocket_features['bond_features'],  # [N_rec, N_rec, 26]
            'rec_uid_vec': pocket_features['residue_uid_vec'],  # [N_rec]
            'rec_backbone_mask': backbone_mask,  # [N_rec] - 1 for backbone, 0 for sidechain
            'rec_missing_mask': missing_mask,  # [N_rec] - True for missing/unknown coordinates
            'esm_embeddings': esm_embeddings,  # [N_rec_tokens, 1536]

            # Ligand - unbatched [N_lig, ...]
            'lig_atom_feats': ligand_features['atom_features'],  # [N_lig, 124]
            'lig_atom_coords': lig_coords_centered,  # [N_lig, 3]
            'lig_atom_ref_coords': ligand_features['ref_coords'],  # [N_lig, 3]
            'lig_bond_feats': ligand_features['bond_features'],  # [N_lig, N_lig, 26]
            'lig_uid_vec': ligand_features['ligand_uid_vec'],  # [N_lig]
            'unimol_embeddings': unimol_embeddings,  # [N_lig, 1536]

            # Metadata for post-processing and batching
            '_backbone_com': backbone_com,
            '_n_rec': N_rec,
            '_n_lig': N_lig,
            '_n_rec_tokens': N_rec_tokens,
        }

        return data

    def to_device(self, batch: Dict[str, torch.Tensor], device: str = None) -> Dict[str, torch.Tensor]:
        '''Move batch tensors to device.'''
        if device is None:
            device = self.device

        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            else:
                result[k] = v
        return result


def process_single_complex(
    protein_path: Union[str, Path],
    ref_ligand_path: Union[str, Path],
    target_smiles: Union[str, List[str]],
    output_dir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    device: str = 'cuda',
) -> Path:
    '''
    Process a single complex and save to output directory.

    Args:
        protein_path: Path to protein CIF file
        ref_ligand_path: Path to reference ligand SDF (defines pocket)
        target_smiles: SMILES string(s) for target ligand(s)
        output_dir: Directory to save processed tensors (default: inference/.tmp/{run_id})
        run_id: Run identifier (auto-generated if None)
        device: Device for embedding extraction

    Returns:
        Path to output directory
    '''
    # Handle output directory
    if output_dir is None:
        output_dir = get_default_output_dir(run_id)
    else:
        output_dir = Path(output_dir)
        if run_id is not None:
            output_dir = output_dir / run_id
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")
        return output_dir
    else:
        processor = InferenceProcessor(device=device)
        data = processor.process(
            protein_path=protein_path,
            ref_ligand_path=ref_ligand_path,
            target_smiles=target_smiles,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors (excluding metadata keys starting with _)
        tensors_to_save = {k: v for k, v in data.items() if not k.startswith('_')}
        torch.save(tensors_to_save, output_dir / 'inference_data.pt')

        # Save metadata separately
        metadata = {k: v for k, v in data.items() if k.startswith('_')}
        if metadata:
            torch.save(metadata, output_dir / 'metadata.pt')

        return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process protein-ligand complex for inference')
    parser.add_argument('--protein', type=str, required=True,
                        help='Path to protein CIF/PDB file')
    parser.add_argument('--ref-ligand', type=str, required=True,
                        help='Path to reference ligand SDF (defines pocket location)')
    parser.add_argument('--smiles', type=str, nargs='+', required=True,
                        help='Target SMILES string(s) to dock')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: inference/.tmp/{run_id})')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run identifier (auto-generated if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for embedding extraction')

    args = parser.parse_args()

    output_path = process_single_complex(
        protein_path=args.protein,
        ref_ligand_path=args.ref_ligand,
        target_smiles=args.smiles,
        output_dir=args.output,
        run_id=args.run_id,
        device=args.device,
    )
    print(f"Processed data saved to: {output_path}")
