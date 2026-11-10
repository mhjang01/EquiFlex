import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from .const import *
from rdkit import Chem
from rdkit.Chem import AllChem


# Basic Encoding Functions #

def atom_name_to_ascii(name: str, max_len: int = 4) -> torch.Tensor:
    '''Convert atom name to ASCII codes, padded/truncated to max_len.'''
    name = name.strip().upper()[:max_len]
    ascii_codes = [ord(c) for c in name]
    # Pad with spaces (ASCII 32)
    while len(ascii_codes) < max_len:
        ascii_codes.append(32)
    return torch.tensor(ascii_codes, dtype=torch.long)


def encode_atom_name_chars(name: str, max_len: int = 4, num_bins: int = 64) -> torch.Tensor:
    '''
    Boltz-style atom name encoding: 4 chars × 64 one-hot = 256 dims.
    ASCII values are taken modulo num_bins to fit in the encoding space.
    '''
    ascii_codes = atom_name_to_ascii(name, max_len)
    one_hots = F.one_hot(ascii_codes % num_bins, num_classes=num_bins)
    return one_hots.flatten().float()  # [256]


def one_hot_encode(value, categories: List[Any]) -> torch.Tensor:
    
    one_hot = torch.zeros(len(categories))
    if value in categories:
        one_hot[categories.index(value)] = 1
        return one_hot
    else:
        # Return all zeros
        return one_hot

def gaussian_rbf(value: float,
                 mean_points: List[float],
                 gamma: float) -> torch.Tensor:

    '''
    Gaussian radial basis function (RBF) encoding.
    Args:
        value: The value to encode.
        mean_points: The mean points of the RBF.
        gamma: The gamma parameter of the RBF.
    Returns:
        The encoded value.
    '''
    # Handle None or non-numeric values
    if value is None or not isinstance(value, (int, float)):
        return torch.zeros(len(mean_points), dtype=torch.float32)

    mean_tensor = torch.tensor(mean_points, dtype=torch.float32)
    return torch.exp(-gamma * (float(value) - mean_tensor) ** 2)

# Chemistry Features #

def encode_neighbor_composition(neighbor_elements: Dict[str, int]) -> torch.Tensor:
    '''
    One-hot the bonded heavy-neighbor count per element class.
    NEIGHBOR_ELEMENT_CLASSES (7) x NEIGHBOR_COUNT_BINS (5, last = "4 or more") = 35 dim.
    '''
    parts = []
    for cls in NEIGHBOR_ELEMENT_CLASSES:
        count = min(neighbor_elements.get(cls, 0), NEIGHBOR_COUNT_BINS[-1])
        parts.append(one_hot_encode(count, NEIGHBOR_COUNT_BINS))
    return torch.cat(parts)  # 35 dim


def encode_morgan_bits(morgan_bits: List[int]) -> torch.Tensor:
    '''Multi-hot the per-atom Morgan environment bits. MORGAN_NBITS dim.'''
    vec = torch.zeros(MORGAN_NBITS, dtype=torch.float32)
    if morgan_bits:
        idx = torch.tensor([b for b in morgan_bits if 0 <= b < MORGAN_NBITS], dtype=torch.long)
        vec[idx] = 1.0
    return vec


def featurize_atoms(atom: List[Dict]) -> torch.Tensor:

    # Metadata
    symbol = atom['symbol']
    name = atom['name']
    index = atom['index']

    features = [
        # --- Discriminating context block: 172 dim (replaces old 256 ASCII name) ---
        # Protein heavy-atom-name identity (ligand atoms -> all-zero): 37 dim
        one_hot_encode(name, PROTEIN_ATOM_NAMES),
        # Source flag: 1 if this atom has no protein name (i.e. ligand atom): 1 dim
        torch.tensor([0.0 if name in PROTEIN_ATOM_NAMES else 1.0], dtype=torch.float32),
        # Element identity one-hot: 16 dim
        one_hot_encode(symbol if symbol in ELEMENT_VOCAB else 'OTHER', ELEMENT_VOCAB),
        # Bonded heavy-neighbor element composition: 35 dim
        encode_neighbor_composition(atom['neighbor_elements']),
        # Gasteiger partial charge (RBF): 16 dim
        gaussian_rbf(atom['gasteiger_charge'], GASTEIGER_CENTERS, GASTEIGER_GAMMA),
        # Per-atom Morgan/ECFP environment (multi-hot): 64 dim
        encode_morgan_bits(atom['morgan_bits']),
        # Pharmacophore flags: 3 dim
        torch.tensor([float(atom['is_donor'])], dtype=torch.float32),
        torch.tensor([float(atom['is_acceptor'])], dtype=torch.float32),
        torch.tensor([float(atom['is_hydrophobe'])], dtype=torch.float32),
        # Stereochemistry: CIP R/S one-hot (2) + (possible) stereocenter flag (1): 3 dim
        one_hot_encode(atom['cip_code'], CIP_CODES),
        torch.tensor([float(atom['is_stereocenter'])], dtype=torch.float32),

        # --- Chemistry features: 124 dim (unchanged) ---
        one_hot_encode(atom['period'], PERIODS), # 7 dim
        one_hot_encode(atom['group'], GROUPS), # 19 dim
        gaussian_rbf(atom['pauling_en'], EN_CENTERS, 1.0), # 8 dim
        one_hot_encode(atom['formal_charge'], FORMAL_CHARGES), # 9 dim
        one_hot_encode(atom['num_radical_electrons'], NUM_RADICAL_ELECTRONS), # 4 dim
        one_hot_encode(atom['hybridization'], HYBRIDIZATION), # 9 dim
        one_hot_encode(atom['total_valence'], VALENCES), # 8 dim
        one_hot_encode(atom['explicit_valence'], VALENCES), # 8 dim
        one_hot_encode(atom['implicit_valence'], VALENCES), # 8 dim
        one_hot_encode(atom['total_num_hs'], HYDROGENS), # 5 dim
        one_hot_encode(atom['num_explicit_hs'], HYDROGENS), # 5 dim
        one_hot_encode(atom['num_implicit_hs'], HYDROGENS), # 5 dim
        one_hot_encode(atom['degree'], DEGREES), # 7 dim
        one_hot_encode(atom['total_degree'], DEGREES), # 7 dim
        one_hot_encode(atom['chiral_tag'], CHIRALITY_TYPES), # 7 dim
        # Boolean features: 8 dim
        torch.tensor([atom['is_aromatic']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_3']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_4']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_5']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_6']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_7']], dtype=torch.float32),
        torch.tensor([atom['is_in_ring_size_8']], dtype=torch.float32),
    ]

    # Total: 299 dim (175 context + 124 chemistry)
    tensor_feats = torch.cat(features)

    return {'name': name, 'symbol': symbol, 'index': index, 'features': tensor_feats}

def featurize_bonds(bond: List[Dict]) -> torch.Tensor:
    
    # Metadata
    begin_idx = bond['begin_atom_idx']
    end_idx = bond['end_atom_idx']
    
    features = [
        one_hot_encode(bond['bond_type'], BOND_TYPES), # 6 dim
        one_hot_encode(bond['stereo'], BOND_STEREOS), # 6 dim
        one_hot_encode(bond['bond_dir'], BOND_DIRS), # 5 dim
        torch.tensor([bond['is_aromatic']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_conjugated']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_3']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_4']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_5']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_6']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_7']], dtype=torch.float32), # 1 dim
        torch.tensor([bond['is_in_ring_size_8']], dtype=torch.float32), # 1 dim
    ]

    # Total: 26 dim (6+6+5+9)
    tensor_feats = torch.cat(features)
    
    return {'begin_atom_idx': begin_idx, 'end_atom_idx': end_idx, 'features': tensor_feats}

# Pre-trained features #

class ESM3Extractor:
    '''
    Extracts ESM-3 embeddings for protein residues.
    Aligns embeddings to pocket residues by matching (chain, resnum) identifiers.
    '''

    def __init__(self, device: str = 'cuda'):
        from esm.models.esm3 import ESM3
        self.model = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device(device))
        self.model = self.model.float()
        self.model.eval()
        self.device = device

    def extract_chain_embedding(self, sequence: str) -> torch.Tensor:
        '''Extract ESM-3 embedding for a single chain. Returns [L, 1536].'''
        from esm.sdk.api import ESMProtein

        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        seq_tokens = protein_tensor.sequence.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model.forward(sequence_tokens=seq_tokens)

        embeddings = output.embeddings.squeeze(0)
        embeddings = embeddings[1:-1]  # Remove BOS/EOS tokens
        return embeddings.cpu()

    def extract_pocket_embeddings(
        self,
        sequence_data: Dict[str, str],
        pocket_residues: Dict[str, Dict],
        system_dir: Optional[Path] = None
    ) -> Dict[str, torch.Tensor]:
        '''
        Extract ESM-3 embeddings for pocket residues with alignment verification.

        Args:
            sequence_data: {chain_id: sequence_string}
            pocket_residues: {chain_id: {(resname, resnum, entity_id): residue_data}}

        Returns:
            Dict mapping (chain_id, resnum, entity_id) -> embedding tensor [1536]
        '''
        pocket_embeddings = {}

        for chain_id, residues in pocket_residues.items():
            if chain_id not in sequence_data:
                print(f"Warning: chain {chain_id} not in sequence_data")
                continue

            sequence = sequence_data[chain_id]
            chain_embedding = self.extract_chain_embedding(sequence)  # [L, 1536]

            for (resname, resnum, entity_id), res_data in residues.items():
                # resnum is 1-indexed, sequence embedding is 0-indexed
                seq_idx = resnum - 1

                if seq_idx < 0 or seq_idx >= len(chain_embedding):
                    print(f"Warning: resnum {resnum} out of bounds for chain {chain_id} (len={len(chain_embedding)})")
                    continue

                # Verify alignment: sequence[seq_idx] should match resname's 1-letter code
                expected_aa = sequence[seq_idx] if seq_idx < len(sequence) else None
                actual_aa = res_data.get('aa_code', resname)

                # Convert 3-letter to 1-letter if needed
                if len(actual_aa) == 3:
                    actual_aa = AA_3_TO_1.get(actual_aa, 'X')

                if expected_aa != actual_aa and expected_aa is not None:
                    if system_dir is not None:
                        system_name = system_dir.name
                        warning_str = f"Warning at {system_name}: "
                    else:
                        warning_str = "Warning: "
                    print(f"{warning_str} alignment mismatch at chain {chain_id} resnum {resnum}: "
                          f"sequence has '{expected_aa}', pocket has '{actual_aa}'")

                pocket_embeddings[(chain_id, resnum, entity_id)] = chain_embedding[seq_idx]

        return pocket_embeddings


class UniMolExtractor:
    '''
    Extracts Uni-Mol embeddings for ligands.
    Matches embeddings to parsed atoms by name: {symbol}{canonical_rank+1} (e.g., C1, N2, O3).

    Requires the padding-invariance patch from ``scripts/patch_unimol.sh``:
    stock unimol_tools lets padded atoms leak into real ones, so a molecule's
    embedding shifts by ~1e2 (1.1B model) depending on the largest molecule in
    its batch. With the patch an embedding does not depend on which molecules
    share its batch, whatever ``batch_size`` (molecules per forward pass).
    '''

    def __init__(self, model_size: str = '1.1B', use_cuda: bool = True, batch_size: int = 32):
        import os
        # Suppress tokenizers parallelism warning (UniMol uses multiprocessing for conformer generation)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from unimol_tools import UniMolRepr
        from unimol_tools.models import transformersv2
        if not getattr(transformersv2, 'PADDING_INVARIANT', False):
            raise RuntimeError(
                'unimol_tools is unpatched, so Uni-Mol v2 embeddings would depend on batch '
                'padding. Run scripts/patch_unimol.sh (details in scripts/unimol_padding.patch).'
            )
        params = {
            'data_type': 'molecule',
            'remove_hs': False,
            'model_name': 'unimolv2',
            'model_size': model_size,
        }
        self.model = UniMolRepr(use_gpu=True, use_cuda=use_cuda, batch_size=batch_size, **params)

    def get_unimol_atom_names(self, smiles: str) -> Optional[List[str]]:
        '''
        Generate atom names in Uni-Mol order (RDKit iteration order).
        Names follow the scheme: {symbol}{canonical_rank+1}

        Returns:
            List of atom names in RDKit iteration order, or None if SMILES is invalid
        '''
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        canonical_ranks = AllChem.CanonicalRankAtoms(mol)
        # Names in RDKit iteration order (same order as Uni-Mol embeddings)
        names = [f"{atom.GetSymbol().upper()}{canonical_ranks[atom.GetIdx()]+1}"
                 for atom in mol.GetAtoms()]
        return names

    def extract_single(
        self,
        smiles: str,
        parsed_atoms: List[Dict]
    ) -> Optional[torch.Tensor]:
        '''Single-ligand form of :meth:`extract_batch`.'''
        return self.extract_batch([smiles], [parsed_atoms])[0]

    def extract_batch(
        self,
        smiles_list: List[str],
        parsed_atoms_list: List[List[Dict]],
        return_status: bool = False,
    ):
        '''
        Batched extraction for multiple ligands.

        Each embedding is matched to its parsed atoms by canonical atom name, so
        the parsed atom order need not be Uni-Mol's (RDKit iteration order of
        the SMILES). Output i is verified to be input i by comparing the atomic
        numbers Uni-Mol reports for it against the SMILES: unimol_tools batches
        internally, and a dropped or reordered entry would otherwise shift every
        later embedding onto the wrong ligand. Invalid SMILES are held out of
        the Uni-Mol call (unimol_tools raises on them) and reported as None.

        Args:
            smiles_list: List of SMILES strings
            parsed_atoms_list: List of atom dicts for each ligand
            return_status: Also return one status string per ligand

        Returns:
            List of tensors (None for failures) ordered like *smiles_list*.
            With *return_status*, a tuple ``(tensors, statuses)`` where a status
            is ``''`` (ok), ``'invalid_smiles'``, ``'too_many_atoms'`` (above
            Uni-Mol's max_atoms crop, 128), ``'atom_name_mismatch'``
            (parsed atoms are not the SMILES' atoms) or ``'planar_conformer'``
            (embedding kept, but Uni-Mol fell back to 2D coordinates).
        '''
        n = len(smiles_list)
        results: List[Optional[torch.Tensor]] = [None] * n
        statuses: List[str] = ['invalid_smiles'] * n
        if n == 0:
            return ([], []) if return_status else []

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        for i, mol in enumerate(mols):
            if mol is None:
                print(f"Warning: invalid SMILES: {smiles_list[i]}")
        if not valid:
            return (results, statuses) if return_status else results

        output = self.model.get_repr(data=[smiles_list[i] for i in valid], return_atomic_reprs=True)
        atomic_reprs = output['atomic_reprs']
        if len(atomic_reprs) != len(valid):
            raise RuntimeError(
                f'UniMol returned {len(atomic_reprs)} molecules for {len(valid)} '
                f'SMILES; batch is misaligned'
            )

        max_atoms = int(self.model.params.get('max_atoms', 128))
        for j, i in enumerate(valid):
            smiles, mol, parsed_atoms = smiles_list[i], mols[i], parsed_atoms_list[i]
            canonical_ranks = AllChem.CanonicalRankAtoms(mol)
            unimol_names = [f"{atom.GetSymbol().upper()}{canonical_ranks[atom.GetIdx()]+1}"
                            for atom in mol.GetAtoms()]
            expected_z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            atomic_repr = torch.tensor(atomic_reprs[j])
            reported_z = [int(z) for z in output['atomic_symbol'][j]]

            # The only legitimate disagreement is Uni-Mol's random crop of molecules
            # above its max_atoms; any other count or element difference means
            # output j is not molecule j.
            if len(expected_z) > max_atoms and len(reported_z) == max_atoms:
                print(f"Warning: {smiles} has {len(expected_z)} atoms, above Uni-Mol's "
                      f"max_atoms={max_atoms}")
                statuses[i] = 'too_many_atoms'
                continue
            if reported_z != expected_z:
                raise RuntimeError(
                    f'UniMol output {j} reports atomic numbers {reported_z} but SMILES '
                    f'{smiles} has {expected_z}; batch is misaligned'
                )

            unimol_name_to_idx = {name: idx for idx, name in enumerate(unimol_names)}
            parsed_names = [atom['name'] for atom in parsed_atoms]
            missing = [name for name in parsed_names if name not in unimol_name_to_idx]
            if missing:
                print(f"Warning: atom names not found for {smiles}: {missing[:5]}...")
                statuses[i] = 'atom_name_mismatch'
                continue

            results[i] = atomic_repr[[unimol_name_to_idx[name] for name in parsed_names]]
            coords = np.asarray(output['atomic_coords'][j])
            statuses[i] = 'planar_conformer' if np.all(coords[:, 2] == 0) else ''

        return (results, statuses) if return_status else results
