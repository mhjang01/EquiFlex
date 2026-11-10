from pathlib import Path
from .const import *
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, rdFingerprintGenerator, rdCIPLabeler
from rdkit import RDConfig
from typing import Union, List, Dict, Tuple, Optional
import pandas as pd
import os
import re

# Pharmacophore feature factory (donor / acceptor / hydrophobe atom typing).
_FDEF_PATH = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

### General ###

def sanitize_sequence(sequence: str) -> str:
    '''
    Sanitize protein sequence by replacing non-standard tokens with 'X'.

    Handles:
    - Parenthesis-enclosed tokens like (UNK), (ACE), (NH2), (MSE), etc.
    - Any remaining non-standard single characters

    Args:
        sequence: Raw sequence string from CIF file

    Returns:
        Sanitized sequence with only standard amino acid codes and 'X'
    '''
    # Replace all parenthesis-enclosed tokens with 'X'
    # Pattern matches: ( followed by any characters followed by )
    sanitized = re.sub(r'\([^)]+\)', 'X', sequence)

    # Replace any remaining non-standard characters with 'X'
    # Standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    result = []
    for char in sanitized:
        if char.upper() in standard_aa:
            result.append(char.upper())
        elif char in '\n\r\t ':  # Skip whitespace
            continue
        else:
            result.append('X')

    return ''.join(result)


def get_rdkit_mol(data: Union[str, Path]):
    if isinstance(data, str):
        mol = Chem.MolFromSmiles(data)
    elif isinstance(data, Path):
        if data.suffix == '.pdb':
            mol = Chem.MolFromPDBFile(data)
        elif data.suffix == '.sdf':
            mol = Chem.SDMolSupplier(data)[0]
        elif data.suffix == '.mol2':
            mol = Chem.MolFromMol2File(data)
        elif data.suffix == '.mol':
            mol = Chem.MolFromMolFile(data)
        else:
            print(f"Invalid file extension: {data.suffix}")
            return None
    else:
        print(f"Invalid data type: {type(data)}")
        return None
    
    if mol is None:
        print(f"Failed to create RDKit molecule from {data}")
        return None
    
    return mol


### For proteins ###

def get_ref_aa_data(aa_code: str):

    aa_path = Path(f'./.refs/{aa_code}.npz')
    if not aa_path.exists():
        # Lazy import to avoid circular dependency
        from .reference import generate_aa_reference_data
        generate_aa_reference_data(aa_code)
    data = np.load(aa_path, allow_pickle=True)
    return data

class ExcludeFlags:
    '''
    Flags for filtering atoms during CIF/PDB parsing.
    Set flags to True to EXCLUDE those entries.
    '''
    def __init__(
        self,
        hetatm: bool = True,
        altloc: bool = True,
        hydrogens: bool = True,
        water: bool = True,
        deuterium: bool = True,
        oxt_oxygen: bool = True
    ):
        self.hetatm = hetatm
        self.altloc = altloc
        self.hydrogens = hydrogens
        self.water = water
        self.deuterium = deuterium
        self.oxt_oxygen = oxt_oxygen

    WATER_RESIDUES = {'HOH', 'WAT', 'DOD', 'H2O', 'OH2'}

    def should_exclude(
        self,
        group_pdb: str,
        atom_id: str,
        comp_id: str,
        altloc: str,
    ) -> bool:
        '''
        Check if an atom record should be excluded based on flags.
        Returns True if the atom should be EXCLUDED.
        '''
        if self.hetatm and group_pdb != 'ATOM': # Exclude all but ATOM records
            return True

        if self.altloc and altloc not in ('.', ''):
            return True

        # Hydrogen filter
        if self.hydrogens and atom_id.strip().startswith('H'):
            return True

        # Water filter
        if self.water and comp_id.upper() in self.WATER_RESIDUES:
            return True

        # Deuterium filter
        if self.deuterium and atom_id.strip().startswith('D'):
            return True
        
        if self.oxt_oxygen and atom_id.strip().endswith('OXT'):
            return True

        return False

    @classmethod
    def default(cls) -> 'ExcludeFlags':
        '''Standard exclusion for protein structure analysis.'''
        return cls()

    @classmethod
    def keep_all(cls) -> 'ExcludeFlags':
        '''Keep everything (no exclusions).'''
        return cls(
            hetatm=False, altloc=False, hydrogens=False,
            water=False, deuterium=False, oxt_oxygen=False
        )

    @classmethod
    def protein_only(cls) -> 'ExcludeFlags':
        '''Strict protein-only parsing (no ligands, waters, or hydrogens).'''
        return cls(
            hetatm=True, altloc=True, hydrogens=True,
            water=True, deuterium=True, oxt_oxygen=True
        )


def gemmi_parse_protein(block, excludeflags: Optional[ExcludeFlags] = None):

    if excludeflags is None:
        excludeflags = ExcludeFlags.default()

    # Extract column data
    pdb_group_data = block.find_values('_atom_site.group_PDB')
    atom_type_data = block.find_values('_atom_site.type_symbol')
    atom_id_data = block.find_values('_atom_site.label_atom_id')
    comp_id_data = block.find_values('_atom_site.label_comp_id')
    asym_id_data = block.find_values('_atom_site.label_asym_id')
    entity_id_data = block.find_values('_atom_site.label_entity_id')
    seq_id_data = block.find_values('_atom_site.label_seq_id')
    col_x_data = block.find_values('_atom_site.Cartn_x')
    col_y_data = block.find_values('_atom_site.Cartn_y')
    col_z_data = block.find_values('_atom_site.Cartn_z')
    altloc_data = block.find_values('_atom_site.label_alt_id')
    # Author numbering, kept alongside label_seq_id so the predicted-pocket
    # overlay can match residues in a gemmi structure (which is author-numbered).
    # Without it process.build_pocket falls back to the label_seq ordinal and the
    # overlay silently misses every residue whose ordinal != author number.
    auth_seq_data = block.find_values('_atom_site.auth_seq_id')
    ins_code_data = block.find_values('_atom_site.pdbx_PDB_ins_code')

    # as_string strips mmCIF quoting. A long sequence is stored as a multi-line
    # text field delimited by ';', and find_values hands back those delimiters
    # verbatim; sanitize_sequence then turns each ';' into an 'X', prepending one
    # residue and shifting every label_seq_id by one against the sequence, so
    # ESM3Extractor attaches each pocket residue the embedding of its predecessor.
    sequences = [gemmi.cif.as_string(s)
                 for s in block.find_values('_entity_poly.pdbx_seq_one_letter_code_can')]

    block.find_values('_struct_asym.id')
    block.find_values('_struct_asym.entity_id')
    
    # crate mappings from asym_id to entity_id
    asym_to_entity = {asym_id: entity_id for asym_id, entity_id in zip(asym_id_data, entity_id_data)}

    # Hierarchical data structure: chain -> residue -> atoms
    structure = {}

    # Both author columns are optional; fall back so zip() never truncates.
    n_rows = len(comp_id_data)
    auth_seq_col = list(auth_seq_data) if len(auth_seq_data) == n_rows else list(seq_id_data)
    ins_code_col = list(ins_code_data) if len(ins_code_data) == n_rows else [''] * n_rows

    for i, (group_pdb, atom_type, atom_id, comp_id, asym_id, entity_id, seq_id, col_x, col_y, col_z, altloc, auth_seq, ins_code) in enumerate(
        zip(pdb_group_data, atom_type_data, atom_id_data, comp_id_data, asym_id_data, entity_id_data, seq_id_data, col_x_data, col_y_data, col_z_data, altloc_data, auth_seq_col, ins_code_col)
    ):
        
        try:
            entity_id = int(asym_to_entity[asym_id])
        except:
            if len(entity_id) == 1:
                entity_id = ord(entity_id.upper()) - ord('A') + 1
            else:
                print(f"Failed to get entity_id for {asym_id}")
                continue
            
        # Use full label_asym_id (e.g. "1.A", "2.A") to keep biological-assembly
        # copies as distinct chains. Stripping the prefix collapses copies into
        # one chain dict and merges atoms from different assemblies under the
        # same res_key, causing stored coords to come from the wrong copy
        # (see bug audit in v2_plinder/pocket_cutoff_violations.json).
        chain_id = asym_id.upper()
        
        # Apply exclusion filters
        if excludeflags.should_exclude(
            group_pdb=group_pdb,
            atom_id=atom_id,
            comp_id=comp_id,
            altloc=altloc,
        ):
            continue
        
        # Sequence processing - sanitize to handle (UNK), (ACE), (MSE), etc.
        sequence_current = sanitize_sequence(sequences[entity_id-1])
                
        if chain_id not in structure:
            structure[chain_id] = {
                (None, 'sequence'): sequence_current
            }
        
        aa_1_letter = AA_3_TO_1.get(comp_id.upper(), 'X')  # Non-standard → 'X' (UNK)
        res_key = (int(seq_id), aa_1_letter)

        if res_key not in structure[chain_id]:
            icode = '' if ins_code in (None, '', '.', '?', False) else str(ins_code).strip()
            try:
                pdb_resnum = int(auth_seq)
            except (TypeError, ValueError):
                pdb_resnum = int(seq_id)
            structure[chain_id][res_key] = {
                'resname': comp_id,
                'resnum': int(seq_id),
                'pdb_resnum': pdb_resnum,  # author numbering, for the pocket overlay
                'pdb_icode': icode,
                'entity_id': int(entity_id),
                'atoms': []
            }
        
        # Add atom data
        structure[chain_id][res_key]['atoms'].append({
            'name': atom_id,
            'element': atom_type,
            'coords': [float(col_x), float(col_y), float(col_z)],
            'altloc': altloc,
        })

    return structure
    
    
def parse_cif(path: Path):
    cif_doc = gemmi.cif.read_file(str(path))
    block = cif_doc.sole_block()
    parsed_data = gemmi_parse_protein(block)

    return parsed_data


def gemmi_parse_pdb(structure: gemmi.Structure, excludeflags: Optional[ExcludeFlags] = None):
    '''
    Parse protein structure from gemmi Structure object (loaded from PDB).
    Returns the same format as gemmi_parse_protein (CIF parser).
    '''
    if excludeflags is None:
        excludeflags = ExcludeFlags.default()

    result = {}

    for model in structure:
        for chain in model:
            # Preserve case: PDB files may contain both 'A' and 'a' chain copies
            # (lowercase = biological-assembly mirror). Uppercasing collapses them
            # into one chain dict and merges atoms from different copies, causing
            # stored coords to be from the wrong copy.
            chain_id = chain.name

            # First pass: collect residues in order (handling insertion codes)
            if chain_id not in result:
                # Collect all residues with their full seqid (num + icode)
                residue_list = []  # List of (num, icode, aa_1) in order
                for residue in chain:
                    if residue.name.upper() in AA_3_TO_1:
                        aa_1 = AA_3_TO_1[residue.name.upper()]
                        icode = residue.seqid.icode.strip() if residue.seqid.icode else ''
                        residue_list.append((residue.seqid.num, icode, aa_1))

                # Build sequence in residue order (no gap filling for insertion codes)
                seq_chars = []
                resid_to_seq_idx = {}  # Map (num, icode) -> seq_idx
                for num, icode, aa_1 in residue_list:
                    resid_key = (num, icode)
                    if resid_key not in resid_to_seq_idx:
                        seq_chars.append(aa_1)
                        resid_to_seq_idx[resid_key] = len(seq_chars) - 1

                sequence = ''.join(seq_chars)

                result[chain_id] = {
                    (None, 'sequence'): sequence,
                    (None, 'resid_to_seq_idx'): resid_to_seq_idx,
                }

            # Second pass: extract residue data
            for residue in chain:
                resname = residue.name.upper()
                resnum = residue.seqid.num
                icode = residue.seqid.icode.strip() if residue.seqid.icode else ''

                # Skip water
                if excludeflags.water and resname in ExcludeFlags.WATER_RESIDUES:
                    continue

                is_hetatm = (residue.het_flag == 'H')
                if excludeflags.hetatm and is_hetatm:
                    continue

                # Only process standard amino acids
                if resname not in AA_3_TO_1:
                    continue

                aa_1_letter = AA_3_TO_1.get(resname, 'X')
                # Include insertion code in key to avoid collisions
                res_key = (resnum, icode, aa_1_letter) if icode else (resnum, aa_1_letter)

                if res_key not in result[chain_id]:
                    # Get seq_idx from mapping (1-indexed for compatibility with CIF)
                    resid_to_seq_idx = result[chain_id].get((None, 'resid_to_seq_idx'), {})
                    resid_key = (resnum, icode)
                    seq_idx = resid_to_seq_idx.get(resid_key, len(resid_to_seq_idx))

                    result[chain_id][res_key] = {
                        'resname': resname,
                        'resnum': seq_idx + 1,  # Store 1-indexed seq position (like CIF seq_id)
                        'pdb_resnum': resnum,   # Keep original PDB numbering
                        'pdb_icode': icode,     # Store insertion code
                        'entity_id': 1,
                        'atoms': []
                    }

                for atom in residue:
                    atom_name = atom.name
                    element = atom.element.name
                    altloc = atom.altloc if atom.altloc else '.'

                    # Apply exclusion filters (excluding hetatm check since we did it above)
                    group_pdb = 'ATOM'  # Already filtered hetatm residues above

                    # Check hydrogen filter
                    if excludeflags.hydrogens and atom_name.strip().startswith('H'):
                        continue

                    # Check deuterium filter
                    if excludeflags.deuterium and atom_name.strip().startswith('D'):
                        continue

                    # Check altloc filter
                    if excludeflags.altloc and altloc not in ('.', '', '\0'):
                        continue

                    # Check OXT filter
                    if excludeflags.oxt_oxygen and atom_name.strip() == 'OXT':
                        continue

                    result[chain_id][res_key]['atoms'].append({
                        'name': atom_name,
                        'element': element,
                        'coords': [atom.pos.x, atom.pos.y, atom.pos.z],
                        'altloc': altloc if altloc != '\0' else '.',
                    })

    # Remove empty residues (all atoms filtered out)
    # Skip metadata keys: (None, 'sequence') and (None, 'resid_to_seq_idx')
    for chain_id in list(result.keys()):
        for key in list(result[chain_id].keys()):
            if key[0] is None:  # Skip metadata keys
                continue
            if len(result[chain_id][key]['atoms']) == 0:
                del result[chain_id][key]

    return result


def parse_pdb(path: Path):
    '''
    Parse PDB file and return structure in same format as parse_cif.
    '''
    structure = gemmi.read_pdb(str(path), split_chain_on_ter=False)
    structure.setup_entities()
    parsed_data = gemmi_parse_pdb(structure)
    return parsed_data


def parse_structure(path: Union[str, Path]):
    '''
    Parse protein structure file (CIF or PDB) automatically based on extension.
    Returns the same format regardless of input type.
    '''
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in ('.cif', '.mmcif'):
        return parse_cif(path)
    elif suffix in ('.pdb', '.ent'):
        return parse_pdb(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .cif, .mmcif, .pdb, or .ent")


### For ligands ###

def compute_mol_context(mol: Chem.Mol) -> Dict:
    '''
    Pre-compute molecule-level descriptors needed for discriminating per-atom
    features. Run once per molecule, then passed to extract_atom_data so each
    atom can read its own context without recomputing.

    Returns:
        gasteiger:   list of partial charges (NaN/Inf -> 0.0), indexed by atom
        donors/acceptors/hydrophobes: sets of atom indices
        atom_bits:   {atom_idx: set(morgan_bit)} for radius-MORGAN_RADIUS
                     environments centered on that atom
    '''
    AllChem.ComputeGasteigerCharges(mol)
    gasteiger = []
    for atom in mol.GetAtoms():
        q = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
        gasteiger.append(0.0 if (q != q or q in (float('inf'), float('-inf'))) else q)

    donors, acceptors, hydrophobes = set(), set(), set()
    for feat in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        family = feat.GetFamily()
        if family == 'Donor':
            donors.update(feat.GetAtomIds())
        elif family == 'Acceptor':
            acceptors.update(feat.GetAtomIds())
        elif family in ('Hydrophobe', 'LumpedHydrophobe'):
            hydrophobes.update(feat.GetAtomIds())

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    morgan_gen.GetFingerprint(mol, additionalOutput=ao)
    atom_bits = {i: set() for i in range(mol.GetNumAtoms())}
    for bit, envs in ao.GetBitInfoMap().items():
        for center_idx, _radius in envs:
            atom_bits[center_idx].add(bit)

    # Stereochemistry: assign CIP (R/S) labels and flag (possible) stereocenters.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    cip_codes = {}
    for atom in mol.GetAtoms():
        cip_codes[atom.GetIdx()] = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else ''
    stereocenters = {
        idx for idx, _ in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    }

    return {
        'gasteiger': gasteiger,
        'donors': donors,
        'acceptors': acceptors,
        'hydrophobes': hydrophobes,
        'atom_bits': atom_bits,
        'cip_codes': cip_codes,
        'stereocenters': stereocenters,
    }


def get_neighbor_element_counts(atom: Chem.Atom) -> Dict[str, int]:
    '''Count bonded heavy-atom neighbors per element class (hydrogens ignored).'''
    counts = {cls: 0 for cls in NEIGHBOR_ELEMENT_CLASSES}
    for nbr in atom.GetNeighbors():
        sym = nbr.GetSymbol()
        if sym == 'H':
            continue
        if sym in counts:
            counts[sym] += 1
        elif sym in ('F', 'Cl', 'Br', 'I'):
            counts['HALOGEN'] += 1
        else:
            counts['OTHER'] += 1
    return counts


def extract_atom_data(atom: Chem.Atom, name: Optional[str] = None,
                      mol_ctx: Optional[Dict] = None) -> Dict:
    '''
    Extracts static/intrinsic features from an atom plus, when ``mol_ctx`` is
    provided, discriminating molecular-context descriptors (Gasteiger charge,
    pharmacophore flags, per-atom Morgan environment).
    '''

    # Name is for amino acids, where names can be a fair discriminator
    if name is None:
        name = atom.GetSymbol()
    
    atomic_num = atom.GetAtomicNum()
    pt_row = PERIODIC_TABLE.loc[PERIODIC_TABLE['atomic_number'] == atomic_num]
    # Handle lanthanides & actinides
    group = pt_row['group'].values[0]
    if np.isnan(group):
        group = 19 # Assign lanthanides & actinides to group 19
    
    data = {
        # Basic identity
        'name': name,
        'symbol': atom.GetSymbol(),
        'index': atom.GetIdx(),
        'atomic_number': atomic_num,

        # Periodic table properties (static)
        'group': int(group),
        'period': int(pt_row['period'].values[0]) if len(pt_row) else None,
        'pauling_en': float(pt_row['pauling_en'].values[0]) if len(pt_row) else None,

        # Valence and bonding
        'formal_charge': atom.GetFormalCharge(),
        'num_radical_electrons': atom.GetNumRadicalElectrons(),
        'hybridization': str(atom.GetHybridization()),
        'total_valence': atom.GetTotalValence(),
        'explicit_valence': atom.GetValence(which=Chem.ValenceType.EXPLICIT),
        'implicit_valence': atom.GetValence(which=Chem.ValenceType.IMPLICIT),

        # Hydrogen info
        'total_num_hs': atom.GetTotalNumHs(),
        'num_explicit_hs': atom.GetNumExplicitHs(),
        'num_implicit_hs': atom.GetNumImplicitHs(),

        # Degree (number of bonded neighbors)
        'degree': atom.GetDegree(),
        'total_degree': atom.GetTotalDegree(),

        # Ring and aromaticity
        'is_aromatic': atom.GetIsAromatic(),
        'is_in_ring': atom.IsInRing(),
        'is_in_ring_size_3': atom.IsInRingSize(3),
        'is_in_ring_size_4': atom.IsInRingSize(4),
        'is_in_ring_size_5': atom.IsInRingSize(5),
        'is_in_ring_size_6': atom.IsInRingSize(6),
        'is_in_ring_size_7': atom.IsInRingSize(7),
        'is_in_ring_size_8': atom.IsInRingSize(8),

        # Stereochemistry
        'chiral_tag': str(atom.GetChiralTag().name),

        # Bonded heavy-neighbor element composition (topology-only, canonical)
        'neighbor_elements': get_neighbor_element_counts(atom),
    }

    # Molecular-context descriptors (require the owning molecule)
    if mol_ctx is not None:
        idx = atom.GetIdx()
        data['gasteiger_charge'] = mol_ctx['gasteiger'][idx]
        data['is_donor'] = idx in mol_ctx['donors']
        data['is_acceptor'] = idx in mol_ctx['acceptors']
        data['is_hydrophobe'] = idx in mol_ctx['hydrophobes']
        data['morgan_bits'] = sorted(mol_ctx['atom_bits'][idx])
        data['cip_code'] = mol_ctx['cip_codes'].get(idx, '')
        data['is_stereocenter'] = idx in mol_ctx['stereocenters']
    else:
        data['gasteiger_charge'] = 0.0
        data['is_donor'] = False
        data['is_acceptor'] = False
        data['is_hydrophobe'] = False
        data['morgan_bits'] = []
        data['cip_code'] = ''
        data['is_stereocenter'] = False

    return data


def extract_bond_data(bond: Chem.Bond) -> Dict:

    data = {
        # Bond endpoints
        'begin_atom_idx': bond.GetBeginAtomIdx(),
        'end_atom_idx': bond.GetEndAtomIdx(),

        # Bond type
        'bond_type': str(bond.GetBondType().name),

        # Properties
        'is_aromatic': bond.GetIsAromatic(),
        'is_conjugated': bond.GetIsConjugated(),
        'is_in_ring': bond.IsInRing(),
        'is_in_ring_size_3': bond.IsInRingSize(3),
        'is_in_ring_size_4': bond.IsInRingSize(4),
        'is_in_ring_size_5': bond.IsInRingSize(5),
        'is_in_ring_size_6': bond.IsInRingSize(6),
        'is_in_ring_size_7': bond.IsInRingSize(7),
        'is_in_ring_size_8': bond.IsInRingSize(8),

        # Stereochemistry
        'stereo': str(bond.GetStereo()),
        'bond_dir': str(bond.GetBondDir()),
    }

    return data

def extract_coordinates(mol: Chem.Mol) -> np.ndarray:
    '''
    Extracts 3D coordinates from a molecule if available.
    Returns array of shape [num_atoms, 3] or None if no conformer.
    '''
    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer(0)
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords, dtype=np.float32)


def extract_ligand_data(mol_path: Path) -> Dict:
    '''
    Extracts comprehensive static data from a ligand file.
    Returns a dictionary with atoms, bonds, coordinates, and metadata.
    '''
    # Molecule names in PLIDER: N.{chain}.sdf
    chain = mol_path.stem.split('.')[-1].upper()

    mol = get_rdkit_mol(mol_path)
    if mol is None:
        print(f"Failed to create RDKit molecule from {mol_path}")
        return None
    
    # Extract atom data
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i]+1) for i in range(mol.GetNumAtoms())}
    mol_ctx = compute_mol_context(mol)
    atoms = [extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}', mol_ctx=mol_ctx)
             for atom in mol.GetAtoms()]

    # Extract bond data
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]

    # Extract coordinates
    coords = extract_coordinates(mol)

    data = {
        'chain': chain,
        'smiles': Chem.MolToSmiles(mol, canonical=True),
        'atoms': atoms,
        'bonds': bonds,
        'coords': coords,
        'mapping': mapping,
        '_mol': mol,
    }

    return data


### Torsion Index Computation ###

def get_chiral_atom_indices(mol) -> np.ndarray:
    '''
    Improper-dihedral quadruples encoding each tetrahedral stereocentre.

    Returns [K, 5] of (a0, a1, a2, centre, is_r) in the SAME atom order as the
    input mol.  a0..a2 are the three highest-CIP-rank substituents, so the SIGN
    of the dihedral over (a0, a1, a2, centre) encodes R vs S -- a pseudoscalar,
    which is the only kind of quantity that can tell enantiomers apart.
    Four-substituent centres also emit the three drop-one quadruples (order
    reversed on even skips to preserve orientation) for redundancy.

    Consumed by ChiralAtomLoss via the inference-time guidance.
    '''
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if not all(a.HasProp('_CIPRank') for a in mol.GetAtoms()):
        return np.empty((0, 5), dtype=np.int64)

    out = []
    for centre_idx, orientation in Chem.FindMolChiralCenters(mol, includeUnassigned=False):
        centre = mol.GetAtomWithIdx(centre_idx)
        if centre.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        nbrs = sorted(
            ((n.GetIdx(), int(n.GetProp('_CIPRank'))) for n in centre.GetNeighbors()),
            key=lambda x: x[1], reverse=True,
        )
        if not 3 <= len(nbrs) <= 4:
            continue
        nbrs = tuple(i for i, _ in nbrs)
        is_r = int(orientation == 'R')
        out.append((*nbrs[:3], centre_idx, is_r))
        if len(nbrs) == 4:
            for skip in range(3):
                sub = nbrs[:skip] + nbrs[skip + 1:]
                quad = sub[::-1] if skip % 2 == 0 else sub
                out.append((*quad, centre_idx, is_r))

    if not out:
        return np.empty((0, 5), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def get_sp2_torsion_indices(
    atoms: List[Dict],
    bonds: List[Dict]
) -> np.ndarray:
    '''
    For pre-computing torsion indices for planar torsions
    '''
    
    hybridization = 'SP2'
    num_atoms = len(atoms)

    # Build adjacency list from bonds
    neighbors = {i: [] for i in range(num_atoms)}
    for bond in bonds:
        src, dst = bond['begin_atom_idx'], bond['end_atom_idx']
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    # Find atoms with target hybridization
    target_atoms = [i for i, atom in enumerate(atoms) if atom['hybridization'] == hybridization]

    torsions = []
    seen_bonds = set()

    for b in target_atoms:
        for c in neighbors[b]:
            if c not in target_atoms:
                continue

            # Avoid duplicate bonds (only process each bond once)
            bond_key = (min(b, c), max(b, c))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)

            # Get neighbors of b (excluding c) and neighbors of c (excluding b)
            a_candidates = [n for n in neighbors[b] if n != c]
            d_candidates = [n for n in neighbors[c] if n != b]

            # Create all valid (a, b, c, d) combinations
            for a in a_candidates:
                for d in d_candidates:
                    if a != d:  # Avoid degenerate torsion
                        torsions.append([a, b, c, d])

    if torsions:
        return np.array(torsions, dtype=np.int32)
    return np.empty((0, 4), dtype=np.int32)


def get_all_torsion_indices(
    atoms: List[Dict],
    bonds: List[Dict]
) -> Dict[str, np.ndarray]:
    '''
    Compute torsion indices for all rotatable bond types.

    Returns a dictionary with:
        - 'sp2_sp2': Torsions across SP2-SP2 bonds (should be planar)
        - 'sp3_sp3': Torsions across SP3-SP3 bonds (rotatable)
        - 'sp2_sp3': Torsions across SP2-SP3 bonds (partially restricted)

    Args:
        atoms: List of atom dictionaries
        bonds: List of bond dictionaries

    Returns:
        Dictionary mapping torsion type to [M, 4] array of atom indices
    '''
    num_atoms = len(atoms)

    # Build adjacency list
    neighbors = {i: [] for i in range(num_atoms)}
    for bond in bonds:
        src, dst = bond['begin_atom_idx'], bond['end_atom_idx']
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    # Categorize atoms by hybridization
    sp2_atoms = set(i for i, a in enumerate(atoms) if a['hybridization'] == 'SP2')
    sp3_atoms = set(i for i, a in enumerate(atoms) if a['hybridization'] == 'SP3')

    torsions = {'sp2_sp2': [], 'sp3_sp3': [], 'sp2_sp3': []}
    seen_bonds = set()

    for bond in bonds:
        b, c = bond['begin_atom_idx'], bond['end_atom_idx']

        bond_key = (min(b, c), max(b, c))
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)

        # Determine bond type
        b_sp2, c_sp2 = b in sp2_atoms, c in sp2_atoms
        b_sp3, c_sp3 = b in sp3_atoms, c in sp3_atoms

        if b_sp2 and c_sp2:
            torsion_type = 'sp2_sp2'
        elif b_sp3 and c_sp3:
            torsion_type = 'sp3_sp3'
        elif (b_sp2 and c_sp3) or (b_sp3 and c_sp2):
            torsion_type = 'sp2_sp3'
        else:
            continue  # Skip other hybridization combinations

        # Get neighbor candidates
        a_candidates = [n for n in neighbors[b] if n != c]
        d_candidates = [n for n in neighbors[c] if n != b]

        for a in a_candidates:
            for d in d_candidates:
                if a != d:
                    torsions[torsion_type].append([a, b, c, d])

    return {
        k: np.array(v, dtype=np.int32) if v else np.empty((0, 4), dtype=np.int32)
        for k, v in torsions.items()
    }


### For complex data generation(Data prep for training) ###

def is_adjacent(conf_1: np.ndarray, conf_2: np.ndarray, distance_threshold: float = 5.0):
    '''
    Checks if any atom in conf_1 is within distance_threshold of any atom in conf_2.
    Accepts arrays of different shapes:
        - conf_1: [N, 3] or [3,] (single point)
        - conf_2: [M, 3] or [3,] (single point)
    '''
    # Ensure 2D arrays
    conf_1 = np.atleast_2d(conf_1)
    conf_2 = np.atleast_2d(conf_2)
    
    # Compute pairwise distances: [N, M]
    # Using broadcasting: conf_1[:, None, :] - conf_2[None, :, :] -> [N, M, 3]
    diff = conf_1[:, None, :] - conf_2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    return np.any(dist < distance_threshold)

def is_clash(conf_1: np.ndarray, conf_2: np.ndarray, clash_threshold: float = 1.0):
    '''
    Checks if any atom in conf_1 is within clash_threshold of any atom in conf_2.
    Accepts arrays of different shapes:
        - conf_1: [N, 3] or [3,] (single point)
        - conf_2: [M, 3] or [3,] (single point)
    '''
    # Ensure 2D arrays
    conf_1 = np.atleast_2d(conf_1)
    conf_2 = np.atleast_2d(conf_2)
    
    # Compute pairwise distances: [N, M]
    # Using broadcasting: conf_1[:, None, :] - conf_2[None, :, :] -> [N, M, 3]
    diff = conf_1[:, None, :] - conf_2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    return np.any(dist < clash_threshold)


def get_valid_ligands(mol_data_list: List[Dict], clash_threshold: float = 1.0, distance_threshold: float = 4.0):
    '''
    Find valid ligands using propagating adjacency search.

    Algorithm:
        1. Sort ligands by number of atoms (descending)
        2. Start with largest ligand as the seed
        3. Iteratively find ligands adjacent to current valid set
        4. Stack coordinates of valid ligands and repeat until no new ligands found

    Args:
        mol_data_list: List of parsed ligand data dictionaries
        clash_threshold: Distance cutoff for clash check
        distance_threshold: Distance cutoff for adjacency check

    Returns:
        List of valid ligand data dictionaries
    '''
    if len(mol_data_list) == 0:
        return []

    if len(mol_data_list) == 1:
        return mol_data_list

    # Sort by number of atoms (descending) - largest first
    sorted_ligands = sorted(mol_data_list, key=lambda x: len(x['atoms']), reverse=True)

    # Initialize with largest ligand as seed
    valid_ligands = [sorted_ligands[0]]
    valid_coords = sorted_ligands[0]['coords'].copy()
    remaining = sorted_ligands[1:]

    # Propagate until no new ligands found
    changed = True
    while changed and remaining:
        changed = False

        for ligand_data in remaining:
            if is_adjacent(valid_coords, ligand_data['coords'], distance_threshold)\
                and not is_clash(valid_coords, ligand_data['coords'], clash_threshold):
                valid_ligands.append(ligand_data)
                # Stack coordinates for next iteration
                valid_coords = np.vstack([valid_coords, ligand_data['coords']])
                changed = True
            else:
                continue

    return valid_ligands

def get_pocket_residues(protein_data: Dict, ligand_data: List[Dict],
                        pocket_cutoff: float = 5.0, clash_cutoff: float = 0.1):
    """Include residue if any atom is within `pocket_cutoff` Å of any ligand atom.
    Drop residue if any of its atoms collides (< `clash_cutoff` Å) with any ligand
    atom — these are metal cofactors / covalent ligand atoms duplicated in both
    pocket and ligand records, which would produce cdist NaN gradients during
    training (∂‖x−y‖/∂x = (x−y)/‖x−y‖ is undefined at coincidence).
    """
    # Extract total conformer data from ligand data (concatenate all ligand coords)
    total_coords = np.concatenate([lig['coords'] for lig in ligand_data], axis=0)

    # Extract sequence data from protein data
    pocket_data, sequence_data = {}, {}

    for chain_id, chain_data in protein_data.items():
        per_chain_pocket_residues = []

        for res_key, res_data in chain_data.items():
            # Skip metadata keys: (None, 'sequence'), (None, 'resid_to_seq_idx')
            if res_key[0] is None:
                if res_key[1] == 'sequence':
                    sequence_data[chain_id] = res_data
                continue

            # Reject residue if any atom collides with a ligand atom — must be
            # checked against the whole residue, not just the contact atom.
            res_atom_coords = np.array([atom['coords'] for atom in res_data['atoms']])
            if is_clash(res_atom_coords, total_coords, clash_cutoff):
                continue

            # Include residue if any atom is within pocket cutoff of any ligand atom
            for atom in res_data['atoms']:
                if is_adjacent(total_coords, atom['coords'], pocket_cutoff):
                    per_chain_pocket_residues.append(res_data)
                    break  # Break for the residue loop

        pocket_data[chain_id] = per_chain_pocket_residues

    return pocket_data, sequence_data


def load_aa_reference(resname: str) -> Optional[Dict]:
    '''
    Load amino acid reference data from npz file and convert to a regular dict.
    Returns UNK if resname is non-standard
    '''
    # Check if resname is in standard amino acids
    if resname not in STD_AA_ATOMS:
        print(f"Non-standard residue: {resname}, falling back to UNK")
        ref_path = Path(f'./.refs/UNK.npz')
    else:
        ref_path = Path(f'./.refs/{resname}.npz')
        
    if not ref_path.exists():
        # Lazy import to avoid circular dependency
        from .reference import generate_aa_reference_data
        generate_aa_reference_data(resname)

    try:
        npz_data = np.load(ref_path, allow_pickle=True)
        # Convert NpzFile to regular dict, handling 0-d arrays
        aa_data = {}
        for key in npz_data.files:
            val = npz_data[key]
            # Convert 0-d numpy arrays to Python scalars
            if isinstance(val, np.ndarray) and val.ndim == 0:
                aa_data[key] = val.item()
            else:
                aa_data[key] = val
        return aa_data
    
    except Exception as e:
        print(f"Failed to load reference data for {resname}: {e}")
        return None

def overwrite_coords_on_reference_data(pocket_data: Dict):

    pocket_residues = {}
    for chain_id, residues in pocket_data.items():
        per_chain_data = {}

        for residue in residues:
            per_residue_temp_data = {}

            resname = residue['resname']
            resnum = residue['resnum']
            entity_id = residue['entity_id']

            # Load reference data (handles non-standard residues gracefully)
            aa_data = load_aa_reference(resname)
            if aa_data is None:
                continue
            
            if resname not in STD_AA_ATOMS:
                atom_orders = NONSTD_AA_ATOMS['UNK']
            else:
                atom_orders = STD_AA_ATOMS[resname]

            # Overwrite coords on reference data, with orders from atom_orders
            for atom_data in residue['atoms']:
                per_residue_temp_data[atom_data['name']] = atom_data['coords']

            # Iterate through atom_orders and add missing atoms with [0, 0, 0]
            missing_masks = np.zeros(len(atom_orders), dtype=bool)
            for i, atom_order in enumerate(atom_orders):
                if atom_order not in per_residue_temp_data:
                    missing_masks[i] = True
                    per_residue_temp_data[atom_order] = [0, 0, 0]

            # Drop additional atoms that don't exist in atom_orders
            per_residue_temp_data = {k: v for k, v in per_residue_temp_data.items() if k in atom_orders}

            # Sort by atom_orders
            per_residue_temp_data = [per_residue_temp_data[atom_order] for atom_order in atom_orders]

            # Create new residue data with both ref_coords and pocket coords
            pocket_residue_key = (resname, resnum, entity_id)
            new_aa_data = aa_data.copy()  # Now works since aa_data is a regular dict
            new_aa_data['coords'] = np.array(per_residue_temp_data)
            new_aa_data['missing_masks'] = missing_masks
            if 'pdb_resnum' in residue:
                new_aa_data['pdb_resnum'] = residue['pdb_resnum']
            if 'pdb_icode' in residue:
                new_aa_data['pdb_icode'] = residue['pdb_icode']
            per_chain_data[pocket_residue_key] = new_aa_data

        pocket_residues[chain_id] = per_chain_data

    return pocket_residues