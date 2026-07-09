# From boltz:

import pandas as pd
from pathlib import Path

################### Atom Features ###################
'''
Always put types where unknown values will be assigned at the end of the list
'''

_CONST_DIR = Path(__file__).parent
PERIODIC_TABLE = pd.read_csv(_CONST_DIR / 'periodictable.csv')

PERIODS = list(range(1, 8)) # 1 ~ 7, unk -> 7
GROUPS = list(range(1, 20)) # 1 ~ 18, unk -> 18
NUM_RADICAL_ELECTRONS = list(range(4)) # 0 ~ 4, unk -> 4
FORMAL_CHARGES = list(range(-4, 5)) # -4 ~ 4, unk -> 9
HYBRIDIZATION = [
    "S",
    "SP",
    "SP2",
    "SP2D",
    "SP3",
    "SP3D",
    "SP3D2",
    "UNSPECIFIED",
    "OTHER"
]
VALENCES = list(range(8)) # 0 ~ 7, unk -> 7
HYDROGENS = list(range(5)) # 0 ~ 4, unk -> 4
DEGREES = list(range(7)) # 0 ~ 6, unk -> 6

CHIRALITY_TYPES = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER"
]

# For rbf encoding of Pauling electronegativity
EN_CENTERS = [i*0.5 for i in range(1, 9)] # 0.5 ~ 4.0

BOND_TYPES = [
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
    "OTHER"
]

BOND_STEREOS = [
    "STEREOANY",
    "STEREOCIS",
    "STEREOE",
    "STEREONONE",
    "STEREOTRANS",
    "STEREOZ"
]

BOND_DIRS = [
    "NONE",
    "BEGINWEDGE",
    "BEGINDASH",
    "ENDDOWNRIGHT",
    "ENDUPRIGHT",
]

########## Discriminating per-atom context features ##########
# Replace the old 256-dim ASCII atom-name block (redundant for ligands,
# collision-prone for receptor, train/inference-inconsistent) with canonical,
# permutation-invariant, source-consistent descriptors.

# Protein heavy-atom-name vocabulary (36 standard names + UNK). One-hot gives a
# clean residue-internal identity for receptor atoms; ligand atoms map to all-zero.
PROTEIN_ATOM_NAMES = [
    'N', 'CA', 'C', 'O', 'CB',
    'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
    'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2',
    'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
    'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
    'SD', 'SG',
    'UNK',
]

# Element one-hot vocabulary (UNK -> 'OTHER').
ELEMENT_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                 'B', 'Se', 'Si', 'Fe', 'Zn', 'Mg', 'OTHER']

# Bonded heavy-neighbor composition: count per element class, one-hot over bins.
NEIGHBOR_ELEMENT_CLASSES = ['C', 'N', 'O', 'S', 'P', 'HALOGEN', 'OTHER']
NEIGHBOR_COUNT_BINS = [0, 1, 2, 3, 4]  # 4 -> "4 or more"

# Gaussian RBF centers for Gasteiger partial charge (covers typical [-0.8, 0.8]).
GASTEIGER_CENTERS = [-0.8 + i * (1.6 / 15) for i in range(16)]  # 16 centers
GASTEIGER_GAMMA = 8.0

# Per-atom Morgan (ECFP) circular environment, folded to a multi-hot bit vector.
MORGAN_RADIUS = 2
MORGAN_NBITS = 64

# Atom stereochemistry: CIP descriptor (R/S; unassigned -> all-zero) plus a flag
# for atoms that are (possibly) stereocenters even if unassigned. Complements the
# raw chiral_tag (CW/CCW) already in the chemistry block with assignment-aware
# stereo identity.
CIP_CODES = ['R', 'S']

# Feature layout. Context block (new discriminating features) precedes the
# unchanged 124-dim chemistry block. Downstream code that slices chemistry
# fields (e.g. vdW radius lookup in model/loss.py) must offset by ATOM_CONTEXT_DIM.
#   context = protein_name(37) + is_ligand(1) + element(16) + neighbors(35)
#           + gasteiger(16) + morgan(64) + pharmacophore(3) + stereo(3) = 175
ATOM_CONTEXT_DIM = (
    len(PROTEIN_ATOM_NAMES) + 1 + len(ELEMENT_VOCAB)
    + len(NEIGHBOR_ELEMENT_CLASSES) * len(NEIGHBOR_COUNT_BINS)
    + len(GASTEIGER_CENTERS) + MORGAN_NBITS + 3
    + len(CIP_CODES) + 1
)  # 175
ATOM_CHEM_DIM = 124
ATOM_FEAT_DIM = ATOM_CONTEXT_DIM + ATOM_CHEM_DIM  # 296

################### Proteins ###################

AA_3_TO_1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'UNK': 'X',
}

# Assert all aas are aligned with cif atom orderings
AA_URL = {
    'ALA': 'https://files.rcsb.org/ligands/download/ALA_ideal.sdf',
    'ARG': 'https://files.rcsb.org/ligands/download/ARG_ideal.sdf',
    'ASN': 'https://files.rcsb.org/ligands/download/ASN_ideal.sdf',
    'ASP': 'https://files.rcsb.org/ligands/download/ASP_ideal.sdf',
    'CYS': 'https://files.rcsb.org/ligands/download/CYS_ideal.sdf',
    'GLN': 'https://files.rcsb.org/ligands/download/GLN_ideal.sdf',
    'GLU': 'https://files.rcsb.org/ligands/download/GLU_ideal.sdf',
    'GLY': 'https://files.rcsb.org/ligands/download/GLY_ideal.sdf',
    'HIS': 'https://files.rcsb.org/ligands/download/HIS_ideal.sdf',
    'ILE': 'https://files.rcsb.org/ligands/download/ILE_ideal.sdf',
    'LEU': 'https://files.rcsb.org/ligands/download/LEU_ideal.sdf',
    'LYS': 'https://files.rcsb.org/ligands/download/LYS_ideal.sdf',
    'MET': 'https://files.rcsb.org/ligands/download/MET_ideal.sdf',
    'PHE': 'https://files.rcsb.org/ligands/download/PHE_ideal.sdf',
    'PRO': 'https://files.rcsb.org/ligands/download/PRO_ideal.sdf',
    'SER': 'https://files.rcsb.org/ligands/download/SER_ideal.sdf',
    'THR': 'https://files.rcsb.org/ligands/download/THR_ideal.sdf',
    'TRP': 'https://files.rcsb.org/ligands/download/TRP_ideal.sdf',
    'TYR': 'https://files.rcsb.org/ligands/download/TYR_ideal.sdf',
    'VAL': 'https://files.rcsb.org/ligands/download/VAL_ideal.sdf',
    'UNK': 'https://files.rcsb.org/ligands/download/UNK_ideal.sdf',
}

STD_AA_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

NONSTD_AA_ATOMS = {
    "UNK": ["N", "CA", "C", "O", "CB"],
}

PROTEIN_BACKBONE_ATOM_NAMES = ["N", "CA", "C", "O"]

REF_SYMMETRIES = {
    "ALA": [],
    "ARG": [],
    "ASN": [],
    "ASP": [[(6, 7), (7, 6)]],
    "CYS": [],
    "GLN": [],
    "GLU": [[(7, 8), (8, 7)]],
    "GLY": [],
    "HIS": [],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PHE": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": [],
    "TYR": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "VAL": []
}