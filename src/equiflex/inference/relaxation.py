"""
Structure writers for sampled protein-ligand complexes.

Pocket/protein models are built with gemmi from amino-acid SDF templates
(prep/.refs/) or an mmCIF/PDB source; ligand mols from stored atom/bond
data (ligands.pkl). Emits mmCIF complexes, protein PDBs, ligand SDFs and
trajectory CIFs, plus RMSD helpers over them.

Dependencies: torch, numpy, gemmi, rdkit, joblib
"""

import functools
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from equiflex.prep.const import STD_AA_ATOMS, NONSTD_AA_ATOMS, PROTEIN_ATOM_NAMES, ELEMENT_VOCAB
except ImportError:
    from ..prep.const import STD_AA_ATOMS, NONSTD_AA_ATOMS, PROTEIN_ATOM_NAMES, ELEMENT_VOCAB


# ---------------------------------------------------------------------------
# Protein file discovery
# ---------------------------------------------------------------------------

def find_protein_path(
    system_name: str,
    source_dirs: List[Union[str, Path]],
) -> Optional[Path]:
    """Locate the original protein PDB by matching ``{system_name}_protein.pdb``
    inside ``{source_dir}/{system_name}/`` for each candidate source directory."""
    for sd in source_dirs:
        candidate = Path(sd) / system_name / f'{system_name}_protein.pdb'
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Element decoding from featurized atom names
# ---------------------------------------------------------------------------

_ELEMENT_FEAT_OFFSET = len(PROTEIN_ATOM_NAMES) + 1  # 37 protein-name dims + 1 source flag


def _decode_element_from_features(atom_feat: torch.Tensor) -> str:
    """Decode element symbol from the ELEMENT_VOCAB one-hot block.

    Layout (see ``prep.featurize.extract_atom_data``): the discriminating-context
    block starts with ``PROTEIN_ATOM_NAMES`` (37) + source flag (1), then the
    ``ELEMENT_VOCAB`` element one-hot (16).  ``'OTHER'`` falls back to carbon.
    """
    block = atom_feat[_ELEMENT_FEAT_OFFSET:_ELEMENT_FEAT_OFFSET + len(ELEMENT_VOCAB)]
    symbol = ELEMENT_VOCAB[int(block.argmax().item())]
    return 'C' if symbol == 'OTHER' else symbol


def _element_from_protein_atom_name(name: str) -> str:
    """Element symbol from standard amino-acid atom name (always 1-char: C, N, O, S)."""
    return name[0]


# ---------------------------------------------------------------------------
# Ligand atom-ordering (SDF order ↔ SMILES-canonical order)
# ---------------------------------------------------------------------------

def _find_sdf_to_smi_mapping(
    smiles: str,
    atom_features: torch.Tensor,
    bond_features: torch.Tensor,
    n_atoms: int,
) -> Optional[List[int]]:
    """Map SDF atom indices (our stored data) to the SMILES-derived mol ordering.

    Returns *smi_to_sdf*: ``smi_to_sdf[smi_idx] = sdf_idx``.
    Returns ``None`` when the mapping cannot be determined (falls back to identity).
    """
    from rdkit import Chem

    smi_mol = Chem.MolFromSmiles(smiles)
    if smi_mol is None or smi_mol.GetNumAtoms() != n_atoms:
        return None

    sdf_elements = [_decode_element_from_features(atom_features[i]) for i in range(n_atoms)]

    _bt_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
        4: Chem.BondType.SINGLE,
        5: Chem.BondType.SINGLE,
    }

    query = Chem.RWMol()
    for elem in sdf_elements:
        try:
            anum = Chem.GetPeriodicTable().GetAtomicNumber(elem)
        except Exception:
            return None
        query.AddAtom(Chem.Atom(anum))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if bond_features[i, j].abs().sum() > 0:
                bt_idx = bond_features[i, j, :6].argmax().item()
                query.AddBond(i, j, _bt_map.get(bt_idx, Chem.BondType.SINGLE))

    query_mol = query.GetMol()
    match = smi_mol.GetSubstructMatch(query_mol)

    if len(match) != n_atoms:
        return None

    # match[sdf_idx] = smi_idx  →  invert to smi_to_sdf
    smi_to_sdf = [0] * n_atoms
    for sdf_idx, smi_idx in enumerate(match):
        smi_to_sdf[smi_idx] = sdf_idx
    return smi_to_sdf


# ---------------------------------------------------------------------------
# Pocket structure reconstruction  (gemmi)
# ---------------------------------------------------------------------------

def _build_pocket_gemmi(
    pocket_feats: Dict,
    coords: torch.Tensor,
    backbone_com: torch.Tensor,
):
    """Build a ``gemmi.Structure`` from pocket features + (centered) coordinates."""
    import gemmi

    residue_keys = pocket_feats['residue_keys']
    n_atoms_per_residue = pocket_feats['n_atoms_per_residue']
    missing_mask = pocket_feats['missing_mask']

    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu()

    abs_coords = (coords + backbone_com).numpy().astype(np.float64)
    if isinstance(missing_mask, torch.Tensor):
        missing_mask = missing_mask.numpy()

    st = gemmi.Structure()
    st.name = 'complex'
    model = gemmi.Model('1')
    chains: Dict[str, 'gemmi.Chain'] = {}
    atom_offset = 0
    serial = 1

    for res_idx, (chain_id, resname, resnum, _entity_id) in enumerate(residue_keys):
        n_atoms = n_atoms_per_residue[res_idx]
        atom_names = STD_AA_ATOMS.get(resname, NONSTD_AA_ATOMS.get(resname, NONSTD_AA_ATOMS.get('UNK', ['N', 'CA', 'C', 'O', 'CB'])))

        if chain_id not in chains:
            chains[chain_id] = gemmi.Chain(chain_id)

        res = gemmi.Residue()
        res.name = resname
        res.seqid = gemmi.SeqId(str(resnum))
        has_atoms = False

        for i in range(min(n_atoms, len(atom_names))):
            gi = atom_offset + i
            if missing_mask[gi]:
                continue
            a = gemmi.Atom()
            a.name = atom_names[i]
            a.element = gemmi.Element(_element_from_protein_atom_name(atom_names[i]))
            x, y, z = abs_coords[gi].tolist()
            a.pos = gemmi.Position(x, y, z)
            a.occ = 1.0
            a.b_iso = 0.0
            a.serial = serial
            serial += 1
            res.add_atom(a)
            has_atoms = True

        if has_atoms:
            chains[chain_id].add_residue(res)
        atom_offset += n_atoms

    for cid in sorted(chains):
        model.add_chain(chains[cid])
    st.add_model(model)
    return st


def _read_structure_sanitised(path_str: str) -> 'gemmi.Structure':
    """``gemmi.read_structure`` with junk PDB segIDs stripped first.

    Columns 67-76 of a PDB ATOM record are the segment identifier, and gemmi
    folds the segment into residue identity.  Five DEKOIS2.0 receptors (ache,
    adrb2, cdk2, fgfr1, pparg) carry a per-atom *partial charge* in that field
    instead -- a different value for every atom:

        ATOM   1148  N   TRP A 105   -8.966  15.224  57.388  1.00 57.53   .346 N
                                                                          ^^^^ cols 67-76

    So every atom becomes its own single-atom residue (adrb2: 4352 residues for
    7134 atoms).  ``remove_alternative_conformations()`` then sees a run of
    residues sharing seqid and name, treats them as alternates of one residue
    and keeps only the first -- collapsing adrb2 to 439 residues of 1 atom each,
    all named ``N``.  Both ``write_complex_cif`` and ``write_protein_pdb`` go
    through this template, so the predicted complex and the receptor handed to
    smina were destroyed together, silently.

    Blank the field before parsing, but only where it looks like a number: a
    real segID is an alphanumeric label, so requiring a '.' leaves genuine
    multi-segment files alone.
    """
    import gemmi

    path = Path(path_str)
    if path.suffix.lower() not in ('.pdb', '.ent'):
        return gemmi.read_structure(path_str)

    text = path.read_text()
    lines = text.splitlines(keepends=True)
    n_fixed = 0
    for i, line in enumerate(lines):
        if not line.startswith(('ATOM  ', 'HETATM')) or len(line) < 76:
            continue
        seg = line[66:76]
        if seg.strip() and '.' in seg:
            lines[i] = line[:66] + ' ' * 10 + line[76:]
            n_fixed += 1
    if not n_fixed:
        return gemmi.read_structure(path_str)

    logger.warning(
        'receptor %s: stripped a numeric per-atom value from PDB segID columns '
        '67-76 on %d atoms; left in place it shatters every residue',
        path.name, n_fixed,
    )
    st = gemmi.read_pdb_string(''.join(lines))
    st.setup_entities()
    return st


@functools.lru_cache(maxsize=8)
def _protein_template(path_str: str, mtime: float, size: int) -> 'gemmi.Structure':
    """Amino-acid-only copy of *path_str*, cached on (path, mtime, size).

    Do not carry crystallographic HETATMs into the predicted complex.  In
    particular, retaining the bound reference ligand would create an invalid
    overlay containing both the reference and predicted ligand.  Use Gemmi's
    residue classification rather than het_flag alone because mmCIF producers
    are not consistent about that flag.

    Callers must ``.clone()`` before mutating: the cached object is shared.
    """
    import gemmi

    source_st = _read_structure_sanitised(path_str)
    # Keep one conformer per atom.  prep.parse already drops altlocs when
    # featurising (ExcludeFlags.altloc), so without this the written receptor
    # disagrees with what the model was conditioned on: the residue arrives with
    # both conformers merged (MET363 in 5TGZ becomes a 16-atom methionine),
    # which Vina scores twice and RDKit refuses to sanitise — silently dropping
    # the whole residue from the torch scorer.
    source_st.remove_alternative_conformations()
    st = gemmi.Structure()
    st.name = source_st.name
    st.cell = source_st.cell
    st.spacegroup_hm = source_st.spacegroup_hm
    model = gemmi.Model('1')
    for source_chain in source_st[0]:
        chain = gemmi.Chain(source_chain.name)
        for residue in source_chain:
            residue_info = gemmi.find_tabulated_residue(residue.name)
            if residue_info and residue_info.is_amino_acid():
                chain.add_residue(residue)
        if len(chain) > 0:
            model.add_chain(chain)
    st.add_model(model)

    # A shattered residue set is indistinguishable from a healthy one
    # downstream -- it just writes a receptor with most atoms missing -- so fail
    # loudly here instead.  Real amino acids average ~8 heavy / ~16 all-atom;
    # anything under 3 means residue grouping broke.
    n_res = sum(len(ch) for ch in st[0])
    n_atoms = sum(len(r) for ch in st[0] for r in ch)
    if n_res and n_atoms / n_res < 3.0:
        raise ValueError(
            f'{path_str}: residue grouping is broken -- {n_atoms} atoms in '
            f'{n_res} amino-acid residues ({n_atoms / n_res:.2f} per residue). '
            'Expected >= 3. Check the PDB segID columns (67-76).'
        )
    return st


def _build_full_protein_gemmi(
    protein_path: Union[str, Path],
    pocket_feats: Dict,
    rec_coords: torch.Tensor,
    backbone_com: torch.Tensor,
) -> 'gemmi.Structure':
    """Load protein-only coordinates and overlay predicted pocket residues.

    Non-pocket residues keep their original crystallographic positions.
    Pocket residues are updated with the sampler-predicted coords.
    Waters, ligands, ions, cofactors, and all other non-amino-acid residues are
    removed before the predicted ligand is added.  This matches the protein-only
    conditioning structure used by the PoseBusters validation path.
    """
    import gemmi

    protein_path = Path(protein_path)
    stat = protein_path.stat()
    # The amino-acid-only structure depends on nothing but the file, while this
    # function is called once per sampled pose (millions of times in a screen).
    # Parse it once per file and clone; cloning 3.5k atoms is far cheaper than
    # re-parsing the mmCIF, and the clone keeps each pose's overlay independent.
    st = _protein_template(str(protein_path), stat.st_mtime, stat.st_size).clone()
    model = st[0]

    residue_keys = pocket_feats['residue_keys']
    n_atoms_per_residue = pocket_feats['n_atoms_per_residue']
    missing_mask = pocket_feats['missing_mask']

    if isinstance(rec_coords, torch.Tensor):
        rec_coords = rec_coords.detach().cpu()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu()

    abs_coords = (rec_coords + backbone_com).numpy().astype(np.float64)
    if isinstance(missing_mask, torch.Tensor):
        missing_mask = missing_mask.numpy()

    # Match gemmi structure numbering: (chain, resname, seqid.num, insertion code).
    # PDB featurization stores training ``resnum`` as 1-based *chain order* index
    # (see ``prep.parse.gemmi_parse_pdb``), not author residue numbers — so we
    # must use ``overlay_residue_keys`` (pdb_resnum / pdb_icode) when present.
    raw_overlay = pocket_feats.get('overlay_residue_keys')
    if raw_overlay is None:
        logger.warning(
            'pocket_features.pt has no overlay_residue_keys; re-featurize pockets '
            'so full-protein CIF overlay uses PDB/mmCIF residue numbers. '
            'Otherwise only residues where ordinal index == author number match.',
        )
    # NCS assemblies can contain chains whose names differ only by case
    # (author chains 'A' and 'a').  Uppercasing chain ids then collapses both
    # onto one pocket entry, so a single predicted side chain is overlaid on
    # every case-variant chain and those residues end up at identical
    # coordinates — which later breaks any step that adds hydrogens
    # (zero-length bond vector).  Match case-sensitively when such a collision
    # exists; otherwise keep the lenient uppercase match, so systems without
    # case-paired chains behave exactly as before.
    model_chain_names = [chain.name for chain in model]
    case_sensitive = len({c.upper() for c in model_chain_names}) != len(set(model_chain_names))

    def _norm_cid(cid: str) -> str:
        return str(cid) if case_sensitive else str(cid).upper()

    if not raw_overlay or len(raw_overlay) != len(residue_keys):
        overlay_keys = [
            (str(cid), str(rn).upper(), int(rnum), '')
            for (cid, rn, rnum, _eid) in residue_keys
        ]
    else:
        overlay_keys = raw_overlay

    pocket_lookup: Dict[Tuple[str, str, int, str], Tuple[int, int]] = {}
    atom_offset = 0
    for res_idx, (chain_id, resname, struct_num, icode) in enumerate(overlay_keys):
        n_atoms = n_atoms_per_residue[res_idx]
        pocket_lookup[(_norm_cid(chain_id), str(resname).upper(),
                       int(struct_num), icode or '')] = (atom_offset, res_idx)
        atom_offset += n_atoms

    for chain in model:
        cid = _norm_cid(chain.name)
        for residue in chain:
            ricode = residue.seqid.icode.strip() if residue.seqid.icode else ''
            key = (cid, residue.name.upper(), residue.seqid.num, ricode)
            if key not in pocket_lookup:
                continue

            base_offset, res_idx = pocket_lookup[key]
            n_atoms = n_atoms_per_residue[res_idx]
            atom_names = STD_AA_ATOMS.get(
                residue.name,
                NONSTD_AA_ATOMS.get(residue.name, NONSTD_AA_ATOMS.get('UNK', ['N', 'CA', 'C', 'O', 'CB'])),
            )

            for i, aname in enumerate(atom_names[:n_atoms]):
                gi = base_offset + i
                if missing_mask[gi]:
                    continue
                atom = residue.find_atom(aname, '\0')
                if atom is not None:
                    x, y, z = abs_coords[gi].tolist()
                    atom.pos = gemmi.Position(x, y, z)

    st.remove_hydrogens()
    return st


# ---------------------------------------------------------------------------
# Ligand mol reconstruction  (RDKit)
# ---------------------------------------------------------------------------

def _build_ligand_mols(
    ligand_feats: Dict,
    coords: torch.Tensor,
    backbone_com: torch.Tensor,
) -> Tuple[list, List[Optional[List[int]]]]:
    """Build RDKit mols (heavy-atom, SMILES order) with coordinates set.

    Returns ``(mols, smi_to_sdf_maps)`` where each map can be used to
    convert between SMILES mol ordering and SDF (stored) ordering.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    smiles_list = ligand_feats['smiles_list']
    n_atoms_per_lig = ligand_feats.get('n_atoms_per_ligand', [coords.shape[0]])

    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu()

    abs_coords = (coords + backbone_com).numpy().astype(np.float64)

    mols, mappings = [], []
    atom_offset = 0

    for lig_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        n_heavy = n_atoms_per_lig[lig_idx]
        lig_af = ligand_feats['atom_features'][atom_offset:atom_offset + n_heavy]
        lig_bf = ligand_feats['bond_features'][atom_offset:atom_offset + n_heavy,
                                                atom_offset:atom_offset + n_heavy]

        smi_to_sdf = _find_sdf_to_smi_mapping(smiles, lig_af, lig_bf, n_heavy)
        mappings.append(smi_to_sdf)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for smi_i in range(mol.GetNumAtoms()):
            sdf_i = smi_to_sdf[smi_i] if smi_to_sdf is not None else smi_i
            x, y, z = abs_coords[atom_offset + sdf_i].tolist()
            conf.SetAtomPosition(smi_i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        mols.append(mol)
        atom_offset += n_heavy

    return mols, mappings


# ---------------------------------------------------------------------------
# CIF I/O
# ---------------------------------------------------------------------------

_BOND_ORDER_MMCIF = {
    'SINGLE':   'SING',
    'DOUBLE':   'DOUB',
    'TRIPLE':   'TRIP',
    'AROMATIC': 'AROM',
}


def _fix_atom_site_pdbx_model_numbers(doc, n_atoms_per_frame: int) -> None:
    """Gemmi's ``make_mmcif_document`` lists all models in one ``atom_site`` loop
    but leaves ``pdbx_PDB_model_num`` as 1 for every row.  ChimeraX needs
    1, 2, … for ``coordsets true``.  Rows are ordered frame-major (all atoms of
    model 0, then model 1, …).
    """
    if n_atoms_per_frame <= 0:
        return
    block = doc.sole_block()
    tab = block.find_mmcif_category('_atom_site')
    if tab is None:
        return
    loop = tab.loop
    w = loop.width()
    tags = list(loop.tags)
    try:
        mi = tags.index('_atom_site.pdbx_PDB_model_num')
    except ValueError:
        return
    nrows = loop.length()
    vals = list(loop.values)
    rows = [vals[i * w:(i + 1) * w] for i in range(nrows)]
    cols = [list(col) for col in zip(*rows)]
    for row in range(nrows):
        cols[mi][row] = str(row // n_atoms_per_frame + 1)
    loop.set_all_values(cols)


def _append_chem_comp_bonds_to_mmcif_block(
    block,
    ligand_mols: list,
    lig_atom_names: List[List[str]],
) -> None:
    """Add ``_chem_comp_bond`` loop for ligand connectivity (single block)."""
    bond_rows: List[List[str]] = []
    for lig_idx, mol in enumerate(ligand_mols):
        names_this = lig_atom_names[lig_idx]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bond.GetBondType().name
            order_str = _BOND_ORDER_MMCIF.get(bt, 'SING')
            arom = 'Y' if bond.GetIsAromatic() else 'N'
            bond_rows.append(['LIG', names_this[i], names_this[j], order_str, arom])

    if bond_rows:
        bond_loop = block.init_loop(
            '_chem_comp_bond.',
            ['comp_id', 'atom_id_1', 'atom_id_2', 'value_order', 'pdbx_aromatic_flag'],
        )
        for row in bond_rows:
            bond_loop.add_row(row)


def _build_gemmi_complex_single_model(
    pocket_feats: Dict,
    ligand_feats: Dict,
    rec_coords: torch.Tensor,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    protein_path: Optional[Union[str, Path]] = None,
):
    """Build a single-model ``gemmi.Structure`` (receptor + ligand) at one timestep."""
    import gemmi

    if protein_path is not None:
        protein_path = Path(protein_path)
        if protein_path.exists():
            st = _build_full_protein_gemmi(
                protein_path, pocket_feats, rec_coords, backbone_com,
            )
        else:
            logger.warning("protein_path %s not found, writing pocket only", protein_path)
            st = _build_pocket_gemmi(pocket_feats, rec_coords, backbone_com)
    else:
        st = _build_pocket_gemmi(pocket_feats, rec_coords, backbone_com)

    model = st[0]
    ligand_mols, _ = _build_ligand_mols(ligand_feats, lig_coords, backbone_com)

    serial = max((a.serial for ch in model for res in ch for a in res), default=0) + 1
    lig_atom_names: List[List[str]] = []

    for lig_idx, mol in enumerate(ligand_mols):
        chain_id = chr(ord('X') - lig_idx) if lig_idx < 3 else f'L{lig_idx}'
        chain = gemmi.Chain(chain_id)
        res = gemmi.Residue()
        res.name = 'LIG'
        res.seqid = gemmi.SeqId(str(lig_idx + 1))
        res.het_flag = 'H'

        conf = mol.GetConformer()
        names_this = []
        for i in range(mol.GetNumAtoms()):
            atom_rd = mol.GetAtomWithIdx(i)
            name = f"{atom_rd.GetSymbol()}{i + 1}"
            names_this.append(name)
            ga = gemmi.Atom()
            ga.name = name
            ga.element = gemmi.Element(atom_rd.GetSymbol())
            pos = conf.GetAtomPosition(i)
            ga.pos = gemmi.Position(pos.x, pos.y, pos.z)
            ga.occ = 1.0
            ga.b_iso = 0.0
            ga.serial = serial
            serial += 1
            res.add_atom(ga)

        lig_atom_names.append(names_this)
        chain.add_residue(res)
        model.add_chain(chain)

    return st, ligand_mols, lig_atom_names


def write_complex_cif(
    pocket_feats: Dict,
    ligand_feats: Dict,
    rec_coords: torch.Tensor,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    output_path: Union[str, Path],
    protein_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write a protein-ligand complex as mmCIF (heavy atoms only).

    Parameters
    ----------
    protein_path : If given, the full protein structure (PDB/CIF) is loaded
        and pocket residues are overlaid with *rec_coords*.  Non-pocket
        residues keep their crystallographic positions.  When ``None``, only
        the pocket residues are written.
    """
    import gemmi

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    st, ligand_mols, lig_atom_names = _build_gemmi_complex_single_model(
        pocket_feats, ligand_feats, rec_coords, lig_coords, backbone_com, protein_path,
    )
    doc = st.make_mmcif_document()
    block = doc.sole_block()
    _append_chem_comp_bonds_to_mmcif_block(block, ligand_mols, lig_atom_names)
    doc.write_file(str(output_path))
    return output_path


def write_protein_pdb(
    pocket_feats: Dict,
    rec_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    output_path: Union[str, Path],
    protein_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the sampled receptor alone as PDB — the file Vina actually needs.

    Equivalent to ``write_complex_cif`` followed by
    ``inference.vina_scoring.extract_protein_pdb``, without the mmCIF round
    trip: scoring a pose only needs the receptor, and re-reading a 400 kB
    complex just to pull the protein back out costs more than writing it once.
    Pocket residues carry *rec_coords*; the rest stay crystallographic.

    Trimming this to a shell around the ligand reproduces the score exactly
    (Vina's terms vanish beyond 8 A) but measured slower: the cost here is
    per-file latency on the shared filesystem, not the byte count.
    """
    import gemmi

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if protein_path is not None and Path(protein_path).exists():
        st = _build_full_protein_gemmi(protein_path, pocket_feats, rec_coords, backbone_com)
    else:
        st = _build_pocket_gemmi(pocket_feats, rec_coords, backbone_com)

    st.write_pdb(str(output_path))
    return output_path


def write_complex_trajectory_cif(
    pocket_feats: Dict,
    ligand_feats: Dict,
    rec_trajectory: torch.Tensor,
    lig_trajectory: torch.Tensor,
    backbone_com: torch.Tensor,
    output_path: Union[str, Path],
    protein_path: Optional[Union[str, Path]] = None,
    stride: int = 1,
) -> Dict[str, Union[str, Path, int]]:
    """Write an integration trajectory as mmCIF for ChimeraX.

    Primary format: **one mmCIF file** with multiple models; gemmi emits
    ``_atom_site.pdbx_PDB_model_num`` so ChimeraX can open it as coordinate
    sets::

        open trajectory.cif coordsets true

    If building the multi-model file fails, falls back to one ``.cif`` per
    frame under ``<stem>_frames/``.

    Parameters
    ----------
    rec_trajectory, lig_trajectory : tensors shaped ``[T, N_rec, 3]`` and
        ``[T, N_lig, 3]`` (centred coordinates, same convention as sampling).
    stride : keep every *stride*-th frame (1 = all frames).
    """
    import gemmi

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(rec_trajectory, torch.Tensor):
        rec_trajectory = rec_trajectory.detach().cpu().float()
    if isinstance(lig_trajectory, torch.Tensor):
        lig_trajectory = lig_trajectory.detach().cpu().float()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu().float()

    T = int(rec_trajectory.shape[0])
    if lig_trajectory.shape[0] != T:
        raise ValueError(
            f"rec_trajectory T={T} != lig_trajectory T={lig_trajectory.shape[0]}"
        )

    indices = list(range(0, T, max(1, int(stride))))
    if not indices:
        indices = [0]

    try:
        st_traj = gemmi.Structure()
        st_traj.name = output_path.stem or 'trajectory'
        ligand_mols_ref = None
        lig_atom_names_ref = None

        for k, t in enumerate(indices):
            st, mols, names = _build_gemmi_complex_single_model(
                pocket_feats,
                ligand_feats,
                rec_trajectory[t],
                lig_trajectory[t],
                backbone_com,
                protein_path,
            )
            if k == 0:
                st_traj.cell = st.cell
                st_traj.spacegroup_hm = st.spacegroup_hm
                ligand_mols_ref = mols
                lig_atom_names_ref = names
            st_traj.add_model(st[0].clone())

        doc = st_traj.make_mmcif_document()
        block = doc.sole_block()
        if ligand_mols_ref is not None and lig_atom_names_ref is not None:
            _append_chem_comp_bonds_to_mmcif_block(
                block, ligand_mols_ref, lig_atom_names_ref,
            )
        n_apf = st_traj[0].count_atom_sites()
        n_expected = len(indices) * n_apf
        atom_tab = doc.sole_block().find_mmcif_category('_atom_site')
        if atom_tab is not None:
            n_atom_rows = atom_tab.loop.length()
            if n_atom_rows != n_expected:
                logger.warning(
                    'atom_site row count %s != %s frames × %s atoms; model numbers may be wrong',
                    n_atom_rows, len(indices), n_apf,
                )
        _fix_atom_site_pdbx_model_numbers(doc, n_apf)
        doc.write_file(str(output_path))
        return {
            'path': output_path,
            'format': 'multi_model_cif',
            'n_frames': len(indices),
            'chimerax_hint': f'open {output_path} coordsets true',
        }
    except Exception as e:
        logger.warning(
            "Multi-model trajectory CIF failed (%s); writing per-frame CIFs",
            e,
        )
        frames_dir = output_path.parent / f'{output_path.stem}_frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        for k, t in enumerate(indices):
            write_complex_cif(
                pocket_feats,
                ligand_feats,
                rec_trajectory[t],
                lig_trajectory[t],
                backbone_com,
                frames_dir / f'frame_{k:04d}.cif',
                protein_path=protein_path,
            )
        return {
            'path': frames_dir,
            'format': 'per_frame_cif',
            'n_frames': len(indices),
            'chimerax_hint': f'open {frames_dir}/frame_*.cif',
        }


def write_complex_cif_from_dir(
    system_dir: Union[str, Path],
    rec_coords: torch.Tensor,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    output_path: Union[str, Path],
) -> Path:
    """Convenience wrapper that loads features from *system_dir*."""
    system_dir = Path(system_dir)
    pf = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    lf = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
    return write_complex_cif(pf, lf, rec_coords, lig_coords, backbone_com, output_path)


# ---------------------------------------------------------------------------
# RMSD from CIF files
# ---------------------------------------------------------------------------

def compute_cif_rmsd(
    pred_path: Union[str, Path],
    gt_path: Union[str, Path],
) -> Dict[str, float]:
    """Compute heavy-atom RMSD between two mmCIF files.

    Atoms are matched by ``(chain_id, residue_name, seqid, atom_name)``.
    Returns per-component RMSD for receptor (standard residues) and
    ligand (``LIG`` residue).
    """
    import gemmi

    pred_st = gemmi.read_structure(str(pred_path))
    gt_st = gemmi.read_structure(str(gt_path))

    def _extract_atoms(st):
        rec, lig = {}, {}
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        key = (chain.name, res.name, str(res.seqid), atom.name)
                        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                        if res.name == 'LIG':
                            lig[key] = pos
                        else:
                            rec[key] = pos
        return rec, lig

    pred_rec, pred_lig = _extract_atoms(pred_st)
    gt_rec, gt_lig = _extract_atoms(gt_st)

    def _rmsd(d1, d2):
        common = sorted(set(d1) & set(d2))
        if not common:
            return float('nan')
        diff = np.stack([d1[k] - d2[k] for k in common])
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    return {
        'receptor_rmsd': _rmsd(pred_rec, gt_rec),
        'ligand_rmsd': _rmsd(pred_lig, gt_lig),
        'n_receptor_atoms_matched': len(set(pred_rec) & set(gt_rec)),
        'n_ligand_atoms_matched': len(set(pred_lig) & set(gt_lig)),
    }


# ---------------------------------------------------------------------------
# Ligand SDF output and coordinate/topology RMSD
# ---------------------------------------------------------------------------

def _stored_atom_order_for_mol(mol, ligand_data: Dict) -> Optional[List[int]]:
    """Return ``stored_idx`` for each atom in *mol*, using canonical-rank names.

    Ligand tensors are indexed in the atom order used during featurization.  The
    stored ``mapping`` names each atom by canonical rank, so it lets us attach
    coordinates to a fresh RDKit molecule parsed from canonical SMILES even when
    RDKit's atom order differs from the stored tensor order.
    """
    from rdkit.Chem import AllChem

    atoms = ligand_data.get('atoms', [])
    mapping = ligand_data.get('mapping')
    if not mapping or mol.GetNumAtoms() != len(atoms):
        return None

    stored_by_key = {}
    for stored_idx, atom_data in enumerate(atoms):
        rank = str(mapping.get(stored_idx, mapping.get(str(stored_idx), '')))
        if not rank:
            return None
        key = (atom_data.get('symbol', '').upper(), rank)
        if key in stored_by_key:
            return None
        stored_by_key[key] = stored_idx

    ranks = AllChem.CanonicalRankAtoms(mol)
    order = []
    for atom in mol.GetAtoms():
        key = (atom.GetSymbol().upper(), str(ranks[atom.GetIdx()] + 1))
        stored_idx = stored_by_key.get(key)
        if stored_idx is None:
            return None
        order.append(stored_idx)
    return order


def _build_ligand_mol_from_source_data(
    ligand_data: Dict,
    abs_coords: np.ndarray,
):
    """Build an RDKit ligand mol from source topology and attach coordinates.

    Returns ``(mol, kekulize_on_write)``.  Prefer the original SMILES graph so
    aromatic/protonation details such as flavin ``[nH]`` survive metric and SDF
    round-trips.  Fall back to stored atom/bond reconstruction for legacy data.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms = ligand_data['atoms']
    mol = None
    coord_order = None
    kekulize_on_write = False

    smiles = ligand_data.get('smiles')
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() == len(atoms):
            coord_order = _stored_atom_order_for_mol(mol, ligand_data)
            if coord_order is None:
                coord_order = list(range(mol.GetNumAtoms()))
            kekulize_on_write = True
        elif mol is not None:
            logger.warning(
                "SMILES atom count %s != stored ligand atom count %s; "
                "falling back to stored topology for %s",
                mol.GetNumAtoms(),
                len(atoms),
                smiles,
            )
            mol = None

    if mol is None:
        bonds = ligand_data['bonds']
        rw_mol = Chem.RWMol()
        for ad in atoms:
            a = Chem.Atom(ad['atomic_number'])
            a.SetFormalCharge(ad['formal_charge'])
            num_explicit_hs = int(ad.get('num_explicit_hs', 0) or 0)
            a.SetNumExplicitHs(num_explicit_hs)
            if num_explicit_hs > 0:
                a.SetNoImplicit(True)
            if ad.get('is_aromatic', False):
                a.SetIsAromatic(True)
            rw_mol.AddAtom(a)

        for bd in bonds:
            rw_mol.AddBond(
                bd['begin_atom_idx'], bd['end_atom_idx'],
                _bond_type_from_str(bd['bond_type']),
            )
            if bd.get('is_aromatic', False):
                b = rw_mol.GetBondBetweenAtoms(
                    bd['begin_atom_idx'], bd['end_atom_idx'],
                )
                b.SetIsAromatic(True)

        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            Chem.SanitizeMol(
                rw_mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        mol = rw_mol.GetMol()
        coord_order = list(range(mol.GetNumAtoms()))
        kekulize_on_write = False

    if abs_coords.shape[0] < mol.GetNumAtoms():
        raise ValueError(
            f"ligand coordinate count {abs_coords.shape[0]} < mol atom count {mol.GetNumAtoms()}"
        )

    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for mol_idx, stored_idx in enumerate(coord_order):
        conf.SetAtomPosition(mol_idx, Point3D(*abs_coords[stored_idx].tolist()))
    mol.AddConformer(conf, assignId=True)
    return mol, kekulize_on_write


def _load_ligand_sdf_mol(path: Union[str, Path], heavy_only: bool = True):
    """Load one ligand SDF as an RDKit mol for metric use."""
    from rdkit import Chem

    path = Path(path)
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    mol = suppl[0] if suppl and len(suppl) else None
    if mol is None:
        suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = suppl[0] if suppl and len(suppl) else None
        if mol is None:
            raise ValueError(f"Failed to read ligand SDF: {path}")
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    return Chem.RemoveHs(mol, sanitize=False) if heavy_only else mol


def compute_ligand_rmsd_from_coords(
    ligand_data: Dict,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    reference_ligand_path: Union[str, Path],
    heavy_only: bool = True,
) -> float:
    """Compute ligand RMSD directly from predicted coords plus source topology.

    This mirrors the safer FlexDock-style metric path: predicted coordinates are
    attached to the original ligand RDKit graph, and RMSD is computed against the
    reference ligand mol.  The predicted SDF file is not parsed or trusted as a
    metric intermediate.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    if isinstance(lig_coords, torch.Tensor):
        lig_coords = lig_coords.detach().cpu()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu()

    abs_coords = (lig_coords + backbone_com).numpy().astype(np.float64)
    pred_mol, _ = _build_ligand_mol_from_source_data(ligand_data, abs_coords)
    ref_mol = _load_ligand_sdf_mol(reference_ligand_path, heavy_only=heavy_only)
    if heavy_only:
        pred_mol = Chem.RemoveHs(pred_mol, sanitize=False)

    try:
        return float(rdMolAlign.CalcRMS(ref_mol, pred_mol))
    except Exception:
        pass

    try:
        pred_rebonded = AllChem.AssignBondOrdersFromTemplate(ref_mol, pred_mol)
        return float(rdMolAlign.CalcRMS(ref_mol, pred_rebonded))
    except Exception:
        pass

    if pred_mol.GetNumAtoms() == ref_mol.GetNumAtoms():
        pred_nums = [a.GetAtomicNum() for a in pred_mol.GetAtoms()]
        ref_nums = [a.GetAtomicNum() for a in ref_mol.GetAtoms()]
        if pred_nums == ref_nums:
            pc = pred_mol.GetConformer()
            rc = ref_mol.GetConformer()
            diff = []
            for i in range(pred_mol.GetNumAtoms()):
                pp = pc.GetAtomPosition(i)
                rp = rc.GetAtomPosition(i)
                diff.append([pp.x - rp.x, pp.y - rp.y, pp.z - rp.z])
            arr = np.asarray(diff, dtype=np.float64)
            return float(np.sqrt((arr ** 2).sum(axis=1).mean()))

    return float('nan')


def write_ligand_sdf(
    ligand_data: Dict,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    output_path: Union[str, Path],
) -> Path:
    """Write a ligand to an SDF file with sampled coordinates.

    Coordinates are de-centred using *backbone_com* before writing.
    """
    from rdkit import Chem

    output_path = Path(output_path)

    if isinstance(lig_coords, torch.Tensor):
        lig_coords = lig_coords.detach().cpu()
    if isinstance(backbone_com, torch.Tensor):
        backbone_com = backbone_com.detach().cpu()

    abs_coords = (lig_coords + backbone_com).numpy().astype(np.float64)
    mol, kekulize_on_write = _build_ligand_mol_from_source_data(ligand_data, abs_coords)

    # Reconstructed fallback mols can retain aromatic flags without a
    # kekulizable graph, while SMILES-derived mols need normal kekulized SDF
    # output so aromatic [nH] atoms round-trip through RDKit.
    writer = Chem.SDWriter(str(output_path))
    writer.SetKekulize(kekulize_on_write)
    writer.write(mol)
    writer.close()
    return output_path


# ---------------------------------------------------------------------------
# Bond type lookup
# ---------------------------------------------------------------------------

_BOND_TYPE_STR = {
    'SINGLE': None,   # filled at first import
    'DOUBLE': None,
    'TRIPLE': None,
    'AROMATIC': None,
}


def _bond_type_from_str(s: str):
    """Lazy-init bond type map (avoids importing rdkit at module level)."""
    from rdkit import Chem
    if _BOND_TYPE_STR['SINGLE'] is None:
        _BOND_TYPE_STR.update({
            'SINGLE': Chem.BondType.SINGLE,
            'DOUBLE': Chem.BondType.DOUBLE,
            'TRIPLE': Chem.BondType.TRIPLE,
            'AROMATIC': Chem.BondType.AROMATIC,
        })
    return _BOND_TYPE_STR.get(s, _BOND_TYPE_STR['SINGLE'])
