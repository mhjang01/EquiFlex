"""
MMFF-based energy minimization for sampled protein-ligand complexes.

Pocket mols are built from amino-acid SDF templates (prep/.refs/),
ligand mols from stored atom/bond data (ligands.pkl).
Backbone atoms are fixed; sidechains and ligand atoms relax.
Uses RDKit MMFF94s with UFF fallback.

Dependencies: torch, numpy, gemmi, rdkit, joblib
"""

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


def _build_full_protein_gemmi(
    protein_path: Union[str, Path],
    pocket_feats: Dict,
    rec_coords: torch.Tensor,
    backbone_com: torch.Tensor,
) -> 'gemmi.Structure':
    """Load full protein and overlay predicted pocket residue coordinates.

    Non-pocket residues keep their original crystallographic positions.
    Pocket residues are updated with the sampler-predicted coords.
    """
    import gemmi

    protein_path = Path(protein_path)
    st = gemmi.read_structure(str(protein_path))
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
    if not raw_overlay or len(raw_overlay) != len(residue_keys):
        overlay_keys = [
            (str(cid).upper(), str(rn).upper(), int(rnum), '')
            for (cid, rn, rnum, _eid) in residue_keys
        ]
    else:
        overlay_keys = raw_overlay

    pocket_lookup: Dict[Tuple[str, str, int, str], Tuple[int, int]] = {}
    atom_offset = 0
    for res_idx, (chain_id, resname, struct_num, icode) in enumerate(overlay_keys):
        n_atoms = n_atoms_per_residue[res_idx]
        pocket_lookup[(chain_id, resname, int(struct_num), icode)] = (atom_offset, res_idx)
        atom_offset += n_atoms

    for chain in model:
        cid = chain.name.upper()
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
# Pocket mol from SDF templates  (for MMFF relaxation)
# ---------------------------------------------------------------------------

_REFS_DIR = Path(__file__).parent.parent / 'prep' / '.refs'

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


def _build_pocket_mol(
    pocket_feats: Dict,
    coords: torch.Tensor,
    backbone_com: torch.Tensor,
):
    """Build pocket RDKit mol from amino-acid SDF templates.

    Returns ``(pocket_mol_h, backbone_indices, gi_to_heavy)`` where

    * *backbone_indices* — atom indices in the H-added mol for backbone
      heavy atoms (N, CA, C, O) and their attached hydrogens.
    * *gi_to_heavy* — ``{gi: heavy_idx}`` mapping a pocket_feats global
      atom index to the corresponding heavy-atom index in *pocket_mol_h*
      (indices are stable after ``AddHs``).
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

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

    template_cache: Dict[str, Optional['Chem.Mol']] = {}
    combined = None
    backbone_heavy_global: set = set()
    gi_to_heavy: Dict[int, int] = {}   # pocket_feats gi → heavy-atom idx in pocket_mol_h
    global_offset = 0
    atom_offset = 0

    for res_idx, (chain_id, resname, resnum, _entity_id) in enumerate(residue_keys):
        n_atoms = n_atoms_per_residue[res_idx]

        if resname not in template_cache:
            sdf_path = _REFS_DIR / f'{resname}.sdf'
            if not sdf_path.exists():
                sdf_path = _REFS_DIR / 'UNK.sdf'
            supp = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
            template_cache[resname] = supp[0] if len(supp) > 0 else None

        tmpl = template_cache[resname]
        if tmpl is None:
            atom_offset += n_atoms
            continue

        present = [i for i in range(n_atoms)
                    if not missing_mask[atom_offset + i] and i < tmpl.GetNumAtoms()]
        if not present:
            atom_offset += n_atoms
            continue

        old_to_new: Dict[int, int] = {}
        res_mol = Chem.RWMol()
        for new_idx, old_idx in enumerate(present):
            src = tmpl.GetAtomWithIdx(old_idx)
            a = Chem.Atom(src.GetAtomicNum())
            a.SetFormalCharge(src.GetFormalCharge())
            res_mol.AddAtom(a)
            old_to_new[old_idx] = new_idx

        for bond in tmpl.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in old_to_new and a2 in old_to_new:
                res_mol.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

        try:
            Chem.SanitizeMol(res_mol)
        except Exception:
            Chem.SanitizeMol(
                res_mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        mol = res_mol.GetMol()

        conf = Chem.Conformer(mol.GetNumAtoms())
        for new_idx, old_idx in enumerate(present):
            gi = atom_offset + old_idx
            conf.SetAtomPosition(new_idx, Point3D(*abs_coords[gi].tolist()))
            gi_to_heavy[gi] = global_offset + old_to_new[old_idx]
        mol.AddConformer(conf, assignId=True)

        for old_idx in present:
            if old_idx < 4:
                backbone_heavy_global.add(global_offset + old_to_new[old_idx])

        global_offset += mol.GetNumAtoms()
        combined = mol if combined is None else Chem.CombineMols(combined, mol)
        atom_offset += n_atoms

    if combined is None:
        raise ValueError("No pocket residues could be built from templates")

    combined_h = Chem.AddHs(combined, addCoords=True)

    backbone_fixed = set(backbone_heavy_global)
    for i in range(combined_h.GetNumAtoms()):
        atom = combined_h.GetAtomWithIdx(i)
        if atom.GetAtomicNum() == 1:
            for nb in atom.GetNeighbors():
                if nb.GetIdx() in backbone_heavy_global:
                    backbone_fixed.add(i)
                    break

    return combined_h, backbone_fixed, gi_to_heavy


# ---------------------------------------------------------------------------
# Ligand mol from stored data  (for MMFF relaxation)
# ---------------------------------------------------------------------------

def _build_ligand_mol_from_data(
    ligand_data: Dict,
    coords_abs: np.ndarray,
):
    """Build an RDKit mol (with H) directly from ``ligands.pkl`` data.

    Atoms are in SDF order — no canonical-rank remapping needed.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms = ligand_data['atoms']
    bonds = ligand_data['bonds']

    mol = Chem.RWMol()
    for ad in atoms:
        a = Chem.Atom(ad['atomic_number'])
        a.SetFormalCharge(ad['formal_charge'])
        a.SetNumExplicitHs(ad.get('num_explicit_hs', 0))
        a.SetNoImplicit(True)
        mol.AddAtom(a)

    for bd in bonds:
        mol.AddBond(
            bd['begin_atom_idx'], bd['end_atom_idx'],
            _bond_type_from_str(bd['bond_type']),
        )
        if bd.get('is_aromatic', False):
            b = mol.GetBondBetweenAtoms(bd['begin_atom_idx'], bd['end_atom_idx'])
            b.SetIsAromatic(True)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    mol = mol.GetMol()

    n_heavy = len(atoms)
    conf = Chem.Conformer(n_heavy)
    for i in range(n_heavy):
        conf.SetAtomPosition(i, Point3D(*coords_abs[i].tolist()))
    mol.AddConformer(conf, assignId=True)

    mol_h = Chem.AddHs(mol, addCoords=True)
    return mol_h


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def relax_complex(
    system_dir: Union[str, Path],
    rec_coords: torch.Tensor,
    lig_coords: torch.Tensor,
    backbone_com: torch.Tensor,
    output_dir: Optional[Union[str, Path]] = None,
    max_iterations: int = 200,
    mmff_variant: str = 'MMFF94s',
    write_structures: bool = True,
    lig_restraint_k: float = 10.0,
) -> Dict:
    """Energy-minimise predicted ligand(s) with MMFF in the pocket context.

    Pocket backbone atoms are fixed; sidechains and ligand atoms move.
    Ligand heavy atoms are harmonically restrained to their predicted positions
    so MMFF resolves clashes without displacing the pose.
    Pocket is built from amino-acid SDF templates (``prep/.refs/``),
    ligand from stored data in ``ligands.pkl``.
    Uses RDKit MMFF (with UFF fallback).  MMFF setup may log ``Can't kekulize mol``
    when kekulization fails; that path is skipped in favour of UFF, or unrelaxed
    coordinates if both fail.  Plain SDF output (``write_ligand_sdf``) does not
    use MMFF and is unaffected.

    Parameters
    ----------
    system_dir : path to the featurised system (must contain
        ``pocket_features.pt``, ``ligand_features.pt``, ``ligands.pkl``).
    rec_coords : [N_rec, 3] centred receptor coordinates.
    lig_coords : [N_lig, 3] centred ligand coordinates.
    backbone_com : [3] backbone centre of mass used for centring.
    max_iterations : MMFF minimisation steps (default 200).
    mmff_variant : ``'MMFF94s'`` (default) or ``'MMFF94'``.
    write_structures : write ``predicted.cif`` and ``relaxed.cif``.
    lig_restraint_k : harmonic force constant (kcal/mol/Å²) applied to ligand
        heavy atoms to prevent large pose displacement (default 10.0).
        Set to 0.0 to disable. Only applied for MMFF (not UFF fallback).

    Returns
    -------
    dict with ``rec_coords``, ``lig_coords`` (centred; relaxed when minimisation
    succeeds, else same as input prediction), ``initial_energy_kcal_mol``,
    ``final_energy_kcal_mol``, and optionally CIF paths.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from joblib import load as joblib_load

    system_dir = Path(system_dir)
    pocket_feats = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
    ligands_data = joblib_load(system_dir / 'ligands.pkl')

    # ── 1. Pocket mol from SDF templates ──────────────────────────────
    pocket_mol_h, backbone_fixed, gi_to_heavy = _build_pocket_mol(
        pocket_feats, rec_coords, backbone_com,
    )
    n_pocket_h = pocket_mol_h.GetNumAtoms()

    # ── 2. Prepare coordinates (absolute) ────────────────────────────
    if isinstance(backbone_com, torch.Tensor):
        bcom_t = backbone_com.detach().cpu()
    else:
        bcom_t = backbone_com
    bcom = bcom_t.numpy() if hasattr(bcom_t, 'numpy') else np.asarray(bcom_t)

    if isinstance(rec_coords, torch.Tensor):
        rec_coords_cpu = rec_coords.detach().cpu()
    else:
        rec_coords_cpu = rec_coords
    abs_rec = (rec_coords_cpu + bcom_t).numpy().astype(np.float64)

    if isinstance(lig_coords, torch.Tensor):
        lig_coords_cpu = lig_coords.detach().cpu()
    else:
        lig_coords_cpu = lig_coords
    abs_lig = (lig_coords_cpu + bcom_t).numpy().astype(np.float64)

    n_atoms_per_lig = ligand_feats.get('n_atoms_per_ligand', [abs_lig.shape[0]])

    relaxed_lig_parts: List[np.ndarray] = []
    initial_energies: List[float] = []
    final_energies: List[float] = []
    # Predicted receptor coords unless a minimisation pass updates them
    relaxed_rec_abs = abs_rec.copy()

    lig_offset = 0
    for lig_idx, lig_data in enumerate(ligands_data):
        n_heavy = n_atoms_per_lig[lig_idx]
        lig_abs = abs_lig[lig_offset:lig_offset + n_heavy]

        try:
            # ── 3. Ligand mol from stored data (SDF order) ───────────
            mol_h = _build_ligand_mol_from_data(lig_data, lig_abs)
            n_lig_h = mol_h.GetNumAtoms()

            heavy_indices = [
                i for i in range(n_lig_h)
                if mol_h.GetAtomWithIdx(i).GetAtomicNum() != 1
            ]

            # ── 4. Combine: ligand first, pocket second ──────────────
            combined = Chem.CombineMols(mol_h, pocket_mol_h)
            try:
                Chem.SanitizeMol(combined)
            except Exception:
                Chem.SanitizeMol(
                    combined,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                )

            # ── 5. Setup force field (MMFF → UFF fallback) ───────────
            # MMFF may log "Can't kekulize mol" when MMFFGetMoleculeProperties fails.
            ff = None
            mp = AllChem.MMFFGetMoleculeProperties(combined, mmffVariant=mmff_variant)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(combined, mp)

            use_uff = ff is None
            if use_uff:
                logger.info("MMFF failed for ligand %d, falling back to UFF", lig_idx)
                ff = AllChem.UFFGetMoleculeForceField(combined)

            if ff is None:
                logger.warning(
                    "All force fields failed for ligand %d — using unrelaxed coords",
                    lig_idx,
                )
                relaxed_lig_parts.append(lig_abs.copy())
                lig_offset += n_heavy
                continue

            for pocket_i in range(n_pocket_h):
                if pocket_i in backbone_fixed:
                    combined_i = n_lig_h + pocket_i
                    if use_uff:
                        ff.AddFixedPoint(combined_i)
                    else:
                        ff.MMFFAddPositionConstraint(combined_i, 0.0, 1e5)

            if not use_uff and lig_restraint_k > 0.0:
                for i in heavy_indices:
                    ff.MMFFAddPositionConstraint(i, 0.0, lig_restraint_k)

            # ── 6. Minimise ──────────────────────────────────────────
            initial_e = ff.CalcEnergy()
            ff.Minimize(maxIts=max_iterations)
            final_e = ff.CalcEnergy()

            initial_energies.append(initial_e)
            final_energies.append(final_e)

            # ── 7. Extract ligand + pocket coords ────────────────────
            conf = combined.GetConformer()
            relaxed_coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in heavy_indices
            ])
            relaxed_lig_parts.append(relaxed_coords)

            for gi, heavy_idx in gi_to_heavy.items():
                combined_idx = n_lig_h + heavy_idx
                p = conf.GetAtomPosition(combined_idx)
                relaxed_rec_abs[gi] = [p.x, p.y, p.z]

        except Exception as e:
            logger.warning(
                "Relaxation failed for ligand %d (%s) — using unrelaxed coords",
                lig_idx,
                e,
            )
            relaxed_lig_parts.append(lig_abs.copy())

        lig_offset += n_heavy

    relaxed_lig_abs = np.concatenate(relaxed_lig_parts, axis=0)
    relaxed_lig_centered = torch.tensor(relaxed_lig_abs - bcom, dtype=torch.float32)
    relaxed_rec_centered = torch.tensor(relaxed_rec_abs - bcom, dtype=torch.float32)

    initial_energy = sum(initial_energies) if initial_energies else 0.0
    final_energy = sum(final_energies) if final_energies else 0.0
    logger.info("MMFF energy: %.2f -> %.2f kcal/mol", initial_energy, final_energy)

    results: Dict = {
        'rec_coords': relaxed_rec_centered,
        'lig_coords': relaxed_lig_centered,
        'initial_energy_kcal_mol': initial_energy,
        'final_energy_kcal_mol': final_energy,
        'minimization_ok': bool(initial_energies),
    }

    # ── 9. Write structures ───────────────────────────────────────────
    if write_structures and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_cif = write_complex_cif(
            pocket_feats, ligand_feats, rec_coords, lig_coords,
            backbone_com, output_dir / 'predicted.cif',
        )
        results['predicted_cif'] = pred_cif

        relaxed_cif = write_complex_cif(
            pocket_feats, ligand_feats,
            relaxed_rec_centered, relaxed_lig_centered,
            backbone_com, output_dir / 'relaxed.cif',
        )
        results['relaxed_cif'] = relaxed_cif

    return results
