"""Torch reimplementation of the AutoDock Vina / Vinardo scoring function.

Reproduces the intermolecular affinity reported by ``gnina --score_only``
(``smina`` lineage) without the external binary. Used to re-score / rank
generated protein-ligand poses.

The torch port and the real AutoDock Vina binary do not agree pose-for-pose:
hydrogen placement (RDKit vs Open Babel) and nitrogen acceptor typing differ.
Benchmark pose selection therefore uses the binary via :func:`vina_binary_score`
(``vina --score_only`` on Open Babel pdbqt files);
the torch port serves guidance-free relative ranking and the ``summary.md``
written by ``run_inference.py``.

The reported Vina affinity is

    affinity = E_inter / (1 + 0.05846 * num_rot)

where ``E_inter`` is the weighted sum over intermolecular heavy-atom pairs
(receptor x ligand) within an 8 A cutoff of five pairwise terms:

    gauss1 / gauss2 / repulsion / hydrophobic / non_dir_h_bond

Term forms, weights and atom parameters are ported verbatim from gnina
(gninasrc/lib/everything.h, builtinscoring.cpp, atom_constants.h).

Only ``xs_radius`` and the hydrophobe / donor / acceptor flags enter the
energy, so aromatic-vs-aliphatic carbon, sulfur acceptor state, etc. are
irrelevant to the score. The single approximate step versus gnina is the
H-bond *acceptor* test for nitrogen (gnina delegates to OpenBabel); see
``_nitrogen_is_acceptor``. If cross-check numbers diverge, inspect N typing
and hydrogen placement first.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Atom-type table (xs_radius, hydrophobe, donor, acceptor) — gnina default_data
# Index 0 is reserved for "no heavy type" / hydrogen (excluded from scoring).

_TYPE_TABLE: List[Tuple[str, float, bool, bool, bool]] = [
    # name,                     xs_radius, hydrophobe, donor, acceptor
    ("Carbon_H",                 1.9, True,  False, False),  # heteroBonded=False
    ("Carbon_P",                 1.9, False, False, False),  # heteroBonded=True
    ("Nitrogen",                 1.8, False, False, False),
    ("NitrogenXSDonor",          1.8, False, True,  False),
    ("NitrogenXSDonorAcceptor",  1.8, False, True,  True),
    ("NitrogenXSAcceptor",       1.8, False, False, True),
    ("Oxygen",                   1.7, False, False, False),
    ("OxygenXSDonor",            1.7, False, True,  False),
    ("OxygenXSDonorAcceptor",    1.7, False, True,  True),
    ("OxygenXSAcceptor",         1.7, False, False, True),
    ("Sulfur",                   2.0, False, False, False),
    ("Phosphorus",               2.1, False, False, False),
    ("Fluorine",                 1.5, True,  False, False),
    ("Chlorine",                 1.8, True,  False, False),
    ("Bromine",                  2.0, True,  False, False),
    ("Iodine",                   2.2, True,  False, False),
    ("Metal",                    1.2, False, True,  False),
    ("Boron",                    1.92, True, False, False),
    ("GenericHeavy",             1.8, False, False, False),
]

_TYPE_INDEX: Dict[str, int] = {name: i for i, (name, *_) in enumerate(_TYPE_TABLE)}

# Vina term weights (builtinscoring.cpp:43-48) and Vinardo (52-56).
_VINA_WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "h_bond": -0.587439,
}
_VINARDO_WEIGHTS = {
    "gauss1": -0.045,
    "gauss2": 0.0,        # vinardo has no second gaussian
    "repulsion": 0.80,
    "hydrophobic": -0.035,
    "h_bond": -0.60,
}

# Per-function geometry parameters (offset, width / good / bad).
_VINA_PARAMS = {
    "gauss1": dict(offset=0.0, width=0.5),
    "gauss2": dict(offset=3.0, width=2.0),
    "hydrophobic": dict(good=0.5, bad=1.5),
    "h_bond": dict(good=-0.7, bad=0.0),
}
_VINARDO_PARAMS = {
    "gauss1": dict(offset=0.0, width=0.8),
    "gauss2": dict(offset=3.0, width=2.0),
    "hydrophobic": dict(good=0.0, bad=2.5),
    "h_bond": dict(good=-0.6, bad=0.0),
}

CUTOFF = 8.0          # angstrom, charge_independent default
NUM_TORS_COEF = 0.05846  # vina conf-independent rotatable-bond normalization


@dataclass
class AtomTyping:
    """Heavy-atom XS types + coordinates for one molecule."""

    coords: torch.Tensor          # [N, 3] float
    types: torch.Tensor           # [N] long, index into _TYPE_TABLE
    names: List[str] = field(default_factory=list)  # debug: per-atom type name


# Atom typing
def _nitrogen_is_acceptor(atom: Chem.Atom) -> bool:
    """Approximate OpenBabel ``IsHbondAcceptor`` for nitrogen.

    Pyridine-like N (aromatic, no H, two ring neighbours) and neutral
    sp3/sp2 amines act as acceptors; protonated, pyrrole-type, amide and
    nitro nitrogens do not. This is the one heuristic step versus gnina.
    """
    if atom.GetFormalCharge() > 0:
        return False
    if atom.GetIsAromatic():
        # pyridine-type lone pair: aromatic, no attached H
        return atom.GetTotalNumHs() == 0
    # amide / sulfonamide N (bonded to a carbonyl/sulfonyl) -> donor, not acceptor
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() in (6, 16):
            for b in nbr.GetBonds():
                other = b.GetOtherAtom(nbr)
                if (
                    other.GetIdx() != atom.GetIdx()
                    and other.GetAtomicNum() == 8
                    and b.GetBondType() == Chem.BondType.DOUBLE
                ):
                    return False
    # nitro / nitrile etc. with no lone pair available
    if atom.GetTotalValence() - atom.GetTotalNumHs() >= 4:
        return False
    return True


def _atom_xs_type(atom: Chem.Atom) -> Optional[str]:
    """Map a heavy RDKit atom (mol must carry explicit H) to an XS type name.

    Follows gnina ``obatom_to_smina_type`` + ``adjust_smina_type``. Returns
    ``None`` for hydrogen (excluded from the heavy-atom score).
    """
    z = atom.GetAtomicNum()
    if z == 1:
        return None

    hbonded = any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
    hetero_bonded = any(
        n.GetAtomicNum() not in (1, 6) for n in atom.GetNeighbors()
    )

    if z == 6:  # carbon: hydrophobe unless bonded to a heteroatom
        return "Carbon_P" if hetero_bonded else "Carbon_H"
    if z == 7:  # nitrogen
        acceptor = _nitrogen_is_acceptor(atom)
        if acceptor:
            return "NitrogenXSDonorAcceptor" if hbonded else "NitrogenXSAcceptor"
        return "NitrogenXSDonor" if hbonded else "Nitrogen"
    if z == 8:  # oxygen: always acceptor-typed, donor if it carries an H
        return "OxygenXSDonorAcceptor" if hbonded else "OxygenXSAcceptor"
    if z == 16:
        return "Sulfur"
    if z == 15:
        return "Phosphorus"
    if z == 9:
        return "Fluorine"
    if z == 17:
        return "Chlorine"
    if z == 35:
        return "Bromine"
    if z == 53:
        return "Iodine"
    if z == 5:
        return "Boron"
    if z in (12, 25, 30, 20, 26, 11, 19, 27, 29, 28, 13, 3):  # common metals
        return "Metal"
    return "GenericHeavy"


def assign_xs_types(
    mol: Chem.Mol,
    conf_id: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> AtomTyping:
    """Type the heavy atoms of *mol* and pull their coordinates.

    *mol* must have explicit hydrogens (``Chem.AddHs(mol, addCoords=True)``)
    and a 3D conformer — hydrogen presence drives the donor classification.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformer; need 3D coordinates to score")
    conf = mol.GetConformer(conf_id)

    coords: List[List[float]] = []
    type_idx: List[int] = []
    names: List[str] = []
    for atom in mol.GetAtoms():
        name = _atom_xs_type(atom)
        if name is None:  # hydrogen
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
        type_idx.append(_TYPE_INDEX[name])
        names.append(name)

    return AtomTyping(
        coords=torch.tensor(coords, dtype=torch.float32, device=device),
        types=torch.tensor(type_idx, dtype=torch.long, device=device),
        names=names,
    )


# Energy terms
def _gaussian(x: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp(-((x / width) ** 2))


def _slope_step(x: torch.Tensor, bad: float, good: float) -> torch.Tensor:
    """Piecewise-linear ramp, 0 at *bad* -> 1 at *good* (gnina slope_step)."""
    if bad < good:
        return ((x - bad) / (good - bad)).clamp(0.0, 1.0)
    return ((x - bad) / (good - bad)).clamp(0.0, 1.0)


def _param_tensors(device: torch.device):
    radii = torch.tensor([t[1] for t in _TYPE_TABLE], device=device)
    hydro = torch.tensor([t[2] for t in _TYPE_TABLE], device=device, dtype=torch.bool)
    donor = torch.tensor([t[3] for t in _TYPE_TABLE], device=device, dtype=torch.bool)
    accept = torch.tensor([t[4] for t in _TYPE_TABLE], device=device, dtype=torch.bool)
    return radii, hydro, donor, accept


def vina_terms(
    rec: AtomTyping,
    lig: AtomTyping,
    scoring: str = "vina",
) -> Dict[str, torch.Tensor]:
    """Unweighted per-term sums over intermolecular heavy-atom pairs.

    Returns a dict with the five raw term sums (gauss1, gauss2, repulsion,
    hydrophobic, h_bond) — the quantities printed by gnina ``--score_only``
    on the ``Term values, before weighting`` line, for cross-checking.
    """
    params = _VINA_PARAMS if scoring == "vina" else _VINARDO_PARAMS
    device = rec.coords.device
    radii, hydro, donor, accept = _param_tensors(device)

    # pairwise distances [Nr, Nl]
    diff = rec.coords[:, None, :] - lig.coords[None, :, :]
    r = diff.norm(dim=-1)

    rt, lt = rec.types, lig.types
    d0 = radii[rt][:, None] + radii[lt][None, :]      # optimal distance
    surf = r - d0                                     # surface distance

    within = r < CUTOFF

    g1 = _gaussian(surf - params["gauss1"]["offset"], params["gauss1"]["width"])
    g2 = _gaussian(surf - params["gauss2"]["offset"], params["gauss2"]["width"])

    rep = torch.where(surf < 0, surf * surf, torch.zeros_like(surf))

    hphob_pair = hydro[rt][:, None] & hydro[lt][None, :]
    hyd = torch.where(
        hphob_pair,
        _slope_step(surf, params["hydrophobic"]["bad"], params["hydrophobic"]["good"]),
        torch.zeros_like(surf),
    )

    # h-bond possible: donor x acceptor in either order
    hb_pair = (donor[rt][:, None] & accept[lt][None, :]) | (
        accept[rt][:, None] & donor[lt][None, :]
    )
    hb = torch.where(
        hb_pair,
        _slope_step(surf, params["h_bond"]["bad"], params["h_bond"]["good"]),
        torch.zeros_like(surf),
    )

    m = within.float()
    return {
        "gauss1": (g1 * m).sum(),
        "gauss2": (g2 * m).sum(),
        "repulsion": (rep * m).sum(),
        "hydrophobic": (hyd * m).sum(),
        "h_bond": (hb * m).sum(),
    }


def vina_score(
    rec: AtomTyping,
    lig: AtomTyping,
    num_rot: int,
    scoring: str = "vina",
) -> Dict[str, float]:
    """Full Vina affinity for one pose.

    Args:
        rec, lig: typed receptor / ligand heavy atoms.
        num_rot: ligand rotatable-bond count (Vina conf-independent input).
        scoring: ``"vina"`` or ``"vinardo"``.

    Returns dict with raw terms, weighted intermolecular energy, the
    rotatable-bond normalization factor, and the final ``affinity`` (kcal/mol).
    """
    weights = _VINA_WEIGHTS if scoring == "vina" else _VINARDO_WEIGHTS
    terms = vina_terms(rec, lig, scoring=scoring)

    e_inter = sum(weights[k] * terms[k] for k in weights)
    norm = 1.0 + NUM_TORS_COEF * num_rot
    affinity = float(e_inter) / norm

    out = {k: float(v) for k, v in terms.items()}
    out.update(
        intermolecular_energy=float(e_inter),
        num_rot=int(num_rot),
        norm_factor=norm,
        affinity=affinity,
    )
    return out


# Loaders / high-level entry points
def count_rotatable_bonds(mol: Chem.Mol) -> int:
    """Strict rotatable-bond count, approximating Vina's active torsions.

    Note: gnina derives ``num_tors`` from OpenBabel's pdbqt torsion tree,
    which differs from RDKit's count for amide / terminal bonds. For an exact
    affinity match use :func:`count_active_torsions` on the ligand file.
    """
    return rdMolDescriptors.CalcNumRotatableBonds(
        mol, rdMolDescriptors.NumRotatableBondsOptions.Strict
    )


def count_active_torsions(
    ligand_path: Union[str, Path],
    obabel: Union[str, Path] = "obabel",
) -> Optional[int]:
    """gnina-exact ``num_tors``: active torsions in OpenBabel's pdbqt tree.

    Returns the number of ``BRANCH`` records OpenBabel emits when converting
    the ligand to pdbqt — identical to the rotatable-bond count smina/gnina
    feed to the conf-independent normalization. Returns ``None`` if the
    ``obabel`` binary is unavailable.
    """
    if shutil.which(str(obabel)) is None:
        return None
    proc = subprocess.run(
        [str(obabel), str(ligand_path), "-opdbqt"],
        capture_output=True,
        text=True,
    )
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("BRANCH"))


def load_ligand(path: Union[str, Path]) -> Chem.Mol:
    """Read a ligand SDF (first record), add explicit H with coordinates."""
    supp = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = next((m for m in supp if m is not None), None)
    if mol is None:
        raise ValueError(f"no valid molecule in {path}")
    return Chem.AddHs(mol, addCoords=True)


def load_protein(path: Union[str, Path]) -> Chem.Mol:
    """Read a protein from PDB / mmCIF, sanitize, add H with coordinates.

    Hydrogen placement here is RDKit's, not OpenBabel's; for an exact gnina
    cross-check feed identically protonated inputs to both, or compare the
    unweighted term sums from :func:`vina_terms`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        mol = Chem.MolFromMmcifFile(str(path), sanitize=False, removeHs=False)
    else:
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"failed to parse protein {path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    return Chem.AddHs(mol, addCoords=True)


def score_files(
    protein_path: Union[str, Path],
    ligand_sdf_path: Union[str, Path],
    scoring: str = "vina",
    num_rot: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Score a pose given a protein file and a ligand SDF (gnina cross-check path).

    Pass *num_rot* to match gnina's reported rotatable-bond count exactly;
    otherwise it is estimated from the ligand topology.
    """
    lig_mol = load_ligand(ligand_sdf_path)
    rec_mol = load_protein(protein_path)
    if num_rot is None:
        num_rot = count_active_torsions(ligand_sdf_path)
        if num_rot is None:  # no obabel; fall back to RDKit estimate
            num_rot = count_rotatable_bonds(lig_mol)

    rec = assign_xs_types(rec_mol, device=device)
    lig = assign_xs_types(lig_mol, device=device)
    return vina_score(rec, lig, num_rot=num_rot, scoring=scoring)


def score_mols(
    rec_mol: Chem.Mol,
    lig_mol: Chem.Mol,
    scoring: str = "vina",
    num_rot: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Score a pose from pre-built, H-added RDKit mols (production path).

    Use with a protonated pocket mol and the ligand mol.
    """
    if num_rot is None:
        num_rot = count_rotatable_bonds(lig_mol)
    rec = assign_xs_types(rec_mol, device=device)
    lig = assign_xs_types(lig_mol, device=device)
    return vina_score(rec, lig, num_rot=num_rot, scoring=scoring)


# Real AutoDock Vina binary (``--score_only``)
_VINA_AFFINITY = re.compile(r"Affinity:\s*(-?\d+\.?\d*)")


def extract_protein_pdb(
    complex_path: Union[str, Path],
    out_pdb: Union[str, Path],
    setup_entities: bool = False,
) -> bool:
    """Write the amino-acid residues of a predicted ``complex.cif`` as PDB.

    The receptor is rebuilt from the pose's own complex file, so the sampled
    side chains — not the input receptor — are what gets scored. Returns
    ``False`` if the file holds no protein residues. *setup_entities* runs
    gemmi's entity assignment first (adds ``TER`` records), as the PoseBusters
    evaluator does; the Vina path leaves it off.
    """
    import gemmi

    source = gemmi.read_structure(str(complex_path))
    protein = gemmi.Structure()
    protein.name = source.name
    protein.add_model(gemmi.Model("1"))
    for source_chain in source[0]:
        chain = gemmi.Chain(source_chain.name)
        for residue in source_chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info and info.is_amino_acid():
                chain.add_residue(residue)
        if len(chain):
            protein[0].add_chain(chain)
    if not len(protein[0]):
        return False
    if setup_entities:
        protein.setup_entities()
    protein.write_pdb(str(out_pdb))
    return True


def vina_binary_score(
    protein_pdb: Union[str, Path],
    ligand_sdf: Union[str, Path],
    vina: Union[str, Path] = "vina",
    obabel: Union[str, Path] = "obabel",
    timeout: float = 300.0,
) -> Dict[str, Optional[object]]:
    """Affinity from the real AutoDock Vina binary in ``--score_only`` mode.

    Both molecules are protonated and converted to pdbqt by Open Babel
    (receptor with ``-xr`` so it carries no torsion tree; ``ROOT``/``BRANCH``
    records Open Babel still emits are stripped because Vina rejects a rigid
    receptor that has them). Returns ``{"affinity": float | None, "error":
    str | None}`` — ``affinity`` is the kcal/mol value on Vina's ``Affinity:``
    line, ``None`` with a reason if any stage failed.
    """
    result: Dict[str, Optional[object]] = {"affinity": None, "error": None}
    for name, exe in (("vina", vina), ("obabel", obabel)):
        if shutil.which(str(exe)) is None:
            result["error"] = f"{name} binary not found: {exe}"
            return result
    try:
        with tempfile.TemporaryDirectory(prefix="vina_score_") as tmp:
            tmp = Path(tmp)
            raw_pdbqt, rec_pdbqt, lig_pdbqt = tmp / "raw.pdbqt", tmp / "rec.pdbqt", tmp / "lig.pdbqt"
            subprocess.run([str(obabel), str(protein_pdb), "-O", str(raw_pdbqt), "-xr", "-h"],
                           capture_output=True, timeout=timeout)
            if not raw_pdbqt.exists():
                result["error"] = "obabel receptor conversion failed"
                return result
            rec_pdbqt.write_text("\n".join(
                line for line in raw_pdbqt.read_text().splitlines()
                if line.startswith(("ATOM", "HETATM", "TER", "END"))
            ) + "\n")
            subprocess.run([str(obabel), str(ligand_sdf), "-O", str(lig_pdbqt), "-h"],
                           capture_output=True, timeout=timeout)
            if not lig_pdbqt.exists():
                result["error"] = "obabel ligand conversion failed"
                return result
            proc = subprocess.run(
                [str(vina), "--receptor", str(rec_pdbqt), "--ligand", str(lig_pdbqt), "--score_only"],
                capture_output=True, text=True, timeout=timeout,
            )
            match = _VINA_AFFINITY.search(proc.stdout)
            if match is None:
                result["error"] = (proc.stderr or proc.stdout)[-200:] or "no Affinity line in vina output"
                return result
            result["affinity"] = float(match.group(1))
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    return result


def vina_binary_score_pose(
    complex_path: Union[str, Path],
    ligand_sdf: Union[str, Path],
    vina: Union[str, Path] = "vina",
    obabel: Union[str, Path] = "obabel",
    timeout: float = 300.0,
) -> Dict[str, Optional[object]]:
    """:func:`vina_binary_score` for one ``sample_<i>/`` output of ``run_inference.py``.

    The receptor is the protein part of *complex_path* (the sampled pocket
    side chains plus the untouched remainder of the input receptor).
    """
    with tempfile.TemporaryDirectory(prefix="vina_pose_") as tmp:
        protein_pdb = Path(tmp) / "protein.pdb"
        if not extract_protein_pdb(complex_path, protein_pdb):
            return {"affinity": None, "error": "no protein residues in complex"}
        return vina_binary_score(protein_pdb, ligand_sdf, vina=vina, obabel=obabel, timeout=timeout)
