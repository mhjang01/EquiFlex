"""GenScore pose scoring against the modelled receptor.

GenScore is a mixture-density network over protein-ligand pair distances: it
sums a per-pair likelihood over every ligand-atom / pocket-residue pair inside
``dist_threshold``, so a pose that places atoms at distances the model considers
typical scores high.

**Receptor frame.** Every pose is scored against the receptor that was sampled
with it, so the pocket carries the modelled sidechains. This is the whole point
of this module. Earlier GenScore runs in this repo cut one pocket from the
crystal receptor and reused it for all samples of a system, while Vinardo saw
each pose's own predicted receptor -- so the two scorers were not comparing the
same structures and GenScore was handed information the sampler never produced.

The cost is real: the pocket graph has to be rebuilt per pose instead of once
per system, and ``prot_to_graph`` is roughly as expensive as scoring a whole
ligand chunk. Scores also stop being strictly commensurable across ligands,
because each is measured against a slightly different pocket; for ranking
ligands in a screen, aggregate per ligand (mean over poses) rather than taking
a single best pose.

GenScore itself is not vendored. It is imported from ``GENSCORE_ROOT``
(default ``~/references/dock/GenScore``), which must hold the upstream package
and ``trained_models/<name>.pth``.
"""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

GENSCORE_ROOT = Path(os.environ.get(
    'GENSCORE_ROOT', Path.home() / 'references' / 'dock' / 'GenScore'))

# GenScore's protein graph is one node per amino-acid residue with
# RES_MAX_NATOMS=24 coordinate slots, so a larger residue (a cofactor such as
# NAP, 48 atoms) makes prot_to_graph raise "negative dimensions are not
# allowed". Cofactors are not representable by calc_res_features either, so the
# pocket is restricted to standard residues; a site lined by a cofactor is
# therefore scored against an incomplete pocket, which score() reports.
STANDARD_RESIDUES = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX',
})


def _patched_forward(self, data_ligand, data_target):
    """``GenScore.forward`` with the pair-batch index built on the data's device.

    Upstream builds it with ``th.tensor(range(B))`` on the CPU and then indexes
    it with a CUDA boolean mask, which older torch tolerated and current torch
    rejects. Everything else is identical to upstream.
    """
    import torch.nn.functional as F
    from torch_geometric.utils import to_dense_batch

    h_l = self.ligand_model(data_ligand)
    h_t = self.target_model(data_target)
    h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
    h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)
    h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
    h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)
    (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
    self.B, self.N_l, self.N_t = B, N_l, N_t

    h_l_x = h_l_x.unsqueeze(-2).repeat(1, 1, N_t, 1)
    h_t_x = h_t_x.unsqueeze(-3).repeat(1, N_l, 1, 1)
    C = torch.cat((h_l_x, h_t_x), -1)
    C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
    C = C[C_mask]
    C = self.MLP(C)

    C_batch = torch.arange(B, device=C_mask.device).unsqueeze(-1).unsqueeze(-1)
    C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]

    pi = F.softmax(self.z_pi(C), -1)
    sigma = F.elu(self.z_sigma(C)) + 1.1
    mu = F.elu(self.z_mu(C)) + 1
    atom_types = self.atom_types(h_l.x)
    bond_types = self.bond_types(
        torch.cat([h_l.x[h_l.edge_index[0]], h_l.x[h_l.edge_index[1]]], axis=1)
    )
    dist = self.compute_euclidean_distances_matrix(
        h_l_pos, h_t_pos.view(B, -1, 3))[C_mask]
    return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch


@functools.lru_cache(maxsize=1)
def _genscore_api():
    """Import GenScore from GENSCORE_ROOT and install the device fix.

    Only the graph builders and the model are imported. Upstream's VSDataset is
    deliberately bypassed: it exists to build one pocket graph and reuse it for
    every ligand, which is the crystal-frame behaviour this module replaces, and
    it drags in prody and openbabel through extract_pocket_prody.
    """
    import sys
    root = str(GENSCORE_ROOT)
    if not GENSCORE_ROOT.is_dir():
        raise FileNotFoundError(
            f'GenScore not found at {root}. Set GENSCORE_ROOT to the upstream '
            'checkout (it must contain GenScore/ and trained_models/).')
    if root not in sys.path:
        sys.path.insert(0, root)

    from GenScore.feats.mol2graph_rdmda_res import load_mol, mol_to_graph, prot_to_graph
    from GenScore.model.model import GatedGCN, GenScore, GraphTransformer
    from GenScore.model.utils import calculate_probablity

    GenScore.forward = _patched_forward
    return dict(load_mol=load_mol, mol_to_graph=mol_to_graph,
                prot_to_graph=prot_to_graph, GenScore=GenScore,
                GraphTransformer=GraphTransformer, GatedGCN=GatedGCN,
                calculate_probablity=calculate_probablity)


def _atom_lines(pdb_text: str) -> List[str]:
    return [l for l in pdb_text.splitlines(keepends=True)
            if l.startswith(('ATOM', 'HETATM'))]


def cut_pocket(
    receptor_pdb: Path | str,
    ligand_coords: np.ndarray,
    out_pdb: Path | str,
    cutoff: float = 10.0,
) -> Dict[str, int]:
    """Write the standard-residue pocket of *receptor_pdb* around *ligand_coords*.

    A residue is kept when any of its atoms lies within *cutoff* of any ligand
    atom. Returns the non-standard residue names that were dropped, with counts,
    so an incomplete pocket can be reported instead of hidden.

    TER records are written at every chain break. Without them RDKit's PDB
    reader bonds the last atom of one chain to the first of the next, which
    produces an over-valent carbon and makes ``MolFromPDBFile`` return None for
    any pocket spanning two chains.
    """
    lines = _atom_lines(Path(receptor_pdb).read_text())
    lig = np.asarray(ligand_coords, dtype=np.float64).reshape(-1, 3)

    # Group atoms by residue: chainID + resSeq + iCode, as in the fixed columns.
    residues: Dict[str, List[str]] = {}
    for line in lines:
        residues.setdefault(line[21:27], []).append(line)

    kept: List[str] = []
    dropped: Dict[str, int] = {}
    for key, res_lines in residues.items():
        resname = res_lines[0][17:20].strip()
        if resname not in STANDARD_RESIDUES:
            dropped[resname] = dropped.get(resname, 0) + len(res_lines)
            continue
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        for l in res_lines])
        # min over the full pairwise block; pockets are small enough that the
        # [n_res_atoms, n_lig_atoms] matrix is cheaper than a tree.
        if np.sqrt(((xyz[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min() <= cutoff:
            kept.extend(res_lines)

    # Close every chain with a TER so no bond is inferred across the break.
    body: List[str] = []
    for i, line in enumerate(kept):
        if i and line[21] != kept[i - 1][21]:
            prev = kept[i - 1]
            body.append(f'TER   {prev[6:11]}      {prev[17:20]} {prev[21]}{prev[22:27]}\n')
        body.append(line)
    if kept:
        last = kept[-1]
        body.append(f'TER   {last[6:11]}      {last[17:20]} {last[21]}{last[22:27]}\n')

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(''.join(body) + 'END\n')
    return dropped


class GenScoreScorer:
    """GenScore over poses, each against its own predicted receptor.

    ``model_name`` is a file in ``GENSCORE_ROOT/trained_models`` without the
    ``.pth``; the encoder is inferred from the name (``GT_*`` -> graph
    transformer, ``GatedGCN_*`` -> gated GCN).
    """

    def __init__(
        self,
        model_name: str = 'GT_0.0_1',
        device: str | torch.device = 'cuda',
        cutoff: float = 10.0,
        dist_threshold: float = 5.0,
        pocket_cache: int = 8,
    ):
        api = _genscore_api()
        self.api = api
        self.device = torch.device(device)
        self.cutoff = cutoff
        self.dist_threshold = dist_threshold
        self.model_name = model_name

        encoder = 'gcn' if model_name.lower().startswith('gatedgcn') else 'gt'
        common = dict(num_hidden_channels=128, dropout_rate=0.15, num_layers=6)
        if encoder == 'gt':
            def mk(nf, ef):
                return api['GraphTransformer'](
                    in_channels=nf, edge_features=ef, activ_fn=torch.nn.SiLU(),
                    transformer_residual=True, num_attention_heads=4,
                    norm_to_apply='batch', **common)
        else:
            def mk(nf, ef):
                return api['GatedGCN'](in_channels=nf, edge_features=ef,
                                       residual=True, equivstable_pe=False, **common)

        weights = GENSCORE_ROOT / 'trained_models' / f'{model_name}.pth'
        if not weights.exists():
            raise FileNotFoundError(f'GenScore weights not found: {weights}')
        model = api['GenScore'](mk(41, 10), mk(41, 5), in_channels=128,
                                hidden_dim=128, n_gaussians=10, dropout_rate=0.15,
                                dist_threhold=dist_threshold).to(self.device)
        model.load_state_dict(torch.load(weights, map_location=self.device)['model_state_dict'])
        model.eval()
        self.model = model
        self._pocket_graphs: Dict[Tuple[str, int], object] = {}
        self._pocket_cache = pocket_cache
        self.dropped_residues: Dict[str, int] = {}
        # Why a pose scored None, keyed by its receptor path. A silent None here
        # would look like a low score to any caller that filters on NaN.
        self.failures: Dict[str, str] = {}
        self.n_relaxed_load = 0

    def _load_pocket_mol(self, path: Path):
        """Load a pocket PDB, tolerating valence errors from predicted geometry.

        Predicted sidechains occasionally place two atoms close enough that
        RDKit's proximity bonding gives an oxygen three bonds, and ``load_mol``
        returns None (then raises inside AssignStereochemistryFrom3D). This is
        not an edge case: it fires on ~20% of PoseBusters poses. ``prot_to_graph``
        only reads coordinates, residue names and connectivity, so skipping the
        valence/property check changes nothing it uses. The strict path is tried
        first so scores match it wherever it succeeds.

        This mirrors ``load_pocket`` in scripts/score_pb_predicted_receptor.py;
        the two must stay in step or their numbers stop being comparable.
        """
        from rdkit import Chem

        try:
            mol = self.api['load_mol'](str(path), explicit_H=False, use_chirality=True)
            if mol is not None:
                return mol
        except Exception:
            pass
        mol = Chem.MolFromPDBFile(str(path), removeHs=True, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        self.n_relaxed_load += 1
        return mol

    def _pocket_graph(self, receptor_pdb: Path, ligand_coords: np.ndarray):
        """Pocket graph for this receptor. Cached by (path, mtime_ns).

        In the honest frame each pose has its own receptor, so the cache only
        helps when the same file is scored more than once -- a crystal-pocket
        control, or several ligands against one fixed structure.
        """
        receptor_pdb = Path(receptor_pdb)
        key = (str(receptor_pdb), receptor_pdb.stat().st_mtime_ns)
        hit = self._pocket_graphs.get(key)
        if hit is not None:
            return hit
        with tempfile.TemporaryDirectory() as td:
            pocket_pdb = Path(td) / 'pocket.pdb'
            dropped = cut_pocket(receptor_pdb, ligand_coords, pocket_pdb, self.cutoff)
            for name, n in dropped.items():
                self.dropped_residues[name] = self.dropped_residues.get(name, 0) + n
            mol = self._load_pocket_mol(pocket_pdb)
            if mol is None:
                raise ValueError(f'pocket of {receptor_pdb.name} could not be read by GenScore')
            graph = self.api['prot_to_graph'](mol, self.cutoff)
        if len(self._pocket_graphs) >= self._pocket_cache:
            self._pocket_graphs.clear()
        self._pocket_graphs[key] = graph
        return graph

    @torch.no_grad()
    def score_poses(
        self,
        poses: Sequence[Tuple[Path | str, object]],
        batch_size: int = 32,
    ) -> List[Optional[float]]:
        """Score (receptor_pdb, ligand_mol) pairs. ``None`` where a pose failed.

        Each ligand is scored against the pocket cut from the receptor given
        alongside it, so a per-pose predicted receptor stays paired with its pose.
        """
        from torch_geometric.data import Batch
        from torch_scatter import scatter_add

        out: List[Optional[float]] = [None] * len(poses)
        pending: List[Tuple[int, object, object]] = []

        for i, (receptor_pdb, lig_mol) in enumerate(poses):
            try:
                lig_graph = self.api['mol_to_graph'](lig_mol, explicit_H=False,
                                                    use_chirality=True)
                coords = lig_mol.GetConformer().GetPositions()
                pocket_graph = self._pocket_graph(Path(receptor_pdb), coords)
                if lig_graph is None:
                    self.failures[str(receptor_pdb)] = 'ligand graph could not be built'
                    continue
            except Exception as exc:
                self.failures[str(receptor_pdb)] = f'{type(exc).__name__}: {exc}'
                continue
            pending.append((i, lig_graph, pocket_graph))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            bgl = Batch.from_data_list([g for _, g, _ in chunk]).to(self.device)
            bgp = Batch.from_data_list([p for _, _, p in chunk]).to(self.device)
            pi, sigma, mu, dist, _, _, batch = self.model(bgl, bgp)
            prob = self.api['calculate_probablity'](pi, sigma, mu, dist)
            prob[torch.where(dist > self.dist_threshold)[0]] = 0.0
            totals = scatter_add(prob, batch, dim=0,
                                 dim_size=len(chunk)).cpu().numpy()
            for (idx, _, _), value in zip(chunk, totals):
                out[idx] = float(value)
        return out

    def score_pose(self, receptor_pdb: Path | str, lig_mol) -> Optional[float]:
        """Score one pose against its own receptor."""
        return self.score_poses([(receptor_pdb, lig_mol)])[0]
