"""Rescore PoseBusters poses in their OWN predicted receptor, not the crystal one.

The earlier GenScore/RTMScore/EquiScore selector runs cut every pocket from the raw
benchmark receptor (`<id>_protein.pdb`). For PoseBusters redock that file is also the
model's inference input, so it leaks nothing -- but it means the scorer never saw the
sidechains EquiFlex predicted, and so could not penalise a ligand pose that clashes
with our own receptor output or reward one that fits it. Since the selected sample is
reported with sidechain columns, the selector should see the complex it would ship.

This cuts one pocket per SAMPLE from that sample's `complex.cif` (predicted receptor,
ligand stripped) instead of one pocket per system from the crystal receptor, and is
otherwise identical to dekois/genscore/score_posebusters.py: 10 A around the pose,
standard residues only, same model and dist_threhold.

Two caveats the output records rather than hides:
  * complex.cif holds only the modelled pocket, so a few residues inside 10 A of the
    pose are absent that the crystal cut would have included (`n_res`).
  * complex.cif contains no cofactors or metals at all, where the crystal cut at
    least kept those that are standard-residue-representable (none, in practice).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

from Bio.PDB import MMCIFParser, PDBIO, Select  # noqa: E402
from scipy.spatial import distance_matrix  # noqa: E402

PB = ROOT / 'data/posebusters/v7'
STANDARD = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX',
}


class _Pocket(Select):
    """Standard residues with any atom within `cut` of the ligand pose."""

    def __init__(self, xyz, cut):
        self.xyz, self.cut = xyz, cut

    def accept_residue(self, residue):
        if residue.get_resname().strip() not in STANDARD:
            return 0
        pos = np.array([a.coord for a in residue.get_atoms()])
        if pos.ndim < 2:
            return 0
        return int(distance_matrix(pos, self.xyz).min() < self.cut)

    def accept_atom(self, atom):
        return int(atom.element != 'H')


def load_pocket(path, load_mol):
    """Load a pocket PDB, tolerating valence errors from predicted geometry.

    Predicted sidechains occasionally place two atoms close enough that RDKit's
    proximity bonding gives an oxygen three bonds, and upstream `load_mol`
    returns None. `prot_to_graph` only reads coordinates, residue names and
    connectivity, so skipping the valence/property check changes nothing it uses;
    the strict path is tried first so scores match it wherever it succeeds.
    """
    try:
        # load_mol raises rather than returning None: it calls
        # AssignStereochemistryFrom3D on the mol MolFromPDBFile failed to build.
        mol = load_mol(str(path), explicit_H=False, use_chirality=True)
        if mol is not None:
            return mol, False
    except Exception:
        pass
    mol = Chem.MolFromPDBFile(str(path), removeHs=True, sanitize=False)
    if mol is None:
        return None, False
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    return mol, True


def write_pocket(cif, xyz, cut, dst):
    """Cut the predicted pocket from complex.cif. Returns residues kept."""
    st = MMCIFParser(QUIET=True).get_structure('p', str(cif))
    io = PDBIO()
    io.set_structure(st)
    sel = _Pocket(xyz, cut)
    io.save(str(dst), sel)
    return sum(1 for r in st.get_residues() if sel.accept_residue(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='v10_41k_off_5A')
    ap.add_argument('--scorer', default='genscore', choices=['genscore', 'rtmscore'])
    ap.add_argument('--model', default=None, help='default: GT_0.0_1 / rtmscore_model1')
    ap.add_argument('--cutoff', type=float, default=10.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split('/'))
    tmp = ROOT / f'.pb_predpocket_{args.run}_{args.scorer}_{i}'
    tmp.mkdir(parents=True, exist_ok=True)

    if args.scorer == 'genscore':
        model_name = args.model or 'GT_0.0_1'
        sys.path.insert(0, str(ROOT / 'dekois/genscore'))
        from score_genscore import GEN, build_model  # noqa: E402  (patches forward)
        from GenScore.data.data import VSDataset  # noqa: E402
        from GenScore.feats.mol2graph_rdmda_res import load_mol  # noqa: E402
        from GenScore.model.utils import calculate_probablity  # noqa: E402
        from torch_geometric.loader import DataLoader  # noqa: E402
        from torch_scatter import scatter_add  # noqa: E402
        device = 'cuda' if th.cuda.is_available() else 'cpu'
        model = build_model(str(GEN / 'trained_models' / f'{model_name}.pth'), 'gt', device)
        col = 'genscore'

        def score(pocket_mol, mols, ids):
            ds = VSDataset(ids=ids, ligs=mols, prot=pocket_mol, cutoff=args.cutoff,
                           parallel=False)
            loader = DataLoader(dataset=ds, batch_size=64, shuffle=False, num_workers=0)
            got, out = [], []
            with th.no_grad():
                for bids, bgp, bgl, _ in loader:
                    bgl, bgp = bgl.to(device), bgp.to(device)
                    pi, sigma, mu, dist, _, _, batch = model(bgl, bgp)
                    prob = calculate_probablity(pi, sigma, mu, dist)
                    prob[th.where(dist > 5.0)[0]] = 0.
                    out.append(scatter_add(prob, batch, dim=0,
                                           dim_size=batch.unique().size(0)).cpu().numpy())
                    got.extend(bids)
            return got, np.concatenate(out)
    else:
        model_name = args.model or 'rtmscore_model1'
        sys.path.insert(0, str(ROOT / 'rtmscore'))
        from score_posebusters import (DIST_THR, RTM, build_model,  # noqa: E402
                                       calculate_probablity, collate)
        from RTMScore.data.data import VSDataset  # noqa: E402
        from RTMScore.feats.mol2graph_rdmda_res import load_mol  # noqa: E402
        from torch.utils.data import DataLoader  # noqa: E402
        device = 'cpu'
        model = build_model(str(RTM / 'trained_models' / f'{model_name}.pth'), device)
        col = 'rtmscore'

        def score(pocket_mol, mols, ids):
            ds = VSDataset(ids=ids, ligs=mols, prot=pocket_mol, cutoff=args.cutoff,
                           parallel=False)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=collate)
            got, out = [], []
            with th.no_grad():
                for bids, bgl, bgp in loader:
                    bgl, bgp = bgl.to(device), bgp.to(device)
                    pi, sigma, mu, dist, _, _, batch = model(bgp, bgl)
                    prob = calculate_probablity(pi, sigma, mu, dist)
                    prob[th.where(dist > DIST_THR)[0]] = 0.
                    acc = th.zeros(bgl.batch_size, dtype=prob.dtype, device=prob.device)
                    out.append(acc.index_add_(0, batch.view(-1).to(prob.device),
                                              prob.view(-1)).cpu().numpy())
                    got.extend(bids)
            return got, np.concatenate(out)

    root = PB / args.run
    systems = sorted(p.name for p in root.iterdir()
                     if p.is_dir() and (p / 'sample_0/ligand.sdf').exists())
    if args.limit:
        systems = systems[:args.limit]
    systems = systems[i::n]
    print(f'{args.run} shard {i}/{n}: {len(systems)} systems, {args.scorer} '
          f'{model_name} on {device}', flush=True)

    rows, failed, t0 = [], [], time.perf_counter()
    for k, sysid in enumerate(systems, 1):
        for sd in sorted((root / sysid).glob('sample_*'),
                         key=lambda p: int(p.name.split('_')[1])):
            cif, sdf = sd / 'complex.cif', sd / 'ligand.sdf'
            if not (cif.exists() and sdf.exists()):
                failed.append((sysid, sd.name, 'missing file'))
                continue
            m = Chem.MolFromMolFile(str(sdf), sanitize=True)
            if m is None:
                failed.append((sysid, sd.name, 'ligand unreadable'))
                continue
            pock = tmp / f'{sysid}_{sd.name}.pdb'
            n_res = write_pocket(cif, m.GetConformer().GetPositions(), args.cutoff, pock)
            try:
                pocket_mol, relaxed = load_pocket(pock, load_mol)
                if pocket_mol is None:
                    raise ValueError('pocket unreadable')
                got, sc = score(pocket_mol, [m], [sd.name])
            except Exception as exc:
                failed.append((sysid, sd.name, f'{exc!r}'[:80]))
                pock.unlink(missing_ok=True)
                continue
            pock.unlink(missing_ok=True)
            for s, v in zip(got, sc):
                rows.append({'system': sysid, 'sample': s, col: float(v),
                             'n_res': n_res, 'relaxed_load': relaxed})
        if k % 25 == 0:
            el = time.perf_counter() - t0
            print(f'{k}/{len(systems)}  {el:.0f}s  '
                  f'eta {el/k*(len(systems)-k)/60:.1f} min  failed {len(failed)}',
                  flush=True)

    df = pd.DataFrame(rows)
    out = args.out or (ROOT / 'data/posebusters/v7' / args.run /
                       f'predrec_{args.scorer}_{model_name}'
                       f'{"" if n == 1 else f"_shard{i}of{n}"}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    tmp.rmdir()
    print(f'\n{len(df)} poses, {df.system.nunique()} systems, '
          f'{time.perf_counter()-t0:.0f}s -> {out}')
    print(df[col].describe().round(3).to_string())
    if failed:
        print(f'{len(failed)} failures, first 10: {failed[:10]}')


if __name__ == '__main__':
    main()
