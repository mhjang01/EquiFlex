"""Backfill `chiral_atom_indices` into ligand_features.pt written before the key existed.

Featurisations produced before ChiralAtomLoss was wired up load fine but leave
`lig_chiral_indices` empty, so `chiral_coef` is silently a no-op. Everything else
in those dirs (embeddings, pocket features, topology) is still valid, so a full
re-featurisation is wasteful — only this one array is missing, and it is
recomputable from data already on disk with no GPU.

ATOM ORDER IS THE WHOLE PROBLEM. The stored `smiles` is the *canonical* form, and
`generate_ligand_reference_data` warns that a canonical-SMILES round-trip permutes
atom indices — so regenerating from it would produce correctly-shaped, silently
misindexed restraints. This script instead rebuilds each ligand from the stored
per-atom/per-bond records in `ligands.pkl`, which are in the original order by
construction, and assigns stereo from the stored conformer.

Every system is gated on a self-check before anything is written: the reference
conformer in `ligand_features.pt` has correct stereochemistry by construction, so
the signed improper at every stereocentre must already sit on its R/S side. A
wrong sign means the indices are misaligned or the R/S flags are inverted; those
centres are dropped and the rest of the molecule is written. That single gate
covers atom-index alignment, multi-ligand offsets, and sign convention at once.

Dropping beats the alternatives: writing an unverified centre would actively drive
it to the WRONG enantiomer, and refusing the whole system would leave the key
absent, which load_system_data now rejects outright. Known drop case: degree-3 SP3
sulfur (sulfoxide / sulfonium), where get_chiral_atom_indices' _CIPRank-ordered
improper is systematically inverted relative to RDKit's assigned R/S. That is a
pre-existing limitation of the index builder, not of this script.

The gate is on the SIGN of phi, not on the value of ChiralAtomLoss. The loss adds
a 30 deg buffer pushing centres away from the planar state, and real conformers
legitimately sit inside that margin (observed down to ~9 deg), which produces a
small nonzero hinge with perfectly correct stereochemistry. Only a sign flip is
evidence of a bad index; the in-margin count is reported as information.

    python scripts/backfill_chiral_indices.py data/5SAK_ZRY --dry-run
    python scripts/backfill_chiral_indices.py data/posebusters/prepared
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load
from rdkit import Chem, RDLogger

from equiflex.model.loss import ChiralAtomLoss
from equiflex.prep.parse import get_chiral_atom_indices

RDLogger.DisableLog('rdApp.*')

_BOND_TYPES = {
    'SINGLE': Chem.BondType.SINGLE,
    'DOUBLE': Chem.BondType.DOUBLE,
    'TRIPLE': Chem.BondType.TRIPLE,
    'AROMATIC': Chem.BondType.AROMATIC,
    'DATIVE': Chem.BondType.DATIVE,
    'UNSPECIFIED': Chem.BondType.UNSPECIFIED,
}



class BackfillError(Exception):
    """Raised when a system cannot be backfilled safely."""


def rebuild_mol(lig: dict) -> Chem.Mol:
    """Rebuild an RDKit mol from stored atom/bond records, preserving atom order.

    Order-preserving by construction: atom i of the result is `lig['atoms'][i]`.
    Stereo is taken from the stored conformer rather than from any chiral tag, so
    the result reflects the geometry the model was actually given.
    """
    atoms, bonds = lig['atoms'], lig['bonds']
    mol = Chem.RWMol()
    for a in atoms:
        atom = Chem.Atom(int(a['atomic_number']))
        atom.SetFormalCharge(int(a.get('formal_charge', 0)))
        atom.SetIsAromatic(bool(a.get('is_aromatic', False)))
        atom.SetNumExplicitHs(int(a.get('num_explicit_hs', 0)))
        mol.AddAtom(atom)

    for b in bonds:
        bt = _BOND_TYPES.get(str(b['bond_type']).upper())
        if bt is None:
            raise BackfillError(f"unknown bond type {b['bond_type']!r}")
        mol.AddBond(int(b['begin_atom_idx']), int(b['end_atom_idx']), bt)

    mol = mol.GetMol()
    coords = np.asarray(lig['ref_coords'], dtype=np.float64)
    if coords.shape[0] != len(atoms):
        raise BackfillError(
            f"ref_coords has {coords.shape[0]} rows for {len(atoms)} atoms")
    conf = Chem.Conformer(len(atoms))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
    mol.AddConformer(conf, assignId=True)

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise BackfillError(f"sanitisation failed: {exc}")

    # Independent check that AddAtom order survived sanitisation.
    for i, a in enumerate(atoms):
        got = mol.GetAtomWithIdx(i).GetSymbol()
        if got != a['symbol']:
            raise BackfillError(
                f"atom order diverged at index {i}: rebuilt {got}, stored {a['symbol']}")

    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def compute_chiral_indices(ligands: list, n_total: int) -> torch.Tensor:
    """Per-ligand indices with the block offsets process_ligands applies.

    Mirrors prep/process.py: columns 0:4 are atom indices and shift by the running
    atom offset; column 4 is the R/S flag and must not shift.
    """
    blocks, offset = [], 0
    for lig in ligands:
        n_atoms = len(lig['atoms'])
        idx = get_chiral_atom_indices(rebuild_mol(lig))
        if len(idx) > 0:
            idx = np.asarray(idx).copy()
            idx[:, :4] += offset
            blocks.append(idx)
        offset += n_atoms

    if offset != n_total:
        raise BackfillError(
            f"ligand atom counts sum to {offset} but features hold {n_total}")
    if not blocks:
        return torch.empty((0, 5), dtype=torch.long)

    out = torch.as_tensor(np.concatenate(blocks), dtype=torch.long)
    if out[:, :4].max() >= n_total or out[:, :4].min() < 0:
        raise BackfillError("chiral atom index out of range after offsetting")
    if not torch.isin(out[:, 4], torch.tensor([0, 1])).all():
        raise BackfillError("R/S flag column is not binary")
    return out


def verify(chiral: torch.Tensor, coords: torch.Tensor) -> tuple:
    """Check the signed improper at every centre of the reference conformer.

    Returns (correct_mask, n_inside_buffer, min_abs_phi_deg). A False entry means
    the improper sits on the wrong side of planar for its assigned R/S label —
    see module docstring for why that, and not the loss value, is the gate.
    """
    if chiral.numel() == 0:
        return torch.zeros(0, dtype=torch.bool), 0, float('nan')
    loss = ChiralAtomLoss(weight=1.0)
    phi = loss._signed_dihedral(
        coords.unsqueeze(0).float(),
        torch.zeros(chiral.shape[0], dtype=torch.long),
        chiral[:, :4].long(),
    )
    is_r = chiral[:, 4].bool()
    correct = (is_r & (phi > 0)) | (~is_r & (phi < 0))
    n_inside = int((correct & (phi.abs() < loss.buffer)).sum())
    return correct, n_inside, float(np.degrees(float(phi.abs().min())))


def describe_centres(ligands: list, chiral: torch.Tensor, rows: torch.Tensor) -> str:
    """Element + degree of the centres at the given rows, for the drop report."""
    mols, offset, out = [], 0, []
    for lig in ligands:
        mols.append((offset, rebuild_mol(lig)))
        offset += len(lig['atoms'])
    for j in rows.tolist():
        centre = int(chiral[j, 3])
        for off, mol in reversed(mols):
            if centre >= off:
                a = mol.GetAtomWithIdx(centre - off)
                out.append(f'{a.GetSymbol()}(deg{a.GetDegree()})')
                break
    return ','.join(sorted(set(out)))


def backfill(sys_dir: Path, force: bool, dry_run: bool) -> tuple:
    """Returns (status, detail)."""
    feats_path = sys_dir / 'ligand_features.pt'
    ligs_path = sys_dir / 'ligands.pkl'
    if not feats_path.exists():
        return 'skip', 'no ligand_features.pt'
    feats = torch.load(feats_path, map_location='cpu', weights_only=False)
    if 'chiral_atom_indices' in feats and not force:
        return 'skip', 'already present'
    if not ligs_path.exists():
        return 'fail', 'no ligands.pkl — needs full re-featurisation'

    ligands = joblib_load(ligs_path)
    if not isinstance(ligands, list):
        ligands = [ligands]

    n_total = feats['atom_features'].shape[0]
    chiral = compute_chiral_indices(ligands, n_total)
    correct, n_inside, min_phi = verify(chiral, feats['coords'])

    # Drop, don't abort. An unverifiable centre gets excluded so the rest of the
    # molecule is still restrained; writing it anyway would actively drive that
    # centre to the WRONG enantiomer, and refusing the whole system would leave
    # the key absent, which the loader now rejects. Both are worse than dropping.
    status, dropped = 'written', ''
    if chiral.numel() and not correct.all():
        rows = (~correct).nonzero().flatten()
        dropped = (f' | DROPPED {len(rows)} unverifiable centre(s) '
                   f'[{describe_centres(ligands, chiral, rows)}]')
        chiral = chiral[correct]
        correct, n_inside, min_phi = verify(chiral, feats['coords'])
        assert correct.all(), 'drop failed to clear the sign check'
        status = 'written-partial'

    detail = f'{chiral.shape[0]} quads'
    if chiral.shape[0]:
        detail += f', min|phi| {min_phi:.1f} deg'
        if n_inside:
            detail += f', {n_inside} inside 30deg margin'
    detail += dropped
    if dry_run:
        return ('would-write' if status == 'written' else 'would-write-partial'), detail

    # Atomic write: never leave a half-written features file behind.
    feats['chiral_atom_indices'] = chiral
    tmp = feats_path.with_suffix('.pt.tmp')
    torch.save(feats, tmp)
    os.replace(tmp, feats_path)
    return status, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+', type=Path,
                    help='dirs searched recursively for ligand_features.pt '
                         '(inference */system_data/ or flat training <system>/ layout)')
    ap.add_argument('--dry-run', action='store_true', help='verify only, write nothing')
    ap.add_argument('--force', action='store_true', help='recompute even if present')
    args = ap.parse_args()

    dirs = sorted({p.parent for root in args.roots
                   for p in root.rglob('ligand_features.pt')})
    if not dirs:
        print('No ligand_features.pt found.')
        return 1
    print(f'Found {len(dirs)} system dir(s)\n')

    counts, failures = {}, []
    for i, sys_dir in enumerate(dirs, 1):
        name = sys_dir.parent.name if sys_dir.name == 'system_data' else sys_dir.name
        try:
            status, detail = backfill(sys_dir, args.force, args.dry_run)
        except (BackfillError, Exception) as exc:
            status, detail = 'fail', str(exc)
        counts[status] = counts.get(status, 0) + 1
        if status == 'fail':
            failures.append((name, detail))
        if i <= 5 or status in ('fail', 'written-partial', 'would-write-partial') or i % 2000 == 0:
            print(f'  [{i}/{len(dirs)}] {name}: {status} — {detail}')

    print('\n' + '=' * 60)
    for k in ('written', 'written-partial', 'would-write', 'would-write-partial', 'skip', 'fail'):
        if k in counts:
            print(f'  {k:12s} {counts[k]}')
    if failures:
        print(f'\n{len(failures)} failure(s); first 10:')
        for name, detail in failures[:10]:
            print(f'  {name}: {detail}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
