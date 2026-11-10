"""Select pocket residues around a reference ligand at the training cutoff.

Residues with any heavy atom within ``--cutoff`` A (default 5.0) of any ligand
atom are included and residues that collide with the ligand (< 0.1 A, e.g. a
metal duplicated in both records) are dropped — the exact
``equiflex.prep.parse.get_pocket_residues`` call used to featurise the training
set, so an inference pocket cannot drift from the one the model was trained on.

Residue numbers are ``residue['resnum']`` as ``parse_structure`` assigns them:
mmCIF ``label_seq_id``, or for PDB input the 1-based index among the chain's
standard amino-acid residues in file order (modified residues such as MSE are
skipped, so this is not the author number and not a plain position count).
Take them from this script rather than counting by hand; they are what
``run_inference.py`` expects.

    python scripts/select_pocket.py --receptor 5SAK_ZRY_protein.pdb \\
                                    --ligand 5SAK_ZRY_ligand.sdf

prints a ``residues:`` block (plus ``ions:`` when free metal ions sit within
the cutoff) to paste into a run_inference YAML, or ``--residues-cli`` prints
the ``--residues A:35 A:37 ...`` form for single-system mode.
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import yaml
from rdkit import Chem

from equiflex.prep.parse import extract_ligand_data, get_pocket_residues, parse_structure

# Charges assigned to standalone metal ions found near the ligand.
METAL_CHARGE = {'NA': 1, 'K': 1, 'MG': 2, 'CA': 2, 'MN': 2, 'FE': 2, 'CO': 2,
                'NI': 2, 'CU': 2, 'ZN': 2, 'CD': 2, 'HG': 2, 'MO': 0, 'W': 0}


def pocket_residues(receptor: Path, ligand: Path, cutoff: float):
    """``{chain_id: [resnum, ...]}`` at *cutoff* A around the ligand heavy atoms."""
    protein_data = parse_structure(receptor)
    if protein_data is None:
        raise RuntimeError(f'failed to parse receptor {receptor}')
    ligand_data = extract_ligand_data(ligand)
    if ligand_data is None:
        raise RuntimeError(f'failed to parse ligand {ligand}')
    pocket, _ = get_pocket_residues(protein_data, [ligand_data], pocket_cutoff=cutoff)
    residues = {}
    for chain_id, chain_residues in pocket.items():
        resnums = sorted({int(r['resnum']) for r in chain_residues})
        if resnums:
            residues[str(chain_id)] = resnums
    if not residues:
        raise RuntimeError(f'no residues within {cutoff} A of the ligand')
    return residues


def pocket_ions(receptor: Path, ligand: Path, cutoff: float):
    """SMILES of standalone metal ions within *cutoff* A of the ligand.

    Only single-atom residues count: a metal inside a cofactor such as HEM is
    part of that cofactor, not a free ion the model should sample.
    """
    mol = Chem.MolFromMolFile(str(ligand))
    if mol is None:
        return []
    coords = mol.GetConformer().GetPositions()
    structure = gemmi.read_structure(str(receptor))
    structure.remove_waters()
    ions = []
    for chain in structure[0]:
        for residue in chain:
            atoms = list(residue)
            if len(atoms) != 1:
                continue
            element = atoms[0].element.name.upper()
            if element not in METAL_CHARGE:
                continue
            pos = np.array([[atoms[0].pos.x, atoms[0].pos.y, atoms[0].pos.z]])
            if np.linalg.norm(coords - pos, axis=1).min() >= cutoff:
                continue
            charge = METAL_CHARGE[element]
            smiles = f'[{element.capitalize()}+{charge}]' if charge > 0 else f'[{element.capitalize()}]'
            ion = Chem.MolFromSmiles(smiles)
            if ion is not None:
                ions.append(Chem.MolToSmiles(ion))
    return ions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--receptor', type=Path, required=True, help='.pdb / .ent / .cif / .mmcif')
    ap.add_argument('--ligand', type=Path, required=True, help='reference ligand SDF')
    ap.add_argument('--cutoff', type=float, default=5.0,
                    help='heavy-atom distance cutoff in A (training value: 5.0)')
    ap.add_argument('--no-ions', action='store_true', help='skip free metal ion detection')
    ap.add_argument('--residues-cli', action='store_true',
                    help='print run_inference.py --residues tokens instead of YAML')
    args = ap.parse_args()

    residues = pocket_residues(args.receptor, args.ligand, args.cutoff)
    ions = [] if args.no_ions else pocket_ions(args.receptor, args.ligand, args.cutoff)

    if args.residues_cli:
        print('--residues ' + ' '.join(f'{c}:{n}' for c, nums in residues.items() for n in nums))
        if ions:
            print('--ions ' + ' '.join(f'"{s}"' for s in ions))
        return

    spec = {'protein': {'path': str(args.receptor), 'residues': residues},
            'ligand': {'smiles': Chem.MolToSmiles(Chem.MolFromMolFile(str(args.ligand))),
                       'reference_sdf': str(args.ligand)}}
    if ions:
        spec['ligand']['ions'] = ions
    print(yaml.safe_dump({args.receptor.stem.removesuffix('_protein'): spec},
                         sort_keys=False, default_flow_style=None, width=120), end='')
    print(f'# pocket_provenance: cutoff={args.cutoff} selector=equiflex.prep.parse.get_pocket_residues')


if __name__ == '__main__':
    main()
