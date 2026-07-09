'''
Patch parsed atom dicts in an existing dataset with the new discriminating
per-atom context fields (neighbor composition, Gasteiger charge, pharmacophore,
Morgan environment, CIP stereo), without re-parsing raw structures.

- Ligand atoms: fields recomputed from the canonical SMILES (round-trip is
  faithful; atom names match by symbol+canonical-rank, verified 1:1).
- Pocket atoms: fields copied from the regenerated amino-acid references
  (.refs/*.npz) by atom name.

After patching the pkls, regenerate ligand_features.pt / pocket_features.pt via
the existing graph-featurization step. ESM3 / UniMol embeddings are untouched.
'''

import sys
from pathlib import Path

sys.path.insert(0, '/home/mhjang/workspace/flex')

from joblib import load, dump, Parallel, delayed
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from equiflex.prep.parse import compute_mol_context, get_neighbor_element_counts, load_aa_reference

_NEW_FIELDS = ('neighbor_elements', 'gasteiger_charge', 'is_donor', 'is_acceptor',
               'is_hydrophobe', 'morgan_bits', 'cip_code', 'is_stereocenter')


def _ligand_fields_from_atom(atom, idx, ctx):
    return {
        'neighbor_elements': get_neighbor_element_counts(atom),
        'gasteiger_charge': ctx['gasteiger'][idx],
        'is_donor': idx in ctx['donors'],
        'is_acceptor': idx in ctx['acceptors'],
        'is_hydrophobe': idx in ctx['hydrophobes'],
        'morgan_bits': sorted(ctx['atom_bits'][idx]),
        'cip_code': ctx['cip_codes'].get(idx, ''),
        'is_stereocenter': idx in ctx['stereocenters'],
    }


def patch_ligands(ligands):
    '''Inject new fields into every ligand atom dict. Returns patched list.'''
    for lig in ligands:
        mol = Chem.MolFromSmiles(lig['smiles'])
        if mol is None:
            raise ValueError(f"unparseable SMILES: {lig['smiles']}")
        ctx = compute_mol_context(mol)
        ranks = AllChem.CanonicalRankAtoms(mol)
        name_to_idx = {f'{a.GetSymbol().upper()}{ranks[a.GetIdx()] + 1}': a.GetIdx()
                       for a in mol.GetAtoms()}

        for atom in lig['atoms']:
            idx = name_to_idx[atom['name']]
            atom.update(_ligand_fields_from_atom(mol.GetAtomWithIdx(idx), idx, ctx))
    return ligands


def patch_pocket(pocket_residues, ref_cache):
    '''Inject new fields into every pocket residue atom dict from references.'''
    for chain_id, residues in pocket_residues.items():
        for res in residues.values():
            aa = res['aa_code']
            if aa not in ref_cache:
                ref = load_aa_reference(aa)
                ref_cache[aa] = {a['name']: a for a in ref['atoms']}
            ref_atoms = ref_cache[aa]
            for atom in res['atoms']:
                ref_atom = ref_atoms[atom['name']]
                atom.update({k: ref_atom[k] for k in _NEW_FIELDS})
    return pocket_residues


def patch_system(system_dir: Path):
    lig_path = system_dir / 'ligands.pkl'
    pkt_path = system_dir / 'pocket_residues.pkl'
    if not lig_path.exists() or not pkt_path.exists():
        return system_dir.name, False, 'missing pkl'
    try:
        ligands = patch_ligands(load(lig_path))
        dump(ligands, lig_path, compress=3)

        pocket = patch_pocket(load(pkt_path), ref_cache={})
        dump(pocket, pkt_path, compress=3)
        return system_dir.name, True, 'ok'
    except Exception as e:
        return system_dir.name, False, f'error: {e}'


def main(data_path: Path, splits=('train', 'val', 'test'), n_jobs=-1):
    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        system_dirs = sorted(d for d in split_dir.iterdir()
                             if d.is_dir() and (d / 'ligands.pkl').exists())
        print(f"[{split}] patching {len(system_dirs)} systems")
        results = Parallel(n_jobs=n_jobs)(
            delayed(patch_system)(d) for d in tqdm(system_dirs, desc=f'patch {split}')
        )
        ok = sum(1 for _, s, _ in results if s)
        fails = [(n, m) for n, s, m in results if not s]
        print(f"[{split}] patched {ok}/{len(system_dirs)}, failed {len(fails)}")
        if fails:
            log = split_dir / 'patch_errors.txt'
            with open(log, 'w') as f:
                for n, m in fails:
                    f.write(f"{n}: {m}\n")
            for n, m in fails[:5]:
                print(f"  {n}: {m}")
            print(f"  (full list -> {log})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data_v2_plinder')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    main(Path(args.data_path), splits=args.splits, n_jobs=args.n_jobs)
