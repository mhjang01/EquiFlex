#!/usr/bin/env python3
"""Build the TPD redock benchmark (molecular glues + PROTACs) from the boltz-tpd ternary set.

Source (``--tpd-root``): ``glue_test/manifest.tsv`` and one reference mmCIF per record
under ``glue_test/reference`` — chains A and B are the two proteins of the ternary and
chain L the deposited degrader copy — plus the Boltz CCD store (``boltz_cache/mols``)
for ligand bond orders and ``data/structures/<pdb>.cif.gz`` for the metal ions the
reference omits.

Output mirrors ``data/posebusters`` so the PoseBusters scripts apply unchanged:

    <out>/raw/<record>/<record>_protein.pdb   chains A + B, plus single-atom metal ions
                                              within --cutoff of the ligand (HETATM)
    <out>/raw/<record>/<record>_ligand.sdf    deposited pose with CCD bond orders
    <out>/benchmark.yaml                      run_inference systems manifest, pocket at
                                              the training cutoff (scripts/select_pocket.py)
    <out>/records.tsv                         per-record metadata (class, release, ...)
    <out>/skipped.tsv                         records left out and why

Ligand atoms absent from the deposited model are rebuilt when they are at most
--max-missing-frac of the heavy atoms (``n_modelled_atoms`` in records.tsv): a terminal
atom is placed opposite its anchor's observed bonds at covalent-radius distance, anything
larger by constrained embedding. Records missing more are skipped.

    python scripts/prepare_tpd.py --tpd-root ~/workspace/boltz-tpd --out tpd

``--db-manifest`` instead takes a ``scripts/build_plp_dataset.py`` TPD manifest
(e.g. ``tpd/full_db/tpd/nonredundant.tsv``) and copies each record's exported receptor
(every contacting chain, structural ions) and complete ligand pose; records whose
ligand was not fully observed have no SDF and are skipped.

    python scripts/prepare_tpd.py --db-manifest tpd/full_db/tpd/nonredundant.tsv --out tpd/bench_db
    python scripts/make_posebusters_inference_config.py --manifest tpd/benchmark.yaml \\
        --raw-root $PWD/tpd/raw --output-root tpd/prepared \\
        --num-samples 50 --out tpd/inference.yaml
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

from select_pocket import METAL_CHARGE, pocket_ions, pocket_residues

RDLogger.DisableLog('rdApp.*')
# The Boltz store pickles the CCD atom names as RDKit atom properties.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

PROTEIN_CHAINS = ('A', 'B')
LIGAND_CHAIN = 'L'
ION_RESNUM = 9001
# Class fallback when the entry's annotations name neither a glue nor a PROTAC.
PROTAC_MIN_ATOMS = 45


def classify(terms: str, n_atoms: int) -> tuple[str, str]:
    """``(glue | protac, terms | size)`` from the PDB annotation terms, else ligand size."""
    terms = set(terms.split(','))
    if 'protac' in terms:
        return 'protac', 'terms'
    if terms & {'molecular_glue', 'imid'}:
        return 'glue', 'terms'
    return ('protac' if n_atoms >= PROTAC_MIN_ATOMS else 'glue'), 'size'


def place_terminal(mol: Chem.Mol, coords: dict, index: int):
    """Position for a missing atom bonded to one observed atom: opposite that atom's observed bonds."""
    atom = mol.GetAtomWithIdx(index)
    anchors = [n for n in atom.GetNeighbors() if n.GetIdx() in coords]
    if atom.GetDegree() != 1 or len(anchors) != 1:
        return None
    anchor = anchors[0]
    centre = coords[anchor.GetIdx()]
    bonds = [centre - coords[o.GetIdx()] for o in anchor.GetNeighbors() if o.GetIdx() in coords]
    if not bonds:
        return None
    direction = np.sum([b / np.linalg.norm(b) for b in bonds], axis=0)
    if np.linalg.norm(direction) < 1e-3:
        return None
    table = Chem.GetPeriodicTable()
    length = table.GetRcovalent(atom.GetAtomicNum()) + table.GetRcovalent(anchor.GetAtomicNum())
    return centre + direction / np.linalg.norm(direction) * length


def build_ligand(ccd_mol: Chem.Mol, residue: gemmi.Residue, max_missing_frac: float):
    """CCD topology on the deposited coordinates; ``(mol, n_modelled)`` or None."""
    mol = Chem.Mol(ccd_mol)
    Chem.SanitizeMol(mol)
    deposited = {atom.name: np.array(atom.pos.tolist()) for atom in residue}
    coords = {}
    for atom in mol.GetAtoms():
        xyz = deposited.get(atom.GetProp('name'))
        if xyz is not None:
            coords[atom.GetIdx()] = xyz
    missing = [i for i in range(mol.GetNumAtoms()) if i not in coords]
    if len(coords) < 3 or len(missing) > max_missing_frac * mol.GetNumAtoms():
        return None

    modelled = {}
    for i in missing:
        xyz = place_terminal(mol, coords, i)
        if xyz is not None:
            modelled[i] = xyz
    remaining = [i for i in missing if i not in modelled]
    if remaining:
        # Embed with the deposited atoms pinned, superpose on them and keep only the
        # missing atoms' positions, so deposited coordinates stay exactly as they are.
        # CCD chiral tags may disagree with the deposited geometry, so chirality is
        # not enforced; stereo is re-read from the final coordinates below.
        embedded = Chem.AddHs(Chem.Mol(mol))
        coord_map = {i: Point3D(*xyz) for i, xyz in coords.items()}
        if AllChem.EmbedMolecule(embedded, coordMap=coord_map, randomSeed=42,
                                 useRandomCoords=True, enforceChirality=False) < 0:
            return None
        embedded = Chem.RemoveHs(embedded)
        pinned = Chem.Mol(mol)
        pinned.RemoveAllConformers()
        pinned_conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in coords.items():
            pinned_conf.SetAtomPosition(i, Point3D(*xyz))
        pinned.AddConformer(pinned_conf, assignId=True)
        rdMolAlign.AlignMol(embedded, pinned, atomMap=[(i, i) for i in coords])
        for i in remaining:
            modelled[i] = np.array(embedded.GetConformer().GetAtomPosition(i))

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in {**coords, **modelled}.items():
        conf.SetAtomPosition(i, Point3D(*map(float, xyz)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol, len(missing)


def write_protein_pdb(reference: gemmi.Structure, deposited: gemmi.Structure,
                      lig_xyz: np.ndarray, cutoff: float, path: Path) -> list[str]:
    """Chains A and B of the reference plus deposited metal ions near the ligand."""
    out = gemmi.Structure()
    out.cell = reference.cell
    model = gemmi.Model('1')
    chains = {}
    for name in PROTEIN_CHAINS:
        chain = gemmi.Chain(name)
        for residue in reference[0][name]:
            chain.add_residue(residue)
        chains[name] = chain

    ions = []
    for source_chain in deposited[0]:
        for residue in source_chain:
            atoms = list(residue)
            if len(atoms) != 1 or atoms[0].element.name.upper() not in METAL_CHARGE:
                continue
            pos = np.array(atoms[0].pos.tolist())
            if np.linalg.norm(lig_xyz - pos, axis=1).min() >= cutoff:
                continue
            ion = residue.clone()
            ion.seqid = gemmi.SeqId(ION_RESNUM + len(ions), ' ')
            ion.het_flag = 'H'
            chains[PROTEIN_CHAINS[0]].add_residue(ion)
            ions.append(residue.name)

    for chain in chains.values():
        model.add_chain(chain)
    out.add_model(model)
    out.setup_entities()
    out.write_pdb(str(path))
    return ions


def build_record(row: dict, tpd_root: Path, raw_root: Path, cutoff: float,
                 max_missing_frac: float) -> dict:
    record, ccd = row['record'], row['ligand']
    reference = gemmi.read_structure(str(tpd_root / 'glue_test' / 'reference' / f'{record}.cif'))
    reference.remove_hydrogens()
    ligand_res = next(res for chain in reference[0] if chain.name == LIGAND_CHAIN for res in chain)
    lig_xyz = np.array([atom.pos.tolist() for atom in ligand_res])

    with (tpd_root / 'boltz_cache' / 'mols' / f'{ccd}.pkl').open('rb') as f:
        ccd_mol = pickle.load(f)
    built = build_ligand(ccd_mol, ligand_res, max_missing_frac)
    if built is None:
        n_obs = len({a.name for a in ligand_res} & {a.GetProp('name') for a in ccd_mol.GetAtoms()})
        raise ValueError(f'ligand has {ccd_mol.GetNumAtoms() - n_obs}/{ccd_mol.GetNumAtoms()} '
                         f'heavy atoms unmodelled')
    mol, n_modelled = built

    deposited = gemmi.read_structure(str(tpd_root / 'data' / 'structures' / f'{row["pdb_id"].lower()}.cif.gz'))
    deposited.remove_hydrogens()
    deposited.remove_alternative_conformations()

    system_dir = raw_root / record
    system_dir.mkdir(parents=True, exist_ok=True)
    protein_path = system_dir / f'{record}_protein.pdb'
    ligand_path = system_dir / f'{record}_ligand.sdf'
    ions_present = write_protein_pdb(reference, deposited, lig_xyz, cutoff, protein_path)
    mol.SetProp('_Name', record)
    Chem.MolToMolFile(mol, str(ligand_path))

    residues = pocket_residues(protein_path, ligand_path, cutoff)
    ions = pocket_ions(protein_path, ligand_path, cutoff)
    smiles = Chem.MolToSmiles(Chem.MolFromMolFile(str(ligand_path)))
    cls, cls_source = classify(row['strong_terms'], mol.GetNumAtoms())

    spec = {'protein': {'path': protein_path.name, 'residues': residues},
            'ligand': {'smiles': smiles, 'reference_sdf': ligand_path.name}}
    if ions:
        spec['ligand']['ions'] = ions
    meta = {'record': record, 'pdb_id': row['pdb_id'], 'ccd': ccd, 'class': cls,
            'class_source': cls_source, 'release': row['release'],
            'resolution': row['resolution'], 'strong_terms': row['strong_terms'],
            'len_A': row['len_A'], 'len_B': row['len_B'],
            'n_lig_atoms': mol.GetNumAtoms(), 'n_modelled_atoms': n_modelled,
            'pocket_A': len(residues.get('A', [])), 'pocket_B': len(residues.get('B', [])),
            'ions': ','.join(ions), 'ions_in_pdb': ','.join(ions_present)}
    return {'spec': spec, 'meta': meta}


def build_from_db(row: dict, raw_root: Path, cutoff: float) -> dict:
    """Benchmark entry from a build_plp_dataset.py manifest row: its exported receptor and ligand pose."""
    record = row['record']
    if not row['ligand_sdf'] or not row['protein_pdb']:
        raise ValueError(f"no complete ligand pose ({row['sdf_status']})")
    system_dir = raw_root / record
    system_dir.mkdir(parents=True, exist_ok=True)
    protein_path = system_dir / f'{record}_protein.pdb'
    ligand_path = system_dir / f'{record}_ligand.sdf'
    shutil.copyfile(row['protein_pdb'], protein_path)
    shutil.copyfile(row['ligand_sdf'], ligand_path)

    residues = pocket_residues(protein_path, ligand_path, cutoff)
    ions = pocket_ions(protein_path, ligand_path, cutoff)
    mol = Chem.MolFromMolFile(str(ligand_path))
    spec = {'protein': {'path': protein_path.name, 'residues': residues},
            'ligand': {'smiles': Chem.MolToSmiles(mol), 'reference_sdf': ligand_path.name}}
    if ions:
        spec['ligand']['ions'] = ions
    meta = {'record': record, 'pdb_id': row['pdb_id'], 'ccd': row['ccd'], 'class': row['class'],
            'class_source': row['class_source'], 'release': row['release'], 'resolution': row['resolution'],
            'space': row['space'], 'covalent': row['covalent'] == 'True', 'n_lig_atoms': mol.GetNumAtoms(),
            'protein_chains': row['protein_chains'],
            'pocket_A': len(residues.get('A', [])), 'pocket_B': len(residues.get('B', [])),
            'pocket_other': sum(len(v) for k, v in residues.items() if k not in PROTEIN_CHAINS),
            'ions': ','.join(ions)}
    return {'spec': spec, 'meta': meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tpd-root', type=Path, default=Path.home() / 'workspace' / 'boltz-tpd')
    parser.add_argument('--out', type=Path, default=Path('tpd'))
    parser.add_argument('--cutoff', type=float, default=5.0,
                        help='pocket / ion heavy-atom cutoff in A (training value: 5.0)')
    parser.add_argument('--max-missing-frac', type=float, default=0.1,
                        help='skip records with more unmodelled ligand heavy atoms than this fraction')
    parser.add_argument('--db-manifest', type=Path,
                        help='build_plp_dataset.py TPD manifest to use instead of --tpd-root glue_test')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    manifest_path = args.db_manifest or args.tpd_root / 'glue_test' / 'manifest.tsv'
    csv.field_size_limit(sys.maxsize)
    with manifest_path.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if args.limit is not None:
        rows = rows[: args.limit]

    raw_root = args.out / 'raw'
    systems, metas, skipped = {}, [], []
    for index, row in enumerate(rows, 1):
        record = row['record']
        try:
            if args.db_manifest:
                built = build_from_db(row, raw_root, args.cutoff)
            else:
                built = build_record(row, args.tpd_root, raw_root, args.cutoff, args.max_missing_frac)
        except Exception as exc:
            skipped.append((record, f'{type(exc).__name__}: {exc}'))
            print(f'[{index}/{len(rows)}] {record}  SKIP  {exc}')
            continue
        systems[record] = built['spec']
        metas.append(built['meta'])
        meta = built['meta']
        print(f'[{index}/{len(rows)}] {record}  {meta["class"]:6s} lig={meta["n_lig_atoms"]:3d}'
              f'{"(+" + str(meta["n_modelled_atoms"]) + " modelled)" if meta.get("n_modelled_atoms") else ""}'
              f'  pocket A={meta["pocket_A"]} B={meta["pocket_B"]}'
              f'{"  ions=" + meta["ions"] if meta["ions"] else ""}')

    benchmark = {
        'name': 'TPD redock benchmark (molecular glues + PROTACs)',
        'version': 1,
        'source': {
            'manifest': str(manifest_path),
        } if args.db_manifest else {
            'manifest': str(manifest_path),
            'reference': 'boltz-tpd glue_test/reference/{ID}.cif (chains A, B + ligand L)',
            'ligand_topology': 'Boltz CCD store (boltz_cache/mols)',
            'ions': f'single-atom metal residues of the deposited entry within {args.cutoff} A of the ligand',
            'max_missing_frac': args.max_missing_frac,
        },
        'pocket_provenance': {
            'cutoff': args.cutoff,
            'selector': 'equiflex.prep.parse.get_pocket_residues',
            'reference': 'deposited {ID}_ligand.sdf',
            'generated_by': 'scripts/prepare_tpd.py (scripts/select_pocket.py)',
            'ions': 'standalone single-atom metal residues within cutoff (scripts/select_pocket.py)',
        },
        'systems': systems,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'benchmark.yaml').open('w') as f:
        yaml.safe_dump(benchmark, f, sort_keys=False, width=120)
    with (args.out / 'records.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(metas[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(metas)
    (args.out / 'skipped.tsv').write_text(''.join(f'{r}\t{why}\n' for r, why in skipped))

    classes = ', '.join(f'{c} {n}' for c, n in sorted(Counter(m['class'] for m in metas).items()))
    print(f'\n{len(systems)} systems -> {args.out / "benchmark.yaml"}  '
          f'({classes}; {sum(1 for m in metas if m.get("n_modelled_atoms"))} with modelled ligand atoms; '
          f'{len(skipped)} skipped)')


if __name__ == '__main__':
    main()
