"""Emit a run_inference.py YAML for the CrossDock cross-docking benchmark.

A case is any directory under ``--data-root`` holding both ``protein.pdb`` and
``ligand.sdf``; the system id is its path relative to the root with ``/``
replaced by ``__`` (``Ensemble/CDK2/1aq1`` -> ``Ensemble__CDK2__1aq1``).
Pocket residues and free metal ions come from ``scripts/select_pocket.py``, so
they use the training definition (``get_pocket_residues``, 5 A heavy atom) and
the ``resnum`` numbering ``run_inference.py`` expects.

    python scripts/make_crossdock_inference_config.py \\
        --checkpoint checkpoints/v10_512/step41000.ckpt \\
        --output-root /mnt/drive/crossdock_v10_41k \\
        --num-samples 10 --branch --branch-factor 4 \\
        --out crossdock/inference_v10_41k_br10x4.yaml
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_pocket import pocket_ions, pocket_residues  # noqa: E402

RDLogger.DisableLog('rdApp.*')

# v10 41k benchmark recipe (data/posebusters/v7/v10_41k_m300_br10x5_5A).
GUIDANCE = {'enabled': True, 't_start': 0.0, 'scale': 300.0, 'schedule': 'linear_decay',
            'chiral_buffer': 0.5236, 'grad_clip': 10.0, 'match_training_weights': True}
BRANCH = {'enabled': False, 'at_t': 0.5, 'factor': 4, 'max_rot_deg': 180.0, 'trans_sigma': 1.0}
SDE = {'enabled': False, 'sigma': 0.15, 'eps_w': 0.01, 'w_cutoff': 0.85, 'mode': 'both',
       'score_drift': False}


def normalize_hypervalent_phosphorus(mol: Chem.Mol) -> Chem.Mol:
    """Charge-separate neutral P with too many terminal P=O bonds (some CrossDock SDFs).

    The chemically equivalent form uses P-O(-) for enough terminal oxygens to keep
    neutral phosphorus at valence five. Restricted to neutral P=O with terminal
    neutral oxygen so unrelated sanitization failures stay visible.
    """
    editable = Chem.RWMol(mol)
    for phosphorus in editable.GetAtoms():
        if phosphorus.GetAtomicNum() != 15 or phosphorus.GetFormalCharge() != 0:
            continue
        order_sum = lambda: sum(b.GetBondTypeAsDouble() for b in phosphorus.GetBonds())
        repaired = False
        candidates = sorted(
            (b.GetOtherAtom(phosphorus).GetIdx() for b in phosphorus.GetBonds()
             if b.GetBondType() == Chem.BondType.DOUBLE
             and b.GetOtherAtom(phosphorus).GetAtomicNum() == 8
             and b.GetOtherAtom(phosphorus).GetDegree() == 1
             and b.GetOtherAtom(phosphorus).GetFormalCharge() == 0))
        for oxygen_index in candidates:
            if order_sum() <= 5:
                break
            editable.GetBondBetweenAtoms(phosphorus.GetIdx(), oxygen_index).SetBondType(
                Chem.BondType.SINGLE)
            editable.GetAtomWithIdx(oxygen_index).SetFormalCharge(-1)
            repaired = True
        if repaired:
            # Exchanged terminal oxygens are resonance-equivalent; an SDF-derived
            # P stereotag would become an arbitrary P@/P@@ in the SMILES.
            phosphorus.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    normalized = editable.GetMol()
    normalized.UpdatePropertyCache(strict=False)
    return normalized


def ligand_smiles(path: Path) -> str:
    mol = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=False)
        if mol is None:
            raise ValueError(f'could not read a molecule from {path}')
        mol = normalize_hypervalent_phosphorus(mol)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    if mol.GetNumConformers() == 0:
        raise ValueError(f'no 3D conformer in {path}')
    # RemoveAllHs, not RemoveHs: RDKit keeps a stereo-defining H on amidine/guanidine
    # N=C bonds, which Uni-Mol then strips, misaligning its embedding with the SMILES.
    return Chem.MolToSmiles(Chem.RemoveAllHs(mol), canonical=True)


def build_system(system_id: str, case_dir: Path, cutoff: float, no_ions: bool):
    receptor, ligand = case_dir / 'protein.pdb', case_dir / 'ligand.sdf'
    spec = {'protein': {'path': str(receptor), 'residues': pocket_residues(receptor, ligand, cutoff)},
            'ligand': {'smiles': ligand_smiles(ligand), 'reference_sdf': str(ligand)}}
    ions = [] if no_ions else pocket_ions(receptor, ligand, cutoff)
    if ions:
        spec['ligand']['ions'] = ions
    return system_id, spec


def _worker(payload):
    system_id, case_dir, cutoff, no_ions = payload
    try:
        return build_system(system_id, Path(case_dir), cutoff, no_ions)
    except Exception as exc:                      # kept per system: one bad SDF must not kill 14k
        return system_id, {'__error__': f'{type(exc).__name__}: {exc}'}


def discover(data_root: Path, subsets):
    cases = []
    for subset in subsets or ['']:
        root = data_root / subset if subset else data_root
        if not root.is_dir():
            raise FileNotFoundError(f'CrossDock subset not found: {root}')
        for protein in sorted(root.rglob('protein.pdb')):
            if (protein.parent / 'ligand.sdf').is_file():
                rel = protein.parent.relative_to(data_root).as_posix()
                cases.append((rel.replace('/', '__'), protein.parent))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, default=Path('/mnt/drive/CrossDock'))
    ap.add_argument('--subset', action='append', help='subset under data-root; repeatable')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--pocket-cutoff', type=float, default=5.0)
    ap.add_argument('--no-ions', action='store_true')
    ap.add_argument('--num-samples', type=int, default=10)
    ap.add_argument('--num-steps', type=int, default=20)
    ap.add_argument('--schedule', choices=['uniform', 'log'], default='uniform')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--guidance-scale', type=float, default=None)
    ap.add_argument('--no-guidance', action='store_true')
    ap.add_argument('--branch', action='store_true')
    ap.add_argument('--branch-at', type=float, default=0.5)
    ap.add_argument('--branch-factor', type=int, default=4)
    ap.add_argument('--sde-sigma', type=float, default=None)
    ap.add_argument('--force-prepare', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    cases = discover(args.data_root.resolve(), args.subset)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit('no directories with protein.pdb + ligand.sdf found')

    systems, skipped = {}, []
    payloads = [(sid, str(d), args.pocket_cutoff, args.no_ions) for sid, d in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            system_id, spec = future.result()
            if '__error__' in spec:
                skipped.append({'system': system_id, 'error': spec['__error__']})
            else:
                spec['output_dir'] = str(args.output_root / system_id)
                systems[system_id] = spec
            if done % 500 == 0:
                print(f'  {done}/{len(payloads)} cases prepared', flush=True)

    guidance = dict(GUIDANCE, enabled=not args.no_guidance)
    if args.guidance_scale is not None:
        guidance['scale'] = args.guidance_scale
    sde = dict(SDE)
    if args.sde_sigma is not None:
        sde.update(enabled=True, sigma=args.sde_sigma)

    config = {
        'checkpoint': args.checkpoint,
        'output_root': str(args.output_root),
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'schedule': args.schedule,
        'seed': args.seed,
        'device': args.device,
        'return_trajectory': False,
        'score_summary': False,
        'force_prepare': args.force_prepare,
        'guidance': guidance,
        'branch': dict(BRANCH, enabled=args.branch, at_t=args.branch_at,
                       factor=args.branch_factor),
        'sde': sde,
        'pocket_provenance': {'cutoff': args.pocket_cutoff,
                              'selector': 'equiflex.prep.parse.get_pocket_residues',
                              'reference': 'CrossDock ligand.sdf',
                              'generated_by': 'scripts/make_crossdock_inference_config.py'},
        'systems': dict(sorted(systems.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, width=120)
    skip_path = args.out.with_suffix('.skipped.json')
    skip_path.write_text(json.dumps(skipped, indent=2) + '\n')
    poses = args.num_samples * (args.branch_factor if args.branch else 1)
    print(f'Wrote {len(systems)} systems to {args.out} '
          f'({poses} poses/system, skipped {len(skipped)} -> {skip_path})')


if __name__ == '__main__':
    main()
