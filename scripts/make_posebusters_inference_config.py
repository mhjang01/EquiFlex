"""Emit a run_inference.py YAML for the PoseBusters benchmark.

``data/posebusters/benchmark.yaml`` is a portable manifest: per system the
receptor file name, the pocket residues at the training cutoff
(``equiflex.prep.parse.get_pocket_residues``, 5.0 A), the ligand SMILES, free
metal ions near the pocket, and the reference ligand SDF. This script resolves
the file names against ``--raw-root`` (``<raw-root>/<id>/<id>_protein.pdb`` and
``<id>_ligand.sdf``) and stamps the benchmark sampling recipe on top.

    python scripts/make_posebusters_inference_config.py \\
        --raw-root /path/to/posebusters_benchmark_set \\
        --out data/posebusters/inference.yaml
    python scripts/run_inference.py --config data/posebusters/inference.yaml

``run_inference.py`` reuses ``<output-root>/<id>/system_data`` when present, so
pass ``--force-prepare`` after changing the pocket definition: a stale
featurisation built at another cutoff would otherwise be reused silently.
"""

import argparse
from pathlib import Path

import yaml

# Benchmark recipe (pb_test/inference_phys*.yaml lineage).
GUIDANCE = {
    'enabled': True,
    't_start': 0.0,
    'scale': 100.0,
    'schedule': 'linear_decay',
    'bond_coef': 1.0,
    'angle_coef': 1.0,
    'vdw_coef': 5.0,
    'vdw_active_coef': 1.0,
    'vdw_max_coef': 0.0,
    'torsion_coef': 0.25,
    'chiral_coef': 1.0,
    'chiral_buffer': 0.5236,
    'grad_clip': 10.0,
}
BRANCH = {'enabled': False, 'at_t': 0.5, 'factor': 2, 'max_rot_deg': 180.0,
          'trans_sigma': 1.0}
SDE = {'enabled': False, 'sigma': 0.1, 'eps_w': 0.01, 'w_cutoff': 0.85, 'mode': 'both'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path('data/posebusters/benchmark.yaml'))
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='root containing <id>/<id>_protein.pdb and <id>/<id>_ligand.sdf')
    parser.add_argument('--out', type=Path, default=Path('data/posebusters/inference.yaml'))
    parser.add_argument('--output-root', type=Path, default=Path('data/posebusters/prepared'),
                        help='where run_inference.py writes per-system outputs')
    parser.add_argument('--checkpoint', default='weights/weights.ckpt')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--schedule', choices=['uniform', 'log'], default='uniform')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-guidance', action='store_true',
                        help='sample without physics guidance (ablation)')
    parser.add_argument('--match-training', action='store_true',
                        help='guidance coefficients follow the checkpoint loss weights '
                             '(bond/angle/vdW/torsion/chiral) instead of the benchmark recipe')
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--sde-sigma', type=float, default=None,
                        help='enable SDE noise injection at this sigma (sampler.SDEConfig)')
    parser.add_argument('--sde-mode', choices=['both', 'receptor', 'ligand'], default='both')
    parser.add_argument('--branch', action='store_true',
                        help='mid-flight branching: factor copies per trajectory')
    parser.add_argument('--branch-at', type=float, default=0.5)
    parser.add_argument('--branch-factor', type=int, default=2)
    parser.add_argument('--branch-max-rot-deg', type=float, default=180.0)
    parser.add_argument('--branch-trans-sigma', type=float, default=1.0)
    parser.add_argument('--sde-score-drift', action='store_true',
                        help='marginal-preserving OU step instead of bare noise')
    parser.add_argument('--score-summary', action='store_true',
                        help='write per-sample summary.md (physics + torch Vina) per system')
    parser.add_argument('--force-prepare', action='store_true',
                        help='re-featurise every system even if system_data/ exists')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    systems = {}
    for system_id, spec in manifest['systems'].items():
        if args.limit is not None and len(systems) >= args.limit:
            break
        spec = {'protein': dict(spec['protein']), 'ligand': dict(spec['ligand'])}
        spec['protein']['path'] = str(
            args.raw_root / system_id / Path(spec['protein'].get('path', f'{system_id}_protein.pdb')).name
        )
        reference = spec['ligand'].get('reference_sdf', f'{system_id}_ligand.sdf')
        spec['ligand']['reference_sdf'] = str(args.raw_root / system_id / Path(reference).name)
        spec['output_dir'] = str(args.output_root / system_id)
        systems[system_id] = spec

    guidance = {**GUIDANCE, 'enabled': not args.no_guidance}
    if args.match_training:
        for key in [k for k in guidance if k.endswith('_coef')]:
            del guidance[key]
        guidance['match_training_weights'] = True
    if args.guidance_scale is not None:
        guidance['scale'] = args.guidance_scale
    if args.grad_clip is not None:
        guidance['grad_clip'] = args.grad_clip

    sde = dict(SDE, mode=args.sde_mode)
    if args.sde_sigma is not None:
        sde.update(enabled=True, sigma=args.sde_sigma)
    sde['score_drift'] = args.sde_score_drift
    branch = dict(BRANCH, enabled=args.branch, at_t=args.branch_at, factor=args.branch_factor,
                  max_rot_deg=args.branch_max_rot_deg, trans_sigma=args.branch_trans_sigma)

    config = {
        'checkpoint': args.checkpoint,
        'output_root': str(args.output_root),
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'schedule': args.schedule,
        'seed': args.seed,
        'device': args.device,
        'return_trajectory': False,
        'score_summary': args.score_summary,
        'force_prepare': args.force_prepare,
        'guidance': guidance,
        'branch': branch,
        'sde': sde,
    }
    if 'pocket_provenance' in manifest:
        config['pocket_provenance'] = manifest['pocket_provenance']
    config['systems'] = systems

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        yaml.safe_dump(config, f, sort_keys=False, width=120)
    print(f'Wrote {len(systems)} systems to {args.out} '
          f'(guidance {"off" if args.no_guidance else "on"}, force_prepare={args.force_prepare})')


if __name__ == '__main__':
    main()
