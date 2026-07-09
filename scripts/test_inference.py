"""
Batch inference over all systems in ./test_data.

For each system, loads pre-featurized data, runs Euler integration
via FlowMatchingSampler, and saves results to ./trajectory_output/test/{system_id}/.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict

from equiflex.model import FlexTrainer
from equiflex.model.config import FullConfig, DataConfig, OptimizerConfig, SchedulerConfig, EMAConfig, LoggingConfig, TrainingConfig
from equiflex.model.loss import LossConfig
from equiflex.model.models import FlexConfig
from equiflex.inference.sampler import FlowMatchingSampler, get_time_schedule

torch.serialization.add_safe_globals([
    FullConfig, DataConfig, OptimizerConfig, SchedulerConfig,
    EMAConfig, LoggingConfig, TrainingConfig, LossConfig, FlexConfig,
])


def load_system_data(system_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load pre-featurized system data and build the unbatched data dict
    expected by FlowMatchingSampler.sample().
    """
    ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
    pocket_feats = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    esm_emb      = torch.load(system_dir / 'esm3_embeddings.pt', weights_only=False)
    unimol_emb   = torch.load(system_dir / 'unimol_embeddings.pt', weights_only=False)

    rec_coords = pocket_feats['coords']            # [N_rec, 3]
    lig_coords = ligand_feats['coords']            # [N_lig, 3]

    # Backbone mask
    rec_backbone_mask = pocket_feats.get('backbone_mask')
    if rec_backbone_mask is None:
        rec_backbone_mask = torch.zeros(rec_coords.shape[0], dtype=torch.float32)
    elif isinstance(rec_backbone_mask, np.ndarray):
        rec_backbone_mask = torch.from_numpy(rec_backbone_mask).float()

    # Center all coordinates by backbone COM
    if rec_backbone_mask.sum() > 0:
        backbone_com = (rec_coords * rec_backbone_mask.unsqueeze(-1)).sum(0) / rec_backbone_mask.sum()
    else:
        backbone_com = rec_coords.mean(0)

    rec_coords_centered = rec_coords - backbone_com
    lig_coords_centered = lig_coords - backbone_com

    # Constraint matrices (zeros if absent)
    def load_constraint(feats_dict, n_atoms):
        cm = feats_dict.get('constraint_matrix', None)
        if cm is None:
            return torch.zeros(n_atoms, n_atoms, 1, dtype=torch.float32)
        if isinstance(cm, np.ndarray):
            cm = torch.from_numpy(cm).float()
        if cm.ndim == 2:
            cm = cm.unsqueeze(-1)
        return cm

    N_rec = rec_coords.shape[0]
    N_lig = lig_coords.shape[0]

    # Torsion indices — not used by sampler but included for completeness
    lig_torsion = ligand_feats.get('sp2_torsion_indices', ligand_feats.get('torsion_indices', None))
    if lig_torsion is not None and isinstance(lig_torsion, np.ndarray):
        lig_torsion = torch.from_numpy(lig_torsion).long()

    return {
        # Receptor [N_rec, ...]
        'rec_atom_feats':        pocket_feats['atom_features'],
        'rec_atom_coords':       rec_coords_centered,
        'rec_atom_ref_coords':   pocket_feats['ref_coords'],
        'rec_bond_feats':        pocket_feats['bond_features'],
        'rec_uid_vec':           pocket_feats['residue_uid_vec'],
        'rec_backbone_mask':     rec_backbone_mask,
        'rec_constraint_matrix': load_constraint(pocket_feats, N_rec),
        'esm_embeddings':        esm_emb,

        # Ligand [N_lig, ...]
        'lig_atom_feats':        ligand_feats['atom_features'],
        'lig_atom_coords':       lig_coords_centered,
        'lig_atom_ref_coords':   ligand_feats['ref_coords'],
        'lig_bond_feats':        ligand_feats['bond_features'],
        'lig_uid_vec':           ligand_feats['ligand_uid_vec'],
        'lig_constraint_matrix': load_constraint(ligand_feats, N_lig),
        'unimol_embeddings':     unimol_emb,

        # Metadata
        '_backbone_com':   backbone_com,
        '_n_rec':          N_rec,
        '_n_lig':          N_lig,
        '_n_rec_tokens':   esm_emb.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description='Batch inference over test_data systems')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to Lightning .ckpt file')
    parser.add_argument('--data-dir',    type=str, default='./test_data')
    parser.add_argument('--output-dir',  type=str, default='./trajectory_output/test')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-steps',   type=int, default=100)
    parser.add_argument('--schedule',    type=str, default='log', choices=['uniform', 'log'])
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--device',      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume',      action='store_true',
                        help='Skip systems that already have saved results')
    parser.add_argument('--return-trajectory', action='store_true', default=True,
                        help='Store full integration trajectory (default: True)')
    parser.add_argument('--no-trajectory', dest='return_trajectory', action='store_false',
                        help='Skip storing the trajectory to save memory')

    relax_group = parser.add_argument_group('relaxation', 'MMFF energy minimisation options')
    relax_group.add_argument('--relax', action='store_true',
                             help='Run MMFF energy minimisation after sampling')
    relax_group.add_argument('--relax-max-iterations', type=int, default=200,
                             help='MMFF minimisation steps (default: 200)')
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / 'ligand_features.pt').exists()
    ])
    print(f"Found {len(system_dirs)} systems in {data_dir}")

    print(f"Loading checkpoint: {args.checkpoint}")
    trainer_module = FlexTrainer.load_from_checkpoint(args.checkpoint, map_location='cpu')
    sampler = FlowMatchingSampler(trainer_module.model, device=args.device)
    print(f"Model loaded. Device: {args.device}")
    if args.relax:
        print(f"Relaxation enabled (MMFF, max_iters={args.relax_max_iterations})")
    print("=" * 60)

    if args.relax:
        from equiflex.inference.relaxation import relax_complex, write_complex_cif

    n_done, n_skip, n_fail = 0, 0, 0

    for system_dir in system_dirs:
        system_id = system_dir.name
        sys_out   = output_dir / system_id
        out_path  = sys_out / 'results.pt'

        if args.resume and out_path.exists():
            n_skip += 1
            continue

        try:
            data    = load_system_data(system_dir)
            results = sampler.sample(
                data=data,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                schedule=args.schedule,
                seed=args.seed,
                return_trajectory=args.return_trajectory,
            )

            sys_out.mkdir(parents=True, exist_ok=True)

            if args.relax:
                pocket_feats = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
                ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
                backbone_com = results['backbone_com']
                B = results['rec_coords_final'].shape[0]
                n_relaxed = 0

                for b in range(B):
                    sample_dir = sys_out / f'sample_{b}'
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    write_complex_cif(
                        pocket_feats, ligand_feats,
                        results['rec_coords_final'][b],
                        results['lig_coords_final'][b],
                        backbone_com,
                        sample_dir / 'predicted.cif',
                    )

                    try:
                        relaxed = relax_complex(
                            system_dir=system_dir,
                            rec_coords=results['rec_coords_final'][b],
                            lig_coords=results['lig_coords_final'][b],
                            backbone_com=backbone_com,
                            output_dir=sample_dir,
                            max_iterations=args.relax_max_iterations,
                            write_structures=True,
                        )
                        results['lig_coords_final'][b] = relaxed['lig_coords']
                        n_relaxed += 1
                    except Exception as e:
                        print(f"  [RELAX FAIL] {system_id} sample {b}: {e}")

                print(f"  Relaxed {n_relaxed}/{B} samples")

            torch.save(results, out_path)
            n_done += 1

            N_rec = data['_n_rec']
            N_lig = data['_n_lig']
            print(f"[{n_done:>4}] {system_id}  rec={N_rec}  lig={N_lig}  -> {sys_out}")

        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {system_id}: {e}")

    print("=" * 60)
    print(f"Done: {n_done}  Skipped: {n_skip}  Failed: {n_fail}")


if __name__ == '__main__':
    main()
