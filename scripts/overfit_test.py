"""
Overfitting test script for Flex model.

Tests model capacity by fitting on a single structure.
If the model can't overfit one sample, it won't generalize.

Expected results after ~1000-5000 steps:
- Loss should drop to near zero
- Inference RMSD should be < 0.5 Å
"""

import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path

from equiflex.model import FlexTrainer, get_trainer, load_config
from equiflex.model.trainer import RMSDEvaluationCallback
from equiflex.model.loader import FlexDataset, SingleSystemSampler, collate_single_system


class SingleSampleDataset(torch.utils.data.Dataset):
    """Dataset that returns the same sample repeatedly."""

    def __init__(self, system_dir: str, length: int = 10000):
        self.system_dir = Path(system_dir)
        self.length = length
        self.sample = self._load_sample()

    def _load_sample(self):
        import numpy as np

        ligand_feats = torch.load(self.system_dir / 'ligand_features.pt', weights_only=False)
        pocket_feats = torch.load(self.system_dir / 'pocket_features.pt', weights_only=False)
        esm_emb = torch.load(self.system_dir / 'esm3_embeddings.pt', weights_only=False)
        unimol_emb = torch.load(self.system_dir / 'unimol_embeddings.pt', weights_only=False)

        # Backbone mask
        rec_backbone_mask = pocket_feats['backbone_mask']

        # CB mask
        rec_cb_mask = pocket_feats.get('cb_mask', None)
        if rec_cb_mask is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_cb_mask = torch.zeros(n_atoms, dtype=torch.float32)

        # Per-residue backbone atom indices for CB placement
        backbone_atom_indices = pocket_feats.get('backbone_atom_indices', None)

        # Missing mask
        rec_missing_mask = pocket_feats.get('missing_mask', None)
        if rec_missing_mask is not None:
            if isinstance(rec_missing_mask, np.ndarray):
                rec_missing_mask = torch.from_numpy(rec_missing_mask).bool()
        else:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_missing_mask = torch.zeros(n_atoms, dtype=torch.bool)

        # Constraint matrices
        rec_constraint_matrix = pocket_feats.get('constraint_matrix', None)
        if rec_constraint_matrix is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_constraint_matrix = torch.zeros(n_atoms, n_atoms, 1, dtype=torch.float32)
        else:
            if isinstance(rec_constraint_matrix, np.ndarray):
                rec_constraint_matrix = torch.from_numpy(rec_constraint_matrix).float()
            if rec_constraint_matrix.ndim == 2:
                rec_constraint_matrix = rec_constraint_matrix.unsqueeze(-1)

        lig_constraint_matrix = ligand_feats.get('constraint_matrix', None)
        if lig_constraint_matrix is None:
            n_atoms = ligand_feats['atom_features'].shape[0]
            lig_constraint_matrix = torch.zeros(n_atoms, n_atoms, 1, dtype=torch.float32)
        else:
            if isinstance(lig_constraint_matrix, np.ndarray):
                lig_constraint_matrix = torch.from_numpy(lig_constraint_matrix).float()
            if lig_constraint_matrix.ndim == 2:
                lig_constraint_matrix = lig_constraint_matrix.unsqueeze(-1)

        # Coordinates
        rec_coords_gt = pocket_feats['coords']
        lig_coords_gt = ligand_feats['coords']

        # Center by backbone COM
        valid_backbone_mask = rec_backbone_mask * (~rec_missing_mask).float()
        if valid_backbone_mask.sum() > 0:
            backbone_com = (rec_coords_gt * valid_backbone_mask.unsqueeze(-1)).sum(dim=0) / valid_backbone_mask.sum()
        else:
            backbone_com = rec_coords_gt.mean(dim=0)

        rec_coords_gt = rec_coords_gt - backbone_com
        lig_coords_gt = lig_coords_gt - backbone_com
        rec_ref_coords = pocket_feats['ref_coords'] - backbone_com
        lig_ref_coords = ligand_feats['ref_coords'] - backbone_com

        return {
            'rec_atom_feats': pocket_feats['atom_features'],
            'rec_atom_coords_gt': rec_coords_gt,
            'rec_atom_ref_coords': rec_ref_coords,
            'rec_bond_feats': pocket_feats['bond_features'],
            'rec_uid_vec': pocket_feats['residue_uid_vec'],
            'rec_backbone_mask': rec_backbone_mask,
            'rec_cb_mask': rec_cb_mask,
            'rec_backbone_atom_indices': backbone_atom_indices,
            'rec_missing_mask': rec_missing_mask,
            'esm_embeddings': esm_emb,
            'lig_atom_feats': ligand_feats['atom_features'],
            'lig_atom_coords_gt': lig_coords_gt,
            'lig_atom_ref_coords': lig_ref_coords,
            'lig_bond_feats': ligand_feats['bond_features'],
            'lig_uid_vec': ligand_feats['ligand_uid_vec'],
            'unimol_embeddings': unimol_emb,
            'rec_constraint_matrix': rec_constraint_matrix,
            'lig_constraint_matrix': lig_constraint_matrix,
            'rec_torsion_indices': pocket_feats.get('sp2_torsion_indices', pocket_feats.get('torsion_indices', torch.empty((0, 4), dtype=torch.long))),
            'lig_torsion_indices': ligand_feats.get('sp2_torsion_indices', ligand_feats.get('torsion_indices', torch.empty((0, 4), dtype=torch.long))),
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample


def collate_single_sample(batch):
    """Collate function for single sample batches."""
    B = len(batch)
    sample = batch[0]

    N_rec = sample['rec_atom_feats'].shape[0]
    N_lig = sample['lig_atom_feats'].shape[0]
    N_rec_tokens = sample['esm_embeddings'].shape[0]

    collated = {
        # System-level [1, ...] — matches V4 chemistry trunk expectations
        'rec_atom_feats': sample['rec_atom_feats'].unsqueeze(0),
        'rec_atom_ref_coords': sample['rec_atom_ref_coords'].unsqueeze(0),
        'rec_constraint_matrix': sample['rec_constraint_matrix'].unsqueeze(0),
        'rec_bond_feats': sample['rec_bond_feats'].unsqueeze(0),
        'rec_uid_vec': sample['rec_uid_vec'].unsqueeze(0),
        'esm_embeddings': sample['esm_embeddings'].unsqueeze(0),
        'lig_atom_feats': sample['lig_atom_feats'].unsqueeze(0),
        'lig_atom_ref_coords': sample['lig_atom_ref_coords'].unsqueeze(0),
        'lig_constraint_matrix': sample['lig_constraint_matrix'].unsqueeze(0),
        'lig_bond_feats': sample['lig_bond_feats'].unsqueeze(0),
        'lig_uid_vec': sample['lig_uid_vec'].unsqueeze(0),
        'unimol_embeddings': sample['unimol_embeddings'].unsqueeze(0),
        # Per-timestep [B, ...]
        'rec_atom_coords_gt': sample['rec_atom_coords_gt'].unsqueeze(0).expand(B, -1, -1).clone(),
        'lig_atom_coords_gt': sample['lig_atom_coords_gt'].unsqueeze(0).expand(B, -1, -1).clone(),
        'rec_backbone_mask': sample['rec_backbone_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_cb_mask': sample['rec_cb_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_missing_mask': sample['rec_missing_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_atom_mask': torch.ones(B, N_rec, dtype=torch.bool),
        'lig_atom_mask': torch.ones(B, N_lig, dtype=torch.bool),
        'batch_size': B,
    }

    # Per-residue backbone atom indices (system-level, no batch expansion)
    bai = sample.get('rec_backbone_atom_indices')
    if bai is not None:
        collated['rec_backbone_atom_indices'] = bai

    # Handle torsion indices: prepend batch index column → [batch_idx, a, b, c, d]
    for key in ('rec_torsion_indices', 'lig_torsion_indices'):
        t = sample[key]
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).long()
        if t is not None and t.numel() > 0:
            parts = []
            for b in range(B):
                batch_col = torch.full((t.shape[0], 1), b, dtype=torch.long)
                parts.append(torch.cat([batch_col, t], dim=1))
            collated[key] = torch.cat(parts, dim=0)
        else:
            collated[key] = torch.empty((0, 5), dtype=torch.long)

    return collated


def main():
    parser = argparse.ArgumentParser(description='Overfit test for Flex model')
    parser.add_argument('--system', type=str, required=True,
                        help='Path to system directory to overfit on')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--max-steps', type=int, default=5000,
                        help='Maximum training steps')
    parser.add_argument('--val-interval', type=int, default=100,
                        help='Validation interval (steps)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (all samples are identical)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--fixed-timestep', type=float, default=None,
                        help='Fix timestep to a constant value (e.g., 0.5) for true single-sample overfitting')
    parser.add_argument('--num-systems', type=int, default=1,
                        help='Number of systems to overfit on (randomly sampled from --system parent dir)')
    parser.add_argument('--num-unrolls', type=int, default=None,
                        help='Override loss.num_unrolls')
    parser.add_argument('--phys-warmup-steps', type=int, default=None,
                        help='Override training.phys_loss_warmup_steps')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging for the overfit run')
    args = parser.parse_args()

    # Verify system exists
    system_path = Path(args.system)
    if not system_path.exists():
        raise FileNotFoundError(f"System directory not found: {args.system}")

    print(f"=" * 60)
    print("OVERFITTING TEST")
    print(f"=" * 60)
    print(f"System: {args.system}")
    print(f"Num systems: {args.num_systems}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fixed timestep: {args.fixed_timestep if args.fixed_timestep else 'None (varying)'}")
    print(f"=" * 60)

    # Load config
    config = load_config(args.config)

    # Override config for overfitting test
    config.training.max_steps = args.max_steps
    config.training.val_check_interval = args.val_interval
    config.data.batch_size = args.batch_size
    config.logging.log_every_n_steps = 10
    config.logging.checkpoint_every_n_steps = args.max_steps  # Only save at end
    config.ema.enabled = False
    config.training.devices = 1
    config.training.accumulate_grad_batches = 1

    if args.num_unrolls is not None:
        config.loss.num_unrolls = args.num_unrolls
    if args.phys_warmup_steps is not None:
        config.training.phys_loss_warmup_steps = args.phys_warmup_steps
    if args.no_wandb:
        config.logging.use_wandb = False

    # Disable compile if requested
    if args.no_compile:
        config.training.use_compile = False

    # Set seed
    pl.seed_everything(config.training.seed)
    torch.set_float32_matmul_precision('high')

    data_dir = system_path.parent

    if args.num_systems == 1:
        system_ids = [system_path.name]
        config.logging.wandb_run_name = f"overfit_{system_path.name}"
    else:
        all_systems = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
        ])
        system_ids = random.sample(all_systems, min(args.num_systems, len(all_systems)))
        config.logging.wandb_run_name = f"overfit_{args.num_systems}systems"
        print(f"Selected systems: {system_ids}")

    print(f"=" * 60)

    if args.num_systems == 1:
        train_dataset = SingleSampleDataset(args.system, length=10000)
        val_dataset = SingleSampleDataset(args.system, length=1)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_single_sample,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_single_sample,
        )

        sample = train_dataset.sample
        print(f"Receptor atoms: {sample['rec_atom_feats'].shape[0]}")
        print(f"Ligand atoms:   {sample['lig_atom_feats'].shape[0]}")
        print(f"ESM tokens:     {sample['esm_embeddings'].shape[0]}")
    else:
        train_dataset = FlexDataset(str(data_dir), system_ids=system_ids)
        val_dataset = FlexDataset(str(data_dir), system_ids=system_ids)

        train_sampler = SingleSystemSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_sampler = SingleSystemSampler(val_dataset, batch_size=args.batch_size, shuffle=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_single_system,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_single_system,
        )
        print(f"Loaded {len(system_ids)} systems")

    print(f"=" * 60)

    # Create model
    trainer_module = FlexTrainer(config, fixed_timestep=args.fixed_timestep)
    print("Loaded FlexTrainer")

    # Create trainer
    trainer = get_trainer(config)
    print("Loaded trainer")

    # Add RMSD callback
    rmsd_callback = RMSDEvaluationCallback(
        train_systems=system_ids,
        val_systems=system_ids,
        test_systems=None,
        data_dir=str(data_dir),
        num_samples=1,
        num_steps=config.training.num_steps,
        eval_every_n_val=1,
    )
    trainer.callbacks.append(rmsd_callback)
    print("Loaded RMSD callback")

    # Train
    print(f"\nStarting overfitting test...")
    print(f"Expected: loss -> ~0, RMSD -> < 0.5 Å")
    print(f"=" * 60)

    trainer.fit(
        trainer_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Final evaluation (single-system only — multi-system RMSD is logged by the callback)
    if args.num_systems == 1:
        print(f"\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        # cuequivariance triangle / fused-LayerNorm kernels are Triton GPU-only.
        # Lightning may leave the module on CPU after fit teardown, so force CUDA.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer_module = trainer_module.to(device)
        trainer_module.eval()

        batch = collate_single_sample([train_dataset.sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            result = trainer_module._euler_integrate(batch, num_steps=config.training.num_steps)

            rec_rmsd = trainer_module._compute_rmsd(
                result['rec_pred'], result['rec_gt'], batch['rec_atom_mask'].float()
            ).item()

            lig_rmsd = trainer_module._compute_rmsd(
                result['lig_pred'], result['lig_gt'], batch['lig_atom_mask'].float()
            ).item()

            rec_backbone_mask = result.get('rec_backbone_mask')
            if rec_backbone_mask is not None:
                sidechain_mask = (~rec_backbone_mask.bool()).float() * batch['rec_atom_mask'].float()
                sc_rmsd = trainer_module._compute_rmsd(
                    result['rec_pred'], result['rec_gt'], sidechain_mask
                ).item() if sidechain_mask.sum() > 0 else 0.0
            else:
                sc_rmsd = rec_rmsd

        print(f"Receptor RMSD:  {rec_rmsd:.4f} Å")
        print(f"Sidechain RMSD: {sc_rmsd:.4f} Å")
        print(f"Ligand RMSD:    {lig_rmsd:.4f} Å")
        print(f"=" * 60)

        print("\nDIAGNOSIS:")
        if lig_rmsd < 0.5:
            print("SUCCESS: Model can overfit. Architecture is working.")
        elif lig_rmsd < 2.0:
            print("PARTIAL: Model is learning but not converged. Try more steps.")
        else:
            print("FAILURE: Model cannot overfit. Check architecture/loss.")


if __name__ == '__main__':
    main()
