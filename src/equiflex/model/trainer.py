"""
PyTorch Lightning trainer for Flex model.

Features:
- WandB logging (optional)
- EMA (Exponential Moving Average) of model weights
- Flow matching training with auxiliary losses
- Unified configuration system
- Test set evaluation with RMSD metrics
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Dict, Optional, Union, List
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import numpy as np
import gc
import logging
from pathlib import Path

from .models import FlexModel, FlexConfig
from .loss import FlexLoss, LossConfig
from .config import FullConfig, load_config
from equiflex.prep.geometry import place_cb_ideal
from equiflex.prep.validation import validate_cached_unimol

from muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
#from normuon import SingleDeviceNorMuonWithAuxAdam

from conflictfree.grad_operator import ConFIG_update_double
from conflictfree.utils import get_gradient_vector, apply_gradient_vector


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class EMACallback(Callback):
    """PyTorch Lightning callback for EMA."""

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = EMA(pl_module.model, self.decay)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update(pl_module.model)

    def on_validation_start(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.apply(pl_module.model)

    def on_validation_end(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.restore(pl_module.model)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state_dict' in checkpoint:
            if self.ema is None:
                self.ema = EMA(pl_module.model, self.decay)
            self.ema.load_state_dict(checkpoint['ema_state_dict'])


class FlexTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for Flex model training.
    """

    def __init__(self, config: FullConfig, fixed_timestep: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = FlexModel(config.model)
        self.loss_fn = FlexLoss(config.loss)

        # ConFIG conflict-free gradient update: requires manual optimization
        self.use_config = getattr(config.training, 'use_config', False)
        if self.use_config:
            self.automatic_optimization = False
            # Per-group grad accumulators (mean over accumulate_grad_batches, then ConFIG)
            self._flow_grad_accum: Optional[torch.Tensor] = None
            self._phys_grad_accum: Optional[torch.Tensor] = None
            self._config_accum_count: int = 0

        # For overfitting tests: fix timestep to a constant value
        self.fixed_timestep = fixed_timestep

        # Apply torch.compile for faster training/inference
        if config.training.use_compile:
            self.model = torch.compile(self.model, mode=config.training.compile_mode)

        # Loss accumulators for averaging over log_every_n_steps.
        # Kept as detached GPU tensors to avoid per-step GPU→CPU sync;
        # materialized on first training step (device known then).
        self._loss_keys = (
            'total', 'flow', 'bond_length', 'bond_angle',
            'vdw', 'vdw_active', 'vdw_max', 'hbond', 'torsion', 'chiral',
        )
        self._loss_accum: Dict[str, torch.Tensor] = {}
        self._loss_count = 0


    def forward(self, batch: Dict[str, torch.Tensor], t: torch.Tensor, return_attn: bool = False,
                q_chem=None, p_chem=None):
        """Forward pass through the model."""
        return self.model(
            rec_atom_feats=batch['rec_atom_feats'],
            rec_atom_coords=batch['rec_atom_coords'],
            rec_atom_ref_coords=batch['rec_atom_ref_coords'],
            rec_constraint_matrix=batch['rec_constraint_matrix'],
            rec_bond_feats=batch['rec_bond_feats'],
            rec_uid_vec=batch['rec_uid_vec'],
            rec_atom_mask=batch['rec_atom_mask'],
            rec_backbone_mask=batch['rec_backbone_mask'],
            esm_embeddings=batch['esm_embeddings'],
            lig_atom_feats=batch['lig_atom_feats'],
            lig_atom_coords=batch['lig_atom_coords'],
            lig_atom_ref_coords=batch['lig_atom_ref_coords'],
            lig_constraint_matrix=batch['lig_constraint_matrix'],
            lig_bond_feats=batch['lig_bond_feats'],
            lig_uid_vec=batch['lig_uid_vec'],
            lig_atom_mask=batch['lig_atom_mask'],
            unimol_embeddings=batch['unimol_embeddings'],
            t=t,
            return_attn=return_attn,
            q_chem=q_chem,
            p_chem=p_chem,
            rec_missing_mask=batch.get('rec_missing_mask'),
            system_index_per_timestep=batch.get('system_index_per_timestep'),
        )

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps for flow matching.

        Sampler is controlled by config.training.timestep_sampler:
          'uniform'      — U(0, 1)
          'logit_normal' — 0.98*LogitNormal(m=0.8, s=1.7) + 0.02*U(0,1),
                           biases toward t=1 for training stability

        If self.fixed_timestep is set, returns a constant (for overfitting tests).
        """
        if self.fixed_timestep is not None:
            return torch.full((batch_size,), self.fixed_timestep, device=device)

        sampler = getattr(self.config.training, 'timestep_sampler', 'logit_normal')

        if sampler == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:
            # Logit-normal: sample z ~ N(m, s^2), then t = sigmoid(z)
            m, s = 0.8, 1.7
            z = torch.randn(batch_size, device=device) * s + m
            t_logit_normal = torch.sigmoid(z)

            # 2% uniform mixture to ensure coverage of full [0, 1] range
            t_uniform = torch.rand(batch_size, device=device)
            mask = torch.bernoulli(torch.full((batch_size,), 0.02, device=device)).bool()
            t = torch.where(mask, t_uniform, t_logit_normal)

        return t.clamp(1e-5, 1 - 1e-5)

    def _prepare_flow_matching(self, batch: Dict[str, torch.Tensor]):
        """
        Prepare flow matching inputs.

        Linear interpolant: x_t = t * x_1 + (1-t) * x_0
        where x_1 = ground truth, x_0 = noise

        Coordinate centering:
        - All coordinates are centered by global backbone COM
        - This allows ligand x_0 to be N(0, I), which is inference-compatible

        Noise initialization (after centering):
        - Backbone atoms: fixed at ground truth (x_0 = x_1)
        - Sidechain atoms: noise centered at per-residue backbone COM, N(residue_backbone_COM, I)
        - Ligand atoms: noise from origin, N(0, I) (= global backbone COM after centering)
        """
        B = batch.get('batch_size', batch['rec_atom_coords_gt'].shape[0])
        N_rec = batch['rec_atom_feats'].shape[1]
        device = batch['rec_atom_feats'].device

        # Sample timesteps from mixture distribution
        t = self._sample_timesteps(B, device)

        # Ground truth coordinates (x_1) — always [B, ...]
        rec_x_1 = batch['rec_atom_coords_gt']
        lig_x_1 = batch['lig_atom_coords_gt']

        # Backbone mask: 1 for backbone atoms, 0 for sidechain
        rec_backbone_mask = batch.get('rec_backbone_mask')
        # rec_uid_vec may be [1, N_rec] (single-system) or [S, N_rec]
        # (padded multi-system). Expand/index it to the timestep batch.
        rec_uid_vec = batch['rec_uid_vec']
        sys_idx = batch.get('system_index_per_timestep')
        if sys_idx is not None and rec_uid_vec.shape[0] != B:
            rec_uid_vec = rec_uid_vec.index_select(0, sys_idx.to(rec_uid_vec.device).long())
        elif rec_uid_vec.shape[0] == 1 and B > 1:
            rec_uid_vec = rec_uid_vec.expand(B, -1)

        if rec_backbone_mask is not None:
            backbone_mask_3d = rec_backbone_mask.unsqueeze(-1)  # [B, N_rec, 1]
            rec_cb_mask = batch.get('rec_cb_mask')
            cb_mask_3d = rec_cb_mask.unsqueeze(-1) if rec_cb_mask is not None else torch.zeros_like(backbone_mask_3d)

            # Global backbone COM for centering; missing atoms hold 0-placeholders.
            com_mask = rec_backbone_mask
            rec_missing_mask = batch.get('rec_missing_mask')
            if rec_missing_mask is not None:
                com_mask = rec_backbone_mask * (~rec_missing_mask.bool())
            backbone_coords_sum = (com_mask.unsqueeze(-1) * rec_x_1).sum(dim=1, keepdim=True)  # [B, 1, 3]
            backbone_count = com_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # [B, 1, 1]
            global_backbone_com = backbone_coords_sum / backbone_count  # [B, 1, 3]

            # Center all coordinates by global backbone COM
            rec_x_1 = rec_x_1 - global_backbone_com
            lig_x_1 = lig_x_1 - global_backbone_com

            # CB placement: compute ideal CB from backbone N, CA, C per residue
            bai = batch.get('rec_backbone_atom_indices')
            if bai is not None:
                n_idx = bai['n_idx'].to(device)    # [n_residues]
                ca_idx = bai['ca_idx'].to(device)   # [n_residues]
                c_idx = bai['c_idx'].to(device)     # [n_residues]
                is_gly = bai['is_gly'].to(device)   # [n_residues]

                # Index backbone coords: rec_x_1 is [B, N_rec, 3], indices are system-level
                n_coords = rec_x_1[:, n_idx]    # [B, n_residues, 3]
                ca_coords = rec_x_1[:, ca_idx]  # [B, n_residues, 3]
                c_coords = rec_x_1[:, c_idx]    # [B, n_residues, 3]

                # Ideal CB position per residue
                cb_ideal = place_cb_ideal(n_coords, ca_coords, c_coords)  # [B, n_residues, 3]

                # For GLY: fall back to CA position as noise center
                gly_mask = is_gly.unsqueeze(0).unsqueeze(-1)  # [1, n_residues, 1]
                cb_center = torch.where(gly_mask, ca_coords, cb_ideal)  # [B, n_residues, 3]

                # Broadcast CB center to all atoms via residue_uid_vec
                atom_noise_center = torch.gather(
                    cb_center, 1,
                    rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                )  # [B, N_rec, 3]
            else:
                # Fallback: per-residue backbone COM (old behavior)
                n_residues = rec_uid_vec.max().item() + 1
                weighted_coords = backbone_mask_3d * rec_x_1
                residue_backbone_sum = torch.zeros(B, n_residues, 3, device=device, dtype=rec_x_1.dtype)
                residue_backbone_count = torch.zeros(B, n_residues, 1, device=device, dtype=rec_x_1.dtype)
                uid_expanded = rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                uid_count = rec_uid_vec.unsqueeze(-1)
                residue_backbone_sum.scatter_add_(1, uid_expanded, weighted_coords)
                residue_backbone_count.scatter_add_(1, uid_count, backbone_mask_3d)
                residue_backbone_count = residue_backbone_count.clamp(min=1.0)
                residue_backbone_com = residue_backbone_sum / residue_backbone_count
                atom_noise_center = torch.gather(
                    residue_backbone_com, 1,
                    rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                )

            sc_sigma = self.config.model.sidechain_prior_sigma
            lig_sigma = self.config.model.ligand_prior_sigma
            # Sidechain noise: N(CB center, sc_sigma^2)
            rec_noise = torch.randn(rec_x_1.shape, device=device, dtype=rec_x_1.dtype) * sc_sigma + atom_noise_center

            # Backbone + CB stay at GT, rest from noise
            fixed_mask = backbone_mask_3d + cb_mask_3d  # both fixed
            rec_x_0 = fixed_mask * rec_x_1 + (1 - fixed_mask) * rec_noise
        else:
            sc_sigma = self.config.model.sidechain_prior_sigma
            lig_sigma = self.config.model.ligand_prior_sigma
            # No backbone mask: center by receptor COM and noise from origin
            global_com = rec_x_1.mean(dim=1, keepdim=True)
            rec_x_1 = rec_x_1 - global_com
            lig_x_1 = lig_x_1 - global_com
            rec_x_0 = torch.randn(rec_x_1.shape, device=device, dtype=rec_x_1.dtype) * sc_sigma

        # Ligand noise: N(backbone_COM, lig_sigma^2) (origin after centering)
        lig_x_0 = torch.randn(lig_x_1.shape, device=device, dtype=lig_x_1.dtype) * lig_sigma

        # Interpolate to get x_t
        t_expanded = t.view(B, 1, 1)

        rec_x_t = t_expanded * rec_x_1 + (1 - t_expanded) * rec_x_0
        lig_x_t = t_expanded * lig_x_1 + (1 - t_expanded) * lig_x_0

        return {
            't': t,
            'rec_x_t': rec_x_t,
            'lig_x_t': lig_x_t,
            'rec_x_1': rec_x_1,
            'lig_x_1': lig_x_1,
            'rec_x_0': rec_x_0,
            'lig_x_0': lig_x_0
        }

    def _expand_padded_batch_for_timesteps(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand [S, ...] system tensors to [S*T, ...] for flow/loss code."""
        if not batch.get('padded_batch', False):
            return batch

        B = int(batch.get('batch_size', batch['rec_atom_coords_gt'].shape[0]))
        sys_idx = batch['system_index_per_timestep'].long()
        out = dict(batch)
        system_fields = (
            'rec_atom_feats', 'rec_atom_ref_coords', 'rec_constraint_matrix',
            'rec_bond_feats', 'rec_uid_vec', 'esm_embeddings',
            'lig_atom_feats', 'lig_atom_ref_coords', 'lig_constraint_matrix',
            'lig_bond_feats', 'lig_uid_vec', 'unimol_embeddings',
        )
        for key in system_fields:
            value = out.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] != B:
                out[key] = value.index_select(0, sys_idx.to(value.device))
        return out

    def _effective_batch_size(self, batch) -> int:
        if isinstance(batch, list):
            return sum(self._effective_batch_size(sub_batch) for sub_batch in batch)
        return int(batch.get('batch_size', batch['rec_atom_coords_gt'].shape[0]))

    def _aggregate_loss_dicts(self, loss_dicts):
        """Average scalar tensor losses from several variable-size systems."""
        if len(loss_dicts) == 1:
            return loss_dicts[0]

        result = {}
        keys = set().union(*(d.keys() for d in loss_dicts))
        for key in keys:
            if key == 'batch_size':
                result[key] = sum(int(d.get(key, 0)) for d in loss_dicts)
                continue
            values = [d[key] for d in loss_dicts if key in d and isinstance(d[key], torch.Tensor) and d[key].ndim == 0]
            if values:
                result[key] = torch.stack(values).mean()
        return result

    def _shared_step(self, batch: Dict[str, torch.Tensor], return_attn: bool = False, aux_scale: float = 1.0, num_unrolls: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Shared computation for training and validation."""
        if isinstance(batch, list):
            losses = [
                self._shared_step(sub_batch, return_attn=return_attn, aux_scale=aux_scale, num_unrolls=num_unrolls)
                for sub_batch in batch
            ]
            return self._aggregate_loss_dicts(losses)

        fm = self._prepare_flow_matching(batch)

        # Create input batch with noisy coords. In padded mode, expand the
        # system-level tensors to the timestep batch for losses/unroll, while
        # q_chem/p_chem remain at [S, ...] and are indexed inside forward_flow.
        input_batch = {
            **batch,
            'rec_atom_coords': fm['rec_x_t'],
            'lig_atom_coords': fm['lig_x_t'],
        }
        input_batch = self._expand_padded_batch_for_timesteps(input_batch)

        # Run chemistry trunk once per distinct system.
        if batch.get('padded_batch', False):
            q_chem, p_chem = self.model.forward_chemistry(
                rec_atom_feats=batch['rec_atom_feats'],
                rec_atom_ref_coords=batch['rec_atom_ref_coords'],
                rec_constraint_matrix=batch['rec_constraint_matrix'],
                rec_bond_feats=batch['rec_bond_feats'],
                rec_uid_vec=batch['rec_uid_vec'],
                esm_embeddings=batch['esm_embeddings'],
                lig_atom_feats=batch['lig_atom_feats'],
                lig_atom_ref_coords=batch['lig_atom_ref_coords'],
                lig_constraint_matrix=batch['lig_constraint_matrix'],
                lig_bond_feats=batch['lig_bond_feats'],
                lig_uid_vec=batch['lig_uid_vec'],
                unimol_embeddings=batch['unimol_embeddings'],
                rec_atom_mask=batch['rec_atom_mask_system'],
                lig_atom_mask=batch['lig_atom_mask_system'],
            )
        else:
            q_chem, p_chem = self.model.forward_chemistry(
                rec_atom_feats=batch['rec_atom_feats'][0:1],
                rec_atom_ref_coords=batch['rec_atom_ref_coords'][0:1],
                rec_constraint_matrix=batch['rec_constraint_matrix'][0:1],
                rec_bond_feats=batch['rec_bond_feats'][0:1],
                rec_uid_vec=batch['rec_uid_vec'][0:1],
                esm_embeddings=batch['esm_embeddings'][0:1],
                lig_atom_feats=batch['lig_atom_feats'][0:1],
                lig_atom_ref_coords=batch['lig_atom_ref_coords'][0:1],
                lig_constraint_matrix=batch['lig_constraint_matrix'][0:1],
                lig_bond_feats=batch['lig_bond_feats'][0:1],
                lig_uid_vec=batch['lig_uid_vec'][0:1],
                unimol_embeddings=batch['unimol_embeddings'][0:1],
                rec_atom_mask=batch['rec_atom_mask'][0:1],
                lig_atom_mask=batch['lig_atom_mask'][0:1],
            )

        # Forward pass (flow trunk only - used for flow loss)
        output = self.forward(input_batch, fm['t'], return_attn=return_attn,
                              q_chem=q_chem, p_chem=p_chem)

        # Extract flow predictions and free output dict to prevent VRAM accumulation
        rec_flow = output['rec_flow']
        lig_flow = output['lig_flow']
        attn_stats = output.get('attn_stats', None)
        del output

        # Combine coordinates and masks
        N_rec = batch['rec_atom_feats'].shape[1]
        N_lig = batch['lig_atom_feats'].shape[1]
        B = batch.get('batch_size', batch['rec_atom_coords_gt'].shape[0])

        x_0 = torch.cat([fm['rec_x_0'], fm['lig_x_0']], dim=1)
        x_1 = torch.cat([fm['rec_x_1'], fm['lig_x_1']], dim=1)
        pred_flow = torch.cat([rec_flow, lig_flow], dim=1)

        # Ground truth flow: v* = x_1 - x_0
        flow_gt = x_1 - x_0

        # Atom mask: valid atoms (not padding) — [B, ...]
        atom_mask = torch.cat([batch['rec_atom_mask'], batch['lig_atom_mask']], dim=1)
        loss_mask = atom_mask

        # Exclude atoms with missing coordinates from loss
        rec_missing_mask = batch.get('rec_missing_mask')
        if rec_missing_mask is not None:
            lig_missing_mask = torch.zeros_like(batch['lig_atom_mask'])
            missing_mask = torch.cat([rec_missing_mask, lig_missing_mask], dim=1)
            loss_mask = loss_mask * (~missing_mask)

        # Combined bond mask (block diagonal, single channel). For padded
        # multi-system batches, build it per system then index to [B, N, N].
        N_atoms = N_rec + N_lig
        rec_bond_mask = (batch['rec_bond_feats'].abs().sum(dim=-1) > 0).float()
        lig_bond_mask = (batch['lig_bond_feats'].abs().sum(dim=-1) > 0).float()
        if batch.get('padded_batch', False):
            S = batch['rec_atom_feats'].shape[0]
            sys_idx = batch['system_index_per_timestep'].to(x_0.device).long()
            bond_mask_system = torch.zeros(S, N_atoms, N_atoms, device=x_0.device, dtype=torch.float32)
            bond_mask_system[:, :N_rec, :N_rec] = rec_bond_mask
            bond_mask_system[:, N_rec:, N_rec:] = lig_bond_mask
            bond_mask_full = bond_mask_system.index_select(0, sys_idx)
        else:
            bond_mask_full = torch.zeros(1, N_atoms, N_atoms, device=x_0.device, dtype=torch.float32)
            bond_mask_full[:, :N_rec, :N_rec] = rec_bond_mask
            bond_mask_full[:, N_rec:, N_rec:] = lig_bond_mask

        # Combined constraint matrix (block diagonal).
        constraint_dim = batch['rec_constraint_matrix'].shape[-1]
        if batch.get('padded_batch', False):
            S = batch['rec_atom_feats'].shape[0]
            sys_idx = batch['system_index_per_timestep'].to(x_0.device).long()
            constraint_system = torch.zeros(
                S, N_atoms, N_atoms, constraint_dim, device=x_0.device, dtype=batch['rec_constraint_matrix'].dtype
            )
            constraint_system[:, :N_rec, :N_rec, :] = batch['rec_constraint_matrix']
            constraint_system[:, N_rec:, N_rec:, :] = batch['lig_constraint_matrix']
            constraint_matrix = constraint_system.index_select(0, sys_idx)
        else:
            constraint_matrix = torch.zeros(
                1, N_atoms, N_atoms, constraint_dim, device=x_0.device, dtype=batch['rec_constraint_matrix'].dtype
            )
            constraint_matrix[:, :N_rec, :N_rec, :] = batch['rec_constraint_matrix']
            constraint_matrix[:, N_rec:, N_rec:, :] = batch['lig_constraint_matrix']

        # Backbone mask: 1 for backbone atoms, 0 for sidechain/ligand atoms — [B, ...]
        rec_backbone_mask = batch.get('rec_backbone_mask')
        if rec_backbone_mask is not None:
            lig_backbone_mask = torch.zeros(B, N_lig, device=x_0.device)
            backbone_mask = torch.cat([rec_backbone_mask, lig_backbone_mask], dim=1)
        else:
            backbone_mask = None

        # CB mask: 1 for CB atoms, 0 for others — [B, ...]
        rec_cb_mask = batch.get('rec_cb_mask')
        if rec_cb_mask is not None:
            lig_cb_mask = torch.zeros(B, N_lig, device=x_0.device)
            cb_mask = torch.cat([rec_cb_mask, lig_cb_mask], dim=1)
        else:
            cb_mask = None

        # Compute losses:
        # - Flow loss: MSE(flow_pred, flow_gt) without unrolling
        # - Auxiliary losses: computed on unrolled x_1_pred (N steps from x_t)
        losses = self.loss_fn(
            model=self.model,
            batch=input_batch,
            pred_flow=pred_flow,
            flow_gt=flow_gt,
            x_1=x_1,
            t=fm['t'],
            rec_x_t=fm['rec_x_t'],
            lig_x_t=fm['lig_x_t'],
            mask=loss_mask.float(),
            bond_mask=bond_mask_full,
            constraint_matrix=constraint_matrix,
            rec_torsion_indices=batch.get('rec_torsion_indices'),
            lig_torsion_indices=batch.get('lig_torsion_indices'),
            lig_chiral_indices=batch.get('lig_chiral_indices'),
            n_rec=N_rec,
            backbone_mask=backbone_mask,
            cb_mask=cb_mask,
            aux_scale=aux_scale,
            q_chem=q_chem,
            p_chem=p_chem,
            num_unrolls=num_unrolls,
        )

        # Add batch size for logging
        losses['batch_size'] = B

        if return_attn and attn_stats is not None:
            losses['attn_stats'] = attn_stats

        return losses

    def _curriculum_num_unrolls(self) -> int:
        """Per-step unroll depth under the PBFM-style curriculum.

        Ramps 1 -> num_unrolls in num_unrolls equal stages across the post-warmup
        portion of training (each stage = 1/n of (max_steps - warmup_steps)).
        Returns the full configured depth when curriculum is disabled.
        """
        n = self.config.loss.num_unrolls
        if not getattr(self.config.loss, 'unroll_curriculum', False) or n <= 1:
            return n

        warmup = max(0, self.config.training.phys_loss_warmup_steps)
        max_steps = self.config.training.max_steps
        span = max(1, max_steps - warmup)
        progress = (self.global_step - warmup) / span  # <0 during warmup
        stage = int(progress * n) + 1                  # 1..n across the span
        return max(1, min(n, stage))

    def _skip_oom_step(self, err: Exception):
        """Recover from a CUDA OOM by dropping the current gradient update.

        Zeroes any partial gradients, resets ConFIG accumulators (so a partially
        accumulated micro-batch does not corrupt the next ConFIG merge), frees the
        cached allocator blocks, and returns ``None`` — which tells Lightning to
        skip the optimizer step for this batch rather than crash the run.
        """
        print(f"WARNING: CUDA OOM at step {self.global_step}; skipping gradient update. {err}")
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if self.use_config:
            self._flow_grad_accum = None
            self._phys_grad_accum = None
            self._config_accum_count = 0
        gc.collect()
        torch.cuda.empty_cache()
        return None

    def backward(self, loss, *args, **kwargs):
        """Guard Lightning's automatic backward against CUDA OOM.

        In automatic-optimization mode (e.g. ablations with use_config=False) the
        backward pass runs here, outside training_step. Returning None from
        training_step only skips forward-OOM batches; a backward-time OOM would
        still crash. On OOM we drop the gradients (zeroed) so the ensuing
        optimizer.step() is a no-op — the update is effectively skipped — instead
        of aborting the run. Manual-optimization (ConFIG) backward propagates
        normally and is handled at the _config_backward_and_step call site.
        """
        if not self.automatic_optimization:
            return super().backward(loss, *args, **kwargs)
        try:
            super().backward(loss, *args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            print(f"WARNING: CUDA OOM in backward at step {self.global_step}; "
                  f"skipping gradient update. {e}")
            optimizers = self.optimizers()
            if not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        # Check if we should collect attention stats this step
        log_interval = self.config.logging.log_every_n_steps
        should_log_attn = self.config.model.return_attn_stats and (self.global_step % log_interval == 0)

        warmup_steps = self.config.training.phys_loss_warmup_steps
        in_warmup = warmup_steps > 0 and self.global_step < warmup_steps
        if in_warmup:
            # ConFIG normalises grad direction; pre-merge scaling has near-zero
            # effect. Skip physics entirely during warmup -> saves unroll
            # compute, avoids polluting ConFIG with random-init physics dir.
            if self.use_config:
                aux_scale = 0.0
            else:
                aux_scale = float(self.global_step) / float(warmup_steps)
        else:
            aux_scale = 1.0

        num_unrolls = self._curriculum_num_unrolls()

        sync_dist = getattr(self.trainer, "world_size", 1) > 1
        B = self._effective_batch_size(batch)
        self.log(
            'train/phys_loss_scale',
            aux_scale,
            on_step=True,
            on_epoch=False,
            batch_size=B,
            logger=True,
            sync_dist=sync_dist,
        )
        self.log(
            'train/num_unrolls',
            float(num_unrolls),
            on_step=True,
            on_epoch=False,
            batch_size=B,
            logger=True,
            sync_dist=sync_dist,
        )

        try:
            losses = self._shared_step(batch, return_attn=should_log_attn, aux_scale=aux_scale, num_unrolls=num_unrolls)
        except torch.cuda.OutOfMemoryError as e:
            # Skip the gradient update for this batch instead of crashing.
            return self._skip_oom_step(e)

        # Check for NaN/Inf in loss
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            print(f"WARNING: NaN/Inf loss detected at step {self.global_step}")
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.item():.6f} (nan={torch.isnan(v).any()}, inf={torch.isinf(v).any()})")
            if self.use_config:
                opt = self.optimizers()
                opt.zero_grad()
                return None
            return losses['total'].new_tensor(0.0, requires_grad=True)

        if self.use_config:
            try:
                self._config_backward_and_step(losses)
            except torch.cuda.OutOfMemoryError as e:
                return self._skip_oom_step(e)

        B = self._effective_batch_size(batch)

        # Lazy-init GPU accumulators on first step (device known here).
        if not self._loss_accum:
            ref = losses['total']
            self._loss_accum = {
                k: torch.zeros((), device=ref.device, dtype=torch.float32)
                for k in self._loss_keys
            }

        # Accumulate as detached GPU tensors — no per-step GPU→CPU sync.
        # Values are already weighted contributions.
        for key in self._loss_keys:
            val = losses.get(key, None) if key != 'total' else losses['total']
            if isinstance(val, torch.Tensor):
                self._loss_accum[key] += val.detach().float()
        self._loss_count += 1

        # Log averaged losses at every n steps (single sync per key here).
        if self.global_step % log_interval == 0 and self._loss_count > 0:
            sync_dist = getattr(self.trainer, "world_size", 1) > 1
            inv_count = 1.0 / self._loss_count
            for key in self._loss_keys:
                avg = (self._loss_accum[key] * inv_count).item()
                prog = key in ('total', 'flow')
                self.log(f'train/{key}_loss', avg, on_step=True, on_epoch=False,
                         prog_bar=prog, batch_size=B, logger=True, sync_dist=sync_dist)
            self.log('global_step', float(self.global_step), on_step=True, on_epoch=False, batch_size=B, logger=True, sync_dist=sync_dist)

            # Log attention entropy stats if available
            attn_stats = losses.get('attn_stats')
            if attn_stats is not None:
                if 'atom_entropy' in attn_stats:
                    self.log('train/attn_atom_entropy', attn_stats['atom_entropy'].item(),
                             on_step=True, on_epoch=False, batch_size=B, logger=True, sync_dist=sync_dist)
                if 'eva_entropy' in attn_stats:
                    self.log('train/attn_eva_entropy', attn_stats['eva_entropy'].item(),
                             on_step=True, on_epoch=False, batch_size=B, logger=True, sync_dist=sync_dist)
                if 'geo_entropy' in attn_stats:
                    self.log('train/attn_geo_entropy', attn_stats['geo_entropy'].item(),
                             on_step=True, on_epoch=False, batch_size=B, logger=True, sync_dist=sync_dist)

            # Reset accumulators in-place (keeps tensors on GPU, no realloc).
            for key in self._loss_accum:
                self._loss_accum[key].zero_()
            self._loss_count = 0

        if self.use_config:
            return None
        return losses['total']

    def _config_backward_and_step(self, losses: Dict[str, torch.Tensor]) -> None:
        """ConFIG path: per-group backward, accumulate grad vectors across
        accumulate_grad_batches micro-batches, apply ConFIG to mean grads,
        then clip + step at the accumulation boundary."""
        opt = self.optimizers()
        sched = self.lr_schedulers()
        clip_val = self.config.training.gradient_clip_val
        accum_n = self.config.training.accumulate_grad_batches

        flow_loss = losses['flow_group']
        phys_loss = losses['physics_group']
        flow_has_grad = flow_loss.requires_grad
        phys_has_grad = phys_loss.requires_grad

        # flow_group and physics_group share the chemistry-trunk subgraph
        # (q_chem/p_chem feed both main forward and unroll). First backward
        # must retain_graph so the second can traverse shared saved tensors.
        both = flow_has_grad and phys_has_grad

        if flow_has_grad:
            opt.zero_grad()
            self.manual_backward(flow_loss, retain_graph=both)
            grad_flow = get_gradient_vector(self.model, none_grad_mode='zero')
            if self._flow_grad_accum is None:
                self._flow_grad_accum = grad_flow
            else:
                self._flow_grad_accum.add_(grad_flow)
                del grad_flow  # free transient buffer immediately

        if phys_has_grad:
            opt.zero_grad()
            self.manual_backward(phys_loss)
            grad_phys = get_gradient_vector(self.model, none_grad_mode='zero')
            if self._phys_grad_accum is None:
                self._phys_grad_accum = grad_phys
            else:
                self._phys_grad_accum.add_(grad_phys)
                del grad_phys

        self._config_accum_count += 1
        if self._config_accum_count < accum_n:
            return

        # Accumulation boundary: ConFIG on mean per-group grads, then step
        mean_flow = (self._flow_grad_accum / accum_n
                     if self._flow_grad_accum is not None else None)
        mean_phys = (self._phys_grad_accum / accum_n
                     if self._phys_grad_accum is not None else None)

        if mean_flow is not None and mean_phys is not None:
            g_config = ConFIG_update_double(mean_flow, mean_phys)
        elif mean_flow is not None:
            g_config = mean_flow
        elif mean_phys is not None:
            g_config = mean_phys
        else:
            self._reset_config_accum()
            return

        apply_gradient_vector(self.model, g_config, none_grad_mode='zero',
                              zero_grad_mode='pad_value')

        if clip_val and clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=clip_val,
                                gradient_clip_algorithm='norm')

        # Only compute diagnostics on log boundaries — clone+norms force
        # GPU→CPU sync, so doing them every accum boundary stalls the pipeline.
        # `opt.step()` below increments `global_step`, so gate on the post-step
        # value to align with the rest of training's log boundaries.
        log_interval = self.config.logging.log_every_n_steps
        log_now = ((self.global_step + 1) % log_interval == 0) or (self.global_step == 0)

        if log_now:
            p0 = next(self.model.parameters())
            p0_before = p0.detach().clone()
            opt.step()
            # Stack norms into a single tensor → one GPU→CPU transfer.
            norms = torch.stack([
                mean_flow.norm() if mean_flow is not None else g_config.new_tensor(0.0),
                mean_phys.norm() if mean_phys is not None else g_config.new_tensor(0.0),
                g_config.norm(),
                (p0 - p0_before).norm(),
            ]).tolist()
            gf, gp, gc, delta = norms
            if sched is not None:
                sched.step()

            ratio = gf / max(gp, 1e-12)
            sync_dist = getattr(self.trainer, "world_size", 1) > 1
            B = 1  # scalar metric; batch_size irrelevant
            self.log('train/config_grad_flow_norm', gf, on_step=True, on_epoch=False,
                     batch_size=B, logger=True, sync_dist=sync_dist)
            self.log('train/config_grad_phys_norm', gp, on_step=True, on_epoch=False,
                     batch_size=B, logger=True, sync_dist=sync_dist)
            self.log('train/config_grad_ratio', ratio, on_step=True, on_epoch=False,
                     batch_size=B, logger=True, sync_dist=sync_dist)
            self.log('train/config_gc_norm', gc, on_step=True, on_epoch=False,
                     batch_size=B, logger=True, sync_dist=sync_dist)
            self.log('train/config_param_delta', delta, on_step=True, on_epoch=False,
                     batch_size=B, logger=True, sync_dist=sync_dist)
        else:
            opt.step()
            if sched is not None:
                sched.step()

        self._reset_config_accum()

    def _reset_config_accum(self) -> None:
        self._flow_grad_accum = None
        self._phys_grad_accum = None
        self._config_accum_count = 0

    def validation_step(self, batch, batch_idx):
        losses = self._shared_step(batch, aux_scale=1.0)
        B = self._effective_batch_size(batch)

        sync_dist = getattr(self.trainer, "world_size", 1) > 1
        for key in ('total', 'flow', 'bond_length', 'bond_angle', 'vdw', 'vdw_active', 'vdw_max', 'hbond', 'torsion', 'chiral'):
            prog = key == 'total'
            self.log(f'val/{key}_loss', losses.get(key, 0), on_step=False, on_epoch=True,
                     prog_bar=prog, batch_size=B, sync_dist=sync_dist)

        return losses['total']

    def test_step(self, batch, batch_idx):
        """Test step - same as validation but logged under test/ prefix."""
        losses = self._shared_step(batch, aux_scale=1.0)
        B = self._effective_batch_size(batch)

        sync_dist = getattr(self.trainer, "world_size", 1) > 1
        for key in ('total', 'flow', 'bond_length', 'bond_angle', 'vdw', 'vdw_active', 'vdw_max', 'hbond', 'torsion', 'chiral'):
            self.log(f'test/{key}_loss', losses.get(key, 0), on_step=False, on_epoch=True,
                     batch_size=B, sync_dist=sync_dist)

        return losses['total']

    @torch.no_grad()
    def _euler_integrate(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 40,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Euler integration from x_0 (noise) to x_1 (prediction).

        Coordinate centering (consistent with _prepare_flow_matching):
        - All coordinates centered by global backbone COM
        - This places origin at backbone COM

        Noise initialization (after centering):
        - Backbone atoms: fixed at ground truth (x_0 = x_1)
        - Sidechain atoms: noise centered at per-residue backbone COM, N(residue_backbone_COM, I)
        - Ligand atoms: noise from origin, N(0, I) (= global backbone COM after centering)

        Args:
            batch: Collated batch with ground truth coordinates
            num_steps: Number of Euler integration steps

        Returns:
            Dict with final predicted coordinates and ground truth
        """
        device = batch['rec_atom_feats'].device
        B = batch.get('batch_size', batch['rec_atom_coords_gt'].shape[0])

        # Ground truth coordinates (x_1)
        rec_x_1 = batch['rec_atom_coords_gt'].clone()
        lig_x_1 = batch['lig_atom_coords_gt'].clone()

        # Backbone mask and residue indices
        rec_backbone_mask = batch.get('rec_backbone_mask')
        rec_cb_mask = batch.get('rec_cb_mask')
        rec_uid_vec = batch['rec_uid_vec']
        if rec_uid_vec.shape[0] == 1 and B > 1:
            rec_uid_vec = rec_uid_vec.expand(B, -1)

        if rec_backbone_mask is not None:
            rec_backbone_mask_3d = rec_backbone_mask.unsqueeze(-1)  # [B, N_rec, 1]
            rec_cb_mask_3d = rec_cb_mask.unsqueeze(-1) if rec_cb_mask is not None else torch.zeros_like(rec_backbone_mask_3d)

            # Global backbone COM for centering; missing atoms hold 0-placeholders.
            com_mask = rec_backbone_mask
            rec_missing_mask = batch.get('rec_missing_mask')
            if rec_missing_mask is not None:
                com_mask = rec_backbone_mask * (~rec_missing_mask.bool())
            backbone_coords_sum = (com_mask.unsqueeze(-1) * rec_x_1).sum(dim=1, keepdim=True)  # [B, 1, 3]
            backbone_count = com_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # [B, 1, 1]
            global_backbone_com = backbone_coords_sum / backbone_count  # [B, 1, 3]

            # Center all coordinates by global backbone COM
            rec_x_1 = rec_x_1 - global_backbone_com
            lig_x_1 = lig_x_1 - global_backbone_com

            # Match _prepare_flow_matching: sidechain noise is centered at ideal CB
            # when backbone atom indices are available, else per-residue backbone COM.
            bai = batch.get('rec_backbone_atom_indices')
            if bai is not None:
                n_idx = bai['n_idx'].to(device)
                ca_idx = bai['ca_idx'].to(device)
                c_idx = bai['c_idx'].to(device)
                is_gly = bai['is_gly'].to(device)

                n_coords = rec_x_1[:, n_idx]
                ca_coords = rec_x_1[:, ca_idx]
                c_coords = rec_x_1[:, c_idx]

                cb_ideal = place_cb_ideal(n_coords, ca_coords, c_coords)
                gly_mask = is_gly.unsqueeze(0).unsqueeze(-1)
                atom_noise_center_by_res = torch.where(gly_mask, ca_coords, cb_ideal)
                atom_noise_center = torch.gather(
                    atom_noise_center_by_res, 1,
                    rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                )
            else:
                n_residues = rec_uid_vec.max().item() + 1
                weighted_coords = rec_backbone_mask_3d * rec_x_1

                residue_backbone_sum = torch.zeros(B, n_residues, 3, device=device, dtype=rec_x_1.dtype)
                residue_backbone_count = torch.zeros(B, n_residues, 1, device=device, dtype=rec_x_1.dtype)

                uid_expanded = rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                uid_count = rec_uid_vec.unsqueeze(-1)

                residue_backbone_sum.scatter_add_(1, uid_expanded, weighted_coords)
                residue_backbone_count.scatter_add_(1, uid_count, rec_backbone_mask_3d)

                residue_backbone_count = residue_backbone_count.clamp(min=1.0)
                residue_backbone_com = residue_backbone_sum / residue_backbone_count
                atom_noise_center = torch.gather(
                    residue_backbone_com, 1,
                    rec_uid_vec.unsqueeze(-1).expand(-1, -1, 3)
                )

            sc_sigma = self.config.model.sidechain_prior_sigma
            lig_sigma = self.config.model.ligand_prior_sigma
            rec_noise = torch.randn(rec_x_1.shape, device=device, dtype=rec_x_1.dtype) * sc_sigma + atom_noise_center

            # Backbone + CB stay at GT, rest from centered noise.
            fixed_mask_3d = torch.clamp(rec_backbone_mask_3d + rec_cb_mask_3d, max=1.0)
            rec_x_0 = fixed_mask_3d * rec_x_1 + (1 - fixed_mask_3d) * rec_noise
        else:
            sc_sigma = self.config.model.sidechain_prior_sigma
            lig_sigma = self.config.model.ligand_prior_sigma
            # No backbone mask: center by receptor COM and noise from origin
            global_com = rec_x_1.mean(dim=1, keepdim=True)
            rec_x_1 = rec_x_1 - global_com
            lig_x_1 = lig_x_1 - global_com
            rec_x_0 = torch.randn(rec_x_1.shape, device=device, dtype=rec_x_1.dtype) * sc_sigma
            rec_backbone_mask_3d = None
            fixed_mask_3d = None

        # Ligand noise: N(backbone_COM, lig_sigma^2) (origin after centering)
        lig_x_0 = torch.randn(lig_x_1.shape, device=device, dtype=lig_x_1.dtype) * lig_sigma

        # Current coordinates (start from x_0)
        rec_coords = rec_x_0.clone()
        lig_coords = lig_x_0.clone()

        # Time schedule: 0 -> 1
        sampler = getattr(self.config.training, 'timestep_sampler', 'logit_normal')
        if sampler == 'uniform':
            timesteps = torch.linspace(0.0, 1.0, num_steps + 1).to(device)
        else:
            t = 1.0 - torch.logspace(-2, 0, num_steps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            timesteps = t.clamp(min=0.0, max=1.0).to(device)

        # Pre-compute chemistry features once
        q_chem, p_chem = self.model.forward_chemistry(
            rec_atom_feats=batch['rec_atom_feats'][0:1],
            rec_atom_ref_coords=batch['rec_atom_ref_coords'][0:1],
            rec_constraint_matrix=batch['rec_constraint_matrix'][0:1],
            rec_bond_feats=batch['rec_bond_feats'][0:1],
            rec_uid_vec=batch['rec_uid_vec'][0:1],
            esm_embeddings=batch['esm_embeddings'][0:1],
            lig_atom_feats=batch['lig_atom_feats'][0:1],
            lig_atom_ref_coords=batch['lig_atom_ref_coords'][0:1],
            lig_constraint_matrix=batch['lig_constraint_matrix'][0:1],
            lig_bond_feats=batch['lig_bond_feats'][0:1],
            lig_uid_vec=batch['lig_uid_vec'][0:1],
            unimol_embeddings=batch['unimol_embeddings'][0:1],
        )

        # Euler integration
        for i in range(num_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            # Create batch with current coordinates
            input_batch = {
                **batch,
                'rec_atom_coords': rec_coords,
                'lig_atom_coords': lig_coords,
            }

            # Get flow prediction (flow trunk only)
            t_batch = t.expand(B)
            output = self.forward(input_batch, t_batch, q_chem=q_chem, p_chem=p_chem)

            # Extract flow predictions
            rec_flow = output['rec_flow']
            lig_flow = output['lig_flow']

            # Euler step
            rec_coords = rec_coords + dt * rec_flow
            lig_coords = lig_coords + dt * lig_flow

            # Keep training-fixed atoms fixed during integration.
            if fixed_mask_3d is not None:
                rec_coords = fixed_mask_3d * rec_x_1 + (1 - fixed_mask_3d) * rec_coords

        return {
            'rec_pred': rec_coords,
            'lig_pred': lig_coords,
            'rec_gt': rec_x_1,
            'lig_gt': lig_x_1,
            'rec_backbone_mask': rec_backbone_mask,
        }

    def _compute_rmsd(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute RMSD between predicted and ground truth coordinates.

        Args:
            pred: [B, N, 3] predicted coordinates
            gt: [B, N, 3] ground truth coordinates
            mask: [B, N] optional mask (1=include, 0=exclude)

        Returns:
            [B] RMSD per sample
        """
        diff = pred - gt  # [B, N, 3]
        sq_dist = (diff ** 2).sum(dim=-1)  # [B, N]

        if mask is not None:
            # Masked mean
            mask = mask.float()
            msd = (sq_dist * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            msd = sq_dist.mean(dim=-1)

        return torch.sqrt(msd)

    def _get_parameter_groups(self, log_adam_2d: bool = False):
        """
        Separate parameters for Muon vs AdamW optimization.

        Muon: 2D hidden weight matrices (attention, MLP weights)
        AdamW: Embeddings, output heads, biases, 1D params
        log_adam_2d: log (rank zero) any ndim>=2 params that ended up on AdamW.
        """
        muon_params = []
        adam_params = []
        adam_2d = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # AdamW for biases and 1D params. Matching on the '.bias' suffix (not the
            # substring) keeps 2D projections such as pair_to_bias / bias_proj /
            # dist_bias on Muon. Resuming a checkpoint saved before this change
            # moves those params between groups, so its optimizer state no longer
            # lines up: set training.reset_optimizer_state.
            is_bias = name.endswith('.bias') or param.ndim < 2

            if is_bias:
                adam_params.append(param)
                if param.ndim >= 2:
                    adam_2d.append(name)
            else:
                muon_params.append(param)

        if log_adam_2d and (self._trainer is None or self.trainer.is_global_zero):
            logging.getLogger(__name__).info("AdamW params with ndim>=2: %s", adam_2d)
        return muon_params, adam_params

    def on_load_checkpoint(self, checkpoint):
        """Reset optimizer moments when training.reset_optimizer_state is set.

        Lightning raises if optimizer_states/lr_schedulers are absent on a fit
        resume, so instead of dropping them the per-param state is cleared and
        each group's param indices are rebuilt for the current grouping. Group
        hyperparameters (incl. the scheduled lr) and the LR scheduler state are
        kept, so the LR curve continues where the checkpoint left off.
        """
        if not self.config.training.reset_optimizer_state:
            return
        if getattr(self.config.optimizer, 'optimizer_type', 'muon').lower() == 'adamw':
            sizes = [sum(1 for _ in self.model.parameters())]
        else:
            sizes = [len(g) for g in self._get_parameter_groups() if g]
        for opt_state in checkpoint.get('optimizer_states', []):
            opt_state['state'] = {}
            start = 0
            for group, n in zip(opt_state['param_groups'], sizes, strict=True):
                group['params'] = list(range(start, start + n))
                start += n
        print(f'reset_optimizer_state: cleared optimizer state, param groups rebuilt with sizes {sizes}')

    def on_train_start(self):
        """Re-aim the linear LR decay at the configured max_steps after a resume.

        Lightning restores the scheduler state_dict from the checkpoint, including
        LinearLR.total_iters, so a shortened training.max_steps in the YAML would
        otherwise leave the LR decaying towards the old endpoint. LinearLR's
        recursive update then continues from the current LR and reaches
        end_factor at the new total, i.e. a straight line from here to the end.
        """
        sched = self.lr_schedulers()
        schedulers = getattr(sched, '_schedulers', None)
        if not schedulers or len(schedulers) < 2:
            return
        decay = schedulers[-1]
        target = max(1, self.config.training.max_steps - self.config.scheduler.warmup_steps)
        if getattr(decay, 'total_iters', target) != target:
            print(f'LR decay re-aimed: total_iters {decay.total_iters} -> {target} '
                  f'(global_step {self.global_step}, max_steps {self.config.training.max_steps})')
            decay.total_iters = target

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Cross-rank parameter-consistency guard (DDP never re-broadcasts weights).

        Every log interval, all-gather a cheap fingerprint of the weights and log
        the largest deviation from rank 0 as train/ddp_param_drift. With manual
        optimization + ConFIG each rank applies its own merged gradient; if the
        two backward passes were not both all-reduced the replicas would drift
        apart silently. Expected value: 0.0 (bit-identical replicas).
        """
        if getattr(self.trainer, 'world_size', 1) <= 1 or self.global_step == 0:
            return
        if self.global_step % self.config.logging.log_every_n_steps != 0:
            return
        if getattr(self, '_drift_logged_step', -1) == self.global_step:
            return
        self._drift_logged_step = self.global_step
        with torch.no_grad():
            probe = torch.stack([p.detach().float().flatten()[:4096].sum() for p in self.model.parameters()])
            gathered = self.all_gather(probe)                       # [world, n_params]
            drift = (gathered - gathered[0]).abs().max()
        self.log('train/ddp_param_drift', drift, on_step=True, on_epoch=False, logger=True, rank_zero_only=True)

    def configure_optimizers(self):
        opt_cfg = self.config.optimizer
        sched_cfg = self.config.scheduler
        train_cfg = self.config.training

        optimizer_type = getattr(opt_cfg, 'optimizer_type', 'muon').lower()

        if optimizer_type == 'adamw':
            # Pure AdamW for all parameters
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.adam_lr,
                betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                eps=opt_cfg.adam_eps,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            # Muon or NorMuon with AdamW for embeddings/biases
            muon_params, adam_params = self._get_parameter_groups(log_adam_2d=True)

            param_groups = []

            if muon_params:
                param_groups.append(dict(
                    params=muon_params,
                    use_muon=True,
                    lr=opt_cfg.muon_lr,
                    momentum=opt_cfg.muon_momentum,
                    weight_decay=opt_cfg.muon_weight_decay,
                ))

            if adam_params:
                param_groups.append(dict(
                    params=adam_params,
                    use_muon=False,
                    lr=opt_cfg.adam_lr,
                    betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                    eps=opt_cfg.adam_eps,
                    weight_decay=opt_cfg.adam_weight_decay,
                ))

            if optimizer_type == 'normuon':
                #optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
                raise NotImplementedError("NorMuon not implemented")
            else:
                # Default to Muon: distributed variant for multi-GPU, single-device otherwise
                world_size = getattr(self.trainer, "world_size", 1)
                if world_size > 1:
                    optimizer = MuonWithAuxAdam(param_groups)
                else:
                    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

            print(f"Using {optimizer_type} optimizer: ")
            print(f"  Muon params: {sum(p.numel() for p in muon_params)}")
            print(f"  Adam params: {sum(p.numel() for p in adam_params)}")
            print(f"  Total params: {sum(p.numel() for p in muon_params) + sum(p.numel() for p in adam_params)}")
            print(f"  Model params: {sum(p.numel() for p in self.model.parameters())}")

        # Linear warmup + linear decay scheduler
        warmup_steps = sched_cfg.warmup_steps
        total_steps = train_cfg.max_steps
        decay_steps = max(1, total_steps - warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=sched_cfg.start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=sched_cfg.end_factor,
            total_iters=decay_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        print(f"LR Schedule: {warmup_steps} warmup + {decay_steps} linear decay")
        print(f"  start_factor={sched_cfg.start_factor}, end_factor={sched_cfg.end_factor}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

class RMSDEvaluationCallback(Callback):
    """
    Callback to evaluate RMSD on fixed sample sets using Euler integration.

    Runs 40-step Euler integration on first 30 sorted systems from each split
    (train, val, test) and logs RMSD metrics.
    """

    def __init__(
        self,
        train_systems: List[str],
        val_systems: List[str],
        test_systems: Optional[List[str]] = None,
        pb_systems: Optional[List[str]] = None,
        data_dir: str = './pdbbind_data',
        test_data_dir: Optional[str] = None,
        pb_data_dir: Optional[str] = None,
        num_samples: int = 30,
        num_steps: int = 40,
        eval_every_n_val: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir or data_dir
        self.pb_data_dir = pb_data_dir
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.eval_every_n_val = eval_every_n_val
        self.val_count = 0

        if test_systems is None and test_data_dir is not None:
            test_path = Path(self.test_data_dir)
            if test_path.exists():
                test_systems = [
                    d.name for d in test_path.iterdir()
                    if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
                ]

        if pb_systems is None and pb_data_dir is not None:
            pb_path = Path(pb_data_dir)
            if pb_path.exists():
                pb_systems = [
                    d.name for d in pb_path.iterdir()
                    if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
                ]

        # Select first N sorted systems for consistent evaluation
        self.train_systems = sorted(train_systems)[:num_samples]
        self.val_systems = sorted(val_systems)[:num_samples]
        self.test_systems = sorted(test_systems)[:num_samples] if test_systems else []
        self.pb_systems = sorted(pb_systems)[:num_samples] if pb_systems else []

        print(f"RMSD Evaluation Callback initialized:")
        print(f"  Train samples: {len(self.train_systems)}")
        print(f"  Val samples: {len(self.val_systems)}")
        print(f"  Test samples: {len(self.test_systems)} (dir={self.test_data_dir})")
        print(f"  PB samples: {len(self.pb_systems)} (dir={self.pb_data_dir})")
        print(f"  Euler steps: {num_steps}")

    def _create_single_sample_batch(self, system_dir: str, device: torch.device) -> Dict[str, torch.Tensor]:
        """Load a single system and create a batch of size 1."""
        from pathlib import Path

        system_path = Path(system_dir)

        ligand_feats = torch.load(system_path / 'ligand_features.pt', weights_only=False)
        pocket_feats = torch.load(system_path / 'pocket_features.pt', weights_only=False)
        esm_emb = torch.load(system_path / 'esm3_embeddings.pt', weights_only=False)
        unimol_emb = torch.load(system_path / 'unimol_embeddings.pt', weights_only=False)
        validate_cached_unimol(system_path, ligand_feats, unimol_emb)

        # Backbone mask
        rec_backbone_mask = pocket_feats.get('backbone_mask', None)
        if rec_backbone_mask is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_backbone_mask = torch.zeros(n_atoms, dtype=torch.float32)

        # CB mask (sidechain root; fixed during generation alongside backbone)
        rec_cb_mask = pocket_feats.get('cb_mask', None)
        if rec_cb_mask is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_cb_mask = torch.zeros(n_atoms, dtype=torch.float32)
        elif isinstance(rec_cb_mask, np.ndarray):
            rec_cb_mask = torch.from_numpy(rec_cb_mask).float()

        # Missing mask
        rec_missing_mask = pocket_feats.get('missing_mask', None)
        if rec_missing_mask is not None:
            if isinstance(rec_missing_mask, np.ndarray):
                rec_missing_mask = torch.from_numpy(rec_missing_mask).bool()
        else:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_missing_mask = torch.zeros(n_atoms, dtype=torch.bool)

        # Coordinates
        rec_coords_gt = pocket_feats['coords']
        lig_coords_gt = ligand_feats['coords']

        # Center by backbone COM
        valid_backbone_mask = rec_backbone_mask * (~rec_missing_mask).float()
        if valid_backbone_mask.sum() > 0:
            backbone_com = (rec_coords_gt * valid_backbone_mask.unsqueeze(-1)).sum(dim=0) / valid_backbone_mask.sum()
        else:
            backbone_com = rec_coords_gt.mean(dim=0)

        rec_coords_gt_centered = rec_coords_gt - backbone_com
        lig_coords_gt_centered = lig_coords_gt - backbone_com

        N_rec = pocket_feats['atom_features'].shape[0]
        N_lig = ligand_feats['atom_features'].shape[0]
        N_rec_tokens = esm_emb.shape[0]

        # Build batch with B=1
        batch = {
            'rec_atom_feats': pocket_feats['atom_features'].unsqueeze(0),
            'rec_atom_coords_gt': rec_coords_gt_centered.unsqueeze(0),
            'rec_atom_ref_coords': pocket_feats['ref_coords'].unsqueeze(0),
            'rec_constraint_matrix': pocket_feats.get('constraint_matrix', None),
            'rec_bond_feats': pocket_feats['bond_features'].unsqueeze(0),
            'rec_uid_vec': pocket_feats['residue_uid_vec'].unsqueeze(0),
            'rec_backbone_mask': rec_backbone_mask.unsqueeze(0),
            'rec_cb_mask': rec_cb_mask.unsqueeze(0),
            'rec_missing_mask': rec_missing_mask.unsqueeze(0),
            'esm_embeddings': esm_emb.unsqueeze(0),
            'rec_atom_mask': torch.ones(1, N_rec, dtype=torch.bool),
            'rec_token_mask': torch.ones(1, N_rec_tokens, dtype=torch.bool),

            'lig_atom_feats': ligand_feats['atom_features'].unsqueeze(0),
            'lig_atom_coords_gt': lig_coords_gt_centered.unsqueeze(0),
            'lig_atom_ref_coords': ligand_feats['ref_coords'].unsqueeze(0),
            'lig_constraint_matrix': ligand_feats.get('constraint_matrix', None),
            'lig_bond_feats': ligand_feats['bond_features'].unsqueeze(0),
            'lig_uid_vec': ligand_feats['ligand_uid_vec'].unsqueeze(0),
            'unimol_embeddings': unimol_emb.unsqueeze(0),
            'lig_atom_mask': torch.ones(1, N_lig, dtype=torch.bool),
        }

        # Ensure constraint matrices exist and have shape [1, N, N, 1]
        rec_constraint_matrix = batch['rec_constraint_matrix']
        if rec_constraint_matrix is None:
            rec_constraint_matrix = torch.zeros(N_rec, N_rec, 1, dtype=torch.float32)
        elif isinstance(rec_constraint_matrix, np.ndarray):
            rec_constraint_matrix = torch.from_numpy(rec_constraint_matrix).float()
        if rec_constraint_matrix.ndim == 2:
            rec_constraint_matrix = rec_constraint_matrix.unsqueeze(-1)
        batch['rec_constraint_matrix'] = rec_constraint_matrix.unsqueeze(0)

        lig_constraint_matrix = batch['lig_constraint_matrix']
        if lig_constraint_matrix is None:
            lig_constraint_matrix = torch.zeros(N_lig, N_lig, 1, dtype=torch.float32)
        elif isinstance(lig_constraint_matrix, np.ndarray):
            lig_constraint_matrix = torch.from_numpy(lig_constraint_matrix).float()
        if lig_constraint_matrix.ndim == 2:
            lig_constraint_matrix = lig_constraint_matrix.unsqueeze(-1)
        batch['lig_constraint_matrix'] = lig_constraint_matrix.unsqueeze(0)

        # Per-residue backbone atom indices for CB-ideal noise centering
        # (must match training-time noise distribution in _prepare_flow_matching).
        bai = pocket_feats.get('backbone_atom_indices', None)
        if bai is not None:
            batch['rec_backbone_atom_indices'] = bai

        # Move to device (handle nested dict for backbone_atom_indices).
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def _evaluate_split(
        self,
        pl_module: 'FlexTrainer',
        system_ids: List[str],
        data_dir: str,
        split_name: str,
    ) -> Dict[str, float]:
        """Evaluate RMSD on a split."""
        from pathlib import Path
        from tqdm import tqdm

        if len(system_ids) == 0:
            return {}

        device = pl_module.device
        rec_rmsds = []
        lig_rmsds = []
        sidechain_rmsds = []
        error_count = 0

        for system_id in tqdm(system_ids, desc=f"RMSD {split_name}", leave=False):
            system_path = Path(data_dir) / system_id

            if not system_path.exists():
                error_count += 1
                continue

            try:
                batch = self._create_single_sample_batch(str(system_path), device)

                # Run Euler integration
                result = pl_module._euler_integrate(batch, num_steps=self.num_steps)

                # Compute RMSDs
                rec_mask = batch['rec_atom_mask']
                lig_mask = batch['lig_atom_mask']

                # Overall receptor RMSD (sidechain only, since backbone is fixed)
                rec_backbone_mask = result.get('rec_backbone_mask')
                if rec_backbone_mask is not None:
                    sidechain_mask = (~rec_backbone_mask.bool()).float() * rec_mask.float()
                    if sidechain_mask.sum() > 0:
                        sc_rmsd = pl_module._compute_rmsd(
                            result['rec_pred'], result['rec_gt'], sidechain_mask
                        )
                        sidechain_rmsds.append(sc_rmsd.item())

                # Full receptor RMSD
                rec_rmsd = pl_module._compute_rmsd(result['rec_pred'], result['rec_gt'], rec_mask.float())
                rec_rmsds.append(rec_rmsd.item())

                # Ligand RMSD
                lig_rmsd = pl_module._compute_rmsd(result['lig_pred'], result['lig_gt'], lig_mask.float())
                lig_rmsds.append(lig_rmsd.item())

            except Exception as exc:
                error_count += 1
                logging.getLogger(__name__).warning(
                    'RMSD evaluation failed: split=%s system=%s: %s',
                    split_name, system_id, exc, exc_info=True)

            # Clean up VRAM
            try:
                del batch
            except NameError:
                pass
            try:
                del result
            except NameError:
                pass
            torch.cuda.empty_cache()

        metrics = {}
        if rec_rmsds:
            metrics[f'{split_name}/rmsd_receptor'] = np.mean(rec_rmsds)
        if lig_rmsds:
            metrics[f'{split_name}/rmsd_ligand'] = np.mean(lig_rmsds)
        if sidechain_rmsds:
            metrics[f'{split_name}/rmsd_sidechain'] = np.mean(sidechain_rmsds)
        if error_count > 0:
            metrics[f'{split_name}/rmsd_errors'] = error_count

        return metrics

    def _evaluate_test_losses(
        self,
        pl_module: 'FlexTrainer',
        test_loader,
        prefix: str = 'test',
    ) -> Dict[str, float]:
        """Mean training losses over a held-out loader, logged as <prefix>/<key>_loss."""
        from tqdm import tqdm

        if test_loader is None:
            return {}

        device = pl_module.device
        loss_keys = ('total', 'flow', 'bond_length', 'bond_angle', 'vdw', 'hbond', 'torsion', 'chiral')
        loss_accum = {k: [] for k in loss_keys}
        error_count = 0

        iterator = iter(test_loader)
        for batch_idx in tqdm(range(len(test_loader)), desc=f"{prefix} losses", leave=False):
            batch = None
            try:
                # Include loading in the guard: bad feature files should be
                # identified and counted, not silently lost or fatal to this callback.
                batch = next(iterator)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.no_grad():
                    losses = pl_module._shared_step(batch)
                for k in loss_keys:
                    val = losses.get(k, 0)
                    loss_accum[k].append(val.item() if isinstance(val, torch.Tensor) else val)
            except StopIteration:
                break
            except Exception as exc:
                error_count += 1
                system_ids = batch.get('system_ids', ['unknown']) if batch is not None else ['see exception']
                logging.getLogger(__name__).warning(
                    'Loss evaluation failed: split=%s batch=%s systems=%s: %s',
                    prefix, batch_idx, system_ids, exc, exc_info=True)

            del batch
            torch.cuda.empty_cache()

        metrics = {}
        if loss_accum['total']:
            for k in loss_keys:
                metrics[f'{prefix}/{k}_loss'] = np.mean(loss_accum[k])
        if error_count > 0:
            metrics[f'{prefix}/loss_errors'] = error_count

        return metrics

    def _get_ema_callback(self, trainer) -> Optional['EMACallback']:
        """Find the EMACallback from trainer's callbacks."""
        for callback in trainer.callbacks:
            if isinstance(callback, EMACallback):
                return callback
        return None

    def on_validation_end(self, trainer, pl_module):
        """Run RMSD evaluation after validation."""
        # Skip sanity check (step 0) - no point evaluating untrained model
        if trainer.global_step == 0:
            return

        self.val_count += 1

        # Only evaluate every N validations
        if self.val_count % self.eval_every_n_val != 0:
            return

        pl_module.eval()

        # Apply EMA weights if available (EMACallback has already restored original weights
        # by the time on_validation_end runs, so we need to re-apply EMA for evaluation)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None and ema_callback.ema is not None:
            ema_callback.ema.apply(pl_module.model)

        try:
            all_metrics = {}

            # Held-out losses: test split and the PoseBusters benchmark
            all_metrics.update(self._evaluate_test_losses(
                pl_module, getattr(pl_module, 'test_dataloader_ref', None), 'test'))
            all_metrics.update(self._evaluate_test_losses(
                pl_module, getattr(pl_module, 'pb_dataloader_ref', None), 'pb'))

            # Evaluate train split RMSD
            train_metrics = self._evaluate_split(
                pl_module, self.train_systems, self.data_dir, 'train_rmsd'
            )
            all_metrics.update(train_metrics)

            # Evaluate val split RMSD
            val_metrics = self._evaluate_split(
                pl_module, self.val_systems, self.data_dir, 'val_rmsd'
            )
            all_metrics.update(val_metrics)

            # Evaluate test split RMSD (can use separate test_data_dir)
            if self.test_systems:
                test_metrics = self._evaluate_split(
                    pl_module, self.test_systems, self.test_data_dir, 'test_rmsd'
                )
                all_metrics.update(test_metrics)

            # Evaluate PoseBusters benchmark RMSD (separate dir)
            if self.pb_systems and self.pb_data_dir is not None:
                pb_metrics = self._evaluate_split(
                    pl_module, self.pb_systems, self.pb_data_dir, 'pb_rmsd'
                )
                all_metrics.update(pb_metrics)

            # Log metrics directly to logger (can't use self.log in on_validation_end)
            if trainer.logger is not None:
                trainer.logger.log_metrics(all_metrics, step=trainer.global_step)

        finally:
            # Restore original weights after evaluation
            if ema_callback is not None and ema_callback.ema is not None:
                ema_callback.ema.restore(pl_module.model)


class WandBWatchCallback(Callback):
    """Callback to set up wandb.watch for gradient/parameter logging."""

    def __init__(self, log: str = 'gradients', log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        
        # 1. Normalize input
        if log is not None:
            log = str(log).strip().lower()
            
        # 2. Strict validation
        if log not in {'gradients', 'parameters', 'all'}:
            log = None
            
        # RENAME variable to avoid collision with self.log() method
        self.watch_mode = log

    def on_fit_start(self, trainer, pl_module):
        # 3. Use the renamed variable
        if self.watch_mode is None:
            return

        if trainer.logger is not None and hasattr(trainer.logger, 'experiment'):
            try:
                run = trainer.logger.experiment
                run.watch(
                    pl_module.model,
                    log=self.watch_mode,  # Pass the renamed variable
                    log_freq=self.log_freq,
                )
            except ValueError as e:
                print(f"Warning: Failed to initialize wandb.watch with log='{self.watch_mode}'. Error: {e}")
            except Exception as e:
                print(f"Warning: An unexpected error occurred in WandBWatchCallback: {e}")


def get_trainer(config: FullConfig) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with appropriate callbacks.

    Args:
        config: FullConfig instance with all training parameters

    Returns:
        pl.Trainer instance
    """
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
    from pytorch_lightning.loggers import WandbLogger

    train_cfg = config.training
    log_cfg = config.logging
    ema_cfg = config.ema

    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{log_cfg.checkpoint_dir}/{log_cfg.wandb_run_name or "flex"}',
        filename='{step}-{val/total_loss:.4f}',
        save_top_k=log_cfg.save_top_k,
        monitor='val/total_loss',
        mode='min',
        save_last=True,
        every_n_train_steps=log_cfg.checkpoint_every_n_steps,
    )
    callbacks.append(checkpoint_callback)

    # LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Model summary with full depth
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # EMA callback
    if ema_cfg.enabled:
        ema_callback = EMACallback(decay=ema_cfg.decay)
        callbacks.append(ema_callback)

    # Logger
    logger = None
    if log_cfg.use_wandb:
        logger = WandbLogger(
            project=log_cfg.wandb_project,
            name=log_cfg.wandb_run_name,
            entity=log_cfg.wandb_entity,
            save_dir='logs',
            config=config.to_dict(),
            log_model=False,
        )

        # Add wandb.watch callback for gradient/parameter logging
        if getattr(log_cfg, 'watch_model', False) and getattr(log_cfg, 'watch_log', None):
            watch_callback = WandBWatchCallback(
                log=log_cfg.watch_log,
                log_freq=getattr(log_cfg, 'watch_log_freq', 1000),
            )
            callbacks.append(watch_callback)

    # With manual optimization (use_config), Lightning forbids gradient_clip_val
    # and accumulate_grad_batches kwargs; we apply both manually in training_step.
    trainer_kwargs = dict(
        max_steps=train_cfg.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=train_cfg.devices,
        precision=train_cfg.precision,
        val_check_interval=train_cfg.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,  # We handle custom log interval in training_step
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,  # Using custom ModelSummary callback
    )
    if not getattr(train_cfg, 'use_config', False):
        trainer_kwargs['gradient_clip_val'] = train_cfg.gradient_clip_val
        trainer_kwargs['accumulate_grad_batches'] = train_cfg.accumulate_grad_batches

    trainer = pl.Trainer(**trainer_kwargs)

    return trainer
