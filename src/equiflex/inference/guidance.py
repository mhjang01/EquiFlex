"""
Inference-time physics guidance for the Flex flow matching sampler.

Implements g^cov-A from "On the Guidance of Flow Matching" (Feng et al., 2025).
Under the affine path assumption (x_t = t*x_1 + (1-t)*x_0), the guidance VF is
the gradient of the energy J at the clean estimate, preconditioned by the
covariance of p(x_1|x_t). cov-A approximates that covariance with a scalar
schedule lambda_t:

    x_1_hat = x_t + (1-t) * v_pred                 # 1-step Euler clean estimate
    g       = -scale * lambda_t * grad_{x1} J(x_1_hat)
    v_corr  = v_pred + g                           # add directly to the velocity

Key properties (vs. the old Tweedie-divide formulation):
  - guidance is added to the velocity field, NOT divided by (1-t);
  - the schedule lambda_t DECAYS toward t=1, where p(x_1|x_t) localizes
    (Σ_{1|t} -> 0) so guidance should weaken — the opposite of the implicit
    step_size/(1-t) schedule that grew toward the data;
  - the gradient is taken at x_1_hat only — no backprop through the flow model
    (cov-A, not cov-G), so no extra NFE.

The corrector reuses BondLengthLoss / BondAngleLoss / VdWClashLoss / TorsionLoss
from model.loss so coefficients match training-time semantics. A gradient-free
Sim-MC alternative (g^sim-MC flavour) is provided via sim_mc_correct.
"""
import math
import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from equiflex.model.loss import (
    BondLengthLoss,
    BondAngleLoss,
    VdWClashLoss,
    VdWActiveClashLoss,
    VdWMaxPenetrationLoss,
    TorsionLoss,
    _vdw_radius_from_features,
)


def guidance_schedule(name: str, t: torch.Tensor) -> torch.Tensor:
    """Decaying-in-t cov-A schedules. t=1 is data; weight -> 0 there because
    p(x_1|x_t) localizes and its covariance vanishes."""
    if name == 'const':
        return torch.ones_like(t)
    if name == 'linear_decay':
        return (1.0 - t).clamp(min=0.0)
    if name == 'cosine_decay':
        return torch.cos(t * math.pi / 2)
    if name == 'exp_decay':
        return (torch.exp(-t) - math.exp(-1)) / (1.0 - math.exp(-1))
    raise ValueError(f"Unknown guidance schedule: {name}")


@dataclass
class GuidanceConfig:
    """cov-A physics guidance hyperparameters.

    scale (lambda prefactor) and schedule together set the guidance strength.
    Because Flex auxiliary losses are mean-normalised (per num_bonds / num_atoms),
    per-atom gradient magnitude is small; scale typically needs to be O(1-1e2).
    Tune scale per system; verified working range depends on loss normalisation.
    """
    enabled: bool = False
    t_start: float = 0.3              # gate: no guidance below this t
    scale: float = 1.0                # cov-A guidance scale lambda
    schedule: str = 'linear_decay'    # decaying schedule in t (see guidance_schedule)
    grad_clip: float = 0.0            # 0 = off; per-atom max ||g||
    bond_coef: float = 1.0
    angle_coef: float = 1.0
    vdw_coef: float = 5.0             # mean squared penetration — force spread over all clashing pairs
    vdw_active_coef: float = 0.0      # soft active-clash count — clears many shallow clashes, saturates on deep
    vdw_max_coef: float = 0.0         # soft-max worst-pair penetration — concentrates force, CAN dent; keep 0 unless needed
    vdw_active_scale: float = 0.10    # Å scale where active-count saturates
    vdw_max_alpha: float = 20.0       # logsumexp sharpness for soft-max penetration
    torsion_coef: float = 1.0         # planarity loss for aromatic-ring sp2 torsions
    bond_buffer: float = 0.05         # absolute slack (Å) for bond flat-bottom
    angle_buffer: float = 0.05        # absolute slack (Å) for 1-3 flat-bottom
    vdw_buffer: float = 0.225
    vdw_cutoff: float = 6.0
    one_minus_t_min: float = 0.05     # sim_mc v_corr denominator floor only
    sim_mc_guidance: bool = False     # gradient-free candidate sampling instead of cov-A
    sim_mc_n: int = 32                # candidates per sample
    sim_mc_std: float = 0.10          # Cartesian proposal sigma (Å)
    sim_mc_scale: float = 5.0         # inverse-temperature for soft candidate averaging
    sim_mc_clash_count_coef: float = 0.0  # extra count penalty for active vdW clashes
    sim_mc_include_original: bool = True


class TweedieGuidance:
    """Stateful x_1-space corrector. Build once per sampling run, call inside Euler loop."""

    def __init__(
        self,
        config: GuidanceConfig,
        batch: Dict[str, torch.Tensor],
        rec_fixed_mask_3d: torch.Tensor,   # [B, N_rec, 1] backbone + CB
        device: str,
        rec_torsion_indices: torch.Tensor = None,   # [K_rec, 4] sp2 torsion atom indices
        lig_torsion_indices: torch.Tensor = None,   # [K_lig, 4]
    ):
        self.config = config
        self.device = device

        rec_feats = batch['rec_atom_feats']
        lig_feats = batch['lig_atom_feats']
        B = rec_fixed_mask_3d.shape[0]
        N_rec = rec_feats.shape[1]
        N_lig = lig_feats.shape[1]
        N = N_rec + N_lig
        self.N_rec = N_rec
        self.N_lig = N_lig

        if rec_feats.shape[0] == 1 and B > 1:
            rec_feats = rec_feats.expand(B, -1, -1)
            lig_feats = lig_feats.expand(B, -1, -1)
        self.atom_feats = torch.cat([rec_feats, lig_feats], dim=1).to(device)

        rec_mask = batch['rec_atom_mask'].float()
        lig_mask = batch['lig_atom_mask'].float()
        rec_missing = batch.get('rec_missing_mask')
        if rec_missing is not None:
            rec_mask = rec_mask * (~rec_missing.bool()).float()
        self.atom_mask = torch.cat([rec_mask, lig_mask], dim=1).to(device)

        rec_bf = batch['rec_bond_feats']
        lig_bf = batch['lig_bond_feats']
        if rec_bf.shape[0] == 1 and B > 1:
            rec_bf = rec_bf.expand(B, -1, -1, -1)
            lig_bf = lig_bf.expand(B, -1, -1, -1)
        rec_bond_mask = (rec_bf.abs().sum(-1) > 0).float()
        lig_bond_mask = (lig_bf.abs().sum(-1) > 0).float()
        bond_mask = torch.zeros(B, N, N, device=device)
        bond_mask[:, :N_rec, :N_rec] = rec_bond_mask
        bond_mask[:, N_rec:, N_rec:] = lig_bond_mask
        self.bond_mask = bond_mask

        eye = torch.eye(N, device=device).unsqueeze(0)
        bm13 = torch.bmm(bond_mask, bond_mask).clamp(max=1.0)
        bm13 = bm13 * (1 - eye)
        self.angle_mask = bm13 * (1 - bond_mask)

        # 1-4 (dihedral) pairs: reachable in 3 bonds but not 1-2 or 1-3. Excluded
        # from the vdW lower bound — their separation is set by torsion geometry,
        # so a full vdW restraint spuriously pushes them apart (matches training).
        bm14 = torch.bmm(bm13, bond_mask).clamp(max=1.0)
        bm14 = bm14 * (1 - eye)
        self.dihedral_mask = bm14 * (1 - bond_mask) * (1 - self.angle_mask)

        rec_cm = batch['rec_constraint_matrix']
        lig_cm = batch['lig_constraint_matrix']
        if rec_cm.ndim == 4:
            rec_cm = rec_cm.squeeze(-1)
        if lig_cm.ndim == 4:
            lig_cm = lig_cm.squeeze(-1)
        if rec_cm.shape[0] == 1 and B > 1:
            rec_cm = rec_cm.expand(B, -1, -1)
            lig_cm = lig_cm.expand(B, -1, -1)
        cm = torch.zeros(B, N, N, device=device)
        cm[:, :N_rec, :N_rec] = rec_cm
        cm[:, N_rec:, N_rec:] = lig_cm
        self.constraint_matrix = cm

        # Receptor backbone + CB frozen; ligand fully free
        fixed_rec = rec_fixed_mask_3d.squeeze(-1).clamp(max=1.0)
        fixed_lig = torch.zeros(B, N_lig, device=device)
        self.fixed_mask = torch.cat([fixed_rec, fixed_lig], dim=1).clamp(max=1.0)

        # Expand sp2 torsion indices to TorsionLoss's [K_total, 5] format
        # (col 0 = batch index, cols 1-4 = atom indices). Lig atom indices stay
        # in their local frame; TorsionLoss.compute offsets them by n_rec.
        self.rec_torsion_indices = self._expand_torsions(rec_torsion_indices, B, device)
        self.lig_torsion_indices = self._expand_torsions(lig_torsion_indices, B, device)

        # Loss modules — drive via explicit coefficients (no schedule, no aux_scale)
        self.bond_loss = BondLengthLoss(weight=1.0, buffer=config.bond_buffer)
        self.angle_loss = BondAngleLoss(weight=1.0, buffer=config.angle_buffer)
        self.vdw_loss = VdWClashLoss(
            weight=1.0, buffer=config.vdw_buffer, cutoff=config.vdw_cutoff,
        )
        self.vdw_active_loss = VdWActiveClashLoss(
            weight=1.0, buffer=config.vdw_buffer, cutoff=config.vdw_cutoff,
            penetration_scale=config.vdw_active_scale,
        )
        self.vdw_max_loss = VdWMaxPenetrationLoss(
            weight=1.0, buffer=config.vdw_buffer, cutoff=config.vdw_cutoff,
            alpha=config.vdw_max_alpha,
        )
        self.torsion_loss = TorsionLoss(weight=1.0)

    @staticmethod
    def _expand_torsions(idx_4col, B, device):
        """[K, 4] -> [B*K, 5] with batch index in col 0; replicate per sample."""
        if idx_4col is None or idx_4col.numel() == 0:
            return torch.empty((0, 5), dtype=torch.long, device=device)
        K = idx_4col.shape[0]
        idx_4col = idx_4col.to(device).long()
        atom_block = idx_4col.unsqueeze(0).expand(B, K, 4).reshape(B * K, 4)
        batch_col = torch.arange(B, device=device).repeat_interleave(K).unsqueeze(-1)
        return torch.cat([batch_col, atom_block], dim=-1)

    def is_active(self, t: torch.Tensor) -> bool:
        return self.config.enabled and bool(t.mean().item() >= self.config.t_start)

    def _has_torsions(self) -> bool:
        return self.rec_torsion_indices.numel() > 0 or self.lig_torsion_indices.numel() > 0

    def _repeat_for_candidates(self, tensor: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        """Repeat a [B, ...] guidance tensor to match flattened [K*B, ...] candidates."""
        if repeat_factor == 1:
            return tensor
        return tensor.unsqueeze(0).expand(repeat_factor, *tensor.shape).reshape(
            repeat_factor * tensor.shape[0], *tensor.shape[1:]
        )

    def _repeat_torsions_for_candidates(self, idx: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        """Repeat [B*K_tor, 5] torsion indices for flattened [K*B, ...] candidates."""
        if repeat_factor == 1 or idx.numel() == 0:
            return idx
        base_B = self.fixed_mask.shape[0]
        chunks = []
        for k in range(repeat_factor):
            shifted = idx.clone()
            shifted[:, 0] = shifted[:, 0] + k * base_B
            chunks.append(shifted)
        return torch.cat(chunks, dim=0)

    def _context(self, x: torch.Tensor, repeat_factor: int = 1) -> Dict[str, torch.Tensor]:
        """Build loss context for either B original samples or flattened K*B candidates."""
        return {
            'x_1_pred': x,
            'pred_dists': torch.cdist(x, x),
            'bond_mask': self._repeat_for_candidates(self.bond_mask, repeat_factor),
            'angle_mask': self._repeat_for_candidates(self.angle_mask, repeat_factor),
            'dihedral_mask': self._repeat_for_candidates(self.dihedral_mask, repeat_factor),
            'mask': self._repeat_for_candidates(self.atom_mask, repeat_factor),
            'atom_feats': self._repeat_for_candidates(self.atom_feats, repeat_factor),
            'constraint_matrix': self._repeat_for_candidates(self.constraint_matrix, repeat_factor),
            'rec_torsion_indices': self._repeat_torsions_for_candidates(
                self.rec_torsion_indices, repeat_factor
            ),
            'lig_torsion_indices': self._repeat_torsions_for_candidates(
                self.lig_torsion_indices, repeat_factor
            ),
            'n_rec': self.N_rec,
        }

    def _physics_energy(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample weighted sum of the physics terms [B]. Single source of
        truth shared by the gradient (cov-A) and Sim-MC paths."""
        cfg = self.config
        energy = ctx['x_1_pred'].new_zeros(ctx['x_1_pred'].shape[0])
        if cfg.bond_coef > 0:
            energy = energy + cfg.bond_coef * self.bond_loss.compute(ctx)
        if cfg.angle_coef > 0:
            energy = energy + cfg.angle_coef * self.angle_loss.compute(ctx)
        if cfg.vdw_coef > 0:
            energy = energy + cfg.vdw_coef * self.vdw_loss.compute(ctx)
        if cfg.vdw_active_coef > 0:
            energy = energy + cfg.vdw_active_coef * self.vdw_active_loss.compute(ctx)
        if cfg.vdw_max_coef > 0:
            energy = energy + cfg.vdw_max_coef * self.vdw_max_loss.compute(ctx)
        if cfg.torsion_coef > 0 and self._has_torsions():
            energy = energy + cfg.torsion_coef * self.torsion_loss.compute(ctx)
        return energy

    def energy(self, x: torch.Tensor, repeat_factor: int = 1) -> torch.Tensor:
        """Per-sample physics energy J for combined receptor/ligand coordinates."""
        ctx = self._context(x, repeat_factor=repeat_factor)
        energy = self._physics_energy(ctx)
        if self.config.sim_mc_clash_count_coef > 0:
            energy = energy + self.config.sim_mc_clash_count_coef * self._active_clash_count(ctx)
        return energy

    def _active_clash_count(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Count active vdW-violating nonbonded pairs per sample."""
        pred_dists = ctx['pred_dists']
        atom_feats = ctx['atom_feats']
        mask = ctx['mask']
        bond_mask = ctx['bond_mask']
        angle_mask = ctx['angle_mask']
        N = pred_dists.shape[1]
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        pair_mask = pair_mask * (1 - torch.eye(N, device=pred_dists.device))
        pair_mask = pair_mask * (1 - bond_mask) * (1 - angle_mask)
        pair_mask = pair_mask * (pred_dists < self.config.vdw_cutoff).float()
        atom_vdw = _vdw_radius_from_features(atom_feats)
        lower_bounds = (atom_vdw.unsqueeze(-1) + atom_vdw.unsqueeze(-2)) * (1.0 - self.config.vdw_buffer)
        active = (lower_bounds > pred_dists).float() * pair_mask
        triu_mask = torch.triu(torch.ones(N, N, device=pred_dists.device), diagonal=1)
        return (active * triu_mask).sum(dim=(-1, -2))

    def guidance_velocity(
        self,
        rec_x1_hat: torch.Tensor,
        lig_x1_hat: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cov-A guidance VF term g = -scale * lambda_t * grad_{x1} J(x_1_hat).

        Returns (rec_g, lig_g) to be ADDED directly to the predicted velocity.
        Gradient is taken at the clean estimate only (no model backprop); the
        schedule decays toward t=1. Fixed atoms (rec backbone + CB) get zero.
        """
        cfg = self.config
        free = (1.0 - self.fixed_mask).unsqueeze(-1)
        x = torch.cat([rec_x1_hat, lig_x1_hat], dim=1).detach()

        with torch.enable_grad():
            x = x.requires_grad_(True)
            ctx = {
                'x_1_pred': x,
                'pred_dists': torch.cdist(x, x),
                'bond_mask': self.bond_mask,
                'angle_mask': self.angle_mask,
                'dihedral_mask': self.dihedral_mask,
                'mask': self.atom_mask,
                'atom_feats': self.atom_feats,
                'constraint_matrix': self.constraint_matrix,
                'rec_torsion_indices': self.rec_torsion_indices,
                'lig_torsion_indices': self.lig_torsion_indices,
                'n_rec': self.N_rec,
            }
            # Sum over batch: samples are independent through the losses, so the
            # gradient at sample b depends only on J_b (matches per-sample cov-A).
            energy = self._physics_energy(ctx).sum()
            grad = torch.autograd.grad(energy, x)[0]

        lam = guidance_schedule(cfg.schedule, t).view(-1, 1, 1)
        g = -cfg.scale * lam * grad * free

        if cfg.grad_clip > 0:
            norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            g = g * (norm.clamp(max=cfg.grad_clip) / norm)

        return g[:, :self.N_rec], g[:, self.N_rec:]

    @torch.no_grad()
    def sim_mc_correct(
        self,
        rec_x1_hat: torch.Tensor,
        lig_x1_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient-free Cartesian Sim-MC x1 correction via soft selection over
        random proposals. Returns corrected (rec, lig) x_1 (converted to velocity
        by the caller)."""
        cfg = self.config
        x = torch.cat([rec_x1_hat, lig_x1_hat], dim=1).detach()
        B, N, _ = x.shape
        free = (1 - self.fixed_mask).unsqueeze(-1)
        K = max(1, int(cfg.sim_mc_n))

        noise = torch.randn(K, B, N, 3, device=x.device, dtype=x.dtype) * cfg.sim_mc_std
        candidates = x.unsqueeze(0) + noise * free.unsqueeze(0)
        if cfg.sim_mc_include_original:
            candidates[0] = x

        flat = candidates.reshape(K * B, N, 3)
        energy = self.energy(flat, repeat_factor=K).view(K, B)
        logits = -cfg.sim_mc_scale * (energy - energy.min(dim=0, keepdim=True).values)
        weights = torch.softmax(logits, dim=0)
        x_corr = (weights[..., None, None] * candidates).sum(dim=0)
        x_corr = self.fixed_mask.unsqueeze(-1) * x + free * x_corr
        return x_corr[:, :self.N_rec], x_corr[:, self.N_rec:]
