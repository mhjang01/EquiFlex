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
from equiflex.model.loss so coefficients match training-time semantics. A gradient-free
Sim-MC alternative (g^sim-MC flavour) is provided via sim_mc_correct.
"""
import logging
import math
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from torch.profiler import record_function

from equiflex.model.loss import (
    BondLengthLoss,
    BondAngleLoss,
    VdWClashLoss,
    VdWHingeLoss,
    VdWActiveClashLoss,
    VdWMaxPenetrationLoss,
    TorsionLoss,
    ChiralAtomLoss,
    _vdw_clash_components,
)

logger = logging.getLogger(__name__)


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
    Measured on a 350-atom pocket at t=0.5 with 0.3-1.5 A of coordinate error
    (i.e. realistic mid-trajectory x_1_hat), p90 of ||lambda_t * grad J|| is
    1.3e-2 - 2.8e-2, so scale ~50 puts grad_clip=1.0 at the p90 of the
    distribution — the clip guards outliers rather than rescaling every atom.

    WARNING: scale and grad_clip are redundant once the clip binds. At scale=1e3
    with grad_clip=1.0, 82-99% of moving atoms saturate the clip mid-trajectory,
    so guidance degenerates to a fixed-magnitude normalised-gradient step and
    scale has no effect — grad_clip becomes the only strength knob. TweedieGuidance
    tracks the clipped fraction (see clip_fraction) so the regime is visible.

    vdw_buffer / vdw_cutoff / bond_buffer / angle_buffer default to the values
    below, but run_inference overrides any of them not set explicitly in the YAML
    with the training-time value recorded in the checkpoint, so the guidance
    energy matches the loss the model was trained against.
    """
    enabled: bool = False
    t_start: float = 0.3              # gate: no guidance below this t
    scale: float = 50.0               # cov-A guidance scale lambda
    schedule: str = 'linear_decay'    # decaying schedule in t (see guidance_schedule)
    grad_clip: float = 1.0            # 0 = off; per-atom max ||g|| (outlier guard)
    bond_coef: float = 1.0
    angle_coef: float = 1.0
    vdw_coef: float = 5.0             # mean squared penetration — force spread over all clashing pairs
    vdw_active_coef: float = 0.0      # soft active-clash count — clears many shallow clashes, saturates on deep
    vdw_max_coef: float = 0.0         # soft-max worst-pair penetration — concentrates force, CAN dent; keep 0 unless needed
    vdw_form: str = 'mean'            # synced from the checkpoint: 'mean' (VdWClashLoss) or 'hinge' (VdWHingeLoss)
    vdw_rec_rec_ratio: float = 0.15   # hinge form only: weight of receptor-receptor pairs relative to ligand pairs
    vdw_active_scale: float = 0.10    # Å scale where active-count saturates
    vdw_max_alpha: float = 20.0       # logsumexp sharpness for soft-max penetration
    torsion_coef: float = 1.0         # planarity loss for aromatic-ring sp2 torsions
    chiral_coef: float = 0.0          # signed-improper restraint on ligand stereocentres
    match_training_weights: bool = False  # run_inference: take every *_coef not set in the YAML from the checkpoint's loss weights
    chiral_buffer: float = 0.5236     # 30 deg margin away from the planar phi=0 state
    bond_buffer: float = 0.05         # absolute slack (Å) for bond flat-bottom
    angle_buffer: float = 0.05        # absolute slack (Å) for 1-3 flat-bottom
    vdw_buffer: float = 0.225
    vdw_cutoff: float = 6.0
    vdw_linear_onset: bool = True     # mean vdW term p^2+2p (legacy) or p^2; synced from checkpoint
    one_minus_t_min: float = 0.05     # sim_mc v_corr denominator floor only
    sim_mc_guidance: bool = False     # gradient-free candidate sampling instead of cov-A
    sim_mc_n: int = 32                # candidates per sample
    sim_mc_std: float = 0.10          # Cartesian proposal sigma (Å)
    sim_mc_scale: float = 5.0         # inverse-temperature for soft candidate averaging
    sim_mc_clash_count_coef: float = 0.0  # extra count penalty for active vdW clashes
    sim_mc_include_original: bool = True
    sim_mc_chunk: int = 8             # candidates scored per pass; bounds peak memory.
                                      # _context repeats 6 [chunk*B, N, N] tensors, so
                                      # cost is linear in chunk (B=10, N=371: ~0.7 GB at
                                      # chunk=8, ~2.7 GB if all K=32 are scored at once).
                                      # Result is chunk-independent.


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
        lig_chiral_indices: torch.Tensor = None,    # [K_chi, 5] (a0,a1,a2,centre,is_r)
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
        # Drop bonds touching a missing receptor atom BEFORE deriving 1-3 / 1-4,
        # so a 1-3 pair whose middle atom is missing also disappears. Mirrors the
        # training-time context builder in FlexLoss.
        atom_pair_mask = self.atom_mask.unsqueeze(-1) * self.atom_mask.unsqueeze(-2)
        bond_mask = bond_mask * atom_pair_mask
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

        if config.bond_coef > 0 or config.angle_coef > 0:
            self._check_bounds(cm, bond_mask, N_rec)

        # Receptor backbone + CB frozen; ligand fully free
        fixed_rec = rec_fixed_mask_3d.squeeze(-1).clamp(max=1.0)
        fixed_lig = torch.zeros(B, N_lig, device=device)
        self.fixed_mask = torch.cat([fixed_rec, fixed_lig], dim=1).clamp(max=1.0)

        # Frozen-frozen pairs are excluded from the vdW terms. Both atoms are
        # immobile, so they contribute exactly zero force — but they DO inflate
        # the reported energy and the 1/n_pairs normalisation that scales the
        # real gradients. The dominant offender is the peptide bond: prep builds
        # the receptor topology per residue (block_diag), so C(i)-N(i+1) is not a
        # bond and reads as a 1.2 A vdW penetration. On a crystal pocket that
        # accounts for 128 of 130 "clashes" and ~55 A of total penetration.
        self.frozen_pair_mask = (
            self.fixed_mask.unsqueeze(-1) * self.fixed_mask.unsqueeze(-2)
        )

        # Diagnostics: fraction of guided atoms whose gradient hit grad_clip.
        self._clip_hits = 0.0
        self._clip_total = 0.0

        # Expand sp2 torsion indices to TorsionLoss's [K_total, 5] format
        # (col 0 = batch index, cols 1-4 = atom indices). Lig atom indices stay
        # in their local frame; TorsionLoss.compute offsets them by n_rec.
        self.rec_torsion_indices = self._expand_torsions(rec_torsion_indices, B, device)
        self.lig_torsion_indices = self._expand_torsions(lig_torsion_indices, B, device)

        # Loss modules — drive via explicit coefficients (no schedule, no aux_scale)
        self.bond_loss = BondLengthLoss(weight=1.0, buffer=config.bond_buffer)
        self.angle_loss = BondAngleLoss(weight=1.0, buffer=config.angle_buffer)
        if config.vdw_form == 'hinge':
            self.vdw_loss = VdWHingeLoss(
                weight=1.0, buffer=config.vdw_buffer, cutoff=config.vdw_cutoff,
                rec_rec_ratio=config.vdw_rec_rec_ratio,
            )
        else:
            self.vdw_loss = VdWClashLoss(
                weight=1.0, buffer=config.vdw_buffer, cutoff=config.vdw_cutoff,
                linear_onset=config.vdw_linear_onset,
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
        self.chiral_loss = ChiralAtomLoss(weight=1.0, buffer=config.chiral_buffer)
        self.lig_chiral_indices = self._expand_chirals(lig_chiral_indices, B, device)

    @staticmethod
    def _check_bounds(cm: torch.Tensor, bond_mask: torch.Tensor, n_rec: int) -> None:
        """Reject a constraint matrix that carries no bounds for bonded pairs.

        BondLengthLoss/BondAngleLoss read `cm` as literal distance bounds
        (lower = cm.T - buffer, upper = cm + buffer), so a zero entry on a real
        bond means "this bond must be <= buffer long" and the restraint collapses
        it. Checked per block: a well-formed receptor must not mask a zeroed
        ligand. Blocks with no bonds at all (e.g. monoatomic ions) are skipped.
        """
        for label, sl in (('receptor', slice(0, n_rec)), ('ligand', slice(n_rec, None))):
            bonded = bond_mask[:, sl, sl]
            n_bonded = float(bonded.sum())
            if n_bonded == 0:
                continue
            has_bound = ((cm[:, sl, sl] > 0) & bonded.bool()).float().sum()
            if has_bound == 0:
                raise ValueError(
                    f"Physics guidance: the {label} constraint matrix is all-zero "
                    f"on its {int(n_bonded / 2)} bonded pairs, so the bond/angle "
                    f"restraints would pull every bond to <= the flat-bottom "
                    f"buffer. Load systems via run_inference.load_system_data, or "
                    f"set bond_coef=angle_coef=0 to run without them."
                )
            missing = int((n_bonded - float(has_bound)) / 2)
            if missing > 0:
                logger.warning(
                    "Physics guidance: %d %s bonded pair(s) have zero distance "
                    "bounds and will be driven toward zero length by BondLengthLoss.",
                    missing, label,
                )

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

    @staticmethod
    def _expand_chirals(idx_5col, B, device):
        """[K, 5] -> [B*K, 6] with batch index in col 0; replicate per sample."""
        if idx_5col is None or len(idx_5col) == 0:
            return torch.empty((0, 6), dtype=torch.long, device=device)
        if not torch.is_tensor(idx_5col):
            idx_5col = torch.as_tensor(idx_5col)
        idx_5col = idx_5col.to(device).long()
        K = idx_5col.shape[0]
        block = idx_5col.unsqueeze(0).expand(B, K, 5).reshape(B * K, 5)
        batch_col = torch.arange(B, device=device).repeat_interleave(K).unsqueeze(-1)
        return torch.cat([batch_col, block], dim=-1)

    def _repeat_chirals_for_candidates(self, idx: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        if repeat_factor == 1 or idx.numel() == 0:
            return idx
        base_B = self.fixed_mask.shape[0]
        chunks = []
        for k in range(repeat_factor):
            shifted = idx.clone()
            shifted[:, 0] = shifted[:, 0] + k * base_B
            chunks.append(shifted)
        return torch.cat(chunks, dim=0)

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
            'nonbonded_exclude_mask': self._repeat_for_candidates(
                self.frozen_pair_mask, repeat_factor
            ),
            'mask': self._repeat_for_candidates(self.atom_mask, repeat_factor),
            'atom_feats': self._repeat_for_candidates(self.atom_feats, repeat_factor),
            'constraint_matrix': self._repeat_for_candidates(self.constraint_matrix, repeat_factor),
            'rec_torsion_indices': self._repeat_torsions_for_candidates(
                self.rec_torsion_indices, repeat_factor
            ),
            'lig_torsion_indices': self._repeat_torsions_for_candidates(
                self.lig_torsion_indices, repeat_factor
            ),
            'lig_chiral_indices': self._repeat_chirals_for_candidates(
                self.lig_chiral_indices, repeat_factor
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
        if cfg.chiral_coef > 0 and self.lig_chiral_indices.numel() > 0:
            energy = energy + cfg.chiral_coef * self.chiral_loss.compute(ctx)
        return energy

    def physics_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample weighted physics energy J at combined [B, N_rec+N_lig, 3]
        coordinates. The objective guidance descends during sampling."""
        return self._physics_energy(self._context(x))

    def physics_report(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-sample [B] energy and vdW violation diagnostics at x.

        n_clash / max_penetration use the same pair mask as the vdW losses, so
        1-2 / 1-3 / 1-4 and frozen-frozen pairs are already excluded.
        """
        ctx = self._context(x)
        terms = _vdw_clash_components(ctx, self.config.vdw_buffer, self.config.vdw_cutoff)
        pen = terms['penetration'] * terms['pair_mask']
        return {
            'energy': self._physics_energy(ctx),
            'n_clash': (pen > 0).float().sum(dim=(-1, -2)),
            'max_penetration': pen.flatten(1).amax(dim=-1),
        }

    def energy(self, x: torch.Tensor, repeat_factor: int = 1) -> torch.Tensor:
        """Per-sample physics energy J for combined receptor/ligand coordinates."""
        ctx = self._context(x, repeat_factor=repeat_factor)
        energy = self._physics_energy(ctx)
        if self.config.sim_mc_clash_count_coef > 0:
            energy = energy + self.config.sim_mc_clash_count_coef * self._active_clash_count(ctx)
        return energy

    def _active_clash_count(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Count active vdW-violating nonbonded pairs per sample.

        Shares _vdw_clash_components with the vdW losses so the 1-4 and
        frozen-frozen exclusions apply here too.
        """
        terms = _vdw_clash_components(ctx, self.config.vdw_buffer, self.config.vdw_cutoff)
        return ((terms['penetration'] > 0).float() * terms['pair_mask']).sum(dim=(-1, -2))

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

        with torch.enable_grad(), record_function("flex::guidance_energy_grad"):
            x = x.requires_grad_(True)
            ctx = {
                'x_1_pred': x,
                'pred_dists': torch.cdist(x, x),
                'bond_mask': self.bond_mask,
                'angle_mask': self.angle_mask,
                'dihedral_mask': self.dihedral_mask,
                'nonbonded_exclude_mask': self.frozen_pair_mask,
                'mask': self.atom_mask,
                'atom_feats': self.atom_feats,
                'constraint_matrix': self.constraint_matrix,
                'rec_torsion_indices': self.rec_torsion_indices,
                'lig_torsion_indices': self.lig_torsion_indices,
                'lig_chiral_indices': self.lig_chiral_indices,
                'n_rec': self.N_rec,
            }
            # Sum over batch: samples are independent through the losses, so the
            # gradient at sample b depends only on J_b (matches per-sample cov-A).
            with record_function("flex::guidance_energy_forward"):
                energy = self._physics_energy(ctx).sum()
            with record_function("flex::guidance_energy_backward"):
                grad = torch.autograd.grad(energy, x)[0]

        lam = guidance_schedule(cfg.schedule, t).view(-1, 1, 1)
        g = -cfg.scale * lam * grad * free

        if cfg.grad_clip > 0:
            norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            # Track saturation: if most moving atoms clip, grad_clip — not scale —
            # is setting the guidance strength (see GuidanceConfig warning).
            moving = free.squeeze(-1) > 0
            self._clip_hits += float(
                ((norm.squeeze(-1) > cfg.grad_clip) & moving).sum()
            )
            self._clip_total += float(moving.sum())
            g = g * (norm.clamp(max=cfg.grad_clip) / norm)

        return g[:, :self.N_rec], g[:, self.N_rec:]

    def clip_fraction(self) -> Optional[float]:
        """Fraction of guided per-atom gradients that hit grad_clip over the run.

        Near 1.0 means the clip binds everywhere and `scale` is inert.
        None if grad_clip is disabled.
        """
        if self._clip_total == 0:
            return None
        return self._clip_hits / self._clip_total

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

        # Score in chunks. _context materialises 6 [k*B, N, N] tensors per pass and
        # five of them (bond/angle/dihedral/frozen-pair/constraint) are identical
        # across candidates, so scoring all K at once just replicates them K-fold.
        # Chunking bounds that; the softmax below sees the same energies either way.
        chunk = max(1, int(cfg.sim_mc_chunk))
        energy = torch.cat([
            self.energy(
                candidates[s:s + chunk].reshape(-1, N, 3),
                repeat_factor=candidates[s:s + chunk].shape[0],
            ).view(-1, B)
            for s in range(0, K, chunk)
        ], dim=0)
        logits = -cfg.sim_mc_scale * (energy - energy.min(dim=0, keepdim=True).values)
        weights = torch.softmax(logits, dim=0)
        x_corr = (weights[..., None, None] * candidates).sum(dim=0)
        x_corr = self.fixed_mask.unsqueeze(-1) * x + free * x_corr
        return x_corr[:, :self.N_rec], x_corr[:, self.N_rec:]
