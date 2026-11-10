"""
Flow matching inference sampler for Flex model (V4).

Integrates the learned velocity field from t=0 (noise) to t=1 (data)
using Euler integration with configurable time scheduling.

Usage:
    sampler = FlowMatchingSampler(model, device='cuda')
    results = sampler.sample(
        data=processed_data,
        num_samples=10,
        num_steps=100,
        schedule='linear',
    )
"""

import math
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Union
import numpy as np
from tqdm import tqdm
from torch.profiler import record_function

from equiflex.prep.geometry import place_cb_ideal
from equiflex.inference.guidance import GuidanceConfig, TweedieGuidance

logger = logging.getLogger(__name__)


@dataclass
class SDEConfig:
    """Drift-free stochastic perturbation on top of the deterministic Euler step.

    The standard Euler update y_{t+dt} = y_t + v*dt is augmented with isotropic
    Gaussian noise whose variance is the exact integral of 2*sigma*w(s) over the
    step [t0, t1], t1 = min(t0+dt, w_cutoff):
        var = 2*sigma*((1+eps_w)*log((t1+eps_w)/(t0+eps_w)) - (t1-t0)),
    where w(s) = (1-s)/(s + eps_w); no noise for t0 >= w_cutoff (avoids noise
    near the data endpoint). sigma=0 reduces to the deterministic ODE.

    Note: ml-simplefold also adds a `w(t) * score(x_t)` drift term derived from
    the linear-interpolant marginal score. We omit that term — empirically it
    over-scales relative to the noise (their formula uses w not w*sigma) and
    competes with our Tweedie physics guidance, distorting aromatics. The pure
    noise injection here suffices to add diversity without degrading geometry.

    Frozen atoms (receptor backbone + CB, and missing receptor atoms whose
    coordinates are placeholders) get neither drift nor noise.

    `score_drift=True` replaces the drift-free kick with the marginal-preserving
    Ornstein-Uhlenbeck step the interpolant actually implies. For x_t = t*x_1 +
    (1-t)*x_0 with x_0 ~ N(mu, s0^2 I), the marginal score is algebraic in the
    predicted velocity (no extra NFE):

        x1_hat = x_t + (1-t)*v
        m      = t*x1_hat + (1-t)*mu          # marginal mean
        grad log p_t(x_t) = -(x_t - m) / ((1-t)^2 * s0^2)

    With diffusion lambda(t) = 2*sigma*s0^2*w(t), the marginal-preserving SDE
    dx = [v + (lambda/2)*grad log p_t] dt + sqrt(lambda) dW has, over a step, the
    exact OU solution (s0^2 cancels in the drift, survives in the noise):

        c = sigma * log((t1+eps_w)/(t0+eps_w))
        x <- m + (x - m)*exp(-c) + N(0, (1-t1)^2 * s0^2 * (1-exp(-2c)))

    The exponential form is unconditionally stable, unlike an Euler drift step
    (w(t) ~ 1/eps_w near t=0 blows up at dt=0.05). Noise vanishes as t->1 on its
    own, so w_cutoff is unused in this mode. The drift-free mode (default) is the
    old behaviour: isotropic kicks that the mean-field velocity re-contracts by
    (1-t_end)/(1-t) -> 0, so they barely change the sample.

    `mode` selects which subsystem receives noise:
        'both'     — receptor (mobile atoms) + ligand (default).
        'receptor' — only receptor mobile atoms; ligand stays deterministic so a
                     good pose is preserved while sidechains re-anneal.
        'ligand'   — only ligand.
    The deterministic drift always acts on all atoms, so the un-noised subsystem
    still co-adapts through the velocity field.
    """

    enabled: bool = False
    sigma: float = 0.15  # noise scale; 0 = deterministic. 0.15 ≈ +50% diversity.
    eps_w: float = 0.01  # epsilon in w(t) denominator (controls noise envelope shape)
    w_cutoff: float = 0.99  # zero noise above this t (avoid endpoint blow-up)
    mode: Literal["both", "receptor", "ligand"] = "both"
    score_drift: bool = False  # marginal-preserving OU step instead of bare noise


@dataclass
class BranchConfig:
    """Mid-flight branching: diversify by splitting trajectories, not by noise.

    At the first timestep with t >= at_t, every live trajectory is duplicated
    `factor` times. Copy 0 is left untouched; each other copy gets a random
    rotation of the ligand about its own centroid plus a Gaussian translation.
    Both copies are then carried to t=1 in one batch, so the first half of the
    path is computed once and the cost is (at_t)*B + (1-at_t)*B*factor steps
    instead of B*factor full trajectories.

    Rotating the in-flight state rotates signal and noise together. The prior is
    isotropic, so a rotated noise draw is an equally valid noise draw: this is
    distributionally the same as rotating the clean estimate and resampling the
    noise, but without a backward jump.

    `at_t` matters: the noise scale at time t is (1-t)*ligand_prior_sigma, and a
    perturbation only survives if it displaces atoms by more than that. At t=0.5
    with sigma 4 A the threshold is ~2 A, so full rotations survive and small
    ones are absorbed. Measured: at t=0.3 hypothesis survival halves.
    """

    enabled: bool = False
    at_t: float = 0.5
    factor: int = 2           # copies per trajectory, including the original
    max_rot_deg: float = 180.0
    trans_sigma: float = 1.0  # Angstrom


def _random_rotations(n: int, max_deg: float, device, dtype) -> torch.Tensor:
    """n uniform random rotation matrices with angle <= max_deg."""
    axis = torch.randn(n, 3, device=device, dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    angle = torch.rand(n, device=device, dtype=dtype) * (max_deg * math.pi / 180.0)
    K = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
    s = angle.view(n, 1, 1).sin()
    c = angle.view(n, 1, 1).cos()
    return eye + s * K + (1 - c) * (K @ K)


def get_time_schedule(
    num_steps: int,
    schedule: Literal["uniform", "log"] = "uniform",
) -> torch.Tensor:
    """
    Generate time schedule from t=0 to t=1.

    Args:
        num_steps: Number of integration steps
        schedule: Time scheduling strategy
            - 'uniform': Uniform spacing in [0, 1]
            - 'log': Log-spaced (dense near t=1, coarse near t=0)

    Returns:
        Tensor of shape [num_steps + 1] with time values from 0 to 1
    """
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, num_steps + 1)
    elif schedule == "log":
        # Matches _euler_integrate in trainer.py
        t = 1.0 - torch.logspace(-2, 0, num_steps + 1).flip(0)
        t = t - t.min()
        t = t / t.max()
        return t.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class FlowMatchingSampler:
    """
    Sampler for FlexModelV4 using Euler integration.

    The V4 chemistry trunk is run once (B=1) before integration.
    The flow trunk is called per timestep with q_chem/p_chem cached.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def _build_batch(
        self,
        data: Dict[str, torch.Tensor],
        B: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Build batched inputs for V4.

        System-level tensors (chemistry trunk inputs) stay at [1, ...].
        Per-timestep tensors (masks) are expanded to [B, ...].
        Coordinate tensors (rec_coords, lig_coords) are handled separately
        in the integration loop.
        """
        N_rec = data["_n_rec"]
        N_lig = data["_n_lig"]

        rec_uid_vec = data["rec_uid_vec"]
        if not isinstance(rec_uid_vec, torch.Tensor):
            rec_uid_vec = torch.tensor(rec_uid_vec, dtype=torch.long)

        lig_uid_vec = data["lig_uid_vec"]
        if not isinstance(lig_uid_vec, torch.Tensor):
            lig_uid_vec = torch.tensor(lig_uid_vec, dtype=torch.long)

        rec_backbone_mask = data.get("rec_backbone_mask")
        if rec_backbone_mask is None:
            rec_backbone_mask = torch.zeros(N_rec, dtype=torch.float32)
        elif not isinstance(rec_backbone_mask, torch.Tensor):
            rec_backbone_mask = torch.tensor(rec_backbone_mask, dtype=torch.float32)

        rec_cb_mask = data.get("rec_cb_mask")
        if rec_cb_mask is None:
            rec_cb_mask = torch.zeros(N_rec, dtype=torch.float32)
        elif not isinstance(rec_cb_mask, torch.Tensor):
            rec_cb_mask = torch.tensor(rec_cb_mask, dtype=torch.float32)

        rec_missing_mask = data.get("rec_missing_mask")
        if rec_missing_mask is None:
            rec_missing_mask = torch.zeros(N_rec, dtype=torch.bool)
        elif not isinstance(rec_missing_mask, torch.Tensor):
            rec_missing_mask = torch.tensor(rec_missing_mask, dtype=torch.bool)
        else:
            rec_missing_mask = rec_missing_mask.bool()

        # No zeros fallback: prep raises on a missing constraint matrix, so every
        # featurised system has one. An all-zero matrix is a valid "no info" pair
        # feature for the model but means "every bond must be 0 A long" to the
        # physics guidance, which reads the same tensor as literal distance
        # bounds. Substituting zeros here converts missing data into a silent
        # geometry collapse; a KeyError is the correct outcome.
        for key in ("rec_constraint_matrix", "lig_constraint_matrix"):
            if data.get(key) is None:
                raise KeyError(
                    f"{key} is missing from the inference data. It is a required "
                    "input (pair feature for the model, distance bounds for the "
                    "physics guidance). Load systems via "
                    "run_inference.load_system_data, which reads the matrices "
                    "prep writes into pocket_features.pt / ligand_features.pt."
                )
        rec_constraint_matrix = data["rec_constraint_matrix"]
        lig_constraint_matrix = data["lig_constraint_matrix"]

        return {
            # System-level [1, ...] — chemistry trunk inputs
            "rec_atom_feats": data["rec_atom_feats"].unsqueeze(0),
            "rec_atom_ref_coords": data["rec_atom_ref_coords"].unsqueeze(0),
            "rec_constraint_matrix": rec_constraint_matrix.unsqueeze(0),
            "rec_bond_feats": data["rec_bond_feats"].unsqueeze(0),
            "rec_uid_vec": rec_uid_vec.unsqueeze(0),
            "esm_embeddings": data["esm_embeddings"].unsqueeze(0),
            "lig_atom_feats": data["lig_atom_feats"].unsqueeze(0),
            "lig_atom_ref_coords": data["lig_atom_ref_coords"].unsqueeze(0),
            "lig_constraint_matrix": lig_constraint_matrix.unsqueeze(0),
            "lig_bond_feats": data["lig_bond_feats"].unsqueeze(0),
            "lig_uid_vec": lig_uid_vec.unsqueeze(0),
            "unimol_embeddings": data["unimol_embeddings"].unsqueeze(0),
            # Per-timestep [B, ...]
            "rec_atom_mask": torch.ones(B, N_rec, dtype=torch.bool),
            "lig_atom_mask": torch.ones(B, N_lig, dtype=torch.bool),
            "rec_backbone_mask": rec_backbone_mask.unsqueeze(0).expand(B, -1).clone(),
            "rec_cb_mask": rec_cb_mask.unsqueeze(0).expand(B, -1).clone(),
            "rec_missing_mask": rec_missing_mask.unsqueeze(0).expand(B, -1).clone(),
        }

    def _init_coords(
        self,
        data: Dict[str, torch.Tensor],
        B: int,
        seeds: Optional[List[int]],
    ):
        """
        Initialize noise coordinates, matching _prepare_flow_matching in trainer.py:
          - Backbone + CB: fixed at GT (CB keeps its GT coordinates)
          - Sidechain (non-CB): N(ideal CB position, I)
          - Ligand: N(0, I)
        """
        N_rec = data["_n_rec"]
        N_lig = data["_n_lig"]
        device = self.device

        rec_gt = data["rec_atom_coords"].to(device)  # [N_rec, 3]
        rec_backbone_mask = data.get("rec_backbone_mask")  # [N_rec]
        if rec_backbone_mask is None:
            rec_backbone_mask = torch.zeros(N_rec, dtype=torch.float32, device=device)
        else:
            rec_backbone_mask = rec_backbone_mask.to(device)

        rec_cb_mask = data.get("rec_cb_mask")
        if rec_cb_mask is None:
            rec_cb_mask = torch.zeros(N_rec, dtype=torch.float32, device=device)
        else:
            rec_cb_mask = rec_cb_mask.to(device)

        rec_uid_vec = data["rec_uid_vec"]
        if not isinstance(rec_uid_vec, torch.Tensor):
            rec_uid_vec = torch.tensor(rec_uid_vec, dtype=torch.long)
        rec_uid_vec = rec_uid_vec.to(device)  # [N_rec]

        backbone_mask_3d = rec_backbone_mask.unsqueeze(-1)  # [N_rec, 1]
        cb_mask_3d = rec_cb_mask.unsqueeze(-1)  # [N_rec, 1]
        fixed_mask = backbone_mask_3d + cb_mask_3d  # [N_rec, 1]

        # Compute noise center: ideal CB from backbone or fallback to COM
        bai = data.get("rec_backbone_atom_indices")
        if bai is not None:
            n_idx = bai["n_idx"].to(device)
            ca_idx = bai["ca_idx"].to(device)
            c_idx = bai["c_idx"].to(device)
            is_gly = bai["is_gly"].to(device)

            n_coords = rec_gt[n_idx]  # [n_residues, 3]
            ca_coords = rec_gt[ca_idx]  # [n_residues, 3]
            c_coords = rec_gt[c_idx]  # [n_residues, 3]

            cb_ideal = place_cb_ideal(n_coords, ca_coords, c_coords)
            cb_center = torch.where(is_gly.unsqueeze(-1), ca_coords, cb_ideal)

            atom_noise_center = cb_center[rec_uid_vec]  # [N_rec, 3]
        else:
            # Fallback: per-residue backbone COM
            n_residues = rec_uid_vec.max().item() + 1
            weighted = backbone_mask_3d * rec_gt
            res_sum = torch.zeros(n_residues, 3, device=device)
            res_count = torch.zeros(n_residues, 1, device=device)
            res_sum.scatter_add_(0, rec_uid_vec.unsqueeze(-1).expand(-1, 3), weighted)
            res_count.scatter_add_(0, rec_uid_vec.unsqueeze(-1), backbone_mask_3d)
            res_count = res_count.clamp(min=1.0)
            res_com = res_sum / res_count
            atom_noise_center = res_com[rec_uid_vec]

        # Prior noise scales must match training (model config).
        cfg = getattr(self.model, 'config', None)
        sc_sigma = getattr(cfg, 'sidechain_prior_sigma', 1.0)
        lig_sigma = getattr(cfg, 'ligand_prior_sigma', 1.0)

        def _randn(shape):
            return torch.randn(shape, device=device)

        if seeds is not None:
            rec_coords_list, lig_coords_list = [], []
            for s in seeds:
                torch.manual_seed(s)
                rec_noise = _randn((N_rec, 3)) * sc_sigma + atom_noise_center
                rec_coords_list.append(
                    fixed_mask * rec_gt + (1 - fixed_mask) * rec_noise
                )
                lig_coords_list.append(_randn((N_lig, 3)) * lig_sigma)
            rec_coords = torch.stack(rec_coords_list, dim=0)
            lig_coords = torch.stack(lig_coords_list, dim=0)
        else:
            rec_noise = _randn((B, N_rec, 3)) * sc_sigma + atom_noise_center.unsqueeze(0)
            rec_coords = (
                fixed_mask.unsqueeze(0) * rec_gt.unsqueeze(0)
                + (1 - fixed_mask.unsqueeze(0)) * rec_noise
            )
            lig_coords = _randn((B, N_lig, 3)) * lig_sigma

        prior = {
            "rec_mu": atom_noise_center.unsqueeze(0),  # [1, N_rec, 3]
            "lig_mu": torch.zeros(1, 1, 3, device=device),  # pocket-COM frame
            "rec_sigma": float(sc_sigma),
            "lig_sigma": float(lig_sigma),
        }
        return rec_coords, lig_coords, prior

    @torch.no_grad()
    def sample(
        self,
        data: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        num_steps: int = 100,
        schedule: Literal["uniform", "log"] = "uniform",
        seed: Optional[Union[int, List[int]]] = None,
        return_trajectory: bool = True,
        guidance: Optional[GuidanceConfig] = None,
        sde: Optional[SDEConfig] = None,
        branch: Optional[BranchConfig] = None,
        progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate structures by integrating the flow.

        Args:
            data: Processed system data (unbatched), as returned by
                run_inference.load_system_data
            num_samples: Number of samples to generate. Ignored if seed is a list.
            num_steps: Number of Euler integration steps
            schedule: 'uniform' or 'log'
            seed: Int (base seed, sample i uses seed+i) or list of ints.
            return_trajectory: Whether to return full coordinate trajectory
            progress: Show the per-step tqdm bar (off for many-system screens)

        Returns:
            - rec_coords_final: [B, N_rec, 3]
            - lig_coords_final: [B, N_lig, 3]
            - timesteps: [num_steps+1]
            - backbone_com: [3] for recovering absolute coordinates
            - seeds: list of seeds used (None if random)
            - rec_trajectory: [B, num_steps+1, N_rec, 3] (if return_trajectory)
            - lig_trajectory: [B, num_steps+1, N_lig, 3] (if return_trajectory)
        """
        if isinstance(seed, list):
            seeds = seed
            B = len(seeds)
        elif seed is not None:
            B = num_samples if num_samples is not None else 10
            seeds = [seed + i for i in range(B)]
        else:
            B = num_samples if num_samples is not None else 10
            seeds = None

        branching = branch is not None and branch.enabled and branch.factor > 1
        if branching and return_trajectory:
            raise ValueError(
                "branching changes the batch size mid-path, so a single stacked "
                "trajectory tensor cannot be returned; call with "
                "return_trajectory=False"
            )
        branched = False

        data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        with record_function("flex::build_batch"):
            batch = self._build_batch(data, B)
            batch = {k: v.to(self.device) for k, v in batch.items()}

        with record_function("flex::init_coords"):
            rec_coords, lig_coords, prior = self._init_coords(data, B, seeds)
        rec_backbone_mask_3d = batch["rec_backbone_mask"].unsqueeze(-1)  # [B, N_rec, 1]
        rec_cb_mask_3d = batch["rec_cb_mask"].unsqueeze(-1)  # [B, N_rec, 1]
        rec_fixed_mask_3d = (
            rec_backbone_mask_3d + rec_cb_mask_3d
        )  # backbone + CB frozen
        # Missing atoms hold placeholder coordinates: never move them either.
        rec_missing_mask_3d = batch["rec_missing_mask"].unsqueeze(-1).float()
        rec_frozen_3d = torch.clamp(rec_fixed_mask_3d + rec_missing_mask_3d, max=1.0)

        guidance_engine: Optional[TweedieGuidance] = None
        if guidance is not None and guidance.enabled:
            with record_function("flex::guidance_setup"):
                guidance_engine = TweedieGuidance(
                    config=guidance,
                    batch=batch,
                    rec_fixed_mask_3d=rec_fixed_mask_3d,
                    device=self.device,
                    rec_torsion_indices=data.get("rec_sp2_torsion_indices"),
                    lig_torsion_indices=data.get("lig_sp2_torsion_indices"),
                    lig_chiral_indices=data.get("lig_chiral_indices"),
                )

        # Pre-compute chemistry features once (two-trunk architecture)
        with record_function("flex::forward_chemistry"):
            q_chem, p_chem = self.model.forward_chemistry(
                rec_atom_feats=batch["rec_atom_feats"],
                rec_atom_ref_coords=batch["rec_atom_ref_coords"],
                rec_constraint_matrix=batch["rec_constraint_matrix"],
                rec_bond_feats=batch["rec_bond_feats"],
                rec_uid_vec=batch["rec_uid_vec"],
                esm_embeddings=batch["esm_embeddings"],
                lig_atom_feats=batch["lig_atom_feats"],
                lig_atom_ref_coords=batch["lig_atom_ref_coords"],
                lig_constraint_matrix=batch["lig_constraint_matrix"],
                lig_bond_feats=batch["lig_bond_feats"],
                lig_uid_vec=batch["lig_uid_vec"],
                unimol_embeddings=batch["unimol_embeddings"],
            )

        timesteps = get_time_schedule(num_steps, schedule).to(self.device)

        if return_trajectory:
            rec_trajectory = [rec_coords.cpu().clone()]
            lig_trajectory = [lig_coords.cpu().clone()]

        for i in tqdm(range(num_steps), desc="Integrating flow", disable=not progress):
            if branching and not branched and float(timesteps[i]) >= branch.at_t:
                with record_function("flex::branch"):
                    n_extra = B * (branch.factor - 1)
                    rot = _random_rotations(
                        n_extra, branch.max_rot_deg, self.device, lig_coords.dtype
                    )
                    extra = lig_coords.repeat(branch.factor - 1, 1, 1)
                    centroid = extra.mean(dim=1, keepdim=True)
                    trans = (
                        torch.randn(n_extra, 3, device=self.device, dtype=extra.dtype)
                        * branch.trans_sigma
                    )
                    extra = (
                        torch.einsum("nij,naj->nai", rot, extra - centroid)
                        + centroid
                        + trans.unsqueeze(1)
                    )
                    lig_coords = torch.cat([lig_coords, extra], dim=0)
                    rec_coords = rec_coords.repeat(branch.factor, 1, 1)
                    B = B * branch.factor
                    batch = {
                        k: v.to(self.device)
                        for k, v in self._build_batch(data, B).items()
                    }
                    rec_fixed_mask_3d = (
                        batch["rec_backbone_mask"].unsqueeze(-1)
                        + batch["rec_cb_mask"].unsqueeze(-1)
                    )
                    rec_frozen_3d = torch.clamp(
                        rec_fixed_mask_3d
                        + batch["rec_missing_mask"].unsqueeze(-1).float(),
                        max=1.0,
                    )
                    if guidance_engine is not None:
                        guidance_engine = TweedieGuidance(
                            config=guidance,
                            batch=batch,
                            rec_fixed_mask_3d=rec_fixed_mask_3d,
                            device=self.device,
                            rec_torsion_indices=data.get("rec_sp2_torsion_indices"),
                            lig_torsion_indices=data.get("lig_sp2_torsion_indices"),
                            lig_chiral_indices=data.get("lig_chiral_indices"),
                        )
                    if seeds is not None:
                        seeds = [s for _ in range(branch.factor) for s in seeds]
                    branched = True
            t = timesteps[i].expand(B)
            dt = timesteps[i + 1] - timesteps[i]

            with record_function("flex::model_forward"):
                output = self.model(
                    rec_atom_feats=batch["rec_atom_feats"],
                    rec_atom_coords=rec_coords,
                    rec_atom_ref_coords=batch["rec_atom_ref_coords"],
                    rec_constraint_matrix=batch["rec_constraint_matrix"],
                    rec_bond_feats=batch["rec_bond_feats"],
                    rec_uid_vec=batch["rec_uid_vec"],
                    rec_atom_mask=batch["rec_atom_mask"],
                    rec_backbone_mask=batch["rec_backbone_mask"],
                    esm_embeddings=batch["esm_embeddings"],
                    lig_atom_feats=batch["lig_atom_feats"],
                    lig_atom_coords=lig_coords,
                    lig_atom_ref_coords=batch["lig_atom_ref_coords"],
                    lig_constraint_matrix=batch["lig_constraint_matrix"],
                    lig_bond_feats=batch["lig_bond_feats"],
                    lig_uid_vec=batch["lig_uid_vec"],
                    lig_atom_mask=batch["lig_atom_mask"],
                    unimol_embeddings=batch["unimol_embeddings"],
                    rec_missing_mask=batch["rec_missing_mask"],
                    t=t,
                    q_chem=q_chem,
                    p_chem=p_chem,
                )

            rec_flow = output["rec_flow"]  # [B, N_rec, 3]
            lig_flow = output["lig_flow"]  # [B, N_lig, 3]

            # Clean estimate from the raw (un-guided) velocity: the score identity
            # holds for the model's own velocity, not the guided one.
            if sde is not None and sde.enabled and sde.score_drift:
                one_minus_t0 = (1 - t).view(B, 1, 1)
                rec_x1_raw = rec_coords + one_minus_t0 * rec_flow
                lig_x1_raw = lig_coords + one_minus_t0 * lig_flow

            # cov-A physics guidance on the clean x_1 estimate (paper Sec 3.3).
            if guidance_engine is not None and guidance_engine.is_active(t):
                with record_function("flex::guidance"):
                    one_minus_t_true = (1 - t).view(B, 1, 1)
                    rec_x1_hat = rec_coords + one_minus_t_true * rec_flow
                    lig_x1_hat = lig_coords + one_minus_t_true * lig_flow

                    # Snap fixed atoms (backbone + CB) back to their actual positions —
                    # rec_coords for them is GT due to per-step projection. Without this,
                    # free atoms' bond/vdW gradients are anchored to drifted fixed neighbors.
                    rec_x1_hat = (
                        rec_fixed_mask_3d * rec_coords
                        + (1 - rec_fixed_mask_3d) * rec_x1_hat
                    )

                    if guidance.sim_mc_guidance:
                        # Gradient-free: correct x_1, convert back through the Tweedie map.
                        one_minus_t_denom = one_minus_t_true.clamp(
                            min=guidance.one_minus_t_min
                        )
                        rec_x1_hat, lig_x1_hat = guidance_engine.sim_mc_correct(
                            rec_x1_hat, lig_x1_hat
                        )
                        rec_flow = (rec_x1_hat - rec_coords) / one_minus_t_denom
                        lig_flow = (lig_x1_hat - lig_coords) / one_minus_t_denom
                    else:
                        # cov-A: add the guidance VF directly to the predicted velocity.
                        rec_g, lig_g = guidance_engine.guidance_velocity(
                            rec_x1_hat, lig_x1_hat, t
                        )
                        rec_flow = rec_flow + rec_g
                        lig_flow = lig_flow + lig_g

            # Backbone + CB (and missing atoms) stay fixed; only other sidechain and ligand move
            with record_function("flex::coord_update"):
                rec_coords = rec_coords + rec_flow * dt * (1 - rec_frozen_3d)
                lig_coords = lig_coords + lig_flow * dt

            # Drift-free SDE perturbation with the exact step variance
            # int_{t0}^{t1} 2*sigma*w(s) ds, w(s) = (1-s)/(s+eps_w), t1 capped at
            # w_cutoff. sigma=0 or t0>=w_cutoff -> no noise.
            # Never on the final step: nothing runs after it, so that noise would
            # be an uncorrected random kick on the returned coordinates — no
            # velocity update and no guidance to absorb it.
            is_last_step = i == num_steps - 1
            if (
                sde is not None and sde.enabled and sde.sigma > 0
                and sde.score_drift and not is_last_step
            ):
                # Marginal-preserving OU step at t1 (see SDEConfig docstring).
                t0 = float(timesteps[i])
                t1 = float(timesteps[i + 1])
                c = sde.sigma * np.log((t1 + sde.eps_w) / (t0 + sde.eps_w))
                decay = float(np.exp(-c))
                noise_frac = float(max(1.0 - np.exp(-2.0 * c), 0.0)) ** 0.5
                if sde.mode in ("both", "receptor"):
                    m = t1 * rec_x1_raw + (1 - t1) * prior["rec_mu"]
                    std = (1 - t1) * prior["rec_sigma"] * noise_frac
                    ou = m + (rec_coords - m) * decay + torch.randn_like(rec_coords) * std
                    rec_coords = (
                        rec_frozen_3d * rec_coords + (1 - rec_frozen_3d) * ou
                    )
                if sde.mode in ("both", "ligand"):
                    m = t1 * lig_x1_raw + (1 - t1) * prior["lig_mu"]
                    std = (1 - t1) * prior["lig_sigma"] * noise_frac
                    lig_coords = (
                        m + (lig_coords - m) * decay + torch.randn_like(lig_coords) * std
                    )
            elif sde is not None and sde.enabled and sde.sigma > 0 and not is_last_step:
                t0 = float(timesteps[i])
                if t0 < sde.w_cutoff:
                    t1 = min(float(timesteps[i + 1]), sde.w_cutoff)
                    var = 2.0 * sde.sigma * (
                        (1.0 + sde.eps_w) * np.log((t1 + sde.eps_w) / (t0 + sde.eps_w)) - (t1 - t0)
                    )
                    noise_amp = max(var, 0.0) ** 0.5
                    if sde.mode in ("both", "receptor"):
                        rec_noise = torch.randn_like(rec_coords) * noise_amp
                        rec_coords = rec_coords + rec_noise * (1 - rec_frozen_3d)
                    if sde.mode in ("both", "ligand"):
                        lig_noise = torch.randn_like(lig_coords) * noise_amp
                        lig_coords = lig_coords + lig_noise

            if return_trajectory:
                rec_trajectory.append(rec_coords.cpu().clone())
                lig_trajectory.append(lig_coords.cpu().clone())

        results = {
            "rec_coords_final": rec_coords.cpu(),
            "lig_coords_final": lig_coords.cpu(),
            "timesteps": timesteps.cpu(),
            "backbone_com": data["_backbone_com"].cpu(),
            "seeds": seeds,
        }

        # Record the same final-coordinate physics energy used by guidance so
        # downstream ranking/reporting does not have to reconstruct its context.
        if guidance_engine is not None:
            with torch.no_grad():
                final_x = torch.cat([rec_coords, lig_coords], dim=1)
                final_ctx = guidance_engine._context(final_x)
                results["physics_loss"] = guidance_engine._physics_energy(final_ctx).cpu()

            clip_frac = guidance_engine.clip_fraction()
            if clip_frac is not None:
                results["guidance_clip_fraction"] = clip_frac
                if clip_frac > 0.5:
                    logger.warning(
                        "guidance: %.0f%% of per-atom gradients hit grad_clip=%.3g — "
                        "grad_clip, not scale=%.3g, is setting guidance strength",
                        100 * clip_frac, guidance.grad_clip, guidance.scale,
                    )

        if return_trajectory:
            results["rec_trajectory"] = torch.stack(
                rec_trajectory, dim=1
            )  # [B, T, N_rec, 3]
            results["lig_trajectory"] = torch.stack(
                lig_trajectory, dim=1
            )  # [B, T, N_lig, 3]

        return results
