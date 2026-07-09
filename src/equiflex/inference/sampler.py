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

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm

from equiflex.prep.geometry import place_cb_ideal
from equiflex.inference.guidance import GuidanceConfig, TweedieGuidance

logger = logging.getLogger(__name__)


@dataclass
class SDEConfig:
    """Drift-free stochastic perturbation on top of the deterministic Euler step.

    The standard Euler update y_{t+dt} = y_t + v*dt is augmented with isotropic
    Gaussian noise:
        y_{t+dt} = y_t + v*dt + sqrt(2*dt*w(t)*sigma) * eps,
    where w(t) = (1-t)/(t + eps_w), zeroed at t >= w_cutoff to avoid noise near
    the data endpoint. sigma scales the noise amplitude; sigma=0 reduces to the
    deterministic ODE.

    Note: ml-simplefold also adds a `w(t) * score(x_t)` drift term derived from
    the linear-interpolant marginal score. We omit that term — empirically it
    over-scales relative to the noise (their formula uses w not w*sigma) and
    competes with our Tweedie physics guidance, distorting aromatics. The pure
    noise injection here suffices to add diversity without degrading geometry.

    Fixed atoms (receptor backbone + CB) get no noise — the fixed-mask
    projection in the sampler zeros the update on those atoms.

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
            - 'log': Log-spaced (more steps near t=0 where changes are larger)

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

        rec_constraint_matrix = data.get("rec_constraint_matrix")
        if rec_constraint_matrix is None:
            rec_constraint_matrix = torch.zeros(N_rec, N_rec, 1, dtype=torch.float32)

        lig_constraint_matrix = data.get("lig_constraint_matrix")
        if lig_constraint_matrix is None:
            lig_constraint_matrix = torch.zeros(N_lig, N_lig, 1, dtype=torch.float32)

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
          - Backbone + CB: fixed at GT (CB reconstructed from ideal geometry)
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

        return rec_coords, lig_coords

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
    ) -> Dict[str, torch.Tensor]:
        """
        Generate structures by integrating the flow.

        Args:
            data: Processed data from InferenceProcessor (unbatched)
            num_samples: Number of samples to generate. Ignored if seed is a list.
            num_steps: Number of Euler integration steps
            schedule: 'uniform' or 'log'
            seed: Int (base seed, sample i uses seed+i) or list of ints.
            return_trajectory: Whether to return full coordinate trajectory

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

        data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        batch = self._build_batch(data, B)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        rec_coords, lig_coords = self._init_coords(data, B, seeds)
        rec_backbone_mask_3d = batch["rec_backbone_mask"].unsqueeze(-1)  # [B, N_rec, 1]
        rec_cb_mask_3d = batch["rec_cb_mask"].unsqueeze(-1)  # [B, N_rec, 1]
        rec_fixed_mask_3d = (
            rec_backbone_mask_3d + rec_cb_mask_3d
        )  # backbone + CB frozen

        guidance_engine: Optional[TweedieGuidance] = None
        if guidance is not None and guidance.enabled:
            guidance_engine = TweedieGuidance(
                config=guidance,
                batch=batch,
                rec_fixed_mask_3d=rec_fixed_mask_3d,
                device=self.device,
                rec_torsion_indices=data.get("rec_sp2_torsion_indices"),
                lig_torsion_indices=data.get("lig_sp2_torsion_indices"),
            )

        # Pre-compute chemistry features once (V4 two-trunk architecture)
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

        for i in tqdm(range(num_steps), desc="Integrating flow"):
            t = timesteps[i].expand(B)
            dt = timesteps[i + 1] - timesteps[i]

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

            # cov-A physics guidance on the clean x_1 estimate (paper Sec 3.3).
            if guidance_engine is not None and guidance_engine.is_active(t):
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

            # Backbone + CB stay fixed; only other sidechain and ligand move
            rec_coords = rec_coords + rec_flow * dt * (1 - rec_fixed_mask_3d)
            lig_coords = lig_coords + lig_flow * dt

            # Drift-free SDE perturbation: y += sqrt(2*dt*w(t)*sigma) * eps,
            # w(t) = (1-t)/(t+eps_w). sigma=0 or t>=w_cutoff → no noise.
            if sde is not None and sde.enabled and sde.sigma > 0:
                t_scalar = float(t.mean().item())
                if t_scalar < sde.w_cutoff:
                    w = (1.0 - t_scalar) / (t_scalar + sde.eps_w)
                    noise_amp = (2.0 * float(dt) * w * sde.sigma) ** 0.5
                    if sde.mode in ("both", "receptor"):
                        rec_noise = torch.randn_like(rec_coords) * noise_amp
                        rec_coords = rec_coords + rec_noise * (1 - rec_fixed_mask_3d)
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

        if return_trajectory:
            results["rec_trajectory"] = torch.stack(
                rec_trajectory, dim=1
            )  # [B, T, N_rec, 3]
            results["lig_trajectory"] = torch.stack(
                lig_trajectory, dim=1
            )  # [B, T, N_lig, 3]

        return results

    def relax_results(
        self,
        results: Dict[str, torch.Tensor],
        system_dir: Union[str, Path],
        sample_indices: Optional[List[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **relax_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Post-process sampling results with MMFF energy minimisation.

        Calls :func:`inference.relaxation.relax_complex` per sample.
        Only ligand coordinates are updated (protein is frozen during minimisation).

        Args:
            results: Dict returned by :meth:`sample`.
            system_dir: Path to the featurised system directory.
            sample_indices: Which samples to relax (default: all).
            output_dir: Base directory for per-sample CIF output.
            **relax_kwargs: Forwarded to :func:`relax_complex`
                (e.g. ``max_iterations``).

        Returns:
            The same *results* dict with ``lig_coords_final`` updated in-place.
        """
        from equiflex.inference.relaxation import relax_complex

        B = results["rec_coords_final"].shape[0]
        if sample_indices is None:
            sample_indices = list(range(B))

        backbone_com = results["backbone_com"]
        energies = []

        for b in sample_indices:
            out = None
            if output_dir is not None:
                out = Path(output_dir) / f"sample_{b}"

            try:
                relaxed = relax_complex(
                    system_dir=system_dir,
                    rec_coords=results["rec_coords_final"][b],
                    lig_coords=results["lig_coords_final"][b],
                    backbone_com=backbone_com,
                    output_dir=out,
                    **relax_kwargs,
                )
                results["lig_coords_final"][b] = relaxed["lig_coords"]
                energies.append(
                    {
                        "sample": b,
                        "initial_energy": relaxed["initial_energy_kcal_mol"],
                        "final_energy": relaxed["final_energy_kcal_mol"],
                    }
                )
            except Exception as e:
                logger.warning("Relaxation failed for sample %d: %s", b, e)

        results["relaxation_energies"] = energies
        return results


def run_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    num_samples: int = 10,
    num_steps: int = 100,
    schedule: Literal["uniform", "log"] = "uniform",
    seed: Optional[Union[int, List[int]]] = None,
    device: str = "cuda",
    return_trajectory: bool = True,
    relax: bool = False,
    system_dir: Optional[str] = None,
    relax_max_iterations: int = 200,
    guidance: Optional[GuidanceConfig] = None,
) -> Path:
    """
    Run inference and save results.

    Args:
        model_path: Path to Lightning checkpoint (.ckpt)
        data_path: Path to processed inference data (.pt file)
        output_path: Directory to save results
        num_samples: Number of samples to generate (ignored if seed is a list)
        num_steps: Number of integration steps
        schedule: Time scheduling strategy ('uniform' or 'log')
        seed: Random seed or list of seeds
        device: Device to run on
        return_trajectory: Whether to store the full integration trajectory
        relax: Run MMFF energy minimisation after sampling
        system_dir: Path to featurised system directory (for relaxation
            and CIF writing). Defaults to ``Path(data_path).parent``.
        relax_max_iterations: MMFF minimisation steps (default 200)

    Returns:
        Path to output directory
    """
    from equiflex.model import FlexTrainer

    trainer_module = FlexTrainer.load_from_checkpoint(model_path, map_location="cpu")
    model = trainer_module.model

    data = torch.load(data_path, map_location="cpu", weights_only=False)

    metadata_path = Path(data_path).parent / "metadata.pt"
    if metadata_path.exists():
        data.update(torch.load(metadata_path, map_location="cpu", weights_only=False))

    sampler = FlowMatchingSampler(model, device=device)
    results = sampler.sample(
        data=data,
        num_samples=num_samples,
        num_steps=num_steps,
        schedule=schedule,
        seed=seed,
        return_trajectory=return_trajectory,
        guidance=guidance,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sys_dir = Path(system_dir) if system_dir else Path(data_path).parent

    if relax:
        from equiflex.inference.relaxation import relax_complex, write_complex_cif

        pocket_feats = torch.load(sys_dir / "pocket_features.pt", weights_only=False)
        ligand_feats = torch.load(sys_dir / "ligand_features.pt", weights_only=False)
        backbone_com = results["backbone_com"]
        B = results["rec_coords_final"].shape[0]

        for b in range(B):
            sample_dir = output_path / f"sample_{b}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            write_complex_cif(
                pocket_feats,
                ligand_feats,
                results["rec_coords_final"][b],
                results["lig_coords_final"][b],
                backbone_com,
                sample_dir / "predicted.cif",
            )

            try:
                relaxed = relax_complex(
                    system_dir=sys_dir,
                    rec_coords=results["rec_coords_final"][b],
                    lig_coords=results["lig_coords_final"][b],
                    backbone_com=backbone_com,
                    output_dir=sample_dir,
                    max_iterations=relax_max_iterations,
                    write_structures=True,
                )
                results["lig_coords_final"][b] = relaxed["lig_coords"]
                logger.info(
                    "Sample %d relaxed: %.2f -> %.2f kcal/mol",
                    b,
                    relaxed["initial_energy_kcal_mol"],
                    relaxed["final_energy_kcal_mol"],
                )
            except Exception as e:
                logger.warning("Relaxation failed for sample %d: %s", b, e)

    torch.save(results, output_path / "inference_results.pt")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run flow matching inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--schedule", type=str, default="uniform", choices=["uniform", "log"]
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--return-trajectory",
        action="store_true",
        default=True,
        help="Store full integration trajectory (default: True)",
    )
    parser.add_argument(
        "--no-trajectory",
        dest="return_trajectory",
        action="store_false",
        help="Skip storing the trajectory to save memory",
    )

    relax_group = parser.add_argument_group(
        "relaxation", "MMFF energy minimisation options"
    )
    relax_group.add_argument(
        "--relax",
        action="store_true",
        help="Run MMFF energy minimisation after sampling",
    )
    relax_group.add_argument(
        "--system-dir",
        type=str,
        default=None,
        help="Path to featurised system dir (default: parent of --data)",
    )
    relax_group.add_argument(
        "--relax-max-iterations",
        type=int,
        default=200,
        help="MMFF minimisation steps (default: 200)",
    )

    guide_group = parser.add_argument_group(
        "guidance", "Tweedie x_1 physics guidance during integration"
    )
    guide_group.add_argument(
        "--guidance", action="store_true", help="Enable Tweedie x_1 physics guidance"
    )
    guide_group.add_argument(
        "--guide-t-start",
        type=float,
        default=0.3,
        help="Skip guidance below this t (default: 0.3)",
    )
    guide_group.add_argument(
        "--guide-scale",
        type=float,
        default=1.0,
        help="cov-A guidance scale lambda; tune per system (default: 1.0)",
    )
    guide_group.add_argument(
        "--guide-schedule",
        type=str,
        default="linear_decay",
        choices=["const", "linear_decay", "cosine_decay", "exp_decay"],
        help="Decaying-in-t cov-A schedule (default: linear_decay)",
    )
    guide_group.add_argument("--guide-bond-coef", type=float, default=1.0)
    guide_group.add_argument("--guide-angle-coef", type=float, default=1.0)
    guide_group.add_argument("--guide-vdw-coef", type=float, default=5.0)
    guide_group.add_argument(
        "--guide-torsion-coef",
        type=float,
        default=1.0,
        help="Aromatic-ring planarity weight (0 disables)",
    )
    guide_group.add_argument(
        "--guide-grad-clip",
        type=float,
        default=0.0,
        help="Per-atom max ||g|| on the guidance VF (Å/step); 0 disables",
    )

    args = parser.parse_args()

    guidance_cfg = None
    if args.guidance:
        guidance_cfg = GuidanceConfig(
            enabled=True,
            t_start=args.guide_t_start,
            scale=args.guide_scale,
            schedule=args.guide_schedule,
            bond_coef=args.guide_bond_coef,
            angle_coef=args.guide_angle_coef,
            vdw_coef=args.guide_vdw_coef,
            torsion_coef=args.guide_torsion_coef,
            grad_clip=args.guide_grad_clip,
        )

    output_path = run_inference(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        schedule=args.schedule,
        seed=args.seed,
        device=args.device,
        return_trajectory=args.return_trajectory,
        relax=args.relax,
        system_dir=args.system_dir,
        relax_max_iterations=args.relax_max_iterations,
        guidance=guidance_cfg,
    )
    print(f"Results saved to: {output_path}")
