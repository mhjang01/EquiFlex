"""Trace physics loss terms during one inference integration.

This is a diagnostic script for already-prepared ``run_inference.py`` output
directories. It replays sampling for one system/sample seed and records raw
auxiliary losses per Euler step.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equiflex.inference.sampler import FlowMatchingSampler, get_time_schedule
from equiflex.inference.guidance import GuidanceConfig, TweedieGuidance
from equiflex.model import FlexTrainer
from equiflex.model.loss import BondAngleLoss, BondLengthLoss, TorsionLoss, VdWClashLoss
from run_inference import load_system_data


def _to_device(data: Dict, device: str) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _expand_torsions(idx_4col, batch_size: int, device: str) -> torch.Tensor:
    if idx_4col is None or idx_4col.numel() == 0:
        return torch.empty((0, 5), dtype=torch.long, device=device)
    idx_4col = idx_4col.to(device).long()
    k = idx_4col.shape[0]
    atom_block = idx_4col.unsqueeze(0).expand(batch_size, k, 4).reshape(batch_size * k, 4)
    batch_col = torch.arange(batch_size, device=device).repeat_interleave(k).unsqueeze(-1)
    return torch.cat([batch_col, atom_block], dim=-1)


def _build_loss_context(batch: Dict, data: Dict, rec_fixed_mask_3d: torch.Tensor) -> Dict:
    device = rec_fixed_mask_3d.device
    batch_size = rec_fixed_mask_3d.shape[0]
    n_rec = batch["rec_atom_feats"].shape[1]
    n_lig = batch["lig_atom_feats"].shape[1]
    n_atoms = n_rec + n_lig

    rec_feats = batch["rec_atom_feats"]
    lig_feats = batch["lig_atom_feats"]
    if rec_feats.shape[0] == 1 and batch_size > 1:
        rec_feats = rec_feats.expand(batch_size, -1, -1)
        lig_feats = lig_feats.expand(batch_size, -1, -1)
    atom_feats = torch.cat([rec_feats, lig_feats], dim=1)

    atom_mask = torch.cat(
        [batch["rec_atom_mask"].float(), batch["lig_atom_mask"].float()],
        dim=1,
    )
    rec_missing = batch.get("rec_missing_mask")
    if rec_missing is not None:
        lig_missing = torch.zeros_like(batch["lig_atom_mask"], dtype=torch.bool)
        atom_mask = atom_mask * (~torch.cat([rec_missing, lig_missing], dim=1)).float()

    rec_bond_mask = (batch["rec_bond_feats"].abs().sum(-1) > 0).float()
    lig_bond_mask = (batch["lig_bond_feats"].abs().sum(-1) > 0).float()
    if rec_bond_mask.shape[0] == 1 and batch_size > 1:
        rec_bond_mask = rec_bond_mask.expand(batch_size, -1, -1)
        lig_bond_mask = lig_bond_mask.expand(batch_size, -1, -1)
    bond_mask = torch.zeros(batch_size, n_atoms, n_atoms, device=device)
    bond_mask[:, :n_rec, :n_rec] = rec_bond_mask
    bond_mask[:, n_rec:, n_rec:] = lig_bond_mask

    atom_pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
    bond_mask = bond_mask * atom_pair_mask
    bond_mask_13 = torch.bmm(bond_mask, bond_mask).clamp(max=1.0)
    bond_mask_13 = bond_mask_13 * (1 - torch.eye(n_atoms, device=device).unsqueeze(0))
    angle_mask = bond_mask_13 * (1 - bond_mask)

    rec_cm = batch["rec_constraint_matrix"]
    lig_cm = batch["lig_constraint_matrix"]
    if rec_cm.ndim == 4:
        rec_cm = rec_cm.squeeze(-1)
    if lig_cm.ndim == 4:
        lig_cm = lig_cm.squeeze(-1)
    if rec_cm.shape[0] == 1 and batch_size > 1:
        rec_cm = rec_cm.expand(batch_size, -1, -1)
        lig_cm = lig_cm.expand(batch_size, -1, -1)
    constraint_matrix = torch.zeros(batch_size, n_atoms, n_atoms, device=device)
    constraint_matrix[:, :n_rec, :n_rec] = rec_cm
    constraint_matrix[:, n_rec:, n_rec:] = lig_cm

    return {
        "atom_feats": atom_feats,
        "atom_mask": atom_mask,
        "bond_mask": bond_mask,
        "angle_mask": angle_mask,
        "constraint_matrix": constraint_matrix,
        "rec_torsion_indices": _expand_torsions(
            data.get("rec_sp2_torsion_indices"), batch_size, device
        ),
        "lig_torsion_indices": _expand_torsions(
            data.get("lig_sp2_torsion_indices"), batch_size, device
        ),
        "n_rec": n_rec,
        "n_lig": n_lig,
        "fixed_mask": torch.cat(
            [
                rec_fixed_mask_3d.squeeze(-1).clamp(max=1.0),
                torch.zeros(batch_size, n_lig, device=device),
            ],
            dim=1,
        ),
    }


def _loss_values(
    x: torch.Tensor,
    base_ctx: Dict,
    bond_loss: BondLengthLoss,
    angle_loss: BondAngleLoss,
    vdw_loss: VdWClashLoss,
    torsion_loss: TorsionLoss,
) -> Dict[str, float]:
    dists = torch.cdist(x, x)
    ctx = {
        "x_1_pred": x,
        "pred_dists": dists,
        "bond_mask": base_ctx["bond_mask"],
        "angle_mask": base_ctx["angle_mask"],
        "mask": base_ctx["atom_mask"],
        "atom_feats": base_ctx["atom_feats"],
        "constraint_matrix": base_ctx["constraint_matrix"],
        "rec_torsion_indices": base_ctx["rec_torsion_indices"],
        "lig_torsion_indices": base_ctx["lig_torsion_indices"],
        "n_rec": base_ctx["n_rec"],
    }

    values = {
        "bond": bond_loss.compute(ctx).mean(),
        "angle": angle_loss.compute(ctx).mean(),
        "vdw": vdw_loss.compute(ctx).mean(),
        "torsion": torsion_loss.compute(ctx).mean(),
    }
    values["physics_unweighted"] = sum(values.values())

    pair_mask = base_ctx["atom_mask"].unsqueeze(-1) * base_ctx["atom_mask"].unsqueeze(-2)
    pair_mask = pair_mask * (1 - base_ctx["bond_mask"]) * (1 - base_ctx["angle_mask"])
    pair_mask = pair_mask * (dists < vdw_loss.cutoff).float()
    atom_vdw = __import__("model.loss", fromlist=["_vdw_radius_from_features"])._vdw_radius_from_features(
        base_ctx["atom_feats"]
    )
    lower = (atom_vdw.unsqueeze(-1) + atom_vdw.unsqueeze(-2)) * (1.0 - vdw_loss.buffer)
    triu = torch.triu(torch.ones(dists.shape[1], dists.shape[2], device=x.device), diagonal=1)
    penetration = torch.relu(lower - dists) * pair_mask * triu
    values["vdw_active_pairs"] = (penetration > 0).float().sum()
    values["vdw_max_penetration_A"] = penetration.max()

    return {k: float(v.detach().cpu().item()) for k, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--schedule", choices=["uniform", "log"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--guide-t-start", type=float, default=0.3)
    parser.add_argument("--guide-scale", type=float, default=1.0)
    parser.add_argument("--guide-schedule", type=str, default="linear_decay",
                        choices=["const", "linear_decay", "cosine_decay", "exp_decay"])
    parser.add_argument("--guide-bond-coef", type=float, default=1.0)
    parser.add_argument("--guide-angle-coef", type=float, default=1.0)
    parser.add_argument("--guide-vdw-coef", type=float, default=5.0)
    parser.add_argument("--guide-torsion-coef", type=float, default=1.0)
    parser.add_argument("--guide-grad-clip", type=float, default=0.0)
    args = parser.parse_args()

    sample_seed = args.seed + args.sample_index
    data = _to_device(load_system_data(args.system_dir), args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = FlexTrainer.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    if "ema_state_dict" in ckpt:
        shadow = ckpt["ema_state_dict"]["shadow"]
        for name, param in trainer.model.named_parameters():
            if name in shadow:
                param.data = shadow[name]
    del ckpt

    sampler = FlowMatchingSampler(trainer.model, device=args.device)
    batch = sampler._build_batch(data, 1)
    batch = {k: v.to(args.device) for k, v in batch.items()}
    rec_coords, lig_coords, _ = sampler._init_coords(data, 1, [sample_seed])
    rec_fixed_mask_3d = (
        batch["rec_backbone_mask"].unsqueeze(-1) + batch["rec_cb_mask"].unsqueeze(-1)
    ).clamp(max=1.0)

    base_ctx = _build_loss_context(batch, data, rec_fixed_mask_3d)
    bond_loss = BondLengthLoss(weight=1.0)
    angle_loss = BondAngleLoss(weight=1.0)
    vdw_loss = VdWClashLoss(weight=1.0)
    torsion_loss = TorsionLoss(weight=1.0)

    guidance_cfg = None
    guidance_engine = None
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
        guidance_engine = TweedieGuidance(
            config=guidance_cfg,
            batch=batch,
            rec_fixed_mask_3d=rec_fixed_mask_3d,
            device=args.device,
            rec_torsion_indices=data.get("rec_sp2_torsion_indices"),
            lig_torsion_indices=data.get("lig_sp2_torsion_indices"),
        )

    model = sampler.model
    q_chem, p_chem = model.forward_chemistry(
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

    timesteps = get_time_schedule(args.num_steps, args.schedule).to(args.device)
    rows = []

    with torch.no_grad():
        for i in tqdm(range(args.num_steps), desc="Tracing integration"):
            t = timesteps[i].expand(1)
            dt = timesteps[i + 1] - timesteps[i]
            output = model(
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

            rec_flow = output["rec_flow"]
            lig_flow = output["lig_flow"]
            rec_x1_hat = rec_coords + (1 - t).view(1, 1, 1) * rec_flow
            lig_x1_hat = lig_coords + (1 - t).view(1, 1, 1) * lig_flow
            rec_x1_hat = rec_fixed_mask_3d * rec_coords + (1 - rec_fixed_mask_3d) * rec_x1_hat

            x_t_before = torch.cat([rec_coords, lig_coords], dim=1)
            x1_before = torch.cat([rec_x1_hat, lig_x1_hat], dim=1)
            xt_vals = _loss_values(x_t_before, base_ctx, bond_loss, angle_loss, vdw_loss, torsion_loss)
            x1_vals = _loss_values(x1_before, base_ctx, bond_loss, angle_loss, vdw_loss, torsion_loss)

            guidance_active = False
            if guidance_engine is not None and guidance_engine.is_active(t):
                guidance_active = True
                if guidance_cfg.sim_mc_guidance:
                    one_minus_t_denom = (1 - t).view(1, 1, 1).clamp(min=guidance_cfg.one_minus_t_min)
                    rec_x1_hat, lig_x1_hat = guidance_engine.sim_mc_correct(rec_x1_hat, lig_x1_hat)
                    rec_flow = (rec_x1_hat - rec_coords) / one_minus_t_denom
                    lig_flow = (lig_x1_hat - lig_coords) / one_minus_t_denom
                else:
                    rec_g, lig_g = guidance_engine.guidance_velocity(rec_x1_hat, lig_x1_hat, t)
                    rec_flow = rec_flow + rec_g
                    lig_flow = lig_flow + lig_g

            rec_coords = rec_coords + rec_flow * dt * (1 - rec_fixed_mask_3d)
            lig_coords = lig_coords + lig_flow * dt
            x_t_after = torch.cat([rec_coords, lig_coords], dim=1)
            after_vals = _loss_values(x_t_after, base_ctx, bond_loss, angle_loss, vdw_loss, torsion_loss)

            row = {
                "step": i,
                "t": float(t.item()),
                "dt": float(dt.item()),
                "guidance_active": int(guidance_active),
            }
            row.update({f"xt_before_{k}": v for k, v in xt_vals.items()})
            row.update({f"x1_hat_{k}": v for k, v in x1_vals.items()})
            row.update({f"xt_after_{k}": v for k, v in after_vals.items()})
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
