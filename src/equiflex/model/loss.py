"""
Loss functions for Flex model training.

Includes:
- Flow matching loss (direct velocity MSE)
- Modular auxiliary losses on unrolled predictions:
  - Bond length (flat-bottom potential)
  - vdW clash (lower-bound on non-bonded distances)
  - Torsion/planarity (aromatic ring coplanarity)
- Standalone utilities: smooth LDDT, gramian loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import csv
from equiflex.prep.const import ATOM_CONTEXT_DIM



def linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Weight = t. Linear ramp, no suppression at low t."""
    return t



def _load_period_group_to_vdw_radius() -> torch.Tensor:
    table = torch.zeros((8, 20), dtype=torch.float32)
    csv_path = Path(__file__).parent.parent / "prep" / "periodictable.csv"
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            period = int(float(row["period"]))
            group = int(float(row["group"]))
            if 1 <= period <= 7 and 1 <= group <= 19:
                vdw_value = row.get("vdw_radius", "2.0")
                vdw_radius = float(vdw_value)
                table[period, group] = vdw_radius
            else:
                print(f"Invalid period or group: {period}, {group}")
    return table


PERIOD_GROUP_TO_VDW_RADIUS = _load_period_group_to_vdw_radius()

# Per-device copies of the radius table. Without the cache every vdW loss call
# issues a host-to-device copy of the table, which is both a per-step cost and a
# CPU-side op that prevents torch.compile from capturing the energy in a CUDA
# graph (inference guidance runs the energy at every integration step).
_VDW_RADIUS_BY_DEVICE: Dict[torch.device, torch.Tensor] = {}


def _vdw_radius_table(device: torch.device) -> torch.Tensor:
    table = _VDW_RADIUS_BY_DEVICE.get(device)
    if table is None:
        table = PERIOD_GROUP_TO_VDW_RADIUS.to(device)
        _VDW_RADIUS_BY_DEVICE[device] = table
    return table


def _vdw_radius_from_features(atom_feats: torch.Tensor) -> torch.Tensor:
    period_one_hot = atom_feats[..., ATOM_CONTEXT_DIM:ATOM_CONTEXT_DIM + 7]
    group_one_hot = atom_feats[..., ATOM_CONTEXT_DIM + 7:ATOM_CONTEXT_DIM + 7 + 19]

    period_valid = period_one_hot.sum(dim=-1) > 0
    group_valid = group_one_hot.sum(dim=-1) > 0
    period_idx = period_one_hot.argmax(dim=-1) + 1
    group_idx = group_one_hot.argmax(dim=-1) + 1

    period_idx = torch.where(period_valid, period_idx, period_idx.new_zeros(()))
    group_idx = torch.where(group_valid, group_idx, group_idx.new_zeros(()))

    lookup = _vdw_radius_table(atom_feats.device)
    return lookup[period_idx, group_idx]



def unroll_to_x1(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    t: torch.Tensor,
    rec_x_t: torch.Tensor,
    lig_x_t: torch.Tensor,
    num_unrolls: int = 4,
    q_chem: Optional[torch.Tensor] = None,
    p_chem: Optional[torch.Tensor] = None,
    init_rec_flow: Optional[torch.Tensor] = None,
    init_lig_flow: Optional[torch.Tensor] = None,
    rec_fixed_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Unroll FM model from x_t to x_1 prediction via Euler steps.

    The first Euler step is `model(x_t, t)`, which the trainer has already
    computed for the flow-matching loss. Pass it via init_rec_flow/init_lig_flow
    to reuse that forward (and its graph) instead of recomputing it — saves one
    model forward per training step, i.e. 1/(num_unrolls+1) of the unroll-related
    activations.

    rec_fixed_mask [B, N_rec] (1 = frozen backbone/CB) holds those receptor atoms
    fixed throughout the unroll, mirroring the sampler which never moves them.
    Without it the unrolled x_1 carries a drifted receptor, so the physics aux
    losses train on geometry inference never produces.
    """
    B = t.shape[0]
    dt = ((1 - t) / num_unrolls).view(B, 1, 1)

    rec_free = None
    if rec_fixed_mask is not None:
        rec_free = (1.0 - rec_fixed_mask).unsqueeze(-1).to(rec_x_t.dtype)

    rec_pred = rec_x_t.clone()
    lig_pred = lig_x_t.clone()
    t_curr = t.clone()

    for step in range(num_unrolls):
        if step == 0 and init_rec_flow is not None and init_lig_flow is not None:
            rec_flow_s, lig_flow_s = init_rec_flow, init_lig_flow
        else:
            output = model(
                rec_atom_feats=batch['rec_atom_feats'],
                rec_atom_coords=rec_pred,
                rec_atom_ref_coords=batch['rec_atom_ref_coords'],
                rec_constraint_matrix=batch.get('rec_constraint_matrix'),
                rec_bond_feats=batch['rec_bond_feats'],
                rec_uid_vec=batch['rec_uid_vec'],
                rec_atom_mask=batch['rec_atom_mask'],
                rec_backbone_mask=batch.get('rec_backbone_mask'),
                esm_embeddings=batch['esm_embeddings'],
                lig_atom_feats=batch['lig_atom_feats'],
                lig_atom_coords=lig_pred,
                lig_atom_ref_coords=batch['lig_atom_ref_coords'],
                lig_constraint_matrix=batch.get('lig_constraint_matrix'),
                lig_bond_feats=batch['lig_bond_feats'],
                lig_uid_vec=batch['lig_uid_vec'],
                lig_atom_mask=batch['lig_atom_mask'],
                unimol_embeddings=batch['unimol_embeddings'],
                t=t_curr,
                return_attn=False,
                q_chem=q_chem,
                p_chem=p_chem,
                rec_missing_mask=batch.get('rec_missing_mask'),
                system_index_per_timestep=batch.get('system_index_per_timestep'),
            )
            rec_flow_s, lig_flow_s = output['rec_flow'], output['lig_flow']
            del output

        rec_step = dt * rec_flow_s
        if rec_free is not None:
            rec_step = rec_step * rec_free
        rec_pred = rec_pred + rec_step
        lig_pred = lig_pred + dt * lig_flow_s
        t_curr = t_curr + dt.view(B)

    return {'rec_pred': rec_pred, 'lig_pred': lig_pred}



def smooth_lddt_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
    pred_dists: Optional[torch.Tensor] = None,
    true_dists: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Smooth LDDT loss. Returns 1 - LDDT per sample [B]."""
    B, N, _ = pred_coords.shape
    if pred_dists is None:
        pred_dists = torch.cdist(pred_coords, pred_coords)
    if true_dists is None:
        true_dists = torch.cdist(true_coords, true_coords)

    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    pair_mask = pair_mask * (1 - torch.eye(N, device=pred_coords.device))
    pair_mask = pair_mask * (true_dists < cutoff).float()

    dist_diff = torch.clamp(torch.abs(true_dists - pred_dists), 0.0, cutoff)
    eps = (
        F.sigmoid(0.5 - dist_diff) +
        F.sigmoid(1.0 - dist_diff) +
        F.sigmoid(2.0 - dist_diff) +
        F.sigmoid(4.0 - dist_diff)
    ) / 4.0

    num = (eps * pair_mask).sum(dim=(-1, -2))
    den = pair_mask.sum(dim=(-1, -2)).clamp(min=1)
    return 1.0 - num / den


def flow_matching_loss_x1(
    pred_x1: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flow matching loss with x1 parameterization and 1/(1-t)^2 weighting."""
    error = pred_x1 - x_1
    one_minus_t = (1 - t).clamp(min=1e-4)
    weight = (1.0 / (one_minus_t ** 2)).unsqueeze(-1).unsqueeze(-1)

    if mask is not None:
        mask = mask.unsqueeze(-1)
        weighted_squared_error = (error ** 2) * weight * mask
        return weighted_squared_error.sum() / mask.sum().clamp(min=1) / 3
    return ((error ** 2) * weight).mean()


def flow_matching_loss_direct(
    pred_flow: torch.Tensor,
    flow_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Direct flow matching loss: MSE(flow_pred, flow_gt)."""
    error = pred_flow - flow_gt
    if mask is not None:
        mask = mask.unsqueeze(-1)
        squared_error = (error ** 2) * mask
        return squared_error.sum() / mask.sum().clamp(min=1) / 3
    return (error ** 2).mean()


def quadratic_linear_penalty(deviation: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Quadratic near zero, linear past cutoff with continuous value/gradient."""
    if cutoff <= 0:
        return deviation ** 2
    cutoff_t = deviation.new_tensor(cutoff)
    quadratic = deviation ** 2
    linear = cutoff_t ** 2 + 2 * cutoff_t * (deviation - cutoff_t)
    return torch.where(deviation <= cutoff_t, quadratic, linear)


class AuxiliaryLoss(nn.Module):
    """Base class for auxiliary losses on unrolled predictions.

    Each subclass implements compute(ctx) -> per-sample loss [B].
    forward() applies timestep schedule, weight, and aux_scale,
    returning the exact contribution to the total loss.
    """

    def __init__(self, weight: float, schedule: Callable = linear_schedule):
        super().__init__()
        self.weight = weight
        self.schedule = schedule

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample raw loss [B]."""
        raise NotImplementedError

    def forward(self,
                t: torch.Tensor,
                ctx: Dict[str, torch.Tensor],
                aux_scale: float = 1.0,
                ) -> torch.Tensor:
        """Returns contribution to total loss (weight * schedule * aux_scale applied)."""
        raw = self.compute(ctx)  # [B]
        t_weight = self.schedule(t)  # [B]
        scaled = (raw * t_weight).mean()
        return self.weight * scaled * aux_scale


class BondLengthLoss(AuxiliaryLoss):
    """Harmonic flat-bottom potential on RDKit bond bounds with absolute slack buffer."""

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.05,
        huber_cutoff: float = 5.0,
        schedule: Callable = linear_schedule,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.huber_cutoff = huber_cutoff

    def compute(self, ctx):
        pred_dists = ctx['pred_dists']
        bond_mask = ctx['bond_mask']
        constraint_matrix = ctx['constraint_matrix']

        N = pred_dists.shape[1]
        triu_mask = torch.triu(torch.ones(N, N, device=pred_dists.device), diagonal=1)
        bond_mask_unique = bond_mask * triu_mask
        num_bonds = bond_mask_unique.sum(dim=(-1, -2)).clamp(min=1)
        bond_dists = pred_dists * bond_mask_unique

        if constraint_matrix.ndim == 3:
            constraint_matrix = constraint_matrix.unsqueeze(-1)
        lower_bounds = constraint_matrix.transpose(-2, -3).squeeze(-1) - self.buffer
        upper_bounds = constraint_matrix.squeeze(-1) + self.buffer
        deviation = F.relu(lower_bounds - bond_dists) + F.relu(bond_dists - upper_bounds)
        loss = quadratic_linear_penalty(deviation, self.huber_cutoff) * bond_mask_unique

        return loss.sum(dim=(-1, -2)) / num_bonds  # [B]


class BondAngleLoss(AuxiliaryLoss):
    """Harmonic flat-bottom potential on 1-3 pair distance bounds with absolute slack buffer."""

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.05,
        huber_cutoff: float = 5.0,
        schedule: Callable = linear_schedule,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.huber_cutoff = huber_cutoff

    def compute(self, ctx):
        pred_dists = ctx['pred_dists']
        angle_mask = ctx['angle_mask']
        constraint_matrix = ctx['constraint_matrix']

        N = pred_dists.shape[1]
        triu_mask = torch.triu(torch.ones(N, N, device=pred_dists.device), diagonal=1)
        angle_mask_unique = angle_mask * triu_mask
        num_pairs = angle_mask_unique.sum(dim=(-1, -2)).clamp(min=1)
        angle_dists = pred_dists * angle_mask_unique

        if constraint_matrix.ndim == 3:
            constraint_matrix = constraint_matrix.unsqueeze(-1)
        lower_bounds = constraint_matrix.transpose(-2, -3).squeeze(-1) - self.buffer
        upper_bounds = constraint_matrix.squeeze(-1) + self.buffer
        deviation = F.relu(lower_bounds - angle_dists) + F.relu(angle_dists - upper_bounds)
        loss = quadratic_linear_penalty(deviation, self.huber_cutoff) * angle_mask_unique

        return loss.sum(dim=(-1, -2)) / num_pairs  # [B]


def _vdw_clash_components(
    ctx: Dict[str, torch.Tensor],
    buffer: float,
    cutoff: float,
) -> Dict[str, torch.Tensor]:
    """Shared vdW clash tensors for mean/count/max loss variants."""
    pred_dists = ctx['pred_dists']
    atom_feats = ctx['atom_feats']
    mask = ctx['mask']
    bond_mask = ctx['bond_mask']
    angle_mask = ctx['angle_mask']
    N = pred_dists.shape[1]

    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    pair_mask = pair_mask * (1 - torch.eye(N, device=pred_dists.device))
    pair_mask = pair_mask * (1 - bond_mask) * (1 - angle_mask)
    # Exclude 1-4 (dihedral) pairs: their separation is set by torsion/angle
    # geometry, and a full vdW lower bound would spuriously push them apart.
    dihedral_mask = ctx.get('dihedral_mask')
    if dihedral_mask is not None:
        pair_mask = pair_mask * (1 - dihedral_mask)
    # Optional caller-supplied exclusion (inference guidance uses it to drop
    # frozen-frozen receptor pairs, which carry no force but do inflate the
    # energy and the 1/n_pairs normalisation). Unused during training.
    exclude_mask = ctx.get('nonbonded_exclude_mask')
    if exclude_mask is not None:
        pair_mask = pair_mask * (1 - exclude_mask)
    pair_mask = pair_mask * (pred_dists < cutoff).float()

    triu_mask = torch.triu(torch.ones(N, N, device=pred_dists.device), diagonal=1)
    pair_mask = pair_mask * triu_mask

    atom_vdw = _vdw_radius_from_features(atom_feats)
    lower_bounds = (atom_vdw.unsqueeze(-1) + atom_vdw.unsqueeze(-2)) * (1.0 - buffer)
    penetration = F.relu(lower_bounds - pred_dists) * pair_mask

    n_pairs = pair_mask.sum(dim=(-1, -2)).clamp(min=1).to(pred_dists.dtype)
    n_atoms = mask.sum(dim=-1).clamp(min=1).to(pred_dists.dtype)

    return {
        'pair_mask': pair_mask,
        'penetration': penetration,
        'n_pairs': n_pairs,
        'n_atoms': n_atoms,
    }


class VdWClashLoss(AuxiliaryLoss):
    """Mean squared vdW penetration over nearby nonbonded pairs.

    linear_onset=True (checkpoints up to equiflex_90m): (p+1)^2 - 1 = p^2 + 2p,
    a quadratic deep-clash ramp with slope 2 at onset so sub-A clashes are still
    pushed apart. linear_onset=False (v6/v7 configs): pure p^2, zero gradient at
    onset, shallow contacts left to the active-count term. Shared with inference
    guidance, which reads the flag back from the checkpoint's LossConfig.
    """

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.10,
        cutoff: float = 6.0,
        schedule: Callable = linear_schedule,
        linear_onset: bool = True,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.cutoff = cutoff
        self.linear_onset = linear_onset

    def compute(self, ctx):
        terms = _vdw_clash_components(ctx, self.buffer, self.cutoff)
        # relu in _vdw_clash_components => non-clashing pairs contribute exactly 0.
        penetration = terms['penetration']
        clash = (penetration + 1.0) ** 2 - 1.0 if self.linear_onset else penetration ** 2
        return clash.sum(dim=(-1, -2)) / terms['n_pairs']


class VdWHingeLoss(AuxiliaryLoss):
    """Unified vdW clash term: hinge penalty on every nonbonded pair, per ligand atom.

        L = [ sum_{lig-rec} h(p) + sum_{lig-lig} h(p) + rec_rec_ratio * sum_{rec-rec} h(p) ] / n_lig,
        h(p) = (p + 1)^2 - 1 = p^2 + 2p,   p = relu((r_i + r_j)(1 - buffer) - d_ij)

    Replaces the mean (sum p^2 / n_pairs) and active-count terms. Measured on
    v7 at step 17k: the mean term averaged ligand clashes away (ligand-receptor
    pairs are ~6% of pocket pairs; 72% of v7 poses clashed with the protein at
    the same mean-term value as the 90M's 26%), and the active term saturated.
    The hinge has slope 2 at onset (half the observed clashes are < 0.3 A deep),
    per-ligand-atom normalisation keeps ligand clashes undiluted, and
    rec_rec_ratio keeps side-chain packing in the loss at a controlled share.
    Receptor pairs whose BOTH atoms are frozen (backbone, CB: ctx['frozen_mask'])
    are dropped: they never move at inference, carry no gradient in training,
    and on PoseBusters crystal poses they were the entire baseline (median 0.34
    with them, 0.000 without; v7 poses 0.13 either way).
    Needs ctx['n_rec']; without it every pair counts as ligand-involved.
    """

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.20,
        cutoff: float = 6.0,
        rec_rec_ratio: float = 0.15,
        schedule: Callable = linear_schedule,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.cutoff = cutoff
        self.rec_rec_ratio = rec_rec_ratio

    def compute(self, ctx):
        terms = _vdw_clash_components(ctx, self.buffer, self.cutoff)
        penetration = terms['penetration']                 # [B, N, N], upper triangle, masked
        pair_mask = terms['pair_mask']
        clash = ((penetration + 1.0) ** 2 - 1.0) * pair_mask
        B, N, _ = clash.shape
        mask = ctx['mask']
        n_rec = ctx.get('n_rec')
        if n_rec is None or n_rec <= 0 or n_rec >= N:
            class_w = torch.ones(N, N, device=clash.device, dtype=clash.dtype)
            n_lig = mask.sum(dim=-1).clamp(min=1)
        else:
            is_lig = torch.zeros(N, device=clash.device, dtype=clash.dtype)
            is_lig[n_rec:] = 1.0
            rec_rec = (1 - is_lig).unsqueeze(-1) * (1 - is_lig).unsqueeze(-2)
            class_w = 1.0 - (1.0 - self.rec_rec_ratio) * rec_rec
            n_lig = (mask * is_lig).sum(dim=-1).clamp(min=1)
            frozen = ctx.get('frozen_mask')
            if frozen is not None:                                    # [B, N], 1 = backbone/CB
                frozen = frozen.to(clash.dtype)
                class_w = class_w * (1.0 - frozen.unsqueeze(-1) * frozen.unsqueeze(-2))
        return (clash * class_w).sum(dim=(-1, -2)) / n_lig


DONOR_COL, ACCEPTOR_COL = 169, 170   # pharmacophore flags in the 299-dim atom feature (featurize.py context block)


class HBondContactLoss(AuxiliaryLoss):
    """Native polar-contact recovery between ligand and receptor.

    From the crystal x_1 take ligand-receptor donor-acceptor heavy-atom pairs
    (donor on one side, acceptor on the other, feature columns 169/170) closer
    than `native_cutoff`; on the unrolled x_1_pred penalise losing them:

        L = (1 / n_native) * sum_{(D,A) native} huber( relu(d_pred(D,A) - native_cutoff) )

    huber(e) = e^2 for e <= huber_cutoff, else huber_cutoff * (2e - huber_cutoff):
    quadratic for contacts that are nearly kept, linear for a mis-docked ligand
    whose contacts all sit 5-10 A away (a pure quadratic gave a loss mean 5x
    its median on v7 poses and would out-shout the flow loss exactly there).
    Supervised by the crystal contact set, so it only penalises breaking real
    H-bonds and never rewards inventing new ones (no collapse risk, no guidance
    analogue). PoseBusters crystals have median 7 such contacts (p10 2, p90 16);
    systems with none contribute 0. Needs ctx['x_1'] and ctx['n_rec'].
    """

    def __init__(self, weight: float = 1.0, native_cutoff: float = 3.5,
                 huber_cutoff: float = 1.0, schedule: Callable = linear_schedule):
        super().__init__(weight, schedule)
        self.native_cutoff = native_cutoff
        self.huber_cutoff = huber_cutoff

    def compute(self, ctx):
        x_1_pred = ctx['x_1_pred']; x_1 = ctx.get('x_1'); n_rec = ctx.get('n_rec')
        B, N, _ = x_1_pred.shape
        if x_1 is None or n_rec is None or n_rec <= 0 or n_rec >= N:
            return x_1_pred.new_zeros(B)
        feats = ctx['atom_feats']                                  # [B or 1, N, F]
        don, acc = feats[..., DONOR_COL], feats[..., ACCEPTOR_COL]
        if don.shape[0] == 1 and B > 1:
            don, acc = don.expand(B, -1), acc.expand(B, -1)
        is_lig = torch.zeros(N, device=x_1_pred.device, dtype=x_1_pred.dtype); is_lig[n_rec:] = 1.0
        polar = (don.unsqueeze(-1) * acc.unsqueeze(-2) + acc.unsqueeze(-1) * don.unsqueeze(-2)).clamp(max=1.0)
        cross = (is_lig.unsqueeze(-1) * (1 - is_lig).unsqueeze(-2))   # ligand row, receptor column
        mask = ctx['mask']
        pair_mask = polar * cross * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        d_gt = torch.cdist(x_1, x_1)
        native = pair_mask * (d_gt <= self.native_cutoff).to(x_1_pred.dtype)
        d_pred = ctx['pred_dists']
        excess = F.relu(d_pred - self.native_cutoff)
        c = self.huber_cutoff
        penalty = torch.where(excess <= c, excess ** 2, c * (2 * excess - c))
        loss = (penalty * native).sum(dim=(-1, -2))
        n_native = native.sum(dim=(-1, -2))
        return torch.where(n_native > 0, loss / n_native.clamp(min=1), loss.new_zeros(B))


class VdWActiveClashLoss(AuxiliaryLoss):
    """Soft active-clash count normalized by valid atoms.

    This complements VdWClashLoss: the mean-pair loss can hide rare clashes in
    large pockets, while this term grows with the number of violating pairs.
    """

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.10,
        cutoff: float = 6.0,
        penetration_scale: float = 0.10,
        schedule: Callable = linear_schedule,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.cutoff = cutoff
        self.penetration_scale = penetration_scale

    def compute(self, ctx):
        terms = _vdw_clash_components(ctx, self.buffer, self.cutoff)
        scale = max(float(self.penetration_scale), 1e-6)
        soft_active = 1.0 - torch.exp(-((terms['penetration'] / scale) ** 2))
        soft_active = soft_active * terms['pair_mask']
        return soft_active.sum(dim=(-1, -2)) / terms['n_atoms']


class VdWMaxPenetrationLoss(AuxiliaryLoss):
    """Soft maximum vdW penetration, squared to match distance-style penalties."""

    def __init__(
        self,
        weight: float = 1.0,
        buffer: float = 0.10,
        cutoff: float = 6.0,
        alpha: float = 20.0,
        schedule: Callable = linear_schedule,
    ):
        super().__init__(weight, schedule)
        self.buffer = buffer
        self.cutoff = cutoff
        self.alpha = alpha

    def compute(self, ctx):
        terms = _vdw_clash_components(ctx, self.buffer, self.cutoff)
        penetration = terms['penetration']
        pair_mask = terms['pair_mask'].bool()
        B = penetration.shape[0]
        flat_pen = penetration.reshape(B, -1)
        flat_mask = pair_mask.reshape(B, -1)

        alpha = max(float(self.alpha), 1e-6)
        masked = flat_pen.masked_fill(~flat_mask, -torch.inf)
        lse = torch.logsumexp(alpha * masked, dim=-1)
        # Subtract the all-zero baseline so no-clash structures get exactly zero.
        baseline = torch.log(terms['n_pairs'])
        soft_max = ((lse - baseline) / alpha).clamp(min=0.0)
        soft_max = torch.where(torch.isfinite(soft_max), soft_max, torch.zeros_like(soft_max))
        return soft_max ** 2


class TorsionLoss(AuxiliaryLoss):
    """Planarity loss for aromatic ring torsions."""

    def __init__(self, weight: float = 1.0, schedule: Callable = linear_schedule):
        super().__init__(weight, schedule)

    def _torsion_planarity(self, coords, torsion_indices, mask=None):
        """Per-sample planarity loss from a single set of torsion indices."""
        B = coords.shape[0]
        if torsion_indices.numel() == 0:
            return coords.new_zeros(B), False

        batch_idx = torsion_indices[:, 0].long()
        atom_indices = torsion_indices[:, 1:].long()
        if mask is not None:
            valid_torsion = mask[batch_idx.unsqueeze(-1), atom_indices].bool().all(dim=-1)
            if not valid_torsion.any():
                return coords.new_zeros(B), False
            batch_idx = batch_idx[valid_torsion]
            atom_indices = atom_indices[valid_torsion]

        p0 = coords[batch_idx, atom_indices[:, 0]]
        p1 = coords[batch_idx, atom_indices[:, 1]]
        p2 = coords[batch_idx, atom_indices[:, 2]]
        p3 = coords[batch_idx, atom_indices[:, 3]]

        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        n1_norm = n1.norm(dim=-1).clamp(min=1e-8)
        n2_norm = n2.norm(dim=-1).clamp(min=1e-8)
        cos_angle = (n1 * n2).sum(dim=-1) / (n1_norm * n2_norm)
        # Signed sin via projection of (n1 x n2) onto b2
        b2_hat = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sin_angle = torch.cross(n1, n2, dim=-1).mul(b2_hat).sum(dim=-1) / (n1_norm * n2_norm)
        # atan2 gives angle from planar; penalize theta^2
        theta = torch.atan2(sin_angle.abs(), cos_angle.abs())
        per_torsion_loss = theta ** 2

        loss_per_sample = coords.new_zeros(B)
        count_per_sample = coords.new_zeros(B)
        loss_per_sample.scatter_add_(0, batch_idx, per_torsion_loss)
        count_per_sample.scatter_add_(0, batch_idx, torch.ones_like(per_torsion_loss))

        return loss_per_sample / count_per_sample.clamp(min=1), True

    def compute(self, ctx):
        x_1_pred = ctx['x_1_pred']
        mask = ctx.get('mask')
        B = x_1_pred.shape[0]
        loss = x_1_pred.new_zeros(B)
        count = 0

        rec_idx = ctx.get('rec_torsion_indices')
        if rec_idx is not None and rec_idx.numel() > 0:
            rec_loss, has_rec_torsions = self._torsion_planarity(x_1_pred, rec_idx, mask)
            if has_rec_torsions:
                loss = loss + rec_loss
                count += 1

        lig_idx = ctx.get('lig_torsion_indices')
        if lig_idx is not None and lig_idx.numel() > 0:
            if ctx.get('n_rec') is not None:
                lig_idx = lig_idx.clone()
                lig_idx[:, 1:] += ctx['n_rec']
            lig_loss, has_lig_torsions = self._torsion_planarity(x_1_pred, lig_idx, mask)
            if has_lig_torsions:
                loss = loss + lig_loss
                count += 1

        if count > 0:
            loss = loss / count
        return loss  # [B]


class ChiralAtomLoss(AuxiliaryLoss):
    """Flat-bottom restraint on the SIGNED improper dihedral at each stereocentre.

    A Cartesian generative model has nothing stopping it from emitting the mirror
    image: torsion-space docking cannot invert a stereocentre, we can.  Every
    parity-EVEN quantity (distances, |angles|, the planarity term in TorsionLoss)
    is blind to this, so the restraint has to be built on a pseudoscalar.

    phi is the dihedral over (n0, n1, n2, centre) where n0..n2 are the three
    highest-CIP-rank substituents.  Its sign comes from the scalar triple product
    b2 . (n_a x n_b), which flips under reflection -- that is the whole point.
    R centres are required to sit at phi > +buffer, S centres at phi < -buffer;
    the buffer keeps the restraint away from the planar phi=0 transition state
    rather than merely on the correct side of it.

    Index format [K, 6]: (batch_idx, a0, a1, a2, centre, is_r).  Same convention
    as TorsionLoss but with the orientation flag appended.
    """

    def __init__(self, weight: float = 1.0, buffer: float = 0.5236,
                 schedule: Callable = linear_schedule):
        super().__init__(weight, schedule)
        self.buffer = buffer  # 0.5236 rad = 30 deg, matching Boltz

    def _signed_dihedral(self, coords, batch_idx, atom_indices):
        p0 = coords[batch_idx, atom_indices[:, 0]]
        p1 = coords[batch_idx, atom_indices[:, 1]]
        p2 = coords[batch_idx, atom_indices[:, 2]]
        p3 = coords[batch_idx, atom_indices[:, 3]]

        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        n1_norm = n1.norm(dim=-1).clamp(min=1e-8)
        n2_norm = n2.norm(dim=-1).clamp(min=1e-8)
        cos_angle = (n1 * n2).sum(dim=-1) / (n1_norm * n2_norm)
        b2_hat = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sin_angle = torch.cross(n1, n2, dim=-1).mul(b2_hat).sum(dim=-1) / (n1_norm * n2_norm)
        # atan2, not arccos: keeps the sign and has no gradient singularity at +-1
        return torch.atan2(sin_angle, cos_angle)

    def compute(self, ctx):
        x_1_pred = ctx['x_1_pred']
        B = x_1_pred.shape[0]
        idx = ctx.get('lig_chiral_indices')
        if idx is None or idx.numel() == 0:
            return x_1_pred.new_zeros(B)

        idx = idx.clone()
        if ctx.get('n_rec') is not None:
            idx[:, 1:5] += ctx['n_rec']

        batch_idx = idx[:, 0].long()
        atom_indices = idx[:, 1:5].long()
        is_r = idx[:, 5].bool()

        mask = ctx.get('mask')
        if mask is not None:
            valid = mask[batch_idx.unsqueeze(-1), atom_indices].bool().all(dim=-1)
            if not valid.any():
                return x_1_pred.new_zeros(B)
            batch_idx, atom_indices, is_r = batch_idx[valid], atom_indices[valid], is_r[valid]

        phi = self._signed_dihedral(x_1_pred, batch_idx, atom_indices)

        # hinge: R wants phi > +buffer, S wants phi < -buffer; flat inside
        violation = torch.where(is_r, self.buffer - phi, phi + self.buffer)
        per_centre = violation.clamp(min=0.0)

        loss = x_1_pred.new_zeros(B)
        count = x_1_pred.new_zeros(B)
        loss.scatter_add_(0, batch_idx, per_centre)
        count.scatter_add_(0, batch_idx, torch.ones_like(per_centre))
        return loss / count.clamp(min=1)



@dataclass
class LossConfig:
    """Configuration for loss functions."""

    # Unrolling
    num_unrolls: int = 2
    # Curriculum: ramp unroll depth 1 -> num_unrolls over training in
    # num_unrolls equal stages (PBFM-style). The trainer passes the per-step
    # value into FlexLoss.forward; this flag only documents intent.
    unroll_curriculum: bool = False

    # Flow
    flow_weight: float = 1.0

    # Bond length
    bond_length_weight: float = 1.0
    bond_buffer: float = 0.05   # absolute slack (Å) absorb XRD vs RDKit envelope
    bond_huber_cutoff: float = 5.0

    # Bond angle (1-3 pairs)
    bond_angle_weight: float = 0.0
    angle_buffer: float = 0.05
    angle_huber_cutoff: float = 5.0

    # vdW clash
    vdw_weight: float = 100.0
    vdw_active_weight: float = 0.0
    vdw_max_weight: float = 0.0
    vdw_buffer: float = 0.10
    vdw_cutoff: float = 6.0   # Å — only pairs closer than this contribute
    vdw_linear_onset: bool = True   # p^2 + 2p (legacy) vs pure p^2 mean term; see VdWClashLoss
    # vdw_form 'mean': VdWClashLoss (sum p^2 / n_pairs); 'hinge': VdWHingeLoss,
    # per-ligand-atom hinge over all pairs with rec-rec at vdw_rec_rec_ratio
    vdw_form: str = 'mean'
    vdw_rec_rec_ratio: float = 0.15

    # Native ligand-receptor H-bond contact recovery (HBondContactLoss)
    hbond_weight: float = 0.0
    hbond_native_cutoff: float = 3.5
    hbond_huber_cutoff: float = 1.0
    vdw_active_scale: float = 0.10  # Å scale where soft active-count term saturates
    vdw_max_alpha: float = 20.0     # logsumexp sharpness for soft max penetration

    # Torsion/planarity
    torsion_weight: float = 1.0

    # Ligand stereocentres (ChiralAtomLoss); needs chiral_atom_indices in the
    # featurised data
    chiral_weight: float = 0.0
    chiral_buffer: float = 0.5236   # rad; 30 deg margin from the planar state


class FlexLoss(nn.Module):
    """Combined loss for Flex: flow matching + modular auxiliary losses.

    Each auxiliary loss is an nn.Module with its own weight and timestep schedule.
    losses dict values are exact contributions to total, so total == sum(losses).
    """

    def __init__(self, config: LossConfig, schedule: Callable = linear_schedule):
        super().__init__()
        self.config = config

        self.aux_losses = nn.ModuleDict()
        if config.bond_length_weight > 0:
            self.aux_losses['bond_length'] = BondLengthLoss(
                config.bond_length_weight, config.bond_buffer,
                config.bond_huber_cutoff, schedule,
            )
        if config.bond_angle_weight > 0:
            self.aux_losses['bond_angle'] = BondAngleLoss(
                config.bond_angle_weight, config.angle_buffer,
                config.angle_huber_cutoff, schedule,
            )
        if config.vdw_weight > 0 and getattr(config, 'vdw_form', 'mean') == 'hinge':
            self.aux_losses['vdw'] = VdWHingeLoss(
                config.vdw_weight, config.vdw_buffer, config.vdw_cutoff,
                getattr(config, 'vdw_rec_rec_ratio', 0.15), schedule,
            )
        elif config.vdw_weight > 0:
            self.aux_losses['vdw'] = VdWClashLoss(
                config.vdw_weight, config.vdw_buffer, config.vdw_cutoff,
                schedule, linear_onset=getattr(config, 'vdw_linear_onset', True),
            )
        if config.vdw_active_weight > 0:
            self.aux_losses['vdw_active'] = VdWActiveClashLoss(
                config.vdw_active_weight, config.vdw_buffer, config.vdw_cutoff,
                config.vdw_active_scale, schedule,
            )
        if config.vdw_max_weight > 0:
            self.aux_losses['vdw_max'] = VdWMaxPenetrationLoss(
                config.vdw_max_weight, config.vdw_buffer, config.vdw_cutoff,
                config.vdw_max_alpha, schedule,
            )
        if getattr(config, 'hbond_weight', 0.0) > 0:
            self.aux_losses['hbond'] = HBondContactLoss(
                config.hbond_weight, getattr(config, 'hbond_native_cutoff', 3.5),
                getattr(config, 'hbond_huber_cutoff', 1.0), schedule,
            )
        if config.torsion_weight > 0:
            self.aux_losses['torsion'] = TorsionLoss(
                config.torsion_weight, schedule,
            )
        if getattr(config, 'chiral_weight', 0.0) > 0:
            self.aux_losses['chiral'] = ChiralAtomLoss(
                config.chiral_weight, config.chiral_buffer, schedule,
            )

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        pred_flow: torch.Tensor,
        flow_gt: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        rec_x_t: torch.Tensor,
        lig_x_t: torch.Tensor,
        mask: torch.Tensor,
        bond_mask: torch.Tensor,
        constraint_matrix: Optional[torch.Tensor] = None,
        rec_torsion_indices: Optional[torch.Tensor] = None,
        lig_torsion_indices: Optional[torch.Tensor] = None,
        lig_chiral_indices: Optional[torch.Tensor] = None,
        n_rec: Optional[int] = None,
        backbone_mask: Optional[torch.Tensor] = None,
        cb_mask: Optional[torch.Tensor] = None,
        aux_scale: float = 1.0,
        q_chem: Optional[torch.Tensor] = None,
        p_chem: Optional[torch.Tensor] = None,
        num_unrolls: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        zero = pred_flow.new_tensor(0.0)

        # Curriculum override: trainer may pass a smaller unroll depth early in
        # training. Falls back to the configured maximum.
        effective_unrolls = self.config.num_unrolls if num_unrolls is None else num_unrolls

        # Flow loss: exclude backbone and CB atoms (both fixed during generation)
        flow_mask = mask
        if backbone_mask is not None:
            flow_mask = flow_mask * (1 - backbone_mask)
        if cb_mask is not None:
            flow_mask = flow_mask * (1 - cb_mask)
        flow_contribution = zero
        if self.config.flow_weight > 0:
            flow_loss = flow_matching_loss_direct(pred_flow, flow_gt, flow_mask)
            flow_contribution = self.config.flow_weight * flow_loss
            losses['flow'] = flow_contribution

        # aux_scale == 0 -> skip unroll + aux entirely (saves ~num_unrolls extra
        # model forwards/backwards). Used during phys-warmup under ConFIG where
        # pre-merge magnitude scaling has no effect on combined direction.
        skip_aux = (aux_scale == 0.0) or (not self.aux_losses) or (effective_unrolls <= 0)

        if skip_aux:
            for name in self.aux_losses.keys():
                losses[name] = zero
            losses['flow_group'] = flow_contribution
            losses['physics_group'] = zero
            losses['total'] = flow_contribution
            return losses

        # Reuse the flow-matching forward as the unroll's first Euler step:
        # pred_flow == model(x_t, t), identical to unroll step 0. Split by n_rec.
        init_rec_flow = init_lig_flow = None
        if n_rec is not None:
            init_rec_flow = pred_flow[:, :n_rec]
            init_lig_flow = pred_flow[:, n_rec:]

        # Freeze backbone+CB during the unroll, exactly as the sampler does, so
        # the unrolled x_1 (and the physics losses on it) reflect the geometry
        # inference actually produces rather than a drifted receptor.
        N_rec = rec_x_t.shape[1]
        rec_fixed_mask = None
        if backbone_mask is not None or cb_mask is not None:
            rec_fixed_mask = rec_x_t.new_zeros(rec_x_t.shape[0], N_rec)
            if backbone_mask is not None:
                rec_fixed_mask = rec_fixed_mask + backbone_mask[:, :N_rec]
            if cb_mask is not None:
                rec_fixed_mask = rec_fixed_mask + cb_mask[:, :N_rec]
            rec_fixed_mask = rec_fixed_mask.clamp(max=1.0)

        rec_fixed_full = None
        if backbone_mask is not None or cb_mask is not None:
            rec_fixed_full = mask.new_zeros(mask.shape)
            if backbone_mask is not None:
                rec_fixed_full = rec_fixed_full + backbone_mask
            if cb_mask is not None:
                rec_fixed_full = rec_fixed_full + cb_mask
            rec_fixed_full = rec_fixed_full.clamp(max=1.0)

        unrolled = unroll_to_x1(
            model, batch, t, rec_x_t, lig_x_t, effective_unrolls,
            q_chem=q_chem, p_chem=p_chem,
            init_rec_flow=init_rec_flow, init_lig_flow=init_lig_flow,
            rec_fixed_mask=rec_fixed_mask,
        )
        x_1_pred = torch.cat([unrolled['rec_pred'], unrolled['lig_pred']], dim=1)

        # Precompute shared tensors. `bond_mask` is the precomputed block-diagonal
        # single-channel mask from the trainer (same as `(bond_feats.abs().sum(-1) > 0)`).
        bond_mask_mat = bond_mask.float() if bond_mask.dtype != torch.float32 else bond_mask
        N = bond_mask_mat.shape[1]

        # Remove topology terms touching missing atoms.  Building the angle mask
        # from valid bonds also removes i-k 1,3 terms whose middle atom is missing.
        atom_pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        if bond_mask_mat.shape[0] == 1 and mask.shape[0] > 1:
            bond_mask_mat = bond_mask_mat.expand(mask.shape[0], -1, -1)
        bond_mask_mat = bond_mask_mat * atom_pair_mask

        # Covalent ligand-receptor links (N-glycans, covalent inhibitors, PLP adducts: 12%
        # of PLINDER training systems have a ligand heavy atom < 1.8 A from the receptor).
        # The block-diagonal bond features do not know them, so without this the vdW terms
        # see a ~1.2 A "clash" in the crystal pose and the H-bond term a fake contact.
        # Treat crystal cross-interface pairs < 1.8 A as bonds; the 1-3 / 1-4 exclusions
        # below then propagate across the link.
        if x_1 is not None and n_rec is not None and 0 < n_rec < N:
            with torch.no_grad():
                cross = torch.cdist(x_1[:, :n_rec], x_1[:, n_rec:]) < 1.8          # [B, n_rec, n_lig]
            if cross.any():
                link = torch.zeros_like(bond_mask_mat)
                link[:, :n_rec, n_rec:] = cross.to(link.dtype)
                link[:, n_rec:, :n_rec] = cross.transpose(1, 2).to(link.dtype)
                bond_mask_mat = (bond_mask_mat + link * atom_pair_mask).clamp(max=1.0)

        eye = torch.eye(N, device=bond_mask_mat.device)
        bond_mask_13 = torch.bmm(bond_mask_mat, bond_mask_mat).clamp(max=1.0)
        bond_mask_13 = bond_mask_13 * (1 - eye)
        angle_mask = bond_mask_13 * (1 - bond_mask_mat)

        # 1-4 (dihedral) pairs: reachable in 3 bonds but not 1-2 or 1-3.
        bond_mask_14 = torch.bmm(bond_mask_13, bond_mask_mat).clamp(max=1.0)
        bond_mask_14 = bond_mask_14 * (1 - eye)
        bond_mask_14 = bond_mask_14 * (1 - bond_mask_mat) * (1 - angle_mask)

        pred_dists = torch.cdist(x_1_pred, x_1_pred)

        ctx = {
            'x_1_pred': x_1_pred,
            'pred_dists': pred_dists,
            'bond_mask': bond_mask_mat,
            'angle_mask': angle_mask,
            'dihedral_mask': bond_mask_14,
            'mask': mask,
            'atom_feats': torch.cat([batch['rec_atom_feats'], batch['lig_atom_feats']], dim=1),
            'constraint_matrix': constraint_matrix,
            'rec_torsion_indices': rec_torsion_indices,
            'lig_torsion_indices': lig_torsion_indices,
            'lig_chiral_indices': lig_chiral_indices,
            'n_rec': n_rec,
            'frozen_mask': rec_fixed_full,
            'x_1': x_1,
        }

        # All aux losses depend on x_1_pred -> unroll graph -> physics_group.
        physics_group = zero
        for name, loss_fn in self.aux_losses.items():
            contribution = loss_fn(t=t, ctx=ctx, aux_scale=aux_scale)
            losses[name] = contribution
            physics_group = physics_group + contribution

        flow_group = flow_contribution
        total = flow_group + physics_group
        losses['flow_group'] = flow_group
        losses['physics_group'] = physics_group
        losses['total'] = total
        return losses
