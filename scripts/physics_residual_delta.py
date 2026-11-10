"""Per-sample physics residuals of integrated poses and their paired deltas between two runs.

Both runs must have been sampled with the same seeds, so sample k of system s starts from
the same x_0 in each. The residuals are the raw (unweighted) auxiliary losses of the
training objective, evaluated on `results.pt` final coordinates with the exact classes
and hyper-parameters of `FlexLoss` (loss section of the training config):

    bond_length, bond_angle, vdw (hinge), torsion (planarity), chiral

Delta = run A - run B, so Delta > 0 means run A leaves a LARGER residual.

    python scripts/physics_residual_delta.py \
        --a v10_nophys_41k_off_5A --b v10_41k_off_5A --config configs/v10_512.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from equiflex.inference.sampler import FlowMatchingSampler  # noqa: E402
from equiflex.model.loss import FlexLoss, LossConfig  # noqa: E402
from run_inference import load_system_data  # noqa: E402

TERMS = ['bond_length', 'bond_angle', 'vdw', 'torsion', 'chiral']
PB = ROOT / 'data/posebusters/v7'


def with_batch_col(idx, B, device):
    """[K, c] per-molecule indices -> [B*K, 1+c] with a leading batch column."""
    idx = idx.to(device).long()
    k = idx.shape[0]
    col = torch.arange(B, device=device).repeat_interleave(k).unsqueeze(-1)
    return torch.cat([col, idx.repeat(B, 1)], dim=-1)


def build_ctx(data, rec, lig):
    """Loss context of FlexLoss.forward for B final poses, minus the unroll and covalent links."""
    device = rec.device
    B = rec.shape[0]
    batch = FlowMatchingSampler._build_batch(None, data, B)
    batch = {k: v.to(device) for k, v in batch.items()}
    n_rec, n_lig = rec.shape[1], lig.shape[1]
    N = n_rec + n_lig

    def blocks(rec_t, lig_t):
        out = torch.zeros(B, N, N, device=device)
        out[:, :n_rec, :n_rec] = rec_t
        out[:, n_rec:, n_rec:] = lig_t
        return out

    mask = torch.cat([batch['rec_atom_mask'].float(), batch['lig_atom_mask'].float()], dim=1)
    mask = mask * (~torch.cat([batch['rec_missing_mask'], torch.zeros_like(batch['lig_atom_mask'])], dim=1)).float()
    pair = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    bond = blocks((batch['rec_bond_feats'].abs().sum(-1) > 0).float(),
                  (batch['lig_bond_feats'].abs().sum(-1) > 0).float()) * pair
    eye = torch.eye(N, device=device)
    b13 = torch.bmm(bond, bond).clamp(max=1.0) * (1 - eye)
    angle = b13 * (1 - bond)
    dihedral = torch.bmm(b13, bond).clamp(max=1.0) * (1 - eye) * (1 - bond) * (1 - angle)

    cm = [batch[k].squeeze(-1) for k in ('rec_constraint_matrix', 'lig_constraint_matrix')]
    frozen = torch.cat([(batch['rec_backbone_mask'] + batch['rec_cb_mask']).clamp(max=1.0),
                        torch.zeros(B, n_lig, device=device)], dim=1)
    x = torch.cat([rec, lig], dim=1)
    return {
        'x_1_pred': x, 'pred_dists': torch.cdist(x, x),
        'bond_mask': bond, 'angle_mask': angle, 'dihedral_mask': dihedral, 'mask': mask,
        'atom_feats': torch.cat([batch['rec_atom_feats'], batch['lig_atom_feats']], dim=1).expand(B, -1, -1),
        'constraint_matrix': blocks(cm[0], cm[1]),
        'rec_torsion_indices': with_batch_col(data['rec_sp2_torsion_indices'], B, device),
        'lig_torsion_indices': with_batch_col(data['lig_sp2_torsion_indices'], B, device),
        'lig_chiral_indices': with_batch_col(data['lig_chiral_indices'], B, device),
        'n_rec': n_rec, 'frozen_mask': frozen,
    }


@torch.no_grad()
def residuals(run, system, loss_fn, data, device):
    res = torch.load(PB / run / system / 'results.pt', map_location=device, weights_only=False)
    ctx = build_ctx(data, res['rec_coords_final'].float(), res['lig_coords_final'].float())
    return {t: loss_fn.aux_losses[t].compute(ctx).cpu().numpy() for t in TERMS}, res['seeds']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--a', required=True, help='run A (e.g. the no-physics ablation)')
    ap.add_argument('--b', required=True, help='run B (reference)')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/v10_512.yaml')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())['loss']
    loss_cfg = LossConfig(**{k: v for k, v in cfg.items() if k in LossConfig.__dataclass_fields__})
    loss_fn = FlexLoss(loss_cfg).to(args.device)
    weights = {t: loss_fn.aux_losses[t].weight for t in TERMS}

    systems = sorted(p.name for p in (PB / args.a).iterdir() if (p / 'results.pt').exists()
                     and (PB / args.b / p.name / 'results.pt').exists())
    rows = []
    for i, s in enumerate(systems):
        data = load_system_data(PB / args.b / s / 'system_data')
        data = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        ra, seeds_a = residuals(args.a, s, loss_fn, data, args.device)
        rb, seeds_b = residuals(args.b, s, loss_fn, data, args.device)
        assert seeds_a == seeds_b, f'{s}: seeds differ {seeds_a} vs {seeds_b}'
        for k, seed in enumerate(seeds_a):
            row = {'system': s, 'sample': f'sample_{k}', 'seed': seed}
            for t in TERMS:
                row[f'{t}_a'], row[f'{t}_b'] = ra[t][k], rb[t][k]
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(systems)}', flush=True)

    df = pd.DataFrame(rows)
    for t in TERMS:
        df[f'{t}_delta'] = df[f'{t}_a'] - df[f'{t}_b']
    for side in 'ab':
        df[f'weighted_{side}'] = sum(weights[t] * df[f'{t}_{side}'] for t in TERMS)
    df['weighted_delta'] = df.weighted_a - df.weighted_b

    out = args.out or ROOT / f'data/posebusters/v7/{args.a}/physics_residual_delta_vs_{args.b}.csv'
    df.to_csv(out, index=False)
    print(f'{len(df)} pairs, {df.system.nunique()} systems -> {out}\nweights {weights}')

    print(f'\nDelta = {args.a} - {args.b}  (positive: A leaves a larger residual)')
    summary = []
    for t in TERMS + ['weighted']:
        d, a, b = df[f'{t}_delta'], df[f'{t}_a' if t != 'weighted' else 'weighted_a'], df[f'{t}_b' if t != 'weighted' else 'weighted_b']
        per_sys = df.groupby('system')[f'{t}_delta'].mean()
        summary.append({
            'term': t, 'mean_a': a.mean(), 'mean_b': b.mean(), 'median_a': a.median(), 'median_b': b.median(),
            'delta_mean': d.mean(), 'delta_median': d.median(),
            'delta_q05': d.quantile(.05), 'delta_q95': d.quantile(.95),
            '%A>B': 100 * (d > 0).mean(), '%tie': 100 * (d == 0).mean(),
            'p_wilcoxon_pairs': wilcoxon(d[d != 0]).pvalue if (d != 0).any() else np.nan,
            'p_wilcoxon_systems': wilcoxon(per_sys[per_sys != 0]).pvalue if (per_sys != 0).any() else np.nan,
        })
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
