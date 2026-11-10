"""Parameter-count and training-step VRAM sweep over FlexModel configurations.

Param counts need no data. ``--vram`` runs one warmup + 3 timed (median) ConFIG training
steps (forward, physics unroll, both backward passes, optimizer step) on a real
system from data/smoke/train, tiled (whole residues, shifted copies) or cropped
to --n-rec/--n-lig atoms, collated with loader.collate_single_system. On OOM the
YAML batch size is halved (//2, //4, //8) and peak memory/time extrapolated
linearly in bs; if bs 1 still OOMs, two smaller receptors are measured and
a + b*N^2 extrapolated to the target N. Extrapolated rows are labelled.

    uv run python scripts/arch_sweep.py --configs configs/v10_512.yaml configs/v10_probe.yaml --vram
    uv run python scripts/arch_sweep.py --configs configs/v10_probe.yaml --grid ns=768 "ns=1024,num_geo_layers=6"
"""

import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # as scripts/dream/train.sbatch

import argparse
import dataclasses
import gc
import math
import re
import time
from pathlib import Path

import torch
import yaml
from conflictfree.grad_operator import ConFIG_update_double
from conflictfree.utils import apply_gradient_vector, get_gradient_vector
from muon import SingleDeviceMuonWithAuxAdam

from equiflex.model.config import FullConfig, load_config
from equiflex.model.loader import FlexDataset, collate_single_system
from equiflex.model.models import FlexModel
from equiflex.model.trainer import FlexTrainer

REPO = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3

# First matching rule wins; anything unmatched lands in 'other'.
PARAM_GROUPS = [
    ('initial/fusion', r'^(initial_embedder|rec_fusion|lig_fusion)\.'),
    ('time/adaLN', r'^(time_embedder|adaln_flow_init|adaln_geo_layers)\.'),
    ('pair_embedder', r'^pair_embedder\.'),
    ('chem FFN', r'^atom_layers\.[^.]+\.(ffn|ffn_norm)\.'),
    ('chem atom layers', r'^atom_layers\.'),
    ('triangle', r'^triangle_layers\.'),
    ('pair transition', r'^pair_transition'),
    ('cross init', r'cross'),
    ('pocket', r'^pocket_interaction\.'),
    ('eva', r'^eva_module\.'),
    ('rij/vec2scalar/norm', r'^geo_attention_layers\.[^.]+\.(rij_|v2s|vnorm|vec_to_scalar|vector_to_scalar|vec_norm|vector_norm|inv_)'),
    ('geo vector gates', r'^geo_attention_layers\.[^.]+\.vector_gate_'),
    ('geo vector/TP', r'^geo_attention_layers\.[^.]+\.vector_'),
    ('geo attention (scalar)', r'^geo_attention_layers\.'),
    ('head', r'^flow_head\.'),
]


def param_groups(model):
    counts = {name: 0 for name, _ in PARAM_GROUPS}
    counts['other'] = 0
    for pname, p in model.named_parameters():
        group = next((name for name, pat in PARAM_GROUPS if re.search(pat, pname)), 'other')
        counts[group] += p.numel()
    return counts


def apply_overrides(cfg: FullConfig, spec: str) -> FullConfig:
    """Apply 'key=val,key=val' FlexConfig overrides (values parsed as YAML)."""
    kv = {}
    for tok in filter(None, spec.split(',')):
        k, v = tok.split('=', 1)
        kv[k.strip()] = yaml.safe_load(v)
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **kv))


# --------------------------------------------------------------------------- synthetic system

def _block_diag(mats):
    n = sum(m.shape[0] for m in mats)
    out = mats[0].new_zeros(n, n, *mats[0].shape[2:])
    o = 0
    for m in mats:
        out[o:o + m.shape[0], o:o + m.shape[0]] = m
        o += m.shape[0]
    return out


def _remap_rows(idx_rows, keep, n_idx_cols):
    """Keep index rows whose atom columns all survive `keep`, remapped to new numbering."""
    new_index = torch.cumsum(keep.long(), 0) - 1
    atoms = idx_rows[:, :n_idx_cols]
    ok = keep[atoms].all(dim=1)
    rows = idx_rows[ok].clone()
    rows[:, :n_idx_cols] = new_index[rows[:, :n_idx_cols]]
    return rows


def resize_system(item, n_rec, n_lig, shift=25.0):
    """Tile/crop a dataset item to ~n_rec receptor atoms (whole residues) and n_lig ligand atoms."""
    item = dict(item)
    off = torch.tensor([shift, 0.0, 0.0])

    # Receptor: tile whole copies, then keep the first k residues with <= n_rec atoms.
    n0 = item['rec_atom_feats'].shape[0]
    n_res0 = item['esm_embeddings'].shape[0]
    reps = math.ceil(n_rec / n0)
    cat = lambda key: torch.cat([item[key]] * reps, 0)
    shifted = lambda key: torch.cat([item[key] + c * off for c in range(reps)], 0)
    uid = torch.cat([item['rec_uid_vec'].long() + c * n_res0 for c in range(reps)], 0)
    bai0 = item['rec_backbone_atom_indices']
    bai = {k: torch.cat([v if k == 'is_gly' else v + c * n0 for c in range(reps)], 0) for k, v in bai0.items()}
    tors = torch.cat([item['rec_torsion_indices'] + c * n0 for c in range(reps)], 0)
    res_sizes = torch.bincount(uid, minlength=reps * n_res0)
    k = int((torch.cumsum(res_sizes, 0) <= n_rec).sum())
    keep = uid < k
    new_index = torch.cumsum(keep.long(), 0) - 1
    item.update({
        'rec_atom_feats': cat('rec_atom_feats')[keep],
        'rec_atom_coords_gt': shifted('rec_atom_coords_gt')[keep],
        'rec_atom_ref_coords': shifted('rec_atom_ref_coords')[keep],
        'rec_constraint_matrix': _block_diag([item['rec_constraint_matrix']] * reps)[keep][:, keep],
        'rec_bond_feats': _block_diag([item['rec_bond_feats']] * reps)[keep][:, keep],
        'rec_uid_vec': uid[keep],
        'rec_backbone_mask': cat('rec_backbone_mask')[keep],
        'rec_cb_mask': cat('rec_cb_mask')[keep],
        'rec_missing_mask': cat('rec_missing_mask')[keep],
        'esm_embeddings': cat('esm_embeddings')[:k],
        'rec_backbone_atom_indices': {
            key: (v[:k] if key == 'is_gly' else new_index[v[:k]]) for key, v in bai.items()
        },
        'rec_torsion_indices': _remap_rows(tors, keep, 4),
    })

    # Ligand: tile shifted copies (same uid -> one chemistry block), crop to n_lig atoms.
    m0 = item['lig_atom_feats'].shape[0]
    reps = math.ceil(n_lig / m0)
    loff = torch.tensor([0.0, 10.0, 0.0])
    keep = torch.arange(reps * m0) < n_lig
    lcat = lambda key: torch.cat([item[key]] * reps, 0)[:n_lig]
    lig_tors = torch.cat([item['lig_torsion_indices'] + c * m0 for c in range(reps)], 0)
    chir = item['lig_chiral_indices']
    lig_chir = torch.cat([torch.cat([chir[:, :4] + c * m0, chir[:, 4:]], 1) for c in range(reps)], 0)
    item.update({
        'lig_atom_feats': lcat('lig_atom_feats'),
        'lig_atom_coords_gt': torch.cat([item['lig_atom_coords_gt'] + c * loff for c in range(reps)], 0)[:n_lig],
        'lig_atom_ref_coords': torch.cat([item['lig_atom_ref_coords'] + c * loff for c in range(reps)], 0)[:n_lig],
        'lig_constraint_matrix': _block_diag([item['lig_constraint_matrix']] * reps)[:n_lig, :n_lig],
        'lig_bond_feats': _block_diag([item['lig_bond_feats']] * reps)[:n_lig, :n_lig],
        'lig_uid_vec': lcat('lig_uid_vec'),
        'unimol_embeddings': lcat('unimol_embeddings'),
        'lig_torsion_indices': _remap_rows(lig_tors, keep, 4),
        'lig_chiral_indices': _remap_rows(lig_chir, keep, 4),
    })
    return item


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- training step

def build_optimizer(tr: FlexTrainer):
    oc = tr.config.optimizer
    muon_params, adam_params = tr._get_parameter_groups()
    return SingleDeviceMuonWithAuxAdam([
        {'params': muon_params, 'use_muon': True, 'lr': oc.muon_lr, 'momentum': oc.muon_momentum,
         'weight_decay': oc.muon_weight_decay},
        {'params': adam_params, 'use_muon': False, 'lr': oc.adam_lr, 'betas': (oc.adam_beta1, oc.adam_beta2),
         'eps': oc.adam_eps, 'weight_decay': oc.adam_weight_decay},
    ])


def train_step(tr: FlexTrainer, opt, batch, accum: dict):
    """One microbatch of FlexTrainer.training_step + _config_backward_and_step (step every call).

    ConFIG grad accumulators are kept in `accum` across calls, as they persist
    between microbatches under accumulate_grad_batches > 1. Returns
    (forward+backward seconds, ConFIG+clip+optimizer seconds).
    """
    cfg = tr.config
    bf16 = cfg.training.precision.startswith('bf16')
    t0 = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        losses = tr._shared_step(batch, aux_scale=1.0, num_unrolls=cfg.loss.num_unrolls)
    if tr.use_config:
        flow, phys = losses['flow_group'], losses['physics_group']
        assert phys.requires_grad, 'physics group inactive: unroll not exercised'
        opt.zero_grad()
        flow.backward(retain_graph=True)
        g = get_gradient_vector(tr.model, none_grad_mode='zero')
        accum['flow'] = g if 'flow' not in accum else accum['flow'].add_(g)
        del g
        opt.zero_grad()
        phys.backward()
        g = get_gradient_vector(tr.model, none_grad_mode='zero')
        accum['phys'] = g if 'phys' not in accum else accum['phys'].add_(g)
        del g
    else:
        opt.zero_grad()
        losses['total'].backward()
    del losses
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if tr.use_config:
        g_config = ConFIG_update_double(accum['flow'], accum['phys'])
        apply_gradient_vector(tr.model, g_config, none_grad_mode='zero', zero_grad_mode='pad_value')
        del g_config
    torch.nn.utils.clip_grad_norm_(tr.model.parameters(), cfg.training.gradient_clip_val)
    opt.step()
    torch.cuda.synchronize()
    return t1 - t0, time.perf_counter() - t1


TIMED_STEPS = 3
METRICS = ('mem', 'sec', 'fb', 'opt')


def measure(cfg: FullConfig, item, batch_size: int):
    """Peak allocated GiB over TIMED_STEPS steps after one warmup step, and median step times (s)."""
    torch.manual_seed(0)
    tr = FlexTrainer(cfg).cuda().train()
    opt = build_optimizer(tr)
    batch = to_device(collate_single_system([item] * batch_size), 'cuda')
    accum = {}
    try:
        train_step(tr, opt, batch, accum)
        torch.cuda.reset_peak_memory_stats()
        times = torch.tensor([train_step(tr, opt, batch, accum) for _ in range(TIMED_STEPS)])
        fb, op = times.median(dim=0).values.tolist()
        total = times.sum(dim=1).median().item()
        return {'mem': torch.cuda.max_memory_allocated() / GIB, 'sec': total, 'fb': fb, 'opt': op}
    finally:
        del tr, opt, batch, accum
        gc.collect()
        torch.cuda.empty_cache()


def try_measure(cfg: FullConfig, item, batch_size: int):
    """measure() or None on CUDA OOM (cleanup after the except block so the traceback's frames are freed)."""
    result = None
    try:
        result = measure(cfg, item, batch_size)
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  bs {batch_size}: ' + ('OOM' if result is None else _fmt(result)), flush=True)
    return result


def _fmt(r):
    return f"{r['mem']:.2f} GiB/{r['sec']:.2f}s"


def _extrapolate(x1, r1, x2, r2, x):
    """Linear in x through (x1, r1), (x2, r2) for every metric."""
    return {k: r1[k] + (r1[k] - r2[k]) / (x1 - x2) * (x - x1) for k in METRICS}


def vram_at_size(cfg: FullConfig, item):
    """Measure at the YAML batch size; on OOM fall back to bs//2, //4, //8 and extrapolate linearly in bs."""
    bs = cfg.data.batch_size
    result = try_measure(cfg, item, bs)
    if result is not None:
        return {**result, 'note': 'measured'}
    points = []
    for div in (2, 4, 8):
        b = bs // div
        if b < 1:
            break
        result = try_measure(cfg, item, b)
        if result is not None:
            points.append((b, result))
        if len(points) == 2:
            break
    if len(points) < 2:
        return None
    (b1, r1), (b2, r2) = points
    note = f'EXTRAPOLATED in bs (OOM at bs {bs}): measured bs{b1} {_fmt(r1)}, bs{b2} {_fmt(r2)}'
    return {**_extrapolate(b1, r1, b2, r2, bs), 'note': note}


def vram_for_config(cfg: FullConfig, template, n_rec: int, n_lig: int):
    """VRAM at the target size; if even the bs ladder OOMs, fit a + b*N^2 on two smaller receptors."""
    item = resize_system(template, n_rec, n_lig)
    n_tot = item['rec_atom_feats'].shape[0] + n_lig
    print(f'  N = {n_tot}', flush=True)
    r = vram_at_size(cfg, item)
    if r is not None:
        return r
    points = []
    for frac in (0.7, 0.5, 0.35):
        sub = resize_system(template, int(n_rec * frac), n_lig)
        n = sub['rec_atom_feats'].shape[0] + n_lig
        print(f'  N = {n}', flush=True)
        r = vram_at_size(cfg, sub)
        if r is not None:
            points.append((n, r))
        if len(points) == 2:
            break
    if len(points) < 2:
        note = f'OOM at N {n_tot}; too few smaller-N points to extrapolate'
        return {**dict.fromkeys(METRICS, float('nan')), 'note': note}
    (n1, r1), (n2, r2) = points
    note = (f'EXTRAPOLATED in N^2 (OOM at N {n_tot} even at bs 1): N{n1} {_fmt(r1)} [{r1["note"]}], '
            f'N{n2} {_fmt(r2)} [{r2["note"]}]')
    return {**_extrapolate(n1 ** 2, r1, n2 ** 2, r2, n_tot ** 2), 'note': note}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=[], help='YAML configs')
    ap.add_argument('--grid', nargs='+', default=[],
                    help="FlexConfig override sets 'key=val,key=val', applied to the first --configs YAML "
                         "(e.g. ns, nv, num_geo_heads, num_atom_layers, num_geo_layers, pair_hidden_dim, "
                         "chem_ffn_mult, any flag)")
    ap.add_argument('--vram', action='store_true', help='measure training-step memory/time on the GPU')
    ap.add_argument('--n-rec', type=int, default=450)
    ap.add_argument('--n-lig', type=int, default=50)
    ap.add_argument('--system-dir', default=str(REPO / 'data/smoke/train'),
                    help='dataset dir; its first system is the synthetic-batch template')
    args = ap.parse_args()

    runs = [(Path(p).stem, load_config(p)) for p in args.configs]
    if args.grid:
        base_name, base = runs[0]
        runs += [(f'{base_name}[{spec}]', apply_overrides(base, spec)) for spec in args.grid]
    if not runs:
        ap.error('give --configs and/or --grid')
    for _, cfg in runs:
        cfg.training.phys_loss_warmup_steps = 0
        cfg.training.use_compile = False

    counts = {}
    for name, cfg in runs:
        counts[name] = param_groups(FlexModel(cfg.model))

    groups = [g for g, _ in PARAM_GROUPS] + ['other']
    print('\n## Parameters (M)\n')
    print('| group | ' + ' | '.join(counts) + ' |')
    print('|---|' + '---:|' * len(counts))
    for g in groups:
        if any(c[g] for c in counts.values()):
            print(f'| {g} | ' + ' | '.join(f'{c[g] / 1e6:.2f}' for c in counts.values()) + ' |')
    print('| **total** | ' + ' | '.join(f'**{sum(c.values()) / 1e6:.2f}**' for c in counts.values()) + ' |')

    if not args.vram:
        return

    ds = FlexDataset(args.system_dir)
    template = ds[0]
    n_rec = resize_system(template, args.n_rec, args.n_lig)['rec_atom_feats'].shape[0]
    gpu = torch.cuda.get_device_properties(0)
    print(f'\n## Training-step VRAM ({gpu.name}, {gpu.total_memory / GIB:.1f} GiB; target A100-40GB)\n')
    print(f'Template {ds.system_dirs[0].name} resized to N_rec={n_rec}, N_lig={args.n_lig} '
          f'(tiled whole residues / ligand copies). One warmup step, then median of {TIMED_STEPS} timed steps; '
          f'physics unroll active (warmup 0); ConFIG double backward + optimizer step.\n')
    rows = []
    for name, cfg in runs:
        print(f'{name}:', flush=True)
        r = vram_for_config(cfg, template, args.n_rec, args.n_lig)
        rows.append(
            f"| {name} | {sum(counts[name].values()) / 1e6:.2f} | {cfg.data.batch_size} | {cfg.training.precision} | "
            f"{cfg.model.use_gradient_checkpointing} | {cfg.loss.num_unrolls} | {r['mem']:.2f} | "
            f"{r['sec']:.2f} | {r['fb']:.2f} | {r['opt']:.3f} | {40.0 - r['mem']:.2f} | {r['note']} |"
        )
    print('\n| config | params (M) | bs | precision | grad ckpt | unrolls | peak (GiB) | step (s) '
          '| fwd+bwd (s) | ConFIG+opt (s) | A100-40GB headroom (GiB) | note |')
    print('|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|')
    print('\n'.join(rows))

if __name__ == '__main__':
    main()
