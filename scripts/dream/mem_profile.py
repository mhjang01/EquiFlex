"""Peak training-step GPU memory versus system size and copies per microbatch.

Builds the same padded batch the training loader produces (S systems x T
timestep copies, pad_systems=True), runs FlexTrainer._shared_step plus one
backward under bf16 autocast, and reports torch.cuda.max_memory_allocated.
The same system is repeated S times so the padded atom count stays fixed;
peak memory then separates into a per-copy cost at a given N and a fixed base.

  python scripts/dream/mem_profile.py --config configs/v10_512.yaml \
      --data-dir data/smoke/train --pick p50 p90 max --copies 4 8 12 16
"""
import argparse
import gc
import os

import numpy as np
import torch

from equiflex.model import FlexTrainer, load_config
from equiflex.model.loader import FlexDataset, collate_padded_systems


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/v10_512.yaml')
    ap.add_argument('--data-dir', default='data/smoke/train')
    ap.add_argument('--pick', nargs='+', default=['p50', 'p90', 'max'],
                    help='size percentiles (pNN/max) or system directory names')
    ap.add_argument('--copies', nargs='+', type=int, default=[4, 8, 12, 16],
                    help='timestep copies per microbatch (batch_size x systems_per_batch)')
    ap.add_argument('--timesteps', type=int, default=4, help='data.batch_size: copies per system')
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda')
    module = FlexTrainer(config).to(device).train()
    n_params = sum(p.numel() for p in module.model.parameters())
    base = torch.cuda.memory_allocated() / 2**30
    print(f'{args.config}: {n_params/1e6:.1f}M params, {base:.2f} GB weights, unrolls {config.loss.num_unrolls}')

    names = [d.name for d in ds.system_dirs] if (ds := FlexDataset(args.data_dir)) else []
    by_name = all(p != 'max' and not p.startswith('p') for p in args.pick)

    def size_of(d):
        return (torch.load(d / 'pocket_features.pt', weights_only=False)['coords'].shape[0]
                + torch.load(d / 'ligand_features.pt', weights_only=False)['coords'].shape[0])

    if by_name:
        # Named picks: size only those, so a large training directory is not scanned.
        picks = [names.index(p) for p in args.pick]
        sizes = {i: size_of(ds.system_dirs[i]) for i in picks}
    else:
        sizes = np.array([size_of(d) for d in ds.system_dirs])
        order = np.argsort(sizes)
        picks = []
        for p in args.pick:
            if p == 'max':
                picks.append(order[-1])
            elif p.startswith('p'):
                picks.append(order[min(len(order) - 1, int(len(order) * int(p[1:]) / 100))])
            else:
                picks.append(names.index(p))

    print(f'{"system":32s} {"atoms":>5s} {"S":>2s} {"T":>2s} {"copies":>6s} {"peak_GB":>8s}  {"per_copy_GB":>11s} {"step_ms":>8s}')
    for idx in picks:
        item = ds[idx]
        prev = None
        for copies in args.copies:
            T = min(args.timesteps, copies)
            S = copies // T
            batch = to_device(collate_padded_systems([item] * S, timesteps_per_system=T), device)
            module.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                # Warm-up pass (kernel selection, allocator growth), then the timed one.
                torch.cuda.synchronize()
                times = []
                for rep in range(2):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        losses = module._shared_step(batch, aux_scale=1.0)
                    losses['total'].backward()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                    if rep == 0:
                        module.zero_grad(set_to_none=True)
                        del losses
                peak = torch.cuda.max_memory_allocated() / 2**30
                per_copy = '' if prev is None else f'{(peak - prev[1]) / (copies - prev[0]):11.2f}'
                print(f'{ds.system_dirs[idx].name:32s} {sizes[idx]:5d} {S:2d} {T:2d} {S*T:6d} {peak:8.2f}  {per_copy:>11s} {times[-1]:8.0f}')
                prev = (copies, peak)
            except torch.cuda.OutOfMemoryError:
                print(f'{ds.system_dirs[idx].name:32s} {sizes[idx]:5d} {S:2d} {T:2d} {S*T:6d} {"OOM":>8s}')
                oom = True
            else:
                oom = False
                del losses
            # The exception's traceback keeps the half-built graph alive until
            # the except block is left, so release it here, after the block.
            del batch
            module.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if oom:
                break


if __name__ == '__main__':
    main()
