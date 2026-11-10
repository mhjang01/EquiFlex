"""Pre-flight check of a training config against the data actually on disk.

Builds the same loaders scripts/train.py builds (train/val from the split CSV,
test, PoseBusters), confirms every split is non-empty so the checkpoint monitor
(val/total_loss) will exist, checks feature keys, size limits and loss flags,
and prints the settings that matter. Exit code 1 on any failure.

    python scripts/dream/verify_train_setup.py --config configs/v10_512.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

from equiflex.model import load_config
from equiflex.model.loader import get_test_dataloader, get_train_val_dataloaders, load_split_from_csv

problems = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        problems.append(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v10_512.yaml')
    ap.add_argument('--max-atoms', type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d, t, lg, lo, m = cfg.data, cfg.training, cfg.logging, cfg.loss, cfg.model

    print(f'config {args.config}')
    print(f'  model ns={m.ns} pair={m.pair_hidden_dim} atom_layers={m.num_atom_layers} geo_layers={m.num_geo_layers} '
          f'grad_ckpt={m.use_gradient_checkpointing}')
    print(f'  loss unrolls={lo.num_unrolls} curriculum={lo.unroll_curriculum} vdw={lo.vdw_weight}/{lo.vdw_active_weight}/'
          f'{lo.vdw_max_weight} buffer={lo.vdw_buffer} cutoff={lo.vdw_cutoff} linear_onset={getattr(lo, "vdw_linear_onset", "MISSING")} '
          f'chiral={getattr(lo, "chiral_weight", "MISSING")}')
    print(f'  train max_steps={t.max_steps} phys_warmup={t.phys_loss_warmup_steps} accum={t.accumulate_grad_batches} '
          f'val_every={t.val_check_interval} ckpt_every={lg.checkpoint_every_n_steps} use_wandb={lg.use_wandb} run={lg.wandb_run_name}')
    print(f'  data batch_size={d.batch_size} systems_per_batch={d.systems_per_batch} workers={d.num_workers}')
    print(f'  dirs data={d.data_dir} test={d.test_data_dir} pb={getattr(d, "pb_data_dir", None)} split={d.split_csv}')
    form = getattr(lo, 'vdw_form', 'mean')
    check(form == 'hinge' or getattr(lo, 'vdw_linear_onset', True) is False,
          f'vdW form {form}' + ('' if form == 'hinge' else ' with vdw_linear_onset false (pure p^2)'))
    check(getattr(lo, 'chiral_weight', 0) > 0, 'chiral_weight > 0')
    check(lg.checkpoint_every_n_steps <= t.val_check_interval, 'checkpoint cadence <= validation cadence')

    print('split resolution')
    tr, va, te = load_split_from_csv(d.split_csv)
    data_dir = Path(d.data_dir)
    found = {n: sum((data_dir / s / 'ligand_features.pt').exists() for s in ids) for n, ids in (('train', tr), ('val', va), ('test', te))}
    print(f'  csv train={len(tr)} val={len(va)} test={len(te)}; found under {data_dir}: {found}')
    check(found['train'] > 0.95 * len(tr), f'train ids resolve under data_dir ({found["train"]}/{len(tr)})')
    check(found['val'] > 0.95 * len(va), f'val ids resolve under data_dir ({found["val"]}/{len(va)}) -> val/total_loss will exist')
    check(found['test'] == 0, f'test ids are NOT under data_dir ({found["test"]} found; the loader would merge them into training)')

    print('loaders (as train.py builds them)')
    train_loader, val_loader, train_sys, val_sys, _ = get_train_val_dataloaders(
        data_dir=d.data_dir, batch_size=d.batch_size, num_workers=0, val_ratio=d.val_ratio, seed=t.seed,
        log_dir=None, val_batches=d.val_batches, split_csv=d.split_csv, split_dir=getattr(d, 'split_dir', None),
        preload=False, systems_per_batch=getattr(d, 'systems_per_batch', 1), pad_systems=getattr(d, 'pad_systems', True))
    check(len(train_sys) > 0, f'train loader: {len(train_sys)} systems, {len(train_loader)} batches')
    check(len(val_sys) > 0, f'val loader: {len(val_sys)} systems, {len(val_loader)} batches')
    test_loader = get_test_dataloader(d.test_data_dir, d.batch_size, 0)
    check(test_loader is not None and len(test_loader.dataset) > 0, f'test loader: {len(test_loader.dataset)} systems')
    pb_dir = getattr(d, 'pb_data_dir', None)
    pb_loader = get_test_dataloader(pb_dir, d.batch_size, 0) if pb_dir else None
    check(pb_loader is not None and len(pb_loader.dataset) > 0, f'pb loader: {len(pb_loader.dataset) if pb_loader else 0} systems')

    print('feature contents (random samples)')
    random.seed(0)
    for name, root, ids in (('train', data_dir, train_sys), ('val', data_dir, val_sys),
                            ('test', Path(d.test_data_dir), [p.name for p in test_loader.dataset.system_dirs]),
                            ('pb', Path(pb_dir), [p.name for p in pb_loader.dataset.system_dirs])):
        sample = random.sample(ids, min(40, len(ids)))
        n_chiral, n_in_pocket, maxn = 0, 0, 0
        for s in sample:
            lf = torch.load(root / s / 'ligand_features.pt', weights_only=False)
            pf = torch.load(root / s / 'pocket_features.pt', weights_only=False)
            n_chiral += 'chiral_atom_indices' in lf
            n_in_pocket += float(torch.cdist(lf['coords'].float(), pf['coords'].float()).min()) < 6.0
            maxn = max(maxn, lf['coords'].shape[0] + pf['coords'].shape[0])
        check(n_chiral == len(sample), f'{name}: chiral_atom_indices present in {n_chiral}/{len(sample)} sampled systems')
        check(n_in_pocket == len(sample), f'{name}: ligand sits in the pocket in {n_in_pocket}/{len(sample)} (crystal pose, not ref conformer)')
        print(f'       {name}: largest sampled system {maxn} atoms')

    print('size limit')
    counts = Path('data/data_v2_plinder_atom_counts.tsv')
    if counts.exists():
        c = pd.read_csv(counts, sep='\t').set_index('system_id')
        big = [s for s in train_sys if s in c.index and c.loc[s, 'n_total'] > args.max_atoms]
        check(not big, f'no training system above {args.max_atoms} atoms ({len(big)} found)')
    else:
        print('  skip: atom-count census not present')

    print('loader sanity: one padded train batch')
    batch = next(iter(train_loader))
    check(batch.get('lig_chiral_indices') is not None, f'batch carries lig_chiral_indices {tuple(batch["lig_chiral_indices"].shape)}')
    check(batch['rec_atom_coords_gt'].shape[0] == d.batch_size * d.systems_per_batch,
          f'batch has {batch["rec_atom_coords_gt"].shape[0]} copies (= batch_size x systems_per_batch)')

    print('\nRESULT:', 'ALL CHECKS PASSED' if not problems else f'{len(problems)} PROBLEM(S)')
    for p in problems:
        print('  -', p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
