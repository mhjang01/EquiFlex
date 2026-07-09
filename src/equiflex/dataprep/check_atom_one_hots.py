'''
Check per-block one-hot fallback rate for atom_features in v2_plinder_5A.

featurize_atoms layout (380 dim total):
  [0:256]   atom_name_chars (256 = 4 chars x 64 bins)  — char encoding, not category one-hot
  [256:263] period (7)             OH
  [263:282] group (19)             OH
  [282:290] pauling_en RBF (8)     — Gaussian RBF, not one-hot
  [290:299] formal_charge (9)      OH
  [299:303] num_radical_electrons (4) OH
  [303:312] hybridization (9)      OH
  [312:320] total_valence (8)      OH
  [320:328] explicit_valence (8)   OH
  [328:336] implicit_valence (8)   OH
  [336:341] total_num_hs (5)       OH
  [341:346] num_explicit_hs (5)    OH
  [346:351] num_implicit_hs (5)    OH
  [351:358] degree (7)             OH
  [358:365] total_degree (7)       OH
  [365:372] chiral_tag (7)         OH
  [372:380] booleans (8)           — single-bit features

`one_hot_encode` returns all zeros when value not in categories. So per block,
sum-over-block == 0 means the atom's value fell out of the category list (i.e.,
fallback case the user wants to avoid).

Reports per split, per-block:
  - n_atoms total
  - n_atoms with fallback (zero block) for ligand & pocket separately
  - fraction
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
import random
from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

NEW_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')
SPLITS = ['train', 'val', 'test']

# (name, start, end)
ONEHOT_BLOCKS = [
    ('period',                256, 263),
    ('group',                 263, 282),
    ('formal_charge',         290, 299),
    ('num_radical_electrons', 299, 303),
    ('hybridization',         303, 312),
    ('total_valence',         312, 320),
    ('explicit_valence',      320, 328),
    ('implicit_valence',      328, 336),
    ('total_num_hs',          336, 341),
    ('num_explicit_hs',       341, 346),
    ('num_implicit_hs',       346, 351),
    ('degree',                351, 358),
    ('total_degree',          358, 365),
    ('chiral_tag',            365, 372),
]


def count_one_system(system_dir, blocks):
    lig_fp = system_dir / 'ligand_features.pt'
    poc_fp = system_dir / 'pocket_features.pt'
    if not (lig_fp.exists() and poc_fp.exists()):
        return None
    try:
        lig = torch.load(lig_fp, weights_only=False, map_location='cpu')['atom_features']
        poc = torch.load(poc_fp, weights_only=False, map_location='cpu')['atom_features']
    except Exception:
        return None

    out = {'n_lig': int(lig.shape[0]), 'n_poc': int(poc.shape[0]),
           'lig_fb': {}, 'poc_fb': {},
           'lig_argmax': {}, 'poc_argmax': {}}
    for name, s, e in blocks:
        lblock = lig[:, s:e]
        pblock = poc[:, s:e]
        # Fallback = entire block is zero (no category matched)
        out['lig_fb'][name] = int((lblock.sum(dim=1) == 0).sum().item())
        out['poc_fb'][name] = int((pblock.sum(dim=1) == 0).sum().item())
        # Argmax distribution (only for non-fallback rows)
        lmask = lblock.sum(dim=1) > 0
        pmask = pblock.sum(dim=1) > 0
        if lmask.any():
            out['lig_argmax'][name] = lblock[lmask].argmax(dim=1).bincount(minlength=e - s).tolist()
        else:
            out['lig_argmax'][name] = [0] * (e - s)
        if pmask.any():
            out['poc_argmax'][name] = pblock[pmask].argmax(dim=1).bincount(minlength=e - s).tolist()
        else:
            out['poc_argmax'][name] = [0] * (e - s)
    return out


def aggregate(results, blocks):
    total_lig = total_poc = 0
    fb_lig = {name: 0 for name, _, _ in blocks}
    fb_poc = {name: 0 for name, _, _ in blocks}
    am_lig = {name: [0] * (e - s) for name, s, e in blocks}
    am_poc = {name: [0] * (e - s) for name, s, e in blocks}
    for r in results:
        if r is None:
            continue
        total_lig += r['n_lig']
        total_poc += r['n_poc']
        for name in fb_lig:
            fb_lig[name] += r['lig_fb'][name]
            fb_poc[name] += r['poc_fb'][name]
            la = r['lig_argmax'][name]; pa = r['poc_argmax'][name]
            for i, v in enumerate(la):
                am_lig[name][i] += v
            for i, v in enumerate(pa):
                am_poc[name][i] += v

    return {
        'n_atoms': {'ligand': total_lig, 'pocket': total_poc},
        'fallback_counts': {'ligand': fb_lig, 'pocket': fb_poc},
        'fallback_fraction': {
            'ligand': {k: (v / total_lig if total_lig else 0.0) for k, v in fb_lig.items()},
            'pocket': {k: (v / total_poc if total_poc else 0.0) for k, v in fb_poc.items()},
        },
        'argmax_distribution': {
            'ligand': am_lig,
            'pocket': am_poc,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--splits', type=str, default='train,val,test')
    ap.add_argument('--sample', type=int, default=5000,
                    help='Sample up to N system dirs per split (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    summary = {}
    for split in splits:
        sd = NEW_ROOT / split
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        if args.sample and len(dirs) > args.sample:
            random.Random(args.seed).shuffle(dirs)
            dirs = dirs[:args.sample]
        print(f'[{split}] checking {len(dirs)} dirs')
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(count_one_system)(d, ONEHOT_BLOCKS) for d in tqdm(dirs, desc=f'oh:{split}')
        )
        summary[split] = aggregate(results, ONEHOT_BLOCKS)

    out_fp = NEW_ROOT / 'atom_one_hot_check.json'
    out_fp.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out_fp}')

    print('\n=== Fallback fraction (all-zero one-hot block) ===')
    for split in splits:
        s = summary[split]
        print(f'\n[{split}] n_lig_atoms={s["n_atoms"]["ligand"]}  n_poc_atoms={s["n_atoms"]["pocket"]}')
        print(f'  {"block":<24}{"lig_fb%":<12}{"poc_fb%":<12}')
        for name, _, _ in ONEHOT_BLOCKS:
            lf = s['fallback_fraction']['ligand'][name]
            pf = s['fallback_fraction']['pocket'][name]
            print(f'  {name:<24}{lf:.4%}    {pf:.4%}')


if __name__ == '__main__':
    main()
