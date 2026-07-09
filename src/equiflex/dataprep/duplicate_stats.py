'''
Duplicate-system stats for v2_plinder.

Three key levels (all ignore geometry):

  loose:
      (multiset canonical SMILES, multiset pocket resname counts)
      Same ligand + same per-AA pocket composition.

  pocket_identity:
      (multiset canonical SMILES, sorted (chain_id, resname, resnum, entity_id))
      Same ligand + identical pocket residues at identical positions.

  sequence_identity:
      (multiset canonical SMILES, sorted (chain_id, full chain sequence))
      Same ligand + identical full-protein sequences.

Outputs JSON to v2_plinder/duplicate_stats.json with summary per key level,
per-split breakdown, and cross-split overlap.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from joblib import Parallel, delayed, load
from tqdm import tqdm

STORAGE_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')
SPLITS = ['train', 'val', 'test']

KEY_LEVELS = ['loose', 'pocket_identity', 'sequence_identity']


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def compute_keys(system_dir: Path):
    '''Return dict of key_level -> hashable_key, plus n_ligs / n_res. None on failure.'''
    lig_fp = system_dir / 'ligands.pkl'
    poc_fp = system_dir / 'pocket_residues.pkl'
    seq_fp = system_dir / 'sequence_data.pkl'
    if not (lig_fp.exists() and poc_fp.exists()):
        return None
    try:
        ligs = load(lig_fp)
        pocket = load(poc_fp)
        seq = load(seq_fp) if seq_fp.exists() else {}
    except Exception:
        return None

    smiles_tup = tuple(sorted(l['smiles'] for l in ligs))

    resname_counts = Counter()
    pocket_residue_ids = []
    n_res = 0
    for chain_id, chain_residues in pocket.items():
        for (resname, resnum, entity_id) in chain_residues.keys():
            resname_counts[resname] += 1
            pocket_residue_ids.append((chain_id, resname, int(resnum), entity_id))
            n_res += 1
    resname_tuple = tuple(sorted(resname_counts.items()))
    pocket_id_tuple = tuple(sorted(pocket_residue_ids))

    if isinstance(seq, dict):
        seq_tuple = tuple(sorted((c, str(s)) for c, s in seq.items()))
    else:
        seq_tuple = ()
    seq_hash = _hash('|'.join(f'{c}:{s}' for c, s in seq_tuple))

    keys = {
        'loose': (smiles_tup, resname_tuple),
        'pocket_identity': (smiles_tup, pocket_id_tuple),
        'sequence_identity': (smiles_tup, seq_hash),
    }
    return {
        'keys': keys,
        'n_ligs': len(ligs),
        'n_res': n_res,
        'resname_counts': dict(resname_counts),
    }


def _process(system_dir: Path):
    r = compute_keys(system_dir)
    if r is None:
        return None
    return (system_dir.name, r)


def collect_split(split: str, n_jobs: int):
    split_dir = STORAGE_ROOT / split
    sys_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name != 'errors']
    print(f'[{split}] scanning {len(sys_dirs)} system dirs')
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process)(sd) for sd in tqdm(sys_dirs, desc=f'key:{split}')
    )
    return [r for r in results if r is not None]


def summarize(entries, level: str):
    '''entries: list of (sys_id, dict). dict has keys/n_ligs/n_res/resname_counts.'''
    buckets = defaultdict(list)
    info_for_key = {}
    for sys_id, info in entries:
        k = info['keys'][level]
        buckets[k].append(sys_id)
        if k not in info_for_key:
            info_for_key[k] = info

    dup_groups = {k: v for k, v in buckets.items() if len(v) > 1}
    n_systems = len(entries)
    n_unique_keys = len(buckets)
    n_in_dup_groups = sum(len(v) for v in dup_groups.values())
    n_redundant = n_in_dup_groups - len(dup_groups)

    hist = Counter(len(v) for v in buckets.values())
    hist = dict(sorted(hist.items()))

    top = sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:10]
    top_summary = []
    for k, sys_ids in top:
        info = info_for_key[k]
        smiles_tup = k[0]
        top_summary.append({
            'count': len(sys_ids),
            'smiles': list(smiles_tup),
            'n_pocket_residues': info['n_res'],
            'resname_counts': info['resname_counts'],
            'examples': sys_ids[:5],
        })

    return {
        'n_systems_scanned': n_systems,
        'n_unique_keys': n_unique_keys,
        'n_duplicate_groups': len(dup_groups),
        'n_systems_in_duplicate_groups': n_in_dup_groups,
        'n_redundant_systems_if_dedup': n_redundant,
        'dedup_fraction': n_redundant / n_systems if n_systems else 0.0,
        'group_size_histogram': hist,
        'top_duplicate_groups': top_summary,
    }


def cross_split_overlap(per_split_entries, level: str):
    key_to_splits = defaultdict(set)
    for split, entries in per_split_entries.items():
        for _, info in entries:
            key_to_splits[info['keys'][level]].add(split)

    overlaps = {s: 0 for s in SPLITS}
    pair_counts = Counter()
    for k, splits in key_to_splits.items():
        if len(splits) > 1:
            for s in splits:
                overlaps[s] += 1
            pair_counts[tuple(sorted(splits))] += 1

    return {
        'keys_in_multiple_splits': sum(1 for s in key_to_splits.values() if len(s) > 1),
        'systems_with_cross_split_twin': overlaps,
        'split_pair_counts': {','.join(k): v for k, v in pair_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--splits', type=str, default='train,val,test')
    parser.add_argument('--out', type=str, default=str(STORAGE_ROOT / 'duplicate_stats.json'))
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    per_split_entries = {}
    for split in splits:
        per_split_entries[split] = collect_split(split, args.n_jobs)

    summary = {}
    for level in KEY_LEVELS:
        summary[level] = {}
        for split in splits:
            summary[level][split] = summarize(per_split_entries[split], level)
        if len(splits) > 1:
            summary[level]['cross_split'] = cross_split_overlap(per_split_entries, level)

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')

    print('\n=== HEADLINE (redundant_if_dedup / scanned, dedup_frac) ===')
    header = f'{"split":<7}' + ''.join(f'{lvl:>26}' for lvl in KEY_LEVELS)
    print(header)
    for split in splits:
        row = f'{split:<7}'
        for level in KEY_LEVELS:
            s = summary[level][split]
            row += f'{s["n_redundant_systems_if_dedup"]:>10} / {s["n_systems_scanned"]:<5} ({s["dedup_fraction"]:.1%})'.rjust(26)
        print(row)

    print('\n=== cross-split (keys appearing in >=2 splits) ===')
    for level in KEY_LEVELS:
        if 'cross_split' in summary[level]:
            cs = summary[level]['cross_split']
            print(f'[{level}] keys_in_multi_split={cs["keys_in_multiple_splits"]}  '
                  f'systems_with_twin={cs["systems_with_cross_split_twin"]}  '
                  f'pairs={cs["split_pair_counts"]}')


if __name__ == '__main__':
    main()
