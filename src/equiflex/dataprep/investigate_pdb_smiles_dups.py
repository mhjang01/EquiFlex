'''
Investigate the (pdb_id, smiles) duplicate collapse.

Hypothesis: most dups within a (pdb, smiles) bucket are homodimer
biological-assembly copies — same protein chain, same ligand, just
different chain copy IDs (1.A vs 2.A etc).

For each bucket >1 system:
  - Decompose system_id: <pdb>__<biounit>__<rec_chains>__<lig_chains>
  - Within bucket, compare biounit, rec_chains, lig_chains across siblings.

Outputs to v2_plinder_5A/logs/dup_investigation.json + top examples.
'''
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SPLIT_PQ = '/mnt/drive/plinder/2024-06/v2/splits/split.parquet'
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE  = '/home/mhjang/workspace/flex/v2_plinder/texts/posebusters_pdb_ccd_ids.txt'
LOG_DIR  = Path('/home/mhjang/workspace/flex/v2_plinder_5A/logs')


def parse_sys(sys_id):
    parts = sys_id.split('__')
    if len(parts) != 4:
        return None
    pdb, biounit, rec, lig = parts
    rec_chains = tuple(sorted(rec.split('_')))
    lig_chains = tuple(sorted(lig.split('_')))
    return pdb, biounit, rec_chains, lig_chains


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_parquet(SPLIT_PQ)
    split_map = dict(zip(split_df['system_id'], split_df['split']))
    poc_map   = dict(zip(split_df['system_id'], split_df['system_proper_pocket_num_residues']))
    chains_map = dict(zip(split_df['system_id'], split_df['system_proper_num_ligand_chains']))

    annot = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'entry_pdb_id', 'ligand_rdkit_canonical_smiles',
        'ligand_num_heavy_atoms', 'ligand_num_unresolved_heavy_atoms',
    ]).to_pandas()
    annot = annot[annot['system_id'].isin(split_map)].copy()
    annot['split'] = annot['system_id'].map(split_map)
    annot['pocket_res'] = annot['system_id'].map(poc_map)
    annot['n_lig_chains'] = annot['system_id'].map(chains_map)

    # Apply filters up to C5 (same as survivor script).
    pb_pdb = set()
    with open(PB_FILE) as f:
        for line in f:
            line = line.strip()
            if line: pb_pdb.add(line.split('_')[0].lower())

    m = (
        (annot['n_lig_chains'] == 1)
        & (annot['pocket_res'] > 3)
        & (annot['ligand_num_heavy_atoms'] > 4)
        & (annot['ligand_num_heavy_atoms'] < 128)
        & (annot['pocket_res'] * 14 < 500)
        & (annot['ligand_num_unresolved_heavy_atoms'] == 0)
        & (~annot['entry_pdb_id'].str.lower().isin(pb_pdb))
    )
    pre_c6 = annot[m].copy()
    print(f'pre-C6 rows: {len(pre_c6)} (train+val+test combined)')

    # Group by (pdb, smiles) bucket
    smi_col = 'ligand_rdkit_canonical_smiles'
    pre_c6['bucket'] = list(zip(pre_c6['entry_pdb_id'], pre_c6[smi_col]))
    bucket_size = pre_c6.groupby('bucket').size()
    print(f'unique buckets: {len(bucket_size)}')
    print(f'systems in buckets: {bucket_size.sum()}')

    # Bucket size histogram
    hist = Counter(bucket_size.values)
    hist_sorted = sorted(hist.items())
    print('\nBucket-size histogram (size: n_buckets):')
    for size, n in hist_sorted[:15]:
        print(f'  size={size:>3}: {n:>6} buckets ({size*n} systems)')
    print(f'  ... max size: {bucket_size.max()}')

    multi = bucket_size[bucket_size > 1]
    print(f'\nBuckets with >1 system: {len(multi)}  '
          f'(redundant rows = {multi.sum() - len(multi)})')

    # For multi-buckets, characterize structure
    same_rec_same_lig = 0   # homodimer-copy: same rec_chains AND same lig_chains? trivially same row
    same_rec_diff_lig = 0   # same protein chain set, different ligand chain (copy of ligand)
    diff_rec_same_lig = 0   # different protein chain set, same ligand chain
    diff_rec_diff_lig = 0   # both differ
    diff_biounit       = 0
    bucket_examples = []

    for bkey, sub in pre_c6.groupby('bucket'):
        if len(sub) < 2: continue
        decomp = [parse_sys(sid) for sid in sub['system_id']]
        if any(d is None for d in decomp): continue
        rec_sets = {d[2] for d in decomp}
        lig_sets = {d[3] for d in decomp}
        biounits = {d[1] for d in decomp}
        if len(biounits) > 1:
            diff_biounit += 1
        if len(rec_sets) == 1 and len(lig_sets) == 1:
            same_rec_same_lig += 1  # shouldn't happen if system_ids unique
        elif len(rec_sets) == 1 and len(lig_sets) > 1:
            same_rec_diff_lig += 1
        elif len(rec_sets) > 1 and len(lig_sets) == 1:
            diff_rec_same_lig += 1
        else:
            diff_rec_diff_lig += 1
        if len(bucket_examples) < 8 and len(sub) >= 5:
            bucket_examples.append({
                'bucket': list(bkey),
                'n_systems': len(sub),
                'rec_chains_seen': sorted({'_'.join(d[2]) for d in decomp}),
                'lig_chains_seen': sorted({'_'.join(d[3]) for d in decomp}),
                'biounits_seen': sorted(biounits),
                'systems': sub['system_id'].head(8).tolist(),
            })

    print(f'\nMulti-bucket structure (n_buckets):')
    print(f'  same rec_chains, diff lig_chains: {same_rec_diff_lig}')
    print(f'  diff rec_chains, same lig_chains: {diff_rec_same_lig}')
    print(f'  diff rec_chains, diff lig_chains: {diff_rec_diff_lig}')
    print(f'  spans different biounits:         {diff_biounit}')

    # By split
    print('\nBy split (pre-C6 / C6-deduped):')
    for split in ['train','val','test']:
        sub = pre_c6[pre_c6['split']==split]
        before = sub['system_id'].nunique()
        after = sub.drop_duplicates(subset=['entry_pdb_id', smi_col])['system_id'].nunique()
        print(f'  {split}: {before:>7} -> {after:>6}  (-{before-after})')

    out = {
        'unique_buckets': int(len(bucket_size)),
        'systems_in_buckets': int(bucket_size.sum()),
        'multi_buckets': int(len(multi)),
        'redundant_rows_if_dedup': int(multi.sum() - len(multi)),
        'bucket_size_hist': {str(k): int(v) for k, v in hist_sorted},
        'multi_bucket_structure': {
            'same_rec_diff_lig': same_rec_diff_lig,
            'diff_rec_same_lig': diff_rec_same_lig,
            'diff_rec_diff_lig': diff_rec_diff_lig,
            'spans_diff_biounits': diff_biounit,
        },
        'examples_size_ge5': bucket_examples,
    }
    (LOG_DIR / 'dup_investigation.json').write_text(json.dumps(out, indent=2))
    print(f'\nWritten: {LOG_DIR/"dup_investigation.json"}')

    # Print top examples inline
    print('\n=== TOP SAMPLE BUCKETS (size>=5) ===')
    for ex in bucket_examples[:5]:
        print(f"\nbucket {ex['bucket'][0]} smiles[:40]={ex['bucket'][1][:40]!r} n={ex['n_systems']}")
        print(f"  rec_chains: {ex['rec_chains_seen']}")
        print(f"  lig_chains: {ex['lig_chains_seen']}")
        print(f"  biounits:   {ex['biounits_seen']}")
        print(f"  samples:    {ex['systems']}")


if __name__ == '__main__':
    main()
