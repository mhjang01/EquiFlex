'''
Apply annotation-derivable post-parse filters (missing10, metal<=2) to the
v2-split pli_qcov<90 + atom<450 target. Produces the candidate set to parse,
and the prep list (candidates not yet complete on disk).

  missing10: remove if pocket_validation unresolved/heavy > 0.10 (only when
             nonnegative & available; keep when missing/buggy/null).
  metal<=2 : remove if >2 metal-ion ligand rows (ligand_is_ion & ccd not in
             {F,CL,BR,I,XE,KR,AR,HE,NE,RN}).
'''
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
REQUIRED = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']
NONMETAL = {'F', 'CL', 'BR', 'I', 'XE', 'KR', 'AR', 'HE', 'NE', 'RN'}


def main():
    target = sorted({l.strip() for l in open(OUTDIR / 'v2split_pli90_atom450_target.txt') if l.strip()})
    tset = set(target)

    a = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'ligand_is_ion', 'ligand_ccd_code',
        'system_pocket_validation_num_unresolved_heavy_atoms',
        'system_pocket_validation_heavy_atom_count']).to_pandas()
    a = a[a['system_id'].isin(tset)].copy()
    a['ccd'] = a['ligand_ccd_code'].astype(str).str.upper()

    # metal<=2
    a['is_metal'] = a['ligand_is_ion'].astype(bool) & (~a['ccd'].isin(NONMETAL))
    metal_cnt = a.groupby('system_id')['is_metal'].sum()
    metal_bad = set(metal_cnt[metal_cnt > 2].index)

    # missing10 (per-system; pocket validation values are system-level, take first)
    pv = a.drop_duplicates('system_id').set_index('system_id')
    unres = pv['system_pocket_validation_num_unresolved_heavy_atoms']
    heavy = pv['system_pocket_validation_heavy_atom_count']
    missing_bad = set()
    for s in target:
        u, h = unres.get(s), heavy.get(s)
        if u is None or h is None or (isinstance(u, float) and np.isnan(u)) or \
           (isinstance(h, float) and np.isnan(h)) or h <= 0 or u < 0:
            continue  # missing/buggy -> keep
        if (u / h) > 0.10:
            missing_bad.add(s)

    candidates = [s for s in target if s not in metal_bad and s not in missing_bad]
    cset = set(candidates)

    complete, incomplete, absent = [], [], []
    for s in candidates:
        d = TR / s
        if not d.is_dir():
            absent.append(s)
        elif all((d / f).exists() for f in REQUIRED):
            complete.append(s)
        else:
            incomplete.append(s)
    to_prepare = sorted(absent + incomplete)

    # on-disk dirs now NOT in candidate set (should be removed to keep dir correct)
    on_disk = {d.name for d in TR.iterdir() if d.is_dir()}
    stale_on_disk = sorted(on_disk - cset)

    (OUTDIR / 'final_candidates.txt').write_text('\n'.join(candidates) + '\n')
    (OUTDIR / 'final_to_prepare.txt').write_text('\n'.join(to_prepare) + '\n')
    (OUTDIR / 'final_stale_on_disk.txt').write_text('\n'.join(stale_on_disk) + '\n')

    summary = {
        'target': len(target),
        'removed_metal_gt2': len(metal_bad),
        'removed_missing10': len(missing_bad),
        'final_candidates': len(candidates),
        'on_disk_now': len(on_disk),
        'already_complete': len(complete),
        'incomplete': len(incomplete),
        'absent_full_parse': len(absent),
        'to_prepare': len(to_prepare),
        'stale_on_disk_to_remove': len(stale_on_disk),
    }
    (OUTDIR / 'final_filter_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
