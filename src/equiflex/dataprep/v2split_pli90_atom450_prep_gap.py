'''
Add total-pocket-atoms<450 to the v2-split pli_qcov<90 target, then recompute prep gap.
Pocket atom count: REAL (parsed 5A pocket_residues, non-missing) where available,
else annotation proxy system_pocket_validation_heavy_atom_count (~97% agreement).
'''
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
REQUIRED = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']
ATOM_MAX = 450


def real_pocket_atoms(s):
    fp = TR / s / 'pocket_residues.pkl'
    if not fp.exists():
        return s, None
    try:
        pk = load(fp)
        n = 0
        for ch in pk.values():
            for res in ch.values():
                n += int((~np.asarray(res['missing_masks'])).sum())
        return s, n
    except Exception:
        return s, None


def main():
    target = [l.strip() for l in open(OUTDIR / 'v2split_pli90_target.txt') if l.strip()]
    proxy = (pq.read_table(ANNOT_PQ, columns=['system_id', 'system_pocket_validation_heavy_atom_count'])
             .to_pandas().drop_duplicates('system_id')
             .set_index('system_id')['system_pocket_validation_heavy_atom_count'].to_dict())

    real = dict(Parallel(n_jobs=16)(delayed(real_pocket_atoms)(s) for s in target))

    kept, dropped, unknown = [], 0, 0
    src_real = src_proxy = 0
    for s in target:
        n = real.get(s)
        if n is not None:
            src_real += 1
        else:
            n = proxy.get(s)
            if n is not None and not np.isnan(n):
                src_proxy += 1
            else:
                unknown += 1
                kept.append(s)   # unknown -> keep (resolve post-parse)
                continue
        if n < ATOM_MAX:
            kept.append(s)
        else:
            dropped += 1

    complete, incomplete, absent = [], [], []
    for s in kept:
        d = TR / s
        if not d.is_dir():
            absent.append(s)
        elif all((d / f).exists() for f in REQUIRED):
            complete.append(s)
        else:
            incomplete.append(s)
    to_prepare = sorted(absent + incomplete)

    (OUTDIR / 'v2split_pli90_atom450_target.txt').write_text('\n'.join(kept) + '\n')
    (OUTDIR / 'v2split_pli90_atom450_to_prepare.txt').write_text('\n'.join(to_prepare) + '\n')
    summary = {
        'target_before_atom450': len(target),
        'pocket_atom_source': {'real_parsed': src_real, 'proxy_annotation': src_proxy, 'unknown_kept': unknown},
        'dropped_atoms_ge450': dropped,
        'target_after_atom450': len(kept),
        'already_complete': len(complete),
        'incomplete_needs_refeaturize': len(incomplete),
        'absent_needs_full_parse': len(absent),
        'total_to_prepare': len(to_prepare),
        'atom_max': ATOM_MAX,
    }
    (OUTDIR / 'v2split_pli90_atom450_prep_gap.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
