'''
Comprehensive NaN/Inf scan over parsed training features + cdist risk.

Per system checks:
  - NaN/Inf in every float tensor of ligand_features.pt / pocket_features.pt
    (atom_features, bond_features, coords, ref_coords, constraint_matrix)
  - NaN/Inf in esm3_embeddings.pt / unimol_embeddings.pt
  - cdist risk on coords used in training: NaN/Inf coords, and min pairwise
    distance over (valid rec + ligand) atoms (dist==0 -> cdist backward NaN)

Output: data_v2_plinder/logs/nan_feature_scan.json
'''
import json
from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

TR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
OUT = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs/nan_feature_scan.json')
MIN_DIST = 1e-3
FLOAT_KEYS = ['atom_features', 'bond_features', 'coords', 'ref_coords', 'constraint_matrix']


def chk(t):
    if not torch.is_tensor(t) or not torch.is_floating_point(t):
        return False, False
    return bool(torch.isnan(t).any()), bool(torch.isinf(t).any())


def scan(sd):
    out = {'system': sd.name, 'nan': [], 'inf': [], 'mindist': None, 'error': None}
    try:
        lig = torch.load(sd / 'ligand_features.pt', weights_only=False, map_location='cpu')
        pkt = torch.load(sd / 'pocket_features.pt', weights_only=False, map_location='cpu')
    except Exception as e:
        out['error'] = f'load_feat: {e!s}'[:100]
        return out
    for tag, d in (('lig', lig), ('pkt', pkt)):
        for k in FLOAT_KEYS:
            if k in d:
                n, i = chk(d[k])
                if n: out['nan'].append(f'{tag}.{k}')
                if i: out['inf'].append(f'{tag}.{k}')
    for emb in ('esm3_embeddings.pt', 'unimol_embeddings.pt'):
        fp = sd / emb
        if fp.exists():
            try:
                e = torch.load(fp, weights_only=False, map_location='cpu')
                tensors = e.values() if isinstance(e, dict) else ([e] if torch.is_tensor(e) else [])
                for t in tensors:
                    n, i = chk(t)
                    if n: out['nan'].append(emb.split('_')[0] + '.nan')
                    if i: out['inf'].append(emb.split('_')[0] + '.inf')
            except Exception as ex:
                out['error'] = f'load_{emb}: {ex!s}'[:80]

    # cdist risk
    rc, lc = pkt.get('coords'), lig.get('coords')
    if torch.is_tensor(rc) and torch.is_tensor(lc) and not out['nan'] and not out['inf']:
        mm = pkt.get('missing_mask')
        if not torch.is_tensor(mm):
            mm = torch.zeros(rc.shape[0], dtype=torch.bool)
        valid = rc[~mm]
        allc = torch.cat([valid, lc], 0) if valid.shape[0] else lc
        if allc.shape[0] > 1:
            dmat = torch.cdist(allc, allc)
            dmat.fill_diagonal_(float('inf'))
            md = float(dmat.min())
            if md < MIN_DIST:
                out['mindist'] = round(md, 6)
    if not out['nan'] and not out['inf'] and out['mindist'] is None and not out['error']:
        return None
    return out


def main():
    sds = sorted(d for d in TR.iterdir()
                 if d.is_dir() and (d / 'ligand_features.pt').exists() and (d / 'pocket_features.pt').exists())
    print(f'scanning {len(sds)} systems', flush=True)
    res = Parallel(n_jobs=-1)(delayed(scan)(d) for d in tqdm(sds, desc='nan-scan'))
    issues = [r for r in res if r]

    from collections import Counter
    nan_c, inf_c = Counter(), Counter()
    for r in issues:
        for k in r['nan']: nan_c[k] += 1
        for k in r['inf']: inf_c[k] += 1
    mindist = [r for r in issues if r['mindist'] is not None]
    errors = [r for r in issues if r['error']]

    summary = {
        'scanned': len(sds),
        'systems_with_any_issue': len(issues),
        'nan_by_field': dict(nan_c),
        'inf_by_field': dict(inf_c),
        'n_cdist_zero_dist': len(mindist),
        'n_load_errors': len(errors),
        'examples': {
            'nan': [r['system'] for r in issues if r['nan']][:20],
            'inf': [r['system'] for r in issues if r['inf']][:20],
            'cdist_zero_dist': [(r['system'], r['mindist']) for r in mindist][:20],
            'errors': [(r['system'], r['error']) for r in errors][:20],
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    # full issue list
    (OUT.parent / 'nan_feature_scan_systems.txt').write_text(
        '\n'.join(r['system'] for r in issues) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
