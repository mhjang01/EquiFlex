'''
Cross-validate atom features, pocket features, and ESM3 embeddings produced
by featurize_5A on v2_plinder_5A.

Checks per system (where artifacts exist):
    ligand_features.pt:
        atom_features.shape[0] == sum(n_atoms_per_ligand)
        coords.shape[0] == ref_coords.shape[0] == atom_features.shape[0]
        bond_features square in dim 0/1
        ligand_uid_vec length matches atom count and labels are 0..n_lig-1 covering all
    pocket_features.pt:
        atom_features.shape[0] == coords.shape[0] == missing_mask.shape[0]
        bond_features square
        residue_uid_vec length == atom count and max+1 == len(residue_keys)
    esm3_embeddings.pt:
        shape == (len(residue_keys), 1536)
        n zero-rows (= missing residue embeddings) reported
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

NEW_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')
SPLITS = ['train', 'val', 'test']


def check_system(system_dir):
    out = {'sys_id': system_dir.name, 'issues': []}
    lig_fp = system_dir / 'ligand_features.pt'
    poc_fp = system_dir / 'pocket_features.pt'
    esm_fp = system_dir / 'esm3_embeddings.pt'

    if not (lig_fp.exists() and poc_fp.exists()):
        out['issues'].append('missing_features'); return out

    try:
        lig = torch.load(lig_fp, weights_only=False, map_location='cpu')
        poc = torch.load(poc_fp, weights_only=False, map_location='cpu')
    except Exception as e:
        out['issues'].append(f'load_err:{e}'); return out

    # --- ligand_features ---
    n_atoms_per = list(map(int, lig.get('n_atoms_per_ligand', [])))
    n_lig_atoms = sum(n_atoms_per)
    af = lig['atom_features']; bf = lig['bond_features']
    if af.shape[0] != n_lig_atoms:
        out['issues'].append(f'lig_af_rows={af.shape[0]} vs sum_n={n_lig_atoms}')
    if lig['coords'].shape[0] != n_lig_atoms:
        out['issues'].append(f'lig_coords_rows={lig["coords"].shape[0]} vs sum_n={n_lig_atoms}')
    if lig['ref_coords'].shape[0] != n_lig_atoms:
        out['issues'].append(f'lig_ref_coords_rows={lig["ref_coords"].shape[0]} vs sum_n={n_lig_atoms}')
    if bf.shape[0] != n_lig_atoms or bf.shape[1] != n_lig_atoms:
        out['issues'].append(f'lig_bf_shape={tuple(bf.shape)} vs N={n_lig_atoms}')
    uid = lig.get('ligand_uid_vec')
    if uid is not None:
        if uid.shape[0] != n_lig_atoms:
            out['issues'].append(f'lig_uid_len={uid.shape[0]} vs {n_lig_atoms}')
        elif n_atoms_per:
            uniq = set(uid.tolist())
            if uniq != set(range(len(n_atoms_per))):
                out['issues'].append(f'lig_uid_set={sorted(uniq)} vs expected 0..{len(n_atoms_per)-1}')

    # --- pocket_features ---
    paf = poc['atom_features']
    n_poc_atoms = paf.shape[0]
    if poc['coords'].shape[0] != n_poc_atoms:
        out['issues'].append(f'poc_coords_rows={poc["coords"].shape[0]} vs {n_poc_atoms}')
    if poc['missing_mask'].shape[0] != n_poc_atoms:
        out['issues'].append(f'poc_mask_rows={poc["missing_mask"].shape[0]} vs {n_poc_atoms}')
    pbf = poc['bond_features']
    if pbf.shape[0] != n_poc_atoms or pbf.shape[1] != n_poc_atoms:
        out['issues'].append(f'poc_bf_shape={tuple(pbf.shape)} vs {n_poc_atoms}')
    res_keys = poc.get('residue_keys', [])
    n_res = len(res_keys)
    ruid = poc.get('residue_uid_vec')
    if ruid is None:
        out['issues'].append('no_residue_uid_vec')
    elif ruid.shape[0] != n_poc_atoms:
        out['issues'].append(f'res_uid_len={ruid.shape[0]} vs {n_poc_atoms}')
    elif n_res:
        max_uid = int(ruid.max().item())
        if max_uid + 1 != n_res:
            out['issues'].append(f'max_res_uid+1={max_uid+1} vs n_res={n_res}')

    out['n_lig_atoms'] = n_lig_atoms
    out['n_poc_atoms'] = n_poc_atoms
    out['n_poc_res'] = n_res

    # --- ESM3 ---
    if esm_fp.exists():
        try:
            esm = torch.load(esm_fp, weights_only=False, map_location='cpu')
        except Exception as e:
            out['issues'].append(f'esm_load_err:{e}')
        else:
            if esm.shape != (n_res, 1536):
                out['issues'].append(f'esm_shape={tuple(esm.shape)} vs ({n_res},1536)')
            zero_rows = int((esm.abs().sum(dim=1) == 0).sum().item())
            out['esm_zero_rows'] = zero_rows
            out['has_esm'] = True
    else:
        out['has_esm'] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--splits', type=str, default='train,val,test')
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    summary = {}
    for split in splits:
        sd = NEW_ROOT / split
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        print(f'[{split}] {len(dirs)} dirs')
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(check_system)(d) for d in tqdm(dirs, desc=f'check:{split}')
        )

        n = len(results)
        n_clean = sum(1 for r in results if not r['issues'])
        n_issue = n - n_clean
        n_esm = sum(1 for r in results if r.get('has_esm'))
        total_zero_rows = sum(r.get('esm_zero_rows', 0) for r in results)
        n_with_zero = sum(1 for r in results if r.get('esm_zero_rows', 0) > 0)
        issue_types = defaultdict(int)
        for r in results:
            for iss in r['issues']:
                tag = iss.split('=', 1)[0].split(':', 1)[0]
                issue_types[tag] += 1
        worst = sorted([r for r in results if r['issues']],
                       key=lambda r: -len(r['issues']))[:10]

        summary[split] = {
            'n_systems': n,
            'n_clean': n_clean,
            'n_with_issues': n_issue,
            'issue_type_counts': dict(issue_types),
            'esm_present': n_esm,
            'systems_with_esm_zero_rows': n_with_zero,
            'total_esm_zero_rows': total_zero_rows,
            'worst_offenders': [
                {'sys_id': r['sys_id'], 'issues': r['issues']} for r in worst
            ],
        }
        print(f'[{split}] clean={n_clean}  issues={n_issue}  esm_present={n_esm}  '
              f'esm_zero_rows total={total_zero_rows} (in {n_with_zero} sys)')
        if issue_types:
            print(f'  issue_types: {dict(issue_types)}')

    out_fp = NEW_ROOT / 'feature_check.json'
    out_fp.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out_fp}')


if __name__ == '__main__':
    main()
