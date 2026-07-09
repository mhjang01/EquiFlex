'''
Final validity audit of v2_plinder_5A_v2split (organic + optional metal-ion):

1. Pocket cutoff: every stored pocket residue has >=1 non-missing atom within
   5.0 A of any ligand atom (organic OR metal counted equally).
2. Feature integrity: ligand_features.pt / pocket_features.pt / esm3 / unimol
   all present and shape-consistent.
3. Ligand composition: exactly 1 organic (>1 atom, not metal-ion SMILES)
   and >= 0 metal-ion atoms. No multi-atom non-organic co-binders.

Outputs v2_plinder_5A_v2split/logs/final_validity_check.json.
'''
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/home/mhjang/workspace/flex')

import numpy as np
import torch
from joblib import Parallel, delayed, load
from tqdm import tqdm

ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
SPLITS = ['train', 'val', 'test']
METAL_PATTERN = re.compile(
    r'^\[(Zn|Mg|Ca|Fe|Mn|Cu|Ni|Co|Cd|Hg|Pb|K|Na|Cs|Rb|Li|Sr|Ba|Be|Al|Cr|Mo|W|V|Pt|Pd|Ru|Rh|Au|Ag|Sn|Sb|As|Y|La|Eu|Gd|Tb|U)([+\-\d@HspBI]*)\]$'
)


def check_one(sd):
    out = {'sys_id': sd.name, 'issues': []}
    needed = ['ligand_features.pt', 'pocket_features.pt',
              'esm3_embeddings.pt', 'unimol_embeddings.pt',
              'ligands.pkl', 'pocket_residues.pkl']
    for f in needed:
        if not (sd / f).exists():
            out['issues'].append(f'missing:{f}')
    if out['issues']:
        return out

    try:
        ligs = load(sd / 'ligands.pkl')
        pocket = load(sd / 'pocket_residues.pkl')
        lf = torch.load(sd / 'ligand_features.pt', weights_only=False, map_location='cpu')
        pf = torch.load(sd / 'pocket_features.pt', weights_only=False, map_location='cpu')
        uni = torch.load(sd / 'unimol_embeddings.pt', weights_only=False, map_location='cpu')
        esm = torch.load(sd / 'esm3_embeddings.pt', weights_only=False, map_location='cpu')
    except Exception as e:
        out['issues'].append(f'load:{e}')
        return out

    # --- Ligand composition ---
    n_organic = 0
    n_metal = 0
    n_other = 0
    for l in ligs:
        n = len(l['atoms']); smi = l['smiles'].strip()
        if n == 1 and METAL_PATTERN.match(smi):
            n_metal += 1
        elif n > 1:
            n_organic += 1
        else:
            n_other += 1
    out['n_organic'] = n_organic
    out['n_metal'] = n_metal
    out['n_other_single_atom'] = n_other
    out['n_ligs_total'] = len(ligs)
    if n_organic != 1:
        out['issues'].append(f'C1_organic_count_{n_organic}')
    if n_other > 0:
        out['issues'].append(f'C_other_count_{n_other}')

    # --- Pocket 5A invariant ---
    lig_xyz = np.concatenate([np.asarray(l['coords']) for l in ligs], axis=0)
    pocket_min_d = []
    for ch, residues in pocket.items():
        for key, r in residues.items():
            c = np.asarray(r['coords']); m = np.asarray(r['missing_masks'])
            v = c[~m]
            if len(v) == 0:
                pocket_min_d.append(np.inf); continue
            d = np.linalg.norm(v[:, None, :] - lig_xyz[None, :, :], axis=2).min()
            pocket_min_d.append(float(d))
    pocket_min_d = np.asarray(pocket_min_d)
    out['n_pocket_res'] = int(len(pocket_min_d))
    out['max_pocket_min_dist_A'] = float(pocket_min_d.max()) if len(pocket_min_d) else 0.0
    if (pocket_min_d > 5.0).any():
        out['issues'].append(f'pocket_violation_max_{pocket_min_d.max():.2f}A')

    # --- Shape consistency ---
    n_lig_atoms = int(lf['atom_features'].shape[0])
    if uni.shape[0] != n_lig_atoms:
        out['issues'].append(f'unimol_shape_{uni.shape[0]}!=lig_{n_lig_atoms}')
    n_poc_atoms = int(pf['atom_features'].shape[0])
    if pf['coords'].shape[0] != n_poc_atoms:
        out['issues'].append(f'poc_coords_shape_{pf["coords"].shape[0]}!=feat_{n_poc_atoms}')
    n_res_keys = len(pf.get('residue_keys', []))
    if esm.shape != (n_res_keys, 1536):
        out['issues'].append(f'esm_shape_{tuple(esm.shape)}!=({n_res_keys},1536)')
    out['n_lig_atoms'] = n_lig_atoms
    out['n_poc_atoms'] = n_poc_atoms

    return out


def main():
    summary = {}
    for split in SPLITS:
        sd = ROOT / split
        if not sd.exists(): continue
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        print(f'\n[{split}] {len(dirs)} dirs')
        recs = Parallel(n_jobs=-1)(delayed(check_one)(d) for d in tqdm(dirs, desc=split))

        n_clean = sum(1 for r in recs if not r['issues'])
        issue_types = Counter()
        for r in recs:
            for iss in r['issues']:
                tag = iss.split(':', 1)[0].split('_C', 1)[0]
                issue_types[tag] += 1

        composition = Counter()
        for r in recs:
            composition[(r.get('n_organic', '?'), r.get('n_metal', '?'),
                         r.get('n_other_single_atom', '?'))] += 1

        max_d = max((r['max_pocket_min_dist_A'] for r in recs if 'max_pocket_min_dist_A' in r), default=0.0)
        viol_sys = sum(1 for r in recs
                        if any(i.startswith('pocket_violation') for i in r['issues']))

        summary[split] = {
            'n_dirs': len(recs),
            'n_clean': n_clean,
            'n_with_issues': len(recs) - n_clean,
            'issue_type_counts': dict(issue_types),
            'composition_dist_top10': sorted(composition.items(), key=lambda x: -x[1])[:10],
            'pocket_5A_violations': viol_sys,
            'pocket_max_min_dist_overall': max_d,
        }
        print(f'  clean={n_clean}  issues={len(recs)-n_clean}')
        print(f'  issue_types: {dict(issue_types)}')
        print(f'  composition top-5 (n_organic, n_metal, n_other): '
              f'{[(c, n) for c, n in sorted(composition.items(), key=lambda x: -x[1])[:5]]}')
        print(f'  pocket-5A violations: {viol_sys}  max_min_dist={max_d:.3f} A')

    out = ROOT / 'logs' / 'final_validity_check.json'
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
