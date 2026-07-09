'''
Validate that unimol_embeddings.pt atom count matches the new ligand_features.pt
for v2_plinder_5A systems.

For each system dir that has both files:
    lig = torch.load(ligand_features.pt)['n_atoms_per_ligand']
    embed = torch.load(unimol_embeddings.pt)
    expected = sum(lig)
    embed.shape[0] should == expected
Also verify SMILES set in new ligands.pkl matches the old (where symlinked).

Reports per-split: matched / mismatched (with examples) / missing-unimol.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import json
from pathlib import Path

import torch
from joblib import Parallel, delayed, load
from tqdm import tqdm

NEW_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')
OLD_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')
SPLITS = ['train', 'val', 'test']


def check_one(system_dir, old_root):
    name = system_dir.name
    unimol_fp = system_dir / 'unimol_embeddings.pt'
    lig_feat_fp = system_dir / 'ligand_features.pt'
    lig_pkl_fp = system_dir / 'ligands.pkl'
    if not unimol_fp.exists():
        return name, 'no_unimol', None, None
    if not lig_feat_fp.exists():
        return name, 'no_lig_features', None, None

    try:
        lig_feat = torch.load(lig_feat_fp, weights_only=False)
        embed = torch.load(unimol_fp, weights_only=False, map_location='cpu')
    except Exception as e:
        return name, f'load_err:{e}', None, None

    n_per_lig = lig_feat.get('n_atoms_per_ligand', [])
    expected = int(sum(n_per_lig))
    actual = int(embed.shape[0])

    # SMILES check vs old (where applicable)
    smiles_match = None
    old_lig_fp = old_root / system_dir.parent.name / name / 'ligands.pkl'
    if old_lig_fp.exists() and lig_pkl_fp.exists():
        try:
            new_smi = sorted(l['smiles'] for l in load(lig_pkl_fp))
            old_smi = sorted(l['smiles'] for l in load(old_lig_fp))
            smiles_match = (new_smi == old_smi)
        except Exception:
            smiles_match = None

    status = 'match' if expected == actual else 'mismatch'
    return name, status, (expected, actual), smiles_match


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
            delayed(check_one)(d, OLD_ROOT) for d in tqdm(dirs, desc=f'check:{split}')
        )

        counts = {'match': 0, 'mismatch': 0, 'no_unimol': 0, 'no_lig_features': 0, 'load_err': 0}
        mismatches = []
        smiles_check = {'match': 0, 'differ': 0, 'na': 0}
        for name, status, sizes, smi in results:
            key = 'load_err' if status.startswith('load_err') else status
            counts[key] = counts.get(key, 0) + 1
            if status == 'mismatch':
                mismatches.append({'sys_id': name, 'expected': sizes[0], 'actual': sizes[1]})
            if status == 'match':
                if smi is True:
                    smiles_check['match'] += 1
                elif smi is False:
                    smiles_check['differ'] += 1
                else:
                    smiles_check['na'] += 1

        summary[split] = {
            'counts': counts,
            'mismatches_first10': mismatches[:10],
            'smiles_consistency_within_matched': smiles_check,
        }
        print(f'[{split}] {counts}  smiles(among matched): {smiles_check}')

    out = NEW_ROOT / 'unimol_alignment_check.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
