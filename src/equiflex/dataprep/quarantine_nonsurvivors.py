'''
Move non-survivor system dirs out of v2_plinder_5A_v2split/{split}/ into
v2_plinder_5A_v2split_rejected/{split}/ so featurize only sees survivors.

Survivor = parsed dir passing:
  C1: ligands.pkl length == 1
  C2: pocket_residues > 3, 4 < ligand_atom_total < 128
  C3: total non-missing pocket atoms < 500
  C5: pdb_id not in PB set
  C6: keep first per (pdb_id, ligand_smiles)
'''
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/home/mhjang/workspace/flex')

import numpy as np
from joblib import Parallel, delayed, load
from tqdm import tqdm

ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
REJECT = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split_rejected')
PB_FILE = '/home/mhjang/workspace/flex/v2_plinder/texts/posebusters_pdb_ccd_ids.txt'
SPLITS = ['train', 'val', 'test']


def check(system_dir):
    lf = system_dir / 'ligands.pkl'
    pf = system_dir / 'pocket_residues.pkl'
    sid = system_dir.name
    if not (lf.exists() and pf.exists()):
        return sid, 'missing_pkl', None
    try:
        ligs = load(lf)
        pocket = load(pf)
    except Exception as e:
        return sid, f'load_err:{e}', None

    if len(ligs) != 1:
        return sid, 'C1_n_ligs', None

    smi = ligs[0]['smiles']
    n_lig_atoms = len(ligs[0]['atoms'])
    if not (4 < n_lig_atoms < 128):
        return sid, 'C2_lig_atoms', smi

    n_pocket_res = sum(len(v) for v in pocket.values())
    if n_pocket_res <= 3:
        return sid, 'C2_pocket_res', smi

    n_pocket_atoms = 0
    for chain in pocket.values():
        for res in chain.values():
            mask = np.asarray(res['missing_masks'])
            n_pocket_atoms += int((~mask).sum())
    if n_pocket_atoms >= 500:
        return sid, 'C3_pocket_atoms', smi

    return sid, 'OK', smi


def main():
    pb_pdb = set()
    with open(PB_FILE) as f:
        for line in f:
            line = line.strip()
            if line: pb_pdb.add(line.split('_')[0].lower())

    summary = {}
    for split in SPLITS:
        sd = ROOT / split
        if not sd.exists(): continue
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        print(f'\n[{split}] checking {len(dirs)} dirs')
        results = Parallel(n_jobs=-1)(
            delayed(check)(d) for d in tqdm(dirs, desc=f'check:{split}')
        )

        # Apply C5 (PB) and C6 (dedup by (pdb, smi))
        kept_keys = {}     # (pdb, smi) -> kept sid
        reasons = {}
        for sid, status, smi in results:
            pdb = sid.split('__')[0].lower()
            if status != 'OK':
                reasons[sid] = status
                continue
            if pdb in pb_pdb:
                reasons[sid] = 'C5_in_PB'
                continue
            key = (pdb, smi)
            if key in kept_keys:
                reasons[sid] = 'C6_dup'
                continue
            kept_keys[key] = sid

        kept_set = set(kept_keys.values())
        n_kept = len(kept_set)
        n_rej  = len(results) - n_kept

        # Move rejects
        rej_split = REJECT / split
        rej_split.mkdir(parents=True, exist_ok=True)
        moved = 0
        for sid, status, _ in results:
            if sid in kept_set: continue
            src = sd / sid
            dst = rej_split / sid
            if dst.exists():  # already moved
                continue
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                print(f'  move fail {sid}: {e}')

        from collections import Counter
        cnt = Counter(reasons.values())
        summary[split] = {
            'total_parsed': len(results),
            'kept': n_kept,
            'rejected': n_rej,
            'moved': moved,
            'reason_counts': dict(cnt),
        }
        print(f'  kept={n_kept}  rejected={n_rej}  moved={moved}')
        print(f'  reasons: {dict(cnt)}')

        # Write final survivors
        (ROOT / 'logs' / f'{split}_final_survivors.txt').write_text(
            '\n'.join(sorted(kept_set)) + '\n'
        )

    out = ROOT / 'logs' / 'quarantine_summary.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
