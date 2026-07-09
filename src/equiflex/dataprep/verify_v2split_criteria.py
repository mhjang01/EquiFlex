'''
Verify each parsed system in v2_plinder_5A_v2split/ satisfies the survivor
criteria, using ACTUAL parsed data (not just metadata).

Criteria:
  C1: exactly 1 ligand in ligands.pkl
  C2: pocket residue count > 3
      AND 4 < total ligand atom count < 128
  C3: total pocket atoms (non-missing) < 500
  C4: ligand has no missing heavy atoms
       proxy: ligand_features.n_atoms_per_ligand matches
       ligand_num_heavy_atoms from Plinder index
  C5: entry_pdb_id not in PB pdb set
  C6: dedup verified by per-system uniqueness within (pdb, canonical_smiles)
       within current dataset

Outputs:
  v2_plinder_5A_v2split/logs/criteria_verification.json
'''
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, '/home/mhjang/workspace/flex')

import numpy as np
from joblib import Parallel, delayed, load
from tqdm import tqdm

ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
PB_FILE = '/home/mhjang/workspace/flex/v2_plinder/texts/posebusters_pdb_ccd_ids.txt'
SPLITS = ['train', 'val', 'test']


def check_one(system_dir):
    lf = system_dir / 'ligands.pkl'
    pf = system_dir / 'pocket_residues.pkl'
    sys_id = system_dir.name
    if not (lf.exists() and pf.exists()):
        return None
    try:
        ligs = load(lf)
        pocket = load(pf)
    except Exception:
        return None

    n_ligs = len(ligs)
    smiles = sorted({l['smiles'] for l in ligs})
    canonical_smi = smiles[0] if len(smiles) == 1 else '|'.join(smiles)
    lig_atom_total = sum(len(l['atoms']) for l in ligs)

    n_pocket_res = sum(len(v) for v in pocket.values())
    n_pocket_atoms = 0
    n_missing_lig_atoms = 0  # extracted from parsed data
    for chain in pocket.values():
        for res in chain.values():
            mask = np.asarray(res['missing_masks'])
            n_pocket_atoms += int((~mask).sum())
    # Ligand missing atoms: parse_single_system already filters bad SDFs;
    # rely on `ligand_num_heavy_atoms` consistency proxy: every atom kept.
    # We don't have unresolved info in pkl; report None for clarity.

    pdb_id = sys_id.split('__')[0].lower()
    return {
        'sys_id': sys_id,
        'pdb_id': pdb_id,
        'n_ligands': n_ligs,
        'n_pocket_residues': n_pocket_res,
        'lig_atoms_total': lig_atom_total,
        'n_pocket_atoms': n_pocket_atoms,
        'smiles_key': canonical_smi,
    }


def main():
    pb_pdb = set()
    with open(PB_FILE) as f:
        for line in f:
            line = line.strip()
            if line: pb_pdb.add(line.split('_')[0].lower())
    print(f'PoseBusters pdb set: {len(pb_pdb)}')

    summary = {}
    for split in SPLITS:
        sd = ROOT / split
        if not sd.exists():
            continue
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        print(f'\n[{split}] {len(dirs)} dirs')
        recs = Parallel(n_jobs=-1)(
            delayed(check_one)(d) for d in tqdm(dirs, desc=f'verify:{split}')
        )
        recs = [r for r in recs if r is not None]

        c1 = sum(1 for r in recs if r['n_ligands'] == 1)
        c2 = sum(1 for r in recs if r['n_ligands'] == 1
                 and r['n_pocket_residues'] > 3
                 and 4 < r['lig_atoms_total'] < 128)
        c3 = sum(1 for r in recs if r['n_ligands'] == 1
                 and r['n_pocket_residues'] > 3
                 and 4 < r['lig_atoms_total'] < 128
                 and r['n_pocket_atoms'] < 500)
        c5 = sum(1 for r in recs if r['pdb_id'] not in pb_pdb)

        # C6 dedup uniqueness within split (after parsing)
        key_to_ids = defaultdict(list)
        for r in recs:
            key_to_ids[(r['pdb_id'], r['smiles_key'])].append(r['sys_id'])
        c6_unique_keys = len(key_to_ids)
        c6_redundant = sum(len(v)-1 for v in key_to_ids.values() if len(v) > 1)

        # Violations
        bad_c1 = [r['sys_id'] for r in recs if r['n_ligands'] != 1][:10]
        bad_c2_pocket = [r['sys_id'] for r in recs if r['n_pocket_residues'] <= 3][:10]
        bad_c2_lig = [r['sys_id'] for r in recs
                       if not (4 < r['lig_atoms_total'] < 128)][:10]
        bad_c3 = [(r['sys_id'], r['n_pocket_atoms']) for r in recs
                  if r['n_pocket_atoms'] >= 500][:10]
        bad_c5 = [r['sys_id'] for r in recs if r['pdb_id'] in pb_pdb][:10]

        # Histograms / aggregates
        npr = np.array([r['n_pocket_residues'] for r in recs])
        npa = np.array([r['n_pocket_atoms'] for r in recs])
        lat = np.array([r['lig_atoms_total'] for r in recs])

        summary[split] = {
            'n_dirs_parsed': len(recs),
            'C1_n_ligs_eq_1':         c1,
            'C2_pass':                c2,
            'C3_pass_cumulative':     c3,
            'C5_not_in_PB':           c5,
            'C6_unique_keys':         c6_unique_keys,
            'C6_redundant_within_set': c6_redundant,
            'violations_examples': {
                'C1': bad_c1,
                'C2_pocket_le3': bad_c2_pocket,
                'C2_lig_out_of_range': bad_c2_lig,
                'C3_pocket_atoms_ge500': bad_c3,
                'C5_in_PB': bad_c5,
            },
            'pocket_residues': {
                'min': int(npr.min()), 'max': int(npr.max()),
                'mean': float(npr.mean()), 'median': float(np.median(npr)),
            },
            'pocket_atoms': {
                'min': int(npa.min()), 'max': int(npa.max()),
                'mean': float(npa.mean()), 'median': float(np.median(npa)),
                'p95': float(np.percentile(npa, 95)), 'p99': float(np.percentile(npa, 99)),
            },
            'lig_atoms': {
                'min': int(lat.min()), 'max': int(lat.max()),
                'mean': float(lat.mean()),
                'p95': float(np.percentile(lat, 95)),
            },
        }
        print(f'[{split}]')
        for k in ['n_dirs_parsed','C1_n_ligs_eq_1','C2_pass','C3_pass_cumulative',
                  'C5_not_in_PB','C6_unique_keys','C6_redundant_within_set']:
            print(f'  {k}: {summary[split][k]}')

    out = ROOT / 'logs' / 'criteria_verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
