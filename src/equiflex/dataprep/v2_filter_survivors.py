'''
Estimate survivor counts on v2 split.parquet under user criteria.

Source:
    split.parquet              -> per-system: split label
    annotation_table.parquet   -> per-(system,ligand) metadata

Criteria:
    C1: system_proper_num_ligand_chains == 1            (one ligand chain)
    C2: system_proper_pocket_num_residues > 3
        AND 4 < ligand_num_heavy_atoms < 128
    C3: total pocket heavy atoms < 500
        NOTE: no direct column; using proxy
              `system_proper_pocket_num_residues * 14 < 500`
              (Trp has 14 heavy atoms; loose upper bound).
              Exact gate applied at parse-time.
    C4: ligand_num_unresolved_heavy_atoms == 0          (no missing atoms)
    C5: entry_pdb_id NOT in PoseBusters PDB set
    C5_1: max(pli_qcov_*__community/component) < 90     (similarity to test)
    C6: dedup per (entry_pdb_id, ligand_rdkit_canonical_smiles,
                   system_proper_pocket_num_residues)   (same pocket+ligand in same pdb)

Writes:
    v2_plinder_5A/logs/v2_filter_survivors.json
    v2_plinder_5A/logs/{train,val,test}_v2_survivors.txt
'''
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SPLIT_PQ = '/mnt/drive/plinder/2024-06/v2/splits/split.parquet'
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE  = '/home/mhjang/workspace/flex/v2_plinder/texts/posebusters_pdb_ccd_ids.txt'
LOG_DIR  = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split/logs')

PLI_QCOV_COLS = [
    'pli_qcov__50__community', 'pli_qcov__70__community',
    'pli_qcov__95__community', 'pli_qcov__100__community',
    'pli_qcov__50__weak__component', 'pli_qcov__70__weak__component',
    'pli_qcov__95__weak__component', 'pli_qcov__100__weak__component',
    'pli_qcov__50__strong__component', 'pli_qcov__70__strong__component',
    'pli_qcov__95__strong__component', 'pli_qcov__100__strong__component',
]


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading split.parquet...')
    split_df = pd.read_parquet(SPLIT_PQ)
    split_map = dict(zip(split_df['system_id'], split_df['split']))
    pocket_res_map = dict(zip(
        split_df['system_id'], split_df['system_proper_pocket_num_residues']))
    num_lig_chains_map = dict(zip(
        split_df['system_id'], split_df['system_proper_num_ligand_chains']))

    needed = [
        'system_id', 'entry_pdb_id',
        'ligand_num_heavy_atoms',
        'ligand_num_unresolved_heavy_atoms',
        'ligand_rdkit_canonical_smiles',
    ] + PLI_QCOV_COLS

    print('Loading annotation_table.parquet columns...')
    avail = [c for c in needed if c in [f.name for f in pq.read_schema(ANNOT_PQ)]]
    missing = [c for c in needed if c not in avail]
    if missing:
        print(f'  missing cols: {missing}')
    annot = pq.read_table(ANNOT_PQ, columns=avail).to_pandas()
    print(f'  annot rows: {len(annot)}')

    # Restrict to v2 systems in train/val/test
    annot = annot[annot['system_id'].isin(split_map)].copy()
    annot['split'] = annot['system_id'].map(split_map)
    annot['system_proper_pocket_num_residues'] = annot['system_id'].map(pocket_res_map)
    annot['system_proper_num_ligand_chains'] = annot['system_id'].map(num_lig_chains_map)
    print(f'  rows for v2 systems: {len(annot)}')

    # PB PDB set
    pb_pdb = set()
    with open(PB_FILE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            pb_pdb.add(line.split('_')[0].lower())
    print(f'  PoseBusters PDB ids: {len(pb_pdb)}')

    # Track per-criterion filter (cumulative).
    def count(df):
        return df.groupby('split')['system_id'].nunique().to_dict()

    stages = []

    base = annot.copy()
    stages.append(('base v2', count(base)))

    # C1: one ligand chain
    c1 = base[base['system_proper_num_ligand_chains'] == 1]
    stages.append(('C1: 1 ligand chain', count(c1)))

    # C2: pocket residues > 3 AND 4 < lig atoms < 128
    c2 = c1[(c1['system_proper_pocket_num_residues'] > 3)
            & (c1['ligand_num_heavy_atoms'] > 4)
            & (c1['ligand_num_heavy_atoms'] < 128)]
    stages.append(('C2: pocket_res>3, 4<lig_atoms<128', count(c2)))

    # C3 (proxy): pocket_res * 14 < 500
    c3 = c2[c2['system_proper_pocket_num_residues'] * 14 < 500]
    stages.append(('C3 proxy: pocket_res*14<500', count(c3)))

    # C4: no missing atoms in ligand
    c4 = c3[c3['ligand_num_unresolved_heavy_atoms'] == 0]
    stages.append(('C4: no unresolved lig atoms', count(c4)))

    # C5: drop PB-overlap PDB ids
    c5 = c4[~c4['entry_pdb_id'].str.lower().isin(pb_pdb)]
    stages.append(('C5: not in PB pdb set', count(c5)))

    # C5_1: max(pli_qcov) < 90 -- numeric similarity to PB/test, lives in
    # /mnt/drive/plinder/2024-06/v2/scores/ (pairwise sharded parquets).
    # Skipped in this estimate; computed by separate pairwise scan if needed.
    c5_1 = c5
    stages.append(('C5_1 skipped (needs pairwise scores)', count(c5_1)))

    # C6: dedup per (entry_pdb_id, ligand_rdkit_canonical_smiles)
    smiles_col = 'ligand_rdkit_canonical_smiles' if 'ligand_rdkit_canonical_smiles' in c5_1.columns else 'ligand_smiles'
    c6 = c5_1.drop_duplicates(subset=['entry_pdb_id', smiles_col])
    stages.append(('C6: dedup (pdb, canonical_smiles)', count(c6)))

    # Print
    print('\n' + '=' * 72)
    print(f'{"stage":<46}{"train":>10}{"val":>8}{"test":>8}')
    print('=' * 72)
    for name, cnt in stages:
        print(f'{name:<46}{cnt.get("train",0):>10}{cnt.get("val",0):>8}{cnt.get("test",0):>8}')

    # Write survivors
    summary = {'criteria': [n for n, _ in stages],
               'counts_by_stage': [{'stage': n, **cnt} for n, cnt in stages]}
    for split in ['train', 'val', 'test']:
        ids = sorted(c6[c6['split'] == split]['system_id'].unique())
        (LOG_DIR / f'{split}_v2_survivors.txt').write_text('\n'.join(ids) + '\n')
        summary[f'{split}_survivor_count'] = len(ids)

    (LOG_DIR / 'v2_filter_survivors.json').write_text(json.dumps(summary, indent=2))
    print(f'\nLogs written to {LOG_DIR}')


if __name__ == '__main__':
    main()
