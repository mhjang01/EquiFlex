'''
Attribute ligand-diversity (unique canonical SMILES) loss to each filter stage
of the v2_strict pipeline, on the v2 split train set.

At each cumulative stage report: n_systems, n_unique_smiles (over proper ligands
of surviving systems), and fractional drop in unique SMILES vs previous stage.
n_systems is validated against v2_strict_survivors_summary.json counts_by_stage.
'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPLIT_PQ = '/mnt/drive/plinder/2024-06/v2/splits/split.parquet'
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
PLI_COMP = ['pli_qcov__95__community', 'pli_qcov__100__community',
            'pli_qcov__95__weak__component', 'pli_qcov__100__weak__component',
            'pli_qcov__95__strong__component', 'pli_qcov__100__strong__component']
SEQ100 = ['protein_seqsim_max__100__community',
          'protein_seqsim_max__100__weak__component',
          'protein_seqsim_max__100__strong__component']


def main():
    sp = pq.read_table(SPLIT_PQ, columns=[
        'system_id', 'split', 'system_proper_pocket_num_residues']).to_pandas()
    train_sys = set(sp[sp['split'] == 'train']['system_id'])
    pocket_res = dict(zip(sp['system_id'], sp['system_proper_pocket_num_residues']))

    cols = ['system_id', 'entry_pdb_id', 'ligand_is_proper', 'ligand_is_ion',
            'ligand_num_heavy_atoms', 'ligand_num_unresolved_heavy_atoms',
            'ligand_matches_smiles_atom_num', 'ligand_rdkit_canonical_smiles'] + PLI_COMP + SEQ100
    a = pq.read_table(ANNOT_PQ, columns=cols).to_pandas()
    a['pdb'] = a['entry_pdb_id'].str.lower()

    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pb_pdbs = {p.split('_')[0].lower() for p in pb}
    pb_rows = a[a['pdb'].isin(pb_pdbs)]
    pb_systems = set(pb_rows['system_id'])
    pb_smiles = set(pb_rows['ligand_rdkit_canonical_smiles'].dropna())
    pb_seq100 = set()
    for c in SEQ100:
        pb_seq100 |= set(pb_rows[c].dropna())
    pb_pli = set()
    for c in PLI_COMP:
        pb_pli |= set(pb_rows[c].dropna())

    tr = a[a['system_id'].isin(train_sys)].copy()

    def uniq_smiles(df):
        # diversity = unique canonical SMILES of PROPER ligands among surviving systems
        prop = df[df['ligand_is_proper'] == True]
        s = prop['ligand_rdkit_canonical_smiles'].dropna()
        return s[s.astype(str).str.len() > 0].nunique()

    stages = []
    def rec(name, df):
        stages.append({'stage': name, 'n_systems': df['system_id'].nunique(),
                       'n_unique_smiles': uniq_smiles(df)})

    rec('0 base v2 train', tr)

    # 1 one proper ligand; co-binders ions only
    g = tr.groupby('system_id')
    n_proper = g['ligand_is_proper'].sum()
    n_bad = g.apply(lambda d: int(((~d['ligand_is_proper'].astype(bool)) &
                                   (~d['ligand_is_ion'].astype(bool))).sum()))
    keep1 = set(n_proper[(n_proper == 1)].index) & set(n_bad[n_bad == 0].index)
    s1 = tr[tr['system_id'].isin(keep1)]
    rec('1 one proper ligand; cobinders ions only', s1)

    # from here, evaluate on the proper-ligand row of each surviving system
    prop = s1[s1['ligand_is_proper'] == True].drop_duplicates('system_id').set_index('system_id')

    # 2 proper ligand heavy atoms < 70
    keep2 = set(prop[prop['ligand_num_heavy_atoms'] < 70].index)
    rec('2 proper ligand heavy_atoms<70', s1[s1['system_id'].isin(keep2)])

    # 3 proper pocket residues > 3
    keep3 = {s for s in keep2 if pocket_res.get(s, 0) > 3}
    rec('3 proper pocket_residues>3', s1[s1['system_id'].isin(keep3)])

    # 4 no missing ligand atoms; atom count matches smiles
    ok4 = prop[(prop['ligand_num_unresolved_heavy_atoms'] == 0) &
               (prop['ligand_matches_smiles_atom_num'] == True)].index
    keep4 = keep3 & set(ok4)
    rec('4 no missing atoms; atom_count==smiles', s1[s1['system_id'].isin(keep4)])

    # 5 remove PB overlap: exact PB system OR (same proper-ligand smiles AND shared seq100 component)
    def shares_seq100(row):
        return any((row.get(c) in pb_seq100) and pd.notna(row.get(c)) for c in SEQ100)
    keep5 = set()
    p4 = prop.loc[list(keep4)]
    for sid, row in p4.iterrows():
        if sid in pb_systems:
            continue
        smi = row['ligand_rdkit_canonical_smiles']
        if (smi in pb_smiles) and shares_seq100(row):
            continue
        keep5.add(sid)
    rec('5 remove PB overlap (exact / ligand+seq100)', s1[s1['system_id'].isin(keep5)])

    # 6 remove shared PB pli_qcov 95/100 component
    keep6 = set()
    p5 = prop.loc[list(keep5)]
    for sid, row in p5.iterrows():
        if any(pd.notna(row.get(c)) and row.get(c) in pb_pli for c in PLI_COMP):
            continue
        keep6.add(sid)
    rec('6 remove shared PB pli_qcov 95/100 component', s1[s1['system_id'].isin(keep6)])

    # 7 dedup (pdb, canonical_smiles, pocket_num_residues)
    p6 = prop.loc[list(keep6)].reset_index()
    p6['pres'] = p6['system_id'].map(pocket_res)
    dedup = p6.drop_duplicates(subset=['pdb', 'ligand_rdkit_canonical_smiles', 'pres'])
    keep7 = set(dedup['system_id'])
    rec('7 dedup (pdb, smiles, pocket_res)', s1[s1['system_id'].isin(keep7)])

    # fractional drops
    base_u = stages[0]['n_unique_smiles']
    prev_u = base_u
    print(f"{'stage':<48}{'n_sys':>9}{'n_uniq':>9}{'drop_uniq':>11}{'frac_step':>11}{'frac_base':>11}")
    for st in stages:
        u = st['n_unique_smiles']
        drop = prev_u - u
        st['drop_unique_from_prev'] = int(drop)
        st['frac_drop_step'] = float(drop / prev_u) if prev_u else 0.0
        st['frac_remaining_of_base'] = float(u / base_u)
        print(f"{st['stage']:<48}{st['n_systems']:>9}{u:>9}{drop:>11}{st['frac_drop_step']*100:>10.1f}%{u/base_u*100:>10.1f}%")
        prev_u = u

    out = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs/ligand_diversity_by_stage.json')
    out.write_text(json.dumps({'base_unique_smiles': base_u, 'stages': stages}, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
