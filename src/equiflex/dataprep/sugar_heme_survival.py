'''
How many train systems / unique SMILES survive after removing sugar (NAG-type)
and heme proper ligands. Reports several definitions.

Train = on-disk fully-featurized set (split_train.txt). Each system has exactly
one proper ligand (one-proper-ligand filter), so proper-ligand CCD is unique.
'''
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/home/mhjang/workspace/flex/data_v2_plinder')
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'

HEME_CCD = {'HEM', 'HEC', 'HEA', 'HEB', 'HEO', 'HAS', 'HNI', 'HE5', 'HEG', 'HDM',
            'DHE', 'HCO', 'HFM', 'HIF', '1FH', '2FH', 'HP5', 'SRM', 'VER', 'COH',
            'HEME', 'MHM', '6CO', 'CLN', 'DDH', 'FEC', 'HNN'}
SUGAR_CCD = {'NAG', 'NDG', 'BMA', 'MAN', 'GLC', 'BGC', 'GAL', 'GLA', 'GLB', 'FUC',
             'FUL', 'FCA', 'FCB', 'SIA', 'SLB', 'NGA', 'A2G', 'BGN', 'GCS', 'GLD',
             'XYP', 'XYS', 'XYL', 'RAM', 'RIB', 'ARA', 'ARB', 'LYX', 'ALL', 'TAL',
             'GUL', 'IDO', 'RBL', 'FRU', 'TAG', 'SOR', 'NGZ', 'NG6', 'NGC', 'GCU',
             'ADA', 'BDP', 'IDS', 'GCD', 'MAV', 'GAL', 'G6D', 'SGN', 'MUB', 'RIP',
             'BXX', 'BXY', 'MGC', 'GLP', 'NGK', 'KDO', 'DAG'}


def main():
    train = sorted({l.strip() for l in open(ROOT / 'split_train.txt') if l.strip()})
    tset = set(train)

    a = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'ligand_is_proper', 'ligand_ccd_code',
        'ligand_rdkit_canonical_smiles', 'ligand_is_oligo', 'ligand_is_cofactor']).to_pandas()
    p = a[(a['system_id'].isin(tset)) & (a['ligand_is_proper'] == True)].drop_duplicates('system_id').set_index('system_id')
    p['ccd'] = p['ligand_ccd_code'].astype(str).str.upper()

    def usmiles(ids):
        s = p.loc[list(ids), 'ligand_rdkit_canonical_smiles'].dropna()
        return s[s.astype(str).str.len() > 0].nunique()

    base_u = usmiles(train)

    masks = {
        'heme_ccd': p['ccd'].isin(HEME_CCD),
        'sugar_ccd': p['ccd'].isin(SUGAR_CCD),
        'is_oligo_flag': p['ligand_is_oligo'] == True,
        'is_cofactor_flag': p['ligand_is_cofactor'] == True,
    }
    report = {'n_train': len(train), 'base_unique_smiles': int(base_u), 'removals': {}}
    for name, m in masks.items():
        rm = set(p.index[m])
        surv = [s for s in train if s not in rm]
        report['removals'][name] = {'removed': len(rm), 'survivors': len(surv),
                                     'unique_smiles': int(usmiles(surv))}

    # combined: sugar (ccd OR oligo) + heme (ccd)
    rm_comb = set(p.index[masks['heme_ccd'] | masks['sugar_ccd'] | masks['is_oligo_flag']])
    surv = [s for s in train if s not in rm_comb]
    report['combined_sugar_and_heme'] = {
        'definition': 'remove proper ligand if ccd in HEME_CCD or SUGAR_CCD or ligand_is_oligo',
        'removed': len(rm_comb), 'survivors': len(surv), 'unique_smiles': int(usmiles(surv)),
        'survivor_frac': len(surv) / len(train)}

    # also the broader "sugar+all cofactors" for reference
    rm_b = set(p.index[masks['heme_ccd'] | masks['sugar_ccd'] | masks['is_oligo_flag'] | masks['is_cofactor_flag']])
    survb = [s for s in train if s not in rm_b]
    report['combined_plus_all_cofactors'] = {
        'definition': 'above + ligand_is_cofactor (removes ATP/NAD/FAD/etc too)',
        'removed': len(rm_b), 'survivors': len(survb), 'unique_smiles': int(usmiles(survb))}

    (ROOT / 'logs' / 'sugar_heme_survival.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
