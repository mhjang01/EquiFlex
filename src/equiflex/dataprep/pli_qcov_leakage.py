'''
Compute max(pli_qcov) for each train system in data_v2_plinder vs test set.

Test set = union of:
  - data_v2_plinder/test  (internal Plinder system_ids)
  - PoseBusters PB systems (Plinder system_ids whose entry_pdb_id matches a PB pdb)

Score source: /mnt/drive/plinder/2024-06/v2/scores/search_db={holo,apo,pred}/*.parquet
              metric='pli_qcov', similarity in [0, 100].

Output: data_v2_plinder/logs/pli_qcov_leakage.json
        + histogram + percentile table
'''
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

DATA = Path('/home/mhjang/workspace/flex/data_v2_plinder')
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'


def main():
    splits = pd.read_csv(DATA / 'splits.csv')
    train = set(splits[splits['split'] == 'train']['system_id'])
    test_internal = set(splits[splits['split'] == 'test']['system_id'])
    print(f'train: {len(train)}  internal test: {len(test_internal)}')

    pb_pdbs = {l.strip().split('_')[0].lower() for l in open(PB_FILE) if l.strip()}
    annot = pq.read_table(ANNOT_PQ, columns=['system_id', 'entry_pdb_id']).to_pandas()
    annot = annot.drop_duplicates('system_id')
    pb_systems = set(annot[annot['entry_pdb_id'].str.lower().isin(pb_pdbs)]['system_id'])
    print(f'PB pdb ids: {len(pb_pdbs)}  PB Plinder system_ids: {len(pb_systems)}')

    test_all = test_internal | pb_systems
    print(f'test set (internal U PB): {len(test_all)}')

    con = duckdb.connect()
    con.execute('PRAGMA threads=16')
    con.execute(f'''
        CREATE TEMP TABLE train_sys (system_id VARCHAR);
        INSERT INTO train_sys VALUES {','.join(f"('{s}')" for s in train)};
    ''')
    con.execute(f'''
        CREATE TEMP TABLE test_sys (system_id VARCHAR);
        INSERT INTO test_sys VALUES {','.join(f"('{s}')" for s in test_all)};
    ''')

    paths = ' UNION ALL '.join(
        f"SELECT query_system, target_system, similarity FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov'" for p in SCORE_DIRS
    )

    print('\nQuerying score shards (this is the slow step)...')
    res = con.execute(f'''
        WITH s AS ({paths})
        SELECT s.query_system AS train_sid, MAX(s.similarity) AS max_pli_qcov
        FROM s
        WHERE s.query_system IN (SELECT system_id FROM train_sys)
          AND s.target_system IN (SELECT system_id FROM test_sys)
        GROUP BY s.query_system
    ''').fetchdf()
    print(f'train systems with >=1 score row: {len(res)} / {len(train)}')

    # Fill missing with 0
    out = pd.DataFrame({'system_id': sorted(train)})
    out = out.merge(res, left_on='system_id', right_on='train_sid', how='left')
    out['max_pli_qcov'] = out['max_pli_qcov'].fillna(0.0)
    out = out[['system_id', 'max_pli_qcov']]
    out.to_csv(DATA / 'logs' / 'pli_qcov_max_per_train.csv', index=False)

    # Stats + histogram
    import numpy as np
    vals = out['max_pli_qcov'].values
    bins = [0, 50, 70, 85, 90, 95, 99, 100.001]
    hist, _ = np.histogram(vals, bins=bins)
    pcts = {f'p{p}': float(np.percentile(vals, p)) for p in [50, 75, 90, 95, 99]}
    summary = {
        'n_train': int(len(train)),
        'n_train_with_score_row': int(len(res)),
        'n_test_set': int(len(test_all)),
        'mean_max_pli_qcov': float(vals.mean()),
        'percentiles': pcts,
        'max': float(vals.max()),
        'histogram': {f'[{bins[i]:g},{bins[i+1]:g})': int(hist[i]) for i in range(len(hist))},
        'fraction_above': {f'>={t}': float((vals >= t).mean()) for t in [50, 70, 85, 90, 95, 99]},
        'count_above':    {f'>={t}': int((vals >= t).sum())   for t in [50, 70, 85, 90, 95, 99]},
    }
    (DATA / 'logs' / 'pli_qcov_leakage.json').write_text(json.dumps(summary, indent=2))
    print('\n=== max(pli_qcov) over train ↔ (internal_test ∪ PB) ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
