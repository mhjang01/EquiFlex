'''
Survivors of data_v2_plinder/train when filtering by max(metric) < 50 for
each candidate metric, vs test set (internal_test ∪ PoseBusters).
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
PB_FILE  = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'

METRICS = ['pli_qcov', 'pli_unique_qcov', 'pocket_lddt', 'pocket_lddt_qcov']
THRESHOLD = 50.0


def main():
    splits = pd.read_csv(DATA/'splits.csv')
    train = set(splits[splits['split']=='train']['system_id'])
    test_internal = set(splits[splits['split']=='test']['system_id'])
    pb_pdbs = {l.strip().split('_')[0].lower() for l in open(PB_FILE) if l.strip()}
    annot = pq.read_table(ANNOT_PQ, columns=['system_id','entry_pdb_id']).to_pandas().drop_duplicates('system_id')
    pb_systems = set(annot[annot['entry_pdb_id'].str.lower().isin(pb_pdbs)]['system_id'])
    test_all = test_internal | pb_systems
    print(f'train={len(train)}  test_set={len(test_all)}')

    con = duckdb.connect()
    con.execute('PRAGMA threads=16')
    con.execute(f'''
        CREATE TEMP TABLE train_sys AS
        SELECT system_id FROM (VALUES {','.join(f"('{s}')" for s in train)}) AS t(system_id);
    ''')
    con.execute(f'''
        CREATE TEMP TABLE test_sys AS
        SELECT system_id FROM (VALUES {','.join(f"('{s}')" for s in test_all)}) AS t(system_id);
    ''')

    summary = {'threshold': THRESHOLD}
    for metric in METRICS:
        print(f'\nquerying {metric}...')
        union_sql = ' UNION ALL '.join(
            f"SELECT query_system, target_system, similarity FROM read_parquet('{p}', union_by_name=True) "
            f"WHERE metric='{metric}'" for p in SCORE_DIRS
        )
        res = con.execute(f'''
            WITH s AS ({union_sql})
            SELECT s.query_system AS sid, MAX(s.similarity) AS m
            FROM s
            WHERE s.query_system IN (SELECT system_id FROM train_sys)
              AND s.target_system IN (SELECT system_id FROM test_sys)
            GROUP BY s.query_system
        ''').fetchdf()
        max_map = dict(zip(res['sid'], res['m']))
        max_vals = [max_map.get(s, 0.0) for s in train]
        import numpy as np
        vals = np.array(max_vals)
        leaked = int((vals >= THRESHOLD).sum())
        survive = int(len(train) - leaked)
        summary[metric] = {
            'n_train_with_score': len(res),
            f'n_above_{THRESHOLD:g}': leaked,
            f'n_survive_at_<{THRESHOLD:g}': survive,
            'percentiles': {f'p{p}': float(np.percentile(vals, p)) for p in [50,75,90,95,99]},
            'mean': float(vals.mean()),
            'max':  float(vals.max()),
        }
        print(f'  {metric}: surviving (<{THRESHOLD}): {survive}  / dropped: {leaked}')

    (DATA/'logs'/'multi_metric_leakage.json').write_text(json.dumps(summary, indent=2))
    print('\n=== Summary ===')
    print(f'{"metric":<24}{"survive (<50)":<16}{"dropped":<10}{"mean":<8}{"p95":<8}{"p99":<8}')
    for m in METRICS:
        s = summary[m]
        print(f'{m:<24}{s[f"n_survive_at_<{THRESHOLD:g}"]:<16}{s[f"n_above_{THRESHOLD:g}"]:<10}'
              f'{s["mean"]:<8.2f}{s["percentiles"]["p95"]:<8.2f}{s["percentiles"]["p99"]:<8.2f}')


if __name__ == '__main__':
    main()
