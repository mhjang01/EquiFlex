'''
pli_qcov leakage stat for past_data/plinder_data/ (legacy flat <pdb>_<chain> dirs).

Mapping <pdb>_<chain> -> Plinder system_ids:
   take all system_ids in annotation_table where entry_pdb_id == <pdb>.
   (Coarse but captures any leakage at the PDB level.)

For each short_id, compute max(pli_qcov) over its Plinder counterparts
against the test set = (data_v2_plinder/test ∪ PoseBusters Plinder sids).

Output:
   past_data/plinder_data/pli_qcov_leakage.json
   past_data/plinder_data/pli_qcov_max_per_short_id.csv
'''
import json
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA  = Path('/home/mhjang/workspace/flex/past_data/plinder_data')
V2    = Path('/home/mhjang/workspace/flex/data_v2_plinder')
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE  = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'


def main():
    short_ids = sorted(d.name for d in DATA.iterdir() if d.is_dir())
    print(f'short_ids in plinder_data: {len(short_ids)}')
    pdb_to_short = defaultdict(list)
    for s in short_ids:
        pdb = s.split('_')[0].lower()
        pdb_to_short[pdb].append(s)
    print(f'unique pdb_ids: {len(pdb_to_short)}')

    annot = pq.read_table(ANNOT_PQ, columns=['system_id', 'entry_pdb_id']).to_pandas()
    annot = annot.drop_duplicates('system_id')
    annot['pdb_lower'] = annot['entry_pdb_id'].str.lower()
    matched = annot[annot['pdb_lower'].isin(pdb_to_short)]
    print(f'matched Plinder system_ids: {len(matched)}')

    sid_to_short = {}
    for _, r in matched.iterrows():
        for s in pdb_to_short[r['pdb_lower']]:
            sid_to_short[r['system_id']] = s

    splits = pd.read_csv(V2 / 'splits.csv')
    test_internal = set(splits[splits['split'] == 'test']['system_id'])
    pb_pdbs = {l.strip().split('_')[0].lower() for l in open(PB_FILE) if l.strip()}
    pb_systems = set(annot[annot['pdb_lower'].isin(pb_pdbs)]['system_id'])
    test_all = test_internal | pb_systems
    print(f'test set (internal U PB): {len(test_all)}')

    query_sids = set(sid_to_short)
    print(f'Plinder query sids (for plinder_data short_ids): {len(query_sids)}')

    con = duckdb.connect()
    con.execute('PRAGMA threads=16')
    con.execute(f'''
        CREATE TEMP TABLE query_sys AS
        SELECT system_id FROM (VALUES {','.join(f"('{s}')" for s in query_sids)})
        AS t(system_id);
    ''')
    con.execute(f'''
        CREATE TEMP TABLE test_sys AS
        SELECT system_id FROM (VALUES {','.join(f"('{s}')" for s in test_all)})
        AS t(system_id);
    ''')

    union_sql = ' UNION ALL '.join(
        f"SELECT query_system, target_system, similarity FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov'" for p in SCORE_DIRS
    )

    print('Querying score shards...')
    res = con.execute(f'''
        WITH s AS ({union_sql})
        SELECT s.query_system AS sid, MAX(s.similarity) AS max_pli_qcov
        FROM s
        WHERE s.query_system IN (SELECT system_id FROM query_sys)
          AND s.target_system IN (SELECT system_id FROM test_sys)
        GROUP BY s.query_system
    ''').fetchdf()
    print(f'rows with score: {len(res)}')

    # Map back to short_id and take max per short_id
    res['short_id'] = res['sid'].map(sid_to_short)
    agg = res.groupby('short_id')['max_pli_qcov'].max().reset_index()
    print(f'short_ids with any score: {len(agg)}')

    out = pd.DataFrame({'short_id': short_ids}).merge(agg, on='short_id', how='left')
    out['max_pli_qcov'] = out['max_pli_qcov'].fillna(0.0)
    out.to_csv(DATA / 'pli_qcov_max_per_short_id.csv', index=False)

    vals = out['max_pli_qcov'].values
    bins = [0, 50, 70, 85, 90, 95, 99, 100.001]
    hist, _ = np.histogram(vals, bins=bins)
    summary = {
        'n_short_ids': int(len(short_ids)),
        'n_unique_pdbs': int(len(pdb_to_short)),
        'n_test_set': int(len(test_all)),
        'mean_max_pli_qcov': float(vals.mean()),
        'percentiles': {f'p{p}': float(np.percentile(vals, p)) for p in [50, 75, 90, 95, 99]},
        'max': float(vals.max()),
        'histogram': {f'[{bins[i]:g},{bins[i+1]:g})': int(hist[i]) for i in range(len(hist))},
        'count_above': {f'>={t}': int((vals >= t).sum())  for t in [50, 70, 85, 90, 95, 99]},
        'fraction_above': {f'>={t}': float((vals >= t).mean()) for t in [50, 70, 85, 90, 95, 99]},
    }
    (DATA / 'pli_qcov_leakage.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
