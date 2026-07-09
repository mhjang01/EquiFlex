'''
PoseBusters leakage audit for the PL50 split (v0/splits/plinder-pl50.parquet).

Same checks as posebusters_leakage_audit.py but query set = PL50 train systems.
Target = PoseBusters Plinder systems. Scores from v2 score parquets
(system_ids are shared across plinder versions).

Output: data_v2_plinder/logs/pl50_posebusters_leakage_audit.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
PL50_PQ = '/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet'
SYSTEMS = Path('/mnt/drive/plinder/2024-06/v2/systems')
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
METRICS = ['pli_qcov', 'pocket_lddt']
THRESH = 50.0


def read_fasta_seqs(fp):
    try:
        text = fp.read_text()
    except OSError:
        return set()
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith('>'):
            if cur:
                seqs.append(''.join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append(''.join(cur))
    return set(s for s in seqs if s)


def pb_seqs(sid):
    fp = SYSTEMS / sid / 'sequences.fasta'
    try:
        if not fp.exists():
            return set()
    except OSError:
        return set()
    return read_fasta_seqs(fp)


def sys_seqs(sid):
    '''Sequence set for any system from its raw plinder sequences.fasta.'''
    return sid, pb_seqs(sid)


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))
    print(f'PL50 train systems: {len(train)}')

    pb_pdbs = {l.strip().split('_')[0].lower() for l in open(PB_FILE) if l.strip()}
    annot = pq.read_table(
        ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_rdkit_canonical_smiles']
    ).to_pandas()
    annot['pdb_lower'] = annot['entry_pdb_id'].str.lower()
    sys_smiles = annot.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))
    annot_sys = annot.drop_duplicates('system_id')
    pb_systems = sorted(set(annot_sys[annot_sys['pdb_lower'].isin(pb_pdbs)]['system_id']))
    print(f'PB Plinder systems: {len(pb_systems)}')

    # ---- metrics: train -> PB via duckdb ----
    con = duckdb.connect()
    con.execute('PRAGMA threads=16')
    con.execute(f"CREATE TEMP TABLE train_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in train)}) AS t(system_id);")
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")
    metric_list = ','.join(f"'{m}'" for m in METRICS)
    union_sql = ' UNION ALL '.join(
        f"SELECT query_system, target_system, metric, similarity "
        f"FROM read_parquet('{p}', union_by_name=True) WHERE metric IN ({metric_list})"
        for p in SCORE_DIRS)
    print('Scanning score parquets...')
    res = con.execute(f'''
        WITH s AS ({union_sql})
        SELECT s.query_system AS sid, s.metric AS metric, MAX(s.similarity) AS m
        FROM s
        WHERE s.query_system IN (SELECT system_id FROM train_sys)
          AND s.target_system IN (SELECT system_id FROM pb_sys)
        GROUP BY s.query_system, s.metric
    ''').fetchdf()
    print(f'rows: {len(res)}')

    summary = {'split': 'PL50', 'n_train': len(train),
               'n_pb_pdb_codes': len(pb_pdbs), 'n_pb_systems': len(pb_systems),
               'threshold': THRESH, 'metrics': {}}
    for metric in METRICS:
        sub = res[res['metric'] == metric]
        max_map = dict(zip(sub['sid'], sub['m']))
        vals = np.array([max_map.get(s, 0.0) for s in train], dtype=float)
        summary['metrics'][metric] = {
            'n_train_scored': int(len(sub)),
            'max': float(vals.max()),
            f'n_ge_{THRESH:g}': int((vals >= THRESH).sum()),
            f'fraction_ge_{THRESH:g}': float((vals >= THRESH).mean()),
            'mean': float(vals.mean()),
            'percentiles': {f'p{p}': float(np.percentile(vals, p)) for p in [50, 75, 90, 95, 99]},
            'count_above': {f'>={t}': int((vals >= t).sum()) for t in [50, 70, 85, 95, 99]},
        }
        pd.DataFrame({'system_id': train, f'max_{metric}_vs_PB': vals}).to_csv(
            OUTDIR / f'pl50_pb_{metric}_max_per_train.csv', index=False)

    # ---- exact match ----
    pb_smiles = set()
    for s in pb_systems:
        pb_smiles |= sys_smiles.get(s, set())
    train_smiles_map = {s: sys_smiles.get(s, set()) for s in train}
    smiles_hits = [s for s in train if train_smiles_map[s] & pb_smiles]

    pb_seq_all = set().union(*Parallel(n_jobs=-1)(delayed(pb_seqs)(s) for s in pb_systems))
    tr = Parallel(n_jobs=-1)(delayed(sys_seqs)(s) for s in train)
    train_seq_map = {sid: ss for sid, ss in tr}
    seq_hits = [s for s in train if train_seq_map[s] & pb_seq_all]
    both_hits = [s for s in train if (train_smiles_map[s] & pb_smiles)
                 and (train_seq_map[s] & pb_seq_all)]

    summary['exact_match'] = {
        'n_pb_unique_smiles': len(pb_smiles),
        'n_pb_unique_sequences': len(pb_seq_all),
        'smiles_exact_match': {'n_train': len(smiles_hits),
                               'fraction': len(smiles_hits) / len(train), 'examples': smiles_hits[:15]},
        'sequence_exact_match': {'n_train': len(seq_hits),
                                 'fraction': len(seq_hits) / len(train), 'examples': seq_hits[:15]},
        'both_smiles_and_sequence': {'n_train': len(both_hits),
                                     'fraction': len(both_hits) / len(train), 'examples': both_hits[:15]},
    }
    out = OUTDIR / 'pl50_posebusters_leakage_audit.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
