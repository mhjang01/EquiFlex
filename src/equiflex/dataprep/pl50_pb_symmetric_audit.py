'''
PL50 train PoseBusters leakage, SYMMETRIC (both directions) + maximal PB recovery.

PB target set:
  - exact (pdb, ccd) subsystems from the FULL annotation index, plus
  - PDB-level fallback for PB pdbs with no exact (pdb,ccd) match
    (recovers CCD synonyms e.g. H4B->BH4, NOJ->DNJ, and local-download gaps).
Only the 3 PDBs truly absent from the v2 index remain unrecoverable.

Leakage metric per train system = MAX(similarity) over BOTH directions:
   (train=query, PB=target)  OR  (PB=query, train=target)
for metric in {pli_qcov, pocket_lddt}, similarity in [0,100].

Output: data_v2_plinder/logs/pl50_pb_symmetric_audit.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
PL50_PQ = '/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet'
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
METRICS = ['pli_qcov', 'pocket_lddt']
THRESH = 50.0


def build_pb_systems():
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)

    annot = pq.read_table(
        ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_ccd_code']
    ).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()

    in_pb_pdb = annot[annot['pdb'].isin(pb_pdbs)]
    # exact (pdb, ccd)
    exact = in_pb_pdb[[(p, c) in pairs for p, c in zip(in_pb_pdb['pdb'], in_pb_pdb['ccd'])]]
    exact_systems = set(exact['system_id'])
    pdbs_with_exact = set(exact['pdb'])

    # fallback: PB pdbs with no exact ccd match -> take all their subsystems
    pdbs_fallback = pb_pdbs - pdbs_with_exact
    fallback = in_pb_pdb[in_pb_pdb['pdb'].isin(pdbs_fallback)]
    fallback_systems = set(fallback['system_id'])

    pb_systems = exact_systems | fallback_systems
    pdbs_unrecoverable = pdbs_fallback - set(in_pb_pdb['pdb'])
    info = {
        'n_pb_pairs': len(pairs),
        'n_pb_pdbs': len(pb_pdbs),
        'n_systems_exact_ccd': len(exact_systems),
        'n_pdbs_with_exact': len(pdbs_with_exact),
        'n_pdbs_fallback_pdblevel': len(pdbs_fallback),
        'n_systems_fallback': len(fallback_systems),
        'n_pb_systems_total': len(pb_systems),
        'pdbs_unrecoverable': sorted(pdbs_unrecoverable),
    }
    return pb_systems, info


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))
    print(f'PL50 train: {len(train)}')

    pb_systems, info = build_pb_systems()
    print(json.dumps(info, indent=2))

    con = duckdb.connect()
    con.execute('PRAGMA threads=16')
    con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute(f"CREATE TEMP TABLE train_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in train)}) AS t(system_id);")
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")
    metric_list = ','.join(f"'{m}'" for m in METRICS)

    # Push train/PB filters into each parquet scan so only the (small) matching
    # rows materialize. dir='fwd' => train=query->PB; dir='rev' => PB=query->train.
    # tsid = the train system regardless of direction.
    fwd = ' UNION ALL '.join(
        f"SELECT query_system AS tsid, metric, similarity AS sim "
        f"FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({metric_list}) "
        f"AND query_system IN (SELECT system_id FROM train_sys) "
        f"AND target_system IN (SELECT system_id FROM pb_sys)"
        for p in SCORE_DIRS)
    rev = ' UNION ALL '.join(
        f"SELECT target_system AS tsid, metric, similarity AS sim "
        f"FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({metric_list}) "
        f"AND target_system IN (SELECT system_id FROM train_sys) "
        f"AND query_system IN (SELECT system_id FROM pb_sys)"
        for p in SCORE_DIRS)

    print('Scanning scores (forward train->PB)...')
    res_dir = con.execute(
        f"SELECT tsid, metric, MAX(sim) AS m FROM ({fwd}) GROUP BY tsid, metric"
    ).fetchdf()
    print(f'forward rows: {len(res_dir)}')
    print('Scanning scores (reverse PB->train)...')
    res_rev = con.execute(
        f"SELECT tsid, metric, MAX(sim) AS m FROM ({rev}) GROUP BY tsid, metric"
    ).fetchdf()
    print(f'reverse rows: {len(res_rev)}')

    # symmetric = max over both directions per (tsid, metric)
    res = (pd.concat([res_dir, res_rev], ignore_index=True)
           .groupby(['tsid', 'metric'], as_index=False)['m'].max())
    print(f'symmetric rows: {len(res)}')

    summary = {'split': 'PL50', 'n_train': len(train), 'pb_selection': info,
               'threshold': THRESH, 'symmetric': {}, 'directed_train_to_pb': {}}

    def stats(df, train):
        out = {}
        for metric in METRICS:
            sub = df[df['metric'] == metric]
            mp = dict(zip(sub['tsid'], sub['m']))
            vals = np.array([mp.get(s, 0.0) for s in train], dtype=float)
            out[metric] = {
                'n_train_scored': int(len(sub)),
                'max': float(vals.max()),
                f'n_ge_{THRESH:g}': int((vals >= THRESH).sum()),
                f'fraction_ge_{THRESH:g}': float((vals >= THRESH).mean()),
                'mean': float(vals.mean()),
                'percentiles': {f'p{p}': float(np.percentile(vals, p)) for p in [50, 75, 90, 95, 99]},
                'count_above': {f'>={t}': int((vals >= t).sum()) for t in [50, 70, 85, 95]},
            }
        return out

    summary['symmetric'] = stats(res, train)
    summary['directed_train_to_pb'] = stats(res_dir, train)

    out = OUTDIR / 'pl50_pb_symmetric_audit.json'
    out.write_text(json.dumps(summary, indent=2))
    print('\n=== SYMMETRIC ===')
    for m in METRICS:
        s = summary['symmetric'][m]
        print(f"  {m}: frac>=50={s['fraction_ge_50']:.4f}  n>=50={s['n_ge_50']}  max={s['max']:.0f}")
    print('=== DIRECTED (train->PB) ===')
    for m in METRICS:
        s = summary['directed_train_to_pb'][m]
        print(f"  {m}: frac>=50={s['fraction_ge_50']:.4f}  n>=50={s['n_ge_50']}  max={s['max']:.0f}")
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
