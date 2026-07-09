'''
Pairwise pli_qcov dedup of PL50 train vs PoseBusters (548-system set).

For each PL50 train system: symmetric MAX pli_qcov to any PB system (direct
pairwise, from the scores parquet). Then for thresholds {90,95,99,100} report
survivors (max < thr): n_systems and n_unique proper-ligand canonical SMILES.

Contrast with component-level removal (which dropped ~45% of unique SMILES).

Output: data_v2_plinder/logs/pl50_pairwise_pliqcov_dedup.json
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
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
THRESHOLDS = [90, 95, 99, 100]


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))
    print(f'PL50 train: {len(train)}')

    annot = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'entry_pdb_id', 'ligand_ccd_code',
        'ligand_rdkit_canonical_smiles', 'ligand_is_proper']).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()

    # corrected 548 PB set
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    a = annot[annot['pdb'].isin(pb_pdbs)]
    exact = a[[(p, c) in pairs for p, c in zip(a['pdb'], a['ccd'])]]
    pdbs_exact = set(exact['pdb'])
    fb = a[a['pdb'].isin(pb_pdbs - pdbs_exact)]
    pb_systems = set(exact['system_id']) | set(fb['system_id'])
    print(f'PB systems: {len(pb_systems)}')

    # proper-ligand canonical smiles per train system
    prop = annot[(annot['system_id'].isin(train)) & (annot['ligand_is_proper'] == True)]
    sys_smiles = prop.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))

    def uniq_smiles(systems):
        u = set()
        for s in systems:
            u |= sys_smiles.get(s, set())
        return len(u)

    base_unique = uniq_smiles(train)
    print(f'base unique proper SMILES: {base_unique}')

    # symmetric pairwise max pli_qcov train<->PB
    con = duckdb.connect()
    con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute(f"CREATE TEMP TABLE train_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in train)}) AS t(system_id);")
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")
    fwd = ' UNION ALL '.join(
        f"SELECT query_system AS tsid, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov' AND query_system IN (SELECT system_id FROM train_sys) "
        f"AND target_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    rev = ' UNION ALL '.join(
        f"SELECT target_system AS tsid, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov' AND target_system IN (SELECT system_id FROM train_sys) "
        f"AND query_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    print('scanning (fwd)...')
    rf = con.execute(f"SELECT tsid, MAX(sim) m FROM ({fwd}) GROUP BY tsid").fetchdf()
    print('scanning (rev)...')
    rr = con.execute(f"SELECT tsid, MAX(sim) m FROM ({rev}) GROUP BY tsid").fetchdf()
    sym = pd.concat([rf, rr]).groupby('tsid', as_index=False)['m'].max()
    maxmap = dict(zip(sym['tsid'], sym['m']))
    vals = np.array([maxmap.get(s, 0.0) for s in train], dtype=float)

    rows = []
    for thr in THRESHOLDS:
        removed = [s for s, v in zip(train, vals) if v >= thr]
        survivors = [s for s, v in zip(train, vals) if v < thr]
        rows.append({
            'threshold': thr,
            'n_removed': len(removed),
            'frac_removed': len(removed) / len(train),
            'n_survivors': len(survivors),
            'unique_smiles_survivors': uniq_smiles(survivors),
            'unique_smiles_retained_frac': uniq_smiles(survivors) / base_unique,
        })

    summary = {
        'split': 'PL50', 'n_train': len(train), 'n_pb_systems': len(pb_systems),
        'base_unique_proper_smiles': base_unique,
        'pairwise_pli_qcov_dedup': rows,
        'note': 'pairwise symmetric MAX pli_qcov vs PB. Compare component-removal: -45% unique SMILES.',
    }
    out = OUTDIR / 'pl50_pairwise_pliqcov_dedup.json'
    out.write_text(json.dumps(summary, indent=2))
    print('\nthr  removed  frac    survivors  uniqSMILES  retained%')
    for r in rows:
        print(f"{r['threshold']:>3} {r['n_removed']:>7} {r['frac_removed']*100:>6.2f}% "
              f"{r['n_survivors']:>9} {r['unique_smiles_survivors']:>10} "
              f"{r['unique_smiles_retained_frac']*100:>8.2f}%")
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
