'''
For PL50 train survivors at pairwise pli_qcov<90 vs PoseBusters, count how many
systems still need parsing (not present, or missing required featurization files)
in data_v2_plinder/train.

Required files (per create_splits): ligand_features.pt, pocket_features.pt,
esm3_embeddings.pt, unimol_embeddings.pt.

Writes:
  data_v2_plinder/logs/pl50_pli90_survivors.txt
  data_v2_plinder/logs/pl50_pli90_to_prepare.txt
  data_v2_plinder/logs/pl50_pli90_prep_gap.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TRAIN_DIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
PL50_PQ = '/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet'
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
REQUIRED = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']
THR = 90


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))

    annot = pq.read_table(ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_ccd_code']).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    a = annot[annot['pdb'].isin(pb_pdbs)]
    exact = a[[(p, c) in pairs for p, c in zip(a['pdb'], a['ccd'])]]
    fb = a[a['pdb'].isin(pb_pdbs - set(exact['pdb']))]
    pb_systems = set(exact['system_id']) | set(fb['system_id'])

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
    print('fwd...'); rf = con.execute(f"SELECT tsid, MAX(sim) m FROM ({fwd}) GROUP BY tsid").fetchdf()
    print('rev...'); rr = con.execute(f"SELECT tsid, MAX(sim) m FROM ({rev}) GROUP BY tsid").fetchdf()
    sym = pd.concat([rf, rr]).groupby('tsid', as_index=False)['m'].max()
    mx = dict(zip(sym['tsid'], sym['m']))
    survivors = [s for s in train if mx.get(s, 0.0) < THR]
    print(f'survivors (pli_qcov<{THR}): {len(survivors)} / {len(train)}')

    # parse status
    complete, incomplete, absent = [], [], []
    for s in survivors:
        d = TRAIN_DIR / s
        if not d.is_dir():
            absent.append(s); continue
        if all((d / f).exists() for f in REQUIRED):
            complete.append(s)
        else:
            incomplete.append(s)
    to_prepare = absent + incomplete

    (OUTDIR / 'pl50_pli90_survivors.txt').write_text('\n'.join(survivors) + '\n')
    (OUTDIR / 'pl50_pli90_to_prepare.txt').write_text('\n'.join(sorted(to_prepare)) + '\n')
    summary = {
        'pl50_train': len(train), 'pb_systems': len(pb_systems),
        'survivors_pli_qcov_lt90': len(survivors),
        'already_complete': len(complete),
        'incomplete_needs_refeaturize': len(incomplete),
        'absent_needs_full_parse': len(absent),
        'total_to_prepare': len(to_prepare),
        'required_files': REQUIRED,
    }
    (OUTDIR / 'pl50_pli90_prep_gap.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Written: pl50_pli90_survivors.txt, pl50_pli90_to_prepare.txt, pl50_pli90_prep_gap.json')


if __name__ == '__main__':
    main()
