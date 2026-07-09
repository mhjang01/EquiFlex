'''
After pairwise pli_qcov dedup of PL50 train vs PoseBusters, recompute the
leakage metrics (pli_qcov 'pli shared' and pocket_lddt) on the survivor set.

Shows that pli_qcov dedup lowers pli_qcov leakage but NOT pocket_lddt (different metric).

Per PL50 train system: symmetric MAX pli_qcov and MAX pocket_lddt vs 548 PB systems.
Survivors at pli_qcov-dedup thresholds {90,95}: recompute fraction>=50 and max
for BOTH metrics.

Output: data_v2_plinder/logs/pl50_postdedup_metrics.json
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
METRICS = ['pli_qcov', 'pocket_lddt']
DEDUP_THRS = [90, 95]


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))

    annot = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'entry_pdb_id', 'ligand_ccd_code']).to_pandas()
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
    ml = ','.join(f"'{m}'" for m in METRICS)
    fwd = ' UNION ALL '.join(
        f"SELECT query_system AS tsid, metric, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({ml}) AND query_system IN (SELECT system_id FROM train_sys) "
        f"AND target_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    rev = ' UNION ALL '.join(
        f"SELECT target_system AS tsid, metric, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({ml}) AND target_system IN (SELECT system_id FROM train_sys) "
        f"AND query_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    print('fwd...'); rf = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({fwd}) GROUP BY tsid, metric").fetchdf()
    print('rev...'); rr = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({rev}) GROUP BY tsid, metric").fetchdf()
    sym = pd.concat([rf, rr]).groupby(['tsid', 'metric'], as_index=False)['m'].max()

    pli = {r.tsid: r.m for r in sym[sym.metric == 'pli_qcov'].itertuples()}
    lddt = {r.tsid: r.m for r in sym[sym.metric == 'pocket_lddt'].itertuples()}
    pli_v = np.array([pli.get(s, 0.0) for s in train])
    lddt_v = np.array([lddt.get(s, 0.0) for s in train])

    def block(mask, label):
        keep = mask
        pv, lv = pli_v[keep], lddt_v[keep]
        n = int(keep.sum())
        return {
            'label': label, 'n_train': n,
            'pli_qcov': {'frac_ge_50': float((pv >= 50).mean()), 'n_ge_50': int((pv >= 50).sum()), 'max': float(pv.max())},
            'pocket_lddt': {'frac_ge_50': float((lv >= 50).mean()), 'n_ge_50': int((lv >= 50).sum()), 'max': float(lv.max())},
        }

    out = {'n_train': len(train), 'n_pb_systems': len(pb_systems), 'results': []}
    out['results'].append(block(np.ones(len(train), bool), 'baseline (no dedup)'))
    for thr in DEDUP_THRS:
        out['results'].append(block(pli_v < thr, f'after pairwise pli_qcov<{thr} dedup'))

    OUTDIR.joinpath('pl50_postdedup_metrics.json').write_text(json.dumps(out, indent=2))
    print(f"\n{'set':<34}{'n':>8}{'pli>=50':>10}{'pli_max':>9}{'lddt>=50':>11}{'lddt_max':>10}")
    for r in out['results']:
        print(f"{r['label']:<34}{r['n_train']:>8}"
              f"{r['pli_qcov']['frac_ge_50']*100:>9.2f}%{r['pli_qcov']['max']:>9.0f}"
              f"{r['pocket_lddt']['frac_ge_50']*100:>10.2f}%{r['pocket_lddt']['max']:>10.0f}")
    print('\nWritten:', OUTDIR / 'pl50_postdedup_metrics.json')


if __name__ == '__main__':
    main()
