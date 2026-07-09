'''
PL50 train: pairwise dedup vs PoseBusters on BOTH pli_qcov AND pocket_lddt at 90.
Remove a train system if max(pli_qcov) >= 90 OR max(pocket_lddt) >= 90 (symmetric, vs 548 PB).

Report survivors, residual leakage (pli>=50 / lddt>=50 / maxes), and ligand
diversity (unique proper-ligand canonical SMILES) retained. Compare to each
filter alone and to component-removal (-45% diversity).

Output: data_v2_plinder/logs/pl50_combined_dedup90.json
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
THR = 90


def main():
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    train = sorted(set(pl[pl['split'] == 'train']['system_id']))

    annot = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'entry_pdb_id', 'ligand_ccd_code',
        'ligand_rdkit_canonical_smiles', 'ligand_is_proper']).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    a = annot[annot['pdb'].isin(pb_pdbs)]
    exact = a[[(p, c) in pairs for p, c in zip(a['pdb'], a['ccd'])]]
    fb = a[a['pdb'].isin(pb_pdbs - set(exact['pdb']))]
    pb_systems = set(exact['system_id']) | set(fb['system_id'])

    prop = annot[(annot['system_id'].isin(train)) & (annot['ligand_is_proper'] == True)]
    sys_smiles = prop.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))

    def uniq(systems):
        u = set()
        for s in systems:
            u |= sys_smiles.get(s, set())
        return len(u)
    base_u = uniq(train)

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

    def report(mask, label):
        keep = [s for s, k in zip(train, mask) if k]
        pv, lv = pli_v[mask], lddt_v[mask]
        u = uniq(keep)
        return {'label': label, 'n_train': int(mask.sum()),
                'frac_systems_kept': float(mask.mean()),
                'unique_smiles': u, 'diversity_retained': u / base_u,
                'residual_pli_ge50': float((pv >= 50).mean()), 'pli_max': float(pv.max()),
                'residual_lddt_ge50': float((lv >= 50).mean()), 'lddt_max': float(lv.max())}

    pli_ok = pli_v < THR
    lddt_ok = lddt_v < THR
    both_ok = pli_ok & lddt_ok
    out = {'n_train': len(train), 'n_pb_systems': len(pb_systems),
           'base_unique_proper_smiles': base_u, 'threshold': THR, 'results': [
        report(np.ones(len(train), bool), 'baseline'),
        report(pli_ok, 'pli_qcov<90 only'),
        report(lddt_ok, 'pocket_lddt<90 only'),
        report(both_ok, 'BOTH pli_qcov<90 AND pocket_lddt<90'),
    ]}
    # overlap diagnostics
    out['removed_counts'] = {
        'pli_ge90': int((~pli_ok).sum()), 'lddt_ge90': int((~lddt_ok).sum()),
        'either_ge90': int((~both_ok).sum()),
        'both_metrics_ge90': int(((~pli_ok) & (~lddt_ok)).sum())}

    OUTDIR.joinpath('pl50_combined_dedup90.json').write_text(json.dumps(out, indent=2))
    print(f"\n{'set':<40}{'n':>8}{'kept%':>8}{'uniqSMI':>9}{'div%':>8}{'pli>=50':>9}{'lddt>=50':>10}")
    for r in out['results']:
        print(f"{r['label']:<40}{r['n_train']:>8}{r['frac_systems_kept']*100:>7.1f}%"
              f"{r['unique_smiles']:>9}{r['diversity_retained']*100:>7.1f}%"
              f"{r['residual_pli_ge50']*100:>8.1f}%{r['residual_lddt_ge50']*100:>9.1f}%")
    print('\nremoved:', out['removed_counts'])
    print('Written:', OUTDIR / 'pl50_combined_dedup90.json')


if __name__ == '__main__':
    main()
