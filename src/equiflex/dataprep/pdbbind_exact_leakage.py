'''
Refined pdbbind PoseBusters leakage: map each pdbbind train system to its SPECIFIC
plinder system(s) by (PDB, RDKit-canonical ligand SMILES) — not PDB alone — so
pli/lddt/seqsim reflect the actual pdbbind complex, not other-ligand PDB-mates.

PB target = 540 exact (PDB,CCD) systems. Metrics symmetric MAX. Merges with the
existing plinder block from leakage_compare_plinder_pdbbind.json.

Output: data_v2_plinder/logs/pdbbind_exact_leakage.json (+ updates comparison json)
'''
import json
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import load, Parallel, delayed
from rdkit import Chem

ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
PDBBIND = Path('/home/mhjang/workspace/flex/data_pdbbind')
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
METRICS = ['pli_qcov', 'pocket_lddt', 'protein_seqsim_max']
THR = 50.0
LOG = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')


def canon(smi):
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    pb_df = pd.read_csv(PDBBIND / 'splits.csv')
    pdb_train = sorted(set(pb_df[pb_df['split'] == 'train']['system_id'].astype(str).str.lower()))

    # pdbbind ligand canonical SMILES per PDB
    def lig(pid):
        fp = PDBBIND / pid / 'ligands.pkl'
        try:
            return pid, {canon(x.get('smiles')) for x in load(fp) if x.get('smiles')} - {None} if fp.exists() else (pid, set())
        except Exception:
            return pid, set()
    pdb_lig = dict(Parallel(n_jobs=-1)(delayed(lig)(p) for p in pdb_train))
    print(f'pdbbind train pdbs={len(pdb_train)}', flush=True)

    # plinder candidate rows for those PDBs, canonicalized
    annot = pq.read_table(ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_ccd_code',
                                             'ligand_rdkit_canonical_smiles']).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()
    cand = annot[annot['pdb'].isin(set(pdb_train))].copy()
    uniq_smiles = {s for s in cand['ligand_rdkit_canonical_smiles'].dropna().unique()}
    canon_map = {s: canon(s) for s in uniq_smiles}
    cand['cs'] = cand['ligand_rdkit_canonical_smiles'].map(canon_map)

    # match (pdb, canonical smiles)
    sid_to_pdb = {}
    matched_pdbs = set()
    for sid, p, cs in zip(cand['system_id'], cand['pdb'], cand['cs']):
        if cs is not None and cs in pdb_lig.get(p, set()):
            sid_to_pdb[sid] = p
            matched_pdbs.add(p)
    query_sids = sorted(sid_to_pdb)
    print(f'matched pdbs={len(matched_pdbs)}/{len(pdb_train)}  query sids={len(query_sids)}', flush=True)

    # PB exact target
    pairs = set((l.strip().split('_')[0].lower(), l.strip().split('_', 1)[1].upper())
                for l in open(PB_FILE) if l.strip())
    ap = annot[annot['pdb'].isin({p for p, _ in pairs})]
    exact = ap[[(p, c) in pairs for p, c in zip(ap['pdb'], ap['ccd'])]]
    pb_systems = sorted(set(exact['system_id']))

    con = duckdb.connect()
    con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")
    con.execute("CREATE TEMP TABLE q (system_id VARCHAR);")
    con.executemany("INSERT INTO q VALUES (?)", [(s,) for s in query_sids])
    ml = ','.join(f"'{m}'" for m in METRICS)
    fwd = ' UNION ALL '.join(
        f"SELECT query_system AS sid, metric, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({ml}) AND query_system IN (SELECT system_id FROM q) "
        f"AND target_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    rev = ' UNION ALL '.join(
        f"SELECT target_system AS sid, metric, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric IN ({ml}) AND target_system IN (SELECT system_id FROM q) "
        f"AND query_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    print('fwd...', flush=True); rf = con.execute(f"SELECT sid, metric, MAX(sim) m FROM ({fwd}) GROUP BY sid, metric").fetchdf()
    print('rev...', flush=True); rr = con.execute(f"SELECT sid, metric, MAX(sim) m FROM ({rev}) GROUP BY sid, metric").fetchdf()
    sym = pd.concat([rf, rr]).groupby(['sid', 'metric'], as_index=False)['m'].max()
    sym['pdb'] = sym['sid'].map(sid_to_pdb)

    block = {'n_units': len(pdb_train), 'matched_pdbs': len(matched_pdbs), 'unit': 'pdb_code(PDB+SMILES matched)'}
    for m in METRICS:
        sub = sym[sym.metric == m]
        per = sub.groupby('pdb')['m'].max().to_dict()
        v = np.array([per.get(p, 0.0) for p in pdb_train])
        block[m] = {'max': float(v.max()), 'frac_ge_50': float((v >= THR).mean()),
                    'n_ge_50': int((v >= THR).sum()),
                    'count_above': {f'>={t}': int((v >= t).sum()) for t in [50, 70, 90, 95]}}

    LOG.joinpath('pdbbind_exact_leakage.json').write_text(json.dumps(block, indent=2))
    print('\n=== pdbbind (PDB+SMILES exact match) vs PoseBusters ===')
    for m in METRICS:
        s = block[m]
        print(f"  {m}: frac>=50={s['frac_ge_50']:.3f} max={s['max']:.0f} ge70={s['count_above']['>=70']} ge90={s['count_above']['>=90']} ge95={s['count_above']['>=95']}")
    print(json.dumps(block, indent=2))


if __name__ == '__main__':
    main()
