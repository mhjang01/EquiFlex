'''
PoseBusters leakage comparison: data_v2_plinder vs data_pdbbind (train splits).

PB target = EXACT (PDB, ligand_CCD) matched PLINDER systems only (540) — true
PoseBusters ligand-receptor complexes; no PDB-level fallback (would mix in wrong
ligands and contaminate similarity).

Metrics (symmetric MAX vs PB, similarity 0-100):
  pli  = pli_qcov
  lddt = pocket_lddt
  seqsim = protein_seqsim_max
Ligand: exact canonical-SMILES overlap + max Tanimoto (ECFP4) vs PB ligands.

Query-system mapping:
  data_v2_plinder: system_id IS a plinder system_id  -> direct, unit = system
  data_pdbbind   : bare PDB code -> all plinder systems with that entry_pdb_id
                   (coarse, PDB-level), unit = PDB code (max over its systems)

Output: data_v2_plinder/logs/leakage_compare_plinder_pdbbind.json
'''
import json
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
METRICS = ['pli_qcov', 'pocket_lddt', 'protein_seqsim_max']
THR = 50.0
OUT = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs/leakage_compare_plinder_pdbbind.json')


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def max_tanimoto(train_smiles_set, pb_fps):
    """For each unique train SMILES, max Tanimoto vs PB. Returns array over unique smiles."""
    vals = []
    for s in train_smiles_set:
        f = fp(s)
        if f is None or not pb_fps:
            vals.append(0.0); continue
        vals.append(max(DataStructs.BulkTanimotoSimilarity(f, pb_fps)))
    return np.array(vals) if vals else np.array([0.0])


def main():
    annot = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'entry_pdb_id', 'ligand_ccd_code',
        'ligand_rdkit_canonical_smiles', 'ligand_is_proper']).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()

    # PB exact (pdb,ccd) systems + their matched ligand SMILES
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    ap = annot[annot['pdb'].isin({p for p, _ in pairs})]
    exact_rows = ap[[(p, c) in pairs for p, c in zip(ap['pdb'], ap['ccd'])]]
    pb_systems = sorted(set(exact_rows['system_id']))
    pb_smiles = set(exact_rows['ligand_rdkit_canonical_smiles'].dropna())
    pb_fps = [x for x in (fp(s) for s in pb_smiles) if x is not None]
    print(f'PB exact systems={len(pb_systems)} pb_smiles={len(pb_smiles)} pb_fps={len(pb_fps)}', flush=True)

    con = duckdb.connect()
    con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")

    def score_scan(query_sids):
        con.execute("DROP TABLE IF EXISTS q;")
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
        rf = con.execute(f"SELECT sid, metric, MAX(sim) m FROM ({fwd}) GROUP BY sid, metric").fetchdf()
        rr = con.execute(f"SELECT sid, metric, MAX(sim) m FROM ({rev}) GROUP BY sid, metric").fetchdf()
        return pd.concat([rf, rr]).groupby(['sid', 'metric'], as_index=False)['m'].max()

    results = {}

    # ---------- data_v2_plinder ----------
    print('=== data_v2_plinder ===', flush=True)
    pl = pd.read_csv('/home/mhjang/workspace/flex/data_v2_plinder/splits.csv')
    pl_train = sorted(set(pl[pl['split'] == 'train']['system_id']))
    sym = score_scan(pl_train)
    block = {}
    for m in METRICS:
        sub = sym[sym.metric == m]; mp = dict(zip(sub['sid'], sub['m']))
        v = np.array([mp.get(s, 0.0) for s in pl_train])
        block[m] = {'max': float(v.max()), 'frac_ge_50': float((v >= THR).mean()),
                    'n_ge_50': int((v >= THR).sum()),
                    'count_above': {f'>={t}': int((v >= t).sum()) for t in [50, 70, 90, 95]}}
    # ligand
    prop = annot[(annot.system_id.isin(set(pl_train))) & (annot.ligand_is_proper == True)]
    pl_smiles_map = prop.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))
    pl_uniq = set().union(*pl_smiles_map.values) if len(pl_smiles_map) else set()
    exact_hits = sum(1 for s in pl_train if pl_smiles_map.get(s, set()) & pb_smiles)
    tani = max_tanimoto(pl_uniq, pb_fps)
    block['ligand'] = {'unit': 'system', 'n': len(pl_train),
                       'exact_smiles_overlap_n': exact_hits, 'exact_smiles_overlap_frac': exact_hits / len(pl_train),
                       'n_unique_smiles': len(pl_uniq),
                       'tanimoto_unique_ge_0.9': int((tani >= 0.9).sum()),
                       'tanimoto_unique_ge_0.7': int((tani >= 0.7).sum()),
                       'tanimoto_unique_ge_0.5': int((tani >= 0.5).sum()),
                       'tanimoto_mean_max': float(tani.mean())}
    block['n_units'] = len(pl_train)
    results['data_v2_plinder'] = block

    # ---------- data_pdbbind ----------
    print('=== data_pdbbind ===', flush=True)
    pb_df = pd.read_csv('/home/mhjang/workspace/flex/data_pdbbind/splits.csv')
    pdb_train = sorted(set(pb_df[pb_df['split'] == 'train']['system_id'].astype(str).str.lower()))
    # map pdb -> plinder sids
    pdb_to_sids = defaultdict(list)
    sub = annot[annot['pdb'].isin(set(pdb_train))]
    for sid, p in zip(sub['system_id'], sub['pdb']):
        pdb_to_sids[p].append(sid)
    query_sids = sorted({s for v in pdb_to_sids.values() for s in v})
    sid_to_pdb = {s: p for p, v in pdb_to_sids.items() for s in v}
    print(f'pdbbind train pdbs={len(pdb_train)} mapped pdbs={len(pdb_to_sids)} query sids={len(query_sids)}', flush=True)
    sym = score_scan(query_sids)
    sym['pdb'] = sym['sid'].map(sid_to_pdb)
    block = {}
    for m in METRICS:
        sub2 = sym[sym.metric == m]
        per_pdb = sub2.groupby('pdb')['m'].max().to_dict()
        v = np.array([per_pdb.get(p, 0.0) for p in pdb_train])
        block[m] = {'max': float(v.max()), 'frac_ge_50': float((v >= THR).mean()),
                    'n_ge_50': int((v >= THR).sum()),
                    'count_above': {f'>={t}': int((v >= t).sum()) for t in [50, 70, 90, 95]}}
    # ligand from pdbbind parsed ligands.pkl
    PB_DIR = Path('/home/mhjang/workspace/flex/data_pdbbind')
    def lig_smiles(pid):
        fpath = PB_DIR / pid / 'ligands.pkl'
        try:
            return {x['smiles'] for x in load(fpath) if x.get('smiles')} if fpath.exists() else set()
        except Exception:
            return set()
    from joblib import Parallel, delayed
    smi_list = Parallel(n_jobs=-1)(delayed(lig_smiles)(p) for p in pdb_train)
    pdb_smiles_map = dict(zip(pdb_train, smi_list))
    pb_uniq = set().union(*smi_list) if smi_list else set()
    exact_hits = sum(1 for p in pdb_train if pdb_smiles_map[p] & pb_smiles)
    tani = max_tanimoto(pb_uniq, pb_fps)
    block['ligand'] = {'unit': 'pdb_code', 'n': len(pdb_train),
                       'exact_smiles_overlap_n': exact_hits, 'exact_smiles_overlap_frac': exact_hits / len(pdb_train),
                       'n_unique_smiles': len(pb_uniq),
                       'tanimoto_unique_ge_0.9': int((tani >= 0.9).sum()),
                       'tanimoto_unique_ge_0.7': int((tani >= 0.7).sum()),
                       'tanimoto_unique_ge_0.5': int((tani >= 0.5).sum()),
                       'tanimoto_mean_max': float(tani.mean())}
    block['n_units'] = len(pdb_train)
    results['data_pdbbind'] = block

    results['pb_exact_systems'] = len(pb_systems)
    results['note'] = ('PB = exact (pdb,ccd) plinder systems (540). pdbbind mapped PDB->plinder '
                       'at PDB level (coarse, overestimates). plinder unit=system, pdbbind unit=PDB.')
    OUT.write_text(json.dumps(results, indent=2))

    # print comparison
    print('\n=== LEAKAGE COMPARISON vs PoseBusters (540 exact) ===')
    print(f"{'metric':<20}{'plinder frac>=50 / max':<30}{'pdbbind frac>=50 / max':<30}")
    for m in METRICS:
        a = results['data_v2_plinder'][m]; b = results['data_pdbbind'][m]
        pa = '{:.3f} / {:.0f}'.format(a['frac_ge_50'], a['max'])
        pb_ = '{:.3f} / {:.0f}'.format(b['frac_ge_50'], b['max'])
        print(f"{m:<20}{pa:<30}{pb_:<30}")
    for k in ['data_v2_plinder', 'data_pdbbind']:
        lg = results[k]['ligand']
        print(f"{k} ligand (unit={lg['unit']}, n={lg['n']}): exact_overlap={lg['exact_smiles_overlap_n']} "
              f"({lg['exact_smiles_overlap_frac']*100:.2f}%)  tanimoto>=0.9={lg['tanimoto_unique_ge_0.9']} "
              f">=0.7={lg['tanimoto_unique_ge_0.7']} (of {lg['n_unique_smiles']} uniq)")
    print('Written:', OUT)


if __name__ == '__main__':
    main()
