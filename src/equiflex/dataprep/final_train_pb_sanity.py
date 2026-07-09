'''
Final PoseBusters leakage sanity check on the prepared train set (on-disk dirs
in data_v2_plinder/train) vs the 548-system PB set.

  pli_qcov / pocket_lddt: symmetric MAX vs PB -> fraction>=50, max, percentiles
  exact ligand SMILES overlap (annot proper-ligand canonical)
  exact protein sequence overlap (parsed sequence_data.pkl <-> PB fasta)

Output: data_v2_plinder/logs/final_train_pb_sanity.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

ROOT = Path('/home/mhjang/workspace/flex/data_v2_plinder')
TR = ROOT / 'train'
LOG = ROOT / 'logs'
SYSTEMS = Path('/mnt/drive/plinder/2024-06/v2/systems')
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
METRICS = ['pli_qcov', 'pocket_lddt']
THR = 50.0


def fasta_seqs(fp):
    try:
        text = fp.read_text()
    except OSError:
        return set()
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith('>'):
            if cur: out.append(''.join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur: out.append(''.join(cur))
    return set(s for s in out if s)


def main():
    train = sorted(d.name for d in TR.iterdir() if d.is_dir())
    print(f'train (on-disk): {len(train)}', flush=True)

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
    pb_systems = sorted(set(exact['system_id']) | set(fb['system_id']))
    print(f'PB systems: {len(pb_systems)}', flush=True)

    # ---- metrics symmetric ----
    con = duckdb.connect()
    con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute("CREATE TEMP TABLE train_sys (system_id VARCHAR);")
    con.executemany("INSERT INTO train_sys VALUES (?)", [(s,) for s in train])
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
    print('fwd...', flush=True); rf = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({fwd}) GROUP BY tsid, metric").fetchdf()
    print('rev...', flush=True); rr = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({rev}) GROUP BY tsid, metric").fetchdf()
    sym = pd.concat([rf, rr]).groupby(['tsid', 'metric'], as_index=False)['m'].max()

    summary = {'n_train': len(train), 'n_pb_systems': len(pb_systems), 'metrics': {}}
    for m in METRICS:
        sub = sym[sym.metric == m]
        mp = dict(zip(sub['tsid'], sub['m']))
        vals = np.array([mp.get(s, 0.0) for s in train])
        summary['metrics'][m] = {
            'max': float(vals.max()),
            f'n_ge_{THR:g}': int((vals >= THR).sum()),
            f'fraction_ge_{THR:g}': float((vals >= THR).mean()),
            'mean': float(vals.mean()),
            'count_above': {f'>={t}': int((vals >= t).sum()) for t in [50, 70, 85, 90, 95]},
        }

    # ---- exact SMILES ----
    prop = annot[(annot['system_id'].isin(set(train))) & (annot['ligand_is_proper'] == True)]
    sys_smiles = prop.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))
    pb_smiles = set(a['ligand_rdkit_canonical_smiles'].dropna())
    smiles_hits = [s for s in train if sys_smiles.get(s, set()) & pb_smiles]

    # ---- exact sequence ----
    pb_seq = set().union(*Parallel(n_jobs=16)(
        delayed(lambda x: fasta_seqs(SYSTEMS / x / 'sequences.fasta'))(s) for s in pb_systems))
    def tseq(s):
        fp = TR / s / 'sequence_data.pkl'
        try:
            return s, set(v for v in load(fp).values() if v) if fp.exists() else set()
        except Exception:
            return s, set()
    tmap = dict(Parallel(n_jobs=16)(delayed(tseq)(s) for s in train))
    seq_hits = [s for s in train if tmap[s] & pb_seq]
    both = sorted(set(smiles_hits) & set(seq_hits))

    summary['exact_match'] = {
        'pb_unique_smiles': len(pb_smiles), 'pb_unique_sequences': len(pb_seq),
        'smiles_overlap': {'n': len(smiles_hits), 'frac': len(smiles_hits) / len(train)},
        'sequence_overlap': {'n': len(seq_hits), 'frac': len(seq_hits) / len(train)},
        'both': {'n': len(both), 'frac': len(both) / len(train)},
    }

    (LOG / 'final_train_pb_sanity.json').write_text(json.dumps(summary, indent=2))
    print('\n=== FINAL TRAIN vs PoseBusters ===')
    for m in METRICS:
        s = summary['metrics'][m]
        print(f"  {m}: max={s['max']:.1f}  frac>=50={s['fraction_ge_50']*100:.2f}%  "
              f">=85={s['count_above']['>=85']}  >=90={s['count_above']['>=90']}  >=95={s['count_above']['>=95']}")
    em = summary['exact_match']
    print(f"  SMILES overlap: {em['smiles_overlap']['n']} ({em['smiles_overlap']['frac']*100:.2f}%)")
    print(f"  sequence overlap: {em['sequence_overlap']['n']} ({em['sequence_overlap']['frac']*100:.2f}%)")
    print(f"  both: {em['both']['n']} ({em['both']['frac']*100:.3f}%)")


if __name__ == '__main__':
    main()
