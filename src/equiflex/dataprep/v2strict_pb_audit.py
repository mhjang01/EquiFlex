'''
Same PoseBusters leakage metrics, computed on the NEWLY PARSED v2_strict train set
(train_v2_strict_atom_lt450_missing10_metal_le2_survivors.txt, n=59776).

Metrics vs corrected 548-system PB set:
  pli_qcov / pocket_lddt : directed (train->PB) + symmetric MAX, fraction>=50, max
  exact ligand SMILES match
  exact protein sequence match (parsed sequence_data.pkl, complete)

Output: data_v2_plinder/logs/v2strict_pb_audit.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TRAIN_TXT = OUTDIR / 'train_v2_strict_atom_lt450_missing10_metal_le2_survivors.txt'
TRAIN_DIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
SYSTEMS = Path('/mnt/drive/plinder/2024-06/v2/systems')
METRICS = ['pli_qcov', 'pocket_lddt']
THRESH = 50.0


def fasta_seqs(fp):
    try:
        text = fp.read_text()
    except OSError:
        return set()
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith('>'):
            if cur:
                out.append(''.join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        out.append(''.join(cur))
    return set(s for s in out if s)


def main():
    train = sorted({l.strip() for l in open(TRAIN_TXT) if l.strip()})
    print(f'v2_strict train: {len(train)}')

    annot = pq.read_table(
        ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_ccd_code', 'ligand_rdkit_canonical_smiles']
    ).to_pandas()
    annot['pdb'] = annot['entry_pdb_id'].str.lower()
    annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()
    sys_smiles = annot.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))

    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    a = annot[annot['pdb'].isin(pb_pdbs)]
    exact = a[[(p, c) in pairs for p, c in zip(a['pdb'], a['ccd'])]]
    pdbs_exact = set(exact['pdb'])
    fb = a[a['pdb'].isin(pb_pdbs - pdbs_exact)]
    pb_systems = set(exact['system_id']) | set(fb['system_id'])
    print(f'PB systems: {len(pb_systems)}')

    # ---- metrics ----
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
    print('forward scan...')
    rf = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({fwd}) GROUP BY tsid, metric").fetchdf()
    print('reverse scan...')
    rr = con.execute(f"SELECT tsid, metric, MAX(sim) m FROM ({rev}) GROUP BY tsid, metric").fetchdf()
    sym = pd.concat([rf, rr], ignore_index=True).groupby(['tsid', 'metric'], as_index=False)['m'].max()

    def stats(df):
        out = {}
        for m in METRICS:
            sub = df[df['metric'] == m]
            mp = dict(zip(sub['tsid'], sub['m']))
            vals = np.array([mp.get(s, 0.0) for s in train], dtype=float)
            out[m] = {'n_scored': int(len(sub)), 'max': float(vals.max()),
                      f'n_ge_{THRESH:g}': int((vals >= THRESH).sum()),
                      f'fraction_ge_{THRESH:g}': float((vals >= THRESH).mean()),
                      'mean': float(vals.mean()),
                      'count_above': {f'>={t}': int((vals >= t).sum()) for t in [50, 70, 85, 95]}}
        return out

    summary = {'split': 'v2_strict_atom_lt450_missing10_metal_le2',
               'n_train': len(train), 'n_pb_systems': len(pb_systems), 'threshold': THRESH,
               'directed_train_to_pb': stats(rf), 'symmetric': stats(sym)}

    # ---- exact match ----
    pb_smiles = set()
    for s in pb_systems:
        pb_smiles |= sys_smiles.get(s, set())
    pb_seq_all = set().union(*Parallel(n_jobs=16)(
        delayed(lambda x: fasta_seqs(SYSTEMS / x / 'sequences.fasta'))(s) for s in pb_systems))

    smiles_hits = [s for s in train if (sys_smiles.get(s, set()) & pb_smiles)]

    def seqset(s):
        fp = TRAIN_DIR / s / 'sequence_data.pkl'
        try:
            if not fp.exists():
                return s, set()
            return s, set(v for v in load(fp).values() if v)
        except Exception:
            return s, set()
    tmap = dict(Parallel(n_jobs=16)(delayed(seqset)(s) for s in train))
    seq_hits = [s for s in train if tmap[s] & pb_seq_all]
    both = sorted(set(smiles_hits) & set(seq_hits))

    summary['exact_match'] = {
        'n_pb_smiles': len(pb_smiles), 'n_pb_sequences': len(pb_seq_all),
        'smiles_exact': {'n': len(smiles_hits), 'fraction': len(smiles_hits) / len(train)},
        'sequence_exact': {'n': len(seq_hits), 'fraction': len(seq_hits) / len(train)},
        'both': {'n': len(both), 'fraction': len(both) / len(train), 'examples': both[:20]},
    }

    fp = OUTDIR / 'v2strict_pb_audit.json'
    fp.write_text(json.dumps(summary, indent=2))
    print('\n=== v2_strict vs PoseBusters ===')
    for mode in ['directed_train_to_pb', 'symmetric']:
        for m in METRICS:
            s = summary[mode][m]
            print(f"  [{mode}] {m}: frac>=50={s['fraction_ge_50']:.4f} n>=50={s['n_ge_50']} max={s['max']:.0f}")
    em = summary['exact_match']
    print(f"  SMILES exact: {em['smiles_exact']['n']} ({em['smiles_exact']['fraction']*100:.2f}%)")
    print(f"  seq exact:    {em['sequence_exact']['n']} ({em['sequence_exact']['fraction']*100:.2f}%)")
    print(f"  both:         {em['both']['n']} ({em['both']['fraction']*100:.2f}%)")
    print(f'\nWritten: {fp}')


if __name__ == '__main__':
    main()
