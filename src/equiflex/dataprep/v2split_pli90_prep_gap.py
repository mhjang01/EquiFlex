'''
Target = v2-split strict pipeline with the PB pli_qcov COMPONENT removal replaced
by PAIRWISE pli_qcov<90 vs PoseBusters. Annotation-level (post-parse filters
atom<450/missing10/metal<=2 applied later, after parsing).

Stages (reconstructed from annotation_table + split.parquet):
  s1 one proper ligand; co-binders ions only
  s2 proper ligand heavy atoms < 70
  s3 proper pocket residues > 3
  s4 no missing ligand atoms; atom count == smiles
  s5 remove PB overlap: exact PB system OR (proper smiles in PB AND shared seqsim100 component)
  PAIRWISE: remove pli_qcov_max(symmetric) >= 90 vs 548 PB systems
  s_dedup: dedup (pdb, canonical_smiles, pocket_num_residues)

Then prep gap vs data_v2_plinder/train (required: ligand_features/pocket_features/
esm3_embeddings/unimol_embeddings .pt).

Writes: data_v2_plinder/logs/v2split_pli90_{target,to_prepare}.txt + _prep_gap.json
'''
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TRAIN_DIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
SPLIT_PQ = '/mnt/drive/plinder/2024-06/v2/splits/split.parquet'
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
SCORE_DIRS = [
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=holo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=apo/*.parquet',
    '/mnt/drive/plinder/2024-06/v2/scores/search_db=pred/*.parquet',
]
SEQ100 = ['protein_seqsim_max__100__community',
          'protein_seqsim_max__100__weak__component',
          'protein_seqsim_max__100__strong__component']
REQUIRED = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']
THR = 90


def main():
    sp = pq.read_table(SPLIT_PQ, columns=['system_id', 'split', 'system_proper_pocket_num_residues']).to_pandas()
    train_all = set(sp[sp['split'] == 'train']['system_id'])
    pocket_res = dict(zip(sp['system_id'], sp['system_proper_pocket_num_residues']))

    cols = ['system_id', 'entry_pdb_id', 'ligand_ccd_code', 'ligand_is_proper', 'ligand_is_ion',
            'ligand_num_heavy_atoms', 'ligand_num_unresolved_heavy_atoms',
            'ligand_matches_smiles_atom_num', 'ligand_rdkit_canonical_smiles'] + SEQ100
    a = pq.read_table(ANNOT_PQ, columns=cols).to_pandas()
    a['pdb'] = a['entry_pdb_id'].str.lower()
    a['ccd'] = a['ligand_ccd_code'].astype(str).str.upper()

    # PB set + PB smiles + PB seq100 components
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    ap = a[a['pdb'].isin(pb_pdbs)]
    exact = ap[[(p, c) in pairs for p, c in zip(ap['pdb'], ap['ccd'])]]
    fb = ap[ap['pdb'].isin(pb_pdbs - set(exact['pdb']))]
    pb_systems = set(exact['system_id']) | set(fb['system_id'])
    pb_smiles = set(ap['ligand_rdkit_canonical_smiles'].dropna())
    pb_seq100 = set()
    for c in SEQ100:
        pb_seq100 |= set(ap[c].dropna())

    tr = a[a['system_id'].isin(train_all)].copy()

    # s1 one proper ligand; co-binders ions only
    g = tr.groupby('system_id')
    n_proper = g['ligand_is_proper'].sum()
    n_bad = g.apply(lambda d: int(((~d['ligand_is_proper'].astype(bool)) & (~d['ligand_is_ion'].astype(bool))).sum()))
    keep = set(n_proper[n_proper == 1].index) & set(n_bad[n_bad == 0].index)
    prop = tr[tr['ligand_is_proper'] == True].drop_duplicates('system_id').set_index('system_id')

    # s2-s4 on proper row
    p = prop.loc[sorted(keep & set(prop.index))]
    p = p[p['ligand_num_heavy_atoms'] < 70]
    p = p[[pocket_res.get(s, 0) > 3 for s in p.index]]
    p = p[(p['ligand_num_unresolved_heavy_atoms'] == 0) & (p['ligand_matches_smiles_atom_num'] == True)]
    print(f's1-4 survivors: {len(p)}')

    # s5 remove PB overlap
    def keep_s5(sid, row):
        if sid in pb_systems:
            return False
        smi = row['ligand_rdkit_canonical_smiles']
        if smi in pb_smiles and any(pd.notna(row.get(c)) and row.get(c) in pb_seq100 for c in SEQ100):
            return False
        return True
    s5 = [sid for sid, row in p.iterrows() if keep_s5(sid, row)]
    print(f's5 (PB overlap removed): {len(s5)}')

    # PAIRWISE pli_qcov<90 vs PB over s5 candidates
    con = duckdb.connect()
    con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_spill'")
    con.execute("CREATE TEMP TABLE q (system_id VARCHAR);")
    con.executemany("INSERT INTO q VALUES (?)", [(s,) for s in s5])
    con.execute(f"CREATE TEMP TABLE pb_sys AS SELECT system_id FROM (VALUES "
                f"{','.join(f'(\'{s}\')' for s in pb_systems)}) AS t(system_id);")
    fwd = ' UNION ALL '.join(
        f"SELECT query_system AS tsid, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov' AND query_system IN (SELECT system_id FROM q) "
        f"AND target_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    rev = ' UNION ALL '.join(
        f"SELECT target_system AS tsid, similarity AS sim FROM read_parquet('{p}', union_by_name=True) "
        f"WHERE metric='pli_qcov' AND target_system IN (SELECT system_id FROM q) "
        f"AND query_system IN (SELECT system_id FROM pb_sys)" for p in SCORE_DIRS)
    print('fwd...'); rf = con.execute(f"SELECT tsid, MAX(sim) m FROM ({fwd}) GROUP BY tsid").fetchdf()
    print('rev...'); rr = con.execute(f"SELECT tsid, MAX(sim) m FROM ({rev}) GROUP BY tsid").fetchdf()
    sym = pd.concat([rf, rr]).groupby('tsid', as_index=False)['m'].max()
    mx = dict(zip(sym['tsid'], sym['m']))
    after_pli = [s for s in s5 if mx.get(s, 0.0) < THR]
    print(f'after pairwise pli_qcov<{THR}: {len(after_pli)}')

    # dedup (pdb, canonical_smiles, pocket_res)
    pp = p.loc[after_pli].reset_index()
    pp['pres'] = pp['system_id'].map(pocket_res)
    dedup = pp.drop_duplicates(subset=['pdb', 'ligand_rdkit_canonical_smiles', 'pres'])
    target = sorted(dedup['system_id'])
    print(f'target after dedup: {len(target)}')

    # prep gap
    complete, incomplete, absent = [], [], []
    for s in target:
        d = TRAIN_DIR / s
        if not d.is_dir():
            absent.append(s)
        elif all((d / f).exists() for f in REQUIRED):
            complete.append(s)
        else:
            incomplete.append(s)
    to_prepare = sorted(absent + incomplete)

    (OUTDIR / 'v2split_pli90_target.txt').write_text('\n'.join(target) + '\n')
    (OUTDIR / 'v2split_pli90_to_prepare.txt').write_text('\n'.join(to_prepare) + '\n')
    summary = {
        'note': 'annotation-level target; post-parse filters (atom<450/missing10/metal<=2) applied AFTER parsing',
        'v2_train_base': len(train_all),
        's1_4_structural': len(p), 's5_pb_overlap_removed': len(s5),
        'after_pairwise_pli_lt90': len(after_pli), 'target_after_dedup': len(target),
        'already_complete': len(complete),
        'incomplete_needs_refeaturize': len(incomplete),
        'absent_needs_full_parse': len(absent),
        'total_to_prepare': len(to_prepare),
    }
    (OUTDIR / 'v2split_pli90_prep_gap.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
