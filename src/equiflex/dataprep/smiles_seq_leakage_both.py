'''
Exact SMILES / protein-sequence leakage vs PoseBusters for BOTH splits:
  - v2 split  : data_v2_plinder/splits.csv (train)         seqs <- parsed sequence_data.pkl
  - PL50 split: v0/splits/plinder-pl50.parquet (train)     seqs <- /mnt/drive systems fasta

PB target = corrected 548-system set (exact (pdb,ccd) + pdb-level fallback,
full v2 index; 3 PDBs unrecoverable).

Exact match = train system shares >=1 identical ligand_rdkit_canonical_smiles
(resp. protein chain sequence) with any PB system.

SMILES come from the full annotation index (complete).
Sequence note: PL50 train reads per-system fasta off the local /mnt/drive subset;
~10% are absent locally -> PL50 sequence count is a LOWER BOUND.

Output: data_v2_plinder/logs/smiles_seq_leakage_both.json
'''
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
V2_SPLITS = '/home/mhjang/workspace/flex/data_v2_plinder/splits.csv'
V2_TRAIN_DIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
PL50_PQ = '/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet'
SYSTEMS = Path('/mnt/drive/plinder/2024-06/v2/systems')
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'


def fasta_seqs(fp):
    try:
        text = fp.read_text()
    except OSError:
        return set()
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith('>'):
            if cur:
                seqs.append(''.join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append(''.join(cur))
    return set(s for s in seqs if s)


def pb_seqs_for(sid):
    return fasta_seqs(SYSTEMS / sid / 'sequences.fasta')


def make_pb(annot):
    pb = [l.strip() for l in open(PB_FILE) if l.strip()]
    pairs = set((p.split('_')[0].lower(), p.split('_', 1)[1].upper()) for p in pb)
    pb_pdbs = set(p for p, _ in pairs)
    a = annot[annot['pdb'].isin(pb_pdbs)]
    exact = a[[(p, c) in pairs for p, c in zip(a['pdb'], a['ccd'])]]
    exact_sys = set(exact['system_id'])
    pdbs_exact = set(exact['pdb'])
    fb = a[a['pdb'].isin(pb_pdbs - pdbs_exact)]
    pb_systems = exact_sys | set(fb['system_id'])
    return pb_systems


# ---- shared annotation load ----
annot = pq.read_table(
    ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_ccd_code', 'ligand_rdkit_canonical_smiles']
).to_pandas()
annot['pdb'] = annot['entry_pdb_id'].str.lower()
annot['ccd'] = annot['ligand_ccd_code'].astype(str).str.upper()
sys_smiles = annot.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
    lambda s: set(x for x in s if isinstance(x, str) and x))

pb_systems = make_pb(annot)
pb_smiles = set()
for s in pb_systems:
    pb_smiles |= sys_smiles.get(s, set())
print(f'PB systems: {len(pb_systems)}  PB unique SMILES: {len(pb_smiles)}')
pb_seq_sets = Parallel(n_jobs=16)(delayed(pb_seqs_for)(s) for s in pb_systems)
pb_seq_all = set().union(*pb_seq_sets) if pb_seq_sets else set()
print(f'PB unique sequences: {len(pb_seq_all)}')


def smiles_hits(train):
    return [s for s in train if (sys_smiles.get(s, set()) & pb_smiles)]


def v2_seq_hits(train):
    def one(s):
        fp = V2_TRAIN_DIR / s / 'sequence_data.pkl'
        try:
            if not fp.exists():
                return None
            d = load(fp)
            ss = set(v for v in d.values() if v)
        except Exception:
            return None
        return s if (ss & pb_seq_all) else None
    r = Parallel(n_jobs=16)(delayed(one)(s) for s in train)
    return [x for x in r if x]


def pl50_seq_hits(train):
    def one(s):
        ss = pb_seqs_for(s)  # same fasta location for any plinder system
        return s if (ss & pb_seq_all) else None
    r = Parallel(n_jobs=16)(delayed(one)(s) for s in train)
    n_missing = sum(1 for s in train if not (SYSTEMS / s / 'sequences.fasta').exists())
    return [x for x in r if x], n_missing


def main():
    out = {'pb': {'n_systems': len(pb_systems), 'n_smiles': len(pb_smiles),
                  'n_sequences': len(pb_seq_all)}, 'splits': {}}

    # ---- v2 split ----
    v2 = pd.read_csv(V2_SPLITS)
    v2_train = sorted(set(v2[v2['split'] == 'train']['system_id']))
    sh = smiles_hits(v2_train)
    print(f'[v2] smiles hits: {len(sh)}')
    qh = v2_seq_hits(v2_train)
    print(f'[v2] seq hits: {len(qh)}')
    both = sorted(set(sh) & set(qh))
    out['splits']['v2_plinder'] = {
        'n_train': len(v2_train),
        'smiles_exact': {'n': len(sh), 'fraction': len(sh) / len(v2_train)},
        'sequence_exact': {'n': len(qh), 'fraction': len(qh) / len(v2_train)},
        'both': {'n': len(both), 'fraction': len(both) / len(v2_train)},
        'seq_source': 'parsed sequence_data.pkl (complete)',
    }

    # ---- PL50 split ----
    pl = pq.read_table(PL50_PQ, columns=['system_id', 'split']).to_pandas()
    pl_train = sorted(set(pl[pl['split'] == 'train']['system_id']))
    sh2 = smiles_hits(pl_train)
    print(f'[pl50] smiles hits: {len(sh2)}')
    qh2, n_missing = pl50_seq_hits(pl_train)
    print(f'[pl50] seq hits: {len(qh2)}  (fasta missing locally: {n_missing})')
    both2 = sorted(set(sh2) & set(qh2))
    out['splits']['pl50'] = {
        'n_train': len(pl_train),
        'smiles_exact': {'n': len(sh2), 'fraction': len(sh2) / len(pl_train)},
        'sequence_exact': {'n': len(qh2), 'fraction': len(qh2) / len(pl_train),
                           'fasta_missing_locally': n_missing,
                           'note': 'LOWER BOUND: ~missing fasta not counted'},
        'both': {'n': len(both2), 'fraction': len(both2) / len(pl_train)},
        'seq_source': '/mnt/drive systems fasta (partial local subset)',
    }

    fp = OUTDIR / 'smiles_seq_leakage_both.json'
    fp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f'\nWritten: {fp}')


if __name__ == '__main__':
    main()
