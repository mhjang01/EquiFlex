'''
Finalize PB leakage audit: metric stats from saved per-train CSVs + exact
SMILES/sequence match (guarded file reads). Avoids re-scanning score parquets.
'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, load

DATA = Path('/home/mhjang/workspace/flex/data_v2_plinder')
SYSTEMS = Path('/mnt/drive/plinder/2024-06/v2/systems')
ANNOT_PQ = '/home/mhjang/.local/share/plinder/2024-06/v2/index/annotation_table.parquet'
PB_FILE = '/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt'
METRICS = ['pli_qcov', 'pocket_lddt']
THRESH = 50.0


def read_fasta_seqs(fp):
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


def train_seqs(sid):
    fp = DATA / 'train' / sid / 'sequence_data.pkl'
    try:
        if not fp.exists():
            return sid, set()
        d = load(fp)
        return sid, set(v for v in d.values() if v)
    except OSError:
        return sid, set()
    except Exception:
        return sid, set()


def pb_seqs(sid):
    fp = SYSTEMS / sid / 'sequences.fasta'
    try:
        if not fp.exists():
            return set()
    except OSError:
        return set()
    return read_fasta_seqs(fp)


def main():
    splits = pd.read_csv(DATA / 'splits.csv')
    train = sorted(set(splits[splits['split'] == 'train']['system_id']))

    pb_pdbs = {l.strip().split('_')[0].lower() for l in open(PB_FILE) if l.strip()}
    annot = pq.read_table(
        ANNOT_PQ, columns=['system_id', 'entry_pdb_id', 'ligand_rdkit_canonical_smiles']
    ).to_pandas()
    annot['pdb_lower'] = annot['entry_pdb_id'].str.lower()
    sys_smiles = annot.groupby('system_id')['ligand_rdkit_canonical_smiles'].apply(
        lambda s: set(x for x in s if isinstance(x, str) and x))
    annot_sys = annot.drop_duplicates('system_id')
    pb_systems = sorted(set(annot_sys[annot_sys['pdb_lower'].isin(pb_pdbs)]['system_id']))

    summary = {
        'n_train': len(train),
        'n_pb_pdb_codes': len(pb_pdbs),
        'n_pb_systems': len(pb_systems),
        'threshold': THRESH,
        'metrics': {},
    }
    # ---- metrics from saved CSVs ----
    for metric in METRICS:
        df = pd.read_csv(DATA / 'logs' / f'pb_{metric}_max_per_train.csv')
        vals = df[f'max_{metric}_vs_PB'].fillna(0.0).values.astype(float)
        summary['metrics'][metric] = {
            'n_train_scored': int((vals > 0).sum()),
            'max': float(vals.max()),
            f'n_ge_{THRESH:g}': int((vals >= THRESH).sum()),
            f'fraction_ge_{THRESH:g}': float((vals >= THRESH).mean()),
            'mean': float(vals.mean()),
            'percentiles': {f'p{p}': float(np.percentile(vals, p)) for p in [50, 75, 90, 95, 99]},
            'count_above': {f'>={t}': int((vals >= t).sum()) for t in [50, 70, 85, 95, 99]},
        }

    # ---- exact match ----
    pb_smiles = set()
    for s in pb_systems:
        pb_smiles |= sys_smiles.get(s, set())
    train_smiles_map = {s: sys_smiles.get(s, set()) for s in train}
    smiles_hits = [s for s in train if train_smiles_map[s] & pb_smiles]

    pb_seq_sets = Parallel(n_jobs=-1)(delayed(pb_seqs)(s) for s in pb_systems)
    pb_seq_all = set().union(*pb_seq_sets) if pb_seq_sets else set()

    tr = Parallel(n_jobs=-1)(delayed(train_seqs)(s) for s in train)
    train_seq_map = {sid: ss for sid, ss in tr}
    seq_hits = [s for s in train if train_seq_map[s] & pb_seq_all]
    both_hits = [s for s in train if (train_smiles_map[s] & pb_smiles)
                 and (train_seq_map[s] & pb_seq_all)]

    summary['exact_match'] = {
        'n_pb_unique_smiles': len(pb_smiles),
        'n_pb_unique_sequences': len(pb_seq_all),
        'smiles_exact_match': {'n_train': len(smiles_hits),
                               'fraction': len(smiles_hits) / len(train),
                               'examples': smiles_hits[:15]},
        'sequence_exact_match': {'n_train': len(seq_hits),
                                 'fraction': len(seq_hits) / len(train),
                                 'examples': seq_hits[:15]},
        'both_smiles_and_sequence': {'n_train': len(both_hits),
                                     'fraction': len(both_hits) / len(train),
                                     'examples': both_hits[:15]},
    }

    out = DATA / 'logs' / 'posebusters_leakage_audit.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
