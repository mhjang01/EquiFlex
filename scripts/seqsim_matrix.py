"""Full Needleman-Wunsch identity matrix: unique training chain sequences x PoseBusters chains.
Saves data/posebusters/seqsim_matrix.npz (identity over alignment length, float16) with the row/column
ids, so training-side filtering at any threshold and PB-side maxima are both one lookup.
    python scripts/seqsim_matrix.py --train-root ~/workspace/equiflex/data_v2_plinder/train --split-csv .../splits_max500.csv
    NOTE: the local training tree was removed on 2026-11-07; train/ now lives only on the
    clusters (dream: /scratch/jyl_snu/mhjang/workspace/equiflex/data_v2_plinder/train).
    rsync it back beside the splits_*.csv manifests before re-running. The computed
    outputs of this script are already on disk under data/.
"""
import argparse, os, pickle
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed

def load_seqs(p):
    try:
        import joblib; d = joblib.load(p)
    except Exception:
        d = pickle.load(open(p, 'rb'))
    return {k: v for k, v in d.items() if isinstance(v, str) and len(v) >= 20}

def row(query, targets):
    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner(); al.mode = 'global'; al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score = -10.0, -0.5
    out = np.zeros(len(targets), dtype=np.float16)
    for i, t in enumerate(targets):
        a = al.align(query, t)[0]; ident = 0
        for (qs, qe), (ts, te) in zip(*a.aligned):
            ident += sum(1 for x, y in zip(query[qs:qe], t[ts:te]) if x == y)
        out[i] = ident / a.length
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--train-root', type=Path, required=True); ap.add_argument('--split-csv', type=Path, default=None, help='restrict to the train split of this CSV; omit to use every system dir under --train-root')
    ap.add_argument('--pb-root', type=Path, default=Path('data/posebusters/featurized')); ap.add_argument('--out', type=Path, default=Path('data/posebusters/seqsim_matrix.npz')); ap.add_argument('--jobs', type=int, default=22)
    a = ap.parse_args()
    if a.split_csv is not None:
        split = pd.read_csv(a.split_csv); ids = split[split.split == 'train'].system_id.tolist()
    else:
        ids = sorted(os.listdir(a.train_root))
    train_ids = [s for s in ids if (a.train_root / s / 'sequence_data.pkl').exists()]
    seqs = Parallel(n_jobs=a.jobs, batch_size=256)(delayed(load_seqs)(a.train_root / s / 'sequence_data.pkl') for s in train_ids)
    uniq = {}
    for sid, d in zip(train_ids, seqs):
        for ch, s in d.items(): uniq.setdefault(s, []).append(f'{sid}:{ch}')
    train_seqs = list(uniq.keys()); members = [';'.join(v) for v in uniq.values()]
    pb = [(sid, ch, s) for sid in sorted(os.listdir(a.pb_root)) for ch, s in load_seqs(a.pb_root / sid / 'sequence_data.pkl').items()]
    pb_seqs = [s for _, _, s in pb]
    print(f'{len(train_seqs)} unique training chains x {len(pb_seqs)} PB chains', flush=True)
    rows = Parallel(n_jobs=a.jobs, verbose=5, batch_size=16)(delayed(row)(q, pb_seqs) for q in train_seqs)
    M = np.stack(rows)                                   # [n_train_uniq, n_pb_chains]
    np.savez_compressed(a.out, identity=M, train_members=np.array(members, dtype=object), pb_system=np.array([p[0] for p in pb]), pb_chain=np.array([p[1] for p in pb]), train_len=np.array([len(s) for s in train_seqs]), pb_len=np.array([len(s) for s in pb_seqs]))
    print('saved', a.out, M.shape, flush=True)

if __name__ == '__main__':
    main()
