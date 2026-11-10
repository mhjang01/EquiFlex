"""Receptor sequence similarity between the training set and PoseBusters, by exact
Needleman-Wunsch (Biopython PairwiseAligner, global, BLOSUM62, gap open -10 / extend -0.5).

Every chain of every PoseBusters system is aligned against every unique chain sequence of
the training split; identity = identical aligned residues / alignment length (gaps count),
also reported over the shorter sequence. Per PB system the maximum over its chains and all
training chains is kept, with the best-matching training system.

    python scripts/seqsim_train_vs_pb.py --train-root ~/workspace/equiflex/data_v2_plinder/train \
        --split-csv ~/workspace/equiflex/data_v2_plinder/splits_max500.csv --out data/posebusters/seqsim_nw.tsv
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


def best_match(query, targets, target_ids):
    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner(); al.mode = 'global'
    al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score = -10.0, -0.5
    best = (0.0, 0.0, -1)
    for i, t in enumerate(targets):
        a = al.align(query, t)[0]
        # identical aligned residues from the alignment coordinates
        ident = 0
        for (qs, qe), (ts, te) in zip(*a.aligned):
            ident += sum(1 for x, y in zip(query[qs:qe], t[ts:te]) if x == y)
        id_aln = ident / a.length                      # over alignment length incl. gaps
        id_min = ident / min(len(query), len(t))       # over the shorter sequence
        if id_aln > best[0]:
            best = (id_aln, id_min, i)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-root', type=Path, required=True)
    ap.add_argument('--split-csv', type=Path, required=True)
    ap.add_argument('--pb-root', type=Path, default=Path('data/posebusters/featurized'))
    ap.add_argument('--out', type=Path, default=Path('data/posebusters/seqsim_nw.tsv'))
    ap.add_argument('--jobs', type=int, default=22)
    args = ap.parse_args()

    split = pd.read_csv(args.split_csv); train_ids = split[split.split == 'train'].system_id.tolist()
    print(f'training systems: {len(train_ids)}', flush=True)
    seqs = Parallel(n_jobs=args.jobs, batch_size=256)(delayed(load_seqs)(args.train_root / s / 'sequence_data.pkl') for s in train_ids if (args.train_root / s / 'sequence_data.pkl').exists())
    uniq = {}
    for sid, d in zip(train_ids, seqs):
        for ch, s in d.items():
            uniq.setdefault(s, []).append(f'{sid}:{ch}')
    targets = list(uniq.keys()); target_ids = [v[0] for v in uniq.values()]
    print(f'unique training chain sequences: {len(targets)} (median length {int(np.median([len(t) for t in targets]))})', flush=True)

    pb = {sid: load_seqs(args.pb_root / sid / 'sequence_data.pkl') for sid in sorted(os.listdir(args.pb_root))}
    queries = [(sid, ch, s) for sid, d in pb.items() for ch, s in d.items()]
    print(f'PoseBusters chains: {len(queries)} over {len(pb)} systems -> {len(queries) * len(targets) / 1e6:.1f}M alignments', flush=True)
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(best_match)(s, targets, target_ids) for _, _, s in queries)

    rows = [dict(system=sid, chain=ch, length=len(s), identity_aln=r[0], identity_min=r[1],
                 best_train=target_ids[r[2]] if r[2] >= 0 else '', n_train_with_seq=len(uniq[targets[r[2]]]) if r[2] >= 0 else 0)
            for (sid, ch, s), r in zip(queries, res)]
    df = pd.DataFrame(rows); df.to_csv(args.out.with_suffix('.chains.tsv'), sep='\t', index=False)
    per = df.sort_values('identity_aln', ascending=False).groupby('system').first().reset_index()
    per.to_csv(args.out, sep='\t', index=False)
    x = per.identity_aln.values * 100
    print(f'\nPer PB system, max NW identity to any training chain (over alignment length), n={len(per)}')
    print('  median %.1f%%, p25 %.1f%%, p75 %.1f%%' % (np.median(x), np.percentile(x, 25), np.percentile(x, 75)))
    for thr in (30, 40, 50, 70, 90, 95):
        print(f'  >= {thr}%: {100 * (x >= thr).mean():5.1f}% of systems ({(x >= thr).sum()})')
    y = per.identity_min.values * 100
    print(f'Same, identity over the shorter sequence: median {np.median(y):.1f}%, >=90%: {100 * (y >= 90).mean():.1f}%')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
