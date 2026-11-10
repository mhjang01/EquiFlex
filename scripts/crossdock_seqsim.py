"""Receptor sequence identity between the v10 training split and the CrossDock benchmark.

Same measure as scripts/seqsim_train_vs_pb.py so the CrossDock stratification is
comparable to the PoseBusters one: exact Needleman-Wunsch (Biopython
PairwiseAligner, global, BLOSUM62, gap open -10 / extend -0.5), identity over
alignment length. An MMseqs2 search narrows each query to its best candidate
training chains first, because aligning 511 queries against every training chain
exactly would cost 12M alignments for a maximum that only the top hits can hold.
Queries MMseqs2 returns nothing for are reported with identity 0 and are the
distant ones by construction.

    python scripts/crossdock_seqsim.py --seqs data/crossdock/seqs.tsv \
        --split-csv ~/workspace/equiflex/data_v2_plinder/splits_v9.csv \
        --train-root ~/workspace/equiflex/data_v2_plinder/train --out data/crossdock/seqsim_nw.tsv
    NOTE: the local training tree was removed on 2026-11-07; train/ now lives only on the
    clusters (dream: /scratch/jyl_snu/mhjang/workspace/equiflex/data_v2_plinder/train).
    rsync it back beside the splits_*.csv manifests before re-running. The computed
    outputs of this script are already on disk under data/.
"""
import argparse
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

TOP_K = 25


def load_seqs(p):
    try:
        import joblib
        d = joblib.load(p)
    except Exception:
        d = pickle.load(open(p, 'rb'))
    return {k: v for k, v in d.items() if isinstance(v, str) and len(v) >= 20}


def read_fasta(path):
    """-> {sequence: header} ; headers carry the 'system:chain;...' members."""
    uniq = {}
    for block in path.read_text().split('>')[1:]:
        head, seq = block.split('\n', 1)
        uniq[seq.replace('\n', '')] = head.strip()
    return uniq


def train_fasta(train_root, split_csv, cache, jobs):
    """-> {sequence: 'system:chain;...'} for the unique chains of the train split."""
    if cache.exists():
        uniq = read_fasta(cache)
        print(f'train chains from cache: {len(uniq)}')
        return uniq
    ids = pd.read_csv(split_csv).query('split == "train"').system_id.tolist()
    ids = [s for s in ids if (train_root / s / 'sequence_data.pkl').exists()]
    print(f'reading {len(ids)} training systems')
    seqs = Parallel(n_jobs=jobs, batch_size=256, verbose=1)(
        delayed(load_seqs)(train_root / s / 'sequence_data.pkl') for s in ids)
    uniq = {}
    for sid, d in zip(ids, seqs):
        for ch, s in d.items():
            uniq.setdefault(s, []).append(f'{sid}:{ch}')
    uniq = {s: ';'.join(v) for s, v in uniq.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('w') as fh:
        for i, (s, members) in enumerate(uniq.items()):
            fh.write(f'>t{i}|{members[:200]}\n{s}\n')
    print(f'unique training chains: {len(uniq)} -> {cache}')
    return uniq


def mmseqs_candidates(query_fasta, target_seqs, threads):
    """-> {query_id: [target sequence, ...]} best hits per query."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tgt = td / 'target.fasta'
        order = list(target_seqs)
        with tgt.open('w') as fh:
            for i, s in enumerate(order):
                fh.write(f'>t{i}\n{s}\n')
        hits = td / 'hits.tsv'
        cmd = ['mmseqs', 'easy-search', str(query_fasta), str(tgt), str(hits), str(td / 'tmp'),
               '--threads', str(threads), '-s', '7.5', '--max-seqs', str(TOP_K * 4),
               '--format-output', 'query,target,fident,bits', '-e', '10000']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not hits.exists() or not hits.stat().st_size:
            return {}
        df = pd.read_csv(hits, sep='\t', names=['query', 'target', 'fident', 'bits'])
    df = df.sort_values('bits', ascending=False).groupby('query').head(TOP_K)
    return {q: [order[int(t[1:])] for t in g.target] for q, g in df.groupby('query')}


def best_nw(query, targets):
    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner()
    al.mode = 'global'
    al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score = -10.0, -0.5
    best = (0.0, 0.0, None)
    for t in targets:
        a = al.align(query, t)[0]
        ident = sum(1 for (qs, qe), (ts, te) in zip(*a.aligned)
                    for x, y in zip(query[qs:qe], t[ts:te]) if x == y)
        id_aln = ident / a.length
        if id_aln > best[0]:
            best = (id_aln, ident / min(len(query), len(t)), t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', type=Path, required=True)
    ap.add_argument('--split-csv', type=Path,
                    help='v10 training split; omit when --target-fasta is given')
    ap.add_argument('--train-root', type=Path,
                    help='v10 training tree; omit when --target-fasta is given')
    ap.add_argument('--target-fasta', type=Path,
                    help='pre-built training chains to compare against, e.g. another '
                         "method's PDBBind set from scripts/pdbbind_train_seqs.py")
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=None)
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()

    chains = pd.read_csv(args.seqs, sep='\t')
    uniq = chains.drop_duplicates('seq_md5')[['seq_md5', 'seq']]
    print(f'{len(chains)} CrossDock chains, {len(uniq)} unique sequences')

    if args.target_fasta:
        train = read_fasta(args.target_fasta)
        print(f'target chains from {args.target_fasta}: {len(train)}')
    elif args.split_csv and args.train_root:
        cache = args.cache or args.out.parent / 'train_chains.fasta'
        train = train_fasta(args.train_root, args.split_csv, cache, args.jobs)
    else:
        ap.error('give either --target-fasta or both --split-csv and --train-root')

    qf = args.out.parent / 'query.fasta'
    with qf.open('w') as fh:
        for r in uniq.itertuples():
            fh.write(f'>{r.seq_md5}\n{r.seq}\n')
    cands = mmseqs_candidates(qf, train, args.jobs)
    print(f'MMseqs2 returned candidates for {len(cands)}/{len(uniq)} queries')

    rows = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(best_nw)(r.seq, cands.get(r.seq_md5, [])) for r in uniq.itertuples())
    uniq = uniq.assign(
        identity_aln=[r[0] for r in rows],
        identity_min=[r[1] for r in rows],
        best_train=[train.get(r[2], '') if r[2] else '' for r in rows],
        n_candidates=[len(cands.get(m, [])) for m in uniq.seq_md5])

    per_chain = chains.merge(uniq.drop(columns='seq'), on='seq_md5')
    per_system = (per_chain.sort_values('identity_aln', ascending=False)
                  .groupby(['subset', 'system'], as_index=False).first()
                  .drop(columns=['seq', 'chain']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_system.to_csv(args.out, sep='\t', index=False)
    uniq.drop(columns='seq').to_csv(args.out.with_suffix('.unique.tsv'), sep='\t', index=False)
    print(f'wrote {args.out} ({len(per_system)} systems)')

    band = pd.cut(per_system.identity_aln, [-.01, .3, .7, .9, 1.01],
                  labels=['<30%', '30-70%', '70-90%', '>=90%'])
    print(pd.crosstab(per_system.subset, band).to_string())


if __name__ == '__main__':
    main()
