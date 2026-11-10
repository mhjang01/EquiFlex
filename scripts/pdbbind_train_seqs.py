"""Receptor chain sequences for DiffBindFR's PDBBind training set.

DiffBindFR trains on the PDBBind v2020 time split, the EquiBind convention its
own eval code reads (`timesplit_test`, zenodo 6408497), so its training ids are
`timesplit_no_lig_overlap_train` plus the val list -- val is included because a
receptor seen in validation is still a receptor the method has seen.

Writes a FASTA of unique chains, the target set for the PDBBind side of the
CrossDock identity analysis, so each method can be stratified against its own
training set instead of only against ours.

    python scripts/pdbbind_train_seqs.py --pdbbind /mnt/drive/pdbbind/total \
        --splits ~/references/dock/EquiBind/data --out data/crossdock/pdbbind_train_chains.fasta
"""
import argparse
from pathlib import Path

from joblib import Parallel, delayed

from crossdock_seqs import chain_sequences

SPLIT_FILES = ('timesplit_no_lig_overlap_train', 'timesplit_no_lig_overlap_val')


def one_system(root, pdb_id):
    pdb = root / pdb_id / f'{pdb_id}_protein.pdb'
    if not pdb.exists():
        return pdb_id, {}
    return pdb_id, chain_sequences(pdb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdbbind', type=Path, default=Path('/mnt/drive/pdbbind/total'))
    ap.add_argument('--splits', type=Path,
                    default=Path.home() / 'references/dock/EquiBind/data')
    ap.add_argument('--out', type=Path,
                    default=Path('data/crossdock/pdbbind_train_chains.fasta'))
    ap.add_argument('--workers', type=int, default=11)
    args = ap.parse_args()

    ids = []
    for name in SPLIT_FILES:
        rows = (args.splits / name).read_text().split()
        print(f'{name}: {len(rows)} ids')
        ids += rows
    ids = sorted(set(ids))
    print(f'{len(ids)} unique training ids')

    out = Parallel(n_jobs=args.workers, batch_size=64, verbose=1)(
        delayed(one_system)(args.pdbbind, i) for i in ids)

    uniq, n_missing = {}, 0
    for pdb_id, seqs in out:
        if not seqs:
            n_missing += 1
        for ch, s in seqs.items():
            uniq.setdefault(s, []).append(f'{pdb_id}:{ch}')
    if n_missing:
        print(f'{n_missing} ids had no readable receptor')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as fh:
        for i, (s, members) in enumerate(uniq.items()):
            fh.write(f'>p{i}|{";".join(members)[:200]}\n{s}\n')
    print(f'wrote {args.out}: {len(uniq)} unique chains')


if __name__ == '__main__':
    main()
