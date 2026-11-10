"""Extract receptor chain sequences from the CrossDock benchmark.

One row per (system, chain). The CrossDock subsets are built from ensembles of a
few targets, so the distinct sequence count is far below the system count; the
unique set written here is what the similarity search runs against.

    python scripts/crossdock_seqs.py --dataset ~/workspace/equiflex/comparison/diffbindfr_run/crossdock_dataset \
        --out data/crossdock/seqs.tsv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}
MIN_LEN = 20


def chain_sequences(pdb_path):
    """-> {chain: sequence} over standard residues, in file order.

    Keyed on (chain, resnum, icode) so insertion codes do not collapse residues,
    and taken from CA records so a residue is counted once.
    """
    seqs, seen = {}, set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
                continue
            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE:
                continue
            chain = line[21].strip() or 'A'
            key = (chain, line[22:26].strip(), line[26].strip())
            if key in seen:
                continue
            seen.add(key)
            seqs.setdefault(chain, []).append(THREE_TO_ONE[resname])
    return {c: ''.join(s) for c, s in seqs.items() if len(s) >= MIN_LEN}


def one_system(subset, system, pdb_path):
    return [{'subset': subset, 'system': system, 'chain': c, 'n_res': len(s),
             'seq_md5': hashlib.md5(s.encode()).hexdigest(), 'seq': s}
            for c, s in chain_sequences(pdb_path).items()]


def iter_systems(dataset):
    for subset_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        children = sorted(p for p in subset_dir.iterdir() if p.is_dir())
        # Ensemble/ holds the per-target directories rather than systems.
        nested = children and not (children[0] / 'protein.pdb').exists()
        for group in (children if nested else [subset_dir]):
            label = f'{subset_dir.name}/{group.name}' if nested else subset_dir.name
            for sys_dir in sorted(p for p in group.iterdir() if p.is_dir()):
                pdb = sys_dir / 'protein.pdb'
                if pdb.exists():
                    yield label, sys_dir.name, pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    jobs = list(iter_systems(args.dataset))
    print(f'systems with a receptor: {len(jobs)}')
    rows = Parallel(n_jobs=args.workers, verbose=1)(
        delayed(one_system)(sub, sys_id, pdb) for sub, sys_id, pdb in jobs)
    df = pd.DataFrame([r for group in rows for r in group])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    uniq = df.drop_duplicates('seq_md5')
    fasta = args.out.with_suffix('.unique.fasta')
    with fasta.open('w') as fh:
        for r in uniq.itertuples():
            fh.write(f'>{r.seq_md5}\n{r.seq}\n')
    print(f'wrote {args.out} ({len(df)} chains over {df.system.nunique()} systems)')
    print(f'wrote {fasta} ({len(uniq)} unique sequences)')
    print(df.groupby('subset').agg(systems=('system', 'nunique'),
                                   chains=('chain', 'size'),
                                   unique_seqs=('seq_md5', 'nunique')).to_string())


if __name__ == '__main__':
    main()
