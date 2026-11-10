"""Filtered training split for data_v2_plinder from the audit table (data/posebusters/train_seqsim_to_pb.tsv).

Filters (each optional):
  --drop-covalent-sugars   sugar-class ligands with a heavy atom < 1.8 A from the receptor (N-glycans)
  --drop-covalent-all      any ligand with a heavy atom < 1.8 A from the receptor
  --max-seqsim X           drop systems with any chain >= X identity to a PoseBusters chain (NW, alignment length)
  --dedup-copies R         within one PDB entry and one (receptor seq-set, ligand) complex, drop a copy whose pocket
                           RMSD and ligand-pose RMSD (after pocket superposition) to an already-kept copy are both < R A
Val and test rows of the source CSV are copied unchanged.
    python scripts/build_filtered_split.py --src splits_max500.csv --out splits_v9.csv --drop-covalent-sugars --max-seqsim 0.70 --dedup-copies 1.0
"""
import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from joblib import Parallel, delayed

def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0); U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))

def load(root, s):
    pf = torch.load(f'{root}/{s}/pocket_features.pt', weights_only=False); lf = torch.load(f'{root}/{s}/ligand_features.pt', weights_only=False)
    n = list(pf['n_atoms_per_residue']); off = np.cumsum([0] + n[:-1])
    atoms = {(k[1], k[2], k[3]): (o, m) for k, o, m in zip(pf['residue_keys'], off, n)}
    return pf['coords'].float().numpy(), lf['coords'].float().numpy(), atoms

def copy_distance(root, a, b):
    """(pocket RMSD, ligand pose RMSD) between two copies, or None if not comparable."""
    try:
        Pa, La, Aa = load(root, a); Pb, Lb, Ab = load(root, b)
        if La.shape != Lb.shape: return None
        common = [k for k in Aa if k in Ab and Aa[k][1] == Ab[k][1]]
        if len(common) < 5: return None
        ia = np.concatenate([np.arange(Aa[k][0], Aa[k][0] + Aa[k][1]) for k in common]); ib = np.concatenate([np.arange(Ab[k][0], Ab[k][0] + Ab[k][1]) for k in common])
        P, Q = Pa[ia], Pb[ib]; R, prmsd = kabsch_rmsd(P, Q)
        La2 = (La - P.mean(0)) @ R.T + Q.mean(0)
        return prmsd, float(np.sqrt(((La2 - Lb) ** 2).sum(1).mean()))
    except Exception:
        return None

def dedup_group(root, systems, thr):
    kept = [systems[0]]
    for s in systems[1:]:
        dup = False
        for k in kept:
            d = copy_distance(root, k, s)
            if d is not None and d[0] < thr and d[1] < thr: dup = True; break
        if not dup: kept.append(s)
    return kept

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit', type=Path, default=Path('data/posebusters/train_seqsim_to_pb.tsv'))
    # The local training set was removed on 2026-11-07; it lives on the clusters
    # (dream: /scratch/jyl_snu/mhjang/workspace/equiflex/data_v2_plinder/train).
    # Pass --root explicitly, or rsync train/ back beside the splits_*.csv manifests.
    ap.add_argument('--root', type=Path, default=Path('data_v2_plinder/train'))
    ap.add_argument('--drop-covalent-sugars', action='store_true'); ap.add_argument('--drop-covalent-all', action='store_true')
    ap.add_argument('--max-seqsim', type=float, default=None); ap.add_argument('--dedup-copies', type=float, default=None)
    a = ap.parse_args()
    src = pd.read_csv(a.src); audit = pd.read_csv(a.audit, sep='\t').set_index('system')
    tr = src[src.split == 'train'].copy(); n0 = len(tr)
    tr = tr[tr.system_id.isin(audit.index)]; au = audit.loc[tr.system_id]
    keep = pd.Series(True, index=tr.system_id)
    cov = au.min_ligrec_A < 1.8
    if a.drop_covalent_all: keep &= ~cov.values
    elif a.drop_covalent_sugars: keep &= ~(cov & au.cat.str.startswith('sugar')).values
    print(f'train {n0} -> after covalent filter {int(keep.sum())}')
    if a.max_seqsim is not None:
        keep &= (au.max_id < a.max_seqsim).values; print(f'  after seqsim < {a.max_seqsim:.2f}: {int(keep.sum())}')
    kept_ids = tr.system_id[keep.values].tolist()
    if a.dedup_copies is not None:
        sub = au.loc[kept_ids]; groups = sub.groupby(['pdb', 'complex']).apply(lambda d: list(d.index)).tolist()
        multi = [g for g in groups if len(g) > 1]; single = [g[0] for g in groups if len(g) == 1]
        print(f'  dedup: {len(multi)} copy groups ({sum(len(g) for g in multi)} systems) ...', flush=True)
        kept_groups = Parallel(n_jobs=22, batch_size=32)(delayed(dedup_group)(str(a.root), g, a.dedup_copies) for g in multi)
        kept_ids = single + [s for g in kept_groups for s in g]
        print(f'  after dedup (< {a.dedup_copies} A pocket AND ligand-pose): {len(kept_ids)}')
    out = pd.concat([src[src.split != 'train'], src[src.system_id.isin(kept_ids)]]).sort_index()
    out.to_csv(a.out, index=False)
    uniq = au.loc[kept_ids]['complex'].nunique()
    print(f'written {a.out}: train {len(kept_ids)} systems / {uniq} unique complexes; val {int((out.split=="val").sum())} test {int((out.split=="test").sum())}')

if __name__ == '__main__':
    main()
