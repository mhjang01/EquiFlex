'''
Training-data prep for the v2-split pli_qcov<90 + atom<450 + missing10 + metal<=2
target. Order (embeddings LAST since slow):

  Stage 1  metadata parse (parse_single_system) for final_to_prepare
  Stage 2  step1 graph features (ligand_features/pocket_features), no embeddings
  Stage 3  real post-parse filter: pocket atoms < 450 (parsed, non-missing) +
           validity; DELETE failures and any dir not in final_candidates
  Stage 4  ESM3 + Uni-Mol embeddings on survivors only
  Stage 5  report (survivors, unique SMILES)

Run from repo root. Logs stage markers to stdout.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')
import json
import shutil
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, load
from tqdm import tqdm

from equiflex.dataprep.metadata_extraction import parse_single_system
from equiflex.dataprep.featurize_data import main as featurize_main

ROOT = Path('/home/mhjang/workspace/flex')
TR = ROOT / 'data_v2_plinder' / 'train'
LOG = ROOT / 'data_v2_plinder' / 'logs'
SRC = Path('/mnt/drive/new_plinder/v2/systems')
ATOM_MAX = 450
REQUIRED_FEAT = ['ligand_features.pt', 'pocket_features.pt']
PARSE_PKLS = ['ligands.pkl', 'pocket_residues.pkl', 'sequence_data.pkl']


def pocket_atoms(sid):
    fp = TR / sid / 'pocket_residues.pkl'
    if not fp.exists():
        return sid, None
    try:
        pk = load(fp)
        n = 0
        for ch in pk.values():
            for res in ch.values():
                n += int((~np.asarray(res['missing_masks'])).sum())
        return sid, n
    except Exception:
        return sid, None


def main():
    candidates = sorted({l.strip() for l in open(LOG / 'final_candidates.txt') if l.strip()})
    to_prepare = sorted({l.strip() for l in open(LOG / 'final_to_prepare.txt') if l.strip()})
    cset = set(candidates)
    print(f'[plan] candidates={len(candidates)} to_prepare={len(to_prepare)}', flush=True)

    # ---------- Stage 1: metadata parse ----------
    print('STAGE1_START metadata parse', flush=True)
    src_dirs, missing_src = [], []
    for sid in to_prepare:
        d = SRC / sid
        (src_dirs.append(d) if d.is_dir() else missing_src.append(sid))
    print(f'[stage1] {len(src_dirs)} source dirs, {len(missing_src)} missing source', flush=True)
    TR.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=-1)(
        delayed(parse_single_system)(sd, TR, True) for sd in tqdm(src_dirs, desc='metadata'))
    print('STAGE1_DONE', flush=True)

    # ---------- Stage 2: step1 graph features (no embeddings) ----------
    print('STAGE2_START step1 graph features', flush=True)
    featurize_main(data_dir=str(TR), n_jobs=-1, skip_existing=True,
                   cleanup_invalid=True, run_esm3=False, run_unimol=False,
                   unimol_size='1.1B', device='cuda', strict=False)
    print('STAGE2_DONE', flush=True)

    # ---------- Stage 3: real atom<450 + validity filter ----------
    print('STAGE3_START post-parse filter', flush=True)
    on_disk = [d.name for d in TR.iterdir() if d.is_dir()]
    pa = dict(Parallel(n_jobs=-1)(delayed(pocket_atoms)(s) for s in tqdm(on_disk, desc='atom450')))
    removed = {'not_candidate': [], 'atoms_ge450': [], 'parse_or_feature_invalid': []}
    for sid in on_disk:
        d = TR / sid
        if sid not in cset:
            removed['not_candidate'].append(sid); shutil.rmtree(d); continue
        n = pa.get(sid)
        has_feat = all((d / f).exists() for f in REQUIRED_FEAT)
        has_pkls = all((d / f).exists() for f in PARSE_PKLS)
        if n is None or not has_pkls:
            removed['parse_or_feature_invalid'].append(sid); shutil.rmtree(d); continue
        if n >= ATOM_MAX:
            removed['atoms_ge450'].append(sid); shutil.rmtree(d); continue
        if not has_feat:
            removed['parse_or_feature_invalid'].append(sid); shutil.rmtree(d); continue
    for k, v in removed.items():
        (LOG / f'prep_removed_{k}.txt').write_text('\n'.join(sorted(v)) + '\n')
    print(f"[stage3] removed: " + ', '.join(f'{k}={len(v)}' for k, v in removed.items()), flush=True)
    print('STAGE3_DONE', flush=True)

    # ---------- Stage 4: embeddings (ESM3 + Uni-Mol) ----------
    print('STAGE4_START embeddings', flush=True)
    featurize_main(data_dir=str(TR), n_jobs=-1, skip_existing=True,
                   cleanup_invalid=False, run_esm3=True, run_unimol=True,
                   unimol_size='1.1B', device='cuda', strict=False)
    print('STAGE4_DONE', flush=True)

    # ---------- Stage 5: report ----------
    final = [d.name for d in TR.iterdir() if d.is_dir()]
    REQ_ALL = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']
    complete = [s for s in final if all((TR / s / f).exists() for f in REQ_ALL)]
    summary = {
        'final_dirs': len(final),
        'fully_featurized': len(complete),
        'missing_source': len(missing_src),
        'removed_not_candidate': len(removed['not_candidate']),
        'removed_atoms_ge450': len(removed['atoms_ge450']),
        'removed_invalid': len(removed['parse_or_feature_invalid']),
    }
    (LOG / 'run_training_prep_summary.json').write_text(json.dumps(summary, indent=2))
    print('PREP_COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
