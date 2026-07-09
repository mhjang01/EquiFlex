"""
Repair 141 flagged systems by dropping clash residues (rec atom < 0.1Å from
any lig atom). Reuses existing pkls — no source CIF needed and no ESM3 model
call (sliced from cached embeddings).

For each flagged system:
  1. Load pocket_residues.pkl, ligand_features.pt, pocket_features.pt, esm3_embeddings.pt
  2. Identify clash residues (per-residue: any atom < 0.1Å from any lig atom)
  3. Drop those residues from pocket_residues dict
  4. Re-run process_pocket_residues → new pocket_features.pt
  5. Slice esm3_embeddings.pt to match new residue order
  6. Backup originals (.bak) and overwrite

Skips ligand_features.pt and unimol_embeddings.pt (ligand unchanged).
After repair, re-adds systems to splits.csv from splits.csv.bak.
"""
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from joblib import load, dump
from tqdm import tqdm

from equiflex.dataprep.featurize_data import process_pocket_residues


def parse_flagged_ids(report_path: Path):
    """Return {system_id: split} from nan_scan_report.txt."""
    out = {}
    cur_split = None
    with open(report_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('=== Split:'):
                cur_split = line.split(':', 1)[1].split('(')[0].strip()
                continue
            if not line or line.startswith('---') or line.startswith('==='):
                continue
            sid = line.split('\t', 1)[0].strip()
            if sid and cur_split:
                out[sid] = cur_split
    return out


def find_clash_residues(pocket_residues: dict, lig_coords: np.ndarray, threshold: float = 0.1):
    """Return set of (chain_id, res_key) tuples for residues with any atom
    within `threshold` Å of any ligand atom."""
    clash = set()
    for chain_id, residues in pocket_residues.items():
        for res_key, res_data in residues.items():
            coords = np.asarray(res_data['coords'])  # [n_atoms, 3]
            if coords.size == 0:
                continue
            # Pairwise distances: [n_atoms, n_lig]
            diff = coords[:, None, :] - lig_coords[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            if (d < threshold).any():
                clash.add((chain_id, res_key))
    return clash


def repair_system(system_dir: Path, threshold: float = 0.1):
    pocket_residues_path = system_dir / 'pocket_residues.pkl'
    pocket_feats_path = system_dir / 'pocket_features.pt'
    ligand_feats_path = system_dir / 'ligand_features.pt'
    esm_path = system_dir / 'esm3_embeddings.pt'

    pocket_residues = load(pocket_residues_path)
    pocket_feats_old = torch.load(pocket_feats_path, weights_only=False, map_location='cpu')
    ligand_feats = torch.load(ligand_feats_path, weights_only=False, map_location='cpu')
    esm_old = torch.load(esm_path, weights_only=False, map_location='cpu')

    lig_coords = np.asarray(ligand_feats['coords'])

    clashes = find_clash_residues(pocket_residues, lig_coords, threshold)
    if not clashes:
        return {'status': 'no_clash_found', 'dropped': 0}

    # Build OLD residue_keys ordering (matches process_pocket_residues iteration:
    # sort chain_id, then sort residues by res_key[1] = resnum).
    old_keys_in_order = []
    drop_indices = []
    keep_indices = []
    idx = 0
    new_pocket_residues = {}
    for chain_id, residues in sorted(pocket_residues.items()):
        kept_residues = {}
        for (resname, resnum, entity_id), res_data in sorted(residues.items(), key=lambda x: x[0][1]):
            res_key = (resname, resnum, entity_id)
            old_keys_in_order.append((chain_id, resname, resnum, entity_id))
            if (chain_id, res_key) in clashes:
                drop_indices.append(idx)
            else:
                keep_indices.append(idx)
                kept_residues[res_key] = res_data
            idx += 1
        if kept_residues:
            new_pocket_residues[chain_id] = kept_residues

    # Sanity: old residue_keys must match what process_pocket_residues produced before
    if old_keys_in_order != [tuple(k) for k in pocket_feats_old['residue_keys']]:
        return {'status': 'residue_key_mismatch',
                'dropped': len(drop_indices),
                'detail': f'old_order_size={len(old_keys_in_order)} feat_size={len(pocket_feats_old["residue_keys"])}'}

    if not new_pocket_residues:
        return {'status': 'all_residues_dropped', 'dropped': len(drop_indices)}

    # Re-featurize pocket
    pocket_feats_new = process_pocket_residues(new_pocket_residues)
    if pocket_feats_new is None:
        return {'status': 'featurize_failed', 'dropped': len(drop_indices)}

    # Slice ESM rows
    keep_t = torch.tensor(keep_indices, dtype=torch.long)
    esm_new = esm_old.index_select(0, keep_t)

    if esm_new.shape[0] != pocket_feats_new['atom_features'].shape[0] and False:
        pass  # ESM is per-residue (1 row/residue), not per-atom — skip this check

    expected_residues = len(pocket_feats_new['residue_keys'])
    if esm_new.shape[0] != expected_residues:
        return {'status': 'esm_size_mismatch',
                'dropped': len(drop_indices),
                'detail': f'esm_rows={esm_new.shape[0]} new_residues={expected_residues}'}

    # Backup + write
    for p in (pocket_residues_path, pocket_feats_path, esm_path):
        bak = p.with_suffix(p.suffix + '.bak')
        if not bak.exists():
            shutil.copy(p, bak)
    dump(new_pocket_residues, pocket_residues_path, compress=3)
    torch.save(pocket_feats_new, pocket_feats_path)
    torch.save(esm_new, esm_path)

    return {'status': 'ok', 'dropped': len(drop_indices),
            'old_residues': len(old_keys_in_order),
            'new_residues': expected_residues}


def restore_splits_for_fixed(splits_path: Path, fixed_ids: dict):
    """Append fixed system_ids back into splits.csv with their original splits."""
    with open(splits_path) as f:
        existing = set()
        header = f.readline()
        rows = [header]
        for line in f:
            sid = line.split(',', 1)[0].strip()
            existing.add(sid)
            rows.append(line)

    added = 0
    for sid, split in fixed_ids.items():
        if sid in existing:
            continue
        rows.append(f'{sid},{split}\n')
        added += 1

    bak = splits_path.with_suffix('.csv.bak2')
    if not bak.exists():
        shutil.copy(splits_path, bak)

    with open(splits_path, 'w') as f:
        f.writelines(rows)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='nan_scan_report.txt')
    ap.add_argument('--data', default='./data_v2_plinder')
    ap.add_argument('--threshold', type=float, default=0.1)
    ap.add_argument('--splits', default='data_v2_plinder/splits.csv')
    ap.add_argument('--no-restore-splits', action='store_true',
                    help='Skip adding repaired systems back to splits.csv')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    flagged = parse_flagged_ids(Path(args.report))
    print(f'Flagged systems: {len(flagged)}')
    if args.limit:
        items = list(flagged.items())[:args.limit]
        flagged = dict(items)

    root = Path(args.data)
    results = {}
    repaired = {}
    for sid, split in tqdm(flagged.items()):
        sys_dir = root / split / sid
        if not sys_dir.exists():
            results[sid] = {'status': 'missing_dir'}
            continue
        if args.dry_run:
            results[sid] = {'status': 'dry_run'}
            continue
        try:
            r = repair_system(sys_dir, threshold=args.threshold)
        except Exception as e:
            r = {'status': f'error: {e!s}'[:120]}
        results[sid] = r
        if r.get('status') == 'ok':
            repaired[sid] = split

    # Summary
    by_status = {}
    for sid, r in results.items():
        by_status.setdefault(r['status'], []).append(sid)
    print('\n=== Repair summary ===')
    for st, ids in by_status.items():
        print(f'  {st}: {len(ids)}')
        if st != 'ok':
            for sid in ids[:5]:
                print(f'    {sid}: {results[sid]}')

    if repaired and not args.no_restore_splits and not args.dry_run:
        added = restore_splits_for_fixed(Path(args.splits), repaired)
        print(f'\nAdded {added} repaired systems back to {args.splits} (backup at .bak2)')


if __name__ == '__main__':
    main()
