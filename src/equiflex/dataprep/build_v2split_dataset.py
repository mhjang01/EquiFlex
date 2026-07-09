'''
Build v2-split dataset under v2_plinder_5A_v2split/ from the survivor lists in
v2_plinder_5A/logs/{train,val,test}_v2_survivors.txt.

Strategy:
  1. For every survivor system_id:
       a. If matching dir exists in v2_plinder_5A/{any_split}/ -> symlink to new path
          (parse uses same 5A + asym_id rules, so artifacts are valid regardless
           of original split label).
       b. Else mark for fresh parse.
  2. Fresh-parse missing systems into the new path.
  3. For systems missing unimol_embeddings.pt, also try symlink from
     v2_plinder/{any_split}/<sys>/unimol_embeddings.pt before regenerating.
  4. Run featurize_data.main on each split with run_unimol=True
     (skip-existing skips already-cached step1/2 artifacts).

Outputs:
  /home/mhjang/workspace/flex/v2_plinder_5A_v2split/{train,val,test}/<sys>/
  /home/mhjang/workspace/flex/v2_plinder_5A_v2split/logs/build_summary.json
'''
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, '/home/mhjang/workspace/flex')

from joblib import Parallel, delayed
from tqdm import tqdm

from equiflex.dataprep.metadata_extraction import parse_single_system
from equiflex.dataprep.featurize_data import main as featurize_main

SOURCE_SYSTEM_DIR = Path('/mnt/drive/new_plinder/v2/systems/')
OLD_5A_ROOT  = Path('/home/mhjang/workspace/flex/v2_plinder_5A')
OLD_PL50_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')
NEW_ROOT    = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
SPLITS      = ['train', 'val', 'test']


def index_dirs(root):
    '''Return dict[system_id] -> abs_path for all system dirs under root/{splits}.'''
    out = {}
    for split in SPLITS:
        sd = root / split
        if not sd.exists():
            continue
        for d in sd.iterdir():
            if d.is_dir() and d.name != 'errors':
                out.setdefault(d.name, d.resolve())
    return out


def link_or_mark_missing(survivors_by_split, idx_5a):
    '''Symlink existing 5A artifacts to new split path; collect missing system_ids.'''
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    missing = {s: [] for s in SPLITS}
    linked = {s: 0 for s in SPLITS}
    src_absent = {s: 0 for s in SPLITS}

    for split, sids in survivors_by_split.items():
        out_split = NEW_ROOT / split
        out_split.mkdir(parents=True, exist_ok=True)
        for sid in sids:
            new_path = out_split / sid
            if new_path.exists() or new_path.is_symlink():
                continue
            src = idx_5a.get(sid)
            if src is None:
                # Need fresh parse if source CIF dir exists.
                if (SOURCE_SYSTEM_DIR / sid).exists():
                    missing[split].append(sid)
                else:
                    src_absent[split] += 1
                continue
            os.symlink(src, new_path)
            linked[split] += 1
    return linked, missing, src_absent


def parse_missing(missing, n_jobs=-1):
    '''Parse fresh from source CIF for missing system_ids.'''
    produced = {s: 0 for s in SPLITS}
    for split, sids in missing.items():
        if not sids: continue
        out_split = NEW_ROOT / split
        sys_dirs = [SOURCE_SYSTEM_DIR / sid for sid in sids if (SOURCE_SYSTEM_DIR / sid).exists()]
        print(f'[parse:{split}] {len(sys_dirs)} dirs to parse')
        Parallel(n_jobs=n_jobs)(
            delayed(parse_single_system)(sd, out_split, True)
            for sd in tqdm(sys_dirs, desc=f'parse:{split}')
        )
        produced[split] = sum(
            1 for d in out_split.iterdir()
            if d.is_dir() and (d / 'pocket_residues.pkl').exists()
        )
    return produced


def symlink_unimol_from_pl50(idx_pl50):
    '''Pull unimol embeddings from old pl50 root where new split lacks them.'''
    linked = {s: 0 for s in SPLITS}
    no_src = {s: 0 for s in SPLITS}
    for split in SPLITS:
        out_split = NEW_ROOT / split
        if not out_split.exists():
            continue
        for sysdir in out_split.iterdir():
            if not sysdir.is_dir():
                continue
            target = sysdir / 'unimol_embeddings.pt'
            if target.exists() or target.is_symlink():
                continue
            src = idx_pl50.get(sysdir.name)
            if src is None:
                no_src[split] += 1
                continue
            src_unimol = src / 'unimol_embeddings.pt'
            if src_unimol.exists():
                os.symlink(src_unimol.resolve(), target)
                linked[split] += 1
            else:
                no_src[split] += 1
    return linked, no_src


def run_featurize():
    for split in SPLITS:
        data_dir = NEW_ROOT / split
        if not data_dir.exists():
            continue
        print(f'\n[featurize:{split}] on {data_dir}')
        featurize_main(
            data_dir=str(data_dir),
            n_jobs=-1,
            skip_existing=True,
            cleanup_invalid=True,
            run_esm3=True,
            run_unimol=True,
            unimol_size='1.1B',
            device='cuda',
            strict=True,
        )


def main():
    log_dir = NEW_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print('Loading survivor lists...')
    survivors = {}
    for split in SPLITS:
        fp = NEW_ROOT / 'logs' / f'{split}_v2_survivors.txt'
        survivors[split] = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
        print(f'  {split}: {len(survivors[split])} survivors')

    print('\nIndexing existing 5A artifacts...')
    idx_5a = index_dirs(OLD_5A_ROOT)
    print(f'  {len(idx_5a)} system_ids in v2_plinder_5A across splits')

    print('\nIndexing old pl50 (v2_plinder/) for UniMol reuse...')
    idx_pl50 = index_dirs(OLD_PL50_ROOT)
    print(f'  {len(idx_pl50)} system_ids in v2_plinder')

    print('\n=== STAGE A: symlink existing 5A dirs ===')
    linked_a, missing, src_absent = link_or_mark_missing(survivors, idx_5a)
    print(f'  linked: {linked_a}')
    print(f'  missing (need fresh parse): {[(s, len(v)) for s, v in missing.items()]}')
    print(f'  source CIF absent on disk: {src_absent}')

    print('\n=== STAGE B: parse missing from source CIF ===')
    produced = parse_missing(missing)
    print(f'  produced pkl dirs from fresh parse: {produced}')

    print('\n=== STAGE C: symlink unimol from old pl50 where missing ===')
    linked_uni, no_src_uni = symlink_unimol_from_pl50(idx_pl50)
    print(f'  unimol linked: {linked_uni}')
    print(f'  unimol no_src: {no_src_uni}')

    summary = {
        'survivor_counts': {s: len(v) for s, v in survivors.items()},
        'symlinked_from_5A': linked_a,
        'missing_after_5A': {s: len(v) for s, v in missing.items()},
        'source_cif_absent': src_absent,
        'fresh_parse_produced': produced,
        'unimol_symlinked_from_pl50': linked_uni,
        'unimol_no_source': no_src_uni,
    }
    (log_dir / 'build_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nBuild summary -> {log_dir/"build_summary.json"}')

    if os.environ.get('SKIP_FEATURIZE') == '1':
        print('\n[SKIP_FEATURIZE=1] stopping after STAGE C for criteria verification.')
        return
    print('\n=== STAGE D: featurize (step1 + step2 + step3) ===')
    run_featurize()

    # Final counts
    final = {}
    for split in SPLITS:
        sd = NEW_ROOT / split
        if not sd.exists(): continue
        dirs = [d for d in sd.iterdir() if d.is_dir() and d.name != 'errors']
        final[split] = {
            'dirs': len(dirs),
            'lig_feat':  sum(1 for d in dirs if (d/'ligand_features.pt').exists()),
            'pocket_feat': sum(1 for d in dirs if (d/'pocket_features.pt').exists()),
            'esm3':  sum(1 for d in dirs if (d/'esm3_embeddings.pt').exists()),
            'unimol': sum(1 for d in dirs if (d/'unimol_embeddings.pt').exists()),
            'complete': sum(1 for d in dirs if all(
                (d/f).exists() for f in
                ['ligand_features.pt','pocket_features.pt','esm3_embeddings.pt','unimol_embeddings.pt']
            )),
        }
    summary['final'] = final
    (log_dir / 'build_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nFINAL counts:')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
