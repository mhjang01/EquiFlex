'''
Featurize v2_plinder_5A:
  - Symlink unimol_embeddings.pt from v2_plinder/ where present (ligand
    embeddings are coord-frame-independent; SDFs unchanged).
  - Run featurize_data.main with run_unimol=False on the 5A pocket dirs.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

import os
from pathlib import Path

from equiflex.dataprep.featurize_data import main as featurize_main

OLD_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder')
NEW_ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')
SPLITS = ['train', 'val', 'test']


def symlink_unimol():
    counts = {}
    for split in SPLITS:
        new_split = NEW_ROOT / split
        old_split = OLD_ROOT / split
        if not new_split.exists():
            continue
        linked = missing = already = 0
        for sysdir in new_split.iterdir():
            if not sysdir.is_dir() or sysdir.name == 'errors':
                continue
            new_unimol = sysdir / 'unimol_embeddings.pt'
            if new_unimol.exists() or new_unimol.is_symlink():
                already += 1
                continue
            old_unimol = old_split / sysdir.name / 'unimol_embeddings.pt'
            if not old_unimol.exists():
                missing += 1
                continue
            os.symlink(old_unimol.resolve(), new_unimol)
            linked += 1
        counts[split] = {'linked': linked, 'already': already, 'no_source': missing}
        print(f'[{split}] symlinked={linked} already={already} no_source={missing}')
    return counts


def main():
    print('=== STEP A: Symlink UniMol embeddings ===')
    symlink_unimol()

    print('\n=== STEP B: Featurize (step1 + step2, skip unimol) ===')
    for split in SPLITS:
        data_dir = NEW_ROOT / split
        if not data_dir.exists():
            print(f'[{split}] missing, skip'); continue
        print(f'\n[{split}] featurize_main on {data_dir}')
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


if __name__ == '__main__':
    main()
