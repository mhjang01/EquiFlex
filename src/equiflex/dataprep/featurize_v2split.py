'''
Featurize v2_plinder_5A_v2split using the patched OOM-safe featurize_data.
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

from pathlib import Path

from equiflex.dataprep.featurize_data import main as featurize_main

ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
SPLITS = ['train', 'val', 'test']


def main():
    for split in SPLITS:
        data_dir = ROOT / split
        if not data_dir.exists():
            continue
        print(f'\n=== featurize:{split} on {data_dir} ===')
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
