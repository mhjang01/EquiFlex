'''
Build v2_plinder_5A_v2split_flat/ from v2_plinder_5A_v2split/{train,val,test}.

Mirrors flatten_remote.sh for the new dataset:
  v2_plinder_5A_v2split_flat/<system_id> -> ../v2_plinder_5A_v2split/<split>/<system_id>
  v2_plinder_5A_v2split_flat/split_train.txt
  v2_plinder_5A_v2split_flat/split_val.txt
  v2_plinder_5A_v2split_flat/splits.csv      (header + system_id,split rows)
'''
import os
from pathlib import Path

SRC = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split')
DST = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split_flat')
SPLITS = ['train', 'val', 'test']


def main():
    DST.mkdir(parents=True, exist_ok=True)
    csv_lines = ['system_id,split']
    counts = {}
    for split in SPLITS:
        src_split = SRC / split
        if not src_split.exists():
            continue
        dirs = [d for d in src_split.iterdir()
                if d.is_dir() and d.name != 'errors'
                and (d / 'ligand_features.pt').exists()]
        sids = sorted(d.name for d in dirs)
        (DST / f'split_{split}.txt').write_text('\n'.join(sids) + '\n')
        for sid in sids:
            csv_lines.append(f'{sid},{split}')
            if split == 'test':
                continue  # mirror existing convention: test not in flat
            dst_link = DST / sid
            if dst_link.exists() or dst_link.is_symlink():
                continue
            src_dir = (src_split / sid).resolve()
            os.symlink(src_dir, dst_link)
        counts[split] = len(sids)
    (DST / 'splits.csv').write_text('\n'.join(csv_lines) + '\n')
    print('Counts:', counts)
    print(f'splits.csv:    {DST/"splits.csv"}  ({sum(counts.values())+1} rows)')
    print(f'split_*.txt:   {DST}/split_{{train,val,test}}.txt')
    n_links = sum(1 for p in DST.iterdir() if p.is_symlink())
    print(f'symlinks: {n_links} (test split skipped per convention)')


if __name__ == '__main__':
    main()
