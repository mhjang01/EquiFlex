'''
Re-prep val/test from PLINDER native v2 split (split.parquet), parsed AS-IS
(no quality/PB filtering). For each split: parse_single_system -> graph features
-> ESM3 + Uni-Mol. Align dirs to native split (remove deprecated).

Output dirs: data_v2_plinder/{val,test}
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

from equiflex.dataprep.metadata_extraction import parse_single_system
from equiflex.dataprep.featurize_data import main as featurize_main

ROOT = Path('/home/mhjang/workspace/flex/data_v2_plinder')
LOG = ROOT / 'logs'
SRC = Path('/mnt/drive/new_plinder/v2/systems')
SPLIT_PQ = '/mnt/drive/plinder/2024-06/v2/splits/split.parquet'
REQ = ['ligand_features.pt', 'pocket_features.pt', 'esm3_embeddings.pt', 'unimol_embeddings.pt']


def main():
    sp = pq.read_table(SPLIT_PQ, columns=['system_id', 'split']).to_pandas()
    report = {}
    for split in ['val', 'test']:
        ids = sorted(set(sp[sp['split'] == split]['system_id']))
        store = ROOT / split
        store.mkdir(parents=True, exist_ok=True)
        idset = set(ids)
        print(f'SPLIT {split}: native={len(ids)}', flush=True)

        # remove deprecated dirs (not in native split)
        removed = 0
        for d in list(store.iterdir()):
            if d.is_dir() and d.name not in idset:
                shutil.rmtree(d); removed += 1
        print(f'[{split}] removed deprecated dirs: {removed}', flush=True)

        # parse missing
        src_dirs = [SRC / i for i in ids if (SRC / i).is_dir()]
        print(f'STAGE_PARSE {split} ({len(src_dirs)} src)', flush=True)
        Parallel(n_jobs=-1)(delayed(parse_single_system)(sd, store, True)
                            for sd in tqdm(src_dirs, desc=f'parse:{split}'))

        # featurize all steps
        print(f'STAGE_FEAT {split}', flush=True)
        featurize_main(data_dir=str(store), n_jobs=-1, skip_existing=True,
                       cleanup_invalid=True, run_esm3=True, run_unimol=True,
                       unimol_size='1.1B', device='cuda', strict=False)

        dirs = [d.name for d in store.iterdir() if d.is_dir()]
        complete = [s for s in dirs if all((store / s / f).exists() for f in REQ)]
        report[split] = {'native': len(ids), 'removed_deprecated': removed,
                         'dirs': len(dirs), 'fully_featurized': len(complete)}
        print(f'DONE {split}: ' + json.dumps(report[split]), flush=True)

    (LOG / 'valtest_prep_summary.json').write_text(json.dumps(report, indent=2))
    print('VALTEST_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
