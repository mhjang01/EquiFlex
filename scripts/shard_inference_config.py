"""Split a run_inference.py config into N shards that can run concurrently.

Each shard keeps the global sampling settings and a slice of ``systems``;
``output_dir`` is already absolute per system, so shards write into the same
output root without colliding. Systems are dealt round-robin so every shard
sees a mix of subsets and finishes at about the same time.

    python scripts/shard_inference_config.py --config crossdock/inference.yaml --shards 3
"""

import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--out-prefix', type=Path, default=None,
                    help='default: <config stem>.shard')
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    systems = config.pop('systems')
    ids = list(systems)
    prefix = args.out_prefix or args.config.with_suffix('').with_name(args.config.stem + '.shard')
    for shard in range(args.shards):
        shard_ids = ids[shard::args.shards]
        out = Path(f'{prefix}{shard}.yaml')
        with out.open('w') as handle:
            yaml.safe_dump({**config, 'systems': {i: systems[i] for i in shard_ids}},
                           handle, sort_keys=False, width=120)
        print(f'{out}: {len(shard_ids)} systems')


if __name__ == '__main__':
    main()
