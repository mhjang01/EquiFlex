"""Drop systems that already have results from a run_inference.py config.

A system counts as done when ``<output_dir>/results.pt`` exists, so an
interrupted batch can be restarted without re-sampling what finished.

    python scripts/filter_done_systems.py --config crossdock/run.shard0.yaml --in-place
"""

import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None, help='default: alongside, .todo.yaml')
    ap.add_argument('--in-place', action='store_true')
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    root = Path(config['output_root'])
    systems = config['systems']
    todo = {sid: spec for sid, spec in systems.items()
            if not (Path(spec.get('output_dir', root / sid)) / 'results.pt').exists()}
    config['systems'] = todo
    out = args.config if args.in_place else (args.out or args.config.with_suffix('.todo.yaml'))
    with out.open('w') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, width=120)
    print(f'{out}: {len(todo)}/{len(systems)} systems left')


if __name__ == '__main__':
    main()
