"""Collect the step-count sweep (scripts/pb_steps_sweep.sh) into one table.

Re-applies eval_posebusters_redock.summarise to each root's per-pose table
(pb_metrics.csv, check columns from posebusters_redock_checks.json) and adds
the median per-system inference time from stats.jsonl.

    python scripts/pb_sweep_summary.py --sweep data/posebusters/sweep --steps 10 20 30 40 50 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_posebusters_redock import summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep', type=Path, default=Path('data/posebusters/sweep'))
    ap.add_argument('--steps', nargs='+', type=int, default=[10, 20, 30, 40, 50, 100])
    ap.add_argument('--selector', default=None, help='print only this selector (e.g. "min Vina")')
    args = ap.parse_args()

    frames = []
    for n in args.steps:
        root = args.sweep / f'steps_{n}'
        if not (root / 'pb_metrics.csv').exists():
            continue
        poses = pd.read_csv(root / 'pb_metrics.csv')
        checks = json.loads((root / 'posebusters_redock_checks.json').read_text())['columns']
        summary = summarise(poses, checks, f'steps_{n}')
        times = np.array([json.loads(l)['inference_s'] for l in open(root / 'stats.jsonl') if '"error"' not in l])
        summary.insert(0, 'steps', n)
        summary['inference_s_median'] = np.median(times)
        summary['inference_s_total'] = times.sum()
        frames.append(summary)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.sweep / 'summary.csv', index=False)
    if args.selector:
        df = df[df['selector'] == args.selector]
    cols = ['steps', 'selector', 'n_sys', 'inference_s_median', 'lig<2A_%', 'pb_valid_%',
            'success_%', 'sc<1A_%', 'sc_success_%']
    print(df[cols].sort_values(['selector', 'steps']).to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    print(f'\n-> {args.sweep / "summary.csv"}')


if __name__ == '__main__':
    main()
