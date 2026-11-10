"""Per-subset table from scripts/eval_posebusters_selected.py output on CrossDock.

System ids are ``<subset>__<case>`` (``Ensemble__CDK2__<case>`` for the ensemble
targets), so the subset is everything before the final ``__``, collapsed to its
first two fields. The denominator is every system in the run config, so a system
with no pose, no score or a PoseBusters error counts as a failure rather than
dropping out.

    python scripts/summarize_crossdock_selected.py \\
        --poses crossdock/results_v10_41k/vinardo_selected_poses.csv \\
        --config crossdock/inference_v10_41k_br10x4.v2.yaml \\
        --out crossdock/results_v10_41k/by_subset.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def subset_of(system_id: str) -> str:
    parts = system_id.split('__')
    return '__'.join(parts[:2]) if len(parts) > 2 else parts[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--poses', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    systems = list(yaml.safe_load(args.config.read_text())['systems'])
    frame = pd.read_csv(args.poses).drop_duplicates('system').set_index('system').reindex(systems)
    frame['subset'] = [subset_of(s) for s in frame.index]

    rows = []
    for subset, grp in list(frame.groupby('subset')) + [('All', frame)]:
        n = len(grp)
        rows.append({
            'subset': subset, 'systems': n,
            'scored': int(grp['rmsd'].notna().sum()),
            'lig_rmsd_median': round(grp['rmsd'].median(), 3),
            'lig<2A_%': round(100 * (grp['rmsd'] < 2).sum() / n, 2),
            'sc_rmsd_median': round(grp['rmsd_sc'].median(), 3),
            'sc<1A_%': round(100 * (grp['rmsd_sc'] < 1).sum() / n, 2),
            'pb_valid_%': round(100 * grp['pb_valid'].fillna(False).sum() / n, 2),
            'lig_success_pb_%': round(100 * grp['lig_success_pb'].fillna(False).sum() / n, 2),
            'sc_success_pb_%': round(100 * grp['sc_success_pb'].fillna(False).sum() / n, 2),
            'joint_success_pb_%': round(100 * grp['joint_success_pb'].fillna(False).sum() / n, 2),
        })
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
