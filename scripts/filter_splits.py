"""
Filter splits.csv to remove systems flagged by scan_nan_data.py.

Reads nan_scan_report.txt, extracts all system IDs (lines under '=== Split ...'),
and rewrites splits.csv minus those rows. Original is backed up to splits.csv.bak.
"""
import re
import shutil
import argparse
from pathlib import Path


def parse_flagged_ids(report_path: Path) -> set[str]:
    ids = set()
    with open(report_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('===') or line.startswith('---'):
                continue
            # Each entry is "<system_id>\t{...detail...}"
            sid = line.split('\t', 1)[0].strip()
            if sid:
                ids.add(sid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='nan_scan_report.txt')
    ap.add_argument('--splits', default='data_v2_plinder/splits.csv')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    flagged = parse_flagged_ids(Path(args.report))
    print(f'Flagged systems: {len(flagged)}')

    splits_path = Path(args.splits)
    kept_lines = []
    removed = []
    with open(splits_path) as f:
        header = f.readline()
        kept_lines.append(header)
        for line in f:
            sid = line.split(',', 1)[0].strip()
            if sid in flagged:
                removed.append(line.rstrip('\n'))
            else:
                kept_lines.append(line)

    print(f'Removing {len(removed)} rows from {splits_path}')
    not_found = flagged - {ln.split(",", 1)[0].strip() for ln in [r for r in removed]}
    if not_found:
        print(f'WARNING: {len(not_found)} flagged IDs not in CSV: {sorted(not_found)[:5]}...')

    if args.dry_run:
        print('Dry run — no files written. Sample removed rows:')
        for r in removed[:5]:
            print(f'  {r}')
        return

    backup = splits_path.with_suffix('.csv.bak')
    shutil.copy(splits_path, backup)
    print(f'Backed up to {backup}')

    with open(splits_path, 'w') as f:
        f.writelines(kept_lines)
    print(f'Wrote {len(kept_lines)-1} rows (was {len(kept_lines)-1+len(removed)})')


if __name__ == '__main__':
    main()
