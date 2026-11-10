#!/usr/bin/env python3
"""Filter the full P–L–P catalog by date before choosing representatives."""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import build_plp_dataset as b


def main():
    csv.field_size_limit(sys.maxsize)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('tpd/full_db_20260630/all_complexes.tsv'))
    parser.add_argument('--set', choices=('plp', 'tpd'), default='plp')
    parser.add_argument('--date-field', choices=('deposition_date', 'release', 'last_revision_date'), default='deposition_date')
    parser.add_argument('--start', default='')
    parser.add_argument('--end', default='2026-06-30')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--ccd-root', default=str(b.t.CCD_ROOT))
    args = parser.parse_args()
    date.fromisoformat(args.end)
    if args.start:
        date.fromisoformat(args.start)
        if args.start > args.end: parser.error('--start must be on or before --end')
    if args.workers < 1: parser.error('--workers must be positive')
    with args.catalog.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    selected, audit = b.select_conformers(rows, args.set, args.date_field, args.start, args.end, args.workers)
    selected = [dict(r, conformer=audit[r['record']]['conformer']) for r in selected]
    if args.export_dir:
        selected = b.export_selected(selected, args.export_dir, args.workers, args.ccd_root)
        selected.sort(key=lambda r: (r['pdb_id'], r['record']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    b.t.write_tsv(args.out, selected)
    summary = {
        'catalog': str(args.catalog.resolve()), 'set': args.set,
        'date_field': args.date_field, 'start_inclusive': args.start, 'end_inclusive': args.end,
        'catalog_rows': len(rows), 'selected_groups': len(selected),
        'workers': args.workers, 'export_directory': str(args.export_dir) if args.export_dir else None,
        'export_errors': sum(r.get('export_status') == 'error' for r in selected),
        'note': 'Date filtering precedes sequence/ligand/pose deduplication. Selection is limited to entries present in the catalog.',
    }
    args.out.with_suffix('.selection.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary['export_errors']: raise SystemExit(1)


if __name__ == '__main__':
    main()
