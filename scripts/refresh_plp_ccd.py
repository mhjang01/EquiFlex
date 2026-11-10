#!/usr/bin/env python3
"""Reevaluate entries affected by newly cached CCDs and refresh dataset exports."""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import build_plp_dataset as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tpd/full_db_20260630'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--download-report', type=Path, action='append', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    if args.archive.exists(): parser.error('Use a new archive directory to preserve previous snapshots')
    csv.field_size_limit(sys.maxsize)
    reports = [json.loads(p.read_text()) for p in args.download_report]
    latest = {r['ccd']: r for report in reports for r in report['results']}
    changed = {ccd for ccd, r in latest.items() if r['status'] == 'validated'}
    if not changed: parser.error('Reports contain no successfully updated CCDs')
    args.archive.mkdir(parents=True)
    for p in args.out.rglob('*'):
        if p.is_file() and p.suffix in ('.tsv', '.json', '.md'):
            dst = args.archive / 'before' / p.relative_to(args.out)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    old_summary = json.loads((args.out/'summary.json').read_text())
    request = json.loads((args.out/'structure_request.json').read_text())
    results = [json.loads(line) for line in args.checkpoint.open()]
    ids = {r['pdb_id'] for r in results}; jobs = []
    for line in args.metadata.open():
        m = json.loads(line)
        if m.get('pdb_id') in ids and changed.intersection(m.get('plausible_ligands', [])):
            m['parse_cutoff'] = request['cutoff']; jobs.append(m)
    print(f'Rechecking {len(jobs)} entries affected by {len(changed)} validated CCD updates with {args.workers} workers', flush=True)
    updates = {}
    with Pool(args.workers, initializer=b.worker_init, initargs=(request['ccd_root'],)) as pool:
        for i, result in enumerate(pool.imap_unordered(b.evaluate_entry, jobs, chunksize=1), 1):
            if 'error' in result: raise RuntimeError(result)
            updates[result['pdb_id']] = result
            if i % 250 == 0: print(f'Rechecked {i}/{len(jobs)}', flush=True)
    checkpoint = args.archive/'structure_results.jsonl'
    with checkpoint.open('w') as f:
        for result in results: f.write(json.dumps(updates.get(result['pdb_id'], result))+'\n')
    links = []
    try:
        for name, target in [('metadata.jsonl', args.metadata), ('structure_results.jsonl', checkpoint)]:
            path = args.out/name
            if path.exists() or path.is_symlink(): raise ValueError(f'Checkpoint location already occupied: {path}')
            path.symlink_to(target.resolve()); links.append(path)
        settings = argparse.Namespace(pdb=json.loads((args.out/'scan_summary.json').read_text())['pdb_root'],
            out=str(args.out.resolve()), cutoff=request['cutoff'], workers=args.workers, resume=False,
            stage='export', keep_homomers=request['keep_homomers'], ccd_root=request['ccd_root'],
            tpd_overrides=str(b.TPD_OVERRIDES))
        b.export(settings, args.out)
    finally:
        for path in links: path.unlink()
    summary = json.loads((args.out/'summary.json').read_text())
    if summary['structural_errors'] or summary['export_errors']: raise RuntimeError('See error manifests before using refreshed dataset')
    def read(name):
        with (args.out/name).open() as f: return list(csv.DictReader(f, delimiter='\t'))
    plp = read('plp/nonredundant.tsv'); tpd = read('tpd/nonredundant.tsv'); recent = read('tpd/post_20240701.tsv')
    selected = {r['record']: r for r in plp + tpd + recent}
    for rows in (plp, tpd, recent):
        assert len(rows) == len({(r['complex_group'], r['conformer']) for r in rows})
        for row in rows:
            assert row['release'] <= request['cutoff'] and row['export_status'] == 'validated'
            for key in ('reference_cif', 'protein_cif', 'protein_pdb', 'ligand_sdf'):
                if row.get(key): assert Path(row[key]).exists(), (row['record'], key)
    assert all(r['homomeric'] == 'False' for r in tpd)
    assert all(json.loads(r['quality_flags']) == [] for r in plp)
    archived = []
    for path in (args.out/'reference').glob('*.cif'):
        if path.stem in selected: continue
        record = path.stem
        for directory, source in [('reference', path), ('ligands', args.out/'ligands'/f'{record}.sdf'), ('raw', args.out/'raw'/record)]:
            if source.exists():
                dst = args.archive/'superseded_exports'/directory/source.name
                dst.parent.mkdir(parents=True, exist_ok=True); source.rename(dst)
        archived.append(record)
    assert {p.stem for p in (args.out/'reference').glob('*.cif')} == set(selected)
    rows = read('all_complexes.tsv'); training = read('plp/entries.tsv')
    funnel = json.loads((args.out/'filter_funnel.json').read_text())
    funnel['contact_catalog'] = {'entries': len({r['pdb_id'] for r in rows}), 'entry_complexes':len(rows), 'unique_sequence_ligand_groups':len({r['complex_group'] for r in rows})}
    base = [r for r in rows if not set(json.loads(r['quality_flags'])) - set(json.loads(r['additional_chemistry_flags']))]
    funnel['before_additional_chemistry'] = {'entry_complexes':len(base), 'unique_groups':len({r['complex_group'] for r in base})}
    funnel['final'] = {'entries':len({r['pdb_id'] for r in training}), 'entry_complexes':len(training), 'unique_groups':len(plp)}
    funnel['overlapping_review_flags'] = dict(Counter(flag for r in rows for flag in json.loads(r['quality_flags'])).most_common())
    (args.out/'filter_funnel.json').write_text(json.dumps(funnel, indent=2))
    audit = {'workers':args.workers, 'ccd_requested':len(latest), 'ccd_validated':len(changed),
             'ccd_failures':[r for r in latest.values() if r['status'] != 'validated'],
             'entries_rechecked':len(jobs), 'previous_summary':old_summary, 'current_summary':summary,
             'latest_checkpoint':str(checkpoint.resolve()), 'superseded_exports_archived':archived,
             'checks':['complete checkpoint', 'unique groups', 'release cutoff', 'coordinate paths exist', 'TPD heteromeric', 'training quality flags clear', 'all exports independently validated', 'coordinate directory matches manifests']}
    (args.out/'ccd_refresh.json').write_text(json.dumps(audit, indent=2))
    (args.out/'validation.json').write_text(json.dumps(audit, indent=2))
    config = json.loads((args.out/'run_configuration.json').read_text())
    config['ccd_update_reports'] = [str(p.resolve()) for p in args.download_report]
    (args.out/'run_configuration.json').write_text(json.dumps(config, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('ccd_failures','previous_summary','current_summary','superseded_exports_archived')}, indent=2), flush=True)


if __name__ == '__main__': main()
