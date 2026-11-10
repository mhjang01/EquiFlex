#!/usr/bin/env python3
"""Fetch official CCD definitions and build trusted local RDKit references.

Run with pdbeccdutils installed, as used by Boltz's process/ccd.py.
"""
import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.request import urlopen


def update_component(job):
    component, destination = job
    out = Path(destination)
    url = f'https://files.rcsb.org/ligands/download/{component}.cif'
    audit = {'ccd': component, 'source': url}
    try:
        from pdbeccdutils.core import ccd_reader
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        cif = out / f'{component}.cif'
        if not cif.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=30) as response: data = response.read()
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(attempt + 1)
            temp = out / f'{component}.cif.tmp'
            temp.write_bytes(data)
            temp.replace(cif)
        result = ccd_reader.read_pdb_cif_file(str(cif))
        if result.errors: raise ValueError((component, result.errors))
        if result.component.id != component: raise ValueError('CCD identifier mismatch')
        mol = result.component.mol
        full_ccd_sanitized = result.sanitized
        if not full_ccd_sanitized:
            # The screen uses only heavy atoms. Validate this exact graph without
            # guessing charges or changing any heavy-atom bond orders.
            mol = Chem.RemoveAllHs(mol)
            Chem.SanitizeMol(mol)
        if not all(a.HasProp('name') for a in mol.GetAtoms()): raise ValueError('CCD atom names missing')
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        temp = out / f'{component}.pkl.tmp'
        temp.write_bytes(pickle.dumps(mol))
        temp.replace(out / f'{component}.pkl')
        audit.update(status='validated', sha256=hashlib.sha256(cif.read_bytes()).hexdigest(),
                     heavy_atoms=mol.GetNumHeavyAtoms(), warnings=result.warnings,
                     sanitized=True, full_ccd_sanitized=full_ccd_sanitized,
                     parser='pdbeccdutils.core.ccd_reader')
        (out / f'{component}.json').write_text(json.dumps(audit, indent=2))
    except Exception as exc:
        audit.update(status='error', error=str(exc))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('components', nargs='*')
    parser.add_argument('--review', type=Path, action='append', default=[], help='Review TSV or CSV; repeat for PLP and TPD')
    parser.add_argument('--base-ccd-root', type=Path, default=Path('/home/mhjang/workspace/boltz-tpd/boltz_cache/mols'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data/pdb_ccd_updates')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    csv.field_size_limit(sys.maxsize)
    components = {c.upper() for c in args.components}
    for path in args.review:
        with path.open() as handle:
            components.update(r['ccd'].upper() for r in csv.DictReader(handle, delimiter='\t' if path.suffix == '.tsv' else ','))
    if not components: parser.error('Supply component IDs or --review manifests')
    if any(not re.fullmatch(r'[A-Z0-9]+', c) for c in components): parser.error('Invalid component ID')
    pending = sorted(c for c in components if not (args.base_ccd_root / f'{c}.pkl').exists() and not (args.out / f'{c}.pkl').exists())
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{len(components)} unique components; {len(pending)} missing; {args.workers} workers', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(update_component, ((c, str(args.out)) for c in pending), chunksize=1), 1):
            results.append(result)
            if i % 50 == 0 or result['status'] == 'error': print(f'{i}/{len(pending)} {result}', flush=True)
    report = {'workers': args.workers, 'review_manifests': [str(p.resolve()) for p in args.review],
              'components_requested': len(components), 'already_present': len(components)-len(pending),
              'missing_requested': len(pending), 'validated': sum(r['status']=='validated' for r in results),
              'errors': sum(r['status']=='error' for r in results), 'results': results}
    destination = args.report or args.out / 'download_report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    if report['errors']: raise SystemExit(1)


if __name__ == '__main__': main()
