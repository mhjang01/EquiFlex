'''
Metadata extraction script for bsite_unique dataset.

Processes the bsite_unique dataset which contains:
- ~62,000 binding site systems
- Each system has: ligand.sdf, protein.pdb
- Directory names: {pdb_id}_{chain_ids}

Pipeline:
    1. Parse protein from PDB file
    2. Extract ligand data from SDF file
    3. Extract pocket residues within cutoff of ligand
    4. Generate reference conformer coordinates

Usage:
    python exec/bsite_unique_extraction.py --data-dir /mnt/drive/bsite_unique --output-dir ./bsite_data
    python exec/bsite_unique_extraction.py --data-dir /mnt/drive/bsite_unique --output-dir ./bsite_data --n-jobs 32
'''

from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

from equiflex.prep.parse import (
    parse_pdb,
    extract_ligand_data,
    get_valid_ligands,
    get_pocket_residues,
    overwrite_coords_on_reference_data,
    get_sp2_torsion_indices,
)
from equiflex.prep.reference import generate_ligand_reference_data
from tqdm import tqdm
from joblib import Parallel, delayed, dump
import numpy as np


def add_ligand_reference_data(lig_data: Dict) -> Dict:
    '''
    Add reference conformer coordinates and torsion indices to ligand data.
    Uses RDKit to generate a reference conformer from SMILES.
    '''
    smiles = lig_data['smiles']
    atoms = lig_data['atoms']
    bonds = lig_data['bonds']

    ref_data = generate_ligand_reference_data(smiles)

    if ref_data is not None and ref_data['ref_coords'] is not None:
        lig_data['ref_coords'] = ref_data['ref_coords']
        lig_data['sp2_torsion_indices'] = ref_data['sp2_torsion_indices']
    else:
        n_atoms = len(atoms)
        lig_data['ref_coords'] = np.zeros((n_atoms, 3), dtype=np.float32)
        lig_data['sp2_torsion_indices'] = get_sp2_torsion_indices(atoms, bonds)

    return lig_data


def parse_single_system(system_dir: Path, storage_dir: Path, skip_existing: bool = True):
    '''
    Parse a single bsite_unique system.

    Args:
        system_dir: Path to system directory (e.g., /mnt/drive/bsite_unique/11ba_AB)
        storage_dir: Base output directory for processed data
        skip_existing: Skip if output already exists

    Returns:
        Tuple of (system_name, success, message)
    '''
    system_name = system_dir.name
    output_dir = storage_dir / system_name

    # Check if already processed
    if skip_existing and output_dir.exists():
        ligands_path = output_dir / 'ligands.pkl'
        pocket_path = output_dir / 'pocket_residues.pkl'
        if ligands_path.exists() and pocket_path.exists():
            return system_name, True, "skipped (exists)"

    output_dir.mkdir(parents=True, exist_ok=True)

    lig_file = system_dir / 'ligand.sdf'
    rec_file = system_dir / 'protein.pdb'

    # Check required files exist
    if not lig_file.exists():
        return system_name, False, "missing ligand.sdf"
    if not rec_file.exists():
        return system_name, False, "missing protein.pdb"

    try:
        # Extract ligand data from SDF
        lig_data = extract_ligand_data(lig_file)
        if lig_data is None:
            return system_name, False, "failed to parse ligand"

        # Add reference conformer and torsion indices
        lig_data = add_ligand_reference_data(lig_data)
        lig_data_list = [lig_data]

        # Get valid ligands (in case of multi-fragment ligands)
        valid_ligs = get_valid_ligands(lig_data_list)
        if len(valid_ligs) == 0:
            return system_name, False, "no valid ligands"

        # Parse protein structure
        parsed_data = parse_pdb(rec_file)
        if not parsed_data:
            return system_name, False, "failed to parse protein"

        # Get pocket residues
        pocket_data, sequence_data = get_pocket_residues(parsed_data, valid_ligs)
        if not pocket_data or all(len(v) == 0 for v in pocket_data.values()):
            return system_name, False, "no pocket residues"

        # Overwrite coordinates on reference data
        pocket_residues = overwrite_coords_on_reference_data(pocket_data)

        # Save processed data
        dump(valid_ligs, output_dir / 'ligands.pkl', compress=3)
        dump(pocket_residues, output_dir / 'pocket_residues.pkl', compress=3)
        dump(sequence_data, output_dir / 'sequence_data.pkl', compress=3)

        return system_name, True, "success"

    except Exception as e:
        return system_name, False, f"error: {e}"


def get_bsite_systems(data_dir: Path, limit: Optional[int] = None) -> List[Path]:
    '''Get list of bsite_unique system directories.'''
    system_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / 'ligand.sdf').exists() and (d / 'protein.pdb').exists()
    ])

    if limit:
        system_dirs = system_dirs[:limit]

    print(f'Found {len(system_dirs)} bsite_unique systems')
    return system_dirs


def main(
    data_dir: str = '/mnt/drive/bsite_unique',
    output_dir: str = '/home/mhjang/workspace/flex/bsite_data',
    n_jobs: int = -1,
    skip_existing: bool = True,
    limit: Optional[int] = None,
):
    data_path = Path(data_dir)
    storage_path = Path(output_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    # Get system directories
    system_dirs = get_bsite_systems(data_path, limit=limit)

    print(f"Processing {len(system_dirs)} systems")
    print(f"Output directory: {storage_path}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Skip existing: {skip_existing}")

    # Process systems in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(parse_single_system)(system_dir, storage_path, skip_existing)
        for system_dir in tqdm(system_dirs, desc="Extracting metadata")
    )

    # Summarize results
    success = sum(1 for _, s, m in results if s and 'success' in m)
    skipped = sum(1 for _, s, m in results if s and 'skipped' in m)
    failed = sum(1 for _, s, _ in results if not s)

    print(f"\nComplete: Success={success}, Skipped={skipped}, Failed={failed}")

    # Log failures
    failures = [(n, m) for n, s, m in results if not s]
    if failures:
        error_file = storage_path / 'extraction_errors.txt'
        with open(error_file, 'w') as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")
        print(f"Errors logged to: {error_file}")
        print(f"Sample failures: {failures[:5]}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract metadata from bsite_unique dataset')
    parser.add_argument('--data-dir', type=str, default='/mnt/drive/bsite_unique',
                        help='Path to bsite_unique data directory')
    parser.add_argument('--output-dir', type=str, default='/home/mhjang/workspace/flex/bsite_data',
                        help='Output directory for processed data')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Reprocess even if output exists')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of systems (for testing)')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        skip_existing=not args.no_skip,
        limit=args.limit,
    )
