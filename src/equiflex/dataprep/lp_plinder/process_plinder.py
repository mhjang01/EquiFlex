"""
Process PLINDER dataset for Flex model training.

Reads protein-ligand structures from PLINDER's curated format and generates
featurized data compatible with the Flex model.

Pipeline:
    1. Parse protein PDB from PLINDER system directory
    2. Extract ligand from SDF files
    3. Extract pocket residues within cutoff of ligand
    4. Featurize atoms/bonds and generate embeddings

Usage:
    # Step 1: Parse structures from PLINDER data
    python exec/lp_plinder/process_plinder.py --step parse --plinder-dir /path/to/plinder --split train

    # Step 2: Featurize parsed data
    python exec/lp_plinder/process_plinder.py --step featurize --output-dir ./plinder_data

    # Run all steps
    python exec/lp_plinder/process_plinder.py --plinder-dir /path/to/plinder --output-dir ./plinder_data

    # Process specific split
    python exec/lp_plinder/process_plinder.py --plinder-dir /path/to/plinder --split train
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from joblib import dump, load, Parallel, delayed
import traceback
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

from equiflex.prep.parse import (
    parse_pdb,
    extract_atom_data,
    extract_bond_data,
    get_pocket_residues as get_pocket_residues_from_parse,
    overwrite_coords_on_reference_data,
)
from equiflex.prep.reference import generate_ligand_reference_data


def get_plinder_systems(plinder_dir: Path, split: str) -> List[Path]:
    """Get all system directories from a PLINDER split."""
    split_dir = plinder_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    systems = [d for d in split_dir.iterdir() if d.is_dir()]
    return sorted(systems)


def extract_ligand_from_sdf(sdf_path: Path) -> Optional[Dict]:
    """Extract ligand data from SDF file (with aligned coordinates)."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next(suppl, None)

    if mol is None:
        return None

    # Extract atom data with canonical naming
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i] + 1) for i in range(mol.GetNumAtoms())}
    atoms = [
        extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}')
        for atom in mol.GetAtoms()
    ]

    # Extract bond data
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]

    # Extract coordinates from SDF (already aligned to protein)
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer(0)
        coords = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=np.float32)
    else:
        return None

    return {
        'chain': 'LIG',
        'smiles': Chem.MolToSmiles(mol, canonical=True),
        'atoms': atoms,
        'bonds': bonds,
        'coords': coords,
        'mapping': mapping,
    }


def add_ligand_reference_data(lig_data: Dict) -> Dict:
    """Add reference conformer coordinates and torsion indices to ligand data."""
    smiles = lig_data['smiles']

    ref_data = generate_ligand_reference_data(smiles)

    if ref_data is not None and ref_data['ref_coords'] is not None:
        lig_data['ref_coords'] = ref_data['ref_coords']
        lig_data['sp2_torsion_indices'] = ref_data['sp2_torsion_indices']
        lig_data['constraint_matrix'] = ref_data['constraint_matrix']
    else:
        raise ValueError("Failed to generate ligand reference data")
        
    return lig_data


def parse_single_system(
    system_dir: Path,
    output_dir: Path,
    skip_existing: bool = True
) -> Tuple[str, bool, str]:
    """Parse a single PLINDER system."""
    system_id = system_dir.name
    out_system_dir = output_dir / system_id

    if skip_existing and out_system_dir.exists():
        ligands_path = out_system_dir / 'ligands.pkl'
        pocket_path = out_system_dir / 'pocket_residues.pkl'
        if ligands_path.exists() and pocket_path.exists():
            return system_id, True, "skipped (exists)"

    out_system_dir.mkdir(parents=True, exist_ok=True)

    # Get receptor PDB
    receptor_pdb = system_dir / 'receptor.pdb'
    if not receptor_pdb.exists():
        return system_id, False, "receptor.pdb not found"

    # Get ligand files
    ligand_dir = system_dir / 'ligand_files'
    if not ligand_dir.exists():
        return system_id, False, "ligand_files directory not found"

    sdf_files = list(ligand_dir.glob('*.sdf'))
    if not sdf_files:
        return system_id, False, "no SDF files found"

    # Parse protein
    try:
        parsed_data = parse_pdb(receptor_pdb)
    except Exception as e:
        return system_id, False, f"parse failed: {e}"

    if not parsed_data:
        return system_id, False, "empty structure"

    # Extract ligands from all SDF files
    ligands = []
    for sdf_path in sdf_files:
        try:
            lig_data = extract_ligand_from_sdf(sdf_path)
            if lig_data is not None:
                # Add reference data
                lig_data = add_ligand_reference_data(lig_data)
                lig_data['source_file'] = sdf_path.name
                ligands.append(lig_data)
        except Exception as e:
            # Skip ligands that fail to process
            continue

    if not ligands:
        return system_id, False, "no valid ligands"

    # Get pocket residues (cutoff is hardcoded in prep/parse.py)
    pocket_residues, sequence_data = get_pocket_residues_from_parse(
        parsed_data, ligands
    )

    if not pocket_residues:
        return system_id, False, "no pocket residues"

    # Overwrite pocket coords on AA reference templates
    pocket_residues = overwrite_coords_on_reference_data(pocket_residues)

    if not pocket_residues or all(len(v) == 0 for v in pocket_residues.values()):
        return system_id, False, "no valid pocket residues after template matching"

    # Save
    dump(ligands, out_system_dir / 'ligands.pkl', compress=3)
    dump(pocket_residues, out_system_dir / 'pocket_residues.pkl', compress=3)
    dump(sequence_data, out_system_dir / 'sequence_data.pkl', compress=3)

    # Copy chain mapping if exists
    chain_mapping_path = system_dir / 'chain_mapping.json'
    if chain_mapping_path.exists():
        with open(chain_mapping_path) as f:
            chain_mapping = json.load(f)
        with open(out_system_dir / 'chain_mapping.json', 'w') as f:
            json.dump(chain_mapping, f)

    return system_id, True, "success"


def run_step1_parsing(
    plinder_dir: Path,
    output_dir: Path,
    split: str,
    limit: Optional[int] = None,
    n_jobs: int = 1,
    skip_existing: bool = True,
):
    """Step 1: Parse structures from PLINDER data."""
    print("=" * 60)
    print(f"STEP 1: Parsing Structures ({split})")
    print("=" * 60)

    # Get all systems in split
    systems = get_plinder_systems(plinder_dir, split)
    print(f"Found {len(systems)} systems in {split} split")

    if limit:
        systems = systems[:limit]
        print(f"Limited to {len(systems)} systems")

    # Process systems
    results = []
    if n_jobs == 1:
        for system_dir in tqdm(systems, desc="Parsing"):
            try:
                result = parse_single_system(system_dir, output_dir, skip_existing)
                results.append(result)
            except Exception as e:
                tb = traceback.format_exc()
                results.append((system_dir.name, False, f"error: {e}\n{tb}"))
    else:
        def process_system(system_dir):
            try:
                return parse_single_system(system_dir, output_dir, skip_existing)
            except Exception as e:
                tb = traceback.format_exc()
                return (system_dir.name, False, f"error: {e}\n{tb}")

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_system)(system_dir)
            for system_dir in tqdm(systems, desc="Parsing")
        )

    # Summary
    success = sum(1 for _, s, m in results if s and 'success' in m)
    skipped = sum(1 for _, s, m in results if s and 'skipped' in m)
    failed = sum(1 for _, s, _ in results if not s)

    print(f"\nStep 1 Complete: Success={success}, Skipped={skipped}, Failed={failed}")

    # Save failures
    failures = [(n, m) for n, s, m in results if not s]
    if failures:
        error_file = output_dir / f'step1_errors_{split}.txt'
        with open(error_file, 'w') as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")
        print(f"Errors saved to {error_file}")


def run_step2_featurization(
    output_dir: Path,
    skip_existing: bool = True,
    skip_esm: bool = False,
    skip_unimol: bool = False,
    n_jobs: int = -1
):
    """Step 2: Featurize parsed data using exec/featurize_data.py logic."""
    print("\n" + "=" * 60)
    print("STEP 2: Featurization")
    print("=" * 60)

    # Import featurization functions
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from featurize_data import (
        run_step1_graph_featurization,
        run_step2_esm3_extraction,
        run_step3_unimol_extraction,
        FeaturizationStats
    )

    # Run graph featurization
    global_stats, valid_dirs = run_step1_graph_featurization(
        output_dir, n_jobs=n_jobs, cleanup_invalid=False, skip_existing=skip_existing
    )

    # Run ESM3 extraction
    if not skip_esm:
        valid_dirs = run_step2_esm3_extraction(
            output_dir, valid_dirs, device='cuda', skip_existing=skip_existing
        )

    # Run UniMol extraction
    if not skip_unimol:
        valid_dirs = run_step3_unimol_extraction(
            output_dir, valid_dirs, skip_existing=skip_existing
        )

    # Print statistics
    global_stats.print_summary()
    print(f"Final valid directories: {len(valid_dirs)}")


def run_cleanup(output_dir: Path, dry_run: bool = False):
    """Remove directories that failed parsing or featurization."""
    print("\n" + "=" * 60)
    print("CLEANUP: Removing Invalid Directories")
    print("=" * 60)

    import shutil

    # Required files for a valid processed directory
    required_files = [
        'ligands.pkl',
        'pocket_residues.pkl',
        'ligand_features.pt',
        'pocket_features.pt',
        'esm3_embeddings.pt',
        'unimol_embeddings.pt',
    ]

    # Find all system directories
    all_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'errors']
    print(f"Found {len(all_dirs)} directories")

    invalid_dirs = []
    valid_dirs = []

    for system_dir in all_dirs:
        missing_files = [f for f in required_files if not (system_dir / f).exists()]
        if missing_files:
            invalid_dirs.append((system_dir, missing_files))
        else:
            valid_dirs.append(system_dir)

    print(f"Valid directories: {len(valid_dirs)}")
    print(f"Invalid directories: {len(invalid_dirs)}")

    if not invalid_dirs:
        print("No directories to clean up.")
        return

    # Show some examples
    print("\nExamples of invalid directories:")
    for system_dir, missing in invalid_dirs[:5]:
        print(f"  {system_dir.name}: missing {missing}")
    if len(invalid_dirs) > 5:
        print(f"  ... and {len(invalid_dirs) - 5} more")

    if dry_run:
        print("\n[DRY RUN] Would remove these directories. Run without --dry-run to actually delete.")
        return

    # Actually remove
    print(f"\nRemoving {len(invalid_dirs)} directories...")
    for system_dir, _ in tqdm(invalid_dirs, desc="Cleaning"):
        try:
            shutil.rmtree(system_dir)
        except Exception as e:
            print(f"Failed to remove {system_dir}: {e}")

    print(f"Cleanup complete. Removed {len(invalid_dirs)} directories.")


def main():
    parser = argparse.ArgumentParser(description='Process PLINDER dataset for Flex model')
    parser.add_argument('--plinder-dir', type=str, default='/home/mhjang/workspace/plinder',
                        help='Root directory containing PLINDER data (with train/val subdirs)')
    parser.add_argument('--output-dir', type=str, default='./plinder_data',
                        help='Output directory for processed data')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val'],
                        help='Only process specific split (default: process train and val)')
    parser.add_argument('--step', type=str, default='all', choices=['parse', 'featurize', 'all', 'cleanup'],
                        help='Which step to run')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip entries that already exist')
    parser.add_argument('--skip-esm', action='store_true',
                        help='Skip ESM embedding generation')
    parser.add_argument('--skip-unimol', action='store_true',
                        help='Skip UniMol embedding generation')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entries per split (for testing)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for parsing')
    parser.add_argument('--dry-run', action='store_true',
                        help='For cleanup: show what would be deleted without actually deleting')
    args = parser.parse_args()

    # Setup paths
    plinder_dir = Path(args.plinder_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not plinder_dir.exists():
        raise FileNotFoundError(f"PLINDER directory not found: {plinder_dir}")

    # Determine which splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val']

    # Run steps
    if args.step in ['parse', 'all']:
        for split in splits:
            run_step1_parsing(
                plinder_dir=plinder_dir,
                output_dir=output_dir,
                split=split,
                limit=args.limit,
                n_jobs=args.n_jobs,
                skip_existing=args.skip_existing,
            )

    if args.step in ['featurize', 'all']:
        run_step2_featurization(
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            skip_esm=args.skip_esm,
            skip_unimol=args.skip_unimol,
            n_jobs=args.n_jobs if args.n_jobs > 0 else -1
        )

    if args.step == 'cleanup':
        run_cleanup(output_dir=output_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
