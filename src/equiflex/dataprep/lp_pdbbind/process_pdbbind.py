"""
Process LP-PDBBind dataset for Flex model training.

Uses local PDBBind structures and the same logic as exec/metadata_extraction.py
and exec/featurize_data.py.

Pipeline:
    1. Parse protein PDB from local PDBBind directory
    2. Extract ligand from SMILES
    3. Extract pocket residues within cutoff of ligand
    4. Featurize atoms/bonds and generate embeddings

Usage:
    # Step 1: Parse structures from local PDBBind data
    python scripts/lp_pdbbind/process_pdbbind.py --step parse --output-dir ./pdbbind

    # Step 2: Featurize parsed data
    python scripts/lp_pdbbind/process_pdbbind.py --step featurize --output-dir ./pdbbind

    # Run all steps
    python scripts/lp_pdbbind/process_pdbbind.py --output-dir ./pdbbind

    # Process specific split
    python scripts/lp_pdbbind/process_pdbbind.py --output-dir ./pdbbind --split train
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from joblib import dump, load, Parallel, delayed
import traceback
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem

# Metal ions to incorporate as ligand entities, with typical formal charge.
ION_CHARGE = {'NA': 1, 'K': 1, 'MG': 2, 'CA': 2, 'MN': 2, 'FE': 2, 'CO': 2,
              'NI': 2, 'CU': 2, 'ZN': 2, 'CD': 2, 'HG': 2, 'MO': 0, 'W': 0}


def extract_pocket_ions(protein_pdb: Path, lig_coords: np.ndarray,
                        cutoff: float = 5.0) -> List[Dict]:
    """Metal HETATM ions within `cutoff` of the ligand, as separate ligand entities.

    GT coords from the structure; ref_coords [0,0,0] (mirrors plinder/test_data).
    """
    try:
        st = gemmi.read_structure(str(protein_pdb))
    except Exception:
        return []
    ions = []
    seen = set()
    for model in st:
        for chain in model:
            for res in chain:
                # Skip multi-atom cofactors (HEM, FES, SF4, F3S, …).
                # A free ion is always a single-atom residue.
                res_atoms = list(res)
                if len(res_atoms) != 1:
                    continue
                at = res_atoms[0]
                el = at.element.name.upper()
                if el not in ION_CHARGE:
                    continue
                p = np.array([at.pos.x, at.pos.y, at.pos.z], dtype=np.float32)
                if np.linalg.norm(lig_coords - p, axis=1).min() > cutoff:
                    continue
                key = (el, tuple(np.round(p, 2)))
                if key in seen:
                    continue
                seen.add(key)
                    c = ION_CHARGE[el]
                    smi = f'[{el.capitalize()}+{c}]' if c > 0 else f'[{el.capitalize()}]'
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    ctx = compute_mol_context(mol)
                    atom = extract_atom_data(mol.GetAtomWithIdx(0), name=f'{el}1', mol_ctx=ctx)
                    ions.append({
                        'chain': el,
                        'smiles': Chem.MolToSmiles(mol),
                        'atoms': [atom],
                        'bonds': [],
                        'coords': p.reshape(1, 3),
                        'ref_coords': np.zeros((1, 3), dtype=np.float32),
                        'mapping': {0: '1'},
                        'sp2_torsion_indices': np.empty((0, 4), dtype=np.int32),
                        'constraint_matrix': np.zeros((1, 1), dtype=np.float32),
                    })
    return ions

from equiflex.prep.parse import (
    parse_pdb,
    extract_atom_data,
    extract_bond_data,
    compute_mol_context,
    get_sp2_torsion_indices,
    get_pocket_residues as get_pocket_residues_from_parse,
    overwrite_coords_on_reference_data,
)
from equiflex.prep.reference import generate_ligand_reference_data


# Constants
POCKET_DISTANCE_CUTOFF = 5.0


def get_local_structure(pdb_id: str, pdbbind_dir: Path) -> Optional[Path]:
    """Get local PDB structure from PDBBind directory."""
    pdb_id = pdb_id.lower()
    system_dir = pdbbind_dir / pdb_id

    # Try protein PDB first
    protein_pdb = system_dir / f"{pdb_id}_protein.pdb"
    if protein_pdb.exists():
        return protein_pdb

    # Fallback to pocket PDB
    pocket_pdb = system_dir / f"{pdb_id}_pocket.pdb"
    if pocket_pdb.exists():
        return pocket_pdb

    return None


def get_ligand_files(pdb_id: str, pdbbind_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Get local ligand MOL2 and SDF files from PDBBind directory."""
    pdb_id = pdb_id.lower()
    system_dir = pdbbind_dir / pdb_id

    mol2_path = system_dir / f"{pdb_id}_ligand.mol2"
    sdf_path = system_dir / f"{pdb_id}_ligand.sdf"

    return (
        mol2_path if mol2_path.exists() else None,
        sdf_path if sdf_path.exists() else None
    )


def extract_ligand_from_mol2(mol2_path: Path) -> Optional[Dict]:
    """Extract ligand data from MOL2 file (with aligned coordinates)."""
    mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)

    if mol is None:
        return None

    # Extract atom data with canonical naming
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i] + 1) for i in range(mol.GetNumAtoms())}
    mol_ctx = compute_mol_context(mol)
    atoms = [
        extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}', mol_ctx=mol_ctx)
        for atom in mol.GetAtoms()
    ]

    # Extract bond data
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]

    # Extract coordinates from MOL2 (already aligned to protein)
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


def extract_ligand_from_sdf(sdf_path: Path) -> Optional[Dict]:
    """Extract ligand data from SDF file (with aligned coordinates)."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next(suppl, None)

    if mol is None:
        return None

    # Extract atom data with canonical naming
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i] + 1) for i in range(mol.GetNumAtoms())}
    mol_ctx = compute_mol_context(mol)
    atoms = [
        extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}', mol_ctx=mol_ctx)
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


def extract_ligand(mol2_path: Optional[Path], sdf_path: Optional[Path]) -> Optional[Dict]:
    """Extract ligand data, trying MOL2 first then SDF as fallback."""
    # Try MOL2 first
    if mol2_path is not None:
        lig_data = extract_ligand_from_mol2(mol2_path)
        if lig_data is not None:
            return lig_data

    # Fallback to SDF
    if sdf_path is not None:
        lig_data = extract_ligand_from_sdf(sdf_path)
        if lig_data is not None:
            return lig_data

    return None


def extract_ligand_from_smiles(smiles: str) -> Optional[Dict]:
    """
    Extract ligand data from SMILES string.
    Follows the same format as extract_ligand_data() in prep/parse.py.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens for 3D generation, then remove
    mol_3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except:
        # Fallback: use zeros
        mol_3d = None

    # Work with non-H mol for features
    mol = Chem.RemoveHs(mol) if mol.GetNumAtoms() > 0 else mol

    # Extract atom data with canonical naming
    canonical_ranks = AllChem.CanonicalRankAtoms(mol)
    mapping = {i: str(canonical_ranks[i] + 1) for i in range(mol.GetNumAtoms())}
    mol_ctx = compute_mol_context(mol)
    atoms = [
        extract_atom_data(atom, f'{atom.GetSymbol().upper()}{mapping[atom.GetIdx()]}', mol_ctx=mol_ctx)
        for atom in mol.GetAtoms()
    ]

    # Extract bond data
    bonds = [extract_bond_data(bond) for bond in mol.GetBonds()]

    # Extract coordinates
    if mol_3d is not None and mol_3d.GetNumConformers() > 0:
        mol_3d_noH = Chem.RemoveHs(mol_3d)
        conf = mol_3d_noH.GetConformer(0)
        coords = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol_3d_noH.GetNumAtoms())
        ], dtype=np.float32)
    else:
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)

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
    atoms = lig_data['atoms']
    bonds = lig_data['bonds']

    ref_data = generate_ligand_reference_data(smiles)

    if ref_data is None or ref_data['ref_coords'] is None:
        raise ValueError("Failed to generate ligand reference data")

    n_atoms = len(atoms)
    if len(ref_data['ref_coords']) == n_atoms:
        lig_data['ref_coords'] = ref_data['ref_coords']
        lig_data['sp2_torsion_indices'] = ref_data['sp2_torsion_indices']
        lig_data['constraint_matrix'] = ref_data['constraint_matrix']
    else:
        # Canonical-SMILES rebuild gave a different atom count than the parsed
        # mol2/sdf ligand (salt/fragment/H divergence). Fall back to bound coords
        # so ref_coords/constraint stay aligned with atom_features (else the pair
        # distogram vs block_mask disagree and training crashes).
        lig_data['ref_coords'] = np.asarray(lig_data['coords'], dtype=np.float32)
        lig_data['sp2_torsion_indices'] = np.empty((0, 4), dtype=np.int32)
        lig_data['constraint_matrix'] = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        
    return lig_data


def get_pocket_residues(
    parsed_data: Dict,
    ligands: List[Dict],
    cutoff: float = POCKET_DISTANCE_CUTOFF
) -> Tuple[Dict, Dict]:
    """
    Extract pocket residues within cutoff distance of ligands.
    Returns (pocket_residues, sequence_data).
    """
    # Collect all ligand coordinates
    all_lig_coords = []
    for lig in ligands:
        if lig['coords'] is not None:
            all_lig_coords.append(lig['coords'])

    if not all_lig_coords:
        return {}, {}

    lig_coords = np.concatenate(all_lig_coords, axis=0)

    pocket_residues = {}
    sequence_data = {}

    for chain_id, chain_data in parsed_data.items():
        # Extract sequence
        seq_key = (None, 'sequence')
        if seq_key in chain_data:
            sequence_data[chain_id] = chain_data[seq_key]

        pocket_residues[chain_id] = {}

        for key, value in chain_data.items():
            if key[0] is None:  # Skip metadata
                continue

            resnum, aa_1_letter = key
            residue_data = value

            if 'atoms' not in residue_data or len(residue_data['atoms']) == 0:
                continue

            # Get residue coordinates
            res_coords = np.array([atom['coords'] for atom in residue_data['atoms']])

            # Check distance to ligand
            diff = res_coords[:, None, :] - lig_coords[None, :, :]
            distances = np.linalg.norm(diff, axis=2)
            min_dist = distances.min()

            if min_dist <= cutoff:
                entity_id = residue_data.get('entity_id', 1)
                pocket_residues[chain_id][(residue_data['resname'], resnum, entity_id)] = residue_data

    # Remove empty chains
    pocket_residues = {k: v for k, v in pocket_residues.items() if v}

    return pocket_residues, sequence_data


def parse_single_system(
    pdb_id: str,
    pdbbind_dir: Path,
    output_dir: Path,
    skip_existing: bool = True
) -> Tuple[str, bool, str]:
    """Parse a single PDBBind entry. SMILES is extracted from MOL2/SDF files."""
    pdb_id = pdb_id.lower()
    system_dir = output_dir / pdb_id

    if skip_existing and system_dir.exists():
        ligands_path = system_dir / 'ligands.pkl'
        pocket_path = system_dir / 'pocket_residues.pkl'
        if ligands_path.exists() and pocket_path.exists():
            return pdb_id, True, "skipped (exists)"

    system_dir.mkdir(parents=True, exist_ok=True)

    # Get local structure
    pdb_path = get_local_structure(pdb_id, pdbbind_dir)
    if pdb_path is None:
        return pdb_id, False, "structure not found"

    # Get ligand files (MOL2 and SDF)
    mol2_path, sdf_path = get_ligand_files(pdb_id, pdbbind_dir)
    if mol2_path is None and sdf_path is None:
        return pdb_id, False, "no ligand file found"

    # Parse protein
    try:
        parsed_data = parse_pdb(pdb_path)
    except Exception as e:
        return pdb_id, False, f"parse failed: {e}"

    if not parsed_data:
        return pdb_id, False, "empty structure"

    # Extract ligand (try MOL2 first, then SDF)
    lig_data = extract_ligand(mol2_path, sdf_path)
    if lig_data is None:
        return pdb_id, False, "failed to parse ligand"

    # Add reference data
    lig_data = add_ligand_reference_data(lig_data)
    ligands = [lig_data]

    # Append nearby metal ions as separate ligand entities (own uid / UniMol).
    ions = extract_pocket_ions(pdb_path, np.asarray(lig_data['coords']))
    ligands.extend(ions)

    # Get pocket residues (using prep/parse.py version which returns list format)
    pocket_residues, sequence_data = get_pocket_residues_from_parse(parsed_data, ligands)

    if not pocket_residues:
        return pdb_id, False, "no pocket residues"

    # Overwrite pocket coords on AA reference templates (provides proper atom features)
    pocket_residues = overwrite_coords_on_reference_data(pocket_residues)

    if not pocket_residues or all(len(v) == 0 for v in pocket_residues.values()):
        return pdb_id, False, "no valid pocket residues after template matching"

    # Save
    dump(ligands, system_dir / 'ligands.pkl', compress=3)
    dump(pocket_residues, system_dir / 'pocket_residues.pkl', compress=3)
    dump(sequence_data, system_dir / 'sequence_data.pkl', compress=3)

    return pdb_id, True, "success"


def run_step1_parsing(
    csv_path: str,
    output_dir: Path,
    pdbbind_dir: Path,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    n_jobs: int = 1,
    skip_existing: bool = True
):
    """Step 1: Parse structures from local PDBBind data."""
    print("=" * 60)
    print("STEP 1: Parsing Structures")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded {len(df)} entries from {csv_path}")

    # Filter by split
    if split:
        df = df[df['new_split'] == split]
        print(f"Filtered to {len(df)} entries in {split} split")

    if limit:
        df = df.head(limit)
        print(f"Limited to {len(df)} entries")

    # Process entries
    results = []
    if n_jobs == 1:
        for pdb_id, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
            try:
                result = parse_single_system(
                    pdb_id, pdbbind_dir, output_dir, skip_existing
                )
                results.append(result)
            except Exception as e:
                tb = traceback.format_exc()
                results.append((pdb_id, False, f"error: {e}\n{tb}"))
    else:
        def process_row(pdb_id, row):
            try:
                return parse_single_system(
                    pdb_id, pdbbind_dir, output_dir, skip_existing
                )
            except Exception as e:
                tb = traceback.format_exc()
                return (pdb_id, False, f"error: {e}\n{tb}")

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_row)(pdb_id, row)
            for pdb_id, row in tqdm(df.iterrows(), total=len(df), desc="Parsing")
        )

    # Summary
    success = sum(1 for _, s, m in results if s and 'success' in m)
    skipped = sum(1 for _, s, m in results if s and 'skipped' in m)
    failed = sum(1 for _, s, _ in results if not s)

    print(f"\nStep 1 Complete: Success={success}, Skipped={skipped}, Failed={failed}")

    # Save failures
    failures = [(n, m) for n, s, m in results if not s]
    if failures:
        with open(output_dir / 'step1_errors.txt', 'w') as f:
            for name, msg in failures:
                f.write(f"{name}: {msg}\n")


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
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'exec'))
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
    all_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
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
    parser = argparse.ArgumentParser(description='Process LP-PDBBind dataset')
    parser.add_argument('--csv', type=str, default='exec/lp_pdbbind/LP_PDBBind.csv',
                        help='Path to LP-PDBBind CSV file')
    parser.add_argument('--output-dir', type=str, default='./pdbbind',
                        help='Output directory for processed data')
    parser.add_argument('--pdbbind-dir', type=str, default='/mnt/drive/pdbbind/total',
                        help='Directory containing local PDBBind structures')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test'],
                        help='Only process entries from specific split')
    parser.add_argument('--step', type=str, default='all', choices=['parse', 'featurize', 'all', 'cleanup'],
                        help='Which step to run')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip entries that already exist')
    parser.add_argument('--skip-esm', action='store_true',
                        help='Skip ESM embedding generation')
    parser.add_argument('--skip-unimol', action='store_true',
                        help='Skip UniMol embedding generation')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entries (for testing)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for parsing')
    parser.add_argument('--dry-run', action='store_true',
                        help='For cleanup: show what would be deleted without actually deleting')
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdbbind_dir = Path(args.pdbbind_dir)
    if not pdbbind_dir.exists():
        raise FileNotFoundError(f"PDBBind directory not found: {pdbbind_dir}")

    # Run steps
    if args.step in ['parse', 'all']:
        run_step1_parsing(
            csv_path=args.csv,
            output_dir=output_dir,
            pdbbind_dir=pdbbind_dir,
            split=args.split,
            limit=args.limit,
            n_jobs=args.n_jobs,
            skip_existing=args.skip_existing
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
