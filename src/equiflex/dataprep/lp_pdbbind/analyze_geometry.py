"""
Detailed geometry analysis for PDBBind data.

Checks for:
1. Atom clashes (distance < 0.01 Å) - likely duplicate/overlapping atoms
2. Bond length violations (too short < 0.8 Å or too long > 2.5 Å)
3. Unreasonable distances (atoms too far apart in supposedly bonded molecules)
4. NaN/Inf coordinates
5. Zero coordinates (all atoms at origin)
6. Coordinate range issues (atoms too far from centroid)

Usage:
    python exec/lp_pdbbind/analyze_geometry.py --data-dir ./pdbbind_data
    python exec/lp_pdbbind/analyze_geometry.py --data-dir ./pdbbind_data --verbose
    python exec/lp_pdbbind/analyze_geometry.py --data-dir ./pdbbind_data --output-csv geometry_issues.csv
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import json


@dataclass
class GeometryIssue:
    """Represents a geometry issue found in the data."""
    system_id: str
    component: str  # 'ligand' or 'pocket'
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    details: str
    value: Optional[float] = None


@dataclass
class GeometryStats:
    """Statistics for a single system."""
    system_id: str

    # Ligand stats
    lig_n_atoms: int = 0
    lig_n_bonds: int = 0
    lig_min_dist: float = float('inf')
    lig_max_dist: float = 0.0
    lig_min_bond_length: float = float('inf')
    lig_max_bond_length: float = 0.0
    lig_mean_bond_length: float = 0.0
    lig_coord_range: float = 0.0
    lig_n_clashes: int = 0
    lig_n_short_bonds: int = 0
    lig_n_long_bonds: int = 0

    # Pocket stats
    pock_n_atoms: int = 0
    pock_n_bonds: int = 0
    pock_min_dist: float = float('inf')
    pock_max_dist: float = 0.0
    pock_min_bond_length: float = float('inf')
    pock_max_bond_length: float = 0.0
    pock_mean_bond_length: float = 0.0
    pock_coord_range: float = 0.0
    pock_n_clashes: int = 0
    pock_n_short_bonds: int = 0
    pock_n_long_bonds: int = 0

    # Issues
    issues: List[GeometryIssue] = field(default_factory=list)


# Thresholds
CLASH_THRESHOLD = 0.01  # Å - atoms closer than this are likely duplicates
MIN_BOND_LENGTH = 0.8   # Å - bonds shorter than this are suspicious
MAX_BOND_LENGTH = 2.5   # Å - bonds longer than this are suspicious
MAX_COORD_RANGE = 100.0 # Å - max distance from centroid
VERY_SHORT_BOND = 0.5   # Å - critically short bond
VERY_LONG_BOND = 3.5    # Å - critically long bond


def compute_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrix."""
    return torch.cdist(coords, coords)


def get_bond_mask(bond_features: torch.Tensor) -> torch.Tensor:
    """Extract bond mask from bond features tensor."""
    # bond_features: [N, N, 26] - non-zero features indicate a bond
    return (bond_features.abs().sum(dim=-1) > 0).float()


def analyze_coordinates(
    coords: torch.Tensor,
    component: str,
    system_id: str,
) -> List[GeometryIssue]:
    """Check for coordinate-level issues."""
    issues = []

    # Check for NaN
    if torch.isnan(coords).any():
        n_nan = torch.isnan(coords).sum().item()
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='nan_coordinates',
            severity='error',
            details=f'{n_nan} NaN values in coordinates',
            value=n_nan
        ))

    # Check for Inf
    if torch.isinf(coords).any():
        n_inf = torch.isinf(coords).sum().item()
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='inf_coordinates',
            severity='error',
            details=f'{n_inf} Inf values in coordinates',
            value=n_inf
        ))

    # Check for all-zero coordinates
    if coords.abs().sum() == 0:
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='all_zero_coordinates',
            severity='error',
            details='All coordinates are zero',
        ))

    # Check coordinate range (atoms too far from centroid)
    if coords.abs().sum() > 0 and not torch.isnan(coords).any():
        centroid = coords.mean(dim=0)
        distances_from_centroid = torch.norm(coords - centroid, dim=1)
        max_dist = distances_from_centroid.max().item()

        if max_dist > MAX_COORD_RANGE:
            issues.append(GeometryIssue(
                system_id=system_id,
                component=component,
                issue_type='coordinate_range_exceeded',
                severity='warning',
                details=f'Max distance from centroid: {max_dist:.2f} Å',
                value=max_dist
            ))

    return issues


def analyze_clashes(
    coords: torch.Tensor,
    distances: torch.Tensor,
    component: str,
    system_id: str,
) -> Tuple[int, List[GeometryIssue]]:
    """Check for atom clashes (overlapping atoms)."""
    issues = []
    n_atoms = coords.shape[0]

    # Mask out diagonal
    mask = 1 - torch.eye(n_atoms, device=coords.device)
    masked_distances = distances + (1 - mask) * 1000  # Large value for diagonal

    # Find clashes
    clash_mask = (masked_distances < CLASH_THRESHOLD) & (mask > 0)
    n_clashes = clash_mask.sum().item() // 2  # Divide by 2 for symmetry

    if n_clashes > 0:
        # Find specific clash pairs
        clash_pairs = torch.where(torch.triu(clash_mask, diagonal=1))
        clash_distances = masked_distances[clash_pairs].tolist()

        severity = 'error' if n_clashes > 5 else 'warning'
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='atom_clashes',
            severity=severity,
            details=f'{n_clashes} atom pairs with distance < {CLASH_THRESHOLD} Å',
            value=n_clashes
        ))

        # Log first few clash details
        if n_clashes <= 5:
            for i in range(min(n_clashes, 5)):
                idx1, idx2 = clash_pairs[0][i].item(), clash_pairs[1][i].item()
                dist = clash_distances[i]
                issues.append(GeometryIssue(
                    system_id=system_id,
                    component=component,
                    issue_type='atom_clash_detail',
                    severity='info',
                    details=f'Atoms {idx1}-{idx2}: {dist:.6f} Å',
                    value=dist
                ))

    return n_clashes, issues


def analyze_bonds(
    coords: torch.Tensor,
    distances: torch.Tensor,
    bond_mask: torch.Tensor,
    component: str,
    system_id: str,
) -> Tuple[Dict, List[GeometryIssue]]:
    """Analyze bond lengths."""
    issues = []
    stats = {
        'n_bonds': 0,
        'min_length': float('inf'),
        'max_length': 0.0,
        'mean_length': 0.0,
        'n_short': 0,
        'n_long': 0,
    }

    # Get upper triangle of bond mask to avoid double counting
    n_atoms = coords.shape[0]
    triu_mask = torch.triu(torch.ones(n_atoms, n_atoms, device=coords.device), diagonal=1)
    bond_mask_unique = bond_mask * triu_mask

    # Get bond distances
    bond_indices = torch.where(bond_mask_unique > 0)
    if len(bond_indices[0]) == 0:
        return stats, issues

    bond_distances = distances[bond_indices].tolist()
    stats['n_bonds'] = len(bond_distances)
    stats['min_length'] = min(bond_distances)
    stats['max_length'] = max(bond_distances)
    stats['mean_length'] = np.mean(bond_distances)

    # Check for short bonds
    short_bonds = [(i, j, d) for i, j, d in zip(bond_indices[0].tolist(),
                                                  bond_indices[1].tolist(),
                                                  bond_distances) if d < MIN_BOND_LENGTH]
    stats['n_short'] = len(short_bonds)

    if short_bonds:
        very_short = [b for b in short_bonds if b[2] < VERY_SHORT_BOND]
        severity = 'error' if very_short else 'warning'
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='short_bonds',
            severity=severity,
            details=f'{len(short_bonds)} bonds shorter than {MIN_BOND_LENGTH} Å (min: {min(b[2] for b in short_bonds):.3f} Å)',
            value=len(short_bonds)
        ))

        # Log details for very short bonds
        for i, j, d in very_short[:5]:
            issues.append(GeometryIssue(
                system_id=system_id,
                component=component,
                issue_type='short_bond_detail',
                severity='info',
                details=f'Bond {i}-{j}: {d:.4f} Å',
                value=d
            ))

    # Check for long bonds
    long_bonds = [(i, j, d) for i, j, d in zip(bond_indices[0].tolist(),
                                                 bond_indices[1].tolist(),
                                                 bond_distances) if d > MAX_BOND_LENGTH]
    stats['n_long'] = len(long_bonds)

    if long_bonds:
        very_long = [b for b in long_bonds if b[2] > VERY_LONG_BOND]
        severity = 'error' if very_long else 'warning'
        issues.append(GeometryIssue(
            system_id=system_id,
            component=component,
            issue_type='long_bonds',
            severity=severity,
            details=f'{len(long_bonds)} bonds longer than {MAX_BOND_LENGTH} Å (max: {max(b[2] for b in long_bonds):.3f} Å)',
            value=len(long_bonds)
        ))

        # Log details for very long bonds
        for i, j, d in very_long[:5]:
            issues.append(GeometryIssue(
                system_id=system_id,
                component=component,
                issue_type='long_bond_detail',
                severity='info',
                details=f'Bond {i}-{j}: {d:.4f} Å',
                value=d
            ))

    return stats, issues


def analyze_system(system_dir: Path) -> Optional[GeometryStats]:
    """Analyze geometry of a single system."""
    system_id = system_dir.name
    stats = GeometryStats(system_id=system_id)

    # Load features
    try:
        ligand_features = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
        pocket_features = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    except Exception as e:
        stats.issues.append(GeometryIssue(
            system_id=system_id,
            component='system',
            issue_type='load_error',
            severity='error',
            details=str(e)[:100]
        ))
        return stats

    # Analyze ligand
    lig_coords = ligand_features.get('coords')
    lig_bonds = ligand_features.get('bond_features')

    if lig_coords is not None:
        stats.lig_n_atoms = lig_coords.shape[0]

        # Coordinate issues
        stats.issues.extend(analyze_coordinates(lig_coords, 'ligand', system_id))

        if not torch.isnan(lig_coords).any() and lig_coords.abs().sum() > 0:
            lig_distances = compute_distances(lig_coords)

            # Distance stats
            n_atoms = lig_coords.shape[0]
            mask = 1 - torch.eye(n_atoms)
            masked_dist = lig_distances + (1 - mask) * 1e6
            stats.lig_min_dist = masked_dist.min().item()
            stats.lig_max_dist = lig_distances.max().item()

            # Centroid range
            centroid = lig_coords.mean(dim=0)
            stats.lig_coord_range = torch.norm(lig_coords - centroid, dim=1).max().item()

            # Clash analysis
            n_clashes, clash_issues = analyze_clashes(lig_coords, lig_distances, 'ligand', system_id)
            stats.lig_n_clashes = n_clashes
            stats.issues.extend(clash_issues)

            # Bond analysis
            if lig_bonds is not None:
                bond_mask = get_bond_mask(lig_bonds)
                bond_stats, bond_issues = analyze_bonds(lig_coords, lig_distances, bond_mask, 'ligand', system_id)
                stats.lig_n_bonds = bond_stats['n_bonds']
                stats.lig_min_bond_length = bond_stats['min_length']
                stats.lig_max_bond_length = bond_stats['max_length']
                stats.lig_mean_bond_length = bond_stats['mean_length']
                stats.lig_n_short_bonds = bond_stats['n_short']
                stats.lig_n_long_bonds = bond_stats['n_long']
                stats.issues.extend(bond_issues)

    # Analyze pocket
    pock_coords = pocket_features.get('coords')
    pock_bonds = pocket_features.get('bond_features')

    if pock_coords is not None:
        stats.pock_n_atoms = pock_coords.shape[0]

        # Coordinate issues
        stats.issues.extend(analyze_coordinates(pock_coords, 'pocket', system_id))

        if not torch.isnan(pock_coords).any() and pock_coords.abs().sum() > 0:
            pock_distances = compute_distances(pock_coords)

            # Distance stats
            n_atoms = pock_coords.shape[0]
            mask = 1 - torch.eye(n_atoms)
            masked_dist = pock_distances + (1 - mask) * 1e6
            stats.pock_min_dist = masked_dist.min().item()
            stats.pock_max_dist = pock_distances.max().item()

            # Centroid range
            centroid = pock_coords.mean(dim=0)
            stats.pock_coord_range = torch.norm(pock_coords - centroid, dim=1).max().item()

            # Clash analysis
            n_clashes, clash_issues = analyze_clashes(pock_coords, pock_distances, 'pocket', system_id)
            stats.pock_n_clashes = n_clashes
            stats.issues.extend(clash_issues)

            # Bond analysis
            if pock_bonds is not None:
                bond_mask = get_bond_mask(pock_bonds)
                bond_stats, bond_issues = analyze_bonds(pock_coords, pock_distances, bond_mask, 'pocket', system_id)
                stats.pock_n_bonds = bond_stats['n_bonds']
                stats.pock_min_bond_length = bond_stats['min_length']
                stats.pock_max_bond_length = bond_stats['max_length']
                stats.pock_mean_bond_length = bond_stats['mean_length']
                stats.pock_n_short_bonds = bond_stats['n_short']
                stats.pock_n_long_bonds = bond_stats['n_long']
                stats.issues.extend(bond_issues)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze geometry of PDBBind data')
    parser.add_argument('--data-dir', type=str, default='./pdbbind_data',
                        help='Directory containing processed PDBBind data')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV file for detailed results')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output JSON file for issues')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed issues for each system')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of systems to analyze')
    parser.add_argument('--move-errors', action='store_true',
                        help='Move systems with errors to data-dir/errors/')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be moved without actually moving')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all system directories
    all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name != 'errors'])
    if args.limit:
        all_dirs = all_dirs[:args.limit]

    print(f"Analyzing geometry of {len(all_dirs)} systems...")

    # Analyze each system
    all_stats = []
    issue_counts = defaultdict(int)
    systems_with_errors = []
    systems_with_warnings = []

    for system_dir in tqdm(all_dirs, desc="Analyzing"):
        stats = analyze_system(system_dir)
        if stats:
            all_stats.append(stats)

            # Count issues
            has_error = False
            has_warning = False
            for issue in stats.issues:
                issue_counts[issue.issue_type] += 1
                if issue.severity == 'error':
                    has_error = True
                elif issue.severity == 'warning':
                    has_warning = True

            if has_error:
                systems_with_errors.append(stats.system_id)
            elif has_warning:
                systems_with_warnings.append(stats.system_id)

            # Print verbose output
            if args.verbose and stats.issues:
                error_issues = [i for i in stats.issues if i.severity == 'error']
                warning_issues = [i for i in stats.issues if i.severity == 'warning']
                if error_issues or warning_issues:
                    print(f"\n{stats.system_id}:")
                    for issue in error_issues + warning_issues:
                        print(f"  [{issue.severity.upper()}] {issue.component}: {issue.issue_type} - {issue.details}")

    # Summary
    print("\n" + "=" * 70)
    print("GEOMETRY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total systems analyzed: {len(all_stats)}")
    print(f"Systems with errors: {len(systems_with_errors)}")
    print(f"Systems with warnings: {len(systems_with_warnings)}")
    print(f"Systems with no issues: {len(all_stats) - len(systems_with_errors) - len(systems_with_warnings)}")

    # Issue breakdown
    print("\n--- Issue Counts ---")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}")

    # Bond length statistics
    lig_bond_lengths = [s.lig_mean_bond_length for s in all_stats if s.lig_n_bonds > 0]
    pock_bond_lengths = [s.pock_mean_bond_length for s in all_stats if s.pock_n_bonds > 0]

    if lig_bond_lengths:
        print("\n--- Ligand Bond Length Statistics ---")
        print(f"  Mean of means: {np.mean(lig_bond_lengths):.3f} Å")
        print(f"  Std of means: {np.std(lig_bond_lengths):.3f} Å")
        min_bond_all = min(s.lig_min_bond_length for s in all_stats if s.lig_n_bonds > 0)
        max_bond_all = max(s.lig_max_bond_length for s in all_stats if s.lig_n_bonds > 0)
        print(f"  Global min: {min_bond_all:.3f} Å")
        print(f"  Global max: {max_bond_all:.3f} Å")

    if pock_bond_lengths:
        print("\n--- Pocket Bond Length Statistics ---")
        print(f"  Mean of means: {np.mean(pock_bond_lengths):.3f} Å")
        print(f"  Std of means: {np.std(pock_bond_lengths):.3f} Å")
        min_bond_all = min(s.pock_min_bond_length for s in all_stats if s.pock_n_bonds > 0)
        max_bond_all = max(s.pock_max_bond_length for s in all_stats if s.pock_n_bonds > 0)
        print(f"  Global min: {min_bond_all:.3f} Å")
        print(f"  Global max: {max_bond_all:.3f} Å")

    # Clash statistics
    lig_clashes = [s.lig_n_clashes for s in all_stats]
    pock_clashes = [s.pock_n_clashes for s in all_stats]

    print("\n--- Clash Statistics ---")
    print(f"  Ligand systems with clashes: {sum(1 for c in lig_clashes if c > 0)}")
    print(f"  Pocket systems with clashes: {sum(1 for c in pock_clashes if c > 0)}")
    print(f"  Total ligand clashes: {sum(lig_clashes)}")
    print(f"  Total pocket clashes: {sum(pock_clashes)}")

    # List systems with errors
    if systems_with_errors:
        print("\n--- Systems with Errors ---")
        for sid in systems_with_errors[:20]:
            print(f"  {sid}")
        if len(systems_with_errors) > 20:
            print(f"  ... and {len(systems_with_errors) - 20} more")

    # Save to CSV
    if args.output_csv:
        import csv
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'system_id',
                'lig_n_atoms', 'lig_n_bonds', 'lig_min_dist', 'lig_max_dist',
                'lig_min_bond', 'lig_max_bond', 'lig_mean_bond',
                'lig_n_clashes', 'lig_n_short_bonds', 'lig_n_long_bonds',
                'pock_n_atoms', 'pock_n_bonds', 'pock_min_dist', 'pock_max_dist',
                'pock_min_bond', 'pock_max_bond', 'pock_mean_bond',
                'pock_n_clashes', 'pock_n_short_bonds', 'pock_n_long_bonds',
                'n_errors', 'n_warnings'
            ])

            for s in all_stats:
                n_errors = sum(1 for i in s.issues if i.severity == 'error')
                n_warnings = sum(1 for i in s.issues if i.severity == 'warning')
                writer.writerow([
                    s.system_id,
                    s.lig_n_atoms, s.lig_n_bonds,
                    f'{s.lig_min_dist:.4f}' if s.lig_min_dist != float('inf') else '',
                    f'{s.lig_max_dist:.4f}' if s.lig_max_dist > 0 else '',
                    f'{s.lig_min_bond_length:.4f}' if s.lig_min_bond_length != float('inf') else '',
                    f'{s.lig_max_bond_length:.4f}' if s.lig_max_bond_length > 0 else '',
                    f'{s.lig_mean_bond_length:.4f}' if s.lig_mean_bond_length > 0 else '',
                    s.lig_n_clashes, s.lig_n_short_bonds, s.lig_n_long_bonds,
                    s.pock_n_atoms, s.pock_n_bonds,
                    f'{s.pock_min_dist:.4f}' if s.pock_min_dist != float('inf') else '',
                    f'{s.pock_max_dist:.4f}' if s.pock_max_dist > 0 else '',
                    f'{s.pock_min_bond_length:.4f}' if s.pock_min_bond_length != float('inf') else '',
                    f'{s.pock_max_bond_length:.4f}' if s.pock_max_bond_length > 0 else '',
                    f'{s.pock_mean_bond_length:.4f}' if s.pock_mean_bond_length > 0 else '',
                    s.pock_n_clashes, s.pock_n_short_bonds, s.pock_n_long_bonds,
                    n_errors, n_warnings
                ])

        print(f"\nSaved statistics to {output_path}")

    # Save issues to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        issues_data = {}
        for s in all_stats:
            if s.issues:
                issues_data[s.system_id] = [
                    {
                        'component': i.component,
                        'type': i.issue_type,
                        'severity': i.severity,
                        'details': i.details,
                        'value': i.value
                    }
                    for i in s.issues
                    if i.severity in ['error', 'warning']
                ]

        with open(output_path, 'w') as f:
            json.dump(issues_data, f, indent=2)

        print(f"Saved issues to {output_path}")

    # Move error systems to errors directory
    if args.move_errors and systems_with_errors:
        error_dir = data_dir / 'errors'

        if args.dry_run:
            print(f"\n[DRY RUN] Would move {len(systems_with_errors)} systems to {error_dir}")
            for sid in systems_with_errors[:10]:
                print(f"  {sid}")
            if len(systems_with_errors) > 10:
                print(f"  ... and {len(systems_with_errors) - 10} more")
        else:
            error_dir.mkdir(exist_ok=True)
            moved = 0
            failed = 0

            print(f"\nMoving {len(systems_with_errors)} systems with errors to {error_dir}...")
            for sid in tqdm(systems_with_errors, desc="Moving"):
                src = data_dir / sid
                dst = error_dir / sid

                if not src.exists():
                    continue

                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as e:
                    print(f"Failed to move {sid}: {e}")
                    failed += 1

            print(f"Moved {moved} systems to {error_dir}")
            if failed > 0:
                print(f"Failed to move {failed} systems")

            remaining = len(all_dirs) - moved
            print(f"{remaining} valid systems remain in {data_dir}")


if __name__ == '__main__':
    main()
