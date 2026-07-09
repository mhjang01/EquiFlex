'''
Plot rotational equivariance test results for a given system.
Tests rotations at 30-degree intervals for x, y, and z axes.

Uses flow matching perspective:
- Backbone atoms (N, CA, C, O) are fixed at ground truth
- Sidechain and ligand atoms are noised
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import sys
sys.path.append('/home/mhjang/workspace/flex')

from equiflex.model.models import FlexModelV4, FlexConfig
from equiflex.model.loader import FlexDataset
from equiflex.model.config import load_config


def rotation_matrix_axis(axis: str, angle_deg: float, device: torch.device) -> torch.Tensor:
    '''Create rotation matrix for given axis and angle in degrees.'''
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)

    if axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=torch.float32, device=device)
    elif axis == 'y':
        R = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32, device=device)
    elif axis == 'z':
        R = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return R


def rotate_coords(coords: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    '''Rotate coordinates by rotation matrix R.'''
    return torch.einsum('ij, ...j -> ...i', R, coords)


def test_equivariance_single_rotation(
    model: FlexModelV4,
    batch: dict,
    R: torch.Tensor,
    device: torch.device,
) -> dict:
    '''
    Test SO(3) equivariance for a single rotation.

    Tests: R @ f(x) == f(R @ x)
    where x includes both x_t (noisy coords) and ref_coords.

    The batch should contain flow-matching prepared coordinates where:
    - Backbone atoms are at/near ground truth
    - Sidechain and ligand atoms are interpolated between noise and GT
    '''
    model.eval()

    with torch.no_grad():

        # First centralize coordinates by batches
        rec_coords = batch['rec_atom_coords'] - batch['rec_atom_coords'].mean(dim=1, keepdim=True)
        lig_coords = batch['lig_atom_coords'] - batch['lig_atom_coords'].mean(dim=1, keepdim=True)

        # Original forward pass
        output_orig = model(
            rec_atom_feats=batch['rec_atom_feats'],
            rec_atom_coords=rec_coords,
            rec_atom_ref_coords=batch['rec_atom_ref_coords'],
            rec_constraint_matrix=batch['rec_constraint_matrix'],
            rec_bond_feats=batch['rec_bond_feats'],
            rec_uid_vec=batch['rec_uid_vec'],
            rec_atom_mask=batch['rec_atom_mask'],
            rec_backbone_mask=batch['rec_backbone_mask'],
            esm_embeddings=batch['esm_embeddings'],
            lig_atom_feats=batch['lig_atom_feats'],
            lig_atom_coords=lig_coords,
            lig_atom_ref_coords=batch['lig_atom_ref_coords'],
            lig_constraint_matrix=batch['lig_constraint_matrix'],
            lig_bond_feats=batch['lig_bond_feats'],
            lig_uid_vec=batch['lig_uid_vec'],
            lig_atom_mask=batch['lig_atom_mask'],
            unimol_embeddings=batch['unimol_embeddings'],
            t=batch['t'],
        )

        # Rotated input forward pass
        rec_coords_rot = rotate_coords(rec_coords, R)
        lig_coords_rot = rotate_coords(lig_coords, R)

        output_rot = model(
            rec_atom_feats=batch['rec_atom_feats'],
            rec_atom_coords=rec_coords_rot,
            rec_atom_ref_coords=batch['rec_atom_ref_coords'],
            rec_constraint_matrix=batch['rec_constraint_matrix'],
            rec_bond_feats=batch['rec_bond_feats'],
            rec_uid_vec=batch['rec_uid_vec'],
            rec_atom_mask=batch['rec_atom_mask'],
            rec_backbone_mask=batch['rec_backbone_mask'],
            esm_embeddings=batch['esm_embeddings'],
            lig_atom_feats=batch['lig_atom_feats'],
            lig_atom_coords=lig_coords_rot,
            lig_atom_ref_coords=batch['lig_atom_ref_coords'],
            lig_constraint_matrix=batch['lig_constraint_matrix'],
            lig_bond_feats=batch['lig_bond_feats'],
            lig_uid_vec=batch['lig_uid_vec'],
            lig_atom_mask=batch['lig_atom_mask'],
            unimol_embeddings=batch['unimol_embeddings'],
            t=batch['t'],
        )

        # Concatenate rec and lig outputs for overall error
        rec_orig = output_orig['rec_flow']
        lig_orig = output_orig['lig_flow']
        rec_rot = output_rot['rec_flow']
        lig_rot = output_rot['lig_flow']

        # Rotate original output for comparison
        rec_orig_rotated = rotate_coords(rec_orig, R)
        lig_orig_rotated = rotate_coords(lig_orig, R)

        # Per-component errors
        rec_error = (rec_rot - rec_orig_rotated).abs()
        lig_error = (lig_rot - lig_orig_rotated).abs()

        # Combined errors
        all_error = torch.cat([rec_error, lig_error], dim=1)
        max_abs_error = all_error.max().item()
        mean_abs_error = all_error.mean().item()

        # Relative error (avoid division by zero)
        all_orig_rotated = torch.cat([rec_orig_rotated, lig_orig_rotated], dim=1)
        v_norm = all_orig_rotated.abs().clamp(min=1e-8)
        rel_error = all_error / v_norm
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        return {
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error,
            'rec_max_error': rec_error.max().item(),
            'rec_mean_error': rec_error.mean().item(),
            'lig_max_error': lig_error.max().item(),
            'lig_mean_error': lig_error.mean().item(),
        }


def prepare_flow_matching_coords(
    rec_coords_gt: torch.Tensor,
    lig_coords_gt: torch.Tensor,
    rec_backbone_mask: torch.Tensor,
    t: float,
    seed: int = 42
) -> tuple:
    '''
    Prepare flow matching coordinates at timestep t.

    Mirrors trainer.py _prepare_flow_matching() logic:
    - Backbone atoms start from ground truth (x_0 = x_1)
    - Sidechain atoms start from noise (x_0 ~ N(0,I))
    - Ligand atoms start from noise (x_0 ~ N(0,I))
    - x_t = t * x_1 + (1-t) * x_0

    Returns:
        (rec_x_t, lig_x_t): Interpolated coordinates at time t
    '''
    torch.manual_seed(seed)

    # Ground truth (x_1)
    rec_x_1 = rec_coords_gt
    lig_x_1 = lig_coords_gt

    # Sample noise
    rec_noise = torch.randn_like(rec_x_1)
    lig_x_0 = torch.randn_like(lig_x_1)

    # Backbone mask: 1 for backbone (fixed), 0 for sidechain (noised)
    # Shape: [B, N_rec] -> [B, N_rec, 1] for broadcasting
    mask = rec_backbone_mask.unsqueeze(-1)  # [B, N_rec, 1]

    # x_0 for receptor: backbone at GT, sidechain from noise
    rec_x_0 = mask * rec_x_1 + (1 - mask) * rec_noise

    # Interpolate: x_t = t * x_1 + (1-t) * x_0
    rec_x_t = t * rec_x_1 + (1 - t) * rec_x_0
    lig_x_t = t * lig_x_1 + (1 - t) * lig_x_0

    return rec_x_t, lig_x_t


def run_equivariance_tests(
    model: FlexModelV4,
    batch: dict,
    device: torch.device,
    angles: list = None,
    t: float = 0.5,
) -> dict:
    '''Run equivariance tests for all axes at specified angles.'''
    if angles is None:
        angles = list(range(0, 360, 30))

    results = {'x': [], 'y': [], 'z': []}

    for axis in ['x', 'y', 'z']:
        for angle in angles:
            R = rotation_matrix_axis(axis, angle, device)
            metrics = test_equivariance_single_rotation(model, batch, R, device)
            metrics['angle'] = angle
            results[axis].append(metrics)
            print(f"  {axis}-axis {angle:3d}°: max_err={metrics['max_abs_error']:.2e}, mean_err={metrics['mean_abs_error']:.2e}")

    return results


def plot_equivariance_results(results: dict, system_name: str, output_path: Path):
    '''Create plots for equivariance test results.'''
    angles = [r['angle'] for r in results['x']]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'SO(3) Equivariance Test: {system_name}', fontsize=14, fontweight='bold')

    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    markers = {'x': 'o', 'y': 's', 'z': '^'}

    # Plot 1: Max absolute error by axis
    ax1 = axes[0]
    for axis in ['x', 'y', 'z']:
        max_errors = [r['max_abs_error'] for r in results[axis]]
        ax1.plot(angles, max_errors, color=colors[axis], marker=markers[axis],
                 label=f'{axis}-axis', linewidth=2, markersize=6)
    ax1.set_xlabel('Rotation Angle (degrees)')
    ax1.set_ylabel('Max Absolute Error')
    ax1.set_title('Maximum Absolute Error vs Rotation Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(angles)
    ax1.set_yscale('log')

    # Plot 2: Mean absolute error by axis
    ax2 = axes[1]
    for axis in ['x', 'y', 'z']:
        mean_errors = [r['mean_abs_error'] for r in results[axis]]
        ax2.plot(angles, mean_errors, color=colors[axis], marker=markers[axis],
                 label=f'{axis}-axis', linewidth=2, markersize=6)
    ax2.set_xlabel('Rotation Angle (degrees)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Mean Absolute Error vs Rotation Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(angles)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def plot_polar_equivariance(results: dict, system_name: str, output_path: Path):
    '''Create polar plot showing error magnitude at different angles.'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    fig.suptitle(f'SO(3) Equivariance (Polar): {system_name}', fontsize=14, fontweight='bold')

    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

    for idx, axis in enumerate(['x', 'y', 'z']):
        ax = axes[idx]
        angles_rad = [np.radians(r['angle']) for r in results[axis]]
        max_errors = [r['max_abs_error'] for r in results[axis]]

        # Close the loop
        angles_rad.append(angles_rad[0])
        max_errors.append(max_errors[0])

        ax.plot(angles_rad, max_errors, color=colors[axis], linewidth=2, marker='o', markersize=4)
        ax.fill(angles_rad, max_errors, color=colors[axis], alpha=0.2)
        ax.set_title(f'{axis.upper()}-axis rotation', fontsize=12)
        ax.set_rlabel_position(45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Polar plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test and plot rotational equivariance')
    parser.add_argument('system_name', type=str, help='Name of the system directory (e.g., 101m__1__1.A__1.C_1.D)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default='/home/mhjang/workspace/flex/data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='/home/mhjang/workspace/flex/plots', help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--angles', type=str, default='0,30,60,90,120,150,180,210,240,270,300,330',
                        help='Comma-separated list of angles to test')
    parser.add_argument('--timestep', type=float, default=0.5, help='Flow matching timestep t')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for noise sampling')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse angles
    angles = [int(a) for a in args.angles.split(',')]
    print(f"Testing angles: {angles}")
    print(f"Flow matching timestep: t={args.timestep}")

    # Load data
    data_dir = Path(args.data_dir)
    system_dir = data_dir / args.system_name

    if not system_dir.exists():
        print(f"Error: System directory not found: {system_dir}")
        return

    print(f"\nLoading system: {args.system_name}")
    dataset = FlexDataset(str(data_dir), system_ids=[args.system_name])
    sample = dataset[0]

    # Add batch dimension and move to device
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
             for k, v in sample.items()}

    # Get backbone mask (precomputed in pocket_features.pt)
    rec_backbone_mask = batch.get('rec_backbone_mask')
    if rec_backbone_mask is None:
        N_rec = batch['rec_atom_feats'].shape[1]
        print("  WARNING: No backbone_mask found, using zeros (all atoms will be noised)")
        rec_backbone_mask = torch.zeros(1, N_rec, dtype=torch.float32, device=device)
    else:
        rec_backbone_mask = rec_backbone_mask.to(device).float()

    # Prepare flow matching coordinates at timestep t
    # Backbone atoms stay at GT, sidechain/ligand atoms interpolate from noise
    rec_x_t, lig_x_t = prepare_flow_matching_coords(
        rec_coords_gt=batch['rec_atom_coords_gt'],
        lig_coords_gt=batch['lig_atom_coords_gt'],
        rec_backbone_mask=rec_backbone_mask,
        t=args.timestep,
        seed=args.seed
    )

    batch['rec_atom_coords'] = rec_x_t
    batch['lig_atom_coords'] = lig_x_t
    batch['t'] = torch.tensor([args.timestep], dtype=torch.float32, device=device)

    # Add mask tensors (all True since no padding in single sample)
    N_rec = batch['rec_atom_feats'].shape[1]
    N_lig = batch['lig_atom_feats'].shape[1]
    batch['rec_atom_mask'] = torch.ones(1, N_rec, dtype=torch.bool, device=device)
    batch['lig_atom_mask'] = torch.ones(1, N_lig, dtype=torch.bool, device=device)

    # Statistics
    n_backbone = int(rec_backbone_mask.sum().item())
    n_sidechain = N_rec - n_backbone
    print(f"  Receptor atoms: {N_rec} (backbone: {n_backbone}, sidechain: {n_sidechain})")
    print(f"  Ligand atoms: {N_lig}")

    # Create model from config
    if args.config:
        print(f"\nLoading config from: {args.config}")
        full_config = load_config(args.config)
        model_config = full_config.model
    else:
        print("\nUsing default small config for testing")
        model_config = FlexConfig(ns=64, nv=8, attenuate_by_dist=False)

    model = FlexModelV4(model_config).to(device)
    model.eval()
    print(f"Model: FlexModelV4 (two-trunk)")
    print(f"Model config: ns={model_config.ns}, nv={model_config.nv}, "
          f"num_atom_layers={model_config.num_atom_layers}, num_geo_layers={model_config.num_geo_layers}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run tests
    print(f"\nRunning equivariance tests at t={args.timestep}...")
    results = run_equivariance_tests(model, batch, device, angles, t=args.timestep)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    safe_name = args.system_name.replace('/', '_').replace('\\', '_')
    t_str = f"_t{args.timestep:.2f}".replace('.', 'p')

    plot_equivariance_results(
        results,
        f"{args.system_name} (t={args.timestep})",
        output_dir / f'equivariance_{safe_name}{t_str}.png'
    )

    plot_polar_equivariance(
        results,
        f"{args.system_name} (t={args.timestep})",
        output_dir / f'equivariance_polar_{safe_name}{t_str}.png'
    )

    # Print summary
    print("\n" + "="*60)
    print(f"SUMMARY (t={args.timestep})")
    print("="*60)
    print(f"Flow matching: backbone fixed, sidechain+ligand noised")

    for axis in ['x', 'y', 'z']:
        max_err = max(r['max_abs_error'] for r in results[axis])
        mean_err = np.mean([r['mean_abs_error'] for r in results[axis]])
        print(f"{axis}-axis: max_error={max_err:.2e}, avg_mean_error={mean_err:.2e}")

    overall_max = max(max(r['max_abs_error'] for r in results[axis]) for axis in ['x', 'y', 'z'])
    print(f"\nOverall max error: {overall_max:.2e}")

    if overall_max < 1e-4:
        print("Equivariance: PASSED")
    elif overall_max < 1e-3:
        print("Equivariance: MARGINAL (numerical precision issues)")
    else:
        print("Equivariance: FAILED")


if __name__ == '__main__':
    main()
