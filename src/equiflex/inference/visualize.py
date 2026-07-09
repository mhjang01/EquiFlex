'''
Visualize inference trajectories from flow matching sampling.

Plots the flow from t=0 (noise) to t=1 (generated structure) using
trajectories saved by the sampler.

Usage:
    python -m inference.visualize --results results.pt --mode trajectory
    python -m inference.visualize --results results.pt --mode final --sample-idx 0
'''

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Union


def get_bonds_from_features(bond_feats: torch.Tensor) -> List[tuple]:
    '''Extract bond pairs from bond features tensor.'''
    bond_mask = (bond_feats.abs().sum(dim=-1) > 0)
    N = bond_mask.shape[0]
    bonds = []
    for i in range(N):
        for j in range(i + 1, N):
            if bond_mask[i, j]:
                bonds.append((i, j))
    return bonds


def get_residue_colors(uid_vec: np.ndarray) -> np.ndarray:
    '''Generate colors for each atom based on residue UID.'''
    unique_uids = np.unique(uid_vec)
    n_residues = len(unique_uids)
    cmap = plt.cm.get_cmap('tab20', max(n_residues, 20))
    uid_to_idx = {uid: i % 20 for i, uid in enumerate(unique_uids)}
    colors = np.array([cmap(uid_to_idx[uid])[:3] for uid in uid_vec])
    return colors


def plot_receptor(
    ax,
    coords: np.ndarray,
    bonds: List[tuple] = None,
    uid_vec: np.ndarray = None,
    alpha: float = 0.8,
    marker_size: float = 25,
    color: str = None,
):
    '''Plot receptor atoms.'''
    if uid_vec is not None and color is None:
        colors = get_residue_colors(uid_vec)
    else:
        colors = color or 'steelblue'

    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=colors, s=marker_size, alpha=alpha, marker='o',
        edgecolors='none', label='Receptor'
    )

    if bonds:
        for i, j in bonds:
            if isinstance(colors, np.ndarray):
                bond_color = (colors[i] + colors[j]) / 2
            else:
                bond_color = colors
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                c=bond_color, alpha=alpha * 0.6, linewidth=0.8
            )


def plot_ligand(
    ax,
    coords: np.ndarray,
    bonds: List[tuple] = None,
    alpha: float = 1.0,
    marker_size: float = 60,
):
    '''Plot ligand atoms as yellow triangles.'''
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c='gold', s=marker_size, alpha=alpha, marker='^',
        edgecolors='darkorange', linewidths=0.5, label='Ligand'
    )

    if bonds:
        for i, j in bonds:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                c='darkorange', alpha=alpha * 0.8, linewidth=1.2
            )


def set_axes_equal(ax, coords: np.ndarray, padding: float = 1.2):
    '''Set equal aspect ratio for 3D axes.'''
    max_range = np.abs(coords).max() * padding
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')


def visualize_trajectory(
    results: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    t_indices: List[int] = None,
    rec_bonds: List[tuple] = None,
    lig_bonds: List[tuple] = None,
    rec_uid_vec: np.ndarray = None,
    save_path: str = None,
):
    '''
    Visualize trajectory for a single sample at selected timesteps.

    Args:
        results: Results dict from sampler containing trajectories
        sample_idx: Which sample to visualize (batch index)
        t_indices: Which timestep indices to plot (default: 5 evenly spaced)
        rec_bonds: Receptor bond list
        lig_bonds: Ligand bond list
        rec_uid_vec: Receptor residue UID vector for coloring
        save_path: Path to save figure
    '''
    rec_traj = results['rec_trajectory'][sample_idx].numpy()  # [T, N_rec, 3]
    lig_traj = results['lig_trajectory'][sample_idx].numpy()  # [T, N_lig, 3]
    timesteps = results['timesteps'].numpy()  # [T]

    T = rec_traj.shape[0]

    if t_indices is None:
        # Default: 5 evenly spaced timesteps
        t_indices = np.linspace(0, T - 1, 5, dtype=int)

    n_cols = len(t_indices)
    fig = plt.figure(figsize=(5 * n_cols, 5))

    for plot_idx, t_idx in enumerate(t_indices):
        ax = fig.add_subplot(1, n_cols, plot_idx + 1, projection='3d')

        rec_coords = rec_traj[t_idx]
        lig_coords = lig_traj[t_idx]
        t_val = timesteps[t_idx]

        plot_receptor(ax, rec_coords, rec_bonds, rec_uid_vec,
                      alpha=0.8, marker_size=25)
        plot_ligand(ax, lig_coords, lig_bonds, alpha=1.0, marker_size=60)

        ax.set_title(f't = {t_val:.3f}', fontsize=14, fontweight='bold')
        # Use per-timestep bounds for better visualization
        timestep_coords = np.concatenate([rec_coords, lig_coords], axis=0)
        set_axes_equal(ax, timestep_coords)

        if plot_idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    plt.suptitle(f'Inference Trajectory (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_final_samples(
    results: Dict[str, torch.Tensor],
    sample_indices: List[int] = None,
    rec_bonds: List[tuple] = None,
    lig_bonds: List[tuple] = None,
    rec_uid_vec: np.ndarray = None,
    save_path: str = None,
):
    '''
    Visualize final structures for multiple samples.

    Args:
        results: Results dict from sampler
        sample_indices: Which samples to plot (default: first 4)
        rec_bonds: Receptor bond list
        lig_bonds: Ligand bond list
        rec_uid_vec: Receptor residue UID vector for coloring
        save_path: Path to save figure
    '''
    rec_final = results['rec_coords_final'].numpy()  # [B, N_rec, 3]
    lig_final = results['lig_coords_final'].numpy()  # [B, N_lig, 3]

    B = rec_final.shape[0]

    if sample_indices is None:
        sample_indices = list(range(min(4, B)))

    n_samples = len(sample_indices)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # Compute global bounds
    all_coords = np.concatenate([
        rec_final.reshape(-1, 3),
        lig_final.reshape(-1, 3)
    ], axis=0)

    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection='3d')

        rec_coords = rec_final[sample_idx]
        lig_coords = lig_final[sample_idx]

        plot_receptor(ax, rec_coords, rec_bonds, rec_uid_vec,
                      alpha=0.8, marker_size=25)
        plot_ligand(ax, lig_coords, lig_bonds, alpha=1.0, marker_size=60)

        ax.set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
        set_axes_equal(ax, all_coords)

        if plot_idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    plt.suptitle('Generated Structures (t=1)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_initial_vs_final(
    results: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    rec_bonds: List[tuple] = None,
    lig_bonds: List[tuple] = None,
    rec_uid_vec: np.ndarray = None,
    save_path: str = None,
):
    '''
    Side-by-side comparison of initial (t=0) and final (t=1) structures.

    Args:
        results: Results dict from sampler containing trajectories
        sample_idx: Which sample to visualize
        rec_bonds: Receptor bond list
        lig_bonds: Ligand bond list
        rec_uid_vec: Receptor residue UID vector for coloring
        save_path: Path to save figure
    '''
    rec_traj = results['rec_trajectory'][sample_idx].numpy()  # [T, N_rec, 3]
    lig_traj = results['lig_trajectory'][sample_idx].numpy()  # [T, N_lig, 3]

    rec_initial = rec_traj[0]
    lig_initial = lig_traj[0]
    rec_final = rec_traj[-1]
    lig_final = lig_traj[-1]

    fig = plt.figure(figsize=(12, 5))

    # Compute bounds separately for each subplot
    initial_coords = np.concatenate([rec_initial, lig_initial], axis=0)
    final_coords = np.concatenate([rec_final, lig_final], axis=0)

    # Initial state (t=0)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_receptor(ax1, rec_initial, rec_bonds, rec_uid_vec,
                  alpha=0.8, marker_size=25)
    plot_ligand(ax1, lig_initial, lig_bonds, alpha=1.0, marker_size=60)
    ax1.set_title('Initial State (t=0)\nBackbone fixed, sidechain+ligand noised', fontsize=12, fontweight='bold')
    set_axes_equal(ax1, initial_coords)
    ax1.legend(loc='upper left', fontsize=7)

    # Final state (t=1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_receptor(ax2, rec_final, rec_bonds, rec_uid_vec,
                  alpha=0.8, marker_size=25)
    plot_ligand(ax2, lig_final, lig_bonds, alpha=1.0, marker_size=60)
    ax2.set_title('Final State (t=1)\nGenerated Structure', fontsize=12, fontweight='bold')
    set_axes_equal(ax2, final_coords)
    ax2.legend(loc='upper left', fontsize=7)

    plt.suptitle(f'Flow Matching Inference (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_trajectories(
    results: Dict[str, torch.Tensor],
    sample_indices: List[int] = None,
    n_timesteps: int = 5,
    save_path: str = None,
):
    '''
    Visualize trajectories for multiple samples in a grid.

    Args:
        results: Results dict from sampler
        sample_indices: Which samples to plot
        n_timesteps: Number of timesteps per sample
        save_path: Path to save figure
    '''
    rec_traj = results['rec_trajectory'].numpy()  # [B, T, N_rec, 3]
    lig_traj = results['lig_trajectory'].numpy()  # [B, T, N_lig, 3]
    timesteps = results['timesteps'].numpy()

    B, T = rec_traj.shape[:2]

    if sample_indices is None:
        sample_indices = list(range(min(3, B)))

    t_indices = np.linspace(0, T - 1, n_timesteps, dtype=int)

    n_rows = len(sample_indices)
    n_cols = n_timesteps

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    # Global bounds
    all_coords = np.concatenate([
        rec_traj.reshape(-1, 3),
        lig_traj.reshape(-1, 3)
    ], axis=0)

    for row, sample_idx in enumerate(sample_indices):
        for col, t_idx in enumerate(t_indices):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')

            rec_coords = rec_traj[sample_idx, t_idx]
            lig_coords = lig_traj[sample_idx, t_idx]
            t_val = timesteps[t_idx]

            plot_receptor(ax, rec_coords, alpha=0.7, marker_size=15, color='steelblue')
            plot_ligand(ax, lig_coords, alpha=0.9, marker_size=40)

            if row == 0:
                ax.set_title(f't={t_val:.2f}', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Sample {sample_idx}', fontsize=10, fontweight='bold')

            set_axes_equal(ax, all_coords)

    plt.suptitle('Flow Matching Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def create_trajectory_gif(
    results: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    rec_bonds: List[tuple] = None,
    lig_bonds: List[tuple] = None,
    rec_uid_vec: np.ndarray = None,
    save_path: str = 'trajectory.gif',
    n_frames: int = 50,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (8, 8),
    elev: float = 20,
    azim_start: float = 0,
    azim_end: float = 360,
    rotate: bool = True,
):
    '''
    Create animated GIF of the inference trajectory.

    Args:
        results: Results dict from sampler containing trajectories
        sample_idx: Which sample to visualize
        rec_bonds: Receptor bond list
        lig_bonds: Ligand bond list
        rec_uid_vec: Receptor residue UID vector for coloring
        save_path: Path to save GIF
        n_frames: Number of frames in the animation
        fps: Frames per second
        dpi: Resolution of frames
        figsize: Figure size (width, height)
        elev: Elevation angle for 3D view
        azim_start: Starting azimuth angle
        azim_end: Ending azimuth angle
        rotate: Whether to rotate the view during animation
    '''
    from matplotlib.animation import FuncAnimation, PillowWriter

    rec_traj = results['rec_trajectory'][sample_idx].numpy()  # [T, N_rec, 3]
    lig_traj = results['lig_trajectory'][sample_idx].numpy()  # [T, N_lig, 3]
    timesteps = results['timesteps'].numpy()  # [T]

    T = rec_traj.shape[0]

    # Select frames evenly spaced through trajectory
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    # Compute global bounds using final structure (centered, reasonable range)
    final_coords = np.concatenate([rec_traj[-1], lig_traj[-1]], axis=0)
    max_range = np.abs(final_coords).max() * 1.3

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()

        t_idx = frame_indices[frame_idx]
        rec_coords = rec_traj[t_idx]
        lig_coords = lig_traj[t_idx]
        t_val = timesteps[t_idx]

        plot_receptor(ax, rec_coords, rec_bonds, rec_uid_vec,
                      alpha=0.8, marker_size=25)
        plot_ligand(ax, lig_coords, lig_bonds, alpha=1.0, marker_size=60)

        ax.set_title(f't = {t_val:.3f}', fontsize=14, fontweight='bold')

        # Set fixed bounds for smooth animation
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')

        # Rotate view if enabled
        if rotate:
            azim = azim_start + (azim_end - azim_start) * frame_idx / n_frames
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=elev, azim=azim_start)

        ax.legend(loc='upper left', fontsize=7)

        return ax,

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    print(f"Saved GIF to {save_path}")

    plt.close()


def create_receptor_gif(
    results: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    rec_bonds: List[tuple] = None,
    rec_uid_vec: np.ndarray = None,
    save_path: str = 'receptor_trajectory.gif',
    n_frames: int = 50,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (8, 8),
    elev: float = 20,
    azim_start: float = 0,
    azim_end: float = 360,
    rotate: bool = True,
):
    '''
    Create animated GIF of the receptor trajectory only.
    '''
    from matplotlib.animation import FuncAnimation, PillowWriter

    rec_traj = results['rec_trajectory'][sample_idx].numpy()  # [T, N_rec, 3]
    timesteps = results['timesteps'].numpy()

    T = rec_traj.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    final_coords = rec_traj[-1]
    max_range = np.abs(final_coords).max() * 1.3

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()

        t_idx = frame_indices[frame_idx]
        rec_coords = rec_traj[t_idx]
        t_val = timesteps[t_idx]

        plot_receptor(ax, rec_coords, rec_bonds, rec_uid_vec,
                      alpha=0.8, marker_size=30)

        ax.set_title(f'Receptor  t = {t_val:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')

        if rotate:
            azim = azim_start + (azim_end - azim_start) * frame_idx / n_frames
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=elev, azim=azim_start)

        return ax,

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    print(f"Saved receptor GIF to {save_path}")
    plt.close()


def create_ligand_gif(
    results: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    lig_bonds: List[tuple] = None,
    save_path: str = 'ligand_trajectory.gif',
    n_frames: int = 50,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (8, 8),
    elev: float = 20,
    azim_start: float = 0,
    azim_end: float = 360,
    rotate: bool = True,
):
    '''
    Create animated GIF of the ligand trajectory only.
    '''
    from matplotlib.animation import FuncAnimation, PillowWriter

    lig_traj = results['lig_trajectory'][sample_idx].numpy()  # [T, N_lig, 3]
    timesteps = results['timesteps'].numpy()

    T = lig_traj.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    final_coords = lig_traj[-1]
    max_range = np.abs(final_coords).max() * 1.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()

        t_idx = frame_indices[frame_idx]
        lig_coords = lig_traj[t_idx]
        t_val = timesteps[t_idx]

        plot_ligand(ax, lig_coords, lig_bonds, alpha=1.0, marker_size=80)

        ax.set_title(f'Ligand  t = {t_val:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')

        if rotate:
            azim = azim_start + (azim_end - azim_start) * frame_idx / n_frames
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=elev, azim=azim_start)

        return ax,

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    print(f"Saved ligand GIF to {save_path}")
    plt.close()


def load_results_with_data(
    results_path: Union[str, Path],
    data_path: Optional[Union[str, Path]] = None,
) -> tuple:
    '''
    Load results and optionally the original data for bond/uid info.

    Returns:
        (results, data) where data may be None
    '''
    results = torch.load(results_path, map_location='cpu')

    data = None
    if data_path is not None:
        data = torch.load(data_path, map_location='cpu')
        # Also load metadata if exists
        metadata_path = Path(data_path).parent / 'metadata.pt'
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location='cpu')
            data.update(metadata)

    return results, data


def main():
    parser = argparse.ArgumentParser(description='Visualize inference trajectories')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results .pt file from sampler')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to inference data .pt file (for bonds/residue info)')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'trajectory', 'final', 'grid'],
                        help='Visualization mode')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure')

    args = parser.parse_args()

    results, data = load_results_with_data(args.results, args.data)

    # Extract bond and uid info if data available
    rec_bonds = None
    lig_bonds = None
    rec_uid_vec = None
    if data is not None:
        if 'rec_bond_feats' in data:
            rec_bonds = get_bonds_from_features(data['rec_bond_feats'])
        if 'lig_bond_feats' in data:
            lig_bonds = get_bonds_from_features(data['lig_bond_feats'])
        if 'rec_uid_vec' in data:
            rec_uid_vec = data['rec_uid_vec'].numpy()

    if args.mode == 'comparison':
        visualize_initial_vs_final(
            results,
            sample_idx=args.sample_idx,
            rec_bonds=rec_bonds,
            lig_bonds=lig_bonds,
            rec_uid_vec=rec_uid_vec,
            save_path=args.save,
        )
    elif args.mode == 'trajectory':
        visualize_trajectory(
            results,
            sample_idx=args.sample_idx,
            rec_bonds=rec_bonds,
            lig_bonds=lig_bonds,
            rec_uid_vec=rec_uid_vec,
            save_path=args.save,
        )
    elif args.mode == 'final':
        visualize_final_samples(
            results,
            rec_bonds=rec_bonds,
            lig_bonds=lig_bonds,
            rec_uid_vec=rec_uid_vec,
            save_path=args.save,
        )
    elif args.mode == 'grid':
        visualize_all_trajectories(
            results,
            save_path=args.save,
        )


if __name__ == '__main__':
    main()
