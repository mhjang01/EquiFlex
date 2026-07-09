"""
Data loading for Flex model training.

Single-system batching: Each batch contains multiple timesteps from ONE system.
This avoids padding complexity and ensures consistent tensor shapes within a batch.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
import random
import json
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm


def _load_system(system_dir: Path):
    """Read all 4 feature/embedding tensors for one system."""
    return {
        'ligand_feats': torch.load(system_dir / 'ligand_features.pt', weights_only=False),
        'pocket_feats': torch.load(system_dir / 'pocket_features.pt', weights_only=False),
        'esm_emb': torch.load(system_dir / 'esm3_embeddings.pt', weights_only=False),
        'unimol_emb': torch.load(system_dir / 'unimol_embeddings.pt', weights_only=False),
    }


class FlexDataset(Dataset):
    """
    Dataset for Flex model training.

    Loads pre-featurized data from system directories containing:
    - ligand_features.pt
    - pocket_features.pt
    - esm3_embeddings.pt
    - unimol_embeddings.pt

    When ``preload=True``, all four tensors per system are read into a Python
    list at __init__ time and reused across __getitem__ calls. With Linux
    ``fork``-based DataLoader workers, the cache is shared copy-on-write, so a
    single in-memory copy backs all workers. ``__getitem__`` MUST NOT mutate
    cached tensors in place (in-place ops trigger COW per worker, defeating the
    purpose).
    """

    def __init__(
        self,
        data_dir: str,
        system_ids: Optional[List[str]] = None,
        preload: bool = False,
        preload_jobs: int = 16,
        preload_desc: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)

        if system_ids is not None:
            self.system_dirs = [self.data_dir / sid for sid in system_ids]
        else:
            self.system_dirs = sorted([
                d for d in self.data_dir.iterdir()
                if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
            ])

        self._cache: Optional[List[Dict]] = None
        if preload:
            desc = preload_desc or f'preload {self.data_dir.name}'
            self._cache = Parallel(n_jobs=preload_jobs, backend='threading')(
                delayed(_load_system)(sd) for sd in tqdm(self.system_dirs, desc=desc)
            )

    def __len__(self) -> int:
        return len(self.system_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            entry = self._cache[idx]
            ligand_feats = entry['ligand_feats']
            pocket_feats = entry['pocket_feats']
            esm_emb = entry['esm_emb']
            unimol_emb = entry['unimol_emb']
        else:
            system_dir = self.system_dirs[idx]
            ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
            pocket_feats = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
            esm_emb = torch.load(system_dir / 'esm3_embeddings.pt', weights_only=False)
            unimol_emb = torch.load(system_dir / 'unimol_embeddings.pt', weights_only=False)

        # Backbone mask: 1 = backbone (fixed), 0 = sidechain (flexible)
        # Precomputed in pocket_features.pt during featurization
        rec_backbone_mask = pocket_feats.get('backbone_mask', None)
        if rec_backbone_mask is None:
            # Fallback for old data without backbone_mask
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_backbone_mask = torch.zeros(n_atoms, dtype=torch.float32)

        # Load torsion indices if available (sp2_torsion_indices for planarity)
        lig_torsion_indices = ligand_feats.get('sp2_torsion_indices', ligand_feats.get('torsion_indices', None))
        if lig_torsion_indices is not None:
            if isinstance(lig_torsion_indices, np.ndarray):
                lig_torsion_indices = torch.from_numpy(lig_torsion_indices).long()
        else:
            lig_torsion_indices = torch.empty((0, 4), dtype=torch.long)

        rec_torsion_indices = pocket_feats.get('sp2_torsion_indices', pocket_feats.get('torsion_indices', None))
        if rec_torsion_indices is not None:
            if isinstance(rec_torsion_indices, np.ndarray):
                rec_torsion_indices = torch.from_numpy(rec_torsion_indices).long()
        else:
            rec_torsion_indices = torch.empty((0, 4), dtype=torch.long)

        # CB mask: 1 = CB atom (fixed during generation), 0 = other
        rec_cb_mask = pocket_feats.get('cb_mask', None)
        if rec_cb_mask is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_cb_mask = torch.zeros(n_atoms, dtype=torch.float32)

        # Per-residue backbone atom indices for CB placement
        backbone_atom_indices = pocket_feats.get('backbone_atom_indices', None)

        # Missing mask: atoms with missing/unknown coordinates (should be excluded from loss)
        rec_missing_mask = pocket_feats.get('missing_mask', None)
        if rec_missing_mask is not None:
            if isinstance(rec_missing_mask, np.ndarray):
                rec_missing_mask = torch.from_numpy(rec_missing_mask).bool()
        else:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_missing_mask = torch.zeros(n_atoms, dtype=torch.bool)

        # Constraint matrices: default to zeros if not present
        rec_constraint_matrix = pocket_feats.get('constraint_matrix', None)
        if rec_constraint_matrix is None:
            n_atoms = pocket_feats['atom_features'].shape[0]
            rec_constraint_matrix = torch.zeros(n_atoms, n_atoms, 1, dtype=torch.float32)
        else:
            if isinstance(rec_constraint_matrix, np.ndarray):
                rec_constraint_matrix = torch.from_numpy(rec_constraint_matrix).float()
            if rec_constraint_matrix.ndim == 2:
                rec_constraint_matrix = rec_constraint_matrix.unsqueeze(-1)

        lig_constraint_matrix = ligand_feats.get('constraint_matrix', None)
        if lig_constraint_matrix is None:
            n_atoms = ligand_feats['atom_features'].shape[0]
            lig_constraint_matrix = torch.zeros(n_atoms, n_atoms, 1, dtype=torch.float32)
        else:
            if isinstance(lig_constraint_matrix, np.ndarray):
                lig_constraint_matrix = torch.from_numpy(lig_constraint_matrix).float()
            if lig_constraint_matrix.ndim == 2:
                lig_constraint_matrix = lig_constraint_matrix.unsqueeze(-1)

        return {
            # Receptor (pocket)
            'rec_atom_feats': pocket_feats['atom_features'],  # [N_rec, 124]
            'rec_atom_coords_gt': pocket_feats['coords'],  # [N_rec, 3]
            'rec_atom_ref_coords': pocket_feats['ref_coords'],  # [N_rec, 3]
            'rec_constraint_matrix': rec_constraint_matrix,  # [N_rec, N_rec, 1]
            'rec_bond_feats': pocket_feats['bond_features'],  # [N_rec, N_rec, 26]
            'rec_uid_vec': pocket_feats['residue_uid_vec'],  # [N_rec]
            'rec_backbone_mask': rec_backbone_mask,  # [N_rec]
            'rec_cb_mask': rec_cb_mask,  # [N_rec]
            'rec_backbone_atom_indices': backbone_atom_indices,  # dict or None
            'rec_missing_mask': rec_missing_mask,  # [N_rec]
            'rec_torsion_indices': rec_torsion_indices,  # [M_rec, 4]
            'esm_embeddings': esm_emb,  # [N_rec_tokens, 1536]

            # Ligand
            'lig_atom_feats': ligand_feats['atom_features'],  # [N_lig, 124]
            'lig_atom_coords_gt': ligand_feats['coords'],  # [N_lig, 3]
            'lig_atom_ref_coords': ligand_feats['ref_coords'],  # [N_lig, 3]
            'lig_constraint_matrix': lig_constraint_matrix,  # [N_lig, N_lig, 1]
            'lig_bond_feats': ligand_feats['bond_features'],  # [N_lig, N_lig, 26]
            'lig_uid_vec': ligand_feats['ligand_uid_vec'],  # [N_lig]
            'lig_torsion_indices': lig_torsion_indices,  # [M_lig, 4]
            'unimol_embeddings': unimol_emb,  # [N_lig, 1536]

            # Metadata
            'system_idx': idx,
        }


class SingleSystemSampler(Sampler[int]):
    """
    Sampler that yields system indices, with each system repeated batch_size times.
    This creates batches where all samples are from the same system (different timesteps).
    """

    def __init__(
        self,
        dataset: FlexDataset,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_systems = len(dataset)

    def __iter__(self) -> Iterator[int]:
        system_indices = list(range(self.num_systems))
        if self.shuffle:
            random.shuffle(system_indices)

        for sys_idx in system_indices:
            # Yield the same system index batch_size times
            for _ in range(self.batch_size):
                yield sys_idx

    def __len__(self) -> int:
        return self.num_systems * self.batch_size


class MultiSystemSampler(Sampler[int]):
    """Yield several systems per loader batch, each repeated for timestep copies.

    The dataloader still receives flat indices, but each contiguous batch contains
    `systems_per_batch` distinct systems and `batch_size` copies per system.
    """

    def __init__(
        self,
        dataset: FlexDataset,
        batch_size: int = 8,
        systems_per_batch: int = 1,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.systems_per_batch = max(1, systems_per_batch)
        self.shuffle = shuffle
        self.num_systems = len(dataset)

    def __iter__(self) -> Iterator[int]:
        system_indices = list(range(self.num_systems))
        if self.shuffle:
            random.shuffle(system_indices)

        for start in range(0, self.num_systems, self.systems_per_batch):
            group = system_indices[start:start + self.systems_per_batch]
            for sys_idx in group:
                for _ in range(self.batch_size):
                    yield sys_idx

    def __len__(self) -> int:
        return self.num_systems * self.batch_size


def collate_single_system(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for V4 single-system batching.

    System-level tensors (features, bonds, constraints, embeddings) stay at [1, ...]
    since they are identical across timesteps. Only per-timestep tensors (GT coords,
    masks) are expanded to [B, ...].
    """
    B = len(batch)
    item = batch[0]

    N_rec = item['rec_atom_feats'].shape[0]
    N_lig = item['lig_atom_feats'].shape[0]

    # Compute backbone COM for centering
    backbone_mask = item['rec_backbone_mask']  # [N_rec]
    rec_coords_gt = item['rec_atom_coords_gt']  # [N_rec, 3]
    missing_mask = item['rec_missing_mask']  # [N_rec], True = missing
    valid_backbone_mask = backbone_mask * (~missing_mask).float()

    if valid_backbone_mask.sum() > 0:
        backbone_com = (rec_coords_gt * valid_backbone_mask.unsqueeze(-1)).sum(dim=0) / valid_backbone_mask.sum()
    else:
        backbone_com = rec_coords_gt.mean(dim=0)

    # Center GT coordinates by backbone COM
    rec_coords_gt_centered = rec_coords_gt - backbone_com
    lig_coords_gt_centered = item['lig_atom_coords_gt'] - backbone_com

    result = {
        # System-level [1, ...] — chemistry trunk inputs (no tiling needed)
        'rec_atom_feats': item['rec_atom_feats'].unsqueeze(0),             # [1, N_rec, feat_dim]
        'rec_atom_ref_coords': item['rec_atom_ref_coords'].unsqueeze(0),   # [1, N_rec, 3]
        'rec_constraint_matrix': item['rec_constraint_matrix'].unsqueeze(0),
        'rec_bond_feats': item['rec_bond_feats'].unsqueeze(0),             # [1, N_rec, N_rec, 26]
        'rec_uid_vec': item['rec_uid_vec'].unsqueeze(0),                   # [1, N_rec]
        'esm_embeddings': item['esm_embeddings'].unsqueeze(0),             # [1, N_rec_tokens, esm_dim]
        'lig_atom_feats': item['lig_atom_feats'].unsqueeze(0),             # [1, N_lig, feat_dim]
        'lig_atom_ref_coords': item['lig_atom_ref_coords'].unsqueeze(0),   # [1, N_lig, 3]
        'lig_constraint_matrix': item['lig_constraint_matrix'].unsqueeze(0),
        'lig_bond_feats': item['lig_bond_feats'].unsqueeze(0),             # [1, N_lig, N_lig, 26]
        'lig_uid_vec': item['lig_uid_vec'].unsqueeze(0),                   # [1, N_lig]
        'unimol_embeddings': item['unimol_embeddings'].unsqueeze(0),       # [1, N_lig, unimol_dim]

        # Per-timestep [B, ...] — flow matching inputs
        'rec_atom_coords_gt': rec_coords_gt_centered.unsqueeze(0).expand(B, -1, -1).clone(),
        'lig_atom_coords_gt': lig_coords_gt_centered.unsqueeze(0).expand(B, -1, -1).clone(),
        'rec_backbone_mask': item['rec_backbone_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_cb_mask': item['rec_cb_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_missing_mask': item['rec_missing_mask'].unsqueeze(0).expand(B, -1).clone(),
        'rec_atom_mask': torch.ones(B, N_rec, dtype=torch.bool),
        'lig_atom_mask': torch.ones(B, N_lig, dtype=torch.bool),

        # Batch size for trainer to derive B
        'batch_size': B,
    }

    # Per-residue backbone atom indices (system-level, no batch expansion)
    bai = item.get('rec_backbone_atom_indices')
    if bai is not None:
        result['rec_backbone_atom_indices'] = bai

    # Torsion indices: replicate with batch index [batch_idx, a, b, c, d]
    rec_torsions = item['rec_torsion_indices']
    if rec_torsions.numel() > 0:
        all_rec_torsions = []
        for i in range(B):
            batch_col = torch.full((rec_torsions.shape[0], 1), i, dtype=torch.long)
            all_rec_torsions.append(torch.cat([batch_col, rec_torsions], dim=1))
        result['rec_torsion_indices'] = torch.cat(all_rec_torsions, dim=0)
    else:
        result['rec_torsion_indices'] = torch.empty((0, 5), dtype=torch.long)

    lig_torsions = item['lig_torsion_indices']
    if lig_torsions.numel() > 0:
        all_lig_torsions = []
        for i in range(B):
            batch_col = torch.full((lig_torsions.shape[0], 1), i, dtype=torch.long)
            all_lig_torsions.append(torch.cat([batch_col, lig_torsions], dim=1))
        result['lig_torsion_indices'] = torch.cat(all_lig_torsions, dim=0)
    else:
        result['lig_torsion_indices'] = torch.empty((0, 5), dtype=torch.long)

    return result


def collate_multi_system(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Collate several variable-size systems without padding.

    Returns a list of regular single-system batches. The trainer processes each
    element sequentially and averages their losses into one optimizer step.
    """
    grouped: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    for item in batch:
        grouped.setdefault(int(item['system_idx']), []).append(item)
    return [collate_single_system(items) for items in grouped.values()]



def _pad_first_dim(x: torch.Tensor, length: int, value=0) -> torch.Tensor:
    out_shape = (length,) + tuple(x.shape[1:])
    out = x.new_full(out_shape, value)
    out[:x.shape[0]] = x
    return out


def _pad_square(x: torch.Tensor, n: int, value=0) -> torch.Tensor:
    out_shape = (n, n) + tuple(x.shape[2:])
    out = x.new_full(out_shape, value)
    out[:x.shape[0], :x.shape[1]] = x
    return out


def _pad_timestep_tensor(x: torch.Tensor, length: int, timesteps: int, value=0) -> torch.Tensor:
    padded = _pad_first_dim(x, length, value=value)
    return padded.unsqueeze(0).expand(timesteps, *padded.shape).clone()


def collate_padded_systems(
    batch: List[Dict[str, torch.Tensor]],
    timesteps_per_system: int,
) -> Dict[str, torch.Tensor]:
    """Pad several systems into one batch and create timestep copies per system.

    System-level chemistry tensors are kept as [S, ...]. Flow/loss tensors are
    repeated to [S * T, ...], and ``system_index_per_timestep`` maps each copy
    back to its chemistry row.
    """
    S = len(batch)
    T = int(timesteps_per_system)
    B = S * T

    max_rec = max(item['rec_atom_feats'].shape[0] for item in batch)
    max_lig = max(item['lig_atom_feats'].shape[0] for item in batch)
    max_esm = max(item['esm_embeddings'].shape[0] for item in batch)

    rec_atom_feats = []
    rec_atom_ref_coords = []
    rec_constraint_matrix = []
    rec_bond_feats = []
    rec_uid_vec = []
    esm_embeddings = []
    lig_atom_feats = []
    lig_atom_ref_coords = []
    lig_constraint_matrix = []
    lig_bond_feats = []
    lig_uid_vec = []
    unimol_embeddings = []

    rec_coords_gt = []
    lig_coords_gt = []
    rec_backbone_mask = []
    rec_cb_mask = []
    rec_missing_mask = []
    rec_atom_mask_system = []
    lig_atom_mask_system = []
    all_rec_torsions = []
    all_lig_torsions = []

    for sys_i, item in enumerate(batch):
        n_rec = item['rec_atom_feats'].shape[0]
        n_lig = item['lig_atom_feats'].shape[0]

        # Match single-system centering: each system is centered by its own
        # valid receptor-backbone COM before timestep copies are formed.
        backbone_mask = item['rec_backbone_mask']
        missing_mask = item['rec_missing_mask']
        rec_gt = item['rec_atom_coords_gt']
        valid_backbone_mask = backbone_mask * (~missing_mask).float()
        if valid_backbone_mask.sum() > 0:
            backbone_com = (rec_gt * valid_backbone_mask.unsqueeze(-1)).sum(dim=0) / valid_backbone_mask.sum()
        else:
            backbone_com = rec_gt.mean(dim=0)

        rec_gt_centered = rec_gt - backbone_com
        lig_gt_centered = item['lig_atom_coords_gt'] - backbone_com

        rec_atom_feats.append(_pad_first_dim(item['rec_atom_feats'], max_rec))
        rec_atom_ref_coords.append(_pad_first_dim(item['rec_atom_ref_coords'], max_rec))
        rec_constraint_matrix.append(_pad_square(item['rec_constraint_matrix'], max_rec))
        rec_bond_feats.append(_pad_square(item['rec_bond_feats'], max_rec))
        rec_uid_vec.append(_pad_first_dim(item['rec_uid_vec'].long(), max_rec, value=0))
        esm_embeddings.append(_pad_first_dim(item['esm_embeddings'], max_esm))

        lig_atom_feats.append(_pad_first_dim(item['lig_atom_feats'], max_lig))
        lig_atom_ref_coords.append(_pad_first_dim(item['lig_atom_ref_coords'], max_lig))
        lig_constraint_matrix.append(_pad_square(item['lig_constraint_matrix'], max_lig))
        lig_bond_feats.append(_pad_square(item['lig_bond_feats'], max_lig))
        lig_uid_vec.append(_pad_first_dim(item['lig_uid_vec'].long(), max_lig, value=0))
        unimol_embeddings.append(_pad_first_dim(item['unimol_embeddings'], max_lig))

        rec_coords_gt.append(_pad_timestep_tensor(rec_gt_centered, max_rec, T))
        lig_coords_gt.append(_pad_timestep_tensor(lig_gt_centered, max_lig, T))
        rec_backbone_mask.append(_pad_timestep_tensor(item['rec_backbone_mask'].float(), max_rec, T))
        rec_cb_mask.append(_pad_timestep_tensor(item['rec_cb_mask'].float(), max_rec, T))
        rec_missing_mask.append(_pad_timestep_tensor(item['rec_missing_mask'].bool(), max_rec, T, value=False))

        rec_mask = torch.zeros(max_rec, dtype=torch.bool)
        rec_mask[:n_rec] = True
        lig_mask = torch.zeros(max_lig, dtype=torch.bool)
        lig_mask[:n_lig] = True
        rec_atom_mask_system.append(rec_mask)
        lig_atom_mask_system.append(lig_mask)

        rec_torsions = item['rec_torsion_indices']
        if rec_torsions.numel() > 0:
            for t_i in range(T):
                batch_col = torch.full((rec_torsions.shape[0], 1), sys_i * T + t_i, dtype=torch.long)
                all_rec_torsions.append(torch.cat([batch_col, rec_torsions], dim=1))

        lig_torsions = item['lig_torsion_indices']
        if lig_torsions.numel() > 0:
            for t_i in range(T):
                batch_col = torch.full((lig_torsions.shape[0], 1), sys_i * T + t_i, dtype=torch.long)
                all_lig_torsions.append(torch.cat([batch_col, lig_torsions], dim=1))

    rec_atom_mask_system = torch.stack(rec_atom_mask_system, dim=0)
    lig_atom_mask_system = torch.stack(lig_atom_mask_system, dim=0)

    return {
        'padded_batch': True,
        'system_batch_size': S,
        'timesteps_per_system': T,
        'system_index_per_timestep': torch.arange(S, dtype=torch.long).repeat_interleave(T),
        'system_idx': torch.tensor([int(item['system_idx']) for item in batch], dtype=torch.long),
        'batch_size': B,

        # System-level [S, ...] chemistry tensors.
        'rec_atom_feats': torch.stack(rec_atom_feats, dim=0),
        'rec_atom_ref_coords': torch.stack(rec_atom_ref_coords, dim=0),
        'rec_constraint_matrix': torch.stack(rec_constraint_matrix, dim=0),
        'rec_bond_feats': torch.stack(rec_bond_feats, dim=0),
        'rec_uid_vec': torch.stack(rec_uid_vec, dim=0),
        'rec_atom_mask_system': rec_atom_mask_system,
        'esm_embeddings': torch.stack(esm_embeddings, dim=0),
        'lig_atom_feats': torch.stack(lig_atom_feats, dim=0),
        'lig_atom_ref_coords': torch.stack(lig_atom_ref_coords, dim=0),
        'lig_constraint_matrix': torch.stack(lig_constraint_matrix, dim=0),
        'lig_bond_feats': torch.stack(lig_bond_feats, dim=0),
        'lig_uid_vec': torch.stack(lig_uid_vec, dim=0),
        'lig_atom_mask_system': lig_atom_mask_system,
        'unimol_embeddings': torch.stack(unimol_embeddings, dim=0),

        # Per-timestep [S*T, ...] tensors.
        'rec_atom_coords_gt': torch.cat(rec_coords_gt, dim=0),
        'lig_atom_coords_gt': torch.cat(lig_coords_gt, dim=0),
        'rec_backbone_mask': torch.cat(rec_backbone_mask, dim=0),
        'rec_cb_mask': torch.cat(rec_cb_mask, dim=0),
        'rec_missing_mask': torch.cat(rec_missing_mask, dim=0),
        'rec_atom_mask': rec_atom_mask_system.repeat_interleave(T, dim=0),
        'lig_atom_mask': lig_atom_mask_system.repeat_interleave(T, dim=0),

        'rec_torsion_indices': (
            torch.cat(all_rec_torsions, dim=0) if all_rec_torsions
            else torch.empty((0, 5), dtype=torch.long)
        ),
        'lig_torsion_indices': (
            torch.cat(all_lig_torsions, dim=0) if all_lig_torsions
            else torch.empty((0, 5), dtype=torch.long)
        ),
    }

def get_dataloader(
    data_dir: str = './pdbbind_data',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    system_ids: Optional[List[str]] = None,
    drop_last: bool = False,
    preload: bool = False,
    preload_desc: Optional[str] = None,
    systems_per_batch: int = 1,
    pad_systems: bool = True,
) -> DataLoader:
    """
    Create dataloader for Flex model with single-system batching.

    Each batch contains batch_size samples from the SAME system,
    allowing different timesteps to be sampled for flow matching.

    Args:
        data_dir: Path to data directory
        batch_size: Number of timesteps per system in each batch
        shuffle: Whether to shuffle systems between epochs
        num_workers: Number of data loading workers
        system_ids: Optional list of specific system IDs to load
        drop_last: Whether to drop last incomplete batch (unused for single-system)
        preload: Read all systems' tensors into RAM at __init__; with Linux
            ``fork``-based workers, COW shares one copy across workers.
        preload_desc: Optional tqdm description for the preload pass.

    Returns:
        DataLoader instance
    """
    dataset = FlexDataset(data_dir, system_ids, preload=preload, preload_desc=preload_desc)

    systems_per_batch = max(1, int(systems_per_batch))
    if systems_per_batch > 1 and pad_systems:
        return DataLoader(
            dataset,
            batch_size=systems_per_batch,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda items: collate_padded_systems(items, timesteps_per_system=batch_size),
            pin_memory=True,
            drop_last=drop_last,
        )

    if systems_per_batch == 1:
        sampler = SingleSystemSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        loader_batch_size = batch_size
        collate_fn = collate_single_system
    else:
        sampler = MultiSystemSampler(
            dataset,
            batch_size=batch_size,
            systems_per_batch=systems_per_batch,
            shuffle=shuffle,
        )
        loader_batch_size = batch_size * systems_per_batch
        collate_fn = collate_multi_system

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=loader_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def load_split_from_csv(
    csv_path: str = 'scripts/lp_pdbbind/LP_PDBBind.csv',
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train/val/test splits from LP_PDBBind CSV file.

    Args:
        csv_path: Path to LP_PDBBind.csv with 'new_split' column

    Returns:
        Tuple of (train_system_ids, val_system_ids, test_system_ids)
    """
    import pandas as pd

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_file)

    # Detect which schema the CSV uses:
    #   v2_plinder: columns = system_id, split
    #   LP_PDBBind: index_col=0 holds the id, split column is 'new_split'
    split_col = 'split' if 'split' in df.columns else 'new_split'
    if 'system_id' in df.columns:
        id_series = df['system_id'].astype(str)
    else:
        df = pd.read_csv(csv_file, index_col=0)
        id_series = df.index.astype(str)

    splits = df[split_col].astype(str).str.strip()
    ids = id_series.str.strip()

    train_systems = ids[splits == 'train'].tolist()
    val_systems = ids[splits == 'val'].tolist()
    test_systems = ids[splits == 'test'].tolist()

    print(f"Loaded splits from {csv_path} (id_col=" +
          ('system_id' if 'system_id' in df.columns else 'index') +
          f", split_col={split_col})")
    print(f"  Train: {len(train_systems)} systems")
    print(f"  Val: {len(val_systems)} systems")
    print(f"  Test: {len(test_systems)} systems")

    return train_systems, val_systems, test_systems


def load_split_from_file(
    split_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Load train/val split from pre-computed split files.

    Expected files in split_dir:
    - split_train.txt: One system ID per line
    - split_val.txt: One system ID per line

    Or:
    - split.json: JSON with 'train' and 'test'/'val' keys

    Args:
        split_dir: Directory containing split files

    Returns:
        Tuple of (train_system_ids, val_system_ids)
    """
    split_path = Path(split_dir)

    # Try JSON first
    json_file = split_path / 'split.json'
    if json_file.exists():
        with open(json_file, 'r') as f:
            split_data = json.load(f)
        train_systems = split_data['train']
        # Support both 'val' and 'test' keys
        val_systems = split_data.get('val', split_data.get('test', []))
        print(f"Loaded split from {json_file}")
        print(f"  Train: {len(train_systems)} systems")
        print(f"  Val: {len(val_systems)} systems")
        if 'stats' in split_data:
            stats = split_data['stats']
            print(f"  Similarity threshold: {stats.get('similarity_threshold', 'N/A')}")
        return train_systems, val_systems

    # Try text files
    train_file = split_path / 'split_train.txt'
    val_file = split_path / 'split_val.txt'

    if train_file.exists() and val_file.exists():
        with open(train_file, 'r') as f:
            train_systems = [line.strip() for line in f if line.strip()]
        with open(val_file, 'r') as f:
            val_systems = [line.strip() for line in f if line.strip()]
        print(f"Loaded split from {train_file.name} and {val_file.name}")
        print(f"  Train: {len(train_systems)} systems")
        print(f"  Val: {len(val_systems)} systems")
        return train_systems, val_systems

    raise FileNotFoundError(
        f"Could not find split files in {split_dir}. "
        f"Expected either 'split.json' or "
        f"'split_train.txt' + 'split_val.txt'"
    )


def split_systems_random(
    data_dir: str = './pdbbind_data',
    val_ratio: float = 0.05,
    seed: int = 42,
    log_dir: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Split systems into train and validation sets using random shuffle.
    
    WARNING: This does NOT account for ligand similarity!
    Use load_split_from_file() with Tanimoto-based splits for proper evaluation.

    Args:
        data_dir: Path to data directory
        val_ratio: Fraction of systems for validation
        seed: Random seed for reproducibility
        log_dir: Directory to save split log file

    Returns:
        Tuple of (train_system_ids, val_system_ids)
    """
    data_path = Path(data_dir)

    # Get all valid system directories
    all_systems = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
    ])

    # Shuffle with fixed seed
    rng = random.Random(seed)
    shuffled = all_systems.copy()
    rng.shuffle(shuffled)

    # Split
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_systems = shuffled[:n_val]
    train_systems = shuffled[n_val:]

    # Log the split
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        split_info = {
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'val_ratio': val_ratio,
            'total_systems': len(all_systems),
            'train_count': len(train_systems),
            'val_count': len(val_systems),
            'val_systems': val_systems,
        }

        log_file = log_path / 'data_split.json'
        with open(log_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"Data split logged to: {log_file}")

    return train_systems, val_systems


# Backward compatibility alias
split_systems = split_systems_random


def get_train_val_dataloaders(
    data_dir: str = './pdbbind_data',
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.05,
    seed: int = 42,
    log_dir: Optional[str] = None,
    val_batches: int = 100,
    split_csv: str = 'scripts/lp_pdbbind/LP_PDBBind.csv',
    split_dir: Optional[str] = None,
    preload: bool = False,
    systems_per_batch: int = 1,
    pad_systems: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], List[str], List[str]]:
    """
    Create train and validation dataloaders.

    By default, uses splits from LP_PDBBind.csv ('new_split' column).
    If split_dir is provided, loads from pre-computed split files instead.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size (timesteps per system)
        num_workers: Number of data loading workers
        val_ratio: Fraction of systems for validation (only used if split_csv is None)
        seed: Random seed for split reproducibility (only used if split_csv is None)
        log_dir: Directory to save split log (only used if split_csv is None)
        val_batches: Number of validation batches to use
        split_csv: Path to LP_PDBBind.csv with 'new_split' column (default)
        split_dir: Directory containing pre-computed split files (overrides split_csv)

    Returns:
        Tuple of (train_loader, val_loader, train_system_ids, val_system_ids, test_system_ids)
        Note: test_system_ids will be empty because test systems are merged into training.
    """
    test_systems = []

    if split_dir is not None:
        # Load pre-computed Tanimoto-based split from directory
        train_systems, val_systems = load_split_from_file(split_dir=split_dir)
    elif split_csv is not None and Path(split_csv).exists():
        # Load from LP_PDBBind CSV (default)
        train_systems, val_systems, test_systems = load_split_from_csv(csv_path=split_csv)
    else:
        # No valid split source found - abort training
        raise FileNotFoundError(
            f"LP-PDBBind CSV not found at '{split_csv}' and no split_dir provided. "
            f"Training requires a valid train-val-test split. "
            f"Either provide a valid split_csv path or set split_dir to a directory containing split files."
        )

    # Merge test into training (external test set used elsewhere)
    if test_systems:
        train_systems = train_systems + test_systems
        test_systems = []

    print(f"Train systems: {len(train_systems)}, Val systems: {len(val_systems)}")

    # Validate that system directories exist
    data_path = Path(data_dir)
    train_systems = [s for s in train_systems if (data_path / s / 'ligand_features.pt').exists()]
    val_systems = [s for s in val_systems if (data_path / s / 'ligand_features.pt').exists()]
    print(f"After validation - Train: {len(train_systems)}, Val: {len(val_systems)}")

    # Training dataloader (full epoch)
    train_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        system_ids=train_systems,
        preload=preload,
        preload_desc='preload train',
        systems_per_batch=systems_per_batch,
        pad_systems=pad_systems,
    )

    # Validation dataloader with limited systems for faster validation
    # Select random subset of val systems for each validation
    val_systems_subset = val_systems[:val_batches] if len(val_systems) > val_batches else val_systems

    val_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to get different systems each validation
        num_workers=num_workers,
        system_ids=val_systems_subset,
        preload=preload,
        preload_desc='preload val',
        systems_per_batch=systems_per_batch,
        pad_systems=pad_systems,
    )

    return train_loader, val_loader, train_systems, val_systems, test_systems


def get_test_dataloader(
    test_data_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Optional[DataLoader]:
    """
    Create test dataloader using test split from LP_PDBBind.csv.

    Args:
        test_data_dir: Optional path to external test data directory
        batch_size: Batch size (timesteps per system)
        num_workers: Number of data loading workers

    Returns:
        DataLoader for test set, or None if test_data_dir is None
    """
    if test_data_dir is None:
        return None

    data_path = Path(test_data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

    test_systems = [
        d.name for d in data_path.iterdir()
        if d.is_dir() and d.name != 'errors' and (d / 'ligand_features.pt').exists()
    ]

    # Validate that system directories exist
    test_systems = [s for s in test_systems if (data_path / s / 'ligand_features.pt').exists()]

    if len(test_systems) == 0:
        raise ValueError(f"No valid test systems found in {test_data_dir}")

    print(f"Test systems (after validation): {len(test_systems)}")

    return get_dataloader(
        data_dir=test_data_dir,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test set
        num_workers=num_workers,
        system_ids=test_systems,
    )
