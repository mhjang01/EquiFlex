"""End-to-end inference: receptor structure (PDB preferred) + pocket residues + SMILES.

``prep.parse.parse_structure`` loads ``.pdb`` / ``.ent`` or ``.cif`` / ``.mmcif``.

Residue numbers in ``--residues`` / YAML must match parsed ``residue['resnum']``:
  * **mmCIF:** ``label_seq_id`` (integer).
  * **PDB:** 1-based index among the chain's standard amino-acid residues in file
    order (modified residues such as MSE are skipped, so it is neither the author
    number nor a plain position count; see ``pdb_resnum`` in
    ``prep.parse.gemmi_parse_pdb`` for raw PDB numbering). Use
    ``scripts/select_pocket.py`` rather than counting by hand.

Supports single-system CLI mode and batch mode via a YAML config file.

Pocket residues are selected at a 5 A heavy-atom cutoff around the ligand
(``prep.parse.get_pocket_residues``, the training definition); use
``scripts/select_pocket.py`` to compute them from a reference ligand.

Single system:
    python scripts/run_inference.py \\
        --checkpoint weights/weights.ckpt \\
        --receptor protein.pdb \\
        --residues A:42 A:43 A:44 B:10 B:11 \\
        --smiles "CC(=O)Nc1ccc(O)cc1" \\
        --guidance \\
        --output-dir ./output

Batch (YAML):
    python scripts/run_inference.py --config batch.yaml

YAML format (globals shown with the benchmark values):
    checkpoint: weights/weights.ckpt
    num_samples: 10
    num_steps: 20
    schedule: uniform               # uniform | log
    seed: 42
    device: cuda
    score_summary: false            # true -> per-sample summary.md (physics + torch Vina)
    return_trajectory: false
    trajectory_stride: 1
    trajectory_workers: 4           # 0 = auto
    force_prepare: false            # true -> re-featurise even if system_data/ exists
    output_root: ./inference_output
    guidance:
      enabled: true
      t_start: 0.0
      scale: 100.0
      schedule: linear_decay
      bond_coef: 1.0
      angle_coef: 1.0
      vdw_coef: 5.0
      vdw_active_coef: 1.0
      vdw_max_coef: 0.0
      torsion_coef: 0.25
      chiral_coef: 1.0
      chiral_buffer: 0.5236
      grad_clip: 10.0
    sde:
      enabled: false

    systems:
      CDK2_inhibitor:
        protein:
          path: /data/cdk2.pdb
          residues:                   # chain -> resnum (see note above)
            A: [10, 12, 80, 81]
        ligand:
          smiles: "CC(=O)Nc1ccc(O)cc1"
          reference_sdf: /data/cdk2_ligand.sdf   # optional; enables ligand RMSD in stats.jsonl
          ions: ["[Zn+2]"]                        # optional free metal ions near the pocket
        output_dir: ./custom_output   # optional override

      EGFR_erlotinib:
        protein:
          path: /data/egfr.pdb
          residues:
            A: [718, 719]
            B: [790, 791]
        ligand:
          smiles: "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"

Per system, ``output_dir`` receives ``system_data/`` (features), ``sample_<i>/complex.cif``
and ``sample_<i>/ligand.sdf`` per sample, ``results.pt`` (raw tensors), optionally
``summary.md`` and ``sample_<i>/trajectory_complex.cif``. One JSON line per system
(timings, atom counts, per-sample RMSD, or the failure reason) is appended to
``<output_root>/stats.jsonl``.
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from joblib import dump, load

from equiflex.prep.parse import parse_structure, overwrite_coords_on_reference_data, compute_mol_context, extract_atom_data
from equiflex.prep.reference import generate_ligand_reference_data
from equiflex.prep.featurize import ESM3Extractor, UniMolExtractor
from equiflex.prep.process import process_ligands, process_pocket_residues
from equiflex.prep.validation import (
    save_unimol_cache, validate_cached_unimol, validate_unimol_shape,
)

from equiflex.model import FlexTrainer
from equiflex.model.config import (
    FullConfig, DataConfig, OptimizerConfig, SchedulerConfig,
    EMAConfig, LoggingConfig, TrainingConfig,
)
from equiflex.model.loss import LossConfig
from equiflex.model.models import FlexConfig
from equiflex.inference.sampler import FlowMatchingSampler
from equiflex.inference.guidance import GuidanceConfig
from equiflex.inference.sampler import SDEConfig, BranchConfig
from equiflex.inference.vina_scoring import score_files, count_active_torsions
from equiflex.inference.relaxation import (
    compute_ligand_rmsd_from_coords,
    write_complex_cif,
    write_complex_trajectory_cif,
    write_ligand_sdf,
)


# Guidance terms whose numeric definition must match the loss the model was
# trained against. Any of these NOT set explicitly in the YAML is overridden
# with the checkpoint's training value by _sync_guidance_to_checkpoint.
_TRAINING_MATCHED_FIELDS = ('bond_buffer', 'angle_buffer', 'vdw_buffer', 'vdw_cutoff', 'vdw_linear_onset', 'vdw_form', 'vdw_rec_rec_ratio')

# guidance coefficient -> training loss weight. Followed only when the guidance
# spec sets ``match_training_weights: true`` (the 90M benchmark recipe uses
# hand-tuned coefficients that differ from its training weights). The native
# H-bond contact term is crystal-supervised, so it has no guidance counterpart.
_TRAINING_COEF_FIELDS = {
    'bond_coef': 'bond_length_weight', 'angle_coef': 'bond_angle_weight',
    'vdw_coef': 'vdw_weight', 'vdw_active_coef': 'vdw_active_weight',
    'vdw_max_coef': 'vdw_max_weight', 'torsion_coef': 'torsion_weight',
    'chiral_coef': 'chiral_weight',
}

# The only schedules ``inference.sampler.get_time_schedule`` implements; anything
# else raises deep inside integration, after featurisation and model loading.
_VALID_SCHEDULES = ('uniform', 'log')


def _build_guidance_config(spec) -> Tuple[Optional[GuidanceConfig], Set[str]]:
    """Build a GuidanceConfig from a dict (YAML) or None.

    Accepts a dict whose keys mirror GuidanceConfig fields. Returns
    ``(None, set())`` if spec is None / empty / explicitly disabled, so the
    sampler runs vanilla. The second element is the set of fields the caller
    set explicitly, so checkpoint syncing can leave them alone.
    """
    if not spec:
        return None, set()
    if not spec.get('enabled', True):
        return None, set()
    fields = {
        'enabled', 't_start', 'scale', 'schedule', 'grad_clip',
        'bond_coef', 'angle_coef', 'vdw_coef', 'torsion_coef',
        'chiral_coef', 'chiral_buffer', 'match_training_weights',
        'vdw_active_coef', 'vdw_max_coef', 'vdw_active_scale', 'vdw_max_alpha',
        'vdw_form', 'vdw_rec_rec_ratio',
        'bond_buffer', 'angle_buffer',
        'vdw_buffer', 'vdw_cutoff', 'vdw_linear_onset',
        'one_minus_t_min',
        'sim_mc_guidance', 'sim_mc_n', 'sim_mc_std', 'sim_mc_scale',
        'sim_mc_clash_count_coef', 'sim_mc_include_original', 'sim_mc_chunk',
    }
    kwargs = {k: v for k, v in spec.items() if k in fields and v is not None}
    kwargs.setdefault('enabled', True)
    return GuidanceConfig(**kwargs), set(kwargs)


def _sync_guidance_to_checkpoint(
    guidance_cfg: Optional[GuidanceConfig],
    explicit: Set[str],
    train_config,
) -> None:
    """Align guidance buffers/cutoffs with the checkpoint's training values.

    The guidance energy is only a meaningful proxy for "what the model was
    trained to consider physical" if its flat-bottom buffers and vdW cutoff
    match training. They are separate defaults on GuidanceConfig, so a
    checkpoint trained with e.g. vdw_buffer=0.15 / vdw_cutoff=4.0 would
    otherwise be guided by a 0.225 / 6.0 energy — a more permissive lower
    bound, and a ~3x weaker per-clash gradient via the 1/n_pairs
    normalisation. YAML-set values win; anything else follows the checkpoint.
    """
    if guidance_cfg is None or train_config is None:
        return
    loss_cfg = getattr(train_config, 'loss', train_config)
    synced = {}
    fields = list(_TRAINING_MATCHED_FIELDS)
    if guidance_cfg.match_training_weights:
        fields += list(_TRAINING_COEF_FIELDS)
    for field in fields:
        if field in explicit:
            continue
        value = getattr(loss_cfg, _TRAINING_COEF_FIELDS.get(field, field), None)
        if value is None or value == getattr(guidance_cfg, field):
            continue
        value = value if isinstance(value, (bool, str)) else float(value)
        setattr(guidance_cfg, field, value)
        synced[field] = value
    if synced:
        pairs = '  '.join(f'{k}={v}' for k, v in synced.items())
        print(f"  guidance   : synced from checkpoint -> {pairs}")


def _build_sde_config(spec) -> Optional[SDEConfig]:
    """Build an SDEConfig from a dict (YAML) or None."""
    if not spec:
        return None
    if not spec.get('enabled', True):
        return None
    fields = {'enabled', 'sigma', 'eps_w', 'w_cutoff', 'mode', 'score_drift'}
    kwargs = {k: v for k, v in spec.items() if k in fields and v is not None}
    kwargs.setdefault('enabled', True)
    return SDEConfig(**kwargs)


def _build_branch_config(spec) -> Optional[BranchConfig]:
    """Build a BranchConfig from a dict (YAML) or None."""
    if not spec:
        return None
    if not spec.get('enabled', True):
        return None
    fields = {'enabled', 'at_t', 'factor', 'max_rot_deg', 'trans_sigma'}
    kwargs = {k: v for k, v in spec.items() if k in fields and v is not None}
    kwargs.setdefault('enabled', True)
    return BranchConfig(**kwargs)


def find_reference_ligand_path(
    receptor_path: Optional[Path],
    system_id: Optional[str] = None,
    explicit_path: Optional[Path] = None,
) -> Optional[Path]:
    """Find a GT/reference ligand SDF for optional coordinate-topology RMSD."""
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        return explicit_path if explicit_path.exists() else None

    if receptor_path is None:
        return None

    receptor_path = Path(receptor_path)
    parent = receptor_path.parent
    stems = []
    if system_id:
        stems.append(system_id)
    stem = receptor_path.stem
    stems.extend([stem, stem.removesuffix('_protein')])

    candidates = []
    for st in dict.fromkeys(x for x in stems if x):
        candidates.extend([
            parent / f'{st}_ligand.sdf',
            parent / f'{st}_ligands.sdf',
        ])
    candidates.extend(sorted(parent.glob('*_ligand.sdf')))

    for candidate in candidates:
        if candidate.exists() and 'start_conf' not in candidate.name:
            return candidate
    return None

torch.serialization.add_safe_globals([
    FullConfig, DataConfig, OptimizerConfig, SchedulerConfig,
    EMAConfig, LoggingConfig, TrainingConfig, LossConfig, FlexConfig,
])

logger = logging.getLogger(__name__)


def _write_trajectory_worker(task):
    """Process-pool worker for one sample trajectory."""
    (sample, pocket_feats, ligand_feats, rec_trajectory, lig_trajectory,
     backbone_com, output_path, protein_path, stride) = task
    meta = write_complex_trajectory_cif(
        pocket_feats, ligand_feats, rec_trajectory, lig_trajectory,
        backbone_com, output_path, protein_path=protein_path, stride=stride,
    )
    return sample, {
        'path': str(meta['path']),
        'format': str(meta['format']),
        'n_frames': int(meta['n_frames']),
        'chimerax': str(meta['chimerax_hint']),
    }


def _write_sample_score_summary(
    output_dir: Path,
    sample_stats: List[Dict],
    results: Dict[str, torch.Tensor],
) -> Path:
    """Write per-sample physics and Vina scores to ``summary.md``."""
    import gemmi
    import tempfile

    physics = results.get('physics_loss')
    rows = []
    notes = []
    for stat in sample_stats:
        sample = int(stat['sample'])
        sample_dir = output_dir / f'sample_{sample}'
        complex_path = sample_dir / 'complex.cif'
        ligand_path = sample_dir / 'ligand.sdf'
        phys = float(physics[sample]) if physics is not None else None
        vina = None
        try:
            source = gemmi.read_structure(str(complex_path))
            protein = gemmi.Structure()
            protein.name = source.name
            protein.add_model(gemmi.Model('1'))
            for source_chain in source[0]:
                chain = gemmi.Chain(source_chain.name)
                for residue in source_chain:
                    info = gemmi.find_tabulated_residue(residue.name)
                    if info and info.is_amino_acid():
                        chain.add_residue(residue)
                if len(chain) > 0:
                    protein[0].add_chain(chain)
            with tempfile.NamedTemporaryFile(suffix='.pdb') as tmp:
                protein.write_pdb(tmp.name)
                nrot = count_active_torsions(ligand_path)
                vina = float(score_files(tmp.name, ligand_path, num_rot=nrot)['affinity'])
        except Exception as exc:
            notes.append(f'- `sample_{sample}` Vina scoring failed: `{exc}`')
        rows.append((sample, phys, vina))

    def fmt(value):
        return 'N/A' if value is None or not np.isfinite(value) else f'{value:.6f}'

    lines = [
        '# Inference score summary',
        '',
        '| Sample | Physics loss | Vina affinity (kcal/mol) |',
        '|---:|---:|---:|',
    ]
    lines.extend(
        f'| sample_{sample} | {fmt(phys)} | {fmt(vina)} |'
        for sample, phys, vina in rows
    )
    if physics is None:
        notes.append('- Physics loss is unavailable because physics guidance was disabled.')
    if notes:
        lines.extend(['', '## Notes', '', *notes])
    summary_path = output_dir / 'summary.md'
    summary_path.write_text('\n'.join(lines) + '\n')
    return summary_path


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def select_pocket_by_residues(
    protein_data: Dict,
    chain_residues: Dict[str, Set[int]],
):
    """Select pocket residues from a parsed structure.

    Parameters
    ----------
    protein_data : parsed structure from ``prep.parse.parse_structure``.
    chain_residues : ``{chain_id: {resnum, ...}}``.  Each *resnum* matches
        parsed ``residue['resnum']`` (CIF ``label_seq_id`` or PDB sequential index).

    Returns ``(pocket_data, sequence_data)`` in the same format as
    ``prep.parse.get_pocket_residues``.
    """
    pocket_data: Dict[str, list] = {}
    sequence_data: Dict[str, str] = {}

    for chain_id, chain_data in protein_data.items():
        if chain_id not in chain_residues:
            # still capture sequence for ESM3 if needed
            for res_key, res_data in chain_data.items():
                if res_key[0] is None and res_key[1] == 'sequence':
                    sequence_data[chain_id] = res_data
            continue

        target_resnums = chain_residues[chain_id]
        per_chain: list = []

        for res_key, res_data in chain_data.items():
            if res_key[0] is None:
                if res_key[1] == 'sequence':
                    sequence_data[chain_id] = res_data
                continue

            resnum = res_data.get('resnum', res_key[0])
            try:
                rn = int(resnum)
            except (ValueError, TypeError):
                continue

            if rn in target_resnums:
                per_chain.append(res_data)

        if per_chain:
            pocket_data[chain_id] = per_chain

    return pocket_data, sequence_data


def build_ligand_from_smiles(smiles: str) -> Dict:
    """Build a full ligand data dict from a SMILES string.

    Uses ``generate_ligand_reference_data`` for atom/bond extraction
    and conformer generation.  Sets ``coords = ref_coords`` since
    there are no experimental coordinates.
    """
    ref = generate_ligand_reference_data(smiles)
    if ref is None:
        raise ValueError(f"Failed to generate ligand data from SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    return {
        'chain': 'L',
        'smiles': canonical_smiles,
        'atoms': ref['atoms'],
        'bonds': ref['bonds'],
        'coords': ref['ref_coords'],
        'ref_coords': ref['ref_coords'],
        'constraint_matrix': ref['constraint_matrix'],
        'sp2_torsion_indices': ref['sp2_torsion_indices'],
        'chiral_atom_indices': ref['chiral_atom_indices'],
        'mapping': ref['mapping'],
    }


def build_ion_ligand(ion_smiles: str) -> Dict:
    """Build a ligand entity for a (mono/poly-atomic) ion from SMILES.

    No coordinates exist at inference (de novo), so coords/ref_coords are zeros;
    the sampler predicts ion positions from noise. Mirrors the separate-entity
    ion format used in training / test_data (own uid, own UniMol row, own block).
    """
    mol = Chem.MolFromSmiles(ion_smiles)
    if mol is None:
        raise ValueError(f"Failed to parse ion SMILES: {ion_smiles}")
    ctx = compute_mol_context(mol)
    n = mol.GetNumAtoms()
    atoms = [extract_atom_data(a, f'{a.GetSymbol().upper()}{i+1}', mol_ctx=ctx)
             for i, a in enumerate(mol.GetAtoms())]
    z = np.zeros((n, 3), dtype=np.float32)
    el = mol.GetAtomWithIdx(0).GetSymbol().upper()
    return {
        'chain': el,
        'smiles': Chem.MolToSmiles(mol),
        'atoms': atoms,
        'bonds': [],
        'coords': z.copy(),
        'ref_coords': z.copy(),
        'mapping': {i: str(i + 1) for i in range(n)},
        'sp2_torsion_indices': np.empty((0, 4), dtype=np.int32),
        'constraint_matrix': np.zeros((n, n), dtype=np.float32),
    }


def load_system_data(system_dir: Path) -> Dict[str, torch.Tensor]:
    """Load featurized system data into the dict expected by FlowMatchingSampler."""
    ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)
    unimol_emb = torch.load(system_dir / 'unimol_embeddings.pt', weights_only=False)
    validate_cached_unimol(system_dir, ligand_feats, unimol_emb)
    return assemble_system_data(
        ligand_feats=ligand_feats,
        pocket_feats=torch.load(system_dir / 'pocket_features.pt', weights_only=False),
        esm_emb=torch.load(system_dir / 'esm3_embeddings.pt', weights_only=False),
        unimol_emb=unimol_emb,
    )


def assemble_system_data(
    ligand_feats: Dict,
    pocket_feats: Dict,
    esm_emb: torch.Tensor,
    unimol_emb: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build the unbatched sampler input from in-memory feature dicts.

    Separate from the on-disk loader so ``run_hts.py`` can pair one
    featurised pocket with many ligands without a system_data/ dir per ligand.
    """
    validate_unimol_shape(ligand_feats, unimol_emb, 'in-memory inference input')
    rec_coords = pocket_feats['coords']
    lig_coords = ligand_feats['coords']

    rec_backbone_mask = pocket_feats.get('backbone_mask')
    if rec_backbone_mask is None:
        rec_backbone_mask = torch.zeros(rec_coords.shape[0], dtype=torch.float32)
    elif isinstance(rec_backbone_mask, np.ndarray):
        rec_backbone_mask = torch.from_numpy(rec_backbone_mask).float()

    rec_cb_mask = pocket_feats.get('cb_mask')
    if rec_cb_mask is None:
        rec_cb_mask = torch.zeros(rec_coords.shape[0], dtype=torch.float32)
    elif isinstance(rec_cb_mask, np.ndarray):
        rec_cb_mask = torch.from_numpy(rec_cb_mask).float()

    rec_missing_mask = pocket_feats.get('missing_mask')
    if rec_missing_mask is None:
        rec_missing_mask = torch.zeros(rec_coords.shape[0], dtype=torch.bool)
    elif isinstance(rec_missing_mask, np.ndarray):
        rec_missing_mask = torch.from_numpy(rec_missing_mask).bool()
    else:
        rec_missing_mask = rec_missing_mask.bool()

    backbone_atom_indices = pocket_feats.get('backbone_atom_indices')
    valid_backbone_mask = rec_backbone_mask * (~rec_missing_mask).float()

    if valid_backbone_mask.sum() > 0:
        backbone_com = (
            (rec_coords * valid_backbone_mask.unsqueeze(-1)).sum(0)
            / valid_backbone_mask.sum()
        )
    else:
        backbone_com = rec_coords.mean(0)

    rec_coords_centered = rec_coords - backbone_com
    lig_coords_centered = lig_coords - backbone_com

    def _load_constraint(feats, label):
        # prep raises on a missing constraint matrix (prep/process.py), so any
        # featurised system has one. No zeros fallback: zeros read as "no pair
        # info" to the model but as "every bond must be 0 A long" to the physics
        # guidance, which consumes the same tensor as literal distance bounds.
        cm = feats.get('constraint_matrix', None)
        if cm is None:
            raise KeyError(
                f"{label}_features.pt has no 'constraint_matrix' — re-run "
                f"featurisation for this system (prep writes it unconditionally)."
            )
        if isinstance(cm, np.ndarray):
            cm = torch.from_numpy(cm).float()
        if cm.ndim == 2:
            cm = cm.unsqueeze(-1)
        return cm

    def _load_indices(feats, key, label):
        # Absent key != empty index set. Absent means the .pt predates the feature
        # (see _system_data_is_current); empty means the molecule genuinely has no
        # such centres. Both reach guidance as a silent no-op, so separate them here.
        if key not in feats:
            raise KeyError(
                f"{label}_features.pt has no '{key}' — this featurisation predates "
                f"the feature, so the guidance term that consumes it would be "
                f"silently skipped. Delete the system_data dir or re-run with "
                f"--force-prepare."
            )
        return feats[key]

    N_rec = rec_coords.shape[0]
    N_lig = lig_coords.shape[0]

    return {
        'rec_atom_feats':        pocket_feats['atom_features'],
        'rec_atom_coords':       rec_coords_centered,
        'rec_atom_ref_coords':   pocket_feats['ref_coords'],
        'rec_bond_feats':        pocket_feats['bond_features'],
        'rec_uid_vec':           pocket_feats['residue_uid_vec'],
        'rec_backbone_mask':     rec_backbone_mask,
        'rec_cb_mask':           rec_cb_mask,
        'rec_missing_mask':      rec_missing_mask,
        'rec_backbone_atom_indices': backbone_atom_indices,
        'rec_constraint_matrix': _load_constraint(pocket_feats, 'pocket'),
        'rec_sp2_torsion_indices': _load_indices(pocket_feats, 'sp2_torsion_indices', 'pocket'),
        'esm_embeddings':        esm_emb,
        'lig_atom_feats':        ligand_feats['atom_features'],
        'lig_atom_coords':       lig_coords_centered,
        'lig_atom_ref_coords':   ligand_feats['ref_coords'],
        'lig_bond_feats':        ligand_feats['bond_features'],
        'lig_uid_vec':           ligand_feats['ligand_uid_vec'],
        'lig_constraint_matrix': _load_constraint(ligand_feats, 'ligand'),
        'lig_sp2_torsion_indices': _load_indices(ligand_feats, 'sp2_torsion_indices', 'ligand'),
        'lig_chiral_indices':     _load_indices(ligand_feats, 'chiral_atom_indices', 'ligand'),
        'unimol_embeddings':     unimol_emb,
        '_backbone_com':   backbone_com,
        '_n_rec':          N_rec,
        '_n_lig':          N_lig,
        '_n_rec_tokens':   esm_emb.shape[0],
    }


# ---------------------------------------------------------------------------
# Featurization pipeline
# ---------------------------------------------------------------------------

def prepare_system(
    receptor_path: Path,
    chain_residues: Dict[str, Set[int]],
    smiles: str,
    system_dir: Path,
    device: str = 'cuda',
    esm_extractor: Optional[ESM3Extractor] = None,
    unimol_extractor: Optional[UniMolExtractor] = None,
    ions: Optional[List[str]] = None,
):
    """Full pipeline: parse -> pocket selection -> featurize -> embed.

    Parameters
    ----------
    chain_residues : ``{chain_id: {resnum, ...}}``.

    When *esm_extractor* / *unimol_extractor* are provided they are
    reused (avoids reloading large models per system in batch mode).
    """
    system_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse receptor
    print(f"  Parsing receptor: {receptor_path}")
    protein_data = parse_structure(receptor_path)
    if protein_data is None:
        raise RuntimeError(f"Failed to parse receptor: {receptor_path}")

    # 2. Select pocket residues
    pocket_data, sequence_data = select_pocket_by_residues(
        protein_data, chain_residues,
    )
    if not pocket_data or all(len(v) == 0 for v in pocket_data.values()):
        req = {c: sorted(rs) for c, rs in chain_residues.items()}
        raise RuntimeError(
            f"No residues matched. Requested: {req}. "
            f"Available chains: {[c for c in protein_data if not isinstance(c, tuple)]}"
        )
    total_res = sum(len(v) for v in pocket_data.values())
    print(f"  Pocket: {total_res} residues from chain(s) {list(pocket_data.keys())}")

    # 3. Align to reference templates
    pocket_residues = overwrite_coords_on_reference_data(pocket_data)

    # 4. Ligand from SMILES
    ligand_data = build_ligand_from_smiles(smiles)
    ligands = [ligand_data]
    print(f"  Ligand: {len(ligand_data['atoms'])} heavy atoms")

    # Append ions as separate ligand entities (own uid / block / UniMol row)
    for ion_smiles in (ions or []):
        ligands.append(build_ion_ligand(ion_smiles))
    if ions:
        print(f"  Ions: {len(ions)} ({', '.join(ions)})")

    # 5. Save PKLs
    dump(ligands, system_dir / 'ligands.pkl')
    dump(pocket_residues, system_dir / 'pocket_residues.pkl')
    dump(sequence_data, system_dir / 'sequence_data.pkl')

    # 6. Featurize
    ligand_features = process_ligands(ligands)
    pocket_features = process_pocket_residues(pocket_residues)
    if pocket_features is None:
        raise RuntimeError("Failed to featurize pocket residues")
    torch.save(ligand_features, system_dir / 'ligand_features.pt')
    torch.save(pocket_features, system_dir / 'pocket_features.pt')

    # 7. ESM3
    own_esm = esm_extractor is None
    if own_esm:
        esm_extractor = ESM3Extractor(device=device)
    pocket_embeddings = esm_extractor.extract_pocket_embeddings(
        sequence_data, pocket_residues, system_dir=system_dir,
    )
    residue_keys = pocket_features['residue_keys']
    emb_list = [
        pocket_embeddings.get((cid, rn, eid), torch.zeros(1536))
        for cid, _, rn, eid in residue_keys
    ]
    torch.save(torch.stack(emb_list), system_dir / 'esm3_embeddings.pt')
    if own_esm:
        del esm_extractor

    # 8. UniMol
    own_unimol = unimol_extractor is None
    if own_unimol:
        unimol_extractor = UniMolExtractor(model_size='1.1B')
    embeddings = unimol_extractor.extract_batch(
        [lig['smiles'] for lig in ligands],
        [lig['atoms'] for lig in ligands],
    )
    n_atoms_per_lig = ligand_features['n_atoms_per_ligand']
    all_emb = []
    for i, emb in enumerate(embeddings):
        all_emb.append(emb if emb is not None else torch.zeros(n_atoms_per_lig[i], 1536))
    save_unimol_cache(system_dir, ligand_features, ligands, torch.cat(all_emb))
    if own_unimol:
        del unimol_extractor


# ---------------------------------------------------------------------------
# Single-system sampling + output writing
# ---------------------------------------------------------------------------

def run_single_system(
    system_dir: Path,
    output_dir: Path,
    sampler: FlowMatchingSampler,
    num_samples: int,
    num_steps: int,
    schedule: str,
    seed: int,
    score_summary: bool = False,
    return_trajectory: bool = False,
    trajectory_stride: int = 1,
    trajectory_workers: int = 0,
    protein_path: Optional[Path] = None,
    reference_ligand_path: Optional[Path] = None,
    guidance: Optional[GuidanceConfig] = None,
    sde: Optional[SDEConfig] = None,
    branch: Optional[BranchConfig] = None,
):
    """Sample and write outputs for one featurized system.

    Returns a stats dict suitable for writing as a JSONL record.
    """
    data = load_system_data(system_dir)
    n_rec = int(data['_n_rec'])
    n_lig = int(data['_n_lig'])
    print(f"  rec={n_rec} atoms, lig={n_lig} atoms")

    t0 = time.perf_counter()
    results = sampler.sample(
        data=data,
        num_samples=num_samples,
        num_steps=num_steps,
        schedule=schedule,
        seed=seed,
        return_trajectory=return_trajectory,
        guidance=guidance,
        sde=sde,
        branch=branch,
    )
    inference_s = round(time.perf_counter() - t0, 3)
    print(f"  inference: {inference_s:.1f}s  ({num_samples} samples × {num_steps} steps)")

    backbone_com = results['backbone_com']
    ligands_data = load(system_dir / 'ligands.pkl')
    pocket_feats = torch.load(system_dir / 'pocket_features.pt', weights_only=False)
    ligand_feats = torch.load(system_dir / 'ligand_features.pt', weights_only=False)

    B = results['lig_coords_final'].shape[0]
    if reference_ligand_path is not None:
        reference_ligand_path = Path(reference_ligand_path)
        if not reference_ligand_path.exists():
            print(f"  reference ligand not found, skipping RMSD: {reference_ligand_path}")
            reference_ligand_path = None
        else:
            print(f"  reference ligand RMSD: {reference_ligand_path}")
            results['ligand_rmsd'] = torch.full((B,), float('nan'))

    sample_stats: List[Dict] = []

    for b in range(B):
        sample_dir = output_dir / f'sample_{b}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        rec_b = results['rec_coords_final'][b]
        lig_b = results['lig_coords_final'][b]

        write_complex_cif(
            pocket_feats, ligand_feats,
            rec_b, lig_b, backbone_com,
            sample_dir / 'complex.cif',
            protein_path=protein_path,
        )
        write_ligand_sdf(
            ligands_data[0], lig_b, backbone_com,
            sample_dir / 'ligand.sdf',
        )

        s_stat: Dict = {'sample': b}
        if reference_ligand_path is not None:
            rmsd = compute_ligand_rmsd_from_coords(
                ligands_data[0], lig_b, backbone_com, reference_ligand_path,
            )
            s_stat['ligand_rmsd'] = round(rmsd, 4) if np.isfinite(rmsd) else None
            results['ligand_rmsd'][b] = float(rmsd)

        sample_stats.append(s_stat)

    if return_trajectory and results.get('rec_trajectory') is not None:
        workers = trajectory_workers or min(B, max(1, os.cpu_count() or 1))
        workers = max(1, min(int(workers), B))
        tasks = [
            (b, pocket_feats, ligand_feats, results['rec_trajectory'][b],
             results['lig_trajectory'][b], backbone_com,
             output_dir / f'sample_{b}' / 'trajectory_complex.cif',
             protein_path, trajectory_stride)
            for b in range(B)
        ]
        # CIF writing is CPU-only. Fork avoids re-importing the model stack in
        # every worker; workers never touch CUDA tensors (results are on CPU).
        context = multiprocessing.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
        ) as pool:
            for sample, meta in pool.map(_write_trajectory_worker, tasks):
                sample_stats[sample]['trajectory'] = meta
                print(
                    f"    sample_{sample}: trajectory {meta['format']} "
                    f"({meta['n_frames']} frames) -> {meta['path']}"
                )

    if score_summary:
        summary_path = _write_sample_score_summary(output_dir, sample_stats, results)
        print(f"  score summary: {summary_path}")
    torch.save(results, output_dir / 'results.pt')

    return {
        'n_rec': n_rec,
        'n_lig': n_lig,
        'n_samples': B,
        'n_steps': num_steps,
        'inference_s': inference_s,
        'samples': sample_stats,
    }


# ---------------------------------------------------------------------------
# YAML config parsing
# ---------------------------------------------------------------------------

def _parse_residues_spec(raw_residues) -> Dict[str, Set[int]]:
    """Convert YAML ``residues`` into ``{chain_id: {resnum, ...}}``.

    Preferred (readable, avoids ambiguous nested lists).  Integers are
    ``residue['resnum']`` after ``parse_structure`` (CIF label_seq_id or PDB chain index)::

        residues:
          A: [36, 99, 100, 103]
          B: [35, 37, 108]
          C: [95, 137, 141]

    Also accepts a list of ``[chain, resnum]`` pairs. Use **flow** pairs so
    PyYAML does not emit the confusing ``- - A`` block form::

        residues:
          - [A, 42]
          - [B, 10]
    """
    if isinstance(raw_residues, dict):
        out: Dict[str, Set[int]] = {}
        for chain_id, nums in raw_residues.items():
            if not isinstance(nums, (list, tuple)):
                raise ValueError(
                    f"residues[{chain_id!r}] must be a list of integers, got {type(nums)}"
                )
            out[str(chain_id)] = {int(n) for n in nums}
        return out

    if isinstance(raw_residues, list):
        chain_residues: Dict[str, Set[int]] = {}
        for entry in raw_residues:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Each residue entry must be [chain, resnum], got: {entry}"
                )
            chain_id, resnum = str(entry[0]), int(entry[1])
            chain_residues.setdefault(chain_id, set()).add(resnum)
        return chain_residues

    raise ValueError(
        f"protein.residues must be a dict chain -> [resnums] or a list of pairs, "
        f"got {type(raw_residues)}"
    )


def parse_yaml_config(config_path: Path) -> Dict:
    """Parse a batch YAML config and return a normalised dict.

    Returns
    -------
    dict with keys:
        ``globals`` — shared settings (checkpoint, num_samples, …)
        ``systems`` — list of per-system dicts with keys:
            id, receptor, chain_residues, smiles, output_dir
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    systems_raw = raw.pop('systems', {})
    if not systems_raw:
        raise ValueError("YAML config has no 'systems' section")

    output_root = Path(raw.pop('output_root', './inference_output'))

    systems = []
    for sys_id, spec in systems_raw.items():
        protein = spec.get('protein', {})
        ligand = spec.get('ligand', {})

        receptor = protein.get('path')
        if receptor is None:
            raise ValueError(f"System '{sys_id}' missing protein.path")
        smiles = ligand.get('smiles')
        if smiles is None:
            raise ValueError(f"System '{sys_id}' missing ligand.smiles")
        residues = protein.get('residues')
        if not residues:
            raise ValueError(f"System '{sys_id}' missing protein.residues")

        sys_out = spec.get('output_dir')
        if sys_out is not None:
            sys_out = Path(sys_out)
        else:
            sys_out = output_root / sys_id

        reference_ligand = (
            ligand.get('reference_sdf')
            or ligand.get('reference_ligand')
            or ligand.get('reference_path')
        )

        systems.append({
            'id': sys_id,
            'receptor': Path(receptor),
            'chain_residues': _parse_residues_spec(residues),
            'smiles': smiles,
            'ions': ligand.get('ions'),  # optional list of ion SMILES, e.g. ['[Mg+2]']
            'reference_ligand': Path(reference_ligand) if reference_ligand else None,
            'output_dir': sys_out,
        })

    return {'globals': raw, 'output_root': output_root, 'systems': systems}


# ---------------------------------------------------------------------------
# Featurisation currency
# ---------------------------------------------------------------------------

_REQUIRED_SYSTEM_FILES = (
    'ligand_features.pt', 'pocket_features.pt',
    'esm3_embeddings.pt', 'unimol_embeddings.pt',
)
# Existence is not enough: a system_data dir written before a feature was added
# loads fine but silently disables the guidance term that consumes it (e.g. a
# pre-chiral_atom_indices ligand_features.pt makes chiral_coef a no-op). Check
# the schema too, so stale dirs re-prepare instead of degrading in silence.
_REQUIRED_FEATURE_KEYS = {
    'ligand_features.pt': ('constraint_matrix', 'sp2_torsion_indices',
                           'chiral_atom_indices'),
    'pocket_features.pt': ('constraint_matrix', 'sp2_torsion_indices',
                           'cb_mask', 'missing_mask'),
}

def system_data_is_current(sys_dir: Path) -> Optional[str]:
    """None if ``sys_dir`` is a usable featurisation, else the reason it must be re-prepared."""
    for fname in _REQUIRED_SYSTEM_FILES:
        if not (sys_dir / fname).exists():
            return f'missing {fname}'
    for fname, keys in _REQUIRED_FEATURE_KEYS.items():
        try:
            feats = torch.load(sys_dir / fname, map_location='cpu', weights_only=False)
        except Exception as exc:
            return f'{fname} unreadable ({exc})'
        absent = [k for k in keys if k not in feats]
        if absent:
            return f'{fname} predates {"/".join(absent)}'
    try:
        ligand_feats = torch.load(sys_dir / 'ligand_features.pt', map_location='cpu', weights_only=False)
        unimol_emb = torch.load(sys_dir / 'unimol_embeddings.pt', map_location='cpu', weights_only=False)
        validate_cached_unimol(sys_dir, ligand_feats, unimol_emb)
    except Exception as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end inference: receptor PDB/CIF + pocket residues + SMILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode selection ────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    # No default: argparse fills defaults even for un-passed members of a
    # mutually exclusive group, so a default here makes `args.config` always
    # truthy and silently swallows --receptor single-system mode.
    mode.add_argument('--config', type=str, default=None,
                      help='Path to YAML config for batch inference')
    mode.add_argument('--receptor', type=str,
                      help='Path to receptor .pdb / .ent / .cif / .mmcif (single-system mode)')

    # Single-system inputs (ignored in --config mode)
    parser.add_argument('--residues', type=str, nargs='+',
                        help='Pocket residues as chain:resnum pairs, '
                             'e.g. A:42 A:43 B:10')
    parser.add_argument('--smiles', type=str,
                        help='Ligand SMILES string')
    parser.add_argument('--ions', type=str, nargs='+',
                        help='Optional ion SMILES added as separate ligand entities, e.g. "[Mg+2]" "[Zn+2]"')
    parser.add_argument('--reference-ligand', type=str,
                        help='Optional reference ligand SDF for coordinate/topology RMSD')
    parser.add_argument('--output-dir', type=str, default='./inference_output')

    # Shared options (also settable in YAML globals)
    parser.add_argument('--checkpoint', type=str, default='weights/weights.ckpt',
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--schedule', type=str, default='uniform',
                        choices=_VALID_SCHEDULES,
                        help='Time schedule for Euler integration '
                             '(inference.sampler.get_time_schedule)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Re-run featurization even if system_data/ already exists')
    parser.add_argument('--return-trajectory', action='store_true', default=False)
    parser.add_argument('--trajectory-stride', type=int, default=1,
                        help='When saving trajectory CIF, keep every N-th integration frame')
    parser.add_argument('--trajectory-workers', type=int, default=0,
                        help='Trajectory writer processes; 0 selects automatically')

    guide_group = parser.add_argument_group(
        'guidance', 'Tweedie x_1 physics guidance during integration')
    guide_group.add_argument('--guidance', action='store_true',
                             help='Enable Tweedie x_1 physics guidance')
    guide_group.add_argument('--guide-t-start', type=float, default=0.0)
    guide_group.add_argument('--guide-scale', type=float, default=100.0,
                             help='cov-A guidance scale lambda (default: 100.0, the benchmark value)')
    guide_group.add_argument('--guide-schedule', type=str, default='linear_decay',
                             choices=['const', 'linear_decay', 'cosine_decay', 'exp_decay'],
                             help='Decaying-in-t cov-A schedule (default: linear_decay)')
    guide_group.add_argument('--guide-bond-coef', type=float, default=1.0)
    guide_group.add_argument('--guide-angle-coef', type=float, default=1.0)
    guide_group.add_argument('--guide-vdw-coef', type=float, default=5.0)
    guide_group.add_argument('--guide-vdw-active-coef', type=float, default=1.0)
    guide_group.add_argument('--guide-vdw-max-coef', type=float, default=0.0)
    guide_group.add_argument('--guide-torsion-coef', type=float, default=0.25)
    guide_group.add_argument('--guide-chiral-coef', type=float, default=1.0)
    guide_group.add_argument('--guide-chiral-buffer', type=float, default=0.5236)
    guide_group.add_argument('--guide-grad-clip', type=float, default=10.0)
    guide_group.add_argument('--guide-sim-mc', action='store_true',
                             help='Use Cartesian Sim-MC x1 proposals instead of gradient guidance')
    guide_group.add_argument('--guide-sim-mc-n', type=int, default=32,
                             help='Number of Sim-MC candidates per sample')
    guide_group.add_argument('--guide-sim-mc-std', type=float, default=0.10,
                             help='Cartesian proposal sigma in Angstrom')
    guide_group.add_argument('--guide-sim-mc-scale', type=float, default=5.0,
                             help='Softmax inverse-temperature for Sim-MC candidate weights')
    guide_group.add_argument('--guide-sim-mc-clash-count-coef', type=float, default=0.0,
                             help='Extra energy penalty per active vdW clash')
    br_group = parser.add_argument_group(
        'branch', 'Mid-flight trajectory branching (diversifies poses)')
    br_group.add_argument('--branch', action='store_true',
                          help='duplicate each trajectory at --branch-at with a '
                               'rotated ligand copy (factor-1 extra copies)')
    br_group.add_argument('--branch-at', type=float, default=0.5,
                          help='time at which to branch (default: 0.5)')
    br_group.add_argument('--branch-factor', type=int, default=2,
                          help='copies per trajectory including the original')
    br_group.add_argument('--branch-max-rot-deg', type=float, default=180.0,
                          help='max ligand rotation applied to extra copies')
    br_group.add_argument('--branch-trans-sigma', type=float, default=1.0,
                          help='Gaussian translation sigma in Angstrom')
    sde_group = parser.add_argument_group(
        'sde', 'Stochastic Euler-Maruyama integration (adds diversity)')
    sde_group.add_argument('--sde', action='store_true',
                           help='Enable SDE integration (sigma>0 injects noise per step)')
    sde_group.add_argument('--sde-sigma', type=float, default=0.1,
                           help='Noise scale; 0 = deterministic. Default: 0.1')
    sde_group.add_argument('--sde-eps-w', type=float, default=0.01,
                           help='Epsilon in diffusion-coefficient denominator')
    sde_group.add_argument('--sde-w-cutoff', type=float, default=0.99,
                           help='Zero diffusion above this t')
    sde_group.add_argument('--sde-score-drift', action='store_true',
                           help='marginal-preserving OU step (adds the score drift '
                                'implied by the interpolant) instead of bare noise')
    sde_group.add_argument('--sde-mode', type=str, default='both',
                           choices=['both', 'receptor', 'ligand'],
                           help='Which subsystem gets SDE noise (default: both)')

    args = parser.parse_args()

    # ── Build system list ─────────────────────────────────────────────
    if args.config:
        cfg = parse_yaml_config(Path(args.config))
        g = cfg['globals']

        checkpoint = g.get('checkpoint', args.checkpoint)
        num_samples = g.get('num_samples', args.num_samples)
        num_steps = g.get('num_steps', args.num_steps)
        schedule = g.get('schedule', args.schedule)
        if schedule not in _VALID_SCHEDULES:
            parser.error(
                f"schedule {schedule!r} in {args.config} is not supported; "
                f"choose one of {list(_VALID_SCHEDULES)}"
            )
        seed = g.get('seed', args.seed)
        device = g.get('device', args.device)
        score_summary = bool(g.get('score_summary', False))
        force_prepare = g.get('force_prepare', args.force_prepare)
        return_traj = g.get('return_trajectory', args.return_trajectory)
        traj_stride = g.get('trajectory_stride', args.trajectory_stride)
        traj_workers = g.get('trajectory_workers', args.trajectory_workers)
        guidance_cfg, guidance_explicit = _build_guidance_config(g.get('guidance'))
        sde_cfg = _build_sde_config(g.get('sde'))
        branch_cfg = _build_branch_config(g.get('branch'))
        systems = cfg['systems']
        stats_root = cfg['output_root']
    else:
        if not args.residues or not args.smiles:
            parser.error("--receptor mode requires --residues and --smiles")
        if not args.checkpoint:
            parser.error("--checkpoint is required")

        # Parse chain:resnum pairs from CLI
        chain_residues: Dict[str, Set[int]] = {}
        for token in args.residues:
            if ':' not in token:
                parser.error(
                    f"Residue '{token}' must be chain:resnum, e.g. A:42"
                )
            chain_id, resnum_str = token.split(':', 1)
            chain_residues.setdefault(chain_id, set()).add(int(resnum_str))

        checkpoint = args.checkpoint
        num_samples = args.num_samples
        num_steps = args.num_steps
        schedule = args.schedule
        seed = args.seed
        device = args.device
        score_summary = False
        force_prepare = args.force_prepare
        return_traj = args.return_trajectory
        traj_stride = args.trajectory_stride
        traj_workers = args.trajectory_workers
        guidance_cfg, guidance_explicit = _build_guidance_config({
            'enabled': args.guidance,
            't_start': args.guide_t_start,
            'scale': args.guide_scale,
            'schedule': args.guide_schedule,
            'bond_coef': args.guide_bond_coef,
            'angle_coef': args.guide_angle_coef,
            'vdw_coef': args.guide_vdw_coef,
            'vdw_active_coef': args.guide_vdw_active_coef,
            'vdw_max_coef': args.guide_vdw_max_coef,
            'torsion_coef': args.guide_torsion_coef,
            'chiral_coef': args.guide_chiral_coef,
            'chiral_buffer': args.guide_chiral_buffer,
            'grad_clip': args.guide_grad_clip,
            'sim_mc_guidance': args.guide_sim_mc,
            'sim_mc_n': args.guide_sim_mc_n,
            'sim_mc_std': args.guide_sim_mc_std,
            'sim_mc_scale': args.guide_sim_mc_scale,
            'sim_mc_clash_count_coef': args.guide_sim_mc_clash_count_coef,
        })
        branch_cfg = _build_branch_config({
            'enabled': args.branch,
            'at_t': args.branch_at,
            'factor': args.branch_factor,
            'max_rot_deg': args.branch_max_rot_deg,
            'trans_sigma': args.branch_trans_sigma,
        })
        sde_cfg = _build_sde_config({
            'enabled': args.sde,
            'sigma': args.sde_sigma,
            'score_drift': args.sde_score_drift,
            'eps_w': args.sde_eps_w,
            'w_cutoff': args.sde_w_cutoff,
            'mode': args.sde_mode,
        })
        stats_root = Path(args.output_dir)

        systems = [{
            'id': 'system',
            'receptor': Path(args.receptor),
            'chain_residues': chain_residues,
            'smiles': args.smiles,
            'ions': args.ions,
            'reference_ligand': Path(args.reference_ligand) if args.reference_ligand else None,
            'output_dir': Path(args.output_dir),
        }]

    if not checkpoint:
        parser.error("checkpoint is required (via --checkpoint or YAML globals)")

    n_sys = len(systems)
    print(f"{'='*60}")
    print(f"Flex Inference — {n_sys} system(s)")
    print(f"  checkpoint : {checkpoint}")
    print(f"  samples    : {num_samples}  steps: {num_steps}  schedule: {schedule}")
    print(f"  device     : {device}")
    print(f"  summaries  : {'physics + Vina' if score_summary else 'disabled'}")
    if return_traj:
        worker_label = traj_workers or 'auto'
        print(f"  trajectory : save CIF (stride={traj_stride}, workers={worker_label})")
    if guidance_cfg is not None:
        guide_mode = "sim-mc" if guidance_cfg.sim_mc_guidance else "cov-A"
        print(
            f"  guidance   : mode={guide_mode} t_start={guidance_cfg.t_start} "
            f"scale={guidance_cfg.scale} schedule={guidance_cfg.schedule} "
            f"bond={guidance_cfg.bond_coef} angle={guidance_cfg.angle_coef} "
            f"vdw={guidance_cfg.vdw_coef} torsion={guidance_cfg.torsion_coef}"
        )
        if guidance_cfg.sim_mc_guidance:
            print(
                f"  sim-mc     : n={guidance_cfg.sim_mc_n} std={guidance_cfg.sim_mc_std} "
                f"scale={guidance_cfg.sim_mc_scale} "
                f"clash_count={guidance_cfg.sim_mc_clash_count_coef}"
            )
    if sde_cfg is not None:
        print(
            f"  sde        : mode={sde_cfg.mode}  sigma={sde_cfg.sigma}  "
            f"eps_w={sde_cfg.eps_w}  w_cutoff={sde_cfg.w_cutoff}"
        )
    print(f"{'='*60}\n")

    # ── Step 1: Prepare systems (skip per-system if data exists) ─────
    stale_reasons = {}
    systems_to_prepare = []
    for s in systems:
        if force_prepare:
            systems_to_prepare.append(s)
            continue
        reason = system_data_is_current(s['output_dir'] / 'system_data')
        if reason is not None:
            systems_to_prepare.append(s)
            stale_reasons[s['id']] = reason

    prepare_failures: Dict[str, str] = {}
    if systems_to_prepare:
        print(f"Preparing {len(systems_to_prepare)}/{n_sys} system(s) "
              f"({'forced' if force_prepare else 'missing or outdated data'})...")
        outdated = {k: v for k, v in stale_reasons.items() if 'predates' in v}
        if outdated:
            sample = list(outdated.items())[:3]
            print(f"  {len(outdated)} system(s) have outdated features and will be "
                  f"re-featurised, e.g. " +
                  '; '.join(f'{k}: {v}' for k, v in sample))
        print("Loading embedding models (shared across systems)...")
        esm_ext = ESM3Extractor(device=device)
        unimol_ext = UniMolExtractor(model_size='1.1B')

        for i, s in enumerate(systems_to_prepare, 1):
            sys_dir = s['output_dir'] / 'system_data'
            print(f"\n[{i}/{len(systems_to_prepare)}] Preparing: {s['id']}")
            try:
                prepare_system(
                    receptor_path=s['receptor'],
                    chain_residues=s['chain_residues'],
                    smiles=s['smiles'],
                    system_dir=sys_dir,
                    device=device,
                    esm_extractor=esm_ext,
                    unimol_extractor=unimol_ext,
                    ions=s.get('ions'),
                )
                print(f"  Preparation complete.")
            except Exception as e:
                prepare_failures[s['id']] = repr(e)
                print(f"  [FAIL] {e}")

        del esm_ext, unimol_ext
        torch.cuda.empty_cache()
    else:
        print(f"All {n_sys} system(s) already prepared — reusing existing data.")

    # ── Step 2: Load model ────────────────────────────────────────────
    print(f"\nLoading model: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    trainer = FlexTrainer.load_from_checkpoint(checkpoint, map_location='cpu')

    _sync_guidance_to_checkpoint(
        guidance_cfg, guidance_explicit, ckpt.get('hyper_parameters', {}).get('config'),
    )

    if 'ema_state_dict' in ckpt:
        shadow = ckpt['ema_state_dict']['shadow']
        for name, param in trainer.model.named_parameters():
            if name in shadow:
                param.data = shadow[name]
        print(f"Applied EMA weights (decay={ckpt['ema_state_dict']['decay']})")
    del ckpt

    sampler = FlowMatchingSampler(trainer.model, device=device)
    del trainer
    print(f"Model loaded. Device: {device}\n")

    # ── Step 3: Sample each system ────────────────────────────────────
    stats_file = Path(stats_root) / 'stats.jsonl'
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    n_done, n_fail = 0, 0
    with open(stats_file, 'a') as sf:
        for i, s in enumerate(systems, 1):
            sys_dir = s['output_dir'] / 'system_data'
            # Gate on the prepare outcome, not on file presence: prepare_system writes
            # ligand/pocket features before the ESM3/UniMol stages, so a failed
            # re-featurisation leaves a fresh pocket_features.pt beside stale
            # embeddings and the sampler would run on the mismatch.
            reason = prepare_failures.get(s['id']) or system_data_is_current(sys_dir)
            if reason is not None:
                print(f"[{i}/{n_sys}] {s['id']}: skipped ({reason})")
                sf.write(json.dumps({'system': s['id'], 'error': reason}) + '\n')
                sf.flush()
                n_fail += 1
                continue

            print(f"[{i}/{n_sys}] Sampling: {s['id']}")
            try:
                stats = run_single_system(
                    system_dir=sys_dir,
                    output_dir=s['output_dir'],
                    sampler=sampler,
                    num_samples=num_samples,
                    num_steps=num_steps,
                    schedule=schedule,
                    seed=seed,
                    score_summary=score_summary,
                    return_trajectory=return_traj,
                    trajectory_stride=traj_stride,
                    trajectory_workers=traj_workers,
                    protein_path=Path(s['receptor']) if s.get('receptor') else None,
                    reference_ligand_path=find_reference_ligand_path(
                        Path(s['receptor']) if s.get('receptor') else None,
                        system_id=s.get('id'),
                        explicit_path=s.get('reference_ligand'),
                    ),
                    guidance=guidance_cfg,
                    sde=sde_cfg,
                    branch=branch_cfg,
                )
                stats['system'] = s['id']
                sf.write(json.dumps(stats) + '\n')
                sf.flush()
                n_done += 1
                print(f"  -> {s['output_dir']}")
            except Exception as e:
                n_fail += 1
                sf.write(json.dumps({'system': s['id'], 'error': repr(e)}) + '\n')
                sf.flush()
                print(f"  [FAIL] {e}")

    print(f"\n{'='*60}")
    print(f"Done: {n_done}  Failed: {n_fail}")
    print(f"Stats: {stats_file}")


if __name__ == '__main__':
    main()
