"""High-throughput screening: one receptor template + a SMILES library.

The pocket is taken from a reference ligand in the template structure (its
CCD / residue name, or an external SDF) at the training cutoff
(``prep.parse.get_pocket_residues``, 5.0 A). Free metal ions next to the
reference ligand are detected automatically and sampled as separate entities.

Work is split so the three large models are never resident together and every
stage resumes from disk:

  1. receptor/   pocket parsed + featurised once, ESM3 embedded once
  2. ligands/    RDKit conformers + featurisation in a process pool, UniMol
                 embedded in batches on the GPU, written as shards
  3. poses/      each ligand sampled against the same pocket; every pose is
                 written and scored with Vina in a background CPU pool while
                 the GPU moves on to the next ligand

    python scripts/run_hts.py --config hts.yaml [--limit N] [--force-prepare]

YAML:
    checkpoint: weights/weights.ckpt
    output_root: ./hts_output
    device: cuda
    prep_device: cuda               # ESM3 / UniMol device; cpu if the GPU cannot hold them (~10 GB)
    seed: 42
    num_samples: 8
    num_steps: 20
    schedule: uniform               # uniform | log
    return_trajectory: false
    trajectory_stride: 1
    keep_samples: best              # best | all   (pose files kept per ligand)
    force_prepare: false
    guidance: {...}                 # same keys as run_inference.py
    sde: {...}

    receptor:
      path: /data/5SAK.pdb          # template holding the reference ligand (.pdb/.cif)
      reference:
        ccd: ZRY                    # residue name of the reference ligand
        chain: A                    # optional: disambiguate multiple copies
        resnum: 404                 # optional: author residue number
        # sdf: /data/ref.sdf        # alternative: external reference ligand
      pocket_cutoff: 5.0
      ions: auto                    # auto | [] | ['[Zn+2]']
      # residues: {A: [35, 79]}     # optional: explicit pocket, skips extraction

    ligands:
      path: library.smi             # .smi/.txt (SMILES [name]) | .csv/.tsv | .sdf
      smiles_column: smiles         # csv/tsv only
      name_column: name             # csv/tsv only, optional
      limit: null
      largest_fragment: true        # drop counter-ions / solvent fragments
      workers: 0                    # conformer + featurisation processes (0 = auto)
      unimol_batch_size: 128        # SMILES per UniMol call
      shard_size: 256               # ligands per shard file

    scoring:
      backend: torch                # torch (vina_scoring port) | binary (vina) | smina (any smina function)
      smina_scoring: vinardo        # smina backend only; also selects the kept pose
      workers: 8
      vina: vina                    # binary backend only
      obabel: obabel                # torsion count (torch) / pdbqt conversion (binary)

Outputs under ``output_root``:
    receptor/pocket.yaml, pocket_features.pt, esm3_embeddings.pt, reference_ligand.pdb
    ligands/manifest.csv, library.yaml, shard_XXXX.pt
    poses/<lig_id>/sample_<i>/{complex.cif, ligand.sdf}
    scores.csv      one row per pose: vina, physics_loss
    ranking.csv     one row per ligand, best Vina first
    errors.tsv      ligands that failed sampling or scoring

Re-running with the same output_root skips prepared shards and scored ligands.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gemmi
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

from equiflex.inference.relaxation import write_complex_cif, write_ligand_sdf, write_protein_pdb
from equiflex.inference.sampler import FlowMatchingSampler
from equiflex.inference.vina_scoring import (
    count_active_torsions, score_files, vina_binary_score,
)
from equiflex.model import FlexTrainer
from equiflex.prep.featurize import ESM3Extractor, UniMolExtractor
from equiflex.prep.parse import get_pocket_residues, overwrite_coords_on_reference_data, parse_structure
from equiflex.prep.process import process_ligands, process_pocket_residues

from run_inference import (
    _VALID_SCHEDULES, _build_branch_config, _build_guidance_config, _build_sde_config,
    _parse_residues_spec,
    _sync_guidance_to_checkpoint, _write_trajectory_worker, assemble_system_data,
    build_ion_ligand, build_ligand_from_smiles, select_pocket_by_residues,
)
from select_pocket import METAL_CHARGE

_POCKET_FILES = ('pocket.yaml', 'pocket_features.pt', 'esm3_embeddings.pt')
_SCORE_COLUMNS = ('lig_id', 'sample', 'vina', 'vinardo', 'genscore', 'physics_loss', 'error')
# Vina/Vinardo are affinities (lower is better); GenScore is a summed pair
# likelihood, so it is maximised. Pose selection and ranking both honour this.
_HIGHER_IS_BETTER = frozenset({'genscore'})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path, limit_override: Optional[int], force_prepare: bool) -> Dict:
    """Read the HTS YAML and fill defaults; raises on missing required keys."""
    raw = yaml.safe_load(path.read_text())
    for key in ('checkpoint', 'receptor', 'ligands'):
        if key not in raw:
            raise ValueError(f'{path}: missing top-level key {key!r}')

    receptor = dict(raw['receptor'])
    if 'path' not in receptor:
        raise ValueError('receptor.path is required')
    if 'reference' not in receptor and 'residues' not in receptor:
        raise ValueError('receptor needs either reference (ccd/sdf) or residues')
    receptor['path'] = str(Path(receptor['path']).resolve())
    receptor.setdefault('pocket_cutoff', 5.0)
    receptor.setdefault('ions', 'auto')

    ligands = dict(raw['ligands'])
    if 'path' not in ligands:
        raise ValueError('ligands.path is required')
    ligands['path'] = str(Path(ligands['path']).resolve())
    ligands.setdefault('smiles_column', 'smiles')
    ligands.setdefault('name_column', 'name')
    ligands.setdefault('largest_fragment', True)
    ligands.setdefault('workers', 0)
    ligands.setdefault('unimol_batch_size', 128)
    ligands.setdefault('shard_size', 256)
    if limit_override is not None:
        ligands['limit'] = limit_override
    ligands.setdefault('limit', None)

    scoring = dict(raw.get('scoring') or {})
    scoring.setdefault('backend', 'torch')
    scoring.setdefault('workers', 8)
    scoring.setdefault('vina', os.environ.get('VINA_BIN', 'vina'))
    scoring.setdefault('obabel', os.environ.get('OBABEL_BIN', 'obabel'))
    scoring.setdefault('smina_scoring', 'vinardo')
    genscore = dict(scoring.get('genscore') or {})
    genscore.setdefault('enabled', True)
    genscore.setdefault('model', 'GT_0.0_1')
    genscore.setdefault('cutoff', 10.0)
    genscore.setdefault('batch_size', 32)
    scoring['genscore'] = genscore
    scoring.setdefault('select_by', 'genscore' if genscore['enabled'] else
                       ('vinardo' if scoring['backend'] == 'dual' else 'vina'))
    if scoring['select_by'] not in ('vina', 'vinardo', 'genscore'):
        raise ValueError(
            f"scoring.select_by must be vina, vinardo or genscore, got {scoring['select_by']!r}")
    if scoring['select_by'] == 'genscore' and not genscore['enabled']:
        raise ValueError('scoring.select_by=genscore but scoring.genscore.enabled is false')
    if scoring['backend'] not in ('torch', 'binary', 'smina', 'dual'):
        raise ValueError(f"scoring.backend must be torch, binary, smina or dual, got {scoring['backend']!r}")
    if scoring['backend'] in ('smina', 'dual') and shutil.which(scoring['vina']) is None:
        raise ValueError(f"scoring.backend={scoring['backend']} but smina not found: {scoring['vina']}")
    if scoring['backend'] == 'binary' and shutil.which(scoring['vina']) is None:
        raise ValueError(f"scoring.backend=binary but vina executable not found: {scoring['vina']}")

    schedule = raw.get('schedule', 'uniform')
    if schedule not in _VALID_SCHEDULES:
        raise ValueError(f'schedule {schedule!r} not in {list(_VALID_SCHEDULES)}')
    keep = raw.get('keep_samples', 'best')
    if keep not in ('best', 'all'):
        raise ValueError(f"keep_samples must be best or all, got {keep!r}")

    return {
        'checkpoint': raw['checkpoint'],
        'output_root': Path(raw.get('output_root', './hts_output')),
        'device': raw.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'prep_device': raw.get('prep_device', raw.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        'seed': int(raw.get('seed', 42)),
        'num_samples': int(raw.get('num_samples', 8)),
        'num_steps': int(raw.get('num_steps', 20)),
        'schedule': schedule,
        'return_trajectory': bool(raw.get('return_trajectory', False)),
        'trajectory_stride': int(raw.get('trajectory_stride', 1)),
        'keep_samples': keep,
        'force_prepare': force_prepare or bool(raw.get('force_prepare', False)),
        'guidance': raw.get('guidance'),
        'sde': raw.get('sde'),
        'branch': raw.get('branch'),
        'receptor': receptor,
        'ligands': ligands,
        'scoring': scoring,
    }


# ---------------------------------------------------------------------------
# Receptor: reference ligand -> pocket -> features (once)
# ---------------------------------------------------------------------------

def _is_hetero(residue: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(residue.name)
    return info is None or not (info.is_amino_acid() or info.is_water())


def reference_from_structure(
    receptor: Path, resname: str, chain: Optional[str], resnum: Optional[int],
) -> Tuple[np.ndarray, str, gemmi.Structure]:
    """Heavy-atom coords of the reference ligand residue in the template file.

    Returns ``(coords, label, single-residue structure)``. With several copies
    of the residue and no chain/resnum filter the first one is used.
    """
    st = gemmi.read_structure(str(receptor))
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    matches = []
    for ch in st[0]:
        for res in ch:
            if res.name.upper() != resname.upper():
                continue
            if chain is not None and ch.name != str(chain):
                continue
            if resnum is not None and res.seqid.num != int(resnum):
                continue
            matches.append((ch.name, res))
    if not matches:
        hetero = sorted({res.name for ch in st[0] for res in ch if _is_hetero(res)})
        raise ValueError(
            f'no residue {resname!r} in {receptor}'
            + (f' (chain={chain}, resnum={resnum})' if chain or resnum else '')
            + f'; hetero residues present: {hetero}'
        )
    labels = [f'{res.name} {cid}:{res.seqid.num}{res.seqid.icode.strip()}' for cid, res in matches]
    if len(matches) > 1:
        print(f'  reference {resname}: {len(matches)} copies {labels}, using {labels[0]} '
              f'(set receptor.reference.chain / resnum to choose)')
    chain_name, res = matches[0]
    coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res], dtype=np.float32)

    ref_st = gemmi.Structure()
    model = gemmi.Model('1')
    ref_chain = gemmi.Chain(chain_name)
    ref_chain.add_residue(res)
    model.add_chain(ref_chain)
    ref_st.add_model(model)
    return coords, labels[0], ref_st


def reference_from_sdf(path: Path) -> Tuple[np.ndarray, str]:
    mol = Chem.MolFromMolFile(str(path))
    if mol is None:
        raise ValueError(f'failed to parse reference ligand {path}')
    return mol.GetConformer().GetPositions().astype(np.float32), f'{path.name} ({mol.GetNumAtoms()} atoms)'


def detect_pocket_ions(receptor: Path, ref_coords: np.ndarray, cutoff: float) -> List[str]:
    """SMILES of single-atom metal residues within *cutoff* A of the reference ligand."""
    st = gemmi.read_structure(str(receptor))
    st.remove_waters()
    ions = []
    for chain in st[0]:
        for residue in chain:
            atoms = list(residue)
            if len(atoms) != 1:
                continue
            element = atoms[0].element.name.upper()
            if element not in METAL_CHARGE:
                continue
            pos = np.array([[atoms[0].pos.x, atoms[0].pos.y, atoms[0].pos.z]])
            if np.linalg.norm(ref_coords - pos, axis=1).min() >= cutoff:
                continue
            charge = METAL_CHARGE[element]
            smiles = f'[{element.capitalize()}+{charge}]' if charge > 0 else f'[{element.capitalize()}]'
            ions.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    return ions


def prepare_receptor(receptor_cfg: Dict, out_dir: Path, device: str, force: bool) -> Dict:
    """Featurise the pocket once. Reused from *out_dir* when its inputs match."""
    meta_path = out_dir / 'pocket.yaml'
    if not force and all((out_dir / f).exists() for f in _POCKET_FILES):
        meta = yaml.safe_load(meta_path.read_text())
        if meta.get('inputs') == receptor_cfg:
            print(f'  reusing pocket from {out_dir} ({meta["n_residues"]} residues, ions={meta["ions"]})')
            return {
                'features': torch.load(out_dir / 'pocket_features.pt', weights_only=False),
                'esm': torch.load(out_dir / 'esm3_embeddings.pt', weights_only=False),
                'ions': list(meta['ions']),
                'residues': meta['residues'],
            }
        print('  pocket inputs changed, re-featurising')
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor = Path(receptor_cfg['path'])
    cutoff = float(receptor_cfg['pocket_cutoff'])
    protein_data = parse_structure(receptor)
    if protein_data is None:
        raise RuntimeError(f'failed to parse receptor {receptor}')

    ref_coords, ref_label = None, None
    reference = receptor_cfg.get('reference')
    if reference:
        if 'sdf' in reference:
            ref_coords, ref_label = reference_from_sdf(Path(reference['sdf']))
        else:
            resname = reference.get('ccd') or reference.get('resname')
            if not resname:
                raise ValueError('receptor.reference needs ccd (residue name) or sdf')
            ref_coords, ref_label, ref_st = reference_from_structure(
                receptor, resname, reference.get('chain'), reference.get('resnum'),
            )
            ref_st.write_pdb(str(out_dir / 'reference_ligand.pdb'))
        print(f'  reference ligand: {ref_label}, {len(ref_coords)} heavy atoms')

    if receptor_cfg.get('residues'):
        chain_residues = _parse_residues_spec(receptor_cfg['residues'])
        pocket_data, sequence_data = select_pocket_by_residues(protein_data, chain_residues)
    else:
        pocket_data, sequence_data = get_pocket_residues(
            protein_data, [{'coords': ref_coords}], pocket_cutoff=cutoff,
        )
    pocket_data = {c: r for c, r in pocket_data.items() if r}
    if not pocket_data:
        raise RuntimeError('no pocket residues selected')
    residues = {c: sorted({int(r['resnum']) for r in rs}) for c, rs in pocket_data.items()}
    n_residues = sum(len(v) for v in residues.values())
    print(f'  pocket: {n_residues} residues on chain(s) {list(residues)} (cutoff {cutoff} A)')

    ions_cfg = receptor_cfg['ions']
    if ions_cfg == 'auto':
        ions = detect_pocket_ions(receptor, ref_coords, cutoff) if ref_coords is not None else []
    else:
        ions = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in (ions_cfg or [])]
    if ions:
        print(f'  ions: {ions}')

    pocket_residues = overwrite_coords_on_reference_data(pocket_data)
    pocket_features = process_pocket_residues(pocket_residues)
    if pocket_features is None:
        raise RuntimeError('failed to featurise pocket residues')

    print('  ESM3 embeddings...')
    esm = ESM3Extractor(device=device)
    pocket_embeddings = esm.extract_pocket_embeddings(sequence_data, pocket_residues, system_dir=out_dir)
    esm_emb = torch.stack([
        pocket_embeddings.get((cid, rn, eid), torch.zeros(1536))
        for cid, _, rn, eid in pocket_features['residue_keys']
    ])
    del esm
    torch.cuda.empty_cache()

    torch.save(pocket_features, out_dir / 'pocket_features.pt')
    torch.save(esm_emb, out_dir / 'esm3_embeddings.pt')
    meta = {
        'inputs': receptor_cfg,
        'reference_label': ref_label,
        'residues': residues,
        'ions': ions,
        'n_residues': n_residues,
        'n_atoms': int(pocket_features['coords'].shape[0]),
        'pocket_provenance': f'cutoff={cutoff} selector=equiflex.prep.parse.get_pocket_residues',
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, default_flow_style=None, width=120))
    return {'features': pocket_features, 'esm': esm_emb, 'ions': ions, 'residues': residues}


# ---------------------------------------------------------------------------
# Ligand library -> feature shards
# ---------------------------------------------------------------------------

def read_library(path: Path, smiles_column: str, name_column: Optional[str],
                 limit: Optional[int]) -> List[Tuple[str, str]]:
    """``[(name, smiles), ...]`` from .smi/.txt, .csv/.tsv or .sdf."""
    suffix = path.suffix.lower()
    entries: List[Tuple[str, str]] = []
    if suffix in ('.smi', '.txt', '.ism'):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            entries.append((parts[1].strip() if len(parts) > 1 else '', parts[0]))
    elif suffix in ('.csv', '.tsv'):
        df = pd.read_csv(path, sep='\t' if suffix == '.tsv' else ',')
        if smiles_column not in df.columns:
            raise ValueError(f'{path}: no column {smiles_column!r} (columns: {list(df.columns)})')
        names = df[name_column].astype(str) if name_column in df.columns else [''] * len(df)
        entries = list(zip(names, df[smiles_column].astype(str)))
    elif suffix == '.sdf':
        for mol in Chem.SDMolSupplier(str(path)):
            if mol is None:
                entries.append(('', ''))
            else:
                entries.append((mol.GetProp('_Name') if mol.HasProp('_Name') else '', Chem.MolToSmiles(mol)))
    else:
        raise ValueError(f'unsupported library format {suffix}; use .smi/.txt, .csv/.tsv or .sdf')
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError(f'{path}: no ligands read')
    return entries


def ligand_id(idx: int, name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')[:40]
    return f'{idx:06d}_{safe}' if safe else f'{idx:06d}'


def _worker_init():
    torch.set_num_threads(1)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def _featurize_ligand_worker(job):
    """Conformer + features for one library entry; returns ``(idx, record | None, error)``."""
    idx, smiles, ions, largest_fragment = job
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('RDKit cannot parse SMILES')
        if largest_fragment:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        ligands = [build_ligand_from_smiles(Chem.MolToSmiles(mol))]
        ligands.extend(build_ion_ligand(ion) for ion in ions)
        features = process_ligands(ligands)
        return idx, {'ligands': ligands, 'features': features, 'smiles': ligands[0]['smiles']}, None
    except Exception as exc:
        return idx, None, repr(exc)


def _embed_shard(unimol: UniMolExtractor, records: List[Dict], ions: List[str],
                 ion_cache: Dict[str, torch.Tensor], batch_size: int) -> None:
    """Attach ``unimol`` ([n_atoms_total, 1536]) and ``unimol_status`` to every record in place.

    A failed embedding (see ``UniMolExtractor.extract_batch`` statuses) is
    replaced by zeros so the ligand is still screened; the status is kept in
    the manifest.
    """
    for ion in ions:
        if ion not in ion_cache:
            atoms = build_ion_ligand(ion)['atoms']
            (emb,), (status,) = unimol.extract_batch([ion], [atoms], return_status=True)
            if emb is None:
                print(f'  ion {ion}: UniMol {status}, using zero embedding')
            ion_cache[ion] = emb if emb is not None else torch.zeros(len(atoms), 1536)

    main_embs: List[Optional[torch.Tensor]] = []
    statuses: List[str] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        embs, chunk_status = unimol.extract_batch(
            [r['ligands'][0]['smiles'] for r in chunk],
            [r['ligands'][0]['atoms'] for r in chunk],
            return_status=True,
        )
        main_embs.extend(embs)
        statuses.extend(chunk_status)
    for record, emb, status in zip(records, main_embs, statuses):
        n_main = len(record['ligands'][0]['atoms'])
        record['unimol_status'] = status
        parts = [emb if emb is not None else torch.zeros(n_main, 1536)]
        parts.extend(ion_cache[ion] for ion in ions)
        record['unimol'] = torch.cat(parts)


def prepare_ligands(lig_cfg: Dict, entries: List[Tuple[str, str]], out_dir: Path,
                    ions: List[str], force: bool, device: str) -> Tuple[pd.DataFrame, Set[str]]:
    """Featurise the library into ``shard_XXXX.pt`` files.

    Returns the manifest and the ids of ligands whose features were rebuilt
    with different input (or forced), so their earlier poses and scores are stale.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(lig_cfg['shard_size'])
    n_shards = math.ceil(len(entries) / shard_size)
    # Settings that change the features themselves. The library path / size is
    # deliberately not part of it: a shard is reused whenever its file exists and
    # the manifest rows for it carry exactly the input SMILES the library now
    # holds at those positions, so a library can be extended (or --limit
    # changed) and only the shards whose contents differ are rebuilt.
    signature = {'largest_fragment': bool(lig_cfg['largest_fragment']),
                 'shard_size': shard_size, 'ions': ions}
    sig_path = out_dir / 'library.yaml'
    manifest_path = out_dir / 'manifest.csv'

    rows: List[Dict] = []
    if not force and sig_path.exists() and manifest_path.exists():
        if yaml.safe_load(sig_path.read_text()) != signature:
            raise RuntimeError(
                f'{out_dir} was prepared with different settings ({sig_path}: '
                f'largest_fragment / shard_size / ions); pass --force-prepare or use a new output_root'
            )
        rows = pd.read_csv(manifest_path, keep_default_na=False).to_dict('records')

    def shard_is_current(k: int) -> bool:
        if not (out_dir / f'shard_{k:04d}.pt').exists():
            return False
        have = sorted((int(r['idx']), str(r['input_smiles'])) for r in rows if int(r['shard']) == k)
        want = [(i, entries[i][1]) for i in range(k * shard_size, min((k + 1) * shard_size, len(entries)))]
        return have == want

    pending = [k for k in range(n_shards) if force or not shard_is_current(k)]
    old_smiles = {r['lig_id']: str(r['input_smiles']) for r in rows}
    rows = [r for r in rows if int(r['shard']) not in pending and int(r['shard']) < n_shards]
    print(f'  {len(entries)} ligands in {n_shards} shard(s) of {shard_size}; '
          f'{n_shards - len(pending)} ready, {len(pending)} to prepare')
    stale: Set[str] = set()
    if not pending:
        return pd.DataFrame(rows), stale

    sig_path.write_text(yaml.safe_dump(signature, sort_keys=False))
    workers = int(lig_cfg['workers']) or max(1, (os.cpu_count() or 2) - 2)
    unimol: Optional[UniMolExtractor] = None
    ion_cache: Dict[str, torch.Tensor] = {}
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_worker_init) as pool:
        for k in pending:
            t0 = time.perf_counter()
            lo, hi = k * shard_size, min((k + 1) * shard_size, len(entries))
            jobs = [(i, entries[i][1], ions, bool(lig_cfg['largest_fragment'])) for i in range(lo, hi)]
            results = list(tqdm(pool.map(_featurize_ligand_worker, jobs, chunksize=4),
                                total=len(jobs), desc=f'shard {k} conformers', leave=False))
            records = []
            for idx, record, error in results:
                name = entries[idx][0] or f'lig{idx}'
                row = {'idx': idx, 'lig_id': ligand_id(idx, name), 'name': name,
                       'input_smiles': entries[idx][1], 'smiles': '', 'shard': k,
                       'n_atoms': 0, 'unimol_status': '', 'error': error or ''}
                if record is not None:
                    record.update(idx=idx, lig_id=row['lig_id'], name=name)
                    row['smiles'] = record['smiles']
                    row['n_atoms'] = len(record['ligands'][0]['atoms'])
                    records.append(record)
                rows.append(row)
                if force or old_smiles.get(row['lig_id']) != row['input_smiles']:
                    stale.add(row['lig_id'])
            if records:
                if unimol is None:
                    print('  loading UniMol...')
                    # UniMolExtractor masks padded atoms, so batching is safe: an
                    # embedding does not depend on which ligands share a shard or chunk.
                    unimol = UniMolExtractor(model_size='1.1B', use_cuda=device != 'cpu')
                _embed_shard(unimol, records, ions, ion_cache, int(lig_cfg['unimol_batch_size']))
                by_id = {r['lig_id']: r for r in records}
                for row in rows:
                    if row['lig_id'] in by_id:
                        row['unimol_status'] = by_id[row['lig_id']]['unimol_status']
            torch.save(records, out_dir / f'shard_{k:04d}.pt')
            manifest = pd.DataFrame(rows).sort_values('idx')
            manifest.to_csv(manifest_path, index=False)
            n_fail = sum(1 for _, r, _ in results if r is None)
            print(f'  shard {k}: {len(records)} ligands featurised, {n_fail} failed, '
                  f'{time.perf_counter() - t0:.0f}s')
    del unimol
    torch.cuda.empty_cache()
    return pd.DataFrame(rows).sort_values('idx'), stale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_sampler(checkpoint: str, device: str, guidance_cfg, guidance_explicit) -> FlowMatchingSampler:
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
        print(f"  applied EMA weights (decay={ckpt['ema_state_dict']['decay']})")
    del ckpt
    sampler = FlowMatchingSampler(trainer.model, device=device)
    del trainer
    return sampler


# ---------------------------------------------------------------------------
# Scoring workers
# ---------------------------------------------------------------------------

def _score_worker(job) -> Dict:
    """Score one pose from its receptor PDB and ligand SDF.

    The receptor is written straight from the sampled coords (see
    ``write_protein_pdb``), so this no longer re-parses a complex.cif to
    recover the protein.
    """
    lig_id, sample, protein_path, ligand_path, backend, vina, obabel, smina_scoring = job
    t0 = time.perf_counter()
    if backend == 'dual':
        # Both functions on the same pose: the torch port for Vina (validated
        # exact against smina) and the smina binary for Vinardo (the port's
        # Vinardo uses Vina's atom radii and is wrong). Selection then needs no
        # re-sampling whichever function is preferred.
        out = {'lig_id': lig_id, 'sample': sample, 'vina': None, 'vinardo': None, 'error': None}
        try:
            num_rot = count_active_torsions(ligand_path, obabel=obabel)
            out['vina'] = float(score_files(protein_path, ligand_path, num_rot=num_rot)['affinity'])
        except Exception as exc:
            out['error'] = f'vina: {exc!r}'
        try:
            cmd = [str(vina), '--score_only', '-r', str(protein_path), '-l', str(ligand_path),
                   '--scoring', smina_scoring or 'vinardo']
            txt = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
            m = re.search(r'^Affinity:\s*(-?\d+\.?\d*)', txt, re.M)
            out['vinardo'] = float(m.group(1)) if m else None
        except Exception as exc:
            out['error'] = f"{out['error'] or ''} vinardo: {exc!r}".strip()
        out['score_s'] = time.perf_counter() - t0
        return out
    if backend == 'smina':
        # smina reads PDB/SDF directly, so the pose needs no pdbqt conversion.
        # Whatever function is named here also drives pose *selection* in
        # collect(), so `smina_scoring: vinardo` ranks the ensemble by Vinardo.
        cmd = [str(vina), '--score_only', '-r', str(protein_path), '-l', str(ligand_path)]
        if smina_scoring and smina_scoring != 'vina':
            cmd += ['--scoring', smina_scoring]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
            m = re.search(r'^Affinity:\s*(-?\d+\.?\d*)', out, re.M)
            return {'lig_id': lig_id, 'sample': sample,
                    'vina': float(m.group(1)) if m else None,
                    'error': None if m else 'smina: no Affinity line',
                    'score_s': time.perf_counter() - t0}
        except Exception as exc:
            return {'lig_id': lig_id, 'sample': sample, 'vina': None, 'error': repr(exc),
                    'score_s': time.perf_counter() - t0}
    if backend == 'binary':
        scored = vina_binary_score(protein_path, ligand_path, vina=vina, obabel=obabel)
        return {'lig_id': lig_id, 'sample': sample, 'vina': scored['affinity'], 'error': scored['error'],
                'score_s': time.perf_counter() - t0}
    try:
        num_rot = count_active_torsions(ligand_path, obabel=obabel)
        affinity = score_files(protein_path, ligand_path, num_rot=num_rot)['affinity']
        return {'lig_id': lig_id, 'sample': sample, 'vina': float(affinity), 'error': None,
                'score_s': time.perf_counter() - t0}
    except Exception as exc:
        return {'lig_id': lig_id, 'sample': sample, 'vina': None, 'error': repr(exc),
                'score_s': time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Screening loop
# ---------------------------------------------------------------------------

def write_ranking(out_root: Path, manifest: pd.DataFrame, score_rows: List[Dict],
                  ligand_meta: Dict[str, Dict], select_by: str = 'vina') -> Optional[pd.DataFrame]:
    scores = pd.DataFrame(score_rows, columns=list(_SCORE_COLUMNS))
    for col in ('vina', 'vinardo', 'genscore', 'physics_loss'):
        scores[col] = pd.to_numeric(scores[col], errors='coerce')
    sel = select_by if select_by in scores.columns and scores[select_by].notna().any() else 'vina'
    ok = scores[scores[sel].notna()]
    if ok.empty:
        return None
    maximise = sel in _HIGHER_IS_BETTER
    cols = ['lig_id', 'sample', 'vina', 'vinardo', 'genscore', 'physics_loss']
    picked = ok.groupby('lig_id')[sel]
    best = ok.loc[picked.idxmax() if maximise else picked.idxmin(), cols]
    best = best.rename(columns={'sample': 'best_sample', 'vina': 'vina_best',
                                'vinardo': 'vinardo_best', 'genscore': 'genscore_best',
                                'physics_loss': 'physics_best'})
    # GenScore's signal is a per-ligand offset, so the mean over poses is the
    # screening column to rank on; the best-pose column is kept beside it.
    agg = ok.groupby('lig_id').agg(vina_mean=('vina', 'mean'), vinardo_mean=('vinardo', 'mean'),
                                   genscore_mean=('genscore', 'mean'),
                                   n_scored=(sel, 'size')).reset_index()
    ranking = (manifest[['lig_id', 'name', 'smiles', 'n_atoms']]
               .merge(best, on='lig_id').merge(agg, on='lig_id')
               .sort_values([f'{sel}_best', 'lig_id'], ascending=[not maximise, True])
               .reset_index(drop=True))
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    ranking['inference_s'] = [ligand_meta.get(l, {}).get('inference_s') for l in ranking['lig_id']]
    ranking['pose'] = [str(out_root / 'poses' / l / f'sample_{int(s)}' / 'ligand.sdf')
                       for l, s in zip(ranking['lig_id'], ranking['best_sample'])]
    ranking.to_csv(out_root / 'ranking.csv', index=False)
    return ranking


def run_screen(cfg: Dict, pocket: Dict, manifest: pd.DataFrame, stale: Set[str],
               sampler: FlowMatchingSampler, guidance_cfg, sde_cfg,
               branch_cfg=None) -> Dict:
    out_root: Path = cfg['output_root']
    poses_root = out_root / 'poses'
    scores_path = out_root / 'scores.csv'
    errors_path = out_root / 'errors.tsv'
    receptor_path = Path(cfg['receptor']['path'])
    scoring = cfg['scoring']
    num_samples = cfg['num_samples']
    # Branching duplicates every trajectory mid-path, so the sampler returns
    # num_samples * factor poses. Every per-pose loop below must use that count.
    branch_factor = branch_cfg.factor if (branch_cfg is not None and branch_cfg.enabled) else 1
    n_poses = num_samples * branch_factor

    # GenScore runs in this process on the GPU, not in the CPU scoring pool:
    # it is a torch model, and every pose needs the pocket re-cut from its own
    # predicted receptor, so there is nothing to share across workers.
    genscore_cfg = scoring.get('genscore') or {}
    gs_scorer = None
    if genscore_cfg.get('enabled'):
        from equiflex.inference.genscore_scoring import GenScoreScorer
        gs_scorer = GenScoreScorer(
            model_name=genscore_cfg['model'],
            device=getattr(sampler, 'device', 'cuda'),
            cutoff=genscore_cfg['cutoff'],
        )
        print(f"  GenScore: model {genscore_cfg['model']}, pocket cut per pose from the "
              f"predicted receptor ({genscore_cfg['cutoff']} A)")

    score_rows: List[Dict] = []
    if scores_path.exists():
        score_rows = pd.read_csv(scores_path, keep_default_na=False).to_dict('records')
    invalidated = {r['lig_id'] for r in score_rows} & stale
    if invalidated:
        print(f'  {len(invalidated)} ligands re-featurised: discarding their poses and scores')
        score_rows = [r for r in score_rows if r['lig_id'] not in invalidated]
        pd.DataFrame(score_rows, columns=list(_SCORE_COLUMNS)).to_csv(scores_path, index=False)
        for lig_id in invalidated:
            shutil.rmtree(poses_root / lig_id, ignore_errors=True)
    done = {r['lig_id'] for r in score_rows}
    todo = manifest[(manifest['error'] == '') & ~manifest['lig_id'].isin(done)]
    print(f'  {len(todo)} ligands to screen ({len(done)} already scored, '
          f'{int((manifest["error"] != "").sum())} failed featurisation)')

    scores_file = scores_path.open('a', newline='')
    writer = csv.DictWriter(scores_file, fieldnames=list(_SCORE_COLUMNS))
    if scores_path.stat().st_size == 0:
        writer.writeheader()
    errors_file = errors_path.open('a')

    def log_error(lig_id: str, stage: str, message: str) -> None:
        errors_file.write(f'{lig_id}\t{stage}\t{message}\n')
        errors_file.flush()

    ligand_meta: Dict[str, Dict] = {}
    pending: Dict[str, List[Future]] = {}
    traj_futures: List[Future] = []
    # Wall-clock per stage on the main process. score_cpu_s is summed from the
    # workers and overlaps the others, so it is not part of the wall total.
    stage = {'sample_s': 0.0, 'write_s': 0.0, 'wait_s': 0.0, 'complex_s': 0.0,
             'genscore_s': 0.0,
             'score_cpu_s': 0.0, 'n_poses': 0}

    def collect(block: bool) -> None:
        for lig_id in list(pending):
            futures = pending[lig_id]
            if not block and not all(f.done() for f in futures):
                continue
            t_wait = time.perf_counter()
            rows = [f.result() for f in futures]
            stage['wait_s'] += time.perf_counter() - t_wait
            stage['score_cpu_s'] += sum(r.get('score_s', 0.0) for r in rows)
            stage['n_poses'] += len(rows)
            physics = ligand_meta[lig_id]['physics']
            gs = ligand_meta[lig_id].get('genscore')
            for row in rows:
                row['physics_loss'] = physics[row['sample']]
                row['genscore'] = None if gs is None else gs[row['sample']]
                if row['error']:
                    log_error(lig_id, 'score', f"sample_{row['sample']}: {row['error']}")
                writer.writerow({k: ('' if row[k] is None else row[k]) for k in _SCORE_COLUMNS})
                score_rows.append(dict(row))
            scores_file.flush()
            sel = scoring.get('select_by', 'vina')
            scored = [r for r in rows if r.get(sel) is not None]
            pose = ligand_meta[lig_id].pop('pose', None)
            keep = {r['sample'] for r in rows}
            if cfg['keep_samples'] == 'best' and scored:
                pick = max if sel in _HIGHER_IS_BETTER else min
                best = pick(scored, key=lambda r: r[sel])['sample']
                keep = {best}
                for row in rows:
                    if row['sample'] != best:
                        shutil.rmtree(poses_root / lig_id / f"sample_{row['sample']}", ignore_errors=True)
            # Only the kept pose reaches the shared filesystem: its directory is
            # created here, the scoring inputs stay on node-local scratch.
            if pose is not None:
                for b in sorted(keep):
                    sample_dir = poses_root / lig_id / f'sample_{b}'
                    try:
                        t_cx = time.perf_counter()
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(scratch_dir / f'{lig_id}_s{b}.sdf',
                                        sample_dir / 'ligand.sdf')
                        write_complex_cif(
                            pocket['features'], pose['features'], pose['rec'][b], pose['lig'][b],
                            pose['com'], sample_dir / 'complex.cif', protein_path=receptor_path,
                        )
                        stage['complex_s'] += time.perf_counter() - t_cx
                    except Exception as exc:
                        log_error(lig_id, 'write_complex', repr(exc))
            for b in range(n_poses):
                (scratch_dir / f'{lig_id}_s{b}.pdb').unlink(missing_ok=True)
                (scratch_dir / f'{lig_id}_s{b}.sdf').unlink(missing_ok=True)
            del pending[lig_id]

    # Node-local (slurm sets TMPDIR per job); the scoring pool runs on this node.
    scratch_dir = Path(tempfile.mkdtemp(prefix='hts_poses_'))
    context = multiprocessing.get_context('spawn')
    pool = ProcessPoolExecutor(int(scoring['workers']), mp_context=context, initializer=_worker_init)
    shards = sorted(todo['shard'].unique())
    t_start = time.perf_counter()
    n_done = 0
    try:
        for shard_idx in shards:
            records = torch.load(out_root / 'ligands' / f'shard_{int(shard_idx):04d}.pt', weights_only=False)
            wanted = set(todo.loc[todo['shard'] == shard_idx, 'lig_id'])
            for record in tqdm([r for r in records if r['lig_id'] in wanted],
                               desc=f'shard {shard_idx} sampling', leave=False):
                lig_id = record['lig_id']
                try:
                    data = assemble_system_data(record['features'], pocket['features'],
                                                pocket['esm'], record['unimol'])
                    t0 = time.perf_counter()
                    results = sampler.sample(
                        data=data, num_samples=num_samples, num_steps=cfg['num_steps'],
                        schedule=cfg['schedule'], seed=cfg['seed'],
                        return_trajectory=cfg['return_trajectory'],
                        guidance=guidance_cfg, sde=sde_cfg, branch=branch_cfg,
                        progress=False,
                    )
                    # CUDA is async: without a sync the GPU time is charged to
                    # whichever later step first reads the output tensors.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_s = time.perf_counter() - t0
                    stage['sample_s'] += inference_s
                    physics = results.get('physics_loss')
                    ligand_meta[lig_id] = {
                        'inference_s': round(inference_s, 3),
                        'physics': [float(physics[b]) if physics is not None else None
                                    for b in range(n_poses)],
                    }
                    # Keep what the kept pose's complex.cif needs; writing all
                    # num_samples complexes up front costs ~16 ms each and
                    # keep_samples='best' deletes all but one right after scoring.
                    ligand_meta[lig_id]['pose'] = {
                        'features': record['features'],
                        'ligand': record['ligands'][0],
                        'rec': results['rec_coords_final'],
                        'lig': results['lig_coords_final'],
                        'com': results['backbone_com'],
                    }
                    futures = []
                    t_write = time.perf_counter()
                    for b in range(n_poses):
                        # Scoring inputs go to node-local scratch: on the shared
                        # filesystem each create costs ~10x what serialising the
                        # file does, and every one of these is deleted again.
                        protein_path = write_protein_pdb(
                            pocket['features'], results['rec_coords_final'][b],
                            results['backbone_com'], scratch_dir / f'{lig_id}_s{b}.pdb',
                            protein_path=receptor_path,
                        )
                        ligand_path = write_ligand_sdf(
                            record['ligands'][0], results['lig_coords_final'][b],
                            results['backbone_com'], scratch_dir / f'{lig_id}_s{b}.sdf',
                        )
                        futures.append(pool.submit(
                            _score_worker,
                            (lig_id, b, protein_path, ligand_path, scoring['backend'],
                             scoring['vina'], scoring['obabel'], scoring.get('smina_scoring')),
                        ))
                        if cfg['return_trajectory'] and results.get('rec_trajectory') is not None:
                            traj_futures.append(pool.submit(_write_trajectory_worker, (
                                b, pocket['features'], record['features'],
                                results['rec_trajectory'][b], results['lig_trajectory'][b],
                                results['backbone_com'], sample_dir / 'trajectory_complex.cif',
                                receptor_path, cfg['trajectory_stride'],
                            )))
                    stage['write_s'] += time.perf_counter() - t_write

                    if gs_scorer is not None:
                        t_gs = time.perf_counter()
                        try:
                            pose_pairs = []
                            for b in range(n_poses):
                                lig_mol = Chem.MolFromMolFile(
                                    str(scratch_dir / f'{lig_id}_s{b}.sdf'),
                                    removeHs=False, sanitize=True)
                                pose_pairs.append((scratch_dir / f'{lig_id}_s{b}.pdb', lig_mol))
                            ligand_meta[lig_id]['genscore'] = gs_scorer.score_poses(
                                pose_pairs, batch_size=genscore_cfg['batch_size'])
                        except Exception as exc:
                            ligand_meta[lig_id]['genscore'] = [None] * n_poses
                            log_error(lig_id, 'genscore', repr(exc))
                        stage['genscore_s'] = stage.get('genscore_s', 0.0) + (
                            time.perf_counter() - t_gs)

                    pending[lig_id] = futures
                    n_done += 1
                except Exception as exc:
                    log_error(lig_id, 'sample', repr(exc))
                    print(f'  [FAIL] {lig_id}: {exc}')
                collect(block=False)
            collect(block=False)
            elapsed = time.perf_counter() - t_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0.0
            print(f'  shard {shard_idx} sampled: {n_done}/{len(todo)} ligands, {rate:.1f} ligands/min')
            write_ranking(out_root, manifest, score_rows, ligand_meta, scoring.get('select_by', 'vina'))
        collect(block=True)
        for future in traj_futures:
            try:
                future.result()
            except Exception as exc:
                log_error('-', 'trajectory', repr(exc))
    finally:
        pool.shutdown(wait=True)
        scores_file.close()
        errors_file.close()
        shutil.rmtree(scratch_dir, ignore_errors=True)

    ranking = write_ranking(out_root, manifest, score_rows, ligand_meta, scoring.get('select_by', 'vina'))

    n = max(stage['n_poses'], 1)
    wall = (stage['sample_s'] + stage['write_s'] + stage['genscore_s']
            + stage['wait_s'] + stage['complex_s'])
    print('\n  stage breakdown (main process, ms/pose):')
    for key in ('sample_s', 'write_s', 'genscore_s', 'wait_s', 'complex_s'):
        print(f"    {key:<11} {stage[key]:8.1f}s  {stage[key] / n * 1000:7.2f} ms  "
              f"{stage[key] / max(wall, 1e-9) * 100:5.1f}%")
    print(f"    {'(total)':<11} {wall:8.1f}s  {wall / n * 1000:7.2f} ms")
    print(f"    scoring CPU in workers (overlapped): {stage['score_cpu_s']:.1f}s  "
          f"{stage['score_cpu_s'] / n * 1000:.2f} ms/pose")

    print(f'\n  scores : {scores_path}')
    if ranking is None:
        print('  no scored poses; see errors.tsv')
        return stage
    print(f'  ranking: {out_root / "ranking.csv"} ({len(ranking)} ligands)')
    print(ranking[[c for c in ('rank','lig_id','genscore_best','genscore_mean','vina_best','vinardo_best','physics_best','smiles') if c in ranking]]
          .head(10).to_string(index=False))
    return stage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=None, help='screen only the first N library entries')
    ap.add_argument('--force-prepare', action='store_true',
                    help='re-featurise the pocket and every ligand shard')
    args = ap.parse_args()

    cfg = load_config(args.config, args.limit, args.force_prepare)
    out_root: Path = cfg['output_root']
    out_root.mkdir(parents=True, exist_ok=True)
    guidance_cfg, guidance_explicit = _build_guidance_config(cfg['guidance'])
    sde_cfg = _build_sde_config(cfg['sde'])
    branch_cfg = _build_branch_config(cfg['branch'])

    print('=' * 60)
    print('Flex HTS')
    print(f"  checkpoint : {cfg['checkpoint']}")
    print(f"  receptor   : {cfg['receptor']['path']}")
    print(f"  library    : {cfg['ligands']['path']}"
          + (f" (limit {cfg['ligands']['limit']})" if cfg['ligands']['limit'] else ''))
    print(f"  samples    : {cfg['num_samples']}  steps: {cfg['num_steps']}  schedule: {cfg['schedule']}")
    if branch_cfg is not None and branch_cfg.enabled:
        print(f"  branch     : at t={branch_cfg.at_t} factor={branch_cfg.factor} "
              f"-> {cfg['num_samples'] * branch_cfg.factor} poses/ligand "
              f"(max_rot={branch_cfg.max_rot_deg} deg, trans_sigma={branch_cfg.trans_sigma} A)")
    print(f"  scoring    : {cfg['scoring']['backend']} Vina, {cfg['scoring']['workers']} workers; "
          f"keep {cfg['keep_samples']}")
    if guidance_cfg is not None:
        print(f"  guidance   : scale={guidance_cfg.scale} schedule={guidance_cfg.schedule} "
              f"t_start={guidance_cfg.t_start}")
    if sde_cfg is not None:
        print(f"  sde        : sigma={sde_cfg.sigma} mode={sde_cfg.mode}")
    print(f"  device     : {cfg['device']}  (prep: {cfg['prep_device']})")
    print(f"  output     : {out_root}")
    print('=' * 60)

    timings: Dict[str, float] = {}

    print('\n[1/4] Receptor')
    t0 = time.perf_counter()
    pocket = prepare_receptor(cfg['receptor'], out_root / 'receptor', cfg['prep_device'], cfg['force_prepare'])
    timings['receptor_s'] = time.perf_counter() - t0

    print('\n[2/4] Ligands')
    t0 = time.perf_counter()
    entries = read_library(Path(cfg['ligands']['path']), cfg['ligands']['smiles_column'],
                           cfg['ligands']['name_column'], cfg['ligands']['limit'])
    manifest, stale = prepare_ligands(cfg['ligands'], entries, out_root / 'ligands', pocket['ions'],
                                      cfg['force_prepare'], cfg['prep_device'])
    timings['ligand_prep_s'] = time.perf_counter() - t0
    timings['n_ligands'] = float(len(manifest))

    print(f"\n[3/4] Model: {cfg['checkpoint']}")
    t0 = time.perf_counter()
    sampler = load_sampler(cfg['checkpoint'], cfg['device'], guidance_cfg, guidance_explicit)
    timings['model_load_s'] = time.perf_counter() - t0

    print('\n[4/4] Screening')
    t0 = time.perf_counter()
    stage = run_screen(cfg, pocket, manifest, stale, sampler, guidance_cfg, sde_cfg, branch_cfg)
    timings['screening_s'] = time.perf_counter() - t0
    if stage:
        timings.update({f'screen_{k}': float(v) for k, v in stage.items()})

    # Wall-clock breakdown, so a resumed run leaves its own timings file behind
    # rather than overwriting the first one's.
    timings['prep_total_s'] = timings['receptor_s'] + timings['ligand_prep_s']
    timings['total_s'] = sum(timings[k] for k in
                             ('receptor_s', 'ligand_prep_s', 'model_load_s', 'screening_s'))
    stamp = time.strftime('%Y%m%dT%H%M%S')
    (out_root / f'timings_{stamp}.json').write_text(json.dumps(timings, indent=2))
    print('\n  timing (s): ' + '  '.join(
        f'{k}={timings[k]:.1f}' for k in ('receptor_s', 'ligand_prep_s', 'model_load_s',
                                          'screening_s', 'total_s')))


if __name__ == '__main__':
    main()
