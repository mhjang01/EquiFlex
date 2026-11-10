"""Cross-file ligand/Uni-Mol validation and cache provenance.

Legacy, shape-consistent inputs remain readable. Newly prepared features carry
an atom-order signature; their embeddings must have matching provenance. An
explicit preparation run regenerates unverified legacy embedding caches rather
than guessing how their rows correspond to the current ligand atoms.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

SIGNATURE_KEY = 'ligand_atom_signature'
UNIMOL_METADATA = 'unimol_embeddings.meta.json'


def ligand_atom_signature(ligands):
    """Identify ordered entities, atoms and connectivity, excluding coordinates."""
    records = []
    for ligand in ligands:
        atoms = [
            {k: atom.get(k) for k in
             ('name', 'symbol', 'atomic_number', 'formal_charge', 'num_explicit_hs')}
            for atom in ligand['atoms']
        ]
        bonds = [
            {k: bond.get(k) for k in ('begin_atom_idx', 'end_atom_idx', 'bond_type')}
            for bond in ligand.get('bonds', [])
        ]
        records.append({'smiles': ligand['smiles'], 'atoms': atoms, 'bonds': bonds})
    payload = json.dumps(records, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_ligand_records(ligands, features, system_id):
    """Reject mixing a regenerated ligand list with features from another list."""
    expected = [len(ligand['atoms']) for ligand in ligands]
    if (list(features['n_atoms_per_ligand']) != expected or
            features['atom_features'].shape[0] != sum(expected) or
            list(features['smiles_list']) != [ligand['smiles'] for ligand in ligands]):
        raise ValueError(f'{system_id}: ligand records and ligand_features disagree; '
                         'rerun graph featurization before embedding extraction')
    signature = ligand_atom_signature(ligands)
    if features.get(SIGNATURE_KEY) != signature:
        raise ValueError(f'{system_id}: ligand atom-order provenance is missing or stale; '
                         'rerun graph featurization before embedding extraction')
    return signature


def validate_unimol_shape(features, embeddings, system_id):
    n = features['atom_features'].shape[0]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f'{system_id}: unimol_embeddings must be a two-dimensional tensor')
    if embeddings.shape[0] != n:
        raise ValueError(
            f'{system_id}: Uni-Mol atom-count mismatch: ligand_features has {n} atoms '
            f'but unimol_embeddings has {embeddings.shape[0]} rows; regenerate embeddings '
            'from the current ligand records (do not blindly truncate)')


def validate_cached_unimol(system_dir, features, embeddings, require_provenance=False):
    system_dir = Path(system_dir)
    validate_unimol_shape(features, embeddings, system_dir.name)
    metadata_path = system_dir / UNIMOL_METADATA
    signature = features.get(SIGNATURE_KEY)
    if not metadata_path.exists():
        if require_provenance or signature is not None:
            raise ValueError(f'{system_dir.name}: Uni-Mol cache has no atom-order provenance; '
                             'regenerate embeddings from the current ligand records')
        return  # Backwards-compatible loading of consistent legacy datasets.
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('schema') != 1 or metadata.get('shape') != list(embeddings.shape):
        raise ValueError(f'{system_dir.name}: Uni-Mol cache metadata is inconsistent')
    if signature is None or metadata.get(SIGNATURE_KEY) != signature:
        raise ValueError(f'{system_dir.name}: Uni-Mol atom-order/identity provenance mismatch; '
                         'regenerate embeddings from the current ligand records')


def unimol_cache_reason(system_dir, features, ligands):
    """Return None only if a preparation-stage cache is safe to reuse."""
    system_dir = Path(system_dir)
    try:
        validate_ligand_records(ligands, features, system_dir.name)
        embeddings = torch.load(system_dir / 'unimol_embeddings.pt',
                                map_location='cpu', weights_only=False)
        validate_cached_unimol(system_dir, features, embeddings, require_provenance=True)
    except Exception as exc:
        return str(exc)
    return None


def save_unimol_cache(system_dir, features, ligands, embeddings):
    """Validate before replacing either cache file; save a tensor plus JSON sidecar."""
    system_dir = Path(system_dir)
    signature = validate_ligand_records(ligands, features, system_dir.name)
    validate_unimol_shape(features, embeddings, system_dir.name)
    metadata = {'schema': 1, SIGNATURE_KEY: signature, 'shape': list(embeddings.shape)}
    tensor_temp = metadata_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=system_dir, suffix='.pt', delete=False) as f:
            tensor_temp = Path(f.name)
        torch.save(embeddings, tensor_temp)
        with tempfile.NamedTemporaryFile(mode='w', dir=system_dir, suffix='.json',
                                         delete=False) as f:
            metadata_temp = Path(f.name)
            json.dump(metadata, f, indent=2)
            f.write('\n')
        os.replace(tensor_temp, system_dir / 'unimol_embeddings.pt')
        os.replace(metadata_temp, system_dir / UNIMOL_METADATA)
    finally:
        for path in (tensor_temp, metadata_temp):
            if path is not None:
                path.unlink(missing_ok=True)
