"""Regressions for stale ion embeddings and atom-order changes in cached inputs."""
from pathlib import Path
from types import SimpleNamespace

from joblib import dump
import pytest
import torch

from equiflex.prep.validation import (
    SIGNATURE_KEY, ligand_atom_signature, save_unimol_cache,
    validate_cached_unimol, unimol_cache_reason,
)
from equiflex.model.loader import FlexDataset
from equiflex.model.trainer import RMSDEvaluationCallback
from equiflex.dataprep import featurize_data
from scripts.run_inference import system_data_is_current


def ligands(names=('C1', 'O2')):
    return [{'smiles': 'CO', 'atoms': [
        {'name': name, 'symbol': name[0], 'atomic_number': 6 if name[0] == 'C' else 8,
         'formal_charge': 0} for name in names],
        'bonds': [{'begin_atom_idx': 0, 'end_atom_idx': 1, 'bond_type': 'SINGLE'}]}]


def features(records, signed=True):
    n = sum(len(ligand['atoms']) for ligand in records)
    result = {'atom_features': torch.zeros(n, 299), 'coords': torch.zeros(n, 3),
              'n_atoms_per_ligand': [len(ligand['atoms']) for ligand in records],
              'smiles_list': [ligand['smiles'] for ligand in records],
              'constraint_matrix': torch.zeros(n, n),
              'sp2_torsion_indices': torch.empty(0, 4, dtype=torch.long),
              'chiral_atom_indices': torch.empty(0, 5, dtype=torch.long)}
    if signed:
        result[SIGNATURE_KEY] = ligand_atom_signature(records)
    return result


def write_system(root, records=None, signed=True):
    root.mkdir(exist_ok=True)
    records = ligands() if records is None else records
    feat = features(records, signed=signed)
    dump(records, root / 'ligands.pkl')
    torch.save(feat, root / 'ligand_features.pt')
    torch.save({'constraint_matrix': torch.zeros(1, 1),
                'sp2_torsion_indices': torch.empty(0, 4, dtype=torch.long),
                'cb_mask': torch.zeros(1), 'missing_mask': torch.zeros(1, dtype=torch.bool)},
               root / 'pocket_features.pt')
    torch.save(torch.zeros(1, 1536), root / 'esm3_embeddings.pt')
    return records, feat


def test_historical_extra_iron_rows_fail_before_model_forward(tmp_path):
    root = tmp_path / '7CNQ_G8X'
    records, feat = write_system(root, signed=False)
    torch.save(torch.ones(3, 1536), root / 'unimol_embeddings.pt')
    with pytest.raises(ValueError, match=r'7CNQ_G8X: Uni-Mol atom-count mismatch.*2 atoms.*3 rows'):
        FlexDataset(str(tmp_path))[0]
    # Preloading cannot bypass validation either.
    with pytest.raises(ValueError, match='7CNQ_G8X'):
        FlexDataset(str(tmp_path), preload=True, preload_jobs=1)
    assert 'atom-count mismatch' in system_data_is_current(root)


def test_consistent_legacy_benchmark_inputs_remain_readable(tmp_path):
    root = tmp_path / 'legacy'
    records, feat = write_system(root, signed=False)
    emb = torch.zeros(2, 1536)
    torch.save(emb, root / 'unimol_embeddings.pt')
    validate_cached_unimol(root, feat, emb)
    assert system_data_is_current(root) is None
    # Explicit preparation cannot certify unknown legacy atom ordering.
    assert 'provenance' in unimol_cache_reason(root, feat, records)


def test_atom_reordering_with_unchanged_size_invalidates_cache(tmp_path):
    records, feat = write_system(tmp_path)
    emb = torch.ones(2, 1536)
    save_unimol_cache(tmp_path, feat, records, emb)
    reordered = ligands(('O2', 'C1'))
    changed = features(reordered)
    assert changed['atom_features'].shape == feat['atom_features'].shape
    torch.save(changed, tmp_path / 'ligand_features.pt')
    dump(reordered, tmp_path / 'ligands.pkl')
    with pytest.raises(ValueError, match='atom-order/identity provenance mismatch'):
        validate_cached_unimol(tmp_path, changed, emb)
    assert 'provenance mismatch' in unimol_cache_reason(tmp_path, changed, reordered)


def test_partial_feature_update_without_embedding_metadata_is_rejected(tmp_path):
    records, feat = write_system(tmp_path)
    torch.save(torch.ones(2, 1536), tmp_path / 'unimol_embeddings.pt')
    assert 'no atom-order provenance' in system_data_is_current(tmp_path)


def test_wrong_new_tensor_cannot_replace_valid_cache(tmp_path):
    records, feat = write_system(tmp_path)
    save_unimol_cache(tmp_path, feat, records, torch.ones(2, 1536))
    before = (tmp_path / 'unimol_embeddings.pt').read_bytes()
    with pytest.raises(ValueError, match='atom-count mismatch'):
        save_unimol_cache(tmp_path, feat, records, torch.zeros(3, 1536))
    assert (tmp_path / 'unimol_embeddings.pt').read_bytes() == before
    assert unimol_cache_reason(tmp_path, feat, records) is None


def test_valid_cache_does_not_initialize_large_unimol_model(tmp_path, monkeypatch):
    records, feat = write_system(tmp_path)
    save_unimol_cache(tmp_path, feat, records, torch.ones(2, 1536))
    def forbidden(**kwargs):
        pytest.fail('An unchanged validated cache must not load Uni-Mol')
    monkeypatch.setattr(featurize_data, 'UniMolExtractor', forbidden)
    assert featurize_data.run_step3_unimol_extraction(tmp_path, [tmp_path]) == [tmp_path]


def test_stale_ion_cache_is_regenerated_instead_of_skipped(tmp_path, monkeypatch):
    records, feat = write_system(tmp_path)
    torch.save(torch.ones(3, 1536), tmp_path / 'unimol_embeddings.pt')
    calls = []
    class Extractor:
        def __init__(self, **kwargs):
            pass
        def extract_batch(self, smiles, atoms):
            calls.append((smiles, atoms))
            return [torch.full((len(a), 1536), 2.0) for a in atoms]
    monkeypatch.setattr(featurize_data, 'UniMolExtractor', Extractor)
    assert featurize_data.run_step3_unimol_extraction(tmp_path, [tmp_path]) == [tmp_path]
    assert len(calls) == 1
    assert torch.load(tmp_path / 'unimol_embeddings.pt', weights_only=False).shape == (2, 1536)
    assert unimol_cache_reason(tmp_path, feat, records) is None


def test_entity_mismatch_is_rejected_even_if_total_size_matches(tmp_path):
    records = [ligands()[0], ligands()[0]]
    records, feat = write_system(tmp_path, records)
    class Extractor:
        def extract_batch(self, *args):
            return [torch.zeros(3, 1536), torch.zeros(1, 1536)]
    errors = []
    pbar = SimpleNamespace(update=lambda n: None)
    ok = featurize_data._run_unimol_subbatch(
        Extractor(), [tmp_path], ['CO', 'CO'], [r['atoms'] for r in records],
        [[2, 2]], errors, pbar)
    assert ok == []
    assert errors and 'expects 2 embedding rows' in errors[0][1]
    assert not (tmp_path / 'unimol_embeddings.pt').exists()


def test_evaluation_logs_load_failure_and_continues(caplog):
    class Loader:
        def __len__(self):
            return 2
        def __iter__(self):
            class Iterator:
                index = 0
                def __iter__(self):
                    return self
                def __next__(self):
                    self.index += 1
                    if self.index == 1:
                        raise ValueError('7CNQ_G8X: Uni-Mol atom-count mismatch')
                    if self.index == 2:
                        return {'system_ids': ['good_system']}
                    raise StopIteration
            return Iterator()
    module = SimpleNamespace(device=torch.device('cpu'),
                             _shared_step=lambda batch: {'total': torch.tensor(2.0)})
    metrics = RMSDEvaluationCallback._evaluate_test_losses(None, module, Loader(), 'pb')
    assert metrics['pb/loss_errors'] == 1
    assert metrics['pb/total_loss'] == 2.0
    assert '7CNQ_G8X' in caplog.text and 'split=pb' in caplog.text
    assert any(record.exc_info is not None for record in caplog.records)
