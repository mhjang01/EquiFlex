"""Focused regression checks for the TPD screen; no mirror or network required."""
import gzip
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import screen_tpd as t


def test_metadata_after_coordinates_and_release_window(tmp_path):
    path = tmp_path / 'test.cif.gz'
    text = '''data_TEST
_struct.title 'PROTAC ternary complex'
#
loop_
_atom_site.id
_atom_site.Cartn_x
1 1.0
#
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2024-06-30
2 2025-07-01
#
'''
    with gzip.open(path, 'wt') as f:
        f.write(text)
    result = t.scan((str(path), '2024-07-01', '2026-06-30'))
    assert result['release'] == '2024-06-30'
    assert result['scope'] == 'outside_window'
    with gzip.open(path, 'wt') as f:
        f.write(text.replace('2024-06-30', '2024-07-01'))
    result = t.scan((str(path), '2024-07-01', '2026-06-30'))
    assert result['tagged']
    assert result['terms']['protac'] == ['title']


def test_missing_release_is_not_replaced_with_deposition(tmp_path):
    path = tmp_path / 'test.cif'
    path.write_text('data_TEST\n_pdbx_database_status.recvd_initial_deposition_date 2025-01-01\n')
    assert t.scan((str(path), '2024-07-01', '2026-06-30'))['error'] == 'missing_release_date'


def test_homology_catches_distinct_entities_and_domain_constructs():
    sequence = 'MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE'
    assert t.homology('1', '2', {'1':sequence, '2':sequence[5:]}, {})[0]
    assert t.homology('1', '2', {'1':'AAAA', '2':'GGGG'}, {'1':{'P12345'}, '2':{'P12345'}})[0]
    assert not t.homology('1', '2', {'1':sequence, '2':'GGGGGGGGGGGGGGGGGGGG'}, {})[0]


def test_cofactor_rejected_without_ccd_store(monkeypatch,tmp_path):
    monkeypatch.setattr(t, 'CCD_ROOT', tmp_path)
    props,reasons,mol=t.chemistry('FAD','FLAVIN-ADENINE DINUCLEOTIDE','C27 H33 N9 O15 P2')
    assert 'cofactor_buffer_or_sugar' in reasons
    assert props['chemistry_source'] == 'formula_only'


import os
import pytest


def local_entry(pdb_id):
    path = Path(os.environ.get('TPD_PDB_ROOT', '/mnt/drive/PDB')) / pdb_id[1:3] / f'{pdb_id}.cif.gz'
    if not path.exists():
        pytest.skip('Local PDB mirror required for integration checks')
    return t.scan((str(path), '1900-01-01', '2030-01-01'))


@pytest.mark.parametrize('pdb_id', ['9hne', '9sai', '9q2d'])
def test_known_local_ternaries(pdb_id):
    results = t.evaluate(local_entry(pdb_id))
    assert not any(r['tier'] == 'error' for r in results)
    assert any(r['tier'] == 'high_confidence' and r['space'].startswith('assembly_') for r in results)


@pytest.mark.parametrize('pdb_id', ['8pqn', '8qjt'])
def test_user_identified_homomer_false_positives(pdb_id):
    results = t.evaluate(local_entry(pdb_id))
    assert not any(r['tier'] in ('error', 'high_confidence') for r in results)
    assert any('homomeric_or_near_identical_proteins' in r['reasons'] for r in results)


def test_export_preserves_observed_ligand_and_structural_zinc(tmp_path):
    from rdkit import Chem
    metadata = local_entry('9q2d')
    record = next(r for r in t.evaluate(metadata) if r['tier'] == 'high_confidence' and r['space'].startswith('assembly_'))
    record['record'] = '9Q2D_export_test'
    for name in ('reference', 'ligands', 'raw'):
        (tmp_path / name).mkdir()
    t.export_complex(record, metadata, tmp_path)
    structure = t.gemmi.read_structure(str(tmp_path / 'reference/9Q2D_export_test.cif'))
    mol = Chem.SDMolSupplier(str(tmp_path / 'ligands/9Q2D_export_test.sdf'))[0]
    assert mol is not None
    coordinates = [a.pos.tolist() for a in structure[0]['L'][0]]
    distances = t.cKDTree(coordinates).query(mol.GetConformer().GetPositions())[0]
    assert max(distances) < 0.001
    assert mol.GetNumHeavyAtoms() == len(coordinates)
    assert 'ZN' in record['structural_ions']
    receptor = t.gemmi.read_structure(str(tmp_path / 'raw/9Q2D_export_test/9Q2D_export_test_protein.pdb'))
    assert 'L' not in {c.name for c in receptor[0]}
    assert any(a.element.name == 'Zn' for c in receptor[0] for r in c for a in r)
