"""Regression tests for broad homomer retention, date selection and coordinate export."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import build_plp_dataset as b


def test_completeness_uses_heavy_atoms_and_distinguishes_unknown():
    from rdkit import Chem
    block = b.t.gemmi.cif.read_string('''data_test
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
LIG C1 C
LIG O1 O
LIG H1 H
LIG D1 D
''').sole_block()
    atoms = []
    for name, element in [('C1', 'C'), ('O1', 'O')]:
        a = b.t.gemmi.Atom(); a.name = name; a.element = b.t.gemmi.Element(element); a.occ = 1
        atoms.append(a)
    result = b.t.ligand_completeness(block, 'LIG', None, atoms)
    assert result['ligand_completeness'] == 1
    assert result['expected_heavy_atoms'] == 2
    assert result['missing_ccd_atoms'] == []
    assert result['completeness_source'] == 'embedded_chem_comp_atom'
    assert b.t.ligand_completeness(block, 'LIG', None, atoms[:1])['ligand_completeness'] == .5
    unknown = b.t.ligand_completeness(block, 'UNK', None, atoms)
    assert unknown['ligand_completeness'] is None
    assert unknown['missing_ccd_atoms'] is None
    # Retained isotopic hydrogens must never enter a heavy-atom denominator.
    mol = Chem.MolFromSmiles('[2H]OC')
    for atom in mol.GetAtoms(): atom.SetProp('name', {'H': 'D1', 'C': 'C1', 'O': 'O1'}[atom.GetSymbol()])
    assert b.t.ligand_completeness(block, 'LIG', mol, atoms)['ligand_completeness'] == 1


def test_10ay_missing_local_ccd_is_not_missing_coordinates():
    path = Path('/home/mhjang/DB/PDB/0a/10ay.cif.gz')
    if not path.exists(): pytest.skip('Local mirror unavailable')
    meta = b.metadata(str(path)); meta['parse_cutoff'] = '2026-06-30'
    result = b.evaluate_entry(meta)
    assert 'error' not in result
    row = next(r for r in result['records'] if r['ccd'] == 'A1C4S')
    assert row['ligand_completeness'] == 1
    assert row['expected_heavy_atoms'] == row['observed_heavy_atoms'] == 38
    assert row['missing_ccd_atoms'] == []
    assert 'ligand_less_than_90_percent_complete' not in row['quality_flags']
    assert 'ligand_not_fully_observed' not in row['quality_flags']


def test_isotopic_hydrogens_removed_from_chemistry_reference(tmp_path, monkeypatch):
    import pickle
    from rdkit import Chem
    mol = Chem.MolFromSmiles('[2H]Oc1ccccc1')
    assert Chem.RemoveHs(mol).GetNumAtoms() > mol.GetNumHeavyAtoms()
    (tmp_path/'TEST.pkl').write_bytes(pickle.dumps(mol))
    monkeypatch.setattr(b.t, 'CCD_ROOT', tmp_path)
    _, _, heavy = b.t.chemistry('TEST', 'test', '')
    assert heavy.GetNumAtoms() == heavy.GetNumHeavyAtoms()
    assert all(a.GetAtomicNum() > 1 for a in heavy.GetAtoms())


def test_additional_whole_archive_chemistry_filters():
    assert b.additional_chemistry_flags('BOG', 'octyl beta-D-glucopyranoside', 'CCCCCCCCOC1OC(CO)C(O)C(O)C1O')
    assert b.additional_chemistry_flags('NEW', 'systematically named glycolipid', 'CCCCCCCCCCCCCCCC(=O)OCC(COC1OC(CO)C(O)C(O)C1O)OC(=O)CCCCCCCCCCCCCCC')
    assert b.additional_chemistry_flags('NEW', 'systematically named polyene pigment', 'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)CCCC2(C)C)CCCC1(C)C')
    assert not b.additional_chemistry_flags('DRUG', 'small aromatic inhibitor', 'CC(=O)Nc1ccc(O)cc1')


def test_date_filter_precedes_representative_selection():
    old = dict(record='OLD', pdb_id='OLD', complex_group='same', release='2024-06-30',
               deposition_date='2024-01-01', resolution=3, ligand_completeness=1,
               plp_training_eligible=True, tpd_tier='high_confidence')
    new = dict(old, record='NEW', pdb_id='NEW', release='2026-07-01',
               deposition_date='2026-06-30', resolution=2)
    assert b.select_rows([old, new]) == [old]
    assert b.select_rows([old, new], date_field='deposition_date') == [new]
    assert b.select_rows([old, new], start='2024-07-01') == []
    assert b.select_rows([dict(old, plp_training_eligible='False')]) == []
    assert b.select_rows([dict(old, plp_training_eligible='True')])


def test_metadata_preserves_deposition_and_revision_dates(tmp_path):
    path = tmp_path / 'example.cif.gz'
    with gzip.open(path, 'wt') as handle:
        handle.write('''data_TEST
_entry.id TEST
_pdbx_database_status.recvd_initial_deposition_date 2024-01-03
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2024-07-01
2 2026-09-30
#
''')
    result = b.metadata(str(path))
    assert result['deposition_date'] == '2024-01-03'
    assert result['release'] == '2024-07-01'
    assert result['last_revision_date'] == '2026-09-30'


def test_removed_alternate_subchain_does_not_break_assembly():
    path = Path('/home/mhjang/DB/PDB/ih/2ihu.cif.gz')
    if not path.exists(): pytest.skip('Local mirror unavailable')
    s = b.t.gemmi.read_structure(str(path))
    b.t.clean_structure_for_assembly(s)
    for assembly in s.assemblies:
        assert len(b.t.gemmi.make_assembly(assembly, s[0], b.t.gemmi.HowToNameCopiedChain.AddNumber))


@pytest.mark.parametrize('pdb_id,homomer,tpd', [('8pqn', True, 'rejected'), ('9hne', False, 'high_confidence')])
def test_local_broad_and_tpd_separation_and_export(pdb_id, homomer, tpd, tmp_path):
    path = Path('/home/mhjang/DB/PDB') / pdb_id[1:3] / f'{pdb_id}.cif.gz'
    if not path.exists(): pytest.skip('Local mirror unavailable')
    b.worker_init(str(b.t.CCD_ROOT))
    meta = b.metadata(str(path)); meta['parse_cutoff'] = '2026-06-30'
    result = b.evaluate_entry(meta)
    assert 'error' not in result
    assert len(result['records']) == 1
    row = result['records'][0]
    assert row['homomeric'] is homomer
    assert row['plp_training_eligible']
    assert row['tpd_tier'] == tpd
    assert row['instance_count'] > 1
    for directory in ('reference', 'ligands', 'raw'): (tmp_path / directory).mkdir()
    exported = b.export_one((row, str(tmp_path)))
    assert exported['export_status'] == 'validated', exported.get('export_error')
    assert Path(exported['protein_cif']).exists()
    assert Path(exported['ligand_sdf']).exists()
    assert len(exported['validated_contacts']) == 2


def local_records(pdb_id):
    path = Path('/home/mhjang/DB/PDB') / pdb_id[1:3] / f'{pdb_id}.cif.gz'
    if not path.exists(): pytest.skip('Local mirror unavailable')
    b.worker_init(str(b.t.CCD_ROOT))
    meta = b.metadata(str(path)); meta['parse_cutoff'] = '2026-06-30'
    result = b.evaluate_entry(meta)
    assert 'error' not in result
    return result['records']


def export(row, tmp_path):
    for directory in ('reference', 'ligands', 'raw'): (tmp_path / directory).mkdir(parents=True)
    exported = b.export_one((row, str(tmp_path)))
    assert exported['export_status'] == 'validated', exported.get('export_error')
    return exported, b.t.gemmi.read_structure(exported['reference_cif'])


def test_export_keeps_deposited_sequence_with_unresolved_residues(tmp_path):
    row, = local_records('9hne')
    exported, st = export(row, tmp_path)
    for name, key in (('A', 'sequence_A'), ('B', 'sequence_B')):
        polymer = st[0][name].get_polymer()
        assert b.t.gemmi.one_letter_code(st.get_entity_of(polymer).full_sequence) == row[key]
    assert any(len(st[0][name]) < len(row[key]) for name, key in (('A', 'sequence_A'), ('B', 'sequence_B')))


def test_declared_covalent_ligand_is_kept_with_its_link(tmp_path):
    row, = local_records('8g46')
    assert row['covalent'] and row['plp_training_eligible']
    assert [(bond['partner_name'], bond['partner_atom']) for bond in row['covalent_bonds']] == [('CYS', 'SG')]
    assert 'clashing_contact' not in row['quality_flags']
    _, st = export(row, tmp_path)
    link, = st.connections
    assert link.type == b.t.gemmi.ConnectionType.Covale and link.partner1.chain_name == 'L'


def test_ligand_bonded_to_another_nonpolymer_goes_to_review():
    row, = local_records('1fg3')
    assert 'ligand_bonded_to_other_nonpolymer' in row['quality_flags']
    assert 'PLP' in [h['name'] for h in row['context_ligands']]


def test_every_contacting_chain_and_cobinder_is_exported(tmp_path):
    row, = local_records('6q0r')
    assert row['protein_chains'] == 3 and row['tpd_tier'] == 'high_confidence'
    exported, st = export(row, tmp_path)
    assert [c.name for c in st[0]][:3] == ['A', 'B', 'C'] and len(exported['validated_contacts']) == 3
    row, = local_records('1dxo')
    _, st = export(row, tmp_path / 'cobinder')
    assert [res.name for res in st[0]['X']] == ['FAD']


def test_chain_without_four_resolved_known_residues_is_not_a_partner():
    assert local_records('6r18') == []


def test_poorer_view_of_the_same_ligand_copy_is_superseded():
    rows = local_records('6gch')
    assert [(row['space'], row['protein_chains']) for row in rows] == [('assembly_1', 3)]


@pytest.mark.parametrize('pdb_id,cls,recruiter', [('9ggl', 'molecular_glue', True), ('8qs2', 'molecular_glue_non_degrader', False), ('10ay', 'molecular_glue', True)])
def test_glues_qualify_with_or_without_a_ligase(pdb_id, cls, recruiter):
    rows = [row for row in local_records(pdb_id) if row['tpd_tier'] == 'high_confidence']
    assert rows and all(row['class'] == cls and row['recruiter_in_pair'] is recruiter for row in rows)


def test_undeposited_bond_to_a_nucleophile_is_covalent_not_a_clash(tmp_path):
    row, = local_records('29hh')
    bond, = row['covalent_bonds']
    assert (bond['partner_name'], bond['partner_atom'], bond['source']) == ('CYS', 'SG', 'inferred_from_distance')
    assert row['covalent'] and row['quality_flags'] == [] and row['resolution'] > 4
    _, st = export(row, tmp_path)
    assert len(st.connections) == 1


def test_tight_contact_to_a_non_nucleophile_is_neither_bond_nor_clash():
    row = next(row for row in local_records('6bn7') if row['ccd'] == 'RN3')
    assert not row['covalent'] and row['tpd_tier'] == 'high_confidence'


def test_same_sequences_and_ligand_in_a_different_pose_stay_separate():
    rows = [dict(row, plp_training_eligible=True) for pdb_id in ('8bb3', '8bb2', '29hg', '29hh') for row in local_records(pdb_id) if row['ccd'] in ('Q3X', 'A1J20')]
    selected, audit = b.select_conformers(rows, workers=2)
    assert sorted(r['pdb_id'] for r in selected) == ['29HG', '8BB2', '8BB3']
    moved = next(a for record, a in audit.items() if record.startswith('29HH'))
    assert moved['representative'].startswith('29HG') and max(moved['partner_ca_rmsd'], moved['ligand_rmsd']) <= b.CONFORMER_RMSD


def test_interchangeable_chains_do_not_split_a_pose():
    chains = [('SEQ', {i: [float(i), 0., 0.] for i in range(8)}), ('SEQ', {i: [float(i), 5., float(i % 2)] for i in range(8)})]
    ligand = {'C1': [1., 2., 0.], 'C2': [2., 2., 1.]}
    assert max(b.pose_difference((chains[::-1], ligand), (chains, ligand))) < 1e-6


def test_tpd_override_promotes_rejects_and_is_idempotent():
    overrides = {'P_1': {'record': 'P_1', 'decision': 'pass', 'class': 'molecular_glue', 'note': 'CRBN glue'},
                 'F_1': {'record': 'F_1', 'decision': 'fail', 'class': '', 'note': 'ligand incomplete'}}
    passed = {'record': 'P_1', 'tpd_tier': 'review', 'tpd_reasons': '["no_explicit_TPD_or_glue_annotation"]', 'class': 'unresolved', 'class_source': 'entry_annotation'}
    failed = {'record': 'F_1', 'tpd_tier': 'review', 'tpd_reasons': ['ligand_less_than_90_percent_complete'], 'class': 'molecular_glue', 'class_source': 'structure_annotation'}
    b.apply_tpd_override(passed, overrides); b.apply_tpd_override(failed, overrides)
    assert (passed['tpd_tier'], passed['tpd_reasons'], passed['class'], passed['class_source']) == ('high_confidence', [], 'molecular_glue', 'manual_review')
    assert passed['manual_review']['screen_reasons'] == ['no_explicit_TPD_or_glue_annotation']
    assert failed['tpd_tier'] == 'rejected' and failed['class'] == 'molecular_glue'
    assert failed['tpd_reasons'] == ['ligand_less_than_90_percent_complete', 'manual_review_rejected']
    # Reapplying to a row read back from TSV keeps the original screen verdict.
    reread = {**failed, 'manual_review': json.dumps(failed['manual_review']), 'tpd_reasons': json.dumps(failed['tpd_reasons'])}
    b.apply_tpd_override(reread, overrides)
    assert reread['tpd_reasons'] == failed['tpd_reasons'] and reread['manual_review']['screen_tier'] == 'review'
    untouched = {'record': 'X_1', 'tpd_tier': 'review'}
    assert b.apply_tpd_override(untouched, overrides) == {'record': 'X_1', 'tpd_tier': 'review'}


def test_shipped_tpd_overrides_are_well_formed():
    overrides = b.load_tpd_overrides(b.TPD_OVERRIDES)
    assert all(o['class'] in ('', 'molecular_glue', 'molecular_glue_non_degrader', 'protac') for o in overrides.values())
    assert all(o['decision'] == 'fail' or o['class'] for o in overrides.values())
