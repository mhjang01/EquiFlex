#!/usr/bin/env python3
"""Mine drug-like P–L–P complexes and a conservative TPD subset from a local PDB mirror."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import screen_tpd as t


def metadata(path):
    try:
        b = t.header(path)
        def first(tag):
            return next(iter(t.vals(b, tag)), '')
        revisions = t.vals(b, '_pdbx_audit_revision_history.revision_date') or t.vals(b, '_database_PDB_rev.date')
        dates = [d for d in revisions if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d)]
        fields = {k: ' '.join(v for tag in tags for v in t.vals(b, tag)) for k, tags in t.FIELDS.items()}
        fields['ligand_name'] = ' '.join(t.vals(b, '_pdbx_entity_nonpoly.name'))
        terms = {k: [f for f, text in fields.items() if re.search(rx, text, re.I)] for k, rx in t.TERMS.items()}
        terms = {k: v for k, v in terms.items() if v}
        proteins = [eid for eid, typ in t.rows(b, '_entity_poly.', ['entity_id', 'type']) if typ in ('polypeptide(L)', 'polypeptide(D)')]
        nonpoly = {ccd for eid, ccd in t.rows(b, '_pdbx_entity_nonpoly.', ['entity_id', 'comp_id'])}
        plausible = []
        for ccd, name, formula in t.rows(b, '_chem_comp.', ['id', 'name', 'formula']):
            if ccd not in nonpoly or ccd in t.EXCLUDED or re.search(t.BAD_NAME, name, re.I):
                continue
            counts = {el: int(n or 1) for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula)}
            if counts.get('C', 0) >= 6 and sum(n for el, n in counts.items() if el != 'H') >= 12:
                plausible.append(ccd)
        return {
            'pdb_id': first('_entry.id').upper() or Path(path).name.split('.')[0].upper(),
            'path': str(path), 'deposition_date': first('_pdbx_database_status.recvd_initial_deposition_date'),
            'release': min(dates) if dates else '', 'last_revision_date': max(dates) if dates else '',
            'protein_entities': proteins, 'plausible_ligands': plausible,
            'fields': fields, 'terms': terms,
            'machinery': bool(re.search(t.LIGASE, fields['entity'] + ' ' + fields['synonym'], re.I)),
        }
    except Exception as exc:
        return {'path': str(path), 'error': str(exc)}


def scan(args, out):
    paths = sorted(Path(args.pdb).glob('*/*.cif.gz'))
    config = {'pdb_root': str(Path(args.pdb).resolve()), 'cutoff': args.cutoff}
    request = out / 'scan_request.json'
    if args.resume and request.exists() and json.loads(request.read_text()) != config:
        raise ValueError('Resume requires the same archive root and cutoff.')
    request.write_text(json.dumps(config, indent=2))
    audit = out / 'metadata.jsonl'
    done = set()
    if args.resume and audit.exists():
        # Truncate only an incomplete final write, leaving completed records intact.
        with audit.open('rb+') as f:
            while line := f.readline():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    f.seek(-len(line), 1); f.truncate(); break
                done.add(m['path'])
    pending = [p for p in paths if str(p) not in done]
    print(f'Metadata: {len(paths):,} files; {len(done):,} cached; {len(pending):,} remaining', flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, audit.open('a' if args.resume else 'w') as f:
        for i, m in enumerate(pool.map(metadata, map(str, pending), chunksize=16), 1):
            f.write(json.dumps(m) + '\n')
            if i % 5000 == 0:
                f.flush()
                print(f'Metadata {len(done)+i:,}/{len(paths):,}; {i/(time.monotonic()-started):.1f} files/s', flush=True)
    counts = Counter(); latest = ''; entries = []; errors = []
    with audit.open() as f:
        for line in f:
            m = json.loads(line)
            if 'error' in m:
                errors.append(m); counts['parse_error'] += 1; continue
            latest = max(latest, m['release'])
            if not m['release']: scope = 'missing_release'
            elif m['release'] > args.cutoff: scope = 'after_cutoff'
            else: scope = 'through_cutoff'
            counts[scope] += 1
            entries.append({k: m[k] for k in ['pdb_id', 'path', 'deposition_date', 'release', 'last_revision_date', 'protein_entities', 'plausible_ligands']})
    t.write_tsv(out / 'entries.tsv', entries)
    t.write_tsv(out / 'metadata_errors.tsv', errors)
    summary = {**config, 'files': len(paths), 'counts': dict(counts), 'latest_initial_release': latest}
    (out / 'scan_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdb', default='/home/mhjang/DB/PDB')
    p.add_argument('--out', default='tpd/full_db_20260630')
    p.add_argument('--cutoff', default='2026-06-30')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage', choices=['metadata', 'structures', 'export', 'all', 'tpd_review'], default='all',
                   help='tpd_review reapplies --tpd-overrides to an exported catalog and exports only newly selected TPD records')
    p.add_argument('--keep-homomers', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ccd-root', default=str(t.CCD_ROOT))
    p.add_argument('--tpd-overrides', default=str(TPD_OVERRIDES))
    args = p.parse_args(); date.fromisoformat(args.cutoff)
    if args.workers < 1: p.error('--workers must be positive')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('metadata', 'all'): scan(args, out)
    if args.stage in ('structures', 'all'): structures(args, out)
    if args.stage in ('export', 'all'): export(args, out)
    if args.stage == 'tpd_review': tpd_review(args, out)


# Chemistry is reused across structures within each worker; never load arbitrary pickles.
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree


@lru_cache(maxsize=4096)
def cached_chemistry(ccd, name, formula):
    return t.chemistry(ccd, name, formula)


def worker_init(ccd_root):
    t.init_chemistry(ccd_root)


# AlphaFold 3 supplement, Table 9; removed only from crystallographic entries.
CRYSTALLIZATION_AIDS = frozenset('SO4 GOL EDO PO4 ACT PEG DMS TRS PGE PG4 FMT EPE MPD MES CD IOD'.split())
# Side-chain atoms that form covalent adducts, with the bond-length window accepted when no link is deposited.
NUCLEOPHILES = {('CYS', 'SG'): (1.5, 2.15), ('SEC', 'SE'): (1.5, 2.3), ('SER', 'OG'): (1.2, 1.75), ('THR', 'OG1'): (1.2, 1.75),
                ('TYR', 'OH'): (1.2, 1.75), ('LYS', 'NZ'): (1.2, 1.75), ('HIS', 'ND1'): (1.2, 1.75), ('HIS', 'NE2'): (1.2, 1.75),
                ('ASP', 'OD1'): (1.2, 1.75), ('ASP', 'OD2'): (1.2, 1.75), ('GLU', 'OE1'): (1.2, 1.75), ('GLU', 'OE2'): (1.2, 1.75)}
MAX_RESOLUTION = 9.0
TIER_RANK = {'high_confidence': 0, 'review': 1, 'glue_or_proximity_review': 2, 'rejected': 3}


def base_subchain(name):
    """Deposited label_asym_id of a subchain copied into an assembly."""
    return re.sub(r'\d+$', '', name)


def covalent_links(structure):
    """Deposited covalent bonds of every residue, keyed by (label asym, residue number, name)."""
    links = [c for c in structure.connections if c.type == t.gemmi.ConnectionType.Covale]
    if not links:
        return {}
    index = {(chain.name, str(res.seqid), res.name): res for chain in structure[0] for res in chain}
    bonds = {}
    for link in links:
        ends = []
        for partner in (link.partner1, link.partner2):
            res = index.get((partner.chain_name, str(partner.res_id.seqid), partner.res_id.name))
            if res is not None and res.find_atom(partner.atom_name, '*') is not None:
                ends.append((res, partner.atom_name))
        if len(ends) < 2:
            continue
        for (res, atom), (other, other_atom) in (ends, ends[::-1]):
            bonds.setdefault((res.subchain, str(res.seqid), res.name), []).append({
                'atom': atom, 'partner_subchain': other.subchain, 'partner_residue': str(other.seqid),
                'partner_name': other.name, 'partner_atom': other_atom, 'partner_entity': other.entity_id,
                'partner_polymer': other.entity_type == t.gemmi.EntityType.Polymer})
    return bonds


def geometry_contacts(model, seqs, allowed, desc, ccd_names=None, bonds=None, aids=frozenset()):
    """Ligand-centred contacts: one result per ligand copy touching at least two protein chains.

    Polymer chains are screened with the AlphaFold 3 bioassembly filters before
    they count as partners or context. `ccd_names` restricts ligand atoms to the
    component's defined names, `bonds` holds deposited covalent links and `aids`
    names non-polymers that never enter the context. A ligand atom at bonding
    distance from a protein nucleophile is a covalent bond even when undeposited.
    """
    ccd_names = ccd_names or {}; bonds = bonds or {}
    coords = []; owners = []; residue_ids = []; residue_names = []; atom_names = []; keys = []; key_indices = {}; info = []
    ligands = []; hets = []; het_coords = []; het_owners = []
    for chain in model:
        for res in chain:
            atoms = [a for a in res if a.occ > 0 and not a.is_hydrogen()]
            if not atoms:
                continue
            if res.entity_type == t.gemmi.EntityType.Polymer:
                key = (chain.name, res.subchain, res.entity_id)
                if key not in key_indices:
                    key_indices[key] = len(keys); keys.append(key)
                    info.append({'residues': 0, 'atoms': 0, 'known': False, 'ca': []})
                ci = key_indices[key]; chain_info = info[ci]
                coords.extend(a.pos.tolist() for a in atoms)
                owners.extend([ci] * len(atoms))
                residue_ids.extend([str(res.seqid)] * len(atoms)); residue_names.extend([res.name] * len(atoms))
                atom_names.extend(a.name for a in atoms)
                chain_info['residues'] += 1; chain_info['atoms'] += len(atoms)
                chain_info['known'] |= res.name not in ('UNK', 'N', 'DN')
                ca = res.find_atom('CA', '*')
                if res.entity_id in seqs and ca is not None and ca.occ > 0 and res.label_seq is not None:
                    chain_info['ca'].append((res.label_seq, ca.pos))
            elif not res.is_water():
                index = len(hets); hets.append((chain.name, res, len(atoms)))
                het_coords.extend(a.pos.tolist() for a in atoms); het_owners.extend([index] * len(atoms))
                if res.entity_type == t.gemmi.EntityType.NonPolymer and res.name in allowed:
                    names = ccd_names.get(res.name)
                    kept = [a for a in atoms if a.name in names] if names else atoms
                    if len(kept) >= 8:
                        ligands.append((index, chain.name, res, kept, sorted(a.name for a in atoms if names and a.name not in names)))
    if sum(key[2] in seqs for key in keys) < 2 or not ligands:
        return
    xyz_polymer = np.asarray(coords); owners = np.asarray(owners)
    tree = cKDTree(xyz_polymer); het_tree = cKDTree(np.asarray(het_coords))
    verdicts = {}

    def chain_filter(ci):
        """Reason an AlphaFold 3 bioassembly filter removes the chain, '' when it stays."""
        if ci in verdicts:
            return verdicts[ci]
        chain_info = info[ci]; reason = ''
        ca = sorted(chain_info['ca'], key=lambda x: x[0])
        if chain_info['residues'] < 4: reason = 'fewer_than_4_resolved_residues'
        elif not chain_info['known']: reason = 'all_unknown_residues'
        elif any(b[0] - a[0] == 1 and a[1].dist(b[1]) > 10 for a, b in zip(ca, ca[1:])): reason = 'consecutive_CA_over_10A'
        else:
            mine = {}; theirs = {}
            for k, indices in enumerate(tree.query_ball_point(xyz_polymer[owners == ci], 1.7)):
                for pi in indices:
                    cj = owners[pi]
                    if cj != ci:
                        mine.setdefault(cj, set()).add(k); theirs.setdefault(cj, set()).add(pi)
            for cj in mine:
                own = len(mine[cj]) / chain_info['atoms']; other = len(theirs[cj]) / info[cj]['atoms']
                if max(own, other) > .3 and (own, -chain_info['atoms'], keys[ci]) > (other, -info[cj]['atoms'], keys[cj]):
                    reason = 'clashing_chain'
        verdicts[ci] = reason
        return reason

    for index, chain_name, res, atoms, removed in ligands:
        xyz = np.asarray([a.pos.tolist() for a in atoms])
        own_bonds = bonds.get((base_subchain(res.subchain), str(res.seqid), res.name), [])
        polymer_bonds = {(bond['atom'], bond['partner_subchain'], bond['partner_residue'], bond['partner_atom']): bond
                         for bond in own_bonds if bond['partner_polymer']}
        by_chain = {}; covalent = []
        for li, indices in enumerate(tree.query_ball_point(xyz, 4.5)):
            for pi in indices:
                ci = owners[pi]
                d = by_chain.setdefault(ci, {'ligand_atoms': set(), 'protein_atoms': set(), 'residues': set(), 'min_distance': 999.0, 'min_nonbonded_distance': 999.0})
                distance = float(np.linalg.norm(xyz[li] - xyz_polymer[pi]))
                d['ligand_atoms'].add(li); d['protein_atoms'].add(pi); d['residues'].add(residue_ids[pi])
                d['min_distance'] = min(d['min_distance'], distance)
                bond = polymer_bonds.get((atoms[li].name, base_subchain(keys[ci][1]), residue_ids[pi], atom_names[pi])) if distance < 2.5 else None
                low, high = NUCLEOPHILES.get((residue_names[pi], atom_names[pi]), (0, 0))
                if bond is None and not low <= distance <= high:
                    d['min_nonbonded_distance'] = min(d['min_nonbonded_distance'], distance)
                else:
                    covalent.append({'ligand_atom': atoms[li].name, 'partner_kind': 'protein' if keys[ci][2] in seqs else 'polymer',
                                     'partner_subchain': keys[ci][1], 'partner_residue': residue_ids[pi],
                                     'partner_name': residue_names[pi], 'partner_atom': atom_names[pi], 'distance': round(distance, 3),
                                     'source': 'struct_conn' if bond else 'inferred_from_distance'})
        contacts = []; context_polymers = []; filtered = Counter()
        for ci, d in by_chain.items():
            reason = chain_filter(ci)
            if reason:
                filtered[reason] += 1; continue
            auth, sub, eid = keys[ci]
            if eid in seqs and len(d['ligand_atoms']) >= 3 and len(d['residues']) >= 2 and d['min_distance'] <= 4.0:
                contacts.append({'chain': auth, 'subchain': sub, 'entity': eid, 'name': desc.get(eid, ''),
                                 **{k: len(v) if isinstance(v, set) else v for k, v in d.items()}})
            else:
                context_polymers.append(sub)
        if len(contacts) < 2:
            continue
        covalent = [bond for bond in covalent if bond['partner_subchain'] in {c['subchain'] for c in contacts} | set(context_polymers)]
        het_bonds = [bond for bond in own_bonds if not bond['partner_polymer']]
        context_ligands = []
        for hi in sorted({het_owners[i] for indices in het_tree.query_ball_point(xyz, 4.5) for i in indices} - {index}):
            _, other, size = hets[hi]
            bond = next((b for b in het_bonds if (b['partner_subchain'], b['partner_residue'], b['partner_name']) == (base_subchain(other.subchain), str(other.seqid), other.name)), None)
            if other.name in aids and bond is None:
                continue
            context_ligands.append({'subchain': other.subchain, 'name': other.name, 'residue': str(other.seqid), 'atoms': size})
        for bond in het_bonds:
            partner = next((h for h in context_ligands if (base_subchain(h['subchain']), h['residue'], h['name']) == (bond['partner_subchain'], bond['partner_residue'], bond['partner_name'])), None)
            covalent.append({'ligand_atom': bond['atom'], 'partner_kind': 'nonpolymer', 'partner_subchain': partner['subchain'] if partner else None,
                             'partner_residue': bond['partner_residue'], 'partner_name': bond['partner_name'], 'partner_atom': bond['partner_atom'], 'distance': None, 'source': 'struct_conn'})
        yield chain_name, res, atoms, contacts, {'context_polymers': context_polymers, 'context_ligands': context_ligands, 'covalent_bonds': covalent,
                                                'removed_non_ccd_atoms': removed, 'filtered_chains': dict(filtered)}


def classify_tpd(m, names, recruiters, homo, quality, assembly_supported):
    """Tier and class of one contact pair.

    Any heteromeric pair bridged by a ligand the depositors call a PROTAC, glue or
    degrader qualifies. Glues with no recognised ligase or degradation wording
    are induced-proximity glues, classed `molecular_glue_non_degrader`.
    """
    reasons = list(quality)
    if homo: reasons.append('homomeric_or_near_identical_proteins')
    scaffold = (any(re.search(r'elongin|transcription elongation factor B|\bTCEB[123]\b|\bELOC\b|\bELOB\b|cullin|RING.box|\bRBX[12]\b', n, re.I) for n in names) and any(recruiters)) or (any(re.search(r'DNA damage.binding protein 1|\bDDB1\b', n, re.I) for n in names) and all(recruiters))
    if scaffold: reasons.append('ligase_scaffold_pair')
    if any(re.search(r'nanobody|antibody|Fab fragment|immunoglobulin.*(?:chain|fragment)|crystallization chaperone|maltose.binding protein|streptavidin|affibody|DARPin', n, re.I) for n in names):
        reasons.append('possible_antibody_or_crystallization_partner')
    core = set(m['terms']) & {'protac', 'glue', 'degrader'}
    if not core: reasons.append('no_explicit_TPD_or_glue_annotation')
    if not assembly_supported: reasons.append('not_supported_in_deposited_biological_assembly')
    tier = 'high_confidence' if not reasons else 'rejected' if homo or scaffold else 'glue_or_proximity_review' if core and not any(recruiters) else 'review'
    specific = {k for k in ('protac', 'glue') if set(m['terms'].get(k, [])) & {'title', 'keywords', 'descriptor'}}
    source = 'structure_annotation' if any(set(m['terms'].get(k, [])) & {'title', 'keywords', 'descriptor'} for k in core) else 'entry_annotation'
    subtype = specific or (set(m['terms']) & {'protac', 'glue'})
    cls = 'protac' if subtype == {'protac'} else 'molecular_glue' if subtype == {'glue'} else 'unresolved'
    if cls == 'molecular_glue' and not any(recruiters) and 'degrader' not in m['terms']: cls = 'molecular_glue_non_degrader'
    return tier, reasons, cls, source


def evaluate_entry(m):
    try:
        b = t.gemmi.cif.read(m['path']).sole_block()
        s = t.gemmi.make_structure_from_block(b)
        t.clean_structure_for_assembly(s)
        desc = dict(t.rows(b, '_entity.', ['id', 'pdbx_description']))
        aliases = dict(desc)
        for eid, name in t.rows(b, '_entity_name_com.', ['entity_id', 'name']) + t.rows(b, '_entity_src_gen.', ['entity_id', 'pdbx_gene_src_gene']):
            aliases[eid] = aliases.get(eid, '') + ' ' + name
        seqs = {e.name: t.gemmi.one_letter_code(e.full_sequence) for e in s.entities if e.polymer_type in (t.gemmi.PolymerType.PeptideL, t.gemmi.PolymerType.PeptideD)}
        uniprots = {}
        for eid, db, acc in t.rows(b, '_struct_ref.', ['entity_id', 'db_name', 'pdbx_db_accession']):
            if db == 'UNP': uniprots.setdefault(eid, set()).add(acc)
        chem = {}; rejected = Counter()
        for ccd, name, formula in t.rows(b, '_chem_comp.', ['id', 'name', 'formula']):
            if ccd not in m['plausible_ligands']: continue
            props, reasons, mol = cached_chemistry(ccd, name, formula)
            if reasons: rejected.update(reasons)
            else: chem[ccd] = (name, props, mol)
        if not chem:
            return {'pdb_id': m['pdb_id'], 'records': [], 'ligand_rejection_counts': dict(rejected), 'outcome': 'no_druglike_ligand'}
        hom_cache = {}; groups = {}; instances = 0; filtered_chains = Counter()
        ccd_names = {ccd: set(t.expected_atoms(b, ccd, mol)[0]) for ccd, (_, _, mol) in chem.items()}
        bonds = covalent_links(s)
        crystallographic = any(re.search(r'DIFFRACTION|CRYSTALLOGRAPHY', method, re.I) for method in t.vals(b, '_exptl.method'))
        aids = CRYSTALLIZATION_AIDS if crystallographic else frozenset()
        def models():
            yield 'asu', s[0]
            for assembly in s.assemblies:
                yield 'assembly_' + assembly.name, t.gemmi.make_assembly(assembly, s[0], t.gemmi.HowToNameCopiedChain.AddNumber)
        for space, model in models():
            assembly = space.startswith('assembly_')
            for lc, res, atoms, contacts, context in geometry_contacts(model, seqs, chem, desc, ccd_names, bonds, aids):
                ccd = res.name; name, props, mol = chem[ccd]
                filtered_chains.update(context.pop('filtered_chains'))
                covalent = context['covalent_bonds']
                completeness_info = t.ligand_completeness(b, ccd, mol, atoms, covalent=bool(covalent))
                completeness = completeness_info['ligand_completeness']
                quality = []
                if completeness is None: quality.append('ligand_completeness_unknown')
                elif completeness < .9: quality.append('ligand_less_than_90_percent_complete')
                if completeness is None or completeness_info['atom_element_mismatches'] or props.get('chemistry_source') != 'local_CCD': quality.append('chemistry_unverified')
                if not all(seqs.get(c['entity']) for c in contacts): quality.append('sequence_unverified')
                if s.resolution <= 0 or s.resolution > MAX_RESOLUTION: quality.append('resolution_missing_or_over_9A')
                if min(a.occ for a in atoms) < .5: quality.append('ligand_occupancy_under_half')
                # Covalent bonds, deposited or inferred, are legitimate; anything else this close is a clash.
                if min(c['min_nonbonded_distance'] for c in contacts) < 1.8: quality.append('clashing_contact')
                if any(bond['partner_kind'] == 'nonpolymer' for bond in covalent): quality.append('ligand_bonded_to_other_nonpolymer')
                # Every pair is classified; the best one names the record's primary chains A and B.
                pairs = []
                for ca, cb in itertools.combinations(contacts, 2):
                    instances += 1
                    ea, eb = ca['entity'], cb['entity']; hk = tuple(sorted([ea, eb]))
                    if hk not in hom_cache: hom_cache[hk] = t.homology(ea, eb, seqs, uniprots)
                    homo, identity, method = hom_cache[hk]
                    recruiters = [bool(re.search(t.LIGASE, aliases.get(e, ''), re.I)) for e in (ea, eb)]
                    verdict = classify_tpd(m, [ca['name'], cb['name']], recruiters, bool(homo), quality, assembly)
                    pairs.append((TIER_RANK[verdict[0]], -min(ca['ligand_atoms'], cb['ligand_atoms']), len(pairs), ca, cb, bool(homo), identity, method, verdict))
                _, _, _, ca, cb, _, identity, method, (tier, reasons, cls, cls_source) = min(pairs, key=lambda p: p[:3])
                ea, eb = ca['entity'], cb['entity']
                contacts = [ca, cb] + sorted((c for c in contacts if c is not ca and c is not cb), key=lambda c: -c['ligand_atoms'])
                for c in contacts[2:]: c['sequence'] = seqs.get(c['entity'], '')
                group = hashlib.sha256(('\n'.join(sorted(seqs.get(c['entity'], '') for c in contacts)) + '\n' + props.get('smiles', ccd)).encode()).hexdigest()[:20]
                broad_reasons = list(quality)
                if completeness is not None and completeness < 1: broad_reasons.append('ligand_not_fully_observed')
                if not assembly: broad_reasons.append('not_supported_in_deposited_biological_assembly')
                row = {
                    'record': f"{m['pdb_id']}_{group}", 'pdb_id': m['pdb_id'], 'complex_group': group,
                    'deposition_date': m['deposition_date'], 'release': m['release'], 'last_revision_date': m['last_revision_date'],
                    'coordinates_revised_after_cutoff': m['last_revision_date'] > m['parse_cutoff'], 'source_path': m['path'], 'space': space, 'ccd': ccd, 'ligand_name': name,
                    'ligand_chain': lc, 'ligand_subchain': res.subchain, 'ligand_residue': str(res.seqid),
                    'chain_A': ca['chain'], 'chain_B': cb['chain'], 'subchain_A': ca['subchain'], 'subchain_B': cb['subchain'],
                    'protein_A': ca['name'], 'protein_B': cb['name'], 'entity_A': ea, 'entity_B': eb,
                    'sequence_A': seqs.get(ea, ''), 'sequence_B': seqs.get(eb, ''),
                    'uniprot_A': sorted(uniprots.get(ea, set())), 'uniprot_B': sorted(uniprots.get(eb, set())),
                    'homomeric': all(p[5] for p in pairs), 'sequence_identity': identity, 'homology_check': method,
                    'protein_chains': len(contacts), 'contacts': contacts, **context,
                    'covalent': any(bond['partner_kind'] == 'protein' for bond in covalent),
                    'observed_heavy_atoms': len(atoms), **completeness_info,
                    'resolution': s.resolution, 'crystallographic': crystallographic, 'quality_flags': broad_reasons,
                    'plp_training_eligible': not broad_reasons,
                    'tpd_tier': tier, 'tpd_reasons': reasons, 'class': cls, 'class_source': cls_source,
                    'recruiter_in_pair': any(re.search(t.LIGASE, aliases.get(e, ''), re.I) for e in (ea, eb)),
                    'terms': m['terms'], 'title': m['fields']['title'], 'citation': m['fields']['citation'],
                    'tpd_tagged': bool(m['terms'] or m['machinery']), **props,
                }
                # Prefer experimentally supported, complete assembly instances before ASU copies.
                score = (bool(broad_reasons), not assembly, len(broad_reasons), -(completeness or 0), -min(ca['ligand_atoms'], cb['ligand_atoms']), lc, str(res.seqid))
                location = [space, lc, res.subchain, str(res.seqid), *(c['subchain'] for c in contacts)]
                if group not in groups:
                    groups[group] = {'row': row, 'score': score, 'copies': [location]}
                else:
                    item = groups[group]; item['copies'].append(location)
                    if score < item['score']: item.update(row=row, score=score)
        # A ligand copy seen with fewer chains in one space is the same site as its richer instance elsewhere.
        richest = {}
        for item in groups.values():
            for location in item['copies']:
                ligand = (base_subchain(location[2]), location[3])
                richest[ligand] = max(richest.get(ligand, 0), len(location))
        records = []; superseded = 0
        for item in groups.values():
            if all(richest[(base_subchain(location[2]), location[3])] > len(location) for location in item['copies']):
                superseded += 1; continue
            row = item['row']; row['instance_count'] = len(item['copies']); row['equivalent_instances'] = item['copies']; records.append(row)
        return {'pdb_id': m['pdb_id'], 'records': records, 'pair_instances': instances, 'filtered_chain_counts': dict(filtered_chains), 'superseded_groups': superseded,
                'ligand_rejection_counts': dict(rejected), 'outcome': 'has_plp' if records else 'no_qualifying_contact_pair'}
    except Exception as exc:
        return {'pdb_id': m['pdb_id'], 'records': [], 'error': str(exc)}


def structures(args, out):
    summary = json.loads((out / 'scan_summary.json').read_text())
    if summary['pdb_root'] != str(Path(args.pdb).resolve()): raise ValueError('Archive root differs from metadata cache.')
    jobs = []
    for line in (out / 'metadata.jsonl').open():
        m = json.loads(line)
        if not m.get('release') or m['release'] > args.cutoff or not m.get('protein_entities') or not m.get('plausible_ligands'): continue
        if not args.keep_homomers and len(m['protein_entities']) < 2: continue
        m['parse_cutoff'] = args.cutoff
        jobs.append(m)
    config = {'cutoff': args.cutoff, 'keep_homomers': args.keep_homomers, 'ccd_root': args.ccd_root}
    config_path = out / 'structure_request.json'
    if args.resume and config_path.exists() and json.loads(config_path.read_text()) != config: raise ValueError('Structural resume configuration mismatch.')
    config_path.write_text(json.dumps(config, indent=2))
    path = out / 'structure_results.jsonl'; done = set()
    if args.resume and path.exists():
        with path.open('rb+') as f:
            while line := f.readline():
                try: result = json.loads(line)
                except json.JSONDecodeError:
                    f.seek(-len(line), 1); f.truncate(); break
                done.add(result['pdb_id'])
    jobs = [m for m in jobs if m['pdb_id'] not in done]
    print(f'Structures: {len(jobs):,} remaining, {len(done):,} cached', flush=True)
    from multiprocessing import Pool
    started = time.monotonic()
    with Pool(args.workers, initializer=worker_init, initargs=(args.ccd_root,)) as pool, path.open('a' if args.resume else 'w') as f:
        for i, result in enumerate(pool.imap_unordered(evaluate_entry, jobs, chunksize=1), 1):
            f.write(json.dumps(result) + '\n')
            if i % 1000 == 0:
                f.flush(); print(f'Structures {len(done)+i:,}; {i/(time.monotonic()-started):.1f} entries/s', flush=True)
    print('Structural scan complete', flush=True)



def truth(value):
    return value is True or str(value).lower() in ('true', '1')


@lru_cache(maxsize=16384)
def additional_chemistry_flags(ccd, name, smiles):
    """Conservative review flags for common whole-archive chemical confounders."""
    flags = []
    excluded = {
        'TRP', 'PHE', 'TYR', 'CHD', 'BGC', 'SIA', 'FBP', 'G6P',
        'C2E', '2BA', '1SY', 'TYD', 'PLG', 'TZD', 'BTI',
        'NHE', 'MPO', 'CXS', 'BOG', 'DMU', 'LMN', 'AV0', 'Q7G', 'HTG', 'BNG',
        'APR', '5AD', 'FNR', 'MTA', 'IMP', 'XMP', '4BW', 'DCS', 'PGU', 'IAC',
        'A8W', 'DU0', 'TRT', 'JCV',
    }
    if ccd in excluded: flags.append('known_metabolite_cofactor_analog_or_additive')
    if re.search(r'phyco.*bilin|bilirubin|biliverdin|\bperidinin\b|di.?acyl.?glycerol|di.o.acyl|diacyl glycerol|maltopyranoside|lauryl maltose|cholic acid|triton|pregnenolone|indol.?3.ylacetic', name, re.I):
        flags.append('lipid_detergent_or_pigment_name')
    if not smiles: return tuple(flags)
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return tuple(flags)
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    rings = mol.GetRingInfo().AtomRings()
    sugar_rings = 0
    for ring in rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if len(ring) not in (5, 6) or any(a.GetIsAromatic() for a in atoms): continue
        if sum(a.GetSymbol() == 'O' for a in atoms) != 1 or any(a.GetSymbol() not in ('C', 'O') for a in atoms): continue
        external_o = {n.GetIdx() for a in atoms for n in a.GetNeighbors() if n.GetIdx() not in ring and n.GetSymbol() == 'O'}
        if len(external_o) >= 3: sugar_rings += 1
    tail = Chem.MolFromSmarts('[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]')
    esters = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2][CX4]')))
    if mol.HasSubstructMatch(tail) and (sugar_rings or (esters >= 2 and counts['N'] <= 1 and counts['C'] >= 18)):
        flags.append('long_chain_lipid_or_glycoside_detergent_motif')
    if sugar_rings and sugar_rings == len(rings) and not (set(counts) - {'C', 'O', 'P', 'S', 'N'}) and counts['N'] <= sugar_rings and counts['C'] <= (6 + 2 * bool(counts['N'])) * sugar_rings:
        flags.append('sugar_or_sugar_phosphate_scaffold')
    if sugar_rings and counts['C'] >= 40 and counts['P'] and not counts['N']:
        flags.append('large_glycophospholipid_scaffold')
    cc_unsaturated = sum(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and b.GetBeginAtom().GetSymbol() == b.GetEndAtom().GetSymbol() == 'C' for b in mol.GetBonds())
    if counts['C'] >= 30 and not (set(counts) - {'C', 'O'}) and cc_unsaturated >= 7 and len(rings) <= 3:
        flags.append('long_polyene_pigment_or_prenyl_cofactor_motif')
    return tuple(flags)


def apply_additional_chemistry(row):
    flags = list(additional_chemistry_flags(row['ccd'], row['ligand_name'], row.get('smiles', '')))
    row['additional_chemistry_flags'] = flags
    if flags:
        row['quality_flags'] = list(dict.fromkeys(row['quality_flags'] + flags))
        row['plp_training_eligible'] = False
        row['tpd_reasons'] = list(dict.fromkeys(row['tpd_reasons'] + flags))
        if row['tpd_tier'] == 'high_confidence': row['tpd_tier'] = 'review'
    return row


TPD_OVERRIDES = Path(__file__).with_name('tpd_review_overrides.tsv')


def load_tpd_overrides(path):
    """Manual TPD review decisions keyed by record: decision (pass/fail), class (blank keeps the screen's), note."""
    with open(path) as f: overrides = {r['record']: r for r in csv.DictReader(f, delimiter='\t')}
    assert all(r['decision'] in ('pass', 'fail') for r in overrides.values()), 'decision must be pass or fail'
    return overrides


def apply_tpd_override(row, overrides):
    """Promote a reviewed record to high confidence or reject it; the screen's verdict is kept in `manual_review`."""
    o = overrides.get(row['record'])
    if o is None: return row
    screen = t.as_list(row.get('manual_review')) or {'screen_tier': row['tpd_tier'], 'screen_reasons': t.as_list(row['tpd_reasons']),
                                                     'screen_class': row['class'], 'screen_class_source': row['class_source']}
    row['manual_review'] = {**screen, 'decision': o['decision'], 'note': o['note']}
    passed = o['decision'] == 'pass'
    row['tpd_tier'] = 'high_confidence' if passed else 'rejected'
    row['tpd_reasons'] = [] if passed else screen['screen_reasons'] + ['manual_review_rejected']
    row['class'], row['class_source'] = (o['class'], 'manual_review') if o['class'] else (screen['screen_class'], screen['screen_class_source'])
    return row


CONFORMER_RMSD = 2.0


def eligible_rows(rows, kind='plp', date_field='release', start='', end='2026-06-30'):
    """Rows of the requested set inside the date window, best first."""
    eligible = []
    for row in rows:
        value = row.get(date_field, '')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) or not (start or '0001-01-01') <= value <= end:
            continue
        if kind == 'plp' and not truth(row['plp_training_eligible']): continue
        if kind == 'tpd' and row['tpd_tier'] != 'high_confidence': continue
        eligible.append(row)
    def quality(row):
        resolution = float(row.get('resolution') or 0)
        return (resolution if resolution > 0 else 999, -float(row['ligand_completeness']), row['release'], row['pdb_id'], row['record'])
    return sorted(eligible, key=quality)


def select_rows(rows, kind='plp', date_field='release', start='', end='2026-06-30'):
    """Apply dates BEFORE deduplication; one record per sequence/ligand group regardless of pose."""
    selected = {}
    for row in eligible_rows(rows, kind, date_field, start, end): selected.setdefault(row['complex_group'], row)
    return sorted(selected.values(), key=lambda r: (r['pdb_id'], r['record']))


def pose(row):
    """C-alpha coordinates of every contact chain and the ligand atoms, read from the source entry."""
    s = t.gemmi.read_structure(row['source_path']); t.clean_structure_for_assembly(s)
    model = s[0] if row['space'] == 'asu' else t.gemmi.make_assembly(
        next(a for a in s.assemblies if 'assembly_' + a.name == row['space']), s[0], t.gemmi.HowToNameCopiedChain.AddNumber)
    contacts = t.as_list(row['contacts'])
    sequences = [row['sequence_A'], row['sequence_B'], *(c['sequence'] for c in contacts[2:])]
    chains = {c['subchain']: {} for c in contacts}; ligand = {}
    for chain in model:
        for res in chain:
            if res.subchain in chains and res.entity_type == t.gemmi.EntityType.Polymer:
                ca = res.find_atom('CA', '*')
                if ca is not None and ca.occ > 0 and res.label_seq is not None: chains[res.subchain][res.label_seq] = ca.pos.tolist()
            elif res.subchain == row['ligand_subchain'] and res.name == row['ccd'] and str(res.seqid) == row['ligand_residue']:
                ligand = {a.name: a.pos.tolist() for a in res if a.occ > 0 and not a.is_hydrogen()}
    return row['record'], ([(seq, chains[c['subchain']]) for seq, c in zip(sequences, contacts)], ligand)


def pose_difference(pose, reference):
    """Partner C-alpha and ligand RMSD after superposing the best-covered chain.

    Chains of equal sequence are interchangeable, so the smallest difference over
    their assignments is returned; poses that cannot be compared are infinitely apart.
    """
    (chains, ligand), (ref_chains, ref_ligand) = pose, reference
    names = sorted(ligand.keys() & ref_ligand.keys()); best = (float('inf'), float('inf'))
    if len(chains) != len(ref_chains) or not names: return best
    rmsd = lambda x, y: float(np.sqrt(((x - y) ** 2).sum(1).mean()))
    for order in itertools.islice(itertools.permutations(range(len(chains))), 720):
        if any(chains[i][0] != ref_chains[j][0] for j, i in enumerate(order)): continue
        pairs = []
        for j, i in enumerate(order):
            common = sorted(chains[i][1].keys() & ref_chains[j][1].keys())
            pairs.append((np.array([chains[i][1][k] for k in common]).reshape(-1, 3), np.array([ref_chains[j][1][k] for k in common]).reshape(-1, 3)))
        anchor = max(range(len(pairs)), key=lambda j: len(pairs[j][0]))
        moving, fixed = pairs[anchor]
        partner = [pair for j, pair in enumerate(pairs) if j != anchor and len(pair[0])]
        if len(moving) < 4 or not partner: continue
        centre, target = moving.mean(0), fixed.mean(0)
        u, _, vt = np.linalg.svd((moving - centre).T @ (fixed - target))
        rotation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        fit = lambda x: (x - centre) @ rotation + target
        found = (rmsd(fit(np.concatenate([p for p, _ in partner])), np.concatenate([q for _, q in partner])),
                 rmsd(fit(np.array([ligand[n] for n in names])), np.array([ref_ligand[n] for n in names])))
        best = min(best, found, key=max)
    return best


def select_conformers(rows, kind='plp', date_field='release', start='', end='2026-06-30', workers=10, poses=None):
    """Representatives of every distinct pose within each sequence/ligand group.

    Members are visited best first; one becomes a further representative when its
    partner chains or its ligand sit more than CONFORMER_RMSD from every
    representative already kept. Returns the representatives and, per eligible
    record, its representative, conformer index and the measured differences.
    """
    groups = {}
    for row in eligible_rows(rows, kind, date_field, start, end): groups.setdefault(row['complex_group'], []).append(row)
    poses = {} if poses is None else poses
    missing = [row for members in groups.values() if len(members) > 1 for row in members if row['record'] not in poses]
    if missing:
        from multiprocessing import Pool
        with Pool(workers) as pool: poses.update(pool.imap_unordered(pose, missing, chunksize=4))
    representatives = []; audit = {}
    for members in groups.values():
        kept = []
        for row in members:
            differences = [pose_difference(poses[row['record']], poses[other['record']]) for other in kept]
            close = [i for i, d in enumerate(differences) if max(d) <= CONFORMER_RMSD]
            if close:
                i = min(close, key=lambda i: max(differences[i]))
                audit[row['record']] = {'representative': kept[i]['record'], 'conformer': i, 'partner_ca_rmsd': round(differences[i][0], 3), 'ligand_rmsd': round(differences[i][1], 3)}
            else:
                audit[row['record']] = {'representative': row['record'], 'conformer': len(kept), 'partner_ca_rmsd': 0.0, 'ligand_rmsd': 0.0}
                kept.append(row)
        representatives.extend(kept)
    return sorted(representatives, key=lambda r: (r['pdb_id'], r['record'])), audit


def export_one(job):
    row, destination = job
    row = dict(row); out = Path(destination)
    try:
        try:
            t.export_complex(row, {'path': row['source_path']}, out)
        except Exception as exc:
            # mmCIF is authoritative; some modified residues cannot fit legacy PDB.
            cif = out / 'reference' / f"{row['record']}.cif"
            sdf = out / 'ligands' / f"{row['record']}.sdf"
            if not cif.exists() or not sdf.exists(): raise
            row['pdb_export_error'] = str(exc)
            broken = out / 'raw' / row['record'] / f"{row['record']}_protein.pdb"
            if broken.exists(): broken.unlink()
        record = row['record']
        reference = out / 'reference' / f'{record}.cif'
        st = t.gemmi.read_structure(str(reference))
        assert len(st[0]['A']) and len(st[0]['B']) and len(st[0]['L']) == 1
        assert all(a.occ > 0 and not a.is_hydrogen() for c in st[0] for res in c for a in res)
        ligand_xyz = np.array([a.pos.tolist() for res in st[0]['L'] for a in res])
        contacts = t.as_list(row['contacts'])
        protein_chains = ['A', 'B', *t.EXTRA_CHAIN_NAMES[:len(contacts) - 2]]
        # The written entities must carry the deposited sequences, numbered by label_seq.
        expected = [row['sequence_A'], row['sequence_B'], *(c['sequence'] for c in contacts[2:])]
        for name, sequence in zip(protein_chains, expected):
            polymer = st[0][name].get_polymer(); full = st.get_entity_of(polymer).full_sequence
            assert t.gemmi.one_letter_code(full) == sequence, (record, name, 'sequence')
            assert all(res.name in full[res.label_seq - 1].split(',') for res in polymer), (record, name, 'label_seq')
        assert len(st.connections) == sum(b['partner_subchain'] is not None for b in t.as_list(row['covalent_bonds'])), (record, 'covalent links')
        verified_contacts = []
        for name in protein_chains:
            xyz = []; resids = []
            for res in st[0][name]:
                xyz.extend(a.pos.tolist() for a in res); resids.extend([str(res.seqid)] * len(res))
            tree = cKDTree(xyz); hits = tree.query_ball_point(ligand_xyz, 4.501)
            n_lig = sum(bool(h) for h in hits); n_res = len({resids[i] for h in hits for i in h})
            closest = float(tree.query(ligand_xyz)[0].min())
            assert n_lig >= 3 and n_res >= 2 and closest <= 4.001, (record, name, n_lig, n_res, closest)
            verified_contacts.append({'chain': name, 'ligand_atoms': n_lig, 'residues': n_res, 'min_distance': closest})
        raw = out / 'raw' / record; raw.mkdir(parents=True, exist_ok=True)
        receptor = st.clone(); del receptor[0]['L']
        while len(receptor.connections): del receptor.connections[0]
        receptor_cif = raw / f'{record}_protein.cif'
        receptor.make_mmcif_document().write_file(str(receptor_cif))
        sdf = out / 'ligands' / f'{record}.sdf'
        if sdf.exists():
            from rdkit import Chem
            mol = Chem.SDMolSupplier(str(sdf))[0]
            assert mol is not None and mol.GetNumHeavyAtoms() == len(ligand_xyz)
            assert cKDTree(ligand_xyz).query(mol.GetConformer().GetPositions())[0].max() < .002
            import shutil
            shutil.copy2(sdf, raw / f'{record}_ligand.sdf')
            row['ligand_sdf'] = str(sdf.resolve())
        elif truth(row['plp_training_eligible']):
            raise ValueError('Training-eligible ligand has no exported SDF')
        row.update(export_status='validated', reference_cif=str(reference.resolve()), protein_cif=str(receptor_cif.resolve()), validated_contacts=verified_contacts)
        pdb = raw / f'{record}_protein.pdb'
        row['protein_pdb'] = str(pdb.resolve()) if pdb.exists() else ''
        row.setdefault('sdf_status', 'not_exported_unverified_or_incomplete_chemistry')
        return row
    except Exception as exc:
        return {**row, 'export_status': 'error', 'export_error': str(exc)}


def export_selected(rows, out, workers, ccd_root):
    for directory in ('reference', 'ligands', 'raw'): (out / directory).mkdir(parents=True, exist_ok=True)
    results = []
    from multiprocessing import Pool
    with Pool(workers, initializer=worker_init, initargs=(ccd_root,)) as pool:
        for i, result in enumerate(pool.imap_unordered(export_one, ((r, str(out)) for r in rows), chunksize=1), 1):
            results.append(result)
            if i % 250 == 0: print(f'Exported and checked {i:,}/{len(rows):,}', flush=True)
    return results


def manifest(selection, audit):
    members = Counter(a['representative'] for a in audit.values())
    return [dict(r, conformer=audit[r['record']]['conformer'], pose_members=members[r['record']]) for r in selection]


def select_tpd(rows, cutoff, workers, poses):
    """Strict TPD rows, their nonredundant representatives and those released from 2024-07-01, each with its audit."""
    strict = [r for r in rows if r['tpd_tier'] == 'high_confidence']
    representatives, audit = select_conformers(rows, 'tpd', end=cutoff, workers=workers, poses=poses)
    recent, recent_audit = select_conformers(rows, 'tpd', start='2024-07-01', end=cutoff, workers=workers, poses=poses)
    return strict, representatives, audit, recent, recent_audit


def write_tpd(out, rows, strict, representatives, audit, recent, recent_audit):
    """Write the TPD manifests and return their summary counts."""
    t.write_tsv(out / 'tpd/entries.tsv', strict)
    t.write_tsv(out / 'tpd/nonredundant.tsv', manifest(representatives, audit))
    t.write_tsv(out / 'tpd/post_20240701.tsv', manifest(recent, recent_audit))
    t.write_tsv(out / 'tpd/duplicates.tsv', [{'record': record, 'complex_group': record.split('_')[1], **a} for record, a in sorted(audit.items())])
    t.write_tsv(out / 'tpd/review.tsv', [r for r in rows if truth(r['tpd_tagged']) and r['tpd_tier'] in ('review', 'glue_or_proximity_review')])
    t.write_tsv(out / 'tpd/rejected.tsv', [r for r in rows if truth(r['tpd_tagged']) and r['tpd_tier'] == 'rejected'])
    return {'tpd_entry_complexes': len(strict), 'tpd_nonredundant': len(representatives), 'tpd_post_20240701_nonredundant': len(recent),
            'tpd_nonredundant_class_counts': dict(Counter(r['class'] for r in representatives)),
            'tpd_sequence_ligand_groups': len({r['complex_group'] for r in representatives}),
            'tpd_manual_review_decisions': dict(Counter(t.as_list(r['manual_review'])['decision'] for r in rows if r.get('manual_review')))}


def tpd_review(args, out):
    """Reapply the manual TPD overrides to `all_complexes.tsv` without rerunning the full export.

    Only TPD selections change: records newly selected are exported and validated
    into `out`, the TPD manifests, the catalog and the TPD counts of
    `summary.json` are rewritten. PLP manifests are left as they are.
    """
    with open(out / 'all_complexes.tsv') as f: rows = list(csv.DictReader(f, delimiter='\t'))
    overrides = load_tpd_overrides(args.tpd_overrides)
    unknown = overrides.keys() - {r['record'] for r in rows}
    if unknown: raise ValueError(f'Overrides name records absent from the catalog: {sorted(unknown)}')
    rows = [apply_tpd_override(r, overrides) for r in rows]
    selection = select_tpd(rows, args.cutoff, args.workers, {})
    pending = {r['record']: r for r in selection[1] + selection[3] if r['export_status'] != 'validated'}
    print(f'Exporting {len(pending)} newly selected TPD records', flush=True)
    exported = {r['record']: r for r in export_selected(list(pending.values()), out, args.workers, args.ccd_root)}
    failed = [r for r in exported.values() if r['export_status'] == 'error']
    if failed: raise RuntimeError(f"Export failed: {[(r['record'], r['export_error']) for r in failed]}")
    rows = [exported.get(r['record'], r) for r in rows]
    by_record = {r['record']: r for r in rows}
    strict, representatives, audit, recent, recent_audit = selection
    counts = write_tpd(out, rows, [by_record[r['record']] for r in strict], [by_record[r['record']] for r in representatives], audit,
                       [by_record[r['record']] for r in recent], recent_audit)
    t.write_tsv(out / 'all_complexes.tsv', rows)
    summary = json.loads((out / 'summary.json').read_text()); summary.update(counts)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2), flush=True)


def export(args, out):
    request = json.loads((out / 'structure_request.json').read_text())
    if request != {'cutoff': args.cutoff, 'keep_homomers': args.keep_homomers, 'ccd_root': args.ccd_root}:
        raise ValueError('Export configuration differs from structural checkpoint.')
    expected = set()
    for line in (out / 'metadata.jsonl').open():
        m = json.loads(line)
        if m.get('release') and m['release'] <= args.cutoff and m.get('protein_entities') and m.get('plausible_ligands'):
            if args.keep_homomers or len(m['protein_entities']) >= 2: expected.add(m['pdb_id'])
    overrides = load_tpd_overrides(args.tpd_overrides)
    processed = set()
    rows = []; errors = []; outcomes = Counter(); rejected = Counter(); filtered_chains = Counter(); instances = 0
    for line in (out / 'structure_results.jsonl').open():
        result = json.loads(line)
        if result['pdb_id'] in processed: raise ValueError('Duplicate entry in structural checkpoint.')
        processed.add(result['pdb_id'])
        if 'error' in result: errors.append(result)
        outcomes[result.get('outcome', 'error')] += 1
        rejected.update(result.get('ligand_rejection_counts', {})); instances += result.get('pair_instances', 0)
        filtered_chains.update(result.get('filtered_chain_counts', {}))
        for row in result['records']:
            if not args.keep_homomers and row['homomeric']: continue
            if row['release'] > args.cutoff: raise ValueError('Structure cache exceeds requested cutoff; rerun/filter explicitly.')
            rows.append(apply_tpd_override(apply_additional_chemistry(row), overrides))
    if processed != expected:
        raise ValueError(f'Structural checkpoint incomplete or incompatible: {len(expected-processed)} missing; {len(processed-expected)} unexpected entries.')
    assert len(rows) == len({r['record'] for r in rows}), 'Repeated structural scan rows; fix checkpoint before exporting.'
    unknown = overrides.keys() - {r['record'] for r in rows}
    if unknown: raise ValueError(f'Overrides name records absent from the catalog: {sorted(unknown)}')
    training = [r for r in rows if r['plp_training_eligible']]
    poses = {}
    representatives, plp_audit = select_conformers(rows, 'plp', end=args.cutoff, workers=args.workers, poses=poses)
    tpd_selection = select_tpd(rows, args.cutoff, args.workers, poses)
    strict, tpd_representatives, tpd_audit, recent_tpd, recent_audit = tpd_selection
    selected = {r['record']: r for r in representatives + tpd_representatives + recent_tpd}
    print(f'Found {len(rows):,} entry/sequence-pair/ligand combinations; {len(representatives):,} nonredundant PLP; {len(tpd_representatives):,} nonredundant TPD', flush=True)
    exported = export_selected(list(selected.values()), out, args.workers, args.ccd_root)
    by_record = {r['record']: r for r in exported}
    for row in rows:
        row.update(by_record.get(row['record'], {'export_status': 'not_selected_for_export'}))
    for directory in ('plp', 'tpd'): (out / directory).mkdir(exist_ok=True)
    t.write_tsv(out / 'all_complexes.tsv', rows)
    t.write_tsv(out / 'plp/entries.tsv', training)
    t.write_tsv(out / 'plp/nonredundant.tsv', manifest(representatives, plp_audit))
    t.write_tsv(out / 'plp/review.tsv', [r for r in rows if not r['plp_training_eligible']])
    tpd_counts = write_tpd(out, rows, *tpd_selection)
    t.write_tsv(out / 'structural_errors.tsv', errors)
    t.write_tsv(out / 'export_errors.tsv', [r for r in exported if r['export_status'] == 'error'])
    chemical_audit = {}
    for row in rows:
        if not row['additional_chemistry_flags']: continue
        audit = chemical_audit.setdefault(row['ccd'], {'ccd': row['ccd'], 'name': row['ligand_name'], 'flags': row['additional_chemistry_flags'], 'entry_complexes': 0})
        audit['entry_complexes'] += 1
    t.write_tsv(out / 'chemistry_review.tsv', sorted(chemical_audit.values(), key=lambda r: (-r['entry_complexes'], r['ccd'])))
    earliest = {}
    for row in training:
        group = earliest.setdefault(row['complex_group'], {'release': row['release'], 'deposition_date': row['deposition_date']})
        for key in group:
            value = row[key]
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value): group[key] = min(group[key] or value, value)
    t.write_tsv(out / 'duplicates.tsv', [
        {'record': r['record'], 'complex_group': r['complex_group'], **plp_audit[r['record']],
         'is_representative': r['record'] == plp_audit[r['record']]['representative'], 'within_entry_instances': r['instance_count'],
         'deposition_date': r['deposition_date'], 'release': r['release'], 'group_first_release': earliest[r['complex_group']]['release'],
         'group_first_deposition': earliest[r['complex_group']]['deposition_date']}
        for r in training])
    summary = {
        'cutoff': args.cutoff, 'hard_cutoff_field': 'initial_release_date', 'workers': args.workers, 'keep_homomers_in_plp': args.keep_homomers,
        'structural_entry_outcomes': dict(outcomes), 'ligand_rejection_counts': dict(rejected), 'contact_pair_instances': instances,
        'per_entry_unique_complexes': len(rows), 'plp_training_entry_complexes': len(training), 'plp_nonredundant': len(representatives),
        'plp_nonredundant_homomeric': sum(r['homomeric'] for r in representatives),
        'plp_nonredundant_heteromeric': sum(not r['homomeric'] for r in representatives),
        **tpd_counts,
        'plp_sequence_ligand_groups': len({r['complex_group'] for r in representatives}),
        'conformer_rmsd_angstrom': CONFORMER_RMSD,
        'plp_nonredundant_covalent': sum(r['covalent'] for r in representatives),
        'plp_nonredundant_over_two_protein_chains': sum(r['protein_chains'] > 2 for r in representatives),
        'plp_nonredundant_with_context_ligands': sum(bool(r['context_ligands']) for r in representatives),
        'chains_removed_near_ligands': dict(filtered_chains),
        'exported_complexes': len(exported), 'validated_exports': sum(r['export_status'] == 'validated' for r in exported),
        'structural_errors': len(errors), 'export_errors': sum(r['export_status'] == 'error' for r in exported),
        'missing_deposition_dates_in_complexes': sum(not re.fullmatch(r'\d{4}-\d{2}-\d{2}', r['deposition_date']) for r in rows),
        'additional_chemistry_review_entry_complexes': sum(bool(r['additional_chemistry_flags']) for r in rows),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    config = {'arguments': vars(args), 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'shared_screen_sha256': hashlib.sha256(Path(t.__file__).read_bytes()).hexdigest(),
              'contacts': {'cutoff_angstrom': 4.5, 'minimum_ligand_atoms_per_chain': 3, 'minimum_residues_per_chain': 2, 'maximum_nearest_distance_angstrom': 4.0},
              'chain_filters': {'minimum_resolved_residues': 4, 'all_unknown_removed': True, 'clash_fraction': 0.3, 'clash_distance_angstrom': 1.7, 'maximum_consecutive_CA_angstrom': 10},
              'covalent': {'source': '_struct_conn covale, else distance to a protein nucleophile', 'clash_below_angstrom': 1.8,
                           'inferred_bond_windows_angstrom': {'/'.join(k): v for k, v in NUCLEOPHILES.items()}},
              'maximum_resolution_angstrom': MAX_RESOLUTION,
              'crystallization_aids_removed_from_crystallographic_context': sorted(CRYSTALLIZATION_AIDS)}
    (out / 'run_configuration.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
