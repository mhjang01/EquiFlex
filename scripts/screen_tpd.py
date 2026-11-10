#!/usr/bin/env python3
"""Auditable local PDB TPD screen; initial release dates, conservative confidence tiers."""
import argparse, csv, gzip, hashlib, itertools, json, re, shutil
from datetime import date
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gemmi
import numpy as np
from scipy.spatial import cKDTree

TERMS = {
 'protac': r'\bprotacs?\b|proteolysis[- ]targeting',
 'glue': r'\bglue[s]?\b',
 'degrader': r'\bdegraders?\b|targeted (?:protein )?degradation|induced (?:protein )?degradation',
 'neosubstrate': r'neo[- ]?substrate',
 'imid': r'thalidomide|lenalidomide|pomalidomide|iberdomide|mezigdomide|celmod',
 'proximity': r'induced proximity|chemically induced dimer|\b(?:lytac|autac|attec|riptac|dubtac|traftac)\b',
}
LIGASE = r'denticleless|DDB1.and.CUL4.associated factor|DDB1.CUL4.associated factor|cereblon|\bcrbn\b|von hippel|\bvhl\b|\bddb1\b|DNA damage.binding protein 1|\bdcaf\d*\b|\bkeap1\b|kelch.like ECH.associated|\bmdm2\b|E3 ubiquitin|ubiquitin.protein ligase|\bbirc\d\b|\bxiap\b|\bciap\b|\bspop\b|\bklhdc\d\b|\bfbxw\d\b|\btrim\d+\b|\brnf\d+\b|\bkbtbd\d+\b|kelch repeat and BTB|F.box|\bbtrc\b|beta.?trcp|\bfbxo\d+\b|glucose.induced degradation protein|\bgid4\b|yippee.like|\bypel5\b'
EXCLUDED = set('HOH DOD GOL EDO PEG PGE PG4 1PE 2PE P6G MPD DMS MES TRS EPE NAG MAN BMA GLC ATP ADP AMP ADN ANP AGS ACP APC GTP GDP GMP GNP GSP GCP UTP UDP UMP UPG CTP CDP CMP TTP TDP TMP NAD NAI NAP NDP NAH FAD FDA FMN SAM SAH COA ACO TPP PLP PMP BTN B12 HEM HEC HEA HEO HEB HEV H4B THF GSH GSS F42 MGD'.split())
BAD_NAME = r'flavin|nicotinamide.*dinucleotide|coenzyme.?a\b|pyridoxal|thiamine.*phosphate|(?:adenosine|guanosine|uridine|cytidine).*(?:triphosphate|diphosphate|monophosphate)|cholest|ergoster|sitoster|stigmaster|chlorophyll|bacteriochloro|pheophyt|caroten|xanthophyll|lutein|lycopene|retinal|biliverdin|porphyrin|\bheme\b|\bhaem\b|phytol|ubiquinone|menaquinone|plastoquinone|phosphatidyl|phosphocholine|cardiolipin|diacylglycerol|monoolein|fatty acid|palmitoyl|oleoyl|stearoyl|myristoyl|maltoside|detergent|cholesterol|quercetin|kaempferol|resveratrol|curcumin|genistein|catechin|auxin|indole.3.acetic|jasmon|gibberell|abscisic|fusicoccin'
FIELDS = {'title':['_struct.title'], 'keywords':['_struct_keywords.pdbx_keywords','_struct_keywords.text'], 'descriptor':['_struct.pdbx_descriptor'], 'citation':['_citation.title'], 'entity':['_entity.pdbx_description'], 'synonym':['_entity_name_com.name']}

def vals(b,tag): return [gemmi.cif.as_string(v) for v in b.find_values(tag)]
def rows(b,cat,tags): return [[gemmi.cif.as_string(v) for v in r] for r in b.find(cat,tags)]
def header(path,start=None,end=None):
    # Revision history can FOLLOW atom_site in older archive files. Strip only
    # coordinate categories, retaining metadata on both sides of the atom loop.
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:
        text=f.read(65536)
        pos=text.find('_pdbx_audit_revision_history.')
        if start and pos>=0:
            begin=text.rfind('\n#',0,pos)
            finish=text.find('\n#',pos)
            if begin>=0 and finish>=0:
                date_block=gemmi.cif.read_string('data_dates\n'+text[begin:finish]).sole_block()
                dates=vals(date_block,'_pdbx_audit_revision_history.revision_date')
                if dates and not start<=min(dates)<=end:return date_block
        text+=f.read()
    for category in ('_atom_site.', '_atom_site_anisotrop.'):
        pos=text.find('\n'+category)
        if pos>=0:
            begin=text.rfind('\nloop_',0,pos)
            end=text.find('\n#',pos)
            if begin>=0 and end>=0:text=text[:begin]+text[end:]
    return gemmi.cif.read_string(text).sole_block()

def scan(job):
    path,start,end=job
    try:
        b=header(path,start,end)
        dates=vals(b,'_pdbx_audit_revision_history.revision_date') or vals(b,'_database_PDB_rev.date')
        dates=[d for d in dates if re.fullmatch(r'\d{4}-\d{2}-\d{2}',d)]
        if not dates: return {'path':str(path),'error':'missing_release_date'}
        release=min(dates)
        base={'pdb_id':Path(path).name.split('.')[0].upper(),'release':release,'path':str(path)}
        if not start<=release<=end: return {**base,'scope':'outside_window'}
        fields={k:' '.join(x for tag in tags for x in vals(b,tag)) for k,tags in FIELDS.items()}
        fields['ligand_name']=' '.join(vals(b,'_pdbx_entity_nonpoly.name'))
        terms={k:[f for f,t in fields.items() if re.search(rx,t,re.I)] for k,rx in TERMS.items()}
        terms={k:v for k,v in terms.items() if v}
        machinery=bool(re.search(LIGASE,fields['entity']+' '+fields['synonym'],re.I))
        return {**base,'scope':'in_window','fields':fields,'terms':terms,'machinery':machinery,'tagged':bool(terms or machinery)}
    except Exception as e: return {'path':str(path),'error':str(e)}

def write_tsv(path,data):
    data=list(data)
    keys=list(dict.fromkeys(k for d in data for k in d))
    if not keys:
        keys=['path','error'] if Path(path).name=='scan_errors.tsv' else ['pdb_id','tier','reasons']
    with open(path,'w') as f:
        w=csv.DictWriter(f,keys,delimiter='\t');w.writeheader()
        for d in data: w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in d.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ccd-root',default='/home/mhjang/workspace/boltz-tpd/boltz_cache/mols');p.add_argument('--pdb',default='/mnt/drive/PDB');p.add_argument('--out',default='tpd/screened')
    p.add_argument('--start',default='2024-07-01');p.add_argument('--end',default='2026-06-30');p.add_argument('--workers',type=int,default=12)
    p.add_argument('--resume-scan',action='store_true');p.add_argument('--reuse-scan',action='store_true');p.add_argument('--scan-only',action='store_true')
    a=p.parse_args()
    if date.fromisoformat(a.start)>date.fromisoformat(a.end):p.error('--start must not follow --end')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    request={'start':a.start,'end':a.end,'pdb_root':str(Path(a.pdb).resolve())}
    request_path=out/'scan_request.json'
    if a.resume_scan and request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError('Cannot resume a scan with different dates or mirror root.')
    if not a.reuse_scan:request_path.write_text(json.dumps(request,indent=2))
    if not a.reuse_scan:
        paths=sorted(Path(a.pdb).glob('*/*.cif.gz'))+sorted(Path(a.pdb).glob('*/*.cif'))
        print(f'Scanning {len(paths)} local entries',flush=True)
        counts=Counter(); maxdate=''; tagged=[];errors=[];window=[]
        previous=[]
        if a.resume_scan and (out/'scan_audit.jsonl').exists():
            for line in (out/'scan_audit.jsonl').read_text().splitlines():
                try: previous.append(json.loads(line))
                except json.JSONDecodeError: pass
            done={m['path'] for m in previous};paths=[p for p in paths if str(p) not in done]
            print('Resuming',len(previous),'finished;',len(paths),'remaining',flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool, (out/'scan_audit.jsonl').open('w') as f:
            for i,m in enumerate(itertools.chain(previous,pool.map(scan,((str(p),a.start,a.end) for p in paths),chunksize=32)),1):
                counts[m.get('scope','error')]+=1;maxdate=max(maxdate,m.get('release',''))
                if m.get('tagged'):tagged.append(m)
                if 'error' in m:errors.append(m)
                if m.get('scope')=='in_window':window.append({k:m[k] for k in ['pdb_id','release','path','terms','machinery','tagged']})
                f.write(json.dumps(m)+'\n')
                if i%10000==0:print(i,dict(counts),'tagged',len(tagged),flush=True)
        (out/'tagged.json').write_text(json.dumps(tagged,indent=2))
        (out/'scan_summary.json').write_text(json.dumps({'total':len(paths)+len(previous),'counts':dict(counts),'latest_local_release':maxdate,'start':a.start,'end':a.end,'pdb_root':str(Path(a.pdb).resolve())},indent=2))
        write_tsv(out/'scan_errors.tsv',errors);write_tsv(out/'window_entries.tsv',window)
    if not a.scan_only:
        summary=json.loads((out/'scan_summary.json').read_text())
        if (summary['start'],summary['end'],summary['pdb_root']) != (a.start,a.end,str(Path(a.pdb).resolve())):
            raise ValueError('Cached scan dates/root differ; rerun without --reuse-scan.')
        analyze(a,out)


CCD_ROOT=Path('/home/mhjang/workspace/boltz-tpd/boltz_cache/mols')

def chemistry(ccd,name,formula):
    import pickle
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    reasons=[]; props={};mol=None
    if ccd in EXCLUDED:reasons.append('cofactor_buffer_or_sugar')
    if re.search(BAD_NAME,name,re.I):reasons.append('natural_product_lipid_or_cofactor_name')
    p=CCD_ROOT/f'{ccd}.pkl'
    if not p.exists():
        p=Path(__file__).resolve().parents[1]/'data'/'pdb_ccd_updates'/f'{ccd}.pkl'
    if p.exists():
        try:
            # RemoveHs retains isotopic and some special hydrogens by default.
            # Both contact geometry and exported ligand poses use heavy atoms.
            mol=Chem.RemoveAllHs(pickle.loads(p.read_bytes()));Chem.SanitizeMol(mol)
            props={'mw':round(Descriptors.MolWt(mol),2),'heavy_atoms':mol.GetNumHeavyAtoms(),'rings':rdMolDescriptors.CalcNumRings(mol),'smiles':Chem.MolToSmiles(mol),'chemistry_source':'local_CCD'}
            if not 150<=props['mw']<=2500:reasons.append('molecular_weight_outside_150_2500')
            if props['heavy_atoms']<12:reasons.append('too_small')
            if not props['rings']:reasons.append('acyclic')
            if any(a.GetSymbol() not in {'C','H','N','O','S','P','F','Cl','Br','I','B','Si','Se'} for a in mol.GetAtoms()):reasons.append('metal_containing')
            if len(Chem.GetMolFrags(mol))!=1:reasons.append('disconnected_ligand')
            if sum(a.GetSymbol()=='P' for a in mol.GetAtoms())>=3:reasons.append('polyphosphate')
        except Exception as e:props['chemistry_error']=str(e);mol=None
    if mol is None:
        counts={el:int(n or 1) for el,n in re.findall(r'([A-Z][a-z]?)(\d*)',formula)}
        props.update(chemistry_source='formula_only',heavy_atoms=sum(n for el,n in counts.items() if el!='H'))
        if counts.get('C',0)<6 or props['heavy_atoms']<12:reasons.append('too_small_or_inorganic')
        if set(counts)-{'C','H','N','O','S','P','F','Cl','Br','I','B','Si','Se'}:reasons.append('metal_containing')
    return props,reasons,mol


def expected_atoms(block, ccd, mol):
    """Heavy-atom names and elements of a component, their source, and its leaving atoms."""
    expected = {a.GetProp('name'): a.GetSymbol() for a in mol.GetAtoms()
                if a.GetAtomicNum() > 1 and a.HasProp('name')} if mol is not None else {}
    leaving = {a.GetProp('name') for a in mol.GetAtoms()
               if a.GetAtomicNum() > 1 and a.HasProp('name') and a.HasProp('leaving_atom') and a.GetBoolProp('leaving_atom')} if mol is not None else set()
    source = 'local_CCD' if expected else 'unavailable'
    if not expected:
        expected = {name: element for comp, name, element in rows(
            block, '_chem_comp_atom.', ['comp_id', 'atom_id', 'type_symbol'])
                    if comp == ccd and element.upper() not in ('H', 'D', 'T')}
        if expected: source = 'embedded_chem_comp_atom'
    return expected, source, leaving


def ligand_completeness(block, ccd, mol, atoms, covalent=False):
    """Compare named heavy atoms; an unavailable reference is unknown, not zero.

    Leaving atoms absent from a covalently bound ligand are not required.
    """
    expected, source, leaving = expected_atoms(block, ccd, mol)
    observed = {a.name: a.element.name for a in atoms if a.occ > 0 and not a.is_hydrogen()}
    absent_leaving = sorted(leaving - set(observed)) if covalent else []
    expected = {k: v for k, v in expected.items() if k not in absent_leaving}
    missing = sorted(set(expected) - set(observed)) if expected else None
    unmatched = sorted(set(observed) - set(expected)) if expected else None
    mismatches = sorted(name for name in expected.keys() & observed.keys()
                        if expected[name].upper() != observed[name].upper())
    return {'ligand_completeness': len(expected.keys() & observed.keys()) / len(expected) if expected else None,
            'completeness_source': source, 'expected_heavy_atoms': len(expected) if expected else None,
            'missing_ccd_atoms': missing, 'unmatched_observed_atoms': unmatched,
            'atom_element_mismatches': mismatches, 'missing_leaving_atoms': absent_leaving}


def homology(e1,e2,seqs,uniprots):
    if e1==e2:return True,1.0,'same_entity'
    if uniprots.get(e1,set()) & uniprots.get(e2,set()):return True,1.0,'shared_UniProt'
    s,t=seqs.get(e1,''),seqs.get(e2,'')
    if not s or not t:return False,None,'missing_sequence'
    from Bio.Align import PairwiseAligner
    a=PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,open_gap_score=-5,extend_gap_score=-0.5)
    alignments=a.align(s,t)
    if alignments.score<=0:return False,0.0,'local_alignment'
    aln=alignments[0]; counts=aln.counts();aligned=counts.identities+counts.mismatches+counts.gaps
    identity=counts.identities/max(1,aligned)
    coverage=sum(y-x for x,y in aln.aligned[0])/min(len(s),len(t))
    return identity>=.9 and coverage>=.8,round(identity,3),'local_alignment'


def evaluate(m):
    try:
        b=gemmi.cif.read(m['path']).sole_block();s=gemmi.make_structure_from_block(b)
        s.remove_hydrogens();s.remove_alternative_conformations()
        desc=dict(rows(b,'_entity.',['id','pdbx_description']))
        role_names=dict(desc)
        for eid,name in rows(b,'_entity_name_com.',['entity_id','name'])+rows(b,'_entity_src_gen.',['entity_id','pdbx_gene_src_gene']):
            role_names[eid]=role_names.get(eid,'')+' '+name
        seqs={e.name:gemmi.one_letter_code(e.full_sequence) for e in s.entities if e.polymer_type in (gemmi.PolymerType.PeptideL,gemmi.PolymerType.PeptideD)}
        uniprots={}
        for eid,db,acc in rows(b,'_struct_ref.',['entity_id','db_name','pdbx_db_accession']):
            if db=='UNP':uniprots.setdefault(eid,set()).add(acc)
        chem={r[0]:(r[1],r[2]) for r in rows(b,'_chem_comp.',['id','name','formula'])}
        chemcache={};homcache={};result=[]
        models=[('asu',s[0])]+[(f'assembly_{a.name}',gemmi.make_assembly(a,s[0],gemmi.HowToNameCopiedChain.AddNumber)) for a in s.assemblies]
        for space,model in models:
            proteins={};ligands=[]
            for chain in model:
                for res in chain:
                    atoms=[a for a in res if a.occ>0 and not a.is_hydrogen()]
                    if not atoms:continue
                    if res.entity_id in seqs:
                        key=(chain.name,res.subchain,res.entity_id)
                        d=proteins.setdefault(key,{'xyz':[],'res':[]})
                        d['xyz'] += [a.pos.tolist() for a in atoms]; d['res'] += [str(res.seqid)]*len(atoms)
                    elif res.entity_type==gemmi.EntityType.NonPolymer and len(atoms)>=8:
                        ligands.append((chain.name,res,atoms))
            for d in proteins.values():d['tree']=cKDTree(d['xyz'])
            for lc,res,atoms in ligands:
                ccd=res.name
                if ccd not in chemcache:chemcache[ccd]=chemistry(ccd,*chem.get(ccd,('','')))
                props,chem_reasons,mol=chemcache[ccd]
                observed_names={a.name for a in atoms}
                ccd_names={a.GetProp('name') for a in mol.GetAtoms() if a.HasProp('name')} if mol is not None else set()
                missing_ccd_atoms=sorted(ccd_names-observed_names)
                unmatched_observed_atoms=sorted(observed_names-ccd_names) if ccd_names else []
                xyz=np.array([a.pos.tolist() for a in atoms]);contacts=[]
                for key,d in proteins.items():
                    hits=d['tree'].query_ball_point(xyz,4.5)
                    ligand_atoms=sum(bool(h) for h in hits);inds={i for h in hits for i in h}
                    if not inds:continue
                    contacts.append({'chain':key[0],'subchain':key[1],'entity':key[2],'name':desc.get(key[2],''),'ligand_atoms':ligand_atoms,'protein_atoms':len(inds),'residues':len({d['res'][i] for i in inds}),'min_distance':round(float(d['tree'].query(xyz)[0].min()),3)})
                robust=[c for c in contacts if c['ligand_atoms']>=3 and c['residues']>=2 and c['min_distance']<=4.0]
                common={'pdb_id':m['pdb_id'],'release':m['release'],'space':space,'ccd':ccd,'ligand_chain':lc,'ligand_subchain':res.subchain,'ligand_residue':str(res.seqid),'ligand_name':chem.get(ccd,('',''))[0],'title':m['fields']['title'],'citation':m['fields']['citation'],'citation_doi':';'.join(vals(b,'_citation.pdbx_database_id_DOI')),'source_path':m['path'],'terms':m['terms'],'contacts':contacts,'observed_heavy_atoms':len(atoms),'missing_ccd_atoms':missing_ccd_atoms,'unmatched_observed_atoms':unmatched_observed_atoms,'resolution':s.resolution,**props}
                if chem_reasons or len(robust)<2:
                    result.append({**common,'tier':'rejected','reasons':chem_reasons+(['fewer_than_two_robust_protein_contacts'] if len(robust)<2 else [])});continue
                for c1,c2 in itertools.combinations(robust,2):
                    e1,e2=c1['entity'],c2['entity'];key=tuple(sorted((e1,e2)))
                    if key not in homcache:homcache[key]=homology(e1,e2,seqs,uniprots)
                    homo,identity,hommethod=homcache[key]
                    reason=[];tier='review'
                    if homo:reason.append('homomeric_or_near_identical_proteins');tier='rejected'
                    names=[c1['name'],c2['name']]
                    recruiters=[bool(re.search(LIGASE,role_names.get(e,''),re.I)) for e in (e1,e2)]
                    scaffold=(any(re.search(r'elongin|transcription elongation factor B|\bTCEB[123]\b|\bELOC\b|\bELOB\b|cullin|RING.box|\bRBX[12]\b',n,re.I) for n in names) and any(recruiters)) or (any(re.search(r'DNA damage.binding protein 1|\bDDB1\b',n,re.I) for n in names) and all(recruiters))
                    if scaffold:reason.append('ligase_scaffold_pair');tier='rejected'
                    if any(re.search(r'nanobody|antibody|Fab fragment|immunoglobulin.*(?:chain|fragment)|crystallization chaperone|maltose.binding protein|streptavidin|affibody|DARPin',n,re.I) for n in names):
                        reason.append('possible_antibody_or_crystallization_partner')
                    terms=set(m['terms']);core=terms&{'protac','glue','degrader'}
                    direct=any(set(m['terms'].get(t,[]))&{'title','keywords','descriptor'} for t in core)
                    if not core:reason.append('no_explicit_TPD_or_glue_annotation')
                    if not any(recruiters):reason.append('no_recognized_recruiter_in_contact_pair')
                    completeness=len(observed_names & ccd_names)/len(ccd_names) if ccd_names else len(atoms)/max(props.get('heavy_atoms',len(atoms)),1)
                    if completeness<.9:reason.append('ligand_less_than_90_percent_complete')
                    if props.get('chemistry_source')!='local_CCD' or not ccd_names or unmatched_observed_atoms:reason.append('chemistry_unverified')
                    if hommethod=='missing_sequence':reason.append('heteromer_unverified')
                    if s.resolution<=0 or s.resolution>4.0:reason.append('resolution_missing_or_over_4A')
                    if min(a.occ for a in atoms)<.5:reason.append('ligand_occupancy_under_half')
                    if min(c1['min_distance'],c2['min_distance'])<1.8:reason.append('covalent_or_clashing_contact')
                    if not direct:reason.append('no_structure_specific_TPD_annotation')
                    if tier!='rejected' and not reason:tier='high_confidence'
                    if tier=='review' and core and not any(recruiters):tier='glue_or_proximity_review'
                    class_terms={t for t in ('protac','glue') if set(m['terms'].get(t,[])) & {'title','keywords','descriptor'}}
                    class_source='structure_annotation' if class_terms else 'entry_annotation'
                    if not class_terms:class_terms=terms & {'protac','glue'}
                    cls='protac' if class_terms=={'protac'} else 'molecular_glue' if class_terms=={'glue'} else 'unresolved'
                    fingerprint=hashlib.sha256(('\n'.join(sorted([seqs.get(e1,''),seqs.get(e2,'')]))+'\n'+props.get('smiles',ccd)).encode()).hexdigest()[:20]
                    result.append({**common,'chain_A':c1['chain'],'chain_B':c2['chain'],'subchain_A':c1['subchain'],'subchain_B':c2['subchain'],'entity_A':e1,'entity_B':e2,'sequence_A':seqs.get(e1,''),'sequence_B':seqs.get(e2,''),'uniprot_A':sorted(uniprots.get(e1,set())),'uniprot_B':sorted(uniprots.get(e2,set())),'protein_A':names[0],'protein_B':names[1],'recruiter_A':recruiters[0],'recruiter_B':recruiters[1],'protein_A_aliases':role_names.get(e1,''),'protein_B_aliases':role_names.get(e2,''),'pair_contacts':[c1,c2],'sequence_identity':identity,'homology_check':hommethod,'tier':tier,'reasons':reason,'class':cls,'class_source':class_source,'ligand_completeness':round(completeness,3),'complex_group':fingerprint})
        return result
    except Exception as e:return [{'pdb_id':m['pdb_id'],'tier':'error','reasons':[str(e)]}]


def clean_structure_for_assembly(s):
    """Remove alternate conformers without leaving dangling assembly subchains."""
    before = {r.subchain for c in s[0] for r in c}
    s.remove_hydrogens(); s.remove_alternative_conformations()
    removed = before - {r.subchain for c in s[0] for r in c}
    if removed:
        for assembly in s.assemblies:
            for generator in assembly.generators:
                generator.subchains = [name for name in generator.subchains if name not in removed]


def as_list(value):
    """Manifest list fields arrive as JSON text when rows are read back from TSV."""
    return json.loads(value) if isinstance(value,str) and value else (value or [])


EXTRA_CHAIN_NAMES='CDEFGHJKMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz0123456789'

def export_complex(r,m,out):
    """Write the ligand-centred complex: every contacting chain, co-binders, ions and covalent links.

    Chains A and B are the primary pair, further letters the other polymer chains
    touching the ligand, L the ligand, X other non-polymers within contact
    distance and I monatomic ions. Polymer entities keep the deposited sequence.
    """
    from rdkit import Chem
    b=gemmi.cif.read(m['path']).sole_block();s=gemmi.make_structure_from_block(b);clean_structure_for_assembly(s)
    if r['space']=='asu':model=s[0]
    else:model=gemmi.make_assembly(next(a for a in s.assemblies if 'assembly_'+a.name==r['space']),s[0],gemmi.HowToNameCopiedChain.AddNumber)
    _,_,mol=chemistry(r['ccd'],r['ligand_name'],'')
    ccd_names=set(expected_atoms(b,r['ccd'],mol)[0])
    contacts=as_list(r.get('contacts'));bonds=as_list(r.get('covalent_bonds'))
    polymers=[r['subchain_A'],r['subchain_B']]+[c['subchain'] for c in contacts[2:]]+as_list(r.get('context_polymers'))
    hets={(h['subchain'],h['name'],h['residue']) for h in as_list(r.get('context_ligands'))}
    if len(polymers)>2+len(EXTRA_CHAIN_NAMES):raise ValueError('Too many polymer chains contact the ligand to name them.')
    chain_names=dict(zip(polymers,['A','B',*EXTRA_CHAIN_NAMES]))
    def observed(res,names=None):
        res=res.clone()
        for i in reversed(range(len(res))):
            if res[i].occ<=0 or (names and res[i].name not in names):del res[i]
        return res
    chains={name:gemmi.Chain(name) for name in [*chain_names.values(),'L','X','I']}
    entity_subchains={};placed={};ion_names=[]
    for chain in model:
        for res in chain:
            key=(res.subchain,res.name,str(res.seqid))
            if res.subchain in chain_names and res.entity_type==gemmi.EntityType.Polymer:
                kept=observed(res)
                if len(kept):
                    chains[chain_names[res.subchain]].add_residue(kept)
                    subs=entity_subchains.setdefault(res.entity_id,[])
                    if res.subchain not in subs:subs.append(res.subchain)
            elif res.subchain==r['ligand_subchain'] and res.name==r['ccd'] and str(res.seqid)==r['ligand_residue']:
                chains['L'].add_residue(observed(res,ccd_names))
            elif key in hets:
                kept=observed(res);target='I' if len(kept)==1 else 'X'
                kept.seqid=gemmi.SeqId(len(chains[target])+1,' ');chains[target].add_residue(kept)
                placed[key]=(target,kept.seqid);ion_names += [res.name]*(target=='I')
    # Keep deposited structural metal ions coordinating the exported polymers.
    tree=cKDTree([a.pos.tolist() for name in chain_names.values() for res in chains[name] for a in res])
    for chain in model:
        for res in chain:
            if (res.subchain,res.name,str(res.seqid)) not in placed and len(res)==1 and res[0].occ>0 and res[0].element.name in {'Zn','Mg','Fe','Mn','Ca','Cu','Ni','Co'} and tree.query(res[0].pos.tolist())[0]<=3.0:
                ion=res.clone();ion.seqid=gemmi.SeqId(len(chains['I'])+1,' ');chains['I'].add_residue(ion);ion_names.append(res.name)
    st=gemmi.Structure();st.cell=s.cell;st.add_model(gemmi.Model('1'))
    for name,chain in chains.items():
        if len(chain) or name in ('A','B','L'):st[0].add_chain(chain)
    # Deposited sequences travel with the chains so unresolved residues stay in the entity.
    for entity_id,subs in entity_subchains.items():
        source=s.get_entity(entity_id);entity=gemmi.Entity(entity_id)
        entity.entity_type=source.entity_type;entity.polymer_type=source.polymer_type
        entity.full_sequence=source.full_sequence;entity.subchains=subs;st.entities.append(entity)
    ligand=st[0]['L'][0]
    for bond in bonds:
        if bond['partner_kind']=='nonpolymer':
            target=placed.get((bond['partner_subchain'],bond['partner_name'],bond['partner_residue']))
        else:
            target=(chain_names[bond['partner_subchain']],gemmi.SeqId(bond['partner_residue'])) if bond['partner_subchain'] in chain_names else None
        if target is None:continue
        link=gemmi.Connection();link.name=f'covale{len(st.connections)+1}';link.type=gemmi.ConnectionType.Covale;link.asu=gemmi.Asu.Same
        link.partner1=gemmi.AtomAddress('L',ligand.seqid,ligand.name,bond['ligand_atom'])
        link.partner2=gemmi.AtomAddress(target[0],target[1],bond['partner_name'],bond['partner_atom'])
        st.connections.append(link)
    r['structural_ions']=ion_names;r['exported_chains']={name:sub for sub,name in chain_names.items()}
    st.setup_entities();st.make_mmcif_document().write_file(str(out/'reference'/f"{r['record']}.cif"))
    if mol is not None:
        atoms={a.name:a for a in ligand}
        absent=[a.GetIdx() for a in mol.GetAtoms() if not a.HasProp('name') or a.GetProp('name') not in atoms]
        r['sdf_status']='complete_observed_pose'
        if absent and bonds and all(mol.GetAtomWithIdx(i).HasProp('leaving_atom') and mol.GetAtomWithIdx(i).GetBoolProp('leaving_atom') for i in absent):
            mol=Chem.RWMol(mol)
            for i in sorted(absent,reverse=True):mol.RemoveAtom(i)
            mol=mol.GetMol();Chem.SanitizeMol(mol);absent=[];r['sdf_status']='covalent_leaving_atoms_removed'
        if not absent:
            conf=Chem.Conformer(mol.GetNumAtoms())
            for a in mol.GetAtoms():conf.SetAtomPosition(a.GetIdx(),atoms[a.GetProp('name')].pos.tolist())
            mol.RemoveAllConformers();mol.AddConformer(conf);mol.SetProp('_Name',r['record'])
            with Chem.SDWriter(str(out/'ligands'/f"{r['record']}.sdf")) as writer:writer.write(mol)
            raw=out/'raw'/r['record'];raw.mkdir(parents=True,exist_ok=True)
            shutil.copy2(out/'ligands'/f"{r['record']}.sdf",raw/f"{r['record']}_ligand.sdf")
            receptor=st.clone();del receptor[0]['L']
            receptor.write_pdb(str(raw/f"{r['record']}_protein.pdb"))
        else:r['sdf_status']='not_exported_missing_CCD_atoms'


def init_chemistry(root):
    global CCD_ROOT
    CCD_ROOT=Path(root)


def refresh_annotations(out):
    """Reapply current annotation rules to cached fields without rereading the mirror."""
    tagged=[];window=[]
    path=out/'scan_audit.jsonl';tmp=out/'scan_audit.refresh.jsonl'
    with path.open() as src,tmp.open('w') as dst:
        for line in src:
            m=json.loads(line)
            if m.get('scope')=='in_window':
                fields=m['fields']
                terms={k:[f for f,text in fields.items() if re.search(rx,text,re.I)] for k,rx in TERMS.items()}
                m['terms']={k:v for k,v in terms.items() if v}
                m['machinery']=bool(re.search(LIGASE,fields['entity']+' '+fields['synonym'],re.I))
                m['tagged']=bool(m['terms'] or m['machinery'])
                if m['tagged']:tagged.append(m)
                window.append({k:m[k] for k in ['pdb_id','release','path','terms','machinery','tagged']})
            dst.write(json.dumps(m)+'\n')
    tmp.replace(path)
    (out/'tagged.json').write_text(json.dumps(tagged,indent=2))
    write_tsv(out/'window_entries.tsv',window)
    return tagged


def analyze(a,out):
    init_chemistry(a.ccd_root)
    (out/'run_configuration.json').write_text(json.dumps({'arguments':vars(a),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gemmi_version':gemmi.__version__,'thresholds':{'contact_A':4.5,'minimum_ligand_atoms_per_partner':3,'minimum_residues_per_partner':2,'minimum_closest_contact_A':4.0,'sequence_identity':0.9,'shorter_sequence_coverage':0.8,'ligand_completeness':0.9,'maximum_resolution_A':4.0,'minimum_occupancy':0.5,'minimum_nonclashing_contact_A':1.8,'mw_range':[150,2500]}},indent=2))
    tagged=refresh_annotations(out);lookup={m['pdb_id']:m for m in tagged}
    print('Analyzing',len(tagged),'tagged entries',flush=True);allrows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init_chemistry,initargs=(a.ccd_root,)) as pool:
        for i,rs in enumerate(pool.map(evaluate,tagged,chunksize=1),1):
            allrows.extend(rs)
            if i%100==0:print('Analyzed',i,'rows',len(allrows),flush=True)
    # Assembly support is required for the high-confidence set. Consolidate ASU and
    # assembly copies by deposited entry, ligand and protein entities.
    assembly_keys={(r['pdb_id'],r['ccd'],tuple(sorted([r.get('entity_A',''),r.get('entity_B','')]))) for r in allrows if r.get('space','').startswith('assembly_') and r.get('tier')=='high_confidence'}
    for r in allrows:
        if r['tier']=='high_confidence' and r['space']=='asu' and (r['pdb_id'],r['ccd'],tuple(sorted([r['entity_A'],r['entity_B']]))) not in assembly_keys:
            r['tier']='review';r['reasons'].append('not_supported_in_deposited_biological_assembly')
    write_tsv(out/'all_ligand_instances.tsv',allrows)
    (out/'all_ligand_instances.json').write_text(json.dumps(allrows,indent=2))
    rank={'high_confidence':0,'review':1,'glue_or_proximity_review':2,'rejected':3,'error':4}
    chosen={}
    for r in sorted(allrows,key=lambda r:(rank[r['tier']],0 if r.get('space','').startswith('assembly_') else 1,-r.get('observed_heavy_atoms',0))):
        if 'complex_group' not in r:continue
        key=(r['pdb_id'],r['complex_group'])
        if key not in chosen:chosen[key]=r.copy()
    candidates=list(chosen.values())
    for r in candidates:r['record']=f"{r['pdb_id']}_{r['chain_A']}_{r['chain_B']}_{r['ccd']}"
    accepted=[r for r in candidates if r['tier']=='high_confidence']
    for d in ['reference','ligands','structures','raw']:(out/d).mkdir(exist_ok=True)
    for r in accepted:
        export_complex(r,lookup[r['pdb_id']],out)
        dest=out/'structures'/Path(lookup[r['pdb_id']]['path']).name
        if not dest.exists():shutil.copy2(lookup[r['pdb_id']]['path'],dest)
    write_tsv(out/'accepted.tsv',accepted)
    (out/'accepted_pdb_ids.txt').write_text('\n'.join(sorted({r['pdb_id'] for r in accepted}))+'\n')
    (out/'accepted_records.txt').write_text('\n'.join(sorted(r['record'] for r in accepted))+'\n')
    write_tsv(out/'review.tsv',[r for r in candidates if 'review' in r['tier']])
    write_tsv(out/'rejected_pairs.tsv',[r for r in candidates if r['tier']=='rejected'])
    write_tsv(out/'analysis_errors.tsv',[r for r in allrows if r['tier']=='error'])
    entry_outcomes=[]
    for m in tagged:
        erows=[r for r in allrows if r['pdb_id']==m['pdb_id']]
        entry_outcomes.append({'pdb_id':m['pdb_id'],'release':m['release'],'title':m['fields']['title'],'terms':m['terms'],'outcomes':dict(Counter(r['tier'] for r in erows)) if erows else {'no_nonpolymer_with_at_least_8_observed_heavy_atoms':1}})
    write_tsv(out/'tagged_entry_outcomes.tsv',entry_outcomes)
    representatives={}
    for r in sorted(accepted,key=lambda r:(r['resolution'] or 999,-r['ligand_completeness'],r['pdb_id'])):representatives.setdefault(r['complex_group'],r)
    write_tsv(out/'nonredundant.tsv',representatives.values())
    prior=Path('tpd/records_human_annot.tsv');comparison=[]
    bypdb={}
    for r in allrows:bypdb.setdefault((r['pdb_id'],r.get('ccd')),[]).append(r)
    if prior.exists():
        for old in csv.DictReader(prior.open(),delimiter='\t'):
            new=bypdb.get((old['pdb_id'],old['ccd']),[])
            original_pair=set(old['record'].split('_')[1:3])
            exact=[r for r in new if r.get('space')=='asu' and {r.get('chain_A'),r.get('chain_B')}==original_pair]
            best=min(exact,key=lambda r:rank[r['tier']]) if exact else None
            entry_best=min(new,key=lambda r:rank[r['tier']]) if new else None
            status=best['tier'] if best else 'outside_window' if not a.start<=old['release']<=a.end else 'rejected' if new else 'not_tagged_or_no_eligible_ligand'
            reasons=best.get('reasons',[]) if best else ['original_chain_pair_not_supported'] if new else []
            comparison.append({'record':old['record'],'pdb_id':old['pdb_id'],'ccd':old['ccd'],'release':old['release'],'human_annot':old.get('human_annot'),'new_status':status,'reasons':reasons,'best_entry_ligand_status':entry_best['tier'] if entry_best else ''})
    write_tsv(out/'existing_dataset_audit.tsv',comparison)
    summary={'tagged_entries':len(tagged),'instance_tiers':dict(Counter(r['tier'] for r in allrows)),'deduplicated_pair_tiers':dict(Counter(r['tier'] for r in candidates)),'accepted_entries':len({r['pdb_id'] for r in accepted}),'accepted_complexes':len(accepted),'nonredundant_complexes':len(representatives),'analysis_errors':sum(r['tier']=='error' for r in allrows),'accepted_class_counts':dict(Counter(r['class'] for r in accepted)),'accepted_sdf_status':dict(Counter(r.get('sdf_status','unknown') for r in accepted)),'review_reason_counts':dict(Counter(reason for r in candidates if 'review' in r['tier'] for reason in r['reasons'])),'existing_dataset_status':dict(Counter(r['new_status'] for r in comparison))}
    scan_summary=json.loads((out/'scan_summary.json').read_text())
    write_tsv(out/'filter_summary.tsv',[
        {'stage':'local_files_scanned','count':scan_summary['total'],'unit':'files'},
        {'stage':'released_in_requested_window','count':scan_summary['counts'].get('in_window',0),'unit':'entries'},
        {'stage':'annotation_or_machinery_tagged','count':len(tagged),'unit':'entries'},
        {'stage':'accepted_entries','count':summary['accepted_entries'],'unit':'entries'},
        {'stage':'accepted_complexes','count':len(accepted),'unit':'entry_sequence_pair_ligand'},
        {'stage':'nonredundant_complexes','count':len(representatives),'unit':'sequence_pair_ligand'},
        {'stage':'review_complexes','count':sum('review' in r['tier'] for r in candidates),'unit':'entry_sequence_pair_ligand'}])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    report=[
        '# Local TPD screening results',
        '',
        f"Initial release window: **{a.start} through {a.end}**, inclusive.",
        f"Newest initial release present in the scanned mirror: **{scan_summary['latest_local_release']}**.",
        '',
        '| Stage | Count |', '|---|---:|',
        f"| Local files scanned | {scan_summary['total']:,} |",
        f"| Entries in the date window | {scan_summary['counts'].get('in_window',0):,} |",
        f"| Entries screened structurally after annotation/machinery tagging | {len(tagged):,} |",
        f"| Accepted PDB entries | {summary['accepted_entries']} |",
        f"| Accepted entry / sequence-pair / ligand combinations | {len(accepted)} |",
        f"| Nonredundant sequence-pair / ligand combinations | {len(representatives)} |",
        f"| Combinations requiring review | {sum('review' in r['tier'] for r in candidates)} |",
        f"| Metadata scan errors | {scan_summary['counts'].get('error',0)} |",
        f"| Structural analysis errors | {summary['analysis_errors']} |",
        '',
        'The accepted set is a conservative computational shortlist, not proof of degradation.',
        'Drug-like chemistry, annotation evidence, heteromeric contacts and assembly support are combined;',
        'the exact rules and limitations are in [README.md](README.md).',
        '',
        'Start with [accepted.tsv](accepted.tsv), [nonredundant.tsv](nonredundant.tsv),',
        '[review.tsv](review.tsv), and [existing_dataset_audit.tsv](existing_dataset_audit.tsv).',
        'Extracted coordinates are in `reference/`, fully observed ligand SDFs in `ligands/`,',
        'and matching receptor/ligand pairs in `raw/`. Original deposited CIFs are in `structures/`.',
        '',
        'Accepted subtype counts: '+json.dumps(summary['accepted_class_counts'],sort_keys=True)+'.',
        'Ligand SDF export counts: '+json.dumps(summary['accepted_sdf_status'],sort_keys=True)+'.',
    ]
    if scan_summary['latest_local_release']<a.end:
        report += ['', '**Coverage limitation:** this mirror does not reach the requested end date. No structures were downloaded to fill the gap.']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
