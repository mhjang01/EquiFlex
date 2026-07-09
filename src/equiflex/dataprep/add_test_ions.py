'''Incorporate nearby metal ions as ligand entities into PoseBusters test_data.
Phase A: append ion ligand dicts + regenerate ligand_features.pt (CPU).
Records augmented systems to test_data/_ion_aug.json for the UniMol phase.
'''
import sys, glob, json
sys.path.insert(0,'/home/mhjang/workspace/flex')
import numpy as np, torch, gemmi
from pathlib import Path
from joblib import load, dump, Parallel, delayed
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from equiflex.prep.parse import compute_mol_context, extract_atom_data
from equiflex.dataprep.featurize_data import process_ligands

PB=Path('/mnt/drive/posebusters')
METAL_CHARGE={'NA':1,'K':1,'MG':2,'CA':2,'MN':2,'FE':2,'CO':2,'NI':2,'CU':2,'ZN':2,'CD':2,'HG':2,'MO':0,'W':0}
CUT=5.0
ION_RESNAMES=set(METAL_CHARGE)

def build_ion(el, coord, chain):
    c=METAL_CHARGE.get(el,2)
    smi=f'[{el.capitalize()}+{c}]' if c>0 else f'[{el.capitalize()}]'
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    ctx=compute_mol_context(mol)
    a=mol.GetAtomWithIdx(0)
    atom=extract_atom_data(a, name=f'{el.upper()}1', mol_ctx=ctx)
    return {'chain':chain,'smiles':Chem.MolToSmiles(mol),'atoms':[atom],'bonds':[],
            'coords':np.array([coord],dtype=np.float32),'mapping':{0:'1'},
            'ref_coords':np.array([[0.,0.,0.]],dtype=np.float32),
            'sp2_torsion_indices':np.empty((0,4),dtype=np.int32),
            'constraint_matrix':np.array([[0.]],dtype=np.float32)}

def find_ions(tid, lig_coords):
    prot=list((PB/tid).glob('*_protein.pdb'))
    if not prot: return []
    st=gemmi.read_structure(str(prot[0]))
    out=[]
    for model in st:
        for ch in model:
            for res in ch:
                atoms=list(res)
                # Only standalone ion residues should be appended as ligand
                # entities. Cofactors such as HEM contain metal atoms too, but
                # those metals are chemically part of the cofactor and must not
                # become free sampled ions.
                # Only standalone single-atom residues can be free ions.
                # Residue name need not match element exactly (FE2, ZN2, etc.
                # are valid PDB residue names for standalone ions).
                if len(atoms)!=1:
                    continue
                at=atoms[0]
                el=at.element.name.upper()
                if el in METAL_CHARGE:
                    p=np.array([at.pos.x,at.pos.y,at.pos.z],dtype=np.float32)
                    if np.min(np.linalg.norm(lig_coords-p,axis=1))<CUT:
                        out.append((el,p,ch.name))
    return out

def is_single_atom_metal_ligand(lig):
    atoms=lig.get('atoms',[])
    return len(atoms)==1 and atoms[0].get('symbol','').upper() in METAL_CHARGE

def process(d):
    d=Path(d); tid=d.name
    ligs=load(d/'ligands.pkl')

    # Rebuild the automatically added ion set instead of skipping existing
    # single-metal ligands. This lets a rerun repair stale outputs generated
    # before cofactor metals (e.g. HEM FE) were excluded.
    base_ligs=[l for l in ligs if not is_single_atom_metal_ligand(l)]
    removed_existing=len(ligs)-len(base_ligs)
    if not base_ligs:
        return (tid,'nolig',0,0,[],0,removed_existing)

    lig_coords=np.concatenate([np.asarray(l['coords'],dtype=np.float32) for l in base_ligs],0)
    ions=find_ions(tid, lig_coords)

    seen=set(); added=0; smis=[]
    repaired=list(base_ligs)
    for el,p,chain in ions:
        key=(el,tuple(np.round(p,2)))
        if key in seen: continue
        seen.add(key)
        ent=build_ion(el,p,chain)
        if ent is None: continue
        repaired.append(ent); added+=1; smis.append(ent['smiles'])

    if removed_existing==0 and added==0:
        return (tid,'noion',0,int(lig_coords.shape[0]),[],len(base_ligs),0)

    dump(repaired, d/'ligands.pkl', compress=3)
    feats=process_ligands(repaired)
    torch.save(feats, d/'ligand_features.pt')
    if removed_existing and added:
        status='repair'
    elif removed_existing:
        status='removed_bad'
    else:
        status='aug'
    return (tid,status,added,int(lig_coords.shape[0]),smis,feats['atom_features'].shape[0],removed_existing)

if __name__=='__main__':
    only=sys.argv[1] if len(sys.argv)>1 else None
    dirs=sorted(glob.glob('test_data/*/'))
    if only: dirs=[d for d in dirs if Path(d).name==only]
    res=Parallel(n_jobs=-1,backend='loky',batch_size=8)(delayed(process)(d) for d in dirs)
    changed=[r for r in res if r[1] in {'aug','repair','removed_bad'}]
    from collections import Counter
    print('status:',dict(Counter(r[1] for r in res)))
    print('total ions added:',sum(r[2] for r in changed))
    print('total stale metal ligands removed:',sum(r[6] for r in changed))
    # save manifest for the UniMol phase / audit
    manifest={r[0]:{'status':r[1],'n_old_atoms':r[3],'ion_smiles':r[4],
                    'new_total':r[5],'removed_existing_metals':r[6]}
              for r in changed}
    json.dump(manifest, open('test_data/_ion_aug.json','w'), indent=2)
    print('manifest systems:',len(manifest))
