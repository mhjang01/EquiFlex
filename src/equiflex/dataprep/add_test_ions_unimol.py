'''Phase B: append UniMol embeddings for the ion atoms added to test_data.
For each system, if unimol rows < ligand atoms, the trailing (ion) ligands lack
embeddings -> compute via UniMol and append. Preserves existing rows exactly.
'''
import sys, glob
sys.path.insert(0,'/home/mhjang/workspace/flex')
import torch
from pathlib import Path
from joblib import load
from equiflex.prep.featurize import UniMolExtractor

def main():
    dirs=sorted(glob.glob('test_data/*/'))
    append_todo=[]; truncate_todo=[]
    for d in dirs:
        lf=torch.load(d+'ligand_features.pt',weights_only=False,mmap=True)
        uni=torch.load(d+'unimol_embeddings.pt',weights_only=False,mmap=True)
        n_atoms=lf['atom_features'].shape[0]
        if n_atoms > uni.shape[0]:
            append_todo.append(d)
        elif n_atoms < uni.shape[0]:
            truncate_todo.append(d)

    print(f"{len(append_todo)} systems need ion UniMol append", flush=True)
    print(f"{len(truncate_todo)} systems need stale UniMol row truncation", flush=True)

    fixed=0
    for d in truncate_todo:
        n_atoms=torch.load(d+'ligand_features.pt',weights_only=False,mmap=True)['atom_features'].shape[0]
        old=torch.load(d+'unimol_embeddings.pt',weights_only=False)
        torch.save(old[:n_atoms].contiguous(), d+'unimol_embeddings.pt')
        fixed+=1

    if append_todo:
        ex=UniMolExtractor(model_size='1.1B')
        for d in append_todo:
            ligs=load(d+'ligands.pkl')
            old=torch.load(d+'unimol_embeddings.pt',weights_only=False)
            need=torch.load(d+'ligand_features.pt',weights_only=False,mmap=True)['atom_features'].shape[0]-old.shape[0]
            # trailing ligands accounting for `need` atoms = the ion ligands
            ion_ligs=[]; acc=0
            for l in reversed(ligs):
                ion_ligs.insert(0,l); acc+=len(l['atoms'])
                if acc>=need: break
            embs=[]
            for l in ion_ligs:
                e=ex.extract_single(l['smiles'], l['atoms'])
                if e is None:
                    e=torch.zeros(len(l['atoms']), old.shape[1], dtype=old.dtype)
                embs.append(e.to(old.dtype))
            new=torch.cat([old]+embs, dim=0)
            assert new.shape[0]==need+old.shape[0]
            torch.save(new, d+'unimol_embeddings.pt')
            fixed+=1
            if fixed%20==0: print(f"  {fixed}/{len(append_todo)+len(truncate_todo)}", flush=True)
    print(f"done: {fixed} systems", flush=True)

if __name__=='__main__': main()
