'''
Build a Flex batch-inference YAML for the internal Plinder test set
(data_v2_plinder/test/), mirroring the format of pb_test/inference_batch.yaml.

For each curated test system the generator emits:
  protein.path      -> source receptor.pdb under SOURCE_SYSTEM_DIR
  protein.residues  -> {auth_chain: [resnum, ...]} where resnum matches the
                       value run_inference's parse_structure(receptor.pdb)
                       reports as residue['resnum'] (1-indexed sequential)
  ligand.smiles     -> the single non-ion ligand
  ligand.ions       -> single-atom metal/ion entities (only if present)

Pocket residues are stored in the curated pocket_residues.pkl under a plinder
asym chain key (e.g. '1.A') with an older resnum convention. We map the asym
chain to its auth chain via chain_mapping.json and translate each pocket resnum
to the current parse's 1-indexed resnum by matching on either resnum or
pdb_resnum (original PDB numbering).
'''
import argparse
import json
import sys
from pathlib import Path

import yaml
from joblib import load
from rdkit import Chem

sys.path.insert(0, '/home/mhjang/workspace/flex')
from equiflex.prep.parse import parse_structure

TEST_ROOT = Path('/home/mhjang/workspace/flex/data_v2_plinder/test')
SOURCE_SYSTEM_DIR = Path('/mnt/drive/new_plinder/v2/systems')

ORGANIC = {'C', 'N', 'O', 'S', 'P', 'H', 'B', 'F', 'Cl', 'Br', 'I', 'Se'}


def is_ion(smiles: str) -> bool:
    '''Single heavy-atom non-organic entity (metal cation / monoatomic ion).'''
    m = Chem.MolFromSmiles(smiles)
    return (m is not None and m.GetNumAtoms() == 1
            and m.GetAtomWithIdx(0).GetSymbol() not in ORGANIC)


THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


def chain_resnum_maps(protein_data):
    '''Per chain, two maps keyed on the curated pocket's resnum convention,
    each yielding (current 1-indexed resnum, aa_code):

      by_pdb : {pdb_resnum -> (resnum, aa)}   original PDB numbering
      by_seq : {resnum     -> (resnum, aa)}   current 1-indexed sequential

    The curated pocket_residues.pkl stores the original PDB numbering, so
    by_pdb is the primary lookup; by_seq is a fallback. run_inference matches
    residues on the 1-indexed ``resnum``, which is what we emit.
    '''
    by_pdb, by_seq = {}, {}
    for chain_id, chain_data in protein_data.items():
        pdb_map, seq_map = {}, {}
        for res_key, res_data in chain_data.items():
            if res_key[0] is None:
                continue
            try:
                rn = int(res_data['resnum'])
            except (KeyError, ValueError, TypeError):
                continue
            aa = res_data.get('aa_code')
            if aa is None and isinstance(res_key[-1], str):
                aa = res_key[-1]
            seq_map[rn] = (rn, aa)
            if 'pdb_resnum' in res_data:
                try:
                    pdb_map[int(res_data['pdb_resnum'])] = (rn, aa)
                except (ValueError, TypeError):
                    pass
        by_pdb[chain_id] = pdb_map
        by_seq[chain_id] = seq_map
    return by_pdb, by_seq


def build_system(sys_id):
    '''Return (spec_dict, problems). spec_dict is None if unusable.'''
    sd = TEST_ROOT / sys_id
    src = SOURCE_SYSTEM_DIR / sys_id
    problems = []

    ligs = load(sd / 'ligands.pkl')
    pocket = load(sd / 'pocket_residues.pkl')
    cmap = json.loads((src / 'chain_mapping.json').read_text())

    mains = [l for l in ligs if not is_ion(l['smiles'])]
    ions = [l['smiles'] for l in ligs if is_ion(l['smiles'])]
    if len(mains) != 1:
        problems.append(f'expected 1 main ligand, got {len(mains)}')
        return None, problems

    protein_data = parse_structure(str(src / 'receptor.pdb'))
    by_pdb, by_seq = chain_resnum_maps(protein_data)

    residues = {}
    for asym_chain, res_dict in pocket.items():
        auth = cmap.get(asym_chain, asym_chain)
        pdb_map = by_pdb.get(auth, {})
        seq_map = by_seq.get(auth, {})
        out = set()
        for res_key in res_dict.keys():
            resname, n = res_key[0], res_key[1]
            want_aa = THREE_TO_ONE.get(resname)
            # Curated pocket resnum is the original PDB numbering: translate via
            # pdb_resnum, verifying the residue identity (resname). Fall back to
            # the 1-indexed convention only on a verified match.
            cand = pdb_map.get(n)
            if cand is None or (want_aa is not None and cand[1] != want_aa):
                seq_cand = seq_map.get(n)
                if seq_cand is not None and (want_aa is None or seq_cand[1] == want_aa):
                    cand = seq_cand
            if cand is None:
                problems.append(f'{asym_chain}->{auth}: resnum {n} ({resname}) unmatched')
                continue
            if want_aa is not None and cand[1] != want_aa:
                problems.append(
                    f'{asym_chain}->{auth}: resnum {n} resname {resname} '
                    f'!= parsed {cand[1]}')
                continue
            out.add(cand[0])
        if out:
            residues[auth] = sorted(out)

    if not residues:
        problems.append('no residues mapped')
        return None, problems

    spec = {
        'protein': {
            'path': str(src / 'receptor.pdb'),
            'residues': residues,
        },
        'ligand': {'smiles': mains[0]['smiles']},
    }
    if ions:
        spec['ligand']['ions'] = ions
    return spec, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path,
                   default=Path('/home/mhjang/workspace/flex/int_test/inference_batch.yaml'))
    p.add_argument('--output-root',
                   default='/home/mhjang/workspace/flex/int_test/int_test_output')
    p.add_argument('--checkpoint',
                   default='/home/mhjang/workspace/flex/checkpoints/32M/last.ckpt')
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--num-steps', type=int, default=50)
    p.add_argument('--schedule', default='uniform')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()

    sys_ids = sorted(d.name for d in TEST_ROOT.iterdir()
                     if d.is_dir() and d.name != 'errors')

    systems = {}
    skipped, with_problems = [], []
    for i, sid in enumerate(sys_ids, 1):
        spec, problems = build_system(sid)
        if problems:
            with_problems.append((sid, problems))
        if spec is None:
            skipped.append(sid)
            continue
        systems[sid] = spec
        if i % 50 == 0:
            print(f'  {i}/{len(sys_ids)} processed')

    print(f'\nsystems emitted: {len(systems)}/{len(sys_ids)}')
    print(f'skipped: {len(skipped)} {skipped[:10]}')
    print(f'systems with residue problems: {len(with_problems)}')
    for sid, probs in with_problems[:15]:
        print(f'  {sid}: {probs[:3]}')

    if args.validate_only:
        return

    config = {
        'checkpoint': args.checkpoint,
        'output_root': args.output_root,
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'schedule': args.schedule,
        'seed': args.seed,
        'device': args.device,
        'relax': False,
        'relax_max_iterations': 1000,
        'return_trajectory': False,
        'guidance': {
            'enabled': False, 't_start': 0.3, 'scale': 200.0,
            'schedule': 'linear_decay', 'bond_coef': 1.0, 'angle_coef': 1.0,
            'vdw_coef': 10.0, 'torsion_coef': 0.0, 'grad_clip': 0.0,
        },
        'sde': {'enabled': False, 'sigma': 0.15, 'eps_w': 0.01, 'w_cutoff': 0.99},
        'systems': systems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f'\nWrote {len(systems)} systems -> {args.out}')


if __name__ == '__main__':
    main()
