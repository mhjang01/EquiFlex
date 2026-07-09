'''
Scan v2_plinder_5A_v2split_rejected/ for systems where the ONLY co-binders
are metal ions (single-atom, SMILES matches `[<Metal>...]`).

A system qualifies if:
    - exactly 1 "organic" ligand (multi-atom, not a metal-ion atom)
    - >= 1 single-atom metal-ion co-binders
    - 0 other co-binders (no multi-atom artifacts, no other organics)

Outputs:
    v2_plinder_5A_v2split/logs/metal_only_cobinder_{train,val,test}.txt
    v2_plinder_5A_v2split/logs/metal_only_cobinder_summary.json
'''
import json
import re
from collections import Counter
from pathlib import Path

from joblib import Parallel, delayed, load
from tqdm import tqdm

REJ  = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split_rejected')
OUT  = Path('/home/mhjang/workspace/flex/v2_plinder_5A_v2split/logs')
SPLITS = ['train', 'val', 'test']

METAL_PATTERN = re.compile(r'^\[(Zn|Mg|Ca|Fe|Mn|Cu|Ni|Co|Cd|Hg|Pb|K|Na|Cs|Rb|Li|Sr|Ba|Be|Al|Cr|Mo|W|V|Pt|Pd|Ru|Rh|Au|Ag|Sn|Sb|As|Y|La|Eu|Gd|Tb|U)([+\-\d@HspBI]*)\]$')


def classify(smi: str, n_atoms: int) -> str:
    if n_atoms == 1 and METAL_PATTERN.match(smi.strip()):
        return 'metal_ion'
    if n_atoms > 1:
        return 'organic'
    return 'other_single_atom'


def check(sd):
    lf = sd / 'ligands.pkl'
    if not lf.exists():
        return None
    try:
        ligs = load(lf)
    except Exception:
        return None
    types = Counter()
    metals = []
    for l in ligs:
        n = len(l['atoms'])
        t = classify(l['smiles'], n)
        types[t] += 1
        if t == 'metal_ion':
            metals.append(l['smiles'])
    return {
        'sys_id': sd.name,
        'n_ligs': len(ligs),
        'types': dict(types),
        'metals': metals,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        sd = REJ / split
        if not sd.exists(): continue
        dirs = [d for d in sd.iterdir() if d.is_dir()]
        print(f'[{split}] scanning {len(dirs)}')
        recs = Parallel(n_jobs=-1)(delayed(check)(d) for d in tqdm(dirs, desc=split))
        recs = [r for r in recs if r is not None]

        qualifying = []
        for r in recs:
            t = r['types']
            if (t.get('organic', 0) == 1
                and t.get('metal_ion', 0) >= 1
                and t.get('other_single_atom', 0) == 0
                and sum(t.values()) == t.get('organic', 0) + t.get('metal_ion', 0)):
                qualifying.append(r)

        metal_hist = Counter()
        for r in qualifying:
            metal_hist[tuple(sorted(r['metals']))] += 1
        n_metals_hist = Counter(len(r['metals']) for r in qualifying)

        ids_fp = OUT / f'metal_only_cobinder_{split}.txt'
        ids_fp.write_text('\n'.join(sorted(r['sys_id'] for r in qualifying)) + '\n')

        summary[split] = {
            'total_rejected_scanned': len(recs),
            'metal_only_cobinder_count': len(qualifying),
            'fraction': len(qualifying)/len(recs) if recs else 0.0,
            'n_metals_distribution': dict(sorted(n_metals_hist.items())),
            'metal_set_top20': [{'metals': list(k), 'n_systems': v}
                                 for k, v in metal_hist.most_common(20)],
            'output_file': str(ids_fp),
        }
        print(f'[{split}] qualifying = {len(qualifying)} / {len(recs)}  '
              f'({100*len(qualifying)/max(1,len(recs)):.1f}%)')
        print(f'         n_metals dist: {dict(sorted(n_metals_hist.items()))}')
        print(f'         top metal combos:')
        for combo, n in metal_hist.most_common(5):
            print(f'             {n:>5}  {combo}')

    (OUT / 'metal_only_cobinder_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nSummary: {OUT/"metal_only_cobinder_summary.json"}')


if __name__ == '__main__':
    main()
