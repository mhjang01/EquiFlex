'''
Summarize final v2-split pli_qcov<90 + atom<450 target survivors, unique ligand
SMILES, and identify deprecated parsed dirs in data_v2_plinder/train (on-disk but
NOT in target). Writes keep/deprecated lists. Does NOT delete (deletion is a
separate confirmed step).
'''
import json
from pathlib import Path

import pyarrow.parquet as pq

OUTDIR = Path('/home/mhjang/workspace/flex/data_v2_plinder/logs')
TR = Path('/home/mhjang/workspace/flex/data_v2_plinder/train')
ANNOT_PQ = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'


def main():
    target = sorted({l.strip() for l in open(OUTDIR / 'v2split_pli90_atom450_target.txt') if l.strip()})
    tset = set(target)
    print(f'target survivors: {len(target)}')

    # unique ligand SMILES (proper ligands) over target
    a = pq.read_table(ANNOT_PQ, columns=[
        'system_id', 'ligand_is_proper', 'ligand_rdkit_canonical_smiles']).to_pandas()
    at = a[a['system_id'].isin(tset)]
    proper = at[at['ligand_is_proper'] == True]['ligand_rdkit_canonical_smiles'].dropna()
    proper = proper[proper.astype(str).str.len() > 0]
    all_lig = at['ligand_rdkit_canonical_smiles'].dropna()
    all_lig = all_lig[all_lig.astype(str).str.len() > 0]
    uniq_proper = proper.nunique()
    uniq_all = all_lig.nunique()
    print(f'unique proper-ligand SMILES: {uniq_proper}')
    print(f'unique all-ligand SMILES:    {uniq_all}')

    # on-disk dirs
    on_disk = {d.name for d in TR.iterdir() if d.is_dir()}
    keep = sorted(on_disk & tset)
    deprecated = sorted(on_disk - tset)
    missing = sorted(tset - on_disk)  # target systems not yet parsed
    print(f'on_disk={len(on_disk)} keep={len(keep)} deprecated={len(deprecated)} target_not_yet_parsed={len(missing)}')

    (OUTDIR / 'v2split_pli90_keep_ondisk.txt').write_text('\n'.join(keep) + '\n')
    (OUTDIR / 'v2split_pli90_deprecated_to_delete.txt').write_text('\n'.join(deprecated) + '\n')

    summary = {
        'target_survivors': len(target),
        'unique_proper_ligand_smiles': int(uniq_proper),
        'unique_all_ligand_smiles': int(uniq_all),
        'on_disk_dirs': len(on_disk),
        'keep_on_disk': len(keep),
        'deprecated_to_delete': len(deprecated),
        'target_not_yet_parsed': len(missing),
    }
    (OUTDIR / 'v2split_pli90_finalize_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
