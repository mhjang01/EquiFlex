"""Aggregate the DiffBindFR CrossDock run and stratify it by training-set sequence identity.

Reads every results_ec.csv under the two output roots (the original whole-subset
run and the chunked rerun), takes each system's MDN top-1 pose, which is
DiffBindFR's own selector, and joins the per-system identity from
scripts/crossdock_seqsim.py. EquiFlex's Vinardo-selected poses are joined on the
same systems when available, so both methods are read on one denominator.

Safe to run while the docking is still going: it reports coverage and summarises
whatever has landed.

    python scripts/crossdock_diffbindfr_summary.py --out crossdock/results_diffbindfr
"""
import argparse
from pathlib import Path

import pandas as pd

# flex/comparison is the protected baseline tree: never deleted during flex cleanup.
OLD_ROOT = Path('/home/mhjang/workspace/flex/comparison/diffbindfr_run/out_crossdock')
NEW_ROOT = Path('/mnt/drive/diffbindfr_crossdock')
EQUIFLEX = Path('/home/mhjang/workspace/equiflex/crossdock/results_v10_41k/vinardo_selected_poses.csv')
SEQSIM = Path('/home/mhjang/workspace/equiflex/data/crossdock/seqsim_nw.tsv')
BANDS = [-.01, .3, .7, .9, 1.01]
LABELS = ['<30%', '30-70%', '70-90%', '>=90%']


# The run directory names the subset; the chunked rerun appends _cNNNNN. System
# ids are not unique across subsets (35 repeat), so the subset is part of the key.
SUBSET_OF_RUN = {'EGFR': 'Ensemble/EGFR', 'FXA': 'Ensemble/FXA', 'CDK2': 'Ensemble/CDK2'}


def subset_of_run(run):
    name = run.split('_c')[0] if '_c' in run and run.rsplit('_c', 1)[1].isdigit() else run
    name = name.replace('_smoke', '')
    return SUBSET_OF_RUN.get(name, name)


def eq_key(s):
    """EquiFlex system id -> 'subset|system'.

    Ids are '<subset>__<system>', and 'Ensemble__<target>__<system>' for the
    three ensemble targets, so splitting on the first '__' loses those.
    """
    p = s.split('__')
    return f'Ensemble/{p[1]}|{p[2]}' if p[0] == 'Ensemble' else f'{p[0]}|{p[1]}'


def load_runs():
    frames = []
    for root in (OLD_ROOT, NEW_ROOT):
        for csv in sorted(root.glob('*/results/results_ec.csv')):
            df = pd.read_csv(csv, usecols=['complex_name', 'l-rmsd', 'l-rmsd_ec',
                                           'sc-rmsd', 'mdn_score', 'smina_score'])
            df['run'] = csv.parent.parent.name
            df['subset'] = subset_of_run(df.run.iloc[0])
            frames.append(df)
    if not frames:
        raise SystemExit('no DiffBindFR results found')
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('crossdock/results_diffbindfr'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    poses = load_runs()
    print(f'{len(poses)} poses over {poses.complex_name.nunique()} systems '
          f'from {poses.run.nunique()} runs')

    # The smoke slice overlaps chunk 0, so a system can appear twice; keep the
    # first run that produced it so each system is counted once.
    poses['key'] = poses.subset + '|' + poses.complex_name
    poses = poses.sort_values(['key', 'run'])
    poses = poses[poses.run == poses.key.map(poses.groupby('key').run.first())]

    top1 = poses.loc[poses.groupby('key').mdn_score.idxmax()].copy()
    top1['lig_success'] = top1['l-rmsd'] < 2
    top1 = top1.merge(poses.groupby('key')['l-rmsd'].min().rename('best_of_n'), on='key')

    seq = pd.read_csv(SEQSIM, sep='\t')
    seq['key'] = seq.subset + '|' + seq.system
    top1 = top1.merge(seq[['key', 'identity_aln', 'identity_min']], on='key', how='left')
    missing = top1.identity_aln.isna().sum()
    if missing:
        print(f'WARNING {missing} systems have no sequence-identity row')

    if EQUIFLEX.exists():
        eq = pd.read_csv(EQUIFLEX, usecols=['system', 'rmsd', 'pb_valid',
                                            'lig_success', 'lig_success_pb'])
        eq['key'] = eq.system.map(eq_key)
        eq = eq.rename(columns={'rmsd': 'eq_rmsd', 'pb_valid': 'eq_pb_valid',
                                'lig_success': 'eq_lig_success',
                                'lig_success_pb': 'eq_joint'})
        top1 = top1.merge(eq.drop(columns='system'), on='key', how='left')

    top1.to_csv(args.out / 'diffbindfr_top1_per_system.csv', index=False)

    def table(df, by):
        g = df.groupby(by, observed=True)
        out = pd.DataFrame({
            'n': g.size(),
            'dbfr_lig<2A_%': 100 * g.lig_success.mean(),
            'dbfr_median': g['l-rmsd'].median(),
            'dbfr_bestofN_%': 100 * g.best_of_n.apply(lambda s: (s < 2).mean()),
        })
        if 'eq_lig_success' in df:
            out['eq_lig<2A_%'] = 100 * g.eq_lig_success.mean()
            out['eq_joint_%'] = 100 * g.eq_joint.mean()
            out['eq_median'] = g.eq_rmsd.median()
            out['n_eq'] = g.eq_lig_success.count()
        return out.round(2)

    by_subset = table(top1, 'subset')
    print('\nBy subset (DiffBindFR MDN top-1 vs EquiFlex Vinardo-selected)')
    print(by_subset.to_string())

    top1['identity_band'] = pd.cut(top1.identity_aln, BANDS, labels=LABELS)
    by_band = table(top1, 'identity_band')
    print('\nBy receptor identity to the v10 training split (alignment-length identity)')
    print(by_band.to_string())

    casf = top1[top1.subset == 'CASF2016']
    if len(casf):
        print('\nCASF2016 only, by identity band (the subset carrying the novel receptors)')
        print(table(casf, 'identity_band').to_string())

    by_subset.to_csv(args.out / 'by_subset.csv')
    by_band.to_csv(args.out / 'by_identity_band.csv')
    print(f'\nwrote {args.out}/diffbindfr_top1_per_system.csv, by_subset.csv, by_identity_band.csv')


if __name__ == '__main__':
    main()
