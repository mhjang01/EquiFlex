"""CrossDock performance stratified against BOTH methods' training sets.

A single identity axis is not a fair generalisation test: measured against the
v10 split, "novel" means novel to EquiFlex only, and DiffBindFR may well have
trained on the same receptor via PDBBind. This crosses the two identity columns
so each method is read against its own training set, and so the only honest
cell -- novel to both -- is reported separately.

Both methods are PB-gated here (RMSD < 2 A and all 27 non-RMSD PoseBusters
checks), which is the DiffBindFR paper's "PB-success" definition.

    python scripts/crossdock_identity_joint.py
"""
import argparse
from pathlib import Path

import pandas as pd

from crossdock_diffbindfr_summary import EQUIFLEX, SEQSIM, eq_key

PB_DBFR = Path('crossdock/results_diffbindfr/pb_mdn_top1.csv')
SEQSIM_PDBBIND = Path('data/crossdock/seqsim_pdbbind.tsv')
BANDS = [-.01, .3, .7, .9, 1.01]
LABELS = ['<30%', '30-70%', '70-90%', '>=90%']
NOVEL = 0.3


def load():
    dbfr = pd.read_csv(PB_DBFR, usecols=['key', 'subset', 'l-rmsd', 'pb_valid',
                                         'dbfr_lig_success', 'dbfr_pb_success'])

    eq = pd.read_csv(EQUIFLEX, usecols=['system', 'rmsd', 'lig_success', 'lig_success_pb'])
    eq['key'] = eq.system.map(eq_key)
    eq = eq.rename(columns={'rmsd': 'eq_rmsd', 'lig_success': 'eq_lig_success',
                            'lig_success_pb': 'eq_pb_success'})
    df = dbfr.merge(eq.drop(columns='system'), on='key', how='left')

    for path, col in ((SEQSIM, 'id_equiflex'), (SEQSIM_PDBBIND, 'id_pdbbind')):
        s = pd.read_csv(path, sep='\t')
        s['key'] = s.subset + '|' + s.system
        df = df.merge(s[['key', 'identity_aln']].rename(columns={'identity_aln': col}),
                      on='key', how='left')
    return df


def rates(df, by, observed=True):
    g = df.groupby(by, observed=observed)
    return pd.DataFrame({
        'n': g.size(),
        'dbfr_pb_success': 100 * g.dbfr_pb_success.mean(),
        'eq_pb_success': 100 * g.eq_pb_success.mean(),
        'dbfr_lig<2A': 100 * g.dbfr_lig_success.mean(),
        'eq_lig<2A': 100 * g.eq_lig_success.mean(),
    }).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('crossdock/results_diffbindfr'))
    args = ap.parse_args()

    df = load()
    print(f'{len(df)} systems; EquiFlex joined for {df.eq_pb_success.notna().sum()}')

    for col, who in (('id_equiflex', 'the v10 split (EquiFlex)'),
                     ('id_pdbbind', 'PDBBind time-split train+val (DiffBindFR)')):
        df[col + '_band'] = pd.cut(df[col], BANDS, labels=LABELS)
        print(f'\n=== by identity to {who}')
        print(rates(df, col + '_band').to_string())

    print('\n=== systems per (EquiFlex band x PDBBind band)')
    print(pd.crosstab(df.id_equiflex_band, df.id_pdbbind_band).to_string())

    # The four quadrants that matter: who has seen the receptor.
    seen_eq = df.id_equiflex >= NOVEL
    seen_db = df.id_pdbbind >= NOVEL
    df['quadrant'] = pd.Series(
        ['seen by both' if e and d else
         'EquiFlex only' if e else
         'DiffBindFR only' if d else
         'novel to both' for e, d in zip(seen_eq, seen_db)], index=df.index)
    print(f'\n=== by who has seen the receptor (>= {NOVEL:.0%} identity)')
    print(rates(df, 'quadrant', observed=True).to_string())

    novel = df[df.quadrant == 'novel to both']
    if len(novel):
        print(f'\n=== novel to both ({len(novel)} systems), by subset')
        print(rates(novel, 'subset').to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / 'identity_joint_per_system.csv', index=False)
    rates(df, 'quadrant').to_csv(args.out / 'by_quadrant.csv')
    print(f"\nwrote {args.out}/identity_joint_per_system.csv, by_quadrant.csv")


if __name__ == '__main__':
    main()
