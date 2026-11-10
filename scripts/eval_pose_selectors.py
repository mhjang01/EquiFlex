"""Compare pose selectors on a PoseBusters run, against existing integration results.

Reads the run's own `pb_metrics.csv` and `posebusters_redock_checks.json`, so no
pose is re-busted and the definitions match
scripts/eval_posebusters_{redock,selected}.py exactly:

    pb_valid    = all checks except the rmsd_ ones True, NaN -> False
    lig_success = rmsd < 2.0
    sc_success  = rmsd_sc < 1.0;  sc_sym_success = rmsd_sc_sym < 1.0
    joint       = lig_success AND sc_sym_success

Leakage-free by construction. The repo's reported `success` is
`lig_success AND pb_valid`, so no selector may look at `pb_valid` -- selecting on
it optimises a conjunct of the reported metric, which inflates the number while
lowering real accuracy. pb_valid is reported as an outcome column only.

External scores are supplied as `--score name=path:column[:low]`, where `low`
marks a score that improves downward. Every pairwise rank-sum consensus of the
supplied scores plus Vinardo is evaluated too. Example:

    python scripts/eval_pose_selectors.py --run v10_41k_off_5A \
        --score GenScore=dekois/genscore/pb_genscore_v10_41k_off_5A_GT_0.0_1.csv:genscore \
        --score EquiScore=equiscore/scores_v10_41k_off_5A.csv:equiscore \
        --score RTMScore=rtmscore/pb_rtmscore_v10_41k_off_5A_rtmscore_model1.csv:rtmscore
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parent.parent
PB = ROOT / 'data/posebusters/v7'
LIG_THR, SC_THR = 2.0, 1.0


def load(run, specs):
    root = PB / run
    df = pd.read_csv(root / 'pb_metrics.csv', low_memory=False)
    checks = [c for c in json.load(open(root / 'posebusters_redock_checks.json'))['columns']
              if not c.startswith('rmsd_')]
    df['pb_valid'] = df.reindex(columns=checks).fillna(False).astype(bool).all(axis=1)
    df['lig_success'] = df.rmsd < LIG_THR
    df['sc_success'] = df.rmsd_sc < SC_THR
    df['sc_sym_success'] = df.rmsd_sc_sym < SC_THR
    df['joint_success'] = df.lig_success & df.sc_sym_success
    df['sample_number'] = df['sample'].str.removeprefix('sample_').astype(int)

    # name -> (column, higher_is_better); vinardo and vina come from pb_metrics itself
    scores = {'vinardo': ('vinardo', False)}
    if 'vina_real' in df.columns and df.vina_real.notna().any():
        scores['vina'] = ('vina_real', False)
    for name, path, col, low in specs:
        s = pd.read_csv(path)[['system', 'sample', col]]
        if col in df.columns:
            raise SystemExit(f'{name}: column {col!r} already in pb_metrics.csv')
        df = df.merge(s, on=['system', 'sample'], how='left')
        scores[name] = (col, not low)
    return df, scores


def parse_spec(raw):
    name, _, rest = raw.partition('=')
    parts = rest.split(':')
    if len(parts) == 2:
        path, col, low = parts[0], parts[1], False
    elif len(parts) == 3 and parts[2] == 'low':
        path, col, low = parts[0], parts[1], True
    else:
        raise SystemExit(f'bad --score spec {raw!r}; want name=path:column[:low]')
    return name, Path(path), col, low


def within_rank(df, col, higher_is_better):
    """Within-system rank, 1 = best. Missing scores rank last."""
    s = df[col] if higher_is_better else -df[col]
    return s.groupby(df.system).rank(ascending=False, na_option='bottom')


def selectors(df, scores):
    base = df.sort_values('sample_number')  # ties break on sample index, as the repo does
    out = {'oracle (min RMSD)': base.loc[base.groupby('system').rmsd.idxmin()]}
    ranks = {}
    for name, (col, hib) in scores.items():
        pick = base.groupby('system')[col]
        out[name] = base.loc[pick.idxmax() if hib else pick.idxmin()]
        ranks[name] = within_rank(base, col, hib)
    out['top-1 (sample_0)'] = base[base.sample_number == 0]
    for a, b in combinations([n for n in scores if n != 'vina'], 2):
        s = (ranks[a] + ranks[b]).rename('s')
        out[f'rank-sum {a}+{b}'] = base.loc[base.join(s).groupby('system').s.idxmin()]
    return out


def mcnemar(a, b):
    """Exact paired test on two boolean Series aligned on system. Returns p, wins, losses."""
    wins = int((a & ~b).sum())
    losses = int((b & ~a).sum())
    if wins + losses == 0:
        return 1.0, 0, 0
    return binomtest(wins, wins + losses, 0.5).pvalue, wins, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--score', action='append', default=[],
                    help='name=path:column[:low]; repeatable')
    ap.add_argument('--ref', default='vinardo', help='selector all others are tested against')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    df, scores = load(args.run, [parse_spec(s) for s in args.score])
    systems = df.system.unique()
    cover = {n: df.groupby('system')[c].apply(lambda s: s.notna().any()).sum()
             for n, (c, _) in scores.items()}
    print(f'{args.run}: {len(df)} poses, {len(systems)} systems, '
          f'{df.sample_number.max()+1} samples/system')
    print('  coverage: ' + ', '.join(f'{n} {v}/{len(systems)}' for n, v in cover.items()))

    picks, rows = {}, []
    for name, s in selectors(df, scores).items():
        s = s.drop_duplicates('system').set_index('system').reindex(systems)
        picks[name] = s
        rows.append(dict(
            selector=name,
            lig_2A=100 * s.lig_success.fillna(False).mean(),
            rmsd_median=s.rmsd.median(),
            rmsd_sc_median=s.rmsd_sc.median(),
            sc_1A=100 * s.sc_success.fillna(False).mean(),
            sc_sym_1A=100 * s.sc_sym_success.fillna(False).mean(),
            joint=100 * s.joint_success.fillna(False).mean(),
            pb_valid=100 * s.pb_valid.fillna(False).mean()))

    ref = picks[args.ref]
    w = max(len(n) for n in picks) + 1
    print(f'\n{"selector":{w}s} {"lig<2A":>7s} {"p(lig)":>8s} {"ligRMSD":>8s} {"scRMSD":>7s} '
          f'{"sc<1A":>7s} {"scsym<1A":>9s} {"joint":>7s} {"p(jnt)":>8s} {"pb_val":>7s}')
    for r in rows:
        name = r['selector']
        p1 = (np.nan if name == args.ref else
              mcnemar(picks[name].lig_success.fillna(False), ref.lig_success.fillna(False))[0])
        p2 = (np.nan if name == args.ref else
              mcnemar(picks[name].joint_success.fillna(False), ref.joint_success.fillna(False))[0])
        r['p_lig'], r['p_joint'] = p1, p2
        print(f'{name:{w}s} {r["lig_2A"]:7.2f} {p1:8.4f} {r["rmsd_median"]:8.3f} '
              f'{r["rmsd_sc_median"]:7.3f} {r["sc_1A"]:7.2f} {r["sc_sym_1A"]:9.2f} '
              f'{r["joint"]:7.2f} {p2:8.4f} {r["pb_valid"]:7.2f}')
    out = args.out or ROOT / f'data/posebusters/v7/{args.run}/selector_comparison.csv'
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f'\npaired wins/losses vs {args.ref} (lig<2A, then joint):')
    for name in picks:
        if name in (args.ref, 'oracle (min RMSD)'):
            continue
        p1, w1, l1 = mcnemar(picks[name].lig_success.fillna(False),
                             ref.lig_success.fillna(False))
        p2, w2, l2 = mcnemar(picks[name].joint_success.fillna(False),
                             ref.joint_success.fillna(False))
        print(f'  {name:{w}s} lig {w1:3d}/{l1:3d} p={p1:.4f}   joint {w2:3d}/{l2:3d} p={p2:.4f}')

    print('\nwithin-system pose-rank corr vs -RMSD (higher = better pose ranking):')
    for name, (col, hib) in scores.items():
        g = df.dropna(subset=[col, 'rmsd'])
        sign = 1.0 if hib else -1.0
        rho = g.groupby('system').apply(
            lambda x: spearmanr(sign * x[col], -x.rmsd).statistic if len(x) > 2 else np.nan)
        print(f'  {name:{w}s} median {rho.median():+.3f}  mean {rho.mean():+.3f}  '
              f'frac>0 {100*(rho > 0).mean():.1f}%')

    print(f'\npose-pick agreement with {args.ref} (% of systems):')
    for name in scores:
        if name == args.ref:
            continue
        print(f'  {name:{w}s} {100*(picks[name]["sample"] == ref["sample"]).mean():5.1f}%')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
