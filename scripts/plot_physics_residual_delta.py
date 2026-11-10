"""Histogram of per-sample physics-residual deltas written by physics_residual_delta.py."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TERMS = [('bond_length', 'Bond length'), ('bond_angle', 'Bond angle (1-3)'), ('vdw', 'vdW clash (hinge)'),
         ('torsion', 'Sp2 planarity'), ('chiral', 'Chirality'), ('weighted', 'Training-weighted sum')]
WORSE, BETTER = '#c4572f', '#2f6f9f'


def panel(ax, df, term, title, label_a, label_b):
    d = df[f'{term}_delta'].to_numpy()
    a, b = df[f'{term}_a'].to_numpy(), df[f'{term}_b'].to_numpy()
    nz = d[d != 0]
    lin = np.quantile(np.abs(nz), 0.10)
    sym = lambda x: np.sign(x) * np.log10(1 + np.abs(x) / lin)
    lim = sym(np.quantile(np.abs(nz), 0.995))
    edges = np.linspace(-lim, lim, 71)
    h, _ = np.histogram(np.clip(sym(nz), -lim, lim), bins=edges)
    w = edges[1] - edges[0]
    c = edges[:-1] + w / 2
    ax.bar(c[c >= 0], h[c >= 0], width=w, color=WORSE, lw=0)
    ax.bar(c[c < 0], h[c < 0], width=w, color=BETTER, lw=0)
    ymax = h.max() * 1.05
    ax.vlines(0, 0, ymax, color='k', lw=0.8)
    ax.vlines(sym(np.median(d)), 0, ymax, color='k', lw=1, ls='--')
    pos = np.array([-lim * 0.95, -lim / 2, 0, lim / 2, lim * 0.95])
    vals = np.sign(pos) * lin * (10 ** np.abs(pos) - 1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f'{v:+.0e}'.replace('e-0', 'e-').replace('e+0', 'e') if v else '0' for v in vals], fontsize=7)
    ax.set_ylim(0, h.max() * 1.85)
    per_sys = df.groupby('system')[f'{term}_delta'].mean()
    p = wilcoxon(per_sys[per_sys != 0]).pvalue
    ax.set_title(title, fontsize=10, loc='left')
    txt = (f'median {np.median(d):+.1e}   mean {d.mean():+.1e}\n'
           f'{100 * (d > 0).mean():.0f}% larger, {100 * (d < 0).mean():.0f}% smaller, '
           f'{100 * (d == 0).mean():.0f}% tie (not drawn)\n'
           f'mean residual {label_a}/{label_b} = {a.mean() / b.mean():.2f}x\n'
           f'per-system Wilcoxon p = {p:.1e}')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top', fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--label-a', default='ablation')
    ap.add_argument('--label-b', default='physics')
    ap.add_argument('--out', type=Path, required=True, help='output stem (writes .png and .pdf)')
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (term, title) in zip(axes.flat, TERMS):
        panel(ax, df, term, title, args.label_a, args.label_b)
    for ax in axes[:, 0]:
        ax.set_ylabel('poses', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(f'Δ residual = {args.label_a} − {args.label_b}  (symlog)', fontsize=8)
    fig.suptitle(f'Per-pose physics-residual delta at equal seed: {df.system.nunique()} systems × '
                 f'{len(df) // df.system.nunique()} = {len(df)} pairs\n'
                 f'red: {args.label_a} larger, blue: smaller, dashed: median; axis clipped at 99.5th percentile of |Δ|', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix('.png'), dpi=200)
    fig.savefig(args.out.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
