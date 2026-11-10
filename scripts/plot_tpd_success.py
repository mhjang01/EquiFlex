#!/usr/bin/env python3
"""Grouped bar chart of TPD redock success by class, split by PoseBusters validity.

One group per ligand class, two bars per group (ligand / receptor side chain).
Each bar is the plain success rate; the solid part is the share that is also
PoseBusters-valid and the hatched part the remainder, so bar height stays the
headline number and the hatch shows what physical validity removes.

    python scripts/plot_tpd_success.py --start 2024-07-01
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# dataviz reference palette, categorical slots 1 and 2 (validated light mode).
SERIES = {'ligand': '#2a78d6', 'sidechain': '#eb6834'}
SURFACE, INK, INK_SOFT, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#d8d7d2'
CLASS_LABELS = {'molecular_glue': 'Molecular glue',
                'molecular_glue_non_degrader': 'Non-degrader glue',
                'protac': 'PROTAC'}
LIG_THR, SC_THR = 2.0, 1.0


def rates(poses: Path, records: Path, start: str) -> pd.DataFrame:
    """Per class: plain and PoseBusters-valid success of the selected pose, in percent."""
    df = pd.read_csv(poses)
    meta = pd.read_csv(records, sep='\t').set_index('record')
    df = df[df.system.isin(meta.index)].copy()
    for column in ('class', 'release'):
        df[column] = df.system.map(meta[column])
    df = df[df.release >= start]
    df['pb'] = df.pb_valid.fillna(False).astype(bool)
    df['lig'] = df.rmsd < LIG_THR
    df['sc'] = df.rmsd_sc < SC_THR
    out = df.groupby('class').apply(lambda g: pd.Series({
        'n': len(g),
        'ligand': 100 * g.lig.mean(), 'ligand_pb': 100 * (g.lig & g.pb).mean(),
        'sidechain': 100 * g.sc.mean(), 'sidechain_pb': 100 * (g.sc & g.pb).mean()}),
        include_groups=False)
    return out.reindex(CLASS_LABELS).dropna(how='all')


def plot(table: pd.DataFrame, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    positions = range(len(table))
    width, offset = 0.3, 0.17

    ax.set_xlim(-0.55, len(table) - 0.45)
    ax.set_ylim(0, 72)
    ax.set_yticks(range(0, 71, 10))
    ax.set_ylabel('Success rate (%)', color=INK_SOFT, fontsize=10, labelpad=8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f'{CLASS_LABELS[c]}\n(n = {int(r.n)})' for c, r in table.iterrows()],
                       color=INK, fontsize=10.5)
    ax.tick_params(axis='both', length=0, colors=INK_SOFT, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_color(INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)

    # A 2px surface gap separates the two fills of a bar; convert it to data units
    # once the layout is final, since the bars themselves do not move the axes.
    fig.canvas.draw()
    gap = 2 / ax.get_window_extent().height * (ax.get_ylim()[1] - ax.get_ylim()[0])

    plt.rcParams['hatch.linewidth'] = 1.1
    for i, (_, row) in enumerate(table.iterrows()):
        for sign, key in ((-1, 'ligand'), (1, 'sidechain')):
            x = i + sign * offset
            colour, solid, total = SERIES[key], row[f'{key}_pb'], row[key]
            # Both fills carry the same stroke so the hatched segment, whose edge
            # draws its border, is not wider than the solid one by that linewidth.
            ax.bar(x, solid, width, facecolor=colour, edgecolor=colour, linewidth=1.1)
            if total - solid > gap:
                ax.bar(x, total - solid - gap, width, bottom=solid + gap,
                       facecolor=SURFACE, edgecolor=colour, hatch='///', linewidth=1.1)
            ax.text(x, total + 1.6, f'{total:.1f}', ha='center', va='bottom',
                    color=INK, fontsize=9.5, fontweight='medium')
            if solid > 7:
                ax.text(x, solid / 2, f'{solid:.1f}', ha='center', va='center',
                        color='white', fontsize=9, fontweight='medium')

    handles = [Patch(facecolor=SERIES['ligand']),
               Patch(facecolor=SURFACE, edgecolor=SERIES['ligand'], hatch='///'),
               Patch(facecolor=SERIES['sidechain']),
               Patch(facecolor=SURFACE, edgecolor=SERIES['sidechain'], hatch='///')]
    labels = [f'Ligand < {LIG_THR:.0f} Å & PB-valid', f'Ligand < {LIG_THR:.0f} Å only',
              f'Side chain < {SC_THR:.0f} Å & PB-valid', f'Side chain < {SC_THR:.0f} Å only']
    legend = ax.legend(handles, labels, ncol=2, loc='upper right', frameon=False,
                       fontsize=9, labelcolor=INK_SOFT, handlelength=1.5,
                       handleheight=1.1, columnspacing=1.6, borderpad=0)
    legend.set_zorder(5)
    ax.set_title(title, color=INK, fontsize=11.5, loc='left', pad=14)

    fig.tight_layout()
    for suffix in ('.png', '.svg'):
        fig.savefig(out.with_suffix(suffix), facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--poses', type=Path, default=Path('tpd/bench_db/vinardo_selected_poses.csv'))
    ap.add_argument('--records', type=Path, default=Path('tpd/bench_db/records.tsv'))
    ap.add_argument('--start', default='2024-07-01', help='keep systems released on or after this date')
    ap.add_argument('--out', type=Path, default=Path('tpd/bench_db/tpd_success'))
    args = ap.parse_args()

    table = rates(args.poses, args.records, args.start)
    plot(table, args.out, f'EquiFlex TPD redock, Vinardo-selected pose '
                          f'(released on or after {args.start})')
    print(table.round(1).to_string())
    print(f'-> {args.out.with_suffix(".png")}, {args.out.with_suffix(".svg")}')


if __name__ == '__main__':
    main()
