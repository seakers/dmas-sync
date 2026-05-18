"""
centralization_plots.py
=======================
Generates all analysis plots for the Centralized vs Decentralized
satellite scheduling experiment.

Usage
-----
    python centralization_plots.py                        # abridged set only
    python centralization_plots.py --full                 # full set only
    python centralization_plots.py --abridged --full      # both

    python centralization_plots.py --csv path/to/file.csv  # custom CSV path

The trial_name is derived from the CSV filename stem by default.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Sequence

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
})


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — LABELS, ORDERING, PALETTES
# ══════════════════════════════════════════════════════════════════════════════

# Algorithm display order: most centralised → most decentralised
ALGO_ORDER: list[str] = ['MILP', 'DP', 'DP-GR', 'GR', 'DP-CBBA', 'CBBA']

# Wong / Okabe-Ito colourblind-safe palette, extended.
# Greedy family: orange pair. CBBA family: blue/teal pair. MILP: rose. DP: grey.
ALGO_PALETTE: dict[str, str] = {
    'MILP':    '#CC79A7',  # muted rose  — centralized baseline
    'DP':      '#999999',  # grey        — preplanner only
    'DP-GR':   '#E69F00',  # amber       — DP + Greedy replanner
    'GR':      '#F0E442',  # yellow      — Greedy replanner only
    'DP-CBBA': '#0072B2',  # deep blue   — DP + CBBA replanner
    'CBBA':    '#56B4E9',  # sky blue    — CBBA replanner only
}

ALGO_LINESTYLES: dict[str, tuple | str] = {
    'MILP':    (2, 2),
    'DP':      (4, 2),
    'DP-GR':   (4, 1, 1, 1),
    'GR':      (1, 2),
    'DP-CBBA': '',          # solid
    'CBBA':    (3, 1),
}

# Connectivity: most restrictive → most permissive
CONN_ORDER: list[str] = ['GS', 'Intraconstellation', 'Interconstellation']
CONN_LABELS: dict[str, str] = {
    'GS':                 'GS Only',
    'Intraconstellation': 'Intra-\nconstellation',
    'Interconstellation': 'Inter-\nconstellation',
}

# Data Processing display labels (rename in plots only, not in data)
DP_ORDER: list[str] = ['Ground', 'Oracle', 'Onboard']
DP_LABELS: dict[str, str] = {
    'Ground':  'Legacy Ground\nDetection',
    'Onboard': 'Onboard\nDetection',
    'Oracle':  'Instant Ground\nDetection',
}
DP_LABELS_SHORT: dict[str, str] = {
    'Ground':  'Legacy Ground',
    'Onboard': 'Onboard',
    'Oracle':  'Instant Ground',
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def label_algorithm(row: pd.Series) -> str:
    pre, rep = row['Preplanner'], row['Replanner']
    if pre == 'Centralized-MILP_priority':
        return 'MILP'
    if pre == 'DP'   and rep == 'None':   return 'DP'
    if pre == 'DP'   and rep == 'CBBA':   return 'DP-CBBA'
    if pre == 'DP'   and rep == 'Greedy': return 'DP-GR'
    if pre == 'None' and rep == 'CBBA':   return 'CBBA'
    if pre == 'None' and rep == 'Greedy': return 'GR'
    return 'Unknown'


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Preplanner'] = df['Preplanner'].fillna('None')
    df['Replanner']  = df['Replanner'].fillna('None')
    df['Algorithm']  = df.apply(label_algorithm, axis=1)

    # Derived metric: reward normalised by Known Task Dual Bound
    # NOTE: Only meaningful *within* a Data Processing mode because the
    # Known Dual Bound varies significantly across modes (Oracle tasks ≫
    # Ground/Onboard tasks).  Cross-mode comparisons of this metric should
    # be interpreted with caution.
    df['Total Obtained Reward [known_norm]'] = (
        df['Total Obtained Reward'] / df['Known Task Reward Dual Bound']
    )

    # Enforce category orders for consistent plotting
    df['Algorithm']       = pd.Categorical(df['Algorithm'],
                                            categories=ALGO_ORDER, ordered=True)
    df['Connectivity']    = pd.Categorical(df['Connectivity'],
                                            categories=CONN_ORDER, ordered=True)
    df['Data Processing'] = pd.Categorical(df['Data Processing'],
                                            categories=DP_ORDER, ordered=True)

    return df.sort_values(['Algorithm', 'Connectivity', 'Data Processing'])


def save_plot(save_dir: str, local_save_dir: str,
              base_dir: str, local_base_dir: str,
              filename: str) -> None:
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        local_path = os.path.join(local_save_dir, filename)
        plt.savefig(local_path, dpi=150, bbox_inches='tight')
    print(f'  Saved → {path}')


def _dp_label(dp: str, short: bool = False) -> str:
    return (DP_LABELS_SHORT if short else DP_LABELS).get(dp, dp)


def _conn_label(conn: str) -> str:
    return CONN_LABELS.get(conn, conn)


# ══════════════════════════════════════════════════════════════════════════════
#  BOX + STRIP  (outline-only style, red median, blue box edges)
# ══════════════════════════════════════════════════════════════════════════════

def make_boxplot(data: pd.DataFrame, metric: str, ax: plt.Axes,
                 ylabel: str | None = None) -> None:
    """
    Outline-only box plot (no fill, blue edges, red median line) with
    individual trial points overlaid as a strip plot.
    """
    sns.boxplot(
        data=data, x='Algorithm', y=metric,
        order=ALGO_ORDER, palette=ALGO_PALETTE,
        hue='Algorithm', hue_order=ALGO_ORDER,
        width=0.5, linewidth=1.2, fliersize=0,
        legend=False, ax=ax,
        # Outline-only: set fill alpha to 0 after drawing
        boxprops=dict(facecolor='none', linewidth=1.2),
        medianprops=dict(color='red', linewidth=1.8),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
    )
    # Re-colour box edges to algorithm colour and keep no fill
    for patch, algo in zip(ax.patches, ALGO_ORDER):
        patch.set_edgecolor(ALGO_PALETTE[algo])
        patch.set_facecolor('none')
        patch.set_linewidth(1.4)

    sns.stripplot(
        data=data, x='Algorithm', y=metric,
        order=ALGO_ORDER, palette=ALGO_PALETTE,
        hue='Algorithm', hue_order=ALGO_ORDER,
        size=5, alpha=0.55, jitter=True, dodge=False,
        legend=False, ax=ax, marker='o',
        edgecolor='grey', linewidth=0.4,
    )

    ax.set_xlabel('Algorithm Configuration')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE A — box + strip by Algorithm
#  FIGURE B — grouped bar charts  (Connectivity × Algorithm, faceted by DP)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric(
    df: pd.DataFrame,
    metrics: str | list[str],
    titles: str | list[str],
    ylabels: str | list[str],
    suptitle: str,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
    lower_ref_col: str | None = None,
    upper_ref_col: str | None = None,
    response_time_note: bool = False,
) -> None:
    """
    Generates Figure A (box+strip by Algorithm) and Figure B (grouped bars
    by Connectivity, faceted by Data Processing) for one or more metrics.

    Parameters
    ----------
    hline_at_one      : draw a dashed reference line at y=1.0
    lower_ref_col     : column name for a lower bound reference bar
    upper_ref_col     : column name for an upper bound reference line
    response_time_note: add caveat annotation about passive observation bias
    """
    if isinstance(metrics, str):
        metrics = [metrics]
        titles  = [titles]
        ylabels = [ylabels]
    n = len(metrics)

    # ── Figure A: box + strip ──────────────────────────────────────────────
    fig, axes = plt.subplots(n, 1,
                              figsize=(10, 5 * n),
                              sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, metric, title, ylabel) in enumerate(
            zip(axes, metrics, titles, ylabels)):
        make_boxplot(df, metric, ax, ylabel=ylabel)
        if hline_at_one:
            ax.axhline(1.0, color='#444444', linestyle='--',
                       linewidth=0.9, alpha=0.6, label='Dual Bound (=1)')
        if df[metric].max() <= 1.05:
            ax.set_ylim(-0.02, 1.05)
        if n > 1:
            ax.set_title(f'({chr(ord("a") + i)})  {title}', loc='left')
            if i < n - 1:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

    if response_time_note:
        axes[-1].annotate(
            '⚠ Response time includes passive nadir observations; '
            'differences between algorithms are attenuated.',
            xy=(0.01, 0.01), xycoords='axes fraction',
            fontsize=7, color='#666666', style='italic',
        )

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_by_algorithm.png')
    plt.close()

    # ── Figure B: grouped bars by Connectivity, faceted by Data Processing ─
    n_dp   = len(DP_ORDER)
    fig_h  = 5 * n
    fig, axes = plt.subplots(n, n_dp,
                              figsize=(5 * n_dp, fig_h),
                              sharey='row')
    if n == 1:
        axes = np.array([axes])        # ensure 2-D indexing

    for row_i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col_i, dp in enumerate(DP_ORDER):
            ax = axes[row_i][col_i]
            sub = df[df['Data Processing'] == dp].copy()

            # Optional lower reference: Primal Bound [norm] per Connectivity
            if lower_ref_col and lower_ref_col in df.columns:
                ref_lower = (
                    sub.groupby('Connectivity')[lower_ref_col]
                    .mean().reindex(CONN_ORDER)
                )
            else:
                ref_lower = None

            # Optional upper reference: Dual Bound per Connectivity (constant)
            if upper_ref_col and upper_ref_col in df.columns:
                ref_upper = (
                    sub.groupby('Connectivity')[upper_ref_col]
                    .mean().reindex(CONN_ORDER)
                )
            else:
                ref_upper = None

            sns.barplot(
                data=sub,
                x='Connectivity', y=metric,
                hue='Algorithm', hue_order=ALGO_ORDER,
                palette=ALGO_PALETTE,
                order=CONN_ORDER,
                errorbar='sd',
                width=0.7, ax=ax,
            )

            # Draw reference lines per x-tick position
            n_conn = len(CONN_ORDER)
            bar_group_w = 0.7
            for xi, conn in enumerate(CONN_ORDER):
                if hline_at_one:
                    ax.plot([xi - bar_group_w / 2, xi + bar_group_w / 2],
                            [1.0, 1.0],
                            color='black', linewidth=1.0,
                            linestyle='--', alpha=0.5)
                if ref_lower is not None and conn in ref_lower.index:
                    val = ref_lower[conn]
                    if pd.notna(val):
                        ax.plot([xi - bar_group_w / 2, xi + bar_group_w / 2],
                                [val, val],
                                color='#888888', linewidth=0.8,
                                linestyle=':', alpha=0.6)

            ax.set_xlabel('Connectivity')
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.set_xticks(range(len(CONN_ORDER)))
            ax.set_xticklabels(
                [_conn_label(c) for c in CONN_ORDER],
                fontsize=8,
            )
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)

            if sub[metric].max() <= 1.05:
                ax.set_ylim(0.0, 1.05)

            # Column title = Data Processing label
            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')

            # Legend only on top-right panel
            if row_i == 0 and col_i == n_dp - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

        if response_time_note:
            axes[row_i][-1].annotate(
                '⚠ Passive obs. bias',
                xy=(1.0, 0.01), xycoords='axes fraction',
                fontsize=6, color='#888888', style='italic', ha='right',
            )

    plt.suptitle(f'{suptitle} — by Connectivity & Detection Mode',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_grouped.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2a — Algorithm Decomposition Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_decomposition_heatmap(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
) -> None:
    """
    3-panel heatmap (one per Data Processing mode).
    Rows = Preplanner  (MILP-row / DP-row / None-row)
    Cols = Replanner   (None / CBBA / Greedy)
    Cell = mean ± std  of `metric`.
    MILP occupies its own row separated by a horizontal line.
    """
    # Rows: Preplanner axis (top = centralized, middle = DP, bottom = None)
    # Cols: Replanner axis, ordered No Replanner -> Greedy -> CBBA
    row_labels = ['MILP (Centralized)', 'DP (Onboard Pre)', 'None (No Pre)']
    col_labels = ['No Replanner', 'Greedy', 'CBBA']

    # Build lookup: algo_label -> (row_i, col_i)
    # col 0 = No Replanner, col 1 = Greedy, col 2 = CBBA
    algo_cell = {
        'MILP':    (0, 0),
        'DP':      (1, 0),
        'DP-GR':   (1, 1),
        'DP-CBBA': (1, 2),
        'GR':      (2, 1),
        'CBBA':    (2, 2),
    }

    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(5 * len(DP_ORDER), 5),
                              sharey=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub  = df[df['Data Processing'] == dp]
        grid = np.full((3, 3), np.nan)
        text = [[''] * 3 for _ in range(3)]

        for algo, (ri, ci) in algo_cell.items():
            vals = sub[sub['Algorithm'] == algo][metric]
            if len(vals) > 0:
                m, s   = vals.mean(), vals.std(ddof=0)
                grid[ri, ci] = m
                text[ri][ci] = f'{m:.3f}\n±{s:.3f}'

        # MILP spans only column 0; columns 1 & 2 of row 0 are empty
        # Mask them so they appear white
        masked = np.ma.masked_invalid(grid)

        im = ax.imshow(masked, cmap='YlOrRd',
                       vmin=df[metric].min(), vmax=df[metric].max(),
                       aspect='auto')

        # Grid text
        for ri in range(3):
            for ci in range(3):
                if text[ri][ci]:
                    ax.text(ci, ri, text[ri][ci],
                            ha='center', va='center',
                            fontsize=8,
                            color='black' if masked[ri, ci] < 0.65 * df[metric].max()
                                         else 'white')

        # Axes labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
        ax.set_xlabel('Replanner', fontsize=9)
        if dp == DP_ORDER[0]:
            ax.set_ylabel('Preplanner', fontsize=9)

        # Separator line between MILP row and DP/None rows
        ax.axhline(0.5, color='white', linewidth=3)

        plt.colorbar(im, ax=ax, shrink=0.8, label=ylabel)

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_heatmap.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2b — Algorithm Decomposition: Box+Strip Faceted by Data Processing
# ══════════════════════════════════════════════════════════════════════════════

def plot_decomposition_boxplots(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
) -> None:
    """
    3-panel (one per Data Processing) box+strip plots, all algorithms.
    Allows statistical comparison across the decomposition space.
    """
    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(5 * len(DP_ORDER), 5),
                              sharey=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub = df[df['Data Processing'] == dp]
        make_boxplot(sub, metric, ax, ylabel=ylabel if dp == DP_ORDER[0] else '')
        if hline_at_one:
            ax.axhline(1.0, color='#444444', linestyle='--',
                       linewidth=0.9, alpha=0.6)
        if sub[metric].max() <= 1.05:
            ax.set_ylim(-0.02, 1.05)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
        ax.set_xlabel('')

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_boxplots.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 6 — Connectivity × Detection Interaction (line plot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_interaction(
    df: pd.DataFrame,
    metrics: list[str],
    ylabels: list[str],
    suptitle: str,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
) -> None:
    """
    Line plot: x = Data Processing (ordered Ground→Onboard→Oracle),
    y = metric mean, one line per Algorithm, faceted by Connectivity (columns).

    Shows how each algorithm benefits as detection capability improves,
    and whether that benefit depends on connectivity level.
    """
    n_metrics = len(metrics)
    n_conn    = len(CONN_ORDER)
    fig, axes = plt.subplots(n_metrics, n_conn,
                              figsize=(5 * n_conn, 4.5 * n_metrics),
                              sharey='row', sharex=True)
    if n_metrics == 1:
        axes = np.array([axes])

    # X positions for Data Processing
    dp_positions = {dp: i for i, dp in enumerate(DP_ORDER)}

    for row_i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col_i, conn in enumerate(CONN_ORDER):
            ax = axes[row_i][col_i]
            sub = df[df['Connectivity'] == conn]

            for algo in ALGO_ORDER:
                asub = sub[sub['Algorithm'] == algo]
                means = asub.groupby('Data Processing')[metric].mean()
                stds  = asub.groupby('Data Processing')[metric].std(ddof=0).fillna(0)

                xs = [dp_positions[dp] for dp in DP_ORDER if dp in means.index]
                ys = [means[dp] for dp in DP_ORDER if dp in means.index]
                es = [stds[dp]  for dp in DP_ORDER if dp in means.index]

                ls_spec = ALGO_LINESTYLES[algo]
                if ls_spec == '':
                    linestyle = 'solid'
                    dashes    = None
                else:
                    linestyle = 'dashed'
                    dashes    = ls_spec

                line, = ax.plot(xs, ys,
                                color=ALGO_PALETTE[algo],
                                linewidth=1.6,
                                marker='o', markersize=5,
                                linestyle=linestyle,
                                label=algo)
                if dashes:
                    line.set_dashes(dashes)

                ax.fill_between(xs,
                                [y - e for y, e in zip(ys, es)],
                                [y + e for y, e in zip(ys, es)],
                                color=ALGO_PALETTE[algo], alpha=0.10)

            if hline_at_one:
                ax.axhline(1.0, color='#444444', linestyle='--',
                           linewidth=0.8, alpha=0.5)

            ax.set_xticks(list(dp_positions.values()))
            ax.set_xticklabels(
                [_dp_label(dp, short=True) for dp in DP_ORDER],
                fontsize=8, rotation=10, ha='right',
            )
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            if sub[metric].max() <= 1.05:
                ax.set_ylim(-0.02, 1.05)

            # Column titles (connectivity) on top row
            if row_i == 0:
                ax.set_title(f'Connectivity: {_conn_label(conn).replace(chr(10), " ")}',
                             fontsize=10, fontweight='bold')

            # Legend on top-right only
            if row_i == 0 and col_i == n_conn - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')

    plt.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_interaction.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 7 — Event Detection Pipeline Cascade
# ══════════════════════════════════════════════════════════════════════════════

def plot_detection_cascade(
    df: pd.DataFrame,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
) -> None:
    """
    For each Connectivity × Data Processing combination, shows three adjacent
    bars per Algorithm: P(Event Detected), P(Event Observed | Detected),
    P(Event Co-observed | Detected). Reveals where the pipeline bottleneck is.
    """
    cascade_metrics = [
        ('P(Event Detected)',                 'P(Event Detected)'),
        ('P(Event Observed | Event Detected)', 'P(Obs | Detected)'),
        ('P(Event Co-observed | Event Detected)', 'P(Co-obs | Detected)'),
    ]
    # Build a long-form tidy dataframe for this plot
    rows = []
    for _, r in df.iterrows():
        for col, label in cascade_metrics:
            rows.append({
                'Algorithm':       r['Algorithm'],
                'Connectivity':    r['Connectivity'],
                'Data Processing': r['Data Processing'],
                'Stage':           label,
                'Value':           r[col],
            })
    long = pd.DataFrame(rows)
    long['Stage'] = pd.Categorical(long['Stage'],
                                    categories=[l for _, l in cascade_metrics],
                                    ordered=True)

    n_dp   = len(DP_ORDER)
    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(n_conn, n_dp,
                              figsize=(5.5 * n_dp, 4.5 * n_conn),
                              sharey=True, sharex=False)

    stage_colors = ['#4477AA', '#EE7733', '#228833']  # colorblind-safe triple

    for row_i, conn in enumerate(CONN_ORDER):
        for col_i, dp in enumerate(DP_ORDER):
            ax = axes[row_i][col_i]
            sub = long[
                (long['Connectivity'] == conn) &
                (long['Data Processing'] == dp)
            ]

            # Group by Algorithm, then cluster the 3 stages
            x_base = np.arange(len(ALGO_ORDER))
            n_stages = len(cascade_metrics)
            width = 0.22
            offsets = np.linspace(-(n_stages - 1) * width / 2,
                                   (n_stages - 1) * width / 2,
                                   n_stages)

            for s_i, (_, stage_label) in enumerate(cascade_metrics):
                vals = []
                for algo in ALGO_ORDER:
                    v = sub[(sub['Algorithm'] == algo) &
                             (sub['Stage'] == stage_label)]['Value']
                    vals.append(v.mean() if len(v) else np.nan)

                ax.bar(x_base + offsets[s_i], vals,
                       width=width, color=stage_colors[s_i],
                       label=stage_label, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

            ax.set_xticks(x_base)
            ax.set_xticklabels(ALGO_ORDER, fontsize=7, rotation=20, ha='right')
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
            ax.set_ylabel('Probability' if col_i == 0 else '')

            # Titles
            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
            if col_i == n_dp - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(
                    f'{_conn_label(conn).replace(chr(10), " ")}',
                    fontsize=9, rotation=270, labelpad=14, va='bottom',
                )

            # Legend once
            if row_i == 0 and col_i == n_dp - 1:
                ax.legend(title='Pipeline Stage', fontsize=7,
                          title_fontsize=8, loc='upper right')

    plt.suptitle('Event Detection → Observation Pipeline by Algorithm',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_cascade.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 8 — Trade-off Scatter: Reward vs Response Time
# ══════════════════════════════════════════════════════════════════════════════

def plot_tradeoff_scatter(
    df: pd.DataFrame,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
) -> None:
    """
    Scatter: x = Average Normalized Response Time to Task,
             y = Total Obtained Reward [norm].
    Algorithm = marker shape, Connectivity = marker size.
    Faceted by Data Processing (3 panels).
    All points labelled; utopia point at (0, 1).
    """
    marker_shapes = {
        'MILP':    'D',
        'DP':      's',
        'DP-GR':   '^',
        'GR':      'v',
        'DP-CBBA': 'o',
        'CBBA':    'P',
    }
    conn_sizes = {
        'GS':                 50,
        'Intraconstellation': 100,
        'Interconstellation': 180,
    }

    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(6 * len(DP_ORDER), 6),
                              sharey=True, sharex=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub = df[df['Data Processing'] == dp]

        # Faint individual trial scatter
        for algo in ALGO_ORDER:
            for conn in CONN_ORDER:
                pts = sub[(sub['Algorithm'] == algo) &
                           (sub['Connectivity'] == conn)]
                if pts.empty:
                    continue
                ax.scatter(
                    pts['Average Normalized Response Time to Task'],
                    pts['Total Obtained Reward [norm]'],
                    color=ALGO_PALETTE[algo],
                    marker=marker_shapes[algo],
                    s=conn_sizes[conn],
                    alpha=0.75, edgecolors='white', linewidths=0.5,
                    zorder=3,
                )

        # Utopia point
        ax.scatter(0, 1, marker='*', s=300, color='gold',
                   edgecolors='black', linewidths=0.8, zorder=6)
        ax.annotate('Utopia', xy=(0, 1), xytext=(0.02, 0.96),
                    fontsize=7, color='black')

        # MILP mean reference lines
        milp_sub = sub[sub['Algorithm'] == 'MILP']
        if not milp_sub.empty:
            mx = milp_sub['Average Normalized Response Time to Task'].mean()
            my = milp_sub['Total Obtained Reward [norm]'].mean()
            ax.axvline(mx, color=ALGO_PALETTE['MILP'],
                       linestyle=':', linewidth=0.9, alpha=0.5)
            ax.axhline(my, color=ALGO_PALETTE['MILP'],
                       linestyle=':', linewidth=0.9, alpha=0.5)
            ax.annotate('MILP\nmean',
                        xy=(mx, my), xytext=(mx + 0.02, my - 0.04),
                        fontsize=6.5, color=ALGO_PALETTE['MILP'])

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Avg. Normalised Response Time to Task')
        ax.set_ylabel('Total Obtained Reward (normalised)' if dp == DP_ORDER[0] else '')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')

    # Combined legend: algorithm (shape) + connectivity (size)
    algo_handles = [
        plt.scatter([], [], color=ALGO_PALETTE[a],
                    marker=marker_shapes[a], s=80, label=a)
        for a in ALGO_ORDER
    ]
    conn_handles = [
        plt.scatter([], [], color='grey',
                    marker='o', s=conn_sizes[c],
                    label=_conn_label(c).replace('\n', ' '))
        for c in CONN_ORDER
    ]
    axes[-1].legend(
        handles=algo_handles + [mpatches.Patch(visible=False)] + conn_handles,
        title='Algorithm / Connectivity',
        fontsize=7, title_fontsize=8, loc='lower right',
    )

    plt.suptitle('Trade-off: Mission Reward vs Response Time',
                 fontsize=12, y=1.01)
    fig.text(0.5, -0.02,
             '⚠ Response time includes passive nadir observations; '
             'inter-algorithm differences are attenuated.',
             ha='center', fontsize=7, color='#666666', style='italic')
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_scatter.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 9 — Communication Load
# ══════════════════════════════════════════════════════════════════════════════

def plot_communication_load(
    df: pd.DataFrame,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
) -> None:
    metrics = [
        ('Total Messages Broadcasted',        'Total Messages Broadcasted'),
        ('Average Messages Broadcasted per Task', 'Avg. Messages per Task'),
    ]
    n_dp = len(DP_ORDER)
    fig, axes = plt.subplots(2, n_dp,
                              figsize=(5 * n_dp, 9),
                              sharey='row')

    for row_i, (metric, ylabel) in enumerate(metrics):
        for col_i, dp in enumerate(DP_ORDER):
            ax = axes[row_i][col_i]
            sub = df[df['Data Processing'] == dp]

            sns.barplot(
                data=sub,
                x='Connectivity', y=metric,
                hue='Algorithm', hue_order=ALGO_ORDER,
                palette=ALGO_PALETTE,
                order=CONN_ORDER,
                errorbar='sd',
                width=0.7, ax=ax,
            )
            ax.set_xlabel('Connectivity')
            ax.set_xticks(range(len(CONN_ORDER)))
            ax.set_xticklabels(
                [_conn_label(c) for c in CONN_ORDER], fontsize=8,
            )
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')

            legend = ax.get_legend()
            if row_i == 0 and col_i == n_dp - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')
            elif legend:
                legend.remove()

    plt.suptitle('Communication Load by Algorithm', fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_communication.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 10 — Seasonal Sensitivity  (full dataset only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_seasonal_sensitivity(
    df: pd.DataFrame,
    save_dir: str, local_save_dir: str,
    base_dir: str, local_base_dir: str,
    filename_stem: str,
) -> None:
    if df['Date'].nunique() < 2:
        print('  [Plot 10] Skipped — only one date in dataset '
              '(requires full multi-date dataset).')
        return

    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(1, n_conn,
                              figsize=(6 * n_conn, 5),
                              sharey=True)

    for ax, conn in zip(axes, CONN_ORDER):
        sub = df[df['Connectivity'] == conn].copy()
        sub = sub.sort_values('Date')

        for algo in ALGO_ORDER:
            asub = sub[sub['Algorithm'] == algo]
            means = asub.groupby('Date')['Total Obtained Reward [norm]'].mean()
            stds  = asub.groupby('Date')['Total Obtained Reward [norm]'].std(ddof=0).fillna(0)

            dates = means.index.tolist()
            ys    = means.values
            es    = stds.values

            ls_spec = ALGO_LINESTYLES[algo]
            if ls_spec == '':
                linestyle = 'solid'
                dashes    = None
            else:
                linestyle = 'dashed'
                dashes    = ls_spec

            line, = ax.plot(dates, ys,
                            color=ALGO_PALETTE[algo],
                            linewidth=1.6, marker='o', markersize=5,
                            linestyle=linestyle, label=algo)
            if dashes:
                line.set_dashes(dashes)

            ax.fill_between(dates,
                            ys - es, ys + es,
                            color=ALGO_PALETTE[algo], alpha=0.10)

        ax.set_title(f'Connectivity: {_conn_label(conn).replace(chr(10), " ")}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Simulation Date')
        ax.set_ylabel('Total Obtained Reward (normalised)' if conn == CONN_ORDER[0]
                       else '')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.set_ylim(-0.02, 1.05)

        if conn == CONN_ORDER[-1]:
            ax.legend(title='Algorithm', fontsize=7,
                      title_fontsize=8, loc='best')

    plt.suptitle('Seasonal Sensitivity of Mission Reward by Algorithm',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, local_save_dir, base_dir, local_base_dir,
              f'{filename_stem}_seasonal.png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(
    csv_path: str,
    trial_name: str,
    subset: str = 'abridged',       # 'abridged' | 'full' | 'both'
) -> None:
    """
    Parameters
    ----------
    csv_path   : path to the compiled results CSV
    trial_name : used to name output directories and file stems
    subset     : which rows to include — 'abridged', 'full', or 'both'
    """
    local_base_dir = os.path.join(
        'experiments', '2_centralized_vs_decentralized', 'analysis')
    base_dir = local_base_dir

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'Compiled results not found at: {csv_path}')

    df_all = load_and_prepare(csv_path)

    date_str = datetime.now().strftime('%Y-%m-%d')

    subsets_to_run: list[tuple[str, pd.DataFrame]] = []

    if subset in ('abridged', 'both'):
        df_ab = df_all[df_all['in_abridged']].copy()
        if df_ab.empty:
            print('  [abridged] No rows found — skipping.')
        else:
            subsets_to_run.append(('abridged', df_ab))

    if subset in ('full', 'both'):
        if 'in_full' not in df_all.columns:
            print('  [full] `in_full` column not found — skipping.')
        else:
            df_full = df_all[df_all['in_full']].copy()
            if df_full.empty:
                print('  [full] No rows found — skipping.')
            else:
                subsets_to_run.append(('full', df_full))

    for tag, df in subsets_to_run:
        stem_prefix = f'{trial_name}_{tag}'
        dirname     = f'{stem_prefix}_P{date_str}'

        save_dir = os.path.join(base_dir, 'plots', 'rq', dirname)
        os.makedirs(save_dir, exist_ok=True)
        local_save_dir = save_dir  # same directory (extend if remote)

        print(f'\n{"="*60}')
        print(f'  Generating plots — subset: {tag}  ({len(df)} trials)')
        print(f'  Output → {save_dir}')
        print(f'{"="*60}')

        # ── Plot 1a — Mission Reward (absolute, log-friendly) ────────────
        print('  Plot 1a — Mission Reward (absolute)')
        plot_metric(
            df,
            metrics='Total Obtained Reward',
            titles='Mission Reward',
            ylabels='Total Obtained Reward',
            suptitle='Mission Reward by Algorithm',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot1a-Mission_Reward',
            upper_ref_col='Task Reward Dual Bound',
        )

        # ── Plot 1b — Mission Reward (normalised by absolute Dual Bound) ─
        print('  Plot 1b — Mission Reward (normalised)')
        plot_metric(
            df,
            metrics='Total Obtained Reward [norm]',
            titles='Mission Reward (normalised by Dual Bound)',
            ylabels='Total Obtained Reward (normalised)',
            suptitle='Normalised Mission Reward by Algorithm',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot1b-Mission_Reward_Norm',
            hline_at_one=True,
            lower_ref_col='Task Reward Primal Bound [norm]',
        )

        # ── Plot 1c — Mission Reward (normalised by Known Dual Bound) ────
        print('  Plot 1c — Mission Reward (known-normalised)')
        plot_metric(
            df,
            metrics='Total Obtained Reward [known_norm]',
            titles='Mission Reward (normalised by Known Dual Bound)',
            ylabels='Total Obtained Reward\n(÷ Known Task Dual Bound)',
            suptitle='Known-Normalised Mission Reward by Algorithm\n'
                     '⚠ Cross-mode comparison limited (Known Dual varies by detection mode)',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot1c-Mission_Reward_KnownNorm',
        )

        # ── Plot 2a — Algorithm Decomposition Heatmap ────────────────────
        print('  Plot 2a — Decomposition heatmap')
        plot_decomposition_heatmap(
            df,
            metric='Total Obtained Reward [norm]',
            ylabel='Reward (normalised)',
            suptitle='Algorithm Decomposition: Preplanner × Replanner\n'
                     '(mean normalised reward)',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot2a-Decomposition',
        )

        # ── Plot 2b — Algorithm Decomposition Box+Strip ───────────────────
        print('  Plot 2b — Decomposition box+strip')
        plot_decomposition_boxplots(
            df,
            metric='Total Obtained Reward [norm]',
            ylabel='Total Obtained Reward (normalised)',
            suptitle='Algorithm Decomposition by Detection Mode',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot2b-Decomposition',
            hline_at_one=True,
        )

        # ── Plot 3 — Task Observation Probability ─────────────────────────
        print('  Plot 3 — Task Observation Probability')
        plot_metric(
            df,
            metrics=['P(Task Observed)',
                     'P(Task Observed | Task Observable)'],
            titles=[r'$P(\mathrm{Task\ Observed})$',
                    r'$P(\mathrm{Task\ Observed\ |\ Observable})$'],
            ylabels=[r'$P(\mathrm{Task\ Observed})$',
                     r'$P(\mathrm{Task\ Observed\ |\ Observable})$'],
            suptitle='Task Observation Probability by Algorithm',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot3-Task_Obs_Probability',
        )

        # ── Plot 4 — Co-observation Quality ───────────────────────────────
        print('  Plot 4 — Co-observation Quality')
        plot_metric(
            df,
            metrics=['P(Event Co-observed)',
                     'P(Event Co-observed | Co-observable)'],
            titles=[r'$P(\mathrm{Event\ Co-observed})$',
                    r'$P(\mathrm{Event\ Co-observed\ |\ Co-observable})$'],
            ylabels=[r'$P(\mathrm{Event\ Co-observed})$',
                     r'$P(\mathrm{Event\ Co-observed\ |\ Co-observable})$'],
            suptitle='Multi-Agent Co-observation Quality by Algorithm',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot4-Coobs_Quality',
        )

        # ── Plot 5 — Response Time ─────────────────────────────────────────
        print('  Plot 5 — Response Time')
        plot_metric(
            df,
            metrics=['Average Normalized Response Time to Task',
                     'Average Normalized Response Time to Event'],
            titles=['Avg. Normalised Task Response Time',
                    'Avg. Normalised Event Response Time'],
            ylabels=['Avg. Normalised Response Time to Task',
                     'Avg. Normalised Response Time to Event'],
            suptitle='Task & Event Response Time by Algorithm',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot5-Response_Time',
            response_time_note=True,
        )

        # ── Plot 6 — Connectivity × Detection Interaction ─────────────────
        print('  Plot 6 — Connectivity × Detection Interaction')
        plot_interaction(
            df,
            metrics=['Total Obtained Reward [norm]',
                     'P(Task Observed | Task Observable)',
                     'P(Event Co-observed | Co-observable)'],
            ylabels=['Reward (normalised)',
                     r'$P(\mathrm{Task\ Obs\ |\ Observable})$',
                     r'$P(\mathrm{Co-obs\ |\ Co-observable})$'],
            suptitle='Effect of Detection Mode by Connectivity Level',
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot6-Interaction',
            hline_at_one=True,
        )

        # ── Plot 7 — Event Detection Pipeline Cascade ──────────────────────
        print('  Plot 7 — Detection Pipeline Cascade')
        plot_detection_cascade(
            df,
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot7-Detection_Cascade',
        )

        # ── Plot 8 — Trade-off Scatter ─────────────────────────────────────
        print('  Plot 8 — Trade-off Scatter')
        plot_tradeoff_scatter(
            df,
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot8-Tradeoff_Scatter',
        )

        # ── Plot 9 — Communication Load ────────────────────────────────────
        print('  Plot 9 — Communication Load')
        plot_communication_load(
            df,
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot9-Communication_Load',
        )

        # ── Plot 10 — Seasonal Sensitivity (full dataset only) ─────────────
        print('  Plot 10 — Seasonal Sensitivity')
        plot_seasonal_sensitivity(
            df,
            save_dir=save_dir, local_save_dir=local_save_dir,
            base_dir=base_dir, local_base_dir=local_base_dir,
            filename_stem=f'{stem_prefix}_Plot10-Seasonal',
        )

    print('\nDONE.')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate analysis plots for centralized vs decentralized experiment.')
    parser.add_argument(
        '--csv',
        default=os.path.join(
            'experiments', '2_centralized_vs_decentralized', 'analysis',
            'compiled', 'full_factorial_trials_2026-05-14_compiled_results.csv'),
        help='Path to the compiled results CSV.')
    parser.add_argument(
        '--trial-name', default='full_factorial_trials_2026-05-14',
        help='Trial name stem used in output filenames.')
    parser.add_argument(
        '--abridged', action='store_true',
        help='Generate plots for the abridged (Feb-only) subset.')
    parser.add_argument(
        '--full', action='store_true',
        help='Generate plots for the full dataset.')
    args = parser.parse_args()

    # Default: abridged only (mirrors previous experiment behaviour)
    if not args.abridged and not args.full:
        args.abridged = True

    subset = ('both'     if args.abridged and args.full else
              'abridged' if args.abridged else
              'full')

    generate_plots(
        csv_path=args.csv,
        trial_name=args.trial_name,
        subset=subset,
    )