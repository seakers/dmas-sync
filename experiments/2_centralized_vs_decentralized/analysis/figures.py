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

Notes (2026-05-17 dataset)
--------------------------
- Data Processing label is now 'Instant' (was 'Oracle') in the raw CSV.
- P(Event Announced Before Expiry) not yet in dataset; Plot 7 stage activates
  automatically once the column appears.
- None-None Primal Bound embedded in Task Reward Primal Bound [norm] at
  compile time — no separate lookup required.
- Known Task Reward bounds excluded (values unreliable).
- Response time now measured from announcement to observation (passive bias
  removed); annotation updated accordingly.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    'font.size':             10,
    'axes.titlesize':        11,
    'axes.labelsize':        10,
    'legend.fontsize':        8,
    'legend.title_fontsize':  9,
})


# =============================================================================
#  CONSTANTS
# =============================================================================

ALGO_ORDER: list[str] = [
    'None-None', 'MILP', 'DP', 'DP-GR', 'GR', 'DP-CBBA', 'CBBA',
]

ALGO_PALETTE: dict[str, str] = {
    'None-None': '#BBBBBB',
    'MILP':      '#CC79A7',
    'DP':        '#999999',
    'DP-GR':     '#E69F00',
    'GR':        '#F0E442',
    'DP-CBBA':   '#0072B2',
    'CBBA':      '#56B4E9',
}

ALGO_LINESTYLES: dict[str, tuple | str] = {
    'None-None': (1, 3),
    'MILP':      (2, 2),
    'DP':        (4, 2),
    'DP-GR':     (4, 1, 1, 1),
    'GR':        (1, 2),
    'DP-CBBA':   '',
    'CBBA':      (3, 1),
}

CONN_ORDER: list[str] = ['GS', 'Intraconstellation', 'Interconstellation']
CONN_LABELS: dict[str, str] = {
    'GS':                 'GS Only',
    'Intraconstellation': 'Intra-\nconstellation',
    'Interconstellation': 'Inter-\nconstellation',
}

# Legacy -> Instant -> Onboard
DP_ORDER: list[str] = ['Ground', 'Instant', 'Onboard']
DP_LABELS: dict[str, str] = {
    'Ground':  'Legacy Ground\nDetection',
    'Instant': 'Instant Ground\nDetection',
    'Onboard': 'Onboard\nDetection',
}
DP_LABELS_SHORT: dict[str, str] = {
    'Ground':  'Legacy Ground',
    'Instant': 'Instant Ground',
    'Onboard': 'Onboard',
}

METRIC_LABELS: dict[str, str] = {
    'Total Obtained Reward [norm]':
        'Total Obtained Reward (normalised)',
    'P(Task Observed)':
        'P(Task Observed)',
    'P(Task Observed | Task Observable)':
        'P(Task Obs | Observable)',
    'P(Event Observed | Event Detected)':
        'P(Event Obs | Detected)',
    'P(Event Observed | Event Observable and Detected)':
        'P(Event Obs | Obs.&Det.)',
    'P(Event Co-observed)':
        'P(Event Co-obs)',
    'P(Event Co-observed | Co-observable)':
        'P(Event Co-obs | Co-obs.)',
    'P(Event Detected)':
        'P(Event Detected)',
    'P(Event Announced)':
        'P(Event Announced)',
    'P(Event Announced Before Expiry)':
        'P(Ann. Before Expiry)',
    'Average Normalized Earliest Response Time to Event':
        'Avg. Norm. Earliest Response Time (Event)',
    'Average Normalized Latest Response Time to Event':
        'Avg. Norm. Latest Response Time (Event)',
    'Average Event Announcement Time [norm]':
        'Avg. Norm. Time to Announcement',
    'Average Normalized Earliest Time to Image Event':
        'Avg. Norm. Earliest Time to Image',
    'Average Normalized Latest Time to Image Event':
        'Avg. Norm. Latest Time to Image',
    'Average Observations per Event':
        'Avg. Observations per Event',
    'Average Observations per Task':
        'Avg. Observations per Task',
}


# =============================================================================
#  DATA LOADING
# =============================================================================

def label_algorithm(row: pd.Series) -> str:
    pre, rep = row['Preplanner'], row['Replanner']
    if pre == 'Centralized-MILP_priority': return 'MILP'
    if pre == 'DP'   and rep == 'None':    return 'DP'
    if pre == 'DP'   and rep == 'CBBA':    return 'DP-CBBA'
    if pre == 'DP'   and rep == 'Greedy':  return 'DP-GR'
    if pre == 'None' and rep == 'CBBA':    return 'CBBA'
    if pre == 'None' and rep == 'Greedy':  return 'GR'
    if pre == 'None' and rep == 'None':    return 'None-None'
    return 'Unknown'


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df['Preplanner'] = df['Preplanner'].fillna('None')
    df['Replanner']  = df['Replanner'].fillna('None')
    df['Algorithm']  = df.apply(label_algorithm, axis=1)
    df['Algorithm']  = pd.Categorical(df['Algorithm'],
                                       categories=ALGO_ORDER, ordered=True)
    df['Connectivity'] = pd.Categorical(df['Connectivity'],
                                         categories=CONN_ORDER, ordered=True)
    df['Data Processing'] = pd.Categorical(df['Data Processing'],
                                            categories=DP_ORDER, ordered=True)
    return df.sort_values(['Algorithm', 'Connectivity', 'Data Processing'])


# =============================================================================
#  UTILITIES
# =============================================================================

def save_plot(save_dir: str, filename: str) -> None:
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved -> {path}')


def _dp_label(dp: str, short: bool = False) -> str:
    return (DP_LABELS_SHORT if short else DP_LABELS).get(dp, dp)


def _conn_label(conn: str) -> str:
    return CONN_LABELS.get(conn, conn)


def _col_ok(df: pd.DataFrame, col: str, ctx: str) -> bool:
    if col not in df.columns:
        print(f'  [{ctx}] Column not found (will activate when added): {col}')
        return False
    return True


def _apply_linestyle(line, algo: str) -> None:
    spec = ALGO_LINESTYLES.get(algo, '')
    if spec != '':
        line.set_dashes(spec)


# =============================================================================
#  BOX + STRIP  (outline-only, red median, algorithm-coloured edges)
# =============================================================================

def make_boxplot(data: pd.DataFrame, metric: str, ax: plt.Axes,
                 ylabel: str | None = None) -> None:
    present = [a for a in ALGO_ORDER if a in data['Algorithm'].values]

    sns.boxplot(
        data=data, x='Algorithm', y=metric,
        order=present, palette=ALGO_PALETTE,
        hue='Algorithm', hue_order=present,
        width=0.5, fliersize=0, legend=False, ax=ax,
        boxprops=dict(facecolor='none', linewidth=1.2),
        medianprops=dict(color='red', linewidth=1.8),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
    )
    for patch, algo in zip(ax.patches, present):
        patch.set_edgecolor(ALGO_PALETTE.get(algo, '#333333'))
        patch.set_facecolor('none')
        patch.set_linewidth(1.4)
        if algo == 'None-None':
            patch.set_hatch('//')

    sns.stripplot(
        data=data, x='Algorithm', y=metric,
        order=present, palette=ALGO_PALETTE,
        hue='Algorithm', hue_order=present,
        size=5, alpha=0.55, jitter=True, dodge=False,
        legend=False, ax=ax, marker='o',
        edgecolor='grey', linewidth=0.4,
    )
    ax.set_xlabel('Algorithm Configuration')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)


# =============================================================================
#  FIGURE A (box+strip) + FIGURE B (grouped bars, x=Connectivity, col=DP)
# =============================================================================

def plot_metric(
    df: pd.DataFrame,
    metrics: str | list[str],
    ylabels: str | list[str],
    suptitle: str,
    save_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
    primal_ref: bool = False,
    response_time_note: bool = False,
) -> None:
    if isinstance(metrics, str):
        metrics = [metrics]
        ylabels = [ylabels]
    n = len(metrics)

    # Figure A
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, metric, ylabel) in enumerate(zip(axes, metrics, ylabels)):
        if not _col_ok(df, metric, f'{filename_stem} FigA'):
            ax.set_visible(False)
            continue
        make_boxplot(df, metric, ax, ylabel=ylabel)
        if hline_at_one:
            ax.axhline(1.0, color='#444444', linestyle='--',
                       linewidth=0.9, alpha=0.6)
        if df[metric].dropna().max() <= 1.05:
            ax.set_ylim(-0.02, 1.05)
        if n > 1:
            ax.set_title(f'({chr(ord("a") + i)})', loc='left')
            if i < n - 1:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

    if response_time_note:
        axes[-1].annotate(
            'Response time measured from announcement to observation.',
            xy=(0.01, 0.01), xycoords='axes fraction',
            fontsize=7, color='#555555', style='italic')

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_by_algorithm.png')
    plt.close()

    # Figure B
    n_dp = len(DP_ORDER)
    fig, axes = plt.subplots(n, n_dp, figsize=(5 * n_dp, 5 * n), sharey='row')
    if n == 1:
        axes = np.array([axes])

    for row_i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        if not _col_ok(df, metric, f'{filename_stem} FigB'):
            for ax in axes[row_i]:
                ax.set_visible(False)
            continue

        for col_i, dp in enumerate(DP_ORDER):
            ax = axes[row_i][col_i]
            sub = df[df['Data Processing'] == dp]
            present = [a for a in ALGO_ORDER if a in sub['Algorithm'].values]

            sns.barplot(
                data=sub, x='Connectivity', y=metric,
                hue='Algorithm', hue_order=present,
                palette=ALGO_PALETTE, order=CONN_ORDER,
                errorbar='sd', width=0.7, ax=ax,
            )

            # None-None primal reference lines per connectivity tick
            if primal_ref and 'Task Reward Primal Bound [norm]' in df.columns:
                nn = df[(df['Algorithm'] == 'None-None') &
                        (df['Data Processing'] == dp)]
                for xi, conn in enumerate(CONN_ORDER):
                    val = nn[nn['Connectivity'] == conn][
                        'Task Reward Primal Bound [norm]'].mean()
                    if pd.notna(val):
                        ax.plot([xi - 0.35, xi + 0.35], [val, val],
                                color='#555555', linewidth=1.0,
                                linestyle=':', alpha=0.7)

            if hline_at_one:
                for xi in range(len(CONN_ORDER)):
                    ax.plot([xi - 0.35, xi + 0.35], [1.0, 1.0],
                            color='black', linewidth=0.8,
                            linestyle='--', alpha=0.4)

            ax.set_xticks(range(len(CONN_ORDER)))
            ax.set_xticklabels([_conn_label(c) for c in CONN_ORDER], fontsize=8)
            ax.set_xlabel('Connectivity')
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
            if sub[metric].dropna().max() <= 1.05:
                ax.set_ylim(0.0, 1.05)
            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')

            legend = ax.get_legend()
            if row_i == 0 and col_i == n_dp - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')
            elif legend:
                legend.remove()

    plt.suptitle(f'{suptitle} -- by Connectivity & Detection Mode',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_grouped.png')
    plt.close()


# =============================================================================
#  PLOT 2a -- Decomposition Heatmap
# =============================================================================

def plot_decomposition_heatmap(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str,
    filename_stem: str,
) -> None:
    row_labels = ['MILP (Centralized)', 'DP (Onboard Pre)', 'None (No Pre)']
    col_labels = ['No Replanner', 'Greedy', 'CBBA']

    algo_cell = {
        'MILP':      (0, 0),
        'DP':        (1, 0),
        'DP-GR':     (1, 1),
        'DP-CBBA':   (1, 2),
        'None-None': (2, 0),
        'GR':        (2, 1),
        'CBBA':      (2, 2),
    }

    vmin, vmax = df[metric].min(), df[metric].max()
    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(5.5 * len(DP_ORDER), 5),
                              sharey=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub  = df[df['Data Processing'] == dp]
        grid = np.full((3, 3), np.nan)
        text = [[''] * 3 for _ in range(3)]

        for algo, (ri, ci) in algo_cell.items():
            vals = sub[sub['Algorithm'] == algo][metric].dropna()
            if len(vals):
                m, s = vals.mean(), vals.std(ddof=0)
                grid[ri, ci] = m
                text[ri][ci] = f'{m:.3f}\n+/-{s:.3f}'

        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap='YlOrRd',
                       vmin=vmin, vmax=vmax, aspect='auto')

        for ri in range(3):
            for ci in range(3):
                if text[ri][ci]:
                    brightness = (masked[ri, ci] - vmin) / max(vmax - vmin, 1e-9)
                    fc = 'white' if brightness > 0.6 else 'black'
                    ax.text(ci, ri, text[ri][ci],
                            ha='center', va='center', fontsize=8, color=fc)

        # Hatch None-None cell
        nn_r, nn_c = algo_cell['None-None']
        ax.add_patch(plt.Rectangle(
            (nn_c - 0.5, nn_r - 0.5), 1, 1,
            fill=False, hatch='//', edgecolor='#555555', linewidth=0.5))
        ax.text(nn_c, nn_r + 0.35, 'Passive Ref',
                ha='center', va='center', fontsize=6,
                color='#555555', style='italic')

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
        ax.set_xlabel('Replanner', fontsize=9)
        if dp == DP_ORDER[0]:
            ax.set_ylabel('Preplanner', fontsize=9)
        ax.axhline(0.5, color='white', linewidth=3)
        plt.colorbar(im, ax=ax, shrink=0.8, label=ylabel)

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_heatmap.png')
    plt.close()


# =============================================================================
#  PLOT 2b -- Decomposition Box+Strip by Data Processing
# =============================================================================

def plot_decomposition_boxplots(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(5 * len(DP_ORDER), 5),
                              sharey=True)
    for ax, dp in zip(axes, DP_ORDER):
        sub = df[df['Data Processing'] == dp]
        make_boxplot(sub, metric, ax,
                     ylabel=ylabel if dp == DP_ORDER[0] else '')
        if hline_at_one:
            ax.axhline(1.0, color='#444444', linestyle='--',
                       linewidth=0.9, alpha=0.6)
        if sub[metric].dropna().max() <= 1.05:
            ax.set_ylim(-0.02, 1.05)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
        ax.set_xlabel('')

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_boxplots.png')
    plt.close()


# =============================================================================
#  PLOT 4b -- Observations per Event/Task  (mean bar, std whisker, median dot)
# =============================================================================

def plot_obs_distribution(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    median_col: str,
    suptitle: str,
    ylabel: str,
    save_dir: str,
    filename_stem: str,
) -> None:
    n_dp = len(DP_ORDER)
    fig, axes = plt.subplots(1, n_dp, figsize=(5 * n_dp, 5), sharey=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub    = df[df['Data Processing'] == dp]
        algos  = [a for a in ALGO_ORDER if a in sub['Algorithm'].values]
        xs     = np.arange(len(algos))
        means  = [sub[sub['Algorithm'] == a][mean_col].mean()   for a in algos]
        stds   = [sub[sub['Algorithm'] == a][std_col].mean()    for a in algos]
        medians= [sub[sub['Algorithm'] == a][median_col].mean() for a in algos]

        for i, (algo, m, s, med) in enumerate(zip(algos, means, stds, medians)):
            color = ALGO_PALETTE.get(algo, '#999999')
            hatch = '//' if algo == 'None-None' else None
            ax.bar(i, m, color=color, alpha=0.75, width=0.6,
                   edgecolor='white', linewidth=0.5, hatch=hatch)
            ax.errorbar(i, m, yerr=s, fmt='none',
                        ecolor='#333333', capsize=4, linewidth=1.0)
            ax.plot(i, med, marker='D', color='red', markersize=5,
                    zorder=5, label='Median' if i == 0 else '')

        ax.set_xticks(xs)
        ax.set_xticklabels(algos, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel if dp == DP_ORDER[0] else '')
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7)

    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}.png')
    plt.close()


# =============================================================================
#  PLOT 5b -- Event Latency Budget (4-component stacked bar)
# =============================================================================

def plot_latency_budget(
    df: pd.DataFrame,
    save_dir: str,
    filename_stem: str,
) -> None:
    """
    Stacked bars normalised 0-1 by event duration.
    Components:
      Detection   = Announcement Time / Event Duration
      Response    = (Earliest Time to Image - Announcement Time) / Duration
      Observation = (Latest - Earliest Time to Image) / Duration
      Slack       = remainder to 1.0

    None-None rows have no Response component (NaN announcement->obs gap).
    """
    ann_col   = 'Average Event Announcement Time [norm]'
    early_col = 'Average Normalized Earliest Time to Image Event'
    late_col  = 'Average Normalized Latest Time to Image Event'

    for c in [ann_col, early_col, late_col]:
        if not _col_ok(df, c, 'Plot 5b'):
            return

    STAGE_COLORS = {
        'Detection':   '#E69F00',
        'Response':    '#56B4E9',
        'Observation': '#009E73',
        'Slack':       '#DDDDDD',
    }
    STAGES = ['Detection', 'Response', 'Observation', 'Slack']

    n_conn = len(CONN_ORDER)
    n_dp   = len(DP_ORDER)
    fig, axes = plt.subplots(n_conn, n_dp,
                              figsize=(5 * n_dp, 4 * n_conn),
                              sharey=True)

    for row_i, conn in enumerate(CONN_ORDER):
        for col_i, dp in enumerate(DP_ORDER):
            ax = axes[row_i][col_i]
            sub   = df[(df['Connectivity'] == conn) &
                       (df['Data Processing'] == dp)]
            algos = [a for a in ALGO_ORDER if a in sub['Algorithm'].values]
            xs    = np.arange(len(algos))
            bottoms = np.zeros(len(algos))

            for stage in STAGES:
                heights = []
                for algo in algos:
                    row = sub[sub['Algorithm'] == algo]
                    if row.empty:
                        heights.append(0.0)
                        continue
                    ann   = row[ann_col].mean()
                    early = row[early_col].mean()
                    late  = row[late_col].mean()

                    if stage == 'Detection':
                        h = ann if pd.notna(ann) else 0.0
                    elif stage == 'Response':
                        # Gap between announcement and first image
                        h = max(early - ann, 0.0) if (
                            pd.notna(early) and pd.notna(ann) and
                            algo != 'None-None') else 0.0
                    elif stage == 'Observation':
                        h = max(late - early, 0.0) if (
                            pd.notna(late) and pd.notna(early)) else 0.0
                    else:  # Slack
                        ann_v   = ann   if pd.notna(ann)   else 0.0
                        resp_v  = max(early - ann_v, 0.0) if (
                            pd.notna(early) and algo != 'None-None') else 0.0
                        obs_v   = max(late - early, 0.0)  if (
                            pd.notna(late) and pd.notna(early)) else 0.0
                        h = max(1.0 - ann_v - resp_v - obs_v, 0.0)

                    heights.append(h)

                ax.bar(xs, heights, bottom=bottoms,
                       color=STAGE_COLORS[stage],
                       label=stage if (row_i == 0 and col_i == 0) else '',
                       edgecolor='white', linewidth=0.4, alpha=0.85)
                bottoms = bottoms + np.array(heights)

            # Algorithm-coloured outline per bar
            for xi, algo in enumerate(algos):
                ax.bar(xi, 1.0, color='none',
                       edgecolor=ALGO_PALETTE.get(algo, '#333333'),
                       linewidth=1.2)

            ax.set_xticks(xs)
            ax.set_xticklabels(algos, rotation=25, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color='#444444', linestyle='--',
                       linewidth=0.7, alpha=0.5)
            ax.set_ylabel('Fraction of Event Duration' if col_i == 0 else '')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
            if col_i == n_dp - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(
                    _conn_label(conn).replace('\n', ' '),
                    fontsize=9, rotation=270, labelpad=14, va='bottom')

    legend_patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s)
                      for s in STAGES]
    fig.legend(handles=legend_patches, title='Event Window Section',
               loc='lower center', ncol=4, fontsize=8,
               title_fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle('Event Latency Budget (normalised by event duration)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_latency_budget.png')
    plt.close()


# =============================================================================
#  PLOT 6 -- Connectivity x Detection Interaction
# =============================================================================

def plot_interaction(
    df: pd.DataFrame,
    metrics: list[str],
    ylabels: list[str],
    suptitle: str,
    save_dir: str,
    filename_stem: str,
    hline_at_one: bool = False,
) -> None:
    n_metrics = len(metrics)
    n_conn    = len(CONN_ORDER)
    dp_pos    = {dp: i for i, dp in enumerate(DP_ORDER)}

    fig, axes = plt.subplots(n_metrics, n_conn,
                              figsize=(5 * n_conn, 4.5 * n_metrics),
                              sharey='row', sharex=True)
    if n_metrics == 1:
        axes = np.array([axes])

    for row_i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        if not _col_ok(df, metric, f'{filename_stem} interaction'):
            for ax in axes[row_i]:
                ax.set_visible(False)
            continue

        for col_i, conn in enumerate(CONN_ORDER):
            ax  = axes[row_i][col_i]
            sub = df[df['Connectivity'] == conn]

            for algo in ALGO_ORDER:
                asub  = sub[sub['Algorithm'] == algo]
                means = asub.groupby('Data Processing')[metric].mean()
                stds  = asub.groupby('Data Processing')[metric].std(ddof=0).fillna(0)
                xs    = [dp_pos[dp] for dp in DP_ORDER if dp in means.index]
                ys    = [means[dp]  for dp in DP_ORDER if dp in means.index]
                es    = [stds[dp]   for dp in DP_ORDER if dp in means.index]
                if not xs:
                    continue

                spec  = ALGO_LINESTYLES[algo]
                line, = ax.plot(xs, ys,
                                color=ALGO_PALETTE[algo], linewidth=1.6,
                                marker='o', markersize=5,
                                linestyle='solid' if spec == '' else 'dashed',
                                label=algo)
                _apply_linestyle(line, algo)
                ax.fill_between(xs,
                                [y - e for y, e in zip(ys, es)],
                                [y + e for y, e in zip(ys, es)],
                                color=ALGO_PALETTE[algo], alpha=0.10)

            if hline_at_one:
                ax.axhline(1.0, color='#444444', linestyle='--',
                           linewidth=0.8, alpha=0.5)

            ax.set_xticks(list(dp_pos.values()))
            ax.set_xticklabels([_dp_label(dp, short=True) for dp in DP_ORDER],
                               fontsize=8, rotation=12, ha='right')
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            sub_vals = sub[metric].dropna()
            if len(sub_vals) and sub_vals.max() <= 1.05:
                ax.set_ylim(-0.02, 1.05)

            if row_i == 0:
                ax.set_title(
                    f'Connectivity: {_conn_label(conn).replace(chr(10), " ")}',
                    fontsize=10, fontweight='bold')
            if row_i == 0 and col_i == n_conn - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')

    plt.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_interaction.png')
    plt.close()


# =============================================================================
#  PLOT 7 -- Event Detection Pipeline Cascade (up to 5 stages)
# =============================================================================

def plot_detection_cascade(
    df: pd.DataFrame,
    save_dir: str,
    filename_stem: str,
) -> None:
    cascade_defs = [
        ('P(Event Detected)',                     'P(Detected)'),
        ('P(Event Announced)',                    'P(Announced)'),
        ('P(Event Announced Before Expiry)',      'P(Ann. Before\nExpiry)'),
        ('P(Event Observed | Event Detected)',    'P(Obs|Det)'),
        ('P(Event Co-observed | Event Detected)', 'P(Co-obs|Det)'),
    ]
    cascade = [(col, lbl) for col, lbl in cascade_defs if col in df.columns]
    skipped = [col for col, _ in cascade_defs if col not in df.columns]
    if skipped:
        print(f'  [Plot 7] Stages pending (column absent): {skipped}')
    if not cascade:
        print('  [Plot 7] No cascade columns found -- skipping.')
        return

    n_stages = len(cascade)
    STAGE_COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#009E73', '#56B4E9'][:n_stages]

    rows = []
    for _, r in df.iterrows():
        for col, lbl in cascade:
            rows.append({
                'Algorithm':       r['Algorithm'],
                'Connectivity':    r['Connectivity'],
                'Data Processing': r['Data Processing'],
                'Stage':           lbl,
                'Value':           r[col],
            })
    long = pd.DataFrame(rows)
    long['Stage'] = pd.Categorical(long['Stage'],
                                    categories=[l for _, l in cascade],
                                    ordered=True)

    n_dp, n_conn = len(DP_ORDER), len(CONN_ORDER)
    fig, axes = plt.subplots(n_conn, n_dp,
                              figsize=(5.5 * n_dp, 4.5 * n_conn),
                              sharey=True)

    width   = 0.80 / n_stages
    x_base  = np.arange(len(ALGO_ORDER))
    offsets = np.linspace(-(n_stages - 1) * width / 2,
                           (n_stages - 1) * width / 2,
                           n_stages)

    for row_i, conn in enumerate(CONN_ORDER):
        for col_i, dp in enumerate(DP_ORDER):
            ax  = axes[row_i][col_i]
            sub = long[(long['Connectivity'] == conn) &
                       (long['Data Processing'] == dp)]

            for s_i, (_, stage_lbl) in enumerate(cascade):
                vals = [float(sub[(sub['Algorithm'] == a) &
                                   (sub['Stage'] == stage_lbl)]['Value'].mean())
                        if a in sub['Algorithm'].values else np.nan
                        for a in ALGO_ORDER]
                ax.bar(x_base + offsets[s_i], vals, width=width,
                       color=STAGE_COLORS[s_i],
                       label=stage_lbl if (row_i == 0 and col_i == 0) else '',
                       alpha=0.85, edgecolor='white', linewidth=0.4)

            ax.set_xticks(x_base)
            ax.set_xticklabels(ALGO_ORDER, fontsize=6.5,
                               rotation=25, ha='right')
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
            ax.set_ylabel('Probability' if col_i == 0 else '')
            if row_i == 0:
                ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')
            if col_i == n_dp - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(_conn_label(conn).replace('\n', ' '),
                              fontsize=9, rotation=270,
                              labelpad=14, va='bottom')

    handles = [mpatches.Patch(color=STAGE_COLORS[i], label=lbl)
               for i, (_, lbl) in enumerate(cascade)]
    fig.legend(handles=handles, title='Pipeline Stage',
               loc='lower center', ncol=n_stages,
               fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.suptitle('Event Detection -> Observation Pipeline by Algorithm',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_cascade.png')
    plt.close()


# =============================================================================
#  PLOT 8 -- Trade-off Scatter
# =============================================================================

def plot_tradeoff_scatter(
    df: pd.DataFrame,
    save_dir: str,
    filename_stem: str,
) -> None:
    x_col = 'Average Normalized Earliest Response Time to Event'
    y_col = 'Total Obtained Reward [norm]'
    if not _col_ok(df, x_col, 'Plot 8'):
        return

    marker_shapes = {
        'None-None': 'X', 'MILP': 'D', 'DP': 's',
        'DP-GR': '^', 'GR': 'v', 'DP-CBBA': 'o', 'CBBA': 'P',
    }
    conn_sizes = {'GS': 50, 'Intraconstellation': 100, 'Interconstellation': 180}

    fig, axes = plt.subplots(1, len(DP_ORDER),
                              figsize=(6 * len(DP_ORDER), 6),
                              sharey=True, sharex=True)

    for ax, dp in zip(axes, DP_ORDER):
        sub = df[df['Data Processing'] == dp]

        for algo in ALGO_ORDER:
            for conn in CONN_ORDER:
                pts = sub[(sub['Algorithm'] == algo) &
                           (sub['Connectivity'] == conn)].dropna(subset=[x_col])
                if pts.empty:
                    continue
                ax.scatter(pts[x_col], pts[y_col],
                           color=ALGO_PALETTE[algo],
                           marker=marker_shapes[algo],
                           s=conn_sizes[conn],
                           alpha=0.5 if algo == 'None-None' else 0.80,
                           edgecolors='white', linewidths=0.5, zorder=3)

        ax.scatter(0, 1, marker='*', s=300, color='gold',
                   edgecolors='black', linewidths=0.8, zorder=6)
        ax.annotate('Utopia', xy=(0, 1), xytext=(0.02, 0.96),
                    fontsize=7, color='black')

        milp_sub = sub[sub['Algorithm'] == 'MILP'].dropna(subset=[x_col])
        if not milp_sub.empty:
            mx, my = milp_sub[x_col].mean(), milp_sub[y_col].mean()
            ax.axvline(mx, color=ALGO_PALETTE['MILP'],
                       linestyle=':', linewidth=0.9, alpha=0.5)
            ax.axhline(my, color=ALGO_PALETTE['MILP'],
                       linestyle=':', linewidth=0.9, alpha=0.5)
            ax.annotate('MILP\nmean', xy=(mx, my),
                        xytext=(mx + 0.02, my - 0.04),
                        fontsize=6.5, color=ALGO_PALETTE['MILP'])

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Avg. Norm. Earliest Response Time (Event)')
        ax.set_ylabel('Total Obtained Reward (normalised)'
                       if dp == DP_ORDER[0] else '')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.set_title(_dp_label(dp), fontsize=10, fontweight='bold')

    algo_handles = [plt.scatter([], [], color=ALGO_PALETTE[a],
                                marker=marker_shapes[a], s=80, label=a)
                    for a in ALGO_ORDER]
    conn_handles = [plt.scatter([], [], color='grey', marker='o',
                                s=conn_sizes[c],
                                label=_conn_label(c).replace('\n', ' '))
                    for c in CONN_ORDER]
    axes[-1].legend(
        handles=algo_handles + [mpatches.Patch(visible=False)] + conn_handles,
        title='Algorithm / Connectivity', fontsize=7,
        title_fontsize=8, loc='lower right')

    plt.suptitle('Trade-off: Mission Reward vs Earliest Response Time',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_scatter.png')
    plt.close()


# =============================================================================
#  PLOT 9 -- Communication Load
# =============================================================================

def plot_communication_load(
    df: pd.DataFrame,
    save_dir: str,
    filename_stem: str,
) -> None:
    metrics = [
        ('Total Messages Broadcasted',            'Total Messages Broadcasted'),
        ('Average Messages Broadcasted per Task',  'Avg. Messages per Task'),
    ]
    n_dp = len(DP_ORDER)
    fig, axes = plt.subplots(2, n_dp, figsize=(5 * n_dp, 9), sharey='row')

    for row_i, (metric, ylabel) in enumerate(metrics):
        if not _col_ok(df, metric, 'Plot 9'):
            for ax in axes[row_i]:
                ax.set_visible(False)
            continue
        for col_i, dp in enumerate(DP_ORDER):
            ax  = axes[row_i][col_i]
            sub = df[df['Data Processing'] == dp]
            present = [a for a in ALGO_ORDER if a in sub['Algorithm'].values]

            sns.barplot(data=sub, x='Connectivity', y=metric,
                        hue='Algorithm', hue_order=present,
                        palette=ALGO_PALETTE, order=CONN_ORDER,
                        errorbar='sd', width=0.7, ax=ax)
            ax.set_xlabel('Connectivity')
            ax.set_xticks(range(len(CONN_ORDER)))
            ax.set_xticklabels([_conn_label(c) for c in CONN_ORDER], fontsize=8)
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
    save_plot(save_dir, f'{filename_stem}_communication.png')
    plt.close()


# =============================================================================
#  PLOT 10 -- Seasonal Sensitivity (full dataset only)
# =============================================================================

def plot_seasonal_sensitivity(
    df: pd.DataFrame,
    save_dir: str,
    filename_stem: str,
) -> None:
    if df['Date'].nunique() < 2:
        print('  [Plot 10] Skipped -- requires full multi-date dataset.')
        return

    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(1, n_conn, figsize=(6 * n_conn, 5), sharey=True)

    for ax, conn in zip(axes, CONN_ORDER):
        sub = df[df['Connectivity'] == conn].sort_values('Date')
        for algo in ALGO_ORDER:
            asub  = sub[sub['Algorithm'] == algo]
            means = asub.groupby('Date')['Total Obtained Reward [norm]'].mean()
            stds  = asub.groupby('Date')['Total Obtained Reward [norm]'].std(
                ddof=0).fillna(0)
            dates, ys, es = means.index.tolist(), means.values, stds.values
            spec  = ALGO_LINESTYLES[algo]
            line, = ax.plot(dates, ys, color=ALGO_PALETTE[algo],
                            linewidth=1.6, marker='o', markersize=5,
                            linestyle='solid' if spec == '' else 'dashed',
                            label=algo)
            _apply_linestyle(line, algo)
            ax.fill_between(dates, ys - es, ys + es,
                            color=ALGO_PALETTE[algo], alpha=0.10)

        ax.set_title(f'Connectivity: {_conn_label(conn).replace(chr(10), " ")}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Simulation Date')
        ax.set_ylabel('Total Obtained Reward (normalised)'
                       if conn == CONN_ORDER[0] else '')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.set_ylim(-0.02, 1.05)
        if conn == CONN_ORDER[-1]:
            ax.legend(title='Algorithm', fontsize=7,
                      title_fontsize=8, loc='best')

    plt.suptitle('Seasonal Sensitivity of Mission Reward by Algorithm',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, f'{filename_stem}_seasonal.png')
    plt.close()


# =============================================================================
#  MAIN DRIVER
# =============================================================================

def generate_plots(csv_path: str, trial_name: str,
                   subset: str = 'abridged') -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    base_dir = os.path.join(
        'experiments', '2_centralized_vs_decentralized', 'analysis')
    df_all   = load_and_prepare(csv_path)
    date_str = datetime.now().strftime('%Y-%m-%d')

    subsets_to_run: list[tuple[str, pd.DataFrame]] = []
    if subset in ('abridged', 'both'):
        df_ab = df_all[df_all['in_abridged']].copy()
        subsets_to_run.append(('abridged', df_ab)) if not df_ab.empty \
            else print('  [abridged] No rows -- skipping.')
    if subset in ('full', 'both'):
        if 'in_full' not in df_all.columns:
            print('  [full] `in_full` column absent -- skipping.')
        else:
            df_f = df_all[df_all['in_full']].copy()
            subsets_to_run.append(('full', df_f)) if not df_f.empty \
                else print('  [full] No rows -- skipping.')

    for tag, df in subsets_to_run:
        stem_prefix = f'{trial_name}_{tag}'
        save_dir = os.path.join(base_dir, 'plots', 'rq',
                                f'{stem_prefix}_P{date_str}')
        os.makedirs(save_dir, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'  Subset: {tag}  ({len(df)} trials)')
        print(f'  Output -> {save_dir}')
        print(f'{"="*60}')

        print('  Plot 1a -- Mission Reward (absolute)')
        plot_metric(df, 'Total Obtained Reward', 'Total Obtained Reward',
                    'Mission Reward by Algorithm', save_dir,
                    f'{stem_prefix}_Plot1a-Mission_Reward')

        print('  Plot 1b -- Mission Reward (normalised)')
        plot_metric(df, 'Total Obtained Reward [norm]',
                    'Total Obtained Reward (normalised)',
                    'Normalised Mission Reward by Algorithm', save_dir,
                    f'{stem_prefix}_Plot1b-Mission_Reward_Norm',
                    hline_at_one=True, primal_ref=True)

        print('  Plot 2a -- Decomposition heatmap')
        plot_decomposition_heatmap(
            df, metric='Total Obtained Reward [norm]',
            ylabel='Reward (normalised)',
            suptitle='Algorithm Decomposition: Preplanner x Replanner',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot2a-Decomposition')

        print('  Plot 2b -- Decomposition box+strip')
        plot_decomposition_boxplots(
            df, metric='Total Obtained Reward [norm]',
            ylabel='Total Obtained Reward (normalised)',
            suptitle='Algorithm Decomposition by Detection Mode',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot2b-Decomposition',
            hline_at_one=True)

        print('  Plot 3 -- Observation Probability (4 panels)')
        plot_metric(
            df,
            metrics=[
                'P(Task Observed)',
                'P(Task Observed | Task Observable)',
                'P(Event Observed | Event Detected)',
                'P(Event Observed | Event Observable and Detected)',
            ],
            ylabels=[
                'P(Task Observed)',
                'P(Task Obs | Observable)',
                'P(Event Obs | Detected)',
                'P(Event Obs | Obs.&Det.)',
            ],
            suptitle='Observation Probability by Algorithm',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot3-Obs_Probability')

        print('  Plot 4a -- Co-observation Quality')
        plot_metric(
            df,
            metrics=['P(Event Co-observed)',
                     'P(Event Co-observed | Co-observable)'],
            ylabels=['P(Event Co-obs)',
                     'P(Event Co-obs | Co-obs.)'],
            suptitle='Co-observation Quality by Algorithm',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot4a-Coobs_Quality')

        print('  Plot 4b -- Observations per Event')
        if all(c in df.columns for c in [
                'Average Observations per Event',
                'Standard Deviation of Observations per Event',
                'Median Observations per Event']):
            plot_obs_distribution(
                df,
                mean_col='Average Observations per Event',
                std_col='Standard Deviation of Observations per Event',
                median_col='Median Observations per Event',
                suptitle='Observations per Event by Algorithm',
                ylabel='Observations per Event',
                save_dir=save_dir,
                filename_stem=f'{stem_prefix}_Plot4b-Obs_per_Event')

        print('  Plot 5a -- Response Time')
        plot_metric(
            df,
            metrics=[
                'Average Normalized Earliest Response Time to Event',
                'Average Normalized Latest Response Time to Event',
                'Average Event Announcement Time [norm]',
            ],
            ylabels=[
                'Avg. Norm. Earliest Response Time (Event)',
                'Avg. Norm. Latest Response Time (Event)',
                'Avg. Norm. Time to Announcement',
            ],
            suptitle='Event Response Time & Announcement Latency by Algorithm',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot5a-Response_Time',
            response_time_note=True)

        print('  Plot 5b -- Event Latency Budget')
        plot_latency_budget(df, save_dir,
                            f'{stem_prefix}_Plot5b-Latency_Budget')

        print('  Plot 6 -- Connectivity x Detection Interaction')
        plot_interaction(
            df,
            metrics=[
                'Total Obtained Reward [norm]',
                'P(Task Observed | Task Observable)',
                'P(Event Co-observed | Co-observable)',
                'Average Normalized Earliest Response Time to Event',
            ],
            ylabels=[
                'Reward (normalised)',
                'P(Task Obs | Observable)',
                'P(Co-obs | Co-obs.)',
                'Avg. Norm. Earliest Response Time',
            ],
            suptitle='Effect of Detection Mode by Connectivity Level',
            save_dir=save_dir,
            filename_stem=f'{stem_prefix}_Plot6-Interaction',
            hline_at_one=True)

        print('  Plot 7 -- Detection Pipeline Cascade')
        plot_detection_cascade(df, save_dir,
                               f'{stem_prefix}_Plot7-Detection_Cascade')

        print('  Plot 8 -- Trade-off Scatter')
        plot_tradeoff_scatter(df, save_dir,
                              f'{stem_prefix}_Plot8-Tradeoff_Scatter')

        print('  Plot 9 -- Communication Load')
        plot_communication_load(df, save_dir,
                                f'{stem_prefix}_Plot9-Communication_Load')

        print('  Plot 10 -- Seasonal Sensitivity')
        plot_seasonal_sensitivity(df, save_dir,
                                  f'{stem_prefix}_Plot10-Seasonal')

    print('\nDONE.')


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        default=os.path.join(
            'experiments', '2_centralized_vs_decentralized', 'analysis',
            'compiled',
            'full_factorial_trials_2026-05-22_compiled_results.csv'))
    parser.add_argument('--trial-name',
                        default='full_factorial_trials_2026-05-22')
    parser.add_argument('--abridged', action='store_true')
    parser.add_argument('--full',     action='store_true')
    args = parser.parse_args()

    if not args.abridged and not args.full:
        args.abridged = True

    subset = ('both'     if args.abridged and args.full else
              'abridged' if args.abridged else 'full')

    generate_plots(csv_path=args.csv, trial_name=args.trial_name,
                   subset=subset)