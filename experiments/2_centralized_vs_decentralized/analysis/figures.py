"""
figures.py
==========
Generates all analysis plots for the Centralized vs Decentralized
satellite scheduling experiment — multi-mission, multi-connectivity,
multi-data-processing factorial dataset.

Algorithm naming convention:
  SC-CBBA    = Sequence-Constrained Consensus Bundle-Based Algorithm
  DP-SC-CBBA = DP preplanner + SC-CBBA replanner
  None-None  excluded from all plots except the decomposition heatmap
              (grayed / hatched).

Colorblind-friendly palette: Wong (2011).

Usage
-----
    python figures.py
    python figures.py --csv path/to/results.csv
    python figures.py --date 2019-02-15
    python figures.py --complete-date 2019-02-15
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.rcParams.update({
    'font.family':           'sans-serif',
    'font.size':             10,
    'axes.titlesize':        11,
    'axes.labelsize':        10,
    'legend.fontsize':        8,
    'legend.title_fontsize':  9,
    'figure.dpi':            150,
})


# =============================================================================
#  CONSTANTS
# =============================================================================

ALGO_ORDER: list[str] = [
    'MILP', 'DP', 'DP-GR', 'GR', 'DP-SC-CBBA', 'SC-CBBA',
]

# Wong (2011) colorblind-safe palette
# MILP = gray (centralised, categorically different)
# DP   = vermillion (distinct from grays and blues)
ALGO_PALETTE: dict[str, str] = {
    'None-None':  '#BBBBBB',   # light gray   (passive reference / hatched)
    'MILP':       '#999999',   # mid gray     (centralised benchmark)
    'DP':         '#D55E00',   # vermillion   (preplanner only)
    'DP-GR':      '#E69F00',   # amber
    'GR':         '#009E73',   # teal
    'DP-SC-CBBA': '#0072B2',   # blue
    'SC-CBBA':    '#56B4E9',   # sky blue
}

ALGO_LINESTYLES: dict[str, tuple | str] = {
    'None-None':  (1, 3),
    'MILP':       (2, 2),
    'DP':         (4, 2),
    'DP-GR':      (4, 1, 1, 1),
    'GR':         (1, 2),
    'DP-SC-CBBA': '',
    'SC-CBBA':    (3, 1),
}

MISSION_ORDER: list[str] = ['Urgency', 'Revisits', 'Co-observations']

# Short labels used in axis ticks / panel titles
MISSION_LABELS: dict[str, str] = {
    'Urgency':         'Urgency',
    'Revisits':        'Revisits',
    'Co-observations': 'Co-observations',
}

# Full labels used in figure titles / suptitles
MISSION_FULL_LABELS: dict[str, str] = {
    'Urgency':         'Urgent Priority Mission',
    'Revisits':        'Revisit Priority Mission',
    'Co-observations': 'Co-observation Priority Mission',
}

CONN_ORDER: list[str] = ['GS', 'Intraconstellation', 'Interconstellation']

CONN_LABELS: dict[str, str] = {
    'GS':                 'GS Only',
    'Intraconstellation': 'Intra-\nconstellation',
    'Interconstellation': 'Inter-\nconstellation',
}

CONN_FULL_LABELS: dict[str, str] = {
    'GS':                 'GS-Only Connectivity',
    'Intraconstellation': 'Intraconstellation Connectivity',
    'Interconstellation': 'Interconstellation Connectivity',
}

# Ground / Onboard = operational; Instant = oracle benchmark
DP_ORDER: list[str] = ['Ground', 'Onboard', 'Instant']

DP_LABELS: dict[str, str] = {
    'Ground':  'Ground\nDetection',
    'Onboard': 'Onboard\nDetection',
    'Instant': 'Instant\n(Benchmark)',
}

DP_LABELS_SHORT: dict[str, str] = {
    'Ground':  'Ground',
    'Onboard': 'Onboard',
    'Instant': 'Instant (Benchmark)',
}

REVISIT_TARGET_S: float = 3600.0

MARKER_SHAPES: dict[str, str] = {
    'None-None':  'X',
    'MILP':       'D',
    'DP':         's',
    'DP-GR':      '^',
    'GR':         'v',
    'DP-SC-CBBA': 'o',
    'SC-CBBA':    'P',
}


# =============================================================================
#  DATA LOADING
# =============================================================================

def label_algorithm(row: pd.Series) -> str:
    pre = row['Preplanner']
    rep = row['Replanner']
    if pre == 'Centralized-MILP_priority':  return 'MILP'
    if pre == 'DP'   and rep == 'None':     return 'DP'
    if pre == 'DP'   and rep in ('SC-CBBA', 'CBBA'):   return 'DP-SC-CBBA'
    if pre == 'DP'   and rep == 'Greedy':   return 'DP-GR'
    if pre == 'None' and rep in ('SC-CBBA', 'CBBA'):   return 'SC-CBBA'
    if pre == 'None' and rep == 'Greedy':   return 'GR'
    if pre == 'None' and rep == 'None':     return 'None-None'
    return 'Unknown'


def load_and_prepare(csv_path: str,
                     trial_csv_path: str | None = None,
                     completeness_threshold: float = 0.90) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df['Preplanner']      = df['Preplanner'].fillna('None')
    df['Replanner']       = df['Replanner'].fillna('None')
    df['Algorithm']       = df.apply(label_algorithm, axis=1)
    df['Algorithm']       = pd.Categorical(
        df['Algorithm'], categories=['None-None'] + ALGO_ORDER, ordered=True)
    df['Connectivity']    = pd.Categorical(
        df['Connectivity'], categories=CONN_ORDER, ordered=True)
    df['Data Processing'] = pd.Categorical(
        df['Data Processing'], categories=DP_ORDER, ordered=True)
    df['Mission']         = pd.Categorical(
        df['Mission'], categories=MISSION_ORDER, ordered=True)

    # --- Complete-date filter (mirrors compiler.py logic) ---
    # Load trial definitions to determine expected count per date
    if trial_csv_path and os.path.exists(trial_csv_path):
        trials = pd.read_csv(trial_csv_path)
        trials['Preplanner'] = trials['Preplanner'].fillna('None')
        trials['Replanner']  = trials['Replanner'].fillna('None')
        # Expected non-MILP trials per date
        expected_per_date = (
            trials[trials['Preplanner'] != 'Centralized-MILP_priority']
            .groupby('Date').size()
        )
        non_milp = df[df['Preplanner'] != 'Centralized-MILP_priority']
        actual_per_date = non_milp.groupby('Date').size()
        complete_dates = [
            d for d in expected_per_date.index
            if actual_per_date.get(d, 0) / expected_per_date[d]
               >= completeness_threshold
        ]
        if complete_dates:
            df = df[df['Date'].isin(complete_dates)].copy()
            print(f'  [load] Complete dates (≥{completeness_threshold:.0%}): '
                  f'{complete_dates}')
        else:
            print(f'  [load] No fully complete dates found at threshold '
                  f'{completeness_threshold:.0%}; using all data.')
    else:
        print('  [load] No trial CSV provided; skipping completeness filter.')

    # --- Derived columns ---
    obs       = df['Total Obtained Task Observations'].replace(0, np.nan)
    tasks_obs = (df['Tasks Observed']
                 if 'Tasks Observed' in df.columns
                 else df['Event-Driven Tasks Observed']).replace(0, np.nan)
    co_able   = df['Events Co-observable'].replace(0, np.nan)
    full_able = df['Events Fully Co-observable'].replace(0, np.nan)

    df['Reward per Observation'] = df['Total Obtained Reward'] / obs
    df['Reward per Task']        = df['Total Obtained Reward'] / tasks_obs
    df['P(Event Tasked Co-observed | Co-observable)'] = (
        df['Events Tasked Co-observed'] / co_able)
    df['P(Event Tasked Fully Co-observed | Fully Co-observable)'] = (
        df['Events Tasked Fully Co-observed'] / full_able)
    df['Average Co-observations per Task'] = df['Event Co-observations'] / df['Events Co-observed']
    df['Average Tasked Co-observations per Task'] = df['Tasked Event Co-observations'] / df['Events Tasked Co-observed']

    # --- MILP normalisation (reward / MILP reward for same Date×Mission×Conn×DP) ---
    milp_rows = df[df['Algorithm'] == 'MILP'][
        ['Date', 'Mission', 'Connectivity', 'Data Processing',
         'Total Obtained Reward']
    ].rename(columns={'Total Obtained Reward': 'MILP Reward'})

    df = df.merge(milp_rows,
                  on=['Date', 'Mission', 'Connectivity', 'Data Processing'],
                  how='left')
    df['Reward / MILP'] = df['Total Obtained Reward'] / df['MILP Reward'].replace(0, np.nan)

    return df.sort_values(['Algorithm', 'Connectivity',
                           'Data Processing', 'Mission'])


# =============================================================================
#  UTILITIES
# =============================================================================

def save_plot(save_dir: str, filename: str) -> None:
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'    Saved -> {path}')


def _col_ok(df: pd.DataFrame, col: str, ctx: str) -> bool:
    if col not in df.columns:
        print(f'  [{ctx}] Column absent -- skipping: {col}')
        return False
    return True


def _apply_linestyle(line, algo: str) -> None:
    spec = ALGO_LINESTYLES.get(algo, '')
    if spec != '':
        line.set_dashes(spec)


def _algo_present(df: pd.DataFrame,
                  exclude_none_none: bool = True) -> list[str]:
    present = [a for a in ALGO_ORDER if a in df['Algorithm'].values]
    if not exclude_none_none and 'None-None' in df['Algorithm'].values:
        present = ['None-None'] + present
    return present


def _make_legend_handles(algos: list[str]) -> list:
    return [mpatches.Patch(color=ALGO_PALETTE.get(a, '#999'), label=a)
            for a in algos]


def _ci95(series: pd.Series) -> float:
    n = series.dropna().__len__()
    if n < 2:
        return 0.0
    return series.sem(ddof=1) * stats.t.ppf(0.975, df=n - 1)


def _winner_label(algo: str) -> str:
    """Return algorithm label with winner indicator."""
    return f'★ {algo}'


def _annotate_winner(ax: plt.Axes, metric: str,
                     data: pd.DataFrame, present: list[str]) -> None:
    """Bold and prefix winner tick label with ★."""
    means = {a: data[data['Algorithm'] == a][metric].mean()
             for a in present
             if not data[data['Algorithm'] == a][metric].dropna().empty}
    if not means:
        return
    winner = max(means, key=means.get)
    # Re-set xticklabels with winner prefixed
    new_labels = [f'★ {a}' if a == winner else a for a in present]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(new_labels, rotation=25, ha='right', fontsize=8)
    # Colour and bold the winner tick
    for lbl in ax.get_xticklabels():
        if lbl.get_text().startswith('★'):
            lbl.set_fontweight('bold')
            lbl.set_color(ALGO_PALETTE.get(winner, '#333'))


# =============================================================================
#  SHARED BOX PLOT BUILDER
#  Uses matplotlib ax.boxplot for reliable alignment with strip points.
#  Visual style: solid filled boxes, thin black outlines (lw=0.8), red median.
# =============================================================================

def make_boxplot(data: pd.DataFrame, metric: str, ax: plt.Axes,
                 ylabel: str | None = None,
                 exclude_none_none: bool = True,
                 annotate_winner: bool = True,
                 milp_line: bool = False) -> None:
    present = _algo_present(data, exclude_none_none=exclude_none_none)
    xs      = np.arange(len(present))
    box_w   = 0.85   # wider to compensate for multi-panel figure layout

    box_data = [data[data['Algorithm'] == a][metric].dropna().values
                for a in present]

    bp = ax.boxplot(
        box_data,
        positions=xs,
        widths=box_w,
        patch_artist=True,
        medianprops=dict(color='red', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8, color='black'),
        capprops=dict(linewidth=0.8, color='black'),
        boxprops=dict(linewidth=0.8, edgecolor='black'),
        flierprops=dict(marker='', markersize=0),
        showfliers=False,
    )

    for patch, algo in zip(bp['boxes'], present):
        patch.set_facecolor(ALGO_PALETTE.get(algo, '#999'))
        patch.set_alpha(0.75)
        if algo == 'None-None':
            patch.set_hatch('//')

    # Strip points at the same integer x-positions — guaranteed aligned
    rng = np.random.default_rng(seed=42)
    for xi, algo in enumerate(present):
        vals = data[data['Algorithm'] == algo][metric].dropna().values
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-box_w * 0.35, box_w * 0.35, size=len(vals))
        ax.scatter(xi + jitter, vals,
                   color=ALGO_PALETTE.get(algo, '#999'),
                   s=14, alpha=0.50, zorder=3,
                   edgecolors='none', linewidths=0)

    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(present, rotation=25, ha='right', fontsize=8)

    if milp_line:
        ax.axhline(1.0, color=ALGO_PALETTE['MILP'], linewidth=1.4,
                   linestyle='--', alpha=0.8, label='MILP = 1.0')

    ax.set_xlabel('')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)

    if annotate_winner and metric in data.columns:
        _annotate_winner(ax, metric, data, present)


# =============================================================================
#  SHARED BAR PLOT BUILDER  (mean ± CI, algorithm-coloured bars)
# =============================================================================

def make_barplot(data: pd.DataFrame, metric: str, ax: plt.Axes,
                 ylabel: str | None = None,
                 exclude_none_none: bool = True,
                 annotate_winner: bool = True,
                 milp_line: bool = False) -> None:
    present = _algo_present(data, exclude_none_none=exclude_none_none)
    xs      = np.arange(len(present))
    bar_w   = 0.70

    for xi, algo in enumerate(present):
        vals = data[data['Algorithm'] == algo][metric].dropna()
        if vals.empty:
            continue
        mean = vals.mean()
        ci   = _ci95(vals)
        ax.bar(xi, mean, width=bar_w,
               color=ALGO_PALETTE.get(algo, '#999'),
               alpha=0.85, edgecolor='white', linewidth=0.5,
               hatch='//' if algo == 'None-None' else None)
        ax.errorbar(xi, mean, yerr=ci, fmt='none',
                    ecolor='#333', capsize=3, linewidth=1.0)

    if milp_line:
        ax.axhline(1.0, color=ALGO_PALETTE['MILP'], linewidth=1.4,
                   linestyle='--', alpha=0.8, label='MILP = 1.0')

    ax.set_xticks(xs)
    ax.set_xticklabels(present, rotation=25, ha='right', fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)

    if annotate_winner and metric in data.columns:
        _annotate_winner(ax, metric, data, present)


# =============================================================================
#  FIGURE 1 — Mission Reward Degradation
#
#  Per scope (all-dates, per-date) four files are produced:
#    a) Box — absolute reward, all DP collapsed
#    b) Box — absolute reward, DP as rows
#    c) Bar — absolute reward, all DP collapsed        (bar equivalent of a)
#    d) Bar — absolute reward, DP as rows              (bar equivalent of b)
#  Per-date additionally:
#    e) Box — MILP-normalised, all DP collapsed
#    f) Box — MILP-normalised, DP as rows
#    g) Bar — MILP-normalised, all DP collapsed
#    h) Bar — MILP-normalised, DP as rows
#
#  Naming suffix key:
#    (no suffix) = box, absolute   | b = box, by-DP
#    _bar        = bar, absolute   | b_bar = bar, by-DP
#    _norm       = box, MILP-norm  | b_norm = box, by-DP, MILP-norm
#    _norm_bar   = bar, MILP-norm  | b_norm_bar = bar, by-DP, MILP-norm
# =============================================================================

def _make_degradation_collapsed(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str,
    filename: str,
    use_barplot: bool = False,
    is_normalised: bool = False,
) -> None:
    """1 row × 3 mission cols, shared y-axis across missions."""
    # For normalised plots: exclude MILP (it IS the reference = 1.0)
    plot_data = data[data['Algorithm'] != 'MILP'].copy() if is_normalised else data

    fig, axes = plt.subplots(
        1, len(MISSION_ORDER),
        figsize=(5 * len(MISSION_ORDER), 5), sharey=True)

    plotter = make_barplot if use_barplot else make_boxplot

    for ax, mission in zip(axes, MISSION_ORDER):
        sub = plot_data[plot_data['Mission'] == mission]
        if sub.empty:
            ax.set_visible(False)
            continue
        plotter(sub, metric, ax,
                ylabel=ylabel if mission == MISSION_ORDER[0] else '',
                milp_line=is_normalised)
        ax.set_title(MISSION_FULL_LABELS[mission], fontsize=10,
                     fontweight='bold')
        if is_normalised and mission == MISSION_ORDER[-1]:
            ax.legend(fontsize=7, loc='best')

    plt.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, filename)
    plt.close()


def _make_degradation_by_dp(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    suptitle: str,
    save_dir: str,
    filename: str,
    use_barplot: bool = False,
    is_normalised: bool = False,
) -> None:
    """3 DP rows × 3 mission cols, shared y-axis within each row."""
    # For normalised plots: exclude MILP
    plot_data = data[data['Algorithm'] != 'MILP'].copy() if is_normalised else data

    n_dp      = len(DP_ORDER)
    n_mission = len(MISSION_ORDER)
    plotter   = make_barplot if use_barplot else make_boxplot

    fig, axes = plt.subplots(
        n_dp, n_mission,
        figsize=(5 * n_mission, 4.5 * n_dp),
        sharey='row')

    for row_i, dp in enumerate(DP_ORDER):
        dp_sub = plot_data[plot_data['Data Processing'] == dp]
        for col_i, mission in enumerate(MISSION_ORDER):
            ax  = axes[row_i][col_i]
            sub = dp_sub[dp_sub['Mission'] == mission]
            if sub.empty:
                ax.set_visible(False)
                continue
            plotter(sub, metric, ax,
                    ylabel=ylabel if col_i == 0 else '',
                    milp_line=is_normalised)
            if row_i == 0:
                ax.set_title(MISSION_FULL_LABELS[mission], fontsize=10,
                             fontweight='bold')
            if col_i == n_mission - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(DP_LABELS[dp], fontsize=9,
                              rotation=270, labelpad=16, va='bottom')

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, filename)
    plt.close()


def plot_mission_degradation(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    data  = df[df['Algorithm'] != 'None-None'].copy()
    abs_m = 'Total Obtained Reward'
    nrm_m = 'Reward / MILP'

    dates = sorted(data['Date'].dropna().unique())

    def _emit(subset, slug_base, date_label):
        """Emit all variants for a given data subset.

        Naming convention: Plot4_X_Y{SPEC}-Title.png
          SPEC: (none) = default, b = by-DP, _bar = bar chart,
                b_bar = by-DP bar, _norm = normalised box,
                b_norm = normalised by-DP box, etc.
        """
        base_title = f'Mission Reward by Algorithm — {date_label}'
        norm_title = f'Mission Reward by Algorithm (MILP-Normalised) — {date_label}'

        # Absolute — collapsed
        _make_degradation_collapsed(
            subset, abs_m, 'Total Obtained Reward',
            base_title, save_dir, f'{slug_base}.png')

        # Absolute — by DP (b spec)
        _make_degradation_by_dp(
            subset, abs_m, 'Total Obtained Reward',
            base_title + '\n(rows = Data Processing mode)',
            save_dir, f'{slug_base}b-Mission_Reward_by_DP.png')

        # Bar equivalents
        _make_degradation_collapsed(
            subset, abs_m, 'Total Obtained Reward',
            base_title, save_dir, f'{slug_base}_bar.png',
            use_barplot=True)
        _make_degradation_by_dp(
            subset, abs_m, 'Total Obtained Reward',
            base_title + '\n(rows = Data Processing mode)',
            save_dir, f'{slug_base}b_bar-Mission_Reward_by_DP.png',
            use_barplot=True)

        # MILP-normalised (only if column available)
        if nrm_m in subset.columns and subset[nrm_m].notna().any():
            _make_degradation_collapsed(
                subset, nrm_m, 'Reward / MILP Reward',
                norm_title, save_dir, f'{slug_base}_norm.png',
                is_normalised=True)
            _make_degradation_by_dp(
                subset, nrm_m, 'Reward / MILP Reward',
                norm_title + '\n(rows = Data Processing mode)',
                save_dir, f'{slug_base}b_norm-Mission_Reward_Norm_by_DP.png',
                is_normalised=True)
            _make_degradation_collapsed(
                subset, nrm_m, 'Reward / MILP Reward',
                norm_title, save_dir, f'{slug_base}_norm_bar.png',
                use_barplot=True, is_normalised=True)
            _make_degradation_by_dp(
                subset, nrm_m, 'Reward / MILP Reward',
                norm_title + '\n(rows = Data Processing mode)',
                save_dir, f'{slug_base}b_norm_bar-Mission_Reward_Norm_by_DP.png',
                use_barplot=True, is_normalised=True)

    # All dates combined
    _emit(data, 'Plot4_1_1-Mission_Reward_All_Dates', 'All Dates & Conditions')

    # One figure per date
    for date_i, date in enumerate(dates, start=2):
        date_sub  = data[data['Date'] == date]
        if date_sub.empty:
            continue
        date_slug = date.replace('-', '')
        _emit(date_sub,
              f'Plot4_1_{date_i}-Mission_Reward_{date_slug}',
              date)


# =============================================================================
#  FIGURE 2 — Decomposition Heatmap  (supplemental)
#  Rows = Mission, Cols = Data Processing
#  Preplanner on y-axis, Replanner on x-axis.
#  Single colorbar per row (rightmost panel only).
#  Per-mission vmin/vmax for better color resolution.
# =============================================================================

def plot_decomposition_heatmap(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Total Obtained Reward'

    row_labels = ['MILP\n(Centralized)', 'DP\n(Onboard)', 'None\n(No Pre)']
    col_labels  = ['No\nReplanner', 'Greedy', 'SC-CBBA']

    algo_cell = {
        'MILP':       (0, 0),
        'DP':         (1, 0),
        'DP-GR':      (1, 1),
        'DP-SC-CBBA': (1, 2),
        'None-None':  (2, 0),
        'GR':         (2, 1),
        'SC-CBBA':    (2, 2),
    }

    n_missions = len(MISSION_ORDER)
    n_dp       = len(DP_ORDER)

    # Global vmin/vmax across all missions and DP modes for consistent colour scale
    vmin = df[metric].min()
    vmax = df[metric].max()

    fig, axes = plt.subplots(
        n_missions, n_dp,
        figsize=(4.5 * n_dp, 4.2 * n_missions),
        sharey='row')

    # Single shared colormap image for one global colorbar
    sm = plt.cm.ScalarMappable(# cmap='YlOrRd',
                                cmap='rocket_r',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    for row_i, mission in enumerate(MISSION_ORDER):
        mission_sub = df[df['Mission'] == mission]

        for col_i, dp in enumerate(DP_ORDER):
            ax  = axes[row_i][col_i]
            sub = mission_sub[mission_sub['Data Processing'] == dp]
            grid = np.full((3, 3), np.nan)
            text = [[''] * 3 for _ in range(3)]

            for algo, (ri, ci) in algo_cell.items():
                vals = sub[sub['Algorithm'] == algo][metric].dropna()
                if len(vals):
                    m, s = vals.mean(), vals.std(ddof=0)
                    grid[ri, ci] = m
                    text[ri][ci] = f'{m:.0f}\n±{s:.0f}'

            masked = np.ma.masked_invalid(grid)
            ax.imshow(masked, cmap='YlOrRd',
                      vmin=vmin, vmax=vmax, aspect='auto')

            for ri in range(3):
                for ci in range(3):
                    if text[ri][ci]:
                        brightness = (masked[ri, ci] - vmin) / max(
                            vmax - vmin, 1e-9)
                        fc = 'white' if brightness > 0.65 else 'black'
                        ax.text(ci, ri, text[ri][ci],
                                ha='center', va='center',
                                fontsize=7, color=fc)

            # None-None: gray + hatch
            nn_r, nn_c = algo_cell['None-None']
            ax.add_patch(plt.Rectangle(
                (nn_c - 0.5, nn_r - 0.5), 1, 1,
                facecolor='#CCCCCC', hatch='//',
                edgecolor='#777777', linewidth=0.8, zorder=2))
            ax.text(nn_c, nn_r, 'Passive\n(ref.)',
                    ha='center', va='center', fontsize=6,
                    color='#444444', style='italic', zorder=3)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(col_labels, fontsize=7)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_xlabel('Replanner', fontsize=8)
            if col_i == 0:
                ax.set_ylabel('Preplanner', fontsize=8)
            ax.axhline(0.5, color='white', linewidth=2)

            # Column headers (DP mode) on top row only
            if row_i == 0:
                ax.set_title(DP_LABELS[dp], fontsize=9, fontweight='bold')

            # Mission label on rightmost column as right-side y-axis label
            if col_i == n_dp - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(MISSION_FULL_LABELS[mission], fontsize=9,
                              rotation=270, labelpad=16, va='bottom')

    # Single global colorbar — placed outside the grid to the right
    # Use tight_layout first, then add colorbar in remaining space
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sm, cax=cbar_ax, label='Total Reward')

    plt.suptitle(
        'Reward Decomposition: Preplanner × Replanner\n',
        # '(rows = Mission Priority; cols = Data Processing; '
        # 'shared colour scale)',
        fontsize=12, y=1.01)
    save_plot(save_dir, 'Plot4_2_1-Decomposition_Heatmap.png')
    plt.close()


# =============================================================================
#  FIGURE 3 — Two-Strategy Scatter  1×2 per mission (3 rows × 2 cols)
#  Left col:  Total Observations vs Reward per Observation
#  Right col: Total Observations vs Reward per Task
#  No Pareto front — per-mission context makes it more readable.
# =============================================================================

def plot_two_strategy_scatter(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    data  = df[df['Algorithm'] != 'None-None'].copy()
    x_col = 'Total Obtained Task Observations'
    panels = [
        ('Reward per Observation',
         'Reward per Observation\n(scheduling efficiency)'),
        ('Reward per Task',
         'Reward per Task\n(selection quality)'),
    ]

    def _pareto_front(xy: np.ndarray) -> np.ndarray:
        """
        Non-dominated set maximising BOTH x (observations) AND y (quality).
        A point dominates another if it has >= x and >= y with at least one strict.
        This highlights strategies that lead either axis — the two extremes
        of volume and quality that together define the frontier of achievable performance.
        """
        is_eff = np.ones(len(xy), dtype=bool)
        for i, c in enumerate(xy):
            if is_eff[i]:
                # Dominated if another point has >= on both axes
                dominated = (xy[:, 0] >= c[0]) & (xy[:, 1] >= c[1])
                dominated[i] = False
                if dominated.any():
                    is_eff[i] = False
        return xy[is_eff]

    n_rows = len(MISSION_ORDER)
    n_cols = len(panels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows),
                             sharey=False, sharex='col')

    for row_i, mission in enumerate(MISSION_ORDER):
        mission_data = data[data['Mission'] == mission]

        for col_i, (y_col, ylabel) in enumerate(panels):
            ax = axes[row_i][col_i]
            if not _col_ok(data, y_col, 'Fig 3'):
                ax.set_visible(False)
                continue

            # All points use circle — mission distinguished by row
            for algo in ALGO_ORDER:
                sub = mission_data[
                    mission_data['Algorithm'] == algo
                ].dropna(subset=[x_col, y_col])
                if sub.empty:
                    continue
                ax.scatter(sub[x_col], sub[y_col],
                           color=ALGO_PALETTE[algo],
                           marker='o',
                           s=70, alpha=0.70,
                           edgecolors='white', linewidths=0.4,
                           zorder=3)

            # Pareto front per panel
            pts = mission_data.dropna(
                subset=[x_col, y_col])[[x_col, y_col]].values
            if len(pts) > 1:
                pareto = _pareto_front(pts)
                pareto = pareto[pareto[:, 0].argsort()]
                ax.step(pareto[:, 0], pareto[:, 1],
                        where='post', color='#333333', linewidth=1.2,
                        linestyle='--', alpha=0.55, zorder=2,
                        label='Pareto front' if col_i == 0 else '')

            ax.set_xlabel(x_col if row_i == n_rows - 1 else '', fontsize=9)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

            # Mission label + y-label on left column
            if col_i == 0:
                ax.set_ylabel(
                    f'{MISSION_FULL_LABELS[mission]}\n\n{ylabel}',
                    fontsize=9)
            else:
                ax.set_ylabel(ylabel, fontsize=9)

            # Column headers on top row only
            if row_i == 0:
                ax.set_title(ylabel.split('\n')[0], fontsize=10,
                             fontweight='bold')

    # Algorithm legend at top
    algo_handles = [
        plt.scatter([], [], color=ALGO_PALETTE[a], marker='o', s=70, label=a)
        for a in ALGO_ORDER
    ]
    pareto_handle = matplotlib.lines.Line2D(
        [], [], color='#333333', linewidth=1.2,
        linestyle='--', alpha=0.55, label='Pareto front')
    fig.legend(handles=algo_handles + [pareto_handle],
               title='Algorithm', loc='upper center',
               ncol=len(ALGO_ORDER) + 1,
               fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

    plt.suptitle(
        'Scheduling Strategy: Volume vs Quality',
        fontsize=12, y=1.04)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_3_1-Two_Strategy_Scatter.png')
    plt.close()


# =============================================================================
#  FIGURE 4 — Data Processing Effect
#  Uses complete_date only. Ground | Onboard  ‖  Instant (benchmark).
#  No star on Instant — vertical separator line is sufficient.
# =============================================================================

def plot_data_processing_effect(
    df: pd.DataFrame,
    save_dir: str,
    complete_date: str = '2019-02-15',
) -> None:
    metric = 'Total Obtained Reward'
    data   = df[(df['Date'] == complete_date) &
                (df['Algorithm'] != 'None-None')].copy()

    if data.empty:
        print(f'  [Fig 4] No data for date {complete_date} -- skipping.')
        return

    x_positions = {'Ground': 0, 'Onboard': 1, 'Instant': 3}
    x_ticks     = [0, 1, 3]
    x_labels    = ['Ground', 'Onboard', 'Instant']

    bar_width = 0.12
    n_algos   = len(ALGO_ORDER)
    offsets   = np.linspace(
        -(n_algos - 1) * bar_width / 2,
         (n_algos - 1) * bar_width / 2, n_algos)

    fig, axes = plt.subplots(
        1, len(MISSION_ORDER),
        figsize=(6 * len(MISSION_ORDER), 5), sharey=False)

    for ax, mission in zip(axes, MISSION_ORDER):
        sub = data[data['Mission'] == mission]

        for algo_i, algo in enumerate(ALGO_ORDER):
            asub = sub[sub['Algorithm'] == algo]
            for dp in DP_ORDER:
                dp_sub = asub[asub['Data Processing'] == dp]
                if dp_sub.empty:
                    continue
                xpos = x_positions[dp] + offsets[algo_i]
                mean = dp_sub[metric].mean()
                ci   = _ci95(dp_sub[metric])
                ax.bar(xpos, mean, width=bar_width,
                       color=ALGO_PALETTE[algo], alpha=0.85,
                       edgecolor='white', linewidth=0.4)
                ax.errorbar(xpos, mean, yerr=ci, fmt='none',
                            ecolor='#333', capsize=2, linewidth=0.8)

        # Separator before Instant — labels pointing AWAY from the line
        sep_x = (x_positions['Onboard'] + x_positions['Instant']) / 2
        ax.axvline(sep_x, color='#555555', linewidth=1.0,
                   linestyle=':', alpha=0.7)
        ax.text(sep_x - 0.05, ax.get_ylim()[1],
                '← Operational',
                fontsize=7, color='#555555', va='top', ha='right')
        ax.text(sep_x + 0.05, ax.get_ylim()[1],
                'Benchmark →',
                fontsize=7, color='#555555', va='top', ha='left')

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Data Processing Mode', fontsize=9)
        ax.set_ylabel('Total Obtained Reward'
                      if mission == MISSION_ORDER[0] else '')
        ax.set_title(MISSION_FULL_LABELS[mission], fontsize=10,
                     fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    handles = _make_legend_handles(ALGO_ORDER)
    fig.legend(handles=handles, title='Algorithm',
               loc='lower center', ncol=len(ALGO_ORDER),
               fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.07))

    plt.suptitle(
        f'Effect of Data Processing Mode on Mission Reward\n'
        f'(date: {complete_date}; averaged across connectivity)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_4_1-Data_Processing_Effect.png')
    plt.close()


# =============================================================================
#  FIGURE 5 — Connectivity Effect
#  Faceted by Data Processing (cols) to reduce variance.
#  One figure per mission.
# =============================================================================

def plot_connectivity_effect(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric   = 'Total Obtained Reward'
    data     = df[df['Algorithm'] != 'None-None'].copy()
    conn_pos = {c: i for i, c in enumerate(CONN_ORDER)}

    for mission_i, mission in enumerate(MISSION_ORDER, start=1):
        mission_data = data[data['Mission'] == mission]

        fig, axes = plt.subplots(
            1, len(DP_ORDER),
            figsize=(5 * len(DP_ORDER), 5), sharey=True)

        for ax, dp in zip(axes, DP_ORDER):
            sub = mission_data[mission_data['Data Processing'] == dp]

            for algo in ALGO_ORDER:
                asub  = sub[sub['Algorithm'] == algo]
                means = asub.groupby(
                    'Connectivity', observed=True)[metric].mean()
                # Use IQR band instead of 95% CI to reduce noise
                q25   = asub.groupby(
                    'Connectivity', observed=True)[metric].quantile(0.25)
                q75   = asub.groupby(
                    'Connectivity', observed=True)[metric].quantile(0.75)

                xs  = [conn_pos[c] for c in CONN_ORDER if c in means.index]
                ys  = [means[c]    for c in CONN_ORDER if c in means.index]
                lo  = [q25[c]      for c in CONN_ORDER if c in means.index]
                hi  = [q75[c]      for c in CONN_ORDER if c in means.index]
                if not xs:
                    continue

                spec = ALGO_LINESTYLES[algo]
                line, = ax.plot(
                    xs, ys,
                    color=ALGO_PALETTE[algo], linewidth=1.8,
                    marker='o', markersize=6,
                    linestyle='solid' if spec == '' else 'dashed',
                    label=algo)
                _apply_linestyle(line, algo)
                ax.fill_between(xs, lo, hi,
                                color=ALGO_PALETTE[algo], alpha=0.13)

            ax.set_xticks(list(conn_pos.values()))
            ax.set_xticklabels(
                [CONN_LABELS[c].replace('\n', ' ') for c in CONN_ORDER],
                fontsize=8, rotation=12, ha='right')
            ax.set_xlabel('Connectivity Architecture', fontsize=9)
            ax.set_ylabel('Total Obtained Reward'
                          if dp == DP_ORDER[0] else '')
            ax.set_title(DP_LABELS[dp], fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

            if dp == DP_ORDER[-1]:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')

        plt.suptitle(
            f'{MISSION_FULL_LABELS[mission]} — '
            f'Effect of Connectivity Architecture\n'
            f'(shading = IQR across dates; cols = data processing mode)',
            fontsize=12, y=1.02)
        plt.tight_layout()
        save_plot(save_dir,
                  f'Plot4_5_{mission_i}-Connectivity_{mission}.png')
        plt.close()


# =============================================================================
#  FIGURE 6a — Urgency: Coverage vs Response Time (scatter)
#  x = Median Normalized Response Time to Task (lower = better)
#  y = P(Task Observed | Task Observable)      (higher = better)
#  One panel per connectivity level, colour = algorithm.
#  Utopia point at bottom-left (fast response, full coverage).
# =============================================================================

def plot_urgency_requirements(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    x_col = 'Median Normalized Response Time to Task'
    y_col = 'P(Task Observed | Task Observable)'
    data  = df[(df['Mission'] == 'Urgency') &
               (df['Algorithm'] != 'None-None')].copy()

    if not _col_ok(data, x_col, 'Fig 6a') or \
       not _col_ok(data, y_col, 'Fig 6a'):
        return

    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(1, n_conn,
                             figsize=(5 * n_conn, 5),
                             sharey=True, sharex=True)

    for ax, conn in zip(axes, CONN_ORDER):
        sub = data[data['Connectivity'] == conn]

        for algo in ALGO_ORDER:
            asub = sub[sub['Algorithm'] == algo].dropna(
                subset=[x_col, y_col])
            if asub.empty:
                continue
            ax.scatter(asub[x_col], asub[y_col],
                       color=ALGO_PALETTE[algo],
                    #    marker=MARKER_SHAPES[algo],
                        # marker=asub['Mission'],
                       s=70, alpha=0.75,
                       edgecolors='white', linewidths=0.4,
                       zorder=3, label=algo)

        # Utopia: low response time, high coverage
        x_utopia = 0.0 # data[x_col].min() * 0.95
        y_utopia = 1.0 # data[y_col].max() * 1.02
        # ax.scatter(x_utopia, min(y_utopia, 1.0), marker='*', s=300,
        #            color='gold', edgecolors='black',
        #            linewidths=0.8, zorder=6)
        # ax.annotate('Utopia', xy=(x_utopia, min(y_utopia, 1.0)),
        #             xytext=(x_utopia + (data[x_col].max() * 0.02),
        #                     min(y_utopia, 1.0) - 0.02),
        #             fontsize=7, color='#333333')

        ax.set_xlabel('Median Norm. Response Time\n(lower = faster)', fontsize=9)
        ax.set_ylabel('P(Task Observed | Observable)'
                      if conn == CONN_ORDER[0] else '')
        ax.set_title(CONN_FULL_LABELS[conn], fontsize=9, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        if conn == CONN_ORDER[-1]:
            ax.legend(title='Algorithm', fontsize=7,
                      title_fontsize=8, loc='best')

    plt.suptitle(
        f'{MISSION_FULL_LABELS["Urgency"]} — Requirement Satisfaction\n'
        f'(efficient algorithms appear bottom-right with high coverage '
        f'and fast response)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_6_1-Urgency_Requirements.png')
    plt.close()


# =============================================================================
#  FIGURE 6b — Revisits: Reobservation Time vs 3600s target
# =============================================================================

def plot_revisit_requirements(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Median Task Reobservation Time [s]'
    data   = df[(df['Mission'] == 'Revisits') &
                (df['Algorithm'] != 'None-None')].copy()

    if not _col_ok(data, metric, 'Fig 6b'):
        return

    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(1, n_conn, figsize=(5 * n_conn, 5), sharey=True)

    for ax, conn in zip(axes, CONN_ORDER):
        sub = data[data['Connectivity'] == conn]
        make_boxplot(sub, metric, ax,
                     ylabel='Median Reobservation Time'
                     if conn == CONN_ORDER[0] else '')
        ax.axhline(REVISIT_TARGET_S, color='#009E73',
                   linewidth=1.4, linestyle='--', alpha=0.85)
        ax.text(0.98, REVISIT_TARGET_S,
                f'Target: {REVISIT_TARGET_S/60:.0f} min',
                transform=ax.get_yaxis_transform(),
                fontsize=7, color='#009E73', va='bottom', ha='right')
        ax.set_title(CONN_FULL_LABELS[conn], fontsize=9, fontweight='bold')
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x/60:.0f} min'))

    plt.suptitle(
        f'{MISSION_FULL_LABELS["Revisits"]} — Requirement Satisfaction\n'
        f'(taskable observations only; green dashed = 60 min target)',
        fontsize=12, y=1.01)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_6_2-Revisit_Requirements.png')
    plt.close()


# =============================================================================
#  FIGURE 6c — Co-observations: Coverage vs Observation Efficiency (scatter)
#  x = Average Observations per Task (proxy for carpet-bombing)
#  y = P(Event Tasked Co-observed | Co-observable)
#  One panel per connectivity level, colour = algorithm.
#
#  NOTE: A future improvement is to compute per-trial the count of repeat
#  same-parameter observations within t_corr (redundant co-obs) from the
#  raw parquet files and add it as a summary column. This would allow a
#  third axis or facet showing how many co-obs were "earned" vs redundant.
# =============================================================================

# def plot_coobs_requirements(
#     df: pd.DataFrame,
#     save_dir: str,
# ) -> None:
#     x_col = 'Average Observations per Task'
#     y_col = 'P(Event Tasked Co-observed | Co-observable)'
#     data  = df[(df['Mission'] == 'Co-observations') &
#                (df['Algorithm'] != 'None-None')].copy()

#     if not _col_ok(data, x_col, 'Fig 6c') or \
#        not _col_ok(data, y_col, 'Fig 6c'):
#         return

#     n_conn = len(CONN_ORDER)
#     fig, axes = plt.subplots(1, n_conn,
#                              figsize=(5 * n_conn, 5),
#                              sharey=True, sharex=True)

#     for ax, conn in zip(axes, CONN_ORDER):
#         sub = data[data['Connectivity'] == conn]

#         for algo in ALGO_ORDER:
#             asub = sub[sub['Algorithm'] == algo].dropna(
#                 subset=[x_col, y_col])
#             if asub.empty:
#                 continue
#             ax.scatter(asub[x_col], asub[y_col],
#                        color=ALGO_PALETTE[algo],
#                     #    marker=MARKER_SHAPES[algo],
#                        s=70, alpha=0.75,
#                        edgecolors='white', linewidths=0.4,
#                        zorder=3, label=algo)
#         ax.set_xlabel('Avg. Observations per Task', fontsize=9)
#         ax.set_ylabel('P(Tasked Co-obs | Co-observable)'
#                       if conn == CONN_ORDER[0] else '')
#         ax.set_title(CONN_FULL_LABELS[conn], fontsize=9, fontweight='bold')
#         ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

#         if conn == CONN_ORDER[-1]:
#             ax.legend(title='Algorithm', fontsize=7,
#                       title_fontsize=8, loc='best')

#     plt.suptitle(
#         f'{MISSION_FULL_LABELS["Co-observations"]} — '
#         f'Requirement Satisfaction\n'
#         f'(NOTE: this plot will gain additional context once per-trial\n'
#         f' redundant co-observation counts are added to the summary)',
#         fontsize=12, y=1.03)
#     plt.tight_layout()
#     save_plot(save_dir, 'Plot4_6_3-Coobs_Requirements.png')
#     plt.close()

def plot_coobs_requirements(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    # x_col = 'Average Co-observations per Task'
    x_col = 'Average Tasked Co-observations per Task'
    y_col = 'P(Event Tasked Co-observed | Co-observable)'
    data  = df[(df['Mission'] == 'Co-observations') &
               (df['Algorithm'] != 'None-None')].copy()
 
    if not _col_ok(data, x_col, 'Fig 6c') or \
       not _col_ok(data, y_col, 'Fig 6c'):
        return
 
    n_conn = len(CONN_ORDER)
    fig, axes = plt.subplots(1, n_conn,
                             figsize=(5 * n_conn, 5),
                             sharey=True, sharex=True)
 
    for ax, conn in zip(axes, CONN_ORDER):
        sub = data[data['Connectivity'] == conn]
 
        for algo in ALGO_ORDER:
            asub = sub[sub['Algorithm'] == algo].dropna(
                subset=[x_col, y_col])
            if asub.empty:
                continue
            ax.scatter(asub[x_col], asub[y_col],
                       color=ALGO_PALETTE[algo],
                    #    marker=MARKER_SHAPES[algo],
                       s=70, alpha=0.75,
                       edgecolors='white', linewidths=0.4,
                       zorder=3, label=algo)
        ax.set_xlabel('Avg. Co-observations per Task', fontsize=9)
        ax.set_ylabel('P(Tasked Co-obs | Co-observable)'
                      if conn == CONN_ORDER[0] else '')
        ax.set_title(CONN_FULL_LABELS[conn], fontsize=9, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
 
        if conn == CONN_ORDER[-1]:
            ax.legend(title='Algorithm', fontsize=7,
                      title_fontsize=8, loc='best')
 
    plt.suptitle(
        f'{MISSION_FULL_LABELS["Co-observations"]} — '
        f'Requirement Satisfaction\n',
        # f'(NOTE: this plot will gain additional context once per-trial\n'
        # f' redundant co-observation counts are added to the summary)',
        fontsize=12, y=1.03)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_6_3-Coobs_Requirements.png')
    plt.close()



# =============================================================================
#  FIGURE 7 — Communication Load (messages, not runtime)
# =============================================================================

def plot_communication_load(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metrics = [
        ('Total Messages Broadcasted',
         'Total Messages Broadcasted',
         'Total_Messages', 1),
        ('P(Message Broadcasted | Bid Message)',
         'P(Bid Message | All Messages)',
         'Bid_Message_Ratio', 2),
    ]
    data = df[df['Algorithm'] != 'None-None'].copy()

    for metric, ylabel, slug, sub_idx in metrics:
        if not _col_ok(data, metric, 'Fig 7'):
            continue

        n_missions = len(MISSION_ORDER)
        n_conn     = len(CONN_ORDER)
        fig, axes  = plt.subplots(
            n_missions, n_conn,
            figsize=(4.5 * n_conn, 4 * n_missions),
            sharey='row')

        for row_i, mission in enumerate(MISSION_ORDER):
            for col_i, conn in enumerate(CONN_ORDER):
                ax  = axes[row_i][col_i]
                sub = data[(data['Mission'] == mission) &
                           (data['Connectivity'] == conn)]
                make_boxplot(sub, metric, ax,
                             ylabel=ylabel if col_i == 0 else '',
                             annotate_winner=False)
                if row_i == 0:
                    ax.set_title(CONN_FULL_LABELS[conn], fontsize=8,
                                 fontweight='bold')
                if col_i == n_conn - 1:
                    ax.yaxis.set_label_position('right')
                    ax.set_ylabel(MISSION_FULL_LABELS[mission], fontsize=8,
                                  rotation=270, labelpad=14, va='bottom')

        plt.suptitle(f'Communication Load — {ylabel}', fontsize=12, y=1.01)
        plt.tight_layout()
        save_plot(save_dir, f'Plot4_7_{sub_idx}-{slug}.png')
        plt.close()


# =============================================================================
#  FIGURE 8 — Seasonal Sensitivity
#  Lines per Algorithm, one panel per Mission.
#  August dates marked with a different marker (peak fire season).
# =============================================================================

def plot_seasonal_sensitivity(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Total Obtained Reward'
    data   = df[df['Algorithm'] != 'None-None'].copy()
    dates  = sorted(data['Date'].dropna().unique())

    if len(dates) < 2:
        print('  [Fig 8] Skipped -- requires multiple dates.')
        return

    fig, axes = plt.subplots(
        1, len(MISSION_ORDER),
        figsize=(6 * len(MISSION_ORDER), 5), sharey=False)

    for ax, mission in zip(axes, MISSION_ORDER):
        sub = data[data['Mission'] == mission].sort_values('Date')

        for algo in ALGO_ORDER:
            asub  = sub[sub['Algorithm'] == algo]
            means = asub.groupby('Date', observed=True)[metric].mean()
            q25   = asub.groupby('Date', observed=True)[metric].quantile(0.25)
            q75   = asub.groupby('Date', observed=True)[metric].quantile(0.75)

            d_list = [d for d in dates if d in means.index]
            ys     = [means[d] for d in d_list]
            lo     = [q25[d]   for d in d_list]
            hi     = [q75[d]   for d in d_list]
            if not ys:
                continue

            spec = ALGO_LINESTYLES[algo]
            xs = list(range(len(d_list)))

            # Draw line first
            line, = ax.plot(xs, ys,
                            color=ALGO_PALETTE[algo], linewidth=1.6,
                            linestyle='solid' if spec == '' else 'dashed',
                            label=algo, zorder=2)
            _apply_linestyle(line, algo)
            ax.fill_between(xs, lo, hi,
                            color=ALGO_PALETTE[algo], alpha=0.10)

            # Overlay markers — star for August, circle otherwise
            for xi, (d, y) in enumerate(zip(d_list, ys)):
                is_august = d[5:7] == '08'   # check MM portion of YYYY-MM-DD
                mkr = '*' if is_august else 'o'
                mks = 11  if is_august else 5
                ax.plot(xi, y, marker=mkr, markersize=mks,
                        color=ALGO_PALETTE[algo],
                        markeredgecolor='white' if not is_august else 'black',
                        markeredgewidth=0.4, zorder=4)

        ax.set_xticks(range(len(d_list)))
        ax.set_xticklabels(d_list, rotation=35, ha='right', fontsize=7)
        ax.set_xlabel('Simulation Date', fontsize=9)
        ax.set_ylabel('Total Obtained Reward'
                      if mission == MISSION_ORDER[0] else '')
        ax.set_title(MISSION_FULL_LABELS[mission], fontsize=10,
                     fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        if mission == MISSION_ORDER[-1]:
            ax.legend(title='Algorithm', fontsize=7,
                      title_fontsize=8, loc='best')

    # Star marker note
    fig.text(0.5, -0.03,
             '★ = August date (peak fire season)',
             ha='center', fontsize=8, color='#555555', style='italic')

    plt.suptitle(
        'Seasonal Sensitivity of Mission Reward\n'
        '(shading = IQR across connectivity × data processing)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_8_1-Seasonal_Sensitivity.png')
    plt.close()


# =============================================================================
#  FIGURE 1b — Unified MILP-Normalised Reward by Algorithm
#  Single plot, all missions combined, no faceting by connectivity/DP/date.
#  Analogous to the prior experiment's normalised reward overview.
#  MILP excluded from bars; gray dashed line at y=1.0 is the MILP reference.
# =============================================================================

def plot_unified_normalised_reward(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Reward / MILP'
    data   = df[(df['Algorithm'] != 'None-None') &
                (df['Algorithm'] != 'MILP')].copy()

    if metric not in data.columns or data[metric].isna().all():
        print('  [Fig 1b] Reward / MILP column absent or all-NaN — skipping.')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    make_boxplot(data, metric, ax,
                 ylabel='Reward / MILP Reward',
                 milp_line=True)

    ax.axhline(1.0, color=ALGO_PALETTE['MILP'], linewidth=1.4,
               linestyle='--', alpha=0.8)
    ax.text(len(_algo_present(data)) - 0.5, 1.01,
            'MILP = 1.0', fontsize=7.5,
            color=ALGO_PALETTE['MILP'], ha='right', va='bottom')
    ax.set_xlabel('Algorithm Configuration', fontsize=10)

    # Add legend for MILP reference line
    milp_handle = matplotlib.lines.Line2D(
        [], [], color=ALGO_PALETTE['MILP'], linewidth=1.4,
        linestyle='--', alpha=0.8, label='MILP reference (1.0)')
    ax.legend(handles=[milp_handle], fontsize=8, loc='upper left')

    plt.suptitle(
        'Mission Reward by Algorithm — MILP-Normalised\n'
        '(all missions, connectivity levels, data processing modes combined)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_1b_1-Mission_Reward_Norm_Unified.png')
    plt.close()

def plot_unified_reward(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Total Obtained Reward'
    data   = df[df['Algorithm'] != 'None-None'].copy()

    if metric not in data.columns or data[metric].isna().all():
        print('  [Fig 1b] Total Obtained Reward column absent or all-NaN — skipping.')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    make_boxplot(data, metric, ax,
                 ylabel='Total Obtained Reward',
                 milp_line=True)

    ax.axhline(1.0, color=ALGO_PALETTE['MILP'], linewidth=1.4,
               linestyle='--', alpha=0.8)
    ax.text(len(_algo_present(data)) - 0.5, 1.01,
            'MILP = 1.0', fontsize=7.5,
            color=ALGO_PALETTE['MILP'], ha='right', va='bottom')
    ax.set_xlabel('Algorithm Configuration', fontsize=10)

    # Add legend for MILP reference line
    milp_handle = matplotlib.lines.Line2D(
        [], [], color=ALGO_PALETTE['MILP'], linewidth=1.4,
        linestyle='--', alpha=0.8, label='MILP reference (1.0)')
    ax.legend(handles=[milp_handle], fontsize=8, loc='upper left')

    plt.suptitle(
        'Mission Reward by Algorithm — MILP-Normalised\n'
        '(all missions, connectivity levels, data processing modes combined)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_1b_2-Mission_Reward_Unified.png')
    plt.close()


# =============================================================================
#  FIGURE 2b — Normalised Decomposition Heatmap
#  Same 3×3 Preplanner×Replanner grid as Fig 2, but values are Reward / MILP.
#  MILP row is retained since MILP is the reference — it will show 1.0.
#  Color scale: yellow = below MILP, red = above MILP, centred at 1.0.
# =============================================================================

def plot_normalised_decomposition_heatmap(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Reward / MILP'

    row_labels = ['MILP\n(Centralized)', 'DP\n(Onboard)', 'None\n(No Pre)']
    col_labels  = ['No\nReplanner', 'Greedy', 'SC-CBBA']

    algo_cell = {
        'MILP':       (0, 0),
        'DP':         (1, 0),
        'DP-GR':      (1, 1),
        'DP-SC-CBBA': (1, 2),
        'None-None':  (2, 0),
        'GR':         (2, 1),
        'SC-CBBA':    (2, 2),
    }

    n_missions = len(MISSION_ORDER)
    n_dp       = len(DP_ORDER)

    # Centre colour scale at 1.0 (MILP reference)
    vals_all = df[metric].dropna()
    if vals_all.empty:
        print('  [Fig 2b] Reward / MILP column absent — skipping.')
        return
    vmax = max(abs(vals_all.max() - 1.0), abs(vals_all.min() - 1.0)) + 0.05
    # Diverging: below 1.0 = yellow/white, above 1.0 = orange/red
    # Use RdYlGn reversed so red = high, centred at 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=1.0 - vmax,
                                           vcenter=1.0,
                                           vmax=1.0 + vmax)
    cmap = 'RdYlGn'   # green = above MILP, red = below MILP

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(
        n_missions, n_dp,
        figsize=(4.5 * n_dp, 4.2 * n_missions),
        sharey='row')

    for row_i, mission in enumerate(MISSION_ORDER):
        mission_sub = df[df['Mission'] == mission]

        for col_i, dp in enumerate(DP_ORDER):
            ax  = axes[row_i][col_i]
            sub = mission_sub[mission_sub['Data Processing'] == dp]
            grid = np.full((3, 3), np.nan)
            text = [[''] * 3 for _ in range(3)]

            for algo, (ri, ci) in algo_cell.items():
                vals = sub[sub['Algorithm'] == algo][metric].dropna()
                if len(vals):
                    m, s = vals.mean(), vals.std(ddof=0)
                    grid[ri, ci] = m
                    text[ri][ci] = f'{m:.2f}\n±{s:.2f}'

            masked = np.ma.masked_invalid(grid)
            ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

            for ri in range(3):
                for ci in range(3):
                    if text[ri][ci]:
                        val = masked[ri, ci]
                        # dark text near centre, light text at extremes
                        fc = 'black' if abs(val - 1.0) < vmax * 0.5 else 'white'
                        ax.text(ci, ri, text[ri][ci],
                                ha='center', va='center',
                                fontsize=7, color=fc)

            # None-None: gray + hatch
            nn_r, nn_c = algo_cell['None-None']
            ax.add_patch(plt.Rectangle(
                (nn_c - 0.5, nn_r - 0.5), 1, 1,
                facecolor='#CCCCCC', hatch='//',
                edgecolor='#777777', linewidth=0.8, zorder=2))
            ax.text(nn_c, nn_r, 'Passive\n(ref.)',
                    ha='center', va='center', fontsize=6,
                    color='#444444', style='italic', zorder=3)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(col_labels, fontsize=7)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_xlabel('Replanner', fontsize=8)
            if col_i == 0:
                ax.set_ylabel('Preplanner', fontsize=8)
            ax.axhline(0.5, color='white', linewidth=2)

            if row_i == 0:
                ax.set_title(DP_LABELS[dp], fontsize=9, fontweight='bold')
            if col_i == n_dp - 1:
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(MISSION_FULL_LABELS[mission], fontsize=9,
                              rotation=270, labelpad=16, va='bottom')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(sm, cax=cbar_ax, label='Reward / MILP Reward')
    cb.ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--')
    cb.ax.text(2.5, 1.0, 'MILP', fontsize=6, va='center', color='black')

    plt.suptitle(
        'Reward Decomposition (MILP-Normalised): Preplanner × Replanner\n',
        # '(green = above MILP; red = below MILP; rows = Mission; '
        # 'cols = Data Processing)',
        fontsize=12, y=1.01)
    save_plot(save_dir, 'Plot4_2b_1-Decomposition_Heatmap_Norm.png')
    plt.close()



# =============================================================================
#  FIGURE 2c — Cumulative Normalised Heatmap
#  Single 3×3 Preplanner×Replanner grid, all missions/connectivity/DP/dates
#  collapsed into one mean Reward/MILP value per cell.
#  Diverging colour scale centred at 1.0 (MILP reference).
# =============================================================================

def plot_cumulative_normalised_heatmap(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    metric = 'Reward / MILP'
    if metric not in df.columns or df[metric].isna().all():
        print('  [Fig 2c] Reward / MILP absent — skipping.')
        return

    row_labels = ['MILP\n(Centralized)', 'DP\n(Onboard)', 'None\n(No Pre)']
    col_labels  = ['No\nReplanner', 'Greedy', 'SC-CBBA']

    algo_cell = {
        'MILP':       (0, 0),
        'DP':         (1, 0),
        'DP-GR':      (1, 1),
        'DP-SC-CBBA': (1, 2),
        'None-None':  (2, 0),
        'GR':         (2, 1),
        'SC-CBBA':    (2, 2),
    }

    grid = np.full((3, 3), np.nan)
    text = [[''] * 3 for _ in range(3)]

    for algo, (ri, ci) in algo_cell.items():
        vals = df[df['Algorithm'] == algo][metric].dropna()
        if len(vals):
            m, s = vals.mean(), vals.std(ddof=0)
            n    = len(vals)
            grid[ri, ci] = m
            text[ri][ci] = f'{m:.2f}\n±{s:.2f}\n(n={n})'

    vals_all = df[metric].dropna()
    vmax = max(abs(vals_all.max() - 1.0), abs(vals_all.min() - 1.0)) + 0.05
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=1.0 - vmax, vcenter=1.0, vmax=1.0 + vmax)
    cmap = 'RdYlGn'

    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

    for ri in range(3):
        for ci in range(3):
            if text[ri][ci]:
                val = masked[ri, ci]
                fc = 'black' if abs(val - 1.0) < vmax * 0.5 else 'white'
                ax.text(ci, ri, text[ri][ci],
                        ha='center', va='center', fontsize=9, color=fc)

    # None-None: gray + hatch
    nn_r, nn_c = algo_cell['None-None']
    ax.add_patch(plt.Rectangle(
        (nn_c - 0.5, nn_r - 0.5), 1, 1,
        facecolor='#CCCCCC', hatch='//',
        edgecolor='#777777', linewidth=0.8, zorder=2))
    ax.text(nn_c, nn_r, 'Passive\n(ref.)',
            ha='center', va='center', fontsize=7,
            color='#444444', style='italic', zorder=3)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Replanner', fontsize=10)
    ax.set_ylabel('Preplanner', fontsize=10)
    ax.axhline(0.5, color='white', linewidth=2)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, label='Reward / MILP Reward')
    cb.ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--')
    # cb.ax.text(2.8, 1.0, 'MILP', fontsize=7, va='center', color='black')

    plt.suptitle(
        'Cumulative Normalised Reward: Preplanner × Replanner\n'
        '(all missions, connectivity levels, data processing modes combined;\n'
        ' green = above MILP, red = below MILP)',
        fontsize=11, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_2c_1-Cumulative_Normalised_Heatmap.png')
    plt.close()

# =============================================================================
#  FIGURE 6d — Co-observations: Strategy Analysis  (1 row × 3 cols)
#  All connectivity levels collapsed into one plot per panel.
#  y-axis: P(Event Co-observed | Co-observable) — coverage probability
#  x-axes:
#    Panel 1: Event Co-observations          (total volume)
#    Panel 2: Unique Event Co-observations   (genuine coordination)
#    Panel 3: Repeated Event Co-observations (redundant passes)
#  Colour = algorithm.
# =============================================================================
 
def plot_coobs_strategy(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    y_col = 'P(Event Co-observed | Co-observable)'
    panels = [
        ('Event Co-observations',
         'Total Tasked Event Co-observations\n(all passes)'),
        ('Unique Tasked Event Co-observations',
         'Unique Tasked Event Co-observations\n(genuine coordination)'),
        ('Repeated Tasked Event Co-observations',
         'Repeated Tasked Event Co-observations\n(redundant passes)'),
    ]
 
    data = df[(df['Mission'] == 'Co-observations') &
              (df['Algorithm'] != 'None-None')].copy()
 
    for col, _ in panels:
        if not _col_ok(data, col, 'Fig 6d'):
            return
    if not _col_ok(data, y_col, 'Fig 6d'):
        return
 
    fig, axes = plt.subplots(
        1, 3,
        figsize=(6 * 3, 5),
        sharey=True,
        # sharex=True
    )
 
    for ax, (x_col, xlabel) in zip(axes, panels):
        for algo in ALGO_ORDER:
            asub = data[data['Algorithm'] == algo].dropna(
                subset=[x_col, y_col])
            if asub.empty:
                continue
            ax.scatter(asub[x_col], asub[y_col],
                       color=ALGO_PALETTE[algo],
                    #    marker=MARKER_SHAPES[algo],
                       s=70, alpha=0.70,
                       edgecolors='white', linewidths=0.4,
                       zorder=3, label=algo)
 
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('P(Event Co-observed | Co-observable)'
                      if ax is axes[0] else '')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
 
    # Legend on last panel
    algo_handles = [
        plt.scatter([], [], color=ALGO_PALETTE[a],
                    #    marker=MARKER_SHAPES[a],
                       s=70, label=a)
        for a in ALGO_ORDER
    ]
    axes[-1].legend(handles=algo_handles, title='Algorithm',
                    fontsize=7, title_fontsize=8, loc='best')
 
    plt.suptitle(
        f'{MISSION_FULL_LABELS["Co-observations"]} — Scheduling Strategy\n',
        # f'(all connectivity levels combined; colour = algorithm)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_6d_1-Coobs_Strategy.png')
    plt.close()

     
# def plot_coobs_strategy(
#     df: pd.DataFrame,
#     save_dir: str,
# ) -> None:
    x_col = 'P(Tasked Co-observation Unique)'
    y_col = 'P(Event Co-observed | Co-observable)'
 
    data = df[(df['Mission'] == 'Co-observations') &
              (df['Algorithm'] != 'None-None')].copy()
 
    if not _col_ok(data, x_col, 'Fig 6d') or \
       not _col_ok(data, y_col, 'Fig 6d'):
        return
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    for algo in ALGO_ORDER:
        asub = data[data['Algorithm'] == algo].dropna(subset=[x_col, y_col])
        if asub.empty:
            continue
        ax.scatter(asub[x_col], asub[y_col],
                   color=ALGO_PALETTE[algo],
                #    marker=MARKER_SHAPES[algo],
                   s=80, alpha=0.75,
                   edgecolors='white', linewidths=0.4,
                   zorder=3, label=algo)
 
    ax.set_xlabel(
        'P(Tasked Co-observation Unique)\n'
        '← carpet-bombing                              efficient coordination →',
        fontsize=9)
    ax.set_ylabel(
        'P(Event Tasked Co-observed | Co-observable)\n'
        '(tasked coverage of co-observable events)',
        fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
 
    algo_handles = [
        plt.scatter([], [], color=ALGO_PALETTE[a],
                    # marker=MARKER_SHAPES[a], 
                    s=80, label=a)
        for a in ALGO_ORDER
    ]
    ax.legend(handles=algo_handles, title='Algorithm',
              fontsize=8, title_fontsize=9, loc='best')
 
    plt.suptitle(
        f'{MISSION_FULL_LABELS["Co-observations"]} — '
        f'Tasked Coverage vs Coordination Efficiency\n'
        f'(all connectivity levels and data processing modes combined)',
        fontsize=11, y=1.02)
    plt.tight_layout()
    save_plot(save_dir, 'Plot4_6d_2-Coobs_Strategy_Merged.png')
    plt.close()



# =============================================================================
#  MAIN DRIVER
# =============================================================================

def generate_plots(
    csv_path: str,
    trial_name: str,
    trial_csv_path: str | None = None,
    filter_date: str | None = None,
    complete_date: str = '2019-02-15',
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df_all   = load_and_prepare(csv_path, trial_csv_path=trial_csv_path)
    date_str = datetime.now().strftime('%Y-%m-%d')

    base_dir = os.path.join(
        'experiments', '2_centralized_vs_decentralized', 'analysis')
    save_dir = os.path.join(
        base_dir, 'plots', f'{trial_name}_P{date_str}')
    os.makedirs(save_dir, exist_ok=True)

    df = df_all[df_all['Date'] == filter_date].copy() \
        if filter_date else df_all

    print(f'\n{"="*65}')
    print(f'  Trial: {trial_name}')
    print(f'  Rows:  {len(df)} trials')
    print(f'  Date filter: {filter_date or "all"}')
    print(f'  Output: {save_dir}')
    print(f'{"="*65}')

    print('  Fig 1a -- Unified Reward (all conditions)')
    plot_unified_reward(df, save_dir)

    print('  Fig 1b -- Unified Normalised Reward (all conditions)')
    plot_unified_normalised_reward(df, save_dir)

    print('  Fig 1 -- Mission Reward Degradation (all + per-date)')
    plot_mission_degradation(df, save_dir)

    print('  Fig 2 -- Decomposition Heatmap (supplemental)')
    plot_decomposition_heatmap(df_all, save_dir)

    print('  Fig 2b -- Normalised Decomposition Heatmap')
    plot_normalised_decomposition_heatmap(df_all, save_dir)

    print('  Fig 2c -- Cumulative Normalised Heatmap')
    plot_cumulative_normalised_heatmap(df_all, save_dir)

    print('  Fig 3 -- Two-Strategy Scatter (3 rows x 2 cols)')
    plot_two_strategy_scatter(df, save_dir)

    print('  Fig 4 -- Data Processing Effect')
    plot_data_processing_effect(df_all, save_dir,
                                complete_date=complete_date)

    print('  Fig 5 -- Connectivity Effect (per mission)')
    plot_connectivity_effect(df, save_dir)

    print('  Fig 6a -- Urgency Requirements')
    plot_urgency_requirements(df, save_dir)

    print('  Fig 6b -- Revisit Requirements')
    plot_revisit_requirements(df, save_dir)

    print('  Fig 6c -- Co-observations Requirements')
    plot_coobs_requirements(df, save_dir)

    print('  Fig 6d -- Co-observations Strategy (unique vs repeated)')
    plot_coobs_strategy(df, save_dir)

    print('  Fig 7 -- Communication Load')
    plot_communication_load(df, save_dir)

    print('  Fig 8 -- Seasonal Sensitivity')
    plot_seasonal_sensitivity(df, save_dir)

    print('\nDONE.')


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate analysis figures for the SC-CBBA study.')
    parser.add_argument(
        '--csv',
        default=os.path.join(
            'experiments', '2_centralized_vs_decentralized', 'analysis',
            'compiled',
            'full_factorial_trials_2026-05-25_compiled_results.csv'))
    parser.add_argument(
        '--trial-name', default='full_factorial_trials_2026-05-25')
    parser.add_argument(
        '--trial-csv',
        default=os.path.join(
            'experiments', '2_centralized_vs_decentralized', 'resources',
            'trials', 'full_factorial_trials_2026-05-25.csv'),
        help='Trial definition CSV for completeness filtering')
    parser.add_argument(
        '--date', default=None,
        help='Filter to a single simulation date (e.g. 2019-02-15)')
    parser.add_argument(
        '--complete-date', default='2019-02-15',
        help='Fully-complete date for Fig 4 (default: 2019-02-15)')
    args = parser.parse_args()

    generate_plots(
        csv_path=args.csv,
        trial_name=args.trial_name,
        trial_csv_path=args.trial_csv,
        filter_date=args.date,
        complete_date=args.complete_date,
    )