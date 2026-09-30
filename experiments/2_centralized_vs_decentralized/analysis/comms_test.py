"""
plot_rq2_figures.py
-------------------
Reproduces all four RQ2 figures for Chapter 4 of the dissertation.

Inputs (edit paths below if needed):
  - full_factorial_trials_2026-05-25_compiled_results.csv
  - accesses_per_task.parquet
  - all_tasks.parquet
  - intervals.npy
  - meta.json

Outputs (saved to OUTPUT_DIR):
  - fig_rq2_connectivity_stability.png
  - fig_rq2_sq_normalized.png
  - fig_rq2_causal_explanation.png
  - fig_rq2_gs_gaps.png

Dependencies:
  pip install pandas numpy matplotlib pyarrow
"""

import json
import os
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

# ================================================================
# PATHS — edit these to match your local directory structure
# ================================================================
ROOT_DIR = os.path.join('.', 'experiments', '2_centralized_vs_decentralized', 'analysis')
DATA_DIR   = os.path.join(ROOT_DIR, 'access_sample')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'plots', 'rq2')

CSV_PATH       = os.path.join(ROOT_DIR, 'compiled', 'full_factorial_trials_2026-05-25_compiled_results.csv')
ACCESSES_PATH  = os.path.join(DATA_DIR, 'accesses_per_task.parquet')
TASKS_PATH     = os.path.join(DATA_DIR, 'all_tasks.parquet')
INTERVALS_PATH = os.path.join(DATA_DIR, 'intervals.npy')
META_PATH      = os.path.join(DATA_DIR, 'meta.json')

# ================================================================
# CONSTANTS — palette, ordering, labels
# ================================================================
ALGO_PALETTE = {
    'None-None':  '#BBBBBB',
    'MILP':       '#999999',
    'DP':         '#D55E00',
    'DP-GR':      '#E69F00',
    'GR':         '#009E73',
    'DP-SC-CBBA': '#0072B2',
    'SC-CBBA':    '#56B4E9',
}

ALGO_ORDER = ['MILP', 'GR', 'DP', 'DP-GR', 'SC-CBBA', 'DP-SC-CBBA']
MISSION_ORDER = ['Urgency', 'Revisits', 'Co-observations']
CONN_ORDER    = ['GS', 'Intraconstellation', 'Interconstellation']
CONN_LABELS   = {
    'GS':                 'Ground',
    'Intraconstellation': 'Intraconst.',
    'Interconstellation': 'Interconst.',
}
DET_ORDER = ['Ground', 'Onboard', 'Instant']

GS_NAMES = [
    'GS Planner',
    'GS Announcer (Algal Blooms)',
    'GS Announcer (High Flow Rivers)',
    'GS Announcer (Wildfires)',
]

SIM_DURATION = 86400.0  # seconds in one simulation day


# ================================================================
# HELPERS
# ================================================================
def line_style(p):
    return '--' if p == 'MILP' else '-'


def line_width(p):
    return 2.5 if p in ['SC-CBBA', 'DP-SC-CBBA', 'MILP'] else 1.5


def load_and_prepare_csv(path):
    """Load the compiled results CSV and add normalized reward columns."""
    df = pd.read_csv(path)
    df['Planner'] = df['Preplanner'].fillna('None') + '+' + df['Replanner'].fillna('None')

    planner_map = {
        'Centralized-MILP_priority+None': 'MILP',
        'DP+None':    'DP',
        'None+Greedy':'GR',
        'DP+Greedy':  'DP-GR',
        'None+CBBA':  'SC-CBBA',
        'DP+CBBA':    'DP-SC-CBBA',
    }
    df['Planner_Label'] = df['Planner'].map(planner_map)

    # Scenario-matched normalization: divide by MILP reward in same scenario
    milp_scenario = (
        df[df['Planner_Label'] == 'MILP']
        [['Mission', 'Data Processing', 'Date', 'Connectivity', 'Total Obtained Reward']]
        .rename(columns={'Total Obtained Reward': 'MILP_Reward'})
    )
    df = df.merge(milp_scenario,
                  on=['Mission', 'Data Processing', 'Date', 'Connectivity'],
                  how='left')
    df['Norm_Scenario'] = df['Total Obtained Reward'] / df['MILP_Reward']

    # Status-quo normalization: divide by Ground-comms Ground-detection MILP,
    # matched by date and mission only
    sq_ref = (
        df[
            (df['Planner_Label'] == 'MILP') &
            (df['Connectivity'] == 'GS') &
            (df['Data Processing'] == 'Ground')
        ]
        [['Date', 'Mission', 'Total Obtained Reward']]
        .rename(columns={'Total Obtained Reward': 'SQ_Reward'})
    )
    df = df.merge(sq_ref, on=['Date', 'Mission'], how='left')
    df['Norm_SQ'] = df['Total Obtained Reward'] / df['SQ_Reward']

    return df


def compute_coverage_geometry(accesses_path, tasks_path):
    """Return normalized earliest access time per accessible task."""
    accesses = pd.read_parquet(accesses_path)
    tasks    = pd.read_parquet(tasks_path)

    earliest = (
        accesses.groupby('task id')['t start']
        .min()
        .reset_index()
        .rename(columns={'task id': 'id', 't start': 'earliest_access'})
    )
    merged = tasks.merge(earliest, on='id', how='left')
    merged['window_duration']    = merged['t end'] - merged['t start']
    merged['earliest_resp_norm'] = (
        (merged['earliest_access'] - merged['t start']) / merged['window_duration']
    )
    return merged[merged['earliest_access'].notna()].copy()


def compute_gs_gaps(intervals_path, meta_path):
    """Return array of per-satellite maximum gap between GS contacts (seconds)."""
    intervals = np.load(intervals_path, allow_pickle=True)
    with open(meta_path) as f:
        meta = json.load(f)

    u_vocab = meta['columns']['u']['vocab']
    v_vocab = meta['columns']['v']['vocab']

    df_conn = pd.DataFrame(
        intervals,
        columns=['start', 'end', 'prefix_max_end', 'u_code', 'v_code']
    )
    df_conn['u'] = df_conn['u_code'].astype(int).astype(str).map(u_vocab)
    df_conn['v'] = df_conn['v_code'].astype(int).astype(str).map(v_vocab)

    all_nodes  = set(df_conn['u'].dropna()) | set(df_conn['v'].dropna())
    satellites = [
        n for n in all_nodes
        if n not in GS_NAMES and 'tdrss' not in str(n).lower()
    ]

    max_gaps = []
    for sat in satellites:
        contacts = df_conn[
            ((df_conn['u'] == sat) & df_conn['v'].isin(GS_NAMES)) |
            ((df_conn['v'] == sat) & df_conn['u'].isin(GS_NAMES))
        ][['start', 'end']].values

        if len(contacts) == 0:
            max_gaps.append(SIM_DURATION)
            continue

        contacts = contacts[contacts[:, 0].argsort()]
        gaps = [contacts[0][0]]
        for i in range(1, len(contacts)):
            gaps.append(contacts[i][0] - contacts[i - 1][1])
        gaps.append(SIM_DURATION - contacts[-1][1])
        max_gaps.append(max(gaps))

    return np.array(max_gaps), len(satellites)


# ================================================================
# FIGURE 1: Connectivity stability (scenario-matched normalization)
# ================================================================
def plot_connectivity_stability(full, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Planner Performance by Communication Architecture\n'
        'Scenario-matched normalized reward (MILP\u00a0=\u00a01 within each scenario)\n\n',
        fontsize=11,
    )

    for mi, mission in enumerate(MISSION_ORDER):
        ax = axes[mi]
        sub = full[full['Mission'] == mission]

        for p in ALGO_ORDER:
            means = [
                sub[(sub['Planner_Label'] == p) & (sub['Connectivity'] == c)
                    ]['Norm_Scenario'].mean()
                for c in CONN_ORDER
            ]
            stds = [
                sub[(sub['Planner_Label'] == p) & (sub['Connectivity'] == c)
                    ]['Norm_Scenario'].std()
                for c in CONN_ORDER
            ]
            ax.plot(
                range(len(CONN_ORDER)), means,
                color=ALGO_PALETTE[p], linewidth=line_width(p),
                linestyle=line_style(p), marker='o', markersize=6, label=p,
            )
            ax.fill_between(
                range(len(CONN_ORDER)),
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=ALGO_PALETTE[p], alpha=0.08,
            )

        ax.axhline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(f"{mission} Priority Mission", fontweight='bold')
        ax.set_xticks(range(len(CONN_ORDER)))
        ax.set_xticklabels([CONN_LABELS[c] for c in CONN_ORDER], fontsize=9)
        ax.set_ylabel('Norm. reward (MILP\u00a0=\u00a01)')
        ax.set_ylim(0.6, 1.5)
        ax.grid(alpha=0.3)


        if mi == 1:
            ax.annotate(
                'Relative standing\nstable across\narchitectures',
                xy=(1, 1.10), xytext=(0.2, 1.38),
                fontsize=7, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor='#CCCCCC', alpha=0.8),
            )
            # ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.42, 1.0))

    handles = [mpatches.Patch(facecolor=ALGO_PALETTE[p], label=p)
               for p in ALGO_ORDER]
    fig.legend(handles=handles, title='Planner', loc='upper center',
               ncol=6, fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.91), framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig_rq2_connectivity_stability.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

PARAM_LABELS: Dict[str, str] = {
    'temperature [k]':                      'Temperature',
    'fire extent':                          'Fire extent',
    'water level':                          'Water level',
    'turbidity':                            'Turbidity',
    'chlorophyll-a concentration [mg/m^3]': 'Chlorophyll-a',
}
 
# X-axis clip: display up to this many minutes (24 hr = 1440 min)
CLIP_MIN = 1440
 
 
# ================================================================
# FIGURE
# ================================================================
# Ground-station median maximum inter-contact gap per constellation type
# (from Table tab:rq2_gs_reachability in the thesis)
SSO_GS_GAP_MIN   = 36   # minutes — water-quality and wildfire constellations
FLOOD_GS_GAP_MIN = 87   # minutes — flood constellation (mid-inclination orbit)
 
# Co-observation decorrelation window (seconds)
T_CORR_S = 300
 
# Fraction of short fire events geometrically infeasible for co-observation
# (computed separately from access data; see discussion in thesis)
INFEASIBLE_FRAC = 0.979 

# Colour mapping per measurement parameter
PARAM_COLORS: Dict[str, str] = {
    'temperature [k]':                      '#EF5350',
    'fire extent':                          '#FF9800',
    'water level':                          '#2196F3',
    'turbidity':                            '#009688',
    'chlorophyll-a concentration [mg/m^3]': '#4CAF50',
}

def plot_duration_gs_annotated(tasks_path: Path, output_dir: Path) -> None:
    tasks = pd.read_parquet(tasks_path)
    tasks['duration_s']   = tasks['t end'] - tasks['t start']
    tasks['duration_min'] = tasks['duration_s'] / 60.0
 
    bins   = np.linspace(0, CLIP_MIN, 60)
    bottom = np.zeros(len(bins) - 1)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        'Event Duration Distribution and Ground-Station Reachability Thresholds\n'
        '(representative trial: 2019-05-15; '
        'x-axis clipped at 24\u00a0hr for readability)',
        fontsize=11,
    )
 
    # Stacked histogram by parameter
    for param, color in PARAM_COLORS.items():
        sub    = tasks[tasks['parameter'] == param]['duration_min'].clip(0, CLIP_MIN)
        if sub.empty:
            continue

        counts, _ = np.histogram(sub, bins=bins)
        ax.bar(bins[:-1], counts, width=bins[1] - bins[0],
               bottom=bottom, color=color, alpha=0.80,
               edgecolor='white', linewidth=0.3,
               label=PARAM_LABELS[param])
        bottom += counts
 
    # GS gap threshold lines
    ax.axvline(SSO_GS_GAP_MIN, color='#D55E00', linewidth=2, linestyle='--',
               label=f'SSO GS median gap ({SSO_GS_GAP_MIN}\u00a0min)')
    ax.axvline(FLOOD_GS_GAP_MIN, color='#0072B2', linewidth=2, linestyle='--',
               label=f'Flood GS median gap ({FLOOD_GS_GAP_MIN}\u00a0min)')
 
    # Shade the at-risk zone (shorter than the worst-case GS gap)
    ax.axvspan(0, FLOOD_GS_GAP_MIN, alpha=0.08, color='#EF5350')
 
    # Fraction of all tasks shorter than flood GS gap
    frac_below_flood = (tasks['duration_min'] < FLOOD_GS_GAP_MIN).mean()
    ax.text(
        FLOOD_GS_GAP_MIN / 2,
        ax.get_ylim()[1] * 0.82,
        f'{frac_below_flood:.0%} of all tasks\nshorter than\nflood GS gap',
        ha='center', va='top', fontsize=8, color='#C62828',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#EF5350', alpha=0.9),
    )
 
    # Co-observation infeasibility annotation targeting the short fire cluster
    ax.annotate(
        f'{INFEASIBLE_FRAC:.1%} of short fire events\n'
        'are geometrically infeasible\n'
        f'for co-observation within {T_CORR_S}\u00a0s\n'
        'regardless of planner',
        xy=(10, 55),
        xytext=(120, 110),
        fontsize=8, color='#B71C1C',
        arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#EF5350', alpha=0.9),
    )
 
    ax.set_xlabel('Event duration (minutes)', fontsize=10)
    ax.set_ylabel('Number of tasks', fontsize=10)
    ax.set_xlim(0, CLIP_MIN)
    ax.set_xticks([0, 60, 120, 180, 240, 360, 480, 720, 1440])
    ax.set_xticklabels([
        '0', '1\u00a0hr', '2\u00a0hr', '3\u00a0hr',
        '4\u00a0hr', '6\u00a0hr', '8\u00a0hr',
        '12\u00a0hr', '24\u00a0hr',
    ])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    out = os.path.join(output_dir , 'fig_rq2_duration_gs_annotated.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')



# ================================================================
# FIGURE 2: Status-quo normalized reward by connectivity
# ================================================================
def plot_sq_normalized(full, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Planner Performance Relative to Operational Baseline\n'
        'Status-quo normalized reward'
        ' (Ground-comms Ground-detection MILP\u00a0=\u00a01, date-matched)\n\n',
        fontsize=11,
    )
 
    for mi, mission in enumerate(MISSION_ORDER):
        ax = axes[mi]
        sub = full[full['Mission'] == mission]
 
        for p in ALGO_ORDER:
            means = [
                sub[(sub['Planner_Label'] == p) & (sub['Connectivity'] == c)
                    ]['Norm_SQ'].mean()
                for c in CONN_ORDER
            ]
            stds = [
                sub[(sub['Planner_Label'] == p) & (sub['Connectivity'] == c)
                    ]['Norm_SQ'].std()
                for c in CONN_ORDER
            ]
            ax.plot(
                range(len(CONN_ORDER)), means,
                color=ALGO_PALETTE[p], linewidth=line_width(p),
                linestyle=line_style(p), marker='o', markersize=6, label=p,
            )
            ax.fill_between(
                range(len(CONN_ORDER)),
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=ALGO_PALETTE[p], alpha=0.08,
            )
 
        ax.axhline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(mission, fontweight='bold')
        ax.set_xticks(range(len(CONN_ORDER)))
        ax.set_xticklabels([CONN_LABELS[c] for c in CONN_ORDER], fontsize=9)
        ax.set_ylabel('Norm. reward (SQ baseline\u00a0=\u00a01)')
        ax.set_ylim(0.6, 2.0)
        ax.grid(alpha=0.3)
 
        milp_means = [
            sub[(sub['Planner_Label'] == 'MILP') & (sub['Connectivity'] == c)
                ]['Norm_SQ'].mean()
            for c in CONN_ORDER
        ]
 
        if mi == 2:
            ax.annotate(
                'MILP gains ~8%\nwith better comms\n(absolute reward effect)',
                xy=(2, milp_means[2]),
                xytext=(1.3, milp_means[2] - 0.50),
                fontsize=7, color=ALGO_PALETTE['MILP'],
                arrowprops=dict(arrowstyle='->', color=ALGO_PALETTE['MILP'], lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor='#CCCCCC', alpha=0.8),
            )
            # ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.42, 1.0))

    handles = [mpatches.Patch(facecolor=ALGO_PALETTE[p], label=p)
               for p in ALGO_ORDER]
    fig.legend(handles=handles, title='Planner', loc='upper center',
               ncol=6, fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.91), framealpha=0.9)
 
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig_rq2_sq_normalized.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")



# ================================================================
# FIGURE 3: Causal explanation (detection regime + coverage geometry)
# ================================================================
def plot_causal_explanation(full, accessible, output_dir):
    gs_only = full[full['Connectivity'] == 'GS']

    ann_means  = [gs_only[gs_only['Data Processing'] == d
                          ]['Average Event Announcement Time [norm]'].mean()
                  for d in DET_ORDER]
    resp_means = [gs_only[gs_only['Data Processing'] == d
                          ]['Average Normalized Response Time to Event'].mean()
                  for d in DET_ORDER]
    atr         = [max(r - a, 0) for r, a in zip(resp_means, ann_means)]
    ann_clipped = [max(a, 0)     for a in ann_means]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Information Latency and Planner Response\n'
        'Why detection regime affects absolute reward but not relative planner standing',
        fontsize=11,
    )

    # --- Left panel: bar decomposition ---
    ax = axes[0]
    x, w = np.arange(len(DET_ORDER)), 0.35

    ax.bar(x - w/2, ann_clipped, w, color='#FFCDD2', edgecolor='gray',
           label='Announcement latency (event start \u2192 task known)')
    ax.bar(x + w/2, atr,         w, color='#C8E6C9', edgecolor='gray',
           label='Planner response (task known \u2192 first observation)')

    ax.annotate(
        'Planners respond within\n~5% of window after\nannouncement',
        xy=(0 + w/2, atr[0]), xytext=(0.2, 0.22),
        fontsize=8, color='#2E7D32',
        arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#C8E6C9', alpha=0.9),
    )
    ax.annotate(
        'Under Instant detection planners\ntake ~49% of window:\norbital geometry is the floor',
        xy=(2 + w/2, atr[2]), xytext=(1.5, 0.55),
        fontsize=8, color='#1B5E20',
        arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#C8E6C9', alpha=0.9),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(DET_ORDER)
    ax.set_ylabel('Normalized time (fraction of event window)')
    ax.set_title('Ground-only comms, pooled over planners and missions')
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # --- Right panel: coverage geometry histogram ---
    ax = axes[1]
    ax.hist(accessible['earliest_resp_norm'].clip(0, 1), bins=40,
            color='#BBDEFB', edgecolor='gray', alpha=0.85,
            label='Tasks by earliest orbital access')
    ax.axvspan(0, 0.30, alpha=0.15, color='#EF5350')
    ax.axvline(0.30, color='#EF5350', linewidth=2, linestyle='--',
               label='Typical Ground/Onboard\nannouncement time (~30%)')
    ax.axvline(0.18, color='#FF9800', linewidth=2, linestyle='--',
               label='Earliest opportunity\nunder Instant (~18%)')

    frac = (accessible['earliest_resp_norm'] < 0.30).mean()
    ax.text(0.15, 52, f'{frac:.0%} of\naccessible\ntasks',
            ha='center', va='center', fontsize=8, color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#EF5350', alpha=0.9))

    ax.set_xlabel('Earliest access time (normalized to event window)')
    ax.set_ylabel('Number of tasks')
    ax.set_title('Constellation coverage geometry\n(sample trial: 2019-05-15)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig_rq2_causal_explanation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# FIGURE 4: Ground station gap distribution
# ================================================================
def plot_gs_gaps(max_gaps, n_satellites, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(max_gaps / 60, bins=25, color='#C8E6C9', edgecolor='gray', alpha=0.9)
    ax.axvline(np.median(max_gaps) / 60, color='#2E7D32', linewidth=2, linestyle='--',
               label=f'Median max gap: {np.median(max_gaps)/60:.0f} min')
    ax.axvline(max_gaps.max() / 60,      color='#EF5350', linewidth=2, linestyle='--',
               label=f'Worst-case gap: {max_gaps.max()/60:.0f} min')

    ax.annotate(
        f'All {n_satellites} satellites reach a GS\nwithin {max_gaps.max()/60:.0f} min at most',
        xy=(max_gaps.max() / 60, 5),
        xytext=(max_gaps.max() / 60 - 32, 18),
        fontsize=9, color='#C62828',
        arrowprops=dict(arrowstyle='->', color='#C62828', lw=1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#EF5350', alpha=0.9),
    )

    ax.set_xlabel('Maximum gap between ground station contacts (minutes)')
    ax.set_ylabel('Number of satellites')
    ax.set_title(
        'Ground station reachability per satellite\n'
        '(sample trial: 2019-05-15, ground-only connectivity)\n\n'
    )
    # ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # handles = [mpatches.Patch(facecolor=ALGO_PALETTE[p], label=p)
    #            for p in ALGO_ORDER]
    fig.legend(#handles=handles, title='Planner', 
               loc='upper center',
               ncol=6, fontsize=8, title_fontsize=9,
               bbox_to_anchor=(0.55, 0.875), framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig_rq2_gs_gaps.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("Loading data...")
    df   = load_and_prepare_csv(CSV_PATH)
    full = df[df['in_full'] == True].copy()

    print("Computing coverage geometry...")
    accessible = compute_coverage_geometry(ACCESSES_PATH, TASKS_PATH)

    print("Computing GS gaps...")
    max_gaps, n_sats = compute_gs_gaps(INTERVALS_PATH, META_PATH)

    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Plotting Figure 1: Connectivity stability...")
    plot_connectivity_stability(full, OUTPUT_DIR)

    print("Plotting Figure 2: Status-quo normalized...")
    plot_sq_normalized(full, OUTPUT_DIR)

    print("Plotting Figure 3: Causal explanation...")
    plot_causal_explanation(full, accessible, OUTPUT_DIR)

    print("Plotting Figure 4: GS gaps...")
    plot_gs_gaps(max_gaps, n_sats, OUTPUT_DIR)

    print('Plotting Figure 5: Duration and GS annotated...')
    plot_duration_gs_annotated(TASKS_PATH, OUTPUT_DIR)


    print("Done.")