"""
RQ3 plotting module: Effects of Communications Architecture.

Generates the three communications-architecture figures for the single-mission
study from the compiled full-factorial results:

  1. Plot3.3.1 - Mission reward vs. constellation size, all five planners,
                 in panels by communication architecture (latency).
  2. Plot3.3.X - Paired reward advantage of the SC-CBBA over each other
                 configuration, in panels by architecture.
  3. Plot3.3.Y - Reward change from adding the periodic preplanner to each
                 reactive planner, by architecture (paired, with 95% CIs).

Mirrors the structure of the RQ1 feasibility module (label_condition / save_plot /
date-stamped output dirs). The advantage and preplanner figures use paired
differences taken within matched (Num Sats, Task Arrival Rate, Scenario, Latency)
cells, so they are independent of the reward-normalization reference.
"""

from datetime import datetime
import os

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# ----------------------------------------------------------------------
#  LABELING / STYLE
# ----------------------------------------------------------------------
def label_condition(row):
    pre = row['Preplanner']
    rep = row['Replanner']
    if pre == 'DP' and pd.isna(rep):     return 'DP'
    if pd.isna(pre) and rep == 'Greedy': return 'GR'
    if pre == 'DP'  and rep == 'Greedy': return 'DP-GR'
    if pd.isna(pre) and rep == 'CBBA':   return 'CBBA'
    if pre == 'DP'  and rep == 'CBBA':   return 'DP-CBBA'
    return 'Unknown'

# internal label -> display label used in legends/titles
DISPLAY = {'DP': 'DP', 'GR': 'GR', 'DP-GR': 'DP-GR',
           'CBBA': 'SC-CBBA', 'DP-CBBA': 'DP-SC-CBBA'}

# fixed plotting order and colors (keyed by internal label)
ALGO_ORDER = ['DP', 'GR', 'DP-GR', 'CBBA', 'DP-CBBA']
PALETTE = {
    # 'DP':      '#7f7f7f',   # gray
    # 'GR':      '#D9A441',   # gold
    # 'DP-GR':   '#7FB3D5',   # light blue
    # 'CBBA':    '#1F8A70',   # green  (SC-CBBA)
    # 'DP-CBBA': '#2C6FB3',   # blue   (DP-SC-CBBA)
    'DP':          '#D55E00',   # vermillion   (preplanner only)
    'DP-GR':       '#E69F00',   # amber
    'GR':          '#009E73',   # teal
    'CBBA':     '#56B4E9',   # sky blue
    'DP-CBBA':  '#0072B2',   # deep blue
}

LATENCY_ORDER = ['Low', 'Medium', 'High']
REWARD_COL = 'Total Obtained Reward [norm]'
MATCH_KEY = ['Num Sats', 'Task Arrival Rate', 'Scenario', 'Latency']


def save_plot(base_dir, local_base_dir, save_dir, local_save_dir, plot_filename):
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        plt.savefig(local_save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: `{save_path}`" +
          (f" and `{local_save_path}`" if base_dir != local_base_dir else ""))


def _format_log_x(ax, num_sats):
    ax.set_xscale('log')
    ax.set_xticks(num_sats)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(min(num_sats), max(num_sats))


# ----------------------------------------------------------------------
#  PLOT 3.3.1 — Reward by architecture, all planners
# ----------------------------------------------------------------------
def plot_reward_by_latency(df, num_sats, paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lat in zip(axes, LATENCY_ORDER):
        sub = df[df['Latency'] == lat]
        agg = sub.groupby(['Algorithm', 'Num Sats'])[REWARD_COL].mean().reset_index()
        for a in ALGO_ORDER:
            s = agg[agg['Algorithm'] == a].sort_values('Num Sats')
            ax.plot(s['Num Sats'], s[REWARD_COL], marker='.', linewidth=1.6,
                    color=sns.desaturate(PALETTE[a], 0.75), label=DISPLAY[a])
        ax.set_title(f'Latency: {lat}')
        ax.set_xlabel('Number of Satellites')
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', linewidth=0.4)
        _format_log_x(ax, num_sats)


    axes[0].set_ylabel('Total Obtained Reward (normalized)')
    axes[-1].legend(title='Algorithm', fontsize=8)
    plt.suptitle('Mission Reward by Communication Architecture', fontsize=13)
    plt.tight_layout()
    save_plot(*paths, 'Plot3.3.1-Reward_by_Latency.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.3.X — SC-CBBA advantage over each baseline, by architecture
# ----------------------------------------------------------------------
def plot_advantage_retention(df, num_sats, paths, target='CBBA'):
    piv = df.pivot_table(index=MATCH_KEY, columns='Algorithm',
                         values=REWARD_COL).reset_index()
    baselines = [a for a in ALGO_ORDER if a != target]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lat in zip(axes, LATENCY_ORDER):
        p = piv[piv['Latency'] == lat]
        for b in baselines:
            d = pd.DataFrame({'N': p['Num Sats'], 'd': p[target] - p[b]}).dropna()
            s = d.groupby('N')['d'].mean()
            ax.plot(s.index, s.values, marker='.', linewidth=1.6,
                    color=sns.desaturate(PALETTE[b], 0.75), label=f'$-$ {DISPLAY[b]}')
        ax.axhline(0.0, color='gray', linewidth=1.0)
        ax.set_title(f'Latency: {lat}')
        ax.set_xlabel('Number of Satellites')
        ax.grid(True, linestyle='--', linewidth=0.4)
        _format_log_x(ax, num_sats)
    axes[0].set_ylabel(rf'Paired reward advantage, {DISPLAY[target]} $-$ baseline')
    axes[-1].legend(title='vs. baseline', fontsize=8)
    plt.suptitle(f'Retention of the {DISPLAY[target]} Advantage by '
                 f'Communication Architecture', fontsize=13)
    plt.tight_layout()
    save_plot(*paths, 'Plot3.3.X-SCCBBA_Advantage_Retention.png')
    plt.close()


# ----------------------------------------------------------------------
#  PLOT 3.3.Y — Preplanner contribution by architecture (paired, 95% CI)
# ----------------------------------------------------------------------
def plot_preplanner_delta(df, paths):
    piv = df.pivot_table(index=MATCH_KEY, columns='Algorithm',
                         values=REWARD_COL).reset_index()
    # families: (delta column, display label, color) — preplanner added to each reactive planner
    piv['d_GR']   = piv['DP-GR']   - piv['GR']
    piv['d_CBBA'] = piv['DP-CBBA'] - piv['CBBA']
    families = [('d_GR', 'GR', sns.desaturate(PALETTE['GR'], 0.75)),
                ('d_CBBA', DISPLAY['CBBA'], sns.desaturate(PALETTE['CBBA'], 0.75))]

    def mean_ci(x):
        x = x.dropna().values
        n = len(x)
        h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
        return x.mean(), h

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(LATENCY_ORDER))
    width = 0.38
    for k, (col, lbl, color) in enumerate(families):
        means, errs = zip(*[mean_ci(piv.loc[piv['Latency'] == lat, col])
                            for lat in LATENCY_ORDER])
        ax.bar(x + (k - 0.5) * width, means, width, yerr=errs, capsize=4,
               color=color, edgecolor='black', linewidth=0.5, label=lbl)
    ax.axhline(0.0, color='gray', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LATENCY_ORDER)
    ax.set_xlabel('Communication Architecture')
    ax.set_ylabel(r'Reward change from adding preplanner, $\Delta$')
    ax.set_title('Contribution of the Periodic Preplanner by Architecture')
    ax.legend(title='Reactive planner')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4)
    plt.tight_layout()
    save_plot(*paths, 'Plot3.3.Y-Preplanner_Delta_by_Latency.png')
    plt.close()


# ----------------------------------------------------------------------
#  ORCHESTRATOR
# ----------------------------------------------------------------------
def generate_plots(trial_name: str, base_dir: str = None) -> None:
    local_base_dir = os.path.join('experiments', '1_cbba_validation', 'analysis')
    if base_dir is None:
        base_dir = local_base_dir
        compiled_results_path = os.path.join(
            base_dir, 'compiled', f'{trial_name}_compiled_results.csv')
    else:
        compiled_results_path = os.path.join(
            base_dir, f'{trial_name}_compiled_results.csv')

    if not os.path.exists(compiled_results_path):
        raise FileNotFoundError(
            f"Compiled results file not found at: `{compiled_results_path}`. "
            "Please run `compiler.py` first.")

    date_str = datetime.now().strftime("%Y-%m-%d")
    dirname = f"{trial_name}_P{date_str}"
    save_dir = os.path.join(base_dir, 'plots', 'rq3', dirname)
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq3', dirname)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(local_save_dir, exist_ok=True)
    paths = (base_dir, local_base_dir, save_dir, local_save_dir)

    df = pd.read_csv(compiled_results_path)
    df['Algorithm'] = df.apply(label_condition, axis=1)
    num_sats = sorted(df['Num Sats'].unique())

    plot_reward_by_latency(df, num_sats, paths)
    plot_advantage_retention(df, num_sats, paths)
    plot_preplanner_delta(df, paths)


if __name__ == '__main__':
    base_dir = '/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local'
    trial_name = 'full_factorial_trials_2026-03-15'

    generate_plots(trial_name,
                    # base_dir=base_dir,
                    )
    print('DONE')