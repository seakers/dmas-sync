"""
RQ1 plotting module: Computational Scalability.

Generates the runtime-versus-message-volume figure that shows the coupling between
simulation runtime and total messages broadcast holding across all three
communication architectures:

  Plot3.1.4 - Simulation runtime vs. total messages broadcast for the SC-CBBA,
              colored by communication architecture, with a per-architecture
              power-law fit (exponent and R^2 shown in the legend).

This uses the SC-CBBA runs across all three latency levels (the connectivity set),
not the full-connectivity-only scalability sweep, since the point of the figure is
that the runtime-message coupling generalizes beyond full connectivity. Mirrors the
structure of the other plotting modules (label_condition / save_plot / date-stamped
output dirs).
"""

from datetime import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



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

# internal label -> display label
DISPLAY = {'DP': 'DP', 'GR': 'GR', 'DP-GR': 'DP-GR',
           'CBBA': 'SC-CBBA', 'DP-CBBA': 'DP-SC-CBBA'}

LATENCY_ORDER = ['Low', 'Medium', 'High']
# LATENCY_COLORS = {
#     'Low':    '#56B4E9',  # sky blue   (Wong)
#     'Medium': '#E69F00',  # amber      (Wong)
#     'High':   '#D55E00',  # vermillion (Wong)
# }
LATENCY_COLORS = {
    'Low': '#009E73', 
    'Medium': '#56B4E9', 
    'High': '#CC79A7'
}

MSG_COL = 'Total Messages Broadcasted'
RUNTIME_COL = 'Simulation Runtime [s]'
PERTASK_COL = 'Average Messages Broadcasted per Task'


def save_plot(base_dir, local_base_dir, save_dir, local_save_dir, plot_filename):
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        plt.savefig(local_save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: `{save_path}`" +
          (f" and `{local_save_path}`" if base_dir != local_base_dir else ""))


def _loglog_fit(x, y):
    """Fit y = a * x^b in log-log space; return (exponent b, intercept a, R^2)."""
    lx = np.log10(x.replace(0, np.nan))
    ly = np.log10(y.replace(0, np.nan))
    mask = lx.notna() & ly.notna()
    b, a = np.polyfit(lx[mask], ly[mask], 1)
    r2 = np.corrcoef(lx[mask], ly[mask])[0, 1] ** 2
    return b, a, r2
 
 
def _logx_integer_ticks(ax, ticks):
    """Log x-axis with plain integer tick labels at the given values."""
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())



# ----------------------------------------------------------------------
#  PLOT 3.1.4 - Runtime vs. message volume by architecture
# ----------------------------------------------------------------------
def plot_runtime_vs_messages_by_latency(df, paths, target='CBBA'):
    sub = df[df['Algorithm'] == target].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    xline = np.linspace(np.log10(sub[MSG_COL].replace(0, np.nan).min()),
                        np.log10(sub[MSG_COL].max()), 50)

    ann_fracs   = [0.30,      0.55,     0.78    ]
    ann_offsets = [(-50, 30), (0, -38),  (-50, 30)]
    for k, lat in enumerate(LATENCY_ORDER):
        s = sub[sub['Latency'] == lat]
        if s.empty:
            continue
        b, a, r2 = _loglog_fit(s[MSG_COL], s[RUNTIME_COL])
        color = LATENCY_COLORS[lat]
        ax.scatter(s[MSG_COL], s[RUNTIME_COL], s=28, color=color, alpha=0.75,
                   edgecolor='none', label=lat)
        ax.plot(10 ** xline, 10 ** (a + b * xline), color=color, lw=1.3, alpha=0.9, linestyle='--')
        ann_idx = int(len(xline) * ann_fracs[k])
        ax.annotate(f'$b={b:.2f}$, $R^2={r2:.2f}$',
                    xy=(10 ** xline[ann_idx], 10 ** (a + b * xline[ann_idx])),
                    xytext=ann_offsets[k], textcoords='offset points',
                    fontsize=7.5, color=color, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=color, linewidth=0.6, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total messages broadcast')
    ax.set_ylabel('Simulation runtime [s]')
    ax.set_title('Runtime vs. Message Volume by Communication Architecture')
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.6)
    ax.legend(title=f'Architecture ({DISPLAY[target]})', fontsize=9)
    plt.tight_layout()
    save_plot(*paths, 'Plot3.1.4-Runtime_vs_Messages_by_Latency.png')
    plt.close()



# ----------------------------------------------------------------------
#  PLOT 3.1.5 - Message economy across architectures
# ----------------------------------------------------------------------
def plot_message_economy_by_architecture(df, paths, target='CBBA'):
    sub = df[df['Algorithm'] == target].copy()
    num_sats = sorted(sub['Num Sats'].unique())
    gammas = sorted(sub['Task Arrival Rate'].unique())
 
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))
 
    # (a) total messages relative to Low, vs constellation size
    piv = sub.pivot_table(index='Num Sats', columns='Latency',
                          values=MSG_COL).reindex(columns=LATENCY_ORDER)
    for lat in ['Medium', 'High']:
        ax_a.plot(piv.index, (piv[lat] / piv['Low']).values, marker='o',
                  color=LATENCY_COLORS[lat], lw=1.8, label=lat)
    ax_a.axhline(1.0, color=LATENCY_COLORS['Low'], lw=1.3, ls='--',
                 label='Low (reference)')
    _logx_integer_ticks(ax_a, num_sats)
    ax_a.set_xlabel('Number of Satellites')
    ax_a.set_ylabel('Total messages, relative to Low')
    ax_a.set_title('(a) Coordination volume vs. constellation size')
    ax_a.grid(True, linestyle='--', linewidth=0.4)
    ax_a.legend(title='Architecture')
 
    # (b) per-task count relative to Low, vs arrival rate (averaged over constellation size)
    piv_p = sub.pivot_table(index='Task Arrival Rate', columns='Latency',
                            values=PERTASK_COL).reindex(columns=LATENCY_ORDER)
    for lat in ['Medium', 'High']:
        ax_b.plot(piv_p.index, (piv_p[lat] / piv_p['Low']).values, marker='o',
                  color=LATENCY_COLORS[lat], lw=1.8, label=lat)
    ax_b.axhline(1.0, color=LATENCY_COLORS['Low'], lw=1.3, ls='--',
                 label='Low (reference)')
    _logx_integer_ticks(ax_b, gammas)
    ax_b.set_xlabel(r'Task arrival rate $\gamma$ [tasks/day]')
    ax_b.set_ylabel('Messages per task, relative to Low')
    ax_b.set_title('(b) Per-task cost vs. task load')
    ax_b.grid(True, linestyle='--', linewidth=0.4)
    ax_b.legend(title='Architecture')
 
    plt.tight_layout()
    save_plot(*paths, 'Plot3.1.5-Message_Economy_by_Architecture.png')
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
    save_dir = os.path.join(base_dir, 'plots', 'rq1', dirname)
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq1', dirname)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(local_save_dir, exist_ok=True)
    paths = (base_dir, local_base_dir, save_dir, local_save_dir)
 
    df = pd.read_csv(compiled_results_path)
    df['Algorithm'] = df.apply(label_condition, axis=1)
 
    plot_runtime_vs_messages_by_latency(df, paths)
    plot_message_economy_by_architecture(df, paths)




if __name__ == '__main__':
    base_dir = '/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local'
    trial_name = 'full_factorial_trials_2026-03-15'

    generate_plots(trial_name,
                    # base_dir=base_dir,
                    )
    print('DONE')