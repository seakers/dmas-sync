from datetime import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def label_condition(row):
    pre = row['Preplanner']
    rep = row['Replanner']
    if pre == 'DP' and pd.isna(rep):     return 'DP'
    if pd.isna(pre) and rep == 'Greedy': return 'GR'
    if pre == 'DP'  and rep == 'Greedy': return 'DP-GR'
    if pd.isna(pre) and rep == 'CBBA':   return 'CBBA'
    if pre == 'DP'  and rep == 'CBBA':   return 'DP-CBBA'
    return 'Unknown'

def save_plot(base_dir, local_base_dir, save_dir, local_save_dir, plot_filename):
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        plt.savefig(local_save_path, dpi=150, bbox_inches='tight')

    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

def generate_plots(trial_name: str,
                   base_dir: str = None) -> None:
    """
    Generates RQ1 feasibility plots for the compiled results of a given trial.
    """

    # ------------------------------------------------------------------
    #  PATHS
    # ------------------------------------------------------------------
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
    
    # define save directory and filename for plot
    # name the output file with the current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    dirname = f"{trial_name}_P{date_str}"

    save_dir = os.path.join(base_dir, 'plots', 'rq1', dirname)
    os.makedirs(save_dir, exist_ok=True)

    # if saving to external directory, also save a copy to the local analysis directory 
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq1', dirname)
    os.makedirs(local_save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  LOAD AND LABEL
    # ------------------------------------------------------------------
    df = pd.read_csv(compiled_results_path)

    df['Algorithm'] = df.apply(label_condition, axis=1)

    algo_order = ['DP', 'GR', 'DP-GR', 'CBBA', 'DP-CBBA']

    # _viridis = sns.color_palette("viridis", n_colors=3)
    # colors = {
    #     'P_success':        _viridis[2],  # yellow-green — task observed
    #     'P_sched_failure':  _viridis[1],  # teal         — observable but missed
    #     'P_access_failure': _viridis[0],  # purple       — never accessible
    # }
    colors = {
        'P_success':        '#0072B2',  # blue       — dark,   observed
        'P_sched_failure':  '#F0E442',  # yellow     — light,  observable not scheduled
        'P_access_failure': '#999999',  # gray       — medium, not accessible
    }

    def _make_agg(group_col):
        # cbba_only = df[df['Algorithm'] == 'CBBA'].copy()
        agg = df.groupby(['Algorithm', group_col]).agg(
            P_observable=('P(Task Observable)', 'mean'),
            P_obs_given_obs=('P(Task Observed | Task Observable)', 'mean'),
        ).reset_index()
        agg['P_access_failure'] = 1 - agg['P_observable']
        agg['P_sched_failure']  = agg['P_observable'] * (1 - agg['P_obs_given_obs'])
        agg['P_success']        = agg['P_observable'] * agg['P_obs_given_obs']
        return agg

    def _draw_f1(agg, panel_col, panel_vals, panel_title_fn, suptitle, filename_stem):
        fig, axes = plt.subplots(1, len(panel_vals), figsize=(3 * len(panel_vals), 5), sharey=True)
        if len(panel_vals) == 1:
            axes = [axes]

        for ax, val in zip(axes, panel_vals):
            sub = agg[agg[panel_col] == val].set_index('Algorithm').reindex(algo_order)

            bottom_sched  = sub['P_success']
            bottom_access = sub['P_success'] + sub['P_sched_failure']

            first = (val == panel_vals[0])
            ax.bar(algo_order, sub['P_success'],
                   color=colors['P_success'],        label='Observed' if first else '')
            ax.bar(algo_order, sub['P_sched_failure'], bottom=bottom_sched,
                   color=colors['P_sched_failure'],  label='Observable, not scheduled' if first else '')
            ax.bar(algo_order, sub['P_access_failure'], bottom=bottom_access,
                   color=colors['P_access_failure'], label='Not accessible' if first else '')

            ax.set_title(panel_title_fn(val))
            ax.tick_params(axis='x', rotation=30)
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4)

        axes[0].set_ylabel('Fraction of Tasks')
        axes[0].legend(loc='upper right', fontsize=8)
        plt.suptitle(suptitle, fontsize=13)
        fig.supxlabel('Algorithm', fontsize=11)
        plt.tight_layout()
        save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
                  f'{filename_stem}.png')
        plt.close()

    # ------------------------------------------------------------------
    #  PLOT F1a — panels by Number of Satellites
    # ------------------------------------------------------------------
    _draw_f1(
        agg=_make_agg('Num Sats'),
        panel_col='Num Sats',
        panel_vals=sorted(df['Num Sats'].unique()),
        panel_title_fn=lambda n: rf'$N_{{sat}} = {n}$',
        suptitle='Task Outcome Decomposition by Number of Satellites',
        filename_stem='PlotF1a-Task_Accessibility_by_NumSats',
    )

    # ------------------------------------------------------------------
    #  PLOT F1b — panels by Task Arrival Rate
    # ------------------------------------------------------------------
    _draw_f1(
        agg=_make_agg('Task Arrival Rate'),
        panel_col='Task Arrival Rate',
        panel_vals=sorted(df['Task Arrival Rate'].unique()),
        panel_title_fn=lambda r: rf'$\lambda = {r}$',
        suptitle='Task Outcome Decomposition by Task Arrival Rate',
        filename_stem='PlotF1b-Task_Accessibility_by_TaskRate',
    )

    # ------------------------------------------------------------------
    # PLOT F1c — decomposition for CBBA only, faceted by Latency
    # x-axis = Num Sats, colored segments same as before
    # This directly shows whether High latency shifts the orange segment upward
    # ------------------------------------------------------------------

    # cbba_only = df[df['Algorithm'] == 'DP-CBBA'].copy()
    cbba_only = df[df['Algorithm'] == 'CBBA'].copy()


    cbba_only['P_access_failure'] = 1 - cbba_only['P(Task Observable)']
    cbba_only['P_sched_failure']  = (cbba_only['P(Task Observable)']
                                    * (1 - cbba_only['P(Task Observed | Task Observable)']))
    cbba_only['P_success']        = (cbba_only['P(Task Observable)']
                                    * cbba_only['P(Task Observed | Task Observable)'])

    agg = cbba_only.groupby(['Latency', 'Num Sats'])[
        ['P_success', 'P_sched_failure', 'P_access_failure']
    ].mean().reset_index()

    latency_order = ['Low', 'Medium', 'High']
    num_sats = sorted(df['Num Sats'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    x = range(len(num_sats))
    width = 0.6

    for ax, latency in zip(axes, latency_order):
        sub = agg[agg['Latency'] == latency].sort_values('Num Sats')

        bot_sched  = sub['P_success'].values
        bot_access = sub['P_success'].values + sub['P_sched_failure'].values

        ax.bar(x, sub['P_success'],        width, color=colors['P_success'],
            label='Observed' if latency == 'Low' else '')
        ax.bar(x, sub['P_sched_failure'],  width, bottom=bot_sched,
            color=colors['P_sched_failure'],
            label='Observable, not scheduled' if latency == 'Low' else '')
        ax.bar(x, sub['P_access_failure'], width, bottom=bot_access,
            color=colors['P_access_failure'],
            label='Not accessible' if latency == 'Low' else '')

        ax.set_xticks(x)
        ax.set_xticklabels(num_sats)
        ax.set_title(f'Latency: {latency}')
        ax.set_xlabel('$N_{sat}$')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.4)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel('Fraction of Tasks')
    axes[0].legend(fontsize=8, loc='best')
    plt.suptitle(
        r'CBBA Task Outcome Decomposition by Latency',
        # r' - Scheduling failure driven by latency; access failure driven by constellation size',
        fontsize=12
    )
    plt.tight_layout()

    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              'PlotF1c-CBBA_Task_Accessibility_by_Latency')
    
    # ------------------------------------------------------------------
    #  PLOT F2 — Scheduling Failure Rate by Latency
    # ------------------------------------------------------------------
    # latency_palette  = {'Low': '#009E73', 'Medium': '#E69F00', 'High': '#D55E00'}    
    latency_palette  = {'Low': '#009E73', 'Medium': '#56B4E9', 'High': '#CC79A7'}
    latency_order    = ['Low', 'Medium', 'High']
    latency_dashes   = {'Low': '', 'Medium': (4, 2), 'High': (1, 2)}
    num_sats         = sorted(df['Num Sats'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # for ax, algo in zip(axes, ['DP-CBBA', 'GR']):
    for ax, algo in zip(axes, ['CBBA', 'GR']):
        is_first = (algo == 'CBBA')
        sub = df[df['Algorithm'] == algo].copy()
        sub['P_sched_failure'] = (
            sub['P(Task Observable)']
            * (1 - sub['P(Task Observed | Task Observable)'])
        )

        summary = sub.groupby(['Latency', 'Num Sats'])['P_sched_failure'].agg(
            ['mean', 'std']
        ).reset_index()

        for latency in latency_order:
            s = summary[summary['Latency'] == latency].sort_values('Num Sats')
            ls = '' if latency_dashes[latency] == '' else latency_dashes[latency]

            ax.plot(s['Num Sats'], s['mean'],
                    color=latency_palette[latency],
                    linestyle='-' if ls == '' else '--',
                    dashes=ls if ls != '' else (None, None),
                    marker='.', linewidth=1.5, label=latency)

            ax.fill_between(s['Num Sats'],
                            s['mean'] - s['std'],
                            s['mean'] + s['std'],
                            color=latency_palette[latency], alpha=0.15)

        ax.set_title(algo)
        # ax.set_xlabel(r'Number of Satellites $N_{sat}$')
        ax.set_xlabel(r'Number of Satellites')
        if is_first: ax.set_ylabel(r'$P(\mathrm{ \neg Task Observed \ | \ Task Observable})$')
        ax.set_ylim(0, 1.02)
        ax.set_xscale("log")
        ax.set_xticks(num_sats)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlim(min(num_sats), max(num_sats))
        ax.legend(title='Latency')
        ax.grid(True, linestyle='--', linewidth=0.4)

    plt.suptitle(
        r'Scheduling Failure Rate by Latency',
        # r' — Low latency near-zero for DP-CBBA; latency effect peaks at $N_{sat}=48$',
        fontsize=12
    )
    plt.tight_layout()

    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
                'PlotF2-Scheduling_Failure_by_Latency')

if __name__ == '__main__':
    local_base_dir = os.path.join('experiments', '1_cbba_validation', 'analysis')
    base_dir = '/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local'

    trial_name = 'full_factorial_trials_2026-03-15'

    generate_plots(trial_name,
                #    base_dir=base_dir,
                   )

    print('DONE')