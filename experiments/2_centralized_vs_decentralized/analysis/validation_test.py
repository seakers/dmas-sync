from datetime import datetime
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
#  HELPERS
# ----------------------------------------------------------------------

def label_condition(row):
    pre = row['Preplanner']
    rep = row['Replanner']

    if pre == 'Centralized-MILP_priority' and pd.isna(rep): return 'MILP'
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


def aggregate_cells(df, metric_cols, group_vars):
    """
    Pre-aggregate raw trials to one mean per (Algorithm, design-variable)
    cell. Removes within-cell noise while preserving the between-cell
    spread that reflects the design space variation.
    """
    return (
        df.groupby(['Algorithm'] + group_vars)[metric_cols]
        .mean()
        .reset_index()
    )


def make_boxplot(data, metric, ax, algo_order, algo_palette, ylabel=None):
    """
    Box plot of pre-aggregated cell means with individual cell points
    overlaid as a strip plot for transparency.
    """
    sns.boxplot(
        data=data, x='Algorithm', y=metric,
        order=algo_order, palette=algo_palette,
        hue='Algorithm', hue_order=algo_order,
        width=0.5, linewidth=0.8, fliersize=0,
        legend=False, ax=ax,
    )
    sns.stripplot(
        data=data, x='Algorithm', y=metric,
        order=algo_order, palette=algo_palette,
        hue='Algorithm', hue_order=algo_order,
        size=4, alpha=0.5, jitter=True, dodge=False,
        legend=False, ax=ax, marker='o', edgecolor='gray', linewidth=0.5,
    )
    ax.set_xlabel('Algorithm Configuration')
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4)


def plot_metric(df, agg, metrics, titles, ylabels,
                suptitle,
                algo_order, algo_palette,
                group_vars,
                base_dir, local_base_dir, save_dir, local_save_dir,
                filename_stem,
                log_x=False, log_y=False,
                lower_bound_col=None, upper_bound_col=None):
    """
    Generates both figures for one or more related metrics.

    Figure A — box + strip by Algorithm, one panel per metric.
               Single metric  -> (9  x 5)
               Two metrics    -> (13 x 5)

    Figure B — grouped bar plots:
               rows=metrics, cols=each design variable in group_vars

    Parameters
    ----------
    metrics    : str or list[str]
    titles     : str or list[str]   panel titles for Figure A
    ylabels    : str or list[str]   y-axis labels
    suptitle   : str                shared super-title for both figures
    group_vars : list of (col, xlabel) pairs for Figure B columns
    """
    # normalise single-metric calls to lists
    if isinstance(metrics, str):
        metrics = [metrics]
        titles  = [titles]
        ylabels = [ylabels]

    n = len(metrics)
    group_cols   = [gv[0] for gv in group_vars]
    group_labels = [gv[1] for gv in group_vars]

    # ------------------------------------------------------------------
    #  Figure A: box + strip by Algorithm
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n, 1, figsize=(9, 5 * n), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, metric, title, ylabel) in enumerate(zip(axes, metrics, titles, ylabels)):
        make_boxplot(agg, metric, ax, algo_order, algo_palette, ylabel=ylabel)
        if log_y: ax.set_yscale('log')
        if agg[metric].max() <= 1: ax.set_ylim(-0.02, 1.02)
        if n > 1:
            ax.set_title(f'({chr(ord("a") + i)})')
            if i < n - 1:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle(f'{suptitle}', fontsize=13)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              f'{filename_stem}_by_algorithm.png')
    plt.close()

    # ------------------------------------------------------------------
    #  Figure B: grouped bar plots (rows=metrics, cols=design variable)
    # ------------------------------------------------------------------

    # inject upper/lower bound columns as pseudo-algorithm rows so they
    # appear as bars alongside algorithm bars in Figure B
    bar_order   = list(algo_order)
    bar_palette = dict(algo_palette)

    if lower_bound_col is not None:
        lower_agg = (
            df.groupby(group_cols)[lower_bound_col]
            .mean()
            .reset_index()
        )
        bar_order   = ['Primal Bound'] + bar_order
        bar_palette = {'Primal Bound': '#888888', **bar_palette}
    else:
        lower_agg = None

    if upper_bound_col is not None:
        upper_agg = (
            df.groupby(group_cols)[upper_bound_col]
            .mean()
            .reset_index()
        )
        bar_order   = bar_order + ['Dual Bound']
        bar_palette = {**bar_palette, 'Dual Bound': '#222222'}
    else:
        upper_agg = None

    n_cols = len(group_vars)
    fig_height = 5 if n == 1 else 10
    fig, axes = plt.subplots(n, n_cols, figsize=(7 * n_cols, fig_height), sharey='row')

    # ensure axes is always 2D for uniform indexing
    if n == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col_idx, (group_col, xlabel) in enumerate(group_vars):
            ax = axes[row_idx][col_idx]

            plot_data = agg.copy()
            if lower_agg is not None:
                lower_rows = lower_agg[group_cols + [lower_bound_col]].copy()
                lower_rows = lower_rows.rename(columns={lower_bound_col: metric})
                lower_rows['Algorithm'] = 'Primal Bound'
                plot_data = pd.concat([lower_rows, plot_data], ignore_index=True)
            if upper_agg is not None:
                upper_rows = upper_agg[group_cols + [upper_bound_col]].copy()
                upper_rows = upper_rows.rename(columns={upper_bound_col: metric})
                upper_rows['Algorithm'] = 'Dual Bound'
                plot_data = pd.concat([plot_data, upper_rows], ignore_index=True)

            sns.barplot(
                data=plot_data, x=group_col, y=metric,
                hue='Algorithm', hue_order=bar_order,
                palette=bar_palette,
                errorbar=None,
                width=0.7,
                ax=ax,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4)
            if log_y: ax.set_yscale('log')
            if plot_data[metric].max() <= 1: ax.set_ylim(0.0, 1.02)

            # single legend on top-right panel only
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='best')
            else:
                ax.get_legend().remove()

    plt.suptitle(f'{suptitle} - Grouped by Design Variable', fontsize=13)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              f'{filename_stem}_grouped.png')
    plt.close()


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------

def generate_plots(trial_name: str,
                   base_dir: str = None) -> None:
    """
    Generates all plots for the compiled results of a given trial.

    For Plots 1, 3, 4 two figures are produced per metric:
      *_by_algorithm.png  — box + strip of cell means, x = Algorithm
      *_grouped.png       — grouped bar plots by Connectivity and Data Processing

    Plots 2, 5, 6 produce a single figure each.
    """

    # ------------------------------------------------------------------
    #  PATHS
    # ------------------------------------------------------------------
    local_base_dir = os.path.join('experiments', '2_centralized_vs_decentralized', 'analysis')
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

    save_dir = os.path.join(base_dir, 'plots', 'rq2', dirname)
    os.makedirs(save_dir, exist_ok=True)

    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq2', dirname)
    os.makedirs(local_save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  LOAD AND LABEL
    # ------------------------------------------------------------------
    df = pd.read_csv(compiled_results_path)

    if 'in_centralization' not in df.columns:
        raise ValueError(
            f"Column `in_centralization` not found. "
            f"Available columns: {df.columns.tolist()}")

    if df.empty:
        print(f"No results found for trial `{trial_name}`.")
        return

    df['Algorithm'] = df.apply(label_condition, axis=1)

    # Design variables for this experiment
    connectivity_order = ['GS', 'Intraconstellation', 'Interconstellation']
    data_proc_order    = ['Onboard', 'Oracle']
    df['Connectivity']    = pd.Categorical(df['Connectivity'],    categories=connectivity_order, ordered=True)
    df['Data Processing'] = pd.Categorical(df['Data Processing'], categories=data_proc_order,    ordered=True)

    group_vars = [
        ('Connectivity',    'Connectivity Type'),
        ('Data Processing', 'Data Processing Mode'),
    ]
    group_cols = [gv[0] for gv in group_vars]

    algo_order = ['MILP', 'DP', 'GR', 'DP-GR', 'CBBA', 'DP-CBBA']

    algo_palette = {
        'MILP':    '#D55E00',  # vermillion  (centralized reference)
        'DP':      '#CC79A7',  # gray
        'GR':      '#E69F00',  # orange
        'DP-GR':   '#56B4E9',  # sky blue
        'CBBA':    '#009E73',  # teal
        'DP-CBBA': '#0072B2',  # deep blue
    }

    linestyles = {
        'MILP':    (6, 2),
        'DP':      (4, 2),
        'GR':      (1, 2),
        'DP-GR':   (4, 1, 1, 1),
        'CBBA':    (3, 1),
        'DP-CBBA': '',
    }

    # Pre-aggregate all metrics once — one mean per design cell
    cell_metrics = [
        'Total Obtained Reward',
        'Total Obtained Reward [norm]',
        'P(Task Observed)',
        'P(Task Observed | Task Observable)',
        'P(Event Co-observed)',
        'P(Event Co-observed | Co-observable)',
        'Average Normalized Response Time to Task',
    ]
    agg = aggregate_cells(df, cell_metrics, group_cols)

    # ------------------------------------------------------------------
    #  PLOT 1a — Mission Reward
    #  Figure A: 1 panel.   Figure B: 1x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='Total Obtained Reward',
        titles='Mission Reward by Algorithm',
        ylabels='Total Obtained Reward',
        suptitle='Mission Reward by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        group_vars=group_vars,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot1a-Mission_Reward',
        lower_bound_col='Task Reward Primal Bound',
        upper_bound_col='Task Reward Dual Bound',
    )

    # ------------------------------------------------------------------
    #  PLOT 1b — Normalized Mission Reward
    #  Figure A: 1 panel.   Figure B: 1x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='Total Obtained Reward [norm]',
        titles='Normalized Mission Reward by Algorithm',
        ylabels='Total Obtained Reward (normalized)',
        suptitle='Normalized Mission Reward by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        group_vars=group_vars,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot1b-Mission_Reward_Normalized',
    )

    # ------------------------------------------------------------------
    #  PLOT 2 — Reward vs Design Variables
    #  Panel 1: reward vs Connectivity (ordered categorical)
    #  Panel 2: reward vs Data Processing
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (x_col, xlabel, x_order) in zip(axes, [
        ('Connectivity',    'Connectivity Type',      connectivity_order),
        ('Data Processing', 'Data Processing Mode',   data_proc_order),
    ]):
        sns.lineplot(
            data=df,
            x=x_col, y='Total Obtained Reward [norm]',
            hue='Algorithm', style='Algorithm',
            hue_order=algo_order, style_order=algo_order,
            palette=algo_palette, dashes=linestyles,
            errorbar=None, ax=ax,
        )

        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Total Obtained Reward (normalized)')
        ax.grid(True, linestyle='--', linewidth=0.4)

        # legend only on the right panel
        if ax is axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title='Algorithm')

    plt.suptitle('Reward vs Design Variables', fontsize=13)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              'Plot2-Reward_vs_Design_Variables.png')
    plt.close()

    # ------------------------------------------------------------------
    #  PLOT 3 — Task Observation Probability
    #  Figure A: 1x2 panels.   Figure B: 2x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics=[
            'P(Task Observed)',
            'P(Task Observed | Task Observable)',
        ],
        titles=[
            r'$P(\mathrm{Task\ Observed})$',
            r'$P(\mathrm{Task\ Observed\ |\ Observable})$',
        ],
        ylabels=[
            r'$P(\mathrm{Task\ Observed})$',
            r'$P(\mathrm{Task\ Observed\ |\ Observable})$',
        ],
        suptitle='Task Observation Probability by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        group_vars=group_vars,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot3-Task_Observation_Probability',
    )

    # ------------------------------------------------------------------
    #  PLOT 4 — Multi-Agent Coordination Quality
    #  Figure A: 1x2 panels.   Figure B: 2x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics=[
            'P(Event Co-observed)',
            'P(Event Co-observed | Co-observable)',
        ],
        titles=[
            r'$P(\mathrm{Event\ Co-observed})$',
            r'$P(\mathrm{Event\ Co-observed\ |\ Co-observable})$',
        ],
        ylabels=[
            r'$P(\mathrm{Event\ Co-observed})$',
            r'$P(\mathrm{Event\ Co-observed\ |\ Co-observable})$',
        ],
        suptitle='Multi-Agent Coordination Quality by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        group_vars=group_vars,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot4-Coordination_Quality',
    )

    # ------------------------------------------------------------------
    #  PLOT 5 — Task Response Time by Algorithm
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='Average Normalized Response Time to Task',
        titles='Average Normalized Task Response Time by Algorithm',
        ylabels='Average Normalized Response Time to Task',
        suptitle='Average Normalized Task Response Time by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        group_vars=group_vars,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot5-Task_Response_Time',
    )

    # ------------------------------------------------------------------
    #  PLOT 6a — Response Time vs Task Observation Quality
    #  PLOT 6b — Response Time vs Co-observation Quality
    # ------------------------------------------------------------------
    summary = df.groupby('Algorithm').agg(
        rt_mean=('Average Normalized Response Time to Task', 'mean'),
        rt_std=('Average Normalized Response Time to Task', 'std'),
        obs_mean=('P(Task Observed)', 'mean'),
        obs_std=('P(Task Observed)', 'std'),
        obs_cond_mean=('P(Task Observed | Task Observable)', 'mean'),
        obs_cond_std=('P(Task Observed | Task Observable)', 'std'),
        cobs_mean=('P(Event Co-observed)', 'mean'),
        cobs_std=('P(Event Co-observed)', 'std'),
        cobs_cond_mean=('P(Event Co-observed | Co-observable)', 'mean'),
        cobs_cond_std=('P(Event Co-observed | Co-observable)', 'std'),
    ).reindex(algo_order)

    def _draw_rt_scatter(axes, panels):
        for i, (ax, (y_col, y_mean, y_std, ylabel)) in enumerate(zip(axes, panels)):
            # --- raw trial scatter (faint context) ---
            for algo in algo_order:
                sub = df[df['Algorithm'] == algo]
                ax.scatter(
                    sub['Average Normalized Response Time to Task'],
                    sub[y_col],
                    color=algo_palette[algo], alpha=0.08, s=18, zorder=1,
                )

            # --- reference lines at MILP (centralized baseline) mean position ---
            ref_rt = summary.loc['MILP', 'rt_mean']
            ref_y  = summary.loc['MILP', y_mean]
            ax.axvline(ref_rt, color=algo_palette['MILP'],
                       linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axhline(ref_y,  color=algo_palette['MILP'],
                       linestyle=':', linewidth=0.8, alpha=0.5)

            # --- utopia point at (0, 1) ---
            ax.scatter(0, 1, marker='*', s=220, color='gold',
                       edgecolors='black', linewidths=0.8, zorder=5)
            ax.annotate('Utopia', xy=(0, 1), xytext=(0.04, 0.93),
                        fontsize=7, color='black', ha='left')

            # --- mean ± std error bars ---
            for algo in algo_order:
                row = summary.loc[algo]
                ax.errorbar(
                    row['rt_mean'], row[y_mean],
                    xerr=row['rt_std'], yerr=row[y_std],
                    fmt='none', color=algo_palette[algo],
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    alpha=0.7, zorder=2,
                )
                ax.scatter(
                    row['rt_mean'], row[y_mean],
                    color=algo_palette[algo], s=120,
                    marker='D', zorder=3, label=algo,
                    edgecolors='white', linewidths=0.8,
                )

            # --- annotate MILP and CBBA mean markers ---
            for algo, ha, offset in [
                ('MILP', 'right', (-0.010,  0.020)),
                ('CBBA', 'left',  ( 0.010,  0.020)),
            ]:
                row = summary.loc[algo]
                ax.annotate(
                    algo,
                    xy=(row['rt_mean'], row[y_mean]),
                    xytext=(row['rt_mean'] + offset[0], row[y_mean] + offset[1]),
                    fontsize=8, ha=ha, color=algo_palette[algo],
                )

            ax.set_ylabel(ylabel)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle='--', linewidth=0.4)

            if i < len(axes) - 1:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel('Avg. Normalized Response Time to Task')

        handles = [
            plt.scatter([], [], color=algo_palette[a], marker='D', s=80, label=a)
            for a in algo_order
        ]
        axes[-1].legend(handles=handles, title='Algorithm', loc='best', fontsize=8)

    # Plot 6a — task observation
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    _draw_rt_scatter(axes, [
        ('P(Task Observed)',
         'obs_mean', 'obs_std',
         r'$P(\mathrm{Task\ Observed})$'),
        ('P(Task Observed | Task Observable)',
         'obs_cond_mean', 'obs_cond_std',
         r'$P(\mathrm{Task\ Observed\ |\ Observable})$'),
    ])
    plt.suptitle('Response Time vs Task Observation Quality', fontsize=12)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              'Plot6a-Response_Time_vs_Task_Observation.png')
    plt.close()

    # Plot 6b — co-observation
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    _draw_rt_scatter(axes, [
        ('P(Event Co-observed)',
         'cobs_mean', 'cobs_std',
         r'$P(\mathrm{Event\ Co-observed})$'),
        ('P(Event Co-observed | Co-observable)',
         'cobs_cond_mean', 'cobs_cond_std',
         r'$P(\mathrm{Event\ Co-observed\ |\ Co-observable})$'),
    ])
    plt.suptitle('Response Time vs Co-observation Quality', fontsize=12)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              'Plot6b-Response_Time_vs_Co_Observation.png')
    plt.close()

# ----------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == '__main__':
    trial_name = 'full_factorial_trials_2026-05-11'

    generate_plots(trial_name)

    print('DONE')