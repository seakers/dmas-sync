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

    if pre == 'DP' and pd.isna(rep):     return 'Preplanning only'
    if pd.isna(pre) and rep == 'Greedy': return 'Greedy only'
    if pre == 'DP'  and rep == 'Greedy': return 'DP + Greedy'
    if pd.isna(pre) and rep == 'CBBA':   return 'CBBA only'
    if pre == 'DP'  and rep == 'CBBA':   return 'DP + CBBA'

    return 'Unknown'


def save_plot(base_dir, local_base_dir, save_dir, local_save_dir, plot_filename):
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if base_dir != local_base_dir:
        plt.savefig(local_save_path, dpi=150, bbox_inches='tight')

    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


def aggregate_cells(df, metric_cols):
    """
    Pre-aggregate raw trials to one mean per (Algorithm, Num Sats,
    Task Arrival Rate) cell. Removes within-cell noise while preserving
    the between-cell spread that reflects the design space variation.
    """
    return (
        df.groupby(['Algorithm', 'Num Sats', 'Task Arrival Rate'])[metric_cols]
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
                base_dir, local_base_dir, save_dir, local_save_dir,
                filename_stem):
    """
    Generates both figures for one or more related metrics.

    Figure A — box + strip by Algorithm, one panel per metric.
               Single metric  -> (9  x 5)
               Two metrics    -> (13 x 5)

    Figure B — grouped bar plots:
               Single metric  -> 1x2  (Num Sats | Task Arrival Rate)
               Two metrics    -> 2x2  rows=metrics, cols=grouping variable

    Parameters
    ----------
    metrics  : str or list[str]
    titles   : str or list[str]   panel titles for Figure A
    ylabels  : str or list[str]   y-axis labels
    suptitle : str                shared super-title for both figures
    """
    # normalise single-metric calls to lists
    if isinstance(metrics, str):
        metrics = [metrics]
        titles  = [titles]
        ylabels = [ylabels]

    n = len(metrics)

    # ------------------------------------------------------------------
    #  Figure A: box + strip by Algorithm
    # ------------------------------------------------------------------
    fig_width = 9 if n == 1 else 13
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        make_boxplot(agg, metric, ax, algo_order, algo_palette, ylabel=ylabel)
        if n > 1: ax.set_title(title)

    plt.suptitle(f'{suptitle}', fontsize=13)
    plt.tight_layout()
    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              f'{filename_stem}_by_algorithm.png')
    plt.close()

    # ------------------------------------------------------------------
    #  Figure B: grouped bar plots (rows=metrics, cols=grouping variable)
    # ------------------------------------------------------------------
    fig_height = 5 if n == 1 else 10
    fig, axes = plt.subplots(n, 2, figsize=(14, fig_height), sharey='row')

    # ensure axes is always 2D for uniform indexing
    if n == 1:
        axes = np.array([axes])

    for row_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col_idx, (group_col, xlabel) in enumerate([
            ('Num Sats',          'Number of Satellites'),
            ('Task Arrival Rate', r'Task Arrival Rate $\lambda$ (1/day)'),
        ]):
            ax = axes[row_idx][col_idx]

            sns.barplot(
                data=agg, x=group_col, y=metric,
                hue='Algorithm', hue_order=algo_order,
                palette=algo_palette,
                # errorbar='sd',
                # errorbar=('pi', 50),  # 50% prediction interval for the mean
                errorbar=None,
                width=0.7, ax=ax,
            )
            # sns.boxplot(
            #     data=agg, x=group_col, y=metric,
            #     hue='Algorithm', hue_order=algo_order,
            #     palette=algo_palette,
            #     width=0.7, ax=ax,
            # )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.4)

            # single legend on top-right panel only
            if row_idx == 0 and col_idx == 1:
                ax.legend(title='Algorithm', fontsize=7,
                          title_fontsize=8, loc='upper right')
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
    Generates all RQ2 plots for the compiled results of a given trial.

    For Plots 1, 3, 4 two figures are produced per metric:
      *_by_algorithm.png  — box + strip of cell means, x = Algorithm
      *_grouped.png       — grouped bar plots by Num Sats and Task Arrival Rate

    Plots 2, 5, 6 produce a single figure each.
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

    save_dir = os.path.join(base_dir, 'plots', 'rq2', trial_name)
    os.makedirs(save_dir, exist_ok=True)

    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq2')
    os.makedirs(local_save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  LOAD AND LABEL
    # ------------------------------------------------------------------
    df = pd.read_csv(compiled_results_path)

    experiment_col = 'in_validation'
    if experiment_col not in df.columns:
        raise ValueError(
            f"Experiment column `{experiment_col}` not found. "
            f"Available columns: {df.columns.tolist()}")

    if df.empty:
        print(f"No results found for trial `{trial_name}`.")
        return

    df['Algorithm'] = df.apply(label_condition, axis=1)

    algo_order = [
        'Preplanning only', 'Greedy only', 'DP + Greedy', 'CBBA only', 'DP + CBBA'
    ]
    # Wong (2011) colorblind-safe palette
    algo_palette = {
        'Preplanning only': '#999999',  # gray
        'Greedy only':      '#E69F00',  # orange
        'DP + Greedy':      '#56B4E9',  # sky blue
        'CBBA only':        '#009E73',  # teal
        'DP + CBBA':        '#0072B2',  # deep blue
    }
    linestyles = {
        'Preplanning only': (4, 2),
        'Greedy only':      (1, 2),
        'DP + Greedy':      (4, 1, 1, 1),
        'CBBA only':        (3, 1),
        'DP + CBBA':        '',
    }

    # Pre-aggregate all metrics once — one mean per design cell
    cell_metrics = [
        'Total Obtained Reward [norm]',
        'P(Task Observed)',
        'P(Task Observed | Task Observable)',
        'P(Event Co-observed)',
        'Average Normalized Response Time to Task',
    ]
    agg = aggregate_cells(df, cell_metrics)

    # ------------------------------------------------------------------
    #  PLOT 1 — Mission Reward
    #  Figure A: 1 panel.   Figure B: 1x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='Total Obtained Reward [norm]',
        titles='Mission Reward by Algorithm',
        ylabels='Total Obtained Reward (normalized)',
        suptitle='Mission Reward by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot1-Mission_Reward',
    )

    # ------------------------------------------------------------------
    #  PLOT 2 — Reward Scaling
    #  Panel 1: reward vs Task Arrival Rate (log scale line plot)
    #  Panel 2: reward vs Num Sats (line plot)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (x_col, xlabel, xscale) in zip(axes, [
        ('Task Arrival Rate', r'Task Arrival Rate $\lambda$ (1/day)', 'log'),
        ('Num Sats',          'Number of Satellites',                 'log'),
    ]):
        sns.lineplot(
            data=df,
            x=x_col, y='Total Obtained Reward [norm]',
            hue='Algorithm', style='Algorithm',
            hue_order=algo_order, style_order=algo_order,
            palette=algo_palette, dashes=linestyles,
            errorbar=('ci', 95), ax=ax,
        )

        ax.set_xscale(xscale)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
                alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Total Obtained Reward (normalized)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.4)

        # legend only on the right panel
        if "task" in x_col.lower():
            ax.get_legend().remove()
        else:
            ax.legend(title='Algorithm')

    plt.suptitle('Reward Scaling', fontsize=13)
    plt.tight_layout()

    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
            'Plot2-Reward_Scaling.png')
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
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot3-Task_Observation_Probability',
    )

    # ------------------------------------------------------------------
    #  PLOT 4 — Multi-Agent Coordination Quality
    #  Figure A: 1 panel.   Figure B: 1x2.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='P(Event Co-observed)',
        titles='Multi-Agent Coordination Quality by Algorithm',
        ylabels=r'$P(\mathrm{Event\ Co-observed})$',
        suptitle='Multi-Agent Coordination Quality by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot4-Coordination_Quality',
    )

    # ------------------------------------------------------------------
    #  PLOT 5 — Task Response Time by Algorithm
    #  Violin of raw trials — response time is less sensitive to Num Sats
    #  and Task Arrival Rate so the full distribution is informative here.
    #  Single figure only.
    # ------------------------------------------------------------------
    plot_metric(
        df, agg,
        metrics='Average Normalized Response Time to Task',
        titles='Average Normalized Task Response Time by Algorithm',
        ylabels=r'$P(\mathrm{Event\ Co-observed})$',
        suptitle='Average Normalized Task Response Time by Algorithm',
        algo_order=algo_order, algo_palette=algo_palette,
        base_dir=base_dir, local_base_dir=local_base_dir,
        save_dir=save_dir, local_save_dir=local_save_dir,
        filename_stem='Plot5-Task_Response_Time',
    )

    # ------------------------------------------------------------------
    #  PLOT 6 — Response Time vs Observation Quality
    #  Scatter of individual trials (low alpha) with per-algorithm
    #  mean ± std error bars. Two panels share x-axis (response time)
    #  to show CBBA only and Greedy only diverging on observation quality
    #  despite similar response times. Reference lines and an arrow
    #  between Greedy only and CBBA only make the key finding explicit.
    # ------------------------------------------------------------------
    summary = df.groupby('Algorithm').agg(
        rt_mean=('Average Normalized Response Time to Task', 'mean'),
        rt_std=('Average Normalized Response Time to Task', 'std'),
        obs_mean=('P(Task Observed)', 'mean'),
        obs_std=('P(Task Observed)', 'std'),
        cobs_mean=('P(Event Co-observed)', 'mean'),
        cobs_std=('P(Event Co-observed)', 'std'),
    ).reindex(algo_order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (y_col, y_mean, y_std, ylabel) in zip(axes, [
        ('P(Task Observed)',
         'obs_mean', 'obs_std',
         r'$P(\mathrm{Task\ Observed})$'),
        ('P(Event Co-observed)',
         'cobs_mean', 'cobs_std',
         r'$P(\mathrm{Event\ Co-observed})$'),
    ]):
        # --- raw trial scatter (kept faint — context only) ---
        for algo in algo_order:
            sub = df[df['Algorithm'] == algo]
            ax.scatter(
                sub['Average Normalized Response Time to Task'],
                sub[y_col],
                color=algo_palette[algo], alpha=0.08, s=18, zorder=1,
            )

        # --- reference lines at Greedy only mean position ---
        greedy_rt = summary.loc['Greedy only', 'rt_mean']
        greedy_y  = summary.loc['Greedy only', y_mean]

        ax.axvline(greedy_rt, color=algo_palette['Greedy only'],
                   linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(greedy_y,  color=algo_palette['Greedy only'],
                   linestyle=':', linewidth=0.8, alpha=0.5)

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

        # --- arrow from Greedy only to CBBA only ---
        ax.annotate(
            '',
            xy=(
                summary.loc['CBBA only', 'rt_mean'],
                summary.loc['CBBA only', y_mean],
            ),
            xytext=(
                summary.loc['Greedy only', 'rt_mean'],
                summary.loc['Greedy only', y_mean],
            ),
            arrowprops=dict(
                arrowstyle='->',
                color='black',
                lw=1.2,
                connectionstyle='arc3,rad=0.2',
            ),
            zorder=4,
        )

        # --- label the gap magnitude next to the arrow midpoint ---
        gap = summary.loc['CBBA only', y_mean] - summary.loc['Greedy only', y_mean]
        mid_x = (summary.loc['CBBA only', 'rt_mean']
                 + summary.loc['Greedy only', 'rt_mean']) / 2
        mid_y = (summary.loc['CBBA only', y_mean]
                 + summary.loc['Greedy only', y_mean]) / 2
        ax.text(
            mid_x + 0.01, mid_y + 0.025,
            rf'$\Delta={gap:+.3f}$',
            fontsize=8, color='black', ha='left',
        )

        # --- algorithm name annotations on mean markers ---
        for algo, ha, offset in [
            ('Greedy only', 'right', (-0.010,  0.020)),
            ('CBBA only',   'left',  ( 0.010,  0.020)),
        ]:
            row = summary.loc[algo]
            ax.annotate(
                algo,
                xy=(row['rt_mean'], row[y_mean]),
                xytext=(
                    row['rt_mean'] + offset[0],
                    row[y_mean]    + offset[1],
                ),
                fontsize=8, ha=ha, color=algo_palette[algo],
            )

        ax.set_xlabel('Avg. Normalized Response Time to Task')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', linewidth=0.4)

    # --- single shared legend on right panel ---
    handles = [
        plt.scatter([], [], color=algo_palette[a], marker='D', s=80, label=a)
        for a in algo_order
    ]
    axes[1].legend(handles=handles, title='Algorithm',
                   loc='lower right', fontsize=8)

    plt.suptitle(
        r'Response Time vs Observation Quality — '
        r'CBBA matches Greedy on speed, outperforms on coordination',
        fontsize=12,
    )
    plt.tight_layout()

    save_plot(base_dir, local_base_dir, save_dir, local_save_dir,
              'Plot6-Response_Time_vs_Observation_Quality.png')
    plt.close()    

# ----------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == '__main__':
    local_base_dir = os.path.join('experiments', '1_cbba_validation', 'analysis')
    base_dir = '/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local'

    trial_name = 'full_factorial_trials_2026-03-15'

    generate_plots(trial_name,
                #    base_dir=base_dir,
                   )

    print('DONE')
    