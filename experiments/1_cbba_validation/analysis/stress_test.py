
from datetime import datetime
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

# ==================================================================
#  ALTERNATE SCATTER + TRENDLINE VERSIONS
# ==================================================================

def _scatter_trendline(ax, df, x_col, y_col, hue_col, color_map, annotate=True, trendline='log-log'):
    """ Scatter plot with per-group power-law trendline (log-log fit). """
    # generate scatter plot 
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col,
                    palette=color_map, ax=ax, zorder=3)
    
    # fit and plot trendline for each group
    for group_val, group in df.groupby(hue_col):
        # unpack x and y values for this group
        x = group[x_col].values
        y = group[y_col].values
        
        # skip if too few points to fit a trendline
        if len(x) < 2: continue
        
        # fit trendline based on specified type
        if trendline == 'log-log':
            # fit in log-log space → power law y = a * x^b
            slope, intercept, r, _, _ = stats.linregress(np.log10(x), np.log10(y))
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_fit = 10**intercept * x_fit**slope
        elif trendline == 'log-x':
            # fit in semi-log space → log-linear y = a + b*log10(x)
            slope, intercept, r, _, _ = stats.linregress(np.log10(x), y)
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_fit = intercept + slope * np.log10(x_fit)
        else:
            raise ValueError(f"Unsupported trendline type: {trendline}")

        # plot trendline
        ax.plot(x_fit, y_fit, color=color_map[group_val], linewidth=1.5,
                linestyle='--', alpha=0.7, label='_nolegend_')

        # if enabled, annotate slope on the trendline
        if annotate:
            if trendline == 'log-log':
                x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
                y_mid = 10**intercept * x_mid**slope
                ax.text(x_mid, y_mid * 1.15,
                        f'$b={slope:.1f},\\ R^2={r**2:.2f}$',
                        # f'$\log(y)={slope:.1f}\log(x){"+" if intercept >= 0 else "-"}{abs(intercept):.1f}$',
                        fontsize=7, color=color_map[group_val], ha='center')
            elif trendline == 'log-x':
                x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
                y_mid = intercept + slope * np.log10(x_mid)
                ax.text(x_mid, y_mid + 0.02,
                        f'$b={slope:.2f},\\ R^2={r**2:.2f}$',
                        fontsize=7, color=color_map[group_val], ha='center')
            else:
                raise ValueError(f"Unsupported trendline type: {trendline}")

def generate_plots(trial_name : str, 
                    base_dir : str = None) -> None:
    """ Generates line plots for the compiled results of a given trial, showing the impact of number of satellites and arrival rate on key metrics. """

    # ------------------------------------------------------------------
    #  PATHS
    # ------------------------------------------------------------------
    # define base directory if not provided
    local_base_dir = os.path.join('experiments','1_cbba_validation','analysis')
    if base_dir is None: 
        base_dir = local_base_dir
        compiled_results_path = os.path.join(base_dir, 'compiled', f'{trial_name}_compiled_results.csv')
    else:
        compiled_results_path = os.path.join(base_dir, f'{trial_name}_compiled_results.csv')
    
    # check if compiled results file exists
    if not os.path.exists(compiled_results_path):
        raise FileNotFoundError(f"Compiled results file not found at: `{compiled_results_path}`. Please run `compiler.py` to compile results summaries first.")
    
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
    #  LOAD AND FILTER
    # ------------------------------------------------------------------
    # load compiled results
    results_df = pd.read_csv(compiled_results_path)

    # filter results to only include rows where experiment_col is True
    experiment_col = "in_stress"
    if experiment_col not in results_df.columns:
        raise ValueError(f"Experiment column `{experiment_col}` not found in results DataFrame. Available columns: {results_df.columns.tolist()}")
    filtered_df = results_df[results_df[experiment_col] == True]

    # filter results to only include rows with CBBA replanner
    filtered_df = filtered_df[filtered_df['Replanner'] == 'CBBA']

    # Fill in missing values
    ## preplanner column
    filtered_df['Preplanner'] = filtered_df['Preplanner'].fillna('No Preplanner')
    ## replanner column
    filtered_df['Replanner'] = filtered_df['Replanner'].fillna('No Replanner')
    ## remaining numeric columns; fill with -1 to indicate missing values 
    filtered_df = filtered_df.fillna(-1)

    # check if filtered results are empty or have too few rows to plot
    if (
        filtered_df.empty 
        # or len(filtered_df) < 3
        ):
        print(f"No full results found for trial `{trial_name}`.")
        return

    # Create a palette keyed to your Num Sats values
    num_sats_values = sorted(filtered_df['Num Sats'].unique())
    n_tasks_values = sorted(filtered_df['Task Arrival Rate'].unique())
    # sats_palette = sns.color_palette("rocket_r", n_colors=len(num_sats_values))
    # tasks_palette = sns.color_palette("rocket_r", n_colors=len(n_tasks_values))
    sats_palette = sns.color_palette("viridis", n_colors=len(num_sats_values))
    tasks_palette = sns.color_palette("viridis", n_colors=len(n_tasks_values))
    sats_color_map = dict(zip(num_sats_values, sats_palette))
    tasks_color_map = dict(zip(n_tasks_values, tasks_palette))

    # ------------------------------------------------------------------
    #  PLOT 1 (line) — Runtime (two-panel: vs. Task Arrival Rate | vs. Num Sats)
    # ------------------------------------------------------------------
    f, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))
    f.suptitle('Simulation Runtime Scaling', fontsize=14)

    # Panel (a): Runtime vs. Task Arrival Rate
    ax1a.set_xscale("log")
    ax1a.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Simulation Runtime [s]',
                x='Task Arrival Rate',
                hue='Num Sats',
                palette=sats_color_map,
                markers=True,
                dashes=False,
                ax=ax1a
            )
    ax1a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax1a.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax1a.set_ylabel('Simulation Runtime (s)')
    ax1a.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax1a.set_title('(a)')

    # Panel (b): Runtime vs. Num Sats
    ax1b.set_xscale("log")
    ax1b.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Simulation Runtime [s]',
                x='Num Sats',
                hue='Task Arrival Rate',
                palette=tasks_color_map,
                markers=True,
                dashes=False,
                ax=ax1b
            )
    ax1b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax1b.set_xlabel('Number of Satellites (log scale)')
    ax1b.set_ylabel('Simulation Runtime (s)')
    # ax1b.set_xlim(10, max(num_sats_values) * 1.1)
    ax1b.set_xlim(min(num_sats_values), max(num_sats_values))
    ax1b.set_xticks(num_sats_values)
    ax1b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1b.xaxis.set_minor_locator(ticker.NullLocator())
    ax1b.set_title('(b)')

    plt.tight_layout()

    # define filename for plot
    plot_filename = f'Plot1-Simulation_Runtime_s-line.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)

    # save plot
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 1 (scatter) — Runtime scatter+trendline
    # ------------------------------------------------------------------
    f, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))
    f.suptitle('Simulation Runtime Scaling', fontsize=14)

    ax1a.set_xscale("log"); ax1a.set_yscale("log")
    _scatter_trendline(ax1a, filtered_df, 'Task Arrival Rate', 'Simulation Runtime [s]',
                       'Num Sats', sats_color_map)
    ax1a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax1a.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax1a.set_ylabel('Simulation Runtime (s)')
    # ax1a.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax1a.set_title('(a)')

    ax1b.set_xscale("log"); ax1b.set_yscale("log")
    _scatter_trendline(ax1b, filtered_df, 'Num Sats', 'Simulation Runtime [s]',
                       'Task Arrival Rate', tasks_color_map)
    ax1b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax1b.set_xlabel('Number of Satellites (log scale)')
    ax1b.set_ylabel('Simulation Runtime (s)')
    ax1b.set_xlim(10, max(num_sats_values) * 1.1)
    ax1b.set_xticks(num_sats_values)
    ax1b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1b.xaxis.set_minor_locator(ticker.NullLocator())
    ax1b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot1-Simulation_Runtime_s-scatter.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 2a (line) — Messages (two-panel: total | per task) vs. Task Arrival Rate
    # ------------------------------------------------------------------
    # f, (ax2aa, ax2ab) = plt.subplots(1, 2, figsize=(16, 6))
    f, (ax2aa, ax2ab) = plt.subplots(2, 1, figsize=(8, 12))
    f.suptitle('Message Scaling with Task Arrival Rate', fontsize=14)

    # Panel (a): Total Messages Broadcasted
    ax2aa.set_xscale("log")
    ax2aa.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Total Messages Broadcasted',
                x='Task Arrival Rate',
                hue='Num Sats',
                palette=sats_color_map,
                markers=True,
                dashes=False,
                ax=ax2aa
            )
    ax2aa.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax2aa.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax2aa.set_xlabel('')
    ax2aa.tick_params(axis='x', labelbottom=False)
    ax2aa.set_ylabel('Total Messages Broadcasted')
    ax2aa.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax2aa.set_title('(a)')

    # Panel (b): Average Messages Broadcasted per Task
    ax2ab.set_xscale("log")
    ax2ab.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Average Messages Broadcasted per Task',
                x='Task Arrival Rate',
                hue='Num Sats',
                palette=sats_color_map,
                markers=True,
                dashes=False,
                legend=False,
                ax=ax2ab
            )
    ax2ab.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax2ab.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax2ab.set_ylabel('Average Messages Broadcasted per Task')
    ax2ab.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax2ab.set_title('(b)')

    plt.tight_layout()

    # define filename for plot
    plot_filename = f'Plot2a-Messages_Broadcasted_vs_Task_Arrival_Rate-line.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)

    # save plot
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 2a (scatter) — Messages vs. Task Arrival Rate scatter+trendline
    # ------------------------------------------------------------------
    # f, (ax2aa, ax2ab) = plt.subplots(1, 2, figsize=(16, 6))
    f, (ax2aa, ax2ab) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    f.suptitle('Message Scaling with Task Arrival Rate', fontsize=14)

    ax2aa.set_xscale("log"); ax2aa.set_yscale("log")
    _scatter_trendline(ax2aa, filtered_df, 'Task Arrival Rate', 'Total Messages Broadcasted',
                       'Num Sats', sats_color_map)
    ax2aa.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax2aa.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax2aa.set_xlabel('')
    ax2aa.tick_params(axis='x', labelbottom=False)
    ax2aa.set_ylabel('Total Messages Broadcasted')
    # ax2aa.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax2aa.set_title('(a)')

    ax2ab.set_xscale("log"); ax2ab.set_yscale("log")
    _scatter_trendline(ax2ab, filtered_df, 'Task Arrival Rate', 'Average Messages Broadcasted per Task',
                       'Num Sats', sats_color_map, annotate=True)
    ax2ab.get_legend().remove()
    ax2ab.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax2ab.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax2ab.set_ylabel('Average Messages Broadcasted per Task')
    # ax2ab.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax2ab.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot2a-Messages_Broadcasted_vs_Task_Arrival_Rate-scatter.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 2b (line) — Messages (two-panel: total | per task) vs. Num Sats
    # ------------------------------------------------------------------
    # f, (ax2ba, ax2bb) = plt.subplots(1, 2, figsize=(16, 6))
    f, (ax2ba, ax2bb) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    f.suptitle('Message Scaling with Number of Satellites', fontsize=14)

    # Panel (a): Total Messages Broadcasted
    ax2ba.set_xscale("log")
    ax2ba.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Total Messages Broadcasted',
                x='Num Sats',
                hue='Task Arrival Rate',
                palette=tasks_color_map,
                markers=True,
                dashes=False,
                legend=False,
                ax=ax2ba
            )
    ax2ba.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax2ba.set_xlabel('Number of Satellites (log scale)')
    ax2aa.set_xlabel('')
    ax2ba.set_ylabel('Total Messages Broadcasted')
    ax2ba.set_xlim(10, max(num_sats_values) * 1.1)
    ax2ba.set_xticks(num_sats_values)
    ax2ba.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2ba.xaxis.set_minor_locator(ticker.NullLocator())
    ax2ba.set_title('(a)')

    # Panel (b): Average Messages Broadcasted per Task
    ax2bb.set_xscale("log")
    ax2bb.set_yscale("log")
    sns.lineplot(data=filtered_df,
                y='Average Messages Broadcasted per Task',
                x='Num Sats',
                hue='Task Arrival Rate',
                palette=tasks_color_map,
                markers=True,
                dashes=False,
                ax=ax2bb
            )
    ax2bb.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax2bb.set_xlabel('Number of Satellites (log scale)')
    ax2bb.set_ylabel('Average Messages Broadcasted per Task')
    # ax2bb.set_xlim(10, max(num_sats_values) * 1.1)
    ax2bb.set_xlim(min(num_sats_values), max(num_sats_values))
    ax2bb.set_xticks(num_sats_values)
    ax2bb.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2bb.xaxis.set_minor_locator(ticker.NullLocator())
    ax2bb.set_title('(b)')

    plt.tight_layout()

    # define filename for plot
    plot_filename = f'Plot2b-Messages_Broadcasted_vs_Num_Sats-line.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)

    # save plot
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")    

    # ------------------------------------------------------------------
    #  PLOT 2b (scatter) — Messages vs. Num Sats scatter+trendline
    # ------------------------------------------------------------------
    # f, (ax2ba, ax2bb) = plt.subplots(1, 2, figsize=(16, 6))
    f, (ax2ba, ax2bb) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    f.suptitle('Message Scaling with Number of Satellites', fontsize=14)

    ax2ba.set_xscale("log"); ax2ba.set_yscale("log")
    _scatter_trendline(ax2ba, filtered_df, 'Num Sats', 'Total Messages Broadcasted',
                       'Task Arrival Rate', tasks_color_map)
    ax2ba.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax2ba.set_xlabel('Number of Satellites (log scale)')
    ax2aa.set_xlabel('')
    ax2ba.set_ylabel('Total Messages Broadcasted')
    ax2ba.set_xlim(10, max(num_sats_values) * 1.1)
    ax2ba.set_xticks(num_sats_values)
    ax2ba.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2ba.xaxis.set_minor_locator(ticker.NullLocator())
    ax2ba.set_title('(a)')

    ax2bb.set_xscale("log"); ax2bb.set_yscale("log")
    _scatter_trendline(ax2bb, filtered_df, 'Num Sats', 'Average Messages Broadcasted per Task',
                       'Task Arrival Rate', tasks_color_map, annotate=True)
    ax2bb.get_legend().remove()
    ax2bb.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax2bb.set_xlabel('Number of Satellites (log scale)')
    ax2bb.set_ylabel('Average Messages Broadcasted per Task')
    ax2bb.set_xlim(10, max(num_sats_values) * 1.1)
    ax2bb.set_xticks(num_sats_values)
    ax2bb.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2bb.xaxis.set_minor_locator(ticker.NullLocator())
    ax2bb.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot2b-Messages_Broadcasted_vs_Num_Sats-scatter.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 3a (scatter) — Runtime vs. Messages, hue = Num Sats
    # ------------------------------------------------------------------
    f, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f.suptitle('Simulation Runtime vs. Messages Broadcasted', fontsize=14)

    # Panel (a): Runtime vs. Total Messages Broadcasted
    ax3a.set_xscale("log")
    ax3a.set_yscale("log")
    _scatter_trendline(ax3a, filtered_df, 'Total Messages Broadcasted', 'Simulation Runtime [s]',
                       'Num Sats', sats_color_map)
    ax3a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax3a.set_xlabel('Total Messages Broadcasted')
    ax3a.set_ylabel('Simulation Runtime (s)')
    ax3a.set_title('(a)')

    # Panel (b): Runtime vs. Average Messages Broadcasted per Task
    ax3b.set_xscale("log")
    ax3b.set_yscale("log")
    _scatter_trendline(ax3b, filtered_df, 'Average Messages Broadcasted per Task', 'Simulation Runtime [s]',
                       'Num Sats', sats_color_map, annotate=True)
    ax3b.get_legend().remove()
    ax3b.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax3b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3b.xaxis.set_minor_locator(ticker.NullLocator())
    ax3b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax3b.set_xlabel('Average Messages Broadcasted per Task (log scale)')
    ax3b.set_ylabel('Simulation Runtime (s)')
    ax3b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot3a-Runtime_vs_Messages_Broadcasted.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 3b (scatter) — Runtime vs. Messages, hue = Task Arrival Rate
    # ------------------------------------------------------------------
    f, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f.suptitle('Simulation Runtime vs. Messages Broadcasted', fontsize=14)

    # Panel (a): Runtime vs. Total Messages Broadcasted
    ax3a.set_xscale("log")
    ax3a.set_yscale("log")
    _scatter_trendline(ax3a, filtered_df, 'Total Messages Broadcasted', 'Simulation Runtime [s]',
                       'Task Arrival Rate', tasks_color_map)
    ax3a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax3a.set_xlabel('Total Messages Broadcasted')
    ax3a.set_ylabel('Simulation Runtime (s)')
    ax3a.set_title('(a)')

    # Panel (b): Runtime vs. Average Messages Broadcasted per Task
    ax3b.set_xscale("log")
    ax3b.set_yscale("log")
    _scatter_trendline(ax3b, filtered_df, 'Average Messages Broadcasted per Task', 'Simulation Runtime [s]',
                       'Task Arrival Rate', tasks_color_map, annotate=True)
    ax3b.get_legend().remove()
    ax3b.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax3b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3b.xaxis.set_minor_locator(ticker.NullLocator())
    ax3b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax3b.set_xlabel('Average Messages Broadcasted per Task (log scale)')
    ax3b.set_ylabel('Simulation Runtime (s)')
    ax3b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot3b-Runtime_vs_Messages_Broadcasted.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 4a — Total Obtained Reward (two-panel: vs. Task Arrival Rate | vs. Num Sats)
    # ------------------------------------------------------------------
    f, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f.suptitle('Total Obtained Reward', fontsize=14)

    # Panel (a): x = Task Arrival Rate, hue = Num Sats
    ax4a.set_xscale("log"); ax4a.set_yscale("log")
    _scatter_trendline(ax4a, filtered_df, 'Task Arrival Rate', 'Total Obtained Reward',
                       'Num Sats', sats_color_map, trendline='log-log')
    ax4a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax4a.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax4a.set_ylabel('Total Obtained Reward')
    # ax4a.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax4a.set_title('(a)')

    # Panel (b): x = Num Sats, hue = Task Arrival Rate
    ax4b.set_xscale("log"); ax4b.set_yscale("log")
    _scatter_trendline(ax4b, filtered_df, 'Num Sats', 'Total Obtained Reward',
                       'Task Arrival Rate', tasks_color_map, trendline='log-log')
    ax4b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax4b.set_xlabel('Number of Satellites (log scale)')
    ax4b.set_ylabel('Total Obtained Reward')
    ax4b.set_xlim(10, max(num_sats_values) * 1.1)
    ax4b.set_xticks(num_sats_values)
    ax4b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4b.xaxis.set_minor_locator(ticker.NullLocator())
    ax4b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot4a-Total_Obtained_Reward.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 4b — Total Obtained Reward [norm] (two-panel: vs. Task Arrival Rate | vs. Num Sats)
    # ------------------------------------------------------------------
    f, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f.suptitle('Total Obtained Reward (normalized)', fontsize=14)

    # Panel (a): x = Task Arrival Rate, hue = Num Sats
    ax4a.set_xscale("log")
    _scatter_trendline(ax4a, filtered_df, 'Task Arrival Rate', 'Total Obtained Reward [norm]',
                       'Num Sats', sats_color_map, trendline='log-x')
    ax4a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax4a.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax4a.set_ylabel('Total Obtained Reward (normalized)')
    # ax4a.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax4a.set_title('(a)')

    # Panel (b): x = Num Sats, hue = Task Arrival Rate
    ax4b.set_xscale("log")
    _scatter_trendline(ax4b, filtered_df, 'Num Sats', 'Total Obtained Reward [norm]',
                       'Task Arrival Rate', tasks_color_map, trendline='log-x')
    ax4b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax4b.set_xlabel('Number of Satellites (log scale)')
    ax4b.set_ylabel('Total Obtained Reward (normalized)')
    ax4b.set_xlim(10, max(num_sats_values) * 1.1)
    ax4b.set_xticks(num_sats_values)
    ax4b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4b.xaxis.set_minor_locator(ticker.NullLocator())
    ax4b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot4b-Total_Obtained_Reward_normalized.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # ------------------------------------------------------------------
    #  PLOT 5a — Utility vs. Task Arrival Rate, hue = Num Sats
    #  PLOT 5b — Utility vs. Num Sats, hue = Task Arrival Rate
    # ------------------------------------------------------------------
    f, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    f.suptitle('Obtained Utility (normalized)', fontsize=14)

    # Panel (a): x = Task Arrival Rate, hue = Num Sats
    ax5a.set_xscale("log")
    _scatter_trendline(ax5a, filtered_df, 'Task Arrival Rate', 'Total Obtained Utility [norm]',
                       'Num Sats', sats_color_map, trendline='log-x')
    ax5a.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax5a.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax5a.set_ylabel('Total Obtained Utility (normalized)')
    # ax5a.set_xlim(min(n_tasks_values), max(n_tasks_values))
    ax5a.set_title('(a)')

    # Panel (b): x = Num Sats, hue = Task Arrival Rate
    ax5b.set_xscale("log")
    _scatter_trendline(ax5b, filtered_df, 'Num Sats', 'Total Obtained Utility [norm]',
                       'Task Arrival Rate', tasks_color_map, trendline='log-x', annotate=True)
    ax5b.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax5b.set_xlabel('Number of Satellites (log scale)')
    ax5b.set_ylabel('Total Obtained Utility (normalized)')
    ax5b.set_xlim(10, max(num_sats_values) * 1.1)
    ax5b.set_xticks(num_sats_values)
    ax5b.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax5b.xaxis.set_minor_locator(ticker.NullLocator())
    ax5b.set_title('(b)')

    plt.tight_layout()
    plot_filename = 'Plot5-Utility_normalized.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local"

    # trial_name = "full_factorial_trials_2026-02-22"
    # trial_name = "full_factorial_trials_2026-02-23"
    trial_name = "full_factorial_trials_2026-03-15"

    # print runtime data
    generate_plots(trial_name, 
                    #    base_dir
                    )

    print('DONE')
