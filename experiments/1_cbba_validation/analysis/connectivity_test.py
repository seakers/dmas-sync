
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

def generate_plots(trial_name : str, 
                    experiment_col : str,
                    base_dir : str = None) -> None:
    """ Generates line plots for the compiled results of a given trial, showing the impact of number of satellites and arrival rate on key metrics. """

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
    
    # load compiled results
    results_df = pd.read_csv(compiled_results_path)

    # filter results to only include rows where experiment_col is True
    if experiment_col not in results_df.columns:
        raise ValueError(f"Experiment column `{experiment_col}` not found in results DataFrame. Available columns: {results_df.columns.tolist()}")
    filtered_df = results_df[results_df[experiment_col] == True]

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
    
    # define save directory and filename for plot
    save_dir = os.path.join(base_dir, 'plots', 'rq3', trial_name)
    os.makedirs(save_dir, exist_ok=True)

    # if saving to external directory, also save a copy to the local analysis directory 
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq3')
    os.makedirs(local_save_dir, exist_ok=True)

    # Create a palette keyed to your Num Sats values
    num_sats_values = sorted(filtered_df['Num Sats'].unique())
    palette = sns.color_palette("rocket_r", n_colors=len(num_sats_values))
    color_map = dict(zip(num_sats_values, palette))

    # --- Runtime vs. Num of Tasks ---
    # filter results to only include rows with CBBA replanner
    cbba_df = filtered_df[filtered_df['Replanner'] == 'CBBA']

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    latency_dashes = {
        'Low':    '',           # solid — no interruption
        'Medium': (4, 2),       # dashed
        'High':   (1, 2),       # dotted — most "broken up"
    }

    sns.lineplot(data=cbba_df, 
                y='Simulation Runtime [s]',
                x='Task Arrival Rate',
                hue='Num Sats',
                style='Latency',
                palette=color_map,
                dashes=latency_dashes,
                ax=ax
            )
        
    # plt.grid(True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    ax.set_ylabel('Simulation Runtime (s)')    
    plt.tight_layout()

    # define filename for plot
    plot_filename = f'{str("Simulation Runtime (s)").replace(" ", "_").replace("(s)","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # --- Reward vs. Latency ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, replanner in zip(axes, ['CBBA', 'Greedy']):
        subset = filtered_df[filtered_df['Replanner'] == replanner]
        summary = subset.groupby(['Num Sats', 'Latency'])['Total Obtained Reward [norm]'].mean().reset_index()
        
        sns.barplot(data=summary, x='Num Sats', y='Total Obtained Reward [norm]',
                    hue='Latency', hue_order=['Low', 'Medium', 'High'],
                    palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
                    errorbar='sd', 
                    ax=ax)
        ax.set_title(rf'Replanner: {replanner}')
        ax.set_xlabel('Number of Satellites')
        ax.set_ylabel('Total Obtained Reward (normalized)')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.legend(title='Latency')

    plt.suptitle(r'Effect of Latency on Mission Reward — RQ3', fontsize=13)
    plt.tight_layout()

    # define filename for plot
    plot_filename = f'{str("Reward vs Latency").replace(" ", "_").replace("(s)","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # # --- Response Time vs. Latency ---
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # for ax, replanner in zip(axes, ['CBBA', 'Greedy']):
    #     subset = filtered_df[filtered_df['Replanner'] == replanner]
    #     summary = subset.groupby(['Num Sats', 'Latency'])['Total Obtained Reward [norm]'].mean().reset_index()
        
    #     sns.barplot(data=summary, 
    #                 x='Num Sats', 
    #                 y='Total Obtained Reward [norm]',
    #                 hue='Latency', hue_order=['Low', 'Medium', 'High'],
    #                 palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
    #                 errorbar='sd',                    
    #                 ax=ax)
    #     ax.set_title(rf'Replanner: {replanner}')
    #     ax.set_xlabel('Number of Satellites')
    #     ax.set_ylabel('Total Obtained Reward (normalized)')
    #     ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    #     ax.legend(title='Latency')

    # plt.suptitle(r'Effect of Latency on Mission Reward — RQ3', fontsize=13)
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'{str("Reward vs Latency").replace(" ", "_").replace("(s)","s")}.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # --- #N Messages vs. Num of Tasks ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # sns.lineplot(data=filtered_df, 
    #             y='Total Messages Broadcasted',
    #             # y='Average Messages Broadcasted per Task',
    #             x='Task Arrival Rate',
    #             hue='Num Sats',
    #             # col='Replanner',
    #             # row='Replanner',
    #             # kind="line",
    #             # palette="rocket_r", 
    #             palette=color_map,
    #             # err_style="bars",
    #             markers=True, 
    #             dashes=False,
    #             ax=ax
    #         )
        
    # # plt.grid(True)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    # ax.set_ylabel('Total Messages Broadcasted')    
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'{str("Total Messages Broadcasted").replace(" ", "_").replace("(s)","s")}.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # # --- #N Messages / Task vs. Num of Tasks ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # sns.lineplot(data=filtered_df, 
    #             y='Average Messages Broadcasted per Task',
    #             x='Task Arrival Rate',
    #             hue='Num Sats',
    #             # col='Replanner',
    #             # row='Replanner',
    #             # kind="line",
    #             # palette="rocket_r", 
    #             palette=color_map,
    #             # err_style="bars",
    #             markers=True, 
    #             dashes=False,
    #             ax=ax
    #         )
        
    # # plt.grid(True)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    # ax.set_ylabel('Average Messages Broadcasted per Task')
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'{str("Average Messages Broadcasted per Task").replace(" ", "_").replace("(s)","s")}.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # # --- Reward vs. Num of Tasks ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # # ax.set_yscale("log")

    # # Scatter plot
    # sns.scatterplot(
    #     data=filtered_df,
    #     y='Total Obtained Reward [norm]',
    #     x='Task Arrival Rate',
    #     hue='Num Sats',
    #     palette=color_map,
    #     ax=ax,
    # )

    # for num_sats, group in filtered_df.groupby('Num Sats'):
    #     x = group['Task Arrival Rate'].values
    #     y = group['Total Obtained Reward [norm]'].values

    #     # Log-linear fit: y = a + b*log10(x)
    #     slope, intercept, r, _, se = stats.linregress(np.log10(x), y)

    #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    #     y_fit = intercept + slope * np.log10(x_fit)

    #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    #             linestyle='--', alpha=0.7, label='_nolegend_')

    #     x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
    #     y_mid = intercept + slope * np.log10(x_mid)
    #     ax.text(x_mid, y_mid + 0.02, rf'$b={slope:.2f}$',
    #             fontsize=7, color=color_map[num_sats], ha='center')

    # # for num_sats, group in filtered_df.groupby('Num Sats'):
    # #     x = group['Task Arrival Rate'].values
    # #     y = group['Total Obtained Reward [norm]'].values
    # #     log_x = np.log10(x)

    # #     slope, intercept, r, p, se = linregress(log_x, y)

    # #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    # #     log_x_fit = np.log10(x_fit)
    # #     y_fit = intercept + slope * log_x_fit

    # #     # 95% confidence interval
    # #     n = len(x)
    # #     t_val = stats.t.ppf(0.975, df=n - 2)
    # #     x_mean = log_x.mean()
    # #     se_line = se * np.sqrt(1/n + (log_x_fit - x_mean)**2 / np.sum((log_x - x_mean)**2))

    # #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    # #             linestyle='--', alpha=0.8, label='_nolegend_')
    # #     ax.fill_between(x_fit,
    # #                     y_fit - t_val * se_line,
    # #                     y_fit + t_val * se_line,
    # #                     color=color_map[num_sats], alpha=0.12)

    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    # ax.set_ylabel('Total Obtained Reward (normalized)')
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'{str("Total Obtained Reward (normalized)").replace(" ", "_").replace("(normalized)","normalized")}.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")
    
    # # --- Utility vs. Num of Tasks ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # # ax.set_yscale("log")

    # # Scatter plot
    # sns.scatterplot(
    #     data=filtered_df,
    #     y='Total Obtained Utility [norm]',
    #     x='Task Arrival Rate',
    #     hue='Num Sats',
    #     palette=color_map,
    #     ax=ax,
    # )

    # for num_sats, group in filtered_df.groupby('Num Sats'):
    #     x = group['Task Arrival Rate'].values
    #     y = group['Total Obtained Utility [norm]'].values

    #     # Log-linear fit: y = a + b*log10(x)
    #     slope, intercept, r, _, se = stats.linregress(np.log10(x), y)

    #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    #     y_fit = intercept + slope * np.log10(x_fit)

    #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    #             linestyle='--', alpha=0.7, label='_nolegend_')

    #     x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
    #     y_mid = intercept + slope * np.log10(x_mid)
    #     ax.text(x_mid, y_mid + 0.02, rf'$b={slope:.2f}$',
    #             fontsize=7, color=color_map[num_sats], ha='center')

    # # for num_sats, group in filtered_df.groupby('Num Sats'):
    # #     x = group['Task Arrival Rate'].values
    # #     y = group['Total Obtained Utility [norm]'].values
    # #     log_x = np.log10(x)

    # #     slope, intercept, r, p, se = linregress(log_x, y)

    # #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    # #     log_x_fit = np.log10(x_fit)
    # #     y_fit = intercept + slope * log_x_fit

    # #     # 95% confidence interval
    # #     n = len(x)
    # #     t_val = stats.t.ppf(0.975, df=n - 2)
    # #     x_mean = log_x.mean()
    # #     se_line = se * np.sqrt(1/n + (log_x_fit - x_mean)**2 / np.sum((log_x - x_mean)**2))

    # #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    # #             linestyle='--', alpha=0.8, label='_nolegend_')
    # #     ax.fill_between(x_fit,
    # #                     y_fit - t_val * se_line,
    # #                     y_fit + t_val * se_line,
    # #                     color=color_map[num_sats], alpha=0.12)

    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Task Arrival Rate $\lambda$ (1/day)')
    # ax.set_ylabel('Total Obtained Utility (normalized)')
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'{str("Total Obtained Utility (normalized)").replace(" ", "_").replace("(normalized)","normalized")}.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # # --- #N Messages vs. Runtime ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # # Scatter plot
    # sns.scatterplot(
    #     data=filtered_df,
    #     x='Total Messages Broadcasted',
    #     y='Simulation Runtime [s]',
    #     hue='Num Sats',
    #     palette=color_map,
    #     ax=ax,
    # )

    # # Power-law trendline per group
    # for num_sats, group in filtered_df.groupby('Num Sats'):
    #     x = group['Total Messages Broadcasted'].values
    #     y = group['Simulation Runtime [s]'].values

    #     # Fit in log-log space → power law y = a * x^b
    #     slope, intercept, r, _, _ = stats.linregress(np.log10(x), np.log10(y))

    #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    #     y_fit = 10**intercept * x_fit**slope

    #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    #             linestyle='--', alpha=0.7, label='_nolegend_')

    #     # Optional: annotate slope on the trendline
    #     x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
    #     y_mid = 10**intercept * x_mid**slope
    #     ax.text(x_mid, y_mid * 1.15, f'b={slope:.1f}',
    #             fontsize=7, color=color_map[num_sats], ha='center')

    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Total Messages Broadcasted')
    # ax.set_ylabel('Simulation Runtime (s)')
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'Relplot-Total_Messages_Broadcasted-vs-Simulation_Runtime.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # # --- #N Messages / Task vs. Runtime ---
    # f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # # Scatter plot
    # sns.scatterplot(
    #     data=filtered_df,
    #     x='Average Messages Broadcasted per Task',
    #     y='Simulation Runtime [s]',
    #     hue='Num Sats',
    #     palette=color_map,
    #     ax=ax,
    # )

    # # Power-law trendline per group
    # for num_sats, group in filtered_df.groupby('Num Sats'):
    #     x = group['Average Messages Broadcasted per Task'].values
    #     y = group['Simulation Runtime [s]'].values

    #     # Fit in log-log space → power law y = a * x^b
    #     slope, intercept, r, _, _ = stats.linregress(np.log10(x), np.log10(y))

    #     x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    #     y_fit = 10**intercept * x_fit**slope

    #     ax.plot(x_fit, y_fit, color=color_map[num_sats], linewidth=1.5,
    #             linestyle='--', alpha=0.7, label='_nolegend_')

    #     # Optional: annotate slope on the trendline
    #     x_mid = 10**((np.log10(x.min()) + np.log10(x.max())) / 2)
    #     y_mid = 10**intercept * x_mid**slope
    #     ax.text(x_mid, y_mid * 1.15, f'b={slope:.1f}',
    #             fontsize=7, color=color_map[num_sats], ha='center')

    # ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    # ax.set_xlabel('Average Messages Broadcasted per Task')
    # ax.set_ylabel('Simulation Runtime (s)')
    # plt.tight_layout()

    # # define filename for plot
    # plot_filename = f'Relplot-Average_Messages_Broadcasted_per_Task-vs-Simulation_Runtime.png'    
    # save_path = os.path.join(save_dir, plot_filename)
    # local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # # save plot 
    # plt.savefig(save_path)
    # if base_dir != local_base_dir:
    #     plt.savefig(local_save_path)

    # # print completion message with paths to saved plots
    # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local"

    trial_name = "full_factorial_trials_2026-02-22"
    # trial_name = "full_factorial_trials_2026-02-23"

    # experiment_col = "in_stress"
    experiment_col = "in_connectivity"
    # experiment_col = "in_validation"

    # print runtime data
    generate_plots(trial_name, 
                       experiment_col,
                    #    base_dir
                    )

    print('DONE')
