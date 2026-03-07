import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def generate_line_plots(trial_name : str, 
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
    save_dir = os.path.join(base_dir, 'plots', 'lines')
    os.makedirs(save_dir, exist_ok=True)

    # if saving to external directory, also save a copy to the local analysis directory 
    local_save_dir = os.path.join(local_base_dir, 'plots', 'lines')
    os.makedirs(local_save_dir, exist_ok=True)

    # --- Normalized Total Obtained Reward vs. Task Arrival Rate ---
    # - number of satellites as color/line style, 
    # - one panel per planner (Greedy, CBBA, Oracle)
    #  GOAL: see both how CBBA scales with load and how close it stays to the oracle upper bound.

    # f, ax = plt.subplots(figsize=(7, 6))
    # ax.set_xscale("log")

    # sns.lineplot(data=filtered_df, 
    #             y='Normalized Total Obtained Reward',
    #             x='Task Arrival Rate',
    #             # hue='Num Sats',
    #             hue='Replanner',
    #             # row='Replanner',
    #             # kind="line",
    #             palette="tab10", 
    #             err_style="bars",
    #             ax=ax
    #             )

    sns.boxplot(data=filtered_df,
                # y='Normalized Total Obtained Reward',
                y='Total Obtained Reward [norm]',
                x='Task Arrival Rate',
                hue="Replanner", 
                # palette=["m", "g"],
            )
    
    # define filename for plot
    plot_filename = f'{trial_name}-{str("Normalized Total Obtained Reward").replace(" ", "_").replace("[norm]","norm")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    x = 1

    # # --- generate line plots for key metrics ---
    # # define metrics to plot
    # y_metrics = [
    #             # 'Total Planned Reward [norm]',   
    #             # 'Total Planned Utility [norm]',      
    #             'P(Task Observed | Task Observable)',
    #             'Average Response Time to Task [s]',
    #             # 'Average Normalized Response Time to Event',
    #             'Average Normalized Response Time to Task',
    #             'Total Messages Broadcasted',
    #             'Average Messages Broadcasted per Task',
    #             # 'Total Obtained Utility',
    #             'Normalized Total Obtained Utility',
    #             # 'Total Obtained Reward',
    #             'Normalized Total Obtained Reward'
    #         ]
    # # define x-axis metric
    # x_metrics = [
    #     'Task Arrival Rate',
    #     'Num Sats',
    # ]

    # for y_metric in y_metrics:
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    #     for j, x_metric in enumerate(x_metrics):
    #         ax = axes[j]
    #         sns.lineplot(
    #             data=filtered_df,
    #             x=x_metric, y=y_metric,
    #             # hue="Ground Segment",
    #             hue = "Replanner",
    #             style="Latency",
    #             ax=ax
    #         )
    #         ax.set_title(f"{y_metric} vs {x_metric}")
    #         ax.tick_params(axis="x", rotation=45)

    #     fig.tight_layout()

        # # define save directory and filename for plot
        # save_dir = os.path.join(base_dir, 'plots', 'lines')
        # os.makedirs(save_dir, exist_ok=True)
        # plot_filename = f'{trial_name}-{y_metric.replace(" ", "_").replace("[norm]","norm")}.png'
        
        # # save plot if desired
        # save_path = os.path.join(save_dir, plot_filename)
        # plt.savefig(save_path)

        # # if saving to external directory, also save a copy to the local analysis directory 
        # local_save_dir = os.path.join(local_base_dir, 'plots', 'lines')
        # os.makedirs(local_save_dir, exist_ok=True)
        # local_save_path = os.path.join(local_save_dir, plot_filename)
        
        # if base_dir != local_base_dir:
        #     plt.savefig(local_save_path)

        # # print completion message with paths to saved plots
        # print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":  
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local"

    # trial_name = "full_factorial_trials_2026-02-22"
    trial_name = "full_factorial_trials_2026-02-23"

    # experiment_col = "in_stress"
    experiment_col = "in_connectivity"
    # experiment_col = "in_validation"

    # generate bar plots for the specified trial and parameters
    generate_line_plots(trial_name, 
                        experiment_col=experiment_col,
                        # base_dir=base_dir
                        )