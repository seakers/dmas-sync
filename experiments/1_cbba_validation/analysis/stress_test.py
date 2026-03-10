
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def print_runtime_data(trial_name : str, 
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
    
    # define save directory and filename for plot
    save_dir = os.path.join(base_dir, 'plots', 'rq1')
    os.makedirs(save_dir, exist_ok=True)

    # if saving to external directory, also save a copy to the local analysis directory 
    local_save_dir = os.path.join(local_base_dir, 'plots', 'rq1')
    os.makedirs(local_save_dir, exist_ok=True)

    # --- Runtime vs. Num of Tasks ---

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.lineplot(data=filtered_df, 
                y='Simulation Runtime [s]',
                x='Task Arrival Rate',
                hue='Num Sats',
                # col='Replanner',
                # row='Replanner',
                # kind="line",
                # palette="tab10", 
                # err_style="bars",
                markers=True, 
                dashes=False,
                ax=ax
            )
        
    plt.grid(True)

    # define filename for plot
    plot_filename = f'{trial_name}-{str("Simulation Runtime [s]").replace(" ", "_").replace("[s]","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # --- #N Messages vs. Num of Tasks ---
    # fig = plt.figure(layout='constrained', figsize=(10, 4))
    # subfigs = fig.subfigures(1, 2, wspace=0.07)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.lineplot(data=filtered_df, 
                y='Total Messages Broadcasted',
                # y='Average Messages Broadcasted per Task',
                x='Task Arrival Rate',
                hue='Num Sats',
                # col='Replanner',
                # row='Replanner',
                # kind="line",
                # palette="tab10", 
                # err_style="bars",
                markers=True, 
                dashes=False,
                ax=ax
            )
        
    plt.grid(True)

    # define filename for plot
    plot_filename = f'{trial_name}-{str("Total Messages Broadcasted").replace(" ", "_").replace("[s]","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # --- #N Messages / Task vs. Num of Tasks ---
    # fig = plt.figure(layout='constrained', figsize=(10, 4))
    # subfigs = fig.subfigures(1, 2, wspace=0.07)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.lineplot(data=filtered_df, 
                y='Average Messages Broadcasted per Task',
                x='Task Arrival Rate',
                hue='Num Sats',
                # col='Replanner',
                # row='Replanner',
                # kind="line",
                # palette="tab10", 
                # err_style="bars",
                markers=True, 
                dashes=False,
                ax=ax
            )
        
    plt.grid(True)

    # define filename for plot
    plot_filename = f'{trial_name}-{str("Average Messages Broadcasted per Task").replace(" ", "_").replace("[s]","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")

    # --- #N Messages / Task vs. Runtime ---
    # fig = plt.figure(layout='constrained', figsize=(10, 4))
    # subfigs = fig.subfigures(1, 2, wspace=0.07)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    sns.relplot(data=filtered_df, 
                y='Simulation Runtime [s]',
                x='Average Messages Broadcasted per Task',
                hue='Num Sats',
                # col='Replanner',
                # row='Replanner',
                # kind="line",
                # palette="tab10", 
                # err_style="bars",
                # markers=True, 
                # dashes=False,
                # ax=ax
            )
        
    plt.grid(True)

    # define filename for plot
    plot_filename = f'{trial_name}-{str("Replot").replace(" ", "_").replace("[s]","s")}.png'
    save_path = os.path.join(save_dir, plot_filename)
    local_save_path = os.path.join(local_save_dir, plot_filename)
    
    # save plot 
    plt.savefig(save_path)
    if base_dir != local_base_dir:
        plt.savefig(local_save_path)

    # print completion message with paths to saved plots
    print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_cbba_validation/2026_02_26_local"

    trial_name = "full_factorial_trials_2026-02-22"
    # trial_name = "full_factorial_trials_2026-02-23"

    experiment_col = "in_stress"
    # experiment_col = "in_connectivity"
    # experiment_col = "in_validation"

    # print runtime data
    print_runtime_data(trial_name, 
                       experiment_col,
                    #    base_dir
                    )

    print('DONE')
