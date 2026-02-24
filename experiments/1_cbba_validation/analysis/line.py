import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def generate_line_plots(trial_name : str, base_dir : str = None) -> None:
    """ Generates line plots for the compiled results of a given trial, showing the impact of number of satellites and arrival rate on key metrics. """

    # define base directory if not provided
    local_base_dir = os.path.join('experiments','1_0_cbba_stress_test','analysis')
    if base_dir is None: 
        base_dir = local_base_dir
        compiled_results_path = os.path.join(base_dir, 'compiled', f'{trial_name}_compiled_results.csv')
    else:
        compiled_results_path = os.path.join(base_dir, f'{trial_name}_compiled_results.csv')
        
    # ensure base dir exists

    
    # check if compiled results file exists
    if not os.path.exists(compiled_results_path):
        raise FileNotFoundError(f"Compiled results file not found at: `{compiled_results_path}`. Please run `compiler.py` to compile results summaries first.")
    
    # load compiled results
    results_df = pd.read_csv(compiled_results_path)

    filtered_df = results_df.fillna(-1)

    if filtered_df.empty or len(filtered_df) < 3:
        print(f"No full results found for trial `{trial_name}`.")
        return

    # --- generate line plots for key metrics ---
    # define metrics to plot
    y_metrics = [
                'Total Planned Reward [norm]',   
                'Total Planned Utility [norm]',      
                'P(Task Observed | Task Observable)',
                'Average Response Time to Task [s]',
                'Total Messages Broadcasted',
                'Average Messages Broadcasted per Task',
            ]
    # define x-axis metric
    x_metrics = [
        'Task Arrival Rate',
        'Num Sats',
    ]

    for y_metric in y_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for j, x_metric in enumerate(x_metrics):
            ax = axes[j]
            sns.lineplot(
                data=filtered_df,
                x=x_metric, y=y_metric,
                hue="Ground Segment",
                ax=ax
            )
            ax.set_title(f"{y_metric} vs {x_metric}")
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        # define save directory and filename for plot
        save_dir = os.path.join(base_dir, 'plots', 'lines')
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = f'{trial_name}-{y_metric.replace(" ", "_").replace("[norm]","norm")}.png'
        
        # save plot if desired
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path)

        # if saving to external directory, also save a copy to the local analysis directory 
        if base_dir != local_base_dir:
            local_save_dir = os.path.join(local_base_dir, 'plots', 'lines')
            os.makedirs(local_save_dir, exist_ok=True)
            local_save_path = os.path.join(local_save_dir, plot_filename)
            plt.savefig(local_save_path)

        # print completion message with paths to saved plots
        print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":  
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_0_cbba_stress_test/results/2026_02_18_tdist60_merged"
    trial_name = "full_factorial_trials"

    # generate bar plots for the specified trial and parameters
    generate_line_plots(trial_name, base_dir=base_dir)