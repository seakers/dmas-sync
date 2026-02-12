import os

import pandas as pd
from matplotlib import pyplot as plt

def generate_bar_plots(trial_name : str, n_sats : int, arrival_rate : int) -> None:
    """ Generates bar plots for the compiled results of a given trial, showing the impact of number of satellites and arrival rate on key metrics. """

    # load compiled results
    compiled_results_path = os.path.join('experiments','1_0_cbba_stress_test','analysis', 'compiled', f'{trial_name}_compiled_results.csv')
    if not os.path.exists(compiled_results_path):
        raise FileNotFoundError(f"Compiled results file not found at: `{compiled_results_path}`. Please run `compiler.py` to compile results summaries first.")
    
    results_df = pd.read_csv(compiled_results_path)

    # filter results for specified n_sats and arrival_rate
    filtered_df = results_df[(results_df['Num Sats'] == n_sats) & (results_df['Task Arrival Rate'] == arrival_rate)]
    filtered_df = filtered_df.fillna("None (In-Orbit Requester)")


    if filtered_df.empty or len(filtered_df) < 3:
        print(f"No full results found for n_sats={n_sats} and arrival_rate={arrival_rate} in trial `{trial_name}`.")
        return

    # --- generate bar plots for key metrics ---
    # define metrics to plot
    metrics_to_plot = ['Total Planned Reward', 
                    #    'Average Task Completion Time', 
                    #    'Max Task Completion Time'
                       ]
    # define x-axis metric
    x_metric = 'Ground Segment'

    for metric in metrics_to_plot:
        plt.figure(figsize=(8,6))
        plt.bar(filtered_df[x_metric], filtered_df[metric])
        plt.xlabel(x_metric)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_metric} (n_sats={n_sats}, arrival_rate={arrival_rate})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

        # save plot if desired
        save_dir = os.path.join('experiments','1_0_cbba_stress_test','analysis','plots', 'bars')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{trial_name}_n_sats_{n_sats}_arrival_rate_{arrival_rate}_{metric.replace(" ", "_")}.png')
        plt.savefig(save_path)
        print(f"Saved plot to: `{save_path}`")


if __name__ == "__main__":  
    # define trial name and parameters to filter results by
    trial_name = "full_factorial_trials"

    # generate bar plots for the specified trial and parameters
    for n_sats in [24, 48, 96]:
        for arrival_rate in [10, 100, 500, 1000]:
            generate_bar_plots(trial_name, n_sats, arrival_rate)