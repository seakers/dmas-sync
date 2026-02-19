import os

import pandas as pd
from matplotlib import pyplot as plt

def generate_bar_plots(trial_name : str, n_sats : int, arrival_rate : int, base_dir : str = None) -> None:
    """ Generates bar plots for the compiled results of a given trial, showing the impact of number of satellites and arrival rate on key metrics. """

    # define base directory if not provided
    local_base_dir = os.path.join('experiments','1_0_cbba_stress_test','analysis')
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


    # filter results for specified n_sats and arrival_rate
    filtered_df = results_df[(results_df['Num Sats'] == n_sats) & (results_df['Task Arrival Rate'] == arrival_rate)]
    filtered_df = filtered_df.fillna(-1)

    if filtered_df.empty or len(filtered_df) < 3:
        print(f"No full results found for n_sats={n_sats} and arrival_rate={arrival_rate} in trial `{trial_name}`.")
        return

    # --- generate bar plots for key metrics ---
    # define metrics to plot
    metrics_to_plot = [
                        'Total Planned Reward', 
                        'Total Planned Reward [norm]',   
                        'Total Planned Utility [norm]',      
                        'P(Task Observed | Task Observable)',
                        'Average Response Time to Task [s]' 
                    #    'Average Task Completion Time', 
                    #    'Max Task Completion Time'
                       ]
    # define x-axis metric
    x_metric = 'Ground Segment'

    ground_segment_order = (
        results_df[x_metric]
        .fillna("None (In-Orbit Requester)")
        .drop_duplicates()
        .tolist()
    )
    ground_segment_order.remove("None (In-Orbit Requester)")
    ground_segment_order.append("None (In-Orbit Requester)")

    filtered_df[x_metric] = pd.Categorical(
        filtered_df[x_metric],
        categories=ground_segment_order,
        ordered=True
    )

    filtered_df = filtered_df.sort_values(x_metric)

    for metric in metrics_to_plot:
        plt.figure(figsize=(8,6))
        plt.bar(filtered_df[x_metric], filtered_df[metric])
        plt.xlabel(x_metric)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_metric} (n_sats={n_sats}, arrival_rate={arrival_rate})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

        # define save directory and filename for plot
        save_dir = os.path.join(base_dir, 'plots', 'bars')
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = f'{trial_name}_n_sats_{n_sats}_arrival_rate_{arrival_rate}_{metric.replace(" ", "_")}.png'
        
        # save plot if desired
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path)

        # if saving to external directory, also save a copy to the local analysis directory 
        if base_dir != local_base_dir:
            local_save_path = os.path.join(local_base_dir, 'plots', 'bars', plot_filename)
            plt.savefig(local_save_path)

        # print completion message with paths to saved plots
        print(f"Saved plot to: `{save_path}` and `{local_save_path}`")


if __name__ == "__main__":  
    # define trial name and parameters to filter results by
    base_dir = "/media/aslan15/easystore/Data/1_0_cbba_stress_test/results/2026_02_18_tdist60_merged"
    trial_name = "full_factorial_trials"

    # generate bar plots for the specified trial and parameters
    for n_sats in [24, 48, 96]:
        for arrival_rate in [10, 100, 500, 1000]:
            generate_bar_plots(trial_name, n_sats, arrival_rate, base_dir=base_dir)