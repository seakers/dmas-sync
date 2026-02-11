import os

import pandas as pd


def main(trial_name : str, scenario_id : int) -> None:
    """ Generates and saves messages vs time plot from experiment results."""

    # define results directory
    results_dir = f'{trial_name}_scenario_{scenario_id}'
    results_path = os.path.join('experiments','1_0_cbba_stress_test','results', results_dir)

    cwd = os.getcwd()


    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results directory not found at: `{results_path}`")

    # load requests 
    requests_file = os.path.join(results_path, 'environment','requests.parquet')
    requests_df = pd.read_parquet(requests_file)

    # load broadcasts
    broadcasts_file = os.path.join(results_path, 'environment','broadcasts.parquet')
    broadcasts_df = pd.read_parquet(broadcasts_file)

    x = 1

if __name__ == "__main__":
    # define trial parameters
    trial_name = "lhs_trials-2_samples-1000_seed"
    scenario_id = 0

    # print runtime data
    main(trial_name, scenario_id)

    print('DONE')