
import os
import pandas as pd

def print_runtime_data(trial_name : str, scenario_id : int) -> None:
    """Load and print runtime data from experiment results."""

    # define results directory
    results_dir = f'{trial_name}_scenario_{scenario_id}'
    results_path = os.path.join('../results', results_dir)

    for dir_name in os.listdir(results_path):
        print(f"Directory: `{dir_name}`")

        dir_path = os.path.join(results_path, dir_name)
        for file_name in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, file_name)):
                continue

            if 'runtime' in file_name and 'parquet' in file_name:
                print(f" - File: `{file_name}`")
                # load and display data
                file_path = os.path.join(dir_path, file_name)
                data = pd.read_parquet(file_path)
                print(data.to_string(index=False))

        if 'env' in dir_name or 'manager' in dir_name:
            x= 1

if __name__ == "__main__":
    # define trial parameters
    trial_name = "lhs_trials-2_samples-1000_seed"
    scenario_id = 0

    # print runtime data
    print_runtime_data(trial_name, scenario_id)

    print('DONE')
