import os

import pandas as pd


def compile_results_summaries(trial_name : str, 
                              base_dir : str = None) -> None:
    """ Loads all `summary.csv` files from the results directory, compiles them into a single DataFrame, and saves it as `compiled_results.csv`. """

    # define base directory if not provided
    if base_dir is None:
        base_dir = os.path.join('experiments','2_centralized_vs_decentralized','results')

    # initialize data list to hold summary dicts
    data = []

    # initialize counters 
    n_results_files = 0
    n_files_loaded = 0

    # iterate through subdirectories in base_dir to find summary.csv files
    for dir_name in os.listdir(base_dir):
        if (os.path.isdir(os.path.join(base_dir, dir_name)) 
                and dir_name.startswith(trial_name)):
            
            if "reduced" in dir_name or "bkp" in dir_name: 
                # skip reduced trials for now since they are not part of the main analysis
                continue  
            
            # define path to summary.csv
            results_summary_path = os.path.join(base_dir, dir_name, 'summary.csv')
            n_results_files += 1

            # check if summary.csv exists
            if not os.path.exists(results_summary_path):
                print(f"[results compiler] summary file not found at: `{results_summary_path}`. Skipping.")
                continue

            # load summary.csv
            try:
                summary_temp_df = pd.read_csv(results_summary_path)
                # print(f"[results compiler] Loaded summary from `{results_summary_path}` with {len(summary_temp_df)} rows.")
            except Exception as e:
                print(f"[results compiler] error loading `{results_summary_path}`: {e}. Skipping.")
                continue

            # convert to dict 
            summary_as_dict = {}
            for _,row in summary_temp_df.iterrows():
                summary_as_dict[row['Metric']] = row['Value']

            trial_id = dir_name.split('_trial-')[-1]
            if "reduced" in trial_id:
                continue # skip reduced trials; they are used for testing and debugging but not part of the main analysis
            summary_as_dict['Trial ID'] = int(trial_id)
            
            # add to data list
            data.append(summary_as_dict)
            n_files_loaded += 1
    
    if not data:
        raise ValueError(f"No valid summary data found for trial `{trial_name}` in `{base_dir}`.")
    
    # compile summary data into DataFrame
    summary_df = pd.DataFrame(data)

    # load trial definition data
    trial_definitions_path = os.path.join('experiments','2_centralized_vs_decentralized','resources', 'trials', f'{trial_name}.csv')
    trials_df = pd.read_csv(trial_definitions_path)

    # merge summary data with trial definitions
    results_df : pd.DataFrame = summary_df.merge(
        trials_df,
        on="Trial ID",
        how="left",          # keep all results; attach params if found
        validate="many_to_one"  # each Scenario ID should map to one row in trials_df
    )

    # fill missing parameter values in trial definitions to None
    results_df["Preplanner"] = results_df["Preplanner"].fillna("None")
    results_df["Replanner"] = results_df["Replanner"].fillna("None")
    
    # # fill in missing values for Ground Segment with "None (In-Orbit Requester)" to indicate no ground segment 
    # results_df["Ground Segment"] = results_df["Ground Segment"].fillna("None (In-Orbit Requester)")
    
    # fill missing probabilities with -1 to indicate not applicable / no data
    for col in results_df.columns:
        if "P(" in col:
            results_df[col] = results_df[col].fillna(-1)  

    # reorder columns for clarity
    id_col = "Trial ID"
    param_cols = [c for c in trials_df.columns if c != id_col]
    result_cols = [c for c in summary_df.columns if c != id_col]

    # desired column order: Params >> Results
    results_df = results_df[[id_col] + param_cols + result_cols]

    # sort by Trial ID for easier comparison across trials
    results_df = results_df.sort_values(by=id_col).reset_index(drop=True)

    # fill Task Reward Dual Bound: propagate first non-NaN within (Scenario, Date)
    if 'Task Reward Dual Bound' in results_df.columns:
        for (group_scenario, group_date), group_data in results_df.groupby(['Scenario', 'Date']):
            group_mask = (results_df['Scenario'] == group_scenario) & (results_df['Date'] == group_date)
            non_nan = group_data['Task Reward Dual Bound'].dropna()
            if not non_nan.empty:
                results_df.loc[group_mask, 'Task Reward Dual Bound'] = non_nan.iloc[0]
            else:
                results_df.loc[group_mask, 'Task Reward Dual Bound'] = results_df.loc[group_mask, 'Task Reward Dual Bound'].fillna('NaN')

    # fill Known Task Reward Dual Bound, Known Task Reward Primal Bound, and Task Reward Primal Bound
    # grouped by (Connectivity, Data Processing, Date)
    subgroup_cols = ['Connectivity', 'Data Processing', 'Date']
    for (group_connectivity, group_dp, group_date), group_data in results_df.groupby(subgroup_cols):
        group_mask = (
            (results_df['Connectivity'] == group_connectivity) &
            (results_df['Data Processing'] == group_dp) &
            (results_df['Date'] == group_date)
        )

        # Known Task Reward Dual Bound: propagate first non-NaN in group
        if 'Known Task Reward Dual Bound' in results_df.columns:
            non_nan = group_data['Known Task Reward Dual Bound'].dropna()
            if not non_nan.empty:
                results_df.loc[group_mask, 'Known Task Reward Dual Bound'] = non_nan.iloc[0]
            else:
                results_df.loc[group_mask, 'Known Task Reward Dual Bound'] = results_df.loc[group_mask, 'Known Task Reward Dual Bound'].fillna('NaN')

        # Known Task Reward Primal Bound and Task Reward Primal Bound:
        # use Total Obtained Utility from the Preplanner=None x Replanner=None row in this group
        none_none_rows = group_data[
            (group_data['Preplanner'] == 'None') & (group_data['Replanner'] == 'None')
        ]['Total Obtained Utility'].dropna()

        if not none_none_rows.empty:
            primal_value = none_none_rows.iloc[0]
            if 'Known Task Reward Primal Bound' in results_df.columns:
                results_df.loc[group_mask, 'Known Task Reward Primal Bound'] = primal_value
            if 'Task Reward Primal Bound' in results_df.columns:
                results_df.loc[group_mask, 'Task Reward Primal Bound'] = primal_value
        else:
            if 'Known Task Reward Primal Bound' in results_df.columns:
                results_df.loc[group_mask, 'Known Task Reward Primal Bound'] = results_df.loc[group_mask, 'Known Task Reward Primal Bound'].fillna('NaN')
            if 'Task Reward Primal Bound' in results_df.columns:
                results_df.loc[group_mask, 'Task Reward Primal Bound'] = results_df.loc[group_mask, 'Task Reward Primal Bound'].fillna('NaN')

    # perform normalization of metrics if desired (e.g. normalize rewards by number of tasks)
    if 'Total Obtained Reward' in results_df.columns and 'Task Reward Dual Bound' in results_df.columns:
        results_df['Total Obtained Reward [norm]'] = results_df['Total Obtained Reward'] / results_df['Task Reward Dual Bound']
        results_df['Task Reward Primal Bound [norm]'] = results_df['Task Reward Primal Bound'] / results_df['Task Reward Dual Bound']
        # results_df['Total Obtained Reward [norm]'] = results_df['Total Obtained Reward'] / results_df['Total Observable Task Priority']
    
    # if 'Total Obtained Reward' in results_df.columns and 'Known Task Reward Dual Bound' in results_df.columns:
        # results_df['Total Obtained Reward [known_norm]'] = results_df['Total Obtained Reward'] / results_df['Known Task Reward Dual Bound']
        # results_df['Known Task Reward Primal Bound [known_norm]'] = results_df['Known Task Reward Primal Bound'] / results_df['Known Task Reward Dual Bound']
        
    if 'Total Obtained Utility' in results_df.columns and 'Task Reward Dual Bound' in results_df.columns:
        results_df['Total Obtained Utility [norm]'] = results_df['Total Obtained Utility'] / results_df['Task Reward Dual Bound']
        # results_df['Total Obtained Utility [norm]'] = results_df['Total Obtained Utility'] / results_df['Total Observable Task Priority']

    if 'Total Planned Reward' in results_df.columns and 'Task Reward Dual Bound' in results_df.columns:
        results_df['Total Planned Reward [norm]'] = results_df['Total Planned Reward'] / results_df['Task Reward Dual Bound']
        # results_df['Total Planned Reward [norm]'] = results_df['Total Planned Reward'] / results_df['Total Observable Task Priority']

    if 'Total Planned Utility' in results_df.columns and 'Task Reward Dual Bound' in results_df.columns:
        results_df['Total Planned Utility [norm]'] = results_df['Total Planned Utility'] / results_df['Task Reward Dual Bound']
        # results_df['Total Planned Utility [norm]'] = results_df['Total Planned Utility'] / results_df['Total Observable Task Priority']

    if 'Total Messages Broadcasted' in results_df.columns and 'Tasks Available' in results_df.columns:
        results_df['Average Messages Broadcasted per Task'] = results_df['Total Messages Broadcasted'] / results_df['Tasks Available']


    # define output paths for compiled results
    compiled_results_filename = f'{trial_name}_compiled_results.csv'
    compiled_results_path = os.path.join(base_dir, compiled_results_filename)
    local_results_dir = os.path.join('experiments','2_centralized_vs_decentralized','analysis', 'compiled')
    os.makedirs(local_results_dir, exist_ok=True)  # ensure local analysis directory exists
    local_results_path = os.path.join(local_results_dir, compiled_results_filename)
    
    # save compiled results to both `base_dir` and local analysis directory for easy access
    results_df.to_csv(compiled_results_path, index=False)
    results_df.to_csv(local_results_path, index=False)
    
    # print completion message with paths to compiled results
    print(f"Compiled results summary for trial `{trial_name}`:")
    print(f" - Number of summary files successfully loaded: {n_files_loaded} / {n_results_files} ({(n_files_loaded/n_results_files)*100:.1f}%)")
    if compiled_results_path != local_results_path:
        print(f" - Compiled results saved to `{compiled_results_path}` and `{local_results_path}`")
    else:
        print(f" - Compiled results saved to `{compiled_results_path}`")

if __name__ == "__main__":
    # define trial parameters
    base_dir = "/home/aslan15/Documents/GitHub/dmas-sync_bkp/results/merged/full_factorial_trials_2026-05-17_archive_v5"

    # trial_name = "full_factorial_trials_2026-05-11"
    trial_name = "full_factorial_trials_2026-05-17"
    
    # compile and save compiled results summaries for this trial
    compile_results_summaries(trial_name, 
                            #   base_dir=base_dir
                              )

    # print completion message
    print('DONE')