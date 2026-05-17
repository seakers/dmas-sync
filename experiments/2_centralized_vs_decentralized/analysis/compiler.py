import os
from tokenize import group

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
                print(f"[results compiler] Loaded summary from `{results_summary_path}` with {len(summary_temp_df)} rows.")
            except Exception as e:
                print(f"[results compiler] error loading `{results_summary_path}`: {e}. Skipping.")
                continue

            # convert to dict 
            summary_as_dict = {}
            for _,row in summary_temp_df.iterrows():
                summary_as_dict[row['Metric']] = row['Value']

            summary_as_dict['Trial ID'] = int(dir_name.split('_trial-')[-1])
            
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

    # fill missing parameter values in trial definitions to None
    results_df["Preplanner"] = results_df["Prelanner"].fillna("None")
    results_df["Replanner"] = results_df["Replanner"].fillna("None")

    # merge summary data with trial definitions
    results_df : pd.DataFrame = summary_df.merge(
        trials_df,
        on="Trial ID",
        how="left",          # keep all results; attach params if found
        validate="many_to_one"  # each Scenario ID should map to one row in trials_df
    )
    
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

    # fill missing dual bounds with the values from the same scenario
    if 'Task Reward Dual Bound' in results_df.columns:
        # group by scenario and date
        for group_key,group_data in results_df.groupby(['Scenario', 'Date']):
            # unpack key
            group_scenario,group_date = group_key
            
            # define group mask
            group_mask = (results_df['Scenario'] == group_scenario) & (results_df['Date'] == group_date)

            # find if any value in the group has real dual and primal bouds
            has_real_dual_bound =  group_data['Task Reward Dual Bound'].notna()
            has_real_primal_bound = group_data['Total Planned Reward'].notna()
            real_values_slice = group_data[has_real_dual_bound & has_real_primal_bound]

            # check if data was found for this scenario and date
            if not real_values_slice.empty:
                # get the dual bound value from the first row of the slice (should be the same for all rows in the slice)
                dual_bound_value = real_values_slice['Task Reward Dual Bound'].iloc[0]
                primal_bound_value = real_values_slice['Total Planned Reward'].iloc[0]

                # fill missing dual bound values in the group with this value
                results_df.loc[group_mask, 'Task Reward Dual Bound'] = dual_bound_value

                # fill missing primal bound values in the group with this value
                results_df.loc[group_mask, 'Task Reward Primal Bound'] = primal_bound_value
            else:
                # no real data found for this scenario and date, fill with string `NaN`
                results_df.loc[group_mask, 'Task Reward Dual Bound'].fillna('NaN', inplace=True)
                results_df.loc[group_mask, 'Task Reward Primal Bound'].fillna('NaN', inplace=True)

            # define subgroups based on Data Processing and fill missing values within each subgroup
            for subgroup_key,subgroup_data in group_data.groupby('Data Processing'):
                # unpack key
                subgroup_processor = subgroup_key

                # define subgroup mask
                subgroup_mask = group_mask & (results_df['Data Processing'] == subgroup_processor)

                # find if any value in the subgroup has real dual and primal bouds
                has_real_dual_bound_subgroup =  subgroup_data['Known Task Reward Dual Bound'].notna()
                has_real_primal_bound_subgroup = subgroup_data['Known Task Reward Primal Bound'].notna()
                real_values_slice_subgroup = subgroup_data[has_real_dual_bound_subgroup & has_real_primal_bound_subgroup]

                # check if data was found for this subgroup
                if not real_values_slice_subgroup.empty:
                    # get the dual bound value from the first row of the slice (should be the same for all rows in the slice)
                    dual_bound_value_subgroup = real_values_slice_subgroup['Known Task Reward Dual Bound'].iloc[0]
                    primal_bound_value_subgroup = real_values_slice_subgroup['Known Task Reward Primal Bound'].iloc[0]

                    # fill missing dual bound values in the subgroup with this value
                    results_df.loc[subgroup_mask, 'Known Task Reward Dual Bound'] = dual_bound_value_subgroup

                    # fill missing primal bound values in the subgroup with this value
                    results_df.loc[subgroup_mask, 'Known Task Reward Primal Bound'] = primal_bound_value_subgroup
                else:
                    # no real data found for this subgroup, fill with string `NaN`
                    results_df.loc[subgroup_mask, 'Known Task Reward Dual Bound'].fillna('NaN', inplace=True)
                    results_df.loc[subgroup_mask, 'Known Task Reward Primal Bound'].fillna('NaN', inplace=True)

    # perform normalization of metrics if desired (e.g. normalize rewards by number of tasks)
    if 'Total Obtained Reward' in results_df.columns and 'Task Reward Dual Bound' in results_df.columns:
        results_df['Total Obtained Reward [norm]'] = results_df['Total Obtained Reward'] / results_df['Task Reward Dual Bound']
        results_df['Task Reward Primal Bound [norm]'] = results_df['Task Reward Primal Bound'] / results_df['Task Reward Dual Bound']
        # results_df['Total Obtained Reward [norm]'] = results_df['Total Obtained Reward'] / results_df['Total Observable Task Priority']
        
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
    base_dir = "/home/aslan15/Documents/GitHub/dmas-sync_bkp/results/merged/full_factorial_trials_2026-05-14_archive"

    # trial_name = "full_factorial_trials_2026-05-11"
    trial_name = "full_factorial_trials_2026-05-14"
    
    # compile and save compiled results summaries for this trial
    compile_results_summaries(trial_name, 
                            #   base_dir=base_dir
                              )

    # print completion message
    print('DONE')