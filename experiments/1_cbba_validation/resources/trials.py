from datetime import datetime
import itertools
import math
import os
from typing import Callable, Dict, List
import pandas as pd
import numpy as np

from dmas.utils.tools import print_scenario_banner

def generate_full_tactorial_trials(params : Dict[str, list]) -> pd.DataFrame:
    # Generate full factorial combinations
    keys = list(params.keys())
    values = list(params.values())

    combinations = list(itertools.product(*values))

    # Create DataFrame
    return pd.DataFrame(combinations, columns=keys)

def tag_and_merge(trial_dfs: Dict[str, pd.DataFrame], param_cols: List[str]) -> pd.DataFrame:
    # Add membership flags
    tagged = []
    for name, df in trial_dfs.items():
        t = df.copy()
        t[f"in_{name}"] = True
        tagged.append(t)

    # concatenate all tagged DataFrames
    all_tagged = pd.concat(tagged, ignore_index=True)

    # For rows that are duplicated across experiments, OR the flags together
    flag_cols = [c for c in all_tagged.columns if c.startswith("in_")]
    merged = (
        all_tagged
        .groupby(param_cols, as_index=False)[flag_cols]
        .max()   # True if present in any of the duplicates
    )

    # Set missing flags to be `False`
    merged[flag_cols] = merged[flag_cols].fillna(False).astype(bool)

    # return merged dataframe
    return merged

def apply_rules(df: pd.DataFrame, rules: List[Callable[[pd.DataFrame], pd.Series]]) -> pd.DataFrame:
    if not rules:
        return df
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        m = rule(df)
        if not isinstance(m, pd.Series) or m.dtype != bool:
            raise ValueError("Each rule must return a boolean pd.Series.")
        mask &= m
    return df.loc[mask].reset_index(drop=True)

if __name__ == "__main__":
    # print welcome
    print_scenario_banner('Experiment generator for Preplanner Parametric Study')

    # define the number of scenarios to generate per combination of parameters
    n_scenarios = 3

    # define number of trials per batch job (for better distribution across experiment)
    batch_size = 20

    # define experiment parameters
    stress_test_params = {
        "Preplanner" : ["None"],
        "Replanner": [
                        # "Greedy", 
                        "CBBA", 
                        # "Oracle"
                    ],
        "Num Sats": [12, 24, 48, 96, 192],
        "Latency": ["Low"],
        "Task Arrival Rate": [10, 50, 100, 500, 1000],
        "Target Distribution": [60.0],
        "Scenario" : range(n_scenarios), 
    }
    connectivity_test_params = {
        "Preplanner" : ["None"],
        "Replanner": [  
                        # "Greedy", 
                        "CBBA", 
                        # "Oracle"
                    ],
        "Num Sats": [12, 24, 48, 96, 192],
        "Latency": ["High", "Medium", "Low"],
        "Task Arrival Rate": [10, 50, 100, 500, 1000],
        "Target Distribution": [60.0],
        "Scenario" : range(n_scenarios),
    }
    validation_test_params = {
        "Preplanner" : [
                        "None", 
                        "DP"
                    ],
        "Replanner": [
                        "None", 
                        "Greedy",
                        "CBBA",
                    ],
        "Num Sats": [12, 24, 48, 96, 192],
        "Latency": ["High", "Medium", "Low"],
        "Task Arrival Rate": [10, 50, 100, 500, 1000],
        "Target Distribution": [60.0],
        "Scenario" : range(n_scenarios),
    }

    # define experiment parameter rules
    rules = [
        # Oracle replanner only allowed with no preplanner
        lambda d: (d["Replanner"] != "Oracle") | (d["Preplanner"] == "None"),
        # None replanne not allowed with `None` preplanner (i.e. if no preplanner, must have a replanner)
        lambda d: (d["Replanner"] != "None") | (d["Preplanner"] != "None") 
    ]

    # generate full enumeration of trials per experiment
    stress_test_trials = generate_full_tactorial_trials(stress_test_params)
    connectivity_test_trials = generate_full_tactorial_trials(connectivity_test_params)
    validation_test_trials = generate_full_tactorial_trials(validation_test_params)

    trial_dfs = {
        "stress": stress_test_trials,
        "connectivity": connectivity_test_trials,
        "validation": validation_test_trials,
    }

    # merge trials, tagging which experiment(s) they belong to
    param_cols = list(stress_test_params.keys())  
    all_trials = tag_and_merge(trial_dfs, param_cols)

    # apply rules to filter out invalid scenarios
    all_trials = apply_rules(all_trials, rules)

    # sort trials by experiment membership and then by parameters
    # 1) Assign each trial to the earliest experiment it belongs to
    #    (stress first, then connectivity, then validation)
    phase = np.select(
        [
            all_trials["in_stress"],
            all_trials["in_connectivity"],
            all_trials["in_validation"],
        ],
        [0, 1, 2],
        default=3,   # should be rare / indicates "belongs to none"
    )
    # proxy for scenario complexity
    degree = all_trials['Num Sats'] * all_trials['Task Arrival Rate']  

    all_trials = all_trials.copy()
    all_trials["phase"] = phase
    all_trials["degree"] = degree

    # Fit-derived coefficients from observed runtime data
    # Model: log(runtime) = intercept + coefs * features
    INTERCEPT = -5.8645
    COEF_LOG_NUM_SATS  =  1.3212
    COEF_LOG_TASK_RATE =  0.7461
    COEF_CBBA          =  0.8475
    COEF_GREEDY        =  0.1327
    COEF_LATENCY_HIGH  =  0.4689
    COEF_LATENCY_MED   =  0.3878
    # (Preplanner DP ~= 0, skip it)

    def estimate_runtime(row):
        log_rt = (
            INTERCEPT
            + COEF_LOG_NUM_SATS  * np.log(row['Num Sats'])
            + COEF_LOG_TASK_RATE * np.log(row['Task Arrival Rate'])
            + COEF_CBBA          * (row['Replanner'] == 'CBBA')
            + COEF_GREEDY        * (row['Replanner'] == 'Greedy')
            + COEF_LATENCY_HIGH  * (row['Latency'] == 'High')
            + COEF_LATENCY_MED   * (row['Latency'] == 'Medium')
        )
        return np.exp(log_rt)

    # 2) Sort by phase, then small-to-large sims
    all_trials['est_runtime'] = all_trials.apply(estimate_runtime, axis=1)

    # Sort by estimated runtime, then stripe across tasks
    n = len(all_trials)
    num_tasks = int(np.ceil(n / batch_size))

    all_trials = all_trials.sort_values('est_runtime', ascending=True).reset_index(drop=True)
    all_trials['task_id'] = all_trials.index % num_tasks
    all_trials['within_task_rank'] = all_trials.index // num_tasks

    all_trials = (
        all_trials
        .sort_values(['task_id', 'within_task_rank'])
        .reset_index(drop=True)
        .drop(columns=['est_runtime', 'within_task_rank', 'phase', 'degree'])  # drop auxiliary columns used for sorting
    )

    # all_trials = (
    #     all_trials
    #     .sort_values(
    #         # by=["phase", "Num Sats", "Task Arrival Rate"],
    #         # by=["Scenario", "Num Sats", "Latency"],
    #         by=['degree', 'Scenario'],
    #         ascending=[True, True],
    #         kind="mergesort",   # stable / reproducible
    #     )
    #     .reset_index(drop=True)
    #     .drop(columns=["phase", "degree"])
    # )
    
    # # Assign stripe and rank
    # all_trials['batch_id'] = all_trials.index % batch_size
    # all_trials['within_batch_rank'] = all_trials.index // batch_size

    # # Diagnostic BEFORE resorting
    # print(all_trials.groupby('batch_id')['degree'].agg(['sum', 'mean', 'max']))

    # n = len(all_trials)
    # num_tasks = math.ceil(n / batch_size)  # 113 tasks

    # # Assign each trial to a task such that task 0 gets ranks 0, 113, 226...
    # # i.e. the first of each complexity decile, task 1 gets ranks 1, 114, 227...
    # all_trials['task_id'] = all_trials.index % num_tasks
    # all_trials['within_task_rank'] = all_trials.index // num_tasks

    # all_trials = (
    #     all_trials
    #     .sort_values(['task_id', 'within_task_rank'])
    #     .reset_index(drop=True)
    # )

    # # Diagnostic AFTER resorting
    # all_trials['batch_id'] = all_trials.index // batch_size  # after final sort
    # print(all_trials.groupby('batch_id')['degree'].agg(['sum', 'mean', 'max']))

    # 3) Trial IDs after ordering
    all_trials.insert(0, "Trial ID", all_trials.index)
    print(f" - Total number of trials: {len(all_trials)}")

    # generate results directory
    out_dir = os.path.join('.', 'experiments','1_cbba_validation', 'resources', 'trials')
    os.makedirs(out_dir, exist_ok=True)

    # name the output file with the current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"full_factorial_trials_{date_str}.csv"

    # save full factorial trials to csv
    full_trials_path = os.path.join(out_dir, filename)
    all_trials.to_csv(full_trials_path, index=False)
    print(f" - Full factorial trials saved to:\n   `{full_trials_path}`")

    # print completion message
    print("\nExperiment generation complete!")