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
    print_scenario_banner('Trial generator for Centralized vs Decentralized Planning Study')
       

    # define experiment parameters
    testing_params = {
        "Preplanner" : [
            "None",
            "DP",
            "Centralized-MILP_priority",
        ],
        "Replanner": [
            "None",
            "Greedy", 
            "CBBA", 
            # "CBBA (Augmented)",
        ],
        "Connectivity": [
            "GS",                   # sats can only talk to ground station (no inter-sat comms)
        ],
        "Mission": [
            "Urgency",
            "Revisits",
            "Co-observations"
        ],
        "Data Processing" : [
            "Instant",              # the ground is able to perfectly identify which tasks are active at each time step, and can communicate this to the satellites (i.e. perfect event detection and classification)
        ],
        "Date" : [       
            "2019-02-15", 
        ], 
    }

    abridged_params = {
        "Preplanner" : [
            "None",
            "DP",
            "Centralized-MILP_priority",
        ],
        "Replanner": [
            "None",
            "Greedy", 
            "CBBA", 
            # "CBBA (Augmented)", # TODO
        ],
        "Connectivity": [
            "GS",                   # sats can only talk to ground station (no inter-sat comms)
            "Intraconstellation",   # sats can talk to each other within the same constellation and to ground stations, but not across constellations
            "Interconstellation",   # sats can talk to each other across constellations and to ground stations using multi-hop ISL messaging or TDRSS relays
        ],
        "Mission": [
            "Urgency",
            "Revisits",
            "Co-observations"
        ],
        "Data Processing" : [
            "Instant",              # the ground is able to perfectly identify which tasks are active at each time step, and can communicate this to the satellites (i.e. perfect event detection and classification)
            "Ground",               # information is processed on the ground, so replanning can only occur after a full round of data collection and downlink (i.e. replanning occurs at a much slower cadence than onboard processing)
            "Onboard",              # sats must discover events using default mission tasks
        ],
        "Date" : [            
            # 2019 dates 
            "2019-02-15",   # Winter NH
        ], 
    }    
    year_2019_params = {
        "Preplanner" : [
            "None",
            "DP",
            "Centralized-MILP_priority",
        ],
        "Replanner": [
            "None",
            "Greedy", 
            "CBBA", 
            # "CBBA (Augmented)",
        ],
        "Connectivity": [
            "GS",                   # sats can only talk to ground station (no inter-sat comms)
            "Intraconstellation",   # sats can talk to each other within the same constellation and to ground stations, but not across constellations
            "Interconstellation",   # sats can talk to each other across constellations and to ground stations using multi-hop ISL messaging or TDRSS relays
        ],
        "Mission": [
            "Urgency",
            "Revisits",
            "Co-observations"
        ],
        "Data Processing" : [
            "Instant",               # the ground is able to perfectly identify which tasks are active at each time step, and can communicate this to the satellites (i.e. perfect event detection and classification)
            "Ground",               # information is processed on the ground, so replanning can only occur after a full round of data collection and downlink (i.e. replanning occurs at a much slower cadence than onboard processing)
            "Onboard",              # sats must discover events using default mission tasks
        ],
        "Date" : [            
            # 2019 dates 
            "2019-02-15",   # Winter NH
            "2019-05-15",   # Spring NH
            "2019-08-10",   # Summer NH / peak fire season
            "2019-11-10"    # Fall NH
        ], 
    }    
    full_params = {
        "Preplanner" : [
            "None",
            "DP",
            "Centralized-MILP_priority",
        ],
        "Replanner": [
            "None",
            "Greedy", 
            "CBBA", 
            # "CBBA (Augmented)", 
        ],
        "Connectivity": [
            "GS",                   # sats can only talk to ground station (no inter-sat comms)
            "Intraconstellation",   # sats can talk to each other within the same constellation and to ground stations, but not across constellations
            "Interconstellation",   # sats can talk to each other across constellations and to ground stations using multi-hop ISL messaging or TDRSS relays
        ],
        "Mission": [
            "Urgency",
            "Revisits",
            "Co-observations"
        ],
        "Data Processing" : [
            "Instant",               # the ground is able to perfectly identify which tasks are active at each time step, and can communicate this to the satellites (i.e. perfect event detection and classification)
            "Ground",               # information is processed on the ground, so replanning can only occur after a full round of data collection and downlink (i.e. replanning occurs at a much slower cadence than onboard processing)
            "Onboard",              # sats must discover events using default mission tasks
        ],
        "Date" : [
            # 2018 dates 
            # TODO include if cases from 2019 show seasonal trends that we want to compare against (e.g. fire season)
            "2018-02-15",   # Winter NH
            "2018-05-15",   # Spring NH
            "2018-08-10",   # Summer NH / peak fire season
            "2018-11-10",   # Fall NH
            
            # 2019 dates 
            "2019-02-15",   # Winter NH
            "2019-05-15",   # Spring NH
            "2019-08-10",   # Summer NH / peak fire season
            "2019-11-10"    # Fall NH
        ], 
    }    

    # define experiment parameter rules
    rules = [
        # centralized preplanners must have `none` replanenrs (i.e. replanning is only relevant for decentralized strategies)
        lambda d: (d["Preplanner"] != "Centralized-MILP_priority") | (d["Replanner"] == "None"),
        lambda d: (d["Preplanner"] != "Centralized-MILP_assignment") | (d["Replanner"] == "None"),
        # None preplanner cannot have a none replanner (i.e. if no initial plan, must have some kind of replanning strategy)
        # lambda d: (d["Preplanner"] != "None") | (d["Replanner"] != "None"),
        # there can only be one None x None trial per date
        lambda d: ~((d["Preplanner"] == "None") & (d["Replanner"] == "None")) | (~d.duplicated(subset=["Date", "Preplanner", "Replanner"])),
    ]

    # 1) generate full enumeration of trials per experiment
    testing_trials = generate_full_tactorial_trials(testing_params)
    abridged_trials = generate_full_tactorial_trials(abridged_params)
    year_2019_trials = generate_full_tactorial_trials(year_2019_params)
    full_trials = generate_full_tactorial_trials(full_params)

    trial_dfs = {
        "testing" : testing_trials,
        "abridged": abridged_trials,
        "year_2019": year_2019_trials,
        # "full": full_trials
    }

    # merge trials, tagging which experiment(s) they belong to
    param_cols = list(abridged_trials.keys())  
    all_trials = tag_and_merge(trial_dfs, param_cols)

    # apply rules to filter out invalid scenarios
    all_trials = apply_rules(all_trials, rules) 

    # 2) sort trials by experiment membership and then by parameters
    #    (abridged first, then full)
    phase = np.select(
        [
            all_trials["in_testing"],
            all_trials["in_abridged"],
            all_trials["in_year_2019"],
            # all_trials["in_full"],
        ],
        [0, 1, 2],
        default=3,   # should be rare / indicates "belongs to none"
    )
    all_trials["Phase"] = phase
    all_trials["_none_none"] = ((all_trials["Preplanner"] == "None") & (all_trials["Replanner"] == "None")).astype(int)
    dp_order = {"Ground": 0, "Onboard": 1, "Instant": 2}
    all_trials["_dp_sort"] = all_trials["Data Processing"].map(dp_order)
    sort_cols = ["Phase", "_none_none"] + [("_dp_sort" if c == "Data Processing" else c) for c in param_cols]
    all_trials = all_trials.sort_values(by=sort_cols).drop(columns=["Phase", "_none_none", "_dp_sort"]).reset_index(drop=True)

    # 3) Trial IDs after ordering
    all_trials.insert(0, "Trial ID", all_trials.index)
    print(f" - Total number of trials: {len(all_trials)}")

    # 4) Add a column that flags whether to calculate the dual for the trial based on scenario, constellation, and date
    # initialize `calcBoundsOpt` column to `0`
    all_trials["calcBoundsOpt"] = 0
    
    # get combinations of scenario, constellation, and date
    combos = all_trials[["Mission", "Date"]].drop_duplicates()
    
    for _,combo_row in combos.iterrows():
        # create mask for trials matching the combo
        mask = ((all_trials["Mission"] == combo_row["Mission"]) &
                (all_trials["Date"] == combo_row["Date"]) 
                # & (all_trials["in_testing"] == combo_row["in_testing"]) &
                # (all_trials["in_abridged"] == combo_row["in_abridged"]) &
                # (all_trials["in_year_2019"] == combo_row["in_year_2019"]) 
                # & (all_trials["in_full"] == combo_row["in_full"])
                )
        
        # get slice of trials matching the combo
        combo_trials = all_trials.loc[mask]
        
        # set `calcBoundsOpt` to non-zero only for the highest trial ids among the slice 
        if not combo_trials.empty:
            max_dp_trial_id = combo_trials["Trial ID"].max()
            all_trials.loc[all_trials["Trial ID"] == max_dp_trial_id, "calcBoundsOpt"] = 2
            
            # # get the max trial id for each data processing type in the combo
            # n_data_processing_types = len(combo_trials["Data Processing"].unique())

            # # set `calcBoundsOpt` to 1 for the max trial id of each data processing type in the combo
            # # and to 2 if the data processing type is "Oracle"
            # for dp_type in combo_trials["Data Processing"].unique():
            #     dp_trials = combo_trials[combo_trials["Data Processing"] == dp_type]
            #     if not dp_trials.empty:
            #         max_dp_trial_id = dp_trials["Trial ID"].max()
            #         all_trials.loc[all_trials["Trial ID"] == max_dp_trial_id, "calcBoundsOpt"] = 2
            #         if dp_type == "Oracle" or dp_type == "Instant":  # treat "Instant" the same as "Oracle" for this purpose since it represents perfect information, just with a different flavor of realism
            #         else:
            #             all_trials.loc[all_trials["Trial ID"] == max_dp_trial_id, "calcBoundsOpt"] = 1

    # generate results directory
    out_dir = os.path.join('.', 'experiments','2_centralized_vs_decentralized', 'resources', 'trials')
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