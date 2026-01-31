import itertools
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats._qmc import LatinHypercube
from tqdm import tqdm

from chess3d.utils import print_banner

def main(params : List[Tuple[str, list]], lhs_samples : int = 1, seed : int = 1000):
    # generate results directory
    out_dir = './trials'
    os.makedirs(out_dir, exist_ok=True)

    # ==============================
    # Full Factorial Enumeration
    # ==============================    
    print("\nGenerating full factorial trials...")
    
    # count total number of trials
    n_total = np.product([len(p[1]) for p in params])
    print(f" - Total possible trials: {n_total}")
    
    # generate all combinations
    columns = [p[0] for p in params]
    columns.insert(0,'Scenario ID')
    
    combinations = list(itertools.product(*[p[1] for p in params]))
    combinations = [[i]+list(comb) for i,comb in enumerate(combinations)]

    # create dataframe
    full_trials_df = pd.DataFrame(combinations, columns=columns)

    # save to csv
    full_trials_path = os.path.join(out_dir, "full_factorial_trials.csv")
    full_trials_df.to_csv(full_trials_path, index=False)
    print(f" - Full factorial trials saved to:\n   `{full_trials_path}`")

    # ==============================
    # Latin Hypercube Sampling
    # ============================== 
    print("\nGenerating Latin Hypercube trials...")

    # calculate lowest-common-multiple for estimating number of samples
    lcm = np.lcm.reduce([len(vals) for _,vals in params])

    # calculate number of samples for lhs
    n_samples = lhs_samples*lcm
    print(f" - Largest common multiple of parameter lengths: {lcm}")
    print(f" - Number of Latin Hypercube samples: {n_samples}")

    # generate hypercube sampler
    sampler : LatinHypercube = LatinHypercube(d=len(params),seed=seed)
    
    # sample latin hypercube
    samples = sampler.integers(l_bounds=[0 for _ in params], 
                            u_bounds=[len(vals) for _,vals in params], 
                            n=n_samples)

    # interpret samples and generate experiments
    columns = [param for param,_ in params]
    columns.insert(0,'Scenario ID')
    data = []
    for sample in tqdm(samples, desc='Generating experiments', leave=False):        
        # create row of values 
        row = []
        row_query = {}

        for param,val_idx in zip(params, sample):
            param_name,vals = param
            value = vals[val_idx]
            row.append(value)
            row_query[param_name] = value

        # find scenario id  
        mask = (full_trials_df[list(row_query)] == pd.Series(row_query)).all(axis=1)
        scenario_id = max(full_trials_df[mask]['Scenario ID'].values)
        row.insert(0, scenario_id)

        # add to list of experiments
        data.append(row)

    # create data frame
    lhs_trials_df = pd.DataFrame(data=data, columns=columns)
    lhs_trials_df.sort_values(by='Scenario ID', inplace=True)

    # save to csv
    lhs_trials_path = os.path.join(out_dir, f"lhs_trials-{lhs_samples}_samples-{seed}_seed.csv")
    lhs_trials_df.to_csv(lhs_trials_path, index=False)
    print(f" - Latin Hypercube trials saved to:\n   `{lhs_trials_path}`")

    # ==============================
    # Test Trials
    # ============================== 
    print("\nGenerating Test trials...")    

    # sample latin hypercube with a `n_sample` of the lowest common multiple
    samples = sampler.integers(l_bounds=[0 for _ in params], 
                            u_bounds=[len(vals) for _,vals in params], 
                            n=lcm)

    # interpret samples and generate experiments
    columns = [param for param,_ in params]
    columns.insert(0,'Scenario ID')
    data = []
    for sample in tqdm(samples, desc='Generating experiments', leave=False):        
        # create row of values 
        row = []
        row_query = {}

        for param,val_idx in zip(params, sample):
            param_name,vals = param
            value = vals[val_idx]
            row.append(value)
            row_query[param_name] = value

        # find scenario id  
        mask = (full_trials_df[list(row_query)] == pd.Series(row_query)).all(axis=1)
        scenario_id = max(full_trials_df[mask]['Scenario ID'].values)
        row.insert(0, scenario_id)

        # add to list of experiments
        data.append(row)

    # limit to 5 trials
    data = data[:5] if len(data) > 5 else data

    # create data frame
    test_trials_df = pd.DataFrame(data=data, columns=columns)
    test_trials_df.sort_values(by='Scenario ID', inplace=True)

    # save to csv
    test_trials_path = os.path.join(out_dir, f"test_trials-{seed}_seed.csv")
    test_trials_df.to_csv(test_trials_path, index=False)
    print(f" - Test trials saved to:\n   `{test_trials_path}`")

    return 

if __name__ == "__main__":
    # print welcome
    print_banner('Experiment generator for Preplanner Parametric Study')

    # define experiment parameters
    params = [
        ("Num Sats", [24, 48, 96, 192]),
        ("Ground Segment", ["Single NEN", "Full NEN", "None"]),
        ("Task Arrival Rate", [10, 100, 500, 1000]),
        ("Target Distribution", [25.0, 60.0, 90.0]),
    ]

    # generate experiments
    main(params, 2)

    # print completion message
    print("\nExperiment generation complete!")