
from typing import List
import pandas as pd
import pygad
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from chess3d.utils import print_banner

from chess3d.constellations import WalkerDeltaConstellation

def pareto_front(df: pd.DataFrame, objectives: dict) -> pd.DataFrame:
    """
    objectives: dict like {"mean_LCC": "max", "frac_LCC": "max", "P": "min"}
    """
    vals = df[list(objectives.keys())].to_numpy()

    # Convert all objectives to maximization
    for j, (_, sense) in enumerate(objectives.items()):
        if "min" in sense:
            vals[:, j] = -vals[:, j]

    # initiate pareto condition array
    is_pareto = np.ones(len(vals), dtype=bool)

    # check each point
    for i,val in tqdm(enumerate(vals), total=len(vals), desc="Finding Pareto front", leave=False):
        # skip non-pareto points
        if not is_pareto[i]: continue

        # any point dominates i?
        dominates = np.all(vals >= val, axis=1) & np.any(vals > val, axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False

    return df[is_pareto].copy()

if __name__ == "__main__":
    """
    Optimizing Walker Delta Constellation Connectivity Experiment

    GOAL: find the optimal Walker Delta constellation parameters (i.e., number of planes and
    phasing parameter) that MAXIMIZE the connectivity within a satellite constellation of a
    given size.
    
    """
    # terminal welcome message
    print_banner(f'Walker Delta Constellation Connectivity Experiment')

    # set inclination and altitude
    inc = 98.0  # [deg]
    alt = 550.0 # [km]

    # define number of satellites for each constellation
    trials = [12, 48, 96, 192]
    # trials = [8]

    trial_params = [None for _ in trials]
       
    # find optimal parameters for each trial
    for i, num_sats in enumerate(trials):
        # define search space
        PF_space = [ (p,f) 
                    for p in range(1, num_sats+1) # up to `num_sats` planes
                    for f in range(0, p)          # phasing param from 0 to p-1
                ]
        
        # define objectives
        objectives = {
            "max lcc time-fraction [norm]": "max", 
            "avg lcc [norm]": "max", 
            "avg num components": "min",
        }
        # objectives = {
        #     "max lcc time-fraction [norm]": "min", 
        #     "avg lcc [norm]": "min", 
        #     "avg num components": "max",
        # }

        if len(PF_space) <= 25: # small search space, evaluate all options
            # initialize tracking variables
            metrics_df = None

            # evaluate all options
            for (num_planes, phasing) in tqdm(PF_space, desc=f"Evaluating all options for {num_sats} sats"):
                # create constellation
                constellation = WalkerDeltaConstellation(alt, inc, num_sats, num_planes, phasing)
                
                # evaluate connectivity
                metrics_series_df,scalar_metrics = constellation.evaluate_connectivity(debug=False)

                # convert to dataframe 
                scalar_metrics_dict = {
                    "alt [km]" : alt,
                    "inc [deg]" : inc,
                    "num sats" : num_sats,
                    "num planes" : num_planes,
                    "phasing param" : phasing,
                }
                scalar_metrics_dict.update({key : [val] for key, val in scalar_metrics.items()})
                scalar_metrics_df = pd.DataFrame(scalar_metrics_dict)

                # store results
                if metrics_df is None:
                    metrics_df = scalar_metrics_df
                else:
                    metrics_df = pd.concat([metrics_df, scalar_metrics_df], ignore_index=True)
    
            # find pareto front of configurations
            pareto_df = pareto_front(metrics_df, objectives)

            # pick best option (highest max lcc)
            best_idx = pareto_df["max lcc [norm]"].idxmax()
            best_params = pareto_df.loc[best_idx]

            # store best params
            trial_params[i] = dict(best_params)
        
        else:
            # define fitness function
            def fitness_func(ga_instance, solution, solution_idx):
                num_planes, phasing = solution
                num_planes = int(num_planes)
                phasing = int(phasing)

                if phasing >= num_planes:
                    # raise ValueError("Phasing parameter must be less than number of planes.")
                    return [-1e6] * len(objectives)  # invalid solution penalty

                # create constellation
                constellation = WalkerDeltaConstellation(alt, inc, num_sats, num_planes, phasing)
                
                # evaluate connectivity
                _,scalar_metrics = constellation.evaluate_connectivity(debug=False)

                # extract relevant metrics
                fitness_dict = {}
                for key, sense in objectives.items():
                    if "max" in sense:
                        fitness_dict[key] = scalar_metrics[key]
                    elif "min" in sense:
                        fitness_dict[key] = -scalar_metrics[key]

                return [fitness_dict[key] for key in sorted(objectives.keys())]

            # define GA population parameters
            num_parents_mating = 10
            sol_per_pop = 20
            num_generations = len(PF_space) // num_parents_mating
            # num_generations = 1 # debug

            # define GA genes
            num_genes = 2
            gene_space = [ list(range(1, num_sats+1)),
                           list(range(0, num_sats))
                          ]
            gene_constraint=[
                                lambda _,values: values,  # num_planes can be any value
                                lambda solution,values: [val for val in values if val<[solution[0]]] # phasing < num_planes
                        ]
            initial_population = [ (p,f) 
                                  for f in range(0,num_sats)
                                  for p in sorted(set( np.linspace(1, num_sats, num=sol_per_pop, dtype=int).tolist() )) 
                                  ]

            # Initiate GA instance
            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                fitness_func=fitness_func,
                                parent_selection_type='nsga2',
                                gene_space=gene_space,
                                gene_constraint=gene_constraint,
                                initial_population=initial_population[:sol_per_pop]
                                )

            # run GA instance
            ga_instance.run()

            # plot fitness evolution
            # fig = ga_instance.plot_fitness(label=sorted(objectives.keys()))

            # extract best solution
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            # print(f"Parameters of the best solution : {solution}")
            # print(f"Fitness value of the best solution = {solution_fitness}")

            # extract best params
            num_planes, phasing = solution
            num_planes = int(num_planes)
            phasing = int(phasing)
            best_params = {
                    "alt [km]" : alt,
                    "inc [deg]" : inc,
                    "num sats" : num_sats,
                    "num planes" : num_planes,
                    "phasing param" : phasing,
                }
            best_params.update({
                key : solution_fitness[i] if "max" in objectives[key] else -solution_fitness[i]
                for i,key in enumerate(sorted(objectives.keys()))
            })

            # store best params
            trial_params[i] = dict(best_params)
    
    # print results
    print("Optimal Parameters Found:")
    for i, params in enumerate(trial_params):
        print(f" - {params['num sats']} sats: {params['num planes']} planes, phasing {params['phasing param']}")

    print('DONE!')
    x = 1

