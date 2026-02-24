import os
import random
from typing import List
import uuid
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from dmas.utils.tools import print_scenario_banner


if __name__ == "__main__":
    # print welcome
    print_scenario_banner('Event generator for Internal Validation Study')

    # set seed
    seed = 1000

    # create random number generator with seed
    rng = np.random.default_rng(seed)

    # load trials
    trials_path = os.path.join('./experiments/1_cbba_validation/resources/trials', f'full_factorial_trials_2026-02-22.csv')
    assert os.path.isfile(trials_path), f'Cannot find trials file at `{trials_path}`'
    trials : pd.DataFrame = pd.read_csv(trials_path)

    # get unique arrival rate and target distribution values
    arrival_rates = trials['Task Arrival Rate'].unique()
    target_distributions = trials['Target Distribution'].unique()
    n_scenarios_per_combination = max(trials['Scenario'].unique()) + 1

    # calculate all combinations of arrival rates and target distributions
    combinations = [(ar, td) for ar in arrival_rates for td in target_distributions]
    print(f"Unique Task Arrival Rates: {arrival_rates}")
    print(f"Unique Target Distributions: {target_distributions}")
    print(f"Total unique combinations of parameters per scenario: {len(combinations)}")
    print(f"Number of scenarios per combination of parameters: {n_scenarios_per_combination}")

    # define and create events output directory
    events_dir = os.path.join('.', 'experiments','1_cbba_validation', 'resources', 'events')
    os.makedirs(events_dir, exist_ok=True)

    # create list of events for each combination of parameters
    for arrival_rate, target_distribution in tqdm(combinations, desc='Generating events for parameter combinations', unit=' combinations'):
        # unpack scenario parameters
        arrival_rate_hz = arrival_rate / 24 / 3600 # [1/s]

        # load matching target grid
        grid_name = f'random_uniform_inland_grid_5000_latbounds--{round(target_distribution,1)}to{round(target_distribution,1)}_seed-{seed}'
        grid_path = os.path.join('./experiments/1_cbba_validation/resources/grids', f'{grid_name}.csv')
        assert os.path.isfile(grid_path), f'Cannot find grid file at `{grid_path}`'
        grid = pd.read_csv(grid_path)

        # sanity check parameters
        if arrival_rate_hz < 0: raise ValueError("`arrival_rate` must be >= 0")

        # generate events for matching scenarios
        for scenario_idx in range(n_scenarios_per_combination):
            T = 3600 * 24   # duration time [s]
            t_start_s = 0.0 # start time [s]
            t_end_s = t_start_s + T     # end time [s]

            # generate event arrival times
            times: List[float] = []

            t = float(t_start_s)
            while True:
                # Exponential inter-arrival time with mean 1/rate_hz
                dt = rng.exponential(scale=1.0 / arrival_rate_hz)
                t += float(dt)
                if t >= t_end_s:
                    break
                times.append(t)

            # randomly select event locations from grid
            n_events = len(times)
            selected_indices = rng.choice(len(grid), size=n_events, replace=True)
            selected_points : pd.DataFrame = grid.iloc[selected_indices]

            # package events into dataframe
            columns = ['gp_index', 'lat [deg]', 'lon [deg]', 'event type', 'start time [s]', 'duration [s]', 'severity', 'measurements', 'id']
            events = []
            for t_start,(gp_idx,gp_row) in tqdm(zip(times, selected_points.iterrows()), desc='Generating events', unit=' events', leave=False):
                # generate event parameters
                duration = rng.uniform(5.0, 15.0) * 60  # [s]
                severity = rng.uniform(5.0, 10.0)       # [norm]
                measurements = ['IMG_A', 'IMG_B', 'IMG_C']
                id = str(uuid.uuid1())

                # save event
                event = [gp_idx, gp_row['lat [deg]'], gp_row['lon [deg]'], 'generic event', t_start, duration, severity, measurements, id]
                events.append(event)

            # compile events into dataframe
            events_df = pd.DataFrame(data=events, columns=columns)

            # sort by start time
            events_df : pd.DataFrame = events_df.sort_values(by='start time [s]').reset_index(drop=True)

            # save events to file
            event_filename = f'events_arrivalrate-{arrival_rate}_targetdist-{target_distribution}_scenario-{scenario_idx}.csv'
            events_path = os.path.join(events_dir, event_filename)
            os.makedirs(os.path.dirname(events_path), exist_ok=True)
            events_df.to_csv(events_path, index=False)
        
    print("All scenario events generated!")