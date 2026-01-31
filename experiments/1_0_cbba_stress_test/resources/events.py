import os
import random
from typing import List
import uuid
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from chess3d.utils import print_banner


if __name__ == "__main__":
    # print welcome
    print_banner('Event generator for Internal Validation Study')

    # set seed
    seed = 1000

    # create random number generator with seed
    rng = np.random.default_rng(seed)

    # load trials
    trials_path = os.path.join('trials', f'full_factorial_trials.csv')
    assert os.path.isfile(trials_path), f'Cannot find trials file at `{trials_path}`'
    trials : pd.DataFrame = pd.read_csv(trials_path)
    
    # collect grid types, number of groundpoints and grid distribution
    for _,row in tqdm(trials.iterrows(), desc='Generating scenario events', unit=' scenarios'):
        # unpack scenario parameters
        scenario_id = row['Scenario ID']
        arrival_rate = row['Task Arrival Rate'] # [1/day]
        arrival_rate_hz = arrival_rate / 24 / 3600 # [1/s]
        distribution = row['Target Distribution']           # [deg]

        # load matching target grid
        grid_name = f'random_uniform_inland_grid_5000_latbounds--{round(distribution,1)}to{round(distribution,1)}_seed-{seed}'
        grid_path = os.path.join('grids', f'{grid_name}.csv')
        assert os.path.isfile(grid_path), f'Cannot find grid file at `{grid_path}`'
        grid = pd.read_csv(grid_path)

        # generate events for this trial
        if arrival_rate_hz < 0: raise ValueError("`arrival_rate` must be >= 0")
                
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

        columns = ['gp_index', 'lat [deg]', 'lon [deg]', 'event type', 'start time [s]', 'duration [s]', 'severity', 'measurements', 'id']
        events = []
        for t_start,(gp_idx,gp_row) in tqdm(zip(times, selected_points.iterrows()), desc='Generating events', unit=' events', leave=False):
            # generate event parameters
            duration = rng.uniform(5.0, 15.0) * 60  # [s]
            severity = rng.uniform(5.0, 10.0)       # [norm]
            measurements = ['IMG_A','IMG_B', 'IMG_C']
            id = str(uuid.uuid1())

            # save event
            event = [gp_idx, gp_row['lat [deg]'], gp_row['lon [deg]'], 'generic event', t_start, duration, severity, measurements, id]
            events.append(event)

        events_df = pd.DataFrame(data=events, columns=columns)

        # save events to file
        events_path = os.path.join('events', f'scenario_{scenario_id}_events.csv')
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        events_df.to_csv(events_path, index=False)
        
    print("All scenario events generated!")