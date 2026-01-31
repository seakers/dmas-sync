import math
import os
import random

import numpy as np
import pandas as pd
import tqdm

from chess3d.mission.mission import GeophysicalEvent
from chess3d.utils import print_welcome, LEVELS


def main(
         grids_names : list,
         overwrite : bool = True,
         seed : int = 1000
         ):    

    sim_events : pd.DataFrame = None
    for grid_name in grids_names:

        # extract event parameters
        sim_duration = 1.0
        event_duration = 1.0
        n_events = 1000 / 2
        min_severity = 1.0
        max_severity = 10.0
        event_type = 'flood' if 'rivers' in grid_name.lower() else 'algal bloom'

        # get grid
        grid_path : str = os.path.join('./grids', f'{grid_name}.csv')
        grid : pd.DataFrame = pd.read_csv(grid_path)

        # run cases
        events = create_events('./events',
                        grid, 
                        sim_duration, 
                        n_events, 
                        event_duration, 
                        min_severity, 
                        max_severity, 
                        event_type, 
                        overwrite,
                        seed)
        
        sim_events = events if sim_events is None else pd.concat([sim_events, events], ignore_index=False)

    # save events
    events_path = os.path.join('./events', f'lake_events_seed-{seed}.csv')

     # save list of events to events path 
    sim_events.to_csv(events_path,index=False)

def create_events(experiments_dir : str,
                  grid : pd.DataFrame, 
                  sim_duration : float, 
                  n_events : int, 
                  event_duration : float, 
                  min_severity : float, 
                  max_severity : float, 
                  event_type : list,
                  overwrite : bool = False,
                  seed : int = 1000
                  ) -> str:
    # set random seed
    random.seed(seed)

    if 'area [m^2]' in grid.columns:
        max_area = grid['area [m^2]'].max()
        min_area = grid['area [m^2]'].min()

        # generate events
        events : list[GeophysicalEvent] = []
        for _ in tqdm.tqdm(range(int(n_events * sim_duration)), 
                        desc=f'Generating Events for events of type `{event_type}`', 
                        leave=False):
            
            while True:
                # generate start time 
                t_start = sim_duration * 24 * 3600 * random.random()
                t_corr = event_duration * 3600 * random.random()
                
                gp_history = set()
                while True:
                    # select a random ground point for this event
                    gp_index = random.randint(0, len(grid)-1)

                    if gp_index in gp_history: continue

                    gp_history.add(gp_index)
                    gp = grid.iloc[gp_index]

                    # check if time overlap exists in the same ground point
                    overlapping_events = [event
                                        for _,event in events
                                        if math.isclose(event.location[0], gp['lat [deg]'], abs_tol=1e-5)
                                        and math.isclose(event.location[1], gp['lon [deg]'], abs_tol=1e-5)
                                        and (event.t_start <= t_start <= event.t_start
                                        or t_start <= event.t_start <= t_start + event_duration * 3600)]
                    
                    # if no overlaps, break random generation cycle
                    if not overlapping_events: break

                    # if all ground points have overlaps at this time, try another start time
                    if len(gp_history) == len(grid): break

                # if no overlaps, break random generation cycle
                if not overlapping_events: break

            # generate severity proportional to target area
            severity = np.interp(gp['area [m^2]'], [min_area, max_area], [min_severity, max_severity])
            if severity < min_severity: severity = min_severity
            if severity > max_severity: severity = max_severity
            
            # create event
            event = GeophysicalEvent(
                event_type,
                severity,
                (gp['lat [deg]'], gp['lon [deg]'], 0.0),
                t_start,
                t_start + event_duration * 3600,
                t_corr,            
            )

            # add to list of events
            events.append((gp_index, event))
    else:
        # generate events
        events : list[GeophysicalEvent] = []
        for _ in tqdm.tqdm(range(int(n_events * sim_duration)), 
                        desc=f'Generating Events for events of type `{event_type}`', 
                        leave=False):
            
            while True:
                # generate start time 
                t_start = sim_duration * 24 * 3600 * random.random()
                t_corr = event_duration * 3600 * random.random()
                
                gp_history = set()
                while True:
                    # select a random ground point for this event
                    gp_index = random.randint(0, len(grid)-1)

                    if gp_index in gp_history: continue

                    gp_history.add(gp_index)
                    gp = grid.iloc[gp_index]

                    # check if time overlap exists in the same ground point
                    overlapping_events = [event
                                        for _,event in events
                                        if math.isclose(event.location[0], gp['lat [deg]'], abs_tol=1e-5)
                                        and math.isclose(event.location[1], gp['lon [deg]'], abs_tol=1e-5)
                                        and (event.t_start <= t_start <= event.t_start
                                        or t_start <= event.t_start <= t_start + event_duration * 3600)]
                    
                    # if no overlaps, break random generation cycle
                    if not overlapping_events: break

                    # if all ground points have overlaps at this time, try another start time
                    if len(gp_history) == len(grid): break

                # if no overlaps, break random generation cycle
                if not overlapping_events: break

            # generate severity proportional to target area
            severity = (max_severity - min_severity) * random.random() + min_severity
            
            # create event
            event = GeophysicalEvent(
                event_type,
                severity,
                (gp['lat [deg]'], gp['lon [deg]'], 0.0),
                t_start,
                t_start + event_duration * 3600,
                t_corr,            
            )

            # add to list of events
            events.append((gp_index, event))

    assert len(events) == n_events

    # convert to list of tuples
    event_list = [(
                   gp_index,
                   event.location[0], 
                   event.location[1], 
                   event.t_start, 
                   event.t_end - event.t_start, 
                   event.severity, 
                   event.event_type, 
                   event.t_corr, 
                   event.id) 
                for gp_index,event in events]

    # compile list of events
    return pd.DataFrame(data=event_list, 
                        columns=[
                            'gp_index',
                            'lat [deg]',
                            'lon [deg]',
                            'start time [s]',
                            'duration [s]',
                            'severity',
                            'event type',
                            'decorrelation time [s]',
                            'id'
                        ])

if __name__ == "__main__":

    # print welcome
    print_welcome('Event generator for Mission Tests')

    grids_names = [
        # "HydroLAKES_polys_v10_simple_1000",
        # "HydroRIVERS_v10_simple_1000",
        "lake_event_points"
    ]

    # run simulation
    main(
         grids_names,
         overwrite=True
         )

    # print DONE
    print(f'Event generation for sims DONE')
    