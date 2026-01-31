import argparse

import os
import random
import shutil

import numpy as np
import pandas as pd
import tqdm

from chess3d.utils import print_welcome, LEVELS


def main(experiments_name : str,
         overwrite : bool = True,
         seed : int = 1000
         ):    

    # make output directory
    events_dir = './'

    # set simulation duration in hours
    sim_duration : float = 24.0 / 24.0

    # extract event parameters
    experiment_name = "lake_events"
    grid_name = f"lake_event_points"
    event_duration = 3600
    n_events = 790
    min_severity = 0.0
    max_severity = 100
    measurement_list = ['sar', 'visual', 'thermal']


    # get grid
    grid_path : str = os.path.join('./', f'{grid_name}.csv')
    grid : pd.DataFrame = pd.read_csv(grid_path)

    # run cases
    create_events(events_dir,
                  experiment_name, 
                  grid, 
                  sim_duration, 
                  n_events, 
                  event_duration, 
                  min_severity, 
                  max_severity, 
                  measurement_list, 
                  overwrite,
                  seed)

def create_events(experiments_dir : str,
                  experiment_name : str, 
                  grid : pd.DataFrame, 
                  sim_duration : float, 
                  n_events : int, 
                  event_duration : float, 
                  min_severity : float, 
                  max_severity : float, 
                  measurements : list,
                  overwrite : bool = False,
                  seed : int = 1000
                  ) -> str:
    # set random seed
    random.seed(seed)
    
    # check if measurements list contains more than one measurement
    if len(measurements) < 2: raise ValueError('`measurements` must include more than one sensor')

    # generate events
    events = []
    for _ in tqdm.tqdm(range(int(n_events * sim_duration)), 
                       desc=f'Generating Events for {experiment_name}', 
                       leave=False):
        
        while True:
            # generate start time 
            t_start = sim_duration * 24 * 3600 * random.random()
            
            gp_history = set()
            while True:
                # select a random ground point for this event
                gp_index = random.randint(0, len(grid)-1)

                if gp_index in gp_history: continue

                gp_history.add(gp_index)
                gp = grid.iloc[gp_index]

                # check if time overlap exists in the same ground point
                overlapping_events = [(t_start_overlap,duration_overlap)
                                    for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                                    if gp_index == gp_index_overlap
                                    and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                                    or   t_start <= t_start_overlap <= t_start + event_duration*3600)]
                
                # if no overlaps, break random generation cycle
                if not overlapping_events: break

                # if all ground points have overlaps at this time, try another start time
                if len(gp_history) == len(grid): break

            # if no overlaps, break random generation cycle
            if not overlapping_events: break

        # generate severity
        severity = max_severity * random.random() + min_severity

        # generate required measurements        
        n_measurements = random.randint(2,len(measurements)-1)
        required_measurements = random.sample(measurements,k=n_measurements)
        measurements_str = '['
        for req in required_measurements: 
            if required_measurements.index(req) == 0:
                measurements_str += req
            else:
                measurements_str += f',{req}'
        measurements_str += ']'
        
        # create event
        event = [
            gp_index,
            gp['lat [deg]'],
            gp['lon [deg]'],
            t_start,
            event_duration * 3600,
            severity,
            measurements_str
        ]

        # add to list of events
        events.append(event)

    # validate event generation constraints
    for gp_index,_,_,t_start,duration,_,_ in tqdm.tqdm(events, 
                       desc=f'Validating {experiment_name} events', 
                       leave=False):
        
        # check if time overlap exists in the same ground point
        overlapping_events = [(t_start_overlap,duration_overlap)
                            for gp_index_overlap,_,_,t_start_overlap,duration_overlap,_,_ in events
                            if gp_index == gp_index_overlap
                            and abs(t_start - t_start_overlap) > 1e-3
                            and (t_start_overlap <= t_start <= t_start_overlap + duration_overlap
                            or   t_start <= t_start_overlap <= t_start + duration)]
        
        assert not overlapping_events

    assert len(events) == n_events

    # compile list of events
    events_df = pd.DataFrame(data=events, columns=['gp_index','lat [deg]','lon [deg]','start time [s]','duration [s]','severity','measurements'])
    
    # save list of events to events path 
    events_df.to_csv(f'{experiment_name}.csv',index=False)

if __name__ == "__main__":
    
    # create argument parser
    parser = argparse.ArgumentParser(prog='3D-CHESS - Parametric Event Generator Study',
                                     description='Generates events for a given grid.',
                                     epilog='- TAMU')
    
    # set parser arguments
    parser.add_argument('-s',
                        '--scenario-name', 
                        help='name of scenario being used to select the location of events',
                        type=str,
                        required=False,
                        default='experiments_seed-1000')
    parser.add_argument('-o', 
                        '--overwrite',
                        default=False,
                        help='results overwrite toggle',
                        required=False,
                        type=bool) 
    
    # parse arguments
    args = parser.parse_args()
    
    # extract arguments
    scenario_name = args.scenario_name
    overwrite = args.overwrite

    # print welcome
    print_welcome('Event generator for Parametric Study')

    # run simulation
    main(scenario_name, 
        #  overwrite
         )

    # print DONE
    print(f'Event generation for sims DONE')
    