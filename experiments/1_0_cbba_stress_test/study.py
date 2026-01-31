

import argparse
import copy
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from dmas.core.constellations import Constellation, WalkerDeltaConstellation
from dmas.core.orbitdata import OrbitData
from dmas.core.simulation import Simulation
from dmas.utils.tools import LEVELS, print_scenario_banner

def get_base_path() -> str:
    # get current working directory
    cwd = os.getcwd()
    
    # ensure script is being run from root directory
    if 'experiments' in cwd: 
        raise EnvironmentError(f"Please run this script from the root `3dchess/` directory, not from within `{cwd}`.")

    # define desired base path for experiment
    base_path = os.path.join('.','experiments','1_0_cbba_stress_test')    
    
    # return base path
    return base_path

def load_trials(base_path : str, trial_filename : str, lower_bound : int, upper_bound : int) -> pd.DataFrame:
    # construct trials file path
    trials_file = os.path.join(base_path, 'resources','trials',f'{trial_filename}.csv')
    assert os.path.exists(trials_file), f"Trials file not found at: {trials_file}"
    
    # load trials list
    trials : pd.DataFrame = pd.read_csv(trials_file)
    n_trials = len(trials)

    # check if bounds are valid
    assert 0 <= lower_bound < n_trials, f"Lower bound {lower_bound} is out of range [0, {n_trials})"
    assert 0 < upper_bound <= n_trials or upper_bound == np.Inf, f"Upper bound {upper_bound} is out of range (0, {n_trials}]"
    assert lower_bound < upper_bound, f"Lower bound {lower_bound} must be less than upper bound {upper_bound}"
    
    # clip upper bound in case it is set to infinity
    upper_bound = min(upper_bound, n_trials)

    # apply bounds
    trials = trials.iloc[lower_bound:upper_bound].reset_index(drop=True)

    # return trials
    return trials

def load_templates(base_path : str) -> Tuple[dict, dict, dict, dict]:
    # load mission specifications template file
    mission_template_file = os.path.join(base_path, 'resources','templates','MissionSpecs.json')
    with open(mission_template_file, 'r') as mission_template_file:
        mission_specs_template : dict = json.load(mission_template_file)

    # load ground operator specifications template file
    ground_operator_template_file = os.path.join(base_path, 'resources','templates','groundOperator.json')
    with open(ground_operator_template_file, 'r') as ground_operator_template_file:
        ground_operator_specs_template : dict = json.load(ground_operator_template_file)

    # load spacecraft specifications template file
    spacecraft_template_file = os.path.join(base_path, 'resources','templates','spacecraft.json')
    with open(spacecraft_template_file, 'r') as spacecraft_template_file:
        spacecraft_specs_template : dict = json.load(spacecraft_template_file)

    # load available instrument specifications 
    instrument_specs_file = os.path.join(base_path, 'resources','templates','instruments.json')
    with open(instrument_specs_file, 'r') as instrument_specs_file:
        instrument_specs : dict = json.load(instrument_specs_file)

    return mission_specs_template, ground_operator_specs_template, \
                spacecraft_specs_template, instrument_specs

def create_scenario_specifications(base_path : str, trial_filename : str, scenario_id : int) -> dict:
    return {
            "connectivity": "LOS",
            "events": {
                "@type": "PREDEF",
                "eventsPath" : os.path.join(base_path, 'resources','events',f'scenario_{scenario_id}_events.csv')
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "scenarioPath" : base_path,
            "name" : f"{trial_filename}_scenario_{scenario_id}",
            "missionsPath" : os.path.join(base_path, 'resources','missions',f'missions.json')
        }

def create_grid_specifications(base_path : str, target_distribution : int) -> dict:
    # construct grid file path
    grid_name = f'random_uniform_inland_grid_5000_latbounds--{target_distribution}to{target_distribution}_seed-1000.csv'
    grid_path = os.path.join(base_path, 'resources','grids', grid_name)
    
    # return grid specifications
    return [{
        "@type": "customGrid",
        "covGridFilePath": grid_path
    }]

def create_propagator_settings_specifications(base_path : str, scenario_id : int, num_sats : float, gnd_segment : str, target_distribution) -> dict:
    # define out_dir name
    # scenario_name = f"nsats-{num_sats}_gndseg-{gnd_segment.lower()}_tgtdist-{int(target_distribution)}"
    scenario_name = f"scenario_{scenario_id}"
    
    # make out_dir if it does not exist
    out_dir = os.path.join(base_path, 'orbit_data', scenario_name)
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # return settings specifications
    return {
            "coverageType": "GRID COVERAGE",
            "outDir" : out_dir,
        }
    
def create_spacecraft_specifications(num_sats : int, 
                                    spacecraft_specs_template : dict, 
                                    instrument_specs : dict,
                                    base_path : str,
                                    scenario_id : int,
                                    gnd_segment : str
                                    ) -> List[dict]:
    # get altitude and inclination from template
    alt = spacecraft_specs_template['orbitState']['state']['sma'] - Constellation.EARTH_RADIUS_KM  # [km]
    inc = spacecraft_specs_template['orbitState']['state']['inc']                                  # [deg]
    
    # create a walker-delta constellation for given number of satellites;
    #   choose number of planes and satellites per plane to approximate a square-ish constellation
    num_planes = min([p for p in range(1, num_sats+1)], 
                    key=lambda p: abs(num_sats/p - np.sqrt(num_sats)))
    # choose a fixed phasing factor
    phasing_factor = 1

    # create constellation instance
    constellation = WalkerDeltaConstellation(alt, inc, num_sats, num_planes, phasing_factor)

    # extract orbital elements
    orbital_elements : List[dict] = constellation.to_orbital_elements()

    # create satellite specifications list
    satellite_specifications = []
    for sat_idx,orbit_state in enumerate(orbital_elements):
        # create satellite specification from template
        satellite_spec = copy.deepcopy(spacecraft_specs_template)

        # planner settings
        # satellite_spec['planner'].pop('preplanner')

        # assign orbit state
        satellite_spec['orbitState']['state'] = orbit_state

        # select instrument 
        instrument_idx = sat_idx % len(instrument_specs)
        instrument_spec = instrument_specs[list(instrument_specs.keys())[instrument_idx]]

        # assign instrument to satellite
        satellite_spec['instrument'] = copy.deepcopy(instrument_spec)

        # determine satellite name and ID
        satellite_spec['name'] = f"{instrument_spec['@id']}_sat_{sat_idx // len(instrument_specs)}"
        satellite_spec['@id'] = f"sat_{sat_idx}"

        # add to list of satellite specifications
        satellite_specifications.append(satellite_spec)

        # check if there was a ground segment specified; remove ground station network if none
        if gnd_segment.lower() == "none": satellite_spec.pop('groundStationNetwork', None)

    # check if there was a ground segment specified
    if gnd_segment.lower() == "none": # no ground segment; assign announcer role to a copy of the first satellite
        # create copy of first satellite spec
        first_satellite_spec = copy.deepcopy(satellite_specifications[0])

        # rename satellite
        first_satellite_spec['name'] += '_announcer'
        first_satellite_spec['@id'] += '_announcer'

        # remove instruments from announcer satellite
        # first_satellite_spec.pop('instrument', None)
        first_satellite_spec['instruments'] = {}

        # remove replanner spec if it exists
        first_satellite_spec['planner'].pop('replanner', None)
        
        # assign announcer preplanner 
        first_satellite_spec['planner'] = setup_announcer_preplanner(base_path, scenario_id)
        
        # add to list of satellite specifications
        satellite_specifications.append(first_satellite_spec)

    # return satellite specifications
    return satellite_specifications

def setup_announcer_preplanner(base_path : str, scenario_id : int) -> dict:
    """ Setup announcer planner configuration for the scenario. """

    # validate event file exists
    assert isinstance(scenario_id, int), "`scenario_id` must be an integer"
    events_path = os.path.join(base_path, 'resources','events',f'scenario_{scenario_id}_events.csv')
    assert os.path.isfile(events_path), \
        f"Event file not found: scenario_{scenario_id}_events.csv"
    
    # return event announcer planner config
    return {
            "preplanner": {
                "@type": "eventAnnouncer",
                "debug": "False",                        
                "eventsPath" : events_path
            }
        }

def load_ground_stations(base_path : str, network_name : str) -> List[dict]:
    # construct ground stations file path
    ground_stations_path = os.path.join(base_path, 'resources','gstations', f"{network_name}.csv")
    assert os.path.isfile(ground_stations_path), f"Ground stations file not found: {ground_stations_path}"
    
    # load ground station network from file
    df = pd.read_csv(ground_stations_path)
    gs_network_df : list[dict] = df.to_dict(orient='records')

    # if no id in file, add index as id
    gs_network = []
    for gs_idx, gs_df in enumerate(gs_network_df):
        gs = {
            "name": gs_df['name'],
            "latitude": gs_df['lat[deg]'],
            "longitude": gs_df['lon[deg]'],
            "altitude": gs_df['alt[km]'],
            "minimumElevation": gs_df['minElevation[deg]'],
            "@id": gs_df['@id'] if '@id' in gs_df else f'{network_name}-{gs_idx}',
            "networkName": "NEN"
        }
        gs_network.append(gs)

    return gs_network

def create_ground_operator_specifications(base_path : str, scenario_id : int, ground_operator_specs_template : dict) -> List[dict]:
    # create ground operator specifications from template
    ground_operator_specs = copy.deepcopy(ground_operator_specs_template)
    
    # set events path
    events_path = os.path.join(base_path, 'resources','events',f'scenario_{scenario_id}_events.csv')
    ground_operator_specs['planner']['preplanner']['eventsPath'] = events_path

    # return ground operator specifications
    return [ground_operator_specs]

def generate_scenario_mission_specs(mission_specs_template : dict, duration : float, step_size : float, 
                                    base_path : str, trial_filename : str, scenario_id : int,
                                    num_sats : int, gnd_segment : str, target_distribution : int,
                                    spacecraft_specs_template : dict, instrument_specs : dict,
                                    ground_operator_specs_template : dict) -> dict:
    
    """ Generate mission specifications for a given scenario. """
    # create mission specifications from template
    mission_specs = copy.deepcopy(mission_specs_template)
    
    # set simulation duration and propagator step size
    mission_specs['duration'] = duration
    mission_specs['propagator']['stepSize'] = step_size

    # set scenario specifications
    mission_specs['scenario'] = create_scenario_specifications(base_path, trial_filename, scenario_id)

    # set target distribution type
    mission_specs['grid'] = create_grid_specifications(base_path, target_distribution)

    # set propagator settings
    mission_specs['settings'] \
        = create_propagator_settings_specifications(base_path, scenario_id, num_sats, gnd_segment, target_distribution)
    
    # create satellite specifications
    mission_specs['spacecraft'] \
        = create_spacecraft_specifications(num_sats, spacecraft_specs_template, instrument_specs, 
                                            base_path, scenario_id, gnd_segment)
    
    # set ground operator specifications if specified
    if gnd_segment.lower() != "none":
        # get network name from ground segment type
        network_name = "gs_nen_1" if "single" in gnd_segment.lower() else "gs_nen_full"
        
        # set up ground stations for coverage calculations
        mission_specs['groundStation'] \
            = load_ground_stations(base_path, network_name)

        # assign ground operator to mission specs
        mission_specs['groundOperator'] \
            = create_ground_operator_specifications(base_path, scenario_id, ground_operator_specs_template)
        
    # return mission specifications
    return mission_specs

def main(trial_filename : str, 
         lower_bound : int, 
         upper_bound : int, 
         level : int, 
         propagate_only : bool,
         overwrite : bool, 
         reevaluate : bool, 
         debug : bool):
    
    # print welcome
    print_scenario_banner(f'CBBA Stress Test Study - {trial_filename}')
    
    # get base path for experiment
    base_path : str = get_base_path()
    
    # load trials
    trials : pd.DataFrame = load_trials(base_path, trial_filename, lower_bound, upper_bound)
    print(f" - Loaded {len(trials)} trials from `{trial_filename}.csv`:  [{lower_bound}:{upper_bound}) ")

    # load templates
    mission_specs_template, ground_operator_specs_template, \
        spacecraft_specs_template, instrument_specs = load_templates(base_path)
    print(f" - Loaded experiment templates from `resources/templates/`")

    # set simulation duration and step size
    duration = 10000 / 3600 / 24.0 if debug else 1.0 # [days]
    duration = min(duration, 1.0)                   # cap at 1 day for sanity
    step_size = 10                                  # [s]

    # iterate through each trial
    for scenario_id,num_sats,gnd_segment,task_arrival_rate,target_distribution in trials.itertuples(index=False):
        # handle nan ground segment case
        gnd_segment = 'None' if not isinstance(gnd_segment, str) else gnd_segment
        
        if scenario_id > 0: print_scenario_banner(f'CBBA Stress Test Study - {trial_filename}')
        print(f"\n--- Running Trial Scenario ID: {scenario_id} ---")
        print(f" - Num Sats: {num_sats}")
        print(f" - Ground Segment: {gnd_segment}")
        print(f" - Task Arrival Rate: {task_arrival_rate} [tasks/day]")
        print(f" - Target Distribution: Lat=(-{target_distribution}°, +{target_distribution}°)")
        print(f" - Propagation Duration: {round(duration*24*3600,3)} [seconds] ({round(duration, 3)} [days])")

        # generate mission specifications for the scenario
        mission_specs : dict = generate_scenario_mission_specs(
            mission_specs_template, duration, step_size, 
            base_path, trial_filename, scenario_id,
            num_sats, gnd_segment, target_distribution,
            spacecraft_specs_template, instrument_specs,
            ground_operator_specs_template
        )

        ## define results output file name
        results_dir = os.path.join(base_path, 'results', f"{trial_filename}_scenario_{scenario_id}")
        results_summary_path = os.path.join(results_dir, 'summary.csv')

        # check if propagation-only toggle was selected
        if propagate_only:
            # if selected; only precompute orbit data
            print(" - Propagating orbit data only...")
            orbitdata_dir = OrbitData.precompute(mission_specs)
            print (f" - Orbit data propagated and stored at: `{orbitdata_dir}`")
            
            # skip to next trial
            continue 

        # initialize simulation mission
        print(" - Running full simulation...\n")
        
        # check if results do not exist or overwrite/reevaluate is set
        if not os.path.isfile(results_summary_path) or overwrite or reevaluate:
            mission : Simulation = Simulation.from_dict(mission_specs, overwrite=overwrite, level=level)

        # check if output directory was properly initalized
        assert os.path.isdir(results_dir), \
            f"Results directory not properly initialized at: {results_dir}"

        # define conditions to execute mission
        execute_conditions = [
            # there is no results directory generated yet
            not os.path.isdir(results_dir), 
            
            # there are incomplete results directories for any agent
            any([len(os.listdir(os.path.join(results_dir, d))) <= 2 
                    for d in os.listdir(results_dir)
                    if os.path.isdir(os.path.join(results_dir, d))
                    and 'manager' not in d]
                ),
            
            # overwrite flag was set
            overwrite
        ]

        # execute mission if any of the conditions are met
        if any(execute_conditions): 
            print (' - Executing simulation mission...')
            mission.execute()
        else:
            print(' - Simulation data found! Skipping execution...')

        # # print results if it hasn't been performed yet or if results need to be reevaluated
        # if not os.path.isfile(results_summary_path) or reevaluate: 
        #     print(' - Printing simulation results...')
        #     mission.process_results()

        # # ensure if summary file was properly generated at the end of the simulation
        # assert os.path.isfile(results_summary_path), \
        #     f"Results summary file not found at: {results_summary_path}"

    # study done
    return

if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(prog='3D-CHESS - CBBA Stress Test Experiment',
                                     description='Study on the performance of CBBA under varying stress conditions',
                                     epilog='- TAMU')
    
    # set parser arguments
    parser.add_argument('-n',
                        '--trial-filename', 
                        help='filename of the set of trials being run',
                        type=str,
                        required=False,
                        default='test_trials-1000_seed')
    parser.add_argument('-l',
                        '--lower-bound', 
                        help='lower bound of simulation indeces to be run (inclusive)',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-u',
                        '--upper-bound', 
                        help='upper bound of simulation indeces to be run (non-inclusive)',
                        type=int,
                        required=False,
                        default=np.Inf)
    parser.add_argument('-p', 
                        '--propagate-only',
                        help='toggles to only precompute orbit data without running full simulation',
                        default=False,
                        required=False,
                        type=bool) 
    parser.add_argument('-o', 
                        '--overwrite',
                        default=False,
                        help='results overwrite toggle',
                        required=False,
                        type=bool) 
    parser.add_argument('-r', 
                        '--reevaluate',
                        help=' results reevaluation toggle',
                        default=False,
                        required=False,
                        type=bool) 
    parser.add_argument('-d', 
                        '--debug',
                        help='toggles to run just one experiment for debugging purposes',
                        default=False,
                        required=False,
                        type=bool) 
    parser.add_argument('-L', 
                        '--level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                        default='WARNING',
                        help='logging level',
                        required=False,
                        type=str) 
    
    # parse arguments
    args = parser.parse_args()
    
    # extract arguments
    trial_filename = args.trial_filename
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    level = LEVELS.get(args.level)
    propagate_only = args.propagate_only
    overwrite = args.overwrite
    reevaluate = args.reevaluate
    debug = args.debug

    # run main study
    main(trial_filename, lower_bound, upper_bound, level, propagate_only, overwrite, reevaluate, debug)

    # print outro
    print('\n' + '='*54)
    print('STUDY COMPLETE!')