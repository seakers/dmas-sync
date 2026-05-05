import copy
import json
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

from dmas.utils.constellations import Constellation, WalkerDeltaConstellation
from .constellations import generate_commercial, generate_walker_delta, generate_tdrss

# ------------------------------------------------------------------
# Study setup helper functions
# ------------------------------------------------------------------

def get_base_path() -> str:
    # get current working directory
    cwd = os.getcwd()
    
    # ensure script is being run from root directory
    if 'experiments' in cwd: 
        raise EnvironmentError(f"Please run this script from the root `3dchess/` directory, not from within `{cwd}`.")

    # define desired base path for experiment
    base_path = os.path.join('.','experiments','2_centralized_vs_decentralized')    
    
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
    assert lower_bound < upper_bound, f"Lower bound {lower_bound} must be less than upper bound {upper_bound}"
    
    # clip upper bound in case it is set to infinity or over n_trials
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

    # load planner specifications
    planner_specs_file = os.path.join(base_path, 'resources','templates','planners.json')
    with open(planner_specs_file, 'r') as planner_specs_file:
        planner_specs : dict = json.load(planner_specs_file)

    return mission_specs_template, ground_operator_specs_template, \
                spacecraft_specs_template, instrument_specs, planner_specs

# -----------------------------------------------------------------------
# Generates MissionSpecs dictionaries from trial parameters and templates
# -----------------------------------------------------------------------

def create_scenario_specifications(base_path : str, 
                                   results_dir : str, 
                                   events_path : str, 
                                   constellation : str,
                                   connectivity : str,
                                ) -> dict:
    return {
            "events": {
                "@type": "PREDEF",
                "eventsPath" : events_path
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "connectivity" : {
                "@type" : "PREDEF",
                "rulesPath" : f"./experiments/2_centralized_vs_decentralized/resources/connectivity/{constellation.lower()}-{connectivity.lower()}.json",
                "relaysEnabled" : True
            },
            "scenarioPath" : base_path,
            "name" : results_dir,
            "missionsPath" : os.path.join(base_path, 'resources','missions',f'missions.json')
        }

def create_grid_specifications(base_path : str, scenario : str, date : str) -> dict:
    # construct grid file path
    lakes_grid_name = f'lake_grid.csv'
    rivers_grid_name = f"river_grid.csv"    
    wildifres_grid_name = f'wildfire_grid_{date}.csv'

    # grid_path = os.path.join(base_path, 'resources','grids', grid_name)
    
    # return grid specifications
    grids = [
        {
            "@type": "customGrid",
            "covGridFilePath": os.path.join(base_path, 'resources','grids', lakes_grid_name)
        },
        {
            "@type": "customGrid",
            "covGridFilePath": os.path.join(base_path, 'resources','grids', rivers_grid_name)
        }
    ]

    if scenario.lower() == "comprehensive":
        grids.append({
            "@type": "customGrid",
            "covGridFilePath": os.path.join(base_path, 'resources','grids', wildifres_grid_name)
        })

    return grids

def create_propagator_settings_specifications(base_path : str, 
                                              scenario : str,
                                              date : str,
                                              reduced : bool) -> dict:
    # define out_dir name
    scenario_name = f"{scenario.lower()}_{date}"
    if reduced: scenario_name += "_reduced"
    
    # make out_dir if it does not exist
    out_dir = os.path.join(base_path, 'orbit_data', scenario_name)
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # return settings specifications
    return {
            "coverageType": "GRID COVERAGE",
            "outDir" : out_dir,
            "saveUnprocessedCoverage" : "False"
        }
    
def create_spacecraft_specifications(
                                     preplanner : str, 
                                     replanner : str, 
                                     scenario : str,
                                     data_processing : str,
                                     constellation : str,
                                     date : str,
                                     spacecraft_specs_template : dict, 
                                     instrument_specs : dict, 
                                     planner_specs : dict,
                                    ) -> List[dict]:
    # define mission types based on data processing type
    if data_processing.lower() == 'onboard':
        mission_type = "monitoring"
    else:
        mission_type = "response"
    
    # generate constellation design based on scenario parameters
    if "commercial" in constellation.lower():
        satellite_specifications = generate_commercial(
            spacecraft_specs_template=spacecraft_specs_template,
            instrument_specs=instrument_specs,
            mission_type = mission_type
        )
    elif "walker" in constellation.lower():
        satellite_specifications = generate_walker_delta(
            spacecraft_specs_template=spacecraft_specs_template,
            instrument_specs=instrument_specs,
            mission_type=mission_type
        )    

    # assign preplanners and replanners 
    for satellite_spec in satellite_specifications:       
        # check if satellite has autonomous planning capabilities (i.e. if it is not a simple relay satellite like TDRSS)
        if 'planner' not in satellite_spec:
            continue  # skip planner assignment for this satellite

        # set planner settings
        if preplanner.lower() != 'none':
            if 'centralized' in preplanner.lower():
                satellite_spec['planner']['preplanner'] = planner_specs['preplanners']['worker']
            else:
                satellite_spec['planner']['preplanner'] = planner_specs['preplanners'][preplanner.lower()]
        if replanner.lower() != 'none':
            satellite_spec['planner']['replanner'] = planner_specs['replanners'][replanner.lower()]
        
        # enforce planning horizon for CBBA preplanner if needed
        if replanner.lower() == 'cbba' and preplanner.lower() == 'none':
            satellite_spec['planner']['preplanner'] = planner_specs['preplanners']['blank']

        # remove planner if no preplanner or replanner specified
        if preplanner.lower() == 'none' and replanner.lower() == 'none':
            satellite_spec.pop('planner', None)  

    # return satellite specifications
    return satellite_specifications

def setup_announcer_preplanner(events_path : str) -> dict:
    """ Setup announcer planner configuration for the scenario. """

    # validate event file exists
    assert os.path.isfile(events_path), \
        f"Event file not found: {events_path}"
    
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


def create_ground_operator_specifications(
        preplanner : str,         
        data_processing : str, 
        events_path : str,
        planner_specs : dict,
        ground_operator_specs_template : dict, 
    ) -> List[dict]:

    # initialize list of operators
    operators = []

    if 'centralized' in preplanner.lower():      
        # create planner ground operator  
        announcer_specs = copy.deepcopy(ground_operator_specs_template['planner'])

        # set planner 
        announcer_specs['planner']['preplanner'] = planner_specs['preplanners'][preplanner.lower()]

        # add to list of operators
        operators.append(announcer_specs)

    if data_processing.lower() == 'oracle':
        # create event announcer ground operator
        announcer_specs = copy.deepcopy(ground_operator_specs_template['announcer'])
    
        # set events path
        announcer_specs['planner']['preplanner']['eventsPath'] = events_path

        # add to list of operators
        operators.append(announcer_specs)

    # return ground operator specifications
    return operators


def generate_scenario_mission_specs(mission_specs_template : dict, 
                                    duration : float, 
                                    step_size : float, 
                                    base_path : str, 
                                    trials_filename : str,
                                    trial_id : int, preplanner : str, replanner : str, connectivity : str, scenario : int, 
                                    data_processing : str, constellation : str, date : str,   
                                    spacecraft_specs_template : dict, instrument_specs : dict,
                                    planner_specs : dict,
                                    ground_operator_specs_template : dict, reduced : bool) -> dict:
    
    """ Generate mission specifications for a given scenario. """
    # create mission specifications from template
    mission_specs = copy.deepcopy(mission_specs_template)

    # define output directory name based on scenario parameters
    results_dir = f"{trials_filename}_trial-{trial_id}"
    if reduced: results_dir += "_reduced"
    
    # define event file path based on scenario parameters
    events_file = f'{scenario.lower()}_case_{date}_.csv'
    events_path = os.path.join(base_path, 'resources', 'events', events_file)

    # set simulation duration and propagator step size
    mission_specs['duration'] = duration
    mission_specs['propagator']['stepSize'] = step_size

    # set scenario specifications
    mission_specs['scenario'] = create_scenario_specifications(base_path, results_dir, events_path, constellation, connectivity)

    # set target distribution type
    mission_specs['grid'] = create_grid_specifications(base_path, scenario, date)

    # set propagator settings
    mission_specs['settings'] \
        = create_propagator_settings_specifications(base_path, scenario, date, reduced)
    
    # create satellite specifications
    mission_specs['spacecraft'] \
        = create_spacecraft_specifications(preplanner, replanner, scenario, data_processing, constellation, date,
                                           spacecraft_specs_template, instrument_specs, planner_specs)
    
    # set network name from ground segment type
    network_name = "gs_nen_full"
    
    # set up ground stations for coverage calculations
    mission_specs['groundStation'] \
        = load_ground_stations(base_path, network_name)

    # assign ground operator to mission specs
    mission_specs['groundOperator'] \
        = create_ground_operator_specifications(preplanner, data_processing, events_path, planner_specs, ground_operator_specs_template)
        
    # return mission specifications
    return mission_specs