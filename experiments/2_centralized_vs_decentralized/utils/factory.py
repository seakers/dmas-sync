import copy
import json
import os
from typing import List, Tuple
import pandas as pd

from .constellations import generate_commercial, generate_walker_delta

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
                                   data_processing : str,
                                ) -> dict:
    
    if "oracle" in data_processing.lower():
        mission_filename = "response.json"
    else:
        mission_filename = "monitoring.json"
    
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
                "rulesPath" : f"./experiments/2_centralized_vs_decentralized/resources/connectivity/{constellation.lower()}_{connectivity.lower()}.json",
                "relaysEnabled" : True
            },
            "scenarioPath" : base_path,
            "name" : results_dir,
            "missionsPath" : os.path.join(base_path, 'resources','missions',mission_filename)
        }

def create_grid_specifications(base_path : str, scenario : str, date : str) -> dict:
    # construct grid file path
    lakes_grid_name = f'algal_bloom_grid.csv'
    rivers_grid_name = f"high_flow_river_grid_{date}.csv"    
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
                                              constellation : str,
                                              date : str,
                                              reduced : bool) -> dict:
    # define out_dir name
    scenario_name = f"{constellation.lower()}_{scenario.lower()}_{date}"
    if reduced: scenario_name += "_reduced"
    
    # make out_dir if it does not exist
    out_dir = os.path.join(base_path, 'orbit_data', scenario_name)
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # define `save_unprocessed` flag
    save_unprocessed = "True" if reduced else "False"

    # return settings specifications
    return {
            "coverageType": "GRID COVERAGE",
            "outDir" : out_dir,
            "saveUnprocessedCoverage" : save_unprocessed
        }
    
def create_spacecraft_specifications(
                                     preplanner : str, 
                                     replanner : str, 
                                     scenario : str,
                                     data_processing : str,
                                     constellation : str,
                                     date : str,
                                     events_path : str,
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

        # assign onboard processing if specified
        if data_processing.lower() == 'onboard':
            # TODO generate events foreach possible mission configuration and assign path here
            if "commercial" in constellation.lower():
                raise NotImplementedError("Onboard data processing is not yet implemented for commercial constellations in this study. Please set data processing type to 'oracle' or 'none' for now.")
            elif "walker" in constellation.lower():                
                sat_mission = satellite_spec['mission']
                if "algal bloom" in sat_mission.lower():
                    event_type = 'algal_bloom'
                elif "high flow" in sat_mission.lower():
                    event_type = 'high_flow_river'
                elif "fire" in sat_mission.lower():
                    event_type = 'wildfire'
                else:
                    raise ValueError(f"Could not identify event type from satellite mission: {sat_mission}")
            else:
                raise ValueError(f"Constellation {constellation} not recognized for onboard data processing assignment.")
            
            events_path = events_path.split('/')[:-1] + [f'{event_type}_{date}.csv']
            events_path = os.path.join(*events_path)
            
            satellite_spec['science'] = {
                "@type": "lookup", 
                "eventsPath" : events_path
            }

    # return satellite specifications
    return satellite_specifications

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
        centralized_planner_specs = copy.deepcopy(ground_operator_specs_template['planner'])

        # set planner 
        centralized_planner_specs['planner']['preplanner'] = planner_specs['preplanners'][preplanner.lower()]

        # add to list of operators
        operators.append(centralized_planner_specs)

    # get event date from events path to construct announcer specs
    *event_dir_path,event_filename = events_path.split('/')
    scenario = event_filename.split('_')[0]
    event_date = event_filename.split('_')[-1].split('.')[0]
   
    # define scenario-specific event types to announce
    announcer_types = []
    if scenario.lower() == 'lakes':
        announcer_types = ['algal_bloom']
    elif scenario.lower() == 'rivers':
        announcer_types = ['high_flow_river']   
    elif scenario.lower() == 'wildfires':
        announcer_types = ['wildfire']
    elif scenario.lower() == 'water-quality':
        announcer_types = ['algal_bloom', 'high_flow_river']
    elif scenario.lower() == 'comprehensive':
        announcer_types = ['algal_bloom', 'high_flow_river', 'wildfire']
    else:
        raise ValueError(f"Scenario `{scenario}` not recognized for ground operator specification generation.")

    # define mission type based on data processing type
    if data_processing.lower() == 'onboard':
        mission_type = "monitoring"
    else:
        mission_type = "response"

    # create event announcers for each relevant event type
    for announcer_type in announcer_types:
        # skip announcer creation if data processing type is onboard; only announce wildfires
        skip_announcer = bool(data_processing.lower() == 'onboard' and announcer_type in ['algal_bloom', 'high_flow_river'])

        # create event announcer ground operator
        announcer_specs = copy.deepcopy(ground_operator_specs_template['announcer'])

        # set events path
        file_name = f"{announcer_type}_{event_date}.csv" if not skip_announcer else "no_events.csv"
        announcer_events_path = os.path.join(*event_dir_path, file_name)
        announcer_specs['planner']['preplanner']['eventsPath'] = announcer_events_path

        # update name of announcer based on event type
        announcer_type_name = announcer_type.replace('_', ' ').title()
        announcer_specs['name'] += f" ({announcer_type_name}s)"
        announcer_specs['@id'] += f"-{announcer_type}s"         

        # add mission to planner specifications based on event type
        if announcer_type == 'algal_bloom':
            announcer_specs['mission'] = f'algal bloom {mission_type}'
        elif announcer_type == 'high_flow_river':
            announcer_specs['mission'] = f'high flow river {mission_type}'
        elif announcer_type == 'wildfire':
            announcer_specs['mission'] = f'wildfire {mission_type}'
        else:
            raise ValueError(f"Announcer type `{announcer_type}` not recognized for mission assignment in ground operator specification generation.")

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
    events_file = f'{scenario.lower()}_case_{date}.csv'
    events_path = os.path.join(base_path, 'resources', 'events', 'processed', events_file)

    # set simulation duration and propagator step size
    mission_specs['duration'] = duration
    mission_specs['propagator']['stepSize'] = step_size

    # set scenario specifications
    mission_specs['scenario'] = create_scenario_specifications(base_path, results_dir, events_path, constellation, connectivity, data_processing)

    # set target distribution type
    mission_specs['grid'] = create_grid_specifications(base_path, scenario, date)

    # set propagator settings
    mission_specs['settings'] \
        = create_propagator_settings_specifications(base_path, scenario, constellation, date, reduced)
    
    # create satellite specifications
    mission_specs['spacecraft'] \
        = create_spacecraft_specifications(preplanner, replanner, scenario, data_processing, constellation, date, events_path,
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