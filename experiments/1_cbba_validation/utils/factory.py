import copy
import json
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

from dmas.utils.constellations import Constellation, WalkerDeltaConstellation

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
    base_path = os.path.join('.','experiments','1_cbba_validation')    
    
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
                                   num_sats : int, 
                                   latency : str
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
                "rulesPath" : f"./experiments/1_cbba_validation/resources/connectivity/nsats-{int(num_sats)}_latency-{latency.lower()}.json",
                "relaysEnabled" : True
            },
            "scenarioPath" : base_path,
            "name" : results_dir,
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

def create_propagator_settings_specifications(base_path : str, 
                                              num_sats : float, 
                                              target_distribution : float,
                                              reduced : bool) -> dict:
    # define out_dir name
    scenario_name = f"nsats-{num_sats}_tgtdist-{int(target_distribution)}"
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
    
def create_spacecraft_specifications(num_sats : int,  
                                     preplanner : str, 
                                     replanner : str, 
                                     spacecraft_specs_template : dict, 
                                     instrument_specs : dict, 
                                     planner_specs : dict
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
        if preplanner.lower() != 'none':
            satellite_spec['planner']['preplanner'] = planner_specs['preplanners'][preplanner.lower()]
        if replanner.lower() != 'none':
            satellite_spec['planner']['replanner'] = planner_specs['replanners'][replanner.lower()]

        # remove planner if no preplanner or replanner specified
        if preplanner.lower() == 'none' and replanner.lower() == 'none':
            satellite_spec.pop('planner', None)  

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

    # add TRDSS satellites to the scenario for increased connectivity
    # define GEO constellation parameters
    geo_alt = 35_786.0  # [km]
    geo_inc = 0.0     # [deg]
    geo_num_sats = 3  # number of GEO satellites
    geo_num_planes = 1  # number of planes in GEO constellation
    geo_phasing_factor = 0  # phasing factor for GEO constellation

    # create GEO constellation instance
    geo_constellation = WalkerDeltaConstellation(geo_alt, geo_inc, geo_num_sats, geo_num_planes, geo_phasing_factor)
    
    # extract orbital elements
    geo_orbital_elements : List[dict] = geo_constellation.to_orbital_elements()

    # create satellite specifications list
    for geo_sat_idx,geo_orbit_state in enumerate(geo_orbital_elements):
        # create satellite specification from template
        geo_satellite_spec : dict = copy.deepcopy(spacecraft_specs_template)

        # do not assign a planner to the GEO satellites; they are just relays in this scenario
        geo_satellite_spec.pop('planner', None)

        # assign orbit state
        geo_satellite_spec['orbitState']['state'] = geo_orbit_state

        # remove instrument; GEO satellites are just relays in this scenario
        geo_satellite_spec.pop('instrument', None)

        # define satellite name and ID
        geo_satellite_spec['name'] = f"tdrss_sat_{geo_sat_idx}"
        geo_satellite_spec['@id'] = f"tdrss_{geo_sat_idx}"

        # add to list of satellite specifications
        satellite_specifications.append(geo_satellite_spec)

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


def create_ground_operator_specifications(ground_operator_specs_template : dict, events_path : str) -> List[dict]:
    # create ground operator specifications from template
    ground_operator_specs = copy.deepcopy(ground_operator_specs_template)
    
    # set events path
    ground_operator_specs['planner']['preplanner']['eventsPath'] = events_path

    # return ground operator specifications
    return [ground_operator_specs]


def generate_scenario_mission_specs(mission_specs_template : dict, duration : float, step_size : float, 
                                    base_path : str, trials_filename : str,
                                    trial_id : int, preplanner : str, replanner : str, 
                                    num_sats : int, latency : str, 
                                    task_arrival_rate : float, target_distribution : int,
                                    scenario_idx : int,
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
    events_file = f'events_arrivalrate-{task_arrival_rate}_targetdist-{target_distribution}_scenario-{scenario_idx}.csv'
    events_path = os.path.join(base_path, 'resources', 'events', events_file)

    # set simulation duration and propagator step size
    mission_specs['duration'] = duration
    mission_specs['propagator']['stepSize'] = step_size

    # set scenario specifications
    mission_specs['scenario'] = create_scenario_specifications(base_path, results_dir, events_path, num_sats, latency)

    # set target distribution type
    mission_specs['grid'] = create_grid_specifications(base_path, target_distribution)

    # set propagator settings
    mission_specs['settings'] \
        = create_propagator_settings_specifications(base_path, num_sats, target_distribution, reduced)
    
    # create satellite specifications
    mission_specs['spacecraft'] \
        = create_spacecraft_specifications(num_sats, preplanner, replanner, 
                                           spacecraft_specs_template, instrument_specs, planner_specs)
    
    # set network name from ground segment type
    network_name = "gs_nen_full"
    
    # set up ground stations for coverage calculations
    mission_specs['groundStation'] \
        = load_ground_stations(base_path, network_name)

    # assign ground operator to mission specs
    mission_specs['groundOperator'] \
        = create_ground_operator_specifications(ground_operator_specs_template, events_path)
        
    # return mission specifications
    return mission_specs