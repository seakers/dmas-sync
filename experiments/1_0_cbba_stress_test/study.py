
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
import traceback
from dataclasses import dataclass

import cProfile
import pstats

import argparse
import copy
import json
import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pyparsing import Dict
from tqdm import tqdm

from dmas.core.constellations import Constellation, WalkerDeltaConstellation
from dmas.core.orbitdata import OrbitData
from dmas.core.simulation import Simulation
from dmas.utils.tools import LEVELS, print_scenario_banner


@dataclass(frozen=True)
class RunConfig:
    duration: float
    step_size: float
    base_path: str
    trial_filename: str
    propagate_only: bool
    overwrite: bool
    evaluate: bool
    level: int
    mission_specs_template: dict
    spacecraft_specs_template: dict
    instrument_specs: dict
    ground_operator_specs_template: dict
    runtime_profiling : bool

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

def run_one_trial(trial_row: Tuple[Any, ...],   # (scenario_id, num_sats, gnd_segment, task_arrival_rate, target_distribution)
                  cfg: RunConfig,
                  pbar_pos: int
                ) -> Dict:
    t0 = time.time()
    scenario_id, num_sats, gnd_segment, task_arrival_rate, target_distribution = trial_row

    # make gnd segment deterministic
    gnd_segment = 'None' if not isinstance(gnd_segment, str) else gnd_segment

    results_dir = os.path.join(cfg.base_path, 'results', f"{cfg.trial_filename}_scenario_{scenario_id}")
    results_summary_path = os.path.join(results_dir, 'summary.csv')

    # create the scenario directory if it doesnt exist already
    os.makedirs(results_dir, exist_ok=True)

    try:
        # generate mission specs
        mission_specs: dict = generate_scenario_mission_specs(
            cfg.mission_specs_template, cfg.duration, cfg.step_size,
            cfg.base_path, cfg.trial_filename, scenario_id,
            num_sats, gnd_segment, target_distribution,
            cfg.spacecraft_specs_template, cfg.instrument_specs,
            cfg.ground_operator_specs_template
        )

        if cfg.propagate_only:
            orbitdata_dir = OrbitData.precompute(mission_specs)
            return {
                "scenario_id": scenario_id,
                "status": "propagated_only",
                "orbitdata_dir": orbitdata_dir,
                "results_dir": results_dir,
                "elapsed_s": time.time() - t0,
            }

        # Verify results_dir exists
        assert os.path.isdir(results_dir), \
            f"Results directory not properly initialized at: {results_dir}"

        # define conditions to execute mission
        execute_conditions = [
            # there is no results directory generated yet
            not os.path.isdir(results_dir),
            
            # or results directory is empty
            len(os.listdir(results_dir)) == 0,

            any(
                (   # or one of the agents does not have a results directory
                    not os.path.isdir(os.path.join(results_dir, d)) or
                    # or the agent's results directory is incomplete
                    len(os.listdir(os.path.join(results_dir, d))) <= 2
                )
                for d in os.listdir(results_dir)
                if '.csv' not in d
            ),
            
            # or an overwrite flag was set
            cfg.overwrite
        ]

        # execute mission  if any of the conditions are met
        if any(execute_conditions):
            # create mission instance
            mission = Simulation.from_dict(
                mission_specs,
                overwrite=cfg.overwrite,
                printouts=False,
                level=cfg.level
            )

            # execute mission
            mission.execute(pbar_pos, pbar_leave=False)

            # set status
            status = "executed"
        else:
            # skip execution
            mission = None

            # set status
            status = "skipped_existing"

        # TODO : Re-enable result processing
        # # print results if it hasn't been performed yet or if results need to be reevaluated
        # if not os.path.isfile(results_summary_path) or cfg.reevaluate: 
        #     print(' - Printing simulation results...')
        #     if mission is None:
        #         # load mission to process results if not already loaded
        #         mission = Simulation.from_dict(
        #                 mission_specs,
        #                 overwrite=cfg.overwrite,
        #                 printouts=False,
        #                 level=cfg.level
        #             )
            
        #     # process results
        #     mission.process_results()

        # # ensure if summary file was properly generated at the end of the simulation
        # assert os.path.isfile(results_summary_path), \
        #     f"Results summary file not found at: {results_summary_path}"

        return {
            "scenario_id": scenario_id,
            "status": status,
            "results_dir": results_dir,
            "results_summary_path": results_summary_path,
            "elapsed_s": time.time() - t0,
        }

    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "results_dir": results_dir,
            "elapsed_s": time.time() - t0,
        }
    
def parallel_run_trials(trials_df : pd.DataFrame, cfg: RunConfig, max_workers: int = None):
    # Convert to plain tuples to avoid pandas pickling quirks in workers
    trial_rows = list(trials_df.itertuples(index=False, name=None))

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 3, len(trial_rows))

    results = []
    try:
        # run trials with progress bar
        # with tqdm(total=len(trial_rows), desc='Performing study', leave=True, mininterval=0.5, unit=' trial') as pbar:
        with tqdm(total=len(trial_rows),
                    desc="Performing study",
                    leave=True,
                    mininterval=0.5,
                    unit=" trial",
                    dynamic_ncols=True,
                    file=sys.stderr,
                    position=0
                ) as pbar:
                        
            # create process pool executor
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                # submit all trials for execution
                fut_map = {ex.submit(run_one_trial, row, cfg, (i % max_workers) + 1): row for i,row in enumerate(trial_rows)}

                # collect results as they complete
                for fut in as_completed(fut_map):
                    row = fut_map[fut]
                    res = fut.result()
                    results.append(res)

                    sid = res.get("scenario_id", "???")
                    status = res.get("status")
                    elapsed = res.get("elapsed_s", None)
                    
                    if status == "error":
                        tqdm.write(f"[scenario {sid}] ERROR after {elapsed:.1f}s: {res.get('error')}")
                    else:
                        tqdm.write(f"[scenario {sid}] {status} in {elapsed:.1f}s")

                    # update progress bar                
                    pbar.update(1)
    
    except KeyboardInterrupt as e:
        ex.shutdown(wait=False)
        raise e
    
    except Exception as e:
        ex.shutdown(wait=False)
        raise e

    finally:
        # ensure the executor is torn down before exiting scope
        ex.shutdown(wait=True)

    # Optional: sort by scenario_id
    results.sort(key=lambda d: d.get("scenario_id", 0))
    return results

def main_parallellized(trial_filename : str, 
                      lower_bound : int, 
                      upper_bound : int, 
                      level : int, 
                      propagate_only : bool,
                      overwrite : bool, 
                      reevaluate : bool, 
                      debug : bool,
                      runtime_profiling : bool):
    
    # print welcome
    print_scenario_banner(f'CBBA Stress Test Study (parallelized) - {trial_filename}')
    
    # TODO : Warn that runtime profiling is not supported in parallelized mode
    if runtime_profiling: print("WARNING: Runtime profiling is not supported in parallelized mode and will be ignored.")

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

    cfg = RunConfig(
        duration=duration,
        step_size=step_size,
        base_path=base_path,
        trial_filename=trial_filename,
        propagate_only=propagate_only,
        overwrite=overwrite,
        evaluate=reevaluate,
        level=level,
        mission_specs_template=mission_specs_template,
        spacecraft_specs_template=spacecraft_specs_template,
        instrument_specs=instrument_specs,
        ground_operator_specs_template=ground_operator_specs_template,
        runtime_profiling=runtime_profiling
    )

    # run trials in parallel
    run_results = parallel_run_trials(trials, cfg)

    # study done
    return run_results

def main(trial_filename : str, 
         lower_bound : int, 
         upper_bound : int, 
         level : int, 
         propagate_only : bool,
         overwrite : bool, 
         evaluate : bool, 
         debug : bool,
         runtime_profiling : bool):
    
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
    duration = 1500 / 3600 / 24.0 if debug else 1.0 # [days]
    duration = min(duration, 1.0)                   # cap at 1 day for sanity
    step_size = 10                                  # [s]

    # iterate through each trial
    for scenario_id,num_sats,gnd_segment,task_arrival_rate,target_distribution in trials.itertuples(index=False):
        
        # handle nan ground segment case
        gnd_segment = 'None' if not isinstance(gnd_segment, str) else gnd_segment
        
        # print scenario banner
        if scenario_id > 0: print_scenario_banner(f'CBBA Stress Test Study - {trial_filename}')
        if debug: tqdm.write("DEBUG MODE ENABLED: Running a single short experiment for debugging purposes")
        tqdm.write(f"\n--- Running Trial Scenario ID: {scenario_id} ---")
        tqdm.write(f" - Num Sats: {num_sats}")
        tqdm.write(f" - Ground Segment: {gnd_segment}")
        tqdm.write(f" - Task Arrival Rate: {task_arrival_rate} [tasks/day]")
        tqdm.write(f" - Target Distribution: Lat=(-{target_distribution}°, +{target_distribution}°)")
        tqdm.write(f" - Propagation Duration: {round(duration*24*3600,3)} [seconds] ({round(duration, 3)} [days])")        

        try:

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

            # check if runtime profiling toggle was selected
            if runtime_profiling:        
                # initialize profiler
                pr = cProfile.Profile()
                # enable profiler
                pr.enable()

            # check if propagation-only toggle was selected
            if propagate_only:
                # if selected; only precompute orbit data
                tqdm.write(" - Propagating orbit data only...")
                orbitdata_dir = OrbitData.precompute(mission_specs)
                tqdm.write(f" - Orbit data propagated and stored at: `{orbitdata_dir}`")
                
                # skip to next trial
                continue 

            # initialize simulation mission
            tqdm.write(" - Running full simulation...\n")

            # define conditions to execute mission
            if os.path.isdir(results_dir):
                # results directory was already generated
                execute_conditions = [            
                    # results directory is empty
                    len(os.listdir(results_dir)) == 0,
                    
                    # there are incomplete results directories for any agent
                    any([len(os.listdir(os.path.join(results_dir, d))) <= 2 
                            for d in os.listdir(results_dir)
                            if os.path.isdir(os.path.join(results_dir, d))
                            and 'manager' not in d]
                        ),
                    
                    # overwrite flag was set
                    overwrite
                ]
            else:
                # there is no results directory generated yet
                execute_conditions = [True]  # force execution if results directory does not exist

            # execute mission if any of the conditions are met
            if any(execute_conditions): 
                if execute_conditions[-1]:
                    tqdm.write(' - Overwrite flag detected; re-running simulation mission...')
                else:
                    tqdm.write(' - Incomplete or missing results detected; running simulation mission...')
                tqdm.write(' - Initializing simulation mission...')
                mission : Simulation = Simulation.from_dict(mission_specs, overwrite=overwrite, level=level)

                # check if output directory was properly initalized
                assert os.path.isdir(results_dir), \
                    f"Results directory not properly initialized at: {results_dir}"
                            
                tqdm.write(' - Executing simulation mission...')
                mission.execute()
            else:
                tqdm.write(' - Simulation data found! Skipping execution...')
                mission = None
            
            # # TODO : Re-enable result processing
            # # print results if it hasn't been performed yet or if results need to be evaluated
            # results_summary_path = os.path.join(results_dir, 'summary.csv')
            # if not os.path.isfile(results_summary_path) or evaluate: 
            #     if evaluate:
            #         tqdm.write(' - Evaluation flag detected; processing simulation results...')
            #     else:
            #         tqdm.write(' - Results summary not found; processing simulation results...')
            #     if mission is None:
            #         # load mission to process results if not already loaded
            #         tqdm.write(' - Initializing simulation mission...')
            #         mission = Simulation.from_dict(
            #                 mission_specs,
            #                 overwrite=overwrite,
            #                 printouts=False,
            #                 level=level
            #             )
                    
            #         # check if output directory was properly initalized
            #         assert os.path.isdir(results_dir), \
            #             f"Results directory not properly initialized at: {results_dir}"

            #     tqdm.write(' - Evaluating simulation results...')
            #     mission.process_results()

            # # ensure if summary file was properly generated at the end of the simulation
            # assert os.path.isfile(results_summary_path), \
            #     f"Results summary file not found at: {results_summary_path}"


            if runtime_profiling:
                # disable profiler
                pr.disable()
                # save to file
                tqdm.write(" - Printing runtime profiling results...")
                runtime_path = os.path.join(results_dir, "profile.out")
                pr.dump_stats(runtime_path)

                # ensure if summary file was properly generated at the end of the simulation
                assert os.path.isfile(runtime_path), \
                    f"Results summary file not found at: `{runtime_path}`"
                tqdm.write(f" - Profiling results saved to: `{runtime_path}`")
        
        except Exception as e:
            tqdm.write(f" - ERROR during scenario {scenario_id} execution: {repr(e)}")
            tqdm.write(traceback.format_exc())
            continue

    # study done
    return True

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
                        help='simulation results overwrite toggle',
                        action='store_true',
                        required=False)
    parser.add_argument('-e', 
                        '--evaluate',
                        help='results evaluation overwrite toggle',
                        action='store_true',
                        required=False)
    parser.add_argument('-r', 
                        '--runtime-profiling',
                        help='toggles to enable runtime profiling',
                        action='store_true',
                        required=False)
    parser.add_argument('-d', 
                        '--debug',
                        help='toggles to run just one experiment for debugging purposes',
                        action='store_true',
                        required=False)
    parser.add_argument('-s', 
                        '--single-threaded',
                        help='toggles to run simulations in single-threaded mode',
                        action='store_true',
                        required=False)
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
    trial_filename : str = args.trial_filename
    lower_bound : int = args.lower_bound
    upper_bound : int = args.upper_bound
    propagate_only : bool = args.propagate_only
    overwrite : bool = args.overwrite
    evaluate : bool = args.evaluate
    runtime_profiling : bool = args.runtime_profiling
    debug : bool = args.debug
    single_threaded : bool = args.single_threaded
    level : int = LEVELS.get(args.level)

    # run main study
    if debug or single_threaded or upper_bound - lower_bound <= 1:
        # if in debug mode, single-threaded mode, or only one trial is being run; 
        # run trials one at a time use non-parallelized version
        main(trial_filename, lower_bound, upper_bound, level, propagate_only, overwrite, evaluate, debug, runtime_profiling)
    else:
        # if more than one trial is being run; use parallelized version
        main_parallellized(trial_filename, lower_bound, upper_bound, level, propagate_only, overwrite, evaluate, debug, runtime_profiling)

    # print outro
    print('\n' + '='*54)
    print('STUDY COMPLETE!')
