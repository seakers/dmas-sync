import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import sys
import tracemalloc

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import RunConfig, SimulationConfig
import os
import logging
import time
import traceback
from typing import Any, List, Tuple, Dict, Optional

from dmas.core.simulation import Simulation
from dmas.utils.constellations import Constellation, WalkerDeltaConstellation
from dmas.utils.orbitdata import OrbitData
from dmas.utils.tools import print_scenario_banner

# ------------------------------------------------------------------
# Run setup helper functions
# ------------------------------------------------------------------

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

def create_propagator_settings_specifications(base_path : str, 
                                              scenario_id : int, 
                                              num_sats : float, 
                                              gnd_segment : str, 
                                              target_distribution) -> dict:
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
            "saveUnprocessedCoverage" : "False"
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

# ------------------------------------------------------------------
# Since-Scenario Run Functions
# ------------------------------------------------------------------

def _scenario_results_dir(cfg : RunConfig, trial_stem: str, scenario_id: int) -> str:
    # You can change naming here if you want it to be independent of the CSV filename.
    return os.path.join(cfg.base_path, "results", f"{trial_stem}_scenario_{scenario_id}")


def _is_simulation_complete(results_dir: str) -> bool:
    """
    Lightweight check: does this scenario look 'done'?
    Keep your existing heuristics, but isolate them in one function.
    """
    # check if the results directory exists
    if not os.path.isdir(results_dir):
        return False

    # check if results directory is empty
    entries = os.listdir(results_dir)
    if len(entries) == 0:
        return False

    # check if each agent directory has more than 2 files (to account for .gitignore)
    for d in entries:
        agent_dir = os.path.join(results_dir, d)
        if not os.path.isdir(agent_dir):
            continue
        if len(os.listdir(agent_dir)) <= 2:
            return False

    # if all checks passed, consider simulation complete
    return True


def _summary_path(results_dir: str) -> str:
    """ Get the path to the summary CSV file for the scenario. """
    return os.path.join(results_dir, "summary.csv")


def _map_log_level(level_str: str) -> int:
    # If Simulation expects an int logging level:
    return getattr(logging, level_str.upper(), logging.INFO)


def run_one_trial(trial_row: Tuple[Any, ...],   # (scenario_id, num_sats, gnd_segment, task_arrival_rate, target_distribution)
                  run_cfg: RunConfig,           # RunConfig
                  sim_cfg: SimulationConfig,    # SimulationConfig
                  pbar_pos: int
                ) -> Dict:
    """ Run one trial scenario based on the provided configuration and trial parameters. """
    # check if profiling is requested; set up profilers if needed
    if sim_cfg.profile_cpu:
        # initialize profiler
        pr = cProfile.Profile()
        # enable profiler
        pr.enable()

    if sim_cfg.profile_mem:
        # initialize memory profiler
        tracemalloc.start()
    
    # Start timer
    t0 = time.time()

    # Unpack trial row
    scenario_id, num_sats, gnd_segment, task_arrival_rate, target_distribution = trial_row

    # make gnd segment deterministic
    gnd_segment = "None" if not isinstance(gnd_segment, str) else gnd_segment

    # A stable name for this CSV, used in folder naming
    trial_stem = os.path.splitext(os.path.basename(sim_cfg.trials_file))[0]

    results_dir = _scenario_results_dir(run_cfg, trial_stem, scenario_id)
    results_summary_path = _summary_path(results_dir)

    # ensure scenario dir exists early (safe even if we skip)
    os.makedirs(results_dir, exist_ok=True)

    # quiet mode: no console spam, no progress bars
    printouts = not sim_cfg.quiet
    pbar_leave = False
    log_level_int = _map_log_level(sim_cfg.log_level)

    try:
        # ------------------------------------------------------------
        # Stage 0: Build mission specs (always needed)
        # ------------------------------------------------------------
        mission_specs: dict = generate_scenario_mission_specs(
            run_cfg.mission_specs_template, run_cfg.duration, run_cfg.step_size,
            run_cfg.base_path, trial_stem, scenario_id,
            num_sats, gnd_segment, target_distribution,
            run_cfg.spacecraft_specs_template, run_cfg.instrument_specs,
            run_cfg.ground_operator_specs_template
        )

        # ------------------------------------------------------------
        # Stage 1: Precompute (controlled by force/only)
        # ------------------------------------------------------------
        orbitdata_dir: Optional[str] = None
        if sim_cfg.only_precompute or sim_cfg.force_precompute:
            orbitdata_dir = OrbitData.precompute(mission_specs, printouts=printouts)
            if sim_cfg.only_precompute:
                return {
                    "scenario_id": scenario_id,
                    "status": "precomputed_only",
                    "orbitdata_dir": orbitdata_dir,
                    "results_dir": results_dir,
                    "elapsed_s": time.time() - t0,
                }

        # ------------------------------------------------------------
        # Stage 2: Simulate / propagate (controlled by force/only + cache)
        # ------------------------------------------------------------
        already_done = _is_simulation_complete(results_dir)
        should_run_sim = sim_cfg.force_simulate or (not already_done)

        if sim_cfg.only_simulate:
            # In only_simulate mode, we run simulation even if cached exists?
            # Usually yes (debug mode should do what you asked), but you can flip this.
            should_run_sim = True

        mission = None
        if should_run_sim:
            mission = Simulation.from_dict(
                mission_specs,
                overwrite=sim_cfg.force_simulate,   # you may want overwrite when forcing sim
                printouts=printouts,
                level=log_level_int,
            )
            mission.execute(pbar_pos, pbar_leave=pbar_leave if not sim_cfg.quiet else False)
            sim_status = "executed"
        else:
            sim_status = "skipped_existing"

        if sim_cfg.only_simulate:
            return {
                "scenario_id": scenario_id,
                "status": "simulated_only" if should_run_sim else "simulated_only_skipped_existing",
                "results_dir": results_dir,
                "elapsed_s": time.time() - t0,
            }

        # ------------------------------------------------------------
        # Stage 3: Postprocess (controlled by force/only + cache)
        # ------------------------------------------------------------
        # TODO enable when ready
        post_status = "postprocess_skipped_not_implemented"
        # # Decide whether postprocess should run
        # summary_exists = os.path.isfile(results_summary_path)
        # should_postprocess = sim_cfg.force_postprocess or (not summary_exists)

        # if sim_cfg.only_postprocess:
        #     # In only_postprocess mode, do it even if exists? usually yes.
        #     should_postprocess = True

        # if should_postprocess:
        #     if mission is None:
        #         # Load mission if needed for processing
        #         mission = Simulation.from_dict(
        #             mission_specs,
        #             overwrite=False,
        #             printouts=printouts,
        #             level=log_level_int
        #         )
        #     # Your code currently has this disabled; enable when ready:
        #     # mission.process_results()
        #     post_status = "postprocess_ran"  # change to "processed" once enabled
        # else:
        #     post_status = "postprocess_skipped_existing"

        # if sim_cfg.only_postprocess:
        #     return {
        #         "scenario_id": scenario_id,
        #         "status": post_status,
        #         "results_dir": results_dir,
        #         "results_summary_path": results_summary_path,
        #         "elapsed_s": time.time() - t0,
        #     }

        # ------------------------------------------------------------
        # Return summary
        # ------------------------------------------------------------
        return {
            "scenario_id": scenario_id,
            "status": sim_status,
            "postprocess_status": post_status,
            "results_dir": results_dir,
            "results_summary_path": results_summary_path,
            "elapsed_s": time.time() - t0,
        }

    except Exception as e:
        if sim_cfg.reduced or sim_cfg.exceptions:
            traceback.print_exc()  # print full traceback for debugging
            raise e  # re-raise for debugging 

        return {
            "scenario_id": scenario_id,
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "results_dir": results_dir,
            "elapsed_s": time.time() - t0,
        }
    
    finally:
        # check if profiling is requested; close it properly
        if sim_cfg.profile_cpu:
            # disable profiler
            pr.disable()
            # save to file
            if not sim_cfg.quiet:
                tqdm.write("================= RUNTIME PROFILING ===================")
                tqdm.write(" - Printing runtime profiling results...")
            runtime_path = os.path.join(results_dir, "runtime.out")
            pr.dump_stats(runtime_path)

            # ensure if summary file was properly generated at the end of the simulation
            assert os.path.isfile(runtime_path), \
                f"Results summary file not found at: `{runtime_path}`"
            if not sim_cfg.quiet:
                tqdm.write(f" - Runtime profiling results saved to: `{runtime_path}`")

        if sim_cfg.profile_mem:
            # capture memory profiling results            
            current, peak = tracemalloc.get_traced_memory()
            # get performance snaptshot
            snapshot = tracemalloc.take_snapshot()
            
            # print to console
            if not sim_cfg.quiet:
                tqdm.write("================= MEMORY ALLOCATION ===================")                
                tqdm.write(f"- Current memory usage is {current / 10**6:.2f}MB; Peak was {peak / 10**6:.2f}MB")
                tqdm.write(" - Traceback Limit : " + str(tracemalloc.get_traceback_limit()) + " Frames")
                tqdm.write(" - Traced Memory (Current, Peak): " + str(tracemalloc.get_traced_memory()))
                tqdm.write(" - Memory Usage by tracemalloc Module : " + str(tracemalloc.get_tracemalloc_memory()) + " bytes")
                tqdm.write(" - Tracing Status : " + str(tracemalloc.is_tracing()))

            # save to file
            memory_path = os.path.join(results_dir, "memory.txt")
            with open(memory_path, 'w') as mem_file:
                mem_file.write("================ OVERALL MEMORY USAGE ================")
                mem_file.write(f"\nCurrent memory usage is {current / 10**6:.2f}MB; Peak was {peak / 10**6:.2f}MB")
                mem_file.write("\n----------------------------------")
                mem_file.write(f"\n - Traceback Limit : {tracemalloc.get_traceback_limit()} Frames")
                mem_file.write(f"\n - Traced Memory (Current, Peak): {tracemalloc.get_traced_memory()}")
                mem_file.write(f"\n - Memory Usage by tracemalloc Module : {tracemalloc.get_tracemalloc_memory()} bytes")
                mem_file.write(f"\n - Tracing Status : {tracemalloc.is_tracing()}")
                                
                mem_file.write("\n================= MEMORY SNAPSHOT ====================\n")
                for stat in snapshot.statistics("lineno"):
                    mem_file.write(f"{stat}\n")
                    mem_file.write(f" - {stat.traceback.format()}\n\n")

            # ensure if summary file was properly generated at the end of the simulation
            assert os.path.isfile(memory_path), \
                f"Memory profiling file not found at: `{memory_path}`"

            if not sim_cfg.quiet:
                tqdm.write(f" - Memory profiling results saved to: `{memory_path}`")
            
            # stop memory profiler
            tracemalloc.stop() 

        if not sim_cfg.quiet and (sim_cfg.profile_cpu or sim_cfg.profile_mem):
            tqdm.write("======================================================")

# ------------------------------------------------------------------
# Run Trials Functions
# ------------------------------------------------------------------

def _tqdm_enabled(sim_cfg: SimulationConfig) -> bool:
    # quiet disables bars; also disable if not attached to TTY and on batch
    return not sim_cfg.quiet


def serial_run_trials(trials_df: pd.DataFrame, run_cfg: RunConfig, sim_cfg: SimulationConfig) -> List[Dict]:
    """Run trials one-by-one in the current process."""
    # get list of trial rows
    trial_rows = list(trials_df.itertuples(index=False, name=None))

    # initialize results list
    results: List[Dict] = []

    # initialize progress bar if enabled
    pbar = None
    if _tqdm_enabled(sim_cfg):
        pbar = tqdm(
            total=len(trial_rows),
            desc="Performing study (serial)",
            leave=True,
            mininterval=0.5,
            unit=" trial",
            dynamic_ncols=True,
            file=sys.stderr,
            position=0,
        )

    # run trials serially
    try:
        # print header for serial execution
        print_scenario_banner("CBBA Stress Test Study - Serial Execution")

        # iterate over trial rows
        for i, row in enumerate(trial_rows):
            if not sim_cfg.quiet:
                print(f"\n=== Running trial {i+1}/{len(trial_rows)}: scenario_id={row[0]}, num_sats={row[1]}, gnd_segment={row[2]}, task_arrival_rate={row[3]}, target_distribution={row[4]} ===")

            # run one trial
            res = run_one_trial(row, run_cfg, sim_cfg, pbar_pos=1)
            
            # store result
            results.append(res)

            # update progress bar if used
            if pbar is not None:
                pbar.update(1)

                sid = res.get("scenario_id", "???")
                status = res.get("status")
                elapsed = res.get("elapsed_s", None)
                
                # check if error was encountered
                if status == "error":
                    # log error message
                    pbar.write(f"[scenario {sid}] ERROR after {elapsed:.1f}s: {res.get('error')}")
                else:
                    # log normal status message
                    pbar.write(f"[scenario {sid}] {status} in {elapsed:.1f}s")
            else:
                # print to console if no progress bar (still respect quiet mode)
                sid = res.get("scenario_id", "???")
                status = res.get("status")
                elapsed = res.get("elapsed_s", None)

                if status == "error":
                    # log error message
                    print(f"[scenario {sid}] ERROR after {elapsed:.1f}s: {res.get('error')}")
                else:
                    # log normal status message
                    print(f"[scenario {sid}] {status} in {elapsed:.1f}s")

    finally:
        # close progress bar if used
        if pbar is not None:
            pbar.close()

    results.sort(key=lambda d: d.get("scenario_id", 0))
    return results


def parallel_run_trials(trials_df: pd.DataFrame, run_cfg: RunConfig, sim_cfg: SimulationConfig) -> List[Dict]:
    """Run trials in parallel using ProcessPoolExecutor."""

    # get list of trial rows
    trial_rows = list(trials_df.itertuples(index=False, name=None))
    
    # check if there are no trials to run; if so, return empty list
    if len(trial_rows) == 0: return []

    # cetermine max_workers
    if sim_cfg.max_workers is not None:
        # use specified max_workers
        max_workers = max(1, int(sim_cfg.max_workers))
    else:
        # conservative default; cap at 3 to avoid overloading typical systems
        max_workers = min(os.cpu_count() or 1, 3, len(trial_rows))

    # if single_thread requested, fall back to serial
    if sim_cfg.single_thread or max_workers <= 1:
        return serial_run_trials(trials_df, run_cfg, sim_cfg)

    # initialize results list
    results: List[Dict] = []

    # initialize progress bar if enabled
    pbar = None
    if _tqdm_enabled(sim_cfg):
        pbar = tqdm(
            total=len(trial_rows),
            desc="Performing study (parallel)",
            leave=True,
            mininterval=0.5,
            unit=" trial",
            dynamic_ncols=True,
            file=sys.stderr,
            position=0,
        )

    # run trials in parallel
    try:
        # print header for serial execution
        print_scenario_banner("CBBA Stress Test Study - Parallel Execution")

        # initialize process pool
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            
            # submit all trials to executor
            fut_map = {
                ex.submit(run_one_trial, row, run_cfg, sim_cfg, (i % max_workers) + 1): row
                # If you update run_one_trial signature to include sim_cfg:
                # ex.submit(run_one_trial, row, run_cfg, sim_cfg, (i % max_workers) + 1): row
                for i, row in enumerate(trial_rows)
            }

            # process results as they complete
            for fut in as_completed(fut_map):
                # get result
                res = fut.result()
                
                # store result
                results.append(res)

                # update progress bar if used
                if pbar is not None:
                    pbar.update(1)
                    
                    sid = res.get("scenario_id", "???")
                    status = res.get("status")
                    elapsed = res.get("elapsed_s", None)

                    # check if error was encountered
                    if status == "error":
                        # log error message
                        pbar.write(f"[scenario {sid}] ERROR after {elapsed:.1f}s: {res.get('error')}")
                    else:
                        # log normal status message
                        pbar.write(f"[scenario {sid}] {status} in {elapsed:.1f}s")
                else:
                    # print to console if no progress bar (still respect quiet mode)
                    sid = res.get("scenario_id", "???")
                    status = res.get("status")
                    elapsed = res.get("elapsed_s", None)

                    if status == "error":
                        # log error message
                        print(f"[scenario {sid}] ERROR after {elapsed:.1f}s: {res.get('error')}")
                    else:
                        # log normal status message
                        print(f"[scenario {sid}] {status} in {elapsed:.1f}s")

    finally:
        if pbar is not None:
            pbar.close()

    results.sort(key=lambda d: d.get("scenario_id", 0))
    return results
