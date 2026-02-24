import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import random
import sys
import tracemalloc

import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import RunConfig, SimulationConfig
import os
import logging
import time
import traceback
from typing import Any, List, Tuple, Dict, Optional

from dmas.core.simulation import Simulation
from dmas.utils.constellations import Constellation, WalkerDeltaConstellation
from dmas.utils.orbitdata import OrbitData
from dmas.utils.tools import print_scenario_banner
from utils.factory import generate_scenario_mission_specs

# ------------------------------------------------------------------
# Since-Scenario Run Functions
# ------------------------------------------------------------------

def _trial_results_dir(cfg : RunConfig, trial_stem: str, trial_id: int) -> str:
    # You can change naming here if you want it to be independent of the CSV filename.
    return os.path.join(cfg.base_path, "results", f"{trial_stem}_trial-{trial_id}")


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
    # Trial ID,Preplanner,Replanner,Num Sats,Latency,Task Arrival Rate,Target Distribution,Scenario,in_stress,in_connectivity,in_validation
    trial_id, preplanner, replanner, num_sats, latency, task_arrival_rate, target_distribution, scenario_idx, *_ = trial_row
    
    # normalize `nan` values to None
    preplanner = "none" if not isinstance(preplanner, str) and pd.isna(preplanner) else preplanner.lower()
    replanner = "none" if not isinstance(replanner, str) and pd.isna(replanner) else replanner.lower()

    # A stable name for this CSV, used in folder naming
    trial_stem = os.path.splitext(os.path.basename(sim_cfg.trials_file))[0]

    results_dir = _trial_results_dir(run_cfg, trial_stem, trial_id)
    if sim_cfg.reduced: results_dir = results_dir + "_reduced"
    results_summary_path = os.path.join(results_dir, "summary.csv")

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
        # generate scenario specifications from templates and trial parameters
        if printouts: tqdm.write(f" - Generated mission specifications for scenario {trial_id}")
        mission_specs : dict = generate_scenario_mission_specs(
            run_cfg.mission_specs_template, run_cfg.duration, run_cfg.step_size,
            run_cfg.base_path, trial_stem, trial_id,
            preplanner, replanner, num_sats, latency, task_arrival_rate, 
            target_distribution, scenario_idx, 
            run_cfg.spacecraft_specs_template, run_cfg.instrument_specs,
            run_cfg.planner_specs, run_cfg.ground_operator_specs_template, 
            sim_cfg.reduced,
        )

        # random wait for staggering
        if not sim_cfg.reduced:
            t_wait = random.uniform(0, 1) * int(trial_id)
            if printouts: tqdm.write(f" - Staggering start time with random wait or {t_wait:.2f} [s]...")
            time.sleep(t_wait)

        # ------------------------------------------------------------
        # Stage 1: Precompute (controlled by force/only)
        # ------------------------------------------------------------
        orbitdata_dir: Optional[str] = None
        if sim_cfg.only_precompute or sim_cfg.force_precompute:
            orbitdata_dir = OrbitData.precompute(mission_specs, printouts=printouts)
            if sim_cfg.only_precompute:
                return {
                    "scenario_id": trial_id,
                    "status": "precomputed_only",
                    "orbitdata_dir": orbitdata_dir,
                    "results_dir": results_dir,
                    "elapsed_s": time.time() - t0,
                }

        # ------------------------------------------------------------
        # Stage 2: Simulate / propagate (controlled by force/only + cache)
        # ------------------------------------------------------------
        already_done = _is_simulation_complete(results_dir)
        should_run_sim = sim_cfg.force_simulate or (not already_done) or sim_cfg.only_simulate

        mission = None
        if should_run_sim:
            mission = Simulation.from_dict(
                mission_specs,
                overwrite=sim_cfg.force_simulate,
                printouts=printouts,
                level=log_level_int,
            )
            mission.execute(pbar_pos, pbar_leave=pbar_leave)
            sim_status = "executed"
        else:
            sim_status = "skipped_existing"

        if sim_cfg.only_simulate:
            return {
                "scenario_id": trial_id,
                "status": "simulated_only" if should_run_sim else "simulated_only_skipped_existing",
                "results_dir": results_dir,
                "elapsed_s": time.time() - t0,
            }

        # ------------------------------------------------------------
        # Stage 3: Postprocess (controlled by force/only + cache)
        # ------------------------------------------------------------
        # Decide whether postprocess should run
        required_processed_files = ['grid_data.parquet', 'events_detected.parquet', 'events_requested.parquet', 'known_tasks.parquet',
                                    'accesses_per_event.parquet', 'accesses_per_task.parquet', 'observations_per_event.parquet', 
                                    'observations_per_task.parquet', 'planned_rewards.parquet', 'execution_costs.parquet']        
        processing_exists = all(os.path.isfile(os.path.join(results_dir, f)) for f in required_processed_files)
        should_postprocess = sim_cfg.force_postprocess or (not processing_exists) or sim_cfg.only_postprocess or should_run_sim

        if should_postprocess:
            if mission is None:
                # Load mission if needed for processing
                mission = Simulation.from_dict(
                    mission_specs,
                    overwrite=False,
                    printouts=printouts,
                    level=log_level_int
                )
            # Your code currently has this disabled; enable when ready:
            mission.process_results(force_process=sim_cfg.force_postprocess, printouts=not sim_cfg.quiet)
            post_status = "postprocessed" 
        else:
            post_status = "postprocessed_skipped_existing"

        if sim_cfg.only_postprocess:
            return {
                "scenario_id": trial_id,
                "status": post_status,
                "results_dir": results_dir,
                "results_summary_path": results_summary_path,
                "elapsed_s": time.time() - t0,
            }
        
        # ------------------------------------------------------------
        # Stage 4: Summarize (controlled by force/only + cache)
        # ------------------------------------------------------------
        summary_exists = os.path.isfile(results_summary_path)
        should_summarize = sim_cfg.force_summarize or (not summary_exists) or should_run_sim

        if should_summarize:
            if mission is None:
                # Load mission if needed for processing
                mission = Simulation.from_dict(
                    mission_specs,
                    overwrite=False,
                    printouts=printouts,
                    level=log_level_int
                )
            mission.summarize_results(force_summarize=sim_cfg.force_summarize, printouts=not sim_cfg.quiet)
            sum_status = "summarized"
        else:
            sum_status = "summarized_skipped_existing"

        if sim_cfg.only_summarize:
            return {
                "scenario_id": trial_id,
                "status": sum_status,
                "results_dir": results_dir,
                "results_summary_path": results_summary_path,
                "elapsed_s": time.time() - t0,
            }

        # ------------------------------------------------------------
        # Return summary
        # ------------------------------------------------------------
        return {
            "scenario_id": trial_id,
            "status": sim_status,
            "postprocess_status": post_status,
            "summarize_status": sum_status,
            "results_dir": results_dir,
            "results_summary_path": results_summary_path,
            "elapsed_s": time.time() - t0,
        }

    except Exception as e:
        if sim_cfg.reduced or sim_cfg.exceptions:
            traceback.print_exc()  # print full traceback for debugging
            raise e  # re-raise to terminate trials

        return {
            "scenario_id": trial_id,
            "status": "error",
            "error": repr(e),
            # "traceback": traceback.format_exc(),
            "traceback": traceback.print_exc(),
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
