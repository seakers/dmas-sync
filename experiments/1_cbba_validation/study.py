
import os
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from pyparsing import Dict

from dmas.utils.tools import LEVELS, print_scenario_banner

from utils.config import SimulationConfig, RunConfig, parse_study_args
from utils.factory import load_trials, load_templates, get_base_path
from utils.run import serial_run_trials, parallel_run_trials

# ------------------------------------------------------------------
# Main Study Script
# ------------------------------------------------------------------

def main_study(sim_cfg: SimulationConfig) -> List[Dict]:
    """
    Loads trials + templates, builds RunConfig, then runs either serial or parallel depending on sim_cfg.
    """
    # normalize and validate sim config
    sim_cfg = sim_cfg.normalize_and_validate()

    # print welcome (respect quiet)
    trial_stem = os.path.splitext(os.path.basename(sim_cfg.trials_file))[0]
    if not sim_cfg.quiet:
        print_scenario_banner("CBBA Stress Test Study - {0}".format(trial_stem))

    # base path for experiment
    base_path: str = get_base_path()

    # load trials (your loader expects filename stem; adapt if needed)
    # If load_trials expects the stem without ".csv", pass trial_stem:
    trials: pd.DataFrame = load_trials(base_path, trial_stem, sim_cfg.trial_start, sim_cfg.trial_end)

    if not sim_cfg.quiet:
        end_disp = sim_cfg.trial_end if sim_cfg.trial_end is not None else "end"
        print(" - Loaded {0} trials from `{1}`: [{2}:{3})".format(len(trials), sim_cfg.trials_file, sim_cfg.trial_start, end_disp))

    # load templates
    mission_specs_template, ground_operator_specs_template, \
        spacecraft_specs_template, instrument_specs, planner_specs = load_templates(base_path)
    if not sim_cfg.quiet:
        print(" - Loaded experiment templates from `resources/templates/`")

    # duration/step size
    duration = 20_000 / 3_600 / 24.0 if sim_cfg.reduced else 1.0  # [days]
    duration = min(duration, 1.0)
    step_size = 10  # [s]

    # Build the per-trial config (NO batch toggles here anymore)
    run_cfg = RunConfig(
        duration=duration,
        step_size=step_size,
        base_path=base_path,
        mission_specs_template=mission_specs_template,
        spacecraft_specs_template=spacecraft_specs_template,
        instrument_specs=instrument_specs,
        ground_operator_specs_template=ground_operator_specs_template,
        planner_specs=planner_specs,
    )

    # Choose runner
    if sim_cfg.single_thread:
        return serial_run_trials(trials, run_cfg, sim_cfg)
    else:
        # raise NotImplementedError("Parallel execution is not tested yet in this version.")
        return parallel_run_trials(trials, run_cfg, sim_cfg)

if __name__ == "__main__":
    
    # parse args
    config = parse_study_args()

    # run study
    main_study(config)
    
    # print outro
    print('\n' + '='*54)
    print('STUDY COMPLETE!')