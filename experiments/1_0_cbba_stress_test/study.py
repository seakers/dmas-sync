
import os
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from pyparsing import Dict

from dmas.utils.tools import LEVELS, print_scenario_banner

from config import SimulationConfig, RunConfig, parse_args
from run import serial_run_trials, parallel_run_trials

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
    mission_specs_template, ground_operator_specs_template, spacecraft_specs_template, instrument_specs = load_templates(base_path)
    if not sim_cfg.quiet:
        print(" - Loaded experiment templates from `resources/templates/`")

    # duration/step size
    duration = 10000 / 3600 / 24.0 if sim_cfg.reduced else 1.0  # [days]
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
    )

    # Choose runner
    if sim_cfg.single_thread:
        return serial_run_trials(trials, run_cfg, sim_cfg)
    else:
        raise NotImplementedError("Parallel execution is not tested yet in this version.")
        return parallel_run_trials(trials, run_cfg, sim_cfg)

if __name__ == "__main__":
    
    # parse args
    config = parse_args()

    # run study
    main_study(config)
    
    # print outro
    print('\n' + '='*54)
    print('STUDY COMPLETE!')