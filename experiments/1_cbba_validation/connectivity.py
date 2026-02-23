import json
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from dmas.utils.orbitdata import OrbitData
from utils.factory import get_base_path, load_templates, generate_scenario_mission_specs

from dmas.utils.tools import print_scenario_banner

def generate_high_latency_connectivity_spec(imaging_sats : list, relay_sats : list, ground_stations : list) -> dict:
    # High Latency Scenario: ISLs off, ISL to TDRSS off, SAT to NEN comms on
    return {
        "default": "deny",

        "groups": {
            "instruments": imaging_sats,
            "relays": relay_sats,
            "groundStations": ground_stations
        },

        "rules": [
            {
            "action": "allow",
            "scope": "between",
            "targets": ["instruments", "groundStations"]
            },
            {
            "action": "deny",
            "scope": "within",
            "targets": "instruments"
            },
            {
            "action": "deny",
            "scope": "within",
            "targets": "relays"
            }
        ],

        "overrides": []
    }

def generate_low_latency_connectivity_spec(imaging_sats : list, relay_sats : list, ground_stations : list) -> dict:
    # Low Latency Scenario: ISLs on, TDRSS comms on, SAT to NEN comms on
    return {
        "default": "deny",

        "groups": {
            "instruments": imaging_sats,
            "relays": relay_sats,
            "groundStations": ground_stations
        },

        "rules": [
            {
            "action": "allow",
            "scope": "between",
            "targets": ["instruments", "relays", "groundStations"]
            },
            {
            "action": "allow",
            "scope": "within",
            "targets": "instruments"
            },
            {
            "action": "allow",
            "scope": "within",
            "targets": "relays"
            }
        ],

        "overrides": []
    }

def generate_message_samples(mission_specs : dict, 
                             num_samples : int = 20_000,
                             reduced : bool = False
                            ) -> List[Tuple[str, str, float]]:
    # initiate samples list
    samples = []

    # reduce number of samples for testing purposes if reduced flag is set
    num_samples *= 0.1 if reduced else 1.0
    num_samples = int(num_samples)

    # unpack mission specs 
    duration = mission_specs['duration'] * 24 * 3600  # convert from days to seconds
    agent_names = [agent['name'] for agent in mission_specs['spacecraft']] + [gs['name'] for gs in mission_specs['groundOperator']]

    # generate message samples with random sender, receiver, and start time
    for _ in range(num_samples):
        # choose a random receiver and sender from the list of agents
        sender, receiver = random.sample(agent_names, 2)

        # random start time within mission duration
        start_time = duration * random.random()

        # add to samples list
        samples.append((sender,receiver,start_time))

    # return list of message samples sorted by start time
    return sorted(samples, key=lambda x: x[2])

def calculate_message_latency(sender : str, 
                              receiver : str, 
                              start_time : float, 
                              scenario_orbitdata : Dict[str, OrbitData]
                            ) -> float:
    
    # get the sender's orbitdata
    orbitdata : OrbitData = scenario_orbitdata[sender]

    # keep track of agents that have already been considered in previous intervals to avoid double counting
    agents_considered = set()
    
    # initialize broadcast time to be scheduled 
    t_broadcast = np.inf

    # get column index of this agent in the comms links table
    u_column_idx = orbitdata.comms_target_indices[sender]

    # iterate through list of intervals in this time period 
    for t_start,t_end, *component_indices in orbitdata.comms_links.iter_rows_raw(t=start_time, 
                                                                                 include_current=True):
        
        # get component index of this agent during this interval
        u_component_idx = int(component_indices[u_column_idx])
        
        # find all matching agents with the same component index and add to output list
        targets = set()
        for v_column_idx,v_component_idx in enumerate(component_indices):
            if v_column_idx != u_column_idx and v_component_idx == u_component_idx: 
                # get target agent name from column index
                target_agent = orbitdata.comms_target_columns[v_column_idx]

                # add to set of targets for this interval
                targets.add(target_agent)

        # skip if target agents have already been considered
        if targets <= agents_considered:
            continue
        
        # mark target agents as considered
        agents_considered.update(targets)

        # check if target receiver is in this set of targets
        if receiver in agents_considered:
            # receiver is in this interval; 
            #   schedule broadcast at the start of this interval, or at the message start time if it's later  
            t_broadcast = max(t_start, start_time)  

            # ensure that the message start time is before the end of the interval of communication links
            assert start_time <= t_end, \
                "Message start time is not within the interval of communication links"

            # stop search
            break

    # return latency as time from message start to scheduled broadcast time
    return t_broadcast - start_time 

def evaluate_scenario_latency(mission_specs : dict,
                              messages : List[Tuple[str, str, float]],
                              T_req : float = np.Inf,
                              printouts : bool = True) -> float:
    # precompute coverage data for scenario mission specs
    orbitdata_dir = OrbitData.precompute(mission_specs, printouts=printouts)
    
    # load precomputed coverage data with relevant mission specs 
    scenario_orbitdata : dict = OrbitData.from_directory(orbitdata_dir, mission_specs, printouts=printouts)

    # intiate list of message latencies
    latencies = np.zeros(len(messages))

    # calculate latency for each message 
    for i, (sender, receiver, start_time) in tqdm(enumerate(messages),
                                                  desc = "Evaluating Message Latencies",
                                                  unit=' msgs',
                                                  disable=not printouts):
        latencies[i] = calculate_message_latency(sender, receiver, start_time, scenario_orbitdata)

    # calculate probability of successful communcations
    p_success = np.sum(np.isfinite(latencies)) / len(latencies)

    # check latency requirement
    if np.isinf(T_req):
        # if infinte, set on time delivery rate to success rate
        p_on_time = p_success
    else:
        # else, calculate on time delivery rate based on latency requirement
        p_on_time = np.sum(latencies <= T_req) / len(latencies)

    # return latency statistics (success rate, mean, median, 95th percentile, 99th percentile)
    return p_success, p_on_time, np.mean(latencies), np.percentile(latencies, 50), np.percentile(latencies, 95), np.percentile(latencies, 99)

if __name__ == "__main__":
    """
    Creates `connectivity.json` for the connectivity test scenarios
    """
    # set reduced duration flag for testing purposes;
    #  (set to False to generate full duration scenarios)
    reduced = True

    # load trial csv
    trials_filename = "full_factorial_trials_2026-02-22.csv"
    trials_path = os.path.join(".", 'experiments','1_cbba_validation', 'resources', "trials", trials_filename)
    trials_df = pd.read_csv(trials_path)

    # print welcome
    trial_stem = os.path.splitext(os.path.basename(trials_filename))[0]
    print_scenario_banner("CBBA Stress Test Study - {0}".format(trial_stem))

    # base path for experiment
    base_path: str = get_base_path()

    # load templates
    mission_specs_template, ground_operator_specs_template, \
        spacecraft_specs_template, instrument_specs, planner_specs \
            = load_templates(base_path)    

    # duration/step size
    duration = 1_000 / 3600 / 24.0 if reduced else 1.0  # [days]
    duration = min(duration, 1.0)
    step_size = 10  # [s]

    # calculate average latency for all trials
    for _, row in trials_df.iterrows():
        # define output dir, ensure it exists
        output_dir = os.path.join(base_path, 'resources', 'connectivity')
        os.makedirs(output_dir, exist_ok=True)

        # unpack row parameters
        trial_id = row['Trial ID']
        preplanner = row['Preplanner']
        replanner = row['Replanner'] 
        num_sats = row['Num Sats']
        latency = row['Latency'].lower()
        task_arrival_rate = row['Task Arrival Rate']
        target_distribution = row['Target Distribution']
        scenario_idx = row['Scenario']

        # define output file paths for connectivity spec and latency results
        low_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-low.json")
        medium_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-medium.json")
        high_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-high.json")

        # check if trial has already been evaluated
        if os.path.exists(low_latency_filename) and os.path.exists(medium_latency_filename) and os.path.exists(high_latency_filename):
            print(f"Latency for trials with num_sats={num_sats} already evaluated; skipping.")
            continue  # if so; skip to next trial
        else:
            print(f"Evaluating Latency for trials with num_sats={num_sats}...")

        # normalize `nan` values to None
        preplanner = "none" if not isinstance(preplanner, str) and pd.isna(preplanner) else preplanner.lower()
        replanner = "none" if not isinstance(replanner, str) and pd.isna(replanner) else replanner.lower()

        # construct scenario mission specs
        generic_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, latency, 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)

        # define connectivity groups: instruments, relays, groundStations
        imaging_sats = [sat['name'] for sat in generic_mission_specs['spacecraft'] if 'instrument' in sat ]
        relay_sats = [sat['name'] for sat in generic_mission_specs['spacecraft'] if 'instrument' not in sat ]
        ground_stations = [gs['name'] for gs in generic_mission_specs['groundOperator'] ]

        # generate a list of message samples for the scenario to be used in latency evaluation
        messages = generate_message_samples(generic_mission_specs, reduced=reduced)

        # generate connectivity rules for latency scenarios (low, medium, high) 
        # 1) High Latency Scenario: ISLs off, ISL to TDRSS off, SAT to NEN comms on
        high_latency_connectivity_spec = generate_high_latency_connectivity_spec(imaging_sats, relay_sats, ground_stations)
        
        # save high latency scenario connectivity spec to file
        with open(high_latency_filename, 'w') as f:
            json.dump(high_latency_connectivity_spec, f, indent=4)

        # create mission specs for high latency scenario 
        high_latency_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, 'high', 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)
        
        # evaluate scenario latency
        high_latency_values = evaluate_scenario_latency(high_latency_mission_specs, messages)
        
        # 2) Low Latency Scenario: ISLs on, TDRSS comms on, SAT to NEN comms on
        low_latency_connectivity_spec = generate_low_latency_connectivity_spec(imaging_sats, relay_sats, ground_stations)

        # save low latency scenario connectivity spec to file
        with open(low_latency_filename, 'w') as f:
            json.dump(low_latency_connectivity_spec, f, indent=4)

        # create mission specs for low latency scenario 
        low_latency_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, 'low', 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)
        
        # evaluate scenario latency
        low_latency_values = evaluate_scenario_latency(low_latency_mission_specs, messages)

        # 3) Medium Latency Scenario: 

        x = 1

    x = 1

