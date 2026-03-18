from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import reduce
import os
from typing import Dict, List, Set, Tuple
import numpy as np
import math
import random

import scipy
from tqdm import tqdm
import pandas as pd

from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import ViewGeometry, SphericalGeometry
from orbitpy.util import Spacecraft

from execsatm.mission import Mission, ObservationOpportunity, PerformanceRequirement
from execsatm.attributes import CapabilityRequirementAttributes, SpatialCoverageRequirementAttributes, TemporalRequirementAttributes, ObservationRequirementAttributes
from execsatm.events import GeophysicalEvent
from execsatm.tasks import GenericObservationTask, DefaultMissionTask, EventObservationTask
from execsatm.objectives import DefaultMissionObjective, EventDrivenObjective
from execsatm.requirements import *
from execsatm.observations import ObservationOpportunity, AtomicObservationOpportunity
from execsatm.utils import Interval

from dmas.models.science.requests import TaskRequest
from dmas.utils.orbitdata import OrbitData
from dmas.utils.tools import SimulationRoles

class ResultsProcessor:
    """
    PROCESSING METHODS
    """
    @staticmethod
    def process_results(results_path : str,
                        compiled_orbitdata : Dict[str, OrbitData], 
                        agent_missions : Dict[str, Mission], 
                        events : List[GeophysicalEvent],
                        agent_specs : Dict[str, object],
                        printouts: bool = True
                    ) -> Tuple:
        """ processes simulation results after execution """
        # extract agent field of view specifications from agent specs object
        cross_track_fovs = {agent : ResultsProcessor._collect_fov_specs(specs) 
                            for agent,specs in agent_specs.items()}

        # load results
        observations_performed_df = ResultsProcessor.__load_observations_performed(results_path, printouts)
        task_reqs = ResultsProcessor.__load_task_requests(results_path, agent_missions, events, printouts)
        events_detected_df, events_detected = ResultsProcessor.__load_events_detected(results_path, compiled_orbitdata, printouts)
        known_tasks_df, known_tasks = ResultsProcessor.__load_known_tasks(results_path, compiled_orbitdata, agent_missions, task_reqs)
        agent_broadcasts_df = ResultsProcessor.__load_broadcast_history(results_path)
        planned_rewards_df, execution_costs_df = ResultsProcessor.__load_planned_utilities(results_path, compiled_orbitdata)
        grid_data_df = ResultsProcessor.__load_grid_data(compiled_orbitdata, printouts)
        
        # compile accessibility information
        ## ground points
        accesses_per_gp = ResultsProcessor.__classify_accesses_per_gp(compiled_orbitdata)

        ## events
        events_per_gp = ResultsProcessor.__classify_events_per_gp(events)
        accesses_per_event_df, accesses_per_event = ResultsProcessor.__compile_events_accessibility(events, compiled_orbitdata, agent_missions, accesses_per_gp, printouts)
        events_requested_df, events_requested = ResultsProcessor.__compile_events_requested(events, task_reqs, printouts)

        ## tasks
        accesses_per_task_df, accesses_per_task = ResultsProcessor.__compile_task_accessibility(compiled_orbitdata, known_tasks, agent_missions, accesses_per_gp, printouts)

        # compile observations
        ## ground points
        observations_per_gp = ResultsProcessor.__classify_observations_per_gp(observations_performed_df)

        ## events
        observations_per_event_df, observations_per_event = ResultsProcessor.__compile_events_observed(events, observations_per_gp, agent_missions, printouts)

        ## tasks
        observations_per_task_df, observations_per_task = ResultsProcessor.__compile_tasks_observed(known_tasks, observations_per_gp, agent_missions, printouts)
        obtained_rewards_df = ResultsProcessor.__compile_obtained_rewards(observations_per_task, agent_missions, agent_specs, cross_track_fovs, compiled_orbitdata, printouts)

        # print compiled data
        processed_results_path = os.path.join(results_path)
        os.makedirs(processed_results_path, exist_ok=True)

        printable_dfs : Dict[str, pd.DataFrame] = {
            'grid_data' : grid_data_df,
            'events_detected' : events_detected_df,
            'events_requested' : events_requested_df,
            'known_tasks' : known_tasks_df,
            'accesses_per_event': accesses_per_event_df,
            'accesses_per_task' : accesses_per_task_df,
            'observations_per_event' : observations_per_event_df,            
            'observations_per_task' : observations_per_task_df,
            'planned_rewards' : planned_rewards_df,
            'obtained_rewards' : obtained_rewards_df,
            'execution_costs' : execution_costs_df,
        }

        for filename,df in printable_dfs.items():
            df.to_parquet(os.path.join(processed_results_path, f'{filename}.parquet'), index=False)

        # return compiled data for summary generation
        return task_reqs, known_tasks, events_per_gp, events_detected, events_requested, \
            agent_broadcasts_df, planned_rewards_df, obtained_rewards_df, execution_costs_df, \
                accesses_per_gp, accesses_per_event, accesses_per_task, \
                    observations_per_gp, observations_per_event, observations_per_task
    
    @staticmethod
    def __compile_obtained_rewards(observations_per_task : Dict[GenericObservationTask, List[dict]], 
                                   agent_missions : Dict[str, Mission], 
                                   agent_specs : Dict[str, object],
                                   cross_track_fovs : Dict[str, float],
                                   compiled_orbitdata : Dict[str, OrbitData],
                                   printouts : bool
                                ) -> pd.DataFrame:
        # initialize list of obtained rewards
        obtained_rewards_data = []

        # iterate tasks and corresponding observations
        for task, observations in observations_per_task.items():
            
            # initialize previous observation time for revisit time calculation
            t_prev = np.NINF

            # iterate observations by start time 
            for n_obs,obs_perf in enumerate(sorted(observations, key=lambda obs: obs['time [s]'])):
                # unpack observation information
                observing_agent = obs_perf['agent name']
                instrument_name : str = obs_perf['instrument']
                # loc = (obs_perf['lat [deg]'], obs_perf['lon [deg]'], obs_perf['grid index'], obs_perf['GP index'])
                # t_img = obs_perf['time [s]']
                t_img = max(obs_perf['t_start'], task.availability.left) 
                d_img = obs_perf['t_end'] - t_img
                assert t_img <= obs_perf['t_end'] + 1e-9, \
                    f"Observation time {t_img} is after end of observation window at {obs_perf['t_end']}."

                # get matching mission for observing agent
                agent_mission : Mission = agent_missions[observing_agent]

                measurement_performance \
                        = ResultsProcessor.__estimate_task_performance_metrics(task,
                                                                            instrument_name,
                                                                            obs_perf['look angle [deg]'],
                                                                            t_img,
                                                                            d_img,
                                                                            agent_specs[observing_agent],
                                                                            cross_track_fovs[observing_agent],
                                                                            compiled_orbitdata[observing_agent],
                                                                            n_obs,
                                                                            t_prev,
                                                                        )

                # evaluate reward for this observation
                reward = max([agent_mission.calc_task_value(task, measurement) 
                                for measurement in measurement_performance.values()]) \
                                    if len(measurement_performance.values()) > 0 else 0.0

                # add reward information to observation performance information
                reward_dict = {
                    'task_id' : task.id,
                    'n_obs' : n_obs,
                    't_img' : t_img,
                    'agent' : observing_agent,
                    'instrument' : instrument_name,
                    'reward' : reward,
                }
                obtained_rewards_data.append(reward_dict)
                
                # update observation performance information with reward
                t_prev = t_img

        # construct obtained rewards dataframe
        if obtained_rewards_data:
            obtained_rewards_df = pd.DataFrame(obtained_rewards_data)
            # sort by task id, observation time
            obtained_rewards_df = obtained_rewards_df.sort_values(by=['task_id', 't_img']).reset_index(drop=True)
        else:
            obtained_rewards_df = pd.DataFrame(columns=['task_id', 'n_obs', 't_img', 'agent', 'instrument', 'reward'])
        
        # ----------------------------------
        # DEBUG PRINTOUTS
        # if printouts:
        #     print(obtained_rewards_df.to_string(index=False))
        # ----------------------------------

        # return obtained rewards dataframe
        return obtained_rewards_df

    @staticmethod
    def load_processed_results(results_path : str,
                                compiled_orbitdata : Dict[str, OrbitData], 
                                agent_missions : Dict[str, Mission], 
                                events : List[GeophysicalEvent],
                                printouts: bool = True
                            ) -> Tuple:
        # TODO 
        # raise NotImplementedError("Loading of processed results into summary generation format not yet implemented.")
        
        # load existing results
        observations_performed_df = ResultsProcessor.__load_observations_performed(results_path, printouts)
        task_reqs = ResultsProcessor.__load_task_requests(results_path, agent_missions, events, printouts)
        _, events_detected = ResultsProcessor.__load_events_detected(results_path, compiled_orbitdata, printouts)
        _, known_tasks = ResultsProcessor.__load_known_tasks(results_path, compiled_orbitdata, agent_missions, task_reqs)
        agent_broadcasts_df = ResultsProcessor.__load_broadcast_history(results_path)
        planned_rewards_df, execution_costs_df = ResultsProcessor.__load_planned_utilities(results_path, compiled_orbitdata)
        
        _, events_requested = ResultsProcessor.__compile_events_requested(events, task_reqs, printouts)
        events_per_gp = ResultsProcessor.__classify_events_per_gp(events)

        accesses_per_gp = ResultsProcessor.__classify_accesses_per_gp(compiled_orbitdata)
        observations_per_gp = ResultsProcessor.__classify_observations_per_gp(observations_performed_df)

        # load preprocessed dataframes if they exist, otherwise raise error     
        loadable_dfs = {
            'obtained_rewards' : None,
            'accesses_per_event': None,
            'accesses_per_task' : None,
            'observations_per_event' : None,
            'observations_per_task' : None,
        }

        for filename in loadable_dfs.keys():
            filepath = os.path.join(results_path, f'{filename}.parquet')
            if os.path.exists(filepath):
                loadable_dfs[filename] = pd.read_parquet(filepath)
            else:
                raise FileNotFoundError(f"Processed results file `{filename}.parquet` not found in path `{results_path}`. Expected file to be located at `{filepath}`.")
        
        obtained_rewards_df = loadable_dfs['obtained_rewards']
        accesses_per_event_df = loadable_dfs['accesses_per_event']
        accesses_per_task_df = loadable_dfs['accesses_per_task']
        observations_per_event_df = loadable_dfs['observations_per_event']
        observations_per_task_df = loadable_dfs['observations_per_task']

        # TODO convert dataframes to lists and dictionaries as needed for summary generation
        accesses_per_event = ResultsProcessor.__accesses_per_event_from_df(events, accesses_per_event_df)
        accesses_per_task = ResultsProcessor.__accesses_per_task_from_df(known_tasks, accesses_per_task_df)
        observations_per_event = ResultsProcessor.__observations_per_event_from_df(events, observations_per_event_df)
        observations_per_task = ResultsProcessor.__observations_per_task_from_df(known_tasks, observations_per_task_df)

        # return compiled data for summary generation
        return task_reqs, known_tasks, events_per_gp, events_detected, events_requested, \
            agent_broadcasts_df, planned_rewards_df, obtained_rewards_df, execution_costs_df, \
                accesses_per_gp, accesses_per_event, accesses_per_task, \
                    observations_per_gp, observations_per_event, observations_per_task

    """
    RESULTS LOADING METHODS
    """
    @staticmethod
    def __load_observations_performed(results_path : str, printouts : bool) -> pd.DataFrame:
        # define results path for the environment
        environment_results_path = os.path.join(results_path, SimulationRoles.ENVIRONMENT.name.lower())

        # collect observations
        try:
            observations_performed_path = os.path.join(environment_results_path, 'measurements.parquet')
            observations_performed = pd.read_parquet(observations_performed_path)
            if printouts: print('Loaded observations performed data successfully!')

        except pd.errors.EmptyDataError:
            columns = ['observer','t_img','lat','lon','range','look','incidence','zenith','instrument_name']
            observations_performed = pd.DataFrame(data=[],columns=columns)
            if printouts: print('Loaded observations performed data successfully. No observations were performed during the simulation.')

        # return observations performed        
        return observations_performed

    @staticmethod    
    def __load_events_detected(results_path : str, 
                               compiled_orbitdata : Dict[str, OrbitData], 
                               printouts : bool
                            ) -> Tuple[pd.DataFrame, List[GeophysicalEvent]]:
        # compile events detected
        if printouts: print('Collecting event detection data...')
        events_detected_df : pd.DataFrame = None
        for agent_name in compiled_orbitdata.keys():            
            # define path to agent's detected events file
            events_detected_path = os.path.join(results_path, agent_name.lower(), 'events_detected.parquet')
            
            # skip if file doesn't exist
            if not os.path.isfile(events_detected_path): continue

            # load detected events            
            events_detected_temp = pd.read_parquet(events_detected_path)

            # concatenate to main dataframe
            events_detected_df = pd.concat([events_detected_df, events_detected_temp], axis=0) \
                if events_detected_df is not None else events_detected_temp

        assert events_detected_df is not None, \
            "Couldn't load Event Detection file for any agent."
        
        # remove duplicates
        events_detected_df = events_detected_df.drop_duplicates().reset_index(drop=True)
        
        # convert to list of GeophysicalEvent
        events_detected : list[GeophysicalEvent] = []
        for _,row in events_detected_df.iterrows():           
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['GP index']),
                row['detection time [s]'],
                row['end time [s]'] - row['start time [s]'],
                row['severity'],
                row['start time [s]'],
                row['id']
            )
            events_detected.append(event)

        return events_detected_df, events_detected
    
    @staticmethod
    def __load_task_requests(results_path : str, 
                             agent_missions : Dict[str, Mission], 
                             events : List[GeophysicalEvent], 
                             printouts : bool
                            ) -> List[TaskRequest]:
        
        # define results path for the environment
        environment_results_path = os.path.join(results_path, SimulationRoles.ENVIRONMENT.name.lower())
        
        # compile measurement requests
        if printouts: print('Collecting measurement request data...')
        try:
            task_reqs_df = pd.read_parquet((os.path.join(environment_results_path, 'requests.parquet')))
        except pd.errors.EmptyDataError:
            columns = ['id','requester','lat [deg]','lon [deg]','severity','t start','t end','t corr','Measurment Types']
            task_reqs_df = pd.DataFrame(data=[],columns=columns)
        
        # remove duplicates
        task_reqs_df = task_reqs_df.drop_duplicates(subset='id').reset_index(drop=True)

        # convert to list of TaskRequest
        task_reqs = []
        for _,row in task_reqs_df.iterrows():
            matching_events : list[GeophysicalEvent] = [
                event for event in events
                if event.id == row['task']['event']['id']
            ]
            assert matching_events, \
                f"No matching event found for measurement request with event id `{row['task']['event']['id']}`"
            
            # get name of agent requesting the task
            requester = row['requester']
            
            # get matching `EventDrivenObjective`
            relevant_objectives = [objective for objective in agent_missions[requester]
                                    if isinstance(objective, EventDrivenObjective)
                                    and objective.parameter == row['task']['parameter']]
            # ensure exactly one matching objective found
            assert relevant_objectives, \
                f"No matching EventDrivenObjective found in mission for requester `{requester}` and parameter `{row['parameter']}`."
            
            # create event-driven task 
            task = EventObservationTask(
                row['task']['parameter'],
                event=matching_events[0],
                objective=relevant_objectives[0]
            )
            
            # create task request
            req = TaskRequest(
                task,
                row['requester'],
                agent_missions[requester].name,
                row['t_req'],
                row['task']['event']['id']
            )

            # add to list of task requests
            task_reqs.append(req)

        return task_reqs
    
    @staticmethod
    def __load_known_tasks(results_path : str,
                           compiled_orbitdata : Dict[str, OrbitData],
                           agent_missions : Dict[str, Mission],
                           task_reqs : List[TaskRequest]
                           ) -> Tuple[pd.DataFrame, List[GenericObservationTask]]:
        # compile default tasks from every agent
        default_tasks_df : pd.DataFrame = None
        for agent_name in compiled_orbitdata.keys():
            # define path to agent's known tasks file
            known_tasks_path = os.path.join(results_path, agent_name.lower(), 'known_tasks.parquet')
            
            # skip if file doesn't exist
            if not os.path.isfile(known_tasks_path): continue
            
            # load default tasks
            default_tasks_temp = pd.read_parquet(known_tasks_path)

            # concatenate to main dataframe
            default_tasks_df = pd.concat([default_tasks_df, default_tasks_temp], axis=0) \
                if not default_tasks_temp.empty else default_tasks_df

        # remove duplicates
        default_tasks_df = default_tasks_df.drop_duplicates().reset_index(drop=True) \
            if default_tasks_df is not None else pd.DataFrame(columns=['id','task type','parameter','lat [deg]','lon [deg]','grid index','gp index','t start','t end','priority'])
        
        # convert to list of tasks
        known_tasks : list[GenericObservationTask] = []
        for _,row in default_tasks_df.iterrows():
            # get name of agent requesting the task
            requester = row['requester']
            if row['task type'] != GenericObservationTask.DEFAULT:
                raise ValueError(f"Unknown task type `{row['task type']}` found in known tasks file.")
            
            # get matching `DefaultMissionObjective`
            relevant_objectives = [objective for objective in agent_missions[requester]
                                    if isinstance(objective, DefaultMissionObjective)
                                    and objective.parameter == row['parameter']]
            # ensure exactly one matching objective found
            assert relevant_objectives, \
                f"No matching DefaultMissionObjective found in mission for requester `{requester}` and parameter `{row['parameter']}`."
            
            task = DefaultMissionTask(
                row['parameter'],
                (row['lat [deg]'], row['lon [deg]'], row['grid index'], row['gp index']),
                row['t end'] - row['t start'],
                row['priority'],
                relevant_objectives[0],
                row['id']
            )
            known_tasks.append(task)

        # suplement with event request tasks
        requested_tasks = [req.task for req in task_reqs 
                            if req.task not in known_tasks]
        known_tasks.extend(requested_tasks)

        event_tasks = []
        for req_task in requested_tasks:
            for lat,lon,grid_index,gp_index in req_task.location:
                row = {
                    'id' : req_task.id,
                    'task type' : req_task.task_type,
                    'parameter' : req_task.parameter,
                    'lat [deg]' : lat,
                    'lon [deg]' : lon,
                    'grid index' : grid_index,
                    'gp index' : gp_index,
                    't start' : req_task.availability.left,
                    't end' : req_task.availability.right,
                    'priority' : req_task.priority,
                }
                event_tasks.append(row)
        if event_tasks:
            event_tasks_df = pd.DataFrame(event_tasks)
        else:
            event_tasks_df = pd.DataFrame(columns=['id','task type','parameter','lat [deg]','lon [deg]','grid index','gp index','t start','t end','priority'])

        # merge default tasks and event observation tasks into known tasks dataframe
        known_tasks_df = pd.DataFrame(columns=['id','task type','parameter','lat [deg]','lon [deg]','grid index','gp index','t start','t end','priority'])
        if default_tasks_df is not None:
            known_tasks_df = pd.concat([known_tasks_df, default_tasks_df], axis=0)
        if not event_tasks_df.empty:
            known_tasks_df = pd.concat([known_tasks_df, event_tasks_df], axis=0)    

        # remove duplicates
        known_tasks = list(set(known_tasks))
        known_tasks_df = known_tasks_df.drop_duplicates(subset='id').reset_index(drop=True) \
            if not known_tasks_df.empty else pd.DataFrame(columns=['id','task type','parameter','lat [deg]','lon [deg]','grid index','gp index','t start','t end','priority'])  
        
        # return known tasks dataframe and list of known tasks
        return known_tasks_df, known_tasks

    @staticmethod
    def __load_broadcast_history(results_path : str) -> pd.DataFrame:
        # define results path for the environment
        environment_results_path = os.path.join(results_path, SimulationRoles.ENVIRONMENT.name.lower())        
        
        # compile broadcast history
        agent_broadcasts_df = pd.read_parquet((os.path.join(environment_results_path, 'broadcasts.parquet')))
    
        # return broadcast history dataframe
        return agent_broadcasts_df
    
    @staticmethod
    def __load_planned_utilities(results_path : str, 
                                 compiled_orbitdata : Dict[str, OrbitData]
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # compile planned agent reward data
        planned_rewards_df = None
        for agent_name in compiled_orbitdata.keys():
            rewards_path = os.path.join(results_path, agent_name.lower(), 'rewards.parquet')
            if not os.path.isfile(rewards_path): continue
            
            # load rewards data
            rewards_temp = pd.read_parquet(rewards_path)

            # add agent name column
            rewards_temp['agent'] = agent_name

            # concatenate to main dataframe
            if planned_rewards_df is None:
                planned_rewards_df = rewards_temp   
            else:
                planned_rewards_df = pd.concat([planned_rewards_df, rewards_temp], axis=0)

        # handle case where no rewards data was printed
        if planned_rewards_df is None:
            # generate empty dataframe with appropriate columns
            planned_rewards_df = pd.DataFrame(columns=['task_id', 'n_obs', 't_img', 'agent_name', 'planned reward', 'performed reward', 'agent'])

        # compile executed agent cost data
        execution_costs_df = None
        alpha = 1e-6
        for agent_name in compiled_orbitdata.keys():
            agent_state_path = os.path.join(results_path, agent_name.lower(), 'state_history.parquet')
            if not os.path.isfile(agent_state_path): continue
            
            # load costs data
            state_temp = pd.read_parquet(agent_state_path)            
            
            A = np.vstack(state_temp["attitude"].to_numpy())          # shape (N, 3)
            dA = np.abs(np.diff(A, axis=0))              # shape (N-1, 3)
            
            cost_temp = pd.DataFrame({
                'time [s]': state_temp['t'][:-1],               # shape (N-1,)
                'agent' : [agent_name] * (len(state_temp)-1),   # shape (N-1,)                            
                'status' : state_temp['status'][:-1],           # shape (N-1,)
                'attitude [deg]' : state_temp['attitude'][:-1], # shape (N-1,) 
                'cost': alpha * np.linalg.norm(dA, axis=1),     # shape (N-1,)
            })

            # concatenate to main dataframe
            if execution_costs_df is None:
                execution_costs_df = cost_temp   
            else:
                execution_costs_df = pd.concat([execution_costs_df, cost_temp], axis=0)

        # return planned rewards and execution costs dataframes
        return planned_rewards_df, execution_costs_df

    """
    COMPILATION AND CLASSIFICATION METHODS
    """
    @staticmethod
    def __load_grid_data(compiled_orbitdata : Dict[str, OrbitData], printouts : bool) -> Set[Tuple[int,int]]:
        # compile grids from agent orbitdata
        grid_data = []
        for agent_orbitdata in tqdm(compiled_orbitdata.values(), 
                                    desc='Compiling target grids from agent orbit data', 
                                    leave=False,
                                    disable=not printouts
                                ):
            # get set of accessible ground points
            gps_accessible_temp : list = [row for row in agent_orbitdata.grid_data]

            # add to main list
            grid_data.extend(gps_accessible_temp)
        
        # convert to dataframe
        grid_data_df_temp = pd.DataFrame(grid_data, columns=['lat [deg]', 'lon [deg]','grid index', 'GP index'])

        # remove duplicates
        grid_data_df = grid_data_df_temp.drop_duplicates(subset=['grid index', 'GP index']).reset_index(drop=True)
        
        # return set of accessible ground points
        return grid_data_df
    
    @staticmethod
    def __classify_accesses_per_gp(
            compiled_orbitdata : Dict[str, OrbitData]
        ) -> Dict[Tuple[int,int], np.ndarray]:

        # initialize list of columns and dictionary of column chunks
        cols = None
        chunks = None  # maps: col -> list of arrays

        # 1) Collect per-sat arrays (no per-row dicts)
        for agent_orbit_data in compiled_orbitdata.values():
            # get access data for this agent 
            access = agent_orbit_data.gp_access_data.lookup_interval(
                0.0,
                include_extras=True,
                decode=True,
            )

            # check if access data is empty
            t = access.get("time [s]", None)
            if t is None or len(t) == 0:
                # if so, skip this agent
                continue

            # check if columns have been initialized
            if cols is None:
                # if not, initialize columns and chunks dictionary
                cols = list(access.keys())
                chunks = {c: [] for c in cols}

            # append arrays for each column
            for c in cols:
                chunks[c].append(np.asarray(access[c]))

        # check if any access data was found
        if cols is None: 
            # no data found; return empty results
            return {}, {} 

        # 2) Concatenate once per column
        big = {}
        for c in cols:
            big[c] = np.concatenate(chunks[c], axis=0) if chunks[c] else np.array([])

        # check if time column is empty after concatenation
        if big["time [s]"].size == 0:
            # no data found; return empty results
            return {}, {}

        # 3) Group by (grid index, GP index) using packed keys
        grid = np.asarray(big["grid index"], dtype=np.int64)
        gp = np.asarray(big["GP index"], dtype=np.int64)

        # fast packed key (assumes both fit in uint32; common for indices)
        keys = (grid.astype(np.uint64) << np.uint64(32)) | (gp.astype(np.uint64) & np.uint64(0xFFFFFFFF))

        order = np.argsort(keys)
        keys_sorted = keys[order]

        # reorder all columns once
        for c in cols:
            big[c] = np.asarray(big[c])[order]

        # group boundaries
        uniq_keys, starts, counts = np.unique(keys_sorted, return_index=True, return_counts=True)

        # 4) Build slice map: (grid,gp) -> slice(start, stop)
        gp_slices = {}  # type: Dict[Tuple[int, int], slice]
        for k, start, cnt in zip(uniq_keys, starts, counts):
            g = int(k >> np.uint64(32))
            p = int(k & np.uint64(0xFFFFFFFF))
            a = int(start)
            b = int(start + cnt)
            gp_slices[(g, p)] = slice(a, b)

        # 5) Build final output: (grid,gp) -> {col: big[col][slice]}
        out = {(g,p): {c: big[c][sl] for c in big} for (g,p), sl in gp_slices.items()}

        # sort by time within each group
        for data in out.values():
            # get sorting permutation based on time
            order = np.argsort(data["time [s]"], kind="mergesort") 

            # apply permutation to every column
            for col in data:
                data[col] = data[col][order]

        # return final output
        return out

    @staticmethod
    def __classify_events_per_gp(events : List[GeophysicalEvent]) -> Dict[Tuple[int,int], List[GeophysicalEvent]]:
        events_per_gp : Dict[Tuple[int,int], List[GeophysicalEvent]] = defaultdict(list)
        for event in events:
            *_, grid_index, gp_index = event.location
            events_per_gp[(int(grid_index), int(gp_index))].append(event)
        return events_per_gp


    @staticmethod
    def __compile_events_accessibility(events : List[GeophysicalEvent], 
                                       compiled_orbitdata : Dict[str, OrbitData], 
                                       agent_missions : Dict[str, Mission], 
                                       accesses_per_gp : Dict[Tuple[int,int], Dict[str, np.ndarray]],
                                       printouts : bool
                                    ) -> Tuple[pd.DataFrame, Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]]]:
        
        # initiate list of compiled events with access information
        accesses_per_event_data = []
        accesses_per_event = dict()

        # look for accesses to each event for each agent
        for event in tqdm(events, 
                            desc='Classifying event accesses, detections, and observations', 
                            leave=False,
                            disable=not printouts):
            # unpackage event
            *_,event_grid_idx,event_gp_idx = event.location
            event_grid_idx = int(event_grid_idx)
            event_gp_idx = int(event_gp_idx)
            event_type = event.event_type.lower()   

            # compile observation requirements for this event type
            instrument_capability_reqs : Dict[str, set] = defaultdict(set)

            # group requirements by agents to avoid double counting
            for agent_name,mission in agent_missions.items():
                for objective in mission:
                    # check if objective matches event type
                    if (isinstance(objective, EventDrivenObjective) 
                        and objective.event_type.lower() == event_type):
                        
                        # collect instrument capability requirements
                        for req in objective:
                            # check if requirement is an instrument capability requirement
                            if (isinstance(req, ExplicitCapabilityRequirement) 
                                and req.attribute == 'instrument'):
                                instrument_capability_reqs[agent_name].update({val.lower() for val in req.valid_values})
            
            if any([len(instrument_capability_reqs[agent_name]) == 0 
                        for agent_name in instrument_capability_reqs]):
                raise NotImplementedError(f"No instrument capability requirements found for event type `{event_type}`. Case not yet supported.")

            # get matching accesses for this location
            location_key = (event_grid_idx, event_gp_idx)
            location_access : Dict[str, np.ndarray] = accesses_per_gp.get(location_key, None)
            
            # check if there are any accesses for this location
            if location_access is None: continue # no accesses found; skip
            
            # unpack access information
            t = location_access["time [s]"]
            agent = location_access["agent name"]
            instr = location_access["instrument"]
            
            # filter data that exists within the task's availability window and matches instrument capability requirements            
            availability_mask = (
                (t >= event.availability.left) &
                (t <= event.availability.right)
            )

            idx = np.nonzero(availability_mask)[0]

            # Build compatibility mask only over the narrowed slice
            compat = np.zeros(idx.shape[0], dtype=bool)
            for j, i in enumerate(idx):
                a = str(agent[i])
                allowed = instrument_capability_reqs.get(a, set())
                compat[j] = str(instr[i]).lower() in allowed

            combined_mask = availability_mask.copy()
            combined_mask[idx] = compat
            location_access_filtered = {col: arr[combined_mask] for col, arr in location_access.items()}
            
            # iterate throufh data to map accesses by agent name and instrument
            t = np.asarray(location_access_filtered["time [s]"], dtype=np.float64)
            agent = np.asarray(location_access_filtered["agent name"])      # str or int
            instr = np.asarray(location_access_filtered["instrument"])      # str or int

            n = t.size
            if n == 0:
                access_intervals : List[Tuple[Interval, str, str]] = []
            else:
                # --- Sort by (agent, instrument, time) ---
                # np.lexsort sorts by last key first => (time) within (instr) within (agent)
                order = np.lexsort((t, instr, agent))
                t_s = t[order]
                a_s = agent[order]
                i_s = instr[order]

                access_intervals : List[Tuple[Interval, str, str]] = []

                # --- Linear scan, merging unit intervals ---
                # We create [t, t+dt(agent)] and merge overlaps/adjacent.
                idx = 0
                while idx < n:
                    a0 = a_s[idx]
                    i0 = i_s[idx]

                    # handle agent_name for dt lookup
                    agent_name = str(a0) if not isinstance(a0, (str, np.str_)) else a0
                    dt = float(compiled_orbitdata[agent_name].time_step)

                    # start a new merged interval at this group's first time
                    left = float(t_s[idx])
                    right = left + dt
                    idx += 1

                    # consume all entries for this (agent, instrument) group
                    while idx < n and a_s[idx] == a0 and i_s[idx] == i0:
                        tt = float(t_s[idx])
                        # if this unit interval overlaps/abuts, extend
                        if tt <= right + 1e-9:
                            nr = tt + dt
                            if nr > right:
                                right = nr
                        else:
                            # close current interval and start a new one
                            access_intervals.append((Interval(left, right), agent_name, i0))
                            left = tt
                            right = tt + dt
                        idx += 1

                    # close last interval for this group
                    access_intervals.append((Interval(left, right), agent_name, i0))

                # sort final intervals by start time
                access_intervals.sort(key=lambda x: x[0].left)                
            
            # add to compiled event access data 
            for interval, agent_name, instrument in access_intervals:
                accesses_per_event_data.append({
                    'event id' : event.id,
                    'event type' : event.event_type,
                    'lat [deg]' : event.location[0],
                    'lon [deg]' : event.location[1],
                    'grid index' : event.location[2],
                    'GP index' : event.location[3],
                    'access start [s]' : interval.left,
                    'access end [s]' : interval.right,
                    'instrument' : instrument,  
                    'agent name' : agent_name
                })
                
            # add to compiled event accessibility list
            if access_intervals: accesses_per_event[event] = access_intervals

        # convert to dataframe
        if accesses_per_event_data:
            accesses_per_event_df = pd.DataFrame(accesses_per_event_data)
        else:
            accesses_per_event_df = pd.DataFrame(columns=['event id','event type','lat [deg]','lon [deg]','grid index','GP index','access start [s]','access end [s]','instrument','agent name'])
        
        # return compiled event accessibility information
        return accesses_per_event_df, accesses_per_event
    
    @staticmethod
    def __compile_events_requested(events : List[GeophysicalEvent], task_reqs : List[TaskRequest], printouts : bool) -> Dict[GeophysicalEvent, List[TaskRequest]]:
        # initialize event to matching requests map
        requests_per_event : Dict[GeophysicalEvent, List[TaskRequest]] = defaultdict(list)
        event_rquests_df_data = []

        for event in tqdm(events, 
                            desc='Classifying event requests', 
                            leave=False,
                            disable=not printouts):
            
            # find measurement requests that match this event
            if any(not isinstance(task_req.task, EventObservationTask) for task_req in task_reqs):
                raise NotImplementedError("Non-event observation tasks are not yet supported in event observation classification.")
            else:
                matching_requests = sorted([task_req for task_req in task_reqs if task_req.task.event == event], 
                                            key= lambda a : a.t_req)
            
            # add to map of events to matching requests
            requests_per_event[event] = matching_requests

            for task_req in matching_requests:
                event_rquests_df_data.append({
                    'event id' : event.id,
                    'event type' : event.event_type,
                    'lat [deg]' : event.location[0],
                    'lon [deg]' : event.location[1],
                    'grid index' : event.location[2],
                    'GP index' : event.location[3],
                    'requester' : task_req.requester,
                    't_req' : task_req.t_req,
                    'parameter' : task_req.task.parameter
                })

        # convert to dataframe
        events_requested_df = pd.DataFrame(event_rquests_df_data)

        # return map of events to matching measurement requests
        return events_requested_df, requests_per_event

    @staticmethod
    def __compile_task_accessibility(compiled_orbitdata : Dict[str, OrbitData], 
                                     known_tasks: List[GenericObservationTask], 
                                     agent_missions: Dict[str, Mission], 
                                     accesses_per_gp : Dict[Tuple[int,int], Dict[str, np.ndarray]],
                                     printouts: bool
                                    ) -> Tuple[pd.DataFrame, List[Tuple[GenericObservationTask, List[Tuple[Interval, str, str]]]]]:
        # initialize map of tasks to access intervals
        accesses_per_task : Dict[GenericObservationTask, list] = defaultdict(list)
        accesses_per_task_df_data = []

        for task in tqdm(known_tasks, desc="Processing task accessibility", leave=False, disable=not printouts):
            # compile observation requirements for this task
            instrument_capability_reqs : Dict[str, set] = defaultdict(set)

            for agent_name, mission in agent_missions.items():
                # find objectives matching this task
                if task.objective not in mission: continue # skip if objective not in mission

                # collect instrument capability requirements
                for req in task.objective:
                    # check if requirement is an instrument capability requirement
                    if (isinstance(req, ExplicitCapabilityRequirement) 
                        and req.attribute == 'instrument'):
                        instrument_capability_reqs[agent_name].update({val.lower() for val in req.valid_values})

            # find all accesses and observations that match this task
            task_access_windows = []
                
            # check all task locations
            for location in task.location:
                # unpack location
                task_lat,task_lon,task_grid_idx, task_gp_idx = location
                task_lat = round(task_lat,6)
                task_lon = round(task_lon,6)     
                task_grid_idx = int(task_grid_idx)
                task_gp_idx = int(task_gp_idx)   
                location_key = (task_grid_idx, task_gp_idx)
                
                # get matching accesses for this location
                location_access : pd.DataFrame = accesses_per_gp.get(location_key, None)
                
                # check if there are any accesses for this location
                if location_access is None: continue # no accesses found; skip
                
                # get matching accesses for this location
                location_access : Dict[str, np.ndarray] = accesses_per_gp.get(location_key, None)
                t = location_access["time [s]"]
                agent = location_access["agent name"]
                instr = location_access["instrument"]
                
                # check if there are any accesses for this location
                if location_access is None: continue # no accesses found; skip
                
                # filter data that exists within the task's availability window and matches instrument capability requirements            
                dt = compiled_orbitdata[next(iter(compiled_orbitdata))].time_step
                availability_mask = (
                    (task.availability.left - dt <= t) &
                    (t <= task.availability.right)
                )

                idx = np.nonzero(availability_mask)[0]

                # Build compatibility mask only over the narrowed slice
                compat = np.zeros(idx.shape[0], dtype=bool)
                for j, i in enumerate(idx):
                    a = str(agent[i])
                    allowed = instrument_capability_reqs.get(a, set())
                    compat[j] = str(instr[i]).lower() in allowed

                combined_mask = availability_mask.copy()
                combined_mask[idx] = compat
                location_access_filtered = {col: arr[combined_mask] for col, arr in location_access.items()}
                
                # iterate through data to map accesses by agent name and instrument
                t = np.asarray(location_access_filtered["time [s]"], dtype=np.float64)
                agent = np.asarray(location_access_filtered["agent name"])      # str or int
                instr = np.asarray(location_access_filtered["instrument"])      # str or int

                assert all(t_i in task.availability 
                           or task.availability.left-dt <= t_i <= task.availability.right 
                           for t_i in t), \
                    f"Some access times {t} are outside of task availability {task.availability} for task {task.id}."

                n = t.size
                if n == 0:
                    access_intervals : List[Tuple[Interval, str, str]] = []
                else:
                    # --- Sort by (agent, instrument, time) ---
                    # np.lexsort sorts by last key first => (time) within (instr) within (agent)
                    order = np.lexsort((t, instr, agent))
                    t_s = t[order]
                    a_s = agent[order]
                    i_s = instr[order]

                    access_intervals : List[Tuple[Interval, str, str]] = []

                    # --- Linear scan, merging unit intervals ---
                    # We create [t, t+dt(agent)] and merge overlaps/adjacent.
                    idx = 0
                    while idx < n:
                        a0 = a_s[idx]
                        i0 = i_s[idx]

                        # handle agent_name for dt lookup
                        agent_name = str(a0) if not isinstance(a0, (str, np.str_)) else a0
                        dt = float(compiled_orbitdata[agent_name].time_step)

                        # start a new merged interval at this group's first time
                        left = float(t_s[idx])
                        right = left + dt
                        idx += 1

                        # consume all entries for this (agent, instrument) group
                        while idx < n and a_s[idx] == a0 and i_s[idx] == i0:
                            tt = float(t_s[idx])
                            # if this unit interval overlaps/abuts, extend
                            if tt <= right + 1e-9:
                                nr = tt + dt
                                if nr > right:
                                    right = nr
                            else:
                                # close current interval and start a new one
                                access_intervals.append((Interval(left, right - dt), agent_name, i0))
                                left = tt
                                right = tt + dt
                            idx += 1

                        # close last interval for this group
                        access_intervals.append((Interval(left, right - dt), agent_name, i0))

                    # sort final intervals by start time
                    access_intervals.sort(key=lambda x: x[0].left)        

                    # discard any intervals that are not within the task's availability window (with some tolerance for time step)
                    # and clip intervals to task availability window
                    access_intervals = [(interval.intersection(task.availability), agent_name, i0) 
                                        for interval, agent_name, i0 in access_intervals 
                                        if interval.overlaps(task.availability)]    
                
                # append to task lists
                task_access_windows.extend(access_intervals)

                # append to task accesses dataframe
                for interval, agent_name, instrument in access_intervals:
                    accesses_per_task_df_data.append({
                        'task id' : task.id,
                        'task type' : task.task_type,
                        'parameter' : task.parameter,
                        'lat [deg]' : location[0],
                        'lon [deg]' : location[1],
                        'grid index' : location[2],
                        'GP index' : location[3],
                        't start' : interval.left,
                        't end' : interval.right,
                        'agent name' : agent_name,
                        'instrument' : instrument
                    })

                for interval,*_ in access_intervals:
                    assert interval.is_subset(task.availability), \
                        f"Access interval {interval} is not a subset of task availability {task.availability} for task {task.id}."

            # if access windows were found for this task, add to map of tasks to access intervals
            if task_access_windows: accesses_per_task[task] = task_access_windows
    
        # convert to dataframe
        if accesses_per_task_df_data:
            accesses_per_task_df = pd.DataFrame(accesses_per_task_df_data)
        else:
            accesses_per_task_df = pd.DataFrame(columns=['task id','task type','parameter','lat [deg]','lon [deg]','grid index','GP index','t start','t end','agent name','instrument'])

        # remove duplicates
        accesses_per_task_df = accesses_per_task_df.drop_duplicates().reset_index(drop=True)

        # return task access information as dataframe and list
        return accesses_per_task_df, accesses_per_task

    @staticmethod
    def __classify_observations_per_gp(observations_performed_df : pd.DataFrame) -> Dict[Tuple[int,int], pd.DataFrame]:
        # classify observations per GP 
        observations_per_gp : Dict[Tuple[int,int], pd.DataFrame] \
                                = {(int(group[0]), int(group[1])) : data
                                for group,data in observations_performed_df.groupby(['grid index', 'GP index'])} \
                                if not observations_performed_df.empty else dict() # handle empty observations case

        # return observations per GP and set of accessible GPs
        return observations_per_gp

    @staticmethod
    def __compile_events_observed(events : List[GeophysicalEvent], 
                                  observations_per_gp : Dict[Tuple, pd.DataFrame], 
                                  agent_missions : Dict[str, Mission], 
                                  printouts : bool
                                ) -> Dict[tuple, pd.DataFrame]:
        # initialize list of observations per event
        observations_per_event = defaultdict(list)
        observations_per_event_df_data = []

        for event in tqdm(events, 
                            desc='Classifying event accesses, detections, and observations', 
                            leave=False,
                            disable=not printouts):
            # unpackage event
            *_,event_grid_idx,event_gp_idx = event.location
            event_grid_idx = int(event_grid_idx)
            event_gp_idx = int(event_gp_idx)
            event_type = event.event_type.lower()   

            # compile observation requirements for this event type
            instrument_capability_reqs : Dict[str, set] = defaultdict(set)

            # group requirements by agents to avoid double counting
            for agent_name,mission in agent_missions.items():
                for objective in mission:
                    # check if objective matches event type
                    if (isinstance(objective, EventDrivenObjective) 
                        and objective.event_type.lower() == event_type):
                        
                        # collect instrument capability requirements
                        for req in objective:
                            # check if requirement is an instrument capability requirement
                            if (isinstance(req, ExplicitCapabilityRequirement) 
                                and req.attribute == 'instrument'):
                                instrument_capability_reqs[agent_name].update({val.lower() for val in req.valid_values})
            
            if any([len(instrument_capability_reqs[agent_name]) == 0 
                        for agent_name in instrument_capability_reqs]):
                raise NotImplementedError(f"No instrument capability requirements found for event type `{event_type}`. Case not yet supported.")

            # get event observations for this event's location
            if (event_grid_idx, event_gp_idx) not in observations_per_gp:
                # no observations were performed at this event's location
                # create empty dataframe with expected columns for consistency
                matching_observations = pd.DataFrame(columns=["t_start", "t_end", "instrument", "agent name"])
            else:
                matching_observations : pd.DataFrame = observations_per_gp[(event_grid_idx, event_gp_idx)]

            # find observations that match the event time
            time_mask = (
                ((event.availability.left <= matching_observations["t_start"]) &
                (matching_observations["t_start"] <= event.availability.right))
                |
                ((event.availability.left <= matching_observations["t_end"]) &
                (matching_observations["t_end"] <= event.availability.right))
                |
                ((matching_observations["t_start"] <= event.availability.left) &
                (event.availability.right <= matching_observations["t_end"]))
            )
            matching_observations = matching_observations[time_mask]

            # find observations that match the event's instrument capability requirements
            inst_lower = matching_observations["instrument"].str.lower()
            agents = matching_observations["agent name"]

            instrument_mask = [
                inst in instrument_capability_reqs.get(agent, [])
                for inst, agent in zip(inst_lower, agents)
            ]
            matching_observations = matching_observations[instrument_mask]

            # check if any observations remain after filtering by time and instrument capability requirements
            if matching_observations.empty: 
                # skip to next event if no matching observations found
                continue

            # Response times
            matching_observations["resp time [s]"] = matching_observations["t_start"] - event.availability.left

            # Normalized
            matching_observations["resp time [norm]"] = matching_observations["resp time [s]"] / event.availability.span()

            # convert to list of dicts for easier handling downstream
            matching_observations = [dict(row) for _,row in matching_observations.iterrows()]
            matching_observations = sorted(matching_observations, key=lambda x: x['t_start']) # sort by observation start time

            n_obs_data = []
            t_rev_data = []
            
            # add to dataframe of observations per event
            prev_row = None
            for n_obs,row in enumerate(matching_observations):
                t_rev = row['t_start'] - prev_row['t_end'] if prev_row is not None else np.Inf
                
                observations_per_event_df_data.append({
                    'event id' : event.id,
                    'event type' : event.event_type,
                    'lat [deg]' : event.location[0],
                    'lon [deg]' : event.location[1],
                    'grid index' : event.location[2],
                    'GP index' : event.location[3],
                    'agent name' : agent_name,
                    **row,
                    'n_obs' : n_obs,
                    't_rev' : t_rev
                })

                n_obs_data.append(n_obs)
                t_rev_data.append(t_rev)
                prev_row = row            

            matching_observations = [{**obs, 'n_obs': n_obs, 't_rev': t_rev} for obs, n_obs, t_rev in zip(matching_observations, n_obs_data, t_rev_data)]
            
            # add to observations per event map
            if matching_observations: observations_per_event[event] = matching_observations        

        # convert to dataframe
        if observations_per_event_df_data:
            observations_per_event_df = pd.DataFrame(observations_per_event_df_data)
        else:
            observations_per_event_df = pd.DataFrame(columns=['event id','event type','lat [deg]','lon [deg]','grid index','GP index','agent name','t_start','t_end','instrument','n_obs','t_rev'])

        # return observations per event
        return observations_per_event_df, observations_per_event

    @staticmethod
    def __compile_tasks_observed(known_tasks: List[GenericObservationTask], 
                                 observations_per_gp: Dict, 
                                 agent_missions: Dict[str, Mission], 
                                 printouts: bool
                                ) -> Tuple[pd.DataFrame, Dict]:
        # initiate task observation data structures
        tasks_observed : Dict[GenericObservationTask, list] = defaultdict(list)
        task_observations_df_data = []

        # itarate through tasks and find matching observations
        for task in tqdm(known_tasks, desc="Processing task observations", leave=False, disable=not printouts):
            # compile observation requirements for this task
            instrument_capability_reqs : Dict[str, set] = defaultdict(set)

            for agent_name, mission in agent_missions.items():
                # find objectives matching this task
                if task.objective not in mission: continue # skip if objective not in mission

                # collect instrument capability requirements
                for req in task.objective:
                    # check if requirement is an instrument capability requirement
                    if (isinstance(req, ExplicitCapabilityRequirement) 
                        and req.attribute == 'instrument'):
                        instrument_capability_reqs[agent_name].update({val.lower() for val in req.valid_values})

            # find all accesses and observations that match this task
            task_observations = []

            if "EventObservationTask-'generic parameter'@(0,2324)-EVENT-e21c2b8e" in task.id:
                x = 1 # breakpoint
                
            # check all task locations
            for location in task.location:
                # unpack location
                task_lat,task_lon,task_grid_idx, task_gp_idx = location
                task_lat = round(task_lat,6)
                task_lon = round(task_lon,6)     
                task_grid_idx = int(task_grid_idx)
                task_gp_idx = int(task_gp_idx)                   

                # find observations performed at task location while task was active                
                if (task_grid_idx, task_gp_idx) not in observations_per_gp:
                    # no observations were performed at this task's location
                    # create empty dataframe with expected columns for consistency
                    matching_observations = pd.DataFrame(columns=["t_start", "t_end", "instrument", "agent name"])
                else:
                    matching_observations : pd.DataFrame = observations_per_gp[(task_grid_idx, task_gp_idx)]
                
                # find observations that match the event time
                time_mask = (
                    ((task.availability.left <= matching_observations["t_start"]) &
                    (matching_observations["t_start"] <= task.availability.right))
                    |
                    ((task.availability.left <= matching_observations["t_end"]) &
                    (matching_observations["t_end"] <= task.availability.right))
                    |
                    ((matching_observations["t_start"] <= task.availability.left) &
                    (task.availability.right <= matching_observations["t_end"]))
                )
                matching_observations = matching_observations[time_mask]

                # find observations that match the event's instrument capability requirements
                inst_lower = matching_observations["instrument"].str.lower()
                agents = matching_observations["agent name"]

                instrument_mask = [
                    inst in instrument_capability_reqs.get(agent, [])
                    for inst, agent in zip(inst_lower, agents)
                ]
                matching_observations = matching_observations[instrument_mask]

                # check if any observations remain after filtering by time and instrument capability requirements
                if matching_observations.empty: 
                    # skip to next event if no matching observations found
                    continue

                # Response times
                matching_observations["resp time [s]"] = matching_observations["t_start"] - task.availability.left

                # Normalized
                matching_observations["resp time [norm]"] = matching_observations["resp time [s]"] / task.availability.span()

                # convert matching observations to list of dicts for easier handling
                matching_observations = [dict(row) for _,row in matching_observations.iterrows()]
                matching_observations = sorted(matching_observations, key=lambda x: x['t_start']) # sort by observation start time

                n_obs_data = []
                t_rev_data = []

                prev_row = None
                for n_obs,row in enumerate(matching_observations):
                    t_rev = row['t_start'] - prev_row['t_end'] if prev_row is not None else np.Inf
                    
                    task_observations_df_data.append({
                        'task id' : task.id,
                        'parameter' : task.parameter,
                        'lat [deg]' : task_lat,
                        'lon [deg]' : task_lon,
                        'grid index' : task_grid_idx,
                        'GP index' : task_gp_idx,
                        **row,
                        'n_obs' : n_obs,
                        't_rev' : t_rev
                    })
                    n_obs_data.append(n_obs)
                    t_rev_data.append(t_rev)
                    prev_row = row

                matching_observations = [{**obs, 'n_obs': n_obs, 't_rev': t_rev} for obs, n_obs, t_rev in zip(matching_observations, n_obs_data, t_rev_data)]
                
                # append to task lists
                task_observations.extend(matching_observations)
            
            if task_observations: 

                # sort by observation start time
                task_observations.sort(key=lambda x: x['t_start']) 

                # check if any overlapping observations were found for this task; 
                # if so, merge to avoid repeated observations
                merged_observations = []
                for obs in task_observations:
                    if not merged_observations:
                        merged_observations.append(obs)
                    else:
                        merged = False
                        for prev_obs in reversed(merged_observations):
                            # check if this observation overlaps with the previous one
                            if obs['t_start'] <= prev_obs['t_end'] + 1e-9 and obs['agent name'] == prev_obs['agent name'] and obs['instrument'] == prev_obs['instrument']:
                                # if so, merge by extending the end time and updating n_obs and t_rev
                                prev_obs['t_end'] = max(prev_obs['t_end'], obs['t_end'])
                                prev_obs['n_obs'] = min(prev_obs['n_obs'], obs['n_obs'])
                                prev_obs['t_rev'] = min(prev_obs['t_rev'], obs['t_rev'])
                                merged = True
                            
                            if merged: break

                        if not merged:
                            merged_observations.append(obs)              

                # assign observations to this task in map of tasks observed
                # tasks_observed[task] = task_observations
                tasks_observed[task] = merged_observations

        # convert task observations data to dataframe
        task_observations_df = pd.DataFrame(task_observations_df_data)
        
        # return task observations dataframe and dictionary of tasks observed with their corresponding observations
        return task_observations_df, tasks_observed
    
    @staticmethod
    def __accesses_per_event_from_df(events : List[GeophysicalEvent], 
                                     accesses_per_event_df : pd.DataFrame
                                    ) -> Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]]:
        accesses_per_event = {}
        for event in events:
            # get accesses for this event from dataframe
            accesses : pd.DataFrame = accesses_per_event_df[accesses_per_event_df['event id'] == event.id]
            
            # compile information for all matching accesses into proper format
            access_tuples= [] # (interval,agent_name,instrument) 
            for _, access in accesses.iterrows():                
                interval = Interval(access['access start [s]'], access['access end [s]'])
                access_tuples.append((interval, access['agent name'], access['instrument']))

            # add to map of events to accesses
            if access_tuples: accesses_per_event[event] = access_tuples

        # return map of events to accesses
        return accesses_per_event
    
    @staticmethod
    def __accesses_per_task_from_df(tasks_known : List[GenericObservationTask], 
                                    accesses_per_task_df : pd.DataFrame
                                ) -> Dict[GenericObservationTask, List[Tuple[Interval, str, str]]]:
        accesses_per_task = {}
        for task in tasks_known:
            # get accesses for this task from dataframe
            accesses : pd.DataFrame = accesses_per_task_df[accesses_per_task_df['task id'] == task.id]
            
            # compile information for all matching accesses into proper format
            access_tuples= [] # (interval,agent_name,instrument) 
            for _, access in accesses.iterrows():                
                interval = Interval(access['t start'], access['t end'])
                access_tuples.append((interval, access['agent name'], access['instrument']))

            # add to map of tasks to accesses
            if access_tuples: accesses_per_task[task] = access_tuples

        # return map of tasks to accesses
        return accesses_per_task

    @staticmethod
    def __observations_per_event_from_df(events : List[GeophysicalEvent], 
                                        observations_per_event_df : pd.DataFrame
                                    ) -> Dict[GeophysicalEvent, List[dict]]:
        observations_per_event = {}
        for event in events:
            # get observations for this event from dataframe
            observations : pd.DataFrame = observations_per_event_df[observations_per_event_df['event id'] == event.id]
            
            # compile information for all matching observations into proper format
            observation_dicts= [] # (interval,agent_name,instrument) 
            for _, observation in observations.iterrows():                
                observation_dicts.append(dict(observation))
            
            # add to map of events to observations
            if observation_dicts: observations_per_event[event] = observation_dicts

        # return map of events to observations
        return observations_per_event
    
    @staticmethod
    def __observations_per_task_from_df(tasks : List[GenericObservationTask],
                                        observations_per_task_df : pd.DataFrame
                                    ) -> Dict[GenericObservationTask, List[dict]]:
        observations_per_task = {}
        for task in tasks:
            # get observations for this task from dataframe
            observations : pd.DataFrame = observations_per_task_df[observations_per_task_df['task id'] == task.id]
            
            # compile information for all matching observations into proper format
            observation_dicts= [] # (interval,agent_name,instrument) 
            for _, observation in observations.iterrows():                
                observation_dicts.append(dict(observation))
            
            # add to map of tasks to observations
            if observation_dicts: observations_per_task[task] = observation_dicts

        # return map of tasks to observations
        return observations_per_task

    """
    RESULTS SUMMARY METHODS
    """   
    @staticmethod
    def summarize_results(results_path : str,
                          compiled_orbitdata : Dict[str, OrbitData], 
                          events : List[GeophysicalEvent],
                          agent_specs : Dict[str, object],
                          agent_missions : Dict[str, Mission],
                          task_reqs : List[TaskRequest],
                          known_tasks : List[GenericObservationTask],
                          events_per_gp : Dict[Tuple[int,int], List[GeophysicalEvent]],
                          events_detected : List[GeophysicalEvent], 
                          events_requested : List[GeophysicalEvent], 
                          agent_broadcasts_df : pd.DataFrame, 
                          planned_rewards_df : pd.DataFrame, 
                          obtained_rewards_df : pd.DataFrame,
                          execution_costs_df : pd.DataFrame,
                          accesses_per_gp : Dict[Tuple[int,int], pd.DataFrame],
                          accesses_per_event : Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]],
                          accesses_per_task : Dict[GenericObservationTask, list],
                          observations_per_gp : Dict[Tuple[int,int], pd.DataFrame], 
                          observations_per_event : Dict[GeophysicalEvent, List[Dict]],
                          observations_per_task : Dict[GenericObservationTask, list],
                          precision : int = 5,
                          printouts : bool = True
                        ) -> pd.DataFrame:      

        # classify re-observations
        events_re_observable, events_re_obs \
            = ResultsProcessor.__classify_event_reobservations(accesses_per_event, observations_per_event)

        # clasify co-observations
        events_co_observable, events_co_observable_fully, events_co_observable_partially, \
            events_co_obs, events_co_obs_fully, events_co_obs_partially \
                = ResultsProcessor.__classify_event_coobservations(accesses_per_event, observations_per_event, known_tasks)            

        # count observations performed
        # n_events, n_unique_event_obs, n_total_event_obs,
        n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                n_tasks, n_total_task_obs, n_event_tasks, n_default_tasks, \
                                    n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                        n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                            n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved \
                                                    = ResultsProcessor.__count_observations(compiled_orbitdata, 
                                                                                            observations_per_gp,
                                                                                            events, 
                                                                                            events_per_gp,
                                                                                            accesses_per_event,
                                                                                            events_detected, 
                                                                                            events_requested,
                                                                                            observations_per_event, 
                                                                                            events_re_observable,
                                                                                            events_re_obs, 
                                                                                            events_co_observable,
                                                                                            events_co_obs, 
                                                                                            events_co_observable_fully,
                                                                                            events_co_obs_fully, 
                                                                                            events_co_observable_partially,
                                                                                            events_co_obs_partially,
                                                                                            known_tasks,
                                                                                            accesses_per_task,
                                                                                            observations_per_task,
                                                                                            printouts)
            
        # count probabilities of observations performed
        p_gp_accessible, p_gp_observed, p_gp_observed_if_accessible, p_event_at_gp, p_event_detected, \
            p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                    p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                        p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                            p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected, \
                                    p_task_observable, p_event_task_observable, p_default_task_observable, p_task_observed, p_event_task_observed, p_default_task_observed, \
                                        p_task_observed_if_observable, p_event_task_observed_if_observable, p_default_task_observed_if_observable, \
                                            p_task_reobserved, p_event_task_reobserved, p_default_task_reobserved, \
                                                p_task_reobserved_if_reobservable, p_event_task_reobserved_if_reobservable, p_default_task_reobserved_if_reobservable \
                                                    = ResultsProcessor.__calc_event_probabilities(compiled_orbitdata, 
                                                                                                  accesses_per_gp,
                                                                                                  observations_per_gp,
                                                                                                  events, 
                                                                                                  events_per_gp,
                                                                                                  accesses_per_event,
                                                                                                  events_detected, 
                                                                                                  events_requested,
                                                                                                  observations_per_event, 
                                                                                                  events_re_observable,
                                                                                                  events_re_obs, 
                                                                                                  events_co_observable,
                                                                                                  events_co_obs, 
                                                                                                  events_co_observable_fully,
                                                                                                  events_co_obs_fully, 
                                                                                                  events_co_observable_partially,
                                                                                                  events_co_obs_partially,
                                                                                                  known_tasks,
                                                                                                  accesses_per_task,
                                                                                                  observations_per_task,
                                                                                                  printouts)
        
        # calculate event revisit times
        t_gp_reobservation = ResultsProcessor.__calc_groundpoint_reobservation_metrics(observations_per_gp)
        t_event_reobservation = ResultsProcessor.__calc_event_reobservation_metrics(observations_per_event)
        t_task_reobservation = ResultsProcessor.__calc_task_reobservation_metrics(observations_per_task) 
        t_response_to_event = ResultsProcessor.__calc_response_time_metrics(observations_per_event)
        t_response_to_event_norm = ResultsProcessor.__calc_response_time_metrics_normalized(observations_per_event)
        t_response_to_task = ResultsProcessor.__calc_response_time_metrics(observations_per_task)
        t_response_to_task_norm = ResultsProcessor.__calc_response_time_metrics_normalized(observations_per_task)

        # calculate utility metrics
        total_planned_reward = np.round(planned_rewards_df['planned reward'].sum(), precision) if planned_rewards_df is not None else 0.0
        total_planned_utility = np.round(planned_rewards_df['planned reward'].sum() - execution_costs_df['cost'].sum(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0

        avg_planned_reward = np.round(planned_rewards_df['planned reward'].mean(), precision) if planned_rewards_df is not None else 0.0
        std_planned_reward = np.round(planned_rewards_df['planned reward'].std(), precision) if planned_rewards_df is not None else 0.0
        median_planned_reward = np.round(planned_rewards_df['planned reward'].median(), precision) if planned_rewards_df is not None else 0.0

        total_obtained_reward = np.round(obtained_rewards_df['reward'].sum(), precision) if obtained_rewards_df is not None else 0.0
        total_obtained_utility = np.round(obtained_rewards_df['reward'].sum() - execution_costs_df['cost'].sum(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0

        reward_primal_bound, reward_dual_bound \
            = ResultsProcessor.__calculate_reward_bounds(compiled_orbitdata, accesses_per_task, agent_specs, agent_missions, obtained_rewards_df, printouts)
        
        total_task_priority, total_observable_task_priority \
            = ResultsProcessor.__calculate_task_priorities(accesses_per_task)

        # validate reward values
        assert total_obtained_reward < reward_dual_bound or abs(total_obtained_reward - reward_dual_bound) <= 1e-6, \
            "Total obtained reward exceeds calculated reward dual bound. Please check reward calculations for errors."
        assert total_obtained_utility < reward_dual_bound or abs(total_obtained_utility - reward_dual_bound) <= 1e-6, \
            "Total obtained utility exceeds calculated reward dual bound. Please check reward and cost calculations for errors."

        # Generate summary
        summary_headers = ['Metric', 'Value']
        summary_data = [
                    # Counters
                    ['Events', n_events],
                    ['Events Observable', n_events_observable],
                    ['Events Observed', n_events_observed],
                    ['Events Detected', n_events_detected],
                    ['Events Requested', n_events_requested],

                    ['Event Observations', n_total_event_obs],
                    
                    ['Events Re-observable', n_events_reobservable],
                    ['Events Re-observed', n_events_reobserved],
                    ['Event Re-observations', n_total_event_re_obs],
                    
                    ['Events Co-observable', n_events_co_observable],
                    ['Events Co-observed', n_events_co_obs],
                    ['Event Co-observations', n_total_event_co_obs],
                    ['Events Fully Co-observable', n_events_co_observable_fully],
                    ['Events Fully Co-observed', n_events_fully_co_obs],
                    ['Event Full Co-observations', n_total_event_fully_co_obs],
                    ['Events Only Partially Co-observable', n_events_co_observable_partially],
                    ['Events Partially Co-observed', n_events_partially_co_obs],
                    ['Event Partial Co-observations', n_total_event_partially_co_obs],
                    
                    ['Tasks Available', n_tasks],
                    ['Event-Driven Tasks Available', n_event_tasks],
                    ['Default Mission Tasks Available', n_default_tasks],
                    
                    ['Tasks Observable', n_tasks_observable],
                    ['Event-Driven Tasks Observable', n_event_tasks_observable],
                    ['Default Mission Tasks Observable', n_default_tasks_observable],
                    
                    ['Tasks Observed', n_tasks_observed],
                    ['Task Observations', n_total_task_obs],
                    ['Event-Driven Tasks Observed', n_event_tasks_observed],
                    ['Default Mission Tasks Observed', n_default_tasks_observed],

                    # Coverage Metrics #TODO add more
                    ['Ground Points', n_gps],
                    ['Ground Points Accessible', n_gps_accessible],
                    ['Ground Points Observed', n_gps_observed],
                    ['Ground Points Reobserved', n_gps_reobserved],
                    ['Ground Point Observations', n_observations],
                    ['Ground Points with Events', n_gps_with_events],

                    ['Average GP Reobservation Time [s]', t_gp_reobservation['mean']],
                    ['Standard Deviation of GP Reobservation Time [s]', t_gp_reobservation['std']],
                    ['Median GP Reobservation Time [s]', t_gp_reobservation['median']],
                    
                    ['Average Event Reobservation Time [s]', t_event_reobservation['mean']],
                    ['Standard Deviation of Event Reobservation Time [s]', t_event_reobservation['std']],
                    ['Median Event Reobservation Time [s]', t_event_reobservation['median']],
                    
                    ['Average Task Reobservation Time [s]', t_task_reobservation['mean']],
                    ['Standard Deviation of Task Reobservation Time [s]', t_task_reobservation['std']],
                    ['Median Task Reobservation Time [s]', t_task_reobservation['median']],                    

                    # Ground-Point Coverage Probabilities
                    ['P(Ground Point Accessible)', np.round(p_gp_accessible,precision)],
                    ['P(Ground Point Observed)', np.round(p_gp_observed,precision)],
                    ['P(Ground Point Observed | Ground Point Accessible)', np.round(p_gp_observed_if_accessible,precision)],
                    ['P(Event at a GP)', np.round(p_event_at_gp,precision)],

                    # Event Observation Probabilities
                    # TODO add co-observation probabilities
                    ['P(Event Observable)', np.round(p_event_observable,precision)],
                    ['P(Event Re-observable)', np.round(p_event_re_observable,precision)],
                    ['P(Event Co-observable)', np.round(p_event_co_observable,precision)],
                    ['P(Event Fully Co-observable)', np.round(p_event_co_observable_fully,precision)],
                    ['P(Event Partially Co-observable)', np.round(p_event_co_observable_partial,precision)],
                    
                    ['P(Event Detected)', np.round(p_event_detected,precision)],
                    ['P(Event Observed)', np.round(p_event_observed,precision)],
                    ['P(Event Re-observed)', np.round(p_event_re_obs,precision)],
                    ['P(Event Co-observed)', np.round(p_event_co_obs,precision)],
                    ['P(Event Fully Co-observed)', np.round(p_event_co_obs_fully,precision)],
                    ['P(Event Partially Co-observed)', np.round(p_event_co_obs_partial,precision)],

                    ['P(Event Observation | Observation)', np.round(p_event_obs_if_obs,precision)],
                    ['P(Event Re-observation | Observation)', np.round(p_event_re_obs_if_obs,precision)],
                    # ['P(Event Co-observation | Observation)', np.round(p_event_co_obs_if_obs,precision)],
                    # ['P(Event Full Co-observation | Observation)', np.round(p_event_co_obs_partially_if_obs,precision)],
                    # ['P(Event Partial Co-observation | Observation)', np.round(p_event_co_obs_fully_if_obs,precision)],

                    ['P(Event Observed | Observable)', np.round(p_event_observed_if_observable,precision)],
                    ['P(Event Re-observed | Re-observable)', np.round(p_event_re_obs_if_re_observable,precision)],
                    ['P(Event Co-observed | Co-observable)', np.round(p_event_co_obs_if_co_observable,precision)],
                    ['P(Event Fully Co-observed | Fully Co-observable)', np.round(p_event_co_obs_fully_if_co_observable_fully,precision)],
                    ['P(Event Partially Co-observed | Partially Co-observable)', np.round(p_event_co_obs_partial_if_co_observable_partially,precision)],
                    
                    ['P(Event Observed | Event Detected)', np.round(p_event_observed_if_detected,precision)],
                    ['P(Event Re-observed | Event Detected)', np.round(p_event_re_obs_if_detected,precision)],
                    ['P(Event Co-observed | Event Detected)', np.round(p_event_co_obs_if_detected,precision)],
                    ['P(Event Co-observed Fully | Event Detected)', np.round(p_event_co_obs_fully_if_detected,precision)],
                    ['P(Event Co-observed Partially | Event Detected)', np.round(p_event_co_obs_partial_if_detected,precision)],

                    ['P(Event Observed | Event Observable and Detected)', np.round(p_event_observed_if_detected,precision)],
                    ['P(Event Re-observed | Event Re-observable and Detected)', np.round(p_event_re_obs_if_detected,precision)],
                    ['P(Event Co-observed | Event Co-observable and Detected)', np.round(p_event_co_obs_if_detected,precision)],
                    ['P(Event Co-observed Fully | Event Fully Co-observable and Detected)', np.round(p_event_co_obs_fully_if_detected,precision)],
                    ['P(Event Co-observed Partially | Event Partially Co-observable and Detected)', np.round(p_event_co_obs_partial_if_detected,precision)],

                    # Task Observation Probabilities
                    # TODO add co-observation probabilities
                    ['P(Task Observable)', np.round(p_task_observable,precision)],
                    ['P(Task Observed)', np.round(p_task_observed,precision)],
                    ['P(Task Observed | Task Observable)', np.round(p_task_observed_if_observable,precision)],
                    ['P(Task Reobserved)', np.round(p_task_reobserved,precision)],
                    ['P(Task Reobserved | Task Reobservable)', np.round(p_task_reobserved_if_reobservable,precision)],
                    
                    ['P(Event-Driven Task Observable)', np.round(p_event_task_observable,precision)],
                    ['P(Event-Driven Task Observed)', np.round(p_event_task_observed,precision)],
                    ['P(Event-Driven Task Observed | Event-Driven Task Observable)', np.round(p_event_task_observed_if_observable,precision)],
                    ['P(Event-Driven Task Reobserved)', np.round(p_event_task_reobserved,precision)],
                    ['P(Event-Driven Task Reobserved | Event-Driven Task Reobservable)', np.round(p_event_task_reobserved_if_reobservable,precision)],
                    
                    ['P(Default Mission Task Observable)', np.round(p_default_task_observable,precision)],
                    ['P(Default Mission Task Observed)', np.round(p_default_task_observed,precision)],
                    ['P(Default Mission Task Observed | Default Mission Task Observable)', np.round(p_default_task_observed_if_observable,precision)],
                    ['P(Default Mission Task Reobserved)', np.round(p_default_task_reobserved,precision)],
                    ['P(Default Mission Task Reobserved | Default Mission Task Reobservable)', np.round(p_default_task_reobserved_if_reobservable,precision)],
                    
                    # Messaging Statistics
                    ['Total Messages Broadcasted', len(agent_broadcasts_df)],
                    ['P(Message Broadcasted | Bid Message )', len(agent_broadcasts_df[agent_broadcasts_df['msg_type']=='BUS']) / len(agent_broadcasts_df) if len(agent_broadcasts_df) > 0 else 0.0],
                    ['P(Message Broadcasted | Measurement Request Message )', len(agent_broadcasts_df[agent_broadcasts_df['msg_type']=='MEASUREMENT_REQ']) / len(agent_broadcasts_df) if len(agent_broadcasts_df) > 0 else 0.0],

                    # Response Time Statistics
                    ['Average Response Time to Event [s]', t_response_to_event['mean']],
                    ['Standard Deviation of Response Time to Event [s]', t_response_to_event['std']],
                    ['Median Response Time to Event [s]', t_response_to_event['median']],

                    ['Average Normalized Response Time to Event', t_response_to_event_norm['mean']],
                    ['Standard Deviation of Normalized Response Time to Event', t_response_to_event_norm['std']],
                    ['Median Normalized Response Time to Event', t_response_to_event_norm['median']],
                    
                    ['Average Response Time to Task [s]', t_response_to_task['mean']],
                    ['Standard Deviation of Response Time to Task [s]', t_response_to_task['std']],
                    ['Median Response Time to Task [s]', t_response_to_task['median']],

                    ['Average Normalized Response Time to Task', t_response_to_task_norm['mean']],
                    ['Standard Deviation of Normalized Response Time to Task', t_response_to_task_norm['std']],
                    ['Median Normalized Response Time to Task', t_response_to_task_norm['median']],

                    # Reward Statistics 
                    ['Total Planned Reward', total_planned_reward],
                    # ['Normalized Total Planned Reward', total_planned_reward / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Total Planned Task Observations', len(planned_rewards_df) if planned_rewards_df is not None else 0],

                    ['Total Obtained Reward', total_obtained_reward],
                    # ['Normalized Total Obtained Reward', total_obtained_reward / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Total Obtained Task Observations', len(obtained_rewards_df) if obtained_rewards_df is not None else 0],
                    
                    ['Average Planned Reward per Task Observation', avg_planned_reward],
                    ['Standard Deviation of Planned Reward per Task Observation', std_planned_reward],
                    ['Median Planned Reward per Task Observation', median_planned_reward],
                    
                    ['Average Planned Reward per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().mean(), precision) if planned_rewards_df is not None else 0.0],
                    ['Standard Deviation of Planned Reward per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().std(), precision) if planned_rewards_df is not None else 0.0],
                    ['Median Planned Reward per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().median(), precision) if planned_rewards_df is not None else 0.0],

                    # Cost Statistics
                    ['Total Execution Cost', np.round(execution_costs_df['cost'].sum(), precision) if execution_costs_df is not None else 0.0],
                    ['Average Execution Cost per Agent', np.round(execution_costs_df.groupby('agent')['cost'].sum().mean(), precision) if execution_costs_df is not None else 0.0],
                    ['Standard Deviation of Execution Cost per Agent', np.round(execution_costs_df.groupby('agent')['cost'].sum().std(), precision) if execution_costs_df is not None else 0.0],
                    ['Median Execution Cost per Agent', np.round(execution_costs_df.groupby('agent')['cost'].sum().median(), precision) if execution_costs_df is not None else 0.0],

                    # Utility Statistics
                    ['Total Planned Utility', total_planned_utility],
                    # ['Normalized Total Planned Utility', total_planned_utility / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Average Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().mean() - execution_costs_df.groupby('agent')['cost'].sum().mean(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Standard Deviation of Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().std() - execution_costs_df.groupby('agent')['cost'].sum().std(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Median Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().median() - execution_costs_df.groupby('agent')['cost'].sum().median(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],

                    ['Total Obtained Utility', total_obtained_utility],
                    # ['Normalized Total Obtained Utility', total_obtained_utility / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Average Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().mean() - execution_costs_df.groupby('agent')['cost'].sum().mean(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Standard Deviation of Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().std() - execution_costs_df.groupby('agent')['cost'].sum().std(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Median Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().median() - execution_costs_df.groupby('agent')['cost'].sum().median(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],

                    # Available Reward and Utility Statistics
                    ['Total Task Priority Available', total_task_priority],
                    ['Total Observable Task Priority', total_observable_task_priority],
                    ['Task Reward Primal Bound', reward_primal_bound],
                    ['Task Reward Dual Bound', reward_dual_bound],

                    # Results dir
                    # ['Results Directory', results_path]
                ]

        return pd.DataFrame(summary_data, columns=summary_headers)    

    @staticmethod
    def __classify_event_reobservations(
            accesses_per_event : Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]],
            observations_per_event : Dict[GeophysicalEvent, List[Dict]]
        ) -> Tuple[Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]], Dict[GeophysicalEvent, List[Dict]]]:
        # classify re-observations
        events_re_observable = {event : accesses for event, accesses in accesses_per_event.items()
                                 if len(accesses) > 1}
        events_re_obs = {event : observations for event, observations in observations_per_event.items()
                            if len(observations) > 1}

        return events_re_observable, events_re_obs

    @staticmethod
    def __classify_event_coobservations(
            accesses_per_event : Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]],
            observations_per_event : Dict[GeophysicalEvent, List[Dict]],
            known_tasks : List[GenericObservationTask],
            t_corr : float = 3600.0 # TODO temp solution
        ) -> Tuple[Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]], Dict[GeophysicalEvent, List[Dict]]]:
        # TODO define co-observation requirement in `execsatm` and use that to classify co-observations rather than hardcoding a decorrelation time threshold as is done here

        # map events to tasks that observe them
        event_to_task : Dict[GeophysicalEvent, GenericObservationTask] = dict()
        for task in known_tasks:
            if isinstance(task, EventObservationTask):
                event = task.event
                if event in event_to_task:
                    raise NotImplementedError(f"Multiple tasks found for event {event.id}. Co-observability classification not yet supported for multiple tasks per event.")
                event_to_task[event] = task
        
        # map events to required obserations for their tasks
        event_to_required_observations : Dict[GeophysicalEvent, Set[Dict]] = defaultdict(set)
        for event, task in event_to_task.items():
            # initialize set of required observations
            required_observations = set()

            for req in task.objective:
                # check for instrument capability requirements
                if isinstance(req, ExplicitCapabilityRequirement) and req.attribute == 'instrument':
                    for val in req.valid_values: required_observations.add(val)
                # TODO include other capability requirements that define a co-observation here

            # add to map of events to required observations
            event_to_required_observations[event] = required_observations

        # classify accesses
        co_observation_acesses = defaultdict(list)
        for event, observations in accesses_per_event.items():
            for a in observations:
                # unpack access
                *_,instrument = a

                # check if instrument is part of those required for this event 
                if instrument.lower() in event_to_required_observations[event]:
                    co_observation_acesses[event].append(a)

            # sort accesses by start time
            co_observation_acesses[event].sort(key=lambda x: x[0].left) 
        
        # initiate event co-observability sets
        events_co_observable = set()
        events_co_observable_fully = set()
        events_co_observable_partially = set()

        # evaluate events co-observable based on accesses
        for event, observations in co_observation_acesses.items():
            # check if more than one type of observation is required for this event
            if len(event_to_required_observations[event]) < 2: 
                # only one or no observation types required;
                #  co-observability classification not applicable
                continue
            
            # initiate group 
            # best_co_obs_group = []
            best_instruments_in_group = set()

            for i,a in enumerate(observations):
                # get accesses within the desired decorreleation time 
                co_obs_group = [b for b in observations[i:]
                                if b[0].left <= a[0].right + t_corr]

                # get instruments in this co-observation group
                instruments_in_group = {a[2].lower() for a in co_obs_group} | {a[2].lower()}
                
                # check if this group is the best so far
                if len(instruments_in_group) > len(best_instruments_in_group):
                    # best_co_obs_group = co_obs_group
                    best_instruments_in_group = instruments_in_group
            
            # check if more than one instrument is available to observe this event
            if len(best_instruments_in_group) > 1:
                # add to set of co-observable events
                events_co_observable.add(event)

                # classify co-observability of this event based on best co-observation group
                if event_to_required_observations[event] == best_instruments_in_group:
                    events_co_observable_fully.add(event)
                else:
                    events_co_observable_partially.add(event)

        # classify observations
        possible_co_observations = defaultdict(list)
        for event, observations in observations_per_event.items():
            for a in observations:
                # check if instrument is part of those required for this event 
                if a['instrument'].lower() in event_to_required_observations[event]:
                    possible_co_observations[event].append(a)

            # sort accesses by start time
            possible_co_observations[event].sort(key=lambda x: x['t_start'])

        # initiate event co-observation sets
        events_co_obs = set()
        events_co_obs_fully = set()
        events_co_obs_partially = set()  

        # evaluate events co-observable based on accesses
        for event, observations in possible_co_observations.items():
            # check if more than one type of observation is required for this event
            if len(event_to_required_observations[event]) < 2: 
                # only one or no observation types required;
                #  co-observability classification not applicable
                continue

            # initiate group 
            # best_co_obs_group = []
            best_instruments_in_group = set()

            for i,a in enumerate(observations):
                # get accesses within the desired decorreleation time 
                co_obs_group = [b for b in observations[i:]
                                if b['t_start'] <= a['t_end'] + t_corr]

                # get instruments in this co-observation group
                instruments_in_group = {a['instrument'].lower() for a in co_obs_group} | {a['instrument'].lower()}

                # check if this group is the best so far
                if len(instruments_in_group) > len(best_instruments_in_group):
                    # best_co_obs_group = co_obs_group
                    best_instruments_in_group = instruments_in_group
            

            # check if more than one instrument was involved in observations of this event
            if len(best_instruments_in_group) > 1:
                # add to set of co-observed events
                events_co_obs.add(event)

                # classify co-observability of this event based on best co-observation group
                if event_to_required_observations[event] == best_instruments_in_group:
                    events_co_obs_fully.add(event)
                else:
                    events_co_obs_partially.add(event)

        return events_co_observable, events_co_observable_fully, events_co_observable_partially, \
            events_co_obs, events_co_obs_fully, events_co_obs_partially

    @staticmethod
    def __count_observations(orbitdata : Dict[str, OrbitData], 
                            observations_per_gp : dict,
                            events : List[GeophysicalEvent],
                            events_per_gp : dict,
                            events_observable : dict,
                            events_detected : dict, 
                            events_requested : dict,
                            events_observed : dict, 
                            events_re_observable : dict,
                            events_re_obs : dict, 
                            events_co_observable : dict,
                            events_co_obs : dict, 
                            events_co_observable_fully : dict,
                            events_co_obs_fully : dict, 
                            events_co_observable_partially : dict,
                            events_co_obs_partially : dict,
                            tasks_known : list,
                            tasks_observable : dict,
                            tasks_observed : dict,
                            printouts : bool = True
                        ) -> tuple:
        # get a representative agent orbitdata for time and ground point information
        agent_orbitdata : OrbitData = next(iter(orbitdata.values()))

        # count number of groundpoints and their accessibility
        n_gps = None
        gps_accessible_compiled = set()
        for _, agent in tqdm(orbitdata.items(), 
                             desc='Counting total and accessible ground points',
                             leave=False, 
                             disable=not printouts
                            ):

            if n_gps is None: n_gps = len(agent.grid_data)

            tab = agent.gp_access_data
            grid = tab._grid_idx.astype(np.int64, copy=False)
            gp   = tab._gp_idx.astype(np.int64, copy=False)

            pairs = np.empty(grid.shape[0], dtype=[('g', np.int64), ('p', np.int64)])
            pairs['g'] = grid
            pairs['p'] = gp

            upairs = np.unique(pairs)
            gps_accessible_compiled.update(zip(upairs['g'].tolist(), upairs['p'].tolist()))

        n_gps_accessible = len(gps_accessible_compiled)
        n_gps_observed = len(observations_per_gp)

        # count number of groun point reobservations
        n_gps_reobserved = len([gp for gp,observations in observations_per_gp.items() 
                                if len(observations) > 1])
        
        # count number of observations performed
        n_observations = sum(len(observations) for observations in observations_per_gp.values())
        
        # count number of events
        n_events = len(events)

        # count number of groundpoints with events
        n_gps_with_events = len(events_per_gp)

        # count events observable
        n_events_observable = len(events_observable)

        # count event detections
        n_events_detected = len(events_detected)

        # count events with meaurement requests
        n_events_requested = len(events_requested)

        # count event observations
        n_events_observed = len(events_observed)
        n_total_event_obs = sum([len(observations) for _,observations in events_observed.items()])
        
        assert n_total_event_obs <= n_observations

        # count events reobservable
        n_events_reobservable = len(events_re_observable)

        # count event reobservations
        n_events_reobserved = len(events_re_obs)
        n_total_event_re_obs = sum([len(re_observations) for _,re_observations in events_re_obs.items()])

        assert n_events_reobserved <= n_events_observed
        assert n_total_event_re_obs <= n_observations

        # count events co-observable
        n_events_co_observable = len(events_co_observable)
        n_events_co_observable_fully = len(events_co_observable_fully)
        n_events_co_observable_partially = len(events_co_observable_partially)

        # count event co-observations
        n_events_co_obs = len(events_co_obs)
        # n_total_event_co_obs = sum([len(co_observations) for _,co_observations in events_co_obs.items()])        
        n_total_event_co_obs = -1 # TODO

        assert n_events_co_obs <= n_events_observed
        assert n_total_event_co_obs <= n_observations

        n_events_fully_co_obs = len(events_co_obs_fully)
        # n_total_event_fully_co_obs = sum([len(full_co_observations) for _,full_co_observations in events_co_obs_fully.items()])        
        n_total_event_fully_co_obs = -1 # TODO
        
        assert n_events_fully_co_obs <= n_events_observed
        assert n_total_event_fully_co_obs <= n_total_event_co_obs

        n_events_partially_co_obs = len(events_co_obs_partially)
        # n_total_event_partially_co_obs = sum([len(partial_co_observations) for _,partial_co_observations in events_co_obs_partially.items()])        
        n_total_event_partially_co_obs = -1 # TODO

        assert n_events_partially_co_obs <= n_events_observed
        assert n_total_event_partially_co_obs <= n_total_event_co_obs

        assert n_events_co_obs == n_events_fully_co_obs + n_events_partially_co_obs
        # assert n_total_event_co_obs == n_total_event_fully_co_obs + n_total_event_partially_co_obs

        # count observations per task
        n_tasks = len(tasks_known)
        n_total_task_obs = sum(len(observations) for observations in tasks_observed.values())
        n_event_tasks = len([task for task in tasks_known if isinstance(task, EventObservationTask)])
        n_default_tasks = len([task for task in tasks_known if isinstance(task, DefaultMissionTask)]) 

        assert n_event_tasks + n_default_tasks <= n_tasks

        n_tasks_observable = len(tasks_observable)
        n_event_tasks_observable = len([task for task in tasks_observable if isinstance(task, EventObservationTask)])
        n_default_tasks_observable = len([task for task in tasks_observable if isinstance(task, DefaultMissionTask)])

        assert n_tasks_observable <= n_tasks        
        
        n_tasks_observed = len(tasks_observed)
        n_event_tasks_observed = len([task for task in tasks_observed if isinstance(task, EventObservationTask)])
        n_default_tasks_observed = len([task for task in tasks_observed if isinstance(task, DefaultMissionTask)])
        
        assert n_tasks_observed <= n_tasks_observable
        assert n_event_tasks_observed + n_default_tasks_observed <= n_tasks_observed

        # count reobservations per task
        n_tasks_reobservable = len([task for task,access_intervals in tasks_observable.items() 
                                    if len(access_intervals) > 1 
                                    or any(access_interval.span() > agent_orbitdata.time_step for access_interval,*_ in access_intervals)])
        n_event_tasks_reobservable = len([task for task,access_intervals in tasks_observable.items() 
                                            if isinstance(task, EventObservationTask)
                                            and (len(access_intervals) > 1
                                                or any(access_interval.span() > agent_orbitdata.time_step*2
                                                       for access_interval,*_ in access_intervals))
                                        ])
        n_default_tasks_reobservable = len([task for task,access_intervals in tasks_observable.items() 
                                            if isinstance(task, DefaultMissionTask)
                                            and (len(access_intervals) > 1
                                                or any(access_interval.span() > agent_orbitdata.time_step*2
                                                       for access_interval,*_ in access_intervals))
                                        ])
        
        assert n_tasks_reobservable <= n_tasks_observable
        assert n_event_tasks_reobservable + n_default_tasks_reobservable <= n_tasks_reobservable

        n_tasks_reobserved = len([task for task,observations in tasks_observed.items() 
                                if len(observations) > 1])
        n_event_tasks_reobserved = len([task for task in tasks_observed if isinstance(task, EventObservationTask)
                                        and len(tasks_observed[task]) > 1])
        n_default_tasks_reobserved = len([task for task in tasks_observed if isinstance(task, DefaultMissionTask)
                                        and len(tasks_observed[task]) > 1])
        
        # --- DEBUG BREAKPOINTS ---
        for task in tasks_observable:
            task_accesses = tasks_observable[task]
            task_observations = tasks_observed[task] if task in tasks_observed else []

            if len(task_accesses) < len(task_observations):
                x = 1
        # -------------------------


        assert n_tasks_reobserved <= n_tasks_reobservable 
        assert n_tasks_reobserved <= n_tasks_observed
        assert n_event_tasks_reobserved + n_default_tasks_reobserved <= n_tasks_reobserved

        # return values
        return n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
                n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                    n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                        n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                            n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                                n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                    n_tasks, n_total_task_obs, n_event_tasks, n_default_tasks, \
                                        n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                            n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                                n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                    n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved

    @staticmethod
    def __calc_event_probabilities(
                                    orbitdata : dict, 
                                    gps_accessible : dict,
                                    observations_per_gp : dict,
                                    events : pd.DataFrame,
                                    events_per_gp : dict,
                                    events_observable : dict,
                                    events_detected : dict, 
                                    events_requested : dict,
                                    events_observed : dict, 
                                    events_re_observable : dict,
                                    events_re_obs : dict, 
                                    events_co_observable : set,
                                    events_co_obs : set, 
                                    events_co_observable_fully : set,
                                    events_co_obs_fully : set, 
                                    events_co_observable_partially : set,
                                    events_co_obs_partially : set,
                                    tasks_known : list,
                                    tasks_observable : dict,
                                    tasks_observed : dict,
                                    printouts : bool = True
                                ) -> tuple:

        # count observations by type
        n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                n_tasks, n_total_task_obs, n_event_tasks, n_default_tasks, \
                                    n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                        n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                            n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved \
                                                    = ResultsProcessor.__count_observations( orbitdata,                                                                                 
                                                                                observations_per_gp,
                                                                                events, 
                                                                                events_per_gp,
                                                                                events_observable,
                                                                                events_detected, 
                                                                                events_requested,
                                                                                events_observed, 
                                                                                events_re_observable,
                                                                                events_re_obs, 
                                                                                events_co_observable,
                                                                                events_co_obs, 
                                                                                events_co_observable_fully,
                                                                                events_co_obs_fully, 
                                                                                events_co_observable_partially,
                                                                                events_co_obs_partially,
                                                                                tasks_known,
                                                                                tasks_observable,
                                                                                tasks_observed,
                                                                                printouts)
                    
        # count number of ground points accessible and observed 
        n_gps_observed_and_accessible = len(observations_per_gp)
        n_gps = n_gps if n_gps is not None else 0

        # count number of obseved and detected events
        n_events_observed_and_observable = len([event for event in events_observed
                                                if event in events_observable])
        n_events_observed_and_detected = len([event for event in events_observed
                                                if event in events_detected])
        n_events_observed_and_observable_and_detected = len([event for event in events_observed
                                                            if event in events_observable
                                                            and event in events_detected])
        n_events_observable_and_detected = len([event for event in events_observable
                                                if event in events_detected])
            
        # count number of re-obseved and events
        n_events_re_obs_and_reobservable = len([event for event in events_re_obs
                                                if event in  events_re_observable])
        n_events_re_obs_and_detected = len([event for event in events_re_obs
                                            if event in events_detected])
        n_events_re_obs_and_reobservable_and_detected = len([event for event in events_re_obs
                                                            if event in  events_re_observable
                                                            and event in events_detected])
        n_events_re_observable_and_detected = len([event for event in events_re_observable
                                                    if event in events_detected])

        # count number of co-observed and detected events 
        n_events_co_obs_and_co_observable \
            = len(events_co_obs.intersection(events_co_observable))
        n_events_co_obs_and_detected \
            = len(events_co_obs.intersection(events_detected))         
        n_events_co_obs_and_co_observable_and_detected \
            = len(events_co_obs.intersection(events_co_observable).intersection(events_detected))
        n_events_co_observable_and_detected \
            = len(events_co_observable.intersection(events_detected))
        
        n_events_fully_co_obs_and_fully_co_observable = len([event for event in events_co_obs_fully
                                                            if event in events_co_observable_fully])
        n_events_fully_co_obs_and_detected = len([event for event in events_co_obs_fully
                                                    if event in events_detected])
        n_events_fully_co_obs_and_fully_co_observable_and_detected = len([event for event in events_co_obs_fully
                                                                            if event in events_co_observable_fully
                                                                            and event in events_detected])
        n_events_fully_co_observable_and_detected = len([event for event in events_co_observable_fully
                                                        if event in events_detected])
        
        n_events_partially_co_obs_and_partially_co_observable = len([event for event in events_co_obs_partially
                                                                    if event in events_co_observable_partially])
        n_events_partially_co_obs_and_detected = len([event for event in events_detected
                                                    if event in events_co_obs_partially])
        n_events_partially_co_obs_and_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                                            if event in events_co_obs_partially
                                                                            and event in events_detected])
        n_events_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                            if event in events_detected])

        # calculate event probabilities
        p_gp_accessible = n_gps_accessible / n_gps if n_gps > 0 else np.NAN
        p_gp_observed = n_gps_observed / n_gps if n_gps > 0 else np.NAN
        p_event_at_gp = n_gps_with_events / n_gps if n_gps > 0 else np.NAN
        p_event_observable = n_events_observable / n_events if n_events > 0 else np.NAN
        p_event_detected = n_events_detected / n_events if n_events > 0 else np.NAN
        p_event_observed = n_events_observed / n_events if n_events > 0 else np.NAN
        p_event_re_observable = n_events_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs = n_events_reobserved / n_events if n_events > 0 else np.NAN
        p_event_co_observable = n_events_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs = n_events_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_fully = n_events_co_observable_fully / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully = n_events_fully_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_partial = n_events_co_observable_partially / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial = n_events_partially_co_obs / n_events if n_events > 0 else np.NAN    

        # calculate event joint probabilities
        p_gp_observed_and_accessible = p_gp_observed

        p_event_observed_and_observable = n_events_observed_and_observable / n_events if n_events > 0 else np.NAN
        p_event_observed_and_detected = n_events_observed_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observed_and_observable_and_detected = n_events_observed_and_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observable_and_detected = n_events_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_re_obs_and_reobservable = n_events_re_obs_and_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_detected = n_events_re_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_reobservable_and_detected = n_events_re_obs_and_reobservable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_observable_and_detected = n_events_re_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_and_co_observable = n_events_co_obs_and_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_detected = n_events_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_co_observable_and_detected = n_events_co_obs_and_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_observable_and_detected = n_events_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_fully_and_co_observable_fully = n_events_fully_co_obs_and_fully_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully_and_detected = n_events_fully_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_obs_and_fully_co_observable_and_detected = n_events_fully_co_obs_and_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_observable_and_detected = n_events_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_partially_and_co_observable_partially = n_events_partially_co_obs_and_partially_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial_and_detected = n_events_partially_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_obs_and_partially_co_observable_and_detected = n_events_partially_co_obs_and_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_observable_and_detected = n_events_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        # calculate event conditional probabilities
        p_gp_observed_if_accessible = p_gp_observed_and_accessible / p_gp_accessible if p_gp_accessible > 0.0 else np.NAN

        p_event_obs_if_obs = n_total_event_obs / n_observations if n_observations > 0 else np.NAN
        p_event_re_obs_if_obs = n_total_event_re_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_if_obs = n_total_event_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_fully_if_obs = n_total_event_fully_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_partially_if_obs = n_total_event_partially_co_obs / n_observations if n_observations > 0 else np.NAN
        
        p_event_observed_if_observable = p_event_observed_and_observable / p_event_observable if p_event_observable > 0.0 else np.NAN
        p_event_observed_if_detected = p_event_observed_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_observed_if_observable_and_detected = p_event_observed_and_observable_and_detected / p_event_observable_and_detected if p_event_observable_and_detected > 0.0 else np.NAN
        
        p_event_re_obs_if_re_observable = p_event_re_obs_and_reobservable / p_event_re_observable if p_event_re_observable > 0.0 else np.NAN
        p_event_re_obs_if_detected = p_event_re_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_re_obs_if_reobservable_and_detected = p_event_re_obs_and_reobservable_and_detected / p_event_re_observable_and_detected if p_event_re_observable_and_detected > 0.0 else np.NAN
        
        p_event_co_obs_if_co_observable = p_event_co_obs_and_co_observable / p_event_co_observable if p_event_co_observable > 0.0 else np.NAN
        p_event_co_obs_if_detected = p_event_co_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_if_co_observable_and_detected = p_event_co_obs_and_co_observable_and_detected / p_event_co_observable_and_detected if p_event_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_fully_if_co_observable_fully = p_event_co_obs_fully_and_co_observable_fully / p_event_co_observable_fully if p_event_co_observable_fully > 0.0 else np.NAN
        p_event_co_obs_fully_if_detected = p_event_co_obs_fully_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_fully_if_co_observable_fully_and_detected = p_event_fully_co_obs_and_fully_co_observable_and_detected / p_event_fully_co_observable_and_detected if p_event_fully_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_partial_if_co_observable_partially = p_event_co_obs_partially_and_co_observable_partially / p_event_co_observable_partial if p_event_co_observable_partial > 0.0 else np.NAN
        p_event_co_obs_partial_if_detected = p_event_co_obs_partial_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_partial_if_co_observable_partially_and_detected = p_event_partially_co_obs_and_partially_co_observable_and_detected / p_event_partially_co_observable_and_detected if p_event_partially_co_observable_and_detected > 0 else np.NAN

        # calculate task propabilites
        p_task_observable = n_tasks_observable / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_observable = n_event_tasks_observable / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_observable = n_default_tasks_observable / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_observed = n_tasks_observed / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_observed = n_event_tasks_observed / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_observed = n_default_tasks_observed / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_observed_if_observable = n_tasks_observed / n_tasks_observable if n_tasks_observable > 0 else np.NAN
        p_event_task_observed_if_observable = n_event_tasks_observed / n_event_tasks_observable if n_event_tasks_observable > 0 else np.NAN
        p_default_task_observed_if_observable = n_default_tasks_observed / n_default_tasks_observable if n_default_tasks_observable > 0 else np.NAN

        # calculate task reobservation probabilities
        p_task_reobserved = n_tasks_reobserved / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_reobserved = n_event_tasks_reobserved / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_reobserved = n_default_tasks_reobserved / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_reobserved_if_reobservable = n_tasks_reobserved / n_tasks_reobservable if n_tasks_reobservable > 0 else np.NAN
        p_event_task_reobserved_if_reobservable = n_event_tasks_reobserved / n_event_tasks_reobservable if n_event_tasks_reobservable > 0 else np.NAN
        p_default_task_reobserved_if_reobservable = n_default_tasks_reobserved / n_default_tasks_reobservable if n_default_tasks_reobservable > 0 else np.NAN

        return p_gp_accessible, p_gp_observed, p_gp_observed_if_accessible, p_event_at_gp, p_event_detected, \
                p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                    p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                        p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                            p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                                p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                    p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected, \
                                        p_task_observable, p_event_task_observable, p_default_task_observable, p_task_observed, p_event_task_observed, p_default_task_observed, p_task_observed_if_observable, p_event_task_observed_if_observable, p_default_task_observed_if_observable, \
                                            p_task_reobserved, p_event_task_reobserved, p_default_task_reobserved, p_task_reobserved_if_reobservable, p_event_task_reobserved_if_reobservable, p_default_task_reobserved_if_reobservable

    @staticmethod
    def __calc_groundpoint_reobservation_metrics(observations_per_gp: Dict[tuple, pd.DataFrame]) -> tuple:
        # event reobservation times
        t_reobservations : list = []
        for _,observations in observations_per_gp.items():
            prev_observation = None

            for _,observation in observations.iterrows():
                if prev_observation is None:
                    prev_observation = observation
                    continue

                # get observation times
                t_start = observation['t_start']
                t_prev_end = prev_observation['t_end']

                # calculate revisit
                t_reobservation = t_start - t_prev_end

                # add to list
                t_reobservations.append(t_reobservation)

                # update previous observation
                prev_observation = observation
        
        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation

    @staticmethod
    def __calc_event_reobservation_metrics(events_observed : dict) -> tuple:
        t_reobservations : list = []
        for event_observations in events_observed.values():
            for observation in event_observations[1:]:
                # get revisit
                t_reobservation = observation['t_rev']

                # add to list
                t_reobservations.append(t_reobservation)

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation
    
    @staticmethod
    def __calc_task_reobservation_metrics(observations_per_task: dict) -> dict:
        t_reobservations : list = []
        for task_observations in observations_per_task.values():
            for observation in task_observations[1:]:
                # get revisit
                t_reobservation = observation['t_rev']

                # add to list
                t_reobservations.append(t_reobservation)

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation

    @staticmethod
    def __calc_response_time_metrics(observations_map: dict) -> dict:
        t_reobservations : list = []
        for observations in observations_map.values():
            for observation in observations[1:]:
                # get revisit
                t_reobservation = observation['resp time [s]']

                # add to list
                t_reobservations.append(t_reobservation)

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation
    
    @staticmethod
    def __calc_response_time_metrics_normalized(observations_map: dict) -> dict:
        t_reobservations : list = []
        for observations in observations_map.values():
            for observation in observations[1:]:
                # get revisit
                t_reobservation = observation['resp time [norm]']

                # add to list
                t_reobservations.append(t_reobservation)

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation  
    
    @staticmethod
    def __calculate_reward_bounds(compiled_orbitdata : Dict[str, OrbitData], 
                                  accesses_per_task: Dict[GenericObservationTask, List], 
                                  agent_specs: Dict[str, object],
                                  agent_missions: Dict[str, Mission],
                                  obtained_rewards_df : pd.DataFrame,
                                  printouts : bool
                                ) -> Tuple[float, float]:
        # extract agent field of view specifications from agent specs object
        cross_track_fovs = {agent : ResultsProcessor._collect_fov_specs(specs) 
                            for agent,specs in agent_specs.items()}

        # calculate observation opportunities from accesses
        atomic_observation_opps : Dict[str, List[ObservationOpportunity]] \
             = ResultsProcessor.single_task_observation_opportunity_from_accesses(
                    compiled_orbitdata, accesses_per_task, cross_track_fovs
                )
        
        # define cluster threshold for tasks
        threshold = 5*60 # seconds
        # threshold = max(task.availability.span() for task in accesses_per_task.keys()) / 2 if accesses_per_task else threshold

        # cluster observation opportunities across agents for each task
        clustered_observation_opps : Dict[GenericObservationTask, List[ObservationOpportunity]] = {}
        for agent_name, observation_opps in atomic_observation_opps.items():        
            # check if observation opportunities are clusterable
            obs_adjacency : Dict[str, set[ObservationOpportunity]] \
                = ResultsProcessor.__check_task_observation_opportunity_clusterability(observation_opps, False, threshold, printouts)
    
            # cluster observation opportunities based on adjacency
            clustered_opps : list[ObservationOpportunity] \
                = ResultsProcessor.cluster_task_observation_opportunities(observation_opps, obs_adjacency, False, printouts)

            # add clustered observation opportunities to the final list of observation opportunities available for scheduling
            clustered_observation_opps[agent_name] = clustered_opps 

            assert all([obs.slew_angles.span()-1e-6 <= cross_track_fovs[agent_name][obs.instrument_name] 
                        for obs in clustered_opps]), \
                f"Observation opportunities have slew angles larger than the maximum allowed field of view."
        
        # merge task across agents
        merged_observation_opps : Dict[str, List[ObservationOpportunity]] = {}
        for agent_name in clustered_observation_opps.keys():
            merged = list(atomic_observation_opps[agent_name])
            merged.extend(clustered_observation_opps[agent_name])
            merged_observation_opps[agent_name] = merged

        # group observation opportunities by tasks
        task_observation_opps : Dict[GenericObservationTask, List[ObservationOpportunity]] = defaultdict(list)
        for agent_name, observation_opps in merged_observation_opps.items():
            for obs_opp in observation_opps:
                for task in obs_opp.tasks:
                    if task not in task_observation_opps:
                        task_observation_opps[task] = []
                    task_observation_opps[task].append((obs_opp, agent_name))

        # sort observation opportunities for each task by observation time
        for task, obs_opp_agent_list in task_observation_opps.items():
            obs_time_pairs = []

            # calculate earliest observation time for the approrpiate task in each observation opportunity
            for obs_opp, agent_name in obs_opp_agent_list:
                obs_opp : ObservationOpportunity
                obs_opp_earliest_time = obs_opp.get_earliest_task_start(task)
                obs_time_pairs.append((obs_opp, agent_name, obs_opp_earliest_time)) 

            # sort observation opportunities by observation time
            sorted_obs_time_pairs = sorted(obs_time_pairs, key=lambda x: x[2])

            # add to final list of observation opportunities for this task, now sorted by observation time
            task_observation_opps[task] = [(obs_opp, agent_name, obs_time) 
                                           for obs_opp, agent_name, obs_time in sorted_obs_time_pairs]
        
        # TODO calculate primal bound using greedy oracle
        primal_bound = 0.0

        # calculate dual bound using a relaxed version of the problem         
        dual_bound = ResultsProcessor.__calculate_dual_bound(
                task_observation_opps, compiled_orbitdata, agent_missions, agent_specs, cross_track_fovs, obtained_rewards_df, printouts
            )

        # return bound values
        return primal_bound, dual_bound
            
    @staticmethod
    def __calculate_task_priorities(accesses_per_task: Dict[GenericObservationTask, List]) -> Tuple:
        total_task_priority = sum([task.priority for task in accesses_per_task.keys()])
        total_observable_task_priority = sum([task.priority*int(len(accesses) > 0) for task,accesses in accesses_per_task.items()])
        # total_observable_task_priority = sum([task.priority*len(accesses) for task,accesses in accesses_per_task.items()])
        
        return total_task_priority, total_observable_task_priority

    # ---------------------------------------------------------------------------
    # Sequence evaluation
    # ---------------------------------------------------------------------------
    
    @staticmethod
    def _evaluate_sequence_at_times(
        task,
        sequence: list[tuple],     # (agent_name, t_earliest_obs, look_angle, obs_opp)
        times: list[tuple],        # (t_obs, d_obs) per observation
        agent_missions: dict,
        agent_specs: dict,
        cross_track_fovs: dict,
        compiled_orbitdata: dict,
        estimate_fn,
    ) -> float:
        total = 0.0
        for n_obs, ((agent_name, _, th, obs_opp), (t_obs, d_obs)) in enumerate(
            zip(sequence, times)
        ):
            t_prev = times[n_obs - 1][0] if n_obs > 0 else 0.0
    
            measurement_performance = estimate_fn(
                task,
                obs_opp.instrument_name,
                th,
                t_obs,
                d_obs,
                agent_specs[agent_name],
                cross_track_fovs[agent_name],
                compiled_orbitdata[agent_name],
                n_obs,
                t_prev,
            )
    
            for measurement in measurement_performance.values():
                measurement[ObservationRequirementAttributes.OBSERVATION_NUMBER.value] = n_obs + 1
                measurement[TemporalRequirementAttributes.REVISIT_TIME.value] = (
                    t_obs - t_prev if n_obs > 0 else 0.0
                )
    
            obs_opp_utility = max(
                agent_missions[agent_name].calc_task_value(task, measurement, False)
                for measurement in measurement_performance.values()
            ) if measurement_performance else 0.0
    
            total += obs_opp_utility
    
        return total
    
    
    # ---------------------------------------------------------------------------
    # Simulated Annealing over sequences (outer loop)
    # ---------------------------------------------------------------------------
    
    @staticmethod
    def __sa_best_sequence(
        task,
        available_obs: list[tuple],
        agent_missions: dict,
        agent_specs: dict,
        cross_track_fovs: dict,
        compiled_orbitdata: dict,
        estimate_fn,
        bo_n_initial: int,
        bo_n_iterations: int,
        bo_n_acq_candidates: int,
        sa_t_initial: float,
        sa_t_final: float,
        sa_cooling: float,
        sa_steps_per_temp: int,
        seed: int | None,
    ) -> tuple[list[tuple], list[tuple], float]:
        """
        Simulated Annealing over valid ordered subsets of available_obs.
        Returns (best_sequence, best_times, best_reward).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
        n = len(available_obs)
        if n == 0:
            return [], [], 0.0
    
        # Precompute mutual exclusivity bitmasks
        mx_mask = [0] * n
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if available_obs[i][3].is_mutually_exclusive(available_obs[j][3]):
        #             mx_mask[i] |= (1 << j)
        #             mx_mask[j] |= (1 << i)
    
        def are_exclusive(i: int, j: int) -> bool:
            return bool(mx_mask[i] & (1 << j))
    
        def is_valid(indices: list[int]) -> bool:
            for k in range(len(indices)):
                for l in range(k + 1, len(indices)):
                    if are_exclusive(indices[k], indices[l]):
                        return False
            for k in range(len(indices) - 1):
                if available_obs[indices[k]][0] > available_obs[indices[k + 1]][0]:
                    return False
            return True
        
        _eval_cache: dict[frozenset, tuple] = {}
        
        # def evaluate(indices: list[int]) -> tuple[list[tuple], float]:
        def evaluate(indices: list[int]) -> tuple[list[tuple], float]:
            key = frozenset(indices)  # note: loses ordering, add tuple key if order matters
            if key in _eval_cache:
                return _eval_cache[key]
            
            # ... run BO ...
            sequence = [
                (available_obs[i][1], available_obs[i][0],
                available_obs[i][2], available_obs[i][3])
                for i in indices
            ]
            t_starts = [
                available_obs[i][3].task_accessibility[task.id].left
                for i in indices
            ]
            t_ends = [
                available_obs[i][3].task_accessibility[task.id].right
                for i in indices
            ]
            d_min = [
                # available_obs[i][3].task_accessibility[task.id].span()
                0.0
                for i in indices
            ]
    
            def objective(times: list[tuple]) -> float:
                return ResultsProcessor._evaluate_sequence_at_times(
                    task, sequence, times,
                    agent_missions, agent_specs,
                    cross_track_fovs, compiled_orbitdata,
                    estimate_fn,
                )
    
            opt = _SequenceTimeOptimiser(
                n_dims=len(indices),
                n_initial=bo_n_initial,
                n_iterations=bo_n_iterations,
                n_acq_candidates=bo_n_acq_candidates,
            )
            # return opt.optimise(t_starts, t_ends, d_min, objective)
            best_times, reward = opt.optimise(t_starts, t_ends, d_min, objective)
            
            _eval_cache[key] = (best_times, reward)
            return best_times, reward
    
        # Greedy initialisation
        sorted_idx = sorted(range(n), key=lambda i: available_obs[i][0])
        current_indices = [sorted_idx[0]]
        excluded = mx_mask[sorted_idx[0]] | (1 << sorted_idx[0])
        for i in sorted_idx[1:]:
            if not (excluded & (1 << i)):
                if available_obs[i][0] >= available_obs[current_indices[-1]][0]:
                    current_indices.append(i)
                    excluded |= mx_mask[i] | (1 << i)
    
        current_times, current_reward = evaluate(current_indices)
        best_indices = list(current_indices)
        best_times   = list(current_times)
        best_reward  = current_reward
        in_sequence  = set(current_indices)

        # if len(available_obs) == 1:
        #     best_sequence = [
        #         (available_obs[i][1], available_obs[i][0],
        #         available_obs[i][2], available_obs[i][3])
        #         for i in best_indices
        #     ]
        #     return best_sequence, best_times, best_reward
    
        # SA main loop
        temperature = sa_t_initial
    
        while temperature > sa_t_final:
            for _ in range(sa_steps_per_temp):
                excluded_from_current = [i for i in range(n) if i not in in_sequence]
    
                ops = []
                if len(current_indices) > 1:
                    ops.append('remove')
                if excluded_from_current:
                    ops.append('add')
                if len(current_indices) > 1 and excluded_from_current:
                    ops.append('swap')
                if not ops:
                    break
    
                op = random.choice(ops)
                candidate = list(current_indices)
    
                if op == 'remove':
                    pos = random.randrange(len(candidate))
                    candidate.pop(pos)
    
                elif op == 'add':
                    new_idx = random.choice(excluded_from_current)
                    valid_positions = []
                    for pos in range(len(candidate) + 1):
                        trial = candidate[:pos] + [new_idx] + candidate[pos:]
                        if is_valid(trial):
                            valid_positions.append(pos)
                    if not valid_positions:
                        continue
                    pos = random.choice(valid_positions)
                    candidate.insert(pos, new_idx)
    
                elif op == 'swap':
                    pos = random.randrange(len(candidate))
                    new_idx = random.choice(excluded_from_current)
                    candidate[pos] = new_idx
                    if not is_valid(candidate):
                        continue
    
                new_times, new_reward = evaluate(candidate)
                delta = new_reward - current_reward
    
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_indices = candidate
                    current_times   = new_times
                    current_reward  = new_reward
                    in_sequence     = set(current_indices)
    
                    if new_reward > best_reward:
                        best_reward  = new_reward
                        best_indices = list(candidate)
                        best_times   = list(new_times)
    
            temperature *= sa_cooling
    
        best_sequence = [
            (available_obs[i][1], available_obs[i][0],
            available_obs[i][2], available_obs[i][3])
            for i in best_indices
        ]
        return best_sequence, best_times, best_reward
    
    
    # ---------------------------------------------------------------------------
    # Main dual bound calculator
    # ---------------------------------------------------------------------------
    
    @staticmethod
    def __calculate_dual_bound_SA(
        task_observation_opps: Dict,
        compiled_orbitdata: Dict,
        agent_missions: Dict,
        agent_specs: Dict,
        cross_track_fovs: Dict,
        obtained_rewards_df: pd.DataFrame,
        printouts: bool,
        bo_n_initial: int = 8,
        bo_n_iterations: int = 20,
        bo_n_acq_candidates: int = 300,
        sa_t_initial: float = 1.0,
        sa_t_final: float = 1e-3,
        sa_cooling: float = 0.95,
        sa_steps_per_temp: int = 15,
        sa_seed: int | None = None,
    ) -> float:
        estimate_fn = ResultsProcessor.__estimate_task_performance_metrics
    
        dual_bound = dict()
        dual_n_obs = dict()
    
        for task, observation_opps in tqdm(
            task_observation_opps.items(),
            desc='Calculating dual bound',
            disable=not printouts or len(task_observation_opps) < 10,
            leave=False,
        ):
            available_obs_times = [
                (obs_time, agent_name,
                (obs_opp.slew_angles.left + obs_opp.slew_angles.right) / 2,
                obs_opp)
                for obs_opp, agent_name, obs_time in observation_opps
            ]
    
            _, best_times, task_utility = ResultsProcessor.__sa_best_sequence(
                task,
                available_obs_times,
                agent_missions,
                agent_specs,
                cross_track_fovs,
                compiled_orbitdata,
                estimate_fn,
                bo_n_initial        = bo_n_initial,
                bo_n_iterations     = bo_n_iterations,
                bo_n_acq_candidates = bo_n_acq_candidates,
                sa_t_initial        = sa_t_initial,
                sa_t_final          = sa_t_final,
                sa_cooling          = sa_cooling,
                sa_steps_per_temp   = sa_steps_per_temp,
                seed                = sa_seed,
            )
    
            dual_bound[task] = task_utility
    
            obs = obtained_rewards_df[obtained_rewards_df['task_id'] == task.id]
            if sum(obs['reward']) - 1e-6 > task_utility:
                x = 1  # set breakpoint here when debugging bound violations
    
        dual_bound_df = pd.DataFrame(
            list(dual_bound.items()), columns=['task', 'reward']
        )
        dual_bound_df['n_obs'] = dual_bound_df['task'].map(dual_n_obs)
        if printouts:
            print(dual_bound_df.to_string(index=False))
    
        return sum(dual_bound.values())

    @staticmethod
    def __calculate_dual_bound(
        task_observation_opps: Dict,
        compiled_orbitdata: Dict,
        agent_missions: Dict,
        agent_specs: Dict,
        cross_track_fovs: Dict,
        obtained_rewards_df: pd.DataFrame,
        printouts: bool,
        bo_n_initial: int = 8,
        bo_n_iterations: int = 20,
    ) -> float:
        estimate_fn = ResultsProcessor.__estimate_task_performance_metrics

        dual_bound = dict()
        dual_n_obs = dict()

        for task, observation_opps in tqdm(
            task_observation_opps.items(),
            desc='Calculating dual bound',
            disable=not printouts or len(task_observation_opps) < 10,
            leave=False,
        ):
            # Build available_obs_times and enumerate feasible sequences
            # (identical to original)
            available_obs_times = [
                (obs_time, agent_name,
                (obs_opp.slew_angles.left + obs_opp.slew_angles.right) / 2,
                obs_opp)
                for obs_opp, agent_name, obs_time in observation_opps
            ]

            # feasible_sequences = ResultsProcessor._find_longest_observation_sequence_for_task(
            feasible_sequences = ResultsProcessor._find_all_maximal_sequences(

                available_obs_times
            )

            def _optimise_sequence(args):
                """Top-level function (must be picklable — no lambda, no closure)."""
                (task, sequence, t_starts, t_ends, d_min,
                agent_missions, agent_specs, cross_track_fovs,
                compiled_orbitdata, estimate_fn,
                bo_n_initial, bo_n_iterations) = args

                def objective(times):
                    return ResultsProcessor._evaluate_sequence_at_times(
                        task, sequence, times,
                        agent_missions, agent_specs,
                        cross_track_fovs, compiled_orbitdata,
                        estimate_fn,
                    )

                opt = _SequenceTimeOptimiser(
                    n_dims=len(sequence),
                    n_initial=bo_n_initial,
                    n_iterations=bo_n_iterations,
                    # n_acq_candidates=bo_n_acq_candidates,
                )
                return opt.optimise(t_starts, t_ends, d_min, objective)


            # Inside __calculate_dual_bound, replace the inner loop:
            all_args = []
            for obs_names, obs_times, obs_look_angles, obs_opps in feasible_sequences:
                sequence = list(zip(obs_names, obs_times, obs_look_angles, obs_opps))
                t_starts = [o.task_accessibility[task.id].left  for o in obs_opps]
                t_ends   = [o.task_accessibility[task.id].right for o in obs_opps]
                d_min    = [o.task_accessibility[task.id].span() for o in obs_opps]
                all_args.append((
                    task, sequence, t_starts, t_ends, d_min,
                    agent_missions, agent_specs, cross_track_fovs,
                    compiled_orbitdata, estimate_fn,
                    bo_n_initial, bo_n_iterations,
                ))

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(_optimise_sequence, all_args))

            task_utility = max(r[1] for r in results)

            # task_utility = 0.0
            # best_sequence = None

            # for obs_names, obs_times, obs_look_angles, obs_opps in tqdm(
            #     feasible_sequences,
            #     desc=f'Finding best observation sequence for task {task.id.split("-")[-1]}',
            #     disable=not printouts or len(feasible_sequences) < 10,
            #     leave=False,
            # ):
            #     sequence = list(zip(obs_names, obs_times, obs_look_angles, obs_opps))

            #     t_starts = [obs_opp.task_accessibility[task.id].left  for obs_opp in obs_opps]
            #     t_ends   = [obs_opp.task_accessibility[task.id].right for obs_opp in obs_opps]
            #     d_min = [0 for _ in obs_opps]
            #     # d_min = [obs_opp.task_min_duration[task.id] \
            #     #             if i > 0 and obs_names[i] == obs_names[i-1] else 0.0
            #     #          for i,obs_opp in enumerate(obs_opps)
            #     #         ]

            #     def objective(times: list[tuple], _seq=sequence) -> float:
            #         return ResultsProcessor._evaluate_sequence_at_times(
            #             task, _seq, times,
            #             agent_missions, agent_specs,
            #             cross_track_fovs, compiled_orbitdata,
            #             estimate_fn,
            #         )

            #     optimiser = _SequenceTimeOptimiser(
            #         n_dims=len(sequence),
            #         n_initial=bo_n_initial,
            #         n_iterations=bo_n_iterations,
            #     )
            #     best_times, sequence_utility = optimiser.optimise(
            #         t_starts, t_ends, d_min, objective
            #     )

            #     if sequence_utility > task_utility:
            #         task_utility = sequence_utility
            #         best_sequence = (obs_names, best_times, obs_look_angles, obs_opps)

            dual_bound[task] = task_utility

            # Sanity check (preserved from original)
            obs = obtained_rewards_df[obtained_rewards_df['task_id'] == task.id]
            if sum(obs['reward']) - 1e-6 > task_utility:
                tqdm.write(f"Warning: Dual bound for task {task.id} is lower ({task_utility:.6f}) than obtained reward ({sum(obs['reward']):.6f}) by {sum(obs['reward']) - task_utility:.6f}. Please check for bugs in the dual bound calculation.")
                x = 1  # set breakpoint here when debugging bound violations
                # a = ResultsProcessor._find_longest_observation_sequence_for_task(
                #     available_obs_times
                # )
                # y = 1  # set breakpoint here when debugging bound violations

        dual_bound_df = pd.DataFrame(
            list(dual_bound.items()), columns=['task', 'reward']
        )
        # dual_bound_df['n_obs'] = dual_bound_df['task'].map(dual_n_obs)
        if printouts:
            print(dual_bound_df.to_string(index=False))

        return sum(dual_bound.values())
    
    # ---------------------------------------------------------------------------
    # Sequence evaluation — estimate_fn called at every t_obs
    # ---------------------------------------------------------------------------

    @staticmethod
    def _evaluate_sequence_at_times(
        task,
        sequence: list[tuple],          # (agent_name, t_earliest_obs, look_angle, obs_opp)
        times: list[tuple],
        agent_missions: Dict[str,Mission],
        agent_specs: dict,
        cross_track_fovs: dict,
        compiled_orbitdata: dict,
        estimate_fn,
    ) -> float:
        """
        Score a sequence at given observation times.

        estimate_fn is called for every observation at every BO iteration
        because measurement_performance depends on t_obs (through d_obs and
        any other time-dependent fields it computes internally).
        """
        total = 0.0
        for n_obs, ((agent_name, _, th, obs_opp), time) in enumerate(
            zip(sequence, times)
        ):
            # unpack observation time details
            t_obs = time[0]
            d_obs = time[1]
            t_prev = times[n_obs-1][0] if n_obs > 0 else 0.0

            # Recompute performance at the current observation time
            measurement_performance : dict = estimate_fn(
                task,
                obs_opp.instrument_name,
                th,
                t_obs,
                d_obs,
                agent_specs[agent_name],
                cross_track_fovs[agent_name],
                compiled_orbitdata[agent_name],
                n_obs,
                t_prev,
            )

            obs_opp_utility = max(
                agent_missions[agent_name].calc_task_value(task, measurement, False)
                for measurement in measurement_performance.values()
            ) if measurement_performance else 0.0

            total += obs_opp_utility

        return total

    """
    @staticmethod
    def __calculate_dual_bound_DEPRECATED(
                        task_observation_opps : Dict[GenericObservationTask, List[ObservationOpportunity]],
                        compiled_orbitdata : Dict[str, OrbitData],
                        agent_missions: Dict[str, Mission],
                        agent_specs: Dict[str, object],
                        cross_track_fovs : Dict[str, float],
                        obtained_rewards_df : pd.DataFrame,
                        printouts : bool
                    ) -> float:
        # initiate bound value to 0
        dual_bound = dict()
        dual_n_obs = dict()

        # remove any requirements that are to be ignored in the reward calculation for this bound calculation
        ignored_requirements_for_bound_calculation = [
            TemporalRequirementAttributes.DURATION.value,
            TemporalRequirementAttributes.REVISIT_TIME.value,
            TemporalRequirementAttributes.CO_OBSERVATION_TIME.value,
            TemporalRequirementAttributes.RESPONSE_TIME.value,
            TemporalRequirementAttributes.RESPONSE_TIME_NORM.value,
            TemporalRequirementAttributes.OBS_TIME.value,

        ]
        for mission in agent_missions.values():
            for obj in mission:
                obj.requirements = {key : req for key,req in obj.requirements.items()
                                    if req.attribute not in ignored_requirements_for_bound_calculation}

        # calculate the utility for each task
        for task, observation_opps in tqdm(task_observation_opps.items(),
                                           desc='Calculating dual bound',
                                           disable=not printouts or len(task_observation_opps) < 10,
                                           leave=False,
                                        ):
            
            # calculate meaurement performance for each observation opportunity for this task and calculate the utility of each observation opportunity for this task
            measurement_performance_per_obs_opp = {}
            for obs_opp, agent_name, obs_time in observation_opps:
                obs_opp : ObservationOpportunity

                # define observation duration, look angle for this observation opportunity, and previous observation time
                d_obs = obs_opp.accessibility.span() - (obs_time - obs_opp.accessibility.left)
                th = (obs_opp.slew_angles.left + obs_opp.slew_angles.right) / 2
                t_prev = 0.0

                # estimate measurement performance for this observation opportunity
                measurement_performance \
                        = ResultsProcessor.__estimate_task_performance_metrics(task,
                                                                            obs_opp.instrument_name,
                                                                            th,
                                                                            obs_time,
                                                                            d_obs,
                                                                            agent_specs[agent_name],
                                                                            cross_track_fovs[agent_name],
                                                                            compiled_orbitdata[agent_name],
                                                                            0,
                                                                            0.0
                                                                        )
                # store measurement performance for this observation opportunity
                measurement_performance_per_obs_opp[(obs_opp, agent_name, obs_time)] = measurement_performance

            # compile observation opportunities for this task across agents and sort by observation time
            available_obs_times = [ 
                (obs_time, agent_name, (obs_opp.slew_angles.left + obs_opp.slew_angles.right) / 2, obs_opp)
                for obs_opp, agent_name, obs_time in observation_opps
            ]

            # generate all possible observation time opportunities
            feasible_sequences = ResultsProcessor._find_longest_observation_sequence_for_task(available_obs_times)
            
            # initialize search for best observation sequence for this task to 0 utility
            task_utility = 0.0
            best_sequence = None

            # find sequence that maximizes value for this agent
            for obs_names,obs_times,obs_look_angles,obs_tasks in tqdm(feasible_sequences,
                                                                      desc=f'Finding best observation sequence for task {task.id.split("-")[-1]}',
                                                                      disable=not printouts or len(feasible_sequences) < 10,
                                                                      leave=False
                                                                    ):
                # initiate sequence value tracker
                sequence_utility = 0.0

                # evaluate sequence value for this agent
                for n_obs,(agent_name,t_obs,_,obs_opp) in enumerate(zip(obs_names,obs_times,obs_look_angles,obs_tasks)):
                    # get matching agent mission
                    agent_mission = agent_missions[agent_name]
                    d_obs = obs_opp.accessibility.span() - (t_obs - obs_opp.accessibility.left)
                    t_prev = obs_times[n_obs-1] if n_obs > 0 else 0.0
                    
                    # get measurement performance for this observation opportunity
                    measurement_performance \
                        = measurement_performance_per_obs_opp[(obs_opp, agent_name, t_obs)]

                    # update observation number and previous observation time 
                    for measurement in measurement_performance.values():
                        measurement[ObservationRequirementAttributes.OBSERVATION_NUMBER.value] = n_obs + 1
                        if n_obs > 0:
                            measurement[TemporalRequirementAttributes.REVISIT_TIME.value] = t_obs - t_prev
                        else:
                            measurement[TemporalRequirementAttributes.REVISIT_TIME.value] = 0.0

                    # calculate utility of this observation opportunity
                    obs_opp_utility = max([agent_mission.calc_task_value(task, measurement, False) 
                                for measurement in measurement_performance.values()]) \
                                    if len(measurement_performance.values()) > 0 else 0.0
                    
                    # add to sequence utility
                    sequence_utility += obs_opp_utility

                # check if this sequence has higher utility than the best found sequence for this task so far
                if sequence_utility > task_utility:
                    task_utility = sequence_utility
                    best_sequence = (obs_names,obs_times,obs_look_angles,obs_tasks)
            
            # add best sequence utility for this task to the dual bound
            dual_bound[task] = task_utility

            # find observations performed for this task in the current solution
            obs = obtained_rewards_df[obtained_rewards_df['task_id'] == task.id]

            # ensure that the best sequence utility for this task is at least as high as the reward obtained for this task in the current solution
            if sum(obs['reward']) > task_utility:
                x = 1

        dual_bound_df = pd.DataFrame(list(dual_bound.items()), columns=['task','reward'])
        dual_bound_df['n_obs'] = dual_bound_df['task'].map(dual_n_obs)
        if printouts: print(dual_bound_df.to_string(index=False))
            
        # return value
        return sum(dual_bound.values())
    """

    @staticmethod
    def __estimate_task_performance_metrics( 
                                            task : GenericObservationTask, 
                                            instrument_name : str,
                                            th_img : float,
                                            t_img : float,
                                            d_img : float,
                                            specs : Spacecraft, 
                                            cross_track_fovs : dict,
                                            orbitdata : OrbitData,
                                            n_obs : int,
                                            t_prev : float,  
                                        ) -> dict:

        # validate inputs
        assert isinstance(task, GenericObservationTask), "Task must be of type `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(th_img, (int,float)), "Image look angle must be a numeric value."
        assert isinstance(t_img, (int,float)), "Image time must be a numeric value."
        assert t_img >= 0, "Image time must be non-negative."
        assert isinstance(d_img, (int,float)), "Image duration must be a numeric value."
        assert d_img >= 0, "Image duration must be non-negative."
        assert all(isinstance(instr, str) for instr in cross_track_fovs.keys()), "Cross-track FOV instrument names must be strings."
        assert all(isinstance(fov, (int,float)) for fov in cross_track_fovs.values()), "Cross-track FOVs must be numeric values."
        assert all(fov >= 0 for fov in cross_track_fovs.values()), "Cross-track FOVs must be non-negative."
        assert isinstance(orbitdata, OrbitData), "Orbit data must be of type `OrbitData`."
        assert n_obs >= 0, "Number of observations must be non-negative."
        assert t_prev <= t_img, "Last observation time must be before the current image time."

        # get access metrics for given observation time, instrument, and look angle
        observation_performances = ResultsProcessor.get_available_accesses(task, instrument_name, th_img, t_img, d_img, orbitdata, cross_track_fovs)

        # check if there are no valid observations for this task
        if any([len(observation_performances[col]) == 0 for col in observation_performances]): 
            # no valid accesses; no reward added
            return dict()
        
        # group observations by location
        observed_location_groups : dict[tuple[int,int], list[int]] = defaultdict(list)
        for i in range(len(observation_performances['time [s]'])):
            # unpack observed target location information
            lat = observation_performances['lat [deg]'][i]
            lon = observation_performances['lon [deg]'][i]
            grid_index = int(observation_performances['grid index'][i])
            gp_index = int(observation_performances['GP index'][i])

            # define location indices
            loc = (lat,lon,grid_index,gp_index)

            # add to location group
            observed_location_groups[loc].append({col.lower() : observation_performances[col][i] 
                                                    for col in observation_performances})
        
        # sort groups by measurement time 
        for loc in observed_location_groups: observed_location_groups[loc].sort(key=lambda a : a['time [s]'])
        
        # get unique task targets
        task_targets : List[tuple] = list({(grid_idx,gp_idx) 
                                           for *_,grid_idx,gp_idx in task.location})

        # keep only one of the observations per location group that matches the task target
        observation_performance_metrics : Dict[tuple[int,int], dict] = {loc : observed_location_groups[loc][0] # keep only first observation
                                                 for loc in observed_location_groups
                                                 if (loc[2],loc[3]) in task_targets
                                                 }       

        # get instrument specifications
        instrument_spec : BasicSensorModel = next(instr 
                                                  for instr in specs.instrument
                                                  if instr.name.lower() == instrument_name.lower()).mode[0]

        # include additional observation information 
        for loc,obs_perf in observation_performance_metrics.items():
            
            # update observation performance information
            obs_perf.update({ 
                SpatialCoverageRequirementAttributes.LOCATION.value : [loc],
                TemporalRequirementAttributes.DURATION.value : d_img,
                TemporalRequirementAttributes.REVISIT_TIME.value : t_img - t_prev,
                #TODO Co-observation time
                TemporalRequirementAttributes.RESPONSE_TIME.value : t_img - task.availability.left,
                TemporalRequirementAttributes.RESPONSE_TIME_NORM.value : (t_img - task.availability.left) / task.availability.span() if task.availability.span() > 0 else 0.0,
                TemporalRequirementAttributes.OBS_TIME.value : t_img,
                "t_end" : t_img + d_img,
                ObservationRequirementAttributes.OBSERVATION_NUMBER.value : n_obs + 1, # including this observation
            })

            # handle special case of first observation
            if n_obs == 0:
                obs_perf[TemporalRequirementAttributes.REVISIT_TIME.value] = 0.0

            # update instrument-specific observation performance information
            if (('vnir' in instrument_name.lower() or 'tir' in instrument_name.lower())
                or ('vnir' in instrument_spec._type.lower() or 'tir' in instrument_spec._type.lower())):
                if isinstance(instrument_spec.spectral_resolution, str):
                    obs_perf.update({
                        ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution.lower()
                    })
                elif isinstance(instrument_spec.spectral_resolution, (int,float)):
                    obs_perf.update({
                        ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution
                    })
                else:
                    raise ValueError('Unsupported type for spectral resolution in instrument specification.')
                
            elif ('altimeter' in instrument_name.lower()
                  or 'altimeter' in instrument_spec._type.lower()):
                obs_perf.update({
                    ObservationRequirementAttributes.ACCURACY.value : observation_performance_metrics[loc][ObservationRequirementAttributes.ACCURACY.value],
                })
            else:
                raise NotImplementedError(f'Calculation of task reward not yet supported for instruments of type `{instrument_name.lower()}`.')

        return observation_performance_metrics
    
    @staticmethod
    def get_available_accesses( 
                               task : GenericObservationTask, 
                               instrument_name : str,
                               th_img : float,
                               t_img : float,
                               d_img : float,
                               orbitdata : OrbitData, 
                               cross_track_fovs : dict
                            ) -> dict:
        # --- 1) Build targets once (prefer vector-friendly form) ---
        # task.location rows look like: (*_, grid_index, gp_index)
        task_targets = {(int(grid_index), int(gp_index)) for *_, grid_index, gp_index in task.location}
        if not task_targets:
            return {}

        # Pack targets into uint64 keys for fast membership
        tgt_keys = np.fromiter(
            ((g << 32) | (p & 0xFFFFFFFF) for (g, p) in task_targets),
            dtype=np.uint64,
            count=len(task_targets),
        )

        # --- 2) One lookup for the whole time window ---
        access = orbitdata.gp_access_data.lookup_interval(
            t_img,
            t_img + d_img,
            include_extras=True,
            filters=None,
            columns=None,
            decode=True,
            exact_time_filter=True,
        )

        # Quick empty guard
        t = np.asarray(access.get("time [s]", []))
        if t.size == 0: return access  # or {}

        # --- 3) Vector masks for FOV + instrument + target membership ---
        # Convert needed columns to arrays once
        grid = np.asarray(access["grid index"], dtype=np.int64)
        gp = np.asarray(access["GP index"], dtype=np.int64)
        offn = np.asarray(access["off-nadir axis angle [deg]"], dtype=np.float64)

        # Instrument column handling
        instr_col = access["instrument"]

        # FOV check
        half_fov = float(cross_track_fovs[instrument_name]) * 0.5
        m_fov = np.abs(offn - float(th_img)) <= half_fov

        # Instrument match
        instr = np.asarray(instr_col)
        m_instr = (instr == instrument_name)

        # Target membership via packed keys
        keys = (grid.astype(np.uint64) << np.uint64(32)) | (gp.astype(np.uint64) & np.uint64(0xFFFFFFFF))
        # np.isin on uint64 is vectorized
        m_target = np.isin(keys, tgt_keys, assume_unique=False)

        mask = m_fov & m_instr & m_target
        if not mask.any():
            # Return same structure but empty lists (or arrays)
            out = {k: [] for k in access.keys()}
            out["eclipse"] = []
            return out

        # --- 4) Slice all columns once ---
        # Keep as arrays for now; convert to lists at the end if your callers need lists.
        obs = {}
        for col, arr in access.items():
            a = np.asarray(arr)
            obs[col] = a[mask]

        # --- 5) Eclipse flags fast (no per-time "t in interval" loops) ---
        eclipse_intervals = orbitdata.eclipse_data.lookup_intervals(t_img, t_img + d_img)

        times = np.asarray(obs["time [s]"], dtype=np.float64)
        eclipse_flags = np.zeros(times.shape[0], dtype=np.int8)

        if eclipse_intervals:
            # Build sorted start/end arrays
            # Assumes interval has .start and .end or something equivalent.
            # Adjust these two lines to your Interval type.
            starts = np.array([float(interval.left) for interval, *_ in eclipse_intervals], dtype=np.float64)
            ends   = np.array([float(interval.right)   for interval, *_ in eclipse_intervals], dtype=np.float64)

            # Sort by start (and reorder ends accordingly)
            order = np.argsort(starts)
            starts = starts[order]
            ends = ends[order]

            # For each time t: find rightmost start <= t, then check t <= corresponding end
            idx = np.searchsorted(starts, times, side="right") - 1
            valid = idx >= 0
            eclipse_flags[valid] = (times[valid] <= ends[idx[valid]]).astype(np.int8)

        # Use your enum key if you need it
        obs["eclipse"] = eclipse_flags

        # --- 6) Convert to python lists if your downstream expects lists ---
        # (If downstream can accept numpy arrays, don't convert; that’s faster.)
        out = {}
        for k, v in obs.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)

        return out

    @staticmethod
    def single_task_observation_opportunity_from_accesses(
                                                          compiled_orbitdata : Dict[str, OrbitData], 
                                                          accesses_per_task: Dict[GenericObservationTask, List[Tuple[Interval, str, str]]], 
                                                          cross_track_fovs : dict,
                                                          threshold : float = 1e-9
                                                        ) -> Dict[str, List[ObservationOpportunity]]:
        """ Creates one instance of a task observation opportunity per each access opportunity 
        for every available task """

        # initiate task observation opportunities map: agent name -> list of observation opportunities for this agent
        agent_task_observation_opps : Dict[str, List[ObservationOpportunity]] = defaultdict(list)

        # iterate throughout all accessible tasks and their corresponding access opportunities
        for task, accesses in accesses_per_task.items():
            for access_interval, agent_name, instrument_name in accesses:
                for *_,grid_idx,gp_idx in task.location:
                    # check if instrument can perform the task
                    if not ResultsProcessor.__can_perform_task(task, instrument_name): 
                        continue # skip if instrument cannot perform task
                    
                    # get matching agent orbit data
                    agent_orbitdata = compiled_orbitdata[agent_name]

                    # extract minimum duration requirement for this task
                    min_duration_req : float = ResultsProcessor.__extract_minimum_duration_req(task, agent_orbitdata)

                    # ensure minimum duration requirement is a positive number
                    assert isinstance(min_duration_req, (int,float)) and min_duration_req >= 0.0, "minimum duration requirement must be a positive number."

                    # check if overlapping access interval is long enough to perform the task
                    if access_interval.span() < min_duration_req - threshold: continue

                    # get matching access interval in agent orbit data
                    target_coverage_data : dict \
                        = agent_orbitdata.gp_access_data.lookup_interval(access_interval.left, 
                                                                        access_interval.right,
                                                                        filters={
                                                                                'instrument': instrument_name,
                                                                                'grid index': grid_idx,
                                                                                'GP index': gp_idx
                                                                            }
                                                                        )
                    # extract off-nadir angle data for matching access interval
                    reduced_th = target_coverage_data['off-nadir axis angle [deg]']
                    
                    # calculate slew angle interval 
                    off_axis_angles = [Interval(off_axis_angle - cross_track_fovs[agent_name][instrument_name]/2,
                                                off_axis_angle + cross_track_fovs[agent_name][instrument_name]/2)
                                                for off_axis_angle in reduced_th]

                    # skip if no off-nadir angle data available for this access interval
                    if not off_axis_angles: 
                        continue 

                    # merge off-nadir angle intervals to get overall slew angle interval for this access interval
                    slew_angles : Interval = reduce(lambda a, b: a.intersection(b), off_axis_angles)
                    
                    # skip if no valid slew angles
                    if slew_angles.is_empty(): 
                        continue  

                    # add task observation opportunity to list of task observation opportunities
                    obs_opp = AtomicObservationOpportunity(task,
                                                            instrument_name,
                                                            # use reduced access interval
                                                            access_interval,
                                                            # use reduced slew angle interval
                                                            slew_angles,
                                                            # take into acount possible tolerance in duration requirement
                                                            min(min_duration_req, access_interval.span()), 
                                                            # TODO calculate maximum duration requirement based on agent capabilities
                                                        )
                    agent_task_observation_opps[agent_name].append(obs_opp)
                        
        # return task observation opportunities
        return agent_task_observation_opps
    
    @staticmethod
    def _collect_fov_specs(specs : Spacecraft) -> dict:
        """ get instrument field of view specifications from agent specs object """
        # validate inputs
        if isinstance(specs, dict):
            if specs.get('instrument', None) is None: return {}
        elif isinstance(specs, Spacecraft):            
            if specs.instrument is None: return {}
        else:
            raise ValueError(f'`specs` needs to be of type `dict` or `Spacecraft`. Is of type `{type(specs)}`.')

        # compile instrument field of view specifications   
        cross_track_fovs = {instrument.name: np.NAN for instrument in specs.instrument}
        for instrument in specs.instrument:
            cross_track_fov = []
            for instrument_model in instrument.mode:
                if isinstance(instrument_model, BasicSensorModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                elif isinstance(instrument_model, PassiveOpticalScannerModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')
            cross_track_fovs[instrument.name] = max(cross_track_fov)

        return cross_track_fovs

    @staticmethod
    def __extract_minimum_duration_req(task : GenericObservationTask, orbitdata : OrbitData) -> float:
        """ Extracts the minimum duration requirement for a given task. """
        
        # check if task has any objectives
        if task.objective is None:
            return orbitdata.time_step # no objectives assigned to this task; assume default minimum duration requirement

        # extract any duration requirements from the task objective
        duration_reqs = [req for req in task.objective
                        if req.attribute == TemporalRequirementAttributes.DURATION.value]
        
        # check if any duration requirements were found
        if not duration_reqs: return orbitdata.time_step # no duration requirement found; return default minimum duration requirement

        # get duration requirement
        duration_req : PerformanceRequirement = duration_reqs[0]

        # extract minimum duration requirement based on requirement type
        if isinstance(duration_req, CategoricalRequirement):
            raise ValueError('Categorical duration requirements are not supported.')
        
        elif isinstance(duration_req, ConstantValueRequirement):
            return duration_req.value # return constant duration requirement value
        
        elif isinstance(duration_req, ExpSaturationRequirement):
            return - (1 / duration_req.sat_rate) * np.log(1 - 0.01) # return duration requirement at 1% saturation

        elif isinstance(duration_req, LogThresholdRequirement):
            return duration_req.threshold # return log threshold duration requirement value

        elif isinstance(duration_req, ExpDecayRequirement):
            return - (1 / duration_req.decay_rate) * np.log(0.01) # return duration requirement at 1% decay

        elif isinstance(duration_req, GaussianRequirement):
            # TODO implement gaussian requirement extraction
            raise NotImplementedError('Gaussian duration requirements are not supported yet.')
        
        elif isinstance(duration_req, TriangleRequirement):
            min_duration = duration_req.reference - (duration_req.width / 2) * (1 - 0.01) # 1% of the triangle height
            return max(min_duration, 0.0) # ensure non-negative duration requirement

        elif isinstance(duration_req, StepsRequirement):
            # filter non-zero scores
            positive_scores = [(idx, score) for idx,score in enumerate(duration_req.scores) if score > 0]

            # check if there are any positive scores        
            if not positive_scores:
                raise ValueError('No positive scores found in `StepsRequirement` for duration requirement.')

            # get interval with minimum positive score
            min_idx, _ = min(positive_scores, key=lambda x: x[1])

            if min_idx == 0:
                return max(duration_req.thresholds[0], 0.0)
            elif min_idx == len(duration_req.thresholds):
                return max(duration_req.thresholds[-1], 0.0)
            else:
                return max(min(duration_req.thresholds[min_idx + 1], duration_req.thresholds[min_idx]), 0.0)
        
        elif isinstance(duration_req, IntervalInterpolationRequirement):
            # TODO implement interval interpolation requirement extraction
            raise NotImplementedError('Interval interpolation duration requirements are not supported yet.')     
       
        # unsupported requirement type; should not reach here
        raise ValueError('Unsupported duration requirement type.')

    @staticmethod
    def __can_perform_task(task : GenericObservationTask, instrument_name : str) -> bool:
        """ Checks if the agent can perform the task at hand with the given instrument """
        # TODO Replace this with KG for better reasoning capabilities; currently assumes instrument has general capability

        # Check if task has specified objectives
        if task.objective is not None:
            # Extract capability requirements from the objective
            capability_reqs = [req for req in task.objective
                               if isinstance(req, CapabilityRequirement)
                               and req.attribute == CapabilityRequirementAttributes.INSTRUMENT.value]
            capability_req: CapabilityRequirement = capability_reqs[0] if capability_reqs else None

            # Evaluate capability requirement
            if capability_req is not None:
                return capability_req.calc_preference(CapabilityRequirementAttributes.INSTRUMENT.value, instrument_name.lower()) >= 0.5

        # No capability objectives specified; check if instrument has general capability
        return True
    
    @staticmethod
    def __check_task_observation_opportunity_clusterability(
            observation_opportunities : List[ObservationOpportunity], 
            must_overlap : bool, 
            threshold : float,
            printouts : bool
        ) -> dict:
        """ 
        Creates adjacency list for a given list of task observation opportunities.

        #### Arguments
        - `observation_opportunities` : A list of task observation opportunities to create the adjacency list for.
        - `must_overlap` : Whether tasks' availability must overlap in availability time to be considered for clustering.
        - `threshold` : The time threshold for clustering tasks in seconds [s].
        """

        # create adjacency list for tasks
        adj : Dict[str, set[ObservationOpportunity]] = {task.id : set() for task in observation_opportunities}
        assert len(adj) == len(observation_opportunities), \
            "Duplicate observation opportunity IDs found when creating adjacency list."

        if observation_opportunities:
            # sort tasks by accessibility
            observation_opportunities.sort(key=lambda a : a.accessibility) 
            
            # get min and max accessibility times
            t_min = observation_opportunities[0].accessibility.left

            # initialize bins
            bins = defaultdict(list)
            
            # group task in bins by accessibility
            for task in tqdm(observation_opportunities, leave=False, desc="Grouping tasks into bins", disable=not printouts):
                task : ObservationOpportunity
                center_time = (task.accessibility.left + task.accessibility.right) / 2 - t_min
                bin_key = int(center_time // threshold)
                bins[bin_key].append(task)

            # populate adjacency list
            with tqdm(total=len(observation_opportunities), desc="Checking task clusterability", leave=False, disable=not printouts) as pbar:
                for b in bins:
                    candidates : list[ObservationOpportunity] \
                          = bins[b] + bins.get(b + 1, []) + bins.get(b - 1, [])
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            t1, t2 = candidates[i], candidates[j]
                            if t1.can_merge(t2, must_overlap=must_overlap, max_duration=threshold):
                                adj[t1.id].add(t2)
                                adj[t2.id].add(t1)
                        pbar.update(1)

        # check if adjacency list is symmetric
        for p in observation_opportunities:
            assert p not in adj[p.id], \
                f'Task {p.id} is in its own adjacency list.'
            for q in adj[p.id]:
                assert p in adj[q.id], \
                    f'Task {p.id} is in the adjacency list of task {q.id} but not vice versa.'

        return adj

    @staticmethod
    def cluster_task_observation_opportunities( 
                                               observation_opportunities : List[ObservationOpportunity], 
                                               adj : Dict[str, Set[ObservationOpportunity]], 
                                               must_overlap : bool,
                                               printouts : bool
                                            ) -> list:
        """ 
        Clusters observation opportunities based on adjacency. 
        
        ```
        while V!=Ø do
            Pick a vertex p with largest degree from V. 
                If such p are not unique, pick the p with highest priority.
            
            while N(p)=Ø do
                Pick a neighbor q of p, q ∈ N(p), such that the number of their common neighbors is maximum. 
                    If such p are not unique, pick the p with least edges being deleted.
                    Again, if such p are still not unique, pick the p with highest priority.
                Combine q and p into a new p
                Delete edges from q and p that are not connected to their common neighbors
                Reset neighbor collection N(p) for the new p
            end while
            
            Output the cluster-task denoted by p
            Delete p from V
        end while
        ```
        
        """         
        # only keep observation opportunities that have at least one clusterable observation opportunity
        v = [obs for obs in observation_opportunities if len(adj[obs.id]) > 0]
        
        # sort observation opportunities by degree of adjacency 
        v : list[ObservationOpportunity] = ResultsProcessor.__sort_by_degree(observation_opportunities, adj)
        
        # combine observation opportunities into clusters
        combined_obs : list[ObservationOpportunity] = []

        with tqdm(total=len(v), desc="Merging overlapping observation opportunities", leave=False, disable=not printouts) as pbar:
            while len(v) > 0:
                # pop first observation opportunity from the list of observation opportunities to be scheduled
                p : ObservationOpportunity = v.pop()

                # get list of neighbors of p sorted by number of common neighbors
                n_p : list[ObservationOpportunity] = ResultsProcessor.__sort_observation_opportunities_by_common_neighbors(p, list(adj[p.id]), adj)

                # initialize clique with p
                clique = set()

                # update progress bar
                pbar.update(1)

                # while there are neighbors of p
                while len(n_p) > 0:
                    # pop first neighbor q from the list of neighbors
                    q : ObservationOpportunity = n_p.pop()

                    # Combine q and p into a new p                 
                    clique.add(q)

                    # find common neighbors of p and q
                    common_neighbors : set[ObservationOpportunity] = adj[p.id].intersection(adj[q.id])
                   
                    # remove edges to p and q that do not include common neighbors
                    for neighbor in adj[p.id].difference(common_neighbors): adj[neighbor.id].discard(p)
                    for neighbor in adj[q.id]: adj[neighbor.id].discard(q)              
                    
                    # update edges of p and q to only include common neighbors
                    adj[p.id].intersection_update(common_neighbors)
                    
                    # remove q from the adjacency list
                    adj.pop(q.id)

                    # remove q from the list of tasks to be scheduled
                    v.remove(q)

                    # Reset neighbor collection N_p for the new p;
                    n_p : list[ObservationOpportunity] = ResultsProcessor.__sort_observation_opportunities_by_common_neighbors(p, list(adj[p.id]), adj)               

                for q in clique: 
                    # TODO: look into ID being used. Ideally we would want a new ID for the combined task.

                    # merge all tasks in the clique into a single task p
                    # p = p.merge(q, must_overlap=must_overlap, max_duration=threshold)  # max duration of 5 minutes
                    p = p.merge(q, must_overlap=must_overlap)

                    # update progress bar
                    pbar.update(1)

                # Update adjacency lists to capture new task requirements and clusterability
                for neighbor in adj[p.id]:
                    # remove p from neighbor's adjacency list and vice versa
                    adj[neighbor.id].discard(p)
                    adj[p.id].discard(neighbor)

                    # reevaluate adjacency
                    # if p.can_merge(neighbor, must_overlap=must_overlap, max_duration=threshold):
                    if p.can_merge(neighbor, must_overlap=must_overlap):
                        # if p and neighbor can still be merged;
                        # add edge back to adjacency list
                        adj[neighbor.id].add(p)
                        adj[p.id].add(neighbor)

                # DEBUGGING--------- 
                # clique.add(p)
                # cliques.append(sorted([observation_opportunities.index(t)+1 for t in clique]))
                # ------------------

                # add merged task to the list of combined tasks
                combined_obs.append(p) 

                # sort remaining task observation opportunities by degree of adjacency 
                v : list[ObservationOpportunity] = ResultsProcessor.__sort_by_degree(v, adj)
        
        # return only observation opportunities that have multiple parents (avoid generating duplicate observation opportunities)
        multiple_task_obs = [obs for obs in combined_obs if len(obs.tasks) > 1] 

        # generate new id's for combined observation opportunities to avoid duplicate id's with single-task observation opportunities
        for obs in multiple_task_obs:
            obs.regenerate_id()

        # return combined observation opportunities
        return multiple_task_obs

    
    @staticmethod
    def __sort_by_degree(obs_opportunities : List[ObservationOpportunity], adjacency : dict) -> list:
        """ Sorts observation opportunities by degree of adjacency. """
        # calculate degree of each observation opportunity
        degrees : dict = {obs : len(adjacency[obs.id]) for obs in obs_opportunities}

        # sort observation opportunities by degree and return
        return sorted(obs_opportunities, key=lambda p: (degrees[p], 
                                                        sum([parent_task.priority for parent_task in p.tasks]), 
                                                        -p.accessibility.left,
                                                        -p.accessibility.span(),
                                                        p.id
                                                        ))

    @staticmethod
    def __sort_observation_opportunities_by_common_neighbors(p : ObservationOpportunity, n_p : list, adjacency : dict) -> list:
        # specify types
        n_p : list[ObservationOpportunity] = n_p
        adjacency : Dict[str, set[ObservationOpportunity]] = adjacency

        # calculate common neighbors
        common_neighbors : dict = {q : adjacency[p.id].intersection(adjacency[q.id]) 
                                   for q in n_p}
        
        # calculate neighbors to delete
        neighbors_to_delete : dict = {q : adjacency[p.id].difference(adjacency[q.id])
                                      for q in n_p}
        
        # sort neighbors by number of common neighbors, number of edges to delete, priority and accessibility
        return sorted(n_p, 
                      key=lambda p: (len(common_neighbors[p]), 
                                     -len(neighbors_to_delete[p]),
                                     sum([parent_task.priority for parent_task in p.tasks]), 
                                     -p.accessibility.left,
                                     -p.accessibility.span(),
                                      p.id
                                    )
                    )

    @staticmethod
    def _find_all_maximal_sequences(
        available_obs: List[tuple],
    ) -> List[Tuple[List[str], List[float], List[float], List]]:
        """
        Returns all maximal feasible ordered sequences regardless of length.
        A sequence is maximal if no further observation can be appended.
        """
        n = len(available_obs)
        if not n:
            return []
    
        mx_mask = [0] * n
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if available_obs[i][3].is_mutually_exclusive(available_obs[j][3]):
        #             mx_mask[i] |= (1 << j)
        #             mx_mask[j] |= (1 << i)
    
        successors = [[] for _ in range(n)]
        for i in range(n):
            t_i = available_obs[i][0]
            for j in range(i + 1, n):
                if available_obs[j][0] >= t_i:
                    successors[i].append(j)
    
        node_info = []
        stack = []
        for i in range(n):
            node_id = len(node_info)
            node_info.append((i, -1))
            stack.append((i, node_id, mx_mask[i] | (1 << i), 1))
    
        leaf_nodes = []
        while stack:
            obs_idx, node_id, excluded, depth = stack.pop()
            valid_succs = [j for j in successors[obs_idx]
                        if not (excluded & (1 << j))]
            if not valid_succs:
                leaf_nodes.append((depth, node_id))
            else:
                for j in valid_succs:
                    new_node_id = len(node_info)
                    node_info.append((j, node_id))
                    stack.append((j, new_node_id,
                                excluded | mx_mask[j] | (1 << j),
                                depth + 1))
    
        def reconstruct(node_id):
            path = []
            while node_id != -1:
                obs_idx, parent = node_info[node_id]
                path.append(available_obs[obs_idx])
                node_id = parent
            path.reverse()
            return path
    
        results = []
        for _, node_id in leaf_nodes:
            seq = reconstruct(node_id)
            results.append((
                [s[1] for s in seq],
                [s[0] for s in seq],
                [s[2] for s in seq],
                [s[3] for s in seq],
            ))
    
        results.sort(key=lambda x: (*x[1],))
        return results
    
    @staticmethod
    def _find_longest_observation_sequence_for_task(
        available_obs: List[tuple]
    ) -> List[Tuple[List[str], List[float]]]:
        """Find feasible observation number sequences for a given task."""
        
        n = len(available_obs)
        if not n:
            return []

        # --- Precompute mutual exclusivity as a bitmask per observation ---
        # mx_mask[i] is a bitmask of all j where obs[i] and obs[j] are mutually exclusive
        mx_mask = [0] * n
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         # if available_obs[i][3].is_mutually_exclusive(available_obs[j][3]):
        #         if (available_obs[i][1] == available_obs[j][1]
        #             and available_obs[i][3].is_mutually_exclusive(available_obs[j][3])):
        #             mx_mask[i] |= (1 << j)
        #             mx_mask[j] |= (1 << i)

        # --- Precompute valid successors for each index ---
        # A successor j of i must satisfy: obs[j][0] >= obs[i][0] and j > i
        successors = [[] for _ in range(n)]
        for i in range(n):
            t_i = available_obs[i][0]
            for j in range(i + 1, n):
                if available_obs[j][0] >= t_i:
                    successors[i].append(j)

        # --- DFS with parent pointers and exclusion bitmask ---
        # Stack items: (current_index, parent_index, excluded_mask, depth)
        # We track parent pointers to reconstruct paths without copying lists
        
        # node_info[node_id] = (obs_index, parent_node_id)
        node_info = []  # (obs_idx, parent_node_id)
        
        # Stack: (obs_idx, parent_node_id, excluded_bitmask, depth)
        stack = []
        for i in range(n):
            node_id = len(node_info)
            node_info.append((i, -1))
            stack.append((i, node_id, mx_mask[i] | (1 << i), 1))

        best_depth = 0
        leaf_nodes = []  # (depth, node_id) for maximal sequences

        while stack:
            obs_idx, node_id, excluded, depth = stack.pop()

            # Find valid successors (not excluded, time-ordered)
            valid_succs = [j for j in successors[obs_idx] 
                        if not (excluded & (1 << j))]

            if not valid_succs:
                leaf_nodes.append((depth, node_id))
            
            for j in valid_succs:
                new_node_id = len(node_info)
                node_info.append((j, node_id))
                stack.append((j, new_node_id, excluded | mx_mask[j] | (1 << j), depth + 1))

        # --- Reconstruct only the longest sequences ---
        def reconstruct(node_id):
            path = []
            while node_id != -1:
                obs_idx, parent = node_info[node_id]
                path.append(available_obs[obs_idx])
                node_id = parent
            path.reverse()
            return path

        results = []
        for _, node_id in leaf_nodes:
            seq = reconstruct(node_id)
            obs_names      = [s[1]   for s in seq]
            obs_times      = [s[0]   for s in seq]
            obs_look_angles = [s[2]  for s in seq]
            obs_opps       = [s[3]   for s in seq]
            results.append((obs_names, obs_times, obs_look_angles, obs_opps))

        results.sort(key=lambda x: (len(x), *x[1]))
        return results

    # @staticmethod
    # def _find_feasible_observation_sequences_for_task_DEPRECATED(
    #                                                   available_obs : List[tuple]
    #                                                 ) -> List[Tuple[List[str], List[float]]]:
    #     """ Find feasible observation number sequences for a given task. """
    #     # initialize feasible sequence tracker
    #     feasible_sequences = []

    #     # count minimum sequence length; use number of occurrences of this agent in available observation times
    #     min_seq_length = 1 # for now, assume minimum sequence length of 1

    #     # create dfs queue
    #     dfs_queue = deque()

    #     # seed dfs with initial observations from this agent
    #     for obs in available_obs: dfs_queue.append([obs])

    #     # perform dfs to find feasible sequences
    #     while dfs_queue:
    #         # pop current sequence from stack
    #         current_sequence = dfs_queue.pop()

    #         # check if current sequence can be accepted
    #         if (len(current_sequence) >= min_seq_length                 # meets minimum length requirements
    #             ):
    #             # sequence can be accepted; decompose sequence into component lists
    #             obs_names = [agent_name for _,agent_name,_,_ in current_sequence]
    #             obs_times = [t_img for t_img,_,_,_ in current_sequence]
    #             obs_look_angles = [look_angle for _,_,look_angle,_ in current_sequence]
    #             obs_opps = [obs_opp for _,_,_,obs_opp in current_sequence]
                
    #             # add to feasible sequences
    #             feasible_sequences.append((obs_names, obs_times, obs_look_angles, obs_opps))               

    #         # check for available successors
    #         successors = [obs for obs in available_obs
    #                       if obs[0] >= current_sequence[-1][0]
    #                       and obs not in current_sequence
    #                     #   and not any(obs[3].is_mutually_exclusive(prev_obs[3]) 
    #                     #                 for prev_obs in current_sequence)
    #                     ]

    #         # queue successors
    #         for obs_next in successors:
    #             # create new sequence with successor added
    #             new_sequence = [obs for obs in current_sequence] + [obs_next]

    #             # add new sequence to dfs stack
    #             dfs_queue.append(new_sequence)

    #     # return feasible sequences
    #     return feasible_sequences

# ---------------------------------------------------------------------------
# Bayesian Optimisation over observation times
# ---------------------------------------------------------------------------

class _SequenceTimeOptimiser:
    """
    Optimises observation times for a fixed ordered sequence of access windows.
    Each delta_i in [0,1] maps to a valid observation time for window i,
    guaranteed to be strictly after t_{i-1}.
    """

    def __init__(
        self,
        n_dims: int,
        n_initial: int = 8,
        n_iterations: int = 20, # 8,
        n_acq_candidates: int = 30, # 300,
        kappa: float = 2.576,
        noise: float = 1e-6,
        length_scale: float = 0.3,
    ):
        self.n_dims = n_dims
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_acq_candidates = n_acq_candidates
        self.kappa = kappa
        self.noise = noise
        self.length_scale = length_scale

        self._X: list[np.ndarray] = []
        self._y: list[float] = []

        # Cholesky cache — rebuilt once per new point, not per _posterior call
        self._L: np.ndarray | None = None       # lower triangular Cholesky of K
        self._alpha: np.ndarray | None = None   # L^{-T} L^{-1} y

    # -----------------------------------------------------------------------
    # Kernel
    # -----------------------------------------------------------------------

    def _k(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.exp(-0.5 * np.dot(d, d) / self.length_scale ** 2))

    def _k_vector(self, x: np.ndarray) -> np.ndarray:
        """Compute k(x, xi) for all xi in self._X in one vectorised call."""
        X = np.array(self._X)           # (n, d)
        diff = X - x                    # (n, d)
        sq_dist = np.einsum('nd,nd->n', diff, diff)  # (n,)
        return np.exp(-0.5 * sq_dist / self.length_scale ** 2)
    
    def _k_matrix(self, X_query: np.ndarray) -> np.ndarray:
        """k(X_query, X_train) — shape (n_query, n_train)"""
        diff = X_query[:, None, :] - np.array(self._X)[None, :, :]  # (q, n, d)
        sq_dist = np.einsum('qnd,qnd->qn', diff, diff)
        return np.exp(-0.5 * sq_dist / self.length_scale ** 2)

    # -----------------------------------------------------------------------
    # Cache management — call once after every new (x, y) pair is added
    # -----------------------------------------------------------------------

    # def _update_cache(self):
    #     n = len(self._X)
    #     K = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(i, n):
    #             v = self._k(self._X[i], self._X[j])
    #             K[i, j] = K[j, i] = v
    #     K += self.noise * np.eye(n)

    #     # Normalise y to zero mean, unit variance before fitting
    #     y = np.array(self._y)
    #     self._y_mean = y.mean()
    #     self._y_std  = y.std() if y.std() > 1e-9 else 1.0
    #     y_norm = (y - self._y_mean) / self._y_std

    #     self._L = np.linalg.cholesky(K)
    #     self._alpha = np.linalg.solve(
    #         self._L.T, np.linalg.solve(self._L, y_norm)  # fit on normalised y
    #     )

    def _update_cache(self):
        """Full refactorisation — only called when cache is cold."""
        X = np.array(self._X)
        diff = X[:, None, :] - X[None, :, :]
        sq_dist = np.einsum('ijd,ijd->ij', diff, diff)
        K = np.exp(-0.5 * sq_dist / self.length_scale ** 2)
        K += self.noise * np.eye(len(self._X))

        y = np.array(self._y)
        self._y_mean = y.mean()
        self._y_std  = y.std() if y.std() > 1e-9 else 1.0
        y_norm = (y - self._y_mean) / self._y_std

        self._L = np.linalg.cholesky(K)
        self._L_inv = np.linalg.solve(self._L, np.eye(len(self._X)))  # always in sync
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_norm))

    # def _add_point(self, x: np.ndarray, y: float):
    #     """Add a new observed point and refresh the cache."""
    #     self._X.append(x.copy())
    #     self._y.append(y)
    #     self._update_cache()

    def _add_point(self, x: np.ndarray, y: float):
        self._X.append(x.copy())
        self._y.append(y)

        if self._L is None:
            self._update_cache()
            return

        # Rank-1 Cholesky update
        n = len(self._X) - 1
        X_prev = np.array(self._X[:-1])
        diff = X_prev - x
        sq_dist = np.einsum('nd,nd->n', diff, diff)
        k_new = np.exp(-0.5 * sq_dist / self.length_scale ** 2)
        k_self = 1.0 + self.noise

        v = self._L_inv @ k_new        # O(n²) — L_inv already cached
        new_diag = math.sqrt(max(k_self - float(v @ v), 1e-9))

        new_L = np.zeros((n + 1, n + 1))
        new_L[:n, :n] = self._L
        new_L[n,  :n] = v
        new_L[n,   n] = new_diag
        self._L = new_L

        # Update L_inv via block matrix inverse formula
        # [ L    0        ]^{-1}   [ L_inv         0        ]
        # [ v^T  new_diag ]      = [ -v^T L_inv / new_diag  1/new_diag ]
        new_L_inv = np.zeros((n + 1, n + 1))
        new_L_inv[:n, :n] = self._L_inv
        new_L_inv[n,  :n] = -(v @ self._L_inv) / new_diag
        new_L_inv[n,   n] = 1.0 / new_diag
        self._L_inv = new_L_inv

        # Recompute alpha (y_mean/y_std change with each point)
        y_arr = np.array(self._y)
        self._y_mean = y_arr.mean()
        self._y_std  = y_arr.std() if y_arr.std() > 1e-9 else 1.0
        y_norm = (y_arr - self._y_mean) / self._y_std
        self._alpha = self._L_inv.T @ (self._L_inv @ y_norm)  # O(n²), no solve needed

    # -----------------------------------------------------------------------
    # Posterior — O(n) per call thanks to cached L and alpha
    # -----------------------------------------------------------------------

    # def _posterior(self, x: np.ndarray) -> tuple[float, float]:
    #     if not self._X:
    #         return 0.0, 1.0
    #     k_star = np.array([self._k(x, xi) for xi in self._X])
    #     mu_norm = float(k_star @ self._alpha)
    #     v = np.linalg.solve(self._L, k_star)
    #     var = max(1.0 - float(v @ v), 1e-9)

    #     # Denormalise mu back to original scale for acquisition and best tracking
    #     mu = mu_norm * self._y_std + self._y_mean
    #     sigma = math.sqrt(var) * self._y_std
    #     return mu, sigma

    # def _posterior(self, x: np.ndarray) -> tuple[float, float]:
    #     if not self._X:
    #         return 0.0, 1.0
    #     k_star = self._k_vector(x)      # vectorised, no Python loop
    #     mu_norm = float(k_star @ self._alpha)
    #     v = np.linalg.solve(self._L, k_star)
    #     var = max(1.0 - float(v @ v), 1e-9)
    #     mu    = mu_norm * self._y_std + self._y_mean
    #     sigma = math.sqrt(var) * self._y_std
    #     return mu, sigma

    def _posterior(self, x: np.ndarray) -> tuple[float, float]:
        if not self._X:
            return 0.0, 1.0
        k_star = self._k_vector(x)
        mu_norm = float(k_star @ self._alpha)
        v = self._L_inv @ k_star        # pure matmul, no solve
        var = max(1.0 - float(v @ v), 1e-9)
        mu    = mu_norm * self._y_std + self._y_mean
        sigma = math.sqrt(var) * self._y_std
        return mu, sigma

    def _acq(self, x: np.ndarray) -> float:
        mu, sigma = self._posterior(x)
        return mu + self.kappa * sigma
    
    def _acq_ei(self, x: np.ndarray) -> float:
        mu, sigma = self._posterior(x)
        best = max(self._y)
        z = (mu - best) / (sigma + 1e-9)
        # standard normal PDF and CDF
        ei = (mu - best) * self._normal_cdf(z) + sigma * self._normal_pdf(z)
        return max(ei, 0.0)
    
    def _acq_ei_batch(self, X_query: np.ndarray) -> np.ndarray:
        """EI for all candidates at once — shape (n_query,)"""
        K_qs = self._k_matrix(X_query)                    # (q, n)
        mu_norm = K_qs @ self._alpha                       # (q,)
        V = (self._L_inv @ K_qs.T).T                      # (q, n)
        var = np.maximum(1.0 - np.einsum('qn,qn->q', V, V), 1e-9)  # (q,)
        mu    = mu_norm * self._y_std + self._y_mean
        sigma = np.sqrt(var) * self._y_std
        best  = max(self._y)
        z = (mu - best) / (sigma + 1e-9)
        ei = (mu - best) * self._ndtr(z) + sigma * self._npdf(z)
        return np.maximum(ei, 0.0)

    @staticmethod
    def _normal_pdf(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    
    @staticmethod
    def _npdf(z: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    @staticmethod
    def _ndtr(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + scipy.special.erf(z / math.sqrt(2)))

    # -----------------------------------------------------------------------
    # Reparameterisation
    # -----------------------------------------------------------------------

    @staticmethod
    def deltas_to_times(
        deltas: np.ndarray,
        t_starts: list[float],
        t_ends: list[float],
        d_min: list[float],
        eps: float = 1e-6,
    ) -> tuple[list[tuple], bool]:
        """
        Map delta_i in [0,1] to a strictly ordered sequence of
        (t_obs, d_obs) pairs. Returns (times, is_feasible).
        """
        times = []
        prev = -math.inf
        feasible = True
        for delta, t_start, t_end, d in zip(deltas, t_starts, t_ends, d_min):
            lo = max(t_start, prev + eps)
            hi = t_end
            if lo > hi:
                feasible = False
                lo = t_start
            t_obs = lo + delta * max(hi - lo, 0.0)
            d_obs = t_end - t_obs   # remaining duration from obs time to end
            times.append((t_obs, d_obs))
            prev = t_obs
        return times, feasible

    # -----------------------------------------------------------------------
    # Main optimise loop
    # -----------------------------------------------------------------------

    def optimise(
        self,
        t_starts: list[float],
        t_ends: list[float],
        d_min: list[float],
        objective,
        eps: float = 1e-6,
    ) -> tuple[list[tuple], float]:
        """Returns (best_times, best_reward)."""
        self._X = []
        self._y = []
        self._L = None
        self._alpha = None
        best_times: list[tuple] = []
        best_reward = -math.inf

        def _eval(deltas: np.ndarray) -> float:
            times, feasible = self.deltas_to_times(deltas, t_starts, t_ends, d_min, eps)
            if feasible:
                return objective(times)
            violation = sum(
                max(0.0, times[i-1][0] - times[i][0] + eps)
                for i in range(1, len(times))
            )
            return -violation

        def _try(deltas: np.ndarray):
            """Evaluate, update cache, update best."""
            nonlocal best_reward, best_times
            r = _eval(deltas)
            self._add_point(deltas, r)      # O(n³) here, once per point
            if r > best_reward:
                best_reward = r
                best_times, _ = self.deltas_to_times(
                    deltas, t_starts, t_ends, d_min, eps
                )

        # --- Deterministic anchors ---
        _try(np.zeros(self.n_dims))         # earliest times
        _try(np.ones(self.n_dims))          # latest times

        # mixed boundary anchors — one per dimension
        for i in range(self.n_dims):
            d = np.full(self.n_dims, 0.5)
            d[i] = 0.0
            _try(d)
            d[i] = 1.0
            _try(d)

        # --- Random initial samples ---
        for _ in range(self.n_initial):
            _try(np.random.uniform(0, 1, self.n_dims))

        # --- BO iterations ---
        for _ in range(self.n_iterations):
            candidates = np.random.uniform(0, 1, (self.n_acq_candidates, self.n_dims))
            acqs = self._acq_ei_batch(candidates)   # one vectorised call
            _try(candidates[np.argmax(acqs)])        
        # for _ in range(self.n_iterations):
        #     candidates = np.random.uniform(0, 1, (self.n_acq_candidates, self.n_dims))
        #     # acqs = np.array([self._acq(c) for c in candidates])
        #     acqs = np.array([self._acq_ei(c) for c in candidates])
            
        #     _try(candidates[np.argmax(acqs)])

        # after main BO loop, before returning
        best_delta = self._X[np.argmax(self._y)]
        for _ in range(10):
            # small perturbations around best, clipped to [0,1]
            perturbed = np.clip(
                best_delta + np.random.normal(0, 0.05, self.n_dims),
                0, 1
            )
            _try(perturbed)

        # also explicitly try snapping each dimension to its nearest boundary
        for i in range(self.n_dims):
            snapped = best_delta.copy()
            snapped[i] = 0.0 if best_delta[i] < 0.5 else 1.0
            _try(snapped)

        return best_times, best_reward
