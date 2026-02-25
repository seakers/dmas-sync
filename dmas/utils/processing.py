from collections import defaultdict
import os
from typing import Dict, List, Set, Tuple
import numpy as np

from tqdm import tqdm
import pandas as pd

from execsatm.mission import Mission
from execsatm.attributes import SpatialCoverageRequirementAttributes, TemporalRequirementAttributes, ObservationRequirementAttributes
from execsatm.events import GeophysicalEvent
from execsatm.tasks import GenericObservationTask, DefaultMissionTask, EventObservationTask
from execsatm.objectives import DefaultMissionObjective, EventDrivenObjective
from execsatm.requirements import ExplicitCapabilityRequirement
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
                        printouts: bool = True
                    ) -> Tuple:
        """ processes simulation results after execution """
        
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
        obtained_rewards_df = ResultsProcessor.__compile_obtained_rewards(observations_per_task, agent_missions, printouts)

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
                                   printouts : bool
                                ) -> pd.DataFrame:
        # initialize list of obtained rewards
        obtained_rewards_data = []

        # iterate tasks and corresponding observations
        for task, observations in observations_per_task.items():
            
            # initialize previous observation time for revisit time calculation
            t_prev = np.NINF
            
            # iterate observations by start time 
            for n_obs,obs_perf in enumerate(sorted(observations, key=lambda obs: obs['t_start'])):
                # unpack observation information
                observing_agent = obs_perf['agent name']
                instrument_name : str = obs_perf['instrument']
                loc = (obs_perf['lat [deg]'], obs_perf['lon [deg]'], obs_perf['grid index'], obs_perf['GP index'])
                d_img = obs_perf['t_end'] - obs_perf['t_start']
                t_img = obs_perf['time [s]']

                # get matching mission for observing agent
                mission : Mission = agent_missions[observing_agent]

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

                # TODO update instrument-specific observation performance information
                # if (('vnir' in instrument_name.lower() or 'tir' in instrument_name.lower())
                #     or ('vnir' in instrument_spec._type.lower() or 'tir' in instrument_spec._type.lower())):
                #     if isinstance(instrument_spec.spectral_resolution, str):
                #         obs_perf.update({
                #             ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution.lower()
                #         })
                #     elif isinstance(instrument_spec.spectral_resolution, (int,float)):
                #         obs_perf.update({
                #             ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution
                #         })
                #     else:
                #         raise ValueError('Unsupported type for spectral resolution in instrument specification.')
                    
                # elif ('altimeter' in instrument_name.lower()
                #     or 'altimeter' in instrument_spec._type.lower()):
                #     obs_perf.update({
                #         ObservationRequirementAttributes.ACCURACY.value : observation_performance_metrics[loc][ObservationRequirementAttributes.ACCURACY.value],
                #     })
                # else:
                #     raise NotImplementedError(f'Calculation of task reward not yet supported for instruments of type `{instrument_name.lower()}`.')

                # evaluate reward for this observation
                reward = mission.calc_task_value(task, obs_perf)

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
        else:
            obtained_rewards_df = pd.DataFrame(columns=['task_id', 'n_obs', 't_img', 'agent', 'instrument', 'reward'])

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
                row['start time [s]'],
                row['end time [s]'] - row['start time [s]'],
                row['severity'],
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

        # suplement with event observation tasks
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
    def __classify_accesses_per_gp(compiled_orbitdata : Dict[str, OrbitData]) -> Dict[Tuple[int,int], pd.DataFrame]:
        # compile acesses from agent orbitdata
        acesses = []
        for agent_orbit_data in compiled_orbitdata.values():
            access_intervals = agent_orbit_data.gp_access_data.lookup_interval(0.0)
            for i in range(len(access_intervals['time [s]'])):
                row ={col : access_intervals[col][i] for col in access_intervals}
                acesses.append(row)
        accesses_df_temp = pd.DataFrame(acesses)
        
        # group accesses by ground point
        accesses_per_gp : Dict[Tuple[int,int], pd.DataFrame] \
                                = {(int(group[0]), int(group[1])) : data
                                for group,data in accesses_df_temp.groupby(['grid index', 'GP index'])} \
                                if not accesses_df_temp.empty else dict() # handle empty accesses case       
       
        # return set of accessible ground points
        return accesses_per_gp

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
                                       accesses_per_gp : Dict[Tuple[int,int], pd.DataFrame],
                                       printouts : bool
                                    ) -> Tuple[pd.DataFrame, Dict[GeophysicalEvent, List[Tuple[Interval, str, str]]]]:
        
        # initiate list of compiled events with access information
        accesses_per_event_data = []
        accesses_per_event = dict()

        # look for accesses to each event for each agent
        for event in tqdm(events, 
                            desc='Classifying event accesses, detections, and observations', 
                            leave=True,
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
            location_access : pd.DataFrame = accesses_per_gp.get(location_key, None)
            
            # check if there are any accesses for this location
            if location_access is None: continue # no accesses found; skip
            
            # filter data that exists within the task's availability window and matches instrument capability requirements
            location_access_filtered : pd.DataFrame \
                = location_access[(location_access['time [s]'] >= event.availability.left) 
                    & (location_access['time [s]'] <= event.availability.right)]
            location_access_filtered : pd.DataFrame \
                = location_access_filtered[location_access_filtered.apply(lambda row: row['instrument'].lower() in instrument_capability_reqs[row['agent name']], axis=1)]

            # initialize map of compiled access intervals
            access_interval_dict : Dict[tuple,List[Interval]] = defaultdict(list)

            # iterate throufh data to map accesses by agent name and instrument
            for (agent_name, instrument), group in location_access_filtered.groupby(['agent name', 'instrument']):
                # get sorted list of access times for this agent and instrument
                matching_accesses = group[['time [s]', 'agent name', 'instrument']].sort_values(by='time [s]').values.tolist()
                
                # compile list of access intervals
                for t_access, agent_name, instrument in matching_accesses:
                    # get propagation time step for this agent
                    time_step = compiled_orbitdata[agent_name].time_step 

                    # create unitary interval for this access time
                    access_interval = Interval(t_access, t_access + time_step)

                    # check if this access overlaps with any previous access
                    merged = False
                    for interval in access_interval_dict[(agent_name,instrument)]:
                        if access_interval.overlaps(interval):
                            # if so, join intervals
                            interval.join(access_interval)
                            merged = True
                            break
                    
                    # if merged, continue to next interval
                    if merged: continue
                    
                    # otherwise, create a new access interval
                    access_interval_dict[(agent_name,instrument)].append(access_interval)

            # flatten to list of access intervals
            access_intervals : List[Tuple[Interval, str, str]] \
                    = sorted([ (interval,agent_name,instrument) 
                                for (agent_name,instrument),intervals in access_interval_dict.items()
                                for interval in intervals ])
            
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
                                     accesses_per_gp : Dict[Tuple[int,int], pd.DataFrame],
                                     printouts: bool
                                    ) -> Tuple[pd.DataFrame, List[Tuple[GenericObservationTask, List[Tuple[Interval, str, str]]]]]:
        # initialize map of tasks to access intervals
        accesses_per_task : Dict[GenericObservationTask, list] = defaultdict(list)
        accesses_per_task_df_data = []

        for task in tqdm(known_tasks, desc="Processing task accessibility", leave=True, disable=not printouts):
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
                
                # filter data that exists within the task's availability window and matches instrument capability requirements
                location_access_filtered : pd.DataFrame \
                    = location_access[(location_access['time [s]'] >= task.availability.left) 
                     & (location_access['time [s]'] <= task.availability.right)]
                location_access_filtered : pd.DataFrame \
                    = location_access_filtered[location_access_filtered.apply(lambda row: row['instrument'].lower() in instrument_capability_reqs[row['agent name']], axis=1)]

                # initialize map of compiled access intervals
                access_interval_dict : Dict[tuple,List[Interval]] = defaultdict(list)

                # iterate throufh data to map accesses by agent name and instrument
                for (agent_name, instrument), group in location_access_filtered.groupby(['agent name', 'instrument']):
                    # get sorted list of access times for this agent and instrument
                    matching_accesses = group[['time [s]', 'agent name', 'instrument']].sort_values(by='time [s]').values.tolist()
                    
                    # compile list of access intervals
                    for t_access, agent_name, instrument in matching_accesses:
                        # get propagation time step for this agent
                        time_step = compiled_orbitdata[agent_name].time_step 

                        # create unitary interval for this access time
                        access_interval = Interval(t_access, t_access + time_step)

                        # check if this access overlaps with any previous access
                        merged = False
                        for interval in access_interval_dict[(agent_name,instrument)]:
                            if access_interval.overlaps(interval):
                                # if so, join intervals
                                interval.join(access_interval)
                                merged = True
                                break
                        
                        # if merged, continue to next interval
                        if merged: continue
                        
                        # otherwise, create a new access interval
                        access_interval_dict[(agent_name,instrument)].append(access_interval)

                # flatten to list of access intervals
                access_intervals : List[Tuple[Interval, str, str]] \
                    = sorted([ (interval,agent_name,instrument) 
                             for (agent_name,instrument),intervals in access_interval_dict.items()
                             for interval in intervals ])                
                
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
                            leave=True,
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
        for task in tqdm(known_tasks, desc="Processing task observations", leave=True, disable=not printouts):
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
            
            if task_observations: tasks_observed[task] = task_observations

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

        # classify observations
        events_re_observable = {event : accesses for event, accesses in accesses_per_event.items()
                                 if len(accesses) > 1}
        events_re_obs = {event : observations for event, observations in observations_per_event.items()
                            if len(observations) > 1}
        
        # TODO: implement co-observability classification
        events_co_observable = dict()
        events_co_observable_fully = dict()
        events_co_observable_partially = dict()
        events_co_obs = dict()
        events_co_obs_fully = dict()
        events_co_obs_partially = dict()        

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

        total_task_priority, total_available_utility = ResultsProcessor.__calculate_total_available_utility(accesses_per_task)

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
                    # ['P(Event Co-observation | Observation)', np.round(p_event_co_obs_if_obs,n_decimals)],
                    # ['P(Event Full Co-observation | Observation)', np.round(p_event_co_obs_partially_if_obs,n_decimals)],
                    # ['P(Event Partial Co-observation | Observation)', np.round(p_event_co_obs_fully_if_obs,n_decimals)],

                    ['P(Event Observed | Observable)', np.round(p_event_observed_if_observable,precision)],
                    ['P(Event Re-observed | Re-observable)', np.round(p_event_re_obs_if_re_observable,precision)],
                    # ['P(Event Co-observed | Co-observable)', np.round(p_event_co_obs_if_co_observable,n_decimals)],
                    # ['P(Event Fully Co-observed | Fully Co-observable)', np.round(p_event_co_obs_fully_if_co_observable_fully,n_decimals)],
                    # ['P(Event Partially Co-observed | Partially Co-observable)', np.round(p_event_co_obs_partial_if_co_observable_partially,n_decimals)],
                    
                    ['P(Event Observed | Event Detected)', np.round(p_event_observed_if_detected,precision)],
                    ['P(Event Re-observed | Event Detected)', np.round(p_event_re_obs_if_detected,precision)],
                    # ['P(Event Co-observed | Event Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed Fully | Event Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    # ['P(Event Co-observed Partially | Event Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

                    ['P(Event Observed | Event Observable and Detected)', np.round(p_event_observed_if_detected,precision)],
                    ['P(Event Re-observed | Event Re-observable and Detected)', np.round(p_event_re_obs_if_detected,precision)],
                    # ['P(Event Co-observed | Event Co-observable and Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed Fully | Event Fully Co-observable and Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    # ['P(Event Co-observed Partially | Event Partially Co-observable and Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

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
                    ['Normalized Total Planned Reward', total_planned_reward / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Total Planned Task Observations', len(planned_rewards_df) if planned_rewards_df is not None else 0],

                    ['Total Obtained Reward', total_obtained_reward],
                    ['Normalized Total Obtained Reward', total_obtained_reward / total_available_utility if total_available_utility > 0 else 0.0],
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
                    ['Normalized Total Planned Utility', total_planned_utility / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Average Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().mean() - execution_costs_df.groupby('agent')['cost'].sum().mean(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Standard Deviation of Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().std() - execution_costs_df.groupby('agent')['cost'].sum().std(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Median Planned Utility per Agent', np.round(planned_rewards_df.groupby('agent')['planned reward'].sum().median() - execution_costs_df.groupby('agent')['cost'].sum().median(), precision) if planned_rewards_df is not None and execution_costs_df is not None else 0.0],

                    ['Total Obtained Utility', total_obtained_utility],
                    ['Normalized Total Obtained Utility', total_obtained_utility / total_available_utility if total_available_utility > 0 else 0.0],
                    ['Average Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().mean() - execution_costs_df.groupby('agent')['cost'].sum().mean(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Standard Deviation of Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().std() - execution_costs_df.groupby('agent')['cost'].sum().std(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],
                    ['Median Obtained Utility per Agent', np.round(obtained_rewards_df.groupby('agent')['reward'].sum().median() - execution_costs_df.groupby('agent')['cost'].sum().median(), precision) if obtained_rewards_df is not None and execution_costs_df is not None else 0.0],

                    # Available Reward and Utility Statistics
                    ['Total Task Priority Available', total_task_priority],
                    ['Total Available Utility', total_available_utility],

                    # Results dir
                    # ['Results Directory', results_path]
                ]

        return pd.DataFrame(summary_data, columns=summary_headers)    

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
        for _,agent_orbitdata_temp in tqdm(orbitdata.items(), desc='Counting total and accessible ground points', leave=False, disable=not printouts):
            # count number of ground points
            n_gps = len([gps for gps in agent_orbitdata_temp.grid_data]) if n_gps is None else n_gps

            # get set of accessible ground points
            gps_accessible : set = {(row['grid index'], row['GP index']) for _,row in agent_orbitdata_temp.gp_access_data}

            # update set of accessible ground points
            gps_accessible_compiled.update(gps_accessible)

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
        n_total_event_co_obs = sum([len(co_observations) for _,co_observations in events_co_obs.items()])        

        assert n_events_co_obs <= n_events_observed
        assert n_total_event_co_obs <= n_observations

        n_events_fully_co_obs = len(events_co_obs_fully)
        n_total_event_fully_co_obs = sum([len(full_co_observations) for _,full_co_observations in events_co_obs_fully.items()])        
        
        assert n_events_fully_co_obs <= n_events_observed
        assert n_total_event_fully_co_obs <= n_total_event_co_obs

        n_events_partially_co_obs = len(events_co_obs_partially)
        n_total_event_partially_co_obs = sum([len(partial_co_observations) for _,partial_co_observations in events_co_obs_partially.items()])        

        assert n_events_partially_co_obs <= n_events_observed
        assert n_total_event_partially_co_obs <= n_total_event_co_obs

        assert n_events_co_obs == n_events_fully_co_obs + n_events_partially_co_obs
        assert n_total_event_co_obs == n_total_event_fully_co_obs + n_total_event_partially_co_obs

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
        n_events_co_obs_and_co_observable = len([event for event in events_co_obs
                                                if event in events_co_observable])
        n_events_co_obs_and_detected = len([event for event in events_co_obs
                                                if event in events_detected])
        n_events_co_obs_and_co_observable_and_detected = len([event for event in events_co_obs
                                                                if event in events_co_observable
                                                                and event in events_detected])
        n_events_co_observable_and_detected = len([event for event in events_co_observable
                                                    if event in events_detected])

        
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
    def __calculate_total_available_utility(accesses_per_task: Dict[GenericObservationTask, List]) -> float:
        # TODO include objectives and priorities in task data structures to properly calculate this;
        #   currently assumes that each access has a maximum performance value of 1 
        total_task_priority = sum([task.priority for task in accesses_per_task.keys()])
        total_available_utility = sum([task.priority*len(accesses) for task,accesses in accesses_per_task.items()])
        
        return total_task_priority, total_available_utility