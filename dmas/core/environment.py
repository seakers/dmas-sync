import copy
import gc
import logging
import os
import time
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd
from tqdm import tqdm

from dmas.core.messages import BusMessage, MeasurementRequestMessage, ObservationResultsMessage, SimulationMessage, message_from_dict
from dmas.models.trackers import DataSink
from dmas.utils.orbitdata import OrbitData

from execsatm.tasks import EventObservationTask
from execsatm.events import GeophysicalEvent
from execsatm.utils import Interval

from dmas.models.actions import *
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState, SimulationAgentTypes
from dmas.utils.tools import MessageTracker, SimulationRoles


class SimulationEnvironment(object):
    """
    ## Simulation Environment

    Environment in charge of creating task requests and notifying agents of their exiance
    Tracks the current state of the agents and checks if they are in communication range 
    of eachother.
    
    """
    
    def __init__(self, 
                results_path : str, 
                scenario_orbitdata : Dict[str, OrbitData],
                sat_list : List[dict],
                gs_list : List[dict],                
                events : List[GeophysicalEvent],
                connectivity_level : str = 'LOS',
                connectivity_relays : bool = False,
                level: int = logging.INFO, 
                logger: logging.Logger = None,
                printouts : bool = True
            ) -> None:
        ...
        # setup results folder:
        env_results_path : str = os.path.join(results_path, SimulationRoles.ENVIRONMENT.value.lower())

        # assign parameters
        self._orbitdata : Dict[str,OrbitData] = scenario_orbitdata
        self._events : List[GeophysicalEvent] = events
        self._results_path : str = env_results_path
        self._scenario_results_path : str = results_path
        self._logger : logging.Logger = logger if logger is not None \
                                            else logging.getLogger(SimulationRoles.ENVIRONMENT.value.lower())
        self._logger.setLevel(level)
        self._printouts : bool = printouts

        # load agent names and classify by type of agent
        self.agents = {}
        agent_names = []
            
        # load satellite names
        sat_names = []
        if sat_list:
            for sat in sat_list:
                sat : dict
                sat_name = sat.get('name')
                sat_names.append(sat_name)
                agent_names.append(sat_name)
        self.agents[SimulationAgentTypes.SATELLITE] = sat_names

        # load GS agent names
        gs_names : list = []
        if gs_list:
            for gs in gs_list:
                gs : dict
                gs_name = gs.get('name')
                gs_names.append(gs_name)
                agent_names.append(gs_name)
        self.agents[SimulationAgentTypes.GROUND_OPERATOR] = gs_names

        # setup agent connectivity
        self._connectivity_level : str = connectivity_level.upper()
        self.__interval_connectivities : List[Tuple[Interval, Set[frozenset], Dict[str, List[str]]]] \
            = SimulationEnvironment.__precompute_connectivity(scenario_orbitdata, printouts) # list of (interval, connectivity_matrix, components_list)

        # initialize parameters
        self._connectivity_relays : bool = connectivity_relays
        self._t_0 = None
        self._t_f = None
        self._t_curr = np.NINF
        self._agent_state_update_times = {}

        # initialize data sinks for historical data
        # self._task_reqs : list[dict] = list()
        self._task_reqs = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="requests")       
        self._observation_history = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="measurements")
        self._broadcasts_history = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="broadcasts")

        # initialize current connectivity to connectivity at t=0.0 
        self._current_connectivity_interval, \
            self._current_connectivity_components, self._current_connectivity_map \
            = self.__get_agent_connectivity(t=0.0) # serve as references for connectivity at current time
        

    """
    ----------------------
    SIMULATION CYCLE METHODS
    ----------------------
    """
    def __update_state(self, t : float) -> None:
        """ Updates the environment state at time `t` """
        # check time progression
        assert t >= self._t_curr, \
            f"Environment time cannot move backwards (current time {self._t_curr}[s], new time {t}[s])"
        
        # update current time
        self._t_curr = t

        # check if connectivity needs to be update
        if t not in self._current_connectivity_interval:
            # update current connectivity matrix and components
            self._current_connectivity_interval, \
                self._current_connectivity_components, self._current_connectivity_map \
                    = self.__get_agent_connectivity(t)
        
        # INSERT ADDITIONAL ENVIRONMENT UPDATE LOGIC HERE

        # end
        return 
    
    def step(self, 
             state_action_pairs : Dict[str, Tuple[SimulationAgentState, AgentAction]], 
             t_curr : float
            ) -> Dict[str, List]:
        """Updates agent states based on the provided actions at time `t` """
        # update internal time and state
        self.__update_state(t_curr)
        
        # initialize agent update storage
        states = dict()
        actions = dict()
        action_statuses = dict()
        msgs : Dict[str, List] = defaultdict(list)
        measurements : dict[str, List] = defaultdict(list)
        
        # iterate through each agent state-action pair
        for agent_name, (state, action) in state_action_pairs.items():                
            # handle action 
            updated_state, updated_action_status, msgs_out, agent_observations \
                = self.__handle_agent_action(state, action, t_curr)

            assert abs(updated_state.get_time() - t_curr) < 1e-6, \
                f"Agent `{agent_name}` state time {updated_state.get_time()}[s] does not match current time {t_curr}[s] after action handling."

            # store updated state and action status
            states[agent_name] = updated_state
            actions[agent_name] = action
            action_statuses[agent_name] = updated_action_status
            
            # store outgoing messages depending on current connectivity
            for receiver in self._current_connectivity_map[agent_name]:
                msgs[receiver].extend(msgs_out)

            # store observations
            if agent_observations:
                measurements[agent_name].extend(agent_observations)

        # compile senses per agent
        agent_percepts : Dict[str, tuple] = dict()
        for agent_name in states.keys():
            agent_percepts[agent_name] = (
                states[agent_name],
                actions[agent_name],
                action_statuses[agent_name],
                msgs[agent_name],
                measurements.get(agent_name, [])
            )

        # return compiled senses
        return agent_percepts

    def __handle_agent_action(self, 
                              state : SimulationAgentState, 
                              action : AgentAction, 
                              t_curr : float
                            ) -> Tuple[SimulationAgentState, AgentAction, List[SimulationMessage], List[dict]]:
        """ Handles the effects of an agent action on the environment """
        if action is None:
            # no action; idle by default
            state.update(t_curr, status=SimulationAgentState.IDLING)
            return state, ActionStatuses.COMPLETED.value, [], []

        # check action start and end times
        if (action.t_start - t_curr) > 1e-6:
            raise RuntimeError(f"agent {state.agent_name} attempted to perform action of type {action.action_type} before it started (start time {action.t_start}[s]) at time {t_curr}[s]")
        if (action.t_end - t_curr) < -1e-6:
            raise RuntimeError(f"agent {state.agent_name} attempted to perform action of type {action.action_type} after it ended (start/end times {action.t_start}[s], {action.t_end}[s]) at time {t_curr}[s]")
        
        # handle action by type
        if (action.action_type == ActionTypes.IDLE.value         
            or action.action_type == ActionTypes.TRAVEL.value
            or action.action_type == ActionTypes.MANEUVER.value): 
            # perform state-updating action
            return self.__perform_state_action(state, action, t_curr)

        elif action.action_type == ActionTypes.BROADCAST.value:
            # perform message broadcast
            return self.__perform_broadcast(state, action, t_curr)

        elif action.action_type == ActionTypes.OBSERVE.value:                              
            # perform observation
            return self.__handle_observation(state, action, t_curr)
        
        elif action.action_type == ActionTypes.WAIT.value:
            # wait for incoming messages
            return self.__perform_wait(state, action, t_curr)
        
        # unknown action type; raise error
        raise ValueError(f"Unknown action type {action.action_type} for agent {state.agent_name}")
        

    def __perform_state_action(self,
                             state : SimulationAgentState,
                             action : AgentAction,
                             t_curr : float) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs the given action on the agent state at time `t_curr` """        
        # update agent state
        action.status,_ = state.perform_action(action, t_curr)

        # return updated state
        return state, action.status, [], []

    def __perform_broadcast(self, 
                          state : SimulationAgentState,
                          action : BroadcastMessageAction, 
                          t_curr : float
                        ) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs a message broadcast action """
        # extract message from action
        if isinstance(action.msg, dict):
            msg_out : SimulationMessage = message_from_dict(**action.msg)
        else:
            msg_out : SimulationMessage = action.msg

        # mark state status as messaging
        state.update(t_curr, status=SimulationAgentState.MESSAGING)

        # save broadcast contents according to message type
        if isinstance(msg_out, BusMessage):
            for msg in msg_out.msgs:
                if isinstance(msg, MeasurementRequestMessage):
                    # save measurement request to history
                    self._task_reqs.append(msg.req)
        elif isinstance(msg_out, MeasurementRequestMessage):
            # save measurement request to history
            self._task_reqs.append(msg_out.req)


        # save broadcast to history 
        if isinstance(msg_out, BusMessage):
            # convert to dict before saving
            msg_dict : dict = msg_out.to_dict()

            # remove additional messages to avoid memory issues; only save main message info
            msg_dict.pop('msgs', None)

            # save main message info to history
            self._broadcasts_history.append(msg_dict)

        else:
            # save broadcast to history directly as dict
            self._broadcasts_history.append(msg_out.to_dict())

        
        # log broadcast event
        return state, ActionStatuses.COMPLETED.value, [msg_out], []
    
    def __handle_observation(self, 
                            state : SimulationAgentState,
                            action : ObservationAction, 
                            t_curr : float
                        ) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs an observation action """
        
        # unpack message
        agent_state_dict = action.req['agent_state']
        instrument_dict = action.req['instrument']
        t_start = max(action.req['t_start'], t_curr)
        t_end = action.req['t_end']

        assert t_end >= t_start, \
            f"Invalid observation action times: t_start={action.req['t_start']}[s], t_end={t_end}[s], t_img={t_curr}[s]"

        # find/generate measurement results
        observation_data = self._query_measurement_data(agent_state_dict, instrument_dict, t_start, t_end)

        # package observation data
        obs_data : tuple = (instrument_dict['name'].lower(), observation_data)

        # update state status
        state.update(t_curr, status=SimulationAgentState.MEASURING)    

        # save observation to history
        resp = action.req.copy()
        resp['observation_data'] = observation_data
        self._observation_history.extend(observation_data)    

        # return packaged results
        return state, ActionStatuses.COMPLETED.value, [], [obs_data]
    
    def __perform_wait(self, 
                     state : SimulationAgentState,
                     action : WaitAction, 
                     t_curr : float
                    ) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs a wait action for incoming messages """
        # update state
        state.update(t_curr, status=SimulationAgentState.WAITING)
        
        # check if task was completed
        completed = t_curr > action.t_end or abs(t_curr - action.t_end) < 1e-6

        # use completion to determine action status 
        status = ActionStatuses.COMPLETED.value if completed else ActionStatuses.ABORTED.value

        return state, status, [], []
    
    """
    ----------------------
    ORBITDATA QUERY METHODS
    ----------------------
    """
    def _query_measurement_data(self,
                                agent_state_dict : dict, 
                                instrument_dict : dict,
                                t_start : float,
                                t_end : float
                                ) -> dict:
        """
        Queries internal models or data and returns observation information being sensed by the agent
        """

        # if isinstance(agent_state, SatelliteAgentState):
        if agent_state_dict['state_type'] == SimulationAgentTypes.SATELLITE.value:
            # get orbit data for the agent
            agent_orbitdata : OrbitData = self._orbitdata[agent_state_dict['agent_name']]

            # get access data for the agent
            raw_access_data : Dict[str, list] = agent_orbitdata.gp_access_data.lookup_interval(t_start, t_end)
                        
            # get satellite's off-axis angle
            satellite_off_axis_angle = agent_state_dict['attitude'][0]
            
            # collect instrument information
            instrument_name = instrument_dict["name"]
            instruments = np.asarray(raw_access_data["instrument"], dtype=str)
            ID_COLS = {'instrument', 'agent name', 'grid index', 'GP index',
           'lat [deg]', 'lon [deg]', 'pnt-opt index'}
            
            # create instrument mask for data filtering
            inst_mask = (instruments == instrument_name)

            # collect data for every instrument model onboard
            obs_data = []
            for instrument_model in instrument_dict['mode']:
                # get observation FOV from instrument model
                if instrument_model['@type'] == 'Basic Sensor':
                    instrument_off_axis_fov = instrument_model['fieldOfViewGeometry']['angleWidth'] / 2.0
                elif instrument_model['@type'] == 'Passive Optical Scanner':
                    instrument_off_axis_fov = instrument_model['fieldOfViewGeometry']['angleWidth'] / 2.0
                else:
                    raise NotImplementedError(f"measurement data query not yet suported for sensor models of type {instrument_model['model_type']}.")

                # query coverage data of everything that is within the field of view of the agent
                # TODO Add along-track angle checking. Currently assumes that only cross-track maneuverability is available

                angles = np.asarray(raw_access_data["off-nadir axis angle [deg]"])
                angles_inst = angles[inst_mask]            # smaller array

                mask = np.abs(angles_inst - satellite_off_axis_angle) <= instrument_off_axis_fov

                matching_data = {col: np.asarray(vals)[inst_mask][mask] 
                                    for col, vals in tqdm(raw_access_data.items(), 
                                                       desc=f"{SimulationRoles.ENVIRONMENT.value}-Filtering access data for instrument {instrument_name}...", 
                                                       leave=False,
                                                       disable=len(raw_access_data)<10 or not self._printouts)
                                }
                
                # convert columns to arrays once
                cols = {k: np.asarray(v) for k, v in matching_data.items()}
                grid = cols['grid index'].astype(np.int64, copy=False)
                gp   = cols['GP index'].astype(np.int64, copy=False)
                time = cols['time [s]']

                # check if there is any data to process
                if len(time) == 0: continue

                # ---- Build unique groups for (grid, gp) efficiently ----
                # Stack into (n,2) and unique rows
                pairs = np.column_stack((grid, gp))  # shape (n,2)
                _, inv = np.unique(pairs, axis=0, return_inverse=True)
                # inv[i] = group id of row i, groups are 0..G-1

                # Sort rows by group id so each group is contiguous
                order = np.argsort(inv, kind="mergesort")
                inv_sorted = inv[order]

                # Find group boundaries in the sorted order
                # starts: indices in `order` where a new group begins
                starts = np.r_[0, np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1]
                ends   = np.r_[starts[1:], len(order)]

                obs_data: list[dict] = []

                # Iterate groups (G is usually much smaller than N)
                for s,e in tqdm(zip(starts, ends), 
                                desc=f"{SimulationRoles.ENVIRONMENT.value}-Merging observation data for instrument {instrument_name}...", 
                                unit=' obs',
                                disable=len(starts)<10 or not self._printouts,
                                leave=False):
                    idx = order[s:e]  # row indices for this group

                    merged = {
                        't_start': float(np.min(time[idx])),
                        't_end':   float(np.max(time[idx])),
                    }

                    # For ID columns: take first value
                    # For other columns: collect list (or scalar if length 1)
                    for col, arr in cols.items():
                        if col in ID_COLS:
                            v = arr[idx[0]]
                            merged[col] = v.item() if hasattr(v, "item") else v
                        else:
                            v = arr[idx]
                            # Convert numpy scalars to Python types if needed
                            lst = [x.item() if hasattr(x, "item") else x for x in v.tolist()]
                            # merged[col] = lst[0] if len(lst) == 1 else lst
                            merged[col] = lst[-1] # always take last value to reflect changes in observed parameters along the observation window (e.g. for moving targets)

                    obs_data.append(dict(merged))

            # return processed observation data
            return obs_data

        else:
            raise NotImplementedError(f"Measurement results query not yet supported for agents with state of type {agent_state_dict['state_type']}")
    

    """
    ----------------------
    RESULTS HANDLING METHODS
    ----------------------
    """
    def log(self, msg : str, level=logging.DEBUG) -> None:
        """
        Logs a message to the desired level.
        """
        try:
            # check if printouts are enabled; if not, skip logging
            if not self._printouts: return

            t = self._t_curr
            t = t if t is None else round(t,3)

            if level is logging.DEBUG:
                self._logger.debug(f'T={t}[s] | {SimulationRoles.ENVIRONMENT.value}: {msg}')
            elif level is logging.INFO:
                self._logger.info(f'T={t}[s] | {SimulationRoles.ENVIRONMENT.value}: {msg}')
            elif level is logging.WARNING:
                self._logger.warning(f'T={t}[s] | {SimulationRoles.ENVIRONMENT.value}: {msg}')
            elif level is logging.ERROR:
                self._logger.error(f'T={t}[s] | {SimulationRoles.ENVIRONMENT.value}: {msg}')
            elif level is logging.CRITICAL:
                self._logger.critical(f'T={t}[s] | {SimulationRoles.ENVIRONMENT.value}: {msg}')
        
        except Exception as e:
            raise e

    def print_results(self) -> None:
        try:
            # set final simulation time
            self.t_f = time.perf_counter()

            # log results compilation start
            self.log('Compiling results...',level=logging.WARNING)

            # compile observations performed
            self._observation_history.close()
            if self._observation_history.empty():
                self.log("No observations were performed during the simulation.", level=logging.WARNING)
                # create empty dataframe with appropriate columns
                observations_performed = pd.DataFrame(data=[])
                observations_performed.to_parquet(f"{self._results_path}/measurements.parquet", index=False)

            # observations_performed : pd.DataFrame = self.compile_observations()

            # # log and save results
            # # self.log(f"MEASUREMENTS RECEIVED:\n{len(observations_performed.values)}\n\n", level=logging.WARNING)
            # observations_performed.to_parquet(f"{self._results_path}/measurements.parquet", index=False)
            
            # commpile list of broadcasts performed
            self._broadcasts_history.close()
            if self._broadcasts_history.empty():
                self.log("No broadcasts were performed during the simulation.", level=logging.WARNING)
                # create empty dataframe with appropriate columns
                broadcasts_performed = pd.DataFrame(data=[])
                broadcasts_performed.to_parquet(f"{self._results_path}/broadcasts.parquet", index=False)
            # broadcasts_performed : pd.DataFrame = self.compile_broadcasts()

            # # log and save results
            # # self.log(f"BROADCASTS RECEIVED:\n{len(broadcasts_performed.values)}\n\n", level=logging.WARNING)
            # broadcasts_performed.to_parquet(f"{self._results_path}/broadcasts.parquet", index=False)

            # compile list of measurement requests 
            self._task_reqs.close()
            if self._task_reqs.empty():
                self.log("No observation requests were performed during the simulation.", level=logging.WARNING)
                # create empty dataframe with appropriate columns
                task_requests = pd.DataFrame(data=[])
                task_requests.to_parquet(f"{self._results_path}/requests.parquet", index=False)
            # measurement_reqs : pd.DataFrame = self.compile_requests()

            # # log and save results
            # # self.log(f"MEASUREMENT REQUESTS RECEIVED:\n{len(measurement_reqs.values)}\n\n", level=logging.WARNING)
            # measurement_reqs.to_parquet(f"{self._results_path}/requests.parquet", index=False)

            # print connectivity history
            self.__print_connectivity_history()

        except Exception as e:
            print('\n','\n','\n')
            print(e)
            raise e     
                    
    # def compile_observations(self) -> pd.DataFrame:
    #     try:
    #         columns = None
    #         data = []
            
    #         for msg_dict in tqdm(self._observation_history, 
    #                         desc='Compiling observations results', 
    #                         leave=False,
    #                         disable=len(self._observation_history)<10 or not self._printouts
    #                         ):
    #             msg = ObservationResultsMessage(**msg_dict)
    #             observation_data : List[dict] = msg.observation_data
    #             observer = msg.dst

    #             for obs in observation_data:
                    
    #                 # find column names 
    #                 if columns is None:
    #                     columns = sorted([key for key in obs])
    #                     columns.insert(0, 'observer')
    #                     # columns.insert(2, 't_img')
    #                     # columns.remove('t_start')
    #                     # columns.remove('t_end')

    #                 # add observation to data list
    #                 obs['observer'] = observer.lower()
    #                 for key in columns:
    #                     val = obs.get(key, None)
    #                     if isinstance(val, list):
    #                         obs[key] = val[0]
    #                         # if len(val) == 1:
    #                         #     obs[key] = val[0]
    #                         # else:
    #                         #     obs[key] = [val[0], val[-1]]

    #                 # obs['t_img'] = [obs['t_start'], obs['t_end']]
    #                 # obs.pop('t_start')
    #                 # obs.pop('t_end')

    #                 data.append([obs[key] for key in columns])

    #         observations_df = pd.DataFrame(data=data, columns=columns)
    #         observations_df = observations_df.sort_index(axis=1)

    #         return observations_df
        
    #     except Exception as e:
    #         print(e.with_traceback())
    #         raise e
    
    # def compile_broadcasts(self) -> pd.DataFrame:
    #     columns = ['t_msg', 'sender', 'message type', 
    #             #    'Message'
    #                ]

    #     data = [[msg['t_msg'], 
    #              msg['src'], 
    #              msg['msg_type'],
    #              #  json.dumps(msg)
    #              ]
    #             for msg in self._broadcasts_history]
            
    #     return pd.DataFrame(data=data, columns=columns)
    
    # def compile_requests(self) -> pd.DataFrame:
    #     # convert measurement request dictionaries to Task Requests
    #     self._task_reqs : list[TaskRequest] = list({
    #         TaskRequest.from_dict(req_dict)
    #         for req_dict in self._task_reqs})

    #     columns = ['request id', 'requester', 'event id', 'parameter', 't_req', 'mission name']
    #     data = [[req.id,
    #              req.requester,
    #              req.task.event.id,
    #              req.task.parameter,
    #              req.t_req,
    #              req.mission_name,
    #              ] 
    #              for req in self._task_reqs
    #              if isinstance(req.task, EventObservationTask)]
        
    #     assert all(isinstance(req.task, EventObservationTask) for req in self._task_reqs), \
    #         'Only `EventObservationTask` measurement requests are currently supported in the results compilation.'

    #     return pd.DataFrame(data=data, columns=columns)    

    def __print_connectivity_history(self) -> None:
        filename = 'connectivity.md'
        filepath = os.path.join(self._results_path, filename)
        with open(filepath, 'w') as f:
            f.write(f"# Agent Connectivity History\n")
            f.write(f"- Connectivity Level: **{self._connectivity_level}**\n\n")
            f.write(f"- Connectivity Relays Enabled: **{self._connectivity_relays}**\n\n")

            for interval, components, component_map in self.__interval_connectivities:
                f.write('---\n')
                f.write(f"**Interval:** {interval} [s]\n\n")
                
                # conn_matrix = conn_matrix_sparse.toarray()
                agent_names = list(component_map.keys())
                agent_to_idx = {agent: idx for idx, agent in enumerate(agent_names)}

                # print connectivity matrix
                f.write("**Connectivity Matrix:**\n\n")
                ## print table header 
                header = "||" + "  |".join([f"`{name:>5}`" for name in agent_names]) + "|\n"
                f.write(header)
                f.write("|-|" + "-|"*len(agent_names) + "\n")
                
                # ## print matrix rows
                # for sender in agent_names:
                #     row = f"|`{sender:>5}`|"
                #     u = agent_to_idx[sender]
                #     for receiver in agent_names:
                #         v = agent_to_idx[receiver]
                #         status = int(bool(conn_matrix[u][v]) or bool(conn_matrix[v][u]))
                #         row += f"{status:>3}|"
                #     f.write(row + "\n")

                f.write("\n**Connected Components:**\n")
                for i,comp in enumerate(sorted(components)):
                    f.write(f" - Component {i}: [`" + "`, `".join(comp) + "`]\n\n")
                f.write("\n\n")

    @staticmethod
    def __display_connectivity_matrix(conn_matrix : Dict[str, Dict[str, int]]) -> None:
        """ Prints connectivity matrix to console """
        agent_names = list(conn_matrix.keys())
        header = "\t\t" + "  ".join([f"{name:>5}" for name in agent_names])
        print(header)
        for sender in agent_names:
            row = f"{sender:>5}\t"
            if len(sender) < 8:
                row += "\t"
            for receiver in agent_names:
                row += f"{conn_matrix[sender][receiver]:>3}\t"
            print(row)

    @staticmethod
    def __display_connected_components(components : List[Set[str]]) -> None:
        """ Prints connected components to console """
        print("Connected Components:")
        for i,comp in enumerate(sorted(components)):
            print(f" - Component {i}: " + ", ".join(comp))

    """
    ---------------------------
    UTILITY METHODS
    ---------------------------
    """
    @staticmethod
    def __precompute_connectivity(orbitdata : Dict[str, OrbitData], printouts : bool) -> List[tuple]:
        """ 
        Precomputes initial connectivity matrix for all agents 
        
        ### Returns 
            - interval_connectivities (`List[tuple]`): list of tuples of the form (interval, connectivity_matrix, components_list)
        """
        # compile event markers for changes in connectivity 
        connectivity_events : List[tuple] = []
        for sender,agent_orbitdata in orbitdata.items():
            for receiver,data in agent_orbitdata.comms_links.items():
                for t_start,t_end,*_ in data:
                    connectivity_events.append( (t_start, sender, receiver, 1) )   # link comes online
                    connectivity_events.append( (t_end, sender, receiver, 0) )     # link goes offline

        # sort events by time
        connectivity_events.sort(key=lambda x: x[0])

        # extract unique event times
        unique_event_times : List[float] = sorted(set([evt[0] for evt in connectivity_events]))

        # group unique event times into intervals 
        connectivity_intervals : List[Interval] = []
        for i,_ in enumerate(unique_event_times):
            t_start = unique_event_times[i]
            t_end = unique_event_times[i+1] if i + 1 < len(unique_event_times) else np.Inf
            connectivity_intervals.append( Interval(t_start, t_end, right_open=True) )

        # initialize previous connectivity matrix
        t_start = unique_event_times[0] if unique_event_times else np.Inf
        prev_interval = Interval(np.NINF, t_start, left_open=True, right_open=True)
        connectivity_intervals.insert(0, prev_interval)
                
        # group events by interval 
        events_per_interval : Dict[Interval, List[tuple]] \
            = {interval : [] for interval in connectivity_intervals}
        
        for evt in tqdm(connectivity_events, 
                        desc='Grouping connectivity events by interval', 
                        unit=' events', 
                        leave=False,
                        disable=not printouts
                    ):
            # group by bisection search, asuuming `connectivity_intervals` is sorted
            low = 0
            high = len(connectivity_intervals) - 1

            while low <= high:
                mid = (low + high) // 2
                interval = connectivity_intervals[mid]

                if evt[0] in interval:
                    events_per_interval[interval].append(evt)
                    break

                if evt[0] < interval.left:
                    high = mid - 1
                else:
                    low = mid + 1  
        
        # initialize interval-connectivity list
        interval_connectivities : List[tuple] = []
        agent_to_idx = {agent: idx for idx, agent in enumerate(orbitdata.keys())}
        idx_to_agent = [agent for agent in agent_to_idx.keys()]
        prev_connectivity_matrix = np.zeros((len(orbitdata), len(orbitdata)), dtype=int)
        
        # create adjacency matrix per interval
        for interval in tqdm(connectivity_intervals, 
                             desc='Precomputing agent connectivity intervals', 
                             unit=' intervals', 
                             leave=False,
                             disable=not printouts
                            ):
            # copy previous connectivity state
            # interval_connectivity_matrix \
            #     = copy.deepcopy(prev_connectivity_matrix)                    
            interval_connectivity_matrix = prev_connectivity_matrix.copy()
            
            # get connectivity events that occur during the interval
            # interval_events = [ evt for evt in connectivity_events if evt[0] in interval ]
            interval_events = events_per_interval[interval]

            # update connectivity matrix based on events
            for _,sender,receiver,status in interval_events:
                u_idx = agent_to_idx[sender]
                v_idx = agent_to_idx[receiver]
                interval_connectivity_matrix[u_idx][v_idx] = status

            # create component list from connectivity matrix
            interval_components = SimulationEnvironment.__get_connected_components(interval_connectivity_matrix, agent_to_idx, idx_to_agent)

            # convert components to dict for easier lookup
            interval_component_map : Dict[str, List[str]] \
                = { sender : [] for sender in idx_to_agent }

            for component in interval_components:
                for sender in component:
                    for receiver in component:
                        if sender != receiver:
                            interval_component_map[sender].append(receiver)

            # convert connectivity matrix to sparse matrix
            sparse_interval_connectivity_matrix = csr_matrix(interval_connectivity_matrix)
            
            # save as upper triangle only to save memory, since connectivity is symmetric 
            sparse_interval_connectivity_matrix = triu(sparse_interval_connectivity_matrix, k=0)   # keep diagonal + upper triangle only

            # store interval connectivity data
            interval_connectivities.append( (interval, 
                                             interval_components, interval_component_map) )

            # set previous connectivity to current for next iteration
            prev_connectivity_matrix = interval_connectivity_matrix

            # DEBUG PRINTOUTS -------------
            # print(F"Connectivity during interval {interval}:")
            # print('-'*50)
            # SimulationEnvironment.__display_connectivity_matrix(interval_connectivity_matrix)
            # print('-'*50)
            # SimulationEnvironment.__display_connected_components(interval_components)
            # print('='*50 + '\n')
            # x = 1 # breakpoint
            # -------------------------------
        
        # return compiled list of interval connectivities
        # return interval_connectivities

        merged_intervals : List[Tuple[Interval, Set[frozenset], Dict[str, List[str]]]] = []
        for interval, components, component_map in tqdm(interval_connectivities, 
                                                        desc='Merging agent connectivity intervals', 
                                                        unit=' intervals', 
                                                        leave=False,
                                                        disable=not printouts
                                                    ):
            if not merged_intervals:
                merged_intervals.append( (interval, components, component_map) )
                continue

            # get previous interval data
            prev_interval, prev_components, prev_component_map = merged_intervals[-1]
            
            # compare components to determine if connectivity changed; if not, merge intervals
            if components == prev_components and prev_interval.overlaps(interval):
                # merge intervals by updating end time of previous interval to end time of current interval
                merged_interval = prev_interval.union(interval)
                merged_intervals[-1] = (merged_interval, prev_components, prev_component_map)

            else:
                # connectivity changed; add current interval as new entry
                merged_intervals.append( (interval, components, component_map) )
        
        return merged_intervals

    
    @staticmethod
    # def __get_connected_components(adj: Dict[str, Dict[str, int]]):
    def __get_connected_components(adj: List[List[int]], agent_to_idx : Dict[str, int], idx_to_agent : List[str]) -> List[List[str]]:
        """
        adj: dict[node] -> dict[neighbor] -> weight/int (nonzero means edge exists)
        Assumes undirected (symmetric) or at least that reachability should be treated undirected.
        Returns: list of components (each is list of node names)
        """
        # initialize BFS variables
        visited = set()
        comps = set()

        # BFS to find components
        for start in range(len(adj)):
            # skip visited nodes
            if start in visited: continue

            # initialize queue with starting node as root
            q = deque([start])

            # mark root as visited 
            visited.add(start)

            # explore subgraph components starting from root
            comp = set()
            while q:
                # pop next node
                u = q.popleft()

                # add to component list
                comp.add(idx_to_agent[u])

                # iterate neighbors; keep only truthy edges
                for v, connected in enumerate(adj[u]):
                    # check if connected to neighbor
                    if not connected: continue
                    
                    # check neighbor has been visited
                    if v not in visited:                        
                        visited.add(v)
                        q.append(v)

            # add component to component list
            comps.add(frozenset(comp))

        # return list of components
        return comps
    
    def __get_agent_connectivity(self, t : float) -> tuple:
        """ Searches and returns the current connectivity matrix and components for agents at time `t` """
        
        # use binary search to find the correct interval
        low = 0
        high = len(self.__interval_connectivities) - 1
        
        while low <= high:
            # find mid index
            mid = (low + high) // 2

            # unpack interval data
            interval, components, components_map \
                = self.__interval_connectivities[mid]

            # return if time t is in the interval
            if t in interval: return interval, components, components_map
            
            # if not, adjust search bounds
            if t < interval.left:
                high = mid - 1
            else:
                low = mid + 1

        # time t not found in any interval
        raise ValueError(f'Time {t}[s] not found in any precomputed connectivity interval.')