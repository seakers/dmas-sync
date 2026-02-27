import logging
import os
import traceback
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd
from tqdm import tqdm

from dmas.core.messages import BusMessage, MeasurementRequestMessage, SimulationMessage, message_from_dict
from dmas.models.trackers import DataSink
from dmas.utils.orbitdata import OrbitData

from execsatm.events import GeophysicalEvent
from execsatm.utils import Interval

from dmas.models.actions import *
from dmas.models.states import SimulationAgentState, SimulationAgentTypes
from dmas.utils.tools import SimulationRoles


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

        # initialize parameters
        self._t_curr = np.NINF
        self._agent_state_update_times = {}

        # initialize data sinks for historical data
        self._task_reqs = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="requests")       
        self._observation_history = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="measurements")
        self._broadcasts_history = DataSink(out_dir=env_results_path, owner_name=SimulationRoles.ENVIRONMENT.value.lower(), data_name="broadcasts")

        # initialize agent connectivity
        self._current_connectivity_interval, \
            self._current_connectivity_components, \
                self._current_connectivity_map \
                    = Interval(np.NINF, 0.0, right_open=True), None, None

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
                self._current_connectivity_components, \
                    self._current_connectivity_map \
                        = self.__get_agent_connectivity(t)
        
        # INSERT ADDITIONAL ENVIRONMENT UPDATE LOGIC HERE

        # end
        return 
    
    def step(self, 
             state_action_pairs : Dict[str, Tuple[SimulationAgentState, AgentAction]], 
             t_curr : float
            ) -> Dict[str, List[Tuple[SimulationAgentState, AgentAction, str, List[SimulationMessage], List[dict]]]]:
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

        # convert to dict before saving
        msg_dict : dict = {
            'src' : msg_out.src,
            'dst' : msg_out.dst,
            'msg_type' : msg_out.msg_type,
            't_broadcast' : t_curr,
            'id' : msg_out.id
        }

        # save broadcast to history directly as dict
        self._broadcasts_history.append(msg_dict)
        
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
        t_start = action.req['t_start']
        t_end = max(action.req['t_end'], t_curr)

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
        # log results compilation start
        self.log('Compiling results...',level=logging.WARNING)

        # compile observations performed
        self._observation_history.close()
        if self._observation_history.empty():
            self.log("No observations were performed during the simulation.", level=logging.WARNING)
            # create empty dataframe with appropriate columns
            observations_performed = pd.DataFrame(data=[])
            observations_performed.to_parquet(f"{self._results_path}/measurements.parquet", index=False)
        
        # commpile list of broadcasts performed
        self._broadcasts_history.close()
        if self._broadcasts_history.empty():
            self.log("No broadcasts were performed during the simulation.", level=logging.WARNING)
            # create empty dataframe with appropriate columns
            broadcasts_performed = pd.DataFrame(data=[])
            broadcasts_performed.to_parquet(f"{self._results_path}/broadcasts.parquet", index=False)

        # compile list of measurement requests 
        self._task_reqs.close()
        if self._task_reqs.empty():
            self.log("No observation requests were performed during the simulation.", level=logging.WARNING)
            # create empty dataframe with appropriate columns
            task_requests = pd.DataFrame(data=[])
            task_requests.to_parquet(f"{self._results_path}/requests.parquet", index=False)

        # print connectivity history
        self.__print_connectivity_history()                    

    def __print_connectivity_history(self) -> None:
        # create user-readible report of connectivity history
        filename = 'connectivity.md'
        filepath = os.path.join(self._results_path, filename)
        with open(filepath, 'w') as f:
            f.write(f"# Agent Connectivity History\n")
            # get orbitdata for any agent 
            #   (all agents share the same comms links and connectivity interval data)
            agent_orbitdata : OrbitData = next(iter(self._orbitdata.values()))

            # initialize temp connectivity data container
            connectivity_data : List[Tuple[Interval, Set[frozenset], Dict[str, Set[str]]]] = []

            # iterate through list of intervals within the simulation data
            # for t_start,t_end,*component_indices in agent_orbitdata.comms_links.iter_rows_raw(t=0, t_max=self._t_curr, include_current=True):
            for t_start,t_end,*component_indices in agent_orbitdata.comms_links.iter_rows_raw_fast(t=0, t_max=self._t_curr, include_current=True):                
                interval, components, component_map \
                    = self.__interpret_agent_connectivity_data(agent_orbitdata, t_start, t_end, *component_indices)
                
                f.write('---\n')
                f.write(f"**Interval:** {interval} [s]\n\n")

                # conn_matrix = conn_matrix_sparse.toarray()
                # agent_names = list(component_map.keys())

                # TODO print connectivity matrix
                # f.write("**Connectivity Matrix:**\n\n")
                # ## print table header 
                # header = "||" + "  |".join([f"`{name:>5}`" for name in agent_names]) + "|\n"
                # f.write(header)
                # f.write("|-|" + "-|"*len(agent_names) + "\n")
                
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
                for component_idx,comp in enumerate(sorted(components)):
                    f.write(f" - Component {component_idx}: [`" + "`, `".join(comp) + "`]\n\n")
                f.write("\n\n")

                connectivity_data.append((interval, components, component_map))

        # compile table of connectivity intervals and components for later analysis
        data : List[dict]= []
        for interval_idx,(interval, components, component_map) in enumerate(connectivity_data):
            for component_idx,component_members in enumerate(components):
                members = sorted(list(component_members))
                data.append({
                    'start [s]' : interval.left,
                    'end [s]' : interval.right,
                    'interval_id' : interval_idx,
                    'component_id' : component_idx,
                    'n_nodes' : len(members),
                    'component' : members
                })
        connectivity_history_df = pd.DataFrame(data=data)

        # save to results folder
        connectivity_history_df.to_parquet(os.path.join(self._results_path, 'connectivity_history.parquet'), index=False)

    # DEPRECATED
    # @staticmethod
    # def __display_connectivity_matrix(conn_matrix : Dict[str, Dict[str, int]]) -> None:
    #     """ Prints connectivity matrix to console """
    #     agent_names = list(conn_matrix.keys())
    #     header = "\t\t" + "  ".join([f"{name:>5}" for name in agent_names])
    #     print(header)
    #     for sender in agent_names:
    #         row = f"{sender:>5}\t"
    #         if len(sender) < 8:
    #             row += "\t"
    #         for receiver in agent_names:
    #             row += f"{conn_matrix[sender][receiver]:>3}\t"
    #         print(row)

    # DEPRECATED
    # @staticmethod
    # def __display_connected_components(components : List[Set[str]]) -> None:
    #     """ Prints connected components to console """
    #     print("Connected Components:")
    #     for i,comp in enumerate(sorted(components)):
    #         print(f" - Component {i}: " + ", ".join(comp))

    """
    ---------------------------
    UTILITY METHODS
    ---------------------------
    """    
    
    def __get_agent_connectivity(self, t : float) -> tuple:
        """ Searches and returns the current connectivity matrix and components for agents at time `t` """
        
        # get orbitdata for any agent 
        #   (all agents share the same comms links and connectivity interval data)
        agent_orbitdata : OrbitData = next(iter(self._orbitdata.values()))

        # iterate through list of intervals in this time period 
        for t_start,t_end,*component_indices in agent_orbitdata.comms_links.iter_rows_raw_fast(t=t, include_current=True):
            # skip if time `t` is not in this interval
            if not (t_start <= t < t_end):
                continue
            
            # return connectivity data for this interval
            return self.__interpret_agent_connectivity_data(agent_orbitdata, t_start, t_end, *component_indices)

        raise ValueError(f"Time {t}[s] not found in any precomputed connectivity interval.")
    
    def __interpret_agent_connectivity_data(self, 
                                            agent_orbitdata : OrbitData, 
                                            t_start : float, 
                                            t_end : float, 
                                            *component_indices
                                        ) -> Tuple[Interval, Set[frozenset], Dict[str, Set[str]]]:
        """ Interprets raw connectivity interval data into a list of components and a connectivity map for the agents in the scenario for the given interval """
        # create connectivity interval
        interval = Interval(t_start, t_end, right_open=True)

        # initialize component membership dict for this interval
        component_membership : Dict[int, Set[str]] = defaultdict(set)

        # group agents by component membership for this interval
        for agent_idx, component_idx in enumerate(component_indices):
            agent_name = agent_orbitdata.comms_target_columns[agent_idx]
            # add agent to component membership dict
            component_membership[component_idx].add(agent_name)

        # convert component membership dict to list of components (as sets)
        components = set(frozenset(members) for members in component_membership.values())

        # create component map for this interval
        component_map : Dict[str, List[str]] \
            = { sender : [] for sender in agent_orbitdata.comms_target_columns }
        
        # populate component map with agents in each component
        for component_idx, members in component_membership.items():
            for member in members:
                # add all other members of the same component to its connectivity list
                component_map[member] = set(members)
                
                # remove self from list of connected agents
                component_map[member].remove(member) 

        # return connectivity data for this interval
        return interval, components, component_map