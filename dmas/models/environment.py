import copy
import logging
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm

from dmas.core.messages import SimulationMessage, message_from_dict
from dmas.core.orbitdata import OrbitData

from execsatm.events import GeophysicalEvent
from execsatm.utils import Interval

from dmas.models.actions import *
from dmas.models.agent import SimulationAgent
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
                connectivity_level : str = 'LOS',
                connectivity_relays : bool = False,
                level: int = logging.INFO, 
                logger: logging.Logger = None
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
        self.__interval_connectivities : List[Tuple[Interval, Dict, List, Dict]] \
            = SimulationEnvironment.__precompute_connectivity(scenario_orbitdata) # list of (interval, connectivity_matrix, components_list)

        # initialize parameters
        self._connectivity_relays : bool = connectivity_relays
        self._t_0 = None
        self._t_f = None
        self._agent_state_update_times = {}
        self._task_reqs : list[dict] = list()        

        self._observation_history = []
        self._broadcasts_history = []

        self._current_connectivity_interval, self._current_connectivity_matrix, \
            self._current_connectivity_components, self._current_connectivity_map \
            = self.__get_agent_connectivity(t=0.0) # serve as references for connectivity at current time
        

    """
    ----------------------
    SIMULATION CYCLE METHODS
    ----------------------
    """
    def update_state(self, t : float) -> None:
        """ Updates the environment state at time `t` """
        
        # check if connectivity needs to be update
        if t not in self._current_connectivity_interval:
            # update current connectivity matrix and components
            self._current_connectivity_interval, self._current_connectivity_matrix, \
                self._current_connectivity_components, self._current_connectivity_map \
                    = self.__get_agent_connectivity(t)
        
        # INSERT ADDITIONAL ENVIRONMENT UPDATE LOGIC HERE

        # end
        return 
    
    def update_agents(self, 
                            state_action_pairs : Dict[str, Tuple[SimulationAgentState, AgentAction]], 
                            t_curr : float) -> Dict[str, List]:
        """Updates agent states based on the provided actions at time `t` """
        # initialize agent update storage
        states = dict()
        actions = dict()
        action_statuses = dict()
        msgs : Dict[str, List] = defaultdict(list)
        observations : dict[str, List] = defaultdict(list)
        
        # iterate through each agent state-action pair
        for agent_name, (state, action) in state_action_pairs.items():            
            # handle action 
            updated_state, updated_action_status, msgs_out, agent_observations \
                = self.__handle_agent_action(state, action, t_curr)

            # store updated state and action status
            states[agent_name] = updated_state
            actions[agent_name] = action
            action_statuses[agent_name] = updated_action_status
            
            # store outgoing messages depending on current connectivity
            for msg in msgs_out:
                # determine recipients based on current connectivity
                for receiver in self._current_connectivity_map[agent_name]:
                    # add message to receiver's inbox
                    msgs[receiver].append(msg)                  

            # store observations
            if agent_observations:
                observations[agent_name].append(agent_observations)

        # compile senses per agent
        senses : Dict[str, tuple] = dict()
        for agent_name in states.keys():
            senses[agent_name] = (
                states[agent_name],
                actions[agent_name],
                action_statuses[agent_name],
                msgs[agent_name],
                observations.get(agent_name, [])
            )

        # return compiled senses
        return senses

    def __handle_agent_action(self, 
                              state : SimulationAgentState, 
                              action : AgentAction, 
                              t_curr : float
                            ) -> Tuple[SimulationAgentState, AgentAction, List[SimulationMessage], List[dict]]:
        """ Handles the effects of an agent action on the environment """
        if action is None:
            # no action; idle by default
            state.update(t_curr, status=SimulationAgentState.IDLING)
            return state, ActionStatuses.COMPLETED.value, [], {}

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
            return self.perform_state_action(state, action, t_curr)

        elif action.action_type == ActionTypes.BROADCAST.value:
            # perform message broadcast
            return self.perform_broadcast(state, action, t_curr)

        elif action.action_type == ActionTypes.OBSERVE.value:                              
            # perform observation
            return self.perform_observation(state, action, t_curr)
        
        elif action.action_type == ActionTypes.WAIT.value:
            # wait for incoming messages
            return self.perform_wait(state, action, t_curr)
        
        # unknown action type; raise error
        raise ValueError(f"Unknown action type {action.action_type} for agent {state.agent_name}")
        

    def perform_state_action(self,
                             state : SimulationAgentState,
                             action : AgentAction,
                             t_curr : float) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs the given action on the agent state at time `t_curr` """
        # update agent state
        action.status,_ = state.perform_action(action, t_curr)

        # return updated state
        return state, action.status, [], []

    def perform_broadcast(self, 
                          state : SimulationAgentState,
                          action : BroadcastMessageAction, 
                          t_curr : float
                        ) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs a message broadcast action """
        # extract message from action
        msg_out : SimulationMessage = message_from_dict(**action.msg)

        # mark state status as messaging
        state.update(t_curr, status=SimulationAgentState.MESSAGING)

        # log broadcast event
        return state, ActionStatuses.COMPLETED.value, [msg_out], []
    
    def perform_observation(self, 
                            state : SimulationAgentState,
                            action : ObservationAction, 
                            t_curr : float
                        ) -> Tuple[SimulationAgentState, str, list, list]:
        """ Performs an observation action """
        # TODO handle observation action parameters
        # RETURN list of (instrument,observation_data) tuples for observations
        raise NotImplementedError("Observation action handling not yet implemented.")
                
        # own_observations : list[tuple] = [(msg.instrument['name'], msg.observation_data) 
        #                                   for msg in incoming_messages 
        #                                   if isinstance(msg, ObservationResultsMessage)
        #                                   and isinstance(msg.instrument, dict)]

        # simulate observation data collection
        # obs_data : dict = {
        #     'observation': f"Data collected by {state.agent_name} at time {t_curr}[s]"
        # }

        # # mark state status as measuring
        # state.update(t_curr, status=SimulationAgentState.MEASURING)

        # # log observation event
        # return state, ActionStatuses.COMPLETED.value, [], obs_data
    
    def perform_wait(self, 
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

    """
    ----------------------
    RESULTS HANDLING METHODS
    ----------------------
    """
    def print_results(self) -> str:
        # TODO 
        ...

    def __print_connectivity_matrix(self, conn_matrix : Dict[str, Dict[str, int]]) -> None:
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

    def __print_connected_components(self, components : List[Set[str]]) -> None:
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
    def __precompute_connectivity(orbitdata : Dict[str, OrbitData]) -> List[tuple]:
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
        prev_connectivity_matrix = {sender : {receiver : 0 for receiver in orbitdata.keys()} 
                             for sender in orbitdata.keys()}
        
        # group events by interval 
        events_per_interval : Dict[Interval, List[tuple]] \
            = {interval : [] for interval in connectivity_intervals}
        
        for evt in tqdm(connectivity_events, desc='Grouping connectivity events by interval', unit=' events', leave=False):
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
        
        # create adjacency matrix per interval
        for interval in tqdm(connectivity_intervals, desc='Precomputing agent connectivity intervals', unit=' intervals', leave=False):
            # copy previous connectivity state
            interval_connectivity_matrix \
                = copy.deepcopy(prev_connectivity_matrix)                    
            
            # get connectivity events that occur during the interval
            # interval_events = [ evt for evt in connectivity_events if evt[0] in interval ]
            interval_events = events_per_interval[interval]

            # update connectivity matrix based on events
            for _,sender,receiver,status in interval_events:
                interval_connectivity_matrix[sender][receiver] = status

            # create component list from connectivity matrix
            interval_components = SimulationEnvironment.get_connected_components(interval_connectivity_matrix)

            # convert components to dict for easier lookup
            interval_component_map : Dict[str, List[str]] \
                = { sender : [] for sender in interval_connectivity_matrix.keys() }

            for component in interval_components:
                for sender in component:
                    for receiver in component:
                        if sender != receiver:
                            interval_component_map[sender].append(receiver)

            # store interval connectivity data
            interval_connectivities.append( (interval, interval_connectivity_matrix, 
                                             interval_components, interval_component_map) )

            # set previous connectivity to current for next iteration
            prev_connectivity_matrix = interval_connectivity_matrix

            # DEBUG PRINTOUTS -------------
            # print(F"Connectivity during interval {interval}:")
            # print('-'*50)
            # self.__print_connectivity_matrix(interval_connectivity_matrix)
            # print('-'*50)
            # self.__print_connected_components(interval_components)
            # print('='*50 + '\n')
            # x = 1 # breakpoint
            # -------------------------------

        # return compiled list of interval connectivities
        return interval_connectivities
    
    @staticmethod
    def get_connected_components(adj: Dict[str, Dict[str, int]]):
        """
        adj: dict[node] -> dict[neighbor] -> weight/int (nonzero means edge exists)
        Assumes undirected (symmetric) or at least that reachability should be treated undirected.
        Returns: list of components (each is list of node names)
        """
        # initialize BFS variables
        visited = set()
        comps = []

        # BFS to find components
        for start in adj.keys():
            # skip visited nodes
            if start in visited: continue

            # initialize queue with starting node as root
            q = deque([start])

            # mark root as visited 
            visited.add(start)

            # explore subgraph components starting from root
            comp = []
            while q:
                # pop next node
                u = q.popleft()

                # add to component list
                comp.append(u)

                # iterate neighbors; keep only truthy edges
                for v, connected in adj[u].items():
                    # check if connected to neighbor
                    if not connected: continue
                    
                    # check neighbor has been visited
                    if v not in visited:                        
                        visited.add(v)
                        q.append(v)

            # add component to component list
            comps.append(comp)

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
            interval, connectivity, components, components_map \
                = self.__interval_connectivities[mid]

            # return if time t is in the interval
            if t in interval: return interval, connectivity, components, components_map
            
            # if not, adjust search bounds
            if t < interval.left:
                high = mid - 1
            else:
                low = mid + 1

        # time t not found in any interval
        raise ValueError(f'Time {t}[s] not found in any precomputed connectivity interval.')