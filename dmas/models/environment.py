import copy
import logging
import os
from typing import Dict, List, Tuple
from collections import deque

import numpy as np
from tqdm import tqdm

from dmas.core.orbitdata import OrbitData

from execsatm.events import GeophysicalEvent
from execsatm.utils import Interval

from dmas.models.states import SimulationAgentTypes


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
        env_results_path : str = os.path.join(results_path, self.get_element_name().lower())

        # assign parameters
        self._orbitdata : Dict[str,OrbitData] = scenario_orbitdata
        self._events : List[GeophysicalEvent] = events
        self._results_path : str = env_results_path
        self._scenario_results_path : str = results_path
        self._logger : logging.Logger = logger if logger is not None \
                                            else logging.getLogger("SimulationEnvironment")
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
        self.__interval_connectivities : List[Tuple[Interval, Dict, List]]\
            = SimulationEnvironment.__precompute_connectivity(scenario_orbitdata) # list of (interval, connectivity_matrix, components_list)

        # initialize parameters
        self._connectivity_relays : bool = connectivity_relays
        self._t_0 = None
        self._t_f = None
        self._agent_state_update_times = {}
        self._task_reqs : list[dict] = list()        

        self._observation_history = []
        self._broadcasts_history = []

        self._current_connectivity_interval, self._current_connectivity_matrix, self._current_connectivity_components\
            = self.__get_agent_connectivity(t=0.0) # serve as references for connectivity at current time

    """
    ----------------------
    Simulation Cycle Methods
    ----------------------
    """
    def update_environment(self, t : float) -> None:
        """ Updates the environment state at time `t` """
        # check if connectivity needs to be updated
        if t not in self.current_connectivity_interval:
            # update current connectivity matrix and components
            self.current_connectivity_interval, self.current_connectivity_matrix, \
                self.current_connectivity_components = self.__get_agent_connectivity(t)
            
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

            # for interval,interval_events in events_per_interval.items():
            #     if evt[0] in interval:
            #         interval_events.append(evt)
            #         break            
        
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

            # store interval connectivity data
            interval_connectivities.append( (interval, interval_connectivity_matrix, interval_components) )

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
            interval, connectivity, components \
                = self.__interval_connectivities[mid]

            # return if time t is in the interval
            if t in interval: return interval, connectivity, components
            
            # if not, adjust search bounds
            if t < interval.left:
                high = mid - 1
            else:
                low = mid + 1

        # time t not found in any interval
        raise ValueError(f'Time {t}[s] not found in any precomputed connectivity interval.')