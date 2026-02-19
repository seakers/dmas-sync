from collections import defaultdict, deque
import copy
from enum import Enum
import gc
import json
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.sparse import csr_matrix, triu
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

from orbitpy.mission import Mission
from execsatm.utils import Interval
from dmas.utils.series import AccessTable, IntervalTable, StateTable, TargetGridTable

class ConnectivityLevels(Enum):
    FULL = 'FULL'   # static fully connected network between all agents
    LOS = 'LOS'     # line-of-sight links between all agents
    ISL = 'ISL'     # inter-satellite links only
    GS = 'GS'       # satellite-to-ground station links only
    NONE = 'NONE'   # no inter-agent connectivity
    PREDEF = 'PREDEF'   # pre-defined connectivity specified in mission specifications
        
class OrbitData:
    """
    Stores and queries data regarding an agent's orbital data. 

    TODO: add support to load ground station agents' data
    """
    JDUT1 = 'JDUT1'

    def __init__(self,
                 agent_name : str,
                 time_step : float,
                 epoch_type : str,
                 epoch : float,
                 duration : float,
                 eclipse_data : IntervalTable,
                 state_data : StateTable,
                 comms_links : IntervalTable,
                 gs_access_data : IntervalTable,
                 gp_access_data : AccessTable,
                 grid_data: TargetGridTable
                ):
        # assign attributes
        self.agent_name = agent_name
        self.time_step = time_step
        self.epoch_type = epoch_type
        self.epoch = epoch
        self.duration = duration

        # agent state data
        self.eclipse_data = eclipse_data
        self.state_data = state_data

        # agent connectivity data
        self.comms_links = comms_links

        comms_targets = list(comms_links._meta['columns'].keys())
        comms_target_indices = {target : idx for idx, target in enumerate(comms_targets)}

        self.comms_targets = set(comms_targets)
        self.comms_targets.discard(agent_name)
        self.comms_target_columns = comms_targets
        self.comms_target_indices = comms_target_indices

        # ground station access data
        self.gs_access_data = gs_access_data
        
        # ground point access data
        self.gp_access_data = gp_access_data

        # grid data
        self.grid_data = grid_data   
    
    """
    LOAD FROM PRE-COMPUTED DATA
    """
    @staticmethod
    def from_directory(orbitdata_dir: str, mission_specs : dict, force_preprocess : bool = False, printouts : bool = True) -> Dict[str, 'OrbitData']:
        # TODO check if schemas have already been generated at the provided directory and load from there if so, o
        # therwise preprocess data and generate schemas before loading
        
        # define binary output directory
        bin_dir = os.path.join(orbitdata_dir, 'bin')
        bin_meta_file = os.path.join(bin_dir, "meta.json")

        # get simulation duration from mission specifications
        simulation_duration = mission_specs['duration']

        # check if preprocessed data already exists
        if not os.path.exists(bin_meta_file) or force_preprocess:
            # metadata for processed binaries does not exist or preprocessing is being forced; 
            #   preprocess data and store as binarys for faster loading in the future
            schemas : dict[str, dict] \
                = OrbitData.preprocess(orbitdata_dir, simulation_duration, printouts=printouts)
        
            # force garbage collection after loading data to free up memory
            gc.collect() 
        else:
            # metadata file exists; load schemas from metadata
            with open(bin_meta_file, 'r') as meta_file:
                schemas : dict[str, dict] = json.load(meta_file)
                if printouts: tqdm.write('Existing preprocessed data found. Loading from binaries...')

        # get connectivity level from mission specifications
        connectivity_specs : dict = mission_specs.get('scenario',{}).get('connectivity', None)
        if connectivity_specs is None: raise ValueError('Connectivity specifications not found in mission specifications under `scenario.connectivity`. Please ensure connectivity specifications are included in the mission specifications.')

        # load connectivity data for each agent and store in dictionary indexed by agent name
        agent_comms_links : IntervalTable = OrbitData.__load_comms_links(orbitdata_dir, connectivity_specs, schemas, printouts)

        data = dict()
        for agent_name, schema in schemas.items():
            # skip comms data
            if 'data' in agent_name or 'comms' in agent_name: continue 

            # unpack agent-specific data from schema
            time_step = schema['time_specs']['time step']
            epoch_type = schema['time_specs']['epoch type']
            epoch = schema['time_specs']['epoch']

            assert schema['time_specs']['duration'] > simulation_duration or abs(schema['time_specs']['duration'] - simulation_duration) < 1e-6, \
                f"Preprocessed data duration ({schema['time_specs']['duration']} [days]) is less than the desired simulation duration ({simulation_duration} [days])."
            duration = min(schema['time_specs']['duration'], simulation_duration)

            # load eclipse data from binary
            eclipse_data = IntervalTable.from_schema(schema['eclipse'], mmap_mode='r')
            
            # load ground station access data from binary
            gs_access_data = IntervalTable.from_schema(schema['gs_access'], mmap_mode='r')

            # load comms link data from binary
            comms_links = agent_comms_links

            # load ground point access data from binary
            gp_access_data = AccessTable.from_schema(schema['gp_access'], mmap_mode='r')

            # load grid data from binary
            grid_data = TargetGridTable.from_schema(schema['grid'], mmap_mode='r')

            # load state data from binary 
            state_data = StateTable.from_schema(schema['state'], mmap_mode='r')

            data[agent_name] = OrbitData(agent_name, time_step, epoch_type, epoch, 
                                            duration, eclipse_data, state_data, comms_links, gs_access_data, 
                                            gp_access_data, grid_data)
            
        # return compiled data
        return data     
    
    @staticmethod
    def __load_comms_links(orbitdata_dir : str, connectivity_specs : dict, schemas : dict, printouts : bool = False) -> IntervalTable:
        # define mission specifications file path
        mission_specs_file = os.path.join(orbitdata_dir, 'MissionSpecs.json')
        assert os.path.exists(mission_specs_file), \
            f'Mission specifications file not found at: {mission_specs_file}'
        
        # load mission specifications to get mission duration and other relevant details
        with open(mission_specs_file, 'r') as mission_specs:
            mission_specs_dict : dict = json.load(mission_specs)

        # get list of satellite names 
        satellite_names = {sat['name'] : sat for sat in mission_specs_dict['spacecraft']}

        # get list of ground opertors
        ground_operators = {gs['name'] : gs for gs in mission_specs_dict.get('groundOperator', [])}

        # combine into list of agent names
        agent_names = { **satellite_names, **ground_operators }

        # get relay specs from mission specifications 
        relay_toggle : bool = mission_specs_dict['scenario'].get('enabledRelays', 'True').lower() == 'true'

        if not relay_toggle: raise NotImplementedError('Currently only supports relay-enabled scenarios.')

        # load comms link data 
        ## check if full connectivity is specified
        if connectivity_specs.get('@type', None) == ConnectivityLevels.FULL.value:
            # generate full comms link data
            comm_data = OrbitData.__generate_full_comms_links(agent_names, schemas)
        else:
            # load preprocessed comms link data from binary
            comm_data = IntervalTable.from_schema(schemas['comms_data'], mmap_mode='r')

        # generate connectivity mask based on connectivity level specified in mission specifications
        connectivity_mask = OrbitData.__generate_connectivity_mask(connectivity_specs, agent_names, ground_operators)


        # convert connectivity intervals to events
        connectivity_events = []
        for t_start,t_end,u,v in comm_data:
            # check if access is valid based on connectivity level
            key = tuple(sorted([u,v]))
            if not connectivity_mask.get(key, False):
                # access between agents `u` and `v` is not allowed;
                #   skip this access interval
                continue

            # decompose interval to event start and event end
            event_start = (t_start, u, v, 1)
            event_end = (t_end, u, v, 0)

            # add events to list of connectivity events
            connectivity_events.extend([event_start, event_end])
        
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
        interval_connectivities : List[dict] = []
        agent_to_idx = {agent: idx for idx, agent in enumerate(agent_names.keys())}
        idx_to_agent = [agent for agent in agent_to_idx.keys()]
        prev_connectivity_matrix = np.zeros((len(agent_names), len(agent_names)), dtype=int)
        
        # create adjacency matrix per interval
        for interval in tqdm(connectivity_intervals, 
                             desc='Precomputing agent connectivity intervals', 
                             unit=' intervals', 
                             leave=False,
                             disable=not printouts
                            ):
            # copy previous connectivity state              
            interval_connectivity_matrix = prev_connectivity_matrix.copy()
            
            # get connectivity events that occur during the interval
            interval_events = events_per_interval[interval]

            # update connectivity matrix based on events
            for _,sender,receiver,status in interval_events:
                u_idx = agent_to_idx[sender]
                v_idx = agent_to_idx[receiver]
                interval_connectivity_matrix[u_idx][v_idx] = status
                interval_connectivity_matrix[v_idx][u_idx] = status

            # create component list from connectivity matrix
            interval_components = OrbitData.__get_connected_components(interval_connectivity_matrix, agent_to_idx, idx_to_agent)

            # convert components to dict for easier lookup
            interval_component_map : Dict[str, int] \
                = { sender : -1 for sender in idx_to_agent }

            for component_idx, component in enumerate(interval_components):
                for member in component:
                    interval_component_map[member] = component_idx
    

            # store interval connectivity data
            interval_connectivities.append({
                'start' : interval.left,
                'end' : interval.right,
                **interval_component_map
            })

            # set previous connectivity to current for next iteration
            prev_connectivity_matrix = interval_connectivity_matrix

        # merge connectivity intervals if the connectivity maps are the same and intervals are contiguous
        merged_interval_connectivities : List[dict] = []
        for interval_connectivity in tqdm(interval_connectivities, 
                                          desc='Merging agent connectivity intervals', 
                                          unit=' intervals', 
                                          leave=False,
                                          disable=not printouts
                                        ):
            if not merged_interval_connectivities:
                merged_interval_connectivities.append(interval_connectivity)
                continue

            prev_interval_connectivity = merged_interval_connectivities[-1]

            # check if interval overlaps
            if abs(prev_interval_connectivity['end'] - interval_connectivity['start']) > 1e-6:
                # if not, add new interval connectivity to list
                merged_interval_connectivities.append(interval_connectivity)
                continue

            # check if connectivity maps are the same and intervals are contiguous
            if all(prev_interval_connectivity[agent] == interval_connectivity[agent] for agent in agent_names.keys()) and prev_interval_connectivity['end'] == interval_connectivity['start']:
                # if so, merge intervals by updating end time of previous interval
                prev_interval_connectivity['end'] = interval_connectivity['end']
            else:
                # otherwise, add new interval connectivity to list
                merged_interval_connectivities.append(interval_connectivity)

        # convert interval connectivity data to dataframe and then to IntervalTable for easier querying
        comms_links_df = pd.DataFrame(merged_interval_connectivities)
        # comms_links_df = pd.DataFrame(interval_connectivities)
        bin_dir = os.path.join(orbitdata_dir, 'bin')
        comms_links_schema = OrbitData.__write_interval_data_table(comms_links_df, bin_dir, 'comms_links', schemas['comms_data']['time_specs']['time step'], start_col='start', end_col='end', allow_overwrite=True)

        # load comms link table as IntervalTable for future querying
        return IntervalTable.from_schema(comms_links_schema, mmap_mode='r')
    
    @staticmethod
    def __generate_full_comms_links(agent_names : List[str], schemas : dict) -> List[Tuple]:
        # get simulation duration from schemas
        simulation_duration = schemas['comms_data']['time_specs']['duration']

        # convert duration from days to seconds
        simulation_duration *= 3600*24
        
        # create list of unique agent pairs from list of agent names
        agent_pairs = set()
        for u_name in agent_names:
            for v_name in agent_names:
                # skip self-pairs
                if u_name == v_name: continue
                
                # pair key 
                key = tuple(sorted([u_name, v_name]))
                agent_pairs.add(key)

        # create comms link data with one interval spanning the entire simulation duration for each unique agent pair
        comms_links = [(0.0, simulation_duration, u, v)
                       for u,v in agent_pairs]
        
        # return comms link data
        return comms_links

    @staticmethod
    def __generate_connectivity_mask(connectivity_dict : dict, agent_names : List[str], ground_operators : List[str]) -> Dict[Tuple[str, str], bool]:
        # get type of connectivity specified in mission specifications
        connectivity_level = connectivity_dict.get('@type', None)

        # create list of unique agent pairs from list of agent names
        agent_pairs = set()
        for u_name in agent_names:
            for v_name in agent_names:
                # skip self-pairs
                if u_name == v_name: continue
                
                # pair key 
                key = tuple(sorted([u_name, v_name]))
                agent_pairs.add(key)

        # initialize connectivity mask as dictionary mapping agent pairs to boolean indicating whether connectivity is allowed based on specified connectivity level
        connectivity_mask : Dict[Tuple[str, str], bool] = dict()

        for u,v in agent_pairs:        
            # check if access is valid based on connectivity level
            if connectivity_level == ConnectivityLevels.NONE.value:
                # no connectivity between any agents
                connectivity_mask[(u,v)] = False

            elif connectivity_level == ConnectivityLevels.GS.value:
                # ignore any links that do not involve a ground station
                connectivity_mask[(u,v)] = u in ground_operators or v in ground_operators

            elif connectivity_level == ConnectivityLevels.ISL.value:
                # ignore any links that involve a ground station                
                connectivity_mask[(u,v)] = u not in ground_operators and v not in ground_operators

            elif connectivity_level == ConnectivityLevels.LOS.value:
                # accept all links
                connectivity_mask[(u,v)] = True            
            
            elif connectivity_level == ConnectivityLevels.FULL.value:
                # accept all links
                connectivity_mask[(u,v)] = True

            else:
                raise NotImplementedError("TODO: need to implement generation of connectivity mask based on connectivity level specified in mission specifications.")
            
        # return connectivity mask
        return connectivity_mask
    
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
        comps = list()

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
            comps.append(frozenset(comp))

        # return list of components
        return comps

    """
    GET NEXT methods
    """
    def get_next_agent_access(self, t: float, target : str = None, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to another agent or ground station after or during time `t` up to a given time `t_max`. """

        # get next access intervals
        future_intervals : List[Tuple[Interval, ... ]] = self.comms_links.lookup_intervals(t, t_max, include_current)

        # check if a target is specified
        if target is None: 
            # if no target specified, consider all future access intervals
            target_intervals = future_intervals
        else:
            # otherwise, filter for target of interest
            target_intervals = [ interval for interval,interval_target in future_intervals
                                if interval_target == target]
        
        # get next access interval (if any)
        next_interval = min(target_intervals, key=lambda x: x.left) if target_intervals else None

        # return next access interval
        return next_interval

    def get_next_gs_access(self, t, t_max: float = np.Inf, include_current: bool = False) -> Tuple[Interval, ...]:
        """ returns the next access interval to a ground station after or during time `t`. """
        return self.gs_access_data.lookup_interval(t, t_max, include_current)

    def get_next_eclipse_interval(self, t: float, t_max: float = np.Inf, include_current: bool = False) -> Tuple[Interval, ...]:
        """ returns the next eclipse interval after or during time `t`. """
        return self.eclipse_data.lookup_interval(t, t_max, include_current)

    def get_next_agent_accesses(self, t: float, t_max: float = np.Inf,  target : str = None, include_current: bool = False) -> List[Tuple[Interval, str]]:
        """ returns a list of the next access interval to another agent after or during time `t` up to a given time `t_max`. """

        # get next access intervals
        future_intervals : List[Tuple[Interval, ... ]] = self.comms_links.lookup_intervals(t, t_max, include_current)

        # if no target specified, return all future access intervals
        if target is None:  
            # initialize compiled list of access intervals
            out = []

            # get column index of this agent in the comms links table
            u_column_idx = self.comms_target_indices[self.agent_name]

            # iterate through list of intervals in this time period 
            for interval, *component_indices in future_intervals:
                
                # get component index of this agent during this interval
                u_component_idx = int(component_indices[u_column_idx])
                
                # find all matching agents with the same component index and add to output list
                for v_column_idx,v_component_idx in enumerate(component_indices):
                    if v_column_idx != u_column_idx and v_component_idx == u_component_idx:
                        out.append((interval, self.comms_target_columns[v_column_idx]))           

            # return output list                        
            return out

        # else if a target is specified, filter for target of interest

        # initialize compiled list of access intervals
        out = []

        # get column index of this agent and the target in the comms links table
        u_column_idx = self.comms_target_indices[self.agent_name]
        v_column_idx = self.comms_target_indices[target]

        # iterate through list of intervals in this time period 
        for interval, *component_indices in future_intervals:
            
            # get component index of this agent during this interval
            u_component_idx = int(component_indices[u_column_idx])
            v_component_idx = int(component_indices[v_column_idx])

            if u_component_idx == v_component_idx:
                out.append((interval, target))                   

        # return output list                    
        return out   
    
    def get_next_gp_access_interval(self, lat: float, lon: float, t: float, t_max : float = np.Inf) -> Interval:
        """
        Returns the next access to a ground point
        """
        # TODO
        raise NotImplementedError('TODO: need to implement.')

    """
    STATE QUERY methods
    """

    def is_eclipse(self, t: float):
        """ checks if a satellite is currently in eclise at time `t`. """
        _,_,eclipse = self.get_orbit_state(t)
        return bool(eclipse)

    def get_position(self, t: float):
        pos, _, _ = self.get_orbit_state(t)
        return pos

    def get_velocity(self, t: float):
        _, vel, _ = self.get_orbit_state(t)
        return vel
        
    def get_orbit_state(self, t: float):
        # get eclipse data
        is_eclipse = self.is_eclipse(t)

        # get position data
        position_data = self.state_data.lookup_value(t)
        
        if not position_data:
            raise ValueError(f'No position data found for time {t} [s].')

        # unpack position and velocity data
        pos = [position_data['x [km]'], position_data['y [km]'], position_data['z [km]']]
        vel = [position_data['vx [km/s]'], position_data['vy [km/s]'], position_data['vz [km/s]']]
        
        return (pos, vel, is_eclipse)

    """
    PRECOMPUTATION OF ORBIT DATA
    """
    def precompute(scenario_specs : dict, overwrite : bool = False, printouts: bool = True) -> str:
        """ Pre-calculates coverage and position data for all agents described in the scenario specifications """
        
        # get desired orbit data path
        scenario_dir = scenario_specs['scenario']['scenarioPath']
        settings_dict : dict = scenario_specs.get('settings', None)
        if settings_dict is None:
            data_dir = None
        else:
            data_dir = settings_dict.get('outDir', None)

        if data_dir is None:
            data_dir = os.path.join(scenario_dir, 'orbit_data')
    
        if not os.path.exists(data_dir):
            # if directory does not exists, create it
            os.mkdir(data_dir)
            changes_to_scenario = True
        else:
            changes_to_scenario : bool = OrbitData._check_changes_to_scenario(scenario_specs, data_dir)

        if not changes_to_scenario and not overwrite:
            # if propagation data files already exist, load results
            if printouts: tqdm.write(' - Existing orbit data found and matches scenario. Loading existing data...')
            
        else:
            # if propagation data files do not exist, propagate and then load results
            if printouts:
                if os.path.exists(data_dir):
                    tqdm.write(' - Existing orbit data does not match scenario.')
                else:
                    tqdm.write(' - Orbit data not found.')

            # clear files if they exist
            if printouts: tqdm.write(' - Clearing \'orbitdata\' directory...')    
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    f_dir = os.path.join(data_dir, f)
                    if os.path.isdir(f_dir):
                        shutil.rmtree(f_dir)
                    else:
                        os.remove(f_dir) 
            if printouts:
                tqdm.write(' - \'orbitdata\' cleared!')

            # set grid 
            grid_dicts : list = scenario_specs.get("grid", None)
            grid_dicts = [grid_dicts] if isinstance(grid_dicts, dict) else grid_dicts
            assert isinstance(grid_dicts, list), 'Grid specifications must be provided as a list of dictionaries.'

            for grid_dict in grid_dicts:
                grid_dict : dict
                if grid_dict is not None:
                    grid_type : str = grid_dict.get('@type', None)
                    
                    if grid_type.lower() == 'customgrid':
                        # do nothing
                        pass
                    elif grid_type.lower() == 'uniform':
                        # create uniform grid
                        lat_spacing = grid_dict.get('lat_spacing', 1)
                        lon_spacing = grid_dict.get('lon_spacing', 1)
                        grid_index  = grid_dicts.index(grid_dict)
                        grid_path : str = OrbitData._create_uniform_grid(scenario_dir, grid_index, lat_spacing, lon_spacing)

                        # set to customgrid
                        grid_dict['@type'] = 'customgrid'
                        grid_dict['covGridFilePath'] = grid_path
                        
                    elif grid_type.lower() in ['cluster', 'clustered']:
                        # create clustered grid
                        n_clusters          = grid_dict.get('n_clusters', 100)
                        n_cluster_points    = grid_dict.get('n_cluster_points', 1)
                        variance            = grid_dict.get('variance', 1)
                        grid_index          = grid_dicts.index(grid_dict)
                        grid_path : str = OrbitData._create_clustered_grid(scenario_dir, grid_index, n_clusters, n_cluster_points, variance)

                        # set to customgrid
                        grid_dict['@type'] = 'customgrid'
                        grid_dict['covGridFilePath'] = grid_path
                        
                    else:
                        raise ValueError(f'Grids of type \'{grid_type}\' not supported.')
                else:
                    pass
            scenario_specs['grid'] = grid_dicts

            # set output directory to orbit data directory
            if scenario_specs.get("settings", None) is not None:
                if scenario_specs["settings"].get("outDir", None) is None:
                    scenario_specs["settings"]["outDir"] = scenario_dir + '/orbit_data/'
            else:
                scenario_specs["settings"] = {}
                scenario_specs["settings"]["outDir"] = scenario_dir + '/orbit_data/'

            # propagate data and save to orbit data directory
            if printouts: tqdm.write("Propagating orbits...")
            mission : Mission = Mission.from_json(scenario_specs, printouts=printouts)  
            mission.execute(printouts=printouts)                
            if printouts: tqdm.write("Propagation done!")

            # save specifications of propagation in the orbit data directory
            with open(os.path.join(data_dir,'MissionSpecs.json'), 'w') as mission_specs:
                mission_specs.write(json.dumps(scenario_specs, indent=4))
                assert os.path.exists(os.path.join(data_dir,'MissionSpecs.json')), \
                    'Mission specifications not saved correctly!'

        # # update mission duration if needed
        # orbitdata_filename = os.path.join(data_dir, 'MissionSpecs.json')
        # ## check if mission specs file already exists
        # original_duration = scenario_specs['duration']
        # if os.path.exists(orbitdata_filename):
        #     with open(orbitdata_filename, 'r') as orbitdata_specs:
        #         # load existing mission specs
        #         orbitdata_dict : dict = json.load(orbitdata_specs)

        #         # update duration to that of the longest mission
        #         scenario_specs['duration'] = max(scenario_specs['duration'], orbitdata_dict['duration'])

        # # save specifications of propagation in the orbit data directory
        # with open(os.path.join(data_dir,'MissionSpecs.json'), 'w') as mission_specs:
        #     mission_specs.write(json.dumps(scenario_specs, indent=4))
        #     assert os.path.exists(os.path.join(data_dir,'MissionSpecs.json')), \
        #         'Mission specifications not saved correctly!'
            
        # check if data in the directory has already been preprocessed 
        has_been_preprocessed = os.path.exists(os.path.join(data_dir, 'bin', 'meta.json'))

        # check if data has not been preprocessed or if there have been changes to the scenario specifications since the last propagation or if overwrite is enabled
        if not has_been_preprocessed or changes_to_scenario or overwrite:
            # preprocess data and store as binarys for faster loading in the future
            if printouts: tqdm.write("Preprocessing orbit data...")
            OrbitData.preprocess(data_dir, scenario_specs['duration'], overwrite=True, printouts=printouts)
            if printouts: tqdm.write("Preprocessing done!")

        # remove raw data and only keep binaries to save space (if enabled)
        if settings_dict is not None:
            save_unprocessed_coverage = settings_dict.get('saveUnprocessedCoverage', "True").lower() == "true"
            if not save_unprocessed_coverage:
                # remove raw coverage data to save space but maintain mission specifications 
                for f in os.listdir(data_dir):
                    f_dir = os.path.join(data_dir, f)
                    if os.path.isdir(f_dir) and f != 'bin':
                        for h in os.listdir(f_dir):
                            os.remove(os.path.join(f_dir, h))
                        os.rmdir(f_dir)

        # # restore original duration value
        # scenario_specs['duration'] = original_duration

        # return orbit data directory
        return data_dir
    
    def _check_changes_to_scenario(scenario_dict : dict, orbitdata_dir : str) -> bool:
        """ 
        Checks if the scenario has already been pre-computed 
        or if relevant changes have been made 
        """
        # check if directory exists
        filename = 'MissionSpecs.json'
        orbitdata_filename = os.path.join(orbitdata_dir, filename)
        if not os.path.exists(orbitdata_filename):
            return True
        
        # copy scenario specs
        scenario_specs : dict = copy.deepcopy(scenario_dict)
            
        # compare specifications
        with open(orbitdata_filename, 'r') as orbitdata_specs:
            orbitdata_dict : dict = json.load(orbitdata_specs)

            scenario_specs.pop('settings')
            orbitdata_dict.pop('settings')
            scenario_specs.pop('scenario')
            orbitdata_dict.pop('scenario')

            if (
                    scenario_specs['epoch'] != orbitdata_dict['epoch']
                or scenario_specs['duration'] > orbitdata_dict['duration']
                or scenario_specs.get('groundStation', None) != orbitdata_dict.get('groundStation', None)
                # or scenario_dict['grid'] != orbitdata_dict['grid']
                # or scenario_specs['scenario']['connectivity'] != orbitdata_dict['scenario']['connectivity']
                ):
                return True
            
            if scenario_specs['grid'] != orbitdata_dict['grid']:
                if len(scenario_specs['grid']) != len(orbitdata_dict['grid']):
                    return True
                
                for i in range(len(scenario_specs['grid'])):
                    scenario_grid : dict = scenario_specs['grid'][i]
                    mission_grid : dict = orbitdata_dict['grid'][i]

                    scenario_gridtype = scenario_grid['@type'].lower()
                    mission_gridtype = mission_grid['@type'].lower()

                    if scenario_gridtype != mission_gridtype == 'customgrid':
                        if scenario_gridtype not in mission_grid['covGridFilePath']:
                            return True

            if scenario_specs['spacecraft'] != orbitdata_dict['spacecraft']:
                if len(scenario_specs['spacecraft']) != len(orbitdata_dict['spacecraft']):
                    return True
                
                for i in range(len(scenario_specs['spacecraft'])):
                    scenario_sat : dict = scenario_specs['spacecraft'][i]
                    mission_sat : dict = orbitdata_dict['spacecraft'][i]
                    
                    if "planner" in scenario_sat:
                        scenario_sat.pop("planner")
                    if "science" in scenario_sat:
                        scenario_sat.pop("science")
                    if "notifier" in scenario_sat:
                        scenario_sat.pop("notifier") 
                    if "missionProfile" in scenario_sat:
                        scenario_sat.pop("missionProfile")
                    if "mission" in scenario_sat:
                        scenario_sat.pop("mission")
                    if "spacecraftBus" in scenario_sat and "components" in scenario_sat["spacecraftBus"]:
                        scenario_sat["spacecraftBus"].pop("components")

                    if "planner" in mission_sat:
                        mission_sat.pop("planner")
                    if "science" in mission_sat:
                        mission_sat.pop("science")
                    if "notifier" in mission_sat:
                        mission_sat.pop("notifier") 
                    if "missionProfile" in mission_sat:
                        mission_sat.pop("missionProfile")
                    if "mission" in mission_sat:
                        mission_sat.pop("mission")
                    if "spacecraftBus" in mission_sat and "components" in mission_sat["spacecraftBus"]:
                        mission_sat["spacecraftBus"].pop("components")

                    if scenario_sat != mission_sat:
                        return True
                        
        return False

    def _create_uniform_grid(scenario_dir : str, grid_index : int, lat_spacing : float, lon_spacing : float) -> str:
        # create uniform grid
        groundpoints = [(lat, lon) 
                        for lat in np.linspace(-90, 90, int(180/lat_spacing)+1)
                        for lon in np.linspace(-180, 180, int(360/lon_spacing)+1)
                        if lon < 180
                        ]
                
        # create datagrame
        df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

        # save to csv
        grid_path : str = os.path.join(scenario_dir, 'resources', f'uniform_grid{grid_index}.csv')
        df.to_csv(grid_path,index=False)

        # return address
        return grid_path

    def _create_clustered_grid(scenario_dir : str, grid_index : int, n_clusters : int, n_cluster_points : int, variance : float) -> str:
        # create clustered grid of gound points
        std = np.sqrt(variance)
        groundpoints = []
        
        for _ in range(n_clusters):
            # find cluster center
            lat_cluster = (90 - -90) * random.random() -90
            lon_cluster = (180 - -180) * random.random() -180
            
            for _ in range(n_cluster_points):
                # sample groundpoint
                lat = random.normalvariate(lat_cluster, std)
                lon = random.normalvariate(lon_cluster, std)
                groundpoints.append((lat,lon))

        # create datagrame
        df = pd.DataFrame(data=groundpoints, columns=['lat [deg]','lon [deg]'])

        # save to csv
        grid_path : str = os.path.join(scenario_dir, 'resources', f'clustered_grid{grid_index}.csv')
        df.to_csv(grid_path,index=False)

        # return address
        return grid_path
    
    """
    DATA PREPROCESSING METHODS
    """
    @staticmethod
    def preprocess(orbitdata_dir: str, simulation_duration : float, overwrite : bool = True, printouts : bool = True) -> dict:
        """
        Loads orbit data from a directory containig a json file specifying the details of the mission being simulated.
        Data must have already been pre-computed and stored in the directory using `orbitpy`'s `csv` output option. 
        The method processes the raw `csv` data, converts it to binary format, and stores it in a structured directory 
        format for faster loading in the future.

        The resulting processed data gets stored as a dictionary, with each entry containing the orbit data of each agent 
        in the mission indexed by the name of the agent.
        """

        # define binary output directory
        bin_dir = os.path.join(orbitdata_dir, 'bin')

        # check if data has already been processed
        if os.path.exists(bin_dir) and overwrite:
            # path exists and overwrite was enabled; remove existing data before re-processing
            for root, dirs, files in os.walk(bin_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        else:
            # create binary output directory if it does not exist
            os.makedirs(bin_dir, exist_ok=True)

        # load raw data as data frames
        time_specs, agents_loaded, eclipse_dfs, state_dfs, \
            gs_access_dfs, comms_link_dfs, gp_access_dfs, \
                grid_data_dfs = OrbitData.__load_csv_data(orbitdata_dir, simulation_duration, printouts)
            
        # create instances of OrbitData for each agent and store in dictionary indexed by agent name
        schemas = dict()
        for *__,agent_name in agents_loaded: 
            # extract relevant data for this agent
            agent_eclipse_data = eclipse_dfs[agent_name]
            agent_state_data = state_dfs[agent_name]
            agent_gs_access_data = gs_access_dfs[agent_name]
            agent_gp_access_data = gp_access_dfs[agent_name]
            
            # define agent-specific binary output directory
            agent_bin_dir = os.path.join(bin_dir, agent_name)
            
            # print interval data to binaries 
            eclipse_meta = OrbitData.__write_interval_data_table(agent_eclipse_data, agent_bin_dir, 'eclipse', time_specs['time step'], allow_overwrite=True)
            gs_meta = OrbitData.__write_interval_data_table(agent_gs_access_data, agent_bin_dir, 'gs_access', time_specs['time step'], allow_overwrite=True)

            # print agent position data to binaries
            state_meta = OrbitData.__write_state_table(agent_state_data, agent_bin_dir, 'cartesian_state', time_specs['time step'], allow_overwrite=True)
            
            # print agent's groundpoint coverage data to binaries
            gp_access_meta = OrbitData.__write_access_table(agent_gp_access_data, agent_bin_dir, 'gp_access', time_step=time_specs['time step'], n_steps=time_specs['n_steps'], allow_overwrite=True)

            # print grid data to binaries
            grid_meta = OrbitData.__write_grid_table(grid_data_dfs, agent_bin_dir, 'grid', allow_overwrite=True)

            # compile schema for this agent and store in dictionary
            schemas[agent_name] = {
                'time_specs' : time_specs,
                'dir' : agent_bin_dir,
                'eclipse': eclipse_meta,
                'gs_access': gs_meta,
                'state': state_meta,
                'gp_access': gp_access_meta,
                'grid': grid_meta
            }

        # compile comms link data for all agents into single dataframe
        comms_data_concat :pd.DataFrame = OrbitData.__compile_agent_comms_data(comms_link_dfs)
        
        # save agent comms link data into single dataframe and print to binary
        comms_meta = OrbitData.__write_interval_data_table(comms_data_concat, bin_dir, 'comms_data', time_specs['time step'], allow_overwrite=True)
        
        # add comms data schema to compiled schemas
        schemas['comms_data'] = comms_meta
        schemas['comms_data']['time_specs'] = time_specs
        
        # save schemas for all agents to metadata file in binary directory for future loading
        with open(os.path.join(bin_dir, "meta.json"), "w") as f:
            json.dump(schemas, f, indent=4)

        # return compiled schemas for all agents
        return schemas
    
    @staticmethod
    def __compile_agent_comms_data(agent_comms_link_data : Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ Collects comms link data for all agents and compiles it into a single dataframe with columns for sender, receiver, start time, and end time. """
        # define required columns for comms data
        comms_req_columns = ["start index", "end index"]

        # compile comms data for all agents
        working_dfs = []
        for (u,v), comms_links in agent_comms_link_data.items():
            # take only the required cols (this avoids mutating the original DF)
            work = comms_links.loc[:, comms_req_columns].copy()

            # force numeric (coerce bad values to NaN)
            for c in comms_req_columns:
                work[c] = pd.to_numeric(work[c], errors="coerce")

            # check for NaNs
            if work[comms_req_columns].isna().any().any():
                # if values must be integers, enforce and fail fast on NaNs
                bad = work[work[comms_req_columns].isna().any(axis=1)].head(5)
                raise ValueError(
                    f"Non-numeric or missing start/end values after coercion for {u}-->{v}. "
                    f"Examples:\n{bad}"
                )

            # cast start and end indices to `int`
            work["start index"] = work["start index"].astype(np.int32)
            work["end index"]   = work["end index"].astype(np.int32)

            # add columns for sender and receiver
            work["u"] = u
            work["v"] = v
            working_dfs.append(work)
        
        # concadenate comms dataframes
        if working_dfs:
            comms_links_concat = (
                pd.concat(working_dfs, ignore_index=True, copy=False)
                .sort_values(by=["start index", "end index"], kind="mergesort")
                .reset_index(drop=True)
            )
        else:
            # if no accesses were found, create empty dataframe with the correct columns
            comms_links_concat = pd.DataFrame(columns=[*comms_req_columns, "u", "v"])

        # return concatenated comms dataframe
        return comms_links_concat
        
    def __load_csv_data(orbitdata_dir: str, simulation_duration : float, printouts : bool = True) -> Tuple:
        """ 
        Loads precomputed `csv` outputs from `orbitpy` and collects it to a tuple of dictionaries 
         maping agents to a `pd.DataFrame` containing that agent's data for each type of data 
          (eclipse, position, ground station access, comms links, ground point access, grid data).
        """
        # ensure that the provided directory exists
        if not os.path.exists(orbitdata_dir):
            raise FileNotFoundError(f'Orbit data directory `{orbitdata_dir}` does not exist.')

        # define progress bar settings
        tqdm_config = {
            'leave': False,
            'disable': not printouts
        }

        # define path to mission specifications json file
        orbitdata_specs : str = os.path.join(orbitdata_dir, 'MissionSpecs.json')

        # open and load mission specifications file
        with open(orbitdata_specs, 'r') as scenario_specs:            
            mission_dict : dict = json.load(scenario_specs)
            
        # extract agent information from mission specifications
        spacecraft_list : List[dict] = mission_dict.get('spacecraft', None)
        ground_ops_list : List[dict] = mission_dict.get('groundOperator', [])

        # compile list of ground stations in the scenario (if any)
        ground_station_list : List[dict] = mission_dict.get('groundStation', [])

        # compile list of all agents to load
        agents_loaded : List[dict] = []
        for i,spacecraft_dict in enumerate(spacecraft_list):
            spacecraft_name : str = spacecraft_dict['name']
            agents_loaded.append(('spacecraft', i, spacecraft_name))
        for i,ground_op_dict in enumerate(ground_ops_list):
            ground_op_name : str = ground_op_dict['name']
            agents_loaded.append(('groundOperator', i, ground_op_name))

        # load time specifications
        position_file = os.path.join(orbitdata_dir, f"sat{agents_loaded[0][1]}", "state_cartesian.csv")
        time_specs =  pd.read_csv(position_file, nrows=3)
        _, epoch_type, _, epoch = time_specs.at[0,time_specs.axes[1][0]].split(' ')
        epoch_type = epoch_type[1 : -1]
        epoch = float(epoch)
        _, _, _, _, time_step = time_specs.at[1,time_specs.axes[1][0]].split(' ')
        time_step = float(time_step)
        _, _, _, _, prop_duration = time_specs.at[2,time_specs.axes[1][0]].split(' ')
        prop_duration = float(prop_duration)
        n_steps = int(simulation_duration * 24 * 3600 // time_step) + 1

        assert simulation_duration <= prop_duration, \
            f'Simulation duration ({simulation_duration} days) exceeds pre-computed propagation duration ({prop_duration} days).'

        # compile time specifications into dictionary
        time_specs = {"epoch": epoch, 
                      "epoch type": epoch_type, 
                      "time step": time_step,
                      "duration" : simulation_duration,
                      "n_steps": n_steps 
                    }
        
        # load agent eclipse data 
        eclipse_dfs : Dict[str, pd.DataFrame] = OrbitData.__load_agent_eclipse_data(orbitdata_dir, agents_loaded, simulation_duration, time_step)

        # load agent position/vel data
        state_dfs : Dict[str, pd.DataFrame] = OrbitData.__load_agent_state_data(orbitdata_dir, agents_loaded, simulation_duration, time_step)

        # load ground station access data
        gs_access_dfs : Dict[str, pd.DataFrame] = OrbitData.__load_agent_gs_access_data(orbitdata_dir, agents_loaded, simulation_duration, time_step, ground_station_list, tqdm_config)

        # load comms link data
        comms_link_dfs : Dict[tuple, pd.DataFrame] = OrbitData.__load_agent_comms_link_data(orbitdata_dir, agents_loaded, simulation_duration, gs_access_dfs, time_step)

        # load ground point access data
        gp_access_dfs : Dict[str, pd.DataFrame] = OrbitData.__load_agent_gp_access_data(orbitdata_dir, agents_loaded, mission_dict, simulation_duration, time_step, spacecraft_list, tqdm_config)

        # load grid data
        grid_data_dfs : List[pd.DataFrame] = OrbitData.__load_grid_data(orbitdata_dir, mission_dict)

        # return compiled data
        return time_specs, agents_loaded, eclipse_dfs, state_dfs, gs_access_dfs, comms_link_dfs, gp_access_dfs, grid_data_dfs

    @staticmethod
    def __load_agent_eclipse_data(orbitdata_path : str, 
                                  agents_to_load : List[tuple],  
                                  simulation_duration : float,
                                  time_step : float
                                ) -> Dict[str, pd.DataFrame]:
        # initialize eclipse data 
        data = dict()
        
        # iterate through agents to load
        for agent_type, spacecraft_idx, agent_name in agents_to_load:
            # load data by agent type
            if agent_type == 'spacecraft':
                # define agent folder
                sat_id = "sat" + str(spacecraft_idx)
                agent_folder = sat_id + '/' 
            
                # load eclipse data
                eclipse_file = os.path.join(orbitdata_path, agent_folder, "eclipses.csv")
                eclipse_data = pd.read_csv(eclipse_file, skiprows=range(3))

                # reduce data to only include intervals within the simulation duration
                max_time_index = int(simulation_duration * 24 * 3600 // time_step)
                eclipse_data = eclipse_data[eclipse_data['start index'] <= max_time_index]
                eclipse_data.loc[eclipse_data['end index'] > max_time_index, 'end index'] = max_time_index
                
            elif agent_type == 'groundOperator':
                # no eclipse data for ground operators; create empty dataframe
                eclipse_data = pd.DataFrame(columns=['start index', 'end index'])

            else:
                raise ValueError(f'Unknown agent type `{agent_type}` for agent `{agent_name}`.')
            
            # store in dictionary
            data[agent_name] = eclipse_data
        
        # return compiled eclipse data
        return data
    
    @staticmethod
    def __load_agent_state_data(orbitdata_path : str, 
                                agents_to_load : List[tuple],  
                                simulation_duration : float,
                                time_step : float
                            ) -> Dict[str, pd.DataFrame]:
        # initialize state data 
        data = dict()
        
        # iterate through agents to load
        for agent_type, spacecraft_idx, agent_name in agents_to_load:
            # load data by agent type
            if agent_type == 'spacecraft':
                # define agent folder
                sat_id = "sat" + str(spacecraft_idx)
                agent_folder = sat_id + '/' 
            
                ## load agent position and velocity state data
                state_file = os.path.join(orbitdata_path, agent_folder, "state_cartesian.csv")
                state_data = pd.read_csv(state_file, skiprows=range(4))

                # reduce data to only include intervals within the simulation duration
                max_time_index = int(simulation_duration * 24 * 3600 // time_step)
                state_data = state_data[state_data['time index'] <= max_time_index]
                
            elif agent_type == 'groundOperator':
                # no state data for ground operators; create empty dataframe
                state_data = pd.DataFrame(columns=['time index','x [km]','y [km]','z [km]','vx [km/s]','vy [km/s]','vz [km/s]'])

            else:
                raise ValueError(f'Unknown agent type `{agent_type}` for agent `{agent_name}`.')
            
            # store in dictionary
            data[agent_name] = state_data
        
        # return compiled state data
        return data
                
    @staticmethod
    def __load_agent_gs_access_data(orbitdata_path : str, 
                                   agents_to_load : List[tuple],  
                                   simulation_duration : float,
                                   time_step : float,
                                   ground_station_list : List[dict],
                                   tqdm_config : dict,
                                ) -> Dict[str, pd.DataFrame]:
        # initialize ground station access data
        data = dict()

        for agent_type, spacecraft_idx, agent_name in agents_to_load:
            if agent_type == 'spacecraft':
                # define agent folder
                sat_id = "sat" + str(spacecraft_idx)
                agent_folder = sat_id + '/' 

                # compile list of ground stations that are part of the desired network                
                gs_network_station_ids : List[str] = [ gs['@id'] for gs in ground_station_list
                                                    # if gs.get('networkName', None) == gs_network_name 
                                                    ]

                # create empty dataframe to store ground station access data for this agent
                gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name', 'gndStn network', 'lat [deg]','lon [deg]'])
                
                # define path to agent's orbit data
                agent_orbitdata_path = os.path.join(orbitdata_path, agent_folder)

                # load ground station access data
                for file in tqdm(os.listdir(agent_orbitdata_path), desc=f'Loading ground station access data for {agent_name}', unit=' file', **tqdm_config):
                    # check if file is a ground station access file
                    if 'gndStn' not in file: continue

                    # get ground station index from file name
                    gndStn, _ = file.split('_')
                    gndStn_index = int(re.sub("[^0-9]", "", gndStn))
                    
                    # check if ground station is part of the desired network
                    gndStn_id = ground_station_list[gndStn_index].get('@id')
                    gs_network_name = ground_station_list[gndStn_index].get('networkName')
                    if gs_network_station_ids and gndStn_id not in gs_network_station_ids:
                        continue

                    # load ground station access data
                    gndStn_access_file = os.path.join(orbitdata_path, agent_folder, file)
                    
                    gndStn_access_data : pd.DataFrame \
                        = OrbitData.__load_gs_access_data(gndStn_access_file, simulation_duration, time_step)

                    nrows, _ = gndStn_access_data.shape

                    # get ground station information
                    gndStn_name = ground_station_list[gndStn_index].get('name')
                    gndStn_lat = ground_station_list[gndStn_index].get('latitude')
                    gndStn_lon = ground_station_list[gndStn_index].get('longitude')

                    # add ground station information to access data
                    gndStn_name_column = [gndStn_name] * nrows
                    gndStn_id_column = [gndStn_id] * nrows
                    gndStn_lat_column = [gndStn_lat] * nrows
                    gndStn_lon_column = [gndStn_lon] * nrows

                    gndStn_access_data['gndStn name'] = gndStn_name_column
                    gndStn_access_data['gndStn id'] = gndStn_id_column
                    gndStn_access_data['gndStn network'] = [gs_network_name] * nrows
                    gndStn_access_data['lat [deg]'] = gndStn_lat_column
                    gndStn_access_data['lon [deg]'] = gndStn_lon_column

                    # append to overall ground station access data
                    if len(gs_access_data) == 0:
                        gs_access_data = gndStn_access_data
                    else:
                        gs_access_data = pd.concat([gs_access_data, gndStn_access_data])

                # sort gs access data by start index and remove duplicates
                gs_access_data = gs_access_data.sort_values(by=['start index']).drop_duplicates().reset_index(drop=True)

            elif agent_type == 'groundOperator':
                # ground operator agent has constant contact with all ground stations in their network
                
                # calculate number of propagation steps
                n_steps = int(simulation_duration  * 24 * 3600 // time_step) + 1

                # create access intervals spanning the entire simulation duration for each ground station in the network
                columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]']
                access_data = [(0, n_steps, ground_station.get('@id'), ground_station.get('name'), ground_station.get('latitude'), ground_station.get('longitude'))
                        for ground_station in ground_station_list
                    if ground_station.get('networkName', None) == agent_name]
                
                # create dataframe from access data
                gs_access_data = pd.DataFrame(data=access_data, columns=columns)
                
            else:
                raise ValueError(f'Unknown agent type `{agent_type}` for agent `{agent_name}`.')

            # store in dictionary
            data[agent_name] = gs_access_data
            
        # return compiled data
        return data
    
    @staticmethod
    def __load_gs_access_data(gs_access_file : str, simulation_duration : float, time_step : float) -> pd.DataFrame:
        # if connectivity.upper() == ConnectivityLevels.FULL.value:
        #     # fully connected network; modify connectivity 
        #     columns = ['start index', 'end index']
            
        #     # generate mission-long connectivity access                    
        #     data = [[0.0, simulation_duration * 24 * 3600 // time_step + 1]]  # full connectivity from start to end of mission
        #     assert data[0][1] > 0.0

        #     # return modified connectivity
        #     gndStn_access_data = pd.DataFrame(data=data, columns=columns)
        
        # elif connectivity.upper() == ConnectivityLevels.LOS.value:
        #     # line-of-sight driven connectivity; load ground station access data
        #     gndStn_access_data = pd.read_csv(gs_access_file, skiprows=range(3))

        #     # limit ground station access data to simulation duration
        #     max_time_index = int(simulation_duration * 24 * 3600 // time_step)
        #     gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
        #     gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index
        
        # elif connectivity.upper() == ConnectivityLevels.GS.value:
        #     # ground station-only connectivity; load ground station access data
        #     gndStn_access_data = pd.read_csv(gs_access_file, skiprows=range(3))

        #     # limit ground station access data to simulation duration
        #     max_time_index = int(simulation_duration * 24 * 3600 // time_step)
        #     gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
        #     gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index

        # elif connectivity.upper() == ConnectivityLevels.ISL.value:
        #     # inter-satellite link-driven connectivity; create empty dataframe
        #     columns = ['start index', 'end index']
        #     gndStn_access_data = pd.DataFrame(data=[], columns=columns)

        # elif connectivity.upper() == ConnectivityLevels.NONE.value:
        #     # no inter-agent connectivity; create empty dataframe
        #     columns = ['start index', 'end index']
        #     gndStn_access_data = pd.DataFrame(data=[], columns=columns)

        # else:
        #     # fallback; unsupported connectivity level
        #     raise ValueError(f'Unsupported connectivity level: {connectivity}.')
        
        # load ground station access data
        gndStn_access_data = pd.read_csv(gs_access_file, skiprows=range(3))

        # limit ground station access data to simulation duration
        max_time_index = int(simulation_duration * 24 * 3600 // time_step)
        gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
        gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index

        # if no access intervals remain after limiting to simulation duration, create empty dataframe with correct columns
        if gndStn_access_data.empty:
            # no ISL access during simulation duration; create empty dataframe
            columns = ['start index', 'end index']
            gndStn_access_data = pd.DataFrame(data=[], columns=columns)

        # return ground station access data
        return gndStn_access_data

    @staticmethod
    def __load_agent_comms_link_data(orbitdata_path : str, 
                                     agents_to_load : List[tuple],  
                                     simulation_duration : float,
                                     gs_access_dfs : Dict[str, pd.DataFrame],
                                     time_step : float,
                                ) -> Dict[tuple, pd.DataFrame]:
        # initialize comms link data
        data = defaultdict(dict)

        # define path to comms data
        comms_path = os.path.join(orbitdata_path, 'comm')

        # iterate through agents to load
        for u_type,u_idx,u_name in agents_to_load:
            for v_type,v_idx,v_name in agents_to_load:
                # skip self-links
                if u_name == v_name: continue
                
                # pair key 
                key = tuple(sorted([u_name, v_name]))
                
                # skip if data for this pair of agents has already been loaded
                key_data = data.get(key, None)
                if key_data is not None: continue 

                # load data by agent types
                if u_type == 'spacecraft' and v_type == 'spacecraft':
                    # generate ISL access file path
                    filename = f"sat{u_idx}_to_sat{v_idx}.csv" 
                    isl_file = os.path.join(comms_path, filename)

                    # check if file exists; if not, try reversed order
                    if not os.path.exists(isl_file):
                        filename = f"sat{v_idx}_to_sat{u_idx}.csv"
                        isl_file = os.path.join(comms_path, filename)

                    # validate that ISL access file exists
                    if not os.path.exists(isl_file):
                        raise FileNotFoundError(f'ISL access file not found for satellite pair ({u_name}, {v_name}). Expected file name: `{filename}`.')

                    # load ISL access data
                    comms_data = OrbitData.__load_isl_data(isl_file, simulation_duration, time_step)

                elif u_type == 'groundOperator' and v_type == 'spacecraft':
                    # get all ground station accesses for this spacecraft
                    comms_data = gs_access_dfs[v_name]                    

                    # filter ground station accesses for the ground operator's network's stations
                    comms_data = comms_data[(comms_data['gndStn network'] == u_name)]

                elif u_type == 'spacecraft' and v_type == 'groundOperator':     
                    # get all ground station accesses for this spacecraft
                    comms_data = gs_access_dfs[u_name]                    
                    
                    # filter ground station accesses for the ground operator's network's stations
                    comms_data = comms_data[(comms_data['gndStn network'] == v_name)]
                    
                elif u_type == 'groundOperator' and v_type == 'groundOperator':
                    # all ground operators can communicate with each other for the entire duration of the simulation
                    columns = ['start index', 'end index']
                    
                    # generate mission-long connectivity access
                    duration = timedelta(days=float(simulation_duration))
                    data = [[0.0, duration.total_seconds() // time_step + 1]]
                    assert data[0][1] > 0.0

                    # return modified connectivity
                    comms_data = pd.DataFrame(data=data, columns=columns)

                else:
                    raise ValueError(f'Unknown agent type `{u_type}` for agent `{u_name}`.')
                
                # store comms data in dictionary
                data[key] = comms_data

        # return compiled comms data
        return data
    
    @staticmethod
    def __load_isl_data(isl_file : str, duration_days : float, time_step : float) -> pd.DataFrame:        
        """ Loads ISL access data from a file and modifies it according to the specified connectivity level. """
        # if connectivity.upper() == ConnectivityLevels.FULL.value:
        #     # fully connected network; modify connectivity 
        #     columns = ['start index', 'end index']
            
        #     # generate mission-long connectivity access
        #     duration = timedelta(days=float(duration_days))
        #     data = [[0.0, duration.total_seconds() // time_step + 1]]
        #     assert data[0][1] > 0.0

        #     # return modified connectivity
        #     isl_data = pd.DataFrame(data=data, columns=columns)

        # elif connectivity.upper() == ConnectivityLevels.LOS.value:
        #     # line-of-sight driven connectivity; load connectivity and store data
        #     # TODO if ISL definition is modified in orbitpy, make sure this case is updated accordingly
        #     isl_data = pd.read_csv(isl_file, skiprows=range(3))

        #     # limit ISL data to simulation duration
        #     max_time_index = int(duration_days * 24 * 3600 // time_step)
        #     isl_data = isl_data[isl_data['start index'] <= max_time_index]
        #     isl_data.loc[isl_data['end index'] > max_time_index, 'end index'] = max_time_index

        # elif connectivity.upper() == ConnectivityLevels.ISL.value:
        #     # inter-satellite link driven connectivity; load connectivity and store data
        #     # TODO if ISL definition is modified in orbitpy, make sure this case is updated accordingly
        #     isl_data = pd.read_csv(isl_file, skiprows=range(3))

        #     # limit ISL data to simulation duration
        #     max_time_index = int(duration_days * 24 * 3600 // time_step)
        #     isl_data = isl_data[isl_data['start index'] <= max_time_index]
        #     isl_data.loc[isl_data['end index'] > max_time_index, 'end index'] = max_time_index

        # elif connectivity.upper() == ConnectivityLevels.GS.value:
        #     # ground station-only connectivity; create empty dataframe
        #     columns = ['start index', 'end index']
        #     isl_data = pd.DataFrame(data=[], columns=columns)

        # elif connectivity.upper() == ConnectivityLevels.NONE.value:
        #     # no inter-agent connectivity; create empty dataframe
        #     columns = ['start index', 'end index']
        #     isl_data = pd.DataFrame(data=[], columns=columns)
        # else:
        #     # fallback case for unsupported connectivity levels
        #     raise ValueError(f'Unsupported connectivity level: {connectivity}')
                
        # # check if ISL data is empty
        # if isl_data.empty:
        #     # no ISL access during simulation duration; create empty dataframe
        #     columns = ['start index', 'end index']
        #     isl_data = pd.DataFrame(data=[], columns=columns)

        # load ISL access data
        isl_data = pd.read_csv(isl_file, skiprows=range(3))

        # limit ISL data to simulation duration
        max_time_index = int(duration_days * 24 * 3600 // time_step)
        isl_data = isl_data[isl_data['start index'] <= max_time_index]
        isl_data.loc[isl_data['end index'] > max_time_index, 'end index'] = max_time_index

        # check if ISL data is empty
        if isl_data.empty:
            # no ISL access during simulation duration; create empty dataframe
            columns = ['start index', 'end index']
            isl_data = pd.DataFrame(data=[], columns=columns)

        # return ISL data
        return isl_data
    
    @staticmethod
    def __load_agent_gp_access_data(orbitdata_path : str, 
                                   agents_to_load : List[tuple],  
                                   mission_dict : dict,
                                   simulation_duration : float,
                                   time_step : float,
                                   spacecraft_list : List[dict],
                                   tqdm_config : dict,
                                ) -> Dict[str, pd.DataFrame]:
        
        # initialize ground point access data
        data = dict()

        # iterate through agents to load
        for agent_type, spacecraft_idx, agent_name in agents_to_load:
            if agent_type == 'spacecraft':
                # define agent folder
                sat_id = "sat" + str(spacecraft_idx)
                agent_folder = sat_id + '/' 

                # calculate maximum time index for simulation duration
                max_time_index = int(simulation_duration * 24 * 3600 // time_step)

                # land coverage data metrics data
                payload = spacecraft_list[spacecraft_idx].get('instrument', None)
                if not isinstance(payload, list):
                    payload = [payload]

                # iintialize dataframe to store ground point access data for this agent
                gp_access_data = pd.DataFrame(columns=['time index','grid index', 'GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                                'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

                for instrument in tqdm(payload, desc=f'Loading land coverage data for {agent_name}', unit=' instrument', **tqdm_config):
                    if instrument is None: continue 

                    i_ins = payload.index(instrument)
                    gp_acces_by_mode = []

                    # TODO implement different viewing modes for payloads
                    # modes = spacecraft_list[spacecraft_idx].get('instrument', None)
                    # if not isinstance(modes, list):
                    #     modes = [0]
                    modes = [0]

                    gp_acces_by_mode = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]','instrument',
                                                                'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])
                    for mode in modes:
                        i_mode = modes.index(mode)
                        gp_access_by_grid = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]',
                                                                'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

                        for grid in mission_dict.get('grid'):
                            i_grid = mission_dict.get('grid').index(grid)
                            metrics_file = os.path.join(orbitdata_path, agent_folder, f'datametrics_instru{i_ins}_mode{i_mode}_grid{i_grid}.csv')
                            
                            try:
                                metrics_data = pd.read_csv(metrics_file, skiprows=range(4))
                                
                                nrows, _ = metrics_data.shape
                                grid_id_column = [i_grid] * nrows
                                metrics_data['grid index'] = grid_id_column

                                if len(gp_access_by_grid) == 0:
                                    gp_access_by_grid = metrics_data
                                else:
                                    gp_access_by_grid = pd.concat([gp_access_by_grid, metrics_data])
                            except pd.errors.EmptyDataError:
                                continue

                        nrows, _ = gp_access_by_grid.shape
                        gp_access_by_grid['pnt-opt index'] = [mode] * nrows

                        if len(gp_acces_by_mode) == 0:
                            gp_acces_by_mode = gp_access_by_grid
                        else:
                            gp_acces_by_mode = pd.concat([gp_acces_by_mode, gp_access_by_grid])
                        # gp_acces_by_mode.append(gp_access_by_grid)

                    nrows, _ = gp_acces_by_mode.shape
                    gp_access_by_grid['instrument'] = [instrument['name']] * nrows
                    # gp_access_data[ins_name] = gp_acces_by_mode

                    if len(gp_access_data) == 0:
                        gp_access_data = gp_acces_by_mode
                    else:
                        gp_access_data = pd.concat([gp_access_data, gp_acces_by_mode])
                
                nrows, _ = gp_access_data.shape
                gp_access_data['agent name'] = [spacecraft_list[spacecraft_idx]['name']] * nrows

                # limit gp access data to simulation duration
                gp_access_data = gp_access_data[gp_access_data['time index'] <= max_time_index]
                
            elif agent_type == 'groundOperator':
                # Ground Operators have no sensing capability; create empty ground point coverage data
                gp_access_data = pd.DataFrame(columns=['time index','grid index', 'GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                        'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

            else:
                raise ValueError(f'Unknown agent type `{agent_type}` for agent `{agent_name}`.')

            # store in dictionary
            data[agent_name] = gp_access_data

        # return compiled ground point access data
        return data

    @staticmethod
    def __load_grid_data(orbitdata_path : str,
                         mission_dict : dict
                        ) -> List[pd.DataFrame]:
        # initialize grid data
        grid_data_compiled = []

        # compile coverage grid information
        for grid in mission_dict.get('grid'):
            grid : dict
            i_grid = mission_dict.get('grid').index(grid)
            
            if grid.get('@type').lower() == 'customgrid':
                grid_file = grid.get('covGridFilePath')
                
            elif grid.get('@type').lower() == 'autogrid':
                grid_file = os.path.join(orbitdata_path, f'grid{i_grid}.csv')
            else:
                raise NotImplementedError(f"Loading of grids of type `{grid.get('@type')} not yet supported.`")

            grid_data = pd.read_csv(grid_file)
            nrows, _ = grid_data.shape
            grid_data['grid index'] = [i_grid] * nrows
            grid_data['GP index'] = [i for i in range(nrows)]
            grid_data_compiled.append(grid_data)

        # return compiled grid data
        return grid_data_compiled
 
    @staticmethod
    def __write_interval_data_table(df: pd.DataFrame,
                                    bin_dir: str,
                                    table_name: str,
                                    time_step : float,
                                    *,
                                    start_col: str = "start index",
                                    end_col: str = "end index",
                                    sort_by_time: bool = True,
                                    string_max_unique: Optional[int] = None, 
                                    allow_overwrite: bool = True,
                                    packed_dtype: np.dtype = np.float64
                                ) -> Dict[str, Any]:
        """
        Writes interval data to memmap-able .npy arrays + meta.json.

        Required columns: start_col, end_col (integer indices).
        Additional columns: numeric/bool only (unless you pre-encode strings).
        """
        # validate output directory
        out_dir = os.path.join(bin_dir, table_name) 
        os.makedirs(out_dir, exist_ok=True)

        # define metadata path
        meta_path = os.path.join(out_dir, "meta.json")
        
        # check if metadata file already exists 
        if (not allow_overwrite) and os.path.exists(meta_path):
            # data exists and overwriting is not allowed; raise error
            raise FileExistsError(f"State table already exists at: {out_dir}")

        # Keep only needed columns (required + extras)
        cols = list(df.columns)
        if start_col not in cols or end_col not in cols:
            raise ValueError("start/end columns not found in df")

        # Copy just the columns we will store
        work = df.copy(deep=False)

        # extract start and end time data 
        start, start_dtype = OrbitData.__extract_time_data(work, start_col, time_step)
        end, end_dtype = OrbitData.__extract_time_data(work, end_col, time_step)

        # sort by time if toggle is enabled
        if sort_by_time:
            # stable sort by start then end
            order = np.lexsort((end, start))
            start = start[order]
            end = end[order]
            work = work.iloc[order].reset_index(drop=True)

        # prefix max end for fast existence checks
        prefix_max_end = np.maximum.accumulate(end)

        # select extra columns (everything except start/end)
        extra_cols = [c for c in work.columns if c != start_col and c != end_col]

        # We'll build extras in deterministic order using safe names (like before)
        extras_safe: list[str] = []
        extras_data: dict[str, np.ndarray] = {}   # safe -> numeric array (already encoded for strings)
        columns_meta: dict[str, Any] = {}         # safe -> metadata (keeps your current structure)

        # encode extras (mostly same as your existing code, but don't write per-column files)
        for col in extra_cols:
            # get a safe name for the column
            safe = OrbitData.__safe_name(col)

            # get data series for working column
            s: pd.Series = work[col]

            # evaluate if column is string-like 
            is_stringish = (
                pd.api.types.is_object_dtype(s.dtype)
                or pd.api.types.is_string_dtype(s.dtype)
                or pd.api.types.is_categorical_dtype(s.dtype)
            )

            # compile data according to column type
            if is_stringish:
                # convert to pandas strings (normalizes NaNs)
                s2 = s.astype("string")

                # get unique string values
                uniques = s2.dropna().unique()

                # check if number of unique strings exceeds max allowed for dictionary encoding
                if string_max_unique is not None and len(uniques) > string_max_unique:
                    raise ValueError(
                        f"Column '{col}' has {len(uniques)} unique strings; too many for dictionary encoding."
                    )

                # get list of unique strings 
                uniq_list = [str(x) for x in uniques]

                # map strings to integer codes (0 reserved for null/NaN)
                str_to_code = {v: i + 1 for i, v in enumerate(uniq_list)}  # 0 reserved for null
                
                # map original string series to integer codes, using 0 for NaN/null values
                mapped = s2.map(lambda x: str_to_code.get(str(x), 0) if pd.notna(x) else 0)

                # convert mapped codes to numpy array
                codes = mapped.to_numpy(dtype=np.int32, na_value=0)

                # save code to string mappings 
                vocab = {"0": None, **{str(i + 1): v for i, v in enumerate(uniq_list)}}

                # store encoded codes as numeric (we'll pack into float array later)
                extras_safe.append(safe)
                extras_data[safe] = codes  # keep as int32 for now
                columns_meta[safe] = {
                    "col_name": col,
                    "kind": "string_dict",
                    "vocab": vocab,
                    "code_dtype": str(codes.dtype),
                }

            else:
                arr = s.to_numpy()
                dtype = OrbitData.__pick_numpy_dtype(s)
                arr = arr.astype(dtype, copy=False)

                extras_safe.append(safe)
                extras_data[safe] = arr
                columns_meta[safe] = {
                    "col_name": col,
                    "kind": "numeric",
                    "dtype_original": str(arr.dtype),
                }

        # number of rows and total columns (start, end, prefix, extras)
        n = int(len(work))
        layout = ["start", "end", "prefix_max_end", *extras_safe]
        k = len(layout)

        # pack into one (N,K) array 
        packed = np.empty((n, k), dtype=np.dtype(packed_dtype))
        
        # pack required columns (start, end, prefix_max_end)
        packed[:, 0] = start.astype(packed_dtype, copy=False)
        packed[:, 1] = end.astype(packed_dtype, copy=False)
        packed[:, 2] = prefix_max_end.astype(packed_dtype, copy=False)

        # pack additional columns in order defined by layout
        for j, safe in enumerate(extras_safe, start=3):
            packed[:, j] = extras_data[safe].astype(packed_dtype, copy=False)

        # save single packed file
        packed_fname = "intervals.npy"
        np.save(os.path.join(out_dir, packed_fname), packed)

        # metadata (keep very similar to yours)
        meta: Dict[str, Any] = {
            "name": table_name,
            "n": n,
            "layout": layout,  # <--- NEW: defines column order in intervals.npy
            "shape": [n, k],
            "packed_dtype": str(np.dtype(packed_dtype)),
            "columns": columns_meta,  # per-extra metadata incl vocab for strings
            "dtypes": {
                "start_original": str(np.dtype(start_dtype)),
                "end_original": str(np.dtype(end_dtype)),
            },
            "files": {"intervals": packed_fname},  # <--- ONLY ONE
            "sorted_by_time": bool(sort_by_time),
            "dir": out_dir,
            "start_col_name": start_col,
            "end_col_name": end_col,
        }

        # write metadata to file
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        return meta
    
    @staticmethod
    def __extract_time_data(df: pd.DataFrame, 
                            time_col: str, 
                            time_step: float, 
                            dtype=None
                        ) -> Tuple[np.ndarray, np.dtype]:
        """ 
        Extracts time data from a dataframe column and converts to a numpy array with 
        appropriate scaling and dtype for efficient storage. 

        If time data is expressed as a list of integers, it converts them to time in seconds
        by multiplying with the time step. 
        """
        if time_col in df.columns:
            # check if start time data is stored as integer indeces
            if "index" in time_col.lower(): 
                # convert existing time index data to numpy array
                t_idx : np.ndarray = df[time_col].to_numpy()
                
                # check if time index is already an integer type
                if not np.issubdtype(t_idx.dtype, np.integer):
                    # if not, convert to int64
                    t_idx = t_idx.astype(np.int64, copy=False)

                # pick an appropriate data type in case it's not already provided
                if dtype is None:
                    dtype = OrbitData.__pick_numpy_dtype(df[time_col] * time_step)

                # convert time index to time in seconds
                t : np.ndarray = t_idx * time_step

            else:
                # convert existing time column to numpy array
                t : np.ndarray= df[time_col].to_numpy()

                # pick an appropriate data type in case it's not already provided
                if dtype is None:
                    dtype = OrbitData.__pick_numpy_dtype(df[time_col])
        
        # return time data as numpy array of the desired dtype
        return t.astype(dtype, copy=False), dtype

    @staticmethod
    def __pick_numpy_dtype(series: pd.Series) -> np.dtype:
        """
        Choose a stable numpy dtype for a dataframe column.
        Keep it simple: ints -> int32/int64, floats -> float32/float64, bool -> bool.
        """
        if series.empty:
            # default to float32 for empty series
            return np.dtype(np.float32)

        # If it's already a known dtype, keep your fast paths
        if pd.api.types.is_bool_dtype(series):
            return np.dtype(np.bool_)
        if pd.api.types.is_integer_dtype(series):
            return np.dtype(np.int32)
        if pd.api.types.is_float_dtype(series):
            return np.dtype(np.float64)

        # Handle object dtype that is actually numeric strings / python floats
        if series.dtype == "object":
            # timedelta objects?
            inferred = pd.api.types.infer_dtype(series, skipna=True)
            if inferred in ("timedelta",):
                # choose a stable representation for storage:
                # store as int64 nanoseconds (recommended)
                return np.dtype("int64")

            # try numeric coercion
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().all() or series.isna().all():
                # numeric (or numeric + NaNs)
                return np.dtype(np.float64)

        raise TypeError(
            f"Unsupported dtype for column '{series.name}': {series.dtype} "
            f"(inferred={pd.api.types.infer_dtype(series, skipna=True)})."
        )

    @staticmethod
    def __safe_name(col: str) -> str:
        return col.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
    

    @staticmethod
    def __write_state_table(df: pd.DataFrame,
                            bin_dir: str,
                            table_name: str,
                            time_step : float,
                            *,
                            t_col: str = "time index",
                            pos_cols: Tuple[str, str, str] = ("x [km]", "y [km]", "z [km]"),
                            vel_cols: Tuple[str, str, str] = ("vx [km/s]", "vy [km/s]", "vz [km/s]"),
                            state_dtype: np.dtype = np.float64,
                            sort_by_time: bool = True,
                            allow_overwrite: bool = True,
                        ) -> Dict[str, Any]:
        """
        Writes a time-indexed position/velocity table to memmap-able .npy arrays + meta.json.

        Files:
        - t.npy (optional if t_col is None)
        - pos.npy shape (N,3)
        - vel.npy shape (N,3)
        - plus any extra numeric columns you decide to add later
        """
        # validate output directory
        out_dir = os.path.join(bin_dir, table_name) 
        os.makedirs(out_dir, exist_ok=True)

        # define metadata path
        meta_path = os.path.join(out_dir, "meta.json")
        
        # check if metadata file already exists 
        if (not allow_overwrite) and os.path.exists(meta_path):
            # data exists and overwriting is not allowed; raise error
            raise FileExistsError(f"State table already exists at: {out_dir}")

        # ensure required columns are present in dataframe
        for c in pos_cols + vel_cols:
            if c not in df.columns:
                raise KeyError(f"Missing required column '{c}'")

        # Copy just the columns we will store
        work = df.copy(deep=False)

        # ensure time column is provided
        if t_col is None:
            raise ValueError("Time column name (t_col) must be provided for state tables.")

        # ensure desired time column exists in data
        if t_col not in work.columns: raise KeyError(f"Missing time column '{t_col}'")
        
        # convert time index to time in seconds and ensure it's the desired dtype
        t,t_dtype = OrbitData.__extract_time_data(work, t_col, time_step)

        # sort time if toggle is enabled
        if sort_by_time:
            order = np.argsort(t, kind="mergesort")  
            work = work.iloc[order].reset_index(drop=True)
            t = t[order]

        # extract position and velocity data from dataframe;
        #  convert to numpy arrays with desired dtype 
        pos = work.loc[:, list(pos_cols)].to_numpy(dtype=state_dtype, copy=False)
        vel = work.loc[:, list(vel_cols)].to_numpy(dtype=state_dtype, copy=False)

        # --- pack into one array (N,7) ---
        # Choose one dtype for the packed array. To keep it simple and memmap-friendly:
        n = int(len(work))
        packed_dtype = np.dtype(state_dtype)

        state = np.empty((n, 7), dtype=packed_dtype)
        state[:, 0] = t.astype(packed_dtype, copy=False)
        state[:, 1:4] = pos
        state[:, 4:7] = vel

        # save single packed file
        state_fname = "state.npy"
        np.save(os.path.join(out_dir, state_fname), state)

        # metadata
        files = {"state": state_fname}
        layout = ["t", *pos_cols, *vel_cols]  # length 7

        meta = {
            "name": table_name,
            "n": n,
            "layout": layout,               # order of columns in state.npy
            "dtypes": {
                "state": str(packed_dtype), # dtype of packed array
                "t_original": str(np.dtype(t_dtype)),  # for reference/debug
            },
            "files": files,
            "sorted_by_time": bool(sort_by_time),
            "dir": out_dir,
            "shape": [n, 7],
        }

        # save metadata to json file
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        # return metadata 
        return meta

    @staticmethod
    def __write_access_table(df: pd.DataFrame,
                                bin_dir: str,
                                table_name: str,
                                time_step : float,
                                n_steps: int,
                                *,
                                t_col: str = 'time index',
                                required_cols: Sequence[str] = ['GP index','grid index','lat [deg]','lon [deg]', 'instrument'],
                                sort_within_time: bool = False,
                                string_max_unique: Optional[int] = None,  # optional guardrail
                                allow_overwrite: bool = True,
                                packed_dtype: np.dtype = np.float64
                            ) -> Dict[str, Any]:
        """
        Writes a ragged table with unknown columns to memmap-friendly binaries:
        - offsets.npy
        - one .npy per column (numeric) OR codes + dict for strings
        - meta.json
        """
        # validate output directory
        out_dir = os.path.join(bin_dir, table_name) 
        os.makedirs(out_dir, exist_ok=True)

        # define metadata path
        meta_path = os.path.join(out_dir, "meta.json")
        
        # check if metadata file already exists 
        if (not allow_overwrite) and os.path.exists(meta_path):
            # data exists and overwriting is not allowed; raise error
            raise FileExistsError(f"State table already exists at: {out_dir}")


        # validate required columns are present in dataframe
        for c in (t_col, *required_cols):
            if c not in df.columns:
                raise KeyError(f"Missing required column: {c}")

        # copy just the columns we will store
        work = df.copy(deep=False)

        # extract time data
        t,t_dtype = OrbitData.__extract_time_data(work, t_col, time_step)
        
        # convert to indices for bucketing
        t_idx = (t / time_step).astype(np.int64, copy=False) 

        # Determine steps
        # T = 0 if len(t_idx) == 0 else int(t_idx.max()) + 1
        if n_steps is None:
            T = 0 if len(t_idx) == 0 else int(t_idx.max()) + 1
        else:
            T = int(n_steps)

        # Filter `t_idx` to [0, T-1]
        if len(t) > 0:
            mask = (t_idx >= 0) & (t_idx < T)
            if not np.all(mask):
                work = work.loc[mask].copy()
                t = t[mask]
                t_idx = t_idx[mask]

        # Sort by time so each bucket is contiguous
        if sort_within_time:
            # stable tie-breaker: keep existing row order by using mergesort later
            order = np.lexsort((np.arange(len(t_idx), dtype=t_dtype), t_idx))
        else:
            order = np.argsort(t_idx, kind="mergesort")

        work = work.iloc[order]
        t = t[order]
        t_idx = t_idx[order]

        # Offsets
        counts = np.bincount(t_idx, minlength=T).astype(np.int64, copy=False)
        offsets = np.empty(T + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        # Decide columns to write (everything except t_col; we will include packed "t" and "t_index" explicitly)
        cols_to_write = [c for c in work.columns if c != t_col]

        # encode additional data
        columns_meta: Dict[str, Any] = {}
        extras_safe: list[str] = []
        extras_data: dict[str, np.ndarray] = {}

        for col in cols_to_write:
            safe = OrbitData.__safe_name(col)
            s: pd.Series = work[col]

            is_stringish = (
                pd.api.types.is_object_dtype(s.dtype)
                or pd.api.types.is_string_dtype(s.dtype)
                or pd.api.types.is_categorical_dtype(s.dtype)
            )

            if is_stringish:
                s2 = s.astype("string")
                uniques = s2.dropna().unique()

                if string_max_unique is not None and len(uniques) > string_max_unique:
                    raise ValueError(
                        f"Column '{col}' has {len(uniques)} unique strings; too many for dictionary encoding."
                    )

                uniq_list = [str(x) for x in uniques]
                str_to_code = {v: i + 1 for i, v in enumerate(uniq_list)}  # 0 reserved for null
                mapped = s2.map(lambda x: str_to_code.get(str(x), 0) if pd.notna(x) else 0)
                codes = mapped.to_numpy(dtype=np.int32, na_value=0)

                vocab = {"0": None, **{str(i + 1): v for i, v in enumerate(uniq_list)}}

                extras_safe.append(safe)
                extras_data[safe] = codes  
                columns_meta[safe] = {
                    "kind": "string_dict",
                    "col_name": col,
                    "vocab": vocab,
                    "code_dtype": str(codes.dtype),
                }
            else:
                arr = s.to_numpy()
                dtype = OrbitData.__pick_numpy_dtype(s)
                arr = arr.astype(dtype, copy=False)

                extras_safe.append(safe)
                extras_data[safe] = arr
                columns_meta[safe] = {
                    "kind": "numeric",
                    "col_name": col,
                    "dtype_original": str(arr.dtype),
                }

        # get number of rows and total columns (t, t_index, extras)
        n_rows = int(offsets[-1])
        layout = ["t", "t_index", *extras_safe]
        k = len(layout)

        # package data 
        rows = np.empty((n_rows, k), dtype=np.dtype(packed_dtype))
        rows[:, 0] = t.astype(packed_dtype, copy=False)
        rows[:, 1] = t_idx.astype(packed_dtype, copy=False)

        for j, safe in enumerate(extras_safe, start=2):
            rows[:, j] = extras_data[safe].astype(packed_dtype, copy=False)

        # write binaries
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "rows.npy"), rows)

        # metadata
        meta: Dict[str, Any] = {
            "name": table_name,
            "n_rows": n_rows,
            "time_step": float(time_step),
            "n_steps": int(T),
            "format": "ragged_offsets_packed_rows",
            "t_col": t_col,
            "required_cols": list(required_cols),
            "layout": layout,
            "shape": [n_rows, k],
            "packed_dtype": str(np.dtype(packed_dtype)),
            "columns": columns_meta,
            "dtypes": {
                "t_original": str(np.dtype(t_dtype)),
            },
            "files": {
                "offsets": "offsets.npy",
                "rows": "rows.npy",
            },
            "dir": out_dir,
        }

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        return meta
    
    @staticmethod
    def __write_grid_table(dfs: List[pd.DataFrame],
                            bin_dir: str,
                            table_name: str,
                            *,                            
                            required_cols: Sequence[str] = ['lat [deg]','lon [deg]', 'grid index', 'GP index'],
                            packed_dtype: np.dtype = np.float64,
                            allow_overwrite: bool = True,
                        ) -> Dict[str, Any]:
        # validate output directory
        out_dir = os.path.join(bin_dir, table_name) 
        os.makedirs(out_dir, exist_ok=True)
        
        # define metadata path
        meta_path = os.path.join(out_dir, "meta.json")
        
        # check if metadata file already exists 
        if (not allow_overwrite) and os.path.exists(meta_path):
            # data exists and overwriting is not allowed; raise error
            raise FileExistsError(f"State table already exists at: {out_dir}")

        # concatenate grid dataframes
        work = pd.concat(dfs, ignore_index=True)

        # validate required columns are present
        for c in required_cols:
            if c not in work.columns:
                raise KeyError(f"Missing required column: {c}")
        if len(work.columns) > len(required_cols):
            raise ValueError(f"Unexpected extra columns found in grid data: {set(work.columns) - set(required_cols)}")

        # get number of rows and columns
        n = int(len(work))
        layout = list(required_cols)  
        k = len(layout)

        # build packed array
        packed = np.empty((n, k), dtype=np.dtype(packed_dtype))
        for j, col in enumerate(layout):
            # robust numeric conversion (avoids object dtype issues)
            s = work[col]
            arr = pd.to_numeric(s, errors="raise").to_numpy(copy=False)
            packed[:, j] = arr.astype(packed_dtype, copy=False)

        # write single file
        packed_fname = "grid.npy"
        np.save(os.path.join(out_dir, packed_fname), packed)

        # metadata
        meta = {
            "name": table_name,
            "n": n,
            "layout": layout,
            "shape": [n, k],
            "packed_dtype": str(np.dtype(packed_dtype)),
            "files": {"grid": packed_fname},
            "dir": out_dir,
        }

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        
        # return metadata
        return meta 
