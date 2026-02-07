from collections import defaultdict
import copy
from dataclasses import dataclass
from enum import Enum
import gc
import json
from math import ceil
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
        
class OrbitData:
    """
    Stores and queries data regarding an agent's orbital data. 

    TODO: add support to load ground station agents' data
    """
    JDUT1 = 'JDUT1'

    def __init__(self,
                 agent_name : str,
                 gs_network_name : str,
                 time_step : float,
                 epoch_type : str,
                 epoch : float,
                 duration : float,
                 eclipse_data : IntervalTable,
                 state_data : StateTable,
                 comms_links : Dict[str, IntervalTable],
                 gs_access_data : IntervalTable,
                 gp_access_data : AccessTable,
                 grid_data: TargetGridTable
                ):
        # assign attributes
        self.agent_name = agent_name
        self.gs_network_name = gs_network_name
        self.time_step = time_step
        self.epoch_type = epoch_type
        self.epoch = epoch
        self.duration = duration

        # agent state data
        self.eclipse_data = eclipse_data
        self.state_data = state_data

        # agent connectivity data
        self.comms_links = comms_links
        self.gs_access_data = gs_access_data
        
        # ground point access data
        self.gp_access_data = gp_access_data

        # grid data
        self.grid_data = grid_data   
    
    """
    LOAD FROM PRE-COMPUTED DATA
    """
    @staticmethod
    def from_directory(orbitdata_dir: str, simulation_duration : float, printouts : bool = True) -> Dict[str, 'OrbitData']:
        # TODO check if schemas have already been generated at the provided directory and load from there if so, o
        # therwise preprocess data and generate schemas before loading
        
        # preprocess data and store as binarys for faster loading in the future
        schemas : dict[str, dict] \
            = OrbitData.preprocess(orbitdata_dir, simulation_duration, printouts=printouts)

        # force garbage collection after loading data to free up memory
        gc.collect() 

        data = dict()
        for agent_name, schema in schemas.items():
            # unpack agent-specific data from schema
            gs_network_name = schema['gs_network_name']
            time_step = schema['time_specs']['time step']
            epoch_type = schema['time_specs']['epoch type']
            epoch = schema['time_specs']['epoch']
            duration = schema['time_specs']['duration']

            # load eclipse data from binary
            eclipse_data = IntervalTable.from_schema(schema['eclipse'], mmap_mode='r')
            
            # load ground station access data from binary
            gs_access_data = IntervalTable.from_schema(schema['gs_access'], mmap_mode='r')

            # load comms link data from binary
            comms_links = dict()
            for link_name, comms_meta in schema['comms_links'].items():
                comms_links[link_name] = IntervalTable.from_schema(comms_meta, mmap_mode='r')

            # load ground point access data from binary
            gp_access_data = AccessTable.from_schema(schema['gp_access'], mmap_mode='r')

            # load grid data from binary
            grid_data = TargetGridTable.from_schema(schema['grid'], mmap_mode='r')

            # load state data from binary 
            state_data = StateTable.from_schema(schema['state'], mmap_mode='r')

            data[agent_name] = OrbitData(agent_name, gs_network_name, time_step, epoch_type, epoch, 
                                            duration, eclipse_data, state_data, comms_links, gs_access_data, 
                                            gp_access_data, grid_data)
            
        # return compiled data
        return data     

    """
    GET NEXT methods
    """
    def get_next_agent_access(self, target : str, t: float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to another agent or ground station after or during time `t` up to a given time `t_max`. """

        # check if target is within the list of known agents
        assert target in self.comms_links.keys(), f'No comms data found for target agent `{target}`.'

        # return next access interval
        return self.__get_next_interval(self.comms_links[target], t, t_max, include_current)

    def get_next_gs_access(self, t, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to a ground station after or during time `t`. """
        return self.__get_next_interval(self.gs_access_data, t, t_max, include_current)

    def get_next_eclipse_interval(self, t: float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next eclipse interval after or during time `t`. """
        return self.__get_next_interval(self.eclipse_data, t, t_max, include_current)

    def __get_next_interval(self, interval_data : IntervalTable, t : float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval from `interval_data` after or during time `t`. """
        # get next intervals
        future_intervals : list[Interval] = interval_data.lookup_interval(t, t_max, include_current)

        # # check if current interval should be included
        # if not include_current:
        #     # exclude intervals that contain time `t`
        #     future_intervals = [interval for interval in future_intervals
        #                         if t < interval.left] # interval starts after time `t`
        # else:
        #     # include current intervals but clip to start at time `t`
        #     future_intervals = [Interval(max(t, interval.left), interval.right) if interval.left <= t <= interval.right else interval
        #                         for interval in future_intervals]

        # check if there are any valid intervals
        if not future_intervals: return None

        # get next interval
        next_interval = future_intervals[0]

        # return the first interval that starts after or at time `t`
        return Interval(max(t, next_interval.left), min(next_interval.right, t_max))

    def get_next_agent_accesses(self, target : str, t: float, t_max: float = np.Inf, include_current: bool = False) -> List[Interval]:
        """ returns a list of the next access interval to another agent or ground station after or during time `t` up to a given time `t_max`. """

        # check if target is within the list of known agents
        assert target in self.comms_links.keys(), f'No comms data found for target agent `{target}`.'

        # get next access intervals
        future_access_intervals = self.__get_next_intervals(self.comms_links[target], t, t_max, include_current)

        # return in interval form
        return [Interval(t_start,t_end) for t_start,t_end in future_access_intervals] if future_access_intervals else []

    def __get_next_intervals(self, interval_data : IntervalTable, t : float, t_max: float = np.Inf, include_current: bool = False) -> List[Tuple[float, float]]:
        # find all intervals that end after time `t` and start before time `t_max`
        future_intervals : List[Interval] = interval_data.lookup_intervals(t, t_max, include_current)
        
        # convert to tuple form
        future_interval_pairs = [(interval.left, interval.right) for interval in future_intervals]
        
        # # check if current interval should be included
        # if not include_current:
        #     # exclude intervals that contain time `t`
        #     future_interval_pairs = [(t_start, t_end) for t_start,t_end in future_interval_pairs
        #                         if t < t_start] # interval starts after time `t`
        # else:
        #     # include current intervals but clip to start at time `t`
        #     future_interval_pairs = [(max(t, t_start), t_end) for t_start,t_end in future_interval_pairs]

        # check if there are any valid intervals
        if not future_interval_pairs: return None
        
        # sort by start time and return
        return sorted(future_interval_pairs, key=lambda interval: interval[0])
    
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
                        for h in os.listdir(f_dir):
                            os.remove(os.path.join(f_dir, h))
                        os.rmdir(f_dir)
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

        # update mission duration if needed
        orbitdata_filename = os.path.join(data_dir, 'MissionSpecs.json')
        ## check if mission specs file exists
        original_duration = scenario_specs['duration']
        if os.path.exists(orbitdata_filename):
            with open(orbitdata_filename, 'r') as orbitdata_specs:
                # load existing mission specs
                orbitdata_dict : dict = json.load(orbitdata_specs)

                # update duration to that of the longest mission
                scenario_specs['duration'] = max(scenario_specs['duration'], orbitdata_dict['duration'])

        # save specifications of propagation in the orbit data directory
        with open(os.path.join(data_dir,'MissionSpecs.json'), 'w') as mission_specs:
            mission_specs.write(json.dumps(scenario_specs, indent=4))
            assert os.path.exists(os.path.join(data_dir,'MissionSpecs.json')), \
                'Mission specifications not saved correctly!'
            
        # restore original duration value
        scenario_specs['duration'] = original_duration

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
                # or scenario_dict['scenario']['connectivity'] != mission_dict['scenario']['connectivity']
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
        for *__,agent_name,gs_network_name in agents_loaded: 
            # extract relevant data for this agent
            agent_eclipse_data = eclipse_dfs[agent_name]
            agent_state_data = state_dfs[agent_name]
            agent_gs_access_data = gs_access_dfs[agent_name]
            agent_gp_access_data = gp_access_dfs[agent_name]
            agent_comms_link_data = { u if u != agent_name else v : comms_links
                                    for (u,v), comms_links in comms_link_dfs.items()
                                    if agent_name in (u,v) }                
            
            # define agent-specific binary output directory
            agent_bin_dir = os.path.join(bin_dir, agent_name)
            
            # print interval data to binaries 
            eclipse_meta = OrbitData.__write_interval_data_table(agent_eclipse_data, agent_bin_dir, 'eclipse', time_specs['time step'], allow_overwrite=True)
            gs_meta = OrbitData.__write_interval_data_table(agent_gs_access_data, agent_bin_dir, 'gs_access', time_specs['time step'], allow_overwrite=True)

            # print comms link data to binaries
            agent_comms_bin_dir = os.path.join(agent_bin_dir, 'comm')
            comms_metas = dict()
            for link_name, comms_links in agent_comms_link_data.items():
                comms_meta = OrbitData.__write_interval_data_table(comms_links, agent_comms_bin_dir, link_name + '_comm', time_specs['time step'], allow_overwrite=True)
                comms_metas[link_name] = comms_meta

            # print agent position data to binaries
            state_meta = OrbitData.__write_state_table(agent_state_data, agent_bin_dir, 'cartesian_state', time_specs['time step'], allow_overwrite=True)
            
            # print agent's groundpoint coverage data to binaries
            gp_access_meta = OrbitData.__write_access_table(agent_gp_access_data, agent_bin_dir, 'gp_access', time_step=time_specs['time step'], n_steps=time_specs['n_steps'], allow_overwrite=True)

            # print grid data to binaries
            grid_meta = OrbitData.__write_grid_table(grid_data_dfs, agent_bin_dir, 'grid', allow_overwrite=True)

            # compile schema for this agent and store in dictionary
            schemas[agent_name] = {
                'gs_network_name': gs_network_name,
                'time_specs' : time_specs,
                'dir' : agent_bin_dir,
                'eclipse': eclipse_meta,
                'gs_access': gs_meta,
                'comms_links': comms_metas,
                'state': state_meta,
                'gp_access': gp_access_meta,
                'grid': grid_meta
            }
            
        with open(os.path.join(bin_dir, "meta.json"), "w") as f:
            json.dump(schemas, f, indent=4)

        # return compiled schemas for all agents
        return schemas
        
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

        # get scenario settings
        scenario_dict : dict = mission_dict.get('scenario', None)

        # get connectivity setting
        connectivity : str = scenario_dict.get('connectivity', None) \
            if scenario_dict else ConnectivityLevels.LOS.value # default to LOS if not specified

        # compile list of all agents to load
        agents_loaded : List[dict] = []
        for i,spacecraft_dict in enumerate(spacecraft_list):
            spacecraft_name : str = spacecraft_dict['name']
            gs_network_name = spacecraft_dict.get('groundStationNetwork', None)
            agents_loaded.append(('spacecraft', i, spacecraft_name, gs_network_name))
        for i,ground_op_dict in enumerate(ground_ops_list):
            ground_op_name : str = ground_op_dict['name']
            agents_loaded.append(('groundOperator', i, ground_op_name, ground_op_name))

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
        gs_access_dfs : Dict[str, pd.DataFrame] = OrbitData.__load_agent_gs_access_data(orbitdata_dir, agents_loaded, simulation_duration, time_step, ground_station_list, connectivity, tqdm_config)

        # load comms link data
        comms_link_dfs : Dict[tuple, pd.DataFrame] = OrbitData.__load_agent_comms_link_data(orbitdata_dir, agents_loaded, simulation_duration, gs_access_dfs, time_step, connectivity)

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
        for agent_type, spacecraft_idx, agent_name, _ in agents_to_load:
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
        for agent_type, spacecraft_idx, agent_name, _ in agents_to_load:
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
                                   connectivity : str,
                                   tqdm_config : dict,
                                ) -> Dict[str, pd.DataFrame]:
        # initialize ground station access data
        data = dict()

        for agent_type, spacecraft_idx, agent_name, gs_network_name in agents_to_load:
            if agent_type == 'spacecraft':
                # define agent folder
                sat_id = "sat" + str(spacecraft_idx)
                agent_folder = sat_id + '/' 

                # compile list of ground stations that are part of the desired network                
                gs_network_station_ids : List[str] = [ gs['@id'] for gs in ground_station_list
                                                    if gs.get('networkName', None) == gs_network_name ]

                # create empty dataframe to store ground station access data for this agent
                gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]'])
                
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
                    if gs_network_station_ids and gndStn_id not in gs_network_station_ids:
                        continue

                    # load ground station access data
                    gndStn_access_file = os.path.join(orbitdata_path, agent_folder, file)
                    
                    gndStn_access_data : pd.DataFrame \
                        = OrbitData.__load_gs_access_data(gndStn_access_file, connectivity, simulation_duration, time_step)

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
    def __load_gs_access_data(gs_access_file : str, connectivity : str, simulation_duration : float, time_step : float) -> pd.DataFrame:
        if connectivity.upper() == ConnectivityLevels.FULL.value:
            # fully connected network; modify connectivity 
            columns = ['start index', 'end index']
            
            # generate mission-long connectivity access                    
            data = [[0.0, simulation_duration * 24 * 3600 // time_step + 1]]  # full connectivity from start to end of mission
            assert data[0][1] > 0.0

            # return modified connectivity
            gndStn_access_data = pd.DataFrame(data=data, columns=columns)
        
        elif connectivity.upper() == ConnectivityLevels.LOS.value:
            # line-of-sight driven connectivity; load ground station access data
            gndStn_access_data = pd.read_csv(gs_access_file, skiprows=range(3))

            # limit ground station access data to simulation duration
            max_time_index = int(simulation_duration * 24 * 3600 // time_step)
            gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
            gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index
        
        elif connectivity.upper() == ConnectivityLevels.GS.value:
            # ground station-only connectivity; load ground station access data
            gndStn_access_data = pd.read_csv(gs_access_file, skiprows=range(3))

            # limit ground station access data to simulation duration
            max_time_index = int(simulation_duration * 24 * 3600 // time_step)
            gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
            gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index

        elif connectivity.upper() == ConnectivityLevels.ISL.value:
            # inter-satellite link-driven connectivity; create empty dataframe
            columns = ['start index', 'end index']
            gndStn_access_data = pd.DataFrame(data=[], columns=columns)

        elif connectivity.upper() == ConnectivityLevels.NONE.value:
            # no inter-agent connectivity; create empty dataframe
            columns = ['start index', 'end index']
            gndStn_access_data = pd.DataFrame(data=[], columns=columns)

        else:
            # fallback; unsupported connectivity level
            raise ValueError(f'Unsupported connectivity level: {connectivity}.')
        
        # return ground station access data
        return gndStn_access_data

    @staticmethod
    def __load_agent_comms_link_data(orbitdata_path : str, 
                                     agents_to_load : List[tuple],  
                                     simulation_duration : float,
                                     gs_access_dfs : Dict[str, pd.DataFrame],
                                     time_step : float,
                                     connectivity : str
                                ) -> Dict[tuple, pd.DataFrame]:
        # initialize comms link data
        data = defaultdict(dict)

        # define path to comms data
        comms_path = os.path.join(orbitdata_path, 'comm')

        # iterate through agents to load
        for u_type,u_idx,u_name,u_gs_network in agents_to_load:
            for v_type,v_idx,v_name,v_gs_network in agents_to_load:
                # skip self-links
                if u_name == v_name: continue
                
                # pair key 
                key = tuple(sorted([u_name, v_name]))
                
                # skip if data for this pair of agents has already been loaded
                if key in data: continue 

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
                    comms_data = OrbitData.__load_isl_data(isl_file, connectivity, simulation_duration, time_step)

                elif u_type == 'groundOperator' and v_type == 'spacecraft':
                    if u_gs_network == v_gs_network: 
                        comms_data = gs_access_dfs[v_name]
                    else:
                        comms_data = pd.DataFrame(columns=['start index', 'end index'])

                elif u_type == 'spacecraft' and v_type == 'groundOperator':                    
                    if u_gs_network == v_gs_network: 
                        comms_data = gs_access_dfs[u_name]
                    else:
                        comms_data = pd.DataFrame(columns=['start index', 'end index'])
                    
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
    def __load_isl_data(isl_file : str, connectivity : str, duration_days : float, time_step : float) -> pd.DataFrame:        
        """ Loads ISL access data from a file and modifies it according to the specified connectivity level. """
        if connectivity.upper() == ConnectivityLevels.FULL.value:
            # fully connected network; modify connectivity 
            columns = ['start index', 'end index']
            
            # generate mission-long connectivity access
            duration = timedelta(days=float(duration_days))
            data = [[0.0, duration.total_seconds() // time_step + 1]]
            assert data[0][1] > 0.0

            # return modified connectivity
            isl_data = pd.DataFrame(data=data, columns=columns)

        elif connectivity.upper() == ConnectivityLevels.LOS.value:
            # line-of-sight driven connectivity; load connectivity and store data
            # TODO if ISL definition is modified in orbitpy, make sure this case is updated accordingly
            isl_data = pd.read_csv(isl_file, skiprows=range(3))

            # limit ISL data to simulation duration
            max_time_index = int(duration_days * 24 * 3600 // time_step)
            isl_data = isl_data[isl_data['start index'] <= max_time_index]
            isl_data.loc[isl_data['end index'] > max_time_index, 'end index'] = max_time_index

        elif connectivity.upper() == ConnectivityLevels.ISL.value:
            # inter-satellite link driven connectivity; load connectivity and store data
            # TODO if ISL definition is modified in orbitpy, make sure this case is updated accordingly
            isl_data = pd.read_csv(isl_file, skiprows=range(3))

            # limit ISL data to simulation duration
            max_time_index = int(duration_days * 24 * 3600 // time_step)
            isl_data = isl_data[isl_data['start index'] <= max_time_index]
            isl_data.loc[isl_data['end index'] > max_time_index, 'end index'] = max_time_index

        elif connectivity.upper() == ConnectivityLevels.GS.value:
            # ground station-only connectivity; create empty dataframe
            columns = ['start index', 'end index']
            isl_data = pd.DataFrame(data=[], columns=columns)

        elif connectivity.upper() == ConnectivityLevels.NONE.value:
            # no inter-agent connectivity; create empty dataframe
            columns = ['start index', 'end index']
            isl_data = pd.DataFrame(data=[], columns=columns)
        else:
            # fallback case for unsupported connectivity levels
            raise ValueError(f'Unsupported connectivity level: {connectivity}')
                
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
        for agent_type, spacecraft_idx, agent_name,_ in agents_to_load:
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
                gp_access_data = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
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

        # Write start/end/prefix
        np.save(os.path.join(out_dir, "start.npy"), start)
        np.save(os.path.join(out_dir, "end.npy"), end)
        np.save(os.path.join(out_dir, "prefix_max_end.npy"), prefix_max_end.astype(np.int32, copy=False))

        # select which columns are pending to be written
        cols_to_write = [c for c in work.columns if c != start_col and c != end_col]

        # initialize metadata dictionary
        meta: Dict[str, Any] = {
            "name": table_name,
            "n": int(len(work)),
            "columns": {},
            "dtypes" : {
                "start" : str(np.dtype(start_dtype)),
                "end" : str(np.dtype(end_dtype)),
                "prefix_max_end" : str(np.dtype(end_dtype))
            },
            "files": {
                "start": "start.npy", 
                "end": "end.npy", 
                "prefix_max_end": "prefix_max_end.npy"
            },
            "sorted_by_time": bool(sort_by_time),
            "dir" : out_dir
        }

        # write extra columns and update metadata
        for col in cols_to_write:
            # get a safe name for the column to use in file names
            safe = OrbitData.__safe_name(col)
            
            # get data series for working column
            s : pd.Series = work[col]

            # check if column is string-like (object, string, or categorical dtype)
            is_stringish = (
                pd.api.types.is_object_dtype(s.dtype)
                or pd.api.types.is_string_dtype(s.dtype)
                or pd.api.types.is_categorical_dtype(s.dtype)
            )

            if is_stringish:
                # convert to pandas strings (normalizes NaNs)
                s2 : pd.Series = s.astype("string")

                # get unique string values
                uniques = s2.dropna().unique()

                # if a limit is set, ensure number of unique strings is not too large for dictionary encoding
                if string_max_unique is not None and len(uniques) > string_max_unique:
                    raise ValueError(f"Column '{col}' has {len(uniques)} unique strings; too many for dictionary encoding.")

                # build string encoding mapping (code 0 reserved for NULL/NaN)
                uniq_list = [str(x) for x in uniques]
                str_to_code = {v: i + 1 for i, v in enumerate(uniq_list)}
                codes = np.zeros(len(s2), dtype=np.int32)

                # Fill codes
                # (vectorized mapping via pandas map)
                mapped = s2.map(lambda x: str_to_code.get(str(x), 0) if pd.notna(x) else 0)
                codes[:] = mapped.to_numpy(dtype=np.int32, na_value=0)

                # define file paths for codes and dictionary
                codes_path = f"col_{safe}__codes.npy"
                dict_path = f"dict_{safe}.json"

                # save codes and dictionary to files
                np.save(os.path.join(out_dir, codes_path), codes)
                with open(os.path.join(out_dir, dict_path), "w") as f:
                    json.dump({"0": None, **{str(i + 1): v for i, v in enumerate(uniq_list)}}, f, indent=4)

                # update metadata for this column
                meta["columns"][safe] = {
                    "col_name" : col,
                    "kind": "string_dict",
                    "dict_file": dict_path,
                }
                meta['files'][safe] = codes_path
                meta['dtypes'][safe] = str(codes.dtype)

            else:
                # convert series to numpy array 
                arr : np.ndarray = s.to_numpy()
                
                # pick an appropriate dtype for storage
                dtype = OrbitData.__pick_numpy_dtype(s)
                arr = arr.astype(dtype, copy=False)

                # save array to .npy file
                col_path = f"col_{safe}.npy"
                np.save(os.path.join(out_dir, col_path), arr)

                # update metadata for this column
                meta["columns"][safe] = {
                    "col_name" : col,
                    "kind": "numeric",
                }
                meta['files'][safe] = col_path
                meta['dtypes'][safe] = str(arr.dtype)

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
        if pd.api.types.is_bool_dtype(series):
            return np.dtype(np.bool_)
        if pd.api.types.is_integer_dtype(series):
            # Use int32 if safe (often time indices fit); else int64
            # If you know your ranges, you can force int32 for space.
            return np.dtype(np.int32)
        if pd.api.types.is_float_dtype(series):
            # float32 is usually fine for many telemetry-ish fields
            return np.dtype(np.float32)
        raise TypeError(f"Unsupported dtype for column '{series.name}': {series.dtype}. "
                        f"Encode strings/categories before writing.")

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
                            state_dtype: np.dtype = np.float32,
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

        # save time, position, and velocity data to .npy files
        np.save(os.path.join(out_dir, "t.npy"), t)
        np.save(os.path.join(out_dir, "pos.npy"), pos)
        np.save(os.path.join(out_dir, "vel.npy"), vel)
        
        # map filenames to columns
        files = {"t": "t.npy", "pos": "pos.npy", "vel": "vel.npy"}

        # store metadata 
        meta = {
            "name": table_name,
            "n": int(len(work)),
            "columns":{
                "pos": list(pos_cols),
                "vel": list(vel_cols)
            },
            "dtypes": {
                "t" : str(np.dtype(t_dtype)),
                "pos": str(np.dtype(state_dtype)),
                "vel": str(np.dtype(state_dtype))
            },
            "files": files,
            "sorted_by_time": bool(sort_by_time),
            "dir" : out_dir
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

        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "t.npy"), t)
        np.save(os.path.join(out_dir, "t_index.npy"), t_idx)

        # Decide which columns to write
        cols_to_write = [c for c in work.columns if c != t_col]

        meta: Dict[str, Any] = {
            "name": table_name,
            "n_rows": int(offsets[-1]),
            "n_steps": int(T),            
            "format": "ragged_csr_columnar",
            "t_col": t_col,
            "columns": {},
            "dtypes" : {},
            "files": {
                "offsets": "offsets.npy",
                "t": "t.npy",
                "t_index": "t_index.npy"
            },
            "required_cols": list(required_cols),
            "dir" : out_dir
        }

        for col in cols_to_write:
            safe = OrbitData.__safe_name(col)
            s : pd.Series = work[col]

            # Treat pandas "string/object" as string-like; also category
            is_stringish = (
                pd.api.types.is_object_dtype(s.dtype)
                or pd.api.types.is_string_dtype(s.dtype)
                or pd.api.types.is_categorical_dtype(s.dtype)
            )

            if is_stringish:
                # Convert to pandas strings; normalize NaNs
                s2 : pd.Series = s.astype("string")

                # Dictionary encode
                uniques = s2.dropna().unique()
                if string_max_unique is not None and len(uniques) > string_max_unique:
                    raise ValueError(f"Column '{col}' has {len(uniques)} unique strings; too many for dictionary encoding.")

                # Build mapping (stable)
                # code 0 reserved for NULL
                uniq_list = [str(x) for x in uniques]
                str_to_code = {v: i + 1 for i, v in enumerate(uniq_list)}
                codes = np.zeros(len(s2), dtype=np.int32)

                # Fill codes
                # (vectorized mapping via pandas map)
                mapped = s2.map(lambda x: str_to_code.get(str(x), 0) if pd.notna(x) else 0)
                codes[:] = mapped.to_numpy(dtype=np.int32, na_value=0)

                codes_path = f"col_{safe}__codes.npy"
                dict_path = f"dict_{safe}.json"
                np.save(os.path.join(out_dir, codes_path), codes)
                with open(os.path.join(out_dir, dict_path), "w") as f:
                    json.dump({"0": None, **{str(i + 1): v for i, v in enumerate(uniq_list)}}, f, indent=4)

                meta["columns"][safe] = {
                    "kind": "string_dict",
                    "col_name": col,
                    "dict_file": dict_path,
                }
                meta["files"][safe] = codes_path
                meta["dtypes"][safe] = str(codes.dtype)
            else:
                # Numeric/bool → store as array
                arr : np.ndarray = s.to_numpy()
                
                # Choose compact dtypes if you want (example: float64->float32)
                # arr = arr.astype(np.float32, copy=False) if arr.dtype == np.float64 else arr
                dtype = OrbitData.__pick_numpy_dtype(s)
                arr : np.ndarray = arr.astype(dtype, copy=False)

                col_path = f"col_{safe}.npy"
                np.save(os.path.join(out_dir, col_path), arr)
                meta["columns"][safe] = {
                    "kind": "numeric",
                    "col_name" : col,
                }
                meta["files"][safe] = col_path
                meta["dtypes"][safe] = str(arr.dtype)

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        return meta
    
    @staticmethod
    def __write_grid_table(dfs: List[pd.DataFrame],
                            bin_dir: str,
                            table_name: str,
                            *,                            
                            required_cols: Sequence[str] = ['lat [deg]','lon [deg]', 'grid index', 'GP index'],
                            dtype: np.dtype = np.float32,
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

        # initiate metadata 
        meta = {
            "name": table_name,
            "n": int(len(work)),
            "columns":{},
            "dtypes": {},
            "files": {},
            "dir" : out_dir
        }

        # extract and save required columns as .npy files
        for col in required_cols:
            # generate a safe name for this column to use in file names
            safe = OrbitData.__safe_name(col)

            # get appropriate dtype for this column
            col_dtype = OrbitData.__pick_numpy_dtype(work[col])

            # convert column to numpy array
            arr = work[col].to_numpy(dtype=col_dtype, copy=False)

            # define file path for this column
            col_path = f"col_{safe}.npy"

            # save to .npy file
            np.save(os.path.join(out_dir, col_path), arr)

            # update metadata with file and dtype info
            meta['columns'][safe] = col
            meta['files'][safe] = col_path
            meta['dtypes'][safe] = str(col_dtype)
        
        # return metadata
        return meta 
