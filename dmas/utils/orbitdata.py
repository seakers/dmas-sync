import copy
from enum import Enum
import json
from math import ceil
import os
import random
import re
import time
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

from orbitpy.mission import Mission
from execsatm.utils import Interval
from dmas.utils.series import TimeIndexedData,IntervalData

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
                 time_data : pd.DataFrame, 
                 eclipse_data : pd.DataFrame, 
                 position_data : pd.DataFrame, 
                 satellite_link_data : Dict[str, pd.DataFrame],
                 ground_operator_link_data : Dict[str, pd.DataFrame],
                 gs_access_data : pd.DataFrame, 
                 gp_access_data : pd.DataFrame, 
                 grid_data : list
                ):
        # name of agent being represented by this object
        self.agent_name = agent_name

        # name of ground station network the agent is associated with
        self.gs_network_name = gs_network_name

        # propagation time specifications
        self.time_step = time_data['time step']
        self.epoch_type = time_data['epoch type']
        self.epoch = time_data['epoch']
        self.duration = time_data['duration']

        # agent position and eclipse information
        self.eclipse_data : IntervalData = IntervalData.from_dataframe(eclipse_data, self.time_step, 'eclipse')
        self.position_data : TimeIndexedData = TimeIndexedData.from_dataframe(position_data, self.time_step, 'position')   

        # TODO validate access data
        assert all([('start index' in df.columns and 'end index' in df.columns) 
                    for df in satellite_link_data.values()]), \
                        'start index or end index column not found in satellite link data'
        
        # access times to other satellites
        self.satellite_links : Dict[str, IntervalData] = {satellite_name : IntervalData.from_dataframe(satellite_link_data[satellite_name], self.time_step, f"{satellite_name.lower()}-isl")
                                                   for satellite_name in satellite_link_data.keys()}

        # access times to ground operators
        self.ground_operator_links : Dict[str, IntervalData] = {network_name : IntervalData.from_dataframe(ground_operator_link_data[network_name], self.time_step, f'{network_name}-gsn_access') 
                                                                for network_name in ground_operator_link_data.keys()}
        
        # inter-agent link access times
        self.comms_links : Dict[str, IntervalData] = {agent_name : IntervalData.from_dataframe(satellite_link_data[agent_name], self.time_step, f"{agent_name.lower()}-comms_link")
                                                     for agent_name in satellite_link_data.keys()}
        self.comms_links.update({network_name : IntervalData.from_dataframe(ground_operator_link_data[network_name], self.time_step, f'{network_name}-comms_link')
                                for network_name in ground_operator_link_data.keys()})

        # access times to ground stations 
        self.ground_station_links : IntervalData = IntervalData.from_dataframe(gs_access_data, self.time_step, 'gs-access')

        # ground point access times
        self.gp_access_data : TimeIndexedData = TimeIndexedData.from_dataframe(gp_access_data, self.time_step, 'gp-access')

        # grid information
        assert isinstance(grid_data, list) and all([isinstance(df, pd.DataFrame) for df in grid_data]), 'grid data must be a list of pandas DataFrames'
        self.grid_data : list[pd.DataFrame] = grid_data
    
    # def get_epoc_in_datetime(self, delta_ut1=0.0) -> datetime:
    #     """
    #     Converts epoc to a datetime in UTC.
        
    #     Parameters
    #     ----------
    #     delta_ut1 : float, optional
    #         UT1-UTC offset in seconds (default 0.0, but usually provided by IERS).
        
    #     Returns
    #     -------
    #     datetime
    #         Corresponding UTC datetime.
    #     """
    #     # check epoc type 
    #     if self.epoc_type == self.JDUT1: # convert JDUT1 to datetime
    #         JD_UNIX_EPOCH = 2440587.5  # JD of 1970-01-01 00:00:00 UTC
    #         days_since_unix = self.epoc - JD_UNIX_EPOCH
    #         seconds_since_unix = days_since_unix * 86400.0 - delta_ut1  # adjust to UTC
    #         return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds_since_unix)

    #     else:
    #         raise NotImplementedError(f"Unsupported epoc type: {self.epoc_type}. Only 'JDUT1' is supported.")

    def copy(self) -> object:
        return OrbitData(self.agent_name, 
                         self.gs_network_name,
                         {'time step': self.time_step, 'epoc type' : self.epoc_type, 'epoc' : self.epoc},
                         self.eclipse_data,
                         self.position_data,
                         self.satellite_links,
                         self.ground_station_links,
                         self.gp_access_data,
                         self.grid_data
                         )
    
    def update_databases(self, t : float) -> None:
        # exclude outdated data
        self.eclipse_data.update_expired_values(t)
        self.position_data.update_expired_values(t)
        for _, comms_link in self.comms_links.items(): comms_link.update_expired_values(t)
        for _, isl in self.satellite_links.items(): isl.update_expired_values(t) 
        self.ground_station_links.update_expired_values(t)
        self.gp_access_data.update_expired_values(t)

    """
    GET NEXT methods
    """
    def get_next_agent_access(self, target : str, t: float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to another agent or ground station after or during time `t` up to a given time `t_max`. """

        # check if target is within the list of known agents
        assert target in self.comms_links.keys(), f'No comms data found for target agent `{target}`.'

        # return next access interval
        return self.__get_next_interval(self.comms_links[target], t, t_max, include_current)

    def get_next_agent_accesses(self, target : str, t: float, t_max: float = np.Inf, include_current: bool = False) -> List[Interval]:
        """ returns a list of the next access interval to another agent or ground station after or during time `t` up to a given time `t_max`. """

        # check if target is within the list of known agents
        assert target in self.comms_links.keys(), f'No comms data found for target agent `{target}`.'

        # get next access intervals
        future_access_intervals = self.__get_next_intervals(self.comms_links[target], t, t_max, include_current)

        # return in interval form
        return [Interval(t_start,t_end) for t_start,t_end in future_access_intervals] if future_access_intervals else []

    def get_next_isl_access_interval(self, target : str, t : float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to another agent after or during time `t`. """

        # check if target is within the list of known satellite agents
        assert target in self.satellite_links.keys(), f'No ISL data found for target agent `{target}`.'

        # return next access interval
        return self.__get_next_interval(self.satellite_links[target], t, t_max, include_current)

    def get_next_ground_operator_access_interval(self, target : str, t : float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to a ground operator agent after or during time `t`. """

        # check if target is within the list of known ground operator agents
        assert target in self.ground_operator_links.keys(), f'No ground operator data found for target agent `{target}`.'

        # return next access interval
        return self.__get_next_interval(self.ground_operator_links[target], t, t_max, include_current)

    def get_next_gs_access(self, t, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval to a ground station after or during time `t`. """
        return self.__get_next_interval(self.ground_station_links, t, t_max, include_current)

    def get_next_eclipse_interval(self, t: float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next eclipse interval after or during time `t`. """
        return self.__get_next_interval(self.eclipse_data, t, t_max, include_current)

    def __get_next_interval(self, interval_data : IntervalData, t : float, t_max: float = np.Inf, include_current: bool = False) -> Interval:
        """ returns the next access interval from `interval_data` after or during time `t`. """
        # get next intervals
        future_intervals : list[Interval] = interval_data.lookup_intervals(t, t_max)

        # check if current interval should be included
        if not include_current:
            # exclude intervals that contain time `t`
            future_intervals = [interval for interval in future_intervals
                                if t < interval.left] # interval starts after time `t`
        else:
            # include current intervals but clip to start at time `t`
            future_intervals = [Interval(max(t, interval.left), interval.right) if interval.left <= t <= interval.right else interval
                                for interval in future_intervals]

        # check if there are any valid intervals
        if not future_intervals: return None

        # get next interval
        next_interval = future_intervals[0]

        # return the first interval that starts after or at time `t`
        return Interval(max(t, next_interval.left), min(next_interval.right, t_max))
        
        # TEMP previous implementation
        # # get next intervals
        # future_intervals: list[tuple[float, float]] = self.__get_next_intervals(interval_data, t, t_max, include_current)

        # # check if there are any valid intervals
        # if not future_intervals: return None

        # # get interval bounds
        # t_start,t_end = future_intervals[0]

        # # return the first interval that starts after or at time `t`
        # return Interval(max(t, t_start), min(t_end, t_max))

    def __get_next_intervals(self, interval_data : IntervalData, t : float, t_max: float = np.Inf, include_current: bool = False) -> List[Tuple[float, float]]:
        # find all intervals that end after time `t` and start before time `t_max`
        future_intervals : List[Interval] = interval_data.lookup_intervals(t, t_max)
        
        # convert to tuple form
        future_interval_pairs = [(interval.left, interval.right) for interval in future_intervals]
        
        # check if current interval should be included
        if not include_current:
            # exclude intervals that contain time `t`
            future_interval_pairs = [(t_start, t_end) for t_start,t_end in future_interval_pairs
                                if t < t_start] # interval starts after time `t`
        else:
            # include current intervals but clip to start at time `t`
            future_interval_pairs = [(max(t, t_start), t_end) for t_start,t_end in future_interval_pairs]

        # check if there are any valid intervals
        if not future_interval_pairs: return None
        
        # sort by start time and return
        return sorted(future_interval_pairs, key=lambda interval: interval[0])
    
    def get_latest_agent_access(self, target : str, t: float, t_max: float, include_current: bool = False) -> Interval:
        """ returns the latest access interval to another agent or ground station after or during time `t`. """

        # check if target is within the list of known agents
        assert target in self.comms_links.keys(), f'No comms data found for target agent `{target}`.'

        # return next access interval
        return self.__get_latest_interval(self.comms_links[target], t, t_max, include_current)

    def __get_latest_interval(self, interval_data : IntervalData, t : float, t_max: float, include_current: bool = False) -> Interval:
        """ returns the latest access interval from `interval_data` after or during time `t`. """
        # find all intervals that end after time `t` and start before time `t_max`
        future_intervals: list[tuple[float, float]] = [(t_start, t_end)
                                                        for t_start,t_end,*_ in interval_data.data
                                                        if t <= t_end # interval ends after or at time `t`
                                                        and t_start <= t_max # interval starts before or at time `t_max`
                                                        ]
        
        # check if current interval should be included
        if not include_current:
            # exclude intervals that contain time `t`
            future_intervals = [(t_start, t_end) for t_start,t_end in future_intervals
                                if t < t_start] # interval starts after time `t`

        # check if there are any valid intervals
        if not future_intervals: return None
        
        # sort by start time
        future_intervals.sort(key=lambda interval: interval[0])

        # get interval bounds
        t_start,t_end = future_intervals[-1]

        # return the last interval that starts after or at time `t`
        return Interval(max(t, t_start), min(t_end, t_max))

    def get_next_gp_access_interval(self, lat: float, lon: float, t: float, t_max : float = np.Inf) -> Interval:
        """
        Returns the next access to a ground point
        """
        # TODO
        raise NotImplementedError('TODO: need to implement.')

    """
    STATE QUERY methods
    """
    def is_accessing_agent(self, target: str, t: float) -> bool:
        """ checks if a satellite is currently accessing another agent at time `t`. """
        # check if the target is the agent itself
        if target in self.agent_name: return True

        # check if target is within the list of known agents
        if target not in self.comms_links.keys(): return False

        # get next access interval
        next_access = self.comms_links[target].lookup(t)

        # check if there is a current access interval
        return next_access is not None

    def is_accessing_ground_station(self, target : str, t: float) -> bool:
        raise NotImplementedError('TODO: implement ground station access check.')
        # t = t/self.time_step
        # nrows, _ = self.gs_access_data.query('`start index` <= @t & @t <= `end index` & `gndStn name` == @target').shape
        # return bool(nrows > 0)

    def is_eclipse(self, t: float):
        """ checks if a satellite is currently in eclise at time `t`. """
        return self.eclipse_data.is_active(t)

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
        position_data = self.position_data.lookup_value(t)
        
        if not position_data:
            raise ValueError(f'No position data found for time {t} [s].')

        # unpack position and velocity data
        pos = [position_data['x [km]'], position_data['y [km]'], position_data['z [km]']]
        vel = [position_data['vx [km/s]'], position_data['vy [km/s]'], position_data['vz [km/s]']]
        
        return (pos, vel, is_eclipse)

    """
    LOAD FROM PRE-COMPUTED DATA
    """
    def from_directory(orbitdata_dir: str, simulation_duration : float) -> Dict[str, 'OrbitData']:
        """
        Loads orbit data from a directory containig a json file specifying the details of the mission being simulated.
        If the data has not been previously propagated, it will do so and store it in the same directory as the json file
        being used.

        The data gets stored as a dictionary, with each entry containing the orbit data of each agent in the mission 
        indexed by the name of the agent.
        """
        orbitdata_specs : str = os.path.join(orbitdata_dir, 'MissionSpecs.json')
        with open(orbitdata_specs, 'r') as scenario_specs:
            
            # load json file as dictionary
            mission_dict : dict = json.load(scenario_specs)
            data = dict()
            spacecraft_list : List[dict] = mission_dict.get('spacecraft', None)
            uav_list : List[dict] = mission_dict.get('uav', None)
            ground_ops_list : List[dict] = mission_dict.get('groundOperator', None)

            # compile list of all agents to load
            agents_to_load : List[dict] = []
            if spacecraft_list: agents_to_load.extend(spacecraft_list)
            if uav_list: agents_to_load.extend(uav_list)
            if ground_ops_list: agents_to_load.extend(ground_ops_list)

            # load pre-computed data for each agent
            for agent in tqdm(agents_to_load, desc='Loading and verifying orbit data', unit='agent', leave=False):
                agent_name = agent.get('name')
                data[agent_name] = OrbitData.load(orbitdata_dir, agent_name, simulation_duration)
            
            # return compiled data
            return data

    def load(orbitdata_path : str, agent_name : str, simulation_duration : float) -> object:
        """
        Loads agent orbit data from pre-computed csv files in scenario directory
        """
        with open(os.path.join(orbitdata_path, 'MissionSpecs.json'), 'r') as mission_specs:
            # load mission specifications json file as dictionary
            mission_dict : dict = json.load(mission_specs)
            
            # get spacecraft and ground station specifications
            spacecraft_list : List[dict] = mission_dict.get('spacecraft', None)
            ground_station_list : List[dict] = mission_dict.get('groundStation', [])
            ground_ops_list : List[dict] = mission_dict.get('groundOperator', [])
            
            # load orbit data for the specified agent
            if agent_name in [spacecraft.get('name') for spacecraft in spacecraft_list]:
                return OrbitData.load_spacecraft_data(agent_name, spacecraft_list, ground_station_list, ground_ops_list, orbitdata_path, mission_dict, simulation_duration)
            elif agent_name in [gstat.get('name') for gstat in ground_ops_list]:
                return OrbitData.load_gstat_data(agent_name, spacecraft_list, ground_station_list, ground_ops_list, orbitdata_path, mission_dict, simulation_duration)
            else:
                raise ValueError(f'Orbitdata for agent `{agent_name}` not found in precomputed data.')
    
    @staticmethod
    def load_spacecraft_data(
                             agent_name : str, 
                             spacecraft_list : List[dict], 
                             ground_station_list: List[dict],
                             ground_ops_list : List[dict],
                             orbitdata_path : str,
                             mission_dict : dict,   
                             simulation_duration : float
                             ) -> object:
        """
        Loads orbit data for a spacecraft from pre-computed csv files in scenario directory
        """

        # get scenario settings
        scenario_dict : dict = mission_dict.get('scenario', None)

        # get connectivity setting
        connectivity : str = scenario_dict.get('connectivity', None) \
            if scenario_dict else ConnectivityLevels.LOS.value # default to LOS if not specified

        # find the desired spacecraft specifications in the mission dictionary
        for spacecraft_idx,spacecraft in enumerate(spacecraft_list):
            # get spacecraft name
            name = spacecraft.get('name')

            # check if this is the desired spacecraft
            if name != agent_name: continue

            # define agent folder
            sat_id = "sat" + str(spacecraft_idx)
            agent_folder = sat_id + '/'

            # load eclipse data
            eclipse_file = os.path.join(orbitdata_path, agent_folder, "eclipses.csv")
            eclipse_data = pd.read_csv(eclipse_file, skiprows=range(3))
            
            # load position data
            position_file = os.path.join(orbitdata_path, agent_folder, "state_cartesian.csv")
            position_data = pd.read_csv(position_file, skiprows=range(4))

            # load propagation time data
            time_data =  pd.read_csv(position_file, nrows=3)
            _, epoch_type, _, epoch = time_data.at[0,time_data.axes[1][0]].split(' ')
            epoch_type = epoch_type[1 : -1]
            epoch = float(epoch)
            _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
            time_step = float(time_step)
            _, _, _, _, prop_duration = time_data.at[2,time_data.axes[1][0]].split(' ')
            prop_duration = float(prop_duration)

            assert simulation_duration <= prop_duration, \
                f'Simulation duration ({simulation_duration} days) exceeds pre-computed propagation duration ({prop_duration} days) for spacecraft `{agent_name}`.'

            time_data = { "epoch": epoch, 
                        "epoch type": epoch_type, 
                        "time step": time_step,
                        "duration" : simulation_duration }

            # limit position and eclipse data to simulation duration
            max_time_index = int(simulation_duration * 24 * 3600 // time_step)
            position_data = position_data[:max_time_index+1]
            eclipse_data = eclipse_data[:max_time_index+1]

            # load inter-satellite link data
            isl_data = dict()
            comms_path = os.path.join(orbitdata_path, 'comm')
            for file in tqdm(os.listdir(comms_path), desc=f'Loading ISL data for {agent_name}', unit='file', leave=False):                
                # remove file extension and split sender and receiver
                isl = re.sub(".csv", "", file)
                sender, _, receiver = isl.split('_')
                
                # generate ISL access file path
                isl_file = os.path.join(comms_path, file)

                # load ISL data depending on whether the current spacecraft is the sender or receiver
                if sat_id == sender:
                    # sat_id in sender
                    receiver_index = int(re.sub("[^0-9]", "", receiver))
                    receiver_name = spacecraft_list[receiver_index].get('name')

                    # load ISL data
                    isl_data[receiver_name] = OrbitData.load_isl_data(isl_file, connectivity, simulation_duration, time_step)
                    
                elif sat_id == receiver:
                    # sat_id in receiver
                    sender_index = int(re.sub("[^0-9]", "", sender))
                    sender_name = spacecraft_list[sender_index].get('name')
                    
                    # load ISL data
                    isl_data[sender_name] = OrbitData.load_isl_data(isl_file, connectivity, simulation_duration, time_step)

            # compile list of ground stations that are part of the desired network
            gs_network_name = spacecraft.get('groundStationNetwork', None)
            gs_network_station_ids : List[str] = [ gs['@id'] for gs in ground_station_list
                                                   if gs.get('networkName', None) == gs_network_name ]

            # load ground station access data
            gs_access_data = pd.DataFrame(columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]'])
            agent_orbitdata_path = os.path.join(orbitdata_path, agent_folder)
            for file in tqdm(os.listdir(agent_orbitdata_path), desc=f'Loading ground station access data for {agent_name}', unit='file', leave=False):
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
                    = OrbitData.load_gstat_comms_data(gndStn_access_file, connectivity, simulation_duration, time_step)

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

            # load agent access to ground operator if one exists
            ground_operator_link_data : Dict[str,pd.DataFrame] = {ground_operator.get('name'): pd.DataFrame(columns=['start index', 'end index'])
                                                                for ground_operator in ground_ops_list}
            if gs_network_name: 
                ground_operator_link_data[gs_network_name] = pd.concat([ground_operator_link_data[gs_network_name], gs_access_data])
                for col in ground_operator_link_data[gs_network_name].columns:
                    if col not in ['start index', 'end index']:
                        ground_operator_link_data[gs_network_name].drop(columns=[col], inplace=True)
                ground_operator_link_data[gs_network_name] = ground_operator_link_data[gs_network_name].drop_duplicates().reset_index(drop=True)

            # land coverage data metrics data
            payload = spacecraft.get('instrument', None)
            if not isinstance(payload, list):
                payload = [payload]

            gp_access_data = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])

            for instrument in tqdm(payload, desc=f'Loading land coverage data for {agent_name}', unit='instrument', leave=False):
                if instrument is None: continue 

                i_ins = payload.index(instrument)
                gp_acces_by_mode = []

                # TODO implement different viewing modes for payloads
                # modes = spacecraft.get('instrument', None)
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
            gp_access_data['agent name'] = [spacecraft['name']] * nrows

            # limit gp access data to simulation duration
            gp_access_data = gp_access_data[gp_access_data['time index'] <= max_time_index]

            grid_data_compiled = []
            for grid in tqdm(mission_dict.get('grid'), desc=f'Loading grid data for {agent_name}', unit='grid', leave=False):
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

            return OrbitData(name, gs_network_name, time_data, eclipse_data, position_data, isl_data, ground_operator_link_data, gs_access_data, gp_access_data, grid_data_compiled)
        
        raise ValueError(f'Orbitdata for satellite `{agent_name}` not found in precalculated data.')
    
    @staticmethod
    def load_isl_data(isl_file : str, connectivity : str, duration_days : float, time_step : float) -> pd.DataFrame:
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
    def load_gstat_comms_data(gndStn_access_file : str, connectivity : str, simulation_duration : float, time_step : float) -> pd.DataFrame:
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
            gndStn_access_data = pd.read_csv(gndStn_access_file, skiprows=range(3))

            # limit ground station access data to simulation duration
            max_time_index = int(simulation_duration * 24 * 3600 // time_step)
            gndStn_access_data = gndStn_access_data[gndStn_access_data['start index'] <= max_time_index]
            gndStn_access_data.loc[gndStn_access_data['end index'] > max_time_index, 'end index'] = max_time_index
        
        elif connectivity.upper() == ConnectivityLevels.GS.value:
            # ground station-only connectivity; load ground station access data
            gndStn_access_data = pd.read_csv(gndStn_access_file, skiprows=range(3))

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
    def load_gstat_data(
                             agent_name : str, 
                             spacecraft_list : List[dict], 
                             ground_station_list: List[dict],
                             ground_ops_list: List[dict],
                             orbitdata_path : str,
                             mission_dict : dict,
                             simulation_duration : float                             
                             ) -> object:
        
        
        # get scenario settings
        scenario_dict : dict = mission_dict.get('scenario', None)

        # get connectivity setting
        connectivity : str = scenario_dict.get('connectivity', None) \
            if scenario_dict else ConnectivityLevels.LOS.value # default to LOS if not specified
        
        # find the desired ground operator specifications in the mission dictionary
        for ground_ops in ground_ops_list:
             # get spacecraft name
            name = ground_ops.get('name')

            # check if this is the desired spacecraft
            if name != agent_name: continue

            # compile list of relevant ground stations for this ground operator
            gs_network_station_indices : List[str] = [ idx for idx,gs in enumerate(ground_station_list)
                                                   if gs.get('networkName', None) == name ]

            # validate that a network name and a list of ground stations is assigned
            assert gs_network_station_indices, f'No ground station network found for ground operator `{agent_name}`.'

            # compile time data
            time_data = None
            for sat_idx, spacecraft in enumerate(spacecraft_list):
                # spacecraft is not part of the ground station network; skip
                if spacecraft.get('groundStationNetwork', None) != name: continue

                # load access time for this spacecraft with each ground_station in the network
                for file in os.listdir(os.path.join(orbitdata_path,"sat" + str(sat_idx))):
                    if 'state' not in file: continue

                    # load propagation time data
                    agent_access_df = os.path.join(orbitdata_path, "sat" + str(sat_idx), file)
                    time_data =  pd.read_csv(agent_access_df, nrows=3)
                    _, epoch_type, _, epoch = time_data.at[0,time_data.axes[1][0]].split(' ')
                    epoch_type = epoch_type[1 : -1]
                    epoch = float(epoch)
                    _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
                    time_step = float(time_step)
                    _, _, _, _, prop_duration = time_data.at[2,time_data.axes[1][0]].split(' ')
                    prop_duration = float(prop_duration)

                    assert simulation_duration <= prop_duration, \
                        f'Simulation duration ({simulation_duration} days) exceeds pre-computed propagation duration ({prop_duration} days) for spacecraft `{agent_name}`.'

                    time_data = { "epoch": epoch, 
                                "epoch type": epoch_type, 
                                "time step": time_step,
                                "duration" : simulation_duration }

                    break # only need to load time data once

                if time_data is not None: break # only need to load time data once

            # ensure time data was found
            assert time_data is not None, \
                f'No propagation data found for any spacecraft in ground station network `{agent_name}`.'

            # calculate number of propagation steps
            n_steps = int(ceil(time_data.get('duration') * 24 * 3600 / time_data.get('time step')))

            # load eclipse data
            # TODO implement eclipse data for ground stations, currently empty
            eclipse_data = pd.DataFrame(columns=['start index', 'end index'])

            # calculate position data
            # TODO implement position data for ground stations, currently empty.
            # Since ground operators consist of a network of ground stations, and the ground stations are static, 
            # there is no single position data to represent the entire network.
            position_data = pd.DataFrame(columns=['time index','x [km]','y [km]','z [km]','vx [km/s]','vy [km/s]','vz [km/s]'])

            # compile satellite link interval data
            satellite_link_data : Dict[str, pd.DataFrame] = dict()
            for sat_idx, spacecraft in enumerate(spacecraft_list):
                # spacecraft is not part of the ground station network; skip
                if spacecraft.get('groundStationNetwork', None) != name: continue

                # initiate access data for this spacecraft
                satellite_access_data = pd.DataFrame(columns=['start index', 'end index'])

                # load access time for this spacecraft with each ground_station in the network
                for file in os.listdir(os.path.join(orbitdata_path,"sat" + str(sat_idx))):
                    if 'gndStn' not in file: continue
                    
                    # get ground station index from file name
                    gndStn_idx = int(re.sub("[^0-9]", "", file))

                    # check if ground station is part of the desired network
                    if gndStn_idx not in gs_network_station_indices: continue 

                    # load ground station access data
                    # agent_access_file = os.path.join(orbitdata_path, "sat" + str(sat_idx), file)
                    if connectivity.upper() == ConnectivityLevels.FULL.value:
                        # fully connected network; modify connectivity 
                        columns = ['start index', 'end index']
                        
                        # generate mission-long connectivity access                    
                        data = [[0.0, simulation_duration * 24 * 3600 // time_step + 1]]  # full connectivity from start to end of mission
                        assert data[0][1] > 0.0

                        # return modified connectivity
                        agent_access_df = pd.DataFrame(data=data, columns=columns)
                    
                    elif connectivity.upper() == ConnectivityLevels.LOS.value:
                        # line-of-sight driven connectivity; load ground station access data
                        agent_access_file = os.path.join(orbitdata_path, "sat" + str(sat_idx), file)
                        agent_access_df = pd.read_csv(agent_access_file, skiprows=range(3))

                        # limit ground station access data to simulation duration
                        max_time_index = int(simulation_duration * 24 * 3600 // time_step)
                        agent_access_df = agent_access_df[agent_access_df['start index'] <= max_time_index]
                        agent_access_df.loc[agent_access_df['end index'] > max_time_index, 'end index'] = max_time_index
                    
                    elif connectivity.upper() == ConnectivityLevels.GS.value:
                        # ground station-only connectivity; load ground station access data
                        agent_access_file = os.path.join(orbitdata_path, "sat" + str(sat_idx), file)
                        agent_access_df = pd.read_csv(agent_access_file, skiprows=range(3))

                        # limit ground station access data to simulation duration
                        max_time_index = int(simulation_duration * 24 * 3600 // time_step)  
                        agent_access_df = agent_access_df[agent_access_df['start index'] <= max_time_index]
                        agent_access_df.loc[agent_access_df['end index'] > max_time_index, 'end index'] = max_time_index

                    elif connectivity.upper() == ConnectivityLevels.ISL.value:
                        # inter-satellite link-driven connectivity; create empty dataframe
                        columns = ['start index', 'end index']
                        agent_access_df = pd.DataFrame(data=[], columns=columns)

                    elif connectivity.upper() == ConnectivityLevels.NONE.value:
                        # no inter-agent connectivity; create empty dataframe
                        columns = ['start index', 'end index']
                        agent_access_df = pd.DataFrame(data=[], columns=columns)

                    else:
                        # fallback; unsupported connectivity level
                        raise ValueError(f'Unsupported connectivity level: {connectivity}.')

                    satellite_access_data = pd.concat([satellite_access_data, agent_access_df])

                # sort by start index and remove duplicates
                satellite_access_data = satellite_access_data.sort_values(by='start index').drop_duplicates().reset_index(drop=True)

                # merge any overlapping intervals
                satellite_access_data = OrbitData.merge_overlapping_intervals(satellite_access_data, merge_touching=True)
                
                if satellite_access_data.empty:
                    # no access during simulation duration; create empty dataframe
                    columns = ['start index', 'end index']
                    satellite_access_data = pd.DataFrame(data=[], columns=columns)
                
                # save access data for this spacecraft
                satellite_link_data[spacecraft.get('name')] = satellite_access_data
            
            # compile ground operator link interval data; assume constant access to all other ground operators
            ground_operator_link_data : Dict[str,pd.DataFrame] = {ground_operator.get('name'): pd.DataFrame(columns=['start index', 'end index'], data=[(0, n_steps-1)])
                                                                  for ground_operator in ground_ops_list
                                                                  if ground_operator.get('name') != name}

            # load ground station access data
            columns=['start index', 'end index', 'gndStn id', 'gndStn name','lat [deg]','lon [deg]']
            data = [(0, n_steps, ground_station.get('@id'), ground_station.get('name'), ground_station.get('latitude'), ground_station.get('longitude'))
                    for ground_station in ground_station_list
                    if ground_station.get('networkName', None) == name]
            gs_access_data = pd.DataFrame(data=data, columns=columns)

            # Ground Operators have no sensing capability; create empty ground point coverage data
            gp_access_data = pd.DataFrame(columns=['time index','GP index','pnt-opt index','lat [deg]','lon [deg]', 'agent','instrument',
                                                            'observation range [km]','look angle [deg]','incidence angle [deg]','solar zenith [deg]'])
        
            # compile coverage grid information
            grid_data_compiled = []
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

            return OrbitData(agent_name, agent_name, time_data, eclipse_data, position_data, satellite_link_data, ground_operator_link_data, gs_access_data, gp_access_data, grid_data_compiled)
                
        raise ValueError(f'Orbitdata for satellite `{agent_name}` not found in precalculated data.')
    
    @staticmethod
    def merge_overlapping_intervals(df: pd.DataFrame, merge_touching: bool = True) -> pd.DataFrame:
        # extract relevant columns and sort by start index
        d = df[["start index", "end index"]].copy()
        d = d.sort_values("start index", kind="mergesort").reset_index(drop=True)

        # early exit if empty
        if d.empty: return d

        if merge_touching:
            # treat [a,b] and [b,c] as overlapping
            new_group = d["start index"].gt(d["end index"].cummax().shift(fill_value=d.loc[0, "end index"]))
        else:
            # touching is NOT overlapping: require strictly greater than previous max end
            new_group = d["start index"].ge(d["end index"].cummax().shift(fill_value=d.loc[0, "end index"]))

        grp = new_group.cumsum()

        out = d.groupby(grp, as_index=False).agg(start=("start index", "min"), end=("end index", "max"))
        
        # rename columns to original names
        out.rename(columns={"start": "start index", "end": "end index"}, inplace=True)

        # return 
        return out

    def precompute(scenario_specs : dict, overwrite : bool = False, printouts: bool = True) -> str:
        """
        Pre-calculates coverage and position data for a given scenario
        """
        
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
            if printouts:
                tqdm.write(' - Existing orbit data found and matches scenario. Loading existing data...')
        else:
            # if propagation data files do not exist, propagate and then load results
            if os.path.exists(data_dir):
                if printouts:
                    tqdm.write(' - Existing orbit data does not match scenario.')
            else:
                if printouts:
                    tqdm.write(' - Orbit data not found.')

            # clear files if they exist
            if printouts:
                tqdm.write(' - Clearing \'orbitdata\' directory...')    
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
            print("Propagating orbits...")
            mission : Mission = Mission.from_json(scenario_specs)  
            mission.execute()                
            print("Propagation done!")

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
    COVERAGE Metrics
    """
    def calculate_percent_coverage(self) -> float:
        n_observed = 0
        n_points = sum([len(grid) for grid in self.grid_data])
        t_0 = time.perf_counter()

        # for grid in self.grid_data:
        #     grid : pd.DataFrame

        #     for lat,lon,grid_index,gp_index in grid.values:

        #         accesses = [(t_img,lat,lon,lat_img,lon_img)
        #                     for t_img, gp_index_img, _, lat_img, lon_img, _, _, _, _, grid_index_img, *_ 
        #                     in self.gp_access_data.values
        #                     if abs(lat - lat_img) < 1e-3
        #                     and abs(lon - lon_img) < 1e-3
        #                     and gp_index == gp_index_img 
        #                     and grid_index == grid_index_img]
                
        #         if accesses:
        #             n_observed += 1

        gp_observed = { (lat,lon) 
                        for grid in self.grid_data
                        for lat,lon,grid_index,gp_index in grid.values
                        for _, gp_index_img, _, _, _, _, _, _, _, grid_index_img, *_ in self.gp_access_data.values
                        if gp_index == gp_index_img 
                        and grid_index == grid_index_img}
        n_observed = len(gp_observed)
        
        dt = time.perf_counter() - t_0        
        
        return n_points, n_observed, float(n_observed) / float(n_points)

"""
TESTING
"""
if __name__ == '__main__':
    scenario_dir = './scenarios/sim_test/'
    
    orbit_data_list = OrbitData.from_directory(scenario_dir)

    # expected val: (grid, point) = 0, 0
    for agent in orbit_data_list:
        lat = 1.0
        lon = 158.0
        t = 210.5

        grid, point, gp_lat, gp_lon = orbit_data_list[agent].find_gp_index(lat, lon)
        print(f'({lat}, {lon}) = G{grid}, P{point}, Lat{gp_lat}, Lon{gp_lon}')

        print(orbit_data_list[agent].is_accessing_ground_point(lat, lon, t))
        break