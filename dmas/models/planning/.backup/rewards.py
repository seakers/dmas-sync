from typing import  Callable

import numpy as np
import pandas as pd

from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker

from chess3d.agents.actions import ObservationAction
from chess3d.agents.science.requests import *

class GridPoint(object):
    """ Describes the reward of performing an observation of a given ground point """
    def __init__(self, 
                 instrument : str,
                 lat : float, 
                 lon : float, 
                 grid_index : int,
                 gp_index : int,
                 initial_reward : float,                  
                 alt : float = 0.0, 
                 observations : list = [],
                 events : list = [],
                 t_update : float = np.NAN
                 ) -> None:
        # set fixed parameters
        self.instrument : str = instrument
        self.target : list[float] = [lat, lon, alt]
        self.grid_index : int = grid_index
        self.gp_index : int = gp_index
        self.initial_reward : float = initial_reward

        # set variable parameters
        self.observations : list[ObservationAction] = [observation for observation in observations]
        self.events : set[GeophysicalEvent] = {event for event in events}
        self.reward : float = initial_reward
        self.t_update : float = t_update
        self.history : list = []

    def update_observations(self, observation_datum : dict, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)

        # chekc if there are any prior observations
        if self.observations:
            # check if observation is more recent than the last one
            if observation_datum['t'] > self.observations[-1]['t']:
                # check if an active event exists
                if any(event.is_active(observation_datum['t']) for event in self.events):
                    # add latest observation to the list
                    self.observations.append(observation_datum)
                else:
                    # update to latest observation
                    self.observations[-1] = observation_datum
        else:
            # add first observation
            self.observations.append(observation_datum)
        
        # update latest update time
        self.t_update = t_update

    def update_events(self, event : GeophysicalEvent, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)
        self.events.add(event)
        self.t_update = t_update

    def update_reward(self, reward : float, t_update : float) -> None:
        assert self.is_update_time_valid(t_update)
        
        # update reward
        self.reward = reward

        # update latest update time
        self.t_update = t_update

        # add to reward history
        lat,lon,_ = self.target
        t_update = t_update if not np.isnan(t_update) else -1
        self.history.append((t_update, self.grid_index, self.gp_index, lat, lon, self.instrument, reward, len(self.observations), len(self.events)))

    def is_update_time_valid(self, t_update : float) -> bool:
        return t_update >= self.t_update or np.isnan(self.t_update)
    
    def reset(self) -> None:
        self.observations = set()
        self.events = set()
        self.reward = self.initial_reward
        self.t_update = np.NAN

    def to_dict(self) -> dict:
        return self.__dict__
    
    def __str__(self) -> str:
        out = 't_update,grid_index,GP index,instrument,reward,n_observations,n_events\n'
        for t_update, grid_index, gp_index, instrument, reward, n_observations, n_events in self.history:
            out += f'{t_update},{grid_index},{gp_index},{instrument},{reward},{n_observations},{n_events}\n'
        
        return out
    
    def __repr__(self) -> str:
        return f'GridPoint_{self.grid_index}_{self.gp_index}_{self.instrument}'

class RewardGrid(object):
    def __init__(self, 
                 reward_function : Callable,
                 specs : object, 
                 grid_data : list, 
                 initial_reward : float,
                 **grid_params : dict,              
                 ) -> None:       
        
        # save reward function
        self.reward_function = reward_function

        # load grid data
        self.grid_data : list[pd.DataFrame] = grid_data
        
        # remove unwanted columns in grid data if any
        for grid in self.grid_data:
            grid : pd.DataFrame
            _ = [grid.pop(unwanted_column) for unwanted_column in grid.columns 
                 if unwanted_column not in ['lat [deg]', 'lon [deg]', 'grid index', 'GP index']]
            # for unwanted_column in unwanted_columns: grid.pop(unwanted_column)

        # ensure no repeats on grid indeces
        assert all([len([int(gp_index) for *_,gp_index in grid_datum.values]) \
                    == len({int(gp_index) for *_,gp_index in grid_datum.values})
                   for grid_datum in self.grid_data])
        
        # save additional parameters
        self.specs = specs
        self.initial_reward = initial_reward
        self.grid_params : dict = grid_params

        self.stats = {}

        # initiate reward grid vectors
        if isinstance(specs, Spacecraft):
            self.rewards = [ [{instrument.name : GridPoint(instrument.name, 
                                                           lat, 
                                                           lon, 
                                                           int(grid_index), 
                                                           int(gp_index),
                                                           initial_reward) 
                               for instrument in specs.instrument} 
                              for lat,lon,*_,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        elif isinstance(specs, dict):
            self.rewards = [ [{instrument['name'] : GridPoint(instrument['name'], 
                                                              lat, 
                                                              lon, 
                                                              int(grid_index), 
                                                              int(gp_index),
                                                              initial_reward)  
                               for instrument in specs['payload']} 
                            for lat,lon,grid_index,gp_index in grid_datum.values] 
                            for grid_datum in self.grid_data]
        else:
            raise ValueError(f'`specs` of type {type(specs)} not supported.')
        
        # create coordinates to indeces map
        self.n_decimals = self.count_decimals(grid_data)
        self.coordinate_map = {(round(lat, self.n_decimals),round(lon, self.n_decimals)) : (int(grid_index),int(gp_index))
                               for grid_datum in self.grid_data
                               for lat,lon,*_,grid_index,gp_index in grid_datum.values}

        # update previous observations and events (if they exist)
        prev_observations = grid_params.get('prev_observations', [])
        prev_events = grid_params.get('prev_events', [])
        self.update(np.NAN, prev_observations, prev_events)

    def count_decimals(self, grid_data : list) -> None:
        decimals = []
        for grid_datum in grid_data:
            grid_datum : pd.DataFrame
            for lat,lon,*_ in grid_datum.values:
                lat_decimals =  len(str(lat).split(".")[1])
                lon_decimals =  len(str(lon).split(".")[1])
                decimals.append(max(lat_decimals,lon_decimals))
        
        return min(max(decimals), 3)

    def get_target_data(self, lat : float, lon : float) -> dict:
        # get appropriate grid point object
        grid_index,gp_index = self.__get_target_indeces(lat,lon)

        # get corresponding grid point
        return self.rewards[grid_index][gp_index]
        

    @runtime_tracker
    def __get_target_indeces(self, lat : float, lon : float) -> tuple:
        # look for indeces in look-up table
        lat_round,lon_round = round(lat,self.n_decimals), round(lon,self.n_decimals)
        if (lat_round,lon_round) in self.coordinate_map:
            # values found; return indeces
            return self.coordinate_map[(lat_round, lon_round)]
        
        # search ground points with matching latitude and longitude
        matches = [(int(grid_index),int(gp_index))
                   for grid_datum in self.grid_data
                   for gp_lat,gp_lon,grid_index,gp_index in grid_datum.values
                   if abs(gp_lat - lat) <= 10**-self.n_decimals
                   and abs(gp_lon - lon) <= 10**-self.n_decimals
                   ]
        
        if matches:
            # values found; return incedes
            return matches.pop() 
        else:
            # values not found; raise exception
            for grid_datum in self.grid_data:
                for gp_lat,gp_lon,grid_index,gp_index in grid_datum.values:
                    if abs(gp_lat - lat) <= 1e-1 and abs(gp_lon - lon) <= 1e-1:
                        x = (grid_index,gp_index)
            raise ValueError(f'Could not find ground point indeces for coordinates ({lat}°,{lon}°,0.0)')
        
    @runtime_tracker
    def reset(self) -> None:
        for grid_rewards in self.rewards:
            for gp_rewards in grid_rewards:
                for _, grid_point in gp_rewards.items():
                    grid_point.reset()

    @runtime_tracker
    def update(self, 
               t : float, 
               observations : list = [], 
               events : list = []) -> None:
        # update events
        for event in events: self.update_event(event, t)

        # update observations
        for instrument,observation_data in observations: 
            for observation_datum in observation_data:
                self.update_observation(instrument, observation_datum, t)

    @runtime_tracker
    def update_observation(self, 
                           instrument : str,
                           observation_datum : dict, 
                           t : float) -> None:
        
        
        # get appropriate grid point object
        lat = observation_datum['lat [deg]']
        lon = observation_datum['lon [deg]']
        grid_index,gp_index = self.__get_target_indeces(lat,lon)

        # check if reward grid point exists
        if instrument not in self.rewards[grid_index][gp_index]:
            # add if needed
            self.rewards[grid_index][gp_index][instrument] \
                = GridPoint(instrument, lat, lon, grid_index, gp_index, self.initial_reward)

        # get corresponding grid point
        grid_point : GridPoint = self.rewards[grid_index][gp_index][instrument]
        
        # estimate current reward
        reward : float = self.propagate_reward(grid_point, t)

        # update current grid point reward
        grid_point.update_reward(reward, t)

        # update grid point observation list
        grid_point.update_observations(observation_datum, t)

        # estimate reward
        reward : float = self.propagate_reward(grid_point, t)

        # update grid point reward
        grid_point.update_reward(reward, t)

    @runtime_tracker
    def update_event(self, event : GeophysicalEvent, t : float) -> None:
        # get appropriate grid point object
        lat,lon,_ = event.location
        grid_index,gp_index = self.__get_target_indeces(lat,lon)

        raise NotImplementedError('Update event not implemented yet.')

        for instrument in event.observation_types:
            # check if reward grid point exists
            if instrument not in self.rewards[grid_index][gp_index]:
                # add if needed
                self.rewards[grid_index][gp_index][instrument] = GridPoint(instrument, lat, lon, grid_index, gp_index, self.initial_reward)

            # get corresponding grid point
            grid_point : GridPoint = self.rewards[grid_index][gp_index][instrument]
            
            # estimate current reward
            reward : float = self.propagate_reward(grid_point, t)

            # update current grid point reward
            grid_point.update_reward(reward, t)

            # update grid point observation list
            grid_point.update_events(event, t)

            # estimate reward
            reward : float = self.propagate_reward(grid_point, t)

            # update grid point reward
            grid_point.update_reward(reward, t)
       
    @runtime_tracker
    def propagate_reward(self, grid_point : GridPoint, t : float) -> float:
        params = grid_point.to_dict()
        params.update(self.grid_params)
        params['t'] = t

        return self.reward_function(**params)
    
    @runtime_tracker
    def estimate_reward(self, observation : ObservationAction) -> float:
        lat,lon,_ = observation.targets
        grid_index,gp_index = self.__get_target_indeces(lat,lon)
        grid_point : GridPoint = self.rewards[grid_index][gp_index][observation.instrument_name]

        return self.propagate_reward(grid_point, observation.t_start)
    
    @runtime_tracker
    def get_history(self) -> list:
        history = []

        for grid_rewards in self.rewards:
            for gp_rewards in grid_rewards:
                for _, grid_point in gp_rewards.items():
                    history.extend(grid_point.history)

        history.sort()

        return history

    def __str__(self) -> str:
        histories = self.get_history()

        out = 't_update,grid_index,GP index,lat,lon,instrument,reward,n_observations,n_events\n'
        for t_update, grid_index, gp_index, instrument, reward, n_observations, n_events in histories:
            out += f'{t_update},{grid_index},{gp_index},{instrument},{reward},{n_observations},{n_events}\n'
        
        return out