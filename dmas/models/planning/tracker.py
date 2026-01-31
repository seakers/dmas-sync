import numpy as np
import pandas as pd
from dmas.core.orbitdata import OrbitData

class ObservationTracker:
    def __init__(self, lat : float, lon : float, grid_index : int, gp_index : int, t_last : float = np.NINF, n_obs : int = 0, latest_observation : dict = None):
        """ 
        Class to track the observation tasks and their history.
        """
        # validate inputs
        assert isinstance(lat, (float, int)), "Latitude must be a float or int."
        assert isinstance(lon, (float, int)), "Longitude must be a float or int."
        assert isinstance(grid_index, int), "Grid index must be an integer."
        assert isinstance(gp_index, int), "Ground point index must be an integer."
        assert isinstance(t_last, (int, float)), "Last observation time must be a float or int."
        assert isinstance(n_obs, int), "Number of observations must be an integer."
        assert n_obs >= 0, "Number of observations must be non-negative."
        assert lat >= -90 and lat <= 90, "Latitude must be between -90 and 90 degrees."
        assert lon >= -180 and lon <= 180, "Longitude must be between -180 and 180 degrees."
        assert grid_index >= 0, "Grid index must be non-negative."
        assert gp_index >= 0, "Ground point index must be non-negative."

        # assign parameters
        self.lat = lat
        self.lon = lon
        self.grid_index = grid_index
        self.gp_index = gp_index
        self.t_last = t_last
        self.n_obs = n_obs
        self.latest_observation = latest_observation
        self.observations : list[dict] = [latest_observation] if latest_observation is not None else []
    
    def update(self, observation : dict) -> None:
        """ Update the observation tracker with a new observation."""        
        # update number of observations at this target
        self.n_obs += 1

        # update list of known observations 
        self.observations.append(observation)

        # update last observation time
        if observation['t_end'] >= self.t_last:
            self.t_last = observation['t_end']
            self.latest_observation = observation

    def __repr__(self):
        return f"ObservationTracker(grid_index={self.grid_index}, gp_index={self.gp_index}, lat={self.lat}, lon={self.lon}, t_last={self.t_last}, n_obs={self.n_obs})"

class ObservationHistory:
    def __init__(self, orbitdata : OrbitData):
        """
        Class to track the observation history of the agent.
        """
        self.history : dict[int, dict[int, ObservationTracker]] = {}
        self.grid_lookup : dict[tuple, tuple] = {}
    
        for gp_index in range(len(orbitdata.grid_data)):
            grid : pd.DataFrame = orbitdata.grid_data[gp_index]
            
            for _,row in grid.iterrows():
                lat = row["lat [deg]"]
                lon = row["lon [deg]"]
                grid_index = int(row["grid index"])
                gp_index = int(row["GP index"])

                # create a new entry for the grid point
                if grid_index not in self.history:
                    self.history[grid_index] = {}
                
                # create a new entry for the grid point
                if gp_index not in self.history[grid_index]:
                    self.history[grid_index][gp_index] = ObservationTracker(lat, lon, grid_index, gp_index) 
                
                # create a lookup table for the grid points
                lat_key = round(row["lat [deg]"], 6)
                lon_key = round(row["lon [deg]"], 6)
                self.grid_lookup[(lat_key, lon_key)] = (
                    int(row["grid index"]),
                    int(row["GP index"])
                )

    def update(self, observations : list) -> None:
        """
        Update the observation history with the new observations.
        """
        for _,observations_data in observations:
            for observation in observations_data:
                grid_index = observation['grid index']
                gp_index = observation['GP index']
                
                tracker : ObservationTracker = self.history[grid_index][gp_index]
                tracker.update(observation)

                # grid_index = observation['grid index']
                # gp_index = observation['GP index']
                # t_end = observation['t_end']
                
                # tracker : ObservationTracker = self.history[grid_index][gp_index]

                # tracker.t_last = t_end
                # tracker.n_obs += 1
                # tracker.latest_observation = observation


    def get_observation_history(self, grid_index : int, gp_index : int) -> ObservationTracker:
        if grid_index in self.history and gp_index in self.history[grid_index]:
            return self.history[grid_index][gp_index]
        else:
            raise ValueError(f"Observation history for grid index {grid_index} and ground point index {gp_index} not found.")

        