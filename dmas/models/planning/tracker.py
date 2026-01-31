import numpy as np
import pandas as pd
from tqdm import tqdm
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
    def __init__(self, trackers : dict, grid_lookup : dict):
        """
        Class to track the observation history of the agent.
        """
        self.trackers : dict[tuple[int,int], ObservationTracker] = trackers
        self.grid_lookup : dict[tuple[float,float], tuple[int,int]] = grid_lookup
        
    @classmethod
    def from_orbitdata(cls, orbitdata : OrbitData) -> 'ObservationHistory':
        # Create an ObservationHistory instance from OrbitData
        trackers: dict[tuple[int,int], ObservationTracker] = {}
        grid_lookup: dict[tuple[float,float], tuple[int,int]] = {}

        # columns to extract
        cols = ["lat [deg]", "lon [deg]", "grid index", "GP index"]

        # parse through the grid data
        for df in tqdm(orbitdata.grid_data, desc="Initializing Observation History", unit=" gp", leave=False):
            # get unique grid points
            sub : pd.DataFrame = df[cols].drop_duplicates(subset=["grid index", "GP index"])

            # iterate through the unique grid points
            for lat, lon, grid_idx, gp_idx in sub.itertuples(index=False, name=None):
                grid_idx = int(grid_idx)
                gp_idx = int(gp_idx)
                lat = float(lat); lon = float(lon)

                key = (grid_idx, gp_idx)
                if key not in trackers:
                    trackers[key] = ObservationTracker(lat, lon, grid_idx, gp_idx)

                lat_key = int(round(lat * 1_000_000))
                lon_key = int(round(lon * 1_000_000))
                grid_lookup[(lat_key, lon_key)] = key

        return cls(trackers, grid_lookup)

    def update(self, observations : list) -> None:
        """
        Update the observation history with the new observations.
        """
        for _,observations_data in observations:
            for observation in observations_data:
                grid_index = observation['grid index']
                gp_index = observation['GP index']
                
                tracker : ObservationTracker = self.trackers[(grid_index, gp_index)]
                tracker.update(observation)

    def get_observation_history(self, grid_index : int, gp_index : int) -> ObservationTracker:
        key = (grid_index, gp_index)
        if key in self.trackers:
            return self.trackers[key]
        else:
            raise ValueError(f"Observation history for grid index {grid_index} and ground point index {gp_index} not found.")

        