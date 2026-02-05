import numpy as np
import pandas as pd
from tqdm import tqdm
from dmas.utils.orbitdata import OrbitData

class ObservationTracker:
    def __init__(self, t_last : float = np.NINF, n_obs : int = 0, latest_observation : dict = None):
        """ 
        Class to track the observation tasks and their history.
        """
        # validate inputs
        assert isinstance(t_last, (int, float)), "Last observation time must be a float or int."
        assert isinstance(n_obs, int), "Number of observations must be an integer."
        assert n_obs >= 0, "Number of observations must be non-negative."

        # initialize trackers and parameters
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
        return f"ObservationTracker(t_last={self.t_last}, n_obs={self.n_obs})"

class ObservationHistory:
    def __init__(self, trackers : dict):
        """
        Class to track the observation history of the agent.
        """
        self.trackers : dict[tuple[int,int], ObservationTracker] = trackers
        
    @classmethod
    def from_orbitdata(cls, orbitdata : OrbitData) -> 'ObservationHistory':
        # Create an ObservationHistory instance from OrbitData
        trackers: dict[tuple[int,int], ObservationTracker] = {}

        # columns to extract
        cols = ["lat [deg]", "lon [deg]", "grid index", "GP index"]

        # parse through the grid data
        for df in tqdm(orbitdata.grid_data, desc="Initializing Observation History", unit=" gp", leave=False):
            # get unique grid points
            sub : pd.DataFrame = df[cols].drop_duplicates(subset=["grid index", "GP index"])

            # iterate through the unique grid points
            arr = sub[cols].to_numpy()
            for lat, lon, grid_idx, gp_idx in arr:
            # for lat, lon, grid_idx, gp_idx in sub.itertuples(index=False, name=None):
                grid_idx = int(grid_idx)
                gp_idx = int(gp_idx)
                lat = float(lat); lon = float(lon)

                key = (grid_idx, gp_idx)
                if key not in trackers:
                    trackers[key] = ObservationTracker()

        return cls(trackers)

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

        