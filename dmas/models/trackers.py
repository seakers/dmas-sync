from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os, uuid
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from yaml import warnings

# @dataclass
# class ObservationTracker:
#     __slots__ = ("t_last", "n_obs", "latest_observation", "observations")

#     def __init__(self, latest_observation: Optional[dict] = None):
#         """ 
#         Class to track the observation tasks and their history.
#         """
#         self.t_last = np.NINF
#         self.n_obs = 0
#         self.latest_observation = latest_observation
#         self.observations = [latest_observation] if latest_observation else []

#     def update(self, observation : dict) -> None:
#         """ Update the observation tracker with a new observation."""        
#         # update number of observations at this target
#         self.n_obs += 1

#         # update list of known observations 
#         self.observations.append(observation)

#         # update last observation time
#         if observation['t_end'] >= self.t_last:
#             self.t_last = observation['t_end']
#             self.latest_observation = observation

#     def __repr__(self):
#         return f"ObservationTracker(t_last={self.t_last}, n_obs={self.n_obs})"

# class ObservationHistory:
#     def __init__(self, trackers : dict):
#         """
#         Class to track the observation history of the agent.
#         """
#         self.trackers : dict[tuple[int,int], ObservationTracker] = trackers
        
#     @classmethod
#     def from_orbitdata(cls, orbitdata : OrbitData) -> 'ObservationHistory':
#         # Create an ObservationHistory instance from OrbitData
#         trackers: dict[tuple[int,int], ObservationTracker] = {}

#         # columns to extract
#         cols = ["lat [deg]", "lon [deg]", "grid index", "GP index"]

#         # parse through the grid data
#         for df in tqdm(orbitdata.grid_data, desc="Initializing Observation History", unit=" gp", leave=False):
#             # get unique grid points
#             sub : pd.DataFrame = df[cols].drop_duplicates(subset=["grid index", "GP index"])

#             # iterate through the unique grid points
#             arr = sub[cols].to_numpy()
#             for lat, lon, grid_idx, gp_idx in arr:
#             # for lat, lon, grid_idx, gp_idx in sub.itertuples(index=False, name=None):
#                 grid_idx = int(grid_idx)
#                 gp_idx = int(gp_idx)
#                 lat = float(lat); lon = float(lon)

#                 key = (grid_idx, gp_idx)
#                 if key not in trackers:
#                     trackers[key] = ObservationTracker()

#         return cls(trackers)

#     def update(self, observations : list) -> None:
#         """
#         Update the observation history with the new observations.
#         """
#         for _,observations_data in observations:
#             for observation in observations_data:
#                 grid_index = observation['grid index']
#                 gp_index = observation['GP index']
                
#                 tracker : ObservationTracker = self.trackers[(grid_index, gp_index)]
#                 tracker.update(observation)

#     def get_observation_history(self, grid_index : int, gp_index : int) -> ObservationTracker:
#         key = (grid_index, gp_index)
#         if key in self.trackers:
#             return self.trackers[key]
#         else:
#             raise ValueError(f"Observation history for grid index {grid_index} and ground point index {gp_index} not found.")

@dataclass
class DataSink:
    """
    Class to manage the storage of data in a parquet file. 
    Data is buffered in memory and flushed to disk when the buffer reaches a specified size or when the sink is closed.

    ### Attributes:
    - `out_dir`: Directory where the parquet file will be stored.
    - `owner_name`: Name of the owner of the data (e.g., agent name).
    - `data_name`: Name of the data being stored (e.g., "observations").
    - `flush_rows`: Number of rows to buffer before flushing to disk (default: 50,000).
    - `_rows`: Internal buffer to store rows of data before flushing.
    - `_writer`: PyArrow ParquetWriter object for writing data to the parquet file.
    - `_path`: Path to the parquet file being written.
    - `_closed`: Flag to indicate whether the sink has been closed.
    """
    

    out_dir: str
    owner_name: str
    data_name: str
    flush_rows: int = 50_000

    _rows: List[Dict[str, Any]] = field(default_factory=list)
    _writer: Optional[pq.ParquetWriter] = None
    _path: Optional[str] = None

    _closed: bool = False

    def append(self, obs: Dict[str, Any]) -> None:
        """ Append a single observation to the sink. Flushes to disk if the buffer exceeds the flush threshold."""
        # append a single observation to the rows buffer
        self._rows.append(obs)

        # flush if we exceed the threshold
        if len(self._rows) >= self.flush_rows:
            self.flush()

    def extend(self, obs_list: List[Dict[str, Any]]) -> None:
        """ Extend the sink with a list of observations. Flushes to disk if the buffer exceeds the flush threshold."""
        # extend current rows with the new list of observations 
        self._rows.extend(obs_list)

        # find index of last flush point
        i_flush = len(self._rows) - len(self._rows) % self.flush_rows

        # flush if we exceed the threshold
        if len(self._rows) >= self.flush_rows:
            self.flush(i_flush)

    def flush(self, i_max : int = -1) -> None:
        """ Flush the buffered data to disk. If `i_max` is specified, only flush up to that index in the buffer."""
        # if sink is empty, do nothing
        if not self._rows: return

        # make sure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # convert rows to a PyArrow table
        table = pa.Table.from_pylist(self._rows[:i_max] if i_max != -1 else self._rows)

        # if writer is not initialized, create it with the schema of the first table
        if self._writer is None:
            self._path = os.path.join(self.out_dir, f"{self.owner_name}_{self.data_name}.parquet")
            self._writer = pq.ParquetWriter(self._path, table.schema, compression="zstd")

        # write the table to the parquet file and clear the rows buffer
        self._writer.write_table(table)

        # clear the rows buffer
        self._rows = self._rows[i_max:] if i_max != -1 else []

    def close(self) -> None:
        """ Close the sink, flushing any remaining data and releasing resources."""
        # cannot close an already closed object
        if self._closed: return

        # flush any remaining data 
        self.flush()

        # close the writer if it exists
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        
        # set closed flag to True
        self._closed = True

    def __del__(self):
        try:
            # check if sink is already closed, if not, close it
            if not getattr(self, "_closed", True): self.close()
        except Exception:
            # if any error occurs during __del__, 
            #   we ignore it to avoid issues during garbage collection
            warnings.warn("An error occurred while deleting DataSink object. \
                          This is likely due to an issue during garbage collection. \
                          The error has been ignored to prevent potential crashes.", 
                          RuntimeWarning)


class ObservationTracker:
    ...

