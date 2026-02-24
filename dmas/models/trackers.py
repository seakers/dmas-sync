from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os, uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from dmas.utils.orbitdata import OrbitData

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

    _buffer: List[Dict[str, Any]] = field(default_factory=list)
    _writer: Optional[pq.ParquetWriter] = None
    _path: Optional[str] = None
    _flush_count: int = 0

    _closed: bool = False

    def append(self, obs: Dict[str, Any]) -> None:
        """ Append a single observation to the sink. Flushes to disk if the buffer exceeds the flush threshold."""
        # ensure sink is open before appending
        self.__ensure_open()
        
        # append a single observation to the buffer
        self._buffer.append(obs)

        # flush if we exceed the threshold
        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def extend(self, obs_list: List[Dict[str, Any]]) -> None:
        """ Extend the sink with a list of observations. Flushes to disk if the buffer exceeds the flush threshold."""
        # ensure sink is open before extending
        self.__ensure_open()
        
        # extend current buffer with the new list of observations 
        self._buffer.extend(obs_list)

        # find index of last flush point
        i_flush = len(self._buffer) - len(self._buffer) % self.flush_rows

        # flush if we exceed the threshold
        if len(self._buffer) >= self.flush_rows:
            self.flush(i_flush)

    def flush(self, i_max : int = -1) -> None:
        """ Flush the buffered data to disk. If `i_max` is specified, only flush up to that index in the buffer."""
        # ensure sink is open before flushing
        self.__ensure_open()
        
        # if sink is empty, do nothing
        if not self._buffer: return

        # ensure i_max is valid
        if i_max != -1 and not (0 < i_max <= len(self._buffer)):
            raise ValueError(f"i_max must be -1 or in the range (0, {len(self._buffer)}], but got {i_max}")

        # make sure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # convert rows to a PyArrow table
        table = pa.Table.from_pylist(self._buffer[:i_max] if i_max != -1 else self._buffer)

        # if writer is not initialized, create it with the schema of the first table
        if self._writer is None:
            self._path = os.path.join(self.out_dir, f"{self.data_name}.parquet")
            self._writer = pq.ParquetWriter(self._path, table.schema, compression="zstd")

        # write the table to the parquet file and clear the buffer
        self._writer.write_table(table)

        # clear the buffer       
        if i_max == -1:
            # if i_max is -1, we flushed all rows, so we can clear the entire buffer
            self._buffer.clear()
        else:
            # if i_max is specified, we only flushed up to that index, so we remove those rows from the buffer
            del self._buffer[:i_max]

        # increment flush count
        self._flush_count += 1

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

    def __ensure_open(self):
        if self._closed:
            raise RuntimeError(f"DataSink({self.owner_name}:{self.data_name}) is closed.")

    def __del__(self):
        try:
            # check if sink is already closed, if not, close it
            if getattr(self, "_writer", None) is not None and not getattr(self, "_closed", True):
                self.close()
        except Exception:
            # if any error occurs during __del__, 
            #   we ignore it to avoid issues during garbage collection
            pass

    def __len__(self):
        return len(self._buffer)
    
    def __iter__(self):
        for row in self._buffer:
            yield row

    def empty(self) -> bool:
        """ Returns True if the sink has no buffered data, False otherwise."""
        return len(self._buffer) == 0 and self._flush_count == 0

    def get_flush_count(self) -> int:
        """ Returns the number of times the sink has been flushed to disk."""
        return self._flush_count

@dataclass
class LatestObservationTracker:

    lut: np.ndarray          # shape (G, P), dtype int32, -1 => not tracked
    G: int
    P: int
    N: int
    t_last: np.ndarray
    n_obs: np.ndarray
    last_actor: np.ndarray
    actor_to_id: Dict[str, int]
    id_to_actor: List[str]

    @classmethod
    def from_orbitdata(
        cls,
        orbitdata: OrbitData,
        actor_name : str,
        quiet: bool = False,
    ) -> "LatestObservationTracker":
        """
        Builds the target index mapping from OrbitData.grid_data.

        Expects:
          orbitdata.grid_data: iterable of pandas DataFrames containing columns:
            ["grid index", "GP index"] (and optionally lat/lon, but not required for tracking)
        """
        
        # if enabled, wrap with tqdm progress bar
        if not quiet:
            grid_iter = tqdm(orbitdata.grid_data, desc="Init ObservationHistory", unit=" df", leave=False)
        else:
            grid_iter = orbitdata.grid_data 

        # iterate through the unique grid points and populate the key_to_k and k_to_key mappings
        targets = [
            (grid_idx, gp_idx)
            for *_,grid_idx,gp_idx in grid_iter
        ]

        # remove duplicate targets while preserving order
        targets_unique = list(set(targets))

        # build from targets
        return cls.from_targets(targets_unique, actor_name=actor_name)

    @classmethod
    def from_targets(cls,
                     targets: Iterable[Tuple[int, int]],
                     actor_name: str,
                     dtype_n_obs=np.uint16,
                     lut_dtype=np.int32,
                    ) -> "LatestObservationTracker":
        
        # Build bounds
        gmax = -1
        pmax = -1
        pairs = []
        for g, p in targets:
            g = int(g); p = int(p)
            pairs.append((g, p))
            if g > gmax: gmax = g
            if p > pmax: pmax = p

        G = gmax + 1
        P = pmax + 1

        lut = np.full((G, P), -1, dtype=lut_dtype)

        # Assign dense k ids only to existing targets
        k = 0
        for g, p in pairs:
            if lut[g, p] == -1:
                lut[g, p] = k
                k += 1
        N = k

        t_last = np.full(N, -np.inf, dtype=np.float64)
        n_obs = np.zeros(N, dtype=dtype_n_obs)
        last_actor = np.full(N, -1, dtype=np.int16)

        actor_to_id = {actor_name: 0}
        id_to_actor = [actor_name]

        return cls(lut, G, P, N, t_last, n_obs, last_actor, actor_to_id, id_to_actor)

    def _get_actor_id(self, actor_name: Optional[str]) -> int:
        if actor_name is None:
            return -1
        aid = self.actor_to_id.get(actor_name)
        if aid is None:
            aid = len(self.actor_to_id)
            self.actor_to_id[actor_name] = aid
            self.id_to_actor.append(actor_name)  # <-- append, not index-assign
        return aid

    def _k(self, grid_idx: int, gp_idx: int) -> int:
        g = int(grid_idx); p = int(gp_idx)
        if g < 0 or p < 0 or g >= self.G or p >= self.P:
            return -1
        return int(self.lut[g, p])

    def __update_one(self, obs: Dict[str, Any]) -> None:
        k = self._k(obs["grid index"], obs["GP index"])
        if k < 0:
            return

        t_end = float(obs["t_end"])
        actor_id = self._get_actor_id(obs.get("agent name"))

        self.n_obs[k] = self.n_obs[k] + 1
        if t_end >= self.t_last[k]:
            self.t_last[k] = t_end
            self.last_actor[k] = actor_id

    def update_many(self, observations: Iterable[Tuple[str,List[Dict[str, Any]]]]) -> None:
        """
        Update arrays for many observation dicts.
        """
        # validate input
        if not isinstance(observations, Iterable):
            raise ValueError("observations must be an iterable of dicts")
        
        # update each observation in the input iterable
        for _,obs_data in observations:
            # get the observation with the latest t_end
            obs = max(obs_data, key=lambda o: o.get("t_end", -np.inf), default=None)  
            
            # `obs_data` is empty; skip 
            if obs is None:
                continue
            
            # update the tracker with this observation
            self.__update_one(obs)

    def lookup(self, grid_index : int, gp_index : int) :
        """
        Lookup the latest observation info for a given grid point.

        Returns a dict with keys:
          - "t_last": time of the latest observation (float)
          - "n_obs": total number of observations (int)
          - "last_actor": name of the last observing agent (str or None)
        """
        k = self._k(grid_index, gp_index)
        if k < 0:
            return {"t_last": -np.inf, "n_obs": -1, "last_actor": None}
        
        actor_id = self.last_actor[k]
        actor_name = self.id_to_actor[actor_id] if actor_id >= 0 else None

        return {
            "t_last": float(self.t_last[k]),
            "n_obs": int(self.n_obs[k]),
            "last_actor": actor_name
        }