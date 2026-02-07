from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.table import table
import numpy as np
import pandas as pd
from tqdm import tqdm

from execsatm.utils import Interval

@dataclass
class AbstractTable(ABC):
    """ Base class for memmap-backed tables. """
    @classmethod
    def load(cls, in_dir: str, mmap_mode: str = "r") -> "AbstractTable":
        # validate inputs
        if not os.path.isdir(in_dir):
            raise ValueError(f"input directory {in_dir} does not exist or is not a directory")
        if not os.path.isfile(os.path.join(in_dir, "meta.json")):
            raise ValueError(f"metadata file meta.json not found in input directory {in_dir}")
        
        # load metadata to get file paths and dtypes
        with open(os.path.join(in_dir, "meta.json"), "r") as f:
            meta = json.load(f)
            # load state data from binary using metadata
            return cls.from_schema(meta, mmap_mode=mmap_mode)
        
    @classmethod
    @abstractmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "AbstractTable":
        """ Creates an instance of the class from a metadata schema. """
        # validate inputs
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dictionary")
        if mmap_mode not in ["r", "r+", "w+", "c"]:
            raise ValueError(f"invalid mmap_mode {mmap_mode}; must be one of 'r', 'r+', 'w+', or 'c'")
    
@dataclass
class TargetGridTable(AbstractTable):
    """
    Memmap-backed target grid table:
      _lat:      (N,1) 
      _lon:      (N,1)
      _grid_idx: (N,1)
      _gp_idx:   (N,1)
    """
    _lat: np.ndarray
    _lon: np.ndarray
    _grid_idx: np.ndarray
    _gp_idx: np.ndarray
    
    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "TargetGridTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)

        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # load data as memmaps
        lat : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["lat_deg"]),  mmap_mode=mmap_mode)
        lon : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["lon_deg"]),  mmap_mode=mmap_mode)
        grid_idx : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["grid_index"]),  mmap_mode=mmap_mode)
        gp_idx : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["GP_index"]),  mmap_mode=mmap_mode)

        # ensure shapes are correct
        assert lat.shape == (schema["n"],), f"expected lat shape {(schema['n'],)}, got {lat.shape}"
        assert lon.shape == (schema["n"],), f"expected lon shape {(schema['n'],)}, got {lon.shape}"
        assert grid_idx.shape == (schema["n"],), f"expected grid_idx shape {(schema['n'],)}, got {grid_idx.shape}"
        assert gp_idx.shape == (schema["n"],), f"expected gp_idx shape {(schema['n'],)}, got {gp_idx.shape}"

        # initiate GroundPointTable object
        return cls(_lat=lat, _lon=lon, _grid_idx=grid_idx, _gp_idx=gp_idx)
    

    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for i in range(len(self._lat)):
            yield (self._lat[i], self._lon[i], self._grid_idx[i], self._gp_idx[i])
            

@dataclass
class StateTable(AbstractTable):
    """
    Memmap-backed time-indexed state table:
      t:        (N,1) 
      pos:      (N,3)
      vel:      (N,3)
    """
    _t: np.ndarray
    _pos: np.ndarray
    _vel: np.ndarray
    
    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "StateTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)

        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # load data as memmaps
        t : np.ndarray   = np.load(os.path.join(in_dir, schema["files"]["t"]),    mmap_mode=mmap_mode)
        pos : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["pos"]),  mmap_mode=mmap_mode)
        vel : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["vel"]),  mmap_mode=mmap_mode)

        # ensure shapes are correct
        assert t.shape == (schema["n"],), f"expected t shape {(schema['n'],)}, got {t.shape}"
        assert pos.shape == (schema["n"], 3), f"expected pos shape {(schema['n'], 3)}, got {pos.shape}"
        assert vel.shape == (schema["n"], 3), f"expected vel shape {(schema['n'], 3)}, got {vel.shape}"

        # initiate StateTable object
        return cls(_t=t, _pos=pos, _vel=vel)

    def lookup(self, t: float) -> tuple:
        """ Returns the state (position and velocity) at time `t` by interpolating between the closest time indices. """

        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        
        # check if t is out of bounds
        if t < self._t[0] - 1e-6:
            # time is positive but before first index; return first state
            return self._pos[0], self._vel[0]
        elif t > self._t[-1] + 1e-6:
            # time is beyond last index; return last state
            return self._pos[-1], self._vel[-1]

        # find location in time array 
        t_idx = np.searchsorted(self._t, t, side="left")

        # check if exact match was found
        if abs(self._t[t_idx] - t) < 1e-6:
            # return state at time index
            return self._pos[t_idx], self._vel[t_idx]

        # interpolate between closest indices
        i0 = max(0, t_idx - 1)
        alpha = (t - self._t[i0]) / (self._t[t_idx] - self._t[i0])
        return self.__linear_interp_state(i0, alpha)
    
    def __linear_interp_state(self, i0: int, alpha: float):
        """
        alpha in [0,1): returns state between i0 and i0+1
        """
        p = (1.0 - alpha) * self._pos[i0] + alpha * self._pos[i0 + 1]
        v = (1.0 - alpha) * self._vel[i0] + alpha * self._vel[i0 + 1]
        return p, v
    
    def lookup_value(self, t : float) -> tuple:
        # TODO 
        pass

@dataclass
class IntervalTable(AbstractTable):
    """
    Memmap-backed interval table.
    """
    _start: np.ndarray              # memmap
    _end: np.ndarray                # memmap
    _prefix_max_end: np.ndarray     # memmap
    _extras: Dict[str, np.ndarray]  # memmap per column

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "IntervalTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)
        
        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # load required data as memmaps
        start : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["start"]), mmap_mode=mmap_mode)
        end : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["end"]), mmap_mode=mmap_mode)
        prefix : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["prefix_max_end"]), mmap_mode=mmap_mode)

        # load any extra columns as memmaps
        extras: Dict[str, np.ndarray] = {}
        for k, fname in schema["files"].items():
            if k in ("start", "end", "prefix_max_end"):
                continue
            extras[k] = np.load(os.path.join(in_dir, fname), mmap_mode=mmap_mode)

        # validate shapes
        assert start.shape == (schema["n"],), f"expected start shape {(schema['n'],)}, got {start.shape}"
        assert end.shape == (schema["n"],), f"expected end shape {(schema['n'],)}, got {end.shape}"
        assert prefix.shape == (schema["n"],), f"expected prefix_max_end shape {(schema['n'],)}, got {prefix.shape}"
        for k, arr in extras.items():
            assert arr.shape == (schema["n"],), f"expected extra column {k} shape {(schema['n'],)}, got {arr.shape}"

        # return `IntervalTable` object
        return cls(_start=start, _end=end, _prefix_max_end=prefix, _extras=extras)    
    
    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for i in range(len(self._start)):
            row = {k: self._extras[k][i] for k in self._extras}
            yield (self._start[i], self._end[i], *row.values())

    def lookup_intervals(self, t : float, t_max: float = np.Inf, include_current: bool = False) -> List[Interval]:
        """
        Returns a list of intervals that start after or at time `t` and end before or at time `t_max`. If `include_current` is True, also includes the interval that contains time `t` if it exists.
        """
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        if not isinstance(t_max, (int, float)) or t_max < 0.0:
            raise ValueError("time t_max must be a non-negative number")
        if t > t_max + 1e-6:
            raise ValueError("time t must be less than or equal to time t_max")
        
        # check if there is any data
        if len(self._start) == 0:
            return []

        # check if t is beyond the end of all intervals
        if t > self._prefix_max_end[-1] + 1e-6:
            # time is beyond the end of all intervals; return empty list
            return []

        # find starting index using prefix max end for efficient search
        idx = np.searchsorted(self._prefix_max_end, t, side="left")

        # set upper bound for search to stop at t_max
        n = len(self._start)

        # iterate through intervals starting from idx until we go past t_max
        intervals = []
        for i in range(idx, n):
            # early stop if start time is beyond t_max
            if self._start[i] > t_max + 1e-6: break

            # check if starts after time t or is current and include_current is True
            if include_current or self._start[i] > t - 1e-6:
                t_start = max(self._start[i], t) if include_current else self._start[i]
                intervals.append(Interval(float(t_start), float(self._end[i])))

        # return list of intervals
        return intervals

    def lookup_interval(self, t : float, include_current: bool = False) -> Optional[Interval]:
        """
        Returns the interval that contains time `t` if it exists. If `include_current` is True, also includes the interval that starts at time `t` if it exists.
        """
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
    
        # check if t is beyond the end of all intervals
        if t > self._prefix_max_end[-1] + 1e-6:
            # time is beyond the end of all intervals; return None
            return None

        # find starting index using prefix max end for efficient search
        idx = np.searchsorted(self._prefix_max_end, t, side="left")
    
    
@dataclass
class AccessTable(AbstractTable):
    """
    Memmap-backed access table.    
    """

    _offsets: np.ndarray     # (T+1,) int64
    _t: np.ndarray           # (M,)   float32
    _t_idx: np.ndarray       # (M,)   int32/int64
    _lat: np.ndarray         # (M,)   float32
    _lon: np.ndarray         # (M,)   float32
    _grid_idx: np.ndarray      # (M,)   int32/int64
    _gp_idx: np.ndarray      # (M,)   int32/int64
    _extras: Dict[str, np.ndarray]  # memmap per extra column
    _meta: Dict[str, Any]

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "AccessTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)

        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # load required data as memmaps
        offsets = np.load(os.path.join(in_dir, schema["files"]["offsets"]), mmap_mode=mmap_mode)
        t = np.load(os.path.join(in_dir, schema["files"]["t"]), mmap_mode=mmap_mode)
        t_idx = np.load(os.path.join(in_dir, schema["files"]["t_index"]), mmap_mode=mmap_mode)
        grid_idx  = np.load(os.path.join(in_dir, schema["files"]["grid_index"]),  mmap_mode=mmap_mode)
        gp_idx  = np.load(os.path.join(in_dir, schema["files"]["GP_index"]),  mmap_mode=mmap_mode)
        lat     = np.load(os.path.join(in_dir, schema["files"]["lat_deg"]),     mmap_mode=mmap_mode)
        lon     = np.load(os.path.join(in_dir, schema["files"]["lon_deg"]),     mmap_mode=mmap_mode)

        # load any extra columns as memmaps
        extras: Dict[str, np.ndarray] = {}
        for k, fname in schema["files"].items():
            if k in ("offsets", "t", "t_index", "grid_index", "GP_index", "lat_deg", "lon_deg"):
                continue
            extras[k] = np.load(os.path.join(in_dir, fname), mmap_mode=mmap_mode)

        return cls(_offsets=offsets, _t=t, _t_idx=t_idx, _lat=lat, _lon=lon, _grid_idx=grid_idx, _gp_idx=gp_idx, _extras=extras, _meta=schema)

    @property
    def n_steps(self) -> int:
        return int(self._meta["n_steps"])

    @property
    def n_rows(self) -> int:
        return int(self._meta["n_rows"])


# class AbstractDataSeries(ABC):
#     """
#     Base class for all database types.
#     """
#     def __init__(self, name : str, columns : list, data : list):
#         self.name = name
#         self.columns = columns
#         self.data = data

#     @abstractmethod
#     def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param') -> 'AbstractDataSeries':
#         """ Creates an instance of the class from a pandas DataFrame. """
    
#     @abstractmethod
#     def update_expired_values(self, t :float) -> None:
#         """ Updates the data by removing all values that are older than time `t`. """    

# class TimeIndexedData(AbstractDataSeries):
#     def __init__(self, 
#                  name : str,
#                  columns : list,
#                  t : list,
#                  data : Dict[str, np.ndarray],
#                  bin_size : float = 3600, # in seconds, default 1 hour
#                  printouts : bool = True
#                 ):
#         """ Stores time-indexed data for fast lookup."""
        
#         # validate inputs
#         assert isinstance(name, str), 'name must be a string'
#         assert len(columns) == len(data), 'number of columns and data do not match'
#         assert all([len(data[col]) == len(t) for col in columns]), 'number of time steps and data do not match'
#         assert isinstance(t, (list, np.ndarray)) and all([isinstance(val, (int, float)) for val in t]), 't must be a list or numpy array of numbers'
        
#         self.name : str = name
#         self.columns : List[str] = columns
#         self.t : List[float] = t
#         self.data : Dict[str, np.ndarray] = data
#         self.bin_size : float = bin_size
                
#         # count number of bins
#         self.n_bins = int(max(t, default=0) // bin_size) + 1

#         # group data indices into bins depending on their time for faster lookup
#         grouped_indices = [[] for _ in range(self.n_bins)]
#         inv_bs = 1.0 / bin_size
#         for i, ti in tqdm(enumerate(t), desc=f'Grouping time-indexed {name} time', unit=' time bins', leave=False, disable=not printouts):
#             b = int(ti * inv_bs)
#             if b >= self.n_bins:
#                 b = self.n_bins - 1
#             grouped_indices[b].append(i)

#         # assign grouped data
#         self.grouped_t = [[t[i] for i in idx] for idx in grouped_indices]
#         self.grouped_data = {
#             col: [[data[col][i] for i in idx] for idx in grouped_indices]
#             for col in columns
#         }

#     def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param', printouts : bool = True) -> 'TimeIndexedData':
#         # validate inputs
#         assert 'time index' in df.columns or 'time [s]' in df.columns, 'time column not found in dataframe'
#         assert time_step > 0.0, 'time step must be greater than 0.0'

#         # get data columns 
#         columns = list(df.columns.values)
        
#         # get appropriate time data
#         if 'time index' in columns:
#             # sort dataframe by time index
#             df = df.sort_values(by=['time index'])

#             # get time column index
#             time_column_index = columns.index('time index')

#             # remove time column from columns list
#             columns.remove('time index')

#             # get time data in seconds
#             t = [val*time_step for val in np.array(df.iloc[:, time_column_index].to_numpy())]

#         elif 'time [s]' in columns:
#             # sort dataframe by time index
#             df = df.sort_values(by=['time [s]'])

#             # get time column index
#             time_column_index = columns.index('time [s]')
            
#             # remove time column from columns list
#             columns.remove('time [s]')

#             # get time data in seconds
#             t = [val for val in np.array(df.iloc[:, time_column_index].to_numpy())]
#         else:
#             raise ValueError('time column not found in dataframe')

#         # get data from dataframe and ignore time column
#         data = {col : np.array(df[col]) for col in df.columns.values}
#         if 'time index' in data:
#             data.pop('time index')
#         elif 'time [s]' in data:
#             data.pop('time [s]')
                
#         # return TimeIndexedData object
#         return TimeIndexedData(name, columns, np.array(t), data, printouts=printouts)

#     def lookup_value(self, t : float, columns : list = None) -> dict:
#         """
#         Returns the value of data at time `t` in seconds
#         """
#         # get desired columns
#         columns = columns if columns is not None else self.columns

#         # Choose bin
#         if not np.isfinite(t):
#             bin_index = len(self.grouped_t) - 1
#         else:
#             bin_index = int(t // self.bin_size)
#             if bin_index >= len(self.grouped_t):
#                 bin_index = len(self.grouped_t) - 1
#             elif bin_index < 0:
#                 bin_index = 0

#         xp = self.grouped_t[bin_index]
#         if len(xp) == 0:
#             # empty bin; fall back (or return empties)
#             return {**{col: np.nan for col in columns}, 'time [s]': t}

#         # Clamp to edges like np.interp does
#         if t <= xp[0]:
#             out = {col: float(self.grouped_data[col][bin_index][0]) for col in columns}
#             out['time [s]'] = t
#             return out
#         if t >= xp[-1]:
#             out = {col: float(self.grouped_data[col][bin_index][-1]) for col in columns}
#             out['time [s]'] = t
#             return out

#         # Find right index once
#         j = int(np.searchsorted(xp, t, side="right"))
#         i = j - 1

#         x0 = xp[i]; x1 = xp[j]
#         w = (t - x0) / (x1 - x0)

#         out = {}
#         for col in columns:
#             fp = self.grouped_data[col][bin_index]
#             y0 = fp[i]; y1 = fp[j]
#             out[col] = float(y0 + w * (y1 - y0))
#         out['time [s]'] = t
#         return out
    
#     def lookup_interval(self, t_start : float, t_end : float, columns : list = None) -> Dict[str, list]:
#         """
#         Returns the value of data between the start and end times in seconds
#         """
#         assert t_start <= t_end, "start time must be less than end time"
#         assert t_start >= 0.0, "start time must be greater than 0.0"

#         columns = self.columns if columns is None else columns

#         if not self.grouped_t:
#             return {col: [] for col in (columns + ['time [s]'])}

#         eps = 1e-6
#         t0 = t_start - eps
#         t1 = t_end + eps

#         # clamp bins safely
#         bin_start = int(t_start // self.bin_size)
#         bin_end = len(self.grouped_t) - 1 if not np.isfinite(t_end) else min(int(t_end // self.bin_size), len(self.grouped_t) - 1)

#         if bin_start < 0:
#             bin_start = 0

#         # If start bin beyond range, just use global arrays (or return empty)
#         if bin_start >= len(self.grouped_t):
#             return {col: [] for col in (columns + ['time [s]'])}

#         # Collect per-bin slices, then concatenate once
#         t_chunks = []
#         idx_slices = []  # store (bin_i, slice(l, r)) so we reuse for each column

#         for i in range(bin_start, bin_end + 1):
#             t_bin = self.grouped_t[i]
#             if len(t_bin) == 0:
#                 continue

#             # t_bin must be sorted for searchsorted
#             l = int(np.searchsorted(t_bin, t0, side="left"))
#             r = int(np.searchsorted(t_bin, t1, side="right"))
#             if r > l:
#                 idx_slices.append((i, l, r))
#                 t_chunks.append(t_bin[l:r])

#         if not idx_slices:
#             return {col: [] for col in (columns + ['time [s]'])}

#         t_out = np.concatenate(t_chunks)

#         out = {'time [s]': t_out.tolist()}

#         for col in columns:
#             v_chunks = []
#             for i, l, r in idx_slices:
#                 v_chunks.append(self.grouped_data[col][i][l:r])
#             out[col] = np.concatenate(v_chunks).tolist()

#         # lengths match by construction
#         return out
        
#     def __iter__(self):
#         """
#         Returns an iterator over the data
#         """
#         for i in range(len(self.t)):
#             row = {col : self.data[col][i] for col in self.columns}
#             t = self.t[i]
#             yield (t,row)

#     def update_expired_values(self, t : float):
#         # only keep values that are still active or that haven't expired yet
#         unexpired_indeces = [(i,t_i) for i, t_i in enumerate(self.t) 
#                             if t_i >= t or abs(t_i - t) <= 1e-6]
        
#         # to avoid empty data, keep the last value if all values have expired
#         if not unexpired_indeces and self.t.size > 0:
#             unexpired_indeces = [(len(self.t)-1, self.t[-1])]
        
#         # update internal data
#         self.t = np.array([t_i for _, t_i in unexpired_indeces])
#         self.data = {col : np.array([self.data[col][i] for i, _ in unexpired_indeces]) 
#                     for col in self.columns}

# class IntervalData(AbstractDataSeries):
#     def __init__(self, 
#                  name : str,
#                  columns : List[str],
#                  data : List[tuple],
#                  bin_size : float = 3600, # in seconds, default 1 hour
#                  printouts : bool = True
#                 ):
#         """ Stores interval-indexed data for fast lookup."""

#         # validate inputs
#         assert isinstance(name, str), 'name must be a string'
#         assert isinstance(columns, list) and all([isinstance(col, str) for col in columns]), 'columns must be a list of strings'
#         assert isinstance(data, list) and all([isinstance(row, tuple) and len(row) >= 2 for row in data]), 'data must be a list of tuples with at least 2 elements (start time, end time)'
#         assert isinstance(bin_size, (int, float)) and bin_size > 0, 'bin_size must be a positive number'

#         # set attributes
#         self.name : str = name
#         self.columns : List[str] = columns
#         self.data : List[tuple] = data
#         self.bin_size : float = bin_size

#         # find maximum time to determine number of bins
#         max_t_start = max([t_start for t_start,*_ in data], default=0.0)
#         max_t_end = max([t_end for _,t_end,*_ in data], default=0.0)
#         max_t = max(max_t_start, max_t_end)

#         # store raw columns in parallel arrays for speed
#         self.n_bins = max(1, int(max_t // self.bin_size) + 1)
#         self.bin_to_data_indices = [[] for _ in range(self.n_bins)]

#         # group data indices into bins depending on their interval for faster lookup
#         for interval_idx, (t_start, t_end, *_) in tqdm(enumerate(data), desc=f'Grouping interval {name} data', unit=' time bins', leave=False, disable=not printouts):
#             b0 = max(0, int(t_start // self.bin_size))
#             b1 = min(self.n_bins - 1, int(t_end // self.bin_size))

#             assert b1 >= b0, \
#                 'invalid bin indices computed for interval data'

#             for b in range(b0, b1 + 1):
#                 self.bin_to_data_indices[b].append(interval_idx)

#         # sort each bin by start time for early stopping during lookup
#         for ids in tqdm(self.bin_to_data_indices, desc=f'Sorting interval {name} data indices', unit=' time bins', leave=False, disable=not printouts):
#             ids.sort(key=lambda idx: data[idx][0])

#     def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param', printouts: bool = True) -> 'IntervalData':
#         assert time_step > 0.0, 'time step must be greater than 0.0'
#         assert 'start index' in df.columns, 'start index column not found in dataframe'
#         assert 'end index' in df.columns, 'end index column not found in dataframe'
        
#         # sort dataframe by time index
#         df.sort_values(by=['start index'])

#         # get time column index
#         if any(['index' in col for col in df.columns.values]):
#             # replace time index with time in seconds
#             columns = [col.replace('index', 'time [s]') for col in df.columns.values]
            
#             # get time data in Inteval format
#             data = [(t_start*time_step, t_end*time_step, *row) 
#                     for t_start,t_end,*row in df.values]
#         else:
#             # get time column index
#             columns = [col for col in df.columns.values]
            
#             # get time data in Inteval format
#             data = [(t_start, t_end, *row) for t_start,t_end,*row in df.values]

#         # return IntervalData object
#         return IntervalData(name, columns, data, printouts=printouts)
    
#     def lookup(self, t : float) -> Tuple:
#         """
#         Returns interval that contains time `t`. Returns None if no interval contains time `t`
#         """        
#         # check if there is any data
#         if len(self.data) == 0: return None

#         # set tolerance for floating point comparisons
#         eps = 1e-6

#         # find appropriate bin to search
#         bin_idx = int(t // self.bin_size)
        
#         # bound bin index
#         if bin_idx < 0: 
#             bin_idx = 0
#         if bin_idx >= self.n_bins: 
#             bin_idx = self.n_bins - 1

#         # define earliest matching interval
#         earliest_interval_idx = None
#         t_start_earliest = np.Inf

#         # search for interval in appropriate bin
#         for data_idx in self.bin_to_data_indices[bin_idx]:
#             # get interval data
#             t_start_i,t_end_i,*_ = self.data[data_idx]

#             # check if interval time starts after `t`
#             if t + eps < t_start_i:  
#                 # early stop because bin list is sorted by start time;
#                 # no need to check further intervals
#                 break

#             # check if `t` is within interval
#             if t_start_i - eps <= t <= t_end_i + eps:
#                 # choose whatever tie-break you want; here earliest start
#                 if earliest_interval_idx is None or t_start_i < t_start_earliest:
#                     earliest_interval_idx = data_idx
#                     t_start_earliest = t_start_i

#         # return the earliest matching interval if found
#         return tuple(self.data[earliest_interval_idx]) \
#             if earliest_interval_idx is not None else None

#     def lookup_intervals(self, t_start : float, t_end : float) -> List[Interval]:
#         """
#         Returns all intervals that overlap with the interval [t_start, t_end]
#         """
#         # validate inputs
#         assert isinstance(t_start, (int,float)) and t_start >= 0.0, 'start time must be a positive number'
#         assert isinstance(t_end, (int,float)) and t_end >= 0.0, 'end time must be a positive number'
#         assert t_start <= t_end, 'start time must be less than end time'

#         # check if there is any data
#         if len(self.data) == 0: return []
        
#         # set tolerance for floating point comparisons
#         eps = 1e-6

#         # set query interval with tolerance
#         q0, q1 = t_start - eps, t_end + eps

#         # find appropriate bins to search
#         b0 = max(int(t_start // self.bin_size), 0)
#         b1 = int(t_end // self.bin_size) if not np.isinf(t_end) else self.n_bins - 1
        
#         # check if start bin is beyond range
#         if b0 >= self.n_bins: 
#             return []

#         # bound bin indices
#         b0 = min(self.n_bins - 1, b0)
#         b1 = min(self.n_bins - 1, b1)

#         assert b1 >= b0, \
#             'invalid bin indices computed for interval data lookup'

#         # store seen interval indices to avoid duplicates
#         seen = set()
        
#         # search for intervals in appropriate bins
#         out = []
#         for bin_idx in range(b0, b1 + 1):
#             # search for intervals in the bin
#             for data_idx in self.bin_to_data_indices[bin_idx]:
#                 # avoid duplicates
#                 if data_idx in seen: continue

#                 # mark interval as seen
#                 seen.add(data_idx)

#                 # unpack interval data
#                 t_start_i,t_end_i,*row = self.data[data_idx]

#                 # check if interval time starts after `t_end`
#                 if t_start_i > q1:  
#                     # early stop because bin list is sorted by start time;
#                     # no need to check further intervals
#                     break

#                 if not (t_end_i < q0 or t_start_i > q1):
#                     out.append((t_start_i, t_end_i, *row))

#         # sort intervals by start time
#         out.sort(key=lambda r: r[0])

#         # return the matching intervals
#         return [Interval(t_start_i, t_end_i) for t_start_i,t_end_i,*_ in out]
    
#     def is_active(self, t : float) -> bool:
#         """
#         Returns True if time `t` is in any of the intervals
#         """
#         current_interval = self.lookup(t)
#         return current_interval is not None
#         # return any([t_start-1e-6 <= t <= t_end+1e-6 for t_start,t_end,*_ in self.data])
    
#     def update_expired_values(self, t : float) -> None:
#         """ 
#         Updates the data by removing all intervals that have ended before time `t`. 
#         """
#         # only keep intervals that are still active or that haven't expired yet
#         data = [(t_start,t_end,*row) for t_start,t_end,*row in self.data
#                     if t <= t_end or abs(t - t_end) <= 1e-6]
        
#         # update internal data if there are any unexpired intervals;
#         #  to avoid empty data, keep the last value if all values have expired
#         self.data = [self.data[-1]] if not data and self.data else data
        
#     def __len__(self):
#         return len(self.data)
    
#     def __iter__(self):
#         """
#         Returns an iterator over the data
#         """
#         for row in self.data:
#             yield row