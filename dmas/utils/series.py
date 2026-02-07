from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple

from matplotlib.table import table
import numpy as np
import pandas as pd
from tqdm import tqdm

from execsatm.utils import Interval


@dataclass
class StateTable:
    """
    Memmap-backed time-indexed state table:
      pos: (N,3)
      vel: (N,3)
      t:   (N,) optional (monotonic int indices or time indices)
    """
    _pos: np.ndarray
    _vel: np.ndarray
    _t: Optional[np.ndarray]
    _meta: Dict

    @classmethod
    def load(cls, in_dir: str, mmap_mode: str = "r") -> "StateTable":
        with open(os.path.join(in_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        pos = np.load(os.path.join(in_dir, meta["files"]["pos"]), mmap_mode=mmap_mode)
        vel = np.load(os.path.join(in_dir, meta["files"]["vel"]), mmap_mode=mmap_mode)

        t = None
        if meta.get("has_t", False):
            t = np.load(os.path.join(in_dir, meta["files"]["t"]), mmap_mode=mmap_mode)

        return cls(_pos=pos, _vel=vel, _t=t, _meta=meta)

    def get_state_at_index(self, i: int):
        return self._pos[i], self._vel[i]

    def get_state_at_time(self, t: float) -> tuple:
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        if self._t is None:
            raise ValueError("This StateTable has no explicit t array; use direct index lookup.")
        
        # calculate time index
        time_step = self._meta.get("time_step", None)
        if time_step is None:
            raise ValueError("This StateTable does not have a time step defined in its metadata; cannot compute time index.")
        
        # Find location in time array 
        idx = np.searchsorted(self._t, t, side="left")

        # check if exact match
        if idx < len(self._t) and abs(self._t[idx] - t) < 1e-6:
            # return state at time index
            return self._pos[idx], self._vel[idx]
        if idx >= len(self._t):
            # time is beyond last index; return last state
            return self._pos[-1], self._vel[-1]

        # interpolate between closest indices
        i0 = max(0, idx - 1)
        alpha = (t - self._t[i0]) / (self._t[idx] - self._t[i0])
        return self.lerp_state(i0, alpha)

    # def get_state_at_time_index_value(self, t_val: int):
    #     if self._t is None:
    #         raise ValueError("This StateTable has no explicit t array; use direct index lookup.")

    #     # Find exact match
    #     i = int(np.searchsorted(self._t, t_val, side="left"))
    #     if i >= len(self._t) or int(self._t[i]) != int(t_val):
    #         raise KeyError(f"time index {t_val} not found")
    #     return self._pos[i], self._vel[i]
    
    # def get_states_in_index_range(self, i0: int, i1: int):
    #     return self._pos[i0:i1], self._vel[i0:i1]
    
    def lerp_state(self, i0: int, alpha: float):
        """
        alpha in [0,1): returns state between i0 and i0+1
        """
        p = (1.0 - alpha) * self._pos[i0] + alpha * self._pos[i0 + 1]
        v = (1.0 - alpha) * self._vel[i0] + alpha * self._vel[i0 + 1]
        return p, v





class AbstractDataSeries(ABC):
    """
    Base class for all database types.
    """
    def __init__(self, name : str, columns : list, data : list):
        self.name = name
        self.columns = columns
        self.data = data

    @abstractmethod
    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param') -> 'AbstractDataSeries':
        """ Creates an instance of the class from a pandas DataFrame. """
    
    @abstractmethod
    def update_expired_values(self, t :float) -> None:
        """ Updates the data by removing all values that are older than time `t`. """    

class TimeIndexedData(AbstractDataSeries):
    def __init__(self, 
                 name : str,
                 columns : list,
                 t : list,
                 data : Dict[str, np.ndarray],
                 bin_size : float = 3600, # in seconds, default 1 hour
                 printouts : bool = True
                ):
        """ Stores time-indexed data for fast lookup."""
        
        # validate inputs
        assert isinstance(name, str), 'name must be a string'
        assert len(columns) == len(data), 'number of columns and data do not match'
        assert all([len(data[col]) == len(t) for col in columns]), 'number of time steps and data do not match'
        assert isinstance(t, (list, np.ndarray)) and all([isinstance(val, (int, float)) for val in t]), 't must be a list or numpy array of numbers'
        
        self.name : str = name
        self.columns : List[str] = columns
        self.t : List[float] = t
        self.data : Dict[str, np.ndarray] = data
        self.bin_size : float = bin_size
                
        # count number of bins
        self.n_bins = int(max(t, default=0) // bin_size) + 1

        # group data indices into bins depending on their time for faster lookup
        grouped_indices = [[] for _ in range(self.n_bins)]
        inv_bs = 1.0 / bin_size
        for i, ti in tqdm(enumerate(t), desc=f'Grouping time-indexed {name} time', unit=' time bins', leave=False, disable=not printouts):
            b = int(ti * inv_bs)
            if b >= self.n_bins:
                b = self.n_bins - 1
            grouped_indices[b].append(i)

        # assign grouped data
        self.grouped_t = [[t[i] for i in idx] for idx in grouped_indices]
        self.grouped_data = {
            col: [[data[col][i] for i in idx] for idx in grouped_indices]
            for col in columns
        }

    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param', printouts : bool = True) -> 'TimeIndexedData':
        # validate inputs
        assert 'time index' in df.columns or 'time [s]' in df.columns, 'time column not found in dataframe'
        assert time_step > 0.0, 'time step must be greater than 0.0'

        # get data columns 
        columns = list(df.columns.values)
        
        # get appropriate time data
        if 'time index' in columns:
            # sort dataframe by time index
            df = df.sort_values(by=['time index'])

            # get time column index
            time_column_index = columns.index('time index')

            # remove time column from columns list
            columns.remove('time index')

            # get time data in seconds
            t = [val*time_step for val in np.array(df.iloc[:, time_column_index].to_numpy())]

        elif 'time [s]' in columns:
            # sort dataframe by time index
            df = df.sort_values(by=['time [s]'])

            # get time column index
            time_column_index = columns.index('time [s]')
            
            # remove time column from columns list
            columns.remove('time [s]')

            # get time data in seconds
            t = [val for val in np.array(df.iloc[:, time_column_index].to_numpy())]
        else:
            raise ValueError('time column not found in dataframe')

        # get data from dataframe and ignore time column
        data = {col : np.array(df[col]) for col in df.columns.values}
        if 'time index' in data:
            data.pop('time index')
        elif 'time [s]' in data:
            data.pop('time [s]')
                
        # return TimeIndexedData object
        return TimeIndexedData(name, columns, np.array(t), data, printouts=printouts)

    def lookup_value(self, t : float, columns : list = None) -> dict:
        """
        Returns the value of data at time `t` in seconds
        """
        # get desired columns
        columns = columns if columns is not None else self.columns

        # Choose bin
        if not np.isfinite(t):
            bin_index = len(self.grouped_t) - 1
        else:
            bin_index = int(t // self.bin_size)
            if bin_index >= len(self.grouped_t):
                bin_index = len(self.grouped_t) - 1
            elif bin_index < 0:
                bin_index = 0

        xp = self.grouped_t[bin_index]
        if len(xp) == 0:
            # empty bin; fall back (or return empties)
            return {**{col: np.nan for col in columns}, 'time [s]': t}

        # Clamp to edges like np.interp does
        if t <= xp[0]:
            out = {col: float(self.grouped_data[col][bin_index][0]) for col in columns}
            out['time [s]'] = t
            return out
        if t >= xp[-1]:
            out = {col: float(self.grouped_data[col][bin_index][-1]) for col in columns}
            out['time [s]'] = t
            return out

        # Find right index once
        j = int(np.searchsorted(xp, t, side="right"))
        i = j - 1

        x0 = xp[i]; x1 = xp[j]
        w = (t - x0) / (x1 - x0)

        out = {}
        for col in columns:
            fp = self.grouped_data[col][bin_index]
            y0 = fp[i]; y1 = fp[j]
            out[col] = float(y0 + w * (y1 - y0))
        out['time [s]'] = t
        return out
    
    def lookup_interval(self, t_start : float, t_end : float, columns : list = None) -> Dict[str, list]:
        """
        Returns the value of data between the start and end times in seconds
        """
        assert t_start <= t_end, "start time must be less than end time"
        assert t_start >= 0.0, "start time must be greater than 0.0"

        columns = self.columns if columns is None else columns

        if not self.grouped_t:
            return {col: [] for col in (columns + ['time [s]'])}

        eps = 1e-6
        t0 = t_start - eps
        t1 = t_end + eps

        # clamp bins safely
        bin_start = int(t_start // self.bin_size)
        bin_end = len(self.grouped_t) - 1 if not np.isfinite(t_end) else min(int(t_end // self.bin_size), len(self.grouped_t) - 1)

        if bin_start < 0:
            bin_start = 0

        # If start bin beyond range, just use global arrays (or return empty)
        if bin_start >= len(self.grouped_t):
            return {col: [] for col in (columns + ['time [s]'])}

        # Collect per-bin slices, then concatenate once
        t_chunks = []
        idx_slices = []  # store (bin_i, slice(l, r)) so we reuse for each column

        for i in range(bin_start, bin_end + 1):
            t_bin = self.grouped_t[i]
            if len(t_bin) == 0:
                continue

            # t_bin must be sorted for searchsorted
            l = int(np.searchsorted(t_bin, t0, side="left"))
            r = int(np.searchsorted(t_bin, t1, side="right"))
            if r > l:
                idx_slices.append((i, l, r))
                t_chunks.append(t_bin[l:r])

        if not idx_slices:
            return {col: [] for col in (columns + ['time [s]'])}

        t_out = np.concatenate(t_chunks)

        out = {'time [s]': t_out.tolist()}

        for col in columns:
            v_chunks = []
            for i, l, r in idx_slices:
                v_chunks.append(self.grouped_data[col][i][l:r])
            out[col] = np.concatenate(v_chunks).tolist()

        # lengths match by construction
        return out
        
    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for i in range(len(self.t)):
            row = {col : self.data[col][i] for col in self.columns}
            t = self.t[i]
            yield (t,row)

    def update_expired_values(self, t : float):
        # only keep values that are still active or that haven't expired yet
        unexpired_indeces = [(i,t_i) for i, t_i in enumerate(self.t) 
                            if t_i >= t or abs(t_i - t) <= 1e-6]
        
        # to avoid empty data, keep the last value if all values have expired
        if not unexpired_indeces and self.t.size > 0:
            unexpired_indeces = [(len(self.t)-1, self.t[-1])]
        
        # update internal data
        self.t = np.array([t_i for _, t_i in unexpired_indeces])
        self.data = {col : np.array([self.data[col][i] for i, _ in unexpired_indeces]) 
                    for col in self.columns}

class IntervalData(AbstractDataSeries):
    def __init__(self, 
                 name : str,
                 columns : List[str],
                 data : List[tuple],
                 bin_size : float = 3600, # in seconds, default 1 hour
                 printouts : bool = True
                ):
        """ Stores interval-indexed data for fast lookup."""

        # validate inputs
        assert isinstance(name, str), 'name must be a string'
        assert isinstance(columns, list) and all([isinstance(col, str) for col in columns]), 'columns must be a list of strings'
        assert isinstance(data, list) and all([isinstance(row, tuple) and len(row) >= 2 for row in data]), 'data must be a list of tuples with at least 2 elements (start time, end time)'
        assert isinstance(bin_size, (int, float)) and bin_size > 0, 'bin_size must be a positive number'

        # set attributes
        self.name : str = name
        self.columns : List[str] = columns
        self.data : List[tuple] = data
        self.bin_size : float = bin_size

        # find maximum time to determine number of bins
        max_t_start = max([t_start for t_start,*_ in data], default=0.0)
        max_t_end = max([t_end for _,t_end,*_ in data], default=0.0)
        max_t = max(max_t_start, max_t_end)

        # store raw columns in parallel arrays for speed
        self.n_bins = max(1, int(max_t // self.bin_size) + 1)
        self.bin_to_data_indices = [[] for _ in range(self.n_bins)]

        # group data indices into bins depending on their interval for faster lookup
        for interval_idx, (t_start, t_end, *_) in tqdm(enumerate(data), desc=f'Grouping interval {name} data', unit=' time bins', leave=False, disable=not printouts):
            b0 = max(0, int(t_start // self.bin_size))
            b1 = min(self.n_bins - 1, int(t_end // self.bin_size))

            assert b1 >= b0, \
                'invalid bin indices computed for interval data'

            for b in range(b0, b1 + 1):
                self.bin_to_data_indices[b].append(interval_idx)

        # sort each bin by start time for early stopping during lookup
        for ids in tqdm(self.bin_to_data_indices, desc=f'Sorting interval {name} data indices', unit=' time bins', leave=False, disable=not printouts):
            ids.sort(key=lambda idx: data[idx][0])

    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param', printouts: bool = True) -> 'IntervalData':
        assert time_step > 0.0, 'time step must be greater than 0.0'
        assert 'start index' in df.columns, 'start index column not found in dataframe'
        assert 'end index' in df.columns, 'end index column not found in dataframe'
        
        # sort dataframe by time index
        df.sort_values(by=['start index'])

        # get time column index
        if any(['index' in col for col in df.columns.values]):
            # replace time index with time in seconds
            columns = [col.replace('index', 'time [s]') for col in df.columns.values]
            
            # get time data in Inteval format
            data = [(t_start*time_step, t_end*time_step, *row) 
                    for t_start,t_end,*row in df.values]
        else:
            # get time column index
            columns = [col for col in df.columns.values]
            
            # get time data in Inteval format
            data = [(t_start, t_end, *row) for t_start,t_end,*row in df.values]

        # return IntervalData object
        return IntervalData(name, columns, data, printouts=printouts)
    
    def lookup(self, t : float) -> Tuple:
        """
        Returns interval that contains time `t`. Returns None if no interval contains time `t`
        """        
        # check if there is any data
        if len(self.data) == 0: return None

        # set tolerance for floating point comparisons
        eps = 1e-6

        # find appropriate bin to search
        bin_idx = int(t // self.bin_size)
        
        # bound bin index
        if bin_idx < 0: 
            bin_idx = 0
        if bin_idx >= self.n_bins: 
            bin_idx = self.n_bins - 1

        # define earliest matching interval
        earliest_interval_idx = None
        t_start_earliest = np.Inf

        # search for interval in appropriate bin
        for data_idx in self.bin_to_data_indices[bin_idx]:
            # get interval data
            t_start_i,t_end_i,*_ = self.data[data_idx]

            # check if interval time starts after `t`
            if t + eps < t_start_i:  
                # early stop because bin list is sorted by start time;
                # no need to check further intervals
                break

            # check if `t` is within interval
            if t_start_i - eps <= t <= t_end_i + eps:
                # choose whatever tie-break you want; here earliest start
                if earliest_interval_idx is None or t_start_i < t_start_earliest:
                    earliest_interval_idx = data_idx
                    t_start_earliest = t_start_i

        # return the earliest matching interval if found
        return tuple(self.data[earliest_interval_idx]) \
            if earliest_interval_idx is not None else None

    def lookup_intervals(self, t_start : float, t_end : float) -> List[Interval]:
        """
        Returns all intervals that overlap with the interval [t_start, t_end]
        """
        # validate inputs
        assert isinstance(t_start, (int,float)) and t_start >= 0.0, 'start time must be a positive number'
        assert isinstance(t_end, (int,float)) and t_end >= 0.0, 'end time must be a positive number'
        assert t_start <= t_end, 'start time must be less than end time'

        # check if there is any data
        if len(self.data) == 0: return []
        
        # set tolerance for floating point comparisons
        eps = 1e-6

        # set query interval with tolerance
        q0, q1 = t_start - eps, t_end + eps

        # find appropriate bins to search
        b0 = max(int(t_start // self.bin_size), 0)
        b1 = int(t_end // self.bin_size) if not np.isinf(t_end) else self.n_bins - 1
        
        # check if start bin is beyond range
        if b0 >= self.n_bins: 
            return []

        # bound bin indices
        b0 = min(self.n_bins - 1, b0)
        b1 = min(self.n_bins - 1, b1)

        assert b1 >= b0, \
            'invalid bin indices computed for interval data lookup'

        # store seen interval indices to avoid duplicates
        seen = set()
        
        # search for intervals in appropriate bins
        out = []
        for bin_idx in range(b0, b1 + 1):
            # search for intervals in the bin
            for data_idx in self.bin_to_data_indices[bin_idx]:
                # avoid duplicates
                if data_idx in seen: continue

                # mark interval as seen
                seen.add(data_idx)

                # unpack interval data
                t_start_i,t_end_i,*row = self.data[data_idx]

                # check if interval time starts after `t_end`
                if t_start_i > q1:  
                    # early stop because bin list is sorted by start time;
                    # no need to check further intervals
                    break

                if not (t_end_i < q0 or t_start_i > q1):
                    out.append((t_start_i, t_end_i, *row))

        # sort intervals by start time
        out.sort(key=lambda r: r[0])

        # return the matching intervals
        return [Interval(t_start_i, t_end_i) for t_start_i,t_end_i,*_ in out]
    
    def is_active(self, t : float) -> bool:
        """
        Returns True if time `t` is in any of the intervals
        """
        current_interval = self.lookup(t)
        return current_interval is not None
        # return any([t_start-1e-6 <= t <= t_end+1e-6 for t_start,t_end,*_ in self.data])
    
    def update_expired_values(self, t : float) -> None:
        """ 
        Updates the data by removing all intervals that have ended before time `t`. 
        """
        # only keep intervals that are still active or that haven't expired yet
        data = [(t_start,t_end,*row) for t_start,t_end,*row in self.data
                    if t <= t_end or abs(t - t_end) <= 1e-6]
        
        # update internal data if there are any unexpired intervals;
        #  to avoid empty data, keep the last value if all values have expired
        self.data = [self.data[-1]] if not data and self.data else data
        
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for row in self.data:
            yield row