from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
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
            yield (float(self._lat[i]), float(self._lon[i]), int(self._grid_idx[i]), int(self._gp_idx[i]))

    def __len__(self):
        return len(self._lat)

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

    def lookup_value(self, t: float) -> tuple:
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
    
    def __len__(self):
        return len(self._t)

@dataclass
class IntervalTable(AbstractTable):
    """
    Memmap-backed interval table.
    """
    _start: np.ndarray              # memmap
    _end: np.ndarray                # memmap
    _prefix_max_end: np.ndarray     # memmap
    _extras: Dict[str, np.ndarray]  # memmap per column
    _meta : Dict[str, Any]          # metadata dictionary

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
        return cls(_start=start, _end=end, _prefix_max_end=prefix, _extras=extras, _meta=schema)    
    
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

    def lookup_interval(self, t : float, t_max : float = np.Inf, include_current: bool = False) -> Optional[Interval]:
        """
        Returns the interval that contains time `t` if it exists. If `include_current` is True, also includes the interval that starts at time `t` if it exists.
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
            return None

        # check if t is beyond the end of all intervals
        if t > self._prefix_max_end[-1] + 1e-6:
            # time is beyond the end of all intervals; return None
            return None

        # find starting index using prefix max end for efficient search
        idx = np.searchsorted(self._prefix_max_end, t, side="left")

        # set upper bound for search to stop at t_max
        n = len(self._start)

        # iterate through intervals starting from idx 
        for i in range(idx, n):
            # early stop if start time is beyond t_max
            if self._start[i] > t_max + 1e-6: break

            # check if starts after time t or is current and include_current is True
            if include_current or self._start[i] > t - 1e-6:
                # first interval that contains time t is found
                t_start = max(self._start[i], t) if include_current else self._start[i]
                
                # return interval as Interval object
                return Interval(float(t_start), float(self._end[i]))

        # fallback, return None
        return None
    
    def __len__(self):
        return len(self._start)
    
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
    
    def __len__(self):
        return len(self._t)
      
    # def get_next_access_intervals(self, target: str, t: float, t_max: float = np.Inf, include_current: bool = False) -> List[Interval]:
    def lookup_interval(self, t_start : float, t_end : float = np.Inf) -> Dict[str, np.ndarray]:
        # validate inputs
        if not isinstance(t_start, (int, float)) or t_start < 0.0:
            raise ValueError("time t_start must be a non-negative number")
        if not isinstance(t_end, (int, float)) or t_end < 0.0:
            raise ValueError("time t_end must be a non-negative number")
        if t_start > t_end + 1e-6:
            raise ValueError("time t_start must be less than or equal to time t_end")
        
        # check if there is any data
        if len(self._t) == 0:
            s0 = slice(0, 0)
            return self.__rows_from_slice(s0, include_extras=True)

        # get start and end indices for time range
        ti0 = self._time_to_index_floor(t_start)
        ti1 = self._time_to_index_floor(t_end)

        # check if start and end time are in the same time index bucket
        if ti0 == ti1:
            # if so, return the rows for that bucket
            return self.lookup_time(t_start, include_extras=True)

        # get slice for time index range
        s = self._slice_for_index_range(ti0, ti1)

        # construct output dictionary
        out = self.__rows_from_slice(s, include_extras=True)

        # filter out any rows that are outside the time range 
        if s.start != s.stop:
            mask = (t_start <= out["time [s]"]) & (out["time [s]"] <= t_end)
            for col in list(out.keys()):
                out[col] = out[col][mask]
        
        # return output
        return out
    
    def _time_to_index_floor(self, t: float) -> int:
        dt = float(self._meta["time_step"])
        return int(t // dt) if not np.isinf(t) else len(self._offsets) - 1
    
    def _slice_for_index_range(self, ti0: int, ti1: int) -> slice:
        if ti1 < ti0:
            return slice(0, 0)
        ti0 = self._clamp_index(ti0)
        ti1 = self._clamp_index(ti1)
        if ti1 < ti0:
            return slice(0, 0)
        a = int(self._offsets[ti0])
        b = int(self._offsets[ti1 + 1])
        return slice(a, b)
    
    def _clamp_index(self, ti: int) -> int:
        T = len(self._offsets) - 1
        if ti < 0:
            return 0
        if ti > T - 1:
            return T - 1
        return ti
           
    def __rows_from_slice(self, s: slice, include_extras: bool = False) -> Dict[str, Any]:
        out = {
            "time [s]": self._t[s].astype(float),
            "lat [deg]": self._lat[s].astype(float),
            "lon [deg]": self._lon[s].astype(float),
            "grid index": self._grid_idx[s],
            "GP index": self._gp_idx[s],
        }

        # add any extra columns
        if include_extras: 
            for k, arr in self._extras.items():
                col = self._meta["columns"].get(k, {}).get("col_name", k)
                # check if column is encoded in metadata
                if 'vocab' in self._meta["columns"].get(k, {}):
                    # decode column using vocab
                    vocab : dict = self._meta["columns"][k]["vocab"]
                    out[col] = np.array([vocab.get(str(code), None) for code in arr[s]])
                else:
                    out[col] = arr[s].astype(float)

        return out
    
    def lookup_time(self, t: float, include_extras: bool = False) -> Dict[str, Any]:
        # Find nearest / first occurrence; but you still need the corresponding time index bucket.
        # If you trust that each bucket has constant t value, you can map by search then use t_idx:
        
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time `t` must be a non-negative number")
        
        # check if there is any data
        if len(self._t) == 0:
            s0 = slice(0, 0)
            return self.__rows_from_slice(s0, include_extras=include_extras)
        
        elif self._t[-1] < t - 1e-6:
            # time is beyond last index; return last row
            s_last = slice(len(self._t) - 1, len(self._t))
            return self.__rows_from_slice(s_last, include_extras=include_extras)
        
        # find location in time array
        i = int(np.searchsorted(self._t, t, side="left"))
        
        # convert to time index and get rows for that time index
        ti = int(self._t_idx[i])
        return self.__rows_at_index(ti, include_extras=include_extras)
    
    def __rows_at_index(self, ti: int, include_extras: bool = False) -> Dict[str, Any]:
        s = self.__slice_for_time_index(ti)
        return self.__rows_from_slice(s, include_extras=include_extras)
    
    def __slice_for_time_index(self, ti: int) -> slice:
        # bounds check (optional but nice)
        if ti < 0 or ti >= (len(self._offsets) - 1):
            return slice(0, 0)  # empty
        a = int(self._offsets[ti])
        b = int(self._offsets[ti + 1])
        return slice(a, b)

    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for i in range(len(self._t)):
            
            out = {
                # "t_index": self._t_idx[i],
                "lat [deg]": float(self._lat[i]),
                "lon [deg]": float(self._lon[i]),
                "grid index": self._grid_idx[i],
                "GP index": self._gp_idx[i],
            }
            # add any extra columns
            for k, arr in self._extras.items():
                col = self._meta["columns"].get(k, {}).get("col_name", k)                
                # check if column is encoded in metadata
                if 'vocab' in self._meta["columns"].get(k, {}):
                    # decode column using vocab
                    vocab : dict = self._meta["columns"][k]["vocab"]
                    # add decoding for current row
                    out[col] = vocab.get(str(arr[i]), None)
                else:
                    # add raw value for current row
                    out[col] = float(arr[i])

            yield float(self._t[i]),out