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
