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
    Memmap-backed target grid table (packed).
    Backing file: one (N, K) memmap called `_buf`.

    schema["layout"] defines column order in grid.npy.
    """
    _buf: np.ndarray      # memmap (N,K)
    _lat: np.ndarray      # view (N,)
    _lon: np.ndarray      # view (N,)
    _grid_idx: np.ndarray # view (N,) (likely float in packed file)
    _gp_idx: np.ndarray   # view (N,) (likely float in packed file)

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "TargetGridTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)
        
        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # ensure required fields are in schema
        if in_dir is None: raise ValueError("schema missing 'dir'")
        if "files" not in schema or "grid" not in schema["files"]:
            raise ValueError("Packed TargetGridTable schema must include files['grid']")

        # get number of rows in table from schema
        n = int(schema["n"])
        layout = schema.get("layout")
        if not layout:
            raise ValueError("Packed TargetGridTable schema must include 'layout'")
        k = len(layout)

        # check if table is empty
        if n == 0:
            # empty table; define empty `ndarray` with correct number of columns based on layout
            dtype = np.dtype(schema.get("packed_dtype", np.float32))
            buf : np.ndarray = np.empty((0, k), dtype=dtype)

            lat : np.ndarray        = buf[:, layout.index("lat [deg]")] if "lat [deg]" in layout else np.empty((0,), dtype=dtype)
            lon : np.ndarray        = buf[:, layout.index("lon [deg]")] if "lon [deg]" in layout else np.empty((0,), dtype=dtype)
            grid_idx : np.ndarray   = buf[:, layout.index("grid index")] if "grid index" in layout else np.empty((0,), dtype=dtype)
            gp_idx : np.ndarray     = buf[:, layout.index("GP index")] if "GP index" in layout else np.empty((0,), dtype=dtype)

            return cls(_buf=buf, _lat=lat, _lon=lon, _grid_idx=grid_idx, _gp_idx=gp_idx)

        # load packed data as memmap
        buf : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["grid"]), mmap_mode=mmap_mode)

        # validate shape of packed data
        if buf.shape[0] != n:
            raise AssertionError(f"expected grid rows {n}, got {buf.shape[0]}")
        if buf.shape[1] != k:
            raise AssertionError(f"expected grid columns {k} based on layout, got {buf.shape[1]}")

        # enumerate columns in layout for indexing
        col = {name: i for i, name in enumerate(layout)}

        # ensure required columns are present in layout
        for req in ("lat [deg]", "lon [deg]", "grid index", "GP index"):
            if req not in col:
                raise ValueError(f"layout missing required column '{req}'")

        # parse required data into views
        lat      : np.ndarray = buf[:, col["lat [deg]"]]
        lon      : np.ndarray = buf[:, col["lon [deg]"]]
        grid_idx : np.ndarray = buf[:, col["grid index"]]
        gp_idx   : np.ndarray = buf[:, col["GP index"]]

        # ensure shapes are correct
        assert lat.shape == (n,), f"expected lat shape {(n,)}, got {lat.shape}"
        assert lon.shape == (n,), f"expected lon shape {(n,)}, got {lon.shape}"
        assert grid_idx.shape == (n,), f"expected grid_idx shape {(n,)}, got {grid_idx.shape}"
        assert gp_idx.shape == (n,), f"expected gp_idx shape {(n,)}, got {gp_idx.shape}"
        assert buf.shape == (n, k), f"expected buf shape {(n, k)}, got {buf.shape}"

        # return `TargetGridTable` object
        return cls(_buf=buf, _lat=lat, _lon=lon, _grid_idx=grid_idx, _gp_idx=gp_idx)

    def __iter__(self):
        for i in range(len(self._lat)):
            yield (float(self._lat[i]), float(self._lon[i]), int(self._grid_idx[i]), int(self._gp_idx[i]))

    def __len__(self):
        return len(self._lat)

@dataclass
class StateTable(AbstractTable):
    """
    Memmap-backed time-indexed state table (packed).
    Backing file: one (N,7) memmap called `_buf`.

    Columns typically: [t, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    """
    _buf: np.ndarray   # memmap (N,7)
    _t: np.ndarray     # view (N,)
    _pos: np.ndarray   # view (N,3)
    _vel: np.ndarray   # view (N,3)

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "StateTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)
        
        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # ensure required fields are in schema
        if in_dir is None: raise ValueError("schema missing 'dir'")
        if "files" not in schema or "state" not in schema["files"]:
            raise ValueError("Packed StateTable schema must include files['state']")

        # get number of rows in table from schema
        n = int(schema["n"])
        layout = schema.get("layout")
        if not layout:
            raise ValueError("Packed StateTable schema must include 'layout'")
        if len(layout) != 7:
            raise ValueError(f"Packed StateTable schema layout must have 7 columns, got {len(layout)}")
        
        # check if table is empty
        if n == 0:
            # empty table; define empty `ndarray` with 7 columns for [t, pos(3), vel(3)]
            dtype = np.dtype(schema.get("packed_dtype", np.float32))
            buf = np.empty((0, 7), dtype=dtype)

            t = buf[:, 0] if "t" in layout else np.empty((0,), dtype=dtype)
            pos = buf[:, 1:4] if all(col in layout for col in ("x [km]", "y [km]", "z [km]")) else np.empty((0, 3), dtype=dtype)
            vel = buf[:, 4:7] if all(col in layout for col in ("vx [km/s]", "vy [km/s]", "vz [km/s]")) else np.empty((0, 3), dtype=dtype)

            return cls(_buf=buf, _t=t, _pos=pos, _vel=vel)

        # load packed data as memmap
        buf : np.ndarray = np.load(os.path.join(in_dir, schema["files"]["state"]), mmap_mode=mmap_mode)

        # validate shape of packed data
        if buf.shape != (n, 7):
            raise AssertionError(f"expected state shape {(n, 7)}, got {buf.shape}")

        # parse required data into views based on layout if provided
        if layout:
            # enumerate columns in layout for indexing
            col = {name: i for i, name in enumerate(layout)}

            # ensure required columns are present in layout
            t : np.ndarray = buf[:, col["t"]]
            pos : np.ndarray = buf[:, [col["x [km]"], col["y [km]"], col["z [km]"]]]
            vel : np.ndarray = buf[:, [col["vx [km/s]"], col["vy [km/s]"], col["vz [km/s]"]]]
        else:
            # fixed convention [t, pos(3), vel(3)]
            t : np.ndarray = buf[:, 0]
            pos : np.ndarray = buf[:, 1:4]
            vel : np.ndarray = buf[:, 4:7]

        # ensure shapes are correct
        assert t.shape == (n,), f"expected t shape {(n,)}, got {t.shape}"
        assert pos.shape == (n, 3), f"expected pos shape {(n, 3)}, got {pos.shape}"
        assert vel.shape == (n, 3), f"expected vel shape {(n, 3)}, got {vel.shape}"
        assert buf.shape == (n, 7), f"expected buf shape {(n, 7)}, got {buf.shape}"        

        # return `StateTable` object
        return cls(_buf=buf, _t=t, _pos=pos, _vel=vel)

    def lookup_value(self, t: float) -> tuple:
        """Returns the state (position and velocity) at time `t` by interpolating between the closest time indices."""
        
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        
        # check if table is empty
        if len(self._t) == 0:
            raise ValueError("`StateTable` is empty")

        # check if t is out of bounds
        if t < self._t[0] - 1e-6:
            # time is positive but before first index; return first known state
            return self._pos[0], self._vel[0]
        elif t > self._t[-1] + 1e-6:
            # time is beyond last index; return last known state
            return self._pos[-1], self._vel[-1]

        # find location in time array 
        t_idx = np.searchsorted(self._t, t, side="left")

        # check if exact match was found
        if t_idx < len(self._t) and abs(self._t[t_idx] - t) < 1e-6:
            # return state at time index
            return self._pos[t_idx], self._vel[t_idx]

        # interpolate between closest indices
        i0 = max(0, t_idx - 1)
        alpha = (t - self._t[i0]) / (self._t[t_idx] - self._t[i0])
        return self.__linear_interp_state(i0, alpha)

    def __linear_interp_state(self, i0: int, alpha: float):
        """ alpha in [0,1): returns state between i0 and i0+1 """
        p = (1.0 - alpha) * self._pos[i0] + alpha * self._pos[i0 + 1]
        v = (1.0 - alpha) * self._vel[i0] + alpha * self._vel[i0 + 1]
        return p, v

    def __len__(self):
        return len(self._t)
    

@dataclass
class IntervalTable(AbstractTable):
    """
    Memmap-backed interval table (packed).
    Backing storage: a single (N, K) memmap called `_buf`.

    schema["layout"] defines column order:
      ["start", "end", "prefix_max_end", <extras...>]
    """
    _buf: np.ndarray                # memmap (N,K)
    _start: np.ndarray              # view (N,)
    _end: np.ndarray                # view (N,)
    _prefix_max_end: np.ndarray     # view (N,)
    _extras: Dict[str, np.ndarray]  # views (N,) by safe-name key
    _meta: Dict[str, Any]           # metadata dictionary
    _col: Dict[str, int]            # name -> column index

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "IntervalTable":
        # validate inputs 
        super().from_schema(schema, mmap_mode=mmap_mode)

        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # ensure required fields are in layout
        if in_dir is None: raise ValueError("schema missing 'dir'")
        if "files" in schema and "intervals" in schema["files"]:
            packed_key = "intervals"
        elif "files" in schema and "packed" in schema["files"]:
            packed_key = "packed"  # optional backward compat name
        else:
            raise ValueError("Packed IntervalTable schema must include files['intervals']")
        
        # get number of rows in table from schema
        n = int(schema["n"])
        layout = schema.get("layout", None)
        if not layout:
            raise ValueError("Packed IntervalTable schema must include 'layout' list")
        k = len(layout)

        # enumerate columns in layout for indexing
        col = {name: i for i, name in enumerate(layout)}

        # check if table is empty
        if n == 0:
            # empty table; define empty `ndarray` with correct number of columns based on layout
            dtype = np.dtype(schema.get("packed_dtype", np.float64))
            buf = np.empty((0, k), dtype=dtype)

            start = buf[:, col["start"]] if "start" in col else np.empty((0,), dtype=dtype)
            end = buf[:, col["end"]] if "end" in col else np.empty((0,), dtype=dtype)
            prefix = buf[:, col["prefix_max_end"]] if "prefix_max_end" in col else np.empty((0,), dtype=dtype)

            extras = {name: buf[:, col[name]] for name in layout if name not in ("start", "end", "prefix_max_end")}
            return cls(_buf=buf, _start=start, _end=end, _prefix_max_end=prefix, _extras=extras, _meta=schema, _col=col)
        
        # load packed data
        buf = np.load(os.path.join(in_dir, schema["files"][packed_key]), mmap_mode=mmap_mode)

        # validate shape of packed data
        if buf.shape[0] != n:
            raise AssertionError(f"expected packed rows {n}, got {buf.shape[0]}")
        if buf.shape[1] != k:
            raise AssertionError(f"expected packed columns {k} based on layout, got {buf.shape[1]}")

        # ensure required columns are present in layout
        for req in ("start", "end", "prefix_max_end"):
            if req not in col: raise ValueError(f"layout missing required column '{req}'")

        # extract required data into packed array
        start = buf[:, col["start"]]
        end = buf[:, col["end"]]
        prefix = buf[:, col["prefix_max_end"]]

        # package additional data
        extras: Dict[str, np.ndarray] = {}
        for name in layout:
            if name in ("start", "end", "prefix_max_end"):
                continue
            extras[name] = buf[:, col[name]]

        # return `IntervalTable` object
        return cls(
            _buf=buf,
            _start=start,
            _end=end,
            _prefix_max_end=prefix,
            _extras=extras,
            _meta=schema,
            _col=col,
        )
    
    def __row_from_index(self, i: int) -> Tuple:
        # initiate row data with start and end times
        row = [float(self._start[i]), float(self._end[i])]

        # add extra columns in order defined by layout
        for safe, arr in sorted(self._extras.items(), key=lambda x: self._col[x[0]]):  
            col_meta = self._meta.get("columns", {}).get(safe, {})

            if "vocab" in col_meta:
                # if column has vocab, decode integer value to string
                vocab: dict = col_meta["vocab"]
                row.append(vocab.get(str(int(arr[i])), None))
            else:
                # otherwise, just append the raw value 
                row.append(float(arr[i]))
                
        # return row as tuple
        return tuple(row)
    
    def __iter__(self):
        for i in range(len(self._start)):
            yield self.__row_from_index(i)

    def iter_rows_raw(self, t: float, t_max: float = np.Inf, include_current: bool = False):        
        # find starting index using prefix max end times
        idx = np.searchsorted(self._prefix_max_end, t, side="left")

        # iterate through intervals starting from `idx`
        for i in range(idx, len(self._start)):
            # get interval start time
            s = self._start[i]
            
            # early stop if start time is beyond `t_max`
            if s > t_max + 1e-6:
                break
            
            # get interval end time
            e = self._end[i]

            # ignore if interval ends before time `t`
            if e < t - 1e-6:
                continue

            # if reached, t <= e and s is either current or in the future;
            # return raw row data if starts after time `t` or is current and `include_current` is True
            if include_current or s > t - 1e-6:
                yield self.__row_from_index(i)


    def __len__(self):
        return len(self._start)
    
    def lookup_intervals(self, t: float, t_max: float = np.Inf, include_current: bool = False) -> List[Tuple[Interval, ...]]:
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

        # check if there is any data stored in the table
        if len(self._start) == 0:
            # no data; return empty list
            return []

        # check if time `t` is beyond the data in the table
        if t > self._prefix_max_end[-1] + 1e-6:
            # time `t` is beyond the end of all intervals; return empty list
            return []

        # search for matching interval index 
        idx = np.searchsorted(self._prefix_max_end, t, side="left")
        
        # iterate through intervals starting from idx until we go past `t_max` 
        intervals: List[Tuple[Interval, ...]] = []
        for i in range(idx, len(self._start)):
            # early stop if start time is beyond `t_max`
            if self._start[i] > t_max + 1e-6: break
            if self._end[i] < t - 1e-6: continue  # skip intervals that end before time `t`

            # check if starts after time `t` or is current and `include_current` is True
            if include_current or self._start[i] > t - 1e-6:
                # valid interval found

                # get row data for this interval index
                row = self.__row_from_index(i)

                # adjust time interval to start at time `t` if needed
                t_start = max(self._start[i], t) if include_current else self._start[i]

                # package interval data 
                interval = (Interval(float(t_start), float(self._end[i])), *row[2:])

                # add to list of intervals to return
                intervals.append(interval)

        # return list of intervals
        return intervals   

    def lookup_interval(self, t: float, t_max: float = np.Inf, include_current: bool = False) -> Tuple[Interval, ...]:
        """
        Returns the interval that contains time `t` if it exists. 
        If `include_current` is True, also includes the interval that starts at time `t` if it exists.
        """
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        if not isinstance(t_max, (int, float)) or t_max < 0.0:
            raise ValueError("time t_max must be a non-negative number")
        if t > t_max + 1e-6:
            raise ValueError("time t must be less than or equal to time t_max")

        # check if there is any data stored in the table
        if len(self._start) == 0:
            # no data; return None
            return (None for _ in range(2 + len(self._extras)))  # (Interval, *extras)

        # check if time `t` is beyond the data in the table
        if t > self._prefix_max_end[-1] + 1e-6:
            # time `t` is beyond the end of all intervals; return None
            return (None for _ in range(2 + len(self._extras)))  # (Interval, *extras)

        # search for matching interval index 
        idx = np.searchsorted(self._prefix_max_end, t, side="left")

        # iterate through intervals starting from idx until we go past `t_max` 
        for i in range(idx, len(self._start)):
            # early stop if start time is beyond `t_max`
            if self._start[i] > t_max + 1e-6: break

            if include_current or self._start[i] > t - 1e-6:            
                # first interval that contains time t is found

                # get row data for this interval index
                row = self.__row_from_index(i)

                # adjust time interval to start at time `t` if needed
                t_start = max(self._start[i], t) if include_current else self._start[i]
                
                # package and reuturn interval data 
                return (Interval(float(t_start), float(self._end[i])), *row[2:])

        # fallback, return None
        return (None for _ in range(2 + len(self._extras)))  # (Interval, *extras)
    
@dataclass
class ConnectivityTable(AbstractTable):
    _adj: np.ndarray                # memmap (N,K,K) adjacency matrices for N intervals and K agents
    _start: np.ndarray              # view (N,)
    _end: np.ndarray                # view (N,)
    _prefix_max_end: np.ndarray     # view (N,)
    _meta: Dict[str, Any]           # metadata dictionary
    _col: Dict[str, int]            # name -> column index

    # @classmethod
    # def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "IntervalTable":
    #     # validate inputs 
    #     super().from_schema(schema, mmap_mode=mmap_mode)

    #     # extract in_dir from schema
    #     in_dir = schema.get("dir", None)

    #     # ensure required fields are in layout
    #     if in_dir is None: raise ValueError("schema missing 'dir'")
    #     if "files" in schema and "intervals" in schema["files"]:
    #         packed_key = "intervals"
    #     elif "files" in schema and "packed" in schema["files"]:
    #         packed_key = "packed"  # optional backward compat name
    #     else:
    #         raise ValueError("Packed IntervalTable schema must include files['intervals']")
        
    #     # get number of rows in table from schema
    #     n = int(schema["n"])
    #     layout = schema.get("layout", None)
    #     if not layout:
    #         raise ValueError("Packed IntervalTable schema must include 'layout' list")
    #     k = len(layout)

    #     # enumerate columns in layout for indexing
    #     col = {name: i for i, name in enumerate(layout)}

    #     # check if table is empty
    #     if n == 0:
    #         # empty table; define empty `ndarray` with correct number of columns based on layout
    #         dtype = np.dtype(schema.get("packed_dtype", np.float64))
    #         buf = np.empty((0, k), dtype=dtype)

    #         start = buf[:, col["start"]] if "start" in col else np.empty((0,), dtype=dtype)
    #         end = buf[:, col["end"]] if "end" in col else np.empty((0,), dtype=dtype)
    #         prefix = buf[:, col["prefix_max_end"]] if "prefix_max_end" in col else np.empty((0,), dtype=dtype)

    #         extras = {name: buf[:, col[name]] for name in layout if name not in ("start", "end", "prefix_max_end")}
    #         return cls(_buf=buf, _start=start, _end=end, _prefix_max_end=prefix, _extras=extras, _meta=schema, _col=col)
        
    #     # load packed data
    #     buf = np.load(os.path.join(in_dir, schema["files"][packed_key]), mmap_mode=mmap_mode)

    #     # validate shape of packed data
    #     if buf.shape[0] != n:
    #         raise AssertionError(f"expected packed rows {n}, got {buf.shape[0]}")
    #     if buf.shape[1] != k:
    #         raise AssertionError(f"expected packed columns {k} based on layout, got {buf.shape[1]}")

    #     # ensure required columns are present in layout
    #     for req in ("start", "end", "prefix_max_end"):
    #         if req not in col: raise ValueError(f"layout missing required column '{req}'")

    #     # extract required data into packed array
    #     start = buf[:, col["start"]]
    #     end = buf[:, col["end"]]
    #     prefix = buf[:, col["prefix_max_end"]]

    #     # package additional data
    #     extras: Dict[str, np.ndarray] = {}
    #     for name in layout:
    #         if name in ("start", "end", "prefix_max_end"):
    #             continue
    #         extras[name] = buf[:, col[name]]

    #     # return `IntervalTable` object
    #     return cls(
    #         _buf=buf,
    #         _start=start,
    #         _end=end,
    #         _prefix_max_end=prefix,
    #         _extras=extras,
    #         _meta=schema,
    #         _col=col,
    #     )

@dataclass
class AccessTable(AbstractTable):
    """
    Memmap-backed access table (ragged offsets + packed rows).

    Files:
      - offsets.npy (T+1,)
      - rows.npy    (M,K) where columns are defined by schema["layout"].
    """
    _offsets: np.ndarray              # (T+1,) int64 memmap
    _rows: np.ndarray                 # (M,K)  memmap
    _col: Dict[str, int]              # name -> column index in _rows
    _extras: Dict[str, np.ndarray]    # views into _rows for extra columns (by safe-name)
    _meta: Dict[str, Any]

    _t: np.ndarray
    _t_idx: np.ndarray
    _lat: np.ndarray
    _lon: np.ndarray
    _grid_idx: np.ndarray
    _gp_idx: np.ndarray

    @classmethod
    def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "AccessTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)

        # extract in_dir from schema
        in_dir = schema.get("dir", None)

        # ensure required fields are in layout
        if in_dir is None:
            raise ValueError("schema missing 'dir'")
        if "files" not in schema:
            raise ValueError("Packed AccessTable schema must include 'files' with 'offsets' and 'rows'")
        files = schema.get("files", {})
        if "offsets" not in files or "rows" not in files:
            raise ValueError("Packed AccessTable schema must include files['offsets'] and files['rows']")

        # get number of rows in table from schema
        n,k = int(schema["shape"][0]), int(schema["shape"][1])
        layout = schema.get("layout", None)
        if not layout:
            raise ValueError("Packed IntervalTable schema must include 'layout' list")
        assert k == len(layout), f"expected {k} columns based on schema shape, got {len(layout)} in layout"

        # check if table is empty
        if n == 0:
            # empty table; define empty `ndarray` for offsets and rows
            offsets = np.empty((0,), dtype=np.int64)
            dtype = np.dtype(schema.get("packed_dtype", np.float64))
            rows = np.empty((0, k), dtype=dtype)
            t = rows[:, layout.index("t")] if "t" in layout else np.empty((0,), dtype=dtype)
            t_idx = rows[:, layout.index("t_index")] if "t_index" in layout else np.empty((0,), dtype=dtype)
            lat = rows[:, layout.index("lat_deg")] if "lat_deg" in layout else np.empty((0,), dtype=dtype)
            lon = rows[:, layout.index("lon_deg")] if "lon_deg" in layout else np.empty((0,), dtype=dtype)
            grid_idx = rows[:, layout.index("grid_index")] if "grid_index" in layout else np.empty((0,), dtype=dtype)
            gp_idx = rows[:, layout.index("GP_index")] if "GP_index" in layout else np.empty((0,), dtype=dtype) 
            extras = {name: rows[:, layout.index(name)] for name in layout if name not in ("t", "t_index", "lat_deg", "lon_deg", "grid_index", "GP_index")}

            return cls(
                _offsets=offsets,
                _rows=rows,
                _col={name: i for i, name in enumerate(layout)},
                _extras=extras,
                _meta=schema,
                _t=t,
                _t_idx=t_idx,
                _lat=lat,
                _lon=lon,
                _grid_idx=grid_idx,
                _gp_idx=gp_idx,
            )

        # load offsets and rows as memmaps
        offsets = np.load(os.path.join(in_dir, files["offsets"]), mmap_mode=mmap_mode)
        rows = np.load(os.path.join(in_dir, files["rows"]), mmap_mode=mmap_mode)
    
        # enumerate columns in layout for indexing
        col = {name: i for i, name in enumerate(layout)}

        # enlist required base cols
        required = ["t", "t_index", "lat_deg", "lon_deg", "grid_index", "GP_index"]
        
        # ensure required columns are present in layout
        missing = [r for r in required if r not in col]
        if missing: raise ValueError(f"layout missing required columns: {missing}")

        # basic shape checks
        if rows.shape[0] != n:
            raise AssertionError(f"expected rows {n}, got {rows.shape[0]}")
        if rows.shape[1] != k:
            raise AssertionError(f"expected columns {k} based on schema shape, got {rows.shape[1]}")
        if offsets.shape[0] != schema['n_steps'] + 1:
            raise AssertionError(f"expected offsets {schema['n_steps'] + 1}, got {offsets.shape[0]}")

        # extract required data into packed array
        t = rows[:, col["t"]]
        t_idx = rows[:, col["t_index"]]
        lat = rows[:, col["lat_deg"]]
        lon = rows[:, col["lon_deg"]]
        grid_idx = rows[:, col["grid_index"]]
        gp_idx = rows[:, col["GP_index"]]

        # package additional data
        base_set = set(required)
        extras: Dict[str, np.ndarray] = {}
        for name in layout:
            if name in base_set:
                continue
            extras[name] = rows[:, col[name]]

        # return `AccessTable` object
        return cls(
            _offsets=offsets,
            _rows=rows,
            _col=col,
            _extras=extras,
            _meta=schema,
            _t=t,
            _t_idx=t_idx,
            _lat=lat,
            _lon=lon,
            _grid_idx=grid_idx,
            _gp_idx=gp_idx,
        )

    def __len__(self):
        return len(self._t)

    def lookup_interval(self, t_start: float, t_end: float = np.Inf, include_extras : bool = True) -> Dict[str, np.ndarray]:
        # validate inputs
        if not isinstance(t_start, (int, float)) or t_start < 0.0:
            raise ValueError("time t_start must be a non-negative number")
        if not isinstance(t_end, (int, float)) or t_end < 0.0:
            raise ValueError("time t_end must be a non-negative number")
        if t_start > t_end + 1e-6:
            raise ValueError("time t_start must be less than or equal to time t_end")

        # check if there is any data
        if len(self._t) == 0:
            s_empty = slice(0, 0)
            return self.__rows_from_slice(s_empty, include_extras=include_extras)

        # get start and end indices for time range
        ti0 = self._time_to_index_floor(t_start)
        ti1 = self._time_to_index_floor(t_end)

        # check if start and end time are in the same time index bucket
        if ti0 == ti1:
            # if so, return the rows for that bucket
            return self.lookup_time(t_start, include_extras=include_extras)

        # get slice for time index range
        s = self._slice_for_index_range(ti0, ti1)

        # construct output dictionary
        out = self.__rows_from_slice(s, include_extras=include_extras)

        # filter out any rows that are outside the time range 
        if s.start != s.stop:
            mask = (t_start <= out["time [s]"]) & (out["time [s]"] <= t_end)
            for k in list(out.keys()):
                out[k] = out[k][mask]

        # return ouput
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
            "grid index": self._grid_idx[s].astype(int, copy=False),
            "GP index": self._gp_idx[s].astype(int, copy=False),
        }

        if include_extras:
            for safe, arr in self._extras.items():
                # safe is the key used in schema["columns"]
                col_meta = self._meta.get("columns", {}).get(safe, {})
                col_name = col_meta.get("col_name", safe)

                if "vocab" in col_meta:
                    # get vocab for column
                    vocab: dict = col_meta["vocab"]

                    # cast to codes if not already int
                    codes = arr[s].astype(np.int32, copy=False)  
                    
                    # decode using vocab
                    out[col_name] = np.array([vocab.get(str(int(c)), None) for c in codes])
                
                else:
                    # no vocab, return raw values as float
                    out[col_name] = arr[s].astype(float)

        # return output dictionary with arrays for each column
        return out

    def lookup_time(self, t: float, include_extras: bool = False) -> Dict[str, Any]:
        """
        Find nearest / first occurrence; but you still need the corresponding time index bucket.
        If you trust that each bucket has constant t value, you can map by search then use t_idx:
        """

        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time `t` must be a non-negative number")

        # check if there is any data    
        if len(self._t) == 0:
            s_empty = slice(0, 0)
            return self.__rows_from_slice(s_empty, include_extras=include_extras)

        # check if time `t` is beyond the data in the table
        if self._t[-1] < t - 1e-6:
            # if so, return last row
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
        # bounds check 
        if ti < 0 or ti >= (len(self._offsets) - 1):
            # time index is out of bounds; return empty slice
            return slice(0, 0)
        
        # compute slice for time index from offsets
        a = int(self._offsets[ti])
        b = int(self._offsets[ti + 1])

        # return slice object
        return slice(a, b)

    def __iter__(self):
        for i in range(len(self._t)):
            out = {
                "lat [deg]": float(self._lat[i]),
                "lon [deg]": float(self._lon[i]),
                "grid index": int(self._grid_idx[i]),
                "GP index": int(self._gp_idx[i]),
            }
            for safe, arr in self._extras.items():
                col_meta = self._meta.get("columns", {}).get(safe, {})
                col_name = col_meta.get("col_name", safe)

                if "vocab" in col_meta:
                    vocab: dict = col_meta["vocab"]
                    out[col_name] = vocab.get(str(int(arr[i])), None)
                else:
                    out[col_name] = float(arr[i])

            yield float(self._t[i]), out    
   
#     def __iter__(self):
#         """
#         Returns an iterator over the data
#         """
#         for i in range(len(self._t)):
            
#             out = {
#                 # "t_index": self._t_idx[i],
#                 "lat [deg]": float(self._lat[i]),
#                 "lon [deg]": float(self._lon[i]),
#                 "grid index": self._grid_idx[i],
#                 "GP index": self._gp_idx[i],
#             }
#             # add any extra columns
#             for k, arr in self._extras.items():
#                 col = self._meta["columns"].get(k, {}).get("col_name", k)                
#                 # check if column is encoded in metadata
#                 if 'vocab' in self._meta["columns"].get(k, {}):
#                     # decode column using vocab
#                     vocab : dict = self._meta["columns"][k]["vocab"]
#                     # add decoding for current row
#                     out[col] = vocab.get(str(arr[i]), None)
#                 else:
#                     # add raw value for current row
#                     out[col] = float(arr[i])

#             yield float(self._t[i]),out