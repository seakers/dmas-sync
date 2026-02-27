from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

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

    _extras_in_order: List[Tuple[str, np.ndarray]]
    _extras_cols_in_order: np.ndarray  # shape (E,), dtype=int64

    @classmethod
    def from_schema(cls, schema: Dict[str, Any], mmap_mode: str = "r") -> "IntervalTable":
        super().from_schema(schema, mmap_mode=mmap_mode)

        in_dir = schema.get("dir", None)
        if in_dir is None:
            raise ValueError("schema missing 'dir'")

        if "files" in schema and "intervals" in schema["files"]:
            packed_key = "intervals"
        elif "files" in schema and "packed" in schema["files"]:
            packed_key = "packed"
        else:
            raise ValueError("Packed IntervalTable schema must include files['intervals']")

        n = int(schema.get("n", 0))
        layout = schema.get("layout", None)
        if not layout:
            raise ValueError("Packed IntervalTable schema must include 'layout' list")
        k = len(layout)

        col = dict((name, i) for i, name in enumerate(layout))

        # Empty table
        if n == 0:
            dtype = np.dtype(schema.get("packed_dtype", np.float64))
            buf = np.empty((0, k), dtype=dtype)

            start = buf[:, col["start"]] if "start" in col else np.empty((0,), dtype=dtype)
            end = buf[:, col["end"]] if "end" in col else np.empty((0,), dtype=dtype)
            prefix = buf[:, col["prefix_max_end"]] if "prefix_max_end" in col else np.empty((0,), dtype=dtype)

            extras = {}
            extras_in_order = []
            extras_cols = []

            for name in layout:
                if name in ("start", "end", "prefix_max_end"):
                    continue
                arr = buf[:, col[name]]
                extras[name] = arr
                extras_in_order.append((name, arr))
                extras_cols.append(col[name])

            return cls(
                _buf=buf,
                _start=start,
                _end=end,
                _prefix_max_end=prefix,
                _extras=extras,
                _meta=schema,
                _col=col,
                _extras_in_order=extras_in_order,
                _extras_cols_in_order=np.array(extras_cols, dtype=np.int64),
            )

        # Load packed data (memmap)
        buf = np.load(os.path.join(in_dir, schema["files"][packed_key]), mmap_mode=mmap_mode)

        if buf.shape[0] != n:
            raise AssertionError("expected packed rows %d, got %d" % (n, buf.shape[0]))
        if buf.shape[1] != k:
            raise AssertionError("expected packed columns %d based on layout, got %d" % (k, buf.shape[1]))

        for req in ("start", "end", "prefix_max_end"):
            if req not in col:
                raise ValueError("layout missing required column '%s'" % req)

        start = buf[:, col["start"]]
        end = buf[:, col["end"]]
        prefix = buf[:, col["prefix_max_end"]]

        extras = {}
        extras_in_order = []
        extras_cols = []

        for name in layout:
            if name in ("start", "end", "prefix_max_end"):
                continue
            arr = buf[:, col[name]]
            extras[name] = arr
            extras_in_order.append((name, arr))
            extras_cols.append(col[name])

        return cls(
            _buf=buf,
            _start=start,
            _end=end,
            _prefix_max_end=prefix,
            _extras=extras,
            _meta=schema,
            _col=col,
            _extras_in_order=extras_in_order,
            _extras_cols_in_order=np.array(extras_cols, dtype=np.int64),
        )
    
    #     @classmethod
#     def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "IntervalTable":
#         # validate inputs 
#         super().from_schema(schema, mmap_mode=mmap_mode)

#         # extract in_dir from schema
#         in_dir = schema.get("dir", None)

#         # ensure required fields are in layout
#         if in_dir is None: raise ValueError("schema missing 'dir'")
#         if "files" in schema and "intervals" in schema["files"]:
#             packed_key = "intervals"
#         elif "files" in schema and "packed" in schema["files"]:
#             packed_key = "packed"  # optional backward compat name
#         else:
#             raise ValueError("Packed IntervalTable schema must include files['intervals']")
        
#         # get number of rows in table from schema
#         n = int(schema["n"])
#         layout = schema.get("layout", None)
#         if not layout:
#             raise ValueError("Packed IntervalTable schema must include 'layout' list")
#         k = len(layout)

#         # enumerate columns in layout for indexing
#         col = {name: i for i, name in enumerate(layout)}

#         # check if table is empty
#         if n == 0:
#             # empty table; define empty `ndarray` with correct number of columns based on layout
#             dtype = np.dtype(schema.get("packed_dtype", np.float64))
#             buf = np.empty((0, k), dtype=dtype)

#             start = buf[:, col["start"]] if "start" in col else np.empty((0,), dtype=dtype)
#             end = buf[:, col["end"]] if "end" in col else np.empty((0,), dtype=dtype)
#             prefix = buf[:, col["prefix_max_end"]] if "prefix_max_end" in col else np.empty((0,), dtype=dtype)

#             extras = {name: buf[:, col[name]] for name in layout if name not in ("start", "end", "prefix_max_end")}
#             return cls(_buf=buf, _start=start, _end=end, _prefix_max_end=prefix, _extras=extras, _meta=schema, _col=col)
        
#         # load packed data
#         buf = np.load(os.path.join(in_dir, schema["files"][packed_key]), mmap_mode=mmap_mode)

#         # validate shape of packed data
#         if buf.shape[0] != n:
#             raise AssertionError(f"expected packed rows {n}, got {buf.shape[0]}")
#         if buf.shape[1] != k:
#             raise AssertionError(f"expected packed columns {k} based on layout, got {buf.shape[1]}")

#         # ensure required columns are present in layout
#         for req in ("start", "end", "prefix_max_end"):
#             if req not in col: raise ValueError(f"layout missing required column '{req}'")

#         # extract required data into packed array
#         start = buf[:, col["start"]]
#         end = buf[:, col["end"]]
#         prefix = buf[:, col["prefix_max_end"]]

#         # package additional data
#         extras: Dict[str, np.ndarray] = {}
#         for name in layout:
#             if name in ("start", "end", "prefix_max_end"):
#                 continue
#             extras[name] = buf[:, col[name]]

#         # return `IntervalTable` object
#         return cls(
#             _buf=buf,
#             _start=start,
#             _end=end,
#             _prefix_max_end=prefix,
#             _extras=extras,
#             _meta=schema,
#             _col=col,
#         )

    def __len__(self) -> int:
        return int(self._start.shape[0])

    # ---------------------------------------------------------------------
    # Iterator (full decoding)
    # ---------------------------------------------------------------------

    def _row_decoded(self, i: int) -> Tuple:
        """
        Decodes vocab columns and converts values to Python floats/strings.
        Use only when you actually need decoded rows.
        """
        # initiate row data with start and end times
        row = [float(self._start[i]), float(self._end[i])]

        # add extra columns 
        columns_meta = self._meta.get("columns", {})
        for safe, arr in self._extras_in_order:
            col_meta = columns_meta.get(safe, {})
            
            if "vocab" in col_meta:
                # if column has vocab, decode integer value to string
                vocab = col_meta["vocab"]
                row.append(vocab.get(str(int(arr[i])), None))
            else:
                # otherwise, just append the raw value 
                row.append(float(arr[i]))

        # return row as tuple
        return tuple(row)

    def __iter__(self):
        for i in range(len(self)):
            yield self._row_decoded(i)

    # ---------------------------------------------------------------------
    # Iterator (no explicit decoding)
    # ---------------------------------------------------------------------

    def iter_rows_raw_fast(
        self,
        t: float,
        t_max: float = np.Inf,
        include_current: bool = False,
    ) -> Iterator[Tuple[Any, ...]]:
        """
        Fast generator that yields:
            (start, end, *extras)
        where each item is a numpy scalar (or python scalar after int()/float()).

        No vocab decoding, no per-row sorting, no per-element float() calls.
        """
        idx = int(np.searchsorted(self._prefix_max_end, t, side="left"))
        start = self._start
        end = self._end
        extras_arrays = [arr for _, arr in self._extras_in_order]

        for i in range(idx, len(start)):
            s = start[i]
            if s > t_max + 1e-6:
                break

            e = end[i]
            if e < t - 1e-6:
                continue

            if include_current or s > t - 1e-6:
                # numpy scalars are fine; caller can cast
                yield (s, e, *[arr[i] for arr in extras_arrays])

    def iter_rows_packed(
        self,
        t: float,
        t_max: float = np.Inf,
        include_current: bool = False,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Fastest generator: yields (row_index, packed_row_view).
        packed_row_view is a (K,) view into the memmap row (no copy).

        Use this when the caller can index into row by _col / known offsets.
        """
        idx = int(np.searchsorted(self._prefix_max_end, t, side="left"))
        start = self._start
        end = self._end
        buf = self._buf

        for i in range(idx, len(start)):
            s = start[i]
            if s > t_max + 1e-6:
                break

            e = end[i]
            if e < t - 1e-6:
                continue

            if include_current or s > t - 1e-6:
                yield i, buf[i]

    def iter_components_packed(
        self,
        t: float,
        t_max: float = np.Inf,
        include_current: bool = False,
        components_slice: Optional[slice] = None,
    ) -> Iterator[Tuple[Any, Any, np.ndarray]]:
        """
        Convenience fast iterator for your comms use-case:
        yields (start, end, components_view)

        components_view is a view into buf[i, components_slice]
        If components_slice is None, it defaults to extras columns in order.

        Avoids allocating tuples for all extras when you only care about them as a vector.
        """
        idx = int(np.searchsorted(self._prefix_max_end, t, side="left"))
        start = self._start
        end = self._end
        buf = self._buf

        if components_slice is None:
            # If your extras are exactly the component indices, this is what you want:
            # components are columns at _extras_cols_in_order.
            cols = self._extras_cols_in_order
        else:
            cols = None  # use slice below

        for i in range(idx, len(start)):
            s = start[i]
            if s > t_max + 1e-6:
                break

            e = end[i]
            if e < t - 1e-6:
                continue

            if include_current or s > t - 1e-6:
                if cols is not None:
                    # Advanced indexing produces a copy; avoid if you can use a slice.
                    # If extras are contiguous in layout, prefer passing components_slice.
                    comp = buf[i, cols]
                else:
                    comp = buf[i, components_slice]
                yield s, e, comp   

#     def iter_rows_raw(self, t: float, t_max: float = np.Inf, include_current: bool = False):        
#         # find starting index using prefix max end times
#         idx = np.searchsorted(self._prefix_max_end, t, side="left")

#         # iterate through intervals starting from `idx`
#         for i in range(idx, len(self._start)):
#             # get interval start time
#             s = self._start[i]
            
#             # early stop if start time is beyond `t_max`
#             if s > t_max + 1e-6:
#                 break
            
#             # get interval end time
#             e = self._end[i]

#             # ignore if interval ends before time `t`
#             if e < t - 1e-6:
#                 continue

#             # if reached, t <= e and s is either current or in the future;
#             # return raw row data if starts after time `t` or is current and `include_current` is True
#             if include_current or s > t - 1e-6:
#                 yield self.__row_from_index(i)

    # ---------------------------------------------------------------------
    # Interval lookups
    # ---------------------------------------------------------------------

    def lookup_intervals(self,
                         t: float,
                         t_max: float = np.Inf,
                         include_current: bool = False,
                        ) -> List[Tuple[Interval, ...]]:
        """
        Returns a list of intervals that start after or at time `t` and end before or at time `t_max`. 
        If `include_current` is True, also includes the interval that contains time `t` if it exists.
        """
        # validate inputs
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time t must be a non-negative number")
        if not isinstance(t_max, (int, float)) or t_max < 0.0:
            raise ValueError("time t_max must be a non-negative number")
        if t > t_max + 1e-6:
            raise ValueError("time t must be <= time t_max")

        # check if there is any data stored in the table
        if len(self._start) == 0:
            return [] # no data; return empty list
        
        # check if time `t` is beyond the data in the table
        if t > self._prefix_max_end[-1] + 1e-6:
            return [] # time `t` is beyond the end of all intervals; return empty list

        # search for matching interval index 
        idx = int(np.searchsorted(self._prefix_max_end, t, side="left"))

        # iterate through intervals starting from idx until we go past `t_max` 
        intervals = []
        for i in range(idx, len(self._start)):
            # early stop if start time is beyond `t_max`
            if self._start[i] > t_max + 1e-6: break

            # skip intervals that end before time `t`
            if self._end[i] < t - 1e-6: continue

            # check if starts after time `t` or is current and `include_current` is True
            if include_current or self._start[i] > t - 1e-6:
                # get row data for this interval index
                row = self._row_decoded(i)

                # adjust time interval to start at time `t` if needed
                t_start = max(float(self._start[i]), t) if include_current else float(self._start[i])
                
                # package interval data 
                interval = (Interval(t_start, float(self._end[i])), *row[2:])
                
                # add to list of intervals to return
                intervals.append(interval)

        # return list of intervals
        return intervals

    def lookup_interval(self,
                        t: float,
                        t_max: float = np.Inf,
                        include_current: bool = False,
                    ) -> Tuple[Any, ...]:
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
            raise ValueError("time t must be <= time t_max")

        # check if there is any data stored in the table
        if len(self._start) == 0:
            return tuple([None] * (2 + len(self._extras)))
        
        # check if time `t` is beyond the data in the table
        if t > self._prefix_max_end[-1] + 1e-6:
            return tuple([None] * (2 + len(self._extras)))

        # search for matching interval index
        idx = int(np.searchsorted(self._prefix_max_end, t, side="left"))

        # iterate through intervals starting from idx until `t_max` is reached
        for i in range(idx, len(self._start)):
            # early stop if start time is beyond `t_max`
            if self._start[i] > t_max + 1e-6: break

            # check if interval contains or is after time `t` 
            if include_current or self._start[i] > t - 1e-6:
                # get row data for this interval index
                row = self._row_decoded(i)

                # adjust time interval to start at time `t` if needed
                t_start = max(float(self._start[i]), t) if include_current else float(self._start[i])
                
                # package and return interval data 
                return (Interval(t_start, float(self._end[i])), *row[2:])

        # fallback, return None
        return tuple([None] * (2 + len(self._extras)))       
    
    
@dataclass
class ConnectivityTable(AbstractTable):
    _adj: np.ndarray                # memmap (N,K,K) adjacency matrices for N intervals and K agents
    _start: np.ndarray              # view (N,)
    _end: np.ndarray                # view (N,)
    _prefix_max_end: np.ndarray     # view (N,)
    _meta: Dict[str, Any]           # metadata dictionary
    _col: Dict[str, int]            # name -> column index

    # TODO develop when non-relay comms are selected for comms use-case
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

FilterSpec = Union[
    Any,                              # equality: value
    Set[Any], List[Any], Tuple[Any],   # membership: {v1,v2} or [v1,v2]
    Tuple[str, Any, Any],              # ("between", lo, hi)
]

@dataclass
class AccessTable(AbstractTable):
    """
    Memmap-backed access table (ragged offsets + packed rows).

    Files:
      - offsets.npy (T+1,)
      - rows.npy    (M,K) where columns are defined by schema["layout"].

    Notes on performance:
      - Filtering is applied BEFORE decoding vocab columns.
      - Vocab decoding uses np.take on a small object array (fast).
      - decode=False returns vocab columns as integer codes (fastest).
      - columns=... allows projection to only the fields you need.
    """
    _offsets: np.ndarray              # (T+1,) int64 memmap
    _rows: np.ndarray                 # (M,K)  memmap
    _col: Dict[str, int]              # safe-name -> column index in _rows
    _extras: Dict[str, np.ndarray]    # safe-name -> view into _rows col
    _meta: Dict[str, Any]

    # Required base columns as views into _rows
    _t: np.ndarray
    _t_idx: np.ndarray
    _lat: np.ndarray
    _lon: np.ndarray
    _grid_idx: np.ndarray
    _gp_idx: np.ndarray

    # Vocab acceleration
    _vocab_arr: Dict[str, np.ndarray]       # safe-name -> object array indexed by code
    _rev_vocab: Dict[str, Dict[Any, int]]   # safe-name -> {label -> code}

    # Optional: share vocab arrays across instances (useful if schema objects are reused)
    _VOCAB_CACHE = {}  # type: Dict[Tuple[str, int], np.ndarray]

    @classmethod
    def _get_vocab_arr_cached(cls, safe: str, vocab: Dict[str, Any]) -> np.ndarray:
        """
        Build (or reuse) an object array such that arr[code] -> label.
        Cache key uses id(vocab) which works well if schema/vocab dict is reused across sats.
        """
        key = (safe, id(vocab))
        arr = cls._VOCAB_CACHE.get(key)
        if arr is not None:
            return arr

        if not vocab:
            arr = np.empty((0,), dtype=object)
            cls._VOCAB_CACHE[key] = arr
            return arr

        codes = [int(k) for k in vocab.keys()]
        max_code = max(codes) if codes else -1
        arr = np.empty((max_code + 1,), dtype=object)
        arr[:] = None
        for k, v in vocab.items():
            arr[int(k)] = v

        cls._VOCAB_CACHE[key] = arr
        return arr

    @classmethod
    def from_schema(cls, schema: Dict[str, Any], mmap_mode: str = "r") -> "AccessTable":
        # validate inputs
        super().from_schema(schema, mmap_mode=mmap_mode)

        in_dir = schema.get("dir", None)
        if in_dir is None:
            raise ValueError("schema missing 'dir'")

        if "files" not in schema:
            raise ValueError("Packed AccessTable schema must include 'files' with 'offsets' and 'rows'")
        files = schema.get("files", {})
        if "offsets" not in files or "rows" not in files:
            raise ValueError("Packed AccessTable schema must include files['offsets'] and files['rows']")

        n = int(schema["shape"][0])
        k = int(schema["shape"][1])
        layout = schema.get("layout", None)
        if not layout:
            raise ValueError("Packed AccessTable schema must include 'layout' list")
        if k != len(layout):
            raise AssertionError("expected %d columns based on schema shape, got %d in layout"
                                 % (k, len(layout)))

        required = ["t", "t_index", "lat_deg", "lon_deg", "grid_index", "GP_index"]
        base_set = set(required)

        # Empty table fast path
        if n == 0:
            offsets = np.empty((0,), dtype=np.int64)
            dtype = np.dtype(schema.get("packed_dtype", np.float64))
            rows = np.empty((0, k), dtype=dtype)

            col = dict((name, i) for i, name in enumerate(layout))

            def empty_view(name: str) -> np.ndarray:
                return rows[:, col[name]] if name in col else np.empty((0,), dtype=dtype)

            t = empty_view("t")
            t_idx = empty_view("t_index")
            lat = empty_view("lat_deg")
            lon = empty_view("lon_deg")
            grid_idx = empty_view("grid_index")
            gp_idx = empty_view("GP_index")

            extras = {}
            for name in layout:
                if name in base_set:
                    continue
                extras[name] = rows[:, col[name]]

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
                _vocab_arr={},
                _rev_vocab={},
            )

        # Load memmaps
        offsets = np.load(os.path.join(in_dir, files["offsets"]), mmap_mode=mmap_mode)
        rows = np.load(os.path.join(in_dir, files["rows"]), mmap_mode=mmap_mode)

        col = dict((name, i) for i, name in enumerate(layout))

        missing = [r for r in required if r not in col]
        if missing:
            raise ValueError("layout missing required columns: %s" % missing)

        if rows.shape[0] != n:
            raise AssertionError("expected rows %d, got %d" % (n, rows.shape[0]))
        if rows.shape[1] != k:
            raise AssertionError("expected columns %d based on schema shape, got %d" % (k, rows.shape[1]))
        if offsets.shape[0] != int(schema["n_steps"]) + 1:
            raise AssertionError("expected offsets %d, got %d" % (int(schema["n_steps"]) + 1, offsets.shape[0]))

        # Base views (no copy)
        t = rows[:, col["t"]]
        t_idx = rows[:, col["t_index"]]
        lat = rows[:, col["lat_deg"]]
        lon = rows[:, col["lon_deg"]]
        grid_idx = rows[:, col["grid_index"]]
        gp_idx = rows[:, col["GP_index"]]

        # Extras (views)
        extras = {}
        for name in layout:
            if name in base_set:
                continue
            extras[name] = rows[:, col[name]]

        # Build vocab accelerators
        vocab_arr = {}
        rev_vocab = {}
        columns_meta = schema.get("columns", {})

        for safe in extras.keys():
            col_meta = columns_meta.get(safe, {})
            vocab = col_meta.get("vocab", None)
            if not vocab:
                continue

            # Fast decoder array (possibly cached)
            arr = cls._get_vocab_arr_cached(safe, vocab)
            vocab_arr[safe] = arr

            # Reverse map for filter-by-label (small dict)
            rv = {}
            for k_str, label in vocab.items():
                rv[label] = int(k_str)
            rev_vocab[safe] = rv

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
            _vocab_arr=vocab_arr,
            _rev_vocab=rev_vocab,
        )

    def __len__(self) -> int:
        return int(self._t.shape[0])

    # ---------------------------
    # Public lookup API
    # ---------------------------

    def lookup_interval(
        self,
        t_start: float,
        t_end: float = np.Inf,
        include_extras: bool = True,
        filters: Optional[Dict[str, FilterSpec]] = None,
        columns: Optional[List[str]] = None,
        decode: bool = True,
        exact_time_filter: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Lookup rows in [t_start, t_end] with optional filters applied BEFORE decoding.

        filters:
          - {"safe_col": value}                equality
          - {"safe_col": {v1,v2}}              membership
          - {"safe_col": ("between", lo, hi)}  numeric range
          For vocab columns you may provide:
            - label (str) or set/list of labels -> auto-mapped to codes
            - integer code(s) directly
        columns:
          - projection list. Uses *output* names:
              base: "time [s]", "lat [deg]", "lon [deg]", "grid index", "GP index"
              extras: schema["columns"][safe]["col_name"] if present else safe
            You can also include safe names for extras.
        decode:
          - if True, vocab extras are decoded to labels (strings)
          - if False, vocab extras are returned as int codes (fast)
        exact_time_filter:
          - if True, apply (t_start <= t <= t_end) at row level.
            This is usually needed because bucket slices may include boundary rows.
        """
        # validate
        if not isinstance(t_start, (int, float)) or t_start < 0.0:
            raise ValueError("time t_start must be a non-negative number")
        if not isinstance(t_end, (int, float)) or t_end < 0.0:
            raise ValueError("time t_end must be a non-negative number")
        if t_start > t_end + 1e-6:
            raise ValueError("time t_start must be <= time t_end")

        if len(self._t) == 0:
            return self.__rows_from_slice(slice(0, 0), include_extras, columns, decode, None)

        ti0 = self._time_to_index_floor(t_start)
        ti1 = self._time_to_index_floor(t_end)

        if ti0 == ti1:
            s = self.__slice_for_time_index(ti0)
        else:
            s = self._slice_for_index_range(ti0, ti1)

        if s.start == s.stop:
            return self.__rows_from_slice(s, include_extras, columns, decode, None)

        # Build mask inside the slice BEFORE decoding
        mask = self._build_mask(
            s=s,
            t_start=t_start,
            t_end=t_end,
            filters=filters,
            exact_time_filter=exact_time_filter,
        )

        return self.__rows_from_slice(s, include_extras, columns, decode, mask)

    def lookup_time(
        self,
        t: float,
        include_extras: bool = False,
        filters: Optional[Dict[str, FilterSpec]] = None,
        columns: Optional[List[str]] = None,
        decode: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Find nearest/first occurrence and return its bucket, with optional filtering.
        """
        if not isinstance(t, (int, float)) or t < 0.0:
            raise ValueError("time `t` must be a non-negative number")

        if len(self._t) == 0:
            return self.__rows_from_slice(slice(0, 0), include_extras, columns, decode, None)

        if float(self._t[-1]) < t - 1e-6:
            s_last = slice(len(self._t) - 1, len(self._t))
            return self.__rows_from_slice(s_last, include_extras, columns, decode, None)

        i = int(np.searchsorted(self._t, t, side="left"))
        ti = int(self._t_idx[i])
        s = self.__slice_for_time_index(ti)

        if s.start == s.stop:
            return self.__rows_from_slice(s, include_extras, columns, decode, None)

        mask = self._build_mask(
            s=s,
            t_start=-np.Inf,
            t_end=np.Inf,
            filters=filters,
            exact_time_filter=False,
        )
        return self.__rows_from_slice(s, include_extras, columns, decode, mask)

    # ---------------------------
    # Core helpers
    # ---------------------------

    def _time_to_index_floor(self, t: float) -> int:
        dt = float(self._meta["time_step"])
        if np.isinf(t):
            return int(len(self._offsets) - 1)
        return int(t // dt)

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
        T = int(len(self._offsets) - 1)
        if ti < 0:
            return 0
        if ti > T - 1:
            return T - 1
        return int(ti)

    def __slice_for_time_index(self, ti: int) -> slice:
        if ti < 0 or ti >= (len(self._offsets) - 1):
            return slice(0, 0)
        a = int(self._offsets[ti])
        b = int(self._offsets[ti + 1])
        return slice(a, b)

    def _build_mask(
        self,
        s: slice,
        t_start: float,
        t_end: float,
        filters: Optional[Dict[str, FilterSpec]],
        exact_time_filter: bool,
    ) -> Optional[np.ndarray]:
        """
        Build a boolean mask for rows within slice s.
        Applied BEFORE decoding to avoid wasted work.

        Note: slice s already narrows to time *buckets*; exact_time_filter
        applies row-level [t_start, t_end] bounds when needed.
        """
        n = int(s.stop - s.start)
        if n <= 0:
            return None

        mask = None  # type: Optional[np.ndarray]

        # Fine-grained time bounds within the bucket slice (often needed)
        if exact_time_filter and (not np.isinf(t_start) or not np.isinf(t_end)):
            tview = self._t[s]
            m = (tview >= t_start) & (tview <= t_end)
            mask = m if mask is None else (mask & m)
            if mask is not None and (not mask.any()):
                return mask

        if not filters:
            return mask

        for key, spec in filters.items():
            arr, is_vocab = self._resolve_filter_array(key, s)

            m = self._mask_for_spec(key, arr, is_vocab, spec)
            mask = m if mask is None else (mask & m)

            if mask is not None and (not mask.any()):
                return mask

        return mask

    def _resolve_filter_array(self, key: str, s: slice) -> Tuple[np.ndarray, bool]:
        """
        Resolve a filter key to an array view inside slice s.
        Recommended: pass safe names for extras ("instrument_code", etc.).
        """
        if key in self._extras:
            return self._extras[key][s], (key in self._vocab_arr)

        # allow filtering on base fields via safe-ish names
        if key in ("t", "time [s]"):
            return self._t[s], False
        if key in ("lat_deg", "lat [deg]"):
            return self._lat[s], False
        if key in ("lon_deg", "lon [deg]"):
            return self._lon[s], False
        if key in ("grid_index", "grid index"):
            return self._grid_idx[s], False
        if key in ("GP_index", "GP index"):
            return self._gp_idx[s], False

        raise KeyError("Unknown filter column: %s" % key)

    def _mask_for_spec(self, safe: str, arr: np.ndarray, is_vocab: bool, spec: FilterSpec) -> np.ndarray:
        """
        Convert a FilterSpec to a boolean mask over arr (already sliced).
        """
        # range
        if isinstance(spec, tuple) and len(spec) == 3 and spec[0] == "between":
            lo = spec[1]
            hi = spec[2]
            return (arr >= lo) & (arr <= hi)

        # membership
        if isinstance(spec, (set, list, tuple)) and not (isinstance(spec, tuple) and len(spec) == 3 and spec[0] == "between"):
            if is_vocab:
                codes = self._labels_or_codes_to_codes(safe, spec)
                return np.isin(arr.astype(np.int32, copy=False), codes)
            return np.isin(arr, spec)

        # equality
        if is_vocab:
            code = self._label_or_code_to_code(safe, spec)
            return arr.astype(np.int32, copy=False) == code
        return arr == spec

    def _label_or_code_to_code(self, safe: str, x: Any) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        rv = self._rev_vocab.get(safe, None)
        if rv is None:
            raise KeyError("No reverse vocab for column: %s" % safe)
        if x not in rv:
            raise KeyError("Unknown vocab label %r for column %s" % (x, safe))
        return int(rv[x])

    def _labels_or_codes_to_codes(self, safe: str, xs: Union[Set[Any], List[Any], Tuple[Any]]) -> np.ndarray:
        out = []  # type: List[int]
        for x in xs:
            out.append(self._label_or_code_to_code(safe, x))
        return np.array(out, dtype=np.int32)

    # ---------------------------
    # Row materialization (fast)
    # ---------------------------

    def __rows_from_slice(
        self,
        s: slice,
        include_extras: bool,
        columns: Optional[List[str]],
        decode: bool,
        mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Materialize rows for slice s with optional boolean mask, projection, and vocab decoding.
        Avoids expensive work when possible:
          - applies mask before decoding
          - avoids astype(float) copies unless needed
        """
        def sel(a: np.ndarray) -> np.ndarray:
            v = a[s]
            if mask is None:
                return v
            return v[mask]

        want = None  # type: Optional[set]
        if columns is not None:
            want = set(columns)

        out = {}  # type: Dict[str, Any]

        # Base columns
        if want is None or "time [s]" in want:
            out["time [s]"] = sel(self._t)
        if want is None or "lat [deg]" in want:
            out["lat [deg]"] = sel(self._lat)
        if want is None or "lon [deg]" in want:
            out["lon [deg]"] = sel(self._lon)
        if want is None or "grid index" in want:
            out["grid index"] = sel(self._grid_idx).astype(np.int32, copy=False)
        if want is None or "GP index" in want:
            out["GP index"] = sel(self._gp_idx).astype(np.int32, copy=False)

        if not include_extras:
            return out

        columns_meta = self._meta.get("columns", {})

        for safe, arr_full in self._extras.items():
            col_meta = columns_meta.get(safe, {})
            col_name = col_meta.get("col_name", safe)

            # projection: accept either display name or safe name for extras
            if want is not None and (col_name not in want and safe not in want):
                continue

            arr = sel(arr_full)

            # vocab?
            if safe in self._vocab_arr:
                codes = arr.astype(np.int32, copy=False)
                if decode:
                    out[col_name] = np.take(self._vocab_arr[safe], codes)
                else:
                    out[col_name] = codes
            else:
                out[col_name] = arr

        return out

    # ---------------------------
    # Iteration
    # ---------------------------

    def __iter__(self):
        """
        Iterates row-by-row (still slower than vectorized methods).
        Uses vocab arrays for O(1) decode per row.
        """
        columns_meta = self._meta.get("columns", {})
        for i in range(len(self._t)):
            out = {
                "lat [deg]": float(self._lat[i]),
                "lon [deg]": float(self._lon[i]),
                "grid index": int(self._grid_idx[i]),
                "GP index": int(self._gp_idx[i]),
            }
            for safe, arr in self._extras.items():
                col_meta = columns_meta.get(safe, {})
                col_name = col_meta.get("col_name", safe)

                if safe in self._vocab_arr:
                    code = int(arr[i])
                    va = self._vocab_arr[safe]
                    out[col_name] = va[code] if 0 <= code < len(va) else None
                else:
                    out[col_name] = float(arr[i])

            yield float(self._t[i]), out

# @dataclass
# class AccessTable(AbstractTable):
#     """
#     Memmap-backed access table (ragged offsets + packed rows).

#     Files:
#       - offsets.npy (T+1,)
#       - rows.npy    (M,K) where columns are defined by schema["layout"].
#     """
#     _offsets: np.ndarray              # (T+1,) int64 memmap
#     _rows: np.ndarray                 # (M,K)  memmap
#     _col: Dict[str, int]              # name -> column index in _rows
#     _extras: Dict[str, np.ndarray]    # views into _rows for extra columns (by safe-name)
#     _meta: Dict[str, Any]

#     _t: np.ndarray
#     _t_idx: np.ndarray
#     _lat: np.ndarray
#     _lon: np.ndarray
#     _grid_idx: np.ndarray
#     _gp_idx: np.ndarray

#     @classmethod
#     def from_schema(cls, schema: Dict, mmap_mode: str = "r") -> "AccessTable":
#         # validate inputs
#         super().from_schema(schema, mmap_mode=mmap_mode)

#         # extract in_dir from schema
#         in_dir = schema.get("dir", None)

#         # ensure required fields are in layout
#         if in_dir is None:
#             raise ValueError("schema missing 'dir'")
#         if "files" not in schema:
#             raise ValueError("Packed AccessTable schema must include 'files' with 'offsets' and 'rows'")
#         files = schema.get("files", {})
#         if "offsets" not in files or "rows" not in files:
#             raise ValueError("Packed AccessTable schema must include files['offsets'] and files['rows']")

#         # get number of rows in table from schema
#         n,k = int(schema["shape"][0]), int(schema["shape"][1])
#         layout = schema.get("layout", None)
#         if not layout:
#             raise ValueError("Packed IntervalTable schema must include 'layout' list")
#         assert k == len(layout), f"expected {k} columns based on schema shape, got {len(layout)} in layout"

#         # check if table is empty
#         if n == 0:
#             # empty table; define empty `ndarray` for offsets and rows
#             offsets = np.empty((0,), dtype=np.int64)
#             dtype = np.dtype(schema.get("packed_dtype", np.float64))
#             rows = np.empty((0, k), dtype=dtype)
#             t = rows[:, layout.index("t")] if "t" in layout else np.empty((0,), dtype=dtype)
#             t_idx = rows[:, layout.index("t_index")] if "t_index" in layout else np.empty((0,), dtype=dtype)
#             lat = rows[:, layout.index("lat_deg")] if "lat_deg" in layout else np.empty((0,), dtype=dtype)
#             lon = rows[:, layout.index("lon_deg")] if "lon_deg" in layout else np.empty((0,), dtype=dtype)
#             grid_idx = rows[:, layout.index("grid_index")] if "grid_index" in layout else np.empty((0,), dtype=dtype)
#             gp_idx = rows[:, layout.index("GP_index")] if "GP_index" in layout else np.empty((0,), dtype=dtype) 
#             extras = {name: rows[:, layout.index(name)] for name in layout if name not in ("t", "t_index", "lat_deg", "lon_deg", "grid_index", "GP_index")}

#             return cls(
#                 _offsets=offsets,
#                 _rows=rows,
#                 _col={name: i for i, name in enumerate(layout)},
#                 _extras=extras,
#                 _meta=schema,
#                 _t=t,
#                 _t_idx=t_idx,
#                 _lat=lat,
#                 _lon=lon,
#                 _grid_idx=grid_idx,
#                 _gp_idx=gp_idx,
#             )

#         # load offsets and rows as memmaps
#         offsets = np.load(os.path.join(in_dir, files["offsets"]), mmap_mode=mmap_mode)
#         rows = np.load(os.path.join(in_dir, files["rows"]), mmap_mode=mmap_mode)
    
#         # enumerate columns in layout for indexing
#         col = {name: i for i, name in enumerate(layout)}

#         # enlist required base cols
#         required = ["t", "t_index", "lat_deg", "lon_deg", "grid_index", "GP_index"]
        
#         # ensure required columns are present in layout
#         missing = [r for r in required if r not in col]
#         if missing: raise ValueError(f"layout missing required columns: {missing}")

#         # basic shape checks
#         if rows.shape[0] != n:
#             raise AssertionError(f"expected rows {n}, got {rows.shape[0]}")
#         if rows.shape[1] != k:
#             raise AssertionError(f"expected columns {k} based on schema shape, got {rows.shape[1]}")
#         if offsets.shape[0] != schema['n_steps'] + 1:
#             raise AssertionError(f"expected offsets {schema['n_steps'] + 1}, got {offsets.shape[0]}")

#         # extract required data into packed array
#         t = rows[:, col["t"]]
#         t_idx = rows[:, col["t_index"]]
#         lat = rows[:, col["lat_deg"]]
#         lon = rows[:, col["lon_deg"]]
#         grid_idx = rows[:, col["grid_index"]]
#         gp_idx = rows[:, col["GP_index"]]

#         # package additional data
#         base_set = set(required)
#         extras: Dict[str, np.ndarray] = {}
#         for name in layout:
#             # skip base column; already packaged
#             if name in base_set: continue

#             # package extra column as view into rows
#             extras[name] = rows[:, col[name]]

#         # return `AccessTable` object
#         return cls(
#             _offsets=offsets,
#             _rows=rows,
#             _col=col,
#             _extras=extras,
#             _meta=schema,
#             _t=t,
#             _t_idx=t_idx,
#             _lat=lat,
#             _lon=lon,
#             _grid_idx=grid_idx,
#             _gp_idx=gp_idx,
#         )

#     def __len__(self):
#         return len(self._t)

#     def lookup_interval(self, t_start: float, t_end: float = np.Inf, include_extras : bool = True) -> Dict[str, np.ndarray]:
#         # validate inputs
#         if not isinstance(t_start, (int, float)) or t_start < 0.0:
#             raise ValueError("time t_start must be a non-negative number")
#         if not isinstance(t_end, (int, float)) or t_end < 0.0:
#             raise ValueError("time t_end must be a non-negative number")
#         if t_start > t_end + 1e-6:
#             raise ValueError("time t_start must be less than or equal to time t_end")

#         # check if there is any data
#         if len(self._t) == 0:
#             s_empty = slice(0, 0)
#             return self.__rows_from_slice(s_empty, include_extras=include_extras)

#         # get start and end indices for time range
#         ti0 = self._time_to_index_floor(t_start)
#         ti1 = self._time_to_index_floor(t_end)

#         # check if start and end time are in the same time index bucket
#         if ti0 == ti1:
#             # if so, return the rows for that bucket
#             return self.lookup_time(t_start, include_extras=include_extras)

#         # get slice for time index range
#         s = self._slice_for_index_range(ti0, ti1)

#         # construct output dictionary
#         out = self.__rows_from_slice(s, include_extras=include_extras)

#         # filter out any rows that are outside the time range 
#         if s.start != s.stop:
#             mask = (t_start <= out["time [s]"]) & (out["time [s]"] <= t_end)
#             for k in list(out.keys()):
#                 out[k] = out[k][mask]

#         # return ouput
#         return out

#     def _time_to_index_floor(self, t: float) -> int:
#         dt = float(self._meta["time_step"])
#         return int(t // dt) if not np.isinf(t) else len(self._offsets) - 1
    
#     def _slice_for_index_range(self, ti0: int, ti1: int) -> slice:
#         if ti1 < ti0:
#             return slice(0, 0)
#         ti0 = self._clamp_index(ti0)
#         ti1 = self._clamp_index(ti1)
#         if ti1 < ti0:
#             return slice(0, 0)
#         a = int(self._offsets[ti0])
#         b = int(self._offsets[ti1 + 1])
#         return slice(a, b)

#     def _clamp_index(self, ti: int) -> int:
#         T = len(self._offsets) - 1
#         if ti < 0:
#             return 0
#         if ti > T - 1:
#             return T - 1
#         return ti    

#     def __rows_from_slice(self, s: slice, include_extras: bool = False) -> Dict[str, Any]:
#         out = {
#             "time [s]": self._t[s].astype(float),
#             "lat [deg]": self._lat[s].astype(float),
#             "lon [deg]": self._lon[s].astype(float),
#             "grid index": self._grid_idx[s].astype(int, copy=False),
#             "GP index": self._gp_idx[s].astype(int, copy=False),
#         }

#         if include_extras:
#             for safe, arr in self._extras.items():
#                 # safe is the key used in schema["columns"]
#                 col_meta = self._meta.get("columns", {}).get(safe, {})
#                 col_name = col_meta.get("col_name", safe)

#                 if "vocab" in col_meta:
#                     # get vocab for column
#                     vocab: dict = col_meta["vocab"]

#                     # cast to codes if not already int
#                     codes = arr[s].astype(np.int32, copy=False)  
                    
#                     # decode using vocab
#                     out[col_name] = np.array([vocab.get(str(int(c)), None) for c in codes])
                
#                 else:
#                     # no vocab, return raw values as float
#                     out[col_name] = arr[s].astype(float)

#         # return output dictionary with arrays for each column
#         return out

#     def lookup_time(self, t: float, include_extras: bool = False) -> Dict[str, Any]:
#         """
#         Find nearest / first occurrence; but you still need the corresponding time index bucket.
#         If you trust that each bucket has constant t value, you can map by search then use t_idx:
#         """

#         # validate inputs
#         if not isinstance(t, (int, float)) or t < 0.0:
#             raise ValueError("time `t` must be a non-negative number")

#         # check if there is any data    
#         if len(self._t) == 0:
#             s_empty = slice(0, 0)
#             return self.__rows_from_slice(s_empty, include_extras=include_extras)

#         # check if time `t` is beyond the data in the table
#         if self._t[-1] < t - 1e-6:
#             # if so, return last row
#             s_last = slice(len(self._t) - 1, len(self._t))
#             return self.__rows_from_slice(s_last, include_extras=include_extras)

#         # find location in time array
#         i = int(np.searchsorted(self._t, t, side="left"))

#         # convert to time index and get rows for that time index
#         ti = int(self._t_idx[i]) 
#         return self.__rows_at_index(ti, include_extras=include_extras)

#     def __rows_at_index(self, ti: int, include_extras: bool = False) -> Dict[str, Any]:
#         s = self.__slice_for_time_index(ti)
#         return self.__rows_from_slice(s, include_extras=include_extras)

#     def __slice_for_time_index(self, ti: int) -> slice:
#         # bounds check 
#         if ti < 0 or ti >= (len(self._offsets) - 1):
#             # time index is out of bounds; return empty slice
#             return slice(0, 0)
        
#         # compute slice for time index from offsets
#         a = int(self._offsets[ti])
#         b = int(self._offsets[ti + 1])

#         # return slice object
#         return slice(a, b)

#     def __iter__(self):
#         for i in range(len(self._t)):
#             out = {
#                 "lat [deg]": float(self._lat[i]),
#                 "lon [deg]": float(self._lon[i]),
#                 "grid index": int(self._grid_idx[i]),
#                 "GP index": int(self._gp_idx[i]),
#             }
#             for safe, arr in self._extras.items():
#                 col_meta = self._meta.get("columns", {}).get(safe, {})
#                 col_name = col_meta.get("col_name", safe)

#                 if "vocab" in col_meta:
#                     vocab: dict = col_meta["vocab"]
#                     out[col_name] = vocab.get(str(int(arr[i])), None)
#                 else:
#                     out[col_name] = float(arr[i])

#             yield float(self._t[i]), out    
            
