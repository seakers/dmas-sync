    
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import json
import math
import os
# import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# from dmas.utils.orbitdata import OrbitData
from execsatm.tasks import GenericObservationTask

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

# @dataclass
# class LatestObservationTracker:

#     lut: np.ndarray          # shape (G, P), dtype int32, -1 => not tracked
#     G: int
#     P: int
#     N: int
#     t_last: np.ndarray
#     n_obs: np.ndarray
#     last_actor: np.ndarray
#     actor_to_id: Dict[str, int]
#     id_to_actor: List[str]

#     @classmethod
#     def from_orbitdata(
#         cls,
#         orbitdata: OrbitData,
#         actor_name : str        
#     ) -> "LatestObservationTracker":
#         """
#         Builds the target index mapping from OrbitData.grid_data.

#         Expects:
#           orbitdata.grid_data: iterable of pandas DataFrames containing columns:
#             ["grid index", "GP index"] (and optionally lat/lon, but not required for tracking)
#         """        

#         # iterate through the unique grid points and populate the key_to_k and k_to_key mappings
#         targets = [
#             (grid_idx, gp_idx)
#             for *_,grid_idx,gp_idx in orbitdata.grid_data
#         ]

#         # remove duplicate targets while preserving order
#         targets_unique = list(set(targets))

#         # build from targets
#         return cls.from_targets(targets_unique, actor_name=actor_name)

#     @classmethod
#     def from_targets(cls,
#                      targets: Iterable[Tuple[int, int]],
#                      actor_name: str,
#                      dtype_n_obs=np.uint16,
#                      lut_dtype=np.int32,
#                     ) -> "LatestObservationTracker":
        
#         # Build bounds
#         gmax = -1
#         pmax = -1
#         pairs = []
#         for g, p in targets:
#             g = int(g); p = int(p)
#             pairs.append((g, p))
#             if g > gmax: gmax = g
#             if p > pmax: pmax = p

#         G = gmax + 1
#         P = pmax + 1

#         lut = np.full((G, P), -1, dtype=lut_dtype)

#         # Assign dense k ids only to existing targets
#         k = 0
#         for g, p in pairs:
#             if lut[g, p] == -1:
#                 lut[g, p] = k
#                 k += 1
#         N = k

#         t_last = np.full(N, -np.inf, dtype=np.float64)
#         n_obs = np.zeros(N, dtype=dtype_n_obs)
#         last_actor = np.full(N, -1, dtype=np.int16)

#         actor_to_id = {actor_name: 0}
#         id_to_actor = [actor_name]

#         return cls(lut, G, P, N, t_last, n_obs, last_actor, actor_to_id, id_to_actor)

#     def _get_actor_id(self, actor_name: Optional[str]) -> int:
#         if actor_name is None:
#             return -1
#         aid = self.actor_to_id.get(actor_name)
#         if aid is None:
#             aid = len(self.actor_to_id)
#             self.actor_to_id[actor_name] = aid
#             self.id_to_actor.append(actor_name)  # <-- append, not index-assign
#         return aid

#     def _k(self, grid_idx: int, gp_idx: int) -> int:
#         g = int(grid_idx); p = int(gp_idx)
#         if g < 0 or p < 0 or g >= self.G or p >= self.P:
#             return -1
#         return int(self.lut[g, p])

#     def __update_one(self, obs: Dict[str, Any]) -> None:
#         k = self._k(obs["grid index"], obs["GP index"])
#         if k < 0:
#             return

#         t_end = float(obs["t_end"])
#         actor_id = self._get_actor_id(obs.get("agent name"))

#         self.n_obs[k] = self.n_obs[k] + 1
#         if t_end >= self.t_last[k]:
#             self.t_last[k] = t_end
#             self.last_actor[k] = actor_id

#     def update_many(self, observations: Iterable[Tuple[str,List[Dict[str, Any]]]]) -> None:
#         """
#         Update arrays for many observation dicts.
#         """
#         # validate input
#         if not isinstance(observations, Iterable):
#             raise ValueError("observations must be an iterable of dicts")
        
#         # update each observation in the input iterable
#         for _,obs_data in observations:
#             # get the observation with the latest t_end
#             obs = max(obs_data, key=lambda o: o.get("t_end", -np.inf), default=None)  
            
#             # `obs_data` is empty; skip 
#             if obs is None:
#                 continue
            
#             # update the tracker with this observation
#             self.__update_one(obs)

#     def lookup(self, grid_index : int, gp_index : int) :
#         """
#         Lookup the latest observation info for a given grid point.

#         Returns a dict with keys:
#           - "t_last": time of the latest observation (float)
#           - "n_obs": total number of observations (int)
#           - "last_actor": name of the last observing agent (str or None)
#         """
#         k = self._k(grid_index, gp_index)
#         if k < 0:
#             return {"t_last": -np.inf, "n_obs": -1, "last_actor": None}
        
#         actor_id = self.last_actor[k]
#         actor_name = self.id_to_actor[actor_id] if actor_id >= 0 else None

#         return {
#             "t_last": float(self.t_last[k]),
#             "n_obs": int(self.n_obs[k]),
#             "last_actor": actor_name
#         }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnknownTaskError(KeyError):
    """
    Raised when the tracker receives an update for a task_id it has never
    seen.  The agent's ``__update_requests_and_tasks`` must register tasks
    before either update path is called.
    """
    def __init__(self, task_id: str, source: str):
        self.task_id = task_id
        self.source  = source
        super().__init__(
            f"[TaskObservationTracker] Unknown task '{task_id}' in {source}. "
            f"Ensure register_task() is called before {source}."
        )


# ---------------------------------------------------------------------------
# Location key type alias
# ---------------------------------------------------------------------------

# (grid_index, gp_index) — the two integer coordinates present in every
# raw measurement entry.  lat/lon are excluded intentionally: they are
# floating-point and therefore unreliable as dict keys.
LocationKey = Tuple[int, int]


# ---------------------------------------------------------------------------
# ObservationRecord
# ---------------------------------------------------------------------------

@dataclass
class ObservationRecord:
    """
    Distilled observation state for a single task.

    ``n_obs`` counts completed observation *passes* for this task,
    regardless of how many target points within the task were covered
    in each pass.  A pass is one call to ``ingest()``.
    """
    t_last:          float         = field(default_factory=lambda: -math.inf)
    n_obs:           int           = 0
    last_actor:      Optional[str] = None
    last_instrument: Optional[str] = None

    def ingest(self, t_end: float, actor: str, instrument: Optional[str]) -> bool:
        """
        Record one completed observation pass for this task.

        Always increments ``n_obs``.  Overwrites the "latest" fields only
        when *t_end* is at least as recent as the stored value.

        Returns True if the latest-observation fields were updated.
        """
        self.n_obs += 1
        if t_end >= self.t_last:
            self.t_last          = t_end
            self.last_actor      = actor
            self.last_instrument = instrument
            return True
        return False

    def merge(self, foreign: Any[ObservationRecord, dict]) -> bool:
        """
        Merge *foreign* into self using peer-share semantics:
          - ``n_obs``          : summed
          - ``t_last``         : max; ties go to *foreign*
          - ``last_actor`` /
            ``last_instrument``: follow whichever side owns the larger t_last

        Returns True if any field changed.
        """
        try:
            merged_n_obs = self.n_obs + foreign['n_obs']

            if foreign['t_last'] >= self.t_last:
                changed = (merged_n_obs != self.n_obs
                        or foreign['t_last'] != self.t_last)
                self.n_obs           = merged_n_obs
                self.t_last          = foreign['t_last']
                self.last_actor      = foreign['last_actor']
                self.last_instrument = foreign['last_instrument']
            else:
                changed    = merged_n_obs != self.n_obs
                self.n_obs = merged_n_obs

        except KeyError:
            foreign : ObservationRecord
            merged_n_obs = self.n_obs + foreign.n_obs

            if foreign.t_last >= self.t_last:
                changed = (merged_n_obs != self.n_obs
                        or foreign.t_last != self.t_last)
                self.n_obs           = merged_n_obs
                self.t_last          = foreign.t_last
                self.last_actor      = foreign.last_actor
                self.last_instrument = foreign.last_instrument
            else:
                changed    = merged_n_obs != self.n_obs
                self.n_obs = merged_n_obs

        return changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_last":          None if math.isinf(self.t_last) else self.t_last,
            "n_obs":           self.n_obs,
            "last_actor":      self.last_actor,
            "last_instrument": self.last_instrument,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObservationRecord":
        raw_t = d.get("t_last")
        return cls(
            t_last          = -math.inf if raw_t is None else float(raw_t),
            n_obs           = int(d.get("n_obs", 0)),
            last_actor      = d.get("last_actor"),
            last_instrument = d.get("last_instrument"),
        )


# ---------------------------------------------------------------------------
# TaskObservationTracker
# ---------------------------------------------------------------------------

@dataclass
class TaskObservationTracker:
    """
    Pure observation-state store for a multi-agent satellite simulation.

    Location → task resolution
    --------------------------
    Raw measurement entries carry a target location ``(grid_idx, gp_idx)``
    but no ``task_id``.  The tracker maintains an internal reverse map built
    from ``task.location`` at registration time so that ingestion can resolve
    locations to all matching tasks without any agent-side pre-processing.

    A location may map to more than one active task (e.g. a default
    monitoring task and an event task sharing a target point).  In that
    case *all* matching tasks are credited.

    ``n_obs`` semantics
    -------------------
    One observation pass (one call to ``update_from_observations`` with
    entries that resolve to a given task) increments ``n_obs`` by exactly
    **one** for that task, regardless of how many target points within the
    task were covered in the pass.

    Responsibilities
    ----------------
    - Maintain ``ObservationRecord`` per registered task.
    - Resolve ``(grid_idx, gp_idx)`` → task_ids via internal location map.
    - Provide two typed update paths: raw sensor data and peer JSON payloads.
    - Encode state for peer broadcast (metadata-free; recipients have task
      details via ``MeasurementRequestMessage``).

    Non-responsibilities (owned by ``SimulationAgent``)
    -------------------------------------------------------
    - Task metadata, expiry, and lifecycle live on ``GenericObservationTask``
      inside ``_known_tasks``.
    - Task registration/removal is driven by the agent; every write path
      raises ``UnknownTaskError`` for unregistered task_ids.
    """

    _owner:       str
    _records:     Dict[str, ObservationRecord]    # task_id  → state
    _loc_to_tasks: Dict[LocationKey, Set[str]]    # (grid, gp) → {task_id, ...}
    _shareable:    Set[str]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, owner_name: str) -> "TaskObservationTracker":
        """Return an empty tracker owned by *owner_name*."""
        return cls(
            _owner        = owner_name,
            _records      = {},
            _loc_to_tasks = {},
            _shareable    = set(),
        )

    # ------------------------------------------------------------------
    # Task lifecycle  (driven entirely by SimulationAgent)
    # ------------------------------------------------------------------

    def register_task(self, task: GenericObservationTask, shareable: bool = False) -> bool:
        """
        Register *task* and index all of its target locations.

        Parameters
        ----------
        shareable:
            If True, this task's observation state will be included in
            ``encode()`` output.  Default False — callers must opt in
            explicitly.  Typical usage:
            - DefaultMissionTask  → shareable=False
            - EventObservationTask → shareable=True
        """
        tid = task.id
        if tid in self._records:
            return False

        # Create observation record
        self._records[tid] = ObservationRecord()

        # Index every target location this task covers
        for loc in task.location:
            # loc = (lat, lon, grid_index, gp_index)
            key: LocationKey = (int(loc[2]), int(loc[3]))
            self._loc_to_tasks.setdefault(key, set()).add(tid)

        # opt in to sharing this task's observation state if requested
        if shareable: self._shareable.add(tid)

        return True

    def remove_task(self, task: GenericObservationTask) -> bool:
        """
        Remove *task* and clean up its location index entries.

        Called by the agent when a task expires or is dropped from
        ``_known_tasks``.  Safe to call on unregistered tasks (no-op).
        Returns True if a record was actually removed.
        """
        tid = task.id
        if tid not in self._records:
            return False

        # Remove from location index
        for loc in task.location:
            key: LocationKey = (int(loc[2]), int(loc[3]))
            task_set = self._loc_to_tasks.get(key)
            if task_set is not None:
                task_set.discard(tid)
                if not task_set:
                    # del self._loc_to_tasks[key]
                    self._loc_to_tasks.pop(key, None)
        
        # remove records for this task
        self._records.pop(tid, None)

        # discard task from shareable set
        self._shareable.discard(tid)

        return True
    
    def is_registered(self, task_id: str) -> bool:
        return task_id in self._records

    def registered_task_ids(self) -> List[str]:
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal: location resolution
    # ------------------------------------------------------------------

    def _resolve_location(self, grid_idx: int, gp_idx: int) -> Set[str]:
        """
        Return the set of task_ids whose coverage includes
        ``(grid_idx, gp_idx)``.  Returns an empty set if the location
        is not covered by any registered task.
        """
        return self._loc_to_tasks.get((int(grid_idx), int(gp_idx)), set())

    # ------------------------------------------------------------------
    # Update path 1 — raw sensor observations from own instruments
    # ------------------------------------------------------------------

    def update_from_observations(
        self,
        measurements: Iterable[List[Dict[str, Any]]],
    ) -> Dict[str, int]:
        """
        Ingest raw measurement data produced by this agent's own sensors.

        Parameters
        ----------
        measurements:
            Iterable of *measurement lists*.  Each list represents one
            observation pass; each dict within the list corresponds to one
            target point covered during that pass.

            Required keys per dict:
              ``grid index`` – int
              ``gp index``   – int
              ``t_end``      – float, time the measurement window closed

            Optional keys:
              ``agent name`` – str (defaults to owner name)
              ``instrument`` – str

        Behaviour
        ---------
        - Every dict in a measurement list is resolved to zero or more
          matching tasks via the internal location map.
        - Each matched task is credited with **one** observation pass for
          the entire measurement list (``n_obs += 1``), using the latest
          ``t_end`` seen across all entries that resolved to that task.
        - Tasks matched by multiple entries in the same pass are therefore
          updated once, not once-per-entry.

        Returns
        -------
        ``{task_id: count}`` — number of passes credited per task in this call.

        Raises
        ------
        UnknownTaskError
            If location resolution returns a task_id that is not in
            ``_records`` (indicates an internal consistency bug).
        """
        ingested: Dict[str, int] = {}

        for measurement in measurements:
            # ── Pass 1: resolve every entry to its matching tasks and
            #    collect the latest t_end seen per task within this pass.
            best_t_end:  Dict[str, float] = {}  # task_id → best t_end this pass
            best_actor:  Dict[str, str]   = {}
            best_instr:  Dict[str, Optional[str]] = {}

            for entry in measurement:
                grid_idx = int(entry.get("grid index", -1))
                gp_idx   = int(entry.get("GP index",   -1))
                t_end    = float(entry.get("t_end", -math.inf))
                actor    = str(entry.get("agent name", self._owner))
                instr    = entry.get("instrument")

                task_ids = self._resolve_location(grid_idx, gp_idx)
                # Silently skip entries whose location isn't covered by any
                # active task — the agent may have pruned the task already.
                if not task_ids:
                    continue

                for tid in task_ids:
                    if not self.is_registered(tid):
                        # Should never happen; indicates a registration bug.
                        raise UnknownTaskError(tid, "update_from_observations()")

                    # Keep only the latest t_end for this task in this pass
                    if t_end > best_t_end.get(tid, -math.inf):
                        best_t_end[tid]  = t_end
                        best_actor[tid]  = actor
                        best_instr[tid]  = instr

            # ── Pass 2: credit each matched task with one observation pass.
            for tid, t_end in best_t_end.items():
                self._records[tid].ingest(
                    t_end      = t_end,
                    actor      = best_actor[tid],
                    instrument = best_instr[tid],
                )
                ingested[tid] = ingested.get(tid, 0) + 1

        return ingested

    # ------------------------------------------------------------------
    # Update path 2 — peer knowledge broadcast
    # ------------------------------------------------------------------

    def update_from_peer(self, payload: object) -> Dict[str, str]:
        """
        Merge knowledge received from a peer agent.

        The payload is a dictionary produced by ``encode()``.  The agent
        must have already processed any accompanying ``MeasurementRequestMessage``
        so that every task_id in the payload is registered here.

        Merge rules
        -----------
        - ``n_obs``          : summed
        - ``t_last``         : max (ties go to foreign)
        - ``last_actor`` /
          ``last_instrument``: follow whichever side holds the larger t_last

        Returns
        -------
        ``{task_id: "updated" | "unchanged"}``

        Raises
        ------
        UnknownTaskError
            If the payload references an unregistered task_id, indicating
            ``__update_requests_and_tasks`` was not called first.
        """
        outcomes: Dict[str, str] = {}

        if isinstance(payload, str):
            payload = json.loads(payload)
            for task_id, state_dict in payload.get("records", {}).items():
                if not self.is_registered(task_id):
                    raise UnknownTaskError(task_id, "update_from_peer()")

                foreign = ObservationRecord.from_dict(state_dict)
                changed = self._records[task_id].merge(foreign)
                outcomes[task_id] = "updated" if changed else "unchanged"

        elif isinstance(payload, dict):
            for task_id, state_dict in payload.get("records", {}).items():
                if not self.is_registered(task_id):
                    raise UnknownTaskError(task_id, "update_from_peer()")

                foreign = ObservationRecord.from_dict(state_dict)
                changed = self._records[task_id].merge(foreign)
                outcomes[task_id] = "updated" if changed else "unchanged"

        elif isinstance(payload, list):
            for tid,t_last,n_obs,last_actor,last_instrument in payload:
                task_id = str(tid)
                if not self.is_registered(task_id):
                    raise UnknownTaskError(task_id, "update_from_peer()")

                foreign = {
                    "t_last": float(t_last) if t_last is not None else -math.inf,
                    "n_obs": int(n_obs),
                    "last_actor": str(last_actor) if last_actor is not None else None,
                    "last_instrument": str(last_instrument) if last_instrument is not None else None,
                }
                changed = self._records[task_id].merge(foreign)
                outcomes[task_id] = "updated" if changed else "unchanged"

        else:
            raise ValueError(f"Unsupported payload type: {type(payload)}. Expected str, dict, or list.")


        return outcomes

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, task_id: str) -> Dict[str, Any]:
        """
        Return observation state for *task_id*.

        Returns sentinel values (``n_obs = -1``, ``t_last = -inf``) for
        unknown tasks rather than raising — lookup is a read path and
        callers often probe defensively.
        """
        record = self._records.get(task_id)
        if record is None:
            return {
                "t_last":          -math.inf,
                "n_obs":           -1,
                "last_actor":      None,
                "last_instrument": None,
            }
        return {
            "t_last":          record.t_last,
            "n_obs":           record.n_obs,
            "last_actor":      record.last_actor,
            "last_instrument": record.last_instrument,
        }

    def tasks_at(self, grid_idx: int, gp_idx: int) -> List[str]:
        """
        Return all task_ids whose coverage includes ``(grid_idx, gp_idx)``.
        Useful for agent-side diagnostics without exposing the internal map.
        """
        return list(self._resolve_location(grid_idx, gp_idx))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        task_ids: Optional[Iterable[str]] = None,
        encoding_type : type = list
    ) -> dict:
        """
        Encode observation state as a JSON string for peer broadcast.

        Task metadata is intentionally excluded — recipients get that via
        ``MeasurementRequestMessage``.

        Parameters
        ----------
        task_ids:
            Subset to encode.  Defaults to all registered tasks.

        Payload schema
        --------------
        {
          "sender":  "<owner>",
          "records": {
            "<task_id>": {
              "t_last": float | null,
              "n_obs": int,
              "last_actor": str | null,
              "last_instrument": str | null
            },
            ...
          }
        }
        """
        if task_ids is not None:
            # Caller-specified subset: respect it exactly, no shareability filter
            ids = [tid for tid in task_ids if tid in self._records]
        else:
            # Default: only shareable tasks
            ids = list(self._shareable)
        
        if encoding_type is list:
            payload = []
            for tid in ids:
                record = self._records.get(tid)
                if record is not None:
                    # Skip tasks with no observations
                    if record.n_obs <= 0: continue  

                    # Encode as a tuple: (task_id, t_last, n_obs, last_actor, last_instrument)
                    datum = (tid, record.t_last, record.n_obs, record.last_actor, record.last_instrument)
                    payload.append(datum)

        elif encoding_type is dict:
            records = {
                tid: self._records[tid].to_dict()
                for tid in ids
                if tid in self._records
            }
            payload = {"sender": self._owner, "records": records}

        elif encoding_type is str:
            records = {
                tid: self._records[tid].to_dict()
                for tid in ids
                if tid in self._records
            }
            payload = json.dumps(
                {"sender": self._owner, "records": records},
                separators=(",", ":"),
            )

        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}. Supported types are dict and list.")
                
        return payload