
from collections import defaultdict
import os
from typing import List, Set, Tuple
import numpy as np
import pandas as pd

from tqdm import tqdm
from abc import abstractmethod

from execsatm.tasks import EventObservationTask
from execsatm.events import GeophysicalEvent
from execsatm.mission import Dict, Mission
from execsatm.objectives import EventDrivenObjective
from execsatm.utils import Interval

from dmas.models.actions import BroadcastMessageAction, WaitAction
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.plan import PeriodicPlan, Plan
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState
from dmas.core.messages import BusMessage, MeasurementRequestMessage
from dmas.utils.orbitdata import OrbitData


class AbstractEventAnnouncerPlanner(AbstractPeriodicPlanner):
    def __init__(self,
                 agent_results_dir : str,
                 events_path : str,
                 simulation_missions : Dict[str,Mission],
                 announce_horizon : float = 3600.0,
                 debug = False,
                 logger = None,
                 printouts : bool = True
                ) -> None:
        """
        # Event Announcer Planner
        Announces geophysical events to other agents in the mission as they become available.
        Uses a rolling planning window of `announce_horizon` seconds so that only the next
        window's worth of broadcasts are scheduled at a time, keeping memory use flat.
        """
        # announce_horizon *= 24.0 # TEMP EXTENSION, REMOVE LATER

        super().__init__(agent_results_dir, announce_horizon, announce_horizon, AbstractPeriodicPlanner.OPPORTUNISTIC, debug, logger, printouts)

        # validate inputs
        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')
        if not isinstance(simulation_missions, Dict):
            raise ValueError('`simulation_missions` must be a dictionary of mission objects.')

        # set properties
        self._agent_results_dir = agent_results_dir
        self._simulation_missions : Dict[str,Mission] = simulation_missions
        self._parent_agent_name : str = None 

        # load predefined events
        self._events_dir : str = events_path
        self._events : list[GeophysicalEvent] = self.load_events(events_path)
        self._event_ids : Dict[str, GeophysicalEvent]= {event.id : event for event in self._events}
        # tracks which comms-column indices have already been covered per event id
        self._announced : Dict[str, set] = {}
    
    def load_events(self, events_path : str) -> pd.DataFrame:        
        events_df : pd.DataFrame = pd.read_csv(events_path)

        events = []
        for _,row in events_df.iterrows():
            # convert event to GeophysicalEvent
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row.get('id',None)
            )
            events.append(event)

        return events
        
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        _ : object,
                        orbitdata : OrbitData,
                        *__
                    ) -> Plan:
        """ Generates a new plan for the agent """            
        # get current time 
        t_curr = state.get_time()

        # set parent agent name if not set already
        if self._parent_agent_name is None:
            self._parent_agent_name = state.agent_name

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, [], orbitdata)
        
        # TODO add maneuvers for pointing-dependent transmissions
        # currently assumes omnidirectional antennas
        
        # wait for next planning period to start
        replan_waits : list = self._schedule_periodic_replan(state, t_curr + self._period)

        # generate plan from actions
        self._plan : PeriodicPlan = PeriodicPlan(broadcasts, replan_waits, t=t_curr, horizon=self._horizon, t_next=t_curr+self._period)    
        
        # return plan and save local copy
        return self._plan.copy()
    
    def _schedule_observations(self, *_) -> list:
        return [] # No scheduling, only announcing events
    
    @abstractmethod
    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        pass

class InstantEventAnnouncerPlanner(AbstractEventAnnouncerPlanner):
    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        # initialzie list of broadcasts to be performed in this planning period
        broadcasts : List[BroadcastMessageAction] = []
        t_curr : float = state.get_time()
        t_window_max : float = t_curr + self._period

        # drop expired events and their announcement records
        self._events = [e for e in self._events if e.is_available(t_curr)]
        active_ids = {e.id for e in self._events}
        self._announced = {eid: covered for eid, covered in self._announced.items() if eid in active_ids}

        # collect events that are available AND start within this planning window
        future_events : List[GeophysicalEvent] = [
            event for event in tqdm(self._events,
                                    desc=f'{state.agent_id}-PREPLANNER: Collecting future events',
                                    leave=False,
                                    disable=(len(self._events) < 10) or not self._printouts)
            if event.is_available(t_curr) and event.availability.left <= t_window_max
        ]

        if not future_events: return broadcasts

        # pre-index objectives by event type to avoid re-scanning missions per event
        objectives_by_type : Dict = defaultdict(list)
        for mission_name, objectives in self._simulation_missions.items():
            for objective in objectives:
                if isinstance(objective, EventDrivenObjective):
                    objectives_by_type[objective.event_type].append((mission_name, objective))

        # build task requests for each event x objective pair
        task_requests : List[Tuple[GeophysicalEvent, TaskRequest]] = []
        for event in tqdm(future_events,
                          desc=f'{state.agent_id}-PREPLANNER: Generating task requests from known events',
                          leave=False,
                          disable=(len(future_events) < 10) or not self._printouts):
            for mission_name, objective in objectives_by_type[event.event_type]:
                objective : EventDrivenObjective
                task = EventObservationTask(objective.parameter, event=event, objective=objective)
                task_request = TaskRequest(task,
                                           requester=state.agent_name,
                                           mission_name=mission_name,
                                           t_req=event.t_start)
                task_request_msg = MeasurementRequestMessage(state.agent_name, state.agent_name, task_request.to_dict())
                task_requests.append((event, task_request, task_request_msg))       

        # group messages by event so coverage tracking is per-event
        event_requests : Dict = defaultdict(list)
        for event, task_req, task_request_msg in task_requests:
            event_requests[event].append((task_req, task_request_msg))

        # comms index for this agent
        u_idx = orbitdata.comms_target_indices[OrbitData.safe_name(state.agent_name)]
        n_cols = len(orbitdata.comms_target_columns)
        n_targets = len(orbitdata.comms_targets)
        col_start = orbitdata.comms_links._col["start"]
        col_end   = orbitdata.comms_links._col["end"]

        # fetch all comm rows that overlap the planning window; include_current captures
        # contacts that started before t_curr but are still active
        all_rows = [row for _, row in orbitdata.comms_links.iter_rows_packed(
            t_curr, t_window_max, include_current=True)]

        # for each event, run coverage logic seeded with already-announced agents
        t_broadcasts : Dict = defaultdict(list)
        for event, reqs in tqdm(event_requests.items(),
                                desc=f'{state.agent_id}-PREPLANNER: Scheduling broadcasts for events',
                                leave=False,
                                disable=(len(event_requests) < 10) or not self._printouts):
            msgs = [msg for _, msg in reqs]
            t_req = min(req.t_req for req, _ in reqs)

            t_window_start = max(event.availability.left, t_req, t_curr)
            t_window_end = min(event.availability.right, t_window_max)

            if t_window_start >= t_window_end:
                continue

            # seed covered agents from previous planning periods
            already_covered = self._announced.get(event.id, set())
            agents_covered = np.zeros(n_cols, dtype=bool)
            for idx in already_covered:
                agents_covered[idx] = True
            n_covered = int(agents_covered.sum())

            if n_covered >= n_targets:
                continue  # all agents already notified in a prior period

            for row in all_rows:
                t_row_start = float(row[col_start])
                t_row_end   = float(row[col_end])
                if t_row_start > t_window_end:
                    break
                if t_row_end < t_window_start:
                    continue  # contact ends before this event's window opens — no usable slot

                comps = row[3:]
                u_comp = comps[u_idx]
                reachable = (comps == u_comp)
                reachable[u_idx] = False

                new_agents = np.where(reachable & ~agents_covered)[0]
                if new_agents.size == 0:
                    continue

                t_broadcast = max(t_row_start, t_window_start)
                t_broadcasts[t_broadcast].extend(msgs)

                agents_covered[new_agents] = True
                n_covered += new_agents.size

                if n_covered >= n_targets:
                    break

            # persist coverage state for this event across planning periods
            self._announced[event.id] = set(np.where(agents_covered)[0].tolist())

        # compile accumulated messages into one BroadcastMessageAction per time slot
        for t_broadcast, task_requests_msgs in t_broadcasts.items():
            assert len(task_requests_msgs) > 0, "No active task requests found for broadcast time."
            bus_broadcast = BusMessage(state.agent_name, state.agent_name, task_requests_msgs)
            broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))

        # return sorted list of broadcasts by time
        return sorted(broadcasts, key=lambda action: action.t_start)    
    
    def print_results(self):
        # clear events list and reload them 
        self._events = self.load_events(self._events_dir)

        # log known and generated requests
        columns = ['id','requester','lat [deg]','lon [deg]','grid index', 'GP index','severity','start time [s]','end time [s]','detection time [s]','event type']        
        data_detected = [(event.id, self._parent_agent_name, event.location[0], event.location[1], event.location[2], event.location[3], event.severity, event.t_start, event.t_start+event.d_exp, event.t_detect, event.event_type)
                for event in self._events]
             
        df = pd.DataFrame(data=data_detected, columns=columns)        
        df.to_parquet(f"{self._agent_results_dir}/events_detected.parquet", index=False)

    
class GroundProcessorEventAnnouncerPlanner(AbstractEventAnnouncerPlanner):
    def __init__(self,
                 agent_results_dir : str,
                 events_path : str,
                 space_segment : List[Dict],
                 simulation_missions : Dict[str,Mission],
                 simulation_orbitdata : Dict[str,OrbitData],
                 announce_horizon : float = 3600.0,
                 debug = False,
                 logger = None,
                 printouts : bool = True
                ) -> None:
        # initialzie parent class with appropriate horizon and period for rolling announcements
        super().__init__(agent_results_dir, events_path, simulation_missions, announce_horizon, debug, logger, printouts)
        
        # def is_taskable(agent_name : str) -> bool:
        #     # if not a satellite, assume taskable (e.g. ground station or processor); only satellites have orbitdata and require access checks for announcements
        #     if agent_name not in space_segment: return False

        #     satellite_spec = next((d for d in space_segment if d['name'] == agent_name), None)
        #     if satellite_spec is None:
        #         raise ValueError(f"Agent {agent_name} not found in space segment.")
            
        #     agent_is_maneuverable = 'maneuver' in satellite_spec['instrument']

        #     return agent_is_maneuverable

        # store orbitdata for use in scheduling broadcasts
        self._agent_orbitdata = {agent_name : agent_orbitdata
                                 for agent_name, agent_orbitdata in simulation_orbitdata.items()
                                 if '-U)' in agent_name # <- only keep orbitdata for non-taskable agents
                                #  if not is_taskable(agent_name) # TODO refine this logic based on actual space segment definitions; for now, assume non-taskable agents are those whose names contain '-U)' as in the provided orbitdata files
                                } 
        
        # assert all('-U)' in agent_name for agent_name in self._agent_orbitdata.keys()), \
        #     "OrbitData must be provided for non-taskable agents only." # <- sanity check 
        
        # map agent name to its mission 
        agent_missions : Dict[str, Mission] \
            = {d['name'] : simulation_missions[d['mission'].lower()] 
                for d in space_segment}
        
        # map event types to list of relevant agents based on their missions
        self._event_type_agents : Dict[str, Set[str]] = defaultdict(set)
        for agent_name, agent_mission in agent_missions.items():
            for objective in agent_mission:
                if isinstance(objective, EventDrivenObjective):
                    self._event_type_agents[objective.event_type].add(agent_name)
        
        # initialize properties
        self._detected_events : Set[GeophysicalEvent] = set() # events for which data has been received
        self._detection_times : Dict[str, float] = {}         # event_id -> t_announce when first detected

    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        broadcasts : List[BroadcastMessageAction] = []
        t_curr : float = state.get_time()
        t_window_max : float = t_curr + self._period

        # drop expired events and their announcement records
        self._events = [e for e in self._events if e.is_available(t_curr)]
        active_ids = {e.id for e in self._events}
        self._announced       = {eid: covered for eid, covered in self._announced.items()       if eid in active_ids}
        self._detection_times = {eid: t       for eid, t       in self._detection_times.items() if eid in active_ids}

        # collect events that are available AND start within this planning window
        future_events : List[GeophysicalEvent] = [
            event for event in tqdm(self._events,
                                    desc=f'{state.agent_id}-PREPLANNER: Collecting future events',
                                    leave=False,
                                    disable=(len(self._events) < 10) or not self._printouts)
            if event.is_available(t_curr) and event.availability.left <= t_window_max
        ]

        if not future_events: return broadcasts

        # split into events we've already detected (t_announce known) and brand-new ones
        new_events : List[GeophysicalEvent] = [e for e in future_events if e.id not in self._detection_times]
        
        # ground processor comms setup
        u_idx = orbitdata.comms_target_indices[OrbitData.safe_name(state.agent_name)]
        n_cols = len(orbitdata.comms_target_columns)
        n_targets = len(orbitdata.comms_targets)
        col_start = orbitdata.comms_links._col["start"]
        col_end   = orbitdata.comms_links._col["end"]

        # ------------------------------------------------------------------
        # Pre-fetch per-agent access data for new (not-yet-detected) events only.
        # Detected events already have their t_announce stored; no access search needed.
        # Search window is [t_curr, t_window_max] — past data for new events is irrelevant
        # because we haven't seen them yet, and future data beyond the horizon is fetched
        # fresh next period.
        # ------------------------------------------------------------------
        agent_gp_accesses : Dict = {}   # agent -> {'times', 'grids', 'gps'} numpy arrays
        agent_downlinks   : Dict = {}   # agent -> (N,2) array of [t_start, t_end] sorted by t_start

        if new_events:
            for agent_name, agent_orbitdata in tqdm(self._agent_orbitdata.items(),
                                                    desc=f'{state.agent_id}-PREPLANNER: collecting access data for non-taskable agents',
                                                    leave=False,
                                                    disable=(len(self._agent_orbitdata) < 10) or not self._printouts,
                                                    total=len(self._agent_orbitdata)
                                                ):
                result = agent_orbitdata.gp_access_data.lookup_interval(
                    t_curr, t_window_max,
                    columns=['time [s]', 'grid index', 'GP index'],
                    decode=False
                )
                agent_gp_accesses[agent_name] = {
                    'times' : result.get('time [s]',   np.array([])),
                    'grids' : result.get('grid index', np.array([])),
                    'gps'   : result.get('GP index',   np.array([])),
                }

                gs_intervals = agent_orbitdata.gs_access_data.lookup_intervals(
                    t_curr, t_max=t_window_max, include_current=True
                )
                if gs_intervals:
                    dl = np.array([(float(iv[0].left), float(iv[0].right))
                                   for iv in gs_intervals])
                    dl = dl[np.argsort(dl[:, 0])]
                else:
                    dl = np.empty((0, 2))
                agent_downlinks[agent_name] = dl

        # pre-index objectives by event type to avoid re-scanning missions per event
        objectives_by_type : Dict = defaultdict(list)
        for mission_name, objectives in self._simulation_missions.items():
            for objective in objectives:
                if isinstance(objective, EventDrivenObjective):
                    objectives_by_type[objective.event_type].append((mission_name, objective))

        # initialize list of broadcasts to be scheduled in this period 
        t_broadcasts : Dict = defaultdict(list)

        for event in tqdm(future_events,
                          desc=f'{state.agent_id}-PREPLANNER: Scheduling broadcasts for events',
                          leave=False,
                          disable=(len(future_events) < 10) or not self._printouts):

            # define search window for this event
            t_window_end = min(event.availability.right, t_window_max)

            # skip if all agents already notified in a prior planning period
            already_covered = self._announced.get(event.id, set())
            agents_covered = np.zeros(n_cols, dtype=bool)
            for idx in already_covered:
                agents_covered[idx] = True
            n_covered = int(agents_covered.sum())

            if n_covered >= n_targets:
                continue

            # ------------------------------------------------------------------
            # Determine `t_detect`.
            #
            # Detected events: t_detect was computed and stored in a prior period;
            # reuse it, collapsed to t_curr if it is in the past.
            #
            # New events: search the pre-fetched access arrays for the earliest
            # obs→downlink chain in [t_curr, t_window_max], then cache the result.
            # ------------------------------------------------------------------
            if event.id in self._detection_times:
                t_detect = max(self._detection_times[event.id], t_curr)
            else:
                grid_idx   = int(event.location[2])
                gp_idx     = int(event.location[3])
                t_detect = np.inf

                for agent_name in self._agent_orbitdata:
                    # check if agent's mission matches this type of event
                    if agent_name not in self._event_type_agents[event.event_type]:
                        continue

                    gp = agent_gp_accesses[agent_name]
                    mask = ((gp['grids'] == grid_idx) &
                            (gp['gps']   == gp_idx)   &
                            (gp['times'] >= event.availability.left) &
                            (gp['times'] <= event.availability.right))
                    obs_times = gp['times'][mask]
                    if obs_times.size == 0:
                        continue
                    t_obs = float(obs_times.min())

                    dl = agent_downlinks[agent_name]
                    if dl.size == 0:
                        continue
                    valid = dl[(dl[:, 1] >= t_obs) & (dl[:, 0] <= event.availability.right)]
                    if valid.size == 0:
                        continue
                    t_downlink = float(max(valid[0, 0], t_obs))
                    if t_downlink > event.availability.right:
                        continue

                    t_detect = min(t_detect, max(t_downlink, t_curr))

                if np.isinf(t_detect):
                    continue  # no observer can deliver data before event expires

                # cache so future periods skip the search
                self._detection_times[event.id] = t_detect
                # if event not in self._detected_events:
                #     self._detected_events.append(event)

            if t_detect >= t_window_end:
                continue  # earliest data arrival falls outside this planning window

            # ------------------------------------------------------------------
            # Build task request messages stamped at t_detect — the time the
            # ground processor actually has the data, not the event start time.
            # ------------------------------------------------------------------
            msgs = []
            for mission_name, objective in objectives_by_type[event.event_type]:
                objective : EventDrivenObjective
                task = EventObservationTask(objective.parameter, event=event, objective=objective)
                task_request = TaskRequest(task,
                                           requester=state.agent_name,
                                           mission_name=mission_name,
                                           t_req=t_detect)
                task_request_msg = MeasurementRequestMessage(state.agent_name, state.agent_name, task_request.to_dict())
                msgs.append(task_request_msg)

            if not msgs:
                continue

            # ------------------------------------------------------------------
            # Schedule broadcasts from ground processor to taskable agents,
            # starting from t_detect.
            # ------------------------------------------------------------------
            all_rows = [row for _, row in orbitdata.comms_links.iter_rows_packed(
                t_detect, t_window_end, include_current=True)]

            for row in all_rows:
                t_row_start = float(row[col_start])
                t_row_end   = float(row[col_end])
                if t_row_start > t_window_end:
                    break
                if t_row_end < t_detect:
                    continue

                comps  = row[3:]
                u_comp = comps[u_idx]
                reachable = (comps == u_comp)
                reachable[u_idx] = False

                new_agents = np.where(reachable & ~agents_covered)[0]
                if new_agents.size == 0:
                    continue

                t_broadcast = max(t_row_start, t_detect)
                t_broadcasts[t_broadcast].extend(msgs)

                agents_covered[new_agents] = True
                n_covered += new_agents.size

                if n_covered >= n_targets:
                    break

            # persist coverage state for this event across planning periods
            self._announced[event.id] = set(np.where(agents_covered)[0].tolist())

        # compile accumulated messages into one BroadcastMessageAction per time slot
        for t_broadcast, msgs in t_broadcasts.items():
            assert len(msgs) > 0, "No active task requests found for broadcast time."
            bus_broadcast = BusMessage(state.agent_name, state.agent_name, msgs)
            broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))

        return sorted(broadcasts, key=lambda action: action.t_start)
        
    def update_percepts(self, state : SimulationAgentState, *args, **kwargs) -> None:
        # update list of detected events based on current time
        t_curr = state.get_time()
        for event_id, t_detect in self._detection_times.items():
            if t_detect <= t_curr:
                event = self._event_ids.get(event_id, None)
                if event:
                    event.t_detect = t_detect
                    self._detected_events.add(event)

    def print_results(self):
        # log known and generated requests
        columns = ['id','requester','lat [deg]','lon [deg]','grid index', 'GP index','severity','start time [s]','end time [s]','detection time [s]','event type']        
        data_detected = [(event.id, self._parent_agent_name, event.location[0], event.location[1], event.location[2], event.location[3], event.severity, event.t_start, event.t_start+event.d_exp, event.t_detect, event.event_type)
                for event in self._detected_events]       
             
        df = pd.DataFrame(data=data_detected, columns=columns)        
        df.to_parquet(f"{self._agent_results_dir}/events_detected.parquet", index=False)
