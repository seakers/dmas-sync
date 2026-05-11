
from collections import defaultdict
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

from tqdm import tqdm

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


class EventAnnouncerPlanner(AbstractPeriodicPlanner):
    def __init__(self,
                 agent_results_dir : str,
                 events_path : str,
                 simulation_missions : Dict[str,Mission],
                 announce_horizon : float = 24 * 3600.0,
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

        # load predefined events
        self.events : list[GeophysicalEvent] = self.load_events(events_path)
        self.simulation_missions : Dict[str,Mission] = simulation_missions
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
    
    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        # initialzie list of broadcasts to be performed in this planning period
        broadcasts : List[BroadcastMessageAction] = []
        t_curr : float = state.get_time()
        t_window_max : float = t_curr + self._period

        # drop expired events and their announcement records
        self.events = [e for e in self.events if e.is_available(t_curr)]
        active_ids = {e.id for e in self.events}
        self._announced = {eid: covered for eid, covered in self._announced.items() if eid in active_ids}

        # collect events that are available AND start within this planning window
        future_events : List[GeophysicalEvent] = [
            event for event in tqdm(self.events,
                                    desc=f'{state.agent_id}-PREPLANNER: Collecting future events',
                                    leave=False,
                                    disable=(len(self.events) < 10) or not self._printouts)
            if event.is_available(t_curr) and event.availability.left <= t_window_max
        ]

        if not future_events: return broadcasts

        # pre-index objectives by event type to avoid re-scanning missions per event
        objectives_by_type : Dict = defaultdict(list)
        for mission_name, objectives in self.simulation_missions.items():
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
        return super().print_results()
    
        # TODO print out list of announcements and how many events were announced, etc.