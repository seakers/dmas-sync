import bisect
import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from dmas.core.messages import BusMessage, MeasurementRequestMessage, SimulationMessage
from dmas.models.actions import AgentAction, BroadcastMessageAction, ObservationAction, WaitAction
from dmas.models.planning.plan import PeriodicPlan, Plan, ReactivePlan
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState

from execsatm.events import GeophysicalEvent
from execsatm.objectives import EventDrivenObjective
from execsatm.tasks import GenericObservationTask, Interval
from execsatm.mission import EventObservationTask, Mission, ObservationOpportunity

from dmas.models.trackers import TaskObservationTracker
from dmas.utils.orbitdata import OrbitData

class FixedPointingDefaultPlanner(AbstractReactivePlanner):
    """
    # Fixed Pointing Default Planner

    A decentralized planner that uses a fixed-pointing strategy for task assignment.
    Events are loaded lazily on the first generate_plan call rather than at init, so
    the raw event data is only in memory for one satellite at a time (agents are stepped
    sequentially).  A one-time full-horizon access check permanently filters out events
    this satellite can never observe; from then on only a bounded rolling window is used
    for scheduling and access caching, keeping per-period memory small.
    """

    def __init__(self,
                 events_path: str = None,
                 mission: Mission = None,
                 fixed_attitude: list = [0.0, 0.0, 0.0],
                 enable_broadcasts: bool = True,
                 replan_horizon: float = 3600.0,
                 debug=False,
                 logger=None,
                 printouts=True):
        super().__init__(debug, logger, printouts)

        self._fixed_attitude = fixed_attitude
        self._enable_broadcasts = enable_broadcasts
        self._replan_horizon = replan_horizon

        # lazy init: only the path is stored; tasks are built on the first generate_plan call
        # mission is not stored — the one passed to generate_plan is used instead
        self._events_path = events_path
        self._initialized: bool = False

        # task state
        self._future_tasks: Set[EventObservationTask] = set()
        self._known_task_ids: Set[str] = set()

        # replanning triggers
        self._new_task_requests: bool = False
        self._new_my_requests: bool = False
        self._next_replan_time: float = -np.inf  # fires immediately on first step

        # self-generated announcements: keyed by task id, kept until the event expires or all targets are covered
        self._pending_announcements: Dict[str, TaskRequest] = {}
        # per-task coverage: set of agent column indices already notified; mirrors _pending_announcements keys
        self._announced: Dict[str, set] = {}

        # intra-period access cache; flushed at the start of every planning period
        self._access_opportunities: dict = {}

        # instrument FOV cache (never changes after first plan)
        self._cross_track_fovs: dict = None

        # initialize with an empty periodic plan to trigger the first replan immediately
        self._plan = PeriodicPlan(t=-1,horizon=replan_horizon,t_next=0.0)

    # ------------------------------------------------------------------
    # Lazy initializer

    def _initialize_from_events(self,
                                 state: SimulationAgentState,
                                 mission: Mission,
                                 orbitdata: OrbitData) -> None:
        """
        Load events from CSV, build EventObservationTasks, and permanently filter to those
        this satellite can observe at any point during the simulation.

        Raw event data and the full-horizon access table are local variables and are released
        the moment this method returns — only the filtered task set is retained.
        """
        if self._events_path is None or mission is None:
            return

        t_curr = state.get_time()

        # load events (held only for the duration of this call)
        events_df = pd.read_csv(self._events_path)
        tasks: Set[EventObservationTask] = set()
        for _, row in events_df.iterrows():
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row.get('id', None)
            )
            for objective in mission:
                if isinstance(objective, EventDrivenObjective) and event.event_type == objective.event_type:
                    tasks.add(EventObservationTask(objective.parameter, event=event, objective=objective))

        # drop tasks that have already expired
        tasks = {task for task in tasks if not task.availability.is_before(t_curr)}
        if not tasks:
            return

        # one-time full-horizon access check: permanently discard events this satellite can never reach
        t_full_end = max(task.availability.right for task in tasks)
        full_horizon = Interval(t_curr, t_full_end)
        full_accesses = super().calculate_access_opportunities(tasks, full_horizon, orbitdata)

        accessible_locs = {loc for loc, opps in full_accesses.items() if opps}
        self._future_tasks = {
            task for task in tasks
            if any((int(gi), int(gp)) in accessible_locs for *__, gi, gp in task.location)
        }

        # merge with any IDs already seen from incoming requests before first plan
        self._known_task_ids |= {task.id for task in self._future_tasks}

        # check if any task has a duplicate ID
        if len(self._known_task_ids) < len(self._future_tasks):
            print("WARNING: Duplicate task IDs detected in events data; ID-based request tracking may be unreliable.")

        # full_accesses and events_df go out of scope here; per-period cache stays empty
        self._new_task_requests = True

    # ------------------------------------------------------------------
    # Percept update

    def update_percepts(self,
                        state: SimulationAgentState,
                        current_plan: Plan,
                        tasks: List[GenericObservationTask],
                        incoming_reqs: List[TaskRequest],
                        misc_messages: List[SimulationMessage],
                        completed_actions: List[AgentAction],
                        aborted_actions: List[AgentAction],
                        pending_actions: List[AgentAction]
                        ) -> None:
        # check for new periodic plan and update preplan if found
        super().update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)

        # process incoming requests for task tracking and replanning triggers
        if incoming_reqs:
            for req in incoming_reqs:
                if req.requester == state.agent_name:
                    # keep self-generated requests until their events expire
                    self._pending_announcements[req.task.id] = req

            new_ids = {req.task.id for req in incoming_reqs} - self._known_task_ids
            if new_ids:
                self._new_task_requests = True
                self._known_task_ids.update(new_ids)

            if any(req.requester == state.agent_name for req in incoming_reqs):
                self._new_my_requests = True

    def needs_planning( self, 
                        state : SimulationAgentState,
                        _ : object,
                        current_plan : Plan,
                        __ : OrbitData
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    
        return (self._new_task_requests 
                or self._new_my_requests 
                or self.needs_periodic_replan(state, current_plan)
            )

    def needs_periodic_replan(self, state, current_plan: Plan) -> bool:
        """
        From periodic planners:
        """
        if (self._plan.t < 0                   # simulation just started
            or state.get_time() >= self._plan.t_next):    # or periodic planning period has been reached
            
            pending_actions = [action for action in current_plan
                               if action.t_start <= self._plan.t_next]
            
            return not bool(pending_actions)     # no actions left to do before the end of the replanning period 
        return False        

    # ------------------------------------------------------------------
    # Access opportunity cache (intra-period incremental fill, no horizon tracking)

    def calculate_access_opportunities(self,
                                       tasks,
                                       planning_horizon: Interval,
                                       orbitdata: OrbitData
                                       ) -> dict:
        if planning_horizon.is_empty():
            return {}

        # only compute locations that are not already in the period cache
        needed_locs = set()
        for task in tasks:
            for *__, grid_index, gp_index in task.location:
                loc = (int(grid_index), int(gp_index))
                if loc not in self._access_opportunities:
                    needed_locs.add(loc)

        if needed_locs:
            tasks_to_compute = [task for task in tasks
                                 if any((int(gi), int(gp)) in needed_locs
                                        for *__, gi, gp in task.location)]
            new_accesses = super().calculate_access_opportunities(tasks_to_compute, planning_horizon, orbitdata)
            self._access_opportunities.update(new_accesses)

        result = {}
        for task in tasks:
            for *__, grid_index, gp_index in task.location:
                loc = (int(grid_index), int(gp_index))
                if loc in self._access_opportunities:
                    result[loc] = self._access_opportunities[loc]
        return result

    # ------------------------------------------------------------------
    # Plan generation

    def generate_plan(self,
                      state: SimulationAgentState,
                      specs: object,
                      current_plan: Plan,
                      orbitdata: OrbitData,
                      mission: Mission,
                      tasks: List[GenericObservationTask],
                      observation_history: TaskObservationTracker
                      ) -> Plan:
        t_curr: float = state.get_time()
        try:
            # --- lazy initialization (once, sequentially across all agents) ---
            if not self._initialized:
                # initialize tasks from event data
                self._initialize_from_events(state, mission, orbitdata)
                # cache cross-track FOVs for all instruments 
                self._cross_track_fovs = self._collect_fov_specs(specs)
                # fix initialized flag 
                self._initialized = True

            # flush access cache only at period boundaries; mid-period replans (triggered by
            # new incoming requests) keep the warm cache and only add missing locations
            if t_curr >= self._plan.t_next:
                self._access_opportunities = {}

            # ==================================================
            # PROCESS TASK REQUESTS

            # expire tasks that are no longer available
            self._future_tasks = {task for task in self._future_tasks
                                   if not task.availability.is_before(t_curr)}

            # bound planning horizon to the rolling replan window
            t_next = self._plan.t_next
            while t_next <= t_curr:
                t_next += self._replan_horizon
            planning_horizon = Interval(t_curr, t_next)

            # access opportunities for future tasks within this window
            access_opportunities: dict = self.calculate_access_opportunities(
                self._future_tasks, planning_horizon, orbitdata)

            # tasks accessible in this window — used for scheduling only, does NOT prune
            # _future_tasks (a task outside this window may still be accessible in the next)
            accessible_this_period: Set[EventObservationTask] = set()
            for task in tqdm(self._future_tasks,
                             desc=f'{state.agent_id}-REPLANNER: Filtering accessible future tasks',
                             leave=False,
                             disable=(len(self._future_tasks) < 10) or not self._printouts):
                for *__, grid_index, gp_index in task.location:
                    if access_opportunities.get((int(grid_index), int(gp_index))):
                        accessible_this_period.add(task)
                        break

            # ==================================================
            # MERGE KNOWN TASKS WITH ACCESSIBLE FUTURE TASKS

            known_events = {task.event for task in tasks if isinstance(task, EventObservationTask)}
            schedulable_tasks = list(tasks)
            for future_task in tqdm(accessible_this_period,
                                    desc=f'{state.agent_id}-REPLANNER: Merging future tasks with known tasks',
                                    leave=False,
                                    disable=(len(accessible_this_period) < 10) or not self._printouts):
                if future_task.event not in known_events:
                    schedulable_tasks.append(future_task)

            # remove duplicates
            schedulable_tasks = list({task.id: task for task in schedulable_tasks}.values())

            # filter to tasks available within the planning horizon
            available_tasks: list[GenericObservationTask] = [
                task for task in schedulable_tasks
                if isinstance(task, GenericObservationTask)
                and task.availability.overlaps(planning_horizon)
            ]

            # access opportunities for all schedulable tasks (incremental; cache is warm)
            access_opportunities: dict[tuple] = self.calculate_access_opportunities(
                available_tasks, planning_horizon, orbitdata)

            # create observation opportunities
            observation_opportunities: list[ObservationOpportunity] = \
                self.create_observation_opportunities_from_accesses(
                    available_tasks, access_opportunities, self._cross_track_fovs, orbitdata)

            # ==================================================
            # GENERATE PLAN

            observation_opportunities.sort(
                key=lambda opp: (opp.accessibility.left, 
                                 -opp.min_duration, 
                                 opp.get_priority())
                            )

            scheduled_actions: list[ObservationAction] = []
            scheduled_t_starts: list[float] = []
            scheduled_t_ends: list[float] = []

            for obs in tqdm(observation_opportunities,
                            desc=f'{state.agent_id}-PLANNER: Pre-Scheduling Observations',
                            leave=False,
                            disable=(len(observation_opportunities) < 10) or not self._printouts):

                idx_prev = bisect.bisect_right(scheduled_t_ends, obs.accessibility.right + 1e-6) - 1
                if idx_prev >= 0:
                    t_prev_obs = scheduled_actions[idx_prev].t_end
                else:
                    t_prev_obs = t_curr

                idx_next = bisect.bisect_left(scheduled_t_starts, obs.accessibility.left - 1e-6)
                if idx_next < len(scheduled_actions):
                    t_next_obs = scheduled_actions[idx_next].t_start
                else:
                    t_next_obs = obs.accessibility.right

                th_img = np.average((obs.slew_angles.left, obs.slew_angles.right))
                t_img = max(t_prev_obs, obs.accessibility.left)
                d_img = obs.min_duration

                if t_img + d_img not in obs.accessibility:
                    continue

                prev_action_feasible: bool = (t_prev_obs <= t_img - 1e-6)
                curr_action_feasible: bool = (abs(th_img) <= self._cross_track_fovs[obs.instrument_name] / 2.0)
                next_action_feasible: bool = (t_img + d_img <= t_next_obs - 1e-6)

                if prev_action_feasible and curr_action_feasible and next_action_feasible:
                    action = ObservationAction(obs.instrument_name, th_img, t_img, d_img, obs)
                    pos = bisect.bisect_left(scheduled_t_starts, t_img)
                    scheduled_t_starts.insert(pos, t_img)
                    scheduled_t_ends.insert(pos, t_img + d_img)
                    scheduled_actions.insert(pos, action)

            observations = scheduled_actions

            # ==================================================
            # BROADCASTS

            broadcasts: List[AgentAction] = []
            if self._enable_broadcasts:
                # expire announcements for events that are no longer active
                self._pending_announcements = {
                    tid: req for tid, req in self._pending_announcements.items()
                    if not req.task.availability.is_before(t_curr)
                }

                if self._pending_announcements:
                    u_idx = orbitdata.comms_target_indices[OrbitData.safe_name(state.agent_name)]
                    n_cols = len(orbitdata.comms_target_columns)
                    n_targets = len(orbitdata.comms_targets)

                    # drop tasks that have already reached every target agent
                    self._pending_announcements = {
                        tid: req for tid, req in self._pending_announcements.items()
                        if len(self._announced.get(tid, set())) < n_targets
                    }
                    # keep _announced in sync with _pending_announcements
                    self._announced = {tid: s for tid, s in self._announced.items()
                                       if tid in self._pending_announcements}

                if self._pending_announcements:
                    task_items = list(self._pending_announcements.items())
                    task_msgs = [
                        MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                        for _, req in task_items
                    ]

                    # bound broadcast window to the replan horizon (or event expiry, whichever comes first)
                    t_window_start = t_curr
                    t_window_end = min(
                        t_next,
                        max(req.task.availability.right for req in self._pending_announcements.values())
                    )

                    # seed agents_covered with the intersection of per-task coverage:
                    # agents who have already received ALL pending tasks need no further targeting
                    covered_sets = [self._announced.get(tid, set()) for tid, _ in task_items]
                    covered_for_all = set.intersection(*covered_sets) if len(covered_sets) > 1 else set(covered_sets[0])
                    agents_covered = np.zeros(n_cols, dtype=bool)
                    for idx in covered_for_all:
                        agents_covered[idx] = True
                    n_covered = int(agents_covered.sum())

                    # collect (t_broadcast, new_agents) so we can update per-task coverage after the sweep
                    broadcast_slots: list = []

                    for _, row in orbitdata.comms_links.iter_rows_packed(
                            t_window_start, t_window_end, include_current=True):
                        t_row_start = float(row[orbitdata.comms_links._col["start"]])
                        comps = row[3:]
                        u_comp = comps[u_idx]
                        reachable = (comps == u_comp)
                        reachable[u_idx] = False
                        new_agents = np.where(reachable & ~agents_covered)[0]
                        if new_agents.size == 0:
                            continue
                        broadcast_slots.append((max(t_row_start, t_window_start), new_agents))
                        agents_covered[new_agents] = True
                        n_covered += new_agents.size
                        if n_covered >= n_targets:
                            break

                    for t_broadcast, new_agents in broadcast_slots:
                        # advance past any observation that contains t_broadcast
                        while True:
                            idx = bisect.bisect_right(scheduled_t_starts, t_broadcast) - 1
                            if idx >= 0 and scheduled_t_ends[idx] > t_broadcast:
                                t_broadcast = scheduled_t_ends[idx]
                            else:
                                break

                        available_msgs = []
                        for msg, (tid, req) in zip(task_msgs, task_items):
                            if t_broadcast in req.task.availability:
                                available_msgs.append(msg)
                                # record these agents as notified for this task
                                self._announced.setdefault(tid, set()).update(new_agents.tolist())

                        if not available_msgs:
                            continue
                        bus_broadcast = BusMessage(state.agent_name, state.agent_name, available_msgs)
                        broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))

                    # remove tasks that have now reached all targets after this period's broadcasts
                    self._pending_announcements = {
                        tid: req for tid, req in self._pending_announcements.items()
                        if len(self._announced.get(tid, set())) < n_targets
                    }
                    self._announced = {tid: s for tid, s in self._announced.items()
                                       if tid in self._pending_announcements}

                # _pending_announcements persist across periods; removed only on expiry or full coverage

            waits = self._schedule_periodic_replan(state, t_next)

            self._plan = ReactivePlan([], t=t_curr, t_next=t_next)
            return ReactivePlan(observations, broadcasts, waits, t=t_curr, t_next=t_next)

        finally:
            # reset planning flags and advance the periodic replan timer
            self._new_task_requests = False
            self._new_my_requests = False

    def _schedule_periodic_replan(self, state : SimulationAgentState, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        # ensure next planning time is in the future
        assert state.get_time() <= t_next, "Next planning time must be in the future."
        # schedule wait action for next planning time
        return [WaitAction(t_next,t_next)] if not np.isinf(t_next) else []

    def print_results(self):
        return super().print_results()
