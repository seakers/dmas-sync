import bisect
import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from dmas.core.messages import BusMessage, MeasurementRequestMessage, SimulationMessage
from dmas.models.actions import AgentAction, BroadcastMessageAction, ObservationAction
from dmas.models.planning.plan import Plan, ReactivePlan
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

    A decentralized planner that uses a fixed-pointing strategy for task assignment. Events are
    loaded directly at initialization from a CSV file, eliminating the need for a separate
    announcer preplanner. The planner generates observations of all accessible future event tasks
    and broadcasts self-generated task requests to other agents.
    """
    def __init__(self,
                 events_path : str = None,
                 mission : Mission = None,
                 fixed_attitude : list = [0.0, 0.0, 0.0],
                 enable_broadcasts : bool = True,
                 debug = False,
                 logger = None,
                 printouts = True):
        """
        Initializes the Fixed Pointing Default Planner.

        Args:
            events_path (str, optional): Path to the CSV file containing geophysical events. Defaults to None.
            mission (Mission, optional): The mission object containing objectives for this agent. Defaults to None.
            fixed_attitude (list, optional): The fixed attitude to point to for all observations. Defaults to [0.0, 0.0, 0.0].
            enable_broadcasts (bool, optional): Whether to enable broadcasts for generated task announcements. Defaults to True.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            logger (logging.Logger, optional): Logger instance for logging. Defaults to None.
            printouts (bool, optional): Whether to enable printouts for debugging. Defaults to True.
        """
        # initialize parent class
        super().__init__(debug, logger, printouts)

        # set parameters
        self._fixed_attitude = fixed_attitude
        self._enable_broadcasts = enable_broadcasts

        # initialize properties
        self._outbox = []
        self._cross_track_fovs : dict = None

        # load events and build future tasks directly if events_path and mission are provided
        if events_path is not None and mission is not None:
            self._future_tasks : Set[EventObservationTask] = self._load_tasks_from_events(events_path, mission)
            self._new_task_requests : bool = True
        else:
            self._future_tasks : Set[EventObservationTask] = set()
            self._new_task_requests : bool = False

        # track task IDs seen so far to avoid replanning on already-known tasks
        self._known_task_ids : Set[str] = {task.id for task in self._future_tasks}

        # cache access opportunities by (grid_index, gp_index); reset when horizon extends
        self._access_opportunities : dict = {}
        self._access_horizon : float = -np.inf

    def _load_tasks_from_events(self, events_path : str, mission : Mission) -> Set[EventObservationTask]:
        """ Loads geophysical events from a CSV and builds EventObservationTask objects for each event x objective pair. """
        if not os.path.isfile(events_path):
            raise ValueError(f'`events_path` must point to an existing file: {events_path}')

        events_df = pd.read_csv(events_path)
        events = []
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
            events.append(event)

        tasks : Set[EventObservationTask] = set()
        for objective in mission:
            if not isinstance(objective, EventDrivenObjective):
                continue
            for event in events:
                if event.event_type == objective.event_type:
                    task = EventObservationTask(objective.parameter, event=event, objective=objective)
                    tasks.add(task)

        return tasks

    def update_percepts(self,
                        state : SimulationAgentState,
                        current_plan : Plan,
                        tasks : List[GenericObservationTask],
                        incoming_reqs: List[TaskRequest],
                        misc_messages : List[SimulationMessage],
                        completed_actions: List[AgentAction],
                        aborted_actions : List[AgentAction],
                        pending_actions : List[AgentAction]
                    ) -> None:
        # update latest performed observations to prevent re-scheduling of recently completed tasks
        performed = {action.obs_opp for action in completed_actions if isinstance(action, ObservationAction)}
        if performed:
            self.latest_performed_observations = performed

        # check for new task requests
        if incoming_reqs:
            # add self-generated requests to outbox for broadcast scheduling in next plan
            my_reqs = [req for req in incoming_reqs if req.requester == state.agent_name]
            self._outbox.extend(my_reqs)

            # only trigger replanning if at least one request carries a task ID not seen before
            new_ids = {req.task.id for req in incoming_reqs} - self._known_task_ids
            if new_ids:
                self._new_task_requests = True
                self._known_task_ids.update(new_ids)

    def needs_planning( self, *args, **kwargs) -> bool:
        """ Replan if there are new task requests with previously unseen task IDs. """
        return self._new_task_requests

    def calculate_access_opportunities(self,
                                       tasks,
                                       planning_horizon : Interval,
                                       orbitdata : OrbitData
                                    ) -> dict:
        if planning_horizon.is_empty():
            return {}

        # reset cache if planning horizon has extended to the right
        if planning_horizon.right > self._access_horizon + 1e-3:
            self._access_opportunities = {}
            self._access_horizon = planning_horizon.right

        # identify locations not yet cached
        needed_locs = set()
        for task in tasks:
            for *__, grid_index, gp_index in task.location:
                loc = (int(grid_index), int(gp_index))
                if loc not in self._access_opportunities:
                    needed_locs.add(loc)

        # compute only for tasks that touch uncached locations, then merge into cache
        if needed_locs:
            tasks_to_compute = [task for task in tasks
                                 if any((int(gi), int(gp)) in needed_locs
                                        for *__, gi, gp in task.location)]
            new_accesses = super().calculate_access_opportunities(tasks_to_compute, planning_horizon, orbitdata)
            self._access_opportunities.update(new_accesses)

        # return the cache entries relevant to the requested tasks
        result = {}
        for task in tasks:
            for *__, grid_index, gp_index in task.location:
                loc = (int(grid_index), int(gp_index))
                if loc in self._access_opportunities:
                    result[loc] = self._access_opportunities[loc]
        return result

    def generate_plan(  self,
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : List[GenericObservationTask],
                        observation_history : TaskObservationTracker,
                    ) -> Plan:
        try:
            # get current time
            t_curr : float = state.get_time()

            # ==================================================
            # PROCESS TASK REQUESTS

            # filter expired tasks
            self._future_tasks = {task for task in self._future_tasks
                                   if not task.availability.is_before(t_curr)}

            # outline planning horizon: from now to the end of the last future task
            if self._future_tasks:
                t_horizon_end = max(task.availability.right for task in self._future_tasks)
            elif tasks:
                t_horizon_end = max(task.availability.right for task in tasks
                                    # if isinstance(task, GenericObservationTask)
                                    )
            else:
                t_horizon_end = t_curr
            planning_horizon = Interval(t_curr, t_horizon_end)

            # calculate coverage opportunities for future tasks
            access_opportunities : dict = self.calculate_access_opportunities(self._future_tasks, planning_horizon, orbitdata)

            # keep only future tasks that have at least one access opportunity
            accessible_future_tasks : Set[EventObservationTask] = set()
            for task in tqdm(self._future_tasks,
                             desc=f'{state.agent_id}-REPLANNER: Filtering accessible future tasks',
                             leave=False,
                             disable=(len(self._future_tasks) < 10) or not self._printouts
                            ):
                for *__, grid_index, gp_index in task.location:
                    if access_opportunities.get((grid_index, gp_index)):
                        accessible_future_tasks.add(task)
                        break
            self._future_tasks = accessible_future_tasks

            # ==================================================
            # MERGE KNOWN TASKS WITH ACTIVE AND VISIBLE FUTURE TASKS

            # pre-build a set of events already covered by known tasks for O(1) lookup
            known_events = {task.event for task in tasks if isinstance(task, EventObservationTask)}

            # start from known tasks; add future tasks only if their event isn't already covered
            schedulable_tasks = list(tasks)
            for future_task in tqdm(self._future_tasks,
                                    desc=f'{state.agent_id}-REPLANNER: Merging future tasks with known tasks',
                                    leave=False,
                                    disable=(len(self._future_tasks) < 10) or not self._printouts
                                   ):
                if future_task.event not in known_events:
                    schedulable_tasks.append(future_task)

            # remove duplicates from schedulable tasks
            schedulable_tasks = list({task.id: task for task in schedulable_tasks}.values())

            # filter only available tasks
            available_tasks : list[GenericObservationTask] = \
                [task for task in schedulable_tasks 
                    if isinstance(task, GenericObservationTask)
                    and task.availability.overlaps(planning_horizon)
                ]
            
            # calculate coverage opportunities for tasks
            access_opportunities : dict[tuple] = self.calculate_access_opportunities(available_tasks, planning_horizon, orbitdata)

            # compile instrument field of view specifications (cached — specs never change)
            if self._cross_track_fovs is None:
                self._cross_track_fovs = self._collect_fov_specs(specs)
            cross_track_fovs = self._cross_track_fovs

            # create task observation opportunities from known tasks and future access opportunities
            observation_opportunities : list[ObservationOpportunity] \
                = self.create_observation_opportunities_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)
            
            # ==================================================
            # GENERATE PLAN

            # sort observation opportunities by start time >> duration >> priority 
            observation_opportunities.sort(
                key=lambda observation_opportunity : (
                    observation_opportunity.accessibility.left, 
                    -observation_opportunity.min_duration, 
                    observation_opportunity.get_priority()
                )
            )

            # generate plan
            # three parallel lists kept sorted by t_start; non-overlapping actions guarantee t_end is also sorted
            scheduled_actions  : list[ObservationAction] = []
            scheduled_t_starts : list[float] = []
            scheduled_t_ends   : list[float] = []

            for obs in tqdm(observation_opportunities,
                            desc=f'{state.agent_id}-PLANNER: Pre-Scheduling Observations',
                            leave=False,
                            disable=(len(observation_opportunities) < 10) or not self._printouts
                            ):

                # latest already-scheduled action ending before this window closes (O(log N))
                idx_prev = bisect.bisect_right(scheduled_t_ends, obs.accessibility.right + 1e-6) - 1
                if idx_prev >= 0:
                    t_prev = scheduled_actions[idx_prev].t_end
                else:
                    t_prev = t_curr

                # earliest already-scheduled action starting after this window opens (O(log N))
                idx_next = bisect.bisect_left(scheduled_t_starts, obs.accessibility.left - 1e-6)
                if idx_next < len(scheduled_actions):
                    t_next = scheduled_actions[idx_next].t_start
                else:
                    t_next = obs.accessibility.right

                # set task observation angle
                th_img = np.average((obs.slew_angles.left, obs.slew_angles.right))

                # select task imaging time and duration
                t_img = max(t_prev, obs.accessibility.left)
                d_img = obs.min_duration

                # check if the observation fits within the task's accessibility window
                if t_img + d_img not in obs.accessibility: continue

                # check if the observation is feasible
                prev_action_feasible : bool = (t_prev <= t_img - 1e-6)
                curr_action_feasible : bool = (abs(th_img) <= cross_track_fovs[obs.instrument_name] / 2.0)
                next_action_feasible : bool = (t_img + d_img <= t_next - 1e-6)

                if prev_action_feasible and curr_action_feasible and next_action_feasible:
                    action = ObservationAction(obs.instrument_name, th_img, t_img, d_img, obs)

                    # insert in sorted position by t_start
                    pos = bisect.bisect_left(scheduled_t_starts, t_img)
                    scheduled_t_starts.insert(pos, t_img)
                    scheduled_t_ends.insert(pos, t_img + d_img)
                    scheduled_actions.insert(pos, action)

            # already sorted by t_start
            observations = scheduled_actions

            # compile broadcasts if enabled
            broadcasts : List[AgentAction] = []
            if self._enable_broadcasts:
                # collect self-generated requests sitting in the outbox
                tasks_to_announce : Dict[str, TaskRequest] = {}
                for req in self._outbox:
                    tasks_to_announce[req.task.id] = req

                if tasks_to_announce:
                    u_idx = orbitdata.comms_target_indices[OrbitData.safe_name(state.agent_name)]
                    n_cols = len(orbitdata.comms_target_columns)
                    n_targets = len(orbitdata.comms_targets)

                    # build the message list once
                    task_msgs = [
                        MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                        for req in tasks_to_announce.values()
                    ]

                    # compute broadcast times once via a single coverage pass over the comm links
                    t_window_start = t_curr
                    t_window_end = max(req.task.availability.right for req in tasks_to_announce.values())

                    agents_covered = np.zeros(n_cols, dtype=bool)
                    n_covered = 0
                    broadcast_times = []

                    for _, row in orbitdata.comms_links.iter_rows_packed(t_window_start, t_window_end, include_current=True):
                        t_row_start = float(row[orbitdata.comms_links._col["start"]])
                        comps = row[3:]

                        u_comp = comps[u_idx]
                        reachable = (comps == u_comp)
                        reachable[u_idx] = False

                        new_agents = np.where(reachable & ~agents_covered)[0]
                        if new_agents.size == 0:
                            continue

                        broadcast_times.append(max(t_row_start, t_window_start))
                        agents_covered[new_agents] = True
                        n_covered += new_agents.size

                        if n_covered >= n_targets:
                            break

                    # pack only still-available messages into each broadcast slot,
                    # advancing each time past any overlapping observation window
                    for t_broadcast in broadcast_times:
                        # advance past any observation that contains t_broadcast
                        while True:
                            idx = bisect.bisect_right(scheduled_t_starts, t_broadcast) - 1
                            if idx >= 0 and scheduled_t_ends[idx] > t_broadcast:
                                t_broadcast = scheduled_t_ends[idx]
                            else:
                                break

                        available_msgs = [
                            msg for msg, req in zip(task_msgs, tasks_to_announce.values())
                            if t_broadcast in req.task.availability
                        ]
                        if not available_msgs:
                            continue
                        bus_broadcast = BusMessage(state.agent_name, state.agent_name, available_msgs)
                        broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))
                        # broadcasts.append(BroadcastMessageAction(bus_broadcast, t_broadcast))

                # clear outbox; contents have been processed into the broadcast schedule
                self._outbox.clear()

            # actions = sorted(observations + broadcasts, key=lambda action: action.t_start)
            # plan = ReactivePlan(actions, t=t_curr, t_next=np.Inf)
            plan = ReactivePlan(observations, broadcasts, t=t_curr, t_next=np.Inf)

            return plan

        finally:
            # reset planning flag
            self._new_task_requests = False
    

    def print_results(self):
        # TODO 
        return super().print_results()
