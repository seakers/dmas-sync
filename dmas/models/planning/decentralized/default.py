import bisect
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from dmas.core.messages import BusMessage, MeasurementRequestMessage, SimulationMessage
from dmas.models.actions import AgentAction, BroadcastMessageAction, ObservationAction
from dmas.models.planning.plan import PeriodicPlan, Plan, ReactivePlan
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState

from execsatm.tasks import GenericObservationTask, Interval
from execsatm.mission import EventObservationTask, Mission, ObservationOpportunity

from dmas.models.trackers import TaskObservationTracker
from dmas.utils.orbitdata import OrbitData

class FixedPointingDefaultPlanner(AbstractReactivePlanner):
    """
    # Fixed Pointing Default Planner

    A decentralized planner that uses a fixed-pointing strategy for task assignment meant to be used
    to discover new events of interest. Instead of relying on default tasks to explore the environment,
    it relies on predefined task announcement broadcasts from an announcement preplanner. 

    This planner listens for future task broadcasts from the announcer and schedules observations for 
    said tasks based on a fixed-pointing strategy (e.g., always point to the same location, or point 
    to the last known location of the event). This allows the parent agent to perform observations of 
    targets with active events and therfore discover new events using its onboard data processin
    (if available). Removes the need of default tasks saturating the environment and the agents' knowledge, 
    which can be unfeasible in real-world scenarios with many events and/or long event durations.
    """
    def __init__(self, 
                 fixed_attitude : list = [0.0, 0.0, 0.0],
                 enable_broadcasts : bool = True, 
                 debug = False, 
                 logger = None, 
                 printouts = True):
        """
        Initializes the Fixed Pointing Default Planner.

        Args:
            fixed_attitude (list, optional): The fixed attitude to point to for all observations. Defaults to [0.0, 0.0, 0.0].
            enable_broadcasts (bool, optional): Whether to enable broadcasts for generated task announcement. Defaults to True.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            logger (logging.Logger, optional): Logger instance for logging. Defaults to None.
            printouts (bool, optional): Whether to enable printouts for debugging. Defaults to True
        
        """
        # initialize parent class
        super().__init__(debug, logger, printouts)

        # set parameters
        self._fixed_attitude = fixed_attitude
        self._enable_broadcasts = enable_broadcasts

        # initialize properties
        self._future_task_requests : Dict[str, TaskRequest] = {}  # task_id -> TaskRequest
        self._future_tasks : Set[EventObservationTask] = set()
        self._new_task_requests : bool = False
        self._outbox = []
        self._cross_track_fovs : dict = None

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
        # check if new periodic plan is available
        if isinstance(current_plan, PeriodicPlan) and abs(state.get_time() - current_plan.t) <= self.EPS:
            # new preplan available; save new preplan
            self._preplan : PeriodicPlan = current_plan.copy()

            # reset future task state for incoming preplan
            self._future_task_requests : Dict[str, TaskRequest] = {}
            self._future_tasks : Set[EventObservationTask] = set()

            # extract one TaskRequest template per task from the preplan's broadcast messages
            # (templates are needed later to reconstruct messages when scheduling broadcasts)
            for action in tqdm(self._preplan,
                               desc=f'{state.agent_id}-REPLANNER: Extracting task requests from preplan',
                               leave=False,
                               disable=(len(list(self._preplan)) < 10) or not self._printouts
                              ):
                if isinstance(action, BroadcastMessageAction):
                    if action.msg['msg_type'] == 'BUS':
                        for msg in action.msg['msgs']:
                            if isinstance(msg, MeasurementRequestMessage):
                                if msg.req['task']['id'] not in self._future_task_requests:
                                    req = TaskRequest.from_dict(msg.req)
                                    self._future_task_requests[req.task.id] = req
                                    self._future_tasks.add(req.task)
                    else:
                        raise NotImplementedError(f"Unexpected message type `{action.msg['msg_type']}` in broadcast action; expected `BUS`.")

            # set flag to trigger new plan generation based on new task requests
            self._new_task_requests = True

        # update latest performed observations to prevent re-scheduling of recently completed tasks
        performed = {action.obs_opp for action in completed_actions if isinstance(action, ObservationAction)}
        if performed:
            self.latest_performed_observations = performed

        # check for new task requests
        if incoming_reqs:
            # add self-generated requests to outbox for broadcast scheduling in next plan
            my_reqs = [req for req in incoming_reqs if req.requester == state.agent_name]
            self._outbox.extend(my_reqs)

            # if a task has been requested by anyone, remove it from our announcer list
            # so we no longer broadcast it (avoids duplicate announcements)
            for req in incoming_reqs:
                if isinstance(req.task, EventObservationTask):
                    to_remove = {tid for tid, r in self._future_task_requests.items()
                                 if isinstance(r.task, EventObservationTask)
                                 and r.task.event == req.task.event}
                else:
                    to_remove = {req.task.id} if req.task.id in self._future_task_requests else set()

                if to_remove:
                    for tid in to_remove:
                        self._future_task_requests.pop(tid)

            # new tasks were requested; update planning flag
            self._new_task_requests = True

    def needs_planning( self, *args, **kwargs) -> bool:
        return self._new_task_requests
        
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
            # PROCESS PRE-COMPUTED PLANS AND TASK REQUESTS

            # filter expired tasks and their request templates
            self._future_tasks = {task for task in self._future_tasks
                                   if not task.availability.is_before(t_curr)}
            self._future_task_requests = {tid: req for tid, req in self._future_task_requests.items()
                                           if not req.task.availability.is_before(t_curr)}

            # outline planning horizon interval
            if t_curr <= self._preplan.t_next:
                planning_horizon = Interval(t_curr, self._preplan.t_next)
            else:
                raise NotImplementedError(f"Current time {t_curr} is beyond the next preplan time {self._preplan.t_next}. This should not happen if replanning is triggered by new task requests from the preplanner's periodic plan, but may happen if replanning is triggered by other events (e.g., task completions). Current implementation does not handle this case yet.")

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

            # drop request templates for tasks that are no longer accessible
            self._future_task_requests = {tid: req for tid, req in self._future_task_requests.items()
                                           if req.task in accessible_future_tasks}

            # ==================================================
            # MERGE KNOWN TASKS WITH ACTIVE AND VISIBLE FUTURE TASKS FROM PREPLAN

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
                    t_prev = state._t

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
                # collect tasks we should announce: still in _future_task_requests (not claimed by others)
                tasks_to_announce : Dict[str, TaskRequest] = {}

                # # proactive: tasks already in the observation plan
                # for obs, _ in observation_sequence:
                #     for task in obs.tasks:
                #         if task.id in self._future_task_requests:
                #             tasks_to_announce[task.id] = self._future_task_requests[task.id]

                # reactive: self-generated requests sitting in the outbox
                for req in self._outbox:
                    tasks_to_announce[req.task.id] = req
                    # if req.task.id in self._future_task_requests:
                    #     tasks_to_announce[req.task.id] = self._future_task_requests[req.task.id]

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

                    # pack only still-available messages into each broadcast slot
                    for t_broadcast in broadcast_times:
                        available_msgs = [
                            msg for msg, req in zip(task_msgs, tasks_to_announce.values())
                            if t_broadcast in req.task.availability
                        ]
                        if not available_msgs:
                            continue
                        bus_broadcast = BusMessage(state.agent_name, state.agent_name, available_msgs)
                        broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))

                # clear outbox — contents have been processed into the broadcast schedule
                self._outbox.clear()

            actions = sorted(observations + broadcasts, key=lambda action: action.t_start)
            plan = ReactivePlan(actions, t=t_curr, t_next=self._preplan.t_next)

            return plan

        finally:
            # reset planning flag
            self._new_task_requests = False
    

    def print_results(self):
        # TODO 
        return super().print_results()
