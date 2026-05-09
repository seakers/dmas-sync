from typing import List, Set

from dmas.core.messages import MeasurementRequestMessage, SimulationMessage
from dmas.models.actions import AgentAction, BroadcastMessageAction
from dmas.models.planning.plan import PeriodicPlan, Plan, ReactivePlan
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState

from execsatm.tasks import GenericObservationTask, Interval
from execsatm.mission import EventObservationTask, Mission

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
                 debug = False, 
                 logger = None, 
                 printouts = True):
        # initialize parent class
        super().__init__(debug, logger, printouts)

        # set parameters
        self._fixed_attitude = fixed_attitude

        # initialize properties
        self._new_task_requests : bool = False
        self._scheduled_broadcasts : List[BroadcastMessageAction] = []
        self._scheduled_requests : Set[TaskRequest] = set()
        self._future_tasks : Set[EventObservationTask] = set()

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

            # reset scheduled broadcasts and requests
            self._scheduled_broadcasts : List[BroadcastMessageAction] = []
            self._scheduled_requests : Set[TaskRequest] = set()
            self._future_tasks : Set[EventObservationTask] = set()

            # extract future task requests and broadcasts from periodic plan
            for action in self._preplan:
                # check action type and extract relevant info
                if isinstance(action, BroadcastMessageAction):
                    # broadcast action; extract message and check if it's a task announcement
                    self._scheduled_broadcasts.append(action)

                    # extract future task from broadcast message if applicable
                    if action.msg['msg_type'] == 'BUS':
                        msgs = action.msg['msgs']
                        for msg in msgs:
                            if isinstance(msg, MeasurementRequestMessage):
                                req = TaskRequest.from_dict(msg.req)
                                self._scheduled_requests.add(req)
                                self._future_tasks.add(req.task)                            
                    else:
                        raise NotImplementedError(f"Unexpected message type `{action.msg['msg_type']}` in broadcast action; expected `BUS`.")

            # set flag to trigger new plan generation based on new task requests
            self._new_task_requests = True
    
        # check for new task requests
        if incoming_reqs:
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

            # PROCESS PRE-COMPUTED PLANS AND TASK REQUESTS -----------------------------------

            # filter expired tasks from future tasks
            self._future_tasks = set([task for task in self._future_tasks 
                                      if not task.availability.is_before(t_curr)])
            
            # remove requests for expired tasks from scheduled requests
            self._scheduled_requests = set([req for req in self._scheduled_requests 
                                            if not req.task.availability.is_before(t_curr)])

            # compile instrument field of view specifications   
            cross_track_fovs : dict = self._collect_fov_specs(specs)

            # compile agility specifications
            max_slew_rate, max_torque = self._collect_agility_specs(specs)

            # Outline planning horizon interval
            if t_curr <= self._preplan.t_next:
                planning_horizon = Interval(t_curr, self._preplan.t_next)
            else:
                # planning_horizon = Interval(t_curr, t_curr + self._preplan._horizon)
                raise NotImplementedError(f"Current time {t_curr} is beyond the next preplan time {self._preplan.t_next}. This should not happen if replanning is triggered by new task requests from the preplanner's periodic plan, but may happen if replanning is triggered by other events (e.g., task completions). Current implementation does not handle this case yet.")
            
            # calculate coverage opportunities for tasks
            access_opportunities : dict[tuple] = self.calculate_access_opportunities(self._future_tasks, planning_horizon, orbitdata)

            # remove tasks from future tasks if they are not accessible 
            accessible_future_tasks : Set[EventObservationTask] = set()
            for task in list(self._future_tasks):

                for *__,grid_index,gp_index in task.location:
                    matching_accesses = access_opportunities.get((task, grid_index, gp_index))
                    if matching_accesses:
                        accessible_future_tasks.add(task)
                        break
            self._future_tasks = accessible_future_tasks

            # remove requests for inaccessible tasks from scheduled requests
            accessible_future_reqs : Set[TaskRequest] = {
                req for req in self._scheduled_requests
                if req.task in accessible_future_tasks
            }
            self._scheduled_requests = accessible_future_reqs

            if accessible_future_tasks:
                x = 1 # breakpoint for debugging
            else:
                x = 1 # breakpoint for debugging

            # ----------------------------

            return ReactivePlan([], t=t_curr, t_next=self._preplan.t_next) # TODO
        
        finally:
            # reset planning flag
            self._new_task_requests = False 
    
    def print_results(self):
        # TODO 
        return super().print_results()
