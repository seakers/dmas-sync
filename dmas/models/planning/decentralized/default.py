from typing import List

from dmas.core.messages import SimulationMessage
from dmas.models.actions import AgentAction, BroadcastMessageAction
from dmas.models.planning.plan import PeriodicPlan, Plan
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState

from execsatm.tasks import GenericObservationTask
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
        self._scheduled_requests : List[TaskRequest] = []
        self._future_tasks : List[EventObservationTask] = []

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
            self._scheduled_requests : List[TaskRequest] = []
            self._future_tasks : List[EventObservationTask] = []

            # extract future task requests and broadcasts from periodic plan
            for action in self._preplan:
                # check action type and extract relevant info
                if isinstance(action, BroadcastMessageAction):
                    # broadcast action; extract message and check if it's a task announcement
                    self._scheduled_broadcasts.append(action)

                    # extract future task from broadcast message if applicable
                    if isinstance(action.message, SimulationMessage) and action.message.type == "task_announcement":
                        future_task = EventObservationTask.from_dict(action.message.content['task'])
                        self._future_tasks.append(future_task)
                elif isinstance(action, TaskRequest) and action not in self._scheduled_requests:
                    self._scheduled_requests.append(action)

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
        
        # extract scheduled tasks

        return [] # TODO
