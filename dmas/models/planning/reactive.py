from abc import abstractmethod
from logging import Logger
from typing import List

from execsatm.tasks import GenericObservationTask
from execsatm.mission import Mission

from dmas.core.messages import SimulationMessage
from dmas.models.actions import AgentAction
from dmas.models.planning.plan import Plan, PeriodicPlan
from dmas.models.planning.planner import AbstractPlanner
from dmas.models.science.requests import TaskRequest
from dmas.models.trackers import LatestObservationTracker
from dmas.models.states import SimulationAgentState
from dmas.utils.orbitdata import OrbitData

class AbstractReactivePlanner(AbstractPlanner):
    """ Repairs previously constructed plans according to external inputs and changes in state. """

    def __init__(self, debug: bool = False, logger: Logger = None, printouts: bool = True) -> None:
        super().__init__(debug, logger, printouts)

        # initialize known preplan and current plan
        self._preplan : PeriodicPlan = PeriodicPlan([])
        self._plan : Plan = None

    @abstractmethod
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

    @abstractmethod
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : List[GenericObservationTask],
                        observation_history : LatestObservationTracker,
                    ) -> Plan:
        pass
