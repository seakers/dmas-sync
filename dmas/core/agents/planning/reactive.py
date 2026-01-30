from abc import abstractmethod
from logging import Logger
from typing import List

from dmas.modules import ClockConfig

from execsatm.tasks import GenericObservationTask
from execsatm.mission import Mission

from chess3d.agents.planning.plan import Plan, PeriodicPlan
from chess3d.agents.planning.planner import AbstractPlanner
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData

class AbstractReactivePlanner(AbstractPlanner):
    """ Repairs previously constructed plans according to external inputs and changes in state. """

    def __init__(self, debug: bool = False, logger: Logger = None) -> None:
        super().__init__(debug, logger)

        self.preplan : PeriodicPlan = None

    # @abstractmethod
    # def update_percepts(self, 
    #                     state : SimulationAgentState,
    #                     current_plan : Plan,
    #                     tasks : List[GenericObservationTask],
    #                     incoming_reqs: list, 
    #                     relay_messages: list, 
    #                     misc_messages : list,
    #                     completed_actions: list,
    #                     aborted_actions : list,
    #                     pending_actions : list
    #                     ) -> None:
        
    #     super().update_percepts(state, incoming_reqs, relay_messages, completed_actions)
        
    #     # update latest preplan
    #     if abs(state.t - current_plan.t) <= 1e-3 and isinstance(current_plan, PeriodicPlan): 
    #         self.preplan : PeriodicPlan = current_plan.copy() 

    @abstractmethod
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : List[GenericObservationTask],
                        observation_history : ObservationHistory,
                    ) -> Plan:
        pass
