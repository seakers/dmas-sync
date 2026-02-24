from dmas.models.actions import AgentAction

from dmas.models.actions import ObservationAction, action_from_dict
from dmas.models.planning.plan import PeriodicPlan, Plan
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.states import SimulationAgentState
from dmas.core.messages import PlanMessage


class WorkerPlanner(AbstractPeriodicPlanner):
    """
    Worker planner class that handles replanning tasks for agents.
    It processes the replanning requests and updates the agent's plan accordingly.
    """
    def __init__(self, dealer_name : str, debug = False, logger = None, printouts : bool = True) -> None:
        super().__init__(debug=debug, logger=logger, sharing=AbstractPeriodicPlanner.NONE, printouts=printouts)
        
        # validate inputs
        assert isinstance(dealer_name, str), "dealer_name must of type `str`"

        self.dealer_name = dealer_name
        self.plan_message : PlanMessage = None

    def update_percepts(self, state, current_plan, tasks, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions):       
        super().update_percepts(state, current_plan, tasks, incoming_reqs, relay_messages, misc_messages, completed_actions, aborted_actions, pending_actions)

        # check if there are any plan messages for this agent
        plan_messages = {msg for msg in misc_messages 
                         if isinstance(msg, PlanMessage) # filter by message type
                         and msg.agent_name == state.agent_name # filter by agent name                         
                        #  and msg.src == self.dealer_name # TODO filter by dealer name
                         }
        
        # update the latest plan message
        for plan_message in plan_messages:
            if self.plan_message is None or plan_message.t_plan > self.plan_message.t_plan:
                self.plan_message = plan_message        
        
    def needs_planning(self, *_) -> bool:
        # only replans if there is an unprocessed plan message
        return self.plan_message is not None

    def generate_plan(self, state : SimulationAgentState, *_) -> Plan:
        # get actions from latest plan message
        actions : list[AgentAction] = [action_from_dict(**action) 
                                       for action in self.plan_message.plan]
        
        # only keep actions that start after the current time
        actions = [action for action in actions if action.t_start >= state._t]

        # DEBUG SECTION -----
        observations = sorted([action for action in actions if isinstance(action, ObservationAction)], key=lambda a: a.t_start)
        if observations:
            dth = abs(observations[0].look_angle - state.attitude[0])
            dt = observations[0].t_start - state._t
            if dt <= dth + 1e-3:
                # f"Observation angle {dth} must be less than or equal to time difference {dt}"
                x = 1
        #--------------------

        # create a plan from plan message actions
        self._plan = PeriodicPlan(actions, t=self.plan_message.t_plan)

        # remove the plan message after processing
        del self.plan_message
        self.plan_message = None

        # return the generated plan
        return self._plan.copy()

    def _schedule_observations(self, *_):
        """ Boilerplate method for scheduling observations."""
        # does not schedule observations for worker agent
        return []
    
    def _schedule_broadcasts(self, *_):
        """ Boilerplate method for scheduling broadcasts."""
        # does not schedule broadcasts for worker agent
        return []
    