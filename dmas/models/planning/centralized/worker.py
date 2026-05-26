from typing import Dict, List, Tuple

from execsatm.tasks import GenericObservationTask, Interval
from tqdm import tqdm

from dmas.models.actions import AgentAction
from dmas.models.actions import ManeuverAction, ObservationAction, action_from_dict
from dmas.models.planning.plan import PeriodicPlan, Plan
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.states import SimulationAgentState
from dmas.core.messages import PlanMessage, message_from_dict


class WorkerPlanner(AbstractPeriodicPlanner):
    """
    Worker planner class that handles replanning tasks for agents.
    It processes the replanning requests and updates the agent's plan accordingly.
    """
    def __init__(self, agent_results_dir : str, dealer_name : str, debug = False, logger = None, printouts : bool = True) -> None:
        super().__init__(agent_results_dir, debug=debug, logger=logger, sharing=AbstractPeriodicPlanner.NONE, printouts=printouts)
        
        # validate inputs
        assert isinstance(dealer_name, str), "dealer_name must of type `str`"

        self.dealer_name = dealer_name
        self.plan_message : PlanMessage = None
        self._new_plan_received : bool = False

    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        tasks : Dict[Tuple,GenericObservationTask],
                        incoming_reqs: Dict[Tuple,Dict], 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list
                    ) -> None:
        # update percepts using parent method
        super().update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)

        # check if there are any plan messages for this agent
        plan_messages = []
        for msg in misc_messages:
            if isinstance(msg,PlanMessage):
                if (msg.agent_name == state.agent_name
                    and msg.src == self.dealer_name):
                    plan_messages.append(msg)
            elif isinstance(msg, dict) and msg.get('msg_type', None) == 'PLAN':
                if (msg.get('agent_name', None) == state.agent_name
                    and msg.get('src', None) == self.dealer_name):
                    plan_msg = message_from_dict(**msg)
                    plan_messages.append(plan_msg)
        
        # update the latest plan message
        for plan_message in plan_messages:
            if self.plan_message is None or plan_message.t_plan >= self.plan_message.t_plan:
                self.plan_message = plan_message        
                self._new_plan_received = True
        
    def needs_planning(self, *_) -> bool:
        # only replans if there is an unprocessed plan message
        # return self.plan_message is not None
        return self._new_plan_received

    def generate_plan(self, state : SimulationAgentState, specs : object, orbitdata : object, *_) -> Plan:
        # get current simulation time
        t_curr = state.get_time()
        
        # get actions from latest plan message
        planner_actions : list[AgentAction] = []
        for action in self.plan_message.plan:
            if isinstance(action, dict):
                planner_actions.append(action_from_dict(**action))
            elif isinstance(action, AgentAction):
                planner_actions.append(action)
            else:
                raise ValueError(f"Unrecognized action format in plan message: {action}")

        # only keep actions that start after the current time
        actions = [action for action in planner_actions if action.t_start >= t_curr]

        # separate observations from maneuvers and other actions
        # maneuvers are discarded — they will be reconstructed from the actual current state
        observations : List[ObservationAction] = sorted(
            [a for a in actions if isinstance(a, ObservationAction)],
            key=lambda a: a.t_start,
        )
        other_actions : List[AgentAction] = [
            a for a in actions if not isinstance(a, (ObservationAction, ManeuverAction))
        ]

        # validate observation sequence against actual current state and prune if infeasible
        valid_observations = self._validate_and_prune_observations(state, specs, observations)

        # reconstruct maneuvers from the actual current state for the pruned observation set
        try:
            maneuvers = self._schedule_maneuvers(state, specs, valid_observations, orbitdata)
        except (ValueError, NotImplementedError):
            maneuvers = []

        # if the satellite is mid-maneuver at plan receipt, insert a zero-duration
        # stop ManeuverAction so the execution engine resets attitude_rates to zero
        # before any gap between plan receipt and the first scheduled ManeuverAction.
        # without this, kinematic_model() propagates the old slew rate through the
        # wait gap and the satellite overshoots the new maneuver's target.
        if any(abs(r) > 1e-6 for r in state.attitude_rates):
            stop = ManeuverAction(
                list(state.attitude),
                list(state.attitude),
                [0.0, 0.0, 0.0],
                t_curr,
                t_curr+1e-5,  # add a tiny epsilon to ensure t_end > t_start for execution engine
            )
            maneuvers.insert(0, stop)

        # create a plan from validated, maneuver-corrected actions
        self._plan = PeriodicPlan(valid_observations, maneuvers, other_actions, t=self.plan_message.t_plan)

        # # remove the plan message after processing
        # del self.plan_message
        # self.plan_message = None
        self._new_plan_received = False

        # DEBUG SECTION ---------------
        # failing_agent_id = 'fl_vnir-fl-t_8'
        # if state.agent_id == failing_agent_id:
        #     t_curr = state.get_time()
        #     t_max = max(action.t_end for action in planner_actions) if planner_actions else t_curr
        #     planning_horizon = Interval(t_curr, t_max)
        #     if len(planner_actions) != len(actions):
        #         tqdm.write(f'[worker t={state._t:.2f}s] dropped {len(planner_actions) - len(actions)} actions from plan message received at t={self.plan_message.t_plan:.2f}s for agent {state.agent_name}')
        #     if 45500.0 in planning_horizon or t_curr >= 45_250.0:
        #         tqdm.write(f'[worker t={t_curr:.2f}s] current state: \n\tattitude={state.attitude}\n\tattitude_rates={state.attitude_rates}')
        #         tqdm.write(f'[worker t={t_curr:.2f}s] received plan: \n{self._plan}')
        #         if t_curr >= 45_500:
        #             x = 1 # DEBUG BREAKPOINT
        # ------------------------------

        # return the generated plan
        return self._plan.copy()
        

    def _validate_and_prune_observations(
        self,
        state : SimulationAgentState,
        specs : object,
        observations : List[ObservationAction],
    ) -> List[ObservationAction]:
        """
        Validates the observation sequence against the actual current satellite state.

        If the sequence is infeasible, the EARLIEST observation is dropped and the
        sequence is re-validated from the current state. This repeats until the remaining
        sequence is fully feasible or no observations remain.

        Dropping the earliest observation (rather than the failing one) is intentional:
        when dead-reckoning drift puts the satellite behind schedule, the first observation
        is usually what creates the cascade of infeasibility for the rest of the sequence.
        """
        valid_obs = list(observations)

        while valid_obs:
            try:
                if self.is_observation_path_valid(state, valid_obs, specs=specs):
                    break
            except NotImplementedError:
                break  # non-satellite agent; skip validation
            valid_obs.pop(0)

        return valid_obs

    def _schedule_observations(self, *_):
        """ Boilerplate method for scheduling observations."""
        # does not schedule observations for worker agent
        return []

    def _schedule_broadcasts(self, *_):
        """ Boilerplate method for scheduling broadcasts."""
        # does not schedule broadcasts for worker agent
        return []
    