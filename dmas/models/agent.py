from collections import defaultdict
import copy
import logging
import os
from typing import Any, Dict, List, Tuple
import uuid
from queue import Queue
import numpy as np
import pandas as pd

from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from execsatm.mission import Mission
from execsatm.tasks import GenericObservationTask, DefaultMissionTask, EventObservationTask
from execsatm.objectives import DefaultMissionObjective
from execsatm.requirements import SpatialCoverageRequirement, SinglePointSpatialRequirement, MultiPointSpatialRequirement, GridSpatialRequirement

from dmas.core.messages import AgentActionMessage, AgentStateMessage, BusMessage, MeasurementBidMessage, MeasurementRequestMessage, ObservationResultsMessage, SimulationMessage, SimulationMessageTypes, message_from_dict
from dmas.models.planning.decentralized.consensus.consensus import ConsensusPlanner
from dmas.utils.orbitdata import OrbitData
from dmas.models.actions import ActionStatuses, AgentAction, BroadcastMessageAction, FutureBroadcastMessageAction, ManeuverAction, ObservationAction, WaitAction, action_from_dict
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.trackers import DataSink, LatestObservationTracker
from dmas.models.science.processing import ObservationDataProcessor
from dmas.models.states import GroundOperatorAgentState, SatelliteAgentState, SimulationAgentState
from dmas.models.science.requests import TaskRequest
from dmas.models.planning.plan import PeriodicPlan, Plan, ReactivePlan
from dmas.utils.tools import SimulationRoles


class SimulationAgent(object):
    def __init__(self, 
                 agent_name : str, 
                 agent_id : str,
                 specs : object,
                 initial_state : SimulationAgentState, 
                 mission : Mission,
                 simulation_results_path : str,
                 orbitdata : OrbitData,
                 processor : ObservationDataProcessor = None, 
                 preplanner : AbstractPeriodicPlanner = None,
                 replanner : AbstractReactivePlanner = None,
                 level : int = logging.INFO, 
                 logger : logging.Logger = None,
                 printouts : bool = True
                ):
        # validate inputs        
        assert isinstance(agent_name, str), "Agent name must be a string."
        assert isinstance(agent_id, (str, type(None))), "Agent ID must be a string or None."
        assert isinstance(specs, object), "Specs must be an object."
        assert isinstance(initial_state, SimulationAgentState), "Initial state must be a SimulationAgentState object."
        assert isinstance(mission, Mission), "Mission must be an execsatm Mission object."
        assert isinstance(simulation_results_path, str), "Simulation results path must be a string."
        agent_results_path = os.path.join(simulation_results_path, agent_name.lower())
        assert os.path.exists(agent_results_path), f"Agent results path {agent_results_path} does not exist."
        assert isinstance(orbitdata, OrbitData), "Orbit data must be an OrbitData object."
        assert processor is None or isinstance(processor, ObservationDataProcessor), "Processor must be a DataProcessor object or None."
        assert preplanner is None or isinstance(preplanner, AbstractPeriodicPlanner), "Preplanner must be an AbstractPeriodicPlanner object or None."
        assert replanner is None or isinstance(replanner, AbstractReactivePlanner), "Replanner must be an AbstractReactivePlanner object or None."
        assert isinstance(level, int), "Logging level must be an integer."
        assert logger is None or isinstance(logger, logging.Logger), "Logger must be a logging.Logger object or None."
        assert isinstance(printouts, bool), "Printouts toggle must be a boolean."

        # assign parameters
        self.name : str = agent_name
        self._id : str = agent_id if agent_id is not None else str(uuid.uuid4())
        self._specs : object = specs
        if isinstance(specs, Spacecraft):
            self._payload = {instrument.name: instrument for instrument in specs.instrument}
        elif isinstance(specs, dict):
            self._payload = {instrument['name']: instrument for instrument in specs['instrument']} if 'instrument' in specs else dict()
        else:
            raise ValueError(f'`specs` must be of type `Spacecraft` or `dict`. Is of type `{type(specs)}`.')

        self._simulation_results_path : str = simulation_results_path
        self._results_path : str = agent_results_path
        self._orbitdata : OrbitData = orbitdata
        self._state : SimulationAgentState = initial_state
        self._mission : Mission = mission
        self._printouts : bool = printouts
        
        self._processor : ObservationDataProcessor = processor
        self._preplanner : AbstractPeriodicPlanner = preplanner
        self._replanner : AbstractReactivePlanner = replanner

        # initialize logger
        self._logger : logging.Logger = logger if logger is not None \
                                            else logging.getLogger(f"Agent-{self.name}")
        self._logger.setLevel(level)

        # initailize other variables
        self._message_inbox : Queue = Queue([])
        self._message_outbox : Queue = Queue([])
        self._plan : Plan = PeriodicPlan(t=-1.0)
        self._plan_history = []
        self._known_tasks : Dict[Tuple, GenericObservationTask] \
            = SimulationAgent.__initialize_default_mission_tasks(mission, orbitdata)
        self._known_reqs : Dict[Tuple, TaskRequest] = dict() # TODO do we need this or is the task list enough?
        
        # initialize trackers and data sinks
        self._observations_tracker = LatestObservationTracker.from_orbitdata(orbitdata, agent_name, quiet=not printouts)
        self._observation_history = DataSink(out_dir=agent_results_path, owner_name=agent_name, data_name="observation_history")
        self._state_history = DataSink(out_dir=agent_results_path, owner_name=agent_name, data_name="state_history")

        # save initial state to history
        self._state_history.append(initial_state.to_dict())

    @staticmethod
    def __initialize_default_mission_tasks(mission : Mission, orbitdata : OrbitData) -> Dict[Tuple, GenericObservationTask]:
        """ 
        Creates default observation tasks for each default mission objective
         based on the spatial requirements of each objective.
        """
        # initialize task list
        tasks = dict()

        # gather targets for each default mission objective
        objective_targets = { objective : [] for objective in mission 
                             # ignore non-default objectives
                             if isinstance(objective, DefaultMissionObjective)
                             }
        
        # iterate through each mission objective
        for objective,targets in objective_targets.items():  
            # collect spatial coverage requirements
            spatial_requirements = [req for req in objective.requirements
                                    if isinstance(req, SpatialCoverageRequirement)]

            # iterate through each spatial requirement
            for req in spatial_requirements:
                if isinstance(req, SinglePointSpatialRequirement):
                    # collect specified target
                    req_targets = [req.target]
                
                elif isinstance(req, MultiPointSpatialRequirement):
                    # collect all specified targets
                    req_targets = [target for target in req.targets]
                
                elif isinstance(req, GridSpatialRequirement):
                    # collect all targets matching this grid requirement
                    req_targets = [
                        (lat, lon, grid_index, gp_index)
                        for lat,lon,grid_index,gp_index in orbitdata.grid_data
                        if grid_index == req.grid_index and gp_index < req.grid_size
                    ]
                else: 
                    raise TypeError(f"Unknown spatial requirement type: {type(req)}")
                    
                # add to list of targets for this objective
                targets.extend(req_targets)

            # check if any spatial coverage requirements were found
            if not spatial_requirements:
                # no spatial coverage requirements found; 
                #   collect all targets from all grids known to this agent
                req_targets = list({
                    (lat, lon, grid_index, gp_index)
                    for lat,lon,grid_index,gp_index in orbitdata.grid_data
                })
                targets.extend(req_targets)
        
        # iterate through each mission objective
        for objective,targets in objective_targets.items():                           
            # create monitoring tasks from each location in this mission objective
            objective_tasks = [DefaultMissionTask(objective.parameter,
                                        location=(lat, lon, grid_index, gp_index),
                                        mission_duration=orbitdata.duration*24*3600,
                                        objective=objective,
                                        )
                        for lat,lon,grid_index,gp_index in targets
                    ]
            
            # add to list of known tasks
            tasks.update({SimulationAgent._task_key(task.to_dict()) : task
                          for task in objective_tasks})

        # return list of created tasks
        return tasks

    """
    ----------------------
    SIMULATION CYCLE METHODS
    ----------------------
    """

    """
    THINK METHOD
    """
    def decide_action(self, 
                      curr_state : SimulationAgentState,
                      prev_action : AgentAction,
                      prev_action_status : str,
                      incoming_messages : List[SimulationMessage],
                      my_measurements : list
                    ) -> Tuple[SimulationAgentState, AgentAction]:
        """ 
        Main thinking method for the agent; processes incoming messages and
            generates next actions to perform.

        #### Parameters
        - `curr_state` : SimulationAgentState
            Current state of the agent.
        - `prev_action` : AgentAction
            Previous action performed by the agent.
        - `prev_action_status` : str
            Status of the previous action performed by the agent.
        - `incoming_messages` : List[SimulationMessage]
            List of incoming messages received by the agent.
        - `my_measurements` : list
            List of measurements performed by the agent.

        """
        # ensure time has advanced
        assert curr_state.get_time() > self._state.get_time() or abs(curr_state.get_time() - self._state.get_time()) < 1e-6, \
            "State time must be greater than or equal to the previous state time."

        # append state to history if time has advanced
        if (abs(curr_state.get_time() - self._state.get_time()) > 1e-6
            or curr_state.status != self._state.status):
            self._state_history.append(curr_state.to_dict())

        # update state
        self._state = curr_state

        # unpack and classify incoming messages
        incoming_reqs, external_measurements, \
            external_states, external_action_statuses, misc_messages \
                = self.__classify_incoming_messages(curr_state, incoming_messages)

        # process action completion
        completed_actions, aborted_actions, pending_actions \
            = self.__process_action_completion([(prev_action, prev_action_status)])
                                                        
        # --- FOR DEBUGGING PURPOSES ONLY: ---
        # x = 1 # breakpoint
        # -------------------------------------

        # update known tasks and requests from incoming tasks requests
        new_reqs, new_tasks = self.__update_requests_and_tasks(incoming_reqs)

        # update plan completion
        self.__update_plan_completion(completed_actions, 
                                    aborted_actions, 
                                    pending_actions, 
                                    curr_state._t)

        # process performed observations
        generated_reqs : List[TaskRequest] = self.__process_measurements(new_reqs, my_measurements)
        new_reqs.extend(generated_reqs)
                        
        # update observation history with my measurements
        self.__update_observation_history(my_measurements)

        # update observations tracker with my measurements and external measurements
        self.__update_observations_tracker(my_measurements, external_measurements)

        # TODO update mission objectives from requests

        # --- Create plan ---
        if self._preplanner is not None:
            # there is a preplanner assigned to this planner
            
            # update preplanner precepts
            self._preplanner.update_percepts(curr_state,
                                            self._plan, 
                                            self._known_tasks.values(),
                                            new_reqs,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            # check if there is a need to construct a new plan
            if self._preplanner.needs_planning(curr_state, 
                                              self._specs, 
                                              self._plan):  
                
                # update tasks for only tasks that are available
                self.__update_tasks(curr_state)
                
                # initialize plan      
                self._plan : Plan = self._preplanner.generate_plan(curr_state, 
                                                            self._specs,
                                                            self._orbitdata,
                                                            self._mission,
                                                            self._known_tasks.values(),
                                                            self._observations_tracker
                                                            )

                # save copy of plan for post-processing
                plan_copy = [action for action in self._plan]
                self._plan_history.append((curr_state._t, plan_copy))
                
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # if self._preplanner._debug: 
                # if state.get_time() < 1:
                if True:
                    # self.__log_plan(self._plan, "PRE-PLAN", logging.WARNING)
                    x = 1 # breakpoint
                # -------------------------------------

        # --- Modify plan ---
        # Check if replanning is needed
        if self._replanner is not None:
            # there is a replanner assigned to this planner

            # update replanner precepts
            self._replanner.update_percepts( curr_state,
                                            self._plan, 
                                            self._known_tasks.values(),
                                            new_reqs,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            if self._replanner.needs_planning(curr_state, 
                                             self._specs,
                                             self._plan,
                                             self._orbitdata):    
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                # x = 1 # breakpoint
                # -------------------------------------

                # update tasks for only tasks that are available
                self.__update_tasks(curr_state)

                # Modify current Plan      
                self._plan : ReactivePlan = self._replanner.generate_plan(curr_state, 
                                                                self._specs,
                                                                self._plan,
                                                                self._orbitdata,
                                                                self._mission,
                                                                self._known_tasks.values(),
                                                                self._observations_tracker
                                                                )

                # update last time plan was updated
                t_plan = curr_state.get_time()

                # save copy of plan for post-processing
                plan_copy = [action for action in self._plan]
                self._plan_history.append((t_plan, plan_copy))
            
                # clear pending actions
                pending_actions = []

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                if True:
                    # self.__log_plan(self._plan, "REPLAN", logging.WARNING)
                    x = 1 # breakpoint
                # -------------------------------------

        # get next actions to perform from current plan
        next_action : AgentAction = self.get_next_planned_action(curr_state)
        
        # ensure an action was returned
        assert next_action is not None, \
            "No next action was returned from `get_next_planned_action()`."

        # change state to indicate the start of the new status 
        # (e.g., maneuvering, observing, waiting, etc.) 
        next_state: SimulationAgentState \
            = self.__prepare_next_state(curr_state, next_action, curr_state._t)
        
        # # save copy to state history
        # self._state_history.append(next_state.to_dict())

        # reset message and observations inbox for next cycle
        incoming_messages.clear(); my_measurements.clear()
        
        incoming_reqs.clear()
        external_measurements.clear()
        external_states.clear()
        external_action_statuses.clear()
        misc_messages.clear()        
        
        # del curr_state    # TODO check if needed
        # del action        # TODO check if needed
        
        # --- FOR DEBUGGING PURPOSES ONLY: ---        
        if True:
            # self.__log_plan(self._plan, "CURRENT PLAN", logging.WARNING)
            # self.__log_plan([next_action], "NEXT ACTION", logging.WARNING)
            x = 1 # breakpoint
        # -------------------------------------        
        
        # return next initial state and next actions to perform
        return next_state, next_action
    
    def __classify_incoming_messages(self, 
                                     state : SimulationAgentState,
                                     incoming_messages : List[SimulationMessage]
                                    ) -> Tuple[List[MeasurementRequestMessage], List[Tuple[str, list]], List[AgentStateMessage], List[AgentActionMessage], List[SimulationMessage]]:
        """ Classify incoming messages into their respective types """

        # check if there exist any bus messages in incoming messages
        bus_messages : List[BusMessage] = [msg for msg in incoming_messages 
                                           if isinstance(msg, BusMessage)]

        # unpack bus messages
        for bus_msg in bus_messages: 
            # add bus' contents to list of incoming messages
            # incoming_messages.extend([message_from_dict(**msg) 
            #                           if isinstance(msg, dict) 
            #                           else msg    
            #                           for msg in bus_msg.msgs])
            incoming_messages.extend(bus_msg.msgs)
            # remove original bus messages 
            incoming_messages.remove(bus_msg)

        # define classified message lists
        incoming_reqs, observation_msgs, \
            external_measurements, external_states, \
                external_action_statuses, misc_messages \
                    = [], [], [], [], [], []

        # classify incoming messages
        for msg in incoming_messages:
            if isinstance(msg, MeasurementRequestMessage):
                incoming_reqs.append(msg)
            elif isinstance(msg, ObservationResultsMessage):
                observation_msgs.append(msg)
                if isinstance(msg.instrument, str):
                    external_measurements.append((msg.instrument, msg.observation_data))
            elif isinstance(msg, AgentStateMessage) and msg.src != state.agent_name:
                external_states.append(msg)
            elif isinstance(msg, AgentActionMessage):
                external_action_statuses.append(msg)
            else:
                misc_messages.append(msg)

        # return classified messages
        return incoming_reqs, external_measurements, \
            external_states, external_action_statuses, list(misc_messages)
    
    def __process_action_completion(self, action_status_pairs : List[Tuple[AgentAction, str]]) -> tuple:
        
        # define classified action lists
        completed_actions = [] # planned action completed
        pending_actions = [] # planned action wasn't completed
        aborted_actions = []# planned action aborted
            
        # classify by action completion
        for action, status in action_status_pairs:
            # skip if no action was performed
            if action is None: continue

            if status == ActionStatuses.COMPLETED.value:
                completed_actions.append(action)
            elif status == ActionStatuses.ABORTED.value:
                aborted_actions.append(action)
            elif status == ActionStatuses.PENDING.value:
                pending_actions.append(action)
            else:
                raise ValueError(f"Unknown action status: {status}")

        # return classified lists
        return completed_actions, aborted_actions, pending_actions

    def __update_plan_completion(self, 
                                completed_actions : list, 
                                aborted_actions : list, 
                                pending_actions : list, 
                                t : float) -> None:
        """
        Updates the plan completion based on the actions performed.
        """
        # update plan completion
        self._plan.update_action_completion(completed_actions, 
                                           aborted_actions, 
                                           pending_actions, 
                                           t)    

    def __process_measurements(self, 
                               incoming_reqs : List[TaskRequest], 
                               measurements : List[Tuple[str, list]]
                               ) -> List[TaskRequest]:
        """
        Processes measurement observations and generates new requests based on the observations.
        """
        # check if there is a data processor assigned
        if self._processor is None: return []

        # process observations and return generated requests
        new_reqs : List[TaskRequest] \
            = self._processor.process_measurements(incoming_reqs, measurements)

        # add to known requests
        self._known_reqs.update({self._req_key(req.to_dict()): req for req in new_reqs})

        # return generated requests
        return new_reqs

    def __update_observation_history(self, my_observations : List[Tuple[str, List[Dict[str, Any]]]]) -> None:
        """
        Updates the observation history with the completed observations.
        """
        for _,obs in my_observations:            
            self._observation_history.extend(obs)

    def __update_observations_tracker(self, 
                                      my_observations : List[Tuple[str, List[Dict[str, Any]]]], 
                                      external_observations : List[Tuple[str, List[Dict[str, Any]]]]) -> None:
        """ Updates the observation history with the completed observations. """
        # update observation history
        if my_observations:
            self._observations_tracker.update_many(my_observations)

        if external_observations:
            raise NotImplementedError("Updating observation history with external observations is not yet implemented.")
            self._observations_tracker.update_many(external_observations)

    def __update_requests_and_tasks(self,
                                    incoming_reqs : List[MeasurementRequestMessage] = []
                                ) -> Tuple[List[TaskRequest], List[GenericObservationTask]]:
        
        # find unique and new requests in incoming requests
        unique_new_reqs = {self._req_key(msg.req): msg.req
                          for msg in incoming_reqs
                          if self._req_key(msg.req) not in self._known_reqs}

        # unpack unique new task requests
        new_reqs = {key : TaskRequest.from_dict(req_dict) 
                    for key,req_dict in unique_new_reqs.items()}
        
        # add new requests to known requests
        self._known_reqs.update(new_reqs)

        # find unique and new tasks in new requests
        new_tasks = {self._task_key(req.task.to_dict()): req.task
                                for req in new_reqs.values()
                                if self._task_key(req.task.to_dict()) not in self._known_tasks}
        
        # find unique and new tasks in incoming requests
        new_task_dicts = {self._task_key(msg.req['task']): msg.req['task']
                                for msg in incoming_reqs
                                if self._task_key(msg.req['task']) not in self._known_tasks
                                and self._task_key(msg.req['task']) not in new_tasks}

        # unpack unique bid tasks
        new_tasks.update({key : GenericObservationTask.from_dict(d) 
                            for key,d in new_task_dicts.items()})

        # add tasks to task list
        self._known_tasks.update(new_tasks)

        # return new_reqs.values(), new_tasks.values()
        return list(new_reqs.values()), list(new_tasks.values())

    def __update_tasks(self, state : SimulationAgentState) -> None:
        """ Updates the list of tasks to only include active tasks. """
        # filter tasks to only include active tasks
        self._known_tasks = {key : task for key,task in self._known_tasks.items() 
                                if task.is_available(state.get_time())}
        
    def __update_requests(self, state : SimulationAgentState) -> None:
        """ Updates the known requests to only include active requests. """
        # filter for request availability
        self._known_reqs = {key : req for key,req in self._known_reqs.items() 
                           if req.task.is_available(state.get_time())}

    def get_next_planned_action(self, state : SatelliteAgentState) -> AgentAction:
        # get current time
        t_curr = state.get_time()

        try:
            # get next set of actions from plan
            actions = self._next_actions_from_plan(t_curr)

            # attach observation requests to any observation actions
            self._attach_observation_requests(actions, state, t_curr)

            # materialize any future-broadcast actions into actual broadcast actions
            actions = self._materialize_future_broadcasts(actions, state, t_curr)

            # merge broadcast actions if needed
            actions = self._merge_broadcast_actions_if_needed(actions, t_curr)

            # validate actions
            self._validate_actions(actions, t_curr)

            # return earliest action
            return min(actions, key=lambda a: a.t_start, default=None)

        except Exception as e:
            self.log(f"Error in `get_next_action()`: {e}", logging.ERROR)
            raise
            
    def _next_actions_from_plan(self, t_curr: float) -> List[AgentAction]:
        # get latest set of actions from plan
        actions = self._plan.get_next_actions(t_curr, False)  # keep your flag
        
        # ensure there are actions to perform
        if not actions: raise RuntimeError("Plan returned no actions.")
        
        # return actions
        return actions

    def _attach_observation_requests(self, 
                                     actions : List[AgentAction], 
                                     state : SatelliteAgentState, 
                                     t_curr : float) -> None:
        # attach observation requests to observation actions
        for action in actions:
            # skip non-observation actions
            if not isinstance(action, ObservationAction): continue

            # get observation duration
            dt = action.t_end - t_curr
            assert dt >= 0, "Observation action must have a non-negative duration."

            # get appropriate instrument from payload
            instrument: Instrument = self._payload[action.instrument_name]

            # create observation request
            req = ObservationResultsMessage(
                self.name,
                SimulationRoles.ENVIRONMENT.value,
                state.to_dict(),
                action.to_dict(),
                instrument.to_dict(),
                action.t_start,
                action.t_end,
            )

            # attach request to action
            action.req = req.to_dict()

    def _materialize_future_broadcasts(self, 
                                       actions: List[AgentAction], 
                                       state: SatelliteAgentState, 
                                       t_curr: float) -> List[AgentAction]:
        # check if any future-broadcast actions are present
        future_broadcasts = [a for a in actions 
                             if isinstance(a, FutureBroadcastMessageAction)]
        
        # return if no future-broadcast actions present
        if not future_broadcasts: return actions

        # update known requests before compiling broadcast messages
        self.__update_requests(state)

        # compile broadcast messages
        msgs : List[SimulationMessage] \
            = self._compile_future_broadcast_messages(future_broadcasts, state, t_curr)

        # remove future broadcast actions from the plan at time t
        for future_broadcast in future_broadcasts: 
            self._plan.remove(future_broadcast, t_curr)

        # find indices of future broadcast actions in current action list
        broadcast_indices = [i for i, a in enumerate(actions) if a in future_broadcasts]

        # remove future broadcast actions from the current action list in reverse order
        for i in sorted(broadcast_indices, reverse=True): actions.pop(i)

        # if nothing to send, just return remaining actions
        if not msgs:
            # get next actions from updated plan
            actions : List[AgentAction] = self._plan.get_next_actions(t_curr, False)

            # return next actions
            return actions

        # create bus message and broadcast action
        bus = BusMessage(state.agent_name, state.agent_name, [m for m in msgs])
        broadcast = BroadcastMessageAction(bus, future_broadcasts[0].t_start)

        # add bus broadcast to plan and return list
        self._plan.add(broadcast, t_curr)

        # insert broadcast action into actions at earliest index of removed future broadcasts
        insert_at = min(broadcast_indices) if broadcast_indices else 0
        actions.insert(insert_at, broadcast)

        return actions

    def _compile_future_broadcast_messages(self,
                                            future_broadcasts: List[FutureBroadcastMessageAction], 
                                            state: SatelliteAgentState, 
                                            t: float
                                        ) -> List[SimulationMessage]:
        # initiate message list
        msgs = []

        # compile messages for each future broadcast action
        for fb in future_broadcasts:
            # determine broadcast type
            bt = fb.broadcast_type

            # add messages based on broadcast type
            if bt == FutureBroadcastMessageAction.STATE:
                msgs.append(AgentStateMessage(state.agent_name, state.agent_name, state.to_dict()))

            elif bt == FutureBroadcastMessageAction.OBSERVATIONS:
                msgs.extend(self._compile_observation_broadcasts(state, t))

            elif bt == FutureBroadcastMessageAction.REQUESTS:
                msgs.extend(self._compile_request_broadcasts(fb, state, t))

            elif bt == FutureBroadcastMessageAction.BIDS:
                msgs.extend(self._compile_bid_broadcasts(state, t))

            else:
                raise NotImplementedError(f"Future broadcast type {bt} not supported.")

        # return compiled messages
        return msgs
    
    def _compile_observation_broadcasts(self, state : SimulationAgentState, t: float):
        # TODO package current state for sharing with others
        latest = self.get_latest_observations(state)

        by_inst = defaultdict(list)
        for obs in latest:
            by_inst[obs["instrument"].lower()].append(obs)

        msgs = []
        for inst, observations in by_inst.items():
            msgs.append(
                ObservationResultsMessage(
                    state.agent_name,
                    state.agent_name,
                    state.to_dict(),
                    {},                     
                    {"name": inst},
                    t,
                    t,
                    observations
                )
            )
        return msgs

    def _compile_request_broadcasts(self, 
                                    future_broadcast : FutureBroadcastMessageAction, 
                                    state : SimulationAgentState, 
                                    t_curr: float):
        # initiate message list
        msgs = []

        # iterate through known requests
        for req in self._known_reqs.values():
            # check if request is active at current time
            if t_curr not in req.task.availability: continue

            # check if request should be included
            if future_broadcast.only_own_info:
                # if only own info, include only own requests
                if req.requester != state.agent_name: continue
            else:
                # include all agents' requests, filtered by desc if provided
                if future_broadcast.desc is not None and req.task not in future_broadcast.desc:
                    continue

            # add measurement request message
            msgs.append(MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict()))
        
        # return compiled messages
        return msgs
    
    def _compile_bid_broadcasts(self, state : SimulationAgentState, t: float) -> List[SimulationMessage]:
        # TODO check if there is a replanner assigned to this agent
        if self._replanner is None:
            raise NotImplementedError("Bid broadcasting is not yet implemented for cases without a replanner.")
        if not isinstance(self._replanner, ConsensusPlanner):
            raise NotImplementedError("Bid broadcasting is only implemented for agents with a replanner of type `ConsensusPlanner`.")
        
        # generate bid messages to share bids in results
        compiled_bid_msgs = [
            MeasurementBidMessage(state.agent_name, state.agent_name, bid.to_dict())
            # MeasurementBidMessage(state.agent_name, state.agent_name, bid)
            for task,bids in self._replanner.results.items()
            if isinstance(task, EventObservationTask)  # only share bids for event-driven tasks
            for bid in bids
        ]

        # return messages list
        return compiled_bid_msgs

    def _merge_broadcast_actions_if_needed(self, 
                                           actions : List[AgentAction], 
                                           t_curr: float):
        # check if there are multiple broadcast actions to merge
        if len(actions) <= 1:
            # not enough actions to merge; return as is
            return actions
        if all([not isinstance(a, BroadcastMessageAction) for a in actions]):
            # no broadcast actions to merge; return as is
            return actions

        # compile all messages into a single bus message
        msgs = []
        merged : List[Tuple[int, AgentAction]] = []
        for i,a in enumerate(actions):
            # skip non-broadcast actions
            if not isinstance(a, BroadcastMessageAction): continue

            # get messages from action
            msg = a.msg

            # unpack bus messages if needed 
            if isinstance(msg,dict) and msg.get("msg_type") == SimulationMessageTypes.BUS.value:
                msgs.extend(msg.get("msgs", []))
            elif isinstance(msg,BusMessage):
                msgs.extend(msg.msgs)
            else:
                msgs.append(msg)

            # else add message as is

            # mark action as merged
            merged.append((i,a))

        # get earliest start and latest end times
        t_start = min(a.t_start for _,a in merged)
        t_end = max(a.t_end for _,a in merged)

        # create merged bus message and broadcast action
        bus = BusMessage(src=self.name, dst=self.name, msgs=msgs)
        merged_action = BroadcastMessageAction(t_start=t_start, t_end=t_end, msg=bus.to_dict())

        # update underlying plan to match returned actions
        for _,a in merged:
            self._plan.remove(a, t_curr)
        self._plan.add(merged_action, t_curr)

        # remove merged actions from current action list
        for _,a in merged:
            actions.remove(a)
        # append merged action to current action list
        i_insert = min(i for i,_ in merged)
        actions.insert(i_insert, merged_action)

        # return merged action in list
        return actions

    def _validate_actions(self, actions : List[AgentAction], t: float):
        assert actions, "No actions were returned from `get_next_actions()`."

        assert all(a.t_start <= t + 1e-3 for a in actions), \
            "All returned actions must start at or before the current time."
        assert all(not isinstance(a, FutureBroadcastMessageAction) for a in actions), \
            "No future broadcast message actions should be present in the output plan."

        d0 = actions[0].t_end - actions[0].t_start
        if np.isinf(d0):
            assert len(actions) == 1, \
                "If an action has infinite duration, it must be the only action returned."
        else:
            assert all(abs((a.t_end - a.t_start) - d0) < 1e-6 for a in actions), \
                "All returned actions must have the same duration."
    
    def get_latest_observations(self, 
                                state : SimulationAgentState,
                                latest_plan_only : bool = True
                                ) -> List[dict]:
        raise NotImplementedError("Error, get_latest_observations() is not yet implemented.")
        return [observation_tracker.latest_observation
                 for _,grid in self._observations_tracker.trackers.items()
                for _, observation_tracker in grid.items()
                if isinstance(observation_tracker, ObservationTracker)
                # check if there is a latest observation
                and observation_tracker.latest_observation is not None
                # only include observations performed by myself 
                and observation_tracker.latest_observation['agent name'] == state.agent_name
                # only include observations performed for the current plan
                and self._plan.t * int(latest_plan_only) <= observation_tracker.latest_observation['t_end'] <= state.get_time()
            ]

    def __prepare_next_state(self, 
                            curr_state : SimulationAgentState, 
                            next_action : AgentAction,
                            t : float
                        ) -> SimulationAgentState:
        """ Update the agent state based on the next actions to perform. """
        # create copy of current state
        next_state : SimulationAgentState = curr_state.copy()
        # next_state : SimulationAgentState = copy.copy(curr_state)
        
        assert abs(next_state.get_time() - t) < 1e-6, \
            "State time must match the provided time."

        # determine new status from next action
        if isinstance(next_action, ManeuverAction):
            next_state.perform_maneuver(next_action, t)
        elif isinstance(next_action, ObservationAction):
            # update state
            next_state.update(t, status=SimulationAgentState.MEASURING)
        elif isinstance(next_action, BroadcastMessageAction):
            # update state
            next_state.update(t, status=SimulationAgentState.MESSAGING)
        elif isinstance(next_action, WaitAction):
            # update state
            next_state.update(t, status=SimulationAgentState.WAITING)

        # return updated state
        return next_state

    """
    ----------------------
    UTILITY METHODS
    ----------------------
    """    

    def get_state(self) -> SimulationAgentState:
        return self._state
    
    @staticmethod
    def _task_key(d : dict) -> tuple:
        return (
            d["task_type"],
            d["parameter"],
            d["priority"],
            d["id"],
        )
    
    @staticmethod
    def _req_key(d : dict) -> tuple:
        return (
            d["task"]["task_type"],
            d["task"]["parameter"],
            d["task"]["priority"],
            d["task"]["id"],
            d["requester"],
        )
    
    def log(self, msg : str, level=logging.DEBUG) -> None:
        """
        Logs a message to the desired level.
        """
        try:
            # check if printouts are enabled
            if not self._printouts: return

            # get current simulation time
            t = self._state.get_time()
            t = t if t is None else round(t,3)

            # log to the appropriate level with agent name and current time
            if level is logging.DEBUG:
                self._logger.debug(f'T={t}[s] | {self.name}: {msg}')
            elif level is logging.INFO:
                self._logger.info(f'T={t}[s] | {self.name}: {msg}')
            elif level is logging.WARNING:
                self._logger.warning(f'T={t}[s] | {self.name}: {msg}')
            elif level is logging.ERROR:
                self._logger.error(f'T={t}[s] | {self.name}: {msg}')
            elif level is logging.CRITICAL:
                self._logger.critical(f'T={t}[s] | {self.name}: {msg}')
        
        except Exception as e:
            raise e
    
    def __log_plan(self, plan : Plan, title : str, level : int = logging.DEBUG) -> None:
        try:
            out = f'\n{title}\n'
            if isinstance(plan, Plan):
                out += str(plan)
            
            else:                
                for action in plan:
                    if isinstance(action, AgentAction):
                        out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"

                    elif isinstance(action, dict):
                        out += f"{action['id'].split('-')[0]}, {action['action_type']}, {action['t_start']}, {action['t_end']}\n"           

            self.log(out, level)
        except Exception as e:
            raise e
        
    def __repr__(self):
        if isinstance(self._state, SatelliteAgentState):
            return f"SatelliteAgent(name={self.name}, id={self._id})"
        elif isinstance(self._state, GroundOperatorAgentState):
            return f"GroundOperatorAgent(name={self.name}, id={self._id})"
        else:
            return f"SimulationAgent(name={self.name}, id={self._id})"

    """
    ----------------------
    RESULTS HANDLING METHODS
    ----------------------
    """
    def print_results(self):
        try:
            # log known default tasks
            columns = ['id', 'task type', 'requester', 'parameter', 'lat [deg]', 'lon [deg]', 'grid index', 'gp index', 't start', 't end', 'priority']
            data = [(task.id,task.task_type, self.name, task.parameter, task.location[0][0], task.location[0][1], task.location[0][2], task.location[0][3],
                    task.availability.left, task.availability.right, task.priority)
                for task in self._known_tasks.values()
                if isinstance(task, DefaultMissionTask)
            ]
            df = pd.DataFrame(data=data, columns=columns)        
            df.to_parquet(f"{self._results_path}/known_tasks.parquet", index=False)

            # log known and generated requests
            columns = ['id','requester','lat [deg]','lon [deg]','grid index', 'GP index','severity','start time [s]','end time [s]','detection time [s]','event type']
            if self._processor is not None:
                data_known = [(event.id, self._processor.event_requesters[event], event.location[0], event.location[1], event.location[2], event.location[3], event.severity, event.t_start, event.t_start+event.d_exp, event.t_detect, event.event_type)
                        for event in self._processor.known_events]
                data_detected = [(event.id, self._processor.event_requesters[event], event.location[0], event.location[1], event.location[2], event.location[3], event.severity, event.t_start, event.t_start+event.d_exp, event.t_detect, event.event_type)
                        for event in self._processor.detected_events]
            else:
                data_known, data_detected = [], []
                
            df = pd.DataFrame(data=data_known, columns=columns)        
            df.to_parquet(f"{self._results_path}/events_known.parquet", index=False)   

            df = pd.DataFrame(data=data_detected, columns=columns)        
            df.to_parquet(f"{self._results_path}/events_detected.parquet", index=False)
        
            # log plan history
            headers = ['plan_index', 't_plan', 'desc', 't_start', 't_end']
            data = []
            
            for i in range(len(self._plan_history)):
                t_plan, plan = self._plan_history[i]
                t_plan : float; plan : list[AgentAction]

                for action in plan:
                    desc = f'{action.action_type}'
                    if isinstance(action, ObservationAction):
                        desc += f'_{action.instrument_name}'
                        
                    line_data = [   i,
                                    np.round(t_plan,3),
                                    desc,
                                    np.round(action.t_start,3 ),
                                    np.round(action.t_end,3 )
                                ]
                    data.append(line_data)

            df = pd.DataFrame(data, columns=headers)
            df.to_parquet(f"{self._results_path}/planner_history.parquet", index=False)
            
            # log observation history
            self._observation_history.close()

            # log state history
            self._state_history.close()

        except Exception as e:
            raise e        