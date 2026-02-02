from collections import defaultdict
import logging
import os
from typing import Dict, List, Tuple
import uuid
from queue import Queue
import numpy as np
import pandas as pd

from instrupy.base import Instrument
from orbitpy.util import Spacecraft

from execsatm.mission import Mission
from execsatm.tasks import GenericObservationTask, DefaultMissionTask
from execsatm.objectives import DefaultMissionObjective
from execsatm.requirements import SpatialCoverageRequirement, SinglePointSpatialRequirement, MultiPointSpatialRequirement, GridSpatialRequirement

from dmas.core.messages import AgentActionMessage, AgentStateMessage, BusMessage, MeasurementRequestMessage, ObservationResultsMessage, SimulationMessage, SimulationMessageTypes, message_from_dict
from dmas.core.orbitdata import OrbitData
from dmas.models.actions import ActionStatuses, AgentAction, BroadcastMessageAction, FutureBroadcastMessageAction, ManeuverAction, ObservationAction, WaitAction, action_from_dict
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.planning.tracker import ObservationHistory, ObservationTracker
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
                 logger : logging.Logger = None
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
        self._state_history : list = []
        self._known_tasks : Dict[Tuple, GenericObservationTask] \
            = SimulationAgent.__initialize_default_mission_tasks(mission, orbitdata)
        self._known_reqs : Dict[Tuple, TaskRequest] = dict() # TODO do we need this or is the task list enough?
        
        # initialize observation history
        self._observation_history = ObservationHistory.from_orbitdata(orbitdata)

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
                        for grid in orbitdata.grid_data
                        for lat,lon,grid_index,gp_index in grid.values
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
                    for grid in orbitdata.grid_data
                    for lat,lon,grid_index,gp_index in grid.values
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
    def think(self, 
              state : SimulationAgentState,
              action : AgentAction,
              action_status : str,
              incoming_messages : List[SimulationMessage],
              observations : list
            ) -> Tuple[SimulationAgentState, AgentAction]:
        """ 
        Main thinking method for the agent; processes incoming messages and
            generates next actions to perform.
        """
        # ensure time has advanced
        assert state.get_time() > self._state.get_time() or abs(state.get_time() - self._state.get_time()) < 1e-6, \
            "State time must be greater than or equal to the previous state time."

        # update state
        self._state = state

        # append state to history if time has advanced
        if abs(state.get_time() - self._state.get_time()) > 1e-6:
            self._state_history.append(state.to_dict())

        # unpack and classify incoming messages
        incoming_reqs, external_observations, \
            external_states, external_action_statuses, misc_messages \
                = self.__classify_incoming_messages(state, incoming_messages)

        # process action completion
        completed_actions, aborted_actions, pending_actions \
            = self.__process_action_completion([(action, action_status)])
                                                        
        # --- FOR DEBUGGING PURPOSES ONLY: ---
        # x = 1 # breakpoint
        # -------------------------------------

        # update known tasks and requests from incoming tasks requests
        new_reqs, new_tasks = self.__update_requests_and_tasks(incoming_reqs)

        # update plan completion
        self.__update_plan_completion(completed_actions, 
                                    aborted_actions, 
                                    pending_actions, 
                                    state._t)

        # process performed observations
        generated_reqs : List[TaskRequest] = self.__process_observations(new_reqs, observations)
        new_reqs.extend(generated_reqs)
                        
        # update observation history
        self.__update_observation_history(observations)

        # TODO update mission objectives from requests

        # --- Create plan ---
        if self._preplanner is not None:
            # there is a preplanner assigned to this planner
            
            # update preplanner precepts
            self._preplanner.update_percepts(state,
                                            self._plan, 
                                            self._known_tasks.values(),
                                            # incoming_reqs,
                                            new_reqs,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            # check if there is a need to construct a new plan
            if self._preplanner.needs_planning(state, 
                                              self._specs, 
                                              self._plan):  
                
                # update tasks for only tasks that are available
                self.__update_tasks(state)
                
                # initialize plan      
                self._plan : Plan = self._preplanner.generate_plan(state, 
                                                            self._specs,
                                                            self._orbitdata,
                                                            self._mission,
                                                            self._known_tasks.values(),
                                                            self._observation_history
                                                            )

                # save copy of plan for post-processing
                plan_copy = [action for action in self._plan]
                self._plan_history.append((state._t, plan_copy))
                
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # if self._preplanner._debug: 
                # if state.get_time() < 1:
                if True:
                    self.__log_plan(self._plan, "PRE-PLAN", logging.WARNING)
                    x = 1 # breakpoint
                # -------------------------------------

        # --- Modify plan ---
        # Check if replanning is needed
        if self._replanner is not None:
            # there is a replanner assigned to this planner

            # update replanner precepts
            self._replanner.update_percepts( state,
                                            self._plan, 
                                            self._known_tasks.values(),
                                            # incoming_reqs,
                                            new_reqs,
                                            misc_messages,
                                            completed_actions,
                                            aborted_actions,
                                            pending_actions
                                        )
            
            if self._replanner.needs_planning(state, 
                                             self._specs,
                                             self._plan,
                                             self._orbitdata):    
                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                # x = 1 # breakpoint
                # -------------------------------------

                # update tasks for only tasks that are available
                self.__update_tasks(state)

                # Modify current Plan      
                self._plan : ReactivePlan = self._replanner.generate_plan(state, 
                                                                self._specs,
                                                                self._plan,
                                                                self._orbitdata,
                                                                self._mission,
                                                                self._known_tasks.values(),
                                                                self._observation_history
                                                                )

                # update last time plan was updated
                t_plan = state.get_time()

                # save copy of plan for post-processing
                plan_copy = [action for action in self._plan]
                self._plan_history.append((t_plan, plan_copy))
            
                # clear pending actions
                pending_actions = []

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                if True:
                # if 95.0 < state.t < 96.0:
                # if state.get_time() > 19 and "1" in state.agent_name:
                    self.__log_plan(self._plan, "REPLAN", logging.WARNING)
                    x = 1 # breakpoint
                # -------------------------------------

        # get next actions to perform
        plan_out = self.get_next_actions(state, True)
        assert len(plan_out) > 0, \
            "No next actions were returned from `get_next_actions()`."
        
        # --- FOR DEBUGGING PURPOSES ONLY: ---        
        # if 95.0 < state.t < 96.0:
        # if state.t > 96.0:
        if True:
            # self.__log_plan(self._plan, "CURRENT PLAN", logging.WARNING)
            # self.__log_plan(plan_out, "NEXT ACTIONS", logging.WARNING)
            x = 1 # breakpoint
        # -------------------------------------        

        # change state to indicate new status (e.g., maneuvering, observing, waiting, etc.)
        # and save to state history
        action_state, action = self.__prepare_state(state, plan_out, state._t)
        
        # return next actions to perform
        return action_state, action
    
    def __prepare_state(self, 
                        state : SimulationAgentState, 
                        plan_out : List[AgentAction], 
                        t : float
                    ) -> Tuple[SimulationAgentState, AgentAction]:
        """ Update the agent state based on the next actions to perform. """
        # create copy of current state
        action_state : SimulationAgentState = state.copy()
        
        assert abs(action_state.get_time() - t) < 1e-6, \
            "State time must match the provided time."

        # get next action to perform
        action = plan_out[0]

        # determine new status from next action
        if isinstance(action, ManeuverAction):
            action_state.perform_maneuver(action, t)
        elif isinstance(action, ObservationAction):
            # update state
            action_state.update(t, status=SimulationAgentState.MEASURING)
        elif isinstance(action, BroadcastMessageAction):
            # update state
            action_state.update(t, status=SimulationAgentState.MESSAGING)
        elif isinstance(action, WaitAction):
            # update state
            action_state.update(t, status=SimulationAgentState.WAITING)

        # return updated state
        return action_state, action

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
            incoming_messages.extend([message_from_dict(**msg) 
                                      if isinstance(msg, dict) 
                                      else msg    
                                      for msg in bus_msg.msgs])
            # remove original bus messages 
            incoming_messages.remove(bus_msg)

        # define classified message lists
        incoming_reqs, observation_msgs, \
            external_observations, external_states, \
                external_action_statuses, misc_messages \
                    = [], [], [], [], [], []

        # classify incoming messages
        for msg in incoming_messages:
            if isinstance(msg, MeasurementRequestMessage):
                incoming_reqs.append(msg)
            elif isinstance(msg, ObservationResultsMessage):
                observation_msgs.append(msg)
                if isinstance(msg.instrument, str):
                    external_observations.append((msg.instrument, msg.observation_data))
            elif isinstance(msg, AgentStateMessage) and msg.src != state.agent_name:
                external_states.append(msg)
            elif isinstance(msg, AgentActionMessage):
                external_action_statuses.append(msg)
            else:
                misc_messages.append(msg)

        # # check for any measurement requests
        # incoming_reqs : list[MeasurementRequestMessage] \
        #     = [msg for msg in incoming_messages 
        #         if isinstance(msg, MeasurementRequestMessage)]
        
        # # check for any observation results messages
        # observation_msgs : List[ObservationResultsMessage] \
        #     = [msg for msg in incoming_messages 
        #         if isinstance(msg, ObservationResultsMessage)]
        
        # # extract observation data from messages
        # external_observations : List[tuple] \
        #     = [(msg.instrument, msg.observation_data) 
        #         for msg in incoming_messages 
        #         if isinstance(msg, ObservationResultsMessage)
        #         and isinstance(msg.instrument, str)]

        # external_states : List[AgentStateMessage] \
        #     = [SimulationAgentState.from_dict(msg.state) 
        #         for msg in incoming_messages 
                # if isinstance(msg, AgentStateMessage)
                # and msg.src != state.agent_name]
        
        # external_action_statuses : List[AgentActionMessage] \
        #     = [msg for msg in incoming_messages
        #         if isinstance(msg, AgentActionMessage)]
                
        # # gather any other miscellaneous messages
        # misc_messages = set(incoming_messages)
        # misc_messages.difference_update(incoming_reqs)
        # misc_messages.difference_update(observation_msgs)
        # misc_messages.difference_update(external_states)
        # misc_messages.difference_update(external_action_statuses)

        # return classified messages
        return incoming_reqs, external_observations, \
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

    def __process_observations(self, 
                               incoming_reqs : List[TaskRequest], 
                               observations : List[Tuple[str, list]]
                               ) -> List[TaskRequest]:
        """
        Processes observations and generates new requests based on the observations.
        """
        # check if there is a data processor assigned
        if self._processor is None: return []

        # process observations and return generated requests
        new_reqs = self._processor.process_observations(incoming_reqs, observations)

        # add to known requests
        self._known_reqs.update({self._req_key(req): req for req in new_reqs})

        # return generated requests
        return new_reqs

    def __update_observation_history(self, observations : list) -> None:
        """
        Updates the observation history with the completed observations.
        """
        # update observation history
        self._observation_history.update(observations)

    def __update_requests_and_tasks(self,
                                    incoming_reqs : List[MeasurementRequestMessage] = []
                                ) -> Tuple[List[TaskRequest], List[GenericObservationTask]]:
        # DEBUG------
        if incoming_reqs:
            x = 1 # breakpoint
        # ------------

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

    # def __update_tasks(self, 
    #                    state : SimulationAgentState,
    #                    incoming_reqs : List[MeasurementRequestMessage] = [], 
    #                    available_only : bool = False
    #                    ) -> None:
    #     """
    #     Updates the list of tasks based on incoming requests and task availability.
    #     """
    #     # get tasks from incoming requests
    #     # event_tasks = [req.task
    #     #                for req in incoming_reqs]
    #     if incoming_reqs:
    #         x = 1 # breakpoint
        # # extract tasks from incoming requests
        # ## find unique and new tasks in incoming requests
        # unique_new_req_tasks = {self._task_key(msg.req['task']): msg.req['task']
        #                         for msg in incoming_reqs
        #                         if self._task_key(msg.req['task']) not in self._known_tasks}

        # ## unpack unique bid tasks
        # event_tasks = {key : GenericObservationTask.from_dict(task_dict) 
        #                 for key,task_dict in unique_new_req_tasks.items()}
        
    #     # TODO filter tasks that can be performed by agent?
    #     # valid_event_tasks = []
    #     # payload_instrument_names = {instrument_name.lower() for instrument_name in self.payload.keys()}
    #     # for event_task in event_tasks_flat:
    #     #     if any([instrument in event_task.objective.valid_instruments 
    #     #             for instrument in payload_instrument_names]):
    #     #         valid_event_tasks.append(event_task)

    #     # add tasks to task list
    #     self._known_tasks.update(event_tasks)
        
    #     # filter tasks to only include active tasks
    #     if available_only: # only consider tasks that are active and available
    #         self._known_tasks = {key : task for key,task in self._known_tasks.items() 
    #                                 if not task.is_expired(state.get_time())}

    # def __update_reqs(self, 
    #                   state : SimulationAgentState,
    #                   incoming_reqs : List[MeasurementRequestMessage] = [], 
    #                   available_only : bool = True
    #                 ) -> None:
    #     """ Updates the known requests based on incoming requests and request availability. """
        
    #     if incoming_reqs:
    #         x = 1 # breakpoint
    #     # extract tasks from incoming requests
    #     ## find unique and new tasks in incoming requests
    #     unique_new_req_tasks = {self._req_key(msg.req): msg.req['task']
    #                             for msg in incoming_reqs
    #                             if self._req_key(msg.req) not in self._known_reqs}
    
    #     # update known requests
    #     self._known_reqs.update(unique_new_req_tasks)

    #     # filter for request availability
    #     if available_only:
    #         self._known_reqs = {key : req for key,req in self._known_reqs.items() 
    #                            if req['task']['availability']['left'] <= state.get_time() <= req['task']['availability']['right']
    #                            }

    def get_next_actions(self, state : SimulationAgentState, earliest : bool = True) -> List[AgentAction]:
        try:
            # get list of next actions from plan
            plan_out : List[AgentAction] = self._plan.get_next_actions(state.get_time(), False)
            
            # check for any observation actions in output plan
            observation_actions = [action for action in plan_out
                                   if isinstance(action, ObservationAction)]
            
            # attach observation requests to observation actions
            for action in observation_actions:
                # get current time and measurement duration
                t = state.get_time()
                dt = action.t_end - t

                assert dt > 0, "Observation action must have a positive duration."
                
                # find relevant instrument information 
                instrument : Instrument = self._payload[action.instrument_name]

                # create observation data request for environment 
                observation_req = ObservationResultsMessage(
                                                self.name,
                                                SimulationRoles.ENVIRONMENT.value,
                                                state.to_dict(),
                                                action.to_dict(),
                                                instrument.to_dict(),
                                                t,
                                                action.t_end,
                                                )
                
                # attach observation request to observation action
                action.req = observation_req.to_dict()

            # check for future broadcast message actions in output plan
            future_broadcasts = [action for action in plan_out
                                if isinstance(action, FutureBroadcastMessageAction)]

            # no future broadcasts; return plan as is
            if not future_broadcasts: 
                return plan_out
            
            # update known tasks and requests to only include active ones
            self.__update_requests(state)

            # if there are future broadcast; compile broadcast information
            msgs : list[SimulationMessage] = []
            for future_broadcast in future_broadcasts:
                
                # create appropriate broadcast message
                if future_broadcast.broadcast_type == FutureBroadcastMessageAction.STATE:
                    msgs.append(AgentStateMessage(state.agent_name, state.agent_name, state.to_dict()))
                    
                elif future_broadcast.broadcast_type == FutureBroadcastMessageAction.OBSERVATIONS:
                    # compile latest observations from the observation history
                    latest_observations : List[ObservationAction] = self.get_latest_observations(state)

                    # index by instrument name
                    instruments_used : set = {latest_observation['instrument'].lower() 
                                            for latest_observation in latest_observations}
                    indexed_observations = {instrument_used: [latest_observation for latest_observation in latest_observations
                                                            if latest_observation['instrument'].lower() == instrument_used]
                                            for instrument_used in instruments_used}

                    # create ObservationResultsMessage for each instrument
                    msgs.extend([ObservationResultsMessage(state.agent_name, 
                                                    state.agent_name, 
                                                    state.to_dict(), 
                                                    {}, 
                                                    {"name" : instrument},
                                                    state.get_time(),
                                                    state.get_time(),
                                                    observations
                                                    )
                            for instrument, observations in indexed_observations.items()])
                    
                    # msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

                elif future_broadcast.broadcast_type == FutureBroadcastMessageAction.REQUESTS:
                    msgs.extend([MeasurementRequestMessage(state.agent_name, state.agent_name, req.to_dict())
                            for req in self._known_reqs.values()
                            if state.get_time() in req.task.availability           # only active or future events
                            and (not future_broadcast.only_own_info
                                 and (future_broadcast.desc is None 
                                      or req.task in future_broadcast.desc))    # include requests from all agents if `only_own_info` is not set
                            or (future_broadcast.only_own_info and 
                                req.requester == state.agent_name)              # only requests created by myself if `only_own_info` is set
                            ])

                else: # unsupported broadcast type
                    raise NotImplementedError(f'Future broadcast type {future_broadcast.broadcast_type} not yet supported.')
            
            # remove future message action from current plan
            for future_broadcast in future_broadcasts: 
                self._plan.remove(future_broadcast, state.get_time())

            # check if requested information from future messages was found
            if not msgs:
                # get next actions from updated plan
                plan_out : List[AgentAction] = self._plan.get_next_actions(state.get_time(), False)

                # return next actions
                return plan_out
            
            else:
                # create bus message if there are messages to broadcast
                msg = BusMessage(state.agent_name, state.agent_name, [msg.to_dict() for msg in msgs])

                # create state broadcast message action
                broadcast = BroadcastMessageAction(msg.to_dict(), future_broadcasts[0].t_start)

                # get indices of future broadcast message actions in output plan
                future_broadcast_indices = [i for i, action in enumerate(plan_out) if action in future_broadcasts]

                # remove future message actions from output plan
                for i in sorted(future_broadcast_indices, reverse=True): plan_out.pop(i)
                
                # add broadcast message action from current plan
                self._plan.add(broadcast, state.get_time())
                
                # replace future message action with broadcast action in out plan
                plan_out.insert(min(future_broadcast_indices), broadcast)    

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(self._plan, "UPDATED-REPLAN", logging.WARNING)
                x = 1 # breakpoint
                # -------------------------------------

                return plan_out
        
        except Exception as e:
            self.log(f'Error in `get_next_actions()`: {e}', logging.ERROR)
            raise e

        finally:
            assert plan_out, \
                "No actions were returned from `get_next_actions()`."
            assert all([action.t_start <= state.get_time() + 1e-3 for action in plan_out]), \
                "All returned actions must start at or before the current time."
             # ensure no future broadcast message actions in output plan
            assert all([not isinstance(action, FutureBroadcastMessageAction) for action in plan_out]), \
                "No future broadcast message actions should be present in the output plan."
            # ensure all output tasks have the same duration
            assert all([((action.t_end - action.t_start) == (plan_out[0].t_end - plan_out[0].t_start)
                        or abs(action.t_end - action.t_start) - (plan_out[0].t_end - plan_out[0].t_start) < 1e-6) 
                        for action in plan_out]), \
                "All returned actions must have the same duration."
            
            # if all actions are a broadcasting action, merge into a single broadcast
            if len(plan_out) > 1 and all([isinstance(action, BroadcastMessageAction) for action in plan_out]):
                # collect start and end times
                t_start = min([action.t_start for action in plan_out]) # should be equal amongst all actions
                t_end = max([action.t_end for action in plan_out])      # should be equal amongst all actions
                
                # collect all messages
                msgs = []
                for action in plan_out:
                    if isinstance(action, BroadcastMessageAction) and action.msg['msg_type'] == SimulationMessageTypes.BUS.value:
                        msgs.extend(action.msg.get('msgs', []))
                    else:
                        msgs.append(action.msg)
                
                # create bus message to hold all messages
                bus_msg = BusMessage(  src=self.name,
                                        dst=self.name,
                                        msgs=msgs
                                    )
                # create single broadcast action
                broadcast_action = BroadcastMessageAction(  t_start=t_start,
                                                            t_end=t_end,
                                                            msg=bus_msg.to_dict()
                                                        )
                # replace multiple broadcasts with single broadcast in original plan
                for action in plan_out: self._plan.remove(action, state.get_time())
                self._plan.add(broadcast_action, state.get_time())

                # update plan out to only include single broadcast action
                plan_out = [broadcast_action]

                return plan_out
    
    def get_latest_observations(self, 
                                state : SimulationAgentState,
                                latest_plan_only : bool = True
                                ) -> List[dict]:
        return [observation_tracker.latest_observation
                 for _,grid in self._observation_history.trackers.items()
                for _, observation_tracker in grid.items()
                if isinstance(observation_tracker, ObservationTracker)
                # check if there is a latest observation
                and observation_tracker.latest_observation is not None
                # only include observations performed by myself 
                and observation_tracker.latest_observation['agent name'] == state.agent_name
                # only include observations performed for the current plan
                and self._plan.t * int(latest_plan_only) <= observation_tracker.latest_observation['t_end'] <= state.get_time()
            ]

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
            t = self._state.get_time()
            t = t if t is None else round(t,3)

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
            print(e)
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
                for task in self._known_tasks
                if isinstance(task, DefaultMissionTask)
            ]
            df = pd.DataFrame(data=data, columns=columns)        
            df.to_parquet(f"{self._results_path}/known_tasks.parquet", index=False)

            # log known and generated requests
            columns = ['id','requester','lat [deg]','lon [deg]','severity','t start','t end','t corr','event type']
            if self._processor is not None:
                data_known = [(event.id, self._processor.event_requesters[event], event.location[0], event.location[1], event.severity, event.t_start, event.t_start+event.d_exp, np.Inf, event.event_type)
                        for event in self._processor.known_events]
                data_detected = [(event.id, self._processor.event_requesters[event], event.location[0], event.location[1], event.severity, event.t_start, event.t_start+event.d_exp, np.Inf, event.event_type)
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
            data = defaultdict(list)
            for observation_tracker in self._observation_history.trackers.values():
                if observation_tracker.n_obs == 0:
                    # no observations for this grid point
                    continue                
                for key, value in observation_tracker.latest_observation.items():
                    if isinstance(value, list):
                        data[key].append(value[0])  # assuming single value lists
                    else:
                        data[key].append(value)
            
            df = pd.DataFrame(data)
            df.to_parquet(f"{self._results_path}/observation_history.parquet", index=False)

            # log performance stats
            runtime_dir = os.path.join(self._results_path, "runtime")
            if not os.path.isdir(runtime_dir): os.mkdir(runtime_dir)

        except Exception as e:
            print(f'AGENT TEARDOWN ERROR: {e}')
            raise e