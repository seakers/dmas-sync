from collections import defaultdict
from typing import Dict, List, Tuple
from abc import abstractmethod

import numpy as np

from orbitpy.util import Spacecraft

from execsatm.observations import ObservationOpportunity
from execsatm.tasks import DefaultMissionTask, GenericObservationTask
from execsatm.mission import Mission
from execsatm.objectives import DefaultMissionObjective
from execsatm.requirements import GridSpatialRequirement, SpatialCoverageRequirement, SinglePointSpatialRequirement, MultiPointSpatialRequirement
from execsatm.utils import Interval
from tqdm import tqdm

from dmas.models.actions import AgentAction, BroadcastMessageAction, FutureBroadcastMessageAction, ManeuverAction, ObservationAction, WaitAction
from dmas.models.planning.plan import Plan, PeriodicPlan
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.trackers import TaskObservationTracker
from dmas.models.states import SatelliteAgentState, SimulationAgentState
from dmas.core.messages import  AgentStateMessage, BusMessage, PlanMessage
from dmas.utils.orbitdata import OrbitData
from dmas.utils.series import TargetGridTable


class DealerPlanner(AbstractPeriodicPlanner):
    """
    Generates plans for other agents at a fixed interval.
    """
    def __init__(self, 
                 agent_results_dir : str,
                 client_ids : Dict[str, str],
                 client_orbitdata : Dict[str, OrbitData], 
                 client_specs : Dict[str, object],
                 client_missions : Dict[str, Mission],
                 horizon = np.Inf, 
                 period = np.Inf, 
                 sharing = AbstractPeriodicPlanner.OPPORTUNISTIC,
                 debug = False, 
                 logger = None,
                 printouts : bool = True):
        super().__init__(agent_results_dir, horizon, period, sharing, debug, logger, printouts)

        # check parameters
        assert isinstance(client_orbitdata, dict), \
            "Clients must be a dictionary mapping agent names to OrbitData instances."
        assert all(isinstance(client, str) for client in client_orbitdata.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(orbitdata, OrbitData) for orbitdata in client_orbitdata.values()), \
            "All clients must be instances of OrbitData."
        assert all(isinstance(client, str) for client in client_specs.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(specs, Spacecraft) for specs in client_specs.values()), \
            "All client specs must be instances of Spacecraft."
        assert all(isinstance(client, str) for client in client_missions.keys()), \
            "All keys in clients must be strings representing agent names."
        assert all(isinstance(mission, Mission) for mission in client_missions.values()), \
            "All client missions must be instances of Mission."
        assert len(client_orbitdata) == len(client_specs), \
            "Clients and client_specs must have the same number of entries ({0}).".format(len(client_orbitdata))
        assert len(client_orbitdata) == len(client_missions), \
            "Clients and client_missions must have the same number of entries ({0}).".format(len(client_orbitdata))
        assert all(client in client_specs for client in client_orbitdata), \
            "Clients and client_specs must have the same keys."
        assert all(client in client_missions for client in client_orbitdata), \
            "Clients and client_missions must have the same keys."
        
        # store client information
        self.client_orbitdata : Dict[str, OrbitData] = \
            {client: client_orbitdata for client, client_orbitdata in client_orbitdata.items()}
        self.client_specs : Dict[str, object] = \
            {client: client_specs for client, client_specs in client_specs.items()}
        self.client_missions : Dict[str, Mission] = \
            {client: client_mission for client, client_mission in client_missions.items()}
        self.cross_track_fovs : Dict[str, Dict[str, float]] = \
            self._collect_client_cross_track_fovs(client_specs)
        self.client_states : Dict[str, SatelliteAgentState] = \
            self.__initiate_client_states(client_ids, client_orbitdata, client_specs)
        self.client_plans : Dict[str, PeriodicPlan] =\
            {client : PeriodicPlan([], t=0.0, horizon=self._horizon, t_next=np.Inf)
                for client in self.client_orbitdata}
        # tracks the last plan actually transmitted to each client (may lag client_plans when a
        # client is unreachable for one or more periods, during which it keeps executing the last
        # plan it received rather than the empty plan the dealer generated for that period)
        self.client_executing_plans : Dict[str, PeriodicPlan] =\
            {client : PeriodicPlan([], t=0.0, horizon=self._horizon, t_next=np.Inf)
                for client in self.client_orbitdata}
        self.mission_tasks : Dict[Mission, List[GenericObservationTask]] \
            = self.__generate_default_mission_tasks(client_missions, client_orbitdata)            
        

    def _collect_client_cross_track_fovs(self, client_specs : Dict[str, Spacecraft]) -> Dict[str, Dict[str, float]]:
        """ get instrument field of view specifications from agent specs object """
        return {client: self._collect_fov_specs(client_specs[client]) 
                for client in client_specs}
    
    def _collect_client_agility_specs(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ 
        Collects and returns the agility specifications from agent specs object 

        Returns:
            `max_client_slew_rates`: A dictionary mapping client names to their maximum slew rates [degrees per second].
            `max_client_torques`: A dictionary mapping client names to their maximum torques [Nm].        
        """
        return ({client : self._collect_agility_specs(self.client_specs[client])[0] 
                 for client in self.client_specs},
                {client : self._collect_agility_specs(self.client_specs[client])[1] 
                 for client in self.client_specs})

    def __initiate_client_states(self, 
                                 client_ids : Dict[str, str],
                                 client_orbitdata : Dict[str, OrbitData], 
                                 client_specs : Dict[str, Spacecraft]
                                 ) -> Dict[str, SatelliteAgentState]:
        """ initiate client agent states at the start of the simulation """
        states: Dict[str, SatelliteAgentState] = {client_name : SatelliteAgentState(client_name, 
                                                                                    client_ids[client_name], 
                                                                                    client_specs[client_name].orbitState.to_dict(), 
                                                                                    client_orbitdata[client_name].time_step)
                                                   for client_name in client_orbitdata
                                                }
        # TODO adjust/propagate states to simulation start time if the simulation epoch is different than the orbitdata/specs epoch
        # TEMP SOLUTION check for cases in which the orbitstate does not match the simulation epoch
        for client_name,state in states.items():
            assert client_orbitdata[client_name].epoch == state.orbit_state['date']['jd'], \
                f"Epoch mismatch between client '{client_name}' orbitdata ({client_orbitdata[client_name].epoch}) and specs ({state.orbit_state['date']['jd']})."
            # TODO check for epoch type mismatch if needed
            # assert client_orbitdata[client_name].epoch_type.lower() == state.orbit_state['date']['@type'].lower(), \
            #     f"Epoch type mismatch between client '{client_name}' orbitdata ({client_orbitdata[client_name].epoch_type}) and specs ({state.orbit_state['date']['@type']})."

        # return states
        return states
    
    def update_percepts(self, 
                        state : SimulationAgentState,
                        current_plan : Plan,
                        tasks : Dict[Tuple,GenericObservationTask],
                        incoming_reqs: Dict[Tuple,Dict], 
                        misc_messages : list,
                        completed_actions: list,
                        aborted_actions : list,
                        pending_actions : list,
                    ):
        # update parent class percepts
        super().update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)

        # check if any client broadcasted their state or plan
        agent_state_messages : list[AgentStateMessage] = [msg for msg in misc_messages 
                                                          if isinstance(msg, AgentStateMessage)
                                                          and msg.src in self.client_states]
        # NOTE consider plan messages from clients?

        # TODO update observation history if needed

        # update the stored state 
        for agent_state_msg in agent_state_messages:
            self.client_states[agent_state_msg.src] = SimulationAgentState.from_dict(agent_state_msg.state) 

        # if no updates were received, estimate states
        if not agent_state_messages:

            # use client_executing_plans (the last plan the client actually received) rather than
            # client_plans (the most recently generated plan, which may be empty for periods when
            # the client was unreachable even though it was still executing a previous multi-period plan)
            already_propagated = set()
            t_curr = state.get_time()

            for client in self.client_orbitdata:
                plan = self.client_executing_plans[client]
                # get actions performed in between last known client state; ignore broadcast and wait actions
                filtered_actions = sorted([action for action in plan.actions
                                            if isinstance(action, (ManeuverAction, ObservationAction)) and
                                            # Original approach: any task started after last known state
                                            # and self.client_states[client]._t <= action.t_start <= state._t
                                            # New approach: only actions that are still in progress at `t_curr` or that start between last known state and `t_curr`
                                            (self.client_states[client]._t < action.t_end or abs(self.client_states[client]._t - action.t_end) < 1e-6)
                                            and (action.t_start < t_curr or abs(action.t_start - t_curr) < 1e-6)
                                            ],
                                           key=lambda a: a.t_start)

                if not filtered_actions:
                    continue

                last_action : AgentAction = filtered_actions[-1]

                if isinstance(last_action, ManeuverAction):
                    # compute attitude directly from the maneuver's stored kinematics rather than
                    # calling perform_action — perform_maneuver first calls self.update(t) which
                    # re-propagates attitude using self.attitude_rates (zeroed by the last projection),
                    # causing the satellite to appear stuck at its pre-maneuver attitude and then
                    # triggering the t>=t_end ABORTED branch with the wrong final attitude
                    t_eval = min(t_curr, last_action.t_end)
                    dt_maneuver = t_eval - last_action.t_start
                    attitude = [last_action.initial_attitude[i] + last_action.attitude_rates[i] * dt_maneuver
                                for i in range(len(last_action.initial_attitude))]
                    # propagate pos/vel first (same pattern as _generate_client_plans) to avoid
                    # double-counting: kinematic_model would re-apply rates*dt on top of our value
                    self.client_states[client] = self.client_states[client].propagate(t_curr)
                    self.client_states[client].attitude = attitude
                    self.client_states[client].attitude_rates = (
                        [0.0, 0.0, 0.0] if t_eval >= last_action.t_end - 1e-6
                        else list(last_action.attitude_rates)
                    )
                    already_propagated.add(client)

                elif isinstance(last_action, ObservationAction):
                    # observation ends with zero rates; outer propagate handles pos/vel (no double-count)
                    self.client_states[client].attitude = [last_action.look_angle, 0.0, 0.0]
                    self.client_states[client].attitude_rates = [0.0, 0.0, 0.0]

            # propagate position and velocity for clients not yet handled above
            self.client_states = {
                client: (client_state if client in already_propagated
                         else client_state.propagate(t_curr))
                for client, client_state in self.client_states.items()
            }
    
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : List[GenericObservationTask],
                        observation_history : TaskObservationTracker,
                    ) -> Plan:
        # get current time and next replanning time
        t_curr = state.get_time()
        t_next = t_curr + self._period
        horizon = self._horizon if self._horizon != np.Inf else self._period        
        
        # update plans for all client agents
        self.client_plans : Dict[str, PeriodicPlan] = self._generate_client_plans(state, specs, orbitdata, mission, tasks, observation_history)

        # update executing plans only for clients that will actually receive their new plan this period;
        # a client that is unreachable continues executing its last received plan, not the new empty one
        _next_accesses = self._calculate_agent_accesses(state, orbitdata)
        for client, _next_access in _next_accesses.items():
            if _next_access is None:
                continue
            _t_broadcast = max(_next_access.left, t_curr)
            if _t_broadcast < t_next:  # broadcast falls within this planning period
                self.client_executing_plans[client] = self.client_plans[client]

        # schedule plan broadcasts to be performed
        plan_broadcasts : list[BroadcastMessageAction] = self._schedule_broadcasts(state, orbitdata)

        # schedule wait for next planning period to start
        replan_waits : list[WaitAction] = self._schedule_periodic_replan(state, t_next)

        # generate plan from actions
        self._plan : PeriodicPlan = PeriodicPlan(plan_broadcasts, replan_waits, t=t_curr, horizon=horizon, t_next=t_next)    

        # return plan and save local copy
        return self._plan.copy()
        
    def _generate_client_plans(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               orbitdata : OrbitData, 
                               mission : Mission, 
                               tasks : List[GenericObservationTask], 
                               observation_history : TaskObservationTracker):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # Outline planning horizon interval
        planning_horizons : Dict[str,Interval] = self._calculate_horizons(state)

        # Calculate next access to each client within the replanning period
        next_accesses : Dict[str, Interval] = self._calculate_agent_accesses(state, orbitdata)

        # get current time and next replanning time
        t_curr = state.get_time()

        # DEBUT SECTION ---------------
        # failing_agent_name = 'Flood Monitoring - Dragonfly Caiman/Komodo-class MS Imager (VNIR-FL-T) Sat 8'
        # failing_agent_id = 'fl_vnir-fl-t_8'
        # if 45500.0 in planning_horizons[failing_agent_name]:
        #     tqdm.write(f'[dealer t={t_curr:.2f}s] next accesses to failing agent {failing_agent_id}: {next_accesses[failing_agent_name]}s')
        #     x = 1 # DEBUG BREAKPOINT
        # ------------------------------

        # check if there are clients reachable in the planning period
        if all(next_access is None for next_access in next_accesses.values()):
            # all clients are unreachable within the planning period; 
            # generate empty plans with only a wait action until the next planning period
            
            # get next replanning time
            t_next = t_curr + self._period
            horizon = self._horizon if self._horizon != np.Inf else self._period

            # return empty plans for each client
            return {client: PeriodicPlan([],
                                         t=t_curr,
                                         horizon=horizon,
                                         t_next=t_next)
                        for client in self.client_orbitdata
                    }

        # Project each client's estimated state forward to t_comms
        # (when they will actually receive the new plan from the dealer).
        # Without this, dummy task orientations and maneuver initial attitudes
        # are computed at state._t, but the client keeps executing its old plan
        # until t_comms, so its actual attitude at plan receipt differs from
        # the dealer's estimate — causing negative-time-step maneuver errors.
        t_curr = state.get_time()
        for client, next_access in next_accesses.items():
            if next_access is None:
                continue
            t_comms_abs = max(next_access.left, t_curr)
            t_state = self.client_states[client]._t
            if t_comms_abs <= t_state + 1e-6:
                continue

            # Find the last old-plan action that determines state at t_comms_abs.
            # Include actions already in progress at t_state (t_start <= t_state < t_end)
            # as well as future actions (t_state < t_start <= t_comms_abs).
            # Without the in-progress check, a maneuver that finishes between t_state and
            # t_comms_abs is missed: propagate() keeps using the non-zero rate and
            # overshoots the maneuver's true endpoint.
            # use client_executing_plans (not client_plans): if the client was unreachable for
            # one or more periods, client_plans[client] is the most recently generated (empty)
            # plan, but the client is still executing the last plan it actually received
            old_plan = self.client_executing_plans[client]
            relevant_actions = sorted(
                [action for action in old_plan.actions
                 if isinstance(action, (ManeuverAction, ObservationAction))
                 and action.t_start <= t_comms_abs   # started before or at comms
                 and action.t_end > t_state],         # not already finished before t_state
                key=lambda a: a.t_start
            )

            # propagate pos/vel and _t first; then overwrite attitude with the
            # plan-derived value (kinematic_model would re-integrate from self._t
            # using self.attitude_rates, double-counting any slew we computed above)
            self.client_states[client] = self.client_states[client].propagate(t_comms_abs)

            if relevant_actions:
                last_action = relevant_actions[-1]
                if isinstance(last_action, ManeuverAction):
                    # compute attitude at t_comms_abs (or at maneuver end if it finishes first)
                    t_query = min(t_comms_abs, last_action.t_end)
                    th = last_action.initial_attitude[0] + last_action.attitude_rates[0] * (t_query - last_action.t_start)
                    self.client_states[client].attitude = [th, 0.0, 0.0]
                elif isinstance(last_action, ObservationAction):
                    self.client_states[client].attitude = [last_action.look_angle, 0.0, 0.0]
                # always assume zero rates at plan receipt: the client-side stop action
                # (inserted by WorkerPlanner) zeroes the rates before any gap drift occurs
                self.client_states[client].attitude_rates = [0.0, 0.0, 0.0]

        # collect only available tasks
        available_mission_tasks : Dict[Mission, GenericObservationTask] = \
            self._collect_available_mission_tasks(planning_horizons, tasks)

        # calculate coverage opportunities for tasks
        target_access_opportunities : Dict[str, List[ List[ Dict[str, tuple]]]] = \
              self._calculate_client_target_access_opportunities(available_mission_tasks, planning_horizons)

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_client_tasks : Dict[str, list[ObservationOpportunity]] = \
              self._create_schedulable_client_tasks(available_mission_tasks, target_access_opportunities)

        # 1: schedule observations for each client
        client_observations : Dict[str, List[ObservationAction]] = \
              self._schedule_client_observations(state, available_mission_tasks, schedulable_client_tasks, observation_history, next_accesses)
                
        # validate observation paths for each client
        for client,observations in client_observations.items():
            assert all(isinstance(obs, ObservationAction) for obs in observations), \
                f'All scheduled observations for client {client} must be instances of `ObservationAction`.'
            assert all(obs.obs_opp.tasks for obs in observations), \
                f'All scheduled observations for client {client} must have a parent task.'
            assert self.is_observation_path_valid(self.client_states[client], observations, None, None, self.client_specs[client]), \
                f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'
            
        # 2: schedule maneuvers for each client
        client_maneuvers : Dict[str, List[ManeuverAction]] = self._schedule_client_maneuvers(client_observations)
        
        # validate maneuver paths for each client
        for client,maneuvers in client_maneuvers.items():
            assert all(isinstance(maneuver, ManeuverAction) for maneuver in maneuvers), \
                f'All scheduled maneuvers for client {client} must be instances of `ManeuverAction`.'
            max_slew_rate,_ = self._collect_agility_specs(self.client_specs[client])
            assert self.is_maneuver_path_valid(self.client_states[client],
                                               self.client_specs[client],
                                               client_observations[client],
                                               maneuvers,
                                               max_slew_rate,
                                               self.cross_track_fovs[client]), \
                f'Generated maneuver path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'

        # 3: schedule broadcasts for each client
        client_broadcasts : Dict[str, List[BroadcastMessageAction]] \
            = self._schedule_client_broadcasts(state, orbitdata)

        # validate broadcast paths for each client
        for client,broadcasts in client_broadcasts.items():
            assert all(isinstance(broadcast, (BroadcastMessageAction, FutureBroadcastMessageAction)) for broadcast in broadcasts), \
                f'All scheduled broadcasts for client {client} must be instances of `BroadcastMessageAction` or `FutureBroadcastMessageAction`.'


        # combine scheduled actions to create plans for each client
        client_plans : Dict[str, PeriodicPlan] = {client: PeriodicPlan(client_observations[client], 
                                                            client_maneuvers[client], 
                                                            client_broadcasts[client], 
                                                            t=state._t, horizon=planning_horizons[client].right, t_next=state._t+self._period)
                                                for client in self.client_orbitdata
                                            }

        # DEBUT SECTION ---------------
        # if 45500.0 in planning_horizons[failing_agent_name]:
        #     tqdm.write(f'[dealer t={t_curr:.2f}s] current state for failing agent {failing_agent_id}: \n\tattitude={self.client_states[failing_agent_name].attitude}\n\tattitude_rates={self.client_states[failing_agent_name].attitude_rates}')
        #     tqdm.write(f'[dealer t={t_curr:.2f}s] new plan for failing agent {failing_agent_id}: {client_plans[failing_agent_name]}')
        #     x = 1 # DEBUG BREAKPOINT
        # ------------------------------

        # return plans
        return client_plans

    def _calculate_horizons(self, state : SimulationAgentState) -> Dict[str, Interval]:
        """ calculate planning horizon intervals for each client """

        # get current simulation time
        t_curr = state.get_time()

        # calculate planning horizon span
        t_next = t_curr + self._horizon

        # initialize and populate dictionary to hold planning horizons for each client
        horizons : Dict[str, Interval] = {
            client : Interval(t_curr, t_next)
            for client in self.client_orbitdata
        }        
            
        # return horizons
        return horizons
    
    def _calculate_agent_accesses(self, state : SimulationAgentState, orbitdata : OrbitData) -> Dict[str, List[Tuple[Interval, str]]]:
        # get current simulation time
        t_curr = state.get_time()

        # calculate next replanning time
        t_next = t_curr + self._period

        # initialize dictionary to hold next access intervals for each client
        next_accesses : Dict[str, Interval] = dict()

        # calculate next access for each client
        for client in self.client_orbitdata:        
            if self._sharing == self.OPPORTUNISTIC:    
                # Try to find the next access after the desired horizon 
                next_access,*_ = orbitdata.get_next_agent_access(t_curr, target=client, t_max=t_next, include_current=True)
            elif self._sharing == self.GROUND_BASED:
                next_access, *_ = self.client_orbitdata[client].get_next_gs_access(t_curr, t_max=t_next, include_current=True)
            
            else:
                raise ValueError(f"Sharing mode `{self._sharing}` is not recognized. Supported modes are: `{self.OPPORTUNISTIC}`, `{self.PERIODIC}`, `{self.GROUND_BASED}`.")

            # set planning horizon interval for this client
            next_accesses[client] = next_access

        # return next accesses
        return next_accesses

    def __generate_default_mission_tasks(self, 
                                         client_missions : Dict[str, Mission], 
                                         client_orbitdata : Dict[str, OrbitData]
                                        ) -> Dict[Mission, List[GenericObservationTask]]:
        """ generate default tasks for all clients based on their missions and orbitdata """

        # map missions to clients
        mission_clients : Dict[Mission, list[str]] = {mission : list({client 
                                                                      for client, m in client_missions.items() 
                                                                      if m == mission}) 
                                                     for mission in client_missions.values()}

        # collect coverage grids for each client
        client_coverage_grids : Dict[str, TargetGridTable] = {client : orbitdata.grid_data 
                                                                 for client,orbitdata in client_orbitdata.items()}
        
        # validate that all clients with the same mission have the same coverage grids in their orbitdata
        for mission, clients in mission_clients.items():
            if len(clients) > 0:
                # check same number of grids
                assert all([len(client_coverage_grids[clients[0]]) == len(client_coverage_grids[client]) for client in clients]), \
                    f"All clients with the same mission must have the same number of coverage grids in their orbitdata."

                # check same grid values
                for i,grid_i in enumerate(client_coverage_grids[clients[0]]):
                    assert all([grid_i == (client_coverage_grids[client][i]) for client in clients]), \
                        f"All clients with the same mission must have the same coverage grids in their orbitdata. Clients {clients} do not."
                    
                # check same mission duration for clients with the same mission
                assert all([client_orbitdata[clients[0]].duration == client_orbitdata[client].duration for client in clients]), \
                    f"All clients with the same mission must have the same mission duration. Clients {clients} do not."

        # map mission durations
        mission_durations : Dict[Mission, float] = {mission : client_orbitdata[clients[0]].duration 
                                                    for mission,clients in mission_clients.items()}

        # map missions to grids
        mission_grids : Dict[Mission, TargetGridTable] = {mission : client_coverage_grids[clients[0]] if len(clients) > 0 else []
                                                             for mission,clients in mission_clients.items()}

        # initialize list of tasks
        tasks : Dict[Mission, List[GenericObservationTask]] = defaultdict(list)

        # for each mission and targets, generate default tasks
        for mission,grids in mission_grids.items():
             # initialize task list
            mission_tasks = []

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
                            for grid in grids
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
                        (lat, lon, int(grid_index), int(gp_index))
                        for lat,lon,grid_index,gp_index in grids
                    })
                    targets.extend(req_targets)
            
            # iterate through each mission objective
            for objective,targets in objective_targets.items():                           
                # create monitoring tasks from each location in this mission objective
                objective_tasks = [DefaultMissionTask(objective.parameter,
                                            location=(lat, lon, grid_index, gp_index),
                                            mission_duration=mission_durations[mission]*24*3600,
                                            objective=objective,
                                            )
                            for lat,lon,grid_index,gp_index in targets
                        ]
                
                # add to list of known tasks
                mission_tasks.extend(objective_tasks)

            tasks[mission] = mission_tasks

        return tasks

    def _collect_available_mission_tasks(self, planning_horizons : Dict[str, Interval], tasks: List[GenericObservationTask]) -> Dict[Mission, List[GenericObservationTask]]:
        """ get all known and active tasks for all clients within the planning horizon """        
        # overall planning horizon is the max of all client planning horizons
        mission_planning_horizon = {mission : max([interval 
                                                    for client,interval in planning_horizons.items()
                                                    if self.client_missions[client] == mission], 
                                                  key=lambda x: x.right)
                                    for mission in self.mission_tasks}
        
        # TODO consider default mission tasks
        default_mission_tasks = {mission: [task for task in tasks
                          if isinstance(task, GenericObservationTask)
                          and task.availability.overlaps(mission_planning_horizon[mission])]
                for mission, tasks in self.mission_tasks.items()}
        if any(default_tasks for default_tasks in default_mission_tasks.values()):
            raise NotImplementedError("Default mission tasks are not yet implemented in the dealer planner.")
                
        incoming_tasks = {mission: [task for task in tasks 
                                    if task.availability.overlaps(mission_planning_horizon[mission])
                                    ]
                        for mission in self.mission_tasks.keys()}      
        
        # return tasks that overlap with the overall planning horizon
        return incoming_tasks

    def _calculate_client_target_access_opportunities(self, 
                                                      available_client_tasks : Dict[Mission, List[GenericObservationTask]], 
                                                      planning_horizons : Dict[str, Interval]
                                                    ) -> Dict[str, List[ List[ Dict[str, tuple]]]]:
        """ calculates future access opportunities for ground targets for all clients within the planning horizon """
        return {client : self.calculate_access_opportunities(available_client_tasks[self.client_missions[client]], 
                                                             planning_horizons[client], 
                                                             client_orbitdata)
                for client, client_orbitdata in self.client_orbitdata.items()}

    def _create_schedulable_client_tasks(self, 
                                          available_tasks : Dict[Mission, List[GenericObservationTask]], 
                                          target_access_opportunities : dict
                                        ) -> Dict:
        return {client : self.create_observation_opportunities_from_accesses(
                                                         available_tasks[self.client_missions[client]],
                                                         client_access_opportunities, 
                                                         self.cross_track_fovs[client], 
                                                         self.client_orbitdata[client])
                for client, client_access_opportunities in target_access_opportunities.items()}

    @abstractmethod
    def _schedule_client_observations(self,
                                      state : SimulationAgentState,
                                      available_client_tasks : Dict[Mission, List[GenericObservationTask]],
                                      schedulable_client_tasks: Dict[str, List[ObservationOpportunity]],
                                      observation_history : TaskObservationTracker,
                                      next_accesses : Dict[str, 'Interval'] = None
                                    ) -> Dict[str, List[ObservationAction]]:
        """ schedules observations for all clients """
    
    
    def _schedule_client_maneuvers(self, client_observations : Dict[str, List[ObservationAction]]) -> Dict[str, List[ManeuverAction]]:
        return {client: self._schedule_maneuvers(self.client_states[client], 
                                                self.client_specs[client], 
                                                client_observations[client],
                                                self.client_orbitdata[client])
                for client in self.client_orbitdata }

    def _schedule_client_broadcasts(self, 
                                    state : SimulationAgentState, 
                                    orbitdata : OrbitData
                                ) -> Dict[str, List[BroadcastMessageAction]]:
        """ 
        Schedules broadcasts for all clients
        
        Instructs them to share their state with the dealer at the next possible accesses before the next planning period. 
        """
        # calculate replanning period for each client
        # replanning_periods : Dict[str,Interval] = self._calculate_horizons(state, orbitdata, self.period)

        # initialize dictionary to hold broadcasts for each client
        client_broadcasts = {client: [] for client in self.client_orbitdata}

        # get current time
        t_curr = state.get_time()
        t_next = t_curr + self._period

        # create future state broadcast action for each client
        for client in self.client_orbitdata.keys():
            
            if self._sharing == self.OPPORTUNISTIC:
                # get access intervals with the client agent within the planning horizon
                access_intervals : List[Tuple[Interval, str]] \
                    = orbitdata.get_next_agent_accesses(t_curr, t_max=t_next, target=client, include_current=True)

                # create broadcast actions for each access interval
                for next_access,*_ in access_intervals:

                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # if access opportunity is beyond the next planning period, skip scheduling    
                    if next_access.right <= t_next: continue

                    # get last access interval and calculate broadcast time
                    t_broadcast : float = max(next_access.left, t_next-5e-3) # ensure broadcast happens before the end of the planning period

                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # add to client broadcast list
                    # client_broadcasts[client].extend([state_msg, observations_msg, task_requests_msg])
                    client_broadcasts[client].extend([state_msg, task_requests_msg])

            elif self._sharing == self.GROUND_BASED:
                # get access intervals with the client agent within the planning horizon
                access_intervals : List[Tuple[Interval, str]] \
                    = self.client_orbitdata[client].get_next_gs_accesses(t_curr, t_max=t_next, include_current=True)

                # create broadcast actions for each access interval
                for next_access,*_ in access_intervals:

                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # if access opportunity is beyond the next planning period, skip scheduling    
                    if next_access.right <= t_next: continue

                    # get last access interval and calculate broadcast time
                    t_broadcast : float = max(next_access.left, t_next-5e-3) # ensure broadcast happens before the end of the planning period

                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # add to client broadcast list
                    # client_broadcasts[client].extend([state_msg, observations_msg, task_requests_msg])
                    client_broadcasts[client].extend([state_msg, task_requests_msg])

            elif self._sharing == self.PERIODIC:
                # determine number of periods within the planning horizon
                n_periods = int(self._horizon // self._period)

                # schedule broadcasts at the end of each period
                for i in range(n_periods):
                    # calculate broadcast time
                    t_broadcast : float = t_curr + self._period * (i + 1) - 5e-3  # ensure broadcast happens before the end of the planning period
                
                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # add to client broadcast list
                    client_broadcasts[client].extend([state_msg, observations_msg, task_requests_msg])

            else:
                raise ValueError(f'Unknown sharing mode `{self._sharing}` specified.')

        return client_broadcasts

    def _schedule_broadcasts(self, state : SimulationAgentState, orbitdata : OrbitData):
        """
        Schedules broadcasts to be performed based on the generated plans for each agent.
        """
        # get current time
        t_curr = state.get_time()
        t_next = t_curr + self._period

        # initialize list to hold broadcast actions
        broadcasts : list[BroadcastMessageAction] = []

        # initialize set of times when broadcasts are scheduled
        t_access_starts = set()    

        for client,client_plan in self.client_plans.items():
            if client_plan.is_empty():
                continue # no actions scheduled for this client; skip broadcast scheduling

            if self._sharing == self.OPPORTUNISTIC:
                # get next access interval
                next_access,*_ = orbitdata.get_next_agent_access(t_curr, target=client, t_max=t_next, include_current=True)

                # if no access opportunities in this planning horizon, skip scheduling
                if not next_access: continue

                # collect access start times for future reference
                t_access_starts.add(next_access.left)

                # calculate broadcast time
                t_broadcast : float = max(next_access.left, t_curr)

                # if broadcast time is beyond the next planning period, skip scheduling
                if t_broadcast >= state._t + self._period: continue

                # schedule broadcasts for the client
                plan_msg = PlanMessage(state.agent_name, client, [action for action in client_plan.actions], state._t)

                # create broadcast action
                plan_broadcast = BroadcastMessageAction(plan_msg, t_broadcast)
                broadcasts.append(plan_broadcast)

            elif self._sharing == self.GROUND_BASED:
                # get next access interval
                next_access,*_ = self.client_orbitdata[client].get_next_gs_access(t_curr, t_max=t_next, include_current=True)

                # if no access opportunities in this planning horizon, skip scheduling
                if not next_access: continue

                # collect access start times for future reference
                t_access_starts.add(next_access.left)

                # calculate broadcast time
                t_broadcast : float = max(next_access.left, t_curr)

                # if broadcast time is beyond the next planning period, skip scheduling
                if t_broadcast >= state._t + self._period: continue

                # schedule broadcasts for the client
                plan_msg = PlanMessage(state.agent_name, client, [action for action in client_plan.actions], state._t)

                # create broadcast action
                plan_broadcast = BroadcastMessageAction(plan_msg, t_broadcast)
                broadcasts.append(plan_broadcast)

            elif self._sharing == self.PERIODIC:
                # determine number of periods within the planning horizon
                n_periods = int(self._horizon // self._period)

                # schedule broadcasts at the end of each period
                for i in range(n_periods):
                    # calculate broadcast time
                    t_broadcast : float = t_curr + self._period * (i + 1) - 5e-3  # ensure broadcast happens before the end of the planning period

                    # schedule broadcasts for the client
                    plan_msg = PlanMessage(state.agent_name, client, [action for action in client_plan.actions], state._t)

                    # create broadcast action
                    plan_broadcast = BroadcastMessageAction(plan_msg, t_broadcast)
                    broadcasts.append(plan_broadcast)

            else:
                raise ValueError(f'Unknown sharing mode `{self._sharing}` specified.')          
           
        # # connection waits; allows for messages to be received right after access start times
        # waits = [WaitAction(t_access_start, t_access_start) for t_access_start in t_access_starts]
        # broadcasts.extend(waits)

        # # TODO test waits functionality
        # if waits: raise NotImplementedError('Waits for messages not yet tested in dealer broadcasts.')

        # return sorted broadcasts by broadcast start time
        return sorted(broadcasts, key=lambda x: x.t_start)
    
    def _schedule_observations(self, *_) -> list:
        """ Boilerplate method for scheduling observations for dealer agent. """
        return [] # dealer does not schedule its own observations, only its clients'
    
    def _schedule_periodic_replan(self, state : SimulationAgentState, t_next : float) -> list:
        """
        Creates a broadcast action that serves as a wait until the next planning time, 
        ensuring the dealer replans at the specified period while allowing other agents to 
        broadcast their states right before the next planning time. 
        
        This is necessary to prevent the dealer's state from drifting too far from the clients' 
        states due to lack of updates, which can cause errors in maneuver execution.
        """    
        # ensure next planning time is in the future
        assert state.get_time() <= t_next, "Next planning time must be in the future."
        # schedule empty broadcast action for next planning time
        blank_bus = BusMessage(state.agent_name, state.agent_name, [])
        return [BroadcastMessageAction(blank_bus, t_next)]


class TestingDealer(DealerPlanner):
    """
    A preplanner that generates plans for testing purposes.
    """
    
    def _generate_client_plans(self, state, specs, orbitdata, mission, tasks, observation_history):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # For testing purposes, just return an generic observation action for each client
        return {client: [ObservationAction('VNIR hyper',
                                           0.0,
                                           state._t+500,
                                           0.0,
                                           )] 
                for client in self.client_orbitdata.keys()}
    