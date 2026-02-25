from typing import Dict, List, Tuple
import logging
import numpy as np
from abc import abstractmethod

from dmas.models.actions import AgentAction

from execsatm.tasks import GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission
from execsatm.utils import Interval

from dmas.models.actions import BroadcastMessageAction, FutureBroadcastMessageAction, ObservationAction, WaitAction
from dmas.models.planning.plan import Plan, PeriodicPlan
from dmas.models.planning.planner import AbstractPlanner
from dmas.models.trackers import DataSink, LatestObservationTracker
from dmas.models.science.requests import TaskRequest
from dmas.models.states import GroundOperatorAgentState, SatelliteAgentState, SimulationAgentState
from dmas.utils.orbitdata import OrbitData

class AbstractPeriodicPlanner(AbstractPlanner):
    """
    # Preplanner

    Conducts operations planning for an agent at the beginning of a planning period. 
    """
    
    # sharing modes
    NONE = 'none'                   # no information sharing
    PERIODIC = 'periodic'           # periodic information sharing  
    OPPORTUNISTIC = 'opportunistic' # opportunistic information sharing based on access opportunities

    def __init__(   self, 
                    horizon : float = np.Inf,
                    period : float = np.Inf,
                    sharing : str = OPPORTUNISTIC,
                    debug : bool = False,
                    logger: logging.Logger = None,
                    printouts : bool = True
                ) -> None:
        """
        ## Preplanner 
        
        Creates an instance of a preplanner class object.

        #### Arguments:
        - horizon (`float`) : planning horizon in seconds [s]
        - period (`float`) : period of replanning in seconds [s]
        - logger (`logging.Logger`) : debugging logger
        """
        # initialize planner
        super().__init__(debug, logger, printouts)    

        # validate inputs
        assert isinstance(horizon, (int, float)) and horizon > 0, "Planning horizon must be a positive number."
        assert isinstance(period, (int, float)) and period > 0, "Replanning period must be a positive number."
        assert horizon >= period, "Planning horizon must be greater than or equal to the replanning period."
        assert sharing in {self.NONE, self.PERIODIC, self.OPPORTUNISTIC}, f"Sharing mode `{sharing}` not recognized."

        # set parameters
        self._horizon = horizon                                      # planning horizon
        self._period = period                                        # replanning period         
        self._sharing = sharing                                      # toggle for sharing plans
        self._plan = PeriodicPlan(t=-1,horizon=horizon,t_next=0.0)   # initialized empty plan
                
        # initialize attributes
        self.pending_reqs_to_broadcast : set[TaskRequest] = set()            # set of observation requests that have not been broadcasted

    def print_results(self):
        return super().print_results()

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
        # update percepts
        super().update_percepts(completed_actions)
    
    
    def needs_planning( self, 
                        state : SimulationAgentState,
                        __ : object,
                        current_plan : Plan
                        ) -> bool:
        """ Determines whether a new plan needs to be initalized """    

        if (self._plan.t < 0                  # simulation just started
            or state.get_time() >= self._plan.t_next):    # or periodic planning period has been reached
            
            pending_actions = [action for action in current_plan
                               if action.t_start <= self._plan.t_next]
            
            return not bool(pending_actions)     # no actions left to do before the end of the replanning period 
        return False

    
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : LatestObservationTracker,
                    ) -> Plan:
        """ Generates a new plan for the agent """
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # Outline planning horizon interval
        planning_horizon = Interval(state.get_time(), state.get_time() + self._horizon)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(available_tasks, planning_horizon, orbitdata)

        # create task observation opportunities from known tasks and future access opportunities
        observation_opportunities : list[ObservationOpportunity] = self.create_observation_opportunities_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)

        # schedule observation tasks
        observations : list = self._schedule_observations(state, specs, orbitdata, observation_opportunities, mission, observation_history)

        assert isinstance(observations, list) and all([isinstance(obs, ObservationAction) for obs in observations]), \
            f'Observation actions not generated correctly. Is of type `{type(observations)}` with elements of type `{type(observations[0])}`.'
        assert self.is_observation_path_valid(state, observations, max_slew_rate, max_torque, specs), \
            f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'

        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

        # generate maneuver and travel actions from measurements
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, orbitdata)
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, state.get_time()+self._period)
        
        # generate plan from actions
        self._plan : PeriodicPlan = PeriodicPlan(observations, maneuvers, broadcasts, replan, t=state.get_time(), horizon=self._horizon, t_next=state.get_time()+self._period)    

        # return plan and save local copy
        return self._plan.copy()
            
    
    def get_available_tasks(self, tasks : list, planning_horizon : Interval) -> list:
        """ Returns a list of tasks that are available at the given time """
        # if not isinstance(tasks, list):
        #     raise ValueError(f'`tasks` needs to be of type `list`. Is of type `{type(tasks)}`.')
        
        # TODO add check for capability of the agent to perform the task?      

        # Check if task is available within the proposed planning horizon
        return [task for task in tasks 
                if isinstance(task, GenericObservationTask)
                and task.availability.overlaps(planning_horizon)]
    
    @abstractmethod
    def _schedule_observations(self, state : SimulationAgentState, specs : object, orbitdata : OrbitData, observation_opportunities : list, mission : Mission, observation_history : LatestObservationTracker) -> list:
        """ Creates a list of observation actions to be performed by the agent """    

    @abstractmethod
    def _schedule_broadcasts(self, state: SimulationAgentState, observations : List[ObservationAction], orbitdata: OrbitData, t : float = None) -> List[BroadcastMessageAction]:
        """ Schedules broadcasts to be done by this agent """
        try:
            if not isinstance(state, (SatelliteAgentState, GroundOperatorAgentState)):
                raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
            elif orbitdata is None:
                raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

            # initialize list of broadcasts to be done
            broadcasts = []       

            if self._sharing == self.NONE: 
                pass # no broadcasts scheduled

            elif self._sharing == self.PERIODIC:        
                # determine current time        
                t_curr : float  = state.get_time() if t is None else t                

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
                    broadcasts.extend([state_msg, observations_msg, task_requests_msg])

            elif self._sharing == self.OPPORTUNISTIC:
                # initialize set of times when broadcasts are scheduled
                t_access_starts = set()     
                
                # get access intervals with a client agent within the planning horizon
                access_intervals : List[Tuple[Interval, str]] = orbitdata.get_next_agent_accesses(state.get_time(), include_current=True)

                # collect access start times for future reference
                t_access_starts.update([access.left for access,_ in access_intervals if not access.is_empty()])

                # create broadcast actions for each access interval
                for next_access,_ in access_intervals:
                    next_access : Interval

                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # if access opportunity is beyond the next planning period, skip scheduling    
                    if next_access.right <= state.get_time() + self._period: continue

                    # get last access interval and calculate broadcast time
                    # t_broadcast : float = max(next_access.left, state.t+self.period-5e-3) # ensure broadcast happens before the end of the planning period
                    t_broadcast : float = max(
                                            min(next_access.left + 5*self.EPS,    # give buffer time for access to start
                                                next_access.right),               # ensure broadcast is before access ends
                                        state.get_time())                                # ensure broadcast is not in the past

                    # generate plan message to share state
                    state_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.STATE, t_broadcast)

                    # generate plan message to share completed observations
                    observations_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.OBSERVATIONS, t_broadcast)

                    # generate plan message to share any task requests generated
                    task_requests_msg = FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t_broadcast)

                    # add to client broadcast list
                    broadcasts.extend([state_msg, observations_msg, task_requests_msg])

                # connection waits; allows for messages to be received right after access start times
                waits = [WaitAction(t_access_start, t_access_start) for t_access_start in t_access_starts]
                broadcasts.extend(waits)

            else:
                raise ValueError(f'Unknown sharing mode `{self._sharing}` specified.')

            # return scheduled broadcasts
            return broadcasts 
        
        finally:
            assert isinstance(broadcasts, list)
            # assert all([isinstance(broadcast, BroadcastMessageAction) for broadcast in broadcasts]), \
            #     f'Broadcasts not scheduled correctly. Is of type `{type(broadcasts)}`.'

    
    def _schedule_periodic_replan(self, state : SimulationAgentState, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        # ensure next planning time is in the future
        assert state.get_time() <= t_next, "Next planning time must be in the future."
        # schedule wait action for next planning time
        return [WaitAction(t_next,t_next)] if not np.isinf(t_next) else []
    
    
    def get_ground_points(self,
                          orbitdata : OrbitData
                        ) -> dict:
        # initiate accestimes 
        all_ground_points = list({
            (grid_index, gp_index, lat, lon)
            for grid_datum in orbitdata.grid_data
            for lat, lon, grid_index, gp_index in grid_datum.values
        })
        
        # organize into a `dict`
        ground_points = dict()
        for grid_index, gp_index, lat, lon in all_ground_points: 
            if grid_index not in ground_points: ground_points[grid_index] = dict()
            if gp_index not in ground_points[grid_index]: ground_points[grid_index][gp_index] = dict()

            ground_points[grid_index][gp_index] = (lat,lon)

        # return grid information
        return ground_points
