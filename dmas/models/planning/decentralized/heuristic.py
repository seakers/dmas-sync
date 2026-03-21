import copy
from typing import List, Tuple
from orbitpy.util import Spacecraft

from tqdm import tqdm

from execsatm.mission import DefaultMissionTask, Mission
from execsatm.observations import Interval, ObservationOpportunity

from dmas.models.planning.plan import Plan, ReactivePlan
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.utils.orbitdata import OrbitData
from dmas.core.messages import *
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.trackers import TaskObservationTracker
from dmas.models.states import *
from dmas.models.actions import *
from dmas.models.science.requests import *
from dmas.models.states import SimulationAgentState


class HeuristicInsertionPeriodicPlanner(AbstractPeriodicPlanner):
    """ Schedules observations iteratively (greedy) based on the highest heuristic-scoring and feasible access point """    
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               orbitdata : OrbitData, 
                               observation_opportunities : list,
                               mission : Mission,
                               observation_history : TaskObservationTracker
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # sort tasks by heuristic
        sorted_observation_opportunities : list[ObservationOpportunity] = self._sort_observation_opportunities_by_heuristic(state, observation_opportunities, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # generate plan
        plan_sequence : List[Tuple[ObservationOpportunity, ObservationAction]] = []

        for obs_opp in tqdm(sorted_observation_opportunities,
                         desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                         leave=False,
                         disable=(len(sorted_observation_opportunities) < 10) or not self._printouts
                        ):
            
            # check if agent has the payload to peform observation
            if obs_opp.instrument_name not in payload: continue

            # get previous and future observation actions' info
            th_prev,t_prev,d_prev,th_next,t_next,d_next \
                = self._get_previous_and_future_observation_info(state, obs_opp, plan_sequence, max_slew_rate)
            
            # set task observation angle
            th_img = np.average((obs_opp.slew_angles.left, obs_opp.slew_angles.right))

            # calculate maneuver times
            m_prev = abs(th_prev - th_img) / max_slew_rate if max_slew_rate else 0.0
            m_next = abs(th_img - th_next) / max_slew_rate if max_slew_rate else 0.0
            
            # select task imaging time and duration # TODO room for improvement? Currently aims for earliest and shortest observation possible
            t_img = max(t_prev + d_prev + m_prev, obs_opp.accessibility.left)
            d_img = obs_opp.min_duration
            
            # check if the observation fits within the task's accessibility window
            if t_img + d_img not in obs_opp.accessibility: continue

            # check if the observation is feasible
            prev_action_feasible : bool = (t_prev + d_prev + m_prev <= t_img - 1e-6)
            next_action_feasible : bool = (t_img + d_img + m_next   <= t_next - 1e-6)
            if prev_action_feasible and next_action_feasible:
                # check if observation is mutually exclusive with any already scheduled observations
                if any(obs_opp.is_mutually_exclusive(obs_j) for obs_j,_ in plan_sequence): continue
                
                # create observation action
                action = ObservationAction(obs_opp.instrument_name, 
                                           th_img, 
                                           t_img, 
                                           d_img,
                                           obs_opp)

                # add to plan sequence
                plan_sequence.append((obs_opp, action))

        # return sorted by start time
        return sorted([action for _,action in plan_sequence], key=lambda a : a.t_start)
    
    def _get_previous_and_future_observation_info(self, 
                                                 state : SimulationAgentState, 
                                                 observation_opportunity : ObservationOpportunity, 
                                                 plan_sequence : list, 
                                                 max_slew_rate : float
                                                ) -> tuple:
        
        # get latest previously scheduled observation
        action_prev : ObservationAction = self.__get_previous_observation_action(observation_opportunity, plan_sequence)

        # get values from previous action
        if action_prev:    
            th_prev = action_prev.look_angle
            t_prev = action_prev.t_end
            d_prev = action_prev.t_end - t_prev
            
        else:
            # no prior observation exists; compare with current state
            th_prev = state.attitude[0]
            t_prev = state._t
            d_prev = 0.0
        
        # get next earliest scheduled observation
        action_next : ObservationAction = self.__get_next_observation_action(observation_opportunity, plan_sequence)

        # get values from next action
        if action_next:
            th_next = action_next.look_angle
            t_next = action_next.t_start
            d_next = action_next.t_end - t_next
        else:
            # no future observation exists; compare with current observation opportunity
            th_next = np.average((observation_opportunity.slew_angles.left, observation_opportunity.slew_angles.right))
            t_next = observation_opportunity.accessibility.right
            d_next = 0.0

        return th_prev, t_prev, d_prev, th_next, t_next, d_next

    def __get_previous_observation_action(self, observation_opportunity : ObservationOpportunity, plan_sequence : list) -> ObservationAction:
        """ find any previously scheduled observation """
        # set types
        observations : list[ObservationAction] = [observation for _,observation in plan_sequence]

        # filter for previous actions
        actions_prev : list[ObservationAction] = [observation for observation in observations
                                                 if observation.t_end - 1e-6 <= observation_opportunity.accessibility.right]

        # return latest observation action
        return max(actions_prev, key=lambda a: a.t_end) if actions_prev else None
    
    def __get_next_observation_action(self, observation_opportunity : ObservationOpportunity, plan_sequence : list) -> ObservationAction:
         # set types
        observations : list[ObservationAction] = [observation for _,observation in plan_sequence]

        # filter for next actions
        actions_next = [observation for observation in observations
                        if observation_opportunity.accessibility.left - 1e-6 <= observation.t_start]
        
        # return earliest observation action
        return min(actions_next, key=lambda a: a.t_start) if actions_next else None

    
    def _sort_observation_opportunities_by_heuristic(self, 
                                state : SimulationAgentState, 
                                observation_opportunities : List[ObservationOpportunity], 
                                specs : Spacecraft, 
                                cross_track_fovs : dict, 
                                orbitdata : OrbitData, 
                                mission : Mission, 
                                observation_history : TaskObservationTracker) -> list:
        """ Sorts tasks by heuristic value """
        
        # return if no observations to schedule
        if not observation_opportunities: return observation_opportunities

        # check if planning horizon is set
        if self._horizon < np.Inf:
            # estimate maximum number of observations in the planning horizon
            min_observation_duration = min([obs.accessibility.span() for obs in observation_opportunities])
            max_number_observations = int(self._horizon / min_observation_duration) if min_observation_duration > 0 else len(observation_opportunities)
            
            # sort observations by accessibility duration (longest first)
            observation_opportunities.sort(key=lambda x: x.accessibility.span(),reverse=True)

            # reduce number of observations to be scheduled by using estimated max number of observations 
            observation_opportunities = observation_opportunities[:max_number_observations + 1]

        # calculate heuristic value for each observation up to the maximum number of observations
        heuristic_vals = [(obs, self._calc_heuristic(obs, specs, cross_track_fovs, orbitdata, mission, observation_history)) 
                          for obs in tqdm(observation_opportunities, 
                                           desc=f"{state.agent_name}-PREPLANNER: Calculating heuristic values", 
                                           leave=False,
                                           disable=(len(observation_opportunities) < 10) or not self._printouts
                                        )]
                
        # sort observations by heuristic value
        sorted_data = sorted(heuristic_vals, key=lambda x: x[1])
        
        # return sorted observations
        return [obs for obs,*_ in sorted_data]
    
    
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        specs : Spacecraft, 
                        cross_track_fovs : dict, 
                        orbitdata : OrbitData, 
                        mission : Mission,
                        observation_history : TaskObservationTracker
                        ) -> tuple:
        """ Heuristic function to sort tasks by their heuristic value. """
        # calculate task priority
        priority = observation_opportunity.get_priority()
        
        # calculate task duration
        duration = observation_opportunity.accessibility.span()
        
        # choose task earliest possible start time
        t_start = observation_opportunity.accessibility.left

        # choose shortest allowable task duration
        duration = observation_opportunity.min_duration 

        # calculate task reward
        obs_reward = self.estimate_observation_opportunity_value(observation_opportunity, t_start, duration, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # return to sort using: highest task reward >> highest priority >> longest duration >> earliest start time
        return -obs_reward, -priority, -duration, t_start
    
    
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations : List[ObservationAction], 
                             orbitdata: OrbitData) -> list:
        
        # do not schedule broadcasts
        return super()._schedule_broadcasts(state, observations, orbitdata)
    
class HeuristicInsertionReactivePlanner(AbstractReactivePlanner):
    """ Repairs previously constructed plans according to external inputs and changes in state. """
    
    def __init__(self, replan_threshold : int, debug = False, logger = None, printouts = True):
        super().__init__(debug, logger, printouts)
        
        # set attributes
        self._replan_threshold : int = replan_threshold
        
        # initialize set of known tasks 
        self._known_tasks = set() 

        # initialize replanning flags 
        self._task_announcements_received = False

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
        # update percepts in parent class
        super().update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)
        
        # get current time 
        t_curr = state.get_time()

        # identify new default tasks
        new_default_tasks = [task for task in tasks 
                             # filter for default mission tasks 
                             if isinstance(task, DefaultMissionTask)
                             # check if task is still available 
                             and task.is_available(t_curr)
                             # and not already known
                             and task not in self._known_tasks]

        # get new and active incoming tasks from requests
        new_req_tasks = [req.task for req in incoming_reqs 
                         # check if task is still available 
                         if req.task.is_available(t_curr)
                         # and not already known
                         and req.task not in self._known_tasks]
        
        # merge new tasks
        new_tasks = new_default_tasks + new_req_tasks

        # check if new tasks exceed threshold
        if len(new_tasks) >= self._replan_threshold: 

            # update known tasks
            self._known_tasks.update(new_tasks)
                    
            # set replan flags
            self._task_announcements_received = True

        # initialize properties
        self.access_opportunity_horizon : Interval = None
        self.access_opportunities : dict[tuple] = None

    def needs_planning(self, *_) -> bool:
        # replan if replan flag was set in `update_percepts`
        return self._task_announcements_received    

    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        _ : Plan,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : List[GenericObservationTask],
                        observation_history : TaskObservationTracker
                    ) -> Plan:
        try:
            # compile instrument field of view specifications   
            cross_track_fovs : dict = self._collect_fov_specs(specs)

            # compile agility specifications
            max_slew_rate, max_torque = self._collect_agility_specs(specs)

            # determine planning horizon 
            planning_horizon = Interval(state.get_time(), self._preplan.t_next)

            # get only available tasks
            available_tasks : list[GenericObservationTask] = self.__get_available_tasks(tasks, planning_horizon)
            
            # calculate coverage opportunities for tasks
            access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, available_tasks, planning_horizon, orbitdata)

            # create task observation opportunities from known tasks and future access opportunities
            observation_opportunities : list[ObservationOpportunity] = self.create_observation_opportunities_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)
           
            # TODO remove observation opportunities for tasks that are already scheduled?
        
            # schedule observation tasks
            observations : list = self._schedule_observations(state, specs, orbitdata, planning_horizon, observation_opportunities, mission, observation_history)

            if not self.is_observation_path_valid(state, observations, max_slew_rate, max_torque, specs):
                x = self.is_observation_path_valid(state, observations, max_slew_rate, max_torque, specs)
                y = 1

            assert isinstance(observations, list) and all([isinstance(obs, ObservationAction) for obs in observations]), \
                f'Observation actions not generated correctly. Is of type `{type(observations)}` with elements of type `{type(observations[0])}`.'
            assert self.is_observation_path_valid(state, observations, max_slew_rate, max_torque, specs), \
                f'Generated observation path/sequence is not valid. Overlaps or mutually exclusive tasks detected.'

            # schedule broadcasts to be perfomed
            broadcasts : list = self._schedule_broadcasts(state, observations, orbitdata)

            # generate maneuver and travel actions from measurements
            maneuvers : list = self._schedule_maneuvers(state, specs, observations, orbitdata)
            
            # wait for next planning period to start
            replan : list = self._schedule_periodic_replan(state, planning_horizon.right)
            
            # generate plan from actions
            self._plan : ReactivePlan = ReactivePlan(observations, maneuvers, broadcasts, replan, 
                                                     t=planning_horizon.left, t_next=planning_horizon.right)    

            # return plan and save local copy
            return self._plan.copy()
        
        finally:
            # reset replan flag
            self._task_announcements_received = False   

    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       available_tasks : List[GenericObservationTask], 
                                       planning_horizon : Interval, 
                                       orbitdata : OrbitData
                                    ) -> dict:
        """ Calculate access opportunities for targets visible in the planning horizon """
        # get current simulation time
        t_curr = state.get_time()

        # check if access opportunities need to be created/recreated
        if self.__need_to_update_access_opportunities(t_curr, planning_horizon):
            if self.__needs_to_reset_access_opportunities(t_curr, planning_horizon):
                # calculate all access opportunities for this planning horizon
                access_opportunities : dict[tuple] \
                    = super().calculate_access_opportunities(available_tasks, planning_horizon, orbitdata)
                
                # set new internal access opportunities
                self.access_opportunities = access_opportunities
                
            else:
                # check which tasks' targets have not had their access opportunities calculated yet  
                tasks_to_calculate = [task for task in available_tasks
                                      if any( tuple(loc) not in self.access_opportunities for _,_,*loc in task.location)]
                
                # calculate access opportunities of missing tasks for this planning horizon
                access_opportunities : dict[tuple] \
                    = super().calculate_access_opportunities(tasks_to_calculate, planning_horizon, orbitdata)
                
                # update internal access opportunities with newly calculated ones
                self.access_opportunities.update(access_opportunities)
            
            # update access opportunity horizon
            self.access_opportunity_horizon : Interval = copy.copy(planning_horizon)
            
            # outline is right-open interval
            self.access_opportunity_horizon.open_right()
           
        # retunrn latest known access opportunities
        return self.access_opportunities

    def __need_to_update_access_opportunities(self, t_curr : float, planning_horizon : Interval) -> bool:
        """ Check if target access opportunities need to be created/recreated. """
        
        # no existing access opportunity horizon
        if self.access_opportunity_horizon is None:
            return True
                
        # simulation has advanced beyond existing access opportunity horizon
        if t_curr not in self.access_opportunity_horizon:
            return True
        
        # planning horizon has extended beyond existing access opportunity horizon
        if planning_horizon.right > self.access_opportunity_horizon.right + self.EPS:
            return True
        
        # new tasks were added 
        if self._task_announcements_received:
            return True

        # else; there is no need to recreate access opportunities
        return False

    def __needs_to_reset_access_opportunities(self, t_curr : float, planning_horizon : Interval) -> bool:                        
        # no existing access opportunity horizon
        if self.access_opportunity_horizon is None:
            return True
        
        # simulation has advanced beyond existing access opportunity horizon
        if t_curr not in self.access_opportunity_horizon:
            return True
        
        # planning horizon has extended beyond existing access opportunity horizon
        if planning_horizon.right > self.access_opportunity_horizon.right + self.EPS:
            return True        

        # else; there is no need to recreate access opportunities
        return False

    
    def __get_available_tasks(self, tasks : list, planning_horizon : Interval) -> list:
        """ Returns a list of tasks that are available at the given time """
        # Check if task is available within the proposed planning horizon
        return [task for task in tasks 
                if isinstance(task, GenericObservationTask)
                and task.availability.overlaps(planning_horizon)]
    

    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               orbitdata : OrbitData, 
                               planning_horizon : Interval,
                               observation_opportunities : list,
                               mission : Mission,
                               observation_history : TaskObservationTracker
                               ) -> list:        
        
        # ---------------------
        # DEBUGGING BREAKPOINTS
        if "a_sat_1" in state.agent_name and state.get_time() > 38561.0:
            x = 1
        # ---------------------

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # sort tasks by heuristic
        sorted_observation_opportunities : list[ObservationOpportunity] \
            = self._sort_observation_opportunities_by_heuristic(state, observation_opportunities, specs, 
                                                                cross_track_fovs, orbitdata, planning_horizon, 
                                                                mission, observation_history)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # initialize observation plan
        t_curr = state.get_time()
        if t_curr < self._preplan.t_next and self._preplan.actions:
            # TODO consider preplanned observations
            # raise NotImplementedError('Incorporating preplanned actions into reactive replanning not yet implemented.')
            plan_sequence : List[Tuple[ObservationOpportunity, ObservationAction]] \
                = [(action.obs_opp, action) for action in self._preplan.actions 
                    if isinstance(action, ObservationAction) and t_curr < action.t_end]
                        
            scheduled_obs_opps = set(obs_opp for obs_opp,_ in plan_sequence)
            current_path = [action for _,action in plan_sequence]
            # sorted_observation_opportunities = [obs_opp for obs_opp in sorted_observation_opportunities 
            #                                     if obs_opp not in scheduled_obs_opps]
        else:
            scheduled_obs_opps = set()
            current_path = []

        for new_obs in tqdm(sorted_observation_opportunities,
                         desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                         leave=False,
                         disable=(len(sorted_observation_opportunities) < 10) or not self._printouts
                        ):
            # check if observation opportunity is already scheduled as a preplanned action
            if new_obs in scheduled_obs_opps: continue
            
            # check if agent has the payload to peform observation
            if new_obs.instrument_name not in payload: continue

            # initialize feasible observation time and select observation look angle for new observation opportunity
            t_img, th_img = None, np.average([new_obs.slew_angles.left, new_obs.slew_angles.right])

            # find possible conflicts in current path
            ## get latest observation before new observation opportunity accessibility
            prev_observations = [action for action in current_path
                                if action.t_end <= new_obs.accessibility.left]
            prev_observation = max(prev_observations, key=lambda action: action.t_end) if prev_observations else None
            ## get earliest observation after new observation opportunity accessibility
            next_observations = [action for action in current_path
                                if action.t_start >= new_obs.accessibility.right]
            next_observation = min(next_observations, key=lambda action: action.t_start) if next_observations else None
            ## find observations that are being performed during new observation opportunity accessibility
            observations_during_task_access = [action for action in current_path
                                            if action.t_start in new_obs.accessibility
                                            or action.t_end in new_obs.accessibility]

            # compile conflicting observations        
            conflicting_observations = {prev_observation, next_observation} if prev_observation else {next_observation} if next_observation else set()
            ## get unique observations during new observation opportunity access
            conflicting_observations.update(observations_during_task_access)
            ## sort conflicting observations by start time
            conflicting_observations = sorted([obs for obs in conflicting_observations 
                                            if obs is not None], key=lambda obs: obs.t_start)

            # set current state as a dummy previous observation
            obs_prev = ObservationAction(new_obs.instrument_name,  state.attitude[0], state._t)

            # check if gaps between observations can accommodate new observation opportunity
            for obs_next in conflicting_observations: 
                # check maneuver time between new task and current observations
                m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate
                m_next = abs(obs_next.look_angle - th_img) / max_slew_rate        
                
                # get earliest and latest feasible observation time
                t_earliest = max(new_obs.accessibility.left, obs_prev.t_end + m_prev)
                t_latest = min(new_obs.accessibility.right, obs_next.t_start - m_next) - new_obs.min_duration

                # check if feasible observation time exists
                ## 1) must be able to maneuver from previous observation to new task
                ## 2) must be able to maneuver from new task to next observation
                ## 3) must fit within new task accessibility window
                earliest_is_feasible = (t_earliest + new_obs.min_duration + m_next <= obs_next.t_start
                                        and obs_prev.t_end + m_prev <= t_earliest
                                        and new_obs.accessibility.left <= t_earliest
                                        and t_earliest + new_obs.min_duration <= new_obs.accessibility.right)
                latest_is_feasible = (t_latest + new_obs.min_duration + m_next <= obs_next.t_start
                                        and obs_prev.t_end + m_prev <= t_latest
                                        and new_obs.accessibility.left <= t_latest 
                                        and t_latest + new_obs.min_duration <= new_obs.accessibility.right)
                
                # if feasible, select observation time
                if earliest_is_feasible:
                    # choose earliest feasible time
                    t_img = t_earliest
                elif latest_is_feasible:
                    # choose latest feasible time
                    t_img = t_latest

                # if feasible time found, break
                if earliest_is_feasible or latest_is_feasible: break    

                # else; update previous observation
                obs_prev = obs_next

            # no conflicting observations were found; compare against current state
            if not conflicting_observations:
                # calculate maneuver time from current state
                m = abs(th_img - state.attitude[0]) / max_slew_rate
                
                # schedule at earliest maneuverable observation time
                t_img = max(new_obs.accessibility.left, state._t + m)

                # check observation time feasibility
                if t_img not in new_obs.accessibility or t_img + new_obs.min_duration not in new_obs.accessibility:
                    t_img = None # no feasible observation time found

            # check if observation time was found
            if t_img is None: 
                # no time found; cannot insert new observation into path
                continue 

            # insert new observation into path
            ## create observation action for new task
            new_observation = ObservationAction(new_obs.instrument_name, th_img, t_img, new_obs.min_duration, new_obs)

            ## create new path with inserted observation
            new_path = [action for action in current_path]
            new_path.append(new_observation)
            new_path = sorted(new_path, key=lambda action: action.t_start)
            
            # assign new path as current path
            current_path = new_path
            # if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs):
            #     current_path = new_path
                # plan_sequence.append((new_obs, new_observation))

            # # return new path if valid
            # return (new_path, [new_observation], []) if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None, None)


            # # get previous and future observation actions' info
            # th_prev,t_prev,d_prev,th_next,t_next,d_next \
            #     = self._get_previous_and_future_observation_info(state, obs_opp, plan_sequence, max_slew_rate)
            
            # # set task observation angle
            # th_img = np.average((obs_opp.slew_angles.left, obs_opp.slew_angles.right))

            # # calculate maneuver times
            # m_prev = abs(th_prev - th_img) / max_slew_rate if max_slew_rate else 0.0
            # m_next = abs(th_img - th_next) / max_slew_rate if max_slew_rate else 0.0
            
            # # select task imaging time and duration # TODO room for improvement? Currently aims for earliest and shortest observation possible
            # t_img = max(t_prev + d_prev + m_prev, obs_opp.accessibility.left)
            # d_img = obs_opp.min_duration
            
            # # check if the observation fits within the task's accessibility window
            # if t_img + d_img not in obs_opp.accessibility: continue

            # # check if the observation is feasible
            # prev_action_feasible : bool = (t_prev + d_prev + m_prev <= t_img - 1e-6)
            # next_action_feasible : bool = (t_img + d_img + m_next   <= t_next - 1e-6)
            # if prev_action_feasible and next_action_feasible:
            #     # # check if observation is mutually exclusive with any already scheduled observations
            #     # if any(obs_opp.is_mutually_exclusive(obs_j) for obs_j,_ in plan_sequence): continue
                
            #     # create observation action
            #     action = ObservationAction(obs_opp.instrument_name, 
            #                                th_img, 
            #                                t_img, 
            #                                d_img,
            #                                obs_opp)

            #     # add to plan sequence
            #     plan_sequence.append((obs_opp, action))

        # return new path
        return current_path 
    
    def _get_previous_and_future_observation_info(self, 
                                                 state : SimulationAgentState, 
                                                 observation_opportunity : ObservationOpportunity, 
                                                 plan_sequence : list, 
                                                 max_slew_rate : float) -> tuple:
        
        # get matching observation opportunities from plan sequence
        obs_actions : list[ObservationAction] = [obs_action for _,obs_action in plan_sequence]

        # get latest previously scheduled observation
        action_prev : ObservationAction = self.__get_previous_observation_action(observation_opportunity, obs_actions)

        # get values from previous action
        if action_prev:    
            th_prev = action_prev.look_angle
            t_prev = action_prev.t_end
            d_prev = action_prev.t_end - t_prev
            
        else:
            # no prior observation exists; compare with current state
            th_prev = state.attitude[0]
            t_prev = state._t
            d_prev = 0.0
        
        # get next earliest scheduled observation
        action_next : ObservationAction = self.__get_next_observation_action(observation_opportunity, obs_actions)

        # get values from next action
        if action_next:
            th_next = action_next.look_angle
            t_next = action_next.t_start
            d_next = action_next.t_end - t_next
        else:
            # no future observation exists; compare with current observation opportunity
            th_next = np.average((observation_opportunity.slew_angles.left, observation_opportunity.slew_angles.right))
            t_next = observation_opportunity.accessibility.right
            d_next = 0.0

        return th_prev, t_prev, d_prev, th_next, t_next, d_next

    def __get_previous_observation_action(self, 
                                          observation_opportunity : ObservationOpportunity, 
                                          obs_actions : List[ObservationAction]
                                        ) -> ObservationAction:
        """ find any previously scheduled observation """
        # filter for previous actions
        actions_prev : list[ObservationAction] = [observation for observation in obs_actions
                                                 if observation.t_start - 1e-6 <= observation_opportunity.accessibility.left]

        # return latest observation action
        return max(actions_prev, key=lambda a: a.t_end) if actions_prev else None
    
    def __get_next_observation_action(self, 
                                      observation_opportunity : ObservationOpportunity, 
                                      obs_actions : List[ObservationAction]
                                    ) -> ObservationAction:        
        # filter for next actions
        actions_next = [observation for observation in obs_actions
                        if observation_opportunity.accessibility.left - 1e-6 <= observation.t_start]
        
        # return earliest observation action
        return min(actions_next, key=lambda a: a.t_start) if actions_next else None
    
    def _sort_observation_opportunities_by_heuristic(self, 
                                state : SimulationAgentState, 
                                observation_opportunities : List[ObservationOpportunity], 
                                specs : Spacecraft, 
                                cross_track_fovs : dict, 
                                orbitdata : OrbitData, 
                                planning_horizon : Interval,
                                mission : Mission, 
                                observation_history : TaskObservationTracker) -> list:
        """ Sorts tasks by heuristic value """
        
        # return if no observations to schedule
        if not observation_opportunities: return observation_opportunities

        # check if planning horizon is set
        if planning_horizon.span() < np.Inf:
            # estimate maximum number of observations in the planning horizon
            min_observation_duration = min([obs.accessibility.span() for obs in observation_opportunities])
            max_number_observations = int(planning_horizon.span() / min_observation_duration) if min_observation_duration > 0 else len(observation_opportunities)
            
            # sort observations by accessibility duration (longest first)
            observation_opportunities.sort(key=lambda x: x.accessibility.span(),reverse=True)

            # reduce number of observations to be scheduled by using estimated max number of observations 
            observation_opportunities = observation_opportunities[:max_number_observations + 1]

        # calculate heuristic value for each observation up to the maximum number of observations
        heuristic_vals = [(obs, self._calc_heuristic(obs, specs, cross_track_fovs, orbitdata, mission, observation_history)) 
                          for obs in tqdm(observation_opportunities, 
                                           desc=f"{state.agent_name}-PREPLANNER: Calculating heuristic values", 
                                           leave=False,
                                           disable=(len(observation_opportunities) < 10) or not self._printouts
                                        )]
                
        # sort observations by heuristic value
        sorted_data = sorted(heuristic_vals, key=lambda x: x[1])
        
        # return sorted observations
        return [obs for obs,*_ in sorted_data]
        
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        specs : Spacecraft, 
                        cross_track_fovs : dict, 
                        orbitdata : OrbitData, 
                        mission : Mission,
                        observation_history : TaskObservationTracker
                        ) -> tuple:
        """ Heuristic function to sort tasks by their heuristic value. """
        # calculate task priority
        priority = observation_opportunity.get_priority()
        
        # calculate task duration
        duration = observation_opportunity.accessibility.span()
        
        # choose task earliest possible start time
        t_start = observation_opportunity.accessibility.left

        # choose shortest allowable task duration
        duration = observation_opportunity.min_duration 

        # calculate task reward
        obs_reward = self.estimate_observation_opportunity_value(observation_opportunity, t_start, duration, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # return to sort using: highest task reward >> highest priority >> longest duration >> earliest start time
        return -obs_reward, -priority, -duration, t_start
    
    def _schedule_broadcasts(self, 
                             state: SimulationAgentState, 
                             observations : List[ObservationAction], 
                             orbitdata: OrbitData) -> list:
        
        # initialize broadcasts list
        broadcasts : list = []

        # TODO add any broadcasts here

        # extract any additional broadcasts from preplan
        pre_planned_broadcasts = [action for action in self._preplan
                                  if isinstance(action,BroadcastMessageAction)]
        
        # add to broadcasts to be performed
        broadcasts.extend(pre_planned_broadcasts)

        # return broadcasts to be performed
        return broadcasts
    
    def _schedule_periodic_replan(self, state : SimulationAgentState, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        # ensure next planning time is in the future
        assert state.get_time() <= t_next, "Next planning time must be in the future."
        # schedule wait action for next planning time
        return [WaitAction(t_next,t_next)] if not np.isinf(t_next) else []

    def print_results(self):
        # nothing to add
        return super().print_results()