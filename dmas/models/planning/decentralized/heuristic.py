from typing import List, Tuple
from orbitpy.util import Spacecraft

from tqdm import tqdm

from execsatm.mission import Mission
from execsatm.observations import ObservationOpportunity

from dmas.utils.orbitdata import OrbitData
from dmas.utils.orbitdata import OrbitData
from dmas.core.messages import *
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.trackers import LatestObservationTracker
from dmas.models.states import *
from dmas.models.actions import *
from dmas.models.science.requests import *
from dmas.models.states import SimulationAgentState

class HeuristicInsertionPlanner(AbstractPeriodicPlanner):
    """ Schedules observations iteratively based on the highest heuristic-scoring and feasible access point """

    
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               orbitdata : OrbitData, 
                               observation_opportunities : list,
                               mission : Mission,
                               observation_history : LatestObservationTracker
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
                                                 max_slew_rate : float) -> tuple:
        
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
                                observation_history : LatestObservationTracker) -> list:
        """ Sorts tasks by heuristic value """
        
        # return if no observations to schedule
        if not observation_opportunities: return observation_opportunities

        # check if planning horizon is set
        if self.horizon < np.Inf:
            # estimate maximum number of observations in the planning horizon
            min_observation_duration = min([obs.accessibility.span() for obs in observation_opportunities])
            max_number_observations = int(self.horizon / min_observation_duration) if min_observation_duration > 0 else len(observation_opportunities)
            
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
                        observation_history : LatestObservationTracker
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