from typing import List
from tqdm import tqdm
import numpy as np

from orbitpy.util import Spacecraft

from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission

from dmas.models.states import SimulationAgentState, SatelliteAgentState
from dmas.models.planning.decentralized.earliest import EarliestAccessPlanner
from dmas.models.planning.tracker import ObservationHistory
from dmas.models.actions import ObservationAction
from dmas.core.orbitdata import OrbitData

class NadirPointingPlanner(EarliestAccessPlanner):
    """ Only points agents in the downward direction """
    
    
    def _schedule_maneuvers(    self, 
                                state : SimulationAgentState, 
                                specs : object,
                                observations : List[ObservationAction],
                                orbitdata : OrbitData = None
                            ) -> list:
        # Nadir pointing planner does not schedule maneuvers
        
        # TEMPORARY ASSERTION
        assert state.attitude[0] <= 1e-3, f'Agent `{state.agent_name}` is not nadir pointing at time {state.t}. Current attitude: {state.attitude}'

        # Check if state is nadir pointing throughout the simulation
        if state.attitude[0] > 1e-3:
            # TODO if not schedule maneuvers to return to nadir pointing
            raise NotImplementedError('Nadir pointing planner can only be used with nadir pointing agents. Correction to nadir pointing not yet implemented.')

        return []
    
    
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object,  
                               orbitdata : OrbitData, 
                               observation_opportunities : List[ObservationOpportunity],
                               mission : Mission,
                               observation_history : ObservationHistory
                               ) -> list:
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # sort observation opportunities by heuristic
        observation_opportunities : list[ObservationOpportunity] = self._sort_observation_opportunities_by_heuristic(state, observation_opportunities, specs, cross_track_fovs, orbitdata, mission, observation_history)

        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # generate plan
        plan_sequence : list[tuple[ObservationOpportunity, ObservationAction]] = []

        for obs in tqdm(observation_opportunities,
                         desc=f'{state.agent_name}-PLANNER: Pre-Scheduling Observations', 
                         leave=False):
            
            # check if agent has the payload to peform observation
            if obs.instrument_name not in payload: continue

            # get previous and future observation actions' info
            th_prev,t_prev,d_prev,th_next,t_next,d_next \
                = self._get_previous_and_future_observation_info(state, obs, plan_sequence, max_slew_rate)
            
            # set task observation angle
            th_img = np.average((obs.slew_angles.left, obs.slew_angles.right))
            
            # select task imaging time and duration # TODO room for improvement? Currently aims for earliest and shortest observation possible
            t_img = max(t_prev + d_prev, obs.accessibility.left)
            d_img = obs.min_duration
            
            # check if the observation fits within the task's accessibility window
            if t_img + d_img not in obs.accessibility: continue

            # check if the observation is feasible
            prev_action_feasible : bool = (t_prev + d_prev <= t_img - 1e-6)
            curr_action_feasible : bool = (abs(th_img) <= cross_track_fovs[obs.instrument_name] / 2.0)
            next_action_feasible : bool = (t_img + d_img <= t_next - 1e-6)         
            
            if prev_action_feasible and curr_action_feasible and next_action_feasible:
                # check if task is mutually exclusive with any already scheduled tasks
                if any(obs.is_mutually_exclusive(task_j) for task_j,_ in plan_sequence): continue
                
                # create observation action
                action = ObservationAction(obs.instrument_name, 
                                           th_img, 
                                           t_img, 
                                           d_img,
                                           obs)

                # add to plan sequence
                plan_sequence.append((obs, action))

        # return sorted by start time
        return sorted([action for _,action in plan_sequence], key=lambda a : a.t_start)
    
    
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  observations : List[ObservationAction],
                                  max_slew_rate : float = None,
                                  max_torque : float = None,
                                  specs : object = None
                                ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        # return True
        if isinstance(state, SatelliteAgentState):
            # validate inputs
            assert isinstance(specs, Spacecraft), 'Agent specs must be provided as a `Spacecraft` object from `orbitpy` package.'
            
            # get pointing agility specifications                
            if max_slew_rate is None or max_torque is None:
                if specs is None: raise ValueError('Either `specs` or both `max_slew_rate` and `max_torque` must be provided.')

                max_slew_rate, max_torque = self._collect_agility_specs(specs)

            # validate agility specifications
            if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')
            if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
            assert max_slew_rate > 0.0
            # assert max_torque > 0.0

            # compile name of instruments onboard spacecraft
            instruments = [instrument.name for instrument in specs.instrument]

            # compile instrument field of view specifications   
            cross_track_fovs : dict = self._collect_fov_specs(specs)

            # check if every observation can be reached from the prior measurement
            for j,observation_j in enumerate(observations):

                # estimate the state of the agent at the given measurement
                th_j = observation_j.look_angle
                t_j = observation_j.t_start
                fov = cross_track_fovs[observation_j.instrument_name]

                # compare to prior measurements and state
                th_i = state.attitude[0]
                
                if j > 0: # there was a prior observation performed
                    # estimate the state of the agent at the prior mesurement
                    observation_i : ObservationAction = observations[j-1]
                    t_i = observation_i.t_end

                else: # there was prior measurement
                    # use agent's current state as previous state
                    t_i = state.t                
               
                # check if desired instrument is contained within the satellite's specifications
                if observation_j.instrument_name not in instruments:
                    return False 

                assert not np.isnan(th_j) and not np.isnan(th_i) # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

                # estimate maneuver time betweem states
                dth_maneuver = abs(th_j - th_i)

                # calculate time between measuremnets
                dt_measurements = t_j - t_i

                # check if observation sequence is correct 
                if dt_measurements < 0.0:
                    return False

                # fov constraint: check if the target is within the instrument's fov
                if dth_maneuver > fov / 2:
                    # target is not in the fov of the instrument; flag current observation plan as unfeasible for rescheduling
                    return False              
                                            
            # if all measurements passed the check; observation path is valid
            return True
        else:
            raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        