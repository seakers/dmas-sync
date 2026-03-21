from collections import defaultdict, deque
from copy import copy
from logging import Logger
import math
from typing import List, Dict, Tuple
from tqdm import tqdm

from orbitpy.util import Spacecraft

from execsatm.observations import DummyObservationOpportunity, ObservationOpportunity
from execsatm.mission import Mission
from execsatm.utils import Interval

from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.trackers import TaskObservationTracker
from dmas.models.states import *
from dmas.models.actions import *
from dmas.models.science.requests import *
from dmas.models.states import SimulationAgentState
from dmas.utils.orbitdata import OrbitData
from dmas.core.messages import *
from dmas.utils.tools import argmax

class DynamicProgrammingPlanner(AbstractPeriodicPlanner):          
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               orbitdata : OrbitData, 
                               observation_opportunities : List[ObservationOpportunity],
                               mission : Mission,
                               observation_history : TaskObservationTracker
                               ) -> List[ObservationAction]:
        
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')
        
        # compile list of instruments available in payload
        payload : dict = {instrument.name: instrument for instrument in specs.instrument}
        
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # get pointing agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # sort observation opportunities by start time
        observation_opportunities : list[ObservationOpportunity] = sorted(observation_opportunities, key=lambda t: (t.accessibility.left, -t.get_priority(), -len(t.tasks)))

        # add dummy observation to represent initial state
        instrument_names = list(payload.keys())
        dummy_observation = DummyObservationOpportunity(instrument_names[0], Interval(state._t,state._t), Interval(state.attitude[0],state.attitude[0]), 0.0)
        observation_opportunities.insert(0,dummy_observation)
        
        # initiate constants
        obs_to_idx = {obs: i for i, obs in enumerate(observation_opportunities)}
        d_obs : list[float]        = [obs.min_duration for obs in observation_opportunities]
        th_obs : list[float]       = [np.average((obs.slew_angles.left, obs.slew_angles.right)) for obs in observation_opportunities]
        slew_times : list[float]    = [[abs(th_obs[i] - th_obs[j]) / max_slew_rate if max_slew_rate else np.Inf
                                        for j,_ in enumerate(observation_opportunities)]
                                        for i,_ in enumerate(observation_opportunities)]        

        # generate adjacency map
        adjacency = self.__create_adjacency_dict(state, 
                                                 observation_opportunities, 
                                                 d_obs, 
                                                 slew_times)
        
        # compile current task observations from observation history
        tasks = {task for obs in observation_opportunities for task in obs.tasks}
        initial_task_states = dict()
        for task in tasks:
            # get latest observation info for this task
            task_history = observation_history.lookup(task.id)

            # unpack observation history for this task
            n_obs_prev = task_history['n_obs'] - 1 
            t_prev = task_history['t_last']

            # store initial observation count and time for this task
            initial_task_states[task.id] = (n_obs_prev, t_prev)
        
        # initialize DP states 
        dp_states : Dict[ObservationOpportunity, dict] = dict()
        for i,obs_i in enumerate(observation_opportunities):
            # initialize task states
            task_states = initial_task_states.copy()

            # check if reachable
            is_reachable = (i == 0) or (obs_i in adjacency.get(observation_opportunities[0], []))

            # calculate arrival time
            if i == 0:
                arrival_time = obs_i.accessibility.left
            else:
                # time to slew from initial attitude to first observation
                arrival_time = max(obs_i.accessibility.left, state._t + slew_times[0][i])  

            # initialize DP state for this observation opportunity
            dp_states[obs_i] = {
                'value': 0.0, 
                'arrival_time': arrival_time, 
                'task_states': task_states, 
                'predecessor': None, 
                'is_reachable': is_reachable
            }

        # initialize search for best path
        best_value = np.NINF
        best_terminal_obs = (None, -1)
        
        # Forward DP over DAG
        for i,obs_i in tqdm(enumerate(observation_opportunities),
                            total=len(observation_opportunities),
                            desc=f'{state.agent_name}-PLANNER: Evaluating Observation Opportunities',
                            leave=False,
                            disable=(len(observation_opportunities) < 10) or not self._printouts
                            ):
            
            # get dp state for current observation opportunity
            dp_state_i : dict = dp_states[obs_i]

            # skip if not reachable from source
            if not dp_state_i['is_reachable']: continue

            # Determine observation time for obs opp i
            # (Must be >= arrival_time and within obs opp i's accessibility window)
            t_obs_i = max(dp_state_i['arrival_time'], obs_i.accessibility.left)
            d_obs_i = d_obs[i]
            
            if t_obs_i > obs_i.accessibility.right:
                continue  # `obs_i` is no longer feasible
            
            # Evaluate reward for each task in `obs_i`
            # using current task states at this node
            current_task_states : dict = dp_state_i['task_states']
            updated_task_states : dict = current_task_states.copy()
            
            # initialize task observation counts and previous observation times for this node
            task_n_obs = dict()
            task_t_prevs = dict()

            for task in obs_i.tasks: 
                # get current observation count and last observation time for this task           
                n_obs_prev,t_prev = current_task_states[task.id]
                task_n_obs[task] = n_obs_prev + 1 
                task_t_prevs[task] = t_prev                
                                
                # Update task state after observation
                updated_task_states[task.id] = (n_obs_prev + 1, t_obs_i)

            # calculate reward for this observation opportunity
            obs_i_reward = self.estimate_observation_opportunity_value(
                obs_i, t_obs_i, d_obs_i, specs, cross_track_fovs, 
                orbitdata, mission, observation_history, task_n_obs, task_t_prevs
            ) if i > 0 else self.EPS  # dummy observation has no reward

            # calculate cumulative reward for this observation opportunity
            current_value = dp_state_i['value'] + obs_i_reward

            # Track best terminal value
            if current_value > best_value:
                best_value = current_value
                best_terminal_obs = (obs_i,t_obs_i,d_obs_i,th_obs[i])
            
            # Propagate to successors
            for obs_j in adjacency[obs_i]:                
                # Determine earliest feasible arrival time at `obs_j`
                # given `obs_i`'s observation time
                j = obs_to_idx[obs_j]
                maneuver_time = slew_times[i][j]
                earliest_arrival_ij = t_obs_i + d_obs_i + maneuver_time
                
                if earliest_arrival_ij > obs_j.accessibility.right:
                    continue  # `obs_j` is no longer feasible from `obs_i`

                # get dp state for `obs_j`
                dp_state_j : dict = dp_states[obs_j]
                
                # Update `obs_j` state if this path is better
                # This is where the suboptimality caveat applies:
                # we keep only the highest value path to `obs_j`
                # even though different paths may have different
                # task_states leading to different future rewards
                if current_value > dp_state_j['value']:
                    dp_state_j['value']         = current_value
                    dp_state_j['arrival_time']  = earliest_arrival_ij
                    dp_state_j['task_states']   = updated_task_states.copy()
                    dp_state_j['predecessor']   = (obs_i, t_obs_i, d_obs_i, th_obs[i])
                    dp_state_j['is_reachable']  = True

        # Step 5: Reconstruct best sequence
        sequence : List[Tuple[ObservationOpportunity, ...]]= []
        current = best_terminal_obs
        while current is not None:
            # add current observation to sequence
            sequence.append(current)

            # move to predecessor
            current_obs = current[0]
            current = dp_states[current_obs]['predecessor']
        
        # ensure at least one observation was selected
        assert sequence, "No feasible observation sequence found."

        # reverse sequence
        sequence.reverse()

        # convert sequence of (obs, t_obs) to sequence of observation actions
        observations : List[ObservationAction] = []
        for obs_k,t_obs_k,d_obs_k,th_k in sequence:
            if obs_k is None:
                continue  # skip dummy observation
            
            # get observation action parameters
            instrument_name = obs_k.instrument_name

            # create observation action and update sequence
            obs_action = ObservationAction(instrument_name, th_k, t_obs_k, d_obs_k, obs_k)

            # add to observations list
            observations.append(obs_action)
        
        # remove dummy observation from sequence
        sequence.pop(0)
        observation_opportunities.pop(0)
        observations.pop(0)
        best_value -= self.EPS 

        # return final observation sequence
        return observations    

    def __create_adjacency_dict(self, 
                                state : SatelliteAgentState, 
                                obs_opps : List[ObservationOpportunity],
                                d_imgs : List[float],
                                slew_times : List[float],
                            ) -> Dict[ObservationOpportunity,List[ObservationOpportunity]]:
        """ creates adjacency dict for DP graph """
        
        # initialize adjacency list
        adjacency : Dict[ObservationOpportunity,List[ObservationOpportunity]] = dict()

        # populate adjacency list
        for i,obs_i in tqdm(enumerate(obs_opps), 
                                desc=f'{state.agent_name}-PLANNER: Generating Adjacency Matrix',
                                leave=False,
                                total=len(obs_opps),
                                disable=(len(obs_opps) < 10) or not self._printouts
                            ):
            
            # find all proceeding observation opportunities 
            succeeding_obs = [ obs_j for j,obs_j in enumerate(obs_opps[i+1:], start=i+1)
                                if obs_i.accessibility.left < obs_j.accessibility.right
                                and obs_i.accessibility.left + d_imgs[i] + slew_times[i][j] < obs_j.accessibility.right - d_imgs[j]
                            #   and not obs_i.is_mutually_exclusive(obs_j)
                            ]
            
            # update adjacency list
            adjacency[obs_i] = succeeding_obs              
        
        # return adjacency list
        return adjacency

    def _schedule_broadcasts(self, state: SimulationAgentState, observations: list, orbitdata: OrbitData) -> list:
        # schedule measurement requests
        return super()._schedule_broadcasts(state, observations, orbitdata)