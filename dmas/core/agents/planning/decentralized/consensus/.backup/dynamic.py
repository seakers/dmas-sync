import numpy as np
from tqdm import tqdm
import concurrent.futures

from dmas.utils import runtime_tracker
from dmas.clocks import *

from chess3d.agents.actions import ObservationAction
from chess3d.orbitdata import OrbitData 
from chess3d.agents.planning.plan import Plan, ReactivePlan
from chess3d.agents.planning.decentralized.consensus.acbba import ACBBAPlanner
from chess3d.agents.planning.decentralized.consensus.bids import Bid
# from chess3d.agents.planning.rewards import RewardGrid
from chess3d.agents.science.requests import TaskRequest
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState
from chess3d.utils import Interval

from orbitpy.util import Spacecraft

class DynamicProgrammingACBBAReplanner(ACBBAPlanner):
    """ Performs a asynchronous consensus-based bundle algorithm but implements a dynamic programming approach to the bundle-building phase """
    
    @runtime_tracker
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        reward_grid ,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : dict = None
                    ) -> list:
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PRE-PLANNING PHASE', state, self.results)
        # -------------------------------
        
        # get requests that can be bid on by this agent
        available_reqs : list[tuple] = self._get_available_requests(state, specs, self.results, self.bundle, orbitdata)

        if available_reqs: 
            # schedule observations
            observations : list = self._schedule_observations(state, specs, reward_grid, self.results, orbitdata)

            # update results and bundle from observation plan 
            self.results, self.bundle, self.planner_changes = self.update_results(state, specs, reward_grid, self.results, self.bundle, observations, orbitdata)

        else:
            # no tasks available to be bid on by this agent; return original results
            observations : list = [ observation for observation in self.plan
                                    if isinstance(observation, ObservationAction)
                                    and self.is_observation_path_valid(state, specs, [observation])]
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # self.log_results('PLANNING PHASE', state, self.results)
        # print(f'bundle:')
        # for req, subtask_index, bid in self.bundle:
        #     req : MeasurementRequest
        #     bid : Bid
        #     req_id_short = req.id.split('-')[0]
        #     print(f'\t{req_id_short}, {subtask_index}, {np.round(bid.t_img,3)}, {np.round(bid.bid)}')
        # print('')
        # -------------------------------

        # generate maneuver and travel actions from observations
        maneuvers : list = self._schedule_maneuvers(state, specs, observations, clock_config, orbitdata)

        # schedule broadcasts
        broadcasts : list = self._schedule_broadcasts(state, current_plan, orbitdata)       

        # generate wait actions 
        waits : list = self._schedule_waits(state)
        
        # compile and generate plan
        self.plan = ReactivePlan(maneuvers, waits, observations, broadcasts, t=state.t, t_next=self.preplan.t_next)

        # return copy
        return self.plan.copy()
    
    @runtime_tracker
    def _schedule_observations(self, 
                               state: SimulationAgentState, 
                               specs: object, 
                               reward_grid,
                               results : dict,
                               orbitdata: OrbitData = None
                               ) -> list:
        try:
            if not isinstance(state, SatelliteAgentState):
                raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
            elif not isinstance(specs, Spacecraft):
                raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')

            # compile access times for this planning horizon
            access_opportunities, ground_points = self.calculate_access_opportunities(state, specs, orbitdata)
            access_opportunities : list; ground_points : dict

            # sort by observation time
            access_opportunities.sort(key=lambda a: a[3])

            # initiate results arrays
            t_imgs = [np.NAN for _ in access_opportunities]
            th_imgs = [np.NAN for _ in access_opportunities]
            rewards = [0.0 for _ in access_opportunities]
            cumulative_rewards = [0.0 for _ in access_opportunities]
            preceeding_observations = [np.NAN for _ in access_opportunities]
            adjancency = [[False for _ in access_opportunities] for _ in access_opportunities]

            # create adjancency matrix      
            with tqdm(total=len(access_opportunities), 
                        desc=f'{state.agent_name}-PLANNING: Generating Adjacency Matrix', 
                        leave=False) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    for j in range(len(access_opportunities)): 
                        executor.submit(self.populate_adjacency_matrix, state, specs, access_opportunities, ground_points, adjancency, j, pbar)

            # calculate optimal path and update results
            n_prev_opp = []
            for j in tqdm(range(len(access_opportunities)), 
                        desc=f'{state.agent_name}-PLANNING: Calculating Optimal Path',
                        leave=False):
                
                # get current observation opportunity
                curr_opportunity : tuple = access_opportunities[j]
                lat,lon = ground_points[curr_opportunity[0]][curr_opportunity[1]]
                curr_target = [lat,lon,0.0]

                # get any possibly prior observation
                prev_opportunities : list[tuple] = [prev_opportunity for prev_opportunity in access_opportunities
                                                    if adjancency[access_opportunities.index(prev_opportunity)][j]
                                                    and not np.isnan(th_imgs[access_opportunities.index(prev_opportunity)])
                                                    ]
                n_prev_opp.append(len(prev_opportunities))

                # calculate all possible observation actions for this observation opportunity
                possible_observations = [ObservationAction( curr_opportunity[2], 
                                                            curr_target, 
                                                            curr_opportunity[5][k], 
                                                            curr_opportunity[4][k])
                                            for k in range(len(curr_opportunity[4]))]

                # update observation time and look angle
                if not prev_opportunities: # there are no previous possible observations
                    possible_observations = [possible_observation for possible_observation in possible_observations
                                            if self.is_observation_path_valid(state, specs, [possible_observation])]

                    if possible_observations:
                        req : TaskRequest = self.get_matching_request_from_observation(possible_observations[0])
                        r_exp = reward_grid.estimate_reward(possible_observations[0])

                        if req:
                            # check if it outbids current winner
                            current_bid : Bid = results[req.id][curr_opportunity[2]]
                            
                            # if outbid, ignore
                            if current_bid.bid > r_exp: 
                                continue
                            elif abs(current_bid.bid - r_exp) <= 1e-3 and current_bid.winner != state.agent_name:
                                continue

                        t_imgs[j] = possible_observations[0].t_start
                        th_imgs[j] = possible_observations[0].look_angle
                        rewards[j] = r_exp
                
                for prev_opportunity in prev_opportunities: # there are previous possible observations
                    # get previous observation opportunity
                    prev_opportunity : tuple
                    i = access_opportunities.index(prev_opportunity)

                    # create previous observation action using known information 
                    lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
                    prev_target = [lat_prev,lon_prev,0.0]
                    prev_observation = ObservationAction(prev_opportunity[2], prev_target, th_imgs[i], t_imgs[i])
                    
                    # get possible observation actions from the current observation opportuinty
                    possible_observations = [possible_observation for possible_observation in possible_observations
                                            if self.is_observation_path_valid(state, specs, [prev_observation, possible_observation])]

                    # check if an observation is possible
                    for possible_observation in possible_observations:
                        req : TaskRequest = self.get_matching_request_from_observation(possible_observation)
                        r_exp = reward_grid.estimate_reward(possible_observation)

                        if req:
                            # check if it outbids current winner
                            current_bid : Bid = results[req.id][curr_opportunity[2]]
                            
                            # if outbid, ignore
                            if current_bid.bid >= r_exp: continue
                            
                        # update imaging time, look angle, and reward
                        t_imgs[j] = possible_observation.t_start
                        th_imgs[j] = possible_observation.look_angle
                        rewards[j] = r_exp

                        # update results
                        if cumulative_rewards[i] + rewards[j] > cumulative_rewards[j] and not np.isnan(t_imgs[i]):
                            cumulative_rewards[j] += cumulative_rewards[i] + rewards[j]
                            preceeding_observations[j] = i

                        break
                
            # extract sequence of observations from results
            visited_observation_opportunities = set()

            while preceeding_observations and np.isnan(preceeding_observations[-1]):
                preceeding_observations.pop()
            if preceeding_observations:
                observation_sequence = [len(preceeding_observations)-1]
            else:
                observation_sequence = [rewards.index(max(rewards))] if rewards and max(rewards) > 0 else []
            
            while preceeding_observations and not np.isnan(preceeding_observations[observation_sequence[-1]]):
                prev_observation_index = preceeding_observations[observation_sequence[-1]]
                
                if prev_observation_index in visited_observation_opportunities:
                    raise AssertionError('invalid sequence of observations generated by DP. Cycle detected.')
                
                visited_observation_opportunities.add(prev_observation_index)
                observation_sequence.append(prev_observation_index)

            observation_sequence.sort()

            observations : list[ObservationAction] = []
            for j in observation_sequence:
                grid_index, gp_indx, instrument_name, *_ = access_opportunities[j]
                lat,lon = ground_points[grid_index][gp_indx]
                target = [lat,lon,0.0]
                observation = ObservationAction(instrument_name, target, th_imgs[j], t_imgs[j])
                observations.append(observation)
            
            return observations
        
        finally:
            assert self.is_observation_path_valid(state, specs, observations)
    
    def get_matching_request_from_observation(self, observation : ObservationAction) -> TaskRequest:
        # extract info from observation
        lat,lon,_ = observation.targets
        main_measurement = observation.instrument_name
        t = observation.t_start

        # find matching requests
        matching_requests = {req for req in self.known_reqs
                             if isinstance(req, TaskRequest)
                             and (req.target[0] - lat) <= 1e-3
                             and (req.target[1] - lon) <= 1e-3
                             and main_measurement in req.observation_types
                             and req.t_start <= t <= req.t_end}
        
        # return findings
        return matching_requests.pop() if matching_requests else None

    @runtime_tracker
    def calculate_access_opportunities(self, 
                               state : SimulationAgentState, 
                               specs : Spacecraft,
                               orbitdata : OrbitData
                               ) -> dict:
        # define planning horizon
        t_start = state.t
        t_end = self.preplan.t_next
        t_index_start = t_start / orbitdata.time_step
        t_index_end = t_end / orbitdata.time_step

        # compile coverage data
        orbitdata_columns : list = list(orbitdata.gp_access_data.columns.values)
        raw_coverage_data = [(t_index*orbitdata.time_step, *_)
                             for t_index, *_ in orbitdata.gp_access_data.values
                             if t_index_start < t_index <= t_index_end]
        raw_coverage_data.sort(key=lambda a : a[0])

        # initiate accestimes 
        access_opportunities = {}
        ground_points = {}
        
        for data in raw_coverage_data:
            t_img = data[orbitdata_columns.index('time index')]
            grid_index = data[orbitdata_columns.index('grid index')]
            gp_index = data[orbitdata_columns.index('GP index')]
            lat = data[orbitdata_columns.index('lat [deg]')]
            lon = data[orbitdata_columns.index('lon [deg]')]
            instrument = data[orbitdata_columns.index('instrument')]
            look_angle = data[orbitdata_columns.index('look angle [deg]')]
            
            # initialize dictionaries if needed
            if grid_index not in access_opportunities:
                access_opportunities[grid_index] = {}
                ground_points[grid_index] = {}
            if gp_index not in access_opportunities[grid_index]:
                access_opportunities[grid_index][gp_index] = {instr.name : [] 
                                                        for instr in specs.instrument}
                ground_points[grid_index][gp_index] = (lat, lon)

            # compile time interval information 
            found = False
            for interval, t, th in access_opportunities[grid_index][gp_index][instrument]:
                interval : Interval
                t : list
                th : list

                if (   (t_img - orbitdata.time_step) in interval
                    or (t_img + orbitdata.time_step)in interval):
                    interval.extend(t_img)
                    t.append(t_img)
                    th.append(look_angle)
                    found = True
                    break                        

            if not found:
                access_opportunities[grid_index][gp_index][instrument].append([Interval(t_img, t_img), [t_img], [look_angle]])

        # convert to `list`
        access_opportunities = [    (grid_index, gp_index, instrument, interval, t, th)
                                    for grid_index in access_opportunities
                                    for gp_index in access_opportunities[grid_index]
                                    for instrument in access_opportunities[grid_index][gp_index]
                                    for interval, t, th in access_opportunities[grid_index][gp_index][instrument]
                                ]
                
        # return access times and grid information
        return access_opportunities, ground_points
        
    @runtime_tracker
    def populate_adjacency_matrix(self, 
                                  state : SimulationAgentState, 
                                  specs : object,
                                  access_opportunities : list, 
                                  ground_points : dict,
                                  adjacency : list,
                                  j : int,
                                  pbar : tqdm = None):
               
        # get current observation
        curr_opportunity : tuple = access_opportunities[j]
        lat_curr,lon_curr = ground_points[curr_opportunity[0]][curr_opportunity[1]]
        curr_target = [lat_curr,lon_curr,0.0]

        # get any possibly prior observation
        prev_opportunities : list[tuple] = [prev_opportunity for prev_opportunity in access_opportunities
                                            if prev_opportunity[3].end <= curr_opportunity[3].end
                                            and prev_opportunity != curr_opportunity
                                            ]

        # construct adjacency matrix
        for prev_opportunity in prev_opportunities:
            prev_opportunity : tuple

            # get previous observation opportunity's target 
            lat_prev,lon_prev = ground_points[prev_opportunity[0]][prev_opportunity[1]]
            prev_target = [lat_prev,lon_prev,0.0]

            # assume earliest observation time from previous observation
            earliest_prev_observation = ObservationAction(prev_opportunity[2], 
                                                        prev_target, 
                                                        prev_opportunity[5][0], 
                                                        prev_opportunity[4][0])
            
            # check if observation can be reached from previous observation
            adjacent = any([self.is_observation_path_valid(state, 
                                                        specs, 
                                                        [earliest_prev_observation,
                                                            ObservationAction(curr_opportunity[2], 
                                                                            curr_target, 
                                                                            curr_opportunity[5][k], 
                                                                            curr_opportunity[4][k])])
                            for k in range(len(curr_opportunity[4]))
                            ])                               

            # update adjacency matrix
            adjacency[access_opportunities.index(prev_opportunity)][j] = adjacent

            # update progress bar
            if pbar is not None: pbar.update(1)    

    def update_results(self, 
                      state : SimulationAgentState, 
                      specs : object, 
                      reward_grid : RewardGrid,
                      results : dict, 
                      bundle : list, 
                      observations : list,
                      orbitdata : OrbitData
                      ) -> tuple:
        try: 
            # initialzie changes and resetted requirement lists
            changes = []
            reset_reqs = []
            
            # check if bundle is full
            if len(self.bundle) >= self.max_bundle_size:
                # no room left in bundle; return original results
                return results, bundle, changes 
            
            # -------------------------------------
            # TEMPORARY FIX: resets bundle and always replans from scratch
            if len(bundle) > 0:
                # reset results
                for req, main_measurement, bid in bundle:
                    # reset bid
                    bid : Bid; bid._reset(state.t)

                    # update results
                    results[req.id][main_measurement] = bid

                    # add to list of resetted requests
                    reset_reqs.append((req,main_measurement))

                # reset bundle 
                bundle = []
            # -------------------------------------
            
            path_reqs = set()
            for observation in observations:
                observation : ObservationAction
                req : TaskRequest = self.get_matching_request_from_observation(observation)

                if req and len(bundle) < self.max_bundle_size: # and len(bundle) < self.max_bundle_size:
                    # update list of requests in path
                    main_measurement : str = observation.instrument_name
                    path_reqs.add((req, main_measurement))

                    # update bid
                    old_bid : Bid = results[req.id][main_measurement]
                    new_bid : Bid = old_bid.copy()
                    new_bid.set(reward_grid.estimate_reward(observation), 
                                observation.t_start, 
                                observation.look_angle, 
                                state.t)

                    # place in bundle
                    bundle.append((req, main_measurement, new_bid))

                    # if changed, update results
                    if old_bid != new_bid:
                        changes.append(new_bid.copy())
                        results[req.id][main_measurement] = new_bid

            # announce that tasks were reset and were not re-added to the plan
            reset_bids : list[Bid] = [results[req.id][main_measurement]
                                    for req,main_measurement in reset_reqs
                                    if (req,main_measurement) not in path_reqs]
            
            changes.extend(reset_bids)

            return results, bundle, changes
        
        finally:
            # ensure that bundle is within the allowed size
            # assert len(bundle) <= self.max_bundle_size

            # check if path is valid
            if self._debug: assert self.is_task_path_valid(state, specs, self.path_from_bundle(bundle), orbitdata)