
    # DRAFT METHOD: bids observation value and not task value
    # def _assign_best_observations_and_revisit_times_to_proposed_path(self,
    #                                                                  state : SimulationAgentState,  
    #                                                                  candidate_path : List[ObservationAction],
    #                                                                  path_changes : List[ObservationAction],
    #                                                                  proposed_bids : Dict[GenericObservationTask, Dict[int, Bid]],
    #                                                                  specs : object,
    #                                                                  cross_track_fovs : dict,
    #                                                                  orbitdata : OrbitData,
    #                                                                  mission : Mission,
    #                                                                  observation_history : ObservationHistory
    #                                                                 ) -> Tuple[Dict[int, Dict[GenericObservationTask, int]], 
    #                                                                             Dict[int, Dict[GenericObservationTask, float]]]:
    #     """ Generate best observation numbers and revisit times for each observation in the proposed path. 
        
    #     ### Returns 
    #         - n_obs_best : Dict[int, Dict[GenericObservationTask, int]] - Best observation numbers for each observation in the proposed path.
    #         - t_prev_best : Dict[int, Dict[GenericObservationTask, float]] - Best previous observation times for each observation in the proposed path.
    #     """
    #     # extract modified task observation opportunities from path changes
    #     modified_observation_opportunities : List[ObservationOpportunity] = [obs_action.obs_opp for obs_action in path_changes]   
    #     modified_tasks : List[GenericObservationTask] =\
    #           sorted({task 
    #                   for obs_opp in modified_observation_opportunities 
    #                   for task in obs_opp.tasks}, key=lambda x: x.id)

    #     # find observation time for proposed task in candidate path
    #     modified_task_obs_times : Dict[GenericObservationTask, List[Tuple[float,str,float,ObservationOpportunity]]] \
    #                 = {task : [
    #                     (action.t_start, state.agent_name, action.look_angle, action.obs_opp) 
    #                     for action in candidate_path 
    #                     if task in action.obs_opp.tasks
    #                 ] for task in modified_tasks}
        
    #     # initialize best observation numbers and previous observation times
    #     n_obs_best : Dict[GenericObservationTask, list[str]] = defaultdict(list)
    #     t_img_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)
    #     t_prev_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)
    #     obs_names_best : Dict[GenericObservationTask, list[str]] = defaultdict(list)
    #     vals_best : Dict[GenericObservationTask, list[float]] = defaultdict(list)

    #     # initialize search for best sequence for each task
    #     best_values : dict = {task : np.NINF for task in modified_tasks}

    #     # find best observation sequences for each parent task
    #     task_sequences : Dict[GenericObservationTask, List[Tuple[tuple,float]]] = {}
    #     for task in modified_tasks:
    #         # assume parent task has been considered in results
    #         assert task in self.results, f"Parent task {task} not being bid on by any agent; cannot generate bids."
            
    #         # count all previously performed observations for this parent task
    #         performed_obs : list[Tuple[float,str,float,ObservationOpportunity]] = \
    #                         [(bid.t_img,bid.owner,np.NAN,None) 
    #                          for bid in self.results[task] 
    #                          if bid.was_performed()]
    #         n_performed_obs : int = len(performed_obs)
    #         latest_performed_obs_time : Tuple[float,str,float,ObservationOpportunity] \
    #             = max(performed_obs, key=lambda obs: obs[0]) if performed_obs else None
            
    #         if performed_obs:
    #             x = 1 # debug breakpoint

    #         # initialize feasible observation sequences for this task
    #         available_obs_times : list[Tuple[float,str,float,ObservationOpportunity]] = []

    #         # get all possible observation opportunities from results
    #         scheduled_obs_times : list[Tuple[float,str,float,ObservationOpportunity]] = \
    #               [(bid.t_img,bid.winner,np.NAN,None) for bid in self.results[task] 
    #                if bid.winner != state.agent_name
    #                and not bid.was_performed()]

    #         # include proposed task imaging time 
    #         available_obs_times.extend(scheduled_obs_times)
    #         available_obs_times.extend(modified_task_obs_times[task])

    #         # sort by observation time
    #         available_obs_times.sort(key=lambda x: x[0])

    #         # collect feasible sequences
    #         feasible_sequences = self._find_feasible_observation_sequences_for_task(state, task, available_obs_times)
            
    #         # evaluate value for each feasible sequence
    #         sequence_values : List[Tuple[Tuple,float]] = sorted([
    #             self.__calc_sequence_value(state,
    #                                       specs,
    #                                       task,
    #                                       feasible_sequence,
    #                                       n_performed_obs,
    #                                       latest_performed_obs_time,
    #                                       cross_track_fovs,
    #                                       orbitdata,
    #                                       mission)
    #             for feasible_sequence in feasible_sequences
    #         ], key=lambda x: x[1])

    #         # add to task sequences
    #         task_sequences[task] = sequence_values

    #     # initiate bid lists for tasks in the proposed path based on best observation numbers and previous observation times
    #     new_bids : Dict[ObservationOpportunity, Dict[GenericObservationTask, Bid]] = defaultdict(dict)

    #     # initiate list of best observation numbers and previous observation times for each observation in candidate path
    #     n_obs_candidate = [dict() for _ in candidate_path]
    #     t_prev_candidate = [dict() for _ in candidate_path]

    #     while True:
    #         # pick top sequence for each modified task


    #     # assign best observation numbers and previous observation times to observations in candidate path
    #     for obs_idx,obs in enumerate(candidate_path):
    #         # check if any tasks in this observation were modified
    #         if any(task in modified_tasks for task in obs.obs_opp.tasks):
    #             # find best sequence for this observation
    #             pass
    #         else:
    #             # no tasks were modified; retain existing bids from results
    #             matching_bids = [bid for bid in proposed_bids[task].values()
    #                                 if abs(bid.t_img - obs.t_start) <= self.EPS
    #                                 and bid.owner == state.agent_name]
                
    #             assert matching_bids, \
    #                 "Matching bid for observation in path not found in results. Was assigned without updating results."
    #             assert len(matching_bids) <= 1, \
    #                 "There should be at most one matching bid for the current time step."

    #             matching_bid : Bid = matching_bids.pop()

    #             # get previous matching observations for this task
    #             prev_bids_self = [bid for bid in proposed_bids[task].values()
    #                                 if bid.t_img < obs.t_start]
    #             previous_bids_other = [bid for bid in self.results[task]
    #                                 if bid.winner != state.agent_name
    #                                 and bid.t_img < obs.t_start]
    #             prev_bids = prev_bids_self + previous_bids_other

    #             # update previous observation counts
    #             n_obs_candidate[obs_idx][task] = matching_bid.n_obs
    #             t_prev_candidate[obs_idx][task] = max((bid.t_img for bid in prev_bids), default=np.NINF)

    #             if matching_bid.n_obs > 0: assert t_prev_candidate[obs_idx][task] >= 0.0, \
    #                 "Previous observation time is not defined for observation number greater than zero."
                
    #         # iterate through matching tasks of this observation
    #         # for task in obs.obs_opp.tasks:
    #         #     # check if sequence was modified for this parent task
    #         #     if task in n_obs_best:
    #         #         # extract observation time and revisit time from best sequences
    #         #         n_obs = n_obs_best[task].pop(0)
    #         #         t_prev = t_prev_best[task].pop(0)
    #         #         val = vals_best[task].pop(0)
    #         #         t_img = t_img_best[task].pop(0)

    #         #         if n_obs > 0: assert t_prev >= 0.0, \
    #         #             "Previous observation time is not defined for observation number greater than zero."

    #         #         # generate new bids for this observation if it is part of path changes
    #         #         new_bid = Bid(task, state.agent_name, n_obs, val, val, state.agent_name, t_img, state.t, main_measurement=obs.instrument_name)
    #         #         new_bids[obs.obs_opp][task] = new_bid

    #         #         # assign best observation number and previous observation time
    #         #         n_obs_candidate[obs_idx][task] = n_obs
    #         #         t_prev_candidate[obs_idx][task] = t_prev
                        
    # def __calc_sequence_value(self,
    #                           state : SimulationAgentState,
    #                           specs : object,
    #                           task : GenericObservationTask,
    #                           feasible_sequence : Tuple[List,List,List,List],
    #                           n_performed_obs : int,
    #                           latest_performed_obs_time : Tuple[float,str,float,ObservationOpportunity],
    #                           cross_track_fovs : dict,
    #                           orbitdata : OrbitData,
    #                           mission : Mission, 
    #                           ) -> float:
        
    #     # unpack feasible sequence
    #     obs_names,obs_times,obs_look_angles,obs_opps = feasible_sequence
        
    #     # initiate sequence value tracker
    #     seq_values = []
    #     t_prev_seq = []
    #     n_obs_seq = []

    #     # evaluate sequence value for this agent
    #     for seq_idx,(agent_name,t_obs,look_angle,spec_task) in enumerate(zip(obs_names,obs_times,obs_look_angles,obs_opps)):
            
    #         # get observation number for this observation
    #         n_obs = seq_idx + n_performed_obs

    #         # get observation number and previous observation time
    #         t_prev = obs_times[seq_idx-1] if seq_idx > 0 else latest_performed_obs_time[0] if n_performed_obs > 0 else np.NINF
            
    #         if n_obs > 0: assert t_prev >= 0.0, \
    #             "Previous observation time is not defined for observation number greater than zero."

    #         # get observation value
    #         if agent_name != state.agent_name: # observation is to be performed by another agent
    #             # get matching bid for this observation
    #             matching_bid : Bid = self.results[task][n_obs]

    #             # ensure matching bid is from correct agent
    #             assert matching_bid.winner == agent_name, \
    #                 "Matching bid winner does not match agent assigned to observation."
    #             assert abs(matching_bid.t_img - t_obs) <= self.EPS, \
    #                 "Matching bid observation time does not match assigned observation time."
                
    #             # get observation value from winning bid
    #             task_value = matching_bid.winning_bid

    #         else: # observation is to be performed by this agent
    #             # assume specific task was defined
    #             assert isinstance(spec_task, ObservationOpportunity), \
    #                 "Task observation opportunity not defined."
                
    #             #   estimate task value for this observation
    #             task_value = self._estimate_task_value(task,
    #                                                 spec_task.instrument_name,
    #                                                 look_angle, 
    #                                                 t_obs,
    #                                                 spec_task.min_duration,
    #                                                 specs, 
    #                                                 cross_track_fovs,
    #                                                 orbitdata,
    #                                                 mission,
    #                                                 n_obs,
    #                                                 t_prev
    #                                                 )
                
    #         # accumulate sequence value
    #         seq_values.append(task_value)     
    #         t_prev_seq.append(t_prev) 
    #         n_obs_seq.append(n_obs)     

    #     # calc total sequence value
    #     total_sequence_value = sum(seq_values)

    #     # find indices of observations not assigned to this agent
    #     indeces_to_remove = [idx 
    #                          for idx,agent_name in enumerate(obs_names)
    #                          if agent_name != state.agent_name] 
        
    #     # remove elements of sequence that are not for this agent
    #     for idx in sorted(indeces_to_remove, reverse=True):
    #         obs_names.pop(idx)
    #         obs_times.pop(idx)
    #         obs_look_angles.pop(idx)
    #         obs_opps.pop(idx)
    #         t_prev_seq.pop(idx)
    #         n_obs_seq.pop(idx)

    #     # ensure filter was successful
    #     assert all([agent_name == state.agent_name for agent_name in obs_names]), \
    #            "Not all observations from other agents were removed from best sequences."

    #     # package new sequence
    #     updated_sequence = (obs_names,obs_times,obs_look_angles,obs_opps,t_prev_seq,n_obs_seq) 
    
    #     # return updated sequence and total sequence value
    #     return total_sequence_value, updated_sequence