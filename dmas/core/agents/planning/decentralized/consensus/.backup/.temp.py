""" Temporary file containing drafts of future functions """

# IN CONSENSUS.PY/PLANNING_PHASE()
    # TODO implement bundle repair strategy
    #     max_path = [path_element for path_element in path]
    #     max_path_utility = sum([u_exp for _,_,_,u_exp in path])
    #     max_req = -1
    #     t_img = None
    #     u_exp = None

    #     while (len(bundle) < self.max_bundle_size   # there is room in the bundle
    #            and len(available_reqs) > 0          # there tasks available
    #            and max_req is not None              # a task was added in the previous iteration
    #            ):
    #         # search for best bid
    #         max_req = None
    #         for req, subtask_index in available_reqs:
    #             proposed_path, proposed_path_utility, \
    #                 proposed_t_img, proposed_utility = self.calc_path_bid(state, results, path, req, subtask_index)

    #             if proposed_path is None:
    #                 continue

    #             current_bid : Bid = results[req.id][subtask_index]

    #             if (max_path_utility < proposed_path_utility
    #                 and current_bid.winning_bid < proposed_utility):
    #                 max_path = [path_element for path_element in proposed_path]
    #                 max_path_utility = proposed_path_utility
    #                 max_req = (req, subtask_index)
    #                 t_img = proposed_t_img
    #                 u_exp = proposed_utility
            
    #         # check if a task to be added was found
    #         if max_req is not None:
    #             # update path 
    #             path = [path_element for path_element in max_path]

    #             # update bids
    #             for req, subtask_index, t_img, u_exp in max_path:
    #                 req : MeasurementRequest

    #                 # update bid
    #                 old_bid : Bid = results[req.id][subtask_index]
    #                 new_bid : Bid = old_bid.copy()
    #                 new_bid.set(u_exp, t_img, state.t)

    #                 # if changed, update results
    #                 if old_bid != new_bid:
    #                     changes.append(new_bid.copy())
    #                     results[req.id][subtask_index] = new_bid

    #             # add to bundle
    #             bid = results[req.id][subtask_index]
    #             bundle.append((req, subtask_index, bid))

# IN CONSENSUS.PY/GET_AVAILABLE_REQUESTS()
    # # find access intervals 
    # n_intervals = self.max_bundle_size - len(bundle)
    # intervals : list = self._get_available_intervals(biddable_requests, n_intervals, orbitdata)

    # # find biddable requests that can be accessed in the next observation intervals
    # available_requests = []
    # for req, main_measurement in biddable_requests:
    #     req : MeasurementRequest
    #     bid : Bid = results[req.id][main_measurement]
    #     t_arrivals = [t*orbitdata.time_step 
    #                   for t,*_ in orbitdata.gp_access_data.values
    #                   for t_start, t_end in intervals
    #                   if req.t_start <= t*orbitdata.time_step <= req.t_end 
    #                   and t_start <= t*orbitdata.time_step <= t_end]

    #     if t_arrivals:
    #         available_requests.append((req, main_measurement))

    # return available_requests

# IN CONSENSUS
# def _get_available_intervals(self, 
#                                  available_requests : list, 
#                                  n_intervals : int,
#                                  orbitdata : OrbitData
#                                  ) -> list:
#         intervals = set()
#         for req, _ in available_requests:
#             req : MeasurementRequest

#             t_arrivals = [t*orbitdata.time_step 
#                           for t,*_ in orbitdata.gp_access_data.values
#                           if req.t_start <= t*orbitdata.time_step <= req.t_end]
            
#             t_start = None
#             t_prev = None
#             for t_arrival in t_arrivals:
#                 if t_prev is None:
#                     t_prev = t_arrival
#                     t_start = t_arrival
#                     continue
                    
#                 if abs(t_arrivals[-1] - t_arrival) <= 1e-6: 
#                     intervals.add((t_start, t_arrival))
#                     continue

#                 dt = t_arrival - t_prev
#                 if abs(orbitdata.time_step - dt) <= 1e-6:
#                     t_prev = t_arrival
#                     continue
#                 else:
#                     intervals.add((t_start, t_prev))
#                     t_prev = t_arrival
#                     t_start = t_arrival

#         # split intervals if they overlap
#         intervals_to_remove = set()
#         intervals_to_add = set()
#         for t_start, t_end in intervals:
#             splits = [ [t_start_j, t_end_j, t_start, t_end] 
#                         for t_start_j, t_end_j in intervals
#                         if (t_start_j < t_start < t_end_j < t_end)
#                         or (t_start < t_start_j < t_end < t_end_j)]
            
#             for t_start_j, t_end_j, t_start, t_end in splits:
#                 intervals_to_remove.add((t_start,t_end))
#                 intervals_to_remove.add((t_start_j,t_end_j))
                
#                 split = [t_start_j, t_end_j, t_start, t_end]
#                 split.sort()

#                 intervals_to_add.add((split[0], split[1]))
#                 intervals_to_add.add((split[1], split[2]))
#                 intervals_to_add.add((split[2], split[3]))
        
#         for interval in intervals_to_remove:
#             intervals.remove(interval)

#         for interval in intervals_to_add:
#             intervals.add(interval)

#         # remove overlaps
#         intervals_to_remove = []
#         for t_start, t_end in intervals:
#             containers = [(t_start_j, t_end_j) 
#                         for t_start_j, t_end_j in intervals
#                         if (t_start_j < t_start and t_end <= t_end_j)
#                         or (t_start_j <= t_start and t_end < t_end_j)]
#             if containers:
#                 intervals_to_remove.append((t_start,t_end))
        
#         for interval in intervals_to_remove:
#             intervals.remove(interval)
        
#         # sort intervals
#         intervals = list(intervals)
#         intervals.sort(key= lambda a: a[0])

#         # return intervals 
#         return intervals[:n_intervals]

# IN ACBBA.PY
# @runtime_tracker
# def calc_path_bid(
#                     self, 
#                     state : SimulationAgentState, 
#                     specs : object,
#                     original_results : dict,
#                     original_path : list, 
#                     req : MeasurementRequest, 
#                     subtask_index : int
#                 ) -> tuple:
#     state : SimulationAgentState = state.copy()
#     winning_path = None
#     winning_path_utility = 0.0
#     winning_t_img = -1
#     winning_utility = 0.0

#     # TODO: Improve runtime efficiency:
#     for i in range(len(original_path)+1):
#         # generate possible path
#         path = [scheduled_obs for scheduled_obs in original_path]
#         path.insert(i, (req, subtask_index, -1, -1))
        
#         # self.log_task_sequence('new proposed path', path, level=logging.WARNING)

#         # recalculate bids for each task in the path if needed
#         for j in range(i, len(path), 1):
#             req_i, subtask_j, t_img_prev, _ = path[j]
                
#             if j == i:
#                 # new request and subtask are being added; recalculate bid

#                 # calculate imaging time
#                 req_i : MeasurementRequest
#                 subtask_j : int
#                 t_img = self.calc_imaging_time(state, specs, path, req_i, subtask_j)

#                 # calc utility
#                 params = {"req" : req_i.to_dict(), 
#                           "subtask_index" : subtask_j, 
#                           "t_img" : t_img}
#                 utility = self.utility_func(**params) if t_img >= 0 else np.NINF

#                 # place bid in path
#                 path[j] = (req_i, subtask_j, t_img, utility)

#             else:
#                 # elements from previous path are being adapted to new path

#                 ## calculate imaging time
#                 req_i : MeasurementRequest
#                 subtask_j : int
#                 t_img = self.calc_imaging_time(state, specs, path, req_i, subtask_j)

#                 if abs(t_img - t_img_prev) <= 1e-3:
#                     # path was unchanged; keep the remaining elements of the previous path                    
#                     break
#                 else:
#                     # calc utility
#                     params = {"req" : req_i.to_dict(), "subtask_index" : subtask_j, "t_img" : t_img}
#                     utility = self.utility_func(**params) if t_img >= 0 else np.NINF

#                     # place bid in path
#                     path[j] = (req_i, subtask_j, t_img, utility)

#         # look for path with the best utility
#         path_utility = self.__sum_path_utility(path)
#         if path_utility > winning_path_utility:
#             winning_path = [scheduled_obs for scheduled_obs in path]
#             winning_path_utility = path_utility
#             _, _, winning_t_img, winning_utility = path[i]
    
#     ## replacement strategy
#     if winning_path is None:
#         # TODO add replacement strategy
#         pass
    
#     # ensure winning path contains desired task 
#     if winning_path:
#         assert len(winning_path) == len(original_path) + 1

#         tasks = [(path_req, path_subtask_index) for path_req, path_subtask_index, _, __ in winning_path]
#         assert (req, subtask_index) in tasks

#         tasks_unique = []
#         for task in tasks:
#             if task not in tasks_unique: tasks_unique.append(task)
            
#         assert len(tasks) == len(tasks_unique)
#         assert len(tasks) == len(winning_path)
#         assert self.is_task_path_valid(state, winning_path)
#     else:
#         x = 1

#     return winning_path, winning_path_utility, winning_t_img, winning_utility