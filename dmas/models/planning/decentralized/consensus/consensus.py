from abc import abstractmethod
from collections import defaultdict, deque
from itertools import chain
from typing import Dict, List, Tuple, Union

import logging

import numpy as np

from dmas.core.messages import SimulationMessage, SimulationMessageTypes
from dmas.models.actions import AgentAction

from execsatm.tasks import DefaultMissionTask, EventObservationTask, GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission

from dmas.models.actions import BroadcastMessageAction, FutureBroadcastMessageAction, ObservationAction, WaitAction
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.trackers import DataSink, LatestObservationTracker
from dmas.models.planning.plan import Plan, PeriodicPlan, ReactivePlan
from dmas.models.planning.decentralized.consensus.bids import Bid
from dmas.models.science.requests import TaskRequest
from dmas.models.states import GroundOperatorAgentState, SatelliteAgentState, SimulationAgentState
from dmas.core.messages import MeasurementBidMessage
from dmas.utils.orbitdata import OrbitData

class ConsensusPlanner(AbstractReactivePlanner):    
    # Replanning models
    # NOTE break down into separate replanner classes that use this class as the parent class?
    HEURISTIC_INSERTION = 'heuristicInsertion'
    DYNAMIC_PROGRAMMING = 'dynamicProgramming'
    MILP = 'mixedIntegerLinearProgramming'
    MODELS = [HEURISTIC_INSERTION, DYNAMIC_PROGRAMMING, MILP]
   
    def __init__(self, 
                 model : str,
                 replan_threshold : int,
                 optimistic_bidding_threshold : int,
                 periodic_overwrite : bool,
                 agent_results_dir : str,
                 debug : bool = False,
                 logger: logging.Logger = None,
                 printouts : bool = True
                ) -> None:
        super().__init__(debug, logger, printouts)
        """
        ## Consensus Couple-Constrained Planner
        
        ### Arguments
        - `model` (str): The consensus replanning model to use. Must be one of the defined models in `MODELS`.
        - `replan_threshold` (int): The minimum number of new urgent tasks required to trigger replanning.
        - `optimistic_bidding_threshold` (int): The number of consensus rounds to wait before abandoning optimistic bids for tasks without new bids.        
        - `periodic_overwrite` (bool): Whether to overwrite results upon the generation of a new periodic plan.


        ## Bundle
        The Bundle is defined as a list of a tuple indicating the specific observation opportunity that was added to the plan, 
        and a dictionary that maps the observation number being bid on.
        
        """

        # TODO implement periodic overwrite functionality
        if periodic_overwrite: raise NotImplementedError("Periodic overwrite functionality not yet implemented.")

        # validate inputs
        assert isinstance(model, str), "Model must be a string."
        assert model in self.MODELS, f"Invalid model '{model}'. Must be one of {self.MODELS}."
        assert isinstance(replan_threshold, int) and replan_threshold > 0, "Replan threshold must be positive integer."
        assert isinstance(optimistic_bidding_threshold, int), "Optimistic bidding threshold must be an integer"
        assert optimistic_bidding_threshold >= 0, "Optimistic bidding threshold must be non-negative"

        # initialize consensus results
        self._bundle : List[Tuple[ObservationOpportunity, Dict[GenericObservationTask, int]]] = list()
        self._path : List[ObservationAction] = list()
        self._results : Dict[GenericObservationTask, List[Bid]] = defaultdict(list)
        self._optimistic_bidding_counters : Dict[GenericObservationTask, List[int]] = defaultdict(list)
        self._id_to_tasks : Dict[str, GenericObservationTask] = dict()

        # initialize urgent tasks and bid inbox/outbox
        self._known_event_tasks : set[GenericObservationTask] = set()
        self._incoming_event_tasks : list[GenericObservationTask] = list()
        self._relevant_updates : List[Bid] = list()

        # set parameters
        self._model = model
        self._replan_threshold = replan_threshold
        self._optimistic_bidding_threshold = optimistic_bidding_threshold
        self._periodic_overwrite = periodic_overwrite

        # replanning flags 
        self._task_announcements_received = False
        self._results_changes_performed = False
        self._bundle_changes_performed = False
        self._performed_bundle_observations = False

        # initialize bid results sinks
        self._observation_rewards = DataSink(out_dir=agent_results_dir, owner_name='ConsensusPlanner', data_name='rewards')

    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """
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
        """ Updates internal knowledge based on incoming percepts """
        # update base percepts
        super().update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)

        # collect bids from incoming messages to inbox
        incoming_bids : List[Union[Bid,dict]] \
            = self.__collect_incoming_bids(misc_messages)

        # collect performed observations from completed actions
        performed_observations : List[ObservationAction] = [action for action in completed_actions 
                                                            if isinstance(action, ObservationAction)]

        # -------------------------------
        # DEBUG PRINTOUTS
        if self._debug and incoming_bids:
            self._log_results('CONSENSUS PHASE - RESULTS (BEFORE)', state, self._results)
            print(f'`{state.agent_name}` - Received {len(incoming_bids)} incoming bids and {len(incoming_reqs)} task requests.')
            self._log_bundle('CONSENSUS PHASE - BUNDLE (BEFORE)', state, self._bundle)
        # -------------------------------

        # perform consensus phase for incoming task bids
        task_updates, results_updates, bundle_updates, performed_bundle_observations \
              = self._consensus_phase(state, incoming_reqs, incoming_bids, tasks, current_plan, performed_observations)

        # update latest observations performed 
        self.latest_performed_observations.update(performed_bundle_observations)

        # assume bundle and results are now consistent
        assert len(self._bundle) == len(self._path), \
            "Bundle and path lengths do not match after consensus phase."

        # -------------------------------
        # DEBUG PRINTOUTS
        # if (task_updates or results_updates or bundle_updates) and self._debug:
        if self._debug:
            self._log_results('CONSENSUS PHASE - RESULTS (AFTER)', state, self._results)
            print(f'`{state.agent_name}` - Performed {len(task_updates)} task updates, {len(results_updates)} results updates, and {len(bundle_updates)} bundle updates.')
            if any([
                len(task_updates) > 0,
                len(results_updates) > 0, 
                len(bundle_updates) > 0
            ]):
                print(f'`{state.agent_name}` - Relevant updates detected; replanning is required.')
            else:
                print(f'`{state.agent_name}` - No relevant updates detected; no replanning required.')
            # self._log_bundle('CONSENSUS PHASE - BUNDLE (AFTER)', state, self.bundle)
            # self._log_path('CONSENSUS PHASE - PATH (AFTER)', state, self.path)
        # -------------------------------

        # set replanning flags
        # 0) there were new tasks that were not previously considered by agent
        self._task_announcements_received = len(task_updates) > 0
        # 1) there were relevant updates to bids/results
        self._results_changes_performed = len(results_updates) > 0
        # 2) incoming bids modified tasks in my bundle
        self._bundle_changes_performed = len(bundle_updates) > 0
        # 3) there were new observations performed by parent agent that were in my bundle
        self._performed_bundle_observations = len(performed_bundle_observations) > 0
        
        # clear incoming percepts after processing
        incoming_bids.clear()
        incoming_reqs.clear()        
        
        # clear updates lists after processing
        task_updates.clear()
        results_updates.clear()
        bundle_updates.clear()    


    def __collect_incoming_bids(self, misc_messages : List[SimulationMessage]) -> List[Union[Bid,dict]]:
        """ Collect bids from incoming messages and requests. """
        # initialize list of incoming bids
        incoming_bids : List[Union[Bid,dict]] = []

        # iterate through incoming messages
        for msg in misc_messages:
            # check if message is a bid message
            if isinstance(msg, MeasurementBidMessage):
                # add bid to incoming bids
                incoming_bids.append(msg.bid)
            # if the message is a dictionary, check if it is a bid
            elif isinstance(msg, dict) and msg.get("msg_type") == SimulationMessageTypes.MEASUREMENT_BID.value:
                # add bid dictionary to incoming bids
                incoming_bids.append(msg['bid'])
        
        # TODO include support for BidResultsMessage when re-enabled?

        # return list of incoming bids
        return incoming_bids

    def _consensus_phase(self,
                        state : SimulationAgentState,
                        incoming_reqs : List[TaskRequest],
                        incoming_bids : List[Union[Bid,dict]],
                        tasks : List[GenericObservationTask],
                        current_plan : Plan,
                        performed_observations : List[ObservationAction]
                    ) -> Tuple[List[Bid], List[Bid], List[Bid], List[ObservationAction]]:
        """ Perform consensus phase to update bids and bundle. """

        # check for new default mission tasks
        new_default_tasks = self.__process_default_tasks(state, tasks)

        # check for new urgent tasks
        new_urgent_task_added \
            = self.__process_incoming_task_requests(state, incoming_reqs, incoming_bids)

        # check if new base plan is available
        preplan_obs, preplan_resets \
            = self.__update_bundle_from_preplan(state, current_plan)
        
        # check if tasks in the bundle were performed by parent agent
        self._bundle, self._path, performed_bundle_updates \
            = self.__update_performed_bundle_observations(state, performed_observations)

        # compare results with incoming bids and update bundle
        comparison_updates = self.__compare_incoming_bids(state, incoming_bids)

        # check if planned tasks expired
        expired_tasks, self._bundle, expired_bundle_updates \
              = self.__remove_expired_tasks(state)

        # TODO make sure this is not needed:
        # # check if bids in results would have been performed by other agents
        # performed_updates = self.__update_performed_bids(state)
        
        # compile updates and return list of updates
        task_updates = list(chain.from_iterable([
                                                    new_default_tasks,
                                                    new_urgent_task_added, 
                                                ]))
        results_updates = list(chain.from_iterable([
                                                    # new_default_tasks,
                                                    # new_urgent_task_added, 
                                                    expired_tasks, 
                                                    preplan_obs,
                                                    performed_bundle_updates,
                                                    comparison_updates, 
                                                    # performed_updates, # TODO re-enable when implemented
                                                   ]))   
        bundle_updates = list(chain.from_iterable([
                                                    preplan_resets,
                                                    expired_bundle_updates
                                                    # performed_bundle_updates
                                                ]))

        # update bundle and enforce constraints iteratively on results
        while True:
            # update bundle from results updates
            self._bundle, self._path, results_bundle_updates \
                = self.__update_bundle_from_results(state)

            # enforce constraints in results
            constraint_violations = self.__check_results_constraints(state)

            # append updates to compiling lists
            bundle_updates.extend(results_bundle_updates)
            results_updates.extend(constraint_violations)

            # check for further updates
            if not results_bundle_updates and not constraint_violations:
                break # no more updates; exit loop       
        
        # collect performed observation opportunities
        performed_bundle_observations : List[ObservationOpportunity] \
            = [obs_opp for obs_opp,_ in performed_bundle_updates]
                
        # return lists of updates
        return task_updates, results_updates, bundle_updates, performed_bundle_observations   

    def __update_bundle_from_preplan(self, 
                                     state : SimulationAgentState, 
                                     current_plan : Plan
                                    ) -> tuple:
        """ Update latest preplan if new plan is available. """
        # check if new periodic plan is available
        if not isinstance(current_plan, PeriodicPlan) or abs(state.get_time() - current_plan.t) > self.EPS:
            # no new preplan available; return no updates
            return [], []
    
        # save new preplan 
        #   DEPRECATED, done at the `AbstractReactivePlanner` level 
        # self._preplan : PeriodicPlan = current_plan.copy()

        # extract observations path from new preplan
        preplan_path : List[ObservationAction] = \
                [action for action in current_plan if isinstance(action, ObservationAction)]

        # ensure all tasks in preplan observations are known in results
        assert all((task in self._results 
                    for obs_action in preplan_path 
                    for task in obs_action.obs_opp.tasks)), \
            "All tasks in preplan observations must be known in results."

        # initiate list of changes to bundle        
        bundle_resets = []
        t_curr = state.get_time()

        # update bundle according to new preplan
        while self._bundle:
            # get next bundle entry
            _,tasks = self._bundle.pop(0)

            for task,n_obs in tasks.items():
                # get existing bid for this task and observation number
                bid_to_reset : Bid = self._results[task][n_obs]

                # reset bid to empty bid
                bid_to_reset.reset(t_curr)

                # add empty bid to results
                self._results[task][n_obs] = bid_to_reset

                # add existing bid to list of bundle updates
                bundle_resets.append(bid_to_reset.copy())

        # reset path
        self._path = []

        # add an element to rests to trigger replanning
        if not bundle_resets:
            bundle_resets.append('Bid') # dummy bid to trigger replanning

        # return updates
        return preplan_path, bundle_resets

    def __process_default_tasks(self, state: SimulationAgentState, tasks: List[GenericObservationTask]) -> List[Bid]:
        """ Processes new default mission tasks and updates results accordingly. """
        # initialize list of newly added bids from new tasks
        new_task_added = []

        # identify new default tasks
        unknown_tasks = [task for task in tasks 
                         if isinstance(task, DefaultMissionTask)
                         and task not in self._results]
        
        # process each default task
        for task in unknown_tasks:
            # initialize results for new default tasks
            self._results[task].append(Bid.make_empty_bid(task, state.agent_name, 0))

            # initialize optimistic bidding counter for new task
            self._optimistic_bidding_counters[task].append(self._optimistic_bidding_threshold)

            # add task to known tasks
            self._id_to_tasks[task.id] = task

            # create empty bid for new task and add to list of changes
            new_task_added.append(self._results[task][-1].to_dict())

        # return list of new task bids added to results
        return new_task_added    

    def __process_incoming_task_requests(self, 
                                       state: SimulationAgentState, 
                                       incoming_reqs : List[TaskRequest],
                                       incoming_bids : List[dict]
                                       ) -> List[Bid]:
        """ Processes new urgent task requests and updates results accordingly. """
        
        # initialize list of newly added bids from new tasks
        new_task_added = []

        # get new and active incoming tasks from requests
        active_req_tasks = set([req.task for req in incoming_reqs 
                                # check if task is still available 
                                if req.task.is_available(state.get_time())
                                # and not already in results
                                and req.task not in self._results])
        
        # extract tasks from incoming bids
        ## find unique tasks in incoming bids
        unique_bid_tasks = list({self.__task_key(bid['task']): bid['task']
                                for bid in incoming_bids}.values())

        ## unpack unique bid tasks
        incoming_bid_tasks = set([GenericObservationTask.from_dict(task_dict) 
                                  for task_dict in unique_bid_tasks
                                  if task_dict['id'] not in self._id_to_tasks])
        ## filter active bid tasks
        active_bid_tasks = set([task for task in incoming_bid_tasks
                                # check if task is still available
                                if task.is_available(state.get_time())
                                #  and not already in results
                                and task not in self._results])
        
        # merge active tasks from requests and bids
        active_tasks = active_req_tasks.union(active_bid_tasks)

        # update urgent tasks
        self._known_event_tasks.update(active_tasks)
        self._incoming_event_tasks.extend(active_tasks)
                
        # remove unavailable tasks from known task lists
        # if any([not task.is_available(state.t) for task in self.known_event_tasks]):
        #     raise NotImplementedError("Removal of unavailable urgent tasks not yet implemented.")
        # self.known_event_tasks = {task for task in self.known_event_tasks if task.is_available(state.t)}
        
        # identify new urgent tasks
        # new_event_tasks = [task for task in self.incoming_event_tasks 
        #                    if task not in self.results]

        # check if new tasks exceed threshold
        if len(self._incoming_event_tasks) < self._replan_threshold: 
            return new_task_added # threshold not met; skip processing and allow for more tasks to accumulate

        # DEBUG PRINTOUTS --------
        # if self._debug and self.incoming_event_tasks:
        #     out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\nReceived incoming urgent tasks:\n'
        #-------------------------

        # threshold met; process new tasks
        for task in self._incoming_event_tasks:
            # check if task is already in results
            if task in self._results: continue # already processed; skip

            # initialize results for new event tasks
            self._results[task].append(Bid.make_empty_bid(task, state.agent_name, 0))

            # initialize optimistic bidding counter for new task
            self._optimistic_bidding_counters[task].append(self._optimistic_bidding_threshold)

            # add task to known tasks
            self._id_to_tasks[task.id] = task

            # # create empty bid for new task and add to list of changes
            # new_task_added.append(Bid(task, state.agent_name, t_bid=state.get_time()))

            # add new task to list of changes
            new_task_added.append(task.to_dict())

        # DEBUG PRINTOUTS --------
        #     if self._debug:
        #         out += f'\t - {repr(task)}\n'

        # if self._debug and new_event_tasks:
        #     print(out)
        # -------------------------

        # return list of new task bids added to results
        return new_task_added
    
    def __remove_expired_tasks(self, state : SimulationAgentState) -> Tuple[list, list, list]:
        """ Remove expired tasks from results. """
        # get current time 
        t_curr = state.get_time()

        # identify expired tasks
        expired_tasks = [task for task in self._results 
                         if not task.is_available(t_curr)]

        # initialize list of removed bids
        expired_bids = []
        
        # remove expired tasks from results and check if any expired task exist in bundle
        bundle_idx_to_remove = None
        for task in expired_tasks:
            # remove task from results
            bids_removed = self._results.pop(task)
           
            # remove optimistic bidding counters
            self._optimistic_bidding_counters.pop(task, None)

            # remove task from known event tasks
            self._id_to_tasks.pop(task.id, None)

            # add removed bids to list
            expired_bids.extend(bids_removed)

            # check if expired task was in the bundle
            for bundle_idx,(_,tasks) in enumerate(self._bundle):
                if (task in tasks 
                    and (bundle_idx_to_remove is None
                         or bundle_idx < bundle_idx_to_remove)):
                    bundle_idx_to_remove = bundle_idx
                    break
            
        # check if no expired task exists in bundle
        if bundle_idx_to_remove is None:
            # return expired bids and do not modify bundle
            return expired_bids, self._bundle, []

        # TODO ensure that the following section is not needed. Bids should not 
        #   be placed for tasks that will expire before they are performed.
        # raise NotImplementedError("Removal of expired tasks from bundle not yet implemented.")

        # initialize list of bundle updates
        bundle_updates = []

        # split bundle at first updated task
        revised_bundle = self._bundle[:bundle_idx_to_remove]

        # iterate through remaining bundle to update bids
        for _, obs_tasks in self._bundle[bundle_idx_to_remove:]:
            # reset subsequent bids for all tasks in bundle if the bidder is still listed as the winner
            for task, n_obs in obs_tasks.items():
                # initiate list of bids being reset for this task
                bids_reset = []

                # reset invalid bid along with all subsequent bids  
                for bid_idx in range(n_obs, len(self._results[task])):
                    
                    # check if this agent is still listed as the winning bidder
                    if not self._results[task][bid_idx].is_bidder_winning():
                        continue # another agent is winning this bid; skip
                    elif self._results[task][bid_idx].was_performed():
                        continue # bid was already performed by this agent; do not reset

                    # get bid to reset and remove from results
                    bid_to_reset : Bid = self._results[task][bid_idx]

                    # reset bid
                    bid_to_reset.reset(t_curr)

                    # update results
                    self._results[task][bid_idx] = bid_to_reset

                    # add to list of resets
                    bids_reset.append(bid_to_reset)

                # add to violations list
                bundle_updates.append(bids_reset)

        # return list of removed bids
        return expired_bids, revised_bundle, bundle_updates
    
    def __update_performed_bundle_observations(self, 
                                               state : SimulationAgentState, 
                                               performed_observations : List[ObservationAction]
                                            ) -> Tuple[list, List[Bid]]:
        """ Checks if planned observations were performed by parent agent and updates results accordingly. """
        
        # initialize list of bundle updates
        bundle_updates : List[Tuple[ObservationOpportunity, List[Bid]]] = []

        # initialize list of performed tasks to remove from bundle
        performed_task_bids = []

        # collect actions in bundle past their imaging time
        observed_opportunities : list[ObservationOpportunity] = [obs_action.obs_opp 
                                                                 for obs_action in performed_observations]

        performed_bundle_tasks = [ (obs_opp, obs_tasks) 
                                    for obs_opp, obs_tasks in self._bundle
                                    if obs_opp in observed_opportunities
                                    # if any([obs_opp == performed_obs for performed_obs in observed_opportunities])
                                    ]

        # iterate through performed bundle to mark bids as performed
        for obs_opp, obs_tasks in performed_bundle_tasks:     
                       
            # imaging time has passed for task bids; assume tasks were performed by parent agent
            assert any([self._results[task][n_obs].winner == state.agent_name for task,n_obs in obs_tasks.items()]), \
                "Cannot mark tasks as performed if this agent is not the winning bidder."
            
            # mark bids as performed
            performed_bids = []
            for task, n_obs in obs_tasks.items():
                bid_to_perform : Bid = self._results[task][n_obs]

                # mark bid as performed
                bid_to_perform.set_performed(state.get_time(), performed=True)

                # update results
                self._results[task][n_obs] = bid_to_perform

                # add to list of performed bids
                performed_bids.append(bid_to_perform.copy())
            
            # add bids to list of bundle updates
            bundle_updates.append((obs_opp, performed_bids))

            # add tasks to list of performed tasks
            performed_task_bids.append((obs_opp, obs_tasks))
               
        # create revised bundle considering newly performed tasks
        revised_bundle = [entry for entry in self._bundle 
                          if entry not in performed_task_bids] \
                            if performed_task_bids else self._bundle
        
        # create revised path considering newly performed tasks
        performed_bundle_obs = [obs_opp for obs_opp,_ in performed_bundle_tasks]
        revised_path = [obs_action for obs_action in self._path
                        if obs_action.obs_opp not in performed_bundle_obs]
        
        # collect rewards for performed bids and add to rewards sink
        for _,performed_bids in bundle_updates:
            for performed_bid in performed_bids:
                performed_bid : Bid
                # calculate reward for this bid

                # compile reward information for this bid 
                reward_dict = {
                    'task_id' : performed_bid.task.id,
                    'n_obs' : performed_bid.n_obs,
                    't_img' : performed_bid.t_img,
                    't_bid' : performed_bid.t_bid,
                    'agent_name' : performed_bid.owner,
                    'planned reward' : performed_bid.owner_bid,
                    # 'performed reward' : performed_bid.winning_bid
                }

                # add to rewards sink
                self._observation_rewards.append(reward_dict)

        # ensure elements in path are yet to be performed
        assert all(obs_action.t_start >= state.get_time() for obs_action in revised_path), \
            "Revised path contains observation actions that have already been performed."
                
        # return revised bundle and list of performed bids
        return revised_bundle, revised_path, bundle_updates

    def __compare_incoming_bids(self,
                       state : SimulationAgentState,
                       incoming_bids : List[Union[Bid, dict]]
                       ) -> Tuple[List[Bid], List[Bid]]:
        """
        Compares incoming bids with existing results and updates results and bundle accordingly. 
         - If incoming bid corresponds to an existing observation number for a task, compare bids and update results with winning bid.
         - If there is a conflict between an incoming bid and an existing bid for the same observation (i.e. both bids were performed), update results with winning bid and add losing bid to list of incoming bids to be processed.
         - If any bid was updated in results, check if corresponding bundle entry needs to be updated.
         - If any bundle entry was updated, check if corresponding path entry needs to be updated.
         - Repeat until no more updates are needed.
        """
        # group bids by bidding agent 
        grouped_bids : Dict[str, List[dict]] = self.__group_incoming_bids(incoming_bids)

        # initialize list of updates done to results
        results_updates = []   

        # keep track of tasks receiving incoming bids
        tasks_from_incoming_bids : Dict[Tuple, GenericObservationTask] = dict()

        # iterate through grouped bids and compare with existing results
        for task_key,incoming_task_bids in grouped_bids.items():
            
            # get task object for this task key from incoming bids
            task = tasks_from_incoming_bids.get(task_key)
            
            # check if a matching task was found
            if task is None:
                # no matching task found; check if task is known in results
                task_dict = incoming_task_bids[0]["task"]
                task_id = task_dict["id"]

                task = self._id_to_tasks.get(task_id)

                if task is None:
                    # task is not known in results; reconstruct task from bids
                    task = GenericObservationTask.from_dict(task_dict)
                    task_is_known = self._results.get(task) is not None
                else:
                    # task is known in results; use existing task object
                    task_is_known = True

                # assign task to task key for future reference in this round
                tasks_from_incoming_bids[task_key] = task
            else:
                # task was found for this task key; use existing task object
                task_is_known = True
            
            # get current results
            current_task_bids : List[Bid] = self._results[task] if task_is_known else []
            current_bidding_counters : List[int] = self._optimistic_bidding_counters[task] if task_is_known else []
            
            # initialize queue of incoming bids for processing
            q = deque(incoming_task_bids)

            # process incoming bids
            while q:
                # get next incoming bid
                incoming_bid : dict = q.popleft()

                # convert Bid object to dictionary if needed
                if isinstance(incoming_bid, Bid):
                    incoming_bid = incoming_bid.to_dict()
                    
                # check if incoming bid observation number exceeds existing bids for this task
                n_obs = incoming_bid['n_obs']
                n_existing_bids = len(current_task_bids)

                if n_obs >= n_existing_bids:
                    # incoming bid observation number exceeds existing bids; 
                    #  initialize missing elements in results with empty bids
                    current_task_bids.extend([
                        Bid.make_empty_bid(task, state.agent_name, n_obs)
                        for n_obs in range(n_existing_bids, n_obs + 1)
                    ])

                    # initialize optimistic bidding counter for new bids
                    current_bidding_counters.extend([
                        self._optimistic_bidding_threshold 
                        for _ in range(n_existing_bids, n_obs + 1)
                    ])

                # get current bid for this task and observation number
                current_bid : Bid = current_task_bids[n_obs]
                
                # compare incoming bid with existing bids for the same task
                updated_bid : Bid = current_bid.update(incoming_bid, state.get_time())

                # update results with modified bid
                current_task_bids[n_obs] = updated_bid

                # check if current bid was modified
                current_was_updated = updated_bid.has_different_winner_values(current_bid)

                # check if bid, winner, or observation time were modified
                if current_was_updated: 
                    # add updated bid to results updates
                    results_updates.append(updated_bid)
                
                # check if previously unperformed bid was performed 
                elif not current_bid.was_performed() and incoming_bid['performed']:
                    # bid was performed; add updated bid to results updates
                    results_updates.append(updated_bid)                 
                
                # check if both bids corresponded to a performed observation
                if current_bid.was_performed() and incoming_bid['performed']:
                    # both bids were performed; check which bid was the one that won the comparison
                    if current_was_updated:
                        # current bid lost
                        loser_bid : Bid = current_bid 
                    elif updated_bid.has_different_winner_values(incoming_bid):
                        # incoming bid lost
                        loser_bid : Bid = incoming_bid
                    else:
                        # both bids are identical; skip
                        continue

                    # --- if reached here there was a bid performed conflict ---
                    #        both bids need to be reflected in results
                    
                    # reconstruct losing bid from dictionary if needed 
                    if isinstance(loser_bid, dict):
                        loser_bid = Bid.from_dict(loser_bid)

                    # modify losing bid to reflect updated observation number 
                    loser_bid.n_obs += 1

                    # check if new bid observation number exceeds existing bids for this task
                    if loser_bid.n_obs >= len(current_task_bids):
                        # add empty bid to results for new observation number
                        current_task_bids.append(
                            Bid(loser_bid.task, state.agent_name, loser_bid.n_obs)
                        )

                        # initialize optimistic bidding counter for new bid
                        current_bidding_counters.append(
                            self._optimistic_bidding_threshold
                        )

                    # add updated bid to list of bids to be processed
                    # bids.append(loser_bid)
                    q.append(loser_bid)

        # -------------------------------
        # DEBUG PRINTOUTS
        # for task, bids in self.results.items():
        #     for bid in bids:
        #         if not bid.has_winner():
        #             x=1 # debug breakpoint    
        
        # if results_updates and self._debug:
        #     self._log_results('CONSENSUS PHASE - RESULTS (AFTER COMPARISON)', state, self._results)
        #     x = 1 # debug breakpoint
        # -------------------------------

        # clear grouped bids after processing
        grouped_bids.clear()

        # return result changes
        return results_updates
    
    def __group_incoming_bids(self,
                              incoming_bids : List[Bid]
                              ) -> Dict[str,List[dict]]:
        """
        Groups incoming bids by task_key.
        """
        # initialize bid grouping: task_key -> list[bid_dict]
        grouped_bids = defaultdict(list)

        # set local bindings 
        task_key_fn = self.__task_key

        # Pass 1: group bids by task key
        for bid in incoming_bids:
            # get task dictionary from bid
            try:
                task = bid["task"]
            except (KeyError, TypeError):
                task = bid.task.to_dict() if isinstance(bid, Bid) else None
            
            if task is None: raise ValueError("Bid does not contain a valid task field.")
            
            # group bid by task key
            tk = task_key_fn(task)
            grouped_bids[tk].append(bid)

        # TODO is this needed? Constraints are checked after all updates are performed, 
        #       so the order of processing incoming bids should not matter.
        # # Pass 2: sort bids for each task key by observation number and owner name
        # for tk, bids in grouped_bids.items():
        #     # sort by bid observation number and owner name
        #     bids.sort(key=lambda bid: 
        #                     (bid['n_obs'], bid['owner']) if isinstance(bid, dict) 
        #                         else (bid.n_obs, bid.owner))

        # return grouped bids
        return grouped_bids
         
    def __update_bundle_from_results(self,
                                     state : SimulationAgentState
                                    ) -> Tuple[list, List[List[Bid]]]:
        """ Update bundle according to latest results. """
                
        # initialize list of bundle updates
        bundle_updates = []

        # count initial bundle size
        init_bundle_size = len(self._bundle)

        # match observation times from path to bundle entries
        t_img_bundle = [np.NAN for _ in self._bundle]
        for idx,(obs_opp, obs_tasks) in enumerate(self._bundle):
            # find matching observation action in path
            matching_actions = [obs_action.t_start for obs_action in self._path
                                if obs_action.obs_opp == obs_opp]
            assert len(matching_actions) == 1, \
                "Each bundle observation opportunity must have a matching observation action in the path."
            t_img_bundle[idx] = matching_actions[0]

        # search for results updates for any tasks in the bundle
        min_updated_idx = min([ idx 
                                # set index `idx` to be removed from bundle if...
                                for idx,(_,obs_tasks) in enumerate(self._bundle)
                                # for an a given bundle entry...
                                if any( 
                                        # 1) there is not a bid in results for this task and observation number
                                        len(self._results[task]) <= n_obs
                                        # 2) or this agent is no longer the winning bidder
                                        or not self._results[task][n_obs].is_bidder_winning() 
                                        # 3) or the imaging time does not match that in the path
                                        or abs(self._results[task][n_obs].t_img - t_img_bundle[idx]) > self.EPS
                                        # 4) or the bid was performed
                                        or self._results[task][n_obs].was_performed()
                                    for task, n_obs in obs_tasks.items())
                                ], 
                                # if no updates found, set to None
                                default=None)
        
        # check if any updates were found
        if min_updated_idx is None:
            # no updates to bids in bundle; return original bundle
            return self._bundle, self._path, bundle_updates

        # split bundle at first updated task
        revised_bundle = self._bundle[:min_updated_idx]

        # get current time
        t_curr = state.get_time()

        # iterate through remaining bundle to update bids
        for _, obs_tasks in self._bundle[min_updated_idx:]:
            # reset subsequent bids for all tasks in bundle if the bidder is still listed as the winner
            for task, n_obs in obs_tasks.items():
                # initiate list of bids being reset for this task
                bids_reset = []

                # reset invalid bid along with all subsequent bids  
                for bid_idx in range(n_obs, len(self._results[task])):
                    
                    # check if this agent is still listed as the winning bidder
                    if not self._results[task][bid_idx].is_bidder_winning():
                        continue # another agent is winning this bid; skip
                    elif self._results[task][bid_idx].was_performed():
                        continue # bid was already performed by this agent; do not reset

                    # get bid to reset and remove from results
                    bid_to_reset : Bid = self._results[task][bid_idx]

                    # reset bid
                    bid_to_reset.reset(t_curr)

                    # update results
                    self._results[task][bid_idx] = bid_to_reset

                    # add to list of resets
                    bids_reset.append(bid_to_reset)

                # add to violations list
                bundle_updates.append(bids_reset)

        assert len(revised_bundle) + len(self._bundle[min_updated_idx:]) == init_bundle_size, \
            "Revised bundle size does not match initial bundle size."
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # for task, bids in self.results.items():
        #     for bid in bids:
        #         if not bid.has_winner():
        #             x=1 # debug breakpoint    
        
        # if bundle_updates and self._debug:
        #     self._log_results('CONSENSUS PHASE - RESULTS (AFTER BUNDLE UPDATE)', state, self.results)
        #     x = 1 # debug breakpoint
        # -------------------------------

        # ensure number of bids match bundle entries
        ## get bids won by this agent from results
        winning_bids = [bid for bids in self._results.values()
                        for bid in bids if bid.winner == state.agent_name
                        and not bid.was_performed()]
        bids_in_bundle = [self._results[task][n_obs] 
                            for _,obs_tasks in revised_bundle
                            for task,n_obs in obs_tasks.items()]
        
        ## count and compare with bundle size
        total_won_bids = len(winning_bids)
        total_bundle_entries = len(bids_in_bundle)
        
        # DEBUG PRINTOUTS --------
        # if self._debug and total_bundle_entries != total_won_bids:
        #     print(f'ERROR: Mismatch between winning bids ({total_won_bids}) and bundle entries ({total_bundle_entries}):')
        #     self._log_results('CONSENSUS PHASE - RESULTS (INVALID)', state, self.results)
        #     self._log_bundle('CONSENSUS PHASE - BUNDLE (INVALID)', state, revised_bundle)        

        #     if total_bundle_entries < total_won_bids:                
        #         missing_bids : Set[Bid] = set(winning_bids) - set(bids_in_bundle)
        #         out = f'Bids in results but not in bundle ({len(missing_bids)}):\n'
        #     else:
        #         missing_bids : Set[Bid] = set(bids_in_bundle) - set(winning_bids)
        #         out = f'Bids in bundle but not in results ({len(missing_bids)}):\n'

        #     for bid in missing_bids:
        #         out += f' - {repr(bid)}\n'
        #     print(out)
        #-------------------------

        assert total_won_bids == total_bundle_entries, \
            "Number of winning bids does not match number of bundle entries."
        
        ## check that every bundle entry corresponds to a winning bid
        for _, obs_tasks in revised_bundle:
            for task, n_obs in obs_tasks.items():
                bid : Bid = self._results[task][n_obs]
                assert bid.winner == state.agent_name, \
                    "Bundle entry does not correspond to a winning bid."

        # collect list of observation opportunities in the revised bundle 
        obs_opps_in_revised_bundle = {obs_opp for obs_opp,_ in revised_bundle}
        
        # revise path according to revised bundle
        revised_path = [obs_action for obs_action in self._path
                        if obs_action.obs_opp in obs_opps_in_revised_bundle]

        # return updated bundle and list of updates
        return revised_bundle, revised_path, bundle_updates
    
    def __check_results_constraints(self, state : SimulationAgentState) -> List[Bid]:
        """ Check results for constraint violations and return list of affected bids. """
        # get current time 
        t_curr = state.get_time()
        
        # initiate list of constraint violations
        bids_in_violation = []

        # check every task for constraint violations
        for bids in self._results.values():            
            # ensure the index of every bid matches their observation number
            for i in range(len(bids)):
                if bids[i].n_obs != i:
                    raise AssertionError("Results bids are not sorted by observation number.")
            
            if len(bids) <= 1: continue # no observation sequence to check for constraints
            
            # initialize search for constraint violations
            invalid_bid_idx : int = None
            
            # check every bid for this task
            for n_obs_idx, bid in enumerate(bids[1:], start=1):
                # get previous bid to compare constraints with
                prev_bid : Bid = bids[n_obs_idx - 1]

                # Constraint 0: Previous bid must be assigned to a winner if the bid has a winner
                if bid.has_winner() and not prev_bid.has_winner():
                    invalid_bid_idx = n_obs_idx
                    break # stop searching for constraint violations for this task

                # Constraint 1: Observation number must be consecutive
                if prev_bid.n_obs + 1 != bid.n_obs:
                    invalid_bid_idx = n_obs_idx
                    break # stop searching for constraint violations for this task

                # Constraint 2: If assigned, imaging time must be after previous imaging time
                if bid.has_winner() and prev_bid.t_img > bid.t_img:
                    invalid_bid_idx = n_obs_idx
                    break # stop searching for constraint violations for this task         
            
            # check if invalid bid was found
            if invalid_bid_idx is None: continue # no violations for this task; continue to next task

            # reset invalid bid along with all subsequent bids
            for bid_idx in range(invalid_bid_idx, len(bids)):
                # get bid to reset 
                bid_to_reset : Bid = bids[bid_idx]

                # reset bid
                bid_to_reset.reset(t_curr)

                # add to violations list
                bids_in_violation.append(bid_to_reset.to_dict())                

        # return list of bids in violation
        return bids_in_violation   
    
    def needs_planning(self, *_) -> bool:
        # -------------------------------
        # # DEBUG BREAKPOINTS
        # if self._task_announcements_received:
        #     x = 1 # breakpoint
        # if self._results_changes_performed:
        #     x = 1  # breakpoint
        # if self._bundle_changes_performed:
        #     x = 1  # breakpoint
        # -------------------------------

        # trigger replan if either...
        return (   
                self._task_announcements_received    # 0) new tasks were announced                 
                or self._results_changes_performed   # 1) there were relevant updates to bids/results
                or self._bundle_changes_performed    # 2) incoming bids modified the bundle
                )

    """
    ---------------------------
    BUNDLE-BUILDING PHASE
    ---------------------------
    """
    
    def generate_plan(self, 
                      state : SimulationAgentState,
                      specs : object,
                      current_plan : Plan,
                      orbitdata : OrbitData,
                      mission : Mission,
                      tasks : List[GenericObservationTask],
                      observation_history : LatestObservationTracker,
                    ) -> Plan:  
        """ Generate new plan according to consensus replanning model. """             
        try:
            # check if agent is capable of scheduling observations
            if isinstance(state, SatelliteAgentState):
                # satellite agents can schedule maneuvers and observations;
                #  generate new bundle and path according to replanning model
                self._bundle, self._path, new_bids = \
                    self.__replan_observations(state, specs, current_plan, orbitdata, mission, tasks, observation_history)
            
                # generate maneuver and travel actions from observations
                maneuvers : list = self._schedule_maneuvers(state, specs, self._path, orbitdata)
            
            elif isinstance(state, GroundOperatorAgentState):
                # ground operator does not schedule observations or bids
                new_bids : list = []

                # ground operator agents do not schedule maneuvers
                maneuvers : list = []

            else:
                # other agent types not supported
                raise NotImplementedError("Consensus planner only implemented for satellite and ground station agents.")

            # schedule broadcasts
            broadcasts : list = self._schedule_broadcasts(state, orbitdata, new_bids)

            # determine next planning time        
            t_next = self._preplan.t_next

            # schedule periodic replan
            preplan_waits : list = self._schedule_periodic_replan(state, t_next)
            
            # compile and generate plan
            self._plan = ReactivePlan(maneuvers, self._path, broadcasts, preplan_waits, t=state.get_time(), t_next=t_next)

            # return final plan
            return self._plan.copy()
        
        finally:
            # reset replanning flags
            self._task_announcements_received = False
            self._results_changes_performed = False
            self._bundle_changes_performed = False

            # reset new event task inbox
            self._incoming_event_tasks = list()

    def __replan_observations(self,
                              state : SimulationAgentState,
                              specs : object,
                              current_plan : Plan,
                              orbitdata : OrbitData,
                              mission : Mission,
                              tasks : List[GenericObservationTask],
                              observation_history : LatestObservationTracker,
                            ) -> tuple:

        # DEBUG PRINTOUTS----------------
        # if self._debug:
        #     self._log_results('PLANNING PHASE - RESULTS (BEFORE)', state, self.results)
        #     self._log_bundle('PLANNING PHASE - BUNDLE (BEFORE)', state, self.bundle)
        #     x = 1 # breakpoint
        # -------------------------------

        # check if relevant changes were made to bundle or tasks
        if not self._task_announcements_received and not self._bundle_changes_performed:
            # check if observation path is still valid given current time and results
            if self.is_observation_path_valid(state, self._path, None, None, specs):
                # no changes to tasks or bundle; do not replan
                return self._bundle, self._path, {}            
            else:
                # path is no longer valid; should not be reachable
                raise AssertionError("Observation path is no longer valid but no relevant changes to bundle were detected.")
    
        # check if a new periodic plan was generated by parent agent
        if isinstance(current_plan, PeriodicPlan) and abs(state.get_time() - current_plan.t) <= self.EPS:
            # ensure current bundle and path were reset during consensus phase
            assert len(self._bundle) == 0, "Current bundle not empty during preplan-based bundle building."
            assert len(self._path) == 0, "Current path not empty during preplan-based bundle building."        
            
            # build initial bundle from periodic preplan
            self._bundle, self._path, new_bids = \
                self._build_bundle_from_preplan(state, specs, current_plan, 
                                                orbitdata, mission, observation_history)
            # TODO ensure bids that all observations that were able to be added to bundle match observations in preplan
           
            # check if new path and bundle are valid
            self.__validate_new_bundle(state, specs, new_bids)

            # update results with initial periodic plan bundle
            self.__update_results_from_bundle(state, new_bids)

             # -------------------------------
            # DEBUG PRINTOUTS
            if self._debug and new_bids:
            # if new_bids:
                self._log_results('PLANNING PHASE - RESULTS (AFTER LOADING PREPLAN)', state, self._results)
                self._log_bundle('PLANNING PHASE - BUNDLE (AFTER LOADING PREPLAN)', state, self._bundle)
                print(f'`{state.agent_name}` - New bundle built with {len(new_bids)} new entries ({len(self._bundle)} total) and {len(self._path)} scheduled observations.')
                x = 1 # breakpoint
            # -------------------------------

        # update bundle and path according to replanning model
        self._bundle, self._path, new_bids = \
            self._bundle_building_phase(state, specs, current_plan, tasks, 
                                        orbitdata, mission, observation_history)
        
        # check if new path and bundle are valid
        self.__validate_new_bundle(state, specs, new_bids)

        # update results
        self.__update_results_from_bundle(state, new_bids)

        # -------------------------------
        # DEBUG PRINTOUTS
        if self._debug and new_bids:
        # if new_bids:
            self._log_results('PLANNING PHASE - RESULTS (AFTER)', state, self._results)
            self._log_bundle('PLANNING PHASE - BUNDLE (AFTER)', state, self._bundle)
            print(f'`{state.agent_name}` - New bundle built with {len(new_bids)} new entries ({len(self._bundle)} total) and {len(self._path)} scheduled observations.')
            x = 1 # breakpoint
        # -------------------------------

        return self._bundle, self._path, new_bids
    
    def __validate_new_bundle(self, state : SimulationAgentState, specs : object, new_bids : dict) -> None:
        """ check if new path is valid """
        assert len(self._bundle) == len(self._path), \
            "New bundle and path lengths do not match."
        assert all([obs_action.t_start >= state.get_time() for obs_action in self._path]), \
            "New observation path contains actions scheduled in the past."        
        assert self.is_observation_path_valid(state, self._path, None, None, specs), \
            "New observation path is not valid."   
        if self._debug: assert all(bid.t_bid <= state.get_time() for bids in new_bids.values() for bid in bids.values()), \
            "New bids must be assigned the correct bid time."

        # ensure every task in the path has a matching bundle entry
        if self._debug:
            for obs_action in self._path:
                matching_bundle_entries = [obs_tasks 
                                            for obs_opp,obs_tasks in self._bundle
                                            if obs_opp == obs_action.obs_opp]
                assert len(matching_bundle_entries) == 1, \
                    "Every observation action in the path must have a matching bundle entry."
 

    @abstractmethod
    def _build_bundle_from_preplan(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_plan : Plan,
                                    orbitdata : OrbitData,
                                    mission : Mission,
                                    observation_history : LatestObservationTracker
                                    ) -> tuple:    
        """ Build bundle from latest periodic preplan. """

    @abstractmethod
    def _bundle_building_phase(self,
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        tasks : List[GenericObservationTask],
                        orbitdata : OrbitData,
                        mission : Mission,
                        observation_history : LatestObservationTracker
                        ) -> tuple:        
        """ 
        Build bundle according to selected replanning model. 
        #### Returns:
            - `new_bundle` : List[List[Tuple[GenericObservationTask, int]]] -- New bundle of bids
            - `new_path` : List[GenericObservationTask] -- New observation path
            - `new_bids` : List[Bid] -- New bids generated during bundle building
        
        """
    
    def __update_results_from_bundle(self, 
                                     state : SimulationAgentState,
                                     new_bids : Dict[GenericObservationTask, Dict[int, Bid]]
                                    ) -> None:
        """ Update results dictionary from new bundle. """
        
        # ==========================================================
        #  UPDATE RESULTS WITH NEW BIDS
        # ==========================================================
        
        # update results from new bids from new bundle 
        for _,obs_tasks in self._bundle:
            for task, n_obs in obs_tasks.items():
                # get new bid for this task and observation number
                bid : Bid = new_bids[task][n_obs]

                try:
                    # get previous bid
                    previous_bid : Bid = self._results[task][n_obs]

                    # reduce any optimistic bidding counters if needed  
                    if (previous_bid > bid                          # lower bid value
                        and previous_bid.t_img > bid.t_img          # earlier imaging time
                        and previous_bid.winner != state.agent_name # was not winning bid
                        and bid.winner == state.agent_name          # is now winning bid
                        ):
                        # bid was worsened; reduce optimistic bidding counter
                        self._optimistic_bidding_counters[task][n_obs] -= 1

                        # ensure bid counters are possitive
                        assert self._optimistic_bidding_counters[task][n_obs] >= 0, \
                            "Bundle-builder attempted optimistic bid for task with optimistic bid attempt counter already at zero."

                    # update existing bid
                    self._results[task][n_obs] = bid

                except IndexError:
                    # count bids within results for this task
                    n_existing_bids = len(self._results[task])                    
                    
                    # append missing bids as empty bids with no winner
                    for i_obs in range(n_existing_bids, n_obs):
                        empty_bid = Bid.make_empty_bid(task, state.agent_name, i_obs)
                        self._results[task].append(empty_bid)
                        self._optimistic_bidding_counters[task].append(self._optimistic_bidding_threshold)

                    # bid for new observation number; add to results
                    self._results[task].append(bid)

                    # add optimistic bidding counter for new bid
                    self._optimistic_bidding_counters[task].append(self._optimistic_bidding_threshold)

        # collect bids that have been removed from the bundle
        bids_in_bundle = { (task, n_obs)
                            for _,obs_tasks in self._bundle
                            for task,n_obs in obs_tasks.items()}
        winning_bids = { (task, n_obs)
                            for task, bids in self._results.items()
                            for n_obs,bid in enumerate(bids)
                            if bid.winner == state.agent_name
                            and not bid.was_performed()}
        bids_removed_from_bundle = winning_bids - bids_in_bundle

        # reset bids for tasks that are no longer in the bundle
        for task, n_obs in bids_removed_from_bundle:
            # get bid to reset
            bid_to_reset : Bid = self._results[task][n_obs]

            # reset bid
            bid_to_reset.reset(state.get_time())

            # update results
            self._results[task][n_obs] = bid_to_reset

        # ==========================================================
        #  ENSURE RESULTS CONSISTENCY
        # ==========================================================

        # ensure number of bids match bundle entries
        ## get bids won by this agent from results
        winning_bids = [bid for bids in self._results.values()
                        for bid in bids if bid.winner == state.agent_name
                        and not bid.was_performed()]
        bids_in_bundle = [self._results[task][n_obs] 
                            for _,obs_tasks in self._bundle 
                            for task,n_obs in obs_tasks.items()]
        
        ## count and compare with bundle size
        total_won_bids = len(winning_bids)
        total_bundle_entries = len(bids_in_bundle)
        
        # # DEBUG PRINTOUTS --------
        # if self._debug and total_bundle_entries != total_won_bids:
        #     print(f'ERROR: Mismatch between winning bids ({total_won_bids}) and bundle entries ({total_bundle_entries}):')
        #     self._log_results('CONSENSUS PHASE - RESULTS (INVALID)', state, self.results)
        #     self._log_bundle('CONSENSUS PHASE - BUNDLE (INVALID)', state, self.bundle)        

        #     if total_bundle_entries < total_won_bids:                
        #         missing_bids : Set[Bid] = set(winning_bids) - set(bids_in_bundle)
        #         out = f'Bids in results but not in bundle ({len(missing_bids)}):\n'
        #     else:
        #         missing_bids : Set[Bid] = set(bids_in_bundle) - set(winning_bids)
        #         out = f'Bids in bundle but not in results ({len(missing_bids)}):\n'

        #     for bid in missing_bids:
        #         out += f' - {repr(bid)}\n'
        #     print(out)
        # #-------------------------

        assert total_won_bids == total_bundle_entries, \
            "Number of winning bids does not match number of bundle entries."
        
        # ensure all bundle bids match results
        for _, obs_tasks in self._bundle:
            for task, n_obs in obs_tasks.items():
                assert task in self._results, \
                    "Bundle task not found in results."
                assert n_obs < len(self._results[task]), \
                    "Bundle observation number exceeds number of bids in results for task."

                assert task in self._optimistic_bidding_counters, \
                    "Bundle task not found in optimistic bidding counters."
                assert n_obs < len(self._optimistic_bidding_counters[task]), \
                    "Bundle observation number exceeds number of optimistic bidding counters for task."
                
                # get matching bid from results
                bid : Bid = self._results[task][n_obs]
                
                # check that every bundle entry corresponds to a winning bid
                assert bid.winner == state.agent_name, \
                    "Bundle entry does not correspond to a winning bid."

        # ensure all bundle bids meet requirements; 
        #   assumes bids outside the bundle will be dealt with durin consensus-phase result updates
        for _, obs_tasks in self._bundle:    
            for task, n_obs in obs_tasks.items():
                bid : Bid = self._results[task][n_obs]

                # check if there is a previous bid to compare with
                if n_obs == 0:
                    # there is no previous bid to compare with; define independent constraints
                    constraints : List[bool] = [
                        # Constraint 0: Current bid must be assigned to a winner
                        bid.has_winner(),
                        # Constraint 1: Observation number must match bundle entry
                        bid.n_obs == n_obs
                    ]
                else:
                    # there is a previous bid to compare with; get previous bid
                    prev_bid : Bid = self._results[task][n_obs-1]
                    
                    # define dependent constraints
                    constraints : List[bool] = [
                        # Constraint 0: Previous bid must be assigned to a winner
                        prev_bid.has_winner(),
                        # Constraint 0.5: Current bid must be assigned to a winner
                        bid.has_winner(),
                        # Constraint 1: Observation number must be consecutive
                        prev_bid.n_obs + 1 == bid.n_obs,
                        # Constraint 2: Imaging time must be after previous imaging time
                        (prev_bid.t_img <= bid.t_img and prev_bid.winner != state.agent_name) \
                            or (prev_bid.t_img < bid.t_img and prev_bid.winner == state.agent_name)
                    ]
                    
                # DEBUG PRINTOUTS --------
                # if self._debug and not all(constraints):
                #     print(f'ERROR: generated invalid bids during bundle-building phase:')
                #     self._log_results('INVALID GENERATED BIDS', state, self.results)
                #     x = 1 # breakpoint
                #-------------------------

                # check if any constraint is violated
                assert all(constraints), \
                    "Generated bids violate constraints; cannot update results."  

    def _calculate_path_utility(self,
                                state : SimulationAgentState,
                                specs : object,
                                cross_track_fovs : Dict[str, float],
                                path : List[ObservationAction],
                                observation_history : LatestObservationTracker,
                                orbitdata : OrbitData,
                                mission : Mission,
                                n_obs : List[Dict[GenericObservationTask, int]],
                                t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> float:
        """ Calculate total expected utility of observation path. """
        
        # calculate path value
        path_value = self._calculate_path_value(specs, cross_track_fovs, path, observation_history, orbitdata, mission, n_obs, t_prev)
        
        # calculate path cost
        path_cost = self._calculate_path_cost(state, specs, path)

        # return path utility
        return path_value - path_cost

    def _calculate_path_value(self,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : LatestObservationTracker,
                              orbitdata : OrbitData,
                              mission : Mission,
                              n_obs : List[Dict[GenericObservationTask, int]],
                              t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> float:
        """ Calculate total expected value of observation path. """
        # calculate and accumulate expected value of observation
        task_values = self._calculate_path_values(specs, cross_track_fovs, path, observation_history, orbitdata, mission, n_obs, t_prev)

        # return total task value
        return sum(task_values) 

    def _calculate_path_values(self,
                              specs : object,
                              cross_track_fovs : Dict[str, float],
                              path : List[ObservationAction],
                              observation_history : LatestObservationTracker,
                              orbitdata : OrbitData,
                              mission : Mission,
                              n_obs : List[Dict[GenericObservationTask, int]],
                              t_prev : List[Dict[GenericObservationTask, float]]
                            ) -> List[float]:
        """ Calculate expected value of each observation in the path. """
        return [self.estimate_observation_opportunity_value(obs.obs_opp,
                                                 obs.t_start,
                                                 obs.obs_opp.min_duration,
                                                 specs,
                                                 cross_track_fovs,
                                                 orbitdata,
                                                 mission,
                                                 observation_history,
                                                 n_obs[obs_idx],
                                                 t_prev[obs_idx])
                        for obs_idx, obs in enumerate(path)]

    def _count_observations_and_revisit_times_from_path(self,
                                                        path : List[ObservationAction]
                                                    ) -> Tuple[List[Dict[GenericObservationTask, int]],
                                                            List[Dict[GenericObservationTask, float]]]:
        """ Calculate observation number and revisit time for tasks in the given path given the known bids. """

        # initialize observation counters and previous observation time trackers
        n_obs = [defaultdict(int) for _ in path]
        t_prev = [defaultdict(lambda: np.NINF) for _ in path]

        # get all parent tasks in the given path
        path_tasks : set[GenericObservationTask]= {task 
                                                   for obs_action in path 
                                                   for task in obs_action.obs_opp.tasks}
        
        # ---HISTORICAL DATA FROM BID RESULTS---
        # initiate observation history for all parent tasks in path
        #  only considers performed bids as historical data
        n_obs_history = {task: 0 for task in path_tasks}
        t_prev_history = {task: np.NINF for task in path_tasks}

        # iterate through previous bids to populate initial observation numbers and previous observation times
        for task in path_tasks:
            # assume parent task is part of results
            assert task in self._results, \
                "Parent task in path must be part of results to count observation numbers and revisit times."

            # get previous matching observations for this task
            peformed_bids = [bid for bid in self._results[task]
                                if bid.was_performed()]
            
            assert all(bid.n_obs == idx for idx, bid in enumerate(peformed_bids)), \
                "Results bids are not sorted by observation number."
            
            # update previous observation counts
            n_obs_history[task] += len(peformed_bids)

            # calculate latest observation time from previous bids
            t_latest = max((bid.t_img for bid in peformed_bids), default=np.NINF)
            
            # update previous observation times
            t_prev_history[task] = max(t_prev_history[task], t_latest)

        # ---PATH DATA---
        # initiate observation counter for all parent tasks in path
        n_obs_in_path = {task: 0 for task in path_tasks}
        t_prev_in_path = {task: np.NINF for task in path_tasks}

        # initiate previous observations and times along path
        for obs_idx, obs in enumerate(path):           
            for task in obs.obs_opp.tasks:
                # update overall observation number and revisit times along path using historical and path data
                n_obs[obs_idx][task] = n_obs_history[task] + n_obs_in_path[task]
                t_prev[obs_idx][task] = max(t_prev_history[task], t_prev_in_path[task])               

                # update previous path observation counts 
                n_obs_in_path[task] += 1
                t_prev_in_path[task] = max(t_prev_in_path[task], obs.t_end)

        # return observation numbers and previous observation times
        return n_obs, t_prev
    
    def _count_observations_and_revisit_times_from_results( self,
                                                            state : SimulationAgentState,
                                                            path : List[ObservationAction]
                                                        ) -> Tuple[List[Dict[GenericObservationTask, int]],
                                                            List[Dict[GenericObservationTask, float]]]:
        

        # initialize observation counters and previous observation time trackers
        n_obs = [dict() for _ in path]
        t_prev = [dict() for _ in path]

        # ---HISTORICAL DATA FROM BID RESULTS---
        # iterate through path to populate observation numbers and previous observation times
        for obs_idx, obs_act in enumerate(path):
            # get trackers for this observation action
            n_obs_i : dict = n_obs[obs_idx]
            t_prev_i : dict = t_prev[obs_idx]

            # iterate through parent tasks
            for task in obs_act.obs_opp.tasks:
                # get current bids for this task
                bids = self._results.get(task, None)

                # ensure parent task is part of results
                assert bids is not None, \
                    "Parent task in path must be part of results to count observation numbers and revisit times."

                # find matching bid and most recent previous bid for this observation task
                matching_bid : Bid = None
                prev_bid : Bid = None
                for bid in bids:
                    # check if a matching bid was found
                    if (abs(bid.t_img - obs_act.t_start) <= self.EPS 
                        and bid.winner == state.agent_name):
                        # ensure only one matching bid exists
                        assert matching_bid is None, \
                            "There should only be one matching bid for the current time step."
                        
                        # assign matching bid
                        matching_bid = bid

                    # check if a previous bid was found
                    elif bid.t_img < obs_act.t_start:
                        if prev_bid is None or bid.t_img > prev_bid.t_img:
                            # assign if most recent previous bid
                            prev_bid = bid

                # ensure matching bid was found
                assert matching_bid is not None, \
                    "Matching bid for observation in path not found in results. Was assigned without updating results."
                
                # update previous observation counts
                n_obs_i[task] = matching_bid.n_obs
                t_prev_i[task] = prev_bid.t_img if prev_bid is not None else np.NINF

        # ensure every parent task in path has values for `n_obs` and `t_prev`
        assert all(
            all(task in n_obs[obs_idx] and task in t_prev[obs_idx]
                for task in obs_action.obs_opp.tasks)
                for obs_idx,obs_action in enumerate(path)), \
            "Not all observation opportunity tasks in path have an assigned observation number values."
        
        # return observation numbers and previous observation times
        return n_obs, t_prev

    def _calculate_path_cost(self,
                             state : SimulationAgentState,
                             _ : object,
                             path : List[ObservationAction]
                            ) -> float:
        """ Calculate total expected cost of observation path. """

        # TODO implement realistic path cost calculation using agility specs to calculate power consumption between maneuvers.

        # initiate previus observation action with dummy action representing the current state
        prev_obs = None

        # compute total angle change
        total_angle_change = 0.0
        for obs in path:
            # get previous look angle
            prev_angle = state.attitude[0] if prev_obs is None else prev_obs.look_angle
            
            # calculate angle change
            total_angle_change += abs(obs.look_angle - prev_angle)

            # update previous observation
            prev_obs = obs
        
        # compute cost from total angle change
        return self.EPS * total_angle_change  # Placeholder implementation       

    """
    BROADCAST SCHEDULING
    """
    def _schedule_broadcasts(self, state: SimulationAgentState, orbitdata: OrbitData, new_bids : dict) -> list:
        """ Schedules broadcasts to be done by this agent """
        # validate inputs
        if not isinstance(state, (SatelliteAgentState, GroundOperatorAgentState)):
            raise NotImplementedError(f'Broadcast scheduling for agents of type `{type(state)}` not yet implemented.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

        # -------------------------------
        # DEBUG BREAKPOINTS
        # x = 1
        # -------------------------------

        # initialize list of broadcasts to be done
        broadcasts : List[AgentAction] = []       

        # generate bid messages to share bids in results
        bidded_tasks = [
            task
            for task,bids in self._results.items()
            if isinstance(task, EventObservationTask)   # only consider bids for event-driven tasks
            if bids                                     # only tasks with bids
        ]
        
        # schedule broadcasts at future access opportunities
        t_broadcasts = self.__schedule_broadcast_times(state, orbitdata, new_bids)

        # crreate broadcast actions for each broadcast time
        for t_broadcast in t_broadcasts:
            # TODO decide whether to broadcast state and observations as well
            
            # check if there are any bid messages to share
            if bidded_tasks:
                broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastMessageAction.BIDS, t_broadcast))

        # include established broadcasts from preplan
        preplan_broadcasts = [action for action in self._preplan.actions
                                # extract only broadcast actions
                                if isinstance(action, BroadcastMessageAction)
                                # exclude broadcasts of future information; 
                                #  these would be redundant with those scheduled here 
                                and not isinstance(action, FutureBroadcastMessageAction)]
        broadcasts.extend(preplan_broadcasts)

        # return scheduled broadcasts
        return sorted(broadcasts, key=lambda action: action.t_start) 

    def __schedule_broadcast_times(self, state : SimulationAgentState, orbitdata : OrbitData, new_bids : dict) -> List[float]:
        """ Schedule broadcast times for sharing bids in results with other agents."""

        # check if any shareble bids to share exist
        if all([not isinstance(task, EventObservationTask) for task in self._results]):
            # No tasks with bids to share; return empty list
            return []
        
        # get current time 
        t_curr : float = state.get_time()

        # define end of planning horizon for broadcast scheduling
        t_next = max(self._preplan.t + self._preplan.horizon, t_curr)

        # check if agent is currently participating in consensus bidding
        include_current = self.__is_participating_in_consensus(new_bids)

        # define set of all communication targets to cover
        all_targets = set(orbitdata.comms_targets)
        remaining = set(all_targets)  # targets we still need to “cover”
        
        # initiate set of broadcast times to be scheduled (avoids duplicates)
        t_broadcasts = set()
        agents_considered = set()

        # get column index of this agent in the comms links table
        u_column_idx = orbitdata.comms_target_indices[state.agent_name]

        # iterate through list of intervals in this time period 
        for t_start,t_end, *component_indices in orbitdata.comms_links.iter_rows_raw(t=t_curr, 
                                                                                     t_max=t_next, 
                                                                                     include_current=include_current):
            
            # get component index of this agent during this interval
            u_component_idx = int(component_indices[u_column_idx])
            
            # find all matching agents with the same component index and add to output list
            targets = set()
            for v_column_idx,v_component_idx in enumerate(component_indices):
                if v_column_idx != u_column_idx and v_component_idx == u_component_idx: 
                    # get target agent name from column index
                    target_agent = orbitdata.comms_target_columns[v_column_idx]

                    # add to set of targets for this interval
                    targets.add(target_agent)

            # skip if target agents have already been considered
            if targets <= agents_considered:
                continue

            # get last access interval and calculate broadcast time
            t_broadcast = t_curr if t_curr > t_start else t_end
            
            # add to list of broadcast times if not already present
            t_broadcasts.add(t_broadcast)

            # mark target agents as considered
            agents_considered.update(targets)

            # check if all targets have been considered 
            remaining.difference_update(targets)
            if not remaining:
                # all targets have been considered; stop scheduling broadcasts
                break

        # return sorted list of broadcast times
        return sorted(t_broadcasts)

    def __is_participating_in_consensus(self, new_bids : dict) -> bool:
        """ Check if this agent is currently participating in consensus bidding. """
        # check if any shareble bids to share exist
        return new_bids or self._performed_bundle_observations
        # return (new_bids 
        #         or self._performed_bundle_observations 
        #         or self._task_announcements_received
        #         or self._bundle_changes_performed)

    """
    REPLAN SCHEDULING
    """
    
    def _schedule_periodic_replan(self, state : SimulationAgentState, t_next : float) -> list:
        """ Creates and schedules a waitForMessage action such that it triggers a periodic replan """
        # ensure next planning time is in the future
        assert state.get_time() <= t_next, "Next planning time must be in the future."
        # schedule wait action for next planning time
        return [WaitAction(t_next,t_next)] if not np.isinf(t_next) else []

    """
    UTILITIES
    """
    def __task_key(self, d : dict) -> tuple:
        """ Key function for sorting task dictionaries consistently. """
        return (
            d["task_type"],
            d["parameter"],
            d["priority"],
            d["id"],
        )

    """
    LOGGING
    """
    def print_results(self):
        super().print_results()

        # flush and close sinks to ensure all logs are printed
        self._observation_rewards.close()

        # clear any cashed data
        self._bundle.clear()
        self._path.clear()
        self._results.clear()
        self._optimistic_bidding_counters.clear()
        self._id_to_tasks.clear()
        self._known_event_tasks.clear()
        self._incoming_event_tasks.clear()
        self._relevant_updates.clear()
        self._preplan = None
        self._plan = None

    def _log_results(self, 
                     dsc : str, 
                     state : SimulationAgentState, 
                     results : Dict[GenericObservationTask, List[Bid]],
                     level=logging.DEBUG, 
                     n_tasks : int = 20) -> None:
        out = f'\nT{np.round(state.get_time(),3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Task ID\t n_obs\tins\t\twinner\tbid\tt_img\tt_bid\tv_opt\tperformed\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 25

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        # sort tasks by if they are event-driven and by ID for consistent logging
        tasks = sorted(results.keys(), key=lambda t: (-int(isinstance(t,EventObservationTask)), repr(t)))
        if len(tasks) <= n_tasks:
            tasks_to_print = tasks
        else:
            non_empty_tasks = [task for task in tasks if results[task]]
            empty_tasks = [task for task in tasks if not results[task]]

            n_empties_to_print = n_tasks - len(non_empty_tasks)
            if n_empties_to_print > 0:
                tasks_to_print = sorted(non_empty_tasks + empty_tasks[:n_empties_to_print],
                                        key=lambda t: (-int(isinstance(t,EventObservationTask)), repr(t)))
            else:
                tasks_to_print = non_empty_tasks[:n_tasks]

        if not tasks_to_print: out += '\t<empty results>\n'

        for i_tasks,task in enumerate(tasks_to_print):
            task : GenericObservationTask
            bids : List[Bid] = results[task]

            if isinstance(task, EventObservationTask):
                req_id_short = f"{task.id.split('-')[-1]}    "
            else:
                req_id_short = f'Default({int(task.location[0][-2])},{int(task.location[0][-1])})'
            
            if isinstance(bids, dict):
                printed_bids = sorted(bids.values(), key=lambda b: b.n_obs)
            elif isinstance(bids, list):
                printed_bids = sorted(bids, key=lambda b: b.n_obs)
            else:
                raise TypeError("Bids in results must be either a list or a dictionary.")   
           
            for i_bid,bid in enumerate(printed_bids):
                bid : Bid
                    
                if bid.winner != bid.NONE: 
                    bid_winner = bid.winner.split('_')
                    if len(bid_winner) >=2:
                        bid_winner = f'{bid_winner[-2]}{bid_winner[-1]}'
                    else:
                        bid_winner = f'{bid_winner[0][0]}{bid_winner[0][-1]}'

                try:
                    if bid.winner != bid.NONE:
                        line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t{bid_winner.lower()}\t{np.round(bid.winning_bid,4)}\t{np.round(bid.t_img,1)}\t{np.round(bid.t_bid,1)}\t{self._optimistic_bidding_counters[bid.task][bid.n_obs]}\t{(bid.performed)}\n'
                    else:
                        line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t\tn/a\t{np.round(bid.winning_bid,4)}\t{np.round(bid.t_img,1)}\t{np.round(bid.t_bid,1)}\t{self._optimistic_bidding_counters[bid.task][bid.n_obs]}\t{(bid.performed)}\n'
                except IndexError:
                    x=  1 # breakpoint
                out += line

            if not bids:
                out += f'{req_id_short} <none>\n'        

            if i_tasks < len(tasks_to_print) - 1:
                for _ in range(L_LINE + L_LINE_PADding):out += '-'
                out += '\n'

        if len(tasks) > n_tasks: 
            out += '...\n'   
         
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        out += f'Total tasks in results: {len(results)}\n'

        print(out)

    def _log_path(self, 
                  dsc : str, 
                  state : SimulationAgentState, 
                  observation_path : List[ObservationAction], 
                  level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.get_time(),3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\tt_img\t Task IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 20

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        if not observation_path: out += '\t<empty path>\n'

        n = 15
        for i,obs in enumerate(observation_path):
            spec_task : ObservationOpportunity = obs.obs_opp
            req_id_short = ""

            for task in spec_task.tasks:
                if isinstance(spec_task, EventObservationTask):
                    req_id_short += spec_task.id.split('-')[-1] + ","
                else:
                    req_id_short += f'Default({int(task.location[0][-2])},{int(task.location[0][-1])}),'

            line = f'{i}\t{np.round(obs.t_start,1)}\t[{req_id_short[:-1]}]\n'
            out += line

            for _ in range(L_LINE + L_LINE_PADding):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + L_LINE_PADding):
                    out += '-'
                out += '\n'
                break

        print(out)

    def _log_bundle(self, 
                    dsc : str, 
                    state : SimulationAgentState, 
                    bundle : List[Tuple[GenericObservationTask, Dict[GenericObservationTask, int]]], 
                    n_rows : int = 15,
                    n_tasks : int = 3,
                    level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.get_time(),3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'i\t Task IDs\n'
        
        # count characters in line for formatting
        L_LINE = len(line)
        L_LINE_PADding = 20

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'

        if not bundle: out += '\t<empty bundle>\n'
        
        for i,(_,tasks) in enumerate(bundle):
            line = f'{i}\t['
            i_task = 0
            for task,n_obs in tasks.items():
                if i_task >= n_tasks: break

                if isinstance(task, EventObservationTask):
                    req_id_short = task.id.split('-')[-1]
                else:
                    req_id_short = f'Default({int(task.location[0][-2])},{int(task.location[0][-1])})'

                line += f'({req_id_short},{n_obs}),'
                i_task += 1 

            if len(tasks) > n_tasks: line += f' ..., (n_tasks={len(tasks)}),'

            line = line[:-1] + ']\n'
            out += line

            for _ in range(L_LINE + L_LINE_PADding):
                out += '-'
            out += '\n'

            if i > n_rows:
                out += f'\t\t...\n'
                break
        
        for _ in range(L_LINE + L_LINE_PADding): out += '='
        out += '\n'
        
        # stats
        out += f'Bundle size: {len(bundle)} observations\n'

        print(out)