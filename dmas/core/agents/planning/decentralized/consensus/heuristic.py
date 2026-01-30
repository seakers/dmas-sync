from abc import abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

from dmas.utils import runtime_tracker
from dmas.clocks import ClockConfig

from execsatm.tasks import DefaultMissionTask, GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission
from execsatm.utils import Interval

from chess3d.agents.planning.decentralized.consensus.consensus import ConsensusPlanner
from chess3d.agents.actions import ObservationAction, WaitAction
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.planning.plan import Plan
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.science.reward import *
from chess3d.agents.states import SimulationAgentState
from chess3d.orbitdata import OrbitData


class HeuristicInsertionConsensusPlanner(ConsensusPlanner):
    """
    # Heuristic Insertion Consensus Planner

    A decentralized consensus planner that utilizes heuristic insertion strategies for planning in multi-agent systems.
    """

    # Heuristics available for insertion model
    EARLIEST_ACCESS = 'earliestAccess'
    TASK_VALUE = 'taskValue'
    TASK_PRIORITY = 'taskPriority'
    HEURISTICS = [EARLIEST_ACCESS, TASK_VALUE, TASK_PRIORITY]

    def __init__(self,
                 heuristic : str = EARLIEST_ACCESS,
                 replan_threshold : int = 1, 
                 optimistic_bidding_threshold : int = 1,
                 periodic_overwrite : bool = False,
                 debug : bool = False, 
                 logger : bool = None):
        super().__init__(ConsensusPlanner.HEURISTIC_INSERTION, replan_threshold, optimistic_bidding_threshold, periodic_overwrite, debug, logger)

        # validate inputs
        assert heuristic in self.HEURISTICS, f"Invalid heuristic '{heuristic}'. Must be one of {self.HEURISTICS}."
                
        # set parameters
        self.heuristic = heuristic

        # initialize properties
        self.observation_opportunities : List[ObservationOpportunity] = None

    @runtime_tracker
    def _build_bundle_from_preplan(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_plan : Plan,
                                    _ : ClockConfig,
                                    orbitdata : OrbitData,
                                    mission : Mission,
                                    observation_history : ObservationHistory
                                    ) -> tuple:    
        """ Build bundle from latest periodic preplan. """
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # extract observations from plan
        preplan_path : List[ObservationAction] = sorted([action for action in current_plan if isinstance(action, ObservationAction)], key=lambda a: a.t_start)
        
        # restrict observation opportunities access and look angles to match those in preplan
        sorted_observation_opportunities : List[ObservationOpportunity] = []
        for obs_action in preplan_path:
            # create new observation opportunity with restricted access and look angles
            restricted_obs_opp = ObservationOpportunity(
                obs_action.obs_opp.tasks,
                obs_action.instrument_name,
                Interval(obs_action.t_start, obs_action.t_end),
                obs_action.t_end-obs_action.t_start,
                Interval(obs_action.look_angle, obs_action.look_angle),
                obs_action.obs_opp.id
            )

            # add to sorted observation opportunities
            sorted_observation_opportunities.append(restricted_obs_opp)

        # build bundle using heuristic insertion method
        proposed_bundle, proposed_path, proposed_bids = \
              self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, sorted_observation_opportunities, orbitdata, mission, observation_history)
        
        if len(preplan_path) != len(proposed_path):
            raise NotImplementedError("Testing for cases where not all observations from preplan are added to bundle not yet performed.")

        # ensure that proposed bundle and path are of same length
        assert len(proposed_bundle) == len(proposed_path), \
            "Proposed bundle and path lengths do not match after building from preplan."

        # restore original observation opportunities in proposed bundle
        for i_path,(bundle_element,path_action) in enumerate(zip(proposed_bundle, proposed_path)):
            # type hints
            bundle_element : Tuple[ObservationOpportunity, Dict[GenericObservationTask, int]]
            path_action : ObservationAction

            # ensure observation opportunities match
            assert bundle_element[0] == path_action.obs_opp, \
                "Observation opportunities in proposed bundle and path do not match after building from preplan."

            # find matching observation action in preplan
            matching_obs_action = None
            while preplan_path:
                # get next observation action in the preplan path
                obs_action : ObservationAction = preplan_path.pop(0)

                # see if it matches the current proposed path action
                if abs(obs_action.t_start-path_action.t_start) < 1e-6 and \
                   abs(obs_action.t_end-path_action.t_end) < 1e-6 and \
                     abs(obs_action.look_angle - path_action.look_angle) < 1e-6 and \
                        obs_action.instrument_name == path_action.instrument_name:
                    matching_obs_action = obs_action
                    break

            # ensure matching action was scheduled and found
            assert matching_obs_action is not None, \
                "Could not find matching observation action in preplan for proposed path action."

            # restore original observation opportunity in bundle
            proposed_bundle[i_path] = (matching_obs_action.obs_opp.copy(), bundle_element[1].copy())

            # restore original observation opportunity in path
            path_action.obs_opp = matching_obs_action.obs_opp.copy()

            # match id for scheduled actions
            path_action.id = matching_obs_action.id       
        
        # -------------------------------
        # DEBUG PRINTOUTS
        # if self._debug:
        #     if not self.is_observation_path_valid(state, proposed_path, None, None, specs):
        #         x =1
        # -------------------------------

        # return proposed bundle and path
        return proposed_bundle, proposed_path, proposed_bids
        
    @runtime_tracker
    def _bundle_building_phase(self,
                       state : SimulationAgentState,
                       specs : object,
                       _ : Plan,
                       tasks : List[GenericObservationTask],
                       __ : ClockConfig,
                       orbitdata : OrbitData,
                       mission : Mission,
                       observation_history : ObservationHistory
                    ) -> tuple:
        
        """ Build new bundle and path according to selected heuristic model. """
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)

        # Outline planning horizon interval
        t_next = max(self.preplan.t + self.preplan.horizon, state.t)
        planning_horizon = Interval(state.t, t_next)
        
        # check if observation opportunities need to be created/recreated
        if self.__need_to_create_observation_opportunities(state, planning_horizon, state.t):
            # calculate observation opportunities
            self.observation_opportunities : List[ObservationOpportunity] \
                = self.__calc_observation_opportunities(state, tasks, planning_horizon, cross_track_fovs, orbitdata)
    
        # generate new plan according to selected model
        if self.heuristic == self.EARLIEST_ACCESS:
            # use earliest-access heuristic
            proposed_bundle, proposed_path, proposed_bids = \
                 self.earliest_access_heuristic_bundle_builder(state, specs, cross_track_fovs, self.observation_opportunities, orbitdata, mission, observation_history)
        
        elif self.heuristic == self.TASK_VALUE:
            # use task-value heuristic
            proposed_bundle, proposed_path, proposed_bids = \
                 self.task_value_heuristic_bundle_builder(state, specs, cross_track_fovs, self.observation_opportunities, orbitdata, mission, observation_history)
        
        elif self.heuristic == self.TASK_PRIORITY:
            # use task-priority heuristic
            proposed_bundle, proposed_path, proposed_bids = \
                 self.task_priority_heuristic_bundle_builder(state, specs, cross_track_fovs, self.observation_opportunities, orbitdata, mission, observation_history)
        else:
            # Fallback for unsupported heuristic
            raise NotImplementedError(f"Heuristic '{self.heuristic}' not supported.")            

        # -------------------------------
        # DEBUG PRINTOUTS
        # if self._debug:
        #     if not self.is_observation_path_valid(state, proposed_path, None, None, specs):
        #       x =1
        # -------------------------------

        return proposed_bundle, proposed_path, proposed_bids

    def __need_to_create_observation_opportunities(self, state : SimulationAgentState, current_planning_horizon : Interval, t : float) -> bool:
        """ Check if observation opportunities need to be created/recreated. """
        # TEMPORARY: always recreate observation opportunities
        return True
        
        # TODO
        # # define recalculation conditions
        # conditions = [
        #     # 0) there is no existing plan
        #     self.plan is None,
        #     # 1) no previous observation opportunities have been calculated
        #     self.observation_opportunities is None,
        #     # 2) available tasks have changed
        #     self.task_announcements_received,
        #     # 3) a new planning horizon has started (since last replan)
        #     self.plan.t_next < current_planning_horizon.right - self.EPS
        # ]

        # # check if any were met
        # return any(conditions)
        
        # if self.plan is None:
        #     return True

        # if self.observation_opportunities is None:
        #     return True
        
        # if self.task_announcements_received:
        #     return True
        
        # if self.plan.t_next < current_planning_horizon.right - self.EPS:
        #     return True

        # # else; no need to recreate observation opportunities
        # return False
    
    def __calc_observation_opportunities(self, 
                                         state : SimulationAgentState, 
                                         tasks : List[GenericObservationTask], 
                                         planning_horizon : Interval, 
                                         cross_track_fovs : dict, 
                                         orbitdata : OrbitData
                                        ) -> List[ObservationOpportunity]:
        """ Get currently stored observation opportunities. """
        # get only available tasks from existing plan and urgent tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
                
        # check if any available tasks exist
        if not available_tasks: return []

        # calculate coverage opportunities for available tasks
        access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

        # create and merge task observation opportunities from scheduled tasks and urgent tasks
        observation_opportunities : List[ObservationOpportunity] = self.create_observation_opportunities_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)
        
        # extract already planned task observation opportunities from current plan
        planned_observation_opportunities = [obs.obs_opp for obs in self.path if isinstance(obs,ObservationAction)]

        # filter tasks that are already in the current plan
        observation_opportunities = [obs_opp for obs_opp in observation_opportunities
                                    if obs_opp not in planned_observation_opportunities]
        
        # return observation opportunities
        return observation_opportunities

    # TEMP DEPRECATED
    # @runtime_tracker
    # def _bundle_building_phase(self,
    #                    state : SimulationAgentState,
    #                    specs : object,
    #                    _ : Plan,
    #                    tasks : List[GenericObservationTask],
    #                    __ : ClockConfig,
    #                    orbitdata : OrbitData,
    #                    mission : Mission,
    #                    observation_history : ObservationHistory
    #                 ) -> tuple:
        
    #     # compile instrument field of view specifications   
    #     cross_track_fovs : dict = self._collect_fov_specs(specs)

    #     # Outline planning horizon interval
    #     t_next = max(self.preplan.t + self.preplan.horizon, state.t)
    #     planning_horizon = Interval(state.t, t_next)
        
    #     # get only available tasks from existing plan and urgent tasks
    #     available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
                
    #     # calculate coverage opportunities for available tasks
    #     access_opportunities : dict[tuple] = self.calculate_access_opportunities(state, planning_horizon, orbitdata)

    #     # create and merge task observation opportunities from scheduled tasks and urgent tasks
    #     observation_opportunities : List[ObservationOpportunity] = self.create_observation_opportunities_from_accesses(available_tasks, access_opportunities, cross_track_fovs, orbitdata)
        
    #     # extract already planned task observation opportunities from current plan
    #     planned_observation_opportunities = [obs.obs_opp for obs in self.path if isinstance(obs,ObservationAction)]

    #     # filter tasks that are already in the current plan
    #     observation_opportunities = [obs_opp for obs_opp in observation_opportunities
    #                                 if obs_opp not in planned_observation_opportunities]
    
    #     # generate new plan according to selected model
    #     if self.heuristic == self.EARLIEST_ACCESS:
    #         # use earliest-access heuristic
    #         proposed_bundle, proposed_path, proposed_bids = \
    #              self.earliest_access_heuristic_bundle_builder(state, specs, cross_track_fovs, observation_opportunities, orbitdata, mission, observation_history)
        
    #     elif self.heuristic == self.TASK_VALUE:
    #         # use task-value heuristic
    #         proposed_bundle, proposed_path, proposed_bids = \
    #              self.task_value_heuristic_bundle_builder(state, specs, cross_track_fovs, observation_opportunities, orbitdata, mission, observation_history)
        
    #     elif self.heuristic == self.TASK_PRIORITY:
    #         # use task-priority heuristic
    #         proposed_bundle, proposed_path, proposed_bids = \
    #              self.task_priority_heuristic_bundle_builder(state, specs, cross_track_fovs, observation_opportunities, orbitdata, mission, observation_history)
    #     else:
    #         # Fallback for unsupported heuristic
    #         raise NotImplementedError(f"Heuristic '{self.heuristic}' not supported.")            

    #     # -------------------------------
    #     # DEBUG PRINTOUTS
    #     # if self._debug:
    #     #     if not self.is_observation_path_valid(state, proposed_path, None, None, specs):
    #     #       x =1
    #     # -------------------------------

    #     return proposed_bundle, proposed_path, proposed_bids
    
    def get_available_tasks(self, tasks: List[GenericObservationTask], planning_horizon : Interval) -> list:
        """ Get only tasks that are available within the planning horizon. """
        # get known tasks that may already be part of the plan
        default_tasks = {task 
                         for task in tasks
                         if isinstance(task, DefaultMissionTask)
                         and task.availability.overlaps(planning_horizon)
                         }

        # get urgent event tasks that are available within planning horizon
        event_tasks = {task 
                        for task in self.known_event_tasks 
                        if task.availability.overlaps(planning_horizon)}
        
        # merge task sets
        available_tasks = {task for task in event_tasks}
        available_tasks.update(default_tasks)

        # return tasks as a merged list
        return list(available_tasks)   

    def earliest_access_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       observation_opportunities : List[ObservationOpportunity],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using earliest-access heuristic. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, ObservationOpportunity, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by earliest access time
        sorted_observation_opportunities = sorted(observation_opportunities, key=lambda task: task.accessibility)
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, sorted_observation_opportunities, orbitdata, mission, observation_history, heuristic_evaluator=lambda task: task.accessibility.left)

    def task_value_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       observation_opportunities : List[ObservationOpportunity],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using task-value as main heuristic for order of task addition in bundle building process. 
         Considers the value of performing the task in isolation. Does not take into account any possible changes 
         in value due to in-schedule interactions.

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, ObservationOpportunity, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by expected task value
        task_values = [(task, self.estimate_observation_opportunity_value(task,
                                                               task.accessibility.left,
                                                               task.min_duration,
                                                               specs,
                                                               cross_track_fovs,
                                                               orbitdata,
                                                               mission,
                                                               observation_history)) for task in observation_opportunities]
        sorted_observation_opportunities = [task for task, _ in sorted(task_values, key=lambda item: (-item[1], item[0].accessibility, item[0].id))]
    
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, sorted_observation_opportunities, orbitdata, mission, observation_history)

    def task_priority_heuristic_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       observation_opportunities : List[ObservationOpportunity],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, list]:
        """ 
        Build bundle using task priority as main heuristic for order of task addition in bundle building process. 
         Considers the intrinsic priority of the tasks being considered in the bundle. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, ObservationOpportunity, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.
        """ 
        # sort urgent tasks by intrinsic task priority
        task_priorities = [(obs, obs.get_priority()) for obs in observation_opportunities]
        sorted_observation_opportunities = [task for task, _ in sorted(task_priorities, key=lambda item: (-item[1], item[0].accessibility, item[0].id))]
        
        # build bundle using heuristic insertion method
        return self.__heuristic_insertion_bundle_builder(state, specs, cross_track_fovs, sorted_observation_opportunities, orbitdata, mission, observation_history)

    def _is_task_mutually_exclusive_with_path(self, task : ObservationOpportunity, path : List[ObservationAction]):
        """ Check if task is mutually exclusive with any observations in the given path. """
        return any([task.is_mutually_exclusive(action.obs_opp) for action in path])

    def __heuristic_insertion_bundle_builder(self,
                                       state : SimulationAgentState,
                                       specs : object,
                                       cross_track_fovs : dict,
                                       sorted_observation_opportunities : List[ObservationOpportunity],
                                       orbitdata : OrbitData,
                                       mission : Mission,
                                       observation_history : ObservationHistory
                                    ) -> Tuple[list, List[ObservationAction], dict]:
        """ 
        Build bundle using a given heuristic. Attempts to insert tasks into existing path, right-shift existing tasks to accommodate for new 
         tasks or replaces tasks in the current plan if it leads to a feasible plan that can increase overall plan utility.  Tasks are added 
         according to heuristic evaluator. 

        #### Returns
        - bundle : List[Tuple[GenericObservationTask, int, ObservationOpportunity, Bid]]
            List of tuples containing (task, observation number, observation time, expected utility).
        - path : List[ObservationAction]
            Updated observation path after bundle building.        
        """

        # initialized bundle from current plan
        proposed_bundle : List[Tuple[ObservationOpportunity, 
                                     Dict[GenericObservationTask, int]]] = \
                [task_tuple for task_tuple in self.bundle]
        
        # initialize proposed path from current plan
        proposed_path : List[ObservationAction] = [obs_action for obs_action in self.path]
        
        # extract existing bids from current bundle
        proposed_bids : Dict[GenericObservationTask, Dict[int, Bid]] = defaultdict(dict)
        for _,obs in proposed_bundle:
            for task,n_obs in obs.items():
                # find matchiong existing bid
                existing_bid = self.results[task][n_obs]

                # ensure this bid is assigned to current agent
                assert existing_bid.winner == state.agent_name, \
                          "Existing bids in bundle do not belong to current agent."
                
                # add to proposed bids
                proposed_bids[task][n_obs] = existing_bid.copy()

        # extract observation number assignments for current path
        n_obs_proposed, t_prev_proposed = self._count_observations_and_revisit_times_from_results(state, proposed_path)

        # calculate current path utility
        current_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, proposed_path, observation_history, orbitdata, mission, n_obs_proposed, t_prev_proposed)

        # -------------------------------
        # DEBUG PRINTOUTS
        # if self._debug:
        #     self._log_results('PROPOSED BIDS (DURING BUNDLE-BUILDING PHASE)', state, proposed_bids)
            # self._log_path('CURRENT PATH (DURING BUNDLE-BUILDING PHASE)', state, proposed_path)
            # self._log_bundle('BUNDLE (DURING BUNDLE-BUILDING PHASE)', state, proposed_bundle)
            # x = 1
        # ------------------------------- 

        # Add tasks to path iteratively based on heuristic
        for proposed_observation in tqdm(sorted_observation_opportunities, desc=f'{state.agent_name}-REPLANNER: Building bundle', leave=False):
            # -------------------------------
            # DEBUG PRINTOUTS
            # if self._debug:
            #     req_id_short = ""

            #     for task in proposed_observation.tasks:
            #         if isinstance(proposed_observation, EventObservationTask):
            #             req_id_short += proposed_observation.id.split('-')[-1] + ","
            #         else:
            #             req_id_short += f'Default({int(task.location[0][-2])},{int(task.location[0][-1])}),'
                
            #     out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n'
            #     out += f'OBSERVATION OPPORTUNITY BEING CONSIDERED FOR BUNDLE ADDITION: \nt={proposed_observation.accessibility} (ParentID(s): [{req_id_short[:-1]}])\n'
            #     print(out)
            # ------------------------------- 
            
            # initialize search for best path for proposed task    
            best_path : List[ObservationAction] = None
            best_path_utility : float = current_path_utility # must outperform current path
            best_bids : Dict[ObservationOpportunity, Dict[GenericObservationTask,Bid]] = None
            # best_abandoned : Dict[GenericObservationTask, Dict[int, Bid]] = None

            # Generate list of candidate paths for this observation opportunity
            candidate_paths = self.__incremental_path_builder(state, specs, proposed_path, proposed_observation)

            # Find best placement in path   
            for candidate_path, obs_added, obs_removed in tqdm(candidate_paths, 
                                                               desc=f'{state.agent_name}-REPLANNER: Evaluating placements for observation {proposed_observation.id.split("-")[0]}', 
                                                               leave=False):
                # -------------------------------
                # DEBUG PRINTOUTS
                # if self._debug:
                #     self._log_path('CANDIDATE PATH (DURING BUNDLE-BUILDING PHASE)', state, candidate_path)
                #     changes_indices = [candidate_path.index(change) if change in candidate_path else None for change in path_changes]
                #     print(f'Changes at: i={changes_indices}')
                # -------------------------------

                # find best observation sequence for each parent task of the proposed task in this candidate path
                n_obs_candidate, t_prev_candidate, bids_candidate \
                    = self._assign_best_observations_and_revisit_times_to_proposed_path(state, candidate_path, obs_added, obs_removed, proposed_bids, specs, cross_track_fovs, orbitdata, mission, observation_history)

                # check if valid bids were found for proposed task
                if bids_candidate is None: continue # no valid bids found; skip

                # get path value for proposed path using best observation sequences
                proposed_path_utility : float = self._calculate_path_utility(state, specs, cross_track_fovs, candidate_path, observation_history, orbitdata, mission, n_obs_candidate, t_prev_candidate)

                # if path does not increase overall utility, skip
                if proposed_path_utility <= best_path_utility: continue
                
                # else: save as best path
                best_path_utility = proposed_path_utility
                best_path = candidate_path
                best_bids = bids_candidate

            # if no best path was found, continue to next proposed task
            if best_path is None: continue
            
            # update proposed path
            proposed_path : List[ObservationAction] = best_path

            # update current path utility
            current_path_utility : float = best_path_utility

            # update bundle
            # updated_proposed_bundle : List[Tuple[ObservationOpportunity, 
            #                                     Dict[GenericObservationTask, int]]] = []
            # # get existing proposed bundle elements if they are still in the proposed path
            # for obs_opp,obs_dict in proposed_bundle:
            #     if any([obs_opp == path_action.obs_opp for path_action in proposed_path]):
            #         updated_proposed_bundle.append((obs_opp, obs_dict))

            ## add new observations to proposed bundle
            obs_dict = {task: bid.n_obs for task,bid in best_bids[proposed_observation].items()}
            proposed_bundle.append((proposed_observation, obs_dict))

            ## collect all observation opportunities in proposed path
            obs_opps_in_path = [obs_action.obs_opp for obs_action in proposed_path]
            
            # find indices of bundle elements not in proposed path
            bundle_elements_to_remove = [
                bundle_idx for bundle_idx,(obs_opp,_) in enumerate(proposed_bundle)
                if obs_opp not in obs_opps_in_path
            ]
            # bundle_elements_to_remove = [ 
            #     bundle_idx for bundle_idx,(obs_opp,_) in enumerate(proposed_bundle)
            #     if all(obs_opp != obs_action.obs_opp for obs_action in proposed_path)
            # ]

            ## remove any existing bids for observations that were removed from the path
            for bundle_idx in sorted(bundle_elements_to_remove, reverse=True):
                proposed_bundle.pop(bundle_idx)

            ## ensure path and bundle lengths match
            assert len(proposed_path) == len(proposed_bundle), \
                "Proposed path and bundle lengths do not match after bundle building phase."

            ## map tasks in bundle to best bids
            matching_bundle_indices = {obs_opp : idx 
                                       for idx,(obs_opp,_) in enumerate(proposed_bundle)
                                       if obs_opp in best_bids}
            
            ## update any values in proposed bids and results
            for obs_opp,idx in matching_bundle_indices.items():
                for task,proposed_bid in best_bids[obs_opp].items():
                    proposed_bundle[idx][1][task] = proposed_bid.n_obs

            # compile list of updated proposed bids
            updated_proposed_bids : Dict[GenericObservationTask, Dict[int, Bid]] = defaultdict(dict)

            # iterate through new proposed bundle and update proposed bids
            for obs,tasks in proposed_bundle:
                # iterate through tasks in observation and update bids
                for task,n_obs in tasks.items():
                    if obs in best_bids and task in best_bids[obs]:
                        # observation was modified; update bids
                        updated_proposed_bids[task][n_obs] = best_bids[obs][task].copy()

                        assert abs(updated_proposed_bids[task][n_obs].t_bid - state.t) < self.EPS, \
                            "Bid time in updated proposed bids does not match current state time."
                    else:
                        # observation was not modified; retain existing proposed bids
                        updated_proposed_bids[task][n_obs] = proposed_bids[task][n_obs].copy()

            # update list of proposed bids
            proposed_bids = updated_proposed_bids

            # -------------------------------
            # DEBUG PRINTOUTS
            # if self._debug:
            #     self._log_results('PROPOSED BIDS (DURING BUNDLE-BUILDING PHASE)', state, proposed_bids)
            #     self._log_path('CURRENT PATH (DURING BUNDLE-BUILDING PHASE)', state, proposed_path)
            #     self._log_bundle('BUNDLE (DURING BUNDLE-BUILDING PHASE)', state, proposed_bundle)
            #     x = 1
            if self._debug:
                if not self.is_observation_path_valid(state, proposed_path, None, None, specs):
                    x =1
            # -------------------------------               

        # -------------------------------
        # DEBUG PRINTOUTS
        if self._debug:
            if not self.is_observation_path_valid(state, proposed_path, None, None, specs):
                x =1
        # -------------------------------

        # return proposed bundle and path
        return proposed_bundle, proposed_path, proposed_bids
        
    """
    BUNDLE-BUILDING PHASE - Path Insertion Methods
    """
    def __incremental_path_builder(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_path : List[ObservationAction],
                                    new_obs : ObservationOpportunity
                                ) -> List[Tuple[List[ObservationAction], List[ObservationAction], List[ObservationAction]]]:
        """ 
        Generates a list of proposed paths by applying the following operators to the path:
            1. Direct Insertion into existing path
            2. Right-shifting existing path to accommodate new task
            3. Replace conflicting task with new urgent task

        #### Returns:
            - `proposed_paths` : List[Tuple[List[ObservationAction], List[ObservationAction], List[ObservationAction]]]
        """
        
        # compile agility specifications
        max_slew_rate, max_torque = self._collect_agility_specs(specs)

        # generate proposed paths
        proposed_paths : List[Tuple[List[ObservationAction], List[ObservationAction], List[ObservationAction]]] = [
            # Option 1: Direct Insertion into existing path
            self._direct_insertion_into_path(state, specs, current_path, new_obs, max_slew_rate, max_torque),

            # Option 2: Right-shifting existing path to accommodate new observation opportunity
            self._right_shift_path_for_new_obs(state, specs, current_path, new_obs, max_slew_rate, max_torque),

            # Option 3: Replace conflicting task with new observation opportunity
            self._replace_conflicting_tasks_with_new_obs(state, specs, current_path, new_obs, max_slew_rate, max_torque),

            # TODO Option 4: Remove all conflicting tasks and insert new task
            # self._remove_conflicting_tasks_and_insert_new_task(state, specs, current_path, new_task, max_slew_rate, max_torque),
        ]

        # ensure new task was included in new paths
        assert not self._debug or all([(path is None or any([action.obs_opp == new_obs for action in path])) for path,*_ in proposed_paths]), \
              "New observation opportunity not included in proposed paths."
        
        # ensure new observation opportunity was included in path changes
        assert not self._debug or all([(path is None or any([action.obs_opp == new_obs for action in path_changes])) for path,path_changes,_ in proposed_paths]), \
              "New observation opportunity not included in proposed path changes."

        # ensure generated paths are valid
        assert not self._debug or all([(path is None or self.is_observation_path_valid(state, path, max_slew_rate, max_torque, specs)) for path,*_ in proposed_paths]), \
              "One or more proposed paths are invalid."
        
        # return proposed paths and the respective observation times for the new observation opportunity in said paths
        return [(path,obs_added,obs_removed) for path,obs_added,obs_removed in proposed_paths if path is not None]
        
    def _direct_insertion_into_path(self,
                                    state : SimulationAgentState,
                                    specs : object,
                                    current_path : List[ObservationAction],
                                    new_obs : ObservationOpportunity,
                                    max_slew_rate : float,
                                    max_torque : float
                                ) -> Tuple[List[ObservationAction], List[ObservationAction], List[ObservationAction]]:
        """ Try to directly insert new task into existing path. """
        # initialize feasible observation time and select observation loook angle for new task
        t_img, th_img = None, np.average([new_obs.slew_angles.left, new_obs.slew_angles.right])

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [action for action in current_path
                                           if action.t_start in new_obs.accessibility
                                           or action.t_end in new_obs.accessibility]
        ## get latest observation before new task accessibility
        prev_observations = [action for action in current_path
                             if action.t_end <= new_obs.accessibility.left]
        prev_observation = max(prev_observations, key=lambda action: action.t_end) if prev_observations else None
        ## get earliest observation after new task accessibility
        next_observations = [action for action in current_path
                             if action.t_start >= new_obs.accessibility.right]
        next_observation = min(next_observations, key=lambda action: action.t_start) if next_observations else None

        # compile conflicting observations        
        conflicting_observations = {prev_observation, next_observation} if prev_observation else {next_observation} if next_observation else set()
        ## get unique observations during new task access
        conflicting_observations.update(observations_during_task_access)
        ## sort conflicting observations by start time
        conflicting_observations = sorted([obs for obs in conflicting_observations 
                                           if obs is not None], key=lambda obs: obs.t_start)

        # set current state as a dummy previous observation
        obs_prev = ObservationAction(new_obs.instrument_name,  state.attitude[0], state.t)

        # check if gaps between observations can accommodate new task
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
            t_img = max(new_obs.accessibility.left, state.t + m)

            # check observation time feasibility
            if t_img not in new_obs.accessibility or t_img + new_obs.min_duration not in new_obs.accessibility:
                t_img = None # no feasible observation time found

        # check if observation time was found
        if t_img is None: return (None, None, None) # no time found; cannot insert new observation into path

        # insert new observation into path
        ## create observation action for new task
        new_observation = ObservationAction(new_obs.instrument_name, th_img, t_img, new_obs.min_duration, new_obs)

        ## create new path with inserted observation
        new_path = [action for action in current_path]
        new_path.append(new_observation)
        new_path = sorted(new_path, key=lambda action: action.t_start)
        
        # return new path if valid
        return (new_path, [new_observation], []) if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None, None)

    def _right_shift_path_for_new_obs(self,
                                        state : SimulationAgentState,
                                        specs : object,
                                        current_path : List[ObservationAction],
                                        new_obs : ObservationOpportunity,
                                        max_slew_rate : float,
                                        max_torque : float
                                    ) -> Tuple[List[ObservationAction], List[ObservationAction]]:
        """ Try to right-shift existing path to accommodate new task. """
        # check if path is empty
        if len(current_path) == 0: 
            # Current path is empty; cannot right-shift path for new task.
            return (None, None, None)

        # check if path is sorted by start time
        assert all(current_path[i].t_start <= current_path[i+1].t_start for i in range(len(current_path)-1)), "Current path is not sorted by start time."

        # select observation look angle for new task
        th_img = np.average([new_obs.slew_angles.left, new_obs.slew_angles.right])

        # find current path observations that occur before the end of the new task's accessibility
        preceeding_observations = [(path_idx,action) for path_idx,action in enumerate(current_path)
                                    if action.t_start <= new_obs.accessibility.right]

        # add a dummy observation at the initial state
        preceeding_observations.insert(0, (-1, ObservationAction(new_obs.instrument_name, state.attitude[0], state.t)))

        # initialize feasible path insertion index and observation time
        i_insert, t_img = None, None

        # iterate through previous observations to find insertion point
        for i_obs,obs_prev in preceeding_observations:
            # check maneuver time between new task and current observation
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate

            # calculate earliest feasible observation time
            t_earliest = max(new_obs.accessibility.left, obs_prev.t_end + m_prev)

            # calculate observation feasibility
            ## 1) must be able to maneuver from previous observation to new task
            ## 2) must fit within new task accessibility window
            is_feasible = (obs_prev.t_end + m_prev <= t_earliest
                           and t_earliest in new_obs.accessibility
                           and t_earliest + new_obs.min_duration in new_obs.accessibility)
            
            # check feasibility
            if not is_feasible: 
                break # cannot insert at this point; stop searching
            
            # update insertion index to next location
            i_insert = i_obs + 1
            # update observation time
            t_img = t_earliest
        
        # check if insertion index was found
        if i_insert is None: return (None, None, None) # no insertion point found; cannot right-shift path for new task

        # initiate new path
        new_path = [action for action in current_path[:i_insert]]
        
        # create new observation action
        new_observation = ObservationAction(new_obs.instrument_name, th_img, t_img, new_obs.min_duration, new_obs)
        
        # add new observation to new path
        new_path.append(new_observation)
        obs_added = [new_observation]
        obs_removed = []

        # right-shift remaining observations
        path_to_shift = [action for action in current_path[i_insert:]]
        for i_curr,obs_curr in enumerate(path_to_shift):
            # check previous observation in path
            obs_prev = new_path[-1]

            # compute maneuver time from previous observation
            m = abs(obs_prev.look_angle - obs_curr.look_angle) / max_slew_rate

            # calculate earliest start time for current observation
            t_earliest = max(obs_prev.t_end + m, obs_curr.obs_opp.accessibility.left)

            # check earliest time if feasible
            is_feasible = (obs_prev.t_end + m <= t_earliest
                           and t_earliest in obs_curr.obs_opp.accessibility
                           and t_earliest + obs_curr.obs_opp.min_duration in obs_curr.obs_opp.accessibility)

            # check of new observation time is earlier the or the same as original
            if t_earliest < obs_curr.t_start or abs(t_earliest - obs_curr.t_start) <= self.EPS:
                # new task start time is earlier than original; 
                #  do not modify remaining plan and add to new path
                new_path.extend(path_to_shift[i_curr:])

                # stop shifting process
                break

            # else if new observation time is feasible, add shifted observation to new path
            elif is_feasible: 
                # create shifted observation action
                shifted_observation = ObservationAction(obs_curr.instrument_name, obs_curr.look_angle, t_earliest, obs_curr.obs_opp.min_duration, obs_curr.obs_opp)

                # add shifted observation to new path
                new_path.append(shifted_observation)
                obs_added.append(shifted_observation)
                obs_removed.append(obs_curr)
                
            # else, could not find a feasible time; abort right-shifting process
            else: 
                return (None, None, None) # cannot right-shift path for new task
            
        # return new path if valid
        return (new_path, obs_added, obs_removed) if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs) else (None, None, None)
    
    def _replace_conflicting_tasks_with_new_obs(self,
                                                 state : SimulationAgentState,
                                                 specs : object,
                                                 current_path : List[ObservationAction],
                                                 new_task : ObservationOpportunity,
                                                 max_slew_rate : float,
                                                 max_torque : float
                                            ) -> Tuple[List[ObservationAction], List[ObservationAction]]:
        """ Try to replace conflicting tasks in existing path with new task. """
        # check if path is empty
        if len(current_path) == 0: 
            # Current path is empty; cannot replace conflicting tasks in path for new task.
            return (None, None, None)

        # find possible conflicts in current path
        ## find observations that are being performed during new task accessibility
        observations_during_task_access = [(obs_idx,obs) for obs_idx,obs in enumerate(current_path)
                                           if obs.t_start in new_task.accessibility
                                           or obs.t_end in new_task.accessibility
                                           or (obs.t_start < new_task.accessibility.left
                                               and obs.t_end > new_task.accessibility.right)]

        conflicting_observations = [obs_tup for obs_tup in observations_during_task_access]

        # check if any conflicting observations were found
        if not conflicting_observations: 
            # no conflicting observations found; cannot replace conflicting tasks in path for new task.
            return (None, None, None)
        
        # select observation loook angle for new task
        th_img = np.average([new_task.slew_angles.left, new_task.slew_angles.right])

        # check if removing conflicting observations can accommodate new task
        for conflict_idx,conflicting_observation in conflicting_observations: 
            # create new proposed path
            new_path = [action for action in current_path]

            # # remove conflicting observation
            # new_path.pop(conflict_idx) 

            # get preceeding observation action
            if conflict_idx == 0:
                # set previous observation as dummy action at current state
                obs_prev = ObservationAction(new_task.instrument_name, state.attitude[0], state.t)
            else:
                # select previous observation from path
                obs_prev = new_path[conflict_idx-1]

            # calculate maneuver time between previous observation and new task
            m_prev = abs(obs_prev.look_angle - th_img) / max_slew_rate

            # estimate earliest feasible observation time
            t_img = max(new_task.accessibility.left, obs_prev.t_end + m_prev)

            # get next observation time
            try:
                t_next = new_path[conflict_idx + 1].t_start
                th_next = new_path[conflict_idx + 1].look_angle
            except IndexError:
                t_next = np.inf # no next observation; set to infinity
                th_next = th_img # no next observation; set to new task look angle

            # calculate maneuver time between new task and next observation
            m_next = abs(th_next - th_img) / max_slew_rate

            # check if earliest observation time is feasible
            feasibility_constraints = [
                # 1) must be able to maneuver from previous observation to new task
                obs_prev.t_end + m_prev <= t_img,   
                # 2) must be able to maneuver from new task to next observation
                t_img + new_task.min_duration + m_next <= t_next, 
                # 3) must fit within new task accessibility window
                t_img in new_task.accessibility,
                t_img + new_task.min_duration in new_task.accessibility
            ]

            # check if new task meets all feasibility constraints
            if not all(feasibility_constraints):  continue # not feasible; try next conflicting observation

            # create new observation action for new task
            new_observation = ObservationAction(new_task.instrument_name, th_img, t_img, new_task.min_duration, new_task)
            
            # replace conflicting observation with new task
            new_path[conflict_idx] = new_observation

            # ensure conflicting observation was replaced
            assert conflicting_observation not in new_path, \
                "Conflicting observation was not properly replaced in new path."

            # return new path if valid
            if self.is_observation_path_valid(state, new_path, max_slew_rate, max_torque, specs):
                return (new_path, [new_observation], [conflicting_observation]) # return new path and changes

        # unable to accommodate new task by replacing conflicting observations
        return (None, None, None)

    """
    BUNDLE-BUILDING PHASE - Bid Generation Methods
    """    
    def _assign_best_observations_and_revisit_times_to_proposed_path(self,
                                                                     state : SimulationAgentState,  
                                                                     candidate_path : List[ObservationAction],
                                                                     obs_added : List[ObservationAction],
                                                                     obs_removed : List[ObservationAction],
                                                                     proposed_bids : Dict[GenericObservationTask, Dict[int, Bid]],
                                                                     specs : object,
                                                                     cross_track_fovs : dict,
                                                                     orbitdata : OrbitData,
                                                                     mission : Mission,
                                                                     observation_history : ObservationHistory
                                                                    ) -> Tuple[Dict[int, Dict[GenericObservationTask, int]], 
                                                                                Dict[int, Dict[GenericObservationTask, float]],
                                                                                Dict[ObservationOpportunity, Dict[GenericObservationTask, Bid]]]:
        """ Generate best observation numbers and revisit times for each observation in the proposed path. 
        
        ### Returns 
            - n_obs_best : Dict[int, Dict[GenericObservationTask, int]] - Best observation numbers for each observation in the proposed path.
            - t_prev_best : Dict[int, Dict[GenericObservationTask, float]] - Best previous observation times for each observation in the proposed path.
        """
        # extract modified task observation opportunities from path changes
        added_tasks : Set[GenericObservationTask] =\
            {task for obs_act in obs_added for task in obs_act.obs_opp.tasks}
        removed_tasks : Set[GenericObservationTask] = \
            {task for obs_act in obs_removed for task in obs_act.obs_opp.tasks}
        modified_tasks : Set[GenericObservationTask] = added_tasks.union(removed_tasks)
        
        modified_tasks_in_path : List[GenericObservationTask] = \
              sorted({task
                      for obs_act in candidate_path
                        for task in obs_act.obs_opp.tasks
                        if task in modified_tasks}, key=lambda x: x.id)

        # find observation time for proposed task in candidate path
        modified_task_obs_times : Dict[GenericObservationTask, List[Tuple[float,str,float,ObservationOpportunity]]] \
                    = {task : [
                        (action.t_start, state.agent_name, action.look_angle, action.obs_opp) 
                        for action in candidate_path 
                        if task in action.obs_opp.tasks
                    ] for task in modified_tasks_in_path}
        
        # initialize best observation numbers and previous observation times
        n_obs_best : Dict[GenericObservationTask, list[str]] = {task : [] for task in modified_tasks_in_path}
        t_img_best : Dict[GenericObservationTask, list[float]] = {task : [] for task in modified_tasks_in_path}
        t_prev_best : Dict[GenericObservationTask, list[float]] = {task : [] for task in modified_tasks_in_path}
        obs_names_best : Dict[GenericObservationTask, list[str]] = {task : [] for task in modified_tasks_in_path}
        vals_best : Dict[GenericObservationTask, list[float]] = {task : [] for task in modified_tasks_in_path}
        
        # initialize search for best sequence for each task
        best_values : dict = {task : np.NINF for task in modified_tasks_in_path}

        # find best observation sequences for each parent task
        for task in modified_tasks_in_path:
            # assume parent task has been considered in results
            assert task in self.results, f"Parent task {task} not being bid on by any agent; cannot generate bids."
            
            # count all previously performed observations for this parent task
            performed_obs : list[Tuple[float,str,float,ObservationOpportunity]] = \
                            [(bid.t_img,bid.owner,np.NAN,None) 
                             for bid in self.results[task] 
                             if bid.was_performed()]
            latest_performed_obs_time : Tuple[float,str,float,ObservationOpportunity] \
                = max(performed_obs, key=lambda obs: obs[0]) if performed_obs else None
            
            # initialize feasible observation sequences for this task
            available_obs_times : list[Tuple[float,str,float,ObservationOpportunity]] = []

            # get all possible observation opportunities from results
            scheduled_obs_times : list[Tuple[float,str,float,ObservationOpportunity]] = \
                  [(bid.t_img,bid.winner,np.NAN,None) for bid in self.results[task] 
                   if bid.winner != state.agent_name
                   and not bid.was_performed()]

            # include proposed task imaging time 
            available_obs_times.extend(scheduled_obs_times)
            available_obs_times.extend(modified_task_obs_times[task])

            # sort by observation time
            available_obs_times.sort(key=lambda x: x[0])

            # check if any repeated observation times exist for this agent
            obs_times_for_agent = [t_img for t_img,agent_name,_,_ in available_obs_times if agent_name == state.agent_name]
            for t_img in obs_times_for_agent:
                if any(abs(t_img - other_t_img) <= self.EPS for other_t_img in obs_times_for_agent if other_t_img != t_img):
                    # repeated observation time found for this agent; raise error
                    raise ValueError(f"Repeated observation time {t_img} found for agent '{state.agent_name}' when generating observation sequences for task '{task.id}'.")

            # collect feasible sequences
            feasible_sequences = self._find_feasible_observation_sequences_for_task(state, task, available_obs_times)
            
            # find sequence that maximizes value for this agent
            for obs_names,obs_times,obs_look_angles,obs_tasks in feasible_sequences:
                # initiate sequence value tracker
                seq_values = []
                t_prev_seq = []
                n_obs_seq = []
                is_sequence_valid = True

                # evaluate sequence value for this agent
                for seq_idx,(agent_name,t_obs,look_angle,spec_task) in enumerate(zip(obs_names,obs_times,obs_look_angles,obs_tasks)):
                    
                    # get observation number for this observation
                    n_obs = seq_idx + len(performed_obs)

                    # get observation number and previous observation time
                    t_prev = obs_times[seq_idx-1] if seq_idx > 0 else latest_performed_obs_time[0] if performed_obs else np.NINF
                    
                    if n_obs > 0: assert t_prev >= 0.0, \
                        "Previous observation time is not defined for observation number greater than zero."

                    # get observation value
                    if agent_name != state.agent_name: # observation is to be performed by another agent
                        # get matching bid for this observation
                        matching_bid : Bid = self.results[task][n_obs]

                        # ensure matching bid is from correct agent
                        assert matching_bid.winner == agent_name, \
                            "Matching bid winner does not match agent assigned to observation."
                        assert abs(matching_bid.t_img - t_obs) <= self.EPS, \
                            "Matching bid observation time does not match assigned observation time."
                        
                        # get observation value from winning bid
                        task_value = matching_bid.winning_bid

                    else: # observation is to be performed by this agent
                        # assume specific task was defined
                        assert isinstance(spec_task, ObservationOpportunity), \
                            "Task observation opportunity not defined."
                        
                        # estimate task value for this observation
                        task_value = self._estimate_task_value(task,
                                                            spec_task.instrument_name,
                                                            look_angle, 
                                                            t_obs,
                                                            spec_task.min_duration,
                                                            specs, 
                                                            cross_track_fovs,
                                                            orbitdata,
                                                            mission,
                                                            n_obs,
                                                            t_prev
                                                            )
                        
                        # compare task value estimate against existing bids for this observation number

                        # if no existing bid for this observation number, accept if:
                        if n_obs >= len(self.results[task]):
                            accept_bid = [
                                # 1) proposed observation value is positive 
                                task_value > 0.0
                            ]
                        # if there is an existing bid, accept if either:
                        else:
                            # get existing bid
                            try:
                                existing_bid : Bid = proposed_bids[task][n_obs]
                            except KeyError:
                                existing_bid : Bid = self.results[task][n_obs]

                            # get mutex bids for this observation
                            following_bids = [
                                                mutex_bid
                                                for mutex_bid in self.results[task]
                                                if mutex_bid.n_obs > n_obs
                                                and mutex_bid.winner != state.agent_name
                                            ]
                            mutex_bids = [existing_bid] + following_bids

                            # determine if bid is accepted
                            accept_bid = [
                                # 1) I am the current bid winner and proposed observation value is positive
                                existing_bid.winner == state.agent_name and task_value > 0.0,
                                # 2) or if proposed observation value outperforms existing winning bids for all mutex bids
                                all(task_value > mutex_bid.winning_bid for mutex_bid in mutex_bids),
                                # 3) or if proposed earlier observation time and optimistic bidding counter allows it
                                t_obs < existing_bid.t_img 
                                    and self.optimistic_bidding_counters[task][n_obs] > 0
                            ]

                        # check if bid is accepted
                        if not any(accept_bid):
                            # bid not accepted; skip this sequence
                            is_sequence_valid = False
                            break

                    # accumulate sequence value
                    seq_values.append(task_value)     
                    t_prev_seq.append(t_prev) 
                    n_obs_seq.append(n_obs)     

                # skip to next sequence if current sequence is invalid
                if not is_sequence_valid: 
                    continue

                # check if length of sequence values matches length of observation times
                if len(seq_values) != len(obs_times): 
                    continue # invalid sequence length; skip

                # compute total sequence value
                total_seq_value = sum(seq_values)                

                # check if this sequence outperforms previous best
                if total_seq_value > best_values[task]:
                    # update best sequence value and sequence
                    best_values[task] = total_seq_value

                    # update to best sequence trackers
                    n_obs_best[task] = n_obs_seq
                    t_img_best[task] = obs_times
                    t_prev_best[task] = t_prev_seq 
                    obs_names_best[task] = obs_names
                    vals_best[task] = seq_values  

        # check if a sequence was found for all modified parent tasks
        if any(value < 0.0 for value in best_values.values()):
            return None, None, None #, None

        # -------------------------------
        # DEBUG BREAKPOINT
        # if self._debug:
        #     x = 1
        # -------------------------------
            
        # filter out observations from other agents in best sequences
        indeces_to_remove = {task : [idx for idx,agent_name in enumerate(obs_names_best[task])
                                            if agent_name != state.agent_name] 
                             for task in modified_tasks_in_path}
        
        for task,indices in indeces_to_remove.items():
            for idx in sorted(indices, reverse=True):
                n_obs_best[task].pop(idx)
                t_img_best[task].pop(idx)
                t_prev_best[task].pop(idx)
                obs_names_best[task].pop(idx)
                vals_best[task].pop(idx)

        # ensure filter was successful
        assert all([all([agent_name == state.agent_name for agent_name in obs_names_best[task]])
                   for task in modified_tasks_in_path]), \
               "Not all observations from other agents were removed from best sequences."
        
        # initiate bid lists for tasks in the proposed path based on best observation numbers and previous observation times
        new_bids : Dict[ObservationOpportunity, Dict[GenericObservationTask, Bid]] = defaultdict(dict)

        # initiate list of best observation numbers and previous observation times for each observation in candidate path
        n_obs_candidate = [dict() for _ in candidate_path]
        t_prev_candidate = [dict() for _ in candidate_path]
        
        # assign best observation numbers and previous observation times to observations in candidate path
        for obs_idx,obs in enumerate(candidate_path):
            # iterate through matching tasks of this observation
            for task in obs.obs_opp.tasks:
                # check if sequence was modified for this parent task
                if task in n_obs_best:
                    # extract observation time and revisit time from best sequences
                    n_obs = n_obs_best[task].pop(0)
                    t_prev = t_prev_best[task].pop(0)
                    val = vals_best[task].pop(0)
                    t_img = t_img_best[task].pop(0)

                    if n_obs > 0: assert t_prev >= 0.0, \
                        "Previous observation time is not defined for observation number greater than zero."

                    # generate new bids for this observation if it is part of path changes
                    new_bid = Bid(task, state.agent_name, n_obs, val, val, state.agent_name, t_img, state.t, None, obs.instrument_name)
                    new_bids[obs.obs_opp][task] = new_bid

                    # assign best observation number and previous observation time
                    n_obs_candidate[obs_idx][task] = n_obs
                    t_prev_candidate[obs_idx][task] = t_prev
                else:
                    # no best sequence found for this parent task; use existing bids from results
                    # get matching bid for this observation task
                    matching_bids = [bid for bid in proposed_bids[task].values()
                                    if abs(bid.t_img - obs.t_start) <= self.EPS
                                    and bid.owner == state.agent_name]
                    
                    assert matching_bids, \
                        "Matching bid for observation in path not found in results. Was assigned without updating results."
                    assert len(matching_bids) <= 1, \
                        "There should be at most one matching bid for the current time step."

                    matching_bid : Bid = matching_bids.pop()

                    # get previous matching observations for this task
                    prev_bids_self = [bid for bid in proposed_bids[task].values()
                                        if bid.t_img < obs.t_start]
                    previous_bids_other = [bid for bid in self.results[task]
                                        if (bid.winner != state.agent_name)
                                        and bid.t_img < obs.t_start]
                    prevoous_bids_perf = [bid for bid in self.results[task]
                                        if bid.was_performed()
                                        and bid.t_img < obs.t_start]
                    prev_bids = prev_bids_self + previous_bids_other + prevoous_bids_perf

                    # update previous observation counts
                    n_obs_candidate[obs_idx][task] = matching_bid.n_obs
                    t_prev_candidate[obs_idx][task] = max((bid.t_img for bid in prev_bids), default=np.NINF)

                    if matching_bid.n_obs > 0: assert t_prev_candidate[obs_idx][task] >= 0.0, \
                        "Previous observation time is not defined for observation number greater than zero."
        
        # TODO assure all best observation numbers have been assigned
        
        # TODO assure assignments are consistent within candidate path

        # return updated observation numbers and previous observation times
        return n_obs_candidate, t_prev_candidate, new_bids #, abandoned_bids

    def _find_feasible_observation_sequences_for_task(self,
                                                      state : SimulationAgentState,
                                                      task : GenericObservationTask,
                                                      available_obs : List[tuple]
                                                    ) -> List[Tuple[List[str], List[float]]]:
        """ Find feasible observation number sequences for a given task. """
        # initialize feasible sequence tracker
        feasible_sequences = []

        # count number of completed observations for this task
        performed_bids = [bid for bid in self.results[task] if bid.was_performed()]

        # count minimum sequence length; use number of occurrences of this agent in available observation times
        min_seq_length = sum(1 for _,agent_name,*_ in available_obs if agent_name == state.agent_name)

        # create dfs queue
        dfs_queue = deque()

        # seed dfs with initial observations from this agent
        for obs in available_obs: dfs_queue.append([obs])

        # perform dfs to find feasible sequences
        while dfs_queue:
            # pop current sequence from stack
            current_sequence = dfs_queue.pop()

            # unpack last proposed observation in sequence
            t_img,agent_obs,*_ = current_sequence[-1]            
    
            # if last observation is from another agent, check consistency with known results
            if agent_obs != state.agent_name:
                # determine observation number for this observation
                n_obs = len(current_sequence) + len(performed_bids) - 1

                # check if bid for this observation exists
                if task not in self.results:
                    # no matching bids exist for this task; cannot add as start point
                    continue
                elif len(self.results[task]) <= n_obs:
                    # matching no bids exist for this observation number; cannot add successor
                    continue
                elif self.results[task][n_obs].winner != agent_obs:
                    # known bid for this observation number is from another agent; cannot add as start point
                    continue
                elif abs(self.results[task][n_obs].t_img - t_img) > self.EPS:
                    # observation time for this observation number does not match; cannot add as start point
                    continue
                # --- IGNORE AND PRUNE ---

            # check if current sequence can be accepted
            if (len(current_sequence) >= min_seq_length                 # meets minimum length requirements
                and sum(1 for _,agent_name,_,_ in current_sequence      # includes minimum number of observations from this agent
                   if agent_name == state.agent_name) >= min_seq_length 
                ):
                # sequence can be accepted; decompose sequence into component lists
                obs_names = [agent_name for _,agent_name,_,_ in current_sequence]
                obs_times = [t_img for t_img,_,_,_ in current_sequence]
                obs_look_angles = [look_angle for _,_,look_angle,_ in current_sequence]
                obs_tasks = [spec_task for _,_,_,spec_task in current_sequence]
                
                # add to feasible sequences
                feasible_sequences.append((obs_names, obs_times, obs_look_angles, obs_tasks))               

            # check for available successors
            successors = [obs for obs in available_obs
                          if obs[0] > current_sequence[-1][0]]

            # queue successors
            for obs_next in successors:
                # create new sequence with successor added
                new_sequence = [obs for obs in current_sequence] + [obs_next]

                # add new sequence to dfs stack
                dfs_queue.append(new_sequence)

        # return feasible sequences
        return feasible_sequences