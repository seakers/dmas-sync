
from collections import defaultdict
from functools import reduce
import logging
import queue
from typing import Dict, Set, Tuple

from instrupy.base import BasicSensorModel
from instrupy.passive_optical_scanner_model import PassiveOpticalScannerModel
from instrupy.util import ViewGeometry, SphericalGeometry
from orbitpy.util import Spacecraft

from pyparsing import List
from tqdm import tqdm

from execsatm.tasks import GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.attributes import CapabilityRequirementAttributes, ObservationRequirementAttributes, SpatialCoverageRequirementAttributes, TemporalRequirementAttributes
from execsatm.mission import Mission
from execsatm.requirements import CapabilityRequirement, CategoricalRequirement, ConstantValueRequirement, ExpDecayRequirement, ExpSaturationRequirement, GaussianRequirement, IntervalInterpolationRequirement, LogThresholdRequirement, PerformancePreferenceStrategies, PerformanceRequirement, StepsRequirement, TriangleRequirement
from execsatm.utils import Interval

from dmas.models.actions import ObservationAction
from dmas.models.planning.plan import Plan
from dmas.models.trackers import ObservationHistory, ObservationTracker
from dmas.models.states import *
from dmas.models.science.requests import *
from dmas.core.messages import *
from dmas.utils.orbitdata import OrbitData

class AbstractPlanner(ABC):
    """ 
    Describes a generic planner that, given a new set of percepts, decides whether to generate a new plan
    """
    # Constants
    EPS = 1e-6
    
    def __init__(self, 
                 debug : bool = False,
                 logger : logging.Logger = None) -> None:
        # initialize object
        super().__init__()

        # check inputs
        if not isinstance(logger,logging.Logger) and logger is not None: 
            raise ValueError(f'`logger` must be of type `Logger`. Is of type `{type(logger)}`.')

        # initialize attributes
        self.latest_performed_observations : Set[ObservationOpportunity] \
            = set()  # set of last performed observations
        
        # set attribute parameters
        self._debug = debug                 # toggles debugging features
        self._logger = logger               # logger for debugging

    @abstractmethod
    def update_percepts( self,
                         completed_actions : list
                        ) -> None:
        """ Updates internal knowledge based on incoming percepts """
        # check if observations are peformed in completed actions
        performed_observations : Set[ObservationOpportunity] \
            = {action.obs_opp for action in completed_actions
                    if isinstance(action, ObservationAction)}

        # if so, update latest observation opportunities measured by this agent
        if performed_observations:
            self.latest_performed_observations = performed_observations

    @abstractmethod
    def needs_planning(self, **kwargs) -> bool:
        """ Determines whether planning is triggered """ 
        
    @abstractmethod
    def generate_plan(self, **kwargs) -> Plan:
        """ Creates a plan for the agent to perform """
    
    def calculate_access_opportunities(self, 
                                       state : SimulationAgentState, 
                                       planning_horizon : Interval,
                                       orbitdata : OrbitData
                                    ) -> dict:
        """ Calculate access opportunities for targets visible in the planning horizon """

        # check planning horizon span
        if planning_horizon.is_empty(): return {}

        # compile coverage data
        raw_coverage_data : dict = orbitdata.gp_access_data.lookup_interval(planning_horizon.left, planning_horizon.right)

        # group by grid index and ground point index
        coverage_idx_by_target : Dict[int, Dict[int, Dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i,t in tqdm(enumerate(raw_coverage_data['time [s]']), 
                        desc=f'{state.agent_name}/PLANNER: Grouping access opportunities', 
                        unit=' time-step', 
                        disable=len(raw_coverage_data['time [s]'])<10,
                        mininterval=0.5, 
                        leave=False):
            # extract relevant data
            grid_index = raw_coverage_data['grid index'][i]
            gp_index = raw_coverage_data['GP index'][i]
            instrument = raw_coverage_data['instrument'][i]

            # place in appropriate dictionary entry
            coverage_idx_by_target[grid_index][gp_index][instrument].append((i,t))

        # initiate merged access opportunities
        access_opportunities : Dict[int, Dict[int, Dict[str, List[tuple]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # merge access interval opportunities
        for grid_idx,gp_accesses in tqdm(coverage_idx_by_target.items(), 
                                         desc=f'{state.agent_name}/PLANNER: Merging access opportunities', 
                                         unit=' target',
                                         disable=len(coverage_idx_by_target)<10,
                                         leave=False, 
                                         mininterval=0.5,
                                        ):
            for gp_idx,instrument_accesses in gp_accesses.items():
                for instrument,access_indices in instrument_accesses.items():
                    # initialize merged access intervals
                    merged_access_intervals : list[tuple] = []
                    interval_indices : list = []

                    # sort access indices
                    access_indices.sort()

                    # initialize first interval
                    t_start = access_indices[0][1]
                    t_end = access_indices[0][1]
                    indices = [access_indices[0][0]]

                    # iterate through access indices and merge intervals
                    for idx,t in access_indices[1:]:
                        if t <= t_end + orbitdata.time_step + self.EPS:
                            # extend current interval
                            t_end = t
                            indices.append(idx)
                        else:
                            # save current interval
                            merged_access_intervals.append( Interval(t_start, t_end) )
                            interval_indices.append(list(indices))
                            
                            # start new interval
                            t_start = t
                            t_end = t
                            indices = [idx]
                    
                    # add last interval
                    merged_access_intervals.append( Interval(t_start, t_end) )
                    interval_indices.append(list(indices))

                    for interval,indices in zip(merged_access_intervals, interval_indices):
                        access_opportunities[grid_idx][gp_idx][instrument].append( (interval, 
                                                                                    [raw_coverage_data['time [s]'][i] for i in indices],
                                                                                    [raw_coverage_data['off-nadir axis angle [deg]'][i] for i in indices]
                                                                                    ) )

        return access_opportunities

    
    def create_observation_opportunities_from_accesses(self, 
                                    available_tasks : List[GenericObservationTask],
                                    access_times : List[tuple], 
                                    cross_track_fovs : dict,
                                    orbitdata : OrbitData,
                                    must_overlap : bool = True,
                                    threshold : float = 5*60
                                    ) -> list:
        """ 
        Creates observation opportunities from precalculated access times of known generic task targets. 

        #### Arguments
        - `available_tasks` : List of known and available generic observation tasks.
        - `access_times` : List of access times for each generic observation task.
        - `cross_track_fovs` : Dictionary of cross-track fields of view for each instrument.
        - `orbitdata` : Precalculated orbit and coverage data for the mission.
        - `must_overlap` : Whether tasks' availability must overlap in availability time to be considered for clustering. (default : True)
        - `threshold` : The time threshold for clustering tasks in seconds [s] (default : 300 seconds = 5 minutes).
        """

        if not must_overlap: raise NotImplementedError('Clustering without overlap is not yet fully implemented.')

        # generate task observation opportunities from access times
        observation_opps : list[ObservationOpportunity] \
            = self.single_task_observation_opportunity_from_accesses(available_tasks, access_times, cross_track_fovs, orbitdata)
        
        # remove duplicates if needed
        observation_opps = list(set(observation_opps))

        # filter out opportunities that have just been performed
        filtered_observation_opps : list[ObservationOpportunity] \
            = [obs for obs in observation_opps 
               if all(not obs.is_mutually_exclusive(performed_obs) 
                      for performed_obs in self.latest_performed_observations)] \
                if self.latest_performed_observations else observation_opps
        
        # check if tasks are clusterable
        task_adjacency : Dict[str, set[ObservationOpportunity]] \
            = self.check_task_observation_opportunity_clusterability(filtered_observation_opps, must_overlap, threshold)
   
        # cluster tasks based on adjacency
        combined_obs : list[ObservationOpportunity] \
            = self.cluster_task_observation_opportunities(filtered_observation_opps, task_adjacency, must_overlap, threshold)

        # add clustered tasks to the final list of tasks available for scheduling
        filtered_observation_opps.extend(combined_obs) 

        assert all([obs.slew_angles.span()-1e-6 <= cross_track_fovs[obs.instrument_name] 
                    for obs in filtered_observation_opps]), \
            f"Tasks have slew angles larger than the maximum allowed field of view."

        # return tasks
        return sorted(filtered_observation_opps, key=lambda x: x.accessibility)
        
    
    def single_task_observation_opportunity_from_accesses(self,
                                                          available_tasks : List[GenericObservationTask],
                                                          access_times : List[tuple], 
                                                          cross_track_fovs : dict,
                                                          orbitdata : OrbitData,
                                                          threshold : float = 1e-9
                                                        ) -> List[ObservationOpportunity]:
        """ Creates one instance of a task observation opportunity per each access opportunity 
        for every available task """

        # initialize list of task observation opportunities
        observation_opps : List[ObservationOpportunity] = []

        # create one instance of an observation opportunity per each access opportunity
        for task in tqdm(available_tasks, desc="Calculating access times to known tasks", leave=False):

            # extract minimum duration requirement for this task
            min_duration_req : float = self.__extract_minimum_duration_req(task, orbitdata)

            # ensure minimum duration requirement is a positive number
            assert isinstance(min_duration_req, (int,float)) and min_duration_req >= 0.0, "minimum duration requirement must be a positive number."

            # collect access interval information for each target location for this task
            for *__,grid_index,gp_index in task.location:
                # ensure grid_index and gp_index are integers
                grid_index,gp_index = int(grid_index), int(gp_index)
                
                # check if target is accessible  
                if grid_index not in access_times or gp_index not in access_times[grid_index]:
                    continue

                # get access times for this ground point and grid index
                matching_access_times = [
                                        (instrument, access_interval, t, th)
                                        for instrument in access_times[grid_index][gp_index]
                                        for access_interval,t,th in access_times[grid_index][gp_index][instrument]
                                        if task.availability.overlaps(access_interval)
                                        ]
                
                # create a task observation opportunity for each access time
                for access_time in matching_access_times:
                    # unpack access time
                    instrument_name,accessibility,_,th = access_time
                    accessibility : Interval
                    
                    if max(th) - min(th) > cross_track_fovs[instrument_name]:
                        # not all of the accessibility is observable with a single pass
                        continue
                        # TODO raise NotImplementedError('No support for tasks that require multiple passes yet.')
                    else:
                        off_axis_angles = [Interval(off_axis_angle - cross_track_fovs[instrument_name]/2,
                                                    off_axis_angle + cross_track_fovs[instrument_name]/2)
                                                    for off_axis_angle in th]
                        slew_angles : Interval = reduce(lambda a, b: a.intersection(b), off_axis_angles)

                    if slew_angles.is_empty(): 
                        continue  # skip if no valid slew angles

                    # check if instrument can perform the task                    
                    if not self.can_perform_task(task, instrument_name): 
                        continue # skip if not

                    # check if access time matches task availability
                    if not task.availability.overlaps(accessibility):
                        continue # skip if not

                    # check if access time is enough to perform the task
                    if min_duration_req > accessibility.span():
                        # check if accessibility span is non-zero
                        if accessibility.span() <= 0.0: 
                            continue # accessibility time is too short; skip
    
                        # check if available timespan longer than the minimum observation duration
                        if accessibility.span() - min_duration_req >= threshold: 
                            continue # is over the threshold; skip

                        # create and add task observation opportunity to list of task observation opportunities with a different minimum observation requirement
                        observation_opps.append(ObservationOpportunity(task,
                                                                        instrument_name,
                                                                        accessibility,
                                                                        accessibility.span(), # slightly shorter than `min_duration_req`
                                                                        slew_angles
                                                                        ))
                    else:
                        # create and add task observation opportunity to list of task observation opportunities
                        observation_opps.append(ObservationOpportunity(task,
                                                                        instrument_name,
                                                                        accessibility,
                                                                        min_duration_req,
                                                                        slew_angles
                                                                        ))

        
        # return list of task observation opportunities
        return observation_opps
    
    def __extract_minimum_duration_req(self, task : GenericObservationTask, orbitdata : OrbitData) -> float:
        """ Extracts the minimum duration requirement for a given task. """
        
        # check if task has any objectives
        if task.objective is None:
            return orbitdata.time_step # no objectives assigned to this task; assume default minimum duration requirement

        # extract any duration requirements from the task objective
        duration_reqs = [req for req in task.objective
                        if req.attribute == TemporalRequirementAttributes.DURATION.value]
        
        # check if any duration requirements were found
        if not duration_reqs: return orbitdata.time_step # no duration requirement found; return default minimum duration requirement

        # get duration requirement
        duration_req : PerformanceRequirement = duration_reqs[0]

        # extract minimum duration requirement based on requirement type
        if isinstance(duration_req, CategoricalRequirement):
            raise ValueError('Categorical duration requirements are not supported.')
        
        elif isinstance(duration_req, ConstantValueRequirement):
            return duration_req.value # return constant duration requirement value
        
        elif isinstance(duration_req, ExpSaturationRequirement):
            return - (1 / duration_req.sat_rate) * np.log(1 - 0.01) # return duration requirement at 1% saturation

        elif isinstance(duration_req, LogThresholdRequirement):
            return duration_req.threshold # return log threshold duration requirement value

        elif isinstance(duration_req, ExpDecayRequirement):
            return - (1 / duration_req.decay_rate) * np.log(0.01) # return duration requirement at 1% decay

        elif isinstance(duration_req, GaussianRequirement):
            # TODO implement gaussian requirement extraction
            raise NotImplementedError('Gaussian duration requirements are not supported yet.')
        
        elif isinstance(duration_req, TriangleRequirement):
            min_duration = duration_req.reference - (duration_req.width / 2) * (1 - 0.01) # 1% of the triangle height
            return max(min_duration, 0.0) # ensure non-negative duration requirement

        elif isinstance(duration_req, StepsRequirement):
            # filter non-zero scores
            positive_scores = [(idx, score) for idx,score in enumerate(duration_req.scores) if score > 0]

            # check if there are any positive scores        
            if not positive_scores:
                raise ValueError('No positive scores found in `StepsRequirement` for duration requirement.')

            # get interval with minimum positive score
            min_idx, _ = min(positive_scores, key=lambda x: x[1])

            if min_idx == 0:
                return max(duration_req.thresholds[0], 0.0)
            elif min_idx == len(duration_req.thresholds):
                return max(duration_req.thresholds[-1], 0.0)
            else:
                return max(min(duration_req.thresholds[min_idx + 1], duration_req.thresholds[min_idx]), 0.0)
        
        elif isinstance(duration_req, IntervalInterpolationRequirement):
            # TODO implement interval interpolation requirement extraction
            raise NotImplementedError('Interval interpolation duration requirements are not supported yet.')     
       
        # unsupported requirement type; should not reach here
        raise ValueError('Unsupported duration requirement type.')
            
    def can_perform_task(self, task : GenericObservationTask, instrument_name : str) -> bool:
        """ Checks if the agent can perform the task at hand with the given instrument """
        # TODO Replace this with KG for better reasoning capabilities; currently assumes instrument has general capability

        # Check if task has specified objectives
        if task.objective is not None:
            # Extract capability requirements from the objective
            capability_reqs = [req for req in task.objective
                               if isinstance(req, CapabilityRequirement)
                               and req.attribute == CapabilityRequirementAttributes.INSTRUMENT.value]
            capability_req: CapabilityRequirement = capability_reqs[0] if capability_reqs else None

            # Evaluate capability requirement
            if capability_req is not None:
                return capability_req.calc_preference(CapabilityRequirementAttributes.INSTRUMENT.value, instrument_name.lower()) >= 0.5

        # No capability objectives specified; check if instrument has general capability
        return True
        
    
    def check_task_observation_opportunity_clusterability(self, observation_opportunities : List[ObservationOpportunity], must_overlap : bool, threshold : float) -> dict:
        """ 
        Creates adjacency list for a given list of task observation opportunities.

        #### Arguments
        - `observation_opportunities` : A list of task observation opportunities to create the adjacency list for.
        - `must_overlap` : Whether tasks' availability must overlap in availability time to be considered for clustering.
        - `threshold` : The time threshold for clustering tasks in seconds [s].
        """

        # create adjacency list for tasks
        adj : Dict[str, set[ObservationOpportunity]] = {task.id : set() for task in observation_opportunities}
        assert len(adj) == len(observation_opportunities), \
            "Duplicate observation opportunity IDs found when creating adjacency list."

        if observation_opportunities:
            # sort tasks by accessibility
            observation_opportunities.sort(key=lambda a : a.accessibility) 
            
            # get min and max accessibility times
            t_min = observation_opportunities[0].accessibility.left

            # initialize bins
            bins = defaultdict(list)
            
            # group task in bins by accessibility
            for task in tqdm(observation_opportunities, leave=False, desc="Grouping tasks into bins"):
                task : ObservationOpportunity
                center_time = (task.accessibility.left + task.accessibility.right) / 2 - t_min
                bin_key = int(center_time // threshold)
                bins[bin_key].append(task)

            # populate adjacency list
            with tqdm(total=len(observation_opportunities), desc="Checking task clusterability", leave=False) as pbar:
                for b in bins:
                    candidates : list[ObservationOpportunity]\
                          = bins[b] + bins.get(b + 1, [])  # optionally add b-1 for symmetry
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            t1, t2 = candidates[i], candidates[j]
                            if t1.can_merge(t2, must_overlap=must_overlap, max_duration=threshold):
                                adj[t1.id].add(t2)
                                adj[t2.id].add(t1)
                        pbar.update(1)

        # check if adjacency list is symmetric
        for p in observation_opportunities:
            assert p not in adj[p.id], \
                f'Task {p.id} is in its own adjacency list.'
            for q in adj[p.id]:
                assert p in adj[q.id], \
                    f'Task {p.id} is in the adjacency list of task {q.id} but not vice versa.'

        return adj

    
    def cluster_task_observation_opportunities(self, 
                                               observation_opportunities : List[ObservationOpportunity], 
                                               adj : Dict[str, Set[ObservationOpportunity]], 
                                               must_overlap : bool, 
                                               threshold : float) -> list:
        """ 
        Clusters observation opportunities based on adjacency. 
        
        ```
        while V!=Ø do
            Pick a vertex p with largest degree from V. 
                If such p are not unique, pick the p with highest priority.
            
            while N(p)=Ø do
                Pick a neighbor q of p, q ∈ N(p), such that the number of their common neighbors is maximum. 
                    If such p are not unique, pick the p with least edges being deleted.
                    Again, if such p are still not unique, pick the p with highest priority.
                Combine q and p into a new p
                Delete edges from q and p that are not connected to their common neighbors
                Reset neighbor collection N(p) for the new p
            end while
            
            Output the cluster-task denoted by p
            Delete p from V
        end while
        ```
        
        """         
        # only keep observation opportunities that have at least one clusterable observation opportunity
        v = [obs for obs in observation_opportunities if len(adj[obs.id]) > 0]
        
        # sort observation opportunities by degree of adjacency 
        v : list[ObservationOpportunity] = self.__sort_by_degree(observation_opportunities, adj)
        
        # combine observation opportunities into clusters
        combined_obs : list[ObservationOpportunity] = []

        with tqdm(total=len(v), desc="Merging overlapping observation opportunities", leave=False) as pbar:
            while len(v) > 0:
                # pop first observation opportunity from the list of observation opportunities to be scheduled
                p : ObservationOpportunity = v.pop()

                # get list of neighbors of p sorted by number of common neighbors
                n_p : list[ObservationOpportunity] = self.__sort_observation_opportunities_by_common_neighbors(p, list(adj[p.id]), adj)

                # initialize clique with p
                clique = set()

                # update progress bar
                pbar.update(1)

                # while there are neighbors of p
                while len(n_p) > 0:
                    # pop first neighbor q from the list of neighbors
                    q : ObservationOpportunity = n_p.pop()

                    # Combine q and p into a new p                 
                    clique.add(q)

                    # find common neighbors of p and q
                    common_neighbors : set[ObservationOpportunity] = adj[p.id].intersection(adj[q.id])
                   
                    # remove edges to p and q that do not include common neighbors
                    for neighbor in adj[p.id].difference(common_neighbors): adj[neighbor.id].discard(p)
                    for neighbor in adj[q.id]: adj[neighbor.id].discard(q)              
                    
                    # update edges of p and q to only include common neighbors
                    adj[p.id].intersection_update(common_neighbors)
                    
                    # remove q from the adjacency list
                    adj.pop(q.id)

                    # remove q from the list of tasks to be scheduled
                    v.remove(q)

                    # Reset neighbor collection N_p for the new p;
                    n_p : list[ObservationOpportunity] = self.__sort_observation_opportunities_by_common_neighbors(p, list(adj[p.id]), adj)               

                for q in clique: 
                    # TODO: look into ID being used. Ideally we would want a new ID for the combined task.

                    # merge all tasks in the clique into a single task p
                    p = p.merge(q, must_overlap=must_overlap, max_duration=threshold)  # max duration of 5 minutes

                    # update progress bar
                    pbar.update(1)

                # DEBUGGING--------- 
                # clique.add(p)
                # cliques.append(sorted([observation_opportunities.index(t)+1 for t in clique]))
                # ------------------

                # add merged task to the list of combined tasks
                combined_obs.append(p) 

                # sort remaining task observation opportunities by degree of adjacency 
                v : list[ObservationOpportunity] = self.__sort_by_degree(v, adj)
        
        # return only observation opportunities that have multiple parents (avoid generating duplicate observation opportunities)
        return [obs for obs in combined_obs if len(obs.tasks) > 1] 

    
    def __sort_by_degree(self, obs_opportunities : List[ObservationOpportunity], adjacency : dict) -> list:
        """ Sorts observation opportunities by degree of adjacency. """
        # calculate degree of each observation opportunity
        degrees : dict = {obs : len(adjacency[obs.id]) for obs in obs_opportunities}

        # sort observation opportunities by degree and return
        return sorted(obs_opportunities, key=lambda p: (degrees[p], sum([parent_task.priority for parent_task in p.tasks]), -p.accessibility.left))

    def __sort_observation_opportunities_by_common_neighbors(self, p : ObservationOpportunity, n_p : list, adjacency : dict) -> list:
        # specify types
        n_p : list[ObservationOpportunity] = n_p
        adjacency : Dict[str, set[ObservationOpportunity]] = adjacency

        # calculate common neighbors
        common_neighbors : dict = {q : adjacency[p.id].intersection(adjacency[q.id]) 
                                   for q in n_p}
        
        # calculate neighbors to delete
        neighbors_to_delete : dict = {q : adjacency[p.id].difference(adjacency[q.id])
                                      for q in n_p}
        
        # sort neighbors by number of common neighbors, number of edges to delete, priority and accessibility
        return sorted(n_p, 
                      key=lambda p: (len(common_neighbors[p]), 
                                     -len(neighbors_to_delete[p]),
                                     sum([parent_task.priority for parent_task in p.tasks]), 
                                     -p.accessibility.left))

    
    def estimate_observation_opportunity_value(self, 
                                     obs : ObservationOpportunity, 
                                     t_img : float,
                                     d_img : float,
                                     specs : object, 
                                     cross_track_fovs : Dict[str, float],
                                     orbitdata : OrbitData,
                                     mission : Mission,
                                     observation_history : ObservationHistory,
                                     task_n_obs : Dict[GenericObservationTask,int] = None,
                                     task_t_prevs : Dict[GenericObservationTask,int] = None
                                ) -> float:
        """ 
        
        Estimates task value based on predicted observation performance. 
        
        #### Arguments
        - `obs` : The observation opportunity to estimate the value for.
        - `t_img` : The time of the observation [s].
        - `d_img` : The duration of the observation [s].
        - `specs` : The agent or spacecraft specifications.
        - `cross_track_fovs` : The cross-track fields of view for each instrument.
        - `orbitdata` : The pre-computed orbit and coverage data for the mission.
        - `mission` : The mission assigned to the agent performing the observation.
        - `observation_history` : The observation history tracker for the agent.
        - `task_n_obs` : A dictionary mapping tasks being observed by this agent to the number of observations planned for them.
        - `task_t_prevs` : A dictionary mapping tasks being observed by this agent to the time of the previous observation planned for them.
        """
        
        # check if previous observation counts and times are provided
        if task_n_obs is None or task_t_prevs is None:
            # no previous observation counts and times provided;
            #  count previous observations for each task in the observation opportunity
            task_n_obs, task_t_prevs = self._count_previous_observations_from_history(obs, t_img, observation_history)
        
        # estimate measurment look angle 
        th_img = np.average([obs.slew_angles.left, obs.slew_angles.right])

        # calculate task reward per parent task
        rewards = {parent_task : self._estimate_task_value(parent_task,
                                                            obs.instrument_name,
                                                            th_img,
                                                            t_img,
                                                            d_img,
                                                            specs,
                                                            cross_track_fovs,
                                                            orbitdata,
                                                            mission,
                                                            task_n_obs[parent_task],
                                                            task_t_prevs[parent_task])
                     for parent_task in obs.tasks}

        # return total reward
        return sum(rewards.values())    
    
    def _count_previous_observations_from_history(self,
                                                   obs : ObservationOpportunity,
                                                   t_img : float,
                                                   observation_history : ObservationHistory,
                                                ) -> Tuple[Dict[GenericObservationTask,int], Dict[GenericObservationTask,float]]:
        """ Counts the number of previous observations for each task in the observation opportunity. """
        # initialize observation counts and previous observation times
        task_n_obs : Dict[GenericObservationTask,int] = {task : 0 for task in obs.tasks} 
        task_t_prev : Dict[GenericObservationTask,int] = {task : np.NINF for task in obs.tasks} 

        # Find tergets per task
        for task in obs.tasks:
            # iterate through task targets
            for *_,grid_index,gp_index in task.location:
                # unpack grid and gp indices
                grid_index,gp_index = int(grid_index), int(gp_index)

                # get past observations for this target before current image time
                target_observation : ObservationTracker = observation_history.get_observation_history(grid_index, gp_index)

                # check if there are no previous observations for this target
                if target_observation is None: continue  

                # count number of previous observations and observation time for this task
                task_n_obs[task] += target_observation.n_obs
                task_t_prev[task] = max(task_t_prev[task], target_observation.t_last) if target_observation.t_last <= t_img else task_t_prev[task]

                # validate previous observation time
                if task_n_obs[task] > 0: assert task_t_prev[task] >= 0.0, "Previous observation time must be non-negative."
                    
        # return observation counts and previous observation times
        return task_n_obs, task_t_prev

    def _estimate_task_value(self,
                            task : GenericObservationTask,
                            instrument_name : str,
                            th_img : float,
                            t_img : float,
                            d_img : float,
                            specs : Spacecraft, 
                            cross_track_fovs : dict,
                            orbitdata : OrbitData,
                            mission : Mission,
                            n_obs : int = 0,
                            t_prev : float = np.NINF
                        ) -> float:
        
        assert isinstance(n_obs, int) and n_obs >= 0, \
            "Number of previous observations must be a non-negative integer."
        assert isinstance(t_prev, (int,float)) and (t_prev <= t_img), \
            "Previous observation time must be less than or equal to the current image time."
        if n_obs > 0: assert t_prev >= 0.0, \
            "Previous observation time must be non-negative if there are previous observations."

        measurement_performance : dict = self.__estimate_task_performance_metrics(task, 
                                                                                 instrument_name, 
                                                                                 th_img, 
                                                                                 t_img, 
                                                                                 d_img, 
                                                                                 specs, 
                                                                                 cross_track_fovs, 
                                                                                 orbitdata, 
                                                                                 n_obs, 
                                                                                 t_prev)

        return max([mission.calc_task_value(task, measurement) 
                    for measurement in measurement_performance.values()]) \
                        if len(measurement_performance.values()) > 0 else 0.0

        
    def __estimate_task_performance_metrics(self, 
                                            task : GenericObservationTask, 
                                            instrument_name : str,
                                            th_img : float,
                                            t_img : float,
                                            d_img : float,
                                            specs : Spacecraft, 
                                            cross_track_fovs : dict,
                                            orbitdata : OrbitData,
                                            n_obs : int,
                                            t_prev : float,  
                                        ) -> dict:

        # validate inputs
        assert isinstance(task, GenericObservationTask), "Task must be of type `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(th_img, (int,float)), "Image look angle must be a numeric value."
        assert isinstance(t_img, (int,float)), "Image time must be a numeric value."
        assert t_img >= 0, "Image time must be non-negative."
        assert isinstance(d_img, (int,float)), "Image duration must be a numeric value."
        assert d_img >= 0, "Image duration must be non-negative."
        assert all(isinstance(instr, str) for instr in cross_track_fovs.keys()), "Cross-track FOV instrument names must be strings."
        assert all(isinstance(fov, (int,float)) for fov in cross_track_fovs.values()), "Cross-track FOVs must be numeric values."
        assert all(fov >= 0 for fov in cross_track_fovs.values()), "Cross-track FOVs must be non-negative."
        assert isinstance(orbitdata, OrbitData), "Orbit data must be of type `OrbitData`."
        assert n_obs >= 0, "Number of observations must be non-negative."
        assert t_prev <= t_img, "Last observation time must be before the current image time."

        # get access metrics for given observation time, instrument, and look angle
        observation_performances = self.get_available_accesses(task, instrument_name, th_img, t_img, d_img, orbitdata, cross_track_fovs)

        # check if there are no valid observations for this task
        if any([len(observation_performances[col]) == 0 for col in observation_performances]): 
            # no valid accesses; no reward added
            return dict()
        
        # group observations by location
        observed_location_groups : dict[tuple[int,int], list[int]] = defaultdict(list)
        for i in range(len(observation_performances['time [s]'])):
            # unpack observed target location information
            lat = observation_performances['lat [deg]'][i]
            lon = observation_performances['lon [deg]'][i]
            grid_index = int(observation_performances['grid index'][i])
            gp_index = int(observation_performances['GP index'][i])

            # define location indices
            loc = (lat,lon,grid_index,gp_index)

            # add to location group
            observed_location_groups[loc].append({col.lower() : observation_performances[col][i] 
                                                    for col in observation_performances})
        
        # sort groups by measurement time 
        for loc in observed_location_groups: observed_location_groups[loc].sort(key=lambda a : a['time [s]'])
        
        # get unique task targets
        task_targets : List[tuple] = list({(grid_idx,gp_idx) 
                                           for *_,grid_idx,gp_idx in task.location})

        # keep only one of the observations per location group that matches the task target
        observation_performance_metrics : Dict[tuple[int,int], dict] = {loc : observed_location_groups[loc][0] # keep only first observation
                                                 for loc in observed_location_groups
                                                 if (loc[2],loc[3]) in task_targets
                                                 }       

        # get instrument specifications
        instrument_spec : BasicSensorModel = next(instr 
                                                  for instr in specs.instrument
                                                  if instr.name.lower() == instrument_name.lower()).mode[0]

        # include additional observation information 
        for loc,obs_perf in observation_performance_metrics.items():
            
            # update observation performance information
            obs_perf.update({ 
                SpatialCoverageRequirementAttributes.LOCATION.value : [loc],
                TemporalRequirementAttributes.DURATION.value : d_img,
                TemporalRequirementAttributes.REVISIT_TIME.value : t_img - t_prev,
                #TODO Co-observation time
                TemporalRequirementAttributes.RESPONSE_TIME.value : t_img - task.availability.left,
                TemporalRequirementAttributes.RESPONSE_TIME_NORM.value : (t_img - task.availability.left) / task.availability.span() if task.availability.span() > 0 else 0.0,
                TemporalRequirementAttributes.OBS_TIME.value : t_img,
                "t_end" : t_img + d_img,
                ObservationRequirementAttributes.OBSERVATION_NUMBER.value : n_obs + 1, # including this observation
            })

            # handle special case of first observation
            if n_obs == 0:
                obs_perf[TemporalRequirementAttributes.REVISIT_TIME.value] = 0.0

            # update instrument-specific observation performance information
            if (('vnir' in instrument_name.lower() or 'tir' in instrument_name.lower())
                or ('vnir' in instrument_spec._type.lower() or 'tir' in instrument_spec._type.lower())):
                if isinstance(instrument_spec.spectral_resolution, str):
                    obs_perf.update({
                        ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution.lower()
                    })
                elif isinstance(instrument_spec.spectral_resolution, (int,float)):
                    obs_perf.update({
                        ObservationRequirementAttributes.SPECTRAL_RESOLUTION.value : instrument_spec.spectral_resolution
                    })
                else:
                    raise ValueError('Unsupported type for spectral resolution in instrument specification.')
                
            elif ('altimeter' in instrument_name.lower()
                  or 'altimeter' in instrument_spec._type.lower()):
                obs_perf.update({
                    ObservationRequirementAttributes.ACCURACY.value : observation_performance_metrics[loc][ObservationRequirementAttributes.ACCURACY.value],
                })
            else:
                raise NotImplementedError(f'Calculation of task reward not yet supported for instruments of type `{instrument_name.lower()}`.')

        return observation_performance_metrics
    
    
    def get_available_accesses(self, 
                               task : GenericObservationTask, 
                               instrument_name : str,
                               th_img : float,
                               t_img : float,
                               d_img : float,
                               orbitdata : OrbitData, 
                               cross_track_fovs : dict
                            ) -> dict:
        """ Uses pre-computed orbitdata to estimate observation metrics for a given task. """
        
        # get task targets
        task_targets = {(int(grid_index), int(gp_index))
                        for *_,grid_index,gp_index in task.location}
        
        # get ground points accessesible during the availability of the task
        raw_access_data : Dict[str,list] \
            = orbitdata.gp_access_data.lookup_interval(t_img, t_img + d_img)

        # extract ground point accesses that are within the agent's field of view
        accessible_gps_data_indeces = [i for i in range(len(raw_access_data['time [s]']))
                                        if abs(raw_access_data['off-nadir axis angle [deg]'][i] - th_img) \
                                            <= cross_track_fovs[instrument_name] / 2
                                        and raw_access_data['instrument'][i] == instrument_name]
        accessible_gps_performances = {col : [raw_access_data[col][i] 
                                              for i in accessible_gps_data_indeces]
                                    for col in raw_access_data}
        
        # extract gp accesses of the desired targets within the task's accessibility and agent's field of view
        valid_access_data_indeces = [i for i in range(len(accessible_gps_performances['time [s]']))
                                    if (int(accessible_gps_performances['grid index'][i]), \
                                        int(accessible_gps_performances['GP index'][i])) in task_targets]
        observation_performances = {col : [accessible_gps_performances[col][i] 
                                           for i in valid_access_data_indeces]
                                    for col in accessible_gps_performances}

        # get agent eclipse data
        agent_eclipse_intervals : list[Interval] \
            = orbitdata.eclipse_data.lookup_intervals(t_img, t_img + d_img)

        # include eclipse data for each observation in the performance metrics
        observation_performances[ObservationRequirementAttributes.ECLIPSE.value] = [
            int(any([t in interval for interval in agent_eclipse_intervals]))
            for t in observation_performances['time [s]']
        ]

        # return estimated observation performances
        return observation_performances

    
    def _create_broadcast_path(self, 
                               state : SimulationAgentState, 
                               orbitdata : OrbitData = None,
                               t_init : float = None
                               ) -> tuple:
        """ Finds the best path for broadcasting a message to all agents using depth-first-search 
        
        ### Arguments:
            - state (`SimulationAgentState`): current state of the agent
            - orbitdata (`OrbitData`): coverage data of agent if it is of type `SatelliteAgent`
            - t_init (`float`): ealiest desired broadcast time
        """
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Broadcast routing path not yet supported for agents of type `{type(state)}`')
        
        # get earliest desired broadcast time 
        t_init = state._t if t_init is None or t_init < state._t else t_init

        # populate list of all agents except the parent agent
        target_agents = [target_agent 
                         for target_agent in orbitdata.comms_links.keys() 
                         if target_agent != state.agent_name]
        
        if not target_agents: 
            # no other agents in the simulation; no need for relays
            return ([], t_init)
        
        # check if broadcast needs to be routed
        earliest_accesses = [   orbitdata.get_next_agent_access(target_agent, t_init) 
                                for target_agent in target_agents]           
        
        same_access_start = [   abs(access.left - earliest_accesses[0].left) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, Interval)]
        same_access_end = [     abs(access.right - earliest_accesses[0].right) < 1e-3
                                for access in earliest_accesses 
                                if isinstance(access, Interval)]

        if all(same_access_start) and all(same_access_end):
            # all agents are accessing eachother at the same time; no need for mesasge relays
            return ([], t_init)   

        # look for relay path using depth-first search

        # initialize queue
        q = queue.Queue()
        
        # initialize min path and min path cost
        min_path = []
        min_times = []
        min_cost = np.Inf

        # add parent agent as the root node
        q.put((state.agent_name, [], [], 0.0))

        while not q.empty():
            # get next node in the search
            _, current_path, current_times, path_cost = q.get()

            # check if path is complete
            if len(target_agents) == len(current_path):
                # check if minimum cost
                if path_cost < min_cost:
                    min_cost = path_cost
                    min_path = [path_element for path_element in current_path]
                    min_times = [path_time for path_time in current_times]

            # add children nodes to queue
            for receiver_agent in [receiver_agent for receiver_agent in target_agents 
                                    if receiver_agent not in current_path
                                    and receiver_agent != state.agent_name
                                    ]:
                # query next access interval to children nodes
                t_access : float = state._t + path_cost

                access_interval : Interval = orbitdata.get_next_agent_access(receiver_agent, t_access)
                
                if access_interval.left < np.Inf:
                    new_path = [path_element for path_element in current_path]
                    new_path.append(receiver_agent)

                    new_cost = access_interval.left - state._t

                    new_times = [path_time for path_time in current_times]
                    new_times.append(new_cost + state._t)

                    q.put((receiver_agent, new_path, new_times, new_cost))

        # check if earliest broadcast time is valid
        if min_times: assert state._t <= min_times[0]

        # return path and broadcast start time
        return (min_path, min_times[0]) if min_path else ([], np.Inf)
    
    
    def _schedule_relay(self, relay_message : SimulationMessage) -> list:
        raise NotImplementedError('Relay scheduling not yet supported.')
        
        # check if relay message has a valid relay path
        assert relay.path

        # find next destination and access time
        next_dst = relay.path.pop(0)
        
        # query next access interval to children nodes
        sender_orbitdata : OrbitData = orbitdata[state.agent_name]
        access_interval : Interval = sender_orbitdata.get_next_agent_access(next_dst, state.t)
        t_start : float = access_interval.start

        if t_start < np.Inf:
            # if found, create broadcast action
            broadcast_action = BroadcastMessageAction(relay.to_dict(), t_start)
            
            # check broadcast start; only add to plan if it's within the planning horizon
            if t_start <= state.t + self.horizon:
                broadcasts.append(broadcast_action)

    
    def _schedule_maneuvers(    self, 
                                state : SimulationAgentState, 
                                specs : object,
                                observations : List[ObservationAction],
                                orbitdata : OrbitData = None
                            ) -> list:
        """
        Generates a list of AgentActions from the current path.

        Agents look to move to their designated measurement target and perform the measurement.

        ## Arguments:
            - state (:obj:`SimulationAgentState`): state of the agent at the start of the path
            - specs (`dict` or `Sapcecraft`): contains information regarding the physical specifications of the agent
            - path (`list`): list of tuples indicating the sequence of observations to be performed and time of observation
            - t_init (`float`): start time for plan
        """

        # validate inputs
        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Maneuver scheduling for agents of type `{type(state)}` not yet implemented.')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `Spacecraft` for agents of state type `{type(state)}`. Is of type `{type(specs)}`.')
        elif orbitdata is None:
            raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')
        assert all([isinstance(observation, ObservationAction) for observation in observations]), "`observations` must be a list of `ObservationAction` objects."

        # compile instrument field of view specifications   
        cross_track_fovs = self._collect_fov_specs(specs)

        # get pointing agility specifications
        max_slew_rate,_ = self._collect_agility_specs(specs)
        
        # initialize maneuver list
        maneuvers : list[ManeuverAction] = []

        for i,curr_observation in tqdm(enumerate(observations), 
                      desc=f'{state.agent_name}-PLANNER: Scheduling Maneuvers', 
                      leave=False):

            # estimate previous state
            if i == 0:
                t_prev = state._t
                prev_state : SatelliteAgentState = state.copy()
                
            else:
                prev_observation : ObservationAction = observations[i-1]
                t_prev = prev_observation.t_end
                prev_state : SimulationAgentState = state.propagate(t_prev)
                prev_state.attitude = [prev_observation.look_angle, 0.0, 0.0]

            # maneuver to point to target
            if isinstance(state, SatelliteAgentState):
                prev_state : SatelliteAgentState
                
                dth_req = abs(curr_observation.look_angle - prev_state.attitude[0])
                dth_max = (curr_observation.t_start - prev_state._t) * max_slew_rate

                if dth_req > dth_max and abs(dth_req - dth_max) >= 1e-6: 
                    # maneuver impossible within timeframe
                    raise ValueError(f'Cannot schedule maneuver. Not enough time between observations')\
                
                # check if attitude maneuver is required
                if abs(dth_req) <= 1e-3: continue # already pointing in the same direction; ignore maneuver

                # calculate attitude duration    
                th_0 = prev_state.attitude[0]
                th_f = curr_observation.look_angle
                slew_rate = (th_f - th_0) / dth_req * max_slew_rate
                dt = abs(th_f - th_0) / max_slew_rate

                # calculate maneuver time
                t_maneuver_start = curr_observation.t_start - dt
                t_maneuver_end = curr_observation.t_start

                # check if mnaeuver time is non-zero
                if abs(t_maneuver_start - t_maneuver_end) >= 1e-3:
                    # maneuver has non-zero duration; perform maneuver
                    maneuvers.append(ManeuverAction([th_0, 0, 0],
                                                    [th_f, 0, 0], 
                                                    [slew_rate, 0, 0],
                                                    t_maneuver_start, 
                                                    t_maneuver_end)) 

        maneuvers.sort(key=lambda a: a.t_start)

        assert self.is_maneuver_path_valid(state, specs, observations, maneuvers, max_slew_rate, cross_track_fovs)

        return maneuvers
    
    def is_maneuver_path_valid(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               observations : List[ObservationAction], 
                               maneuvers : List[ManeuverAction],
                               max_slew_rate : float,
                               cross_track_fovs : dict
                               ) -> bool:

        for observation in observations:
            observation : ObservationAction

            # get fov for this observation's instrument
            cross_track_fov : float = cross_track_fovs[observation.instrument_name]

            # check if previous maneuvers were performed
            prev_maneuvers = [maneuver for maneuver in maneuvers
                              if maneuver.t_start <= observation.t_start]
            prev_maneuvers.sort(key=lambda a : a.t_start)

            if prev_maneuvers: # there was a maneuver performed before this observation
                # get latest maneuver
                latest_maneuver : ManeuverAction = prev_maneuvers.pop()

                # check status of completion of this maneuver
                t_0 = latest_maneuver.t_start 
                t_f = min(latest_maneuver.t_end, observation.t_start)
                slew_rate = latest_maneuver.attitude_rates[0]
                th_0 = latest_maneuver.initial_attitude[0]
                th_f = th_0 + slew_rate * (t_f - t_0)

                # compare resulting attitude to intended look angle
                dth = abs(observation.look_angle - th_f) 

            else: # there were no maneuvers performed before this observation
                # compare to initial state
                dth = abs(observation.look_angle - state.attitude[0])

            if dth > cross_track_fov / 2.0 and abs(dth - cross_track_fov / 2.0) >= 1e-6:
                # latest state does not point towards the target at the intended look angle
                return False

        # all maneuvers passed checks; path is valid        
        return True
    
    def _collect_fov_specs(self, specs : Spacecraft) -> dict:
        """ get instrument field of view specifications from agent specs object """
        # validate inputs
        assert isinstance(specs, Spacecraft), f'`specs` needs to be of type `Spacecraft`. Is of type `{type(specs)}`.'

        # compile instrument field of view specifications   
        cross_track_fovs = {instrument.name: np.NAN for instrument in specs.instrument}
        for instrument in specs.instrument:
            cross_track_fov = []
            for instrument_model in instrument.mode:
                if isinstance(instrument_model, BasicSensorModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                elif isinstance(instrument_model, PassiveOpticalScannerModel):
                    instrument_fov : ViewGeometry = instrument_model.get_field_of_view()
                    instrument_fov_geometry : SphericalGeometry = instrument_fov.sph_geom
                    if instrument_fov_geometry.shape == 'RECTANGULAR':
                        cross_track_fov.append(instrument_fov_geometry.angle_width)
                    else:
                        raise NotImplementedError(f'Extraction of FOV for instruments with view geometry of shape `{instrument_fov_geometry.shape}` not yet implemented.')
                else:
                    raise NotImplementedError(f'measurement data query not yet suported for sensor models of type {type(instrument_model)}.')
            cross_track_fovs[instrument.name] = max(cross_track_fov)

        return cross_track_fovs

    def _collect_agility_specs(self, specs : Spacecraft) -> Tuple[float, float]:
        """ get pointing agility specifications from agent specs object """

        # validate inputs
        if not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `Spacecraft`. Is of type `{type(specs)}`.')

        # get attitude determination and control specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        if adcs_specs is None: raise ValueError('ADCS component specifications missing from agent specs object.')

        # get pointing agility specifications
        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None

        # return pointing agility specifications
        return max_slew_rate, max_torque

    
    def is_observation_path_valid(self, 
                                  state : SimulationAgentState, 
                                  observations : List[ObservationAction],
                                  max_slew_rate : float = None,
                                  max_torque : float = None,
                                  specs : object = None,
                                  ) -> bool:
        """ Checks if a given sequence of observations can be performed by a given agent """
        try:
            # Validate inputs
            assert isinstance(observations, list), "Observations must be a list."
            assert all(isinstance(obs, ObservationAction) for obs in observations), "All elements in observations must be of type ObservationAction."
            observations : list[ObservationAction] = observations

            if isinstance(state, SatelliteAgentState) :
                # get pointing agility specifications                
                if max_slew_rate is None or max_torque is None:
                    if specs is None: raise ValueError('Either `specs` or both `max_slew_rate` and `max_torque` must be provided.')
                    max_slew_rate, max_torque = self._collect_agility_specs(specs)

                # validate agility specifications
                if max_slew_rate is None: raise ValueError('ADCS `maxRate` specification missing from agent specs object.')
                if max_torque is None: raise ValueError('ADCS `maxTorque` specification missing from agent specs object.')
                assert max_slew_rate > 0.0
                # assert max_torque > 0.0

                # construct observation sequence parameter list
                observation_parameters = []
                for j,observation_j in enumerate(observations):
                    # estimate the state of the agent at the given measurement
                    observation_j : ObservationAction
                    th_j = observation_j.look_angle
                    t_j = observation_j.t_start
                    d_j = observation_j.t_end - t_j

                    # compare to prior measurements
                    if j > 0: # there was a prior observation performed
                        # estimate the state of the agent at the prior mesurement
                        observation_i : ObservationAction = observations[j-1]
                        th_i = observation_i.look_angle
                        t_i = observation_i.t_start
                        d_i = observation_i.t_end - t_i

                    else: # there was no prior measurement
                        # use agent's current state as previous state
                        th_i = state.attitude[0]
                        t_i = state._t
                        d_i = 0.0

                    observation_parameters.append((t_i, d_i, th_i, t_j, d_j, th_j, max_slew_rate))

                # check if observations sequence is valid
                if any([not self.is_observation_pair_valid(*params) 
                            for params in observation_parameters]):
                    for idx, params in enumerate(observation_parameters):
                        if not self.is_observation_pair_valid(*params):
                            x = 1   
                    return False

                # ensure no mutually exclusive tasks are present in observation sequence
                return all(
                    not obs_i.obs_opp.is_mutually_exclusive(obs_j.obs_opp)
                    for i, obs_i in enumerate(observations)
                    for j, obs_j in enumerate(observations)
                    if i < j
                )
            else:
                raise NotImplementedError(f'Observation path validity check for agents with state type {type(state)} not yet implemented.')
        finally:
            # DEBUG SECTION
            pass
            for pair_idx,(t_i,d_i,th_i,t_j,d_j,th_j,max_slew_rate) in enumerate(observation_parameters):
                if not self.is_observation_pair_valid(t_i, d_i, th_i, t_j, d_j, th_j, max_slew_rate):
                    x = 1

    def is_observation_pair_valid(self, 
                                  t_i, d_i, th_i, 
                                  t_j, d_j, th_j,
                                  max_slew_rate) -> bool:
        # check inputs
        assert not np.isnan(th_j) and not np.isnan(th_i) # TODO: add case where the target is not visible by the agent at the desired time according to the precalculated orbitdata

        # calculate maneuver time betweem states
        dt_maneuver = abs(th_j - th_i) / max_slew_rate
        
        # calculate time between measuremnets
        dt_measurements = t_j - (t_i + d_i)

        return ((dt_measurements > dt_maneuver 
                or abs(dt_measurements - dt_maneuver) < 1e-6)   # there is enough time to maneuver
                and dt_measurements >= -1e-6)                   # measurement time is after the previous measurement
                 
    # def _print_observation_sequence(self, 
    #                                 state : SatelliteAgentState, 
    #                                 path : list, 
    #                                 orbitdata : OrbitData = None
    #                                 ) -> None :
    #     """ Debugging tool. Prints current observation sequence being considered. """

    #     if not isinstance(state, SatelliteAgentState):
    #         raise NotImplementedError('Observation sequence printouts for non-satellite agents not yet supported.')
    #     elif orbitdata is None:
    #         raise ValueError(f'`orbitdata` required for agents of type `{type(state)}`.')

    #     out = f'\n{state.agent_name}:\n\n\ntarget\tinstr\tt_img\tth\tdt_mmt\tdt_mvr\tValid?\n'

    #     out_temp = [f"N\A       ",
    #                 f"N\A",
    #                 f"\t{np.round(state.t,3)}",
    #                 f"\t{np.round(state.attitude[0],3)}",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\t-",
    #                 f"\n"
    #                 ]
    #     out += ''.join(out_temp)

    #     for i in range(len(path)):
    #         if i > 0:
    #             measurement_prev : ObservationAction = path[i-1]
    #             t_prev = measurement_i.t_end
    #             lat,lon,_ = measurement_prev.target
    #             obs_prev = orbitdata.get_groundpoint_access_data(lat, lon, measurement_prev.instrument_name, t_prev)
    #             th_prev = obs_prev['look angle [deg]']
    #         else:
    #             t_prev = state.t
    #             th_prev = state.attitude[0]

    #         measurement_i : ObservationAction = path[i]
    #         t_i = measurement_i.t_start
    #         lat,lon,alt = measurement_i.target
    #         obs_i = orbitdata.get_groundpoint_access_data(lat, lon, measurement_i.instrument_name, t_i)
    #         th_i = obs_i['look angle [deg]']

    #         dt_maneuver = abs(th_i - th_prev) / state.max_slew_rate
    #         dt_measurements = t_i - t_prev

    #         out_temp = [f"({round(lat,3)}, {round(lon,3)}, {round(alt,3)})",
    #                         f"  {measurement_i.instrument_name}",
    #                         f"\t{np.round(measurement_i.t_start,3)}",
    #                         f"\t{np.round(th_i,3)}",
    #                         f"\t{np.round(dt_measurements,3)}",
    #                         f"\t{np.round(dt_maneuver,3)}",
    #                         f"\t{dt_maneuver <= dt_measurements}",
    #                         f"\n"
    #                         ]
    #         out += ''.join(out_temp)
    #     out += f'\nn measurements: {len(path)}\n'

    #     print(out)
