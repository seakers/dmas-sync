from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

from execsatm.tasks import EventObservationTask, GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission

from dmas.models.planning.decentralized.consensus.heuristic import HeuristicInsertionConsensusPlanner
from dmas.models.planning.decentralized.consensus.augmented.consensus import AugmentedConsensusPlanner
from dmas.models.actions import ObservationAction
from dmas.models.trackers import TaskObservationTracker
from dmas.models.planning.plan import Plan
from dmas.models.planning.decentralized.consensus.bids import Bid
from dmas.models.states import SimulationAgentState
from dmas.utils.orbitdata import OrbitData


class AugmentedHeuristicInsertionConsensusPlanner(HeuristicInsertionConsensusPlanner,
                                                   AugmentedConsensusPlanner):
    """
    Augments HeuristicInsertionConsensusPlanner with co-observation value.

    MRO (Python C3):
      AugmentedHeuristicInsertionConsensusPlanner
        -> HeuristicInsertionConsensusPlanner
        -> AugmentedConsensusPlanner
        -> ConsensusPlanner
        -> AbstractReactivePlanner

    Two concrete additions over the base heuristic planner:
    1. modified_tasks is expanded to include co-obs partner tasks so that
       their sequences are re-evaluated whenever a related task changes.
    2. The bid-acceptance check is coalition-aware: a new bid for task T at
       time t must not only beat the existing bid for that slot, but also
       must not reduce the aggregate coalition value (existing bid +
       partner bids) compared to what is already committed.
    """

    def __init__(self,
                 agent_results_dir: str,
                 co_obs_window: float = 300.0,
                 heuristic: str = HeuristicInsertionConsensusPlanner.EARLIEST_ACCESS,
                 replan_threshold: int = 1,
                 optimistic_bidding_threshold: int = 1,
                 periodic_overwrite: bool = False,
                 debug: bool = False,
                 logger: bool = None,
                 printouts: bool = True
                ):
        """
        Parameters
        ----------
        co_obs_window : float
            Maximum allowable time delta [s] between two observations of
            different instrument types to count as a co-observation.
            Default: 300 s (5 min). Override to match mission requirements.
        """
        super().__init__(agent_results_dir, heuristic, replan_threshold,
                         optimistic_bidding_threshold, periodic_overwrite,
                         debug, logger, printouts)
        self._co_obs_window: float = co_obs_window

    # ------------------------------------------------------------------
    # AUGMENTED BUNDLE-BUILDING HELPERS
    # ------------------------------------------------------------------

    def _extend_modified_tasks(self,
                                modified_tasks: Set[GenericObservationTask]
                               ) -> Set[GenericObservationTask]:
        """Expand modified_tasks to include co-obs partner tasks.

        When a task belonging to event E changes, all tasks derived from E
        but observed by a different instrument must also be re-evaluated,
        because the co-obs coalition value of their sequences may shift.
        """
        partner_tasks: Set[GenericObservationTask] = set()
        for task in modified_tasks:
            if not isinstance(task, EventObservationTask) or task.event is None:
                continue
            for param, tasks in self._event_to_tasks_by_instrument.get(task.event.id, {}).items():
                if param != task.parameter:
                    partner_tasks.update(tasks)
        return modified_tasks.union(partner_tasks)

    def _coalition_accept_bid(self,
                               task: GenericObservationTask,
                               n_obs: int,
                               task_value: float,
                               t_obs: float,
                               state: SimulationAgentState,
                               proposed_bids: Dict
                              ) -> List[bool]:
        """Return the list of acceptance conditions for a proposed bid.

        Augments the base two-condition check (beats current / optimistic)
        with a coalition-value neutrality requirement: accepting the new bid
        must not reduce the aggregate coalition value relative to the current
        commitment.

        Conditions (any one being True → bid is accepted):
          1. This agent already holds the slot and value is positive.
          2. New bid beats the current slot winner AND the net coalition value
             does not decrease (no regression for the system as a whole).
          3. Proposed time is earlier AND optimistic counter allows it.
        """
        if n_obs >= len(self._results[task]):
            return [task_value > 0.0]

        try:
            existing_bid: Bid = self._results[task][n_obs]
        except KeyError:
            existing_bid: Bid = proposed_bids[task][n_obs]

        # --- base acceptance conditions ---
        beats_current = task_value > existing_bid.winning_bid

        immediate_next = next(
            (b for b in self._results[task]
             if b.n_obs == n_obs + 1 and b.winner != state.agent_name),
            None
        )
        beats_immediate_next = immediate_next is None or task_value > immediate_next.winning_bid

        # --- coalition neutrality (augmented) ---
        # Coalition value at the existing bid's time vs the proposed time.
        # A new bid at t_obs is acceptable if the total value (my bid +
        # remaining partner bids) does not drop below the current coalition.
        coalition_current = self._compute_coalition_value(task, existing_bid.t_img, self._co_obs_window)
        coalition_proposed = self._compute_coalition_value(task, t_obs, self._co_obs_window)
        net_change = (task_value + coalition_proposed) - (existing_bid.winning_bid + coalition_current)
        coalition_neutral = net_change >= 0.0

        return [
            # 1) already winning, positive value → always keep
            existing_bid.winner == state.agent_name and task_value > 0.0,
            # 2) beats current slot winner, immediate next bid, AND no coalition regression
            beats_current and beats_immediate_next and coalition_neutral,
            # 3) earlier observation time with optimistic counter
            t_obs < existing_bid.t_img and self._optimistic_bidding_counters[task][n_obs] > 0
        ]

    # ------------------------------------------------------------------
    # OVERRIDDEN BUNDLE-BUILDING PHASE METHOD
    # ------------------------------------------------------------------

    def _assign_best_observations_and_revisit_times_to_proposed_path(self,
                                                                     state: SimulationAgentState,
                                                                     candidate_path: List[ObservationAction],
                                                                     obs_added: List[ObservationAction],
                                                                     obs_removed: List[ObservationAction],
                                                                     proposed_bids: Dict[GenericObservationTask, Dict[int, Bid]],
                                                                     specs: object,
                                                                     cross_track_fovs: dict,
                                                                     orbitdata: OrbitData,
                                                                     mission: Mission,
                                                                     observation_history: TaskObservationTracker
                                                                    ) -> Tuple[Dict[int, Dict[GenericObservationTask, int]],
                                                                               Dict[int, Dict[GenericObservationTask, float]],
                                                                               Dict[ObservationOpportunity, Dict[GenericObservationTask, Bid]]]:
        """Generate best observation numbers and revisit times for each observation in the proposed path.

        Augmented over the base class version in two ways:
          [A] modified_tasks is extended to include co-obs partner tasks.
          [B] bid acceptance uses coalition-aware conditions.

        Everything else is identical to HeuristicInsertionConsensusPlanner.
        """
        t_curr = state.get_time()

        # extract modified task observation opportunities from path changes
        added_tasks: Set[GenericObservationTask] = \
            {task for obs_act in obs_added for task in obs_act.obs_opp.tasks}
        removed_tasks: Set[GenericObservationTask] = \
            {task for obs_act in obs_removed for task in obs_act.obs_opp.tasks}
        modified_tasks: Set[GenericObservationTask] = added_tasks.union(removed_tasks)

        # [A] AUGMENTED: expand modified_tasks to cascade into co-obs partner tasks
        modified_tasks = self._extend_modified_tasks(modified_tasks)

        modified_tasks_in_path: List[GenericObservationTask] = \
            sorted({task
                    for obs_act in candidate_path
                    for task in obs_act.obs_opp.tasks
                    if task in modified_tasks}, key=lambda x: x.id)

        # find observation time for proposed task in candidate path
        action_tasks_start_times = [
            action.obs_opp.get_earliest_starts(action.t_start)
            for action in candidate_path
        ]

        modified_task_obs_times: Dict[GenericObservationTask, List[Tuple[float, str, float, ObservationOpportunity]]] \
            = {task: [
                (start_times[task], state.agent_name, action.look_angle, action.obs_opp)
                for action, start_times in zip(candidate_path, action_tasks_start_times)
                if task in action.obs_opp.tasks
            ] for task in modified_tasks_in_path}

        # initialize best observation numbers and previous observation times
        n_obs_best: Dict[GenericObservationTask, list] = {task: [] for task in modified_tasks_in_path}
        t_img_best: Dict[GenericObservationTask, list] = {task: [] for task in modified_tasks_in_path}
        t_prev_best: Dict[GenericObservationTask, list] = {task: [] for task in modified_tasks_in_path}
        obs_names_best: Dict[GenericObservationTask, list] = {task: [] for task in modified_tasks_in_path}
        vals_best: Dict[GenericObservationTask, list] = {task: [] for task in modified_tasks_in_path}

        best_values: dict = {task: np.NINF for task in modified_tasks_in_path}

        # find best observation sequences for each parent task
        for task in modified_tasks_in_path:
            assert task in self._results, \
                f"Parent task {task} not being bid on by any agent; cannot generate bids."

            last_performed_bid = max([bid for bid in self._results[task] if bid.was_performed()],
                                     key=lambda bid: bid.n_obs, default=None)
            n_obs_last_performed = last_performed_bid.n_obs if last_performed_bid else -1

            latest_performed_obs_time: Tuple[float, str, float, ObservationOpportunity] \
                = (last_performed_bid.t_img, last_performed_bid.owner, np.NAN, None) if last_performed_bid else None

            available_obs_times: list = []

            scheduled_obs_times: list = \
                [(bid.t_img, bid.winner, np.NAN, None) for bid in self._results[task]
                 if bid.winner != state.agent_name and bid.has_winner()
                 and not bid.was_performed()
                 and bid.n_obs > n_obs_last_performed]

            available_obs_times.extend(scheduled_obs_times)
            available_obs_times.extend(modified_task_obs_times[task])
            available_obs_times.sort(key=lambda x: x[0])

            obs_times_for_agent = [t_img for t_img, agent_name, _, _ in available_obs_times
                                   if agent_name == state.agent_name]
            for t_img in obs_times_for_agent:
                if any(abs(t_img - other_t_img) <= self.EPS
                       for other_t_img in obs_times_for_agent if other_t_img != t_img):
                    raise ValueError(
                        f"Repeated observation time {t_img} found for agent '{state.agent_name}' "
                        f"when generating observation sequences for task '{task.id}'."
                    )

            feasible_sequences = self._find_feasible_observation_sequences_for_task(
                state, task, available_obs_times
            )

            for obs_names, obs_times, obs_look_angles, obs_tasks in feasible_sequences:
                seq_values = []
                t_prev_seq = []
                n_obs_seq = []
                is_sequence_valid = True

                for seq_idx, (agent_name, t_obs, look_angle, obs_opp) in enumerate(
                        zip(obs_names, obs_times, obs_look_angles, obs_tasks)):

                    n_obs = seq_idx + (n_obs_last_performed + 1)
                    t_prev = (obs_times[seq_idx - 1] if seq_idx > 0
                              else latest_performed_obs_time[0] if last_performed_bid else np.NINF)

                    if n_obs > 0:
                        assert t_prev >= 0.0, \
                            "Previous observation time is not defined for observation number greater than zero."

                    if agent_name != state.agent_name:
                        matching_bid: Bid = self._results[task][n_obs]
                        assert matching_bid.winner == agent_name, \
                            "Matching bid winner does not match agent assigned to observation."
                        assert abs(matching_bid.t_img - t_obs) <= self.EPS, \
                            "Matching bid observation time does not match assigned observation time."
                        task_value = matching_bid.winning_bid
                    else:
                        assert isinstance(obs_opp, ObservationOpportunity), \
                            "Task observation opportunity not defined."

                        task_value = self._estimate_task_value(
                            task, obs_opp.instrument_name, look_angle, t_obs,
                            obs_opp.min_duration, specs, cross_track_fovs,
                            orbitdata, mission, n_obs, t_prev
                        )

                        # [B] AUGMENTED: coalition-aware bid acceptance
                        accept_bid = self._coalition_accept_bid(
                            task, n_obs, task_value, t_obs, state, proposed_bids
                        )

                        if not any(accept_bid):
                            is_sequence_valid = False
                            break

                    seq_values.append(task_value)
                    t_prev_seq.append(t_prev)
                    n_obs_seq.append(n_obs)

                if not is_sequence_valid:
                    continue

                if len(seq_values) != len(obs_times):
                    continue

                total_seq_value = sum(seq_values)

                if total_seq_value > best_values[task]:
                    best_values[task] = total_seq_value
                    n_obs_best[task] = n_obs_seq
                    t_img_best[task] = obs_times
                    t_prev_best[task] = t_prev_seq
                    obs_names_best[task] = obs_names
                    vals_best[task] = seq_values

        if any(value < 0.0 for value in best_values.values()):
            return None, None, None

        # filter out observations from other agents in best sequences
        indices_to_remove = {task: [idx for idx, agent_name in enumerate(obs_names_best[task])
                                    if agent_name != state.agent_name]
                             for task in modified_tasks_in_path}

        for task, indices in indices_to_remove.items():
            for idx in sorted(indices, reverse=True):
                n_obs_best[task].pop(idx)
                t_img_best[task].pop(idx)
                t_prev_best[task].pop(idx)
                obs_names_best[task].pop(idx)
                vals_best[task].pop(idx)

        assert all([all([agent_name == state.agent_name for agent_name in obs_names_best[task]])
                    for task in modified_tasks_in_path]), \
            "Not all observations from other agents were removed from best sequences."

        new_bids: Dict[ObservationOpportunity, Dict[GenericObservationTask, Bid]] = defaultdict(dict)
        n_obs_candidate = [dict() for _ in candidate_path]
        t_prev_candidate = [dict() for _ in candidate_path]

        for obs_idx, obs_act in enumerate(candidate_path):
            obs_t_img = obs_act.obs_opp.get_earliest_starts(obs_act.t_start)

            for task in obs_act.obs_opp.tasks:
                t_start = obs_t_img[task]

                if task in n_obs_best:
                    n_obs = n_obs_best[task].pop(0)
                    t_prev = t_prev_best[task].pop(0)
                    val = vals_best[task].pop(0)
                    t_img = t_img_best[task].pop(0)

                    if (t_img + obs_act.obs_opp.task_min_duration[task.id] < t_curr
                            or abs(t_img + obs_act.obs_opp.task_min_duration[task.id] - t_curr) < 1e-6):
                        continue
                    if t_img < t_curr:
                        matching_bids = [bid for bid in proposed_bids[task].values()
                                         if abs(bid.t_img - t_start) <= self.EPS
                                         and bid.owner == state.agent_name]

                        assert matching_bids, \
                            "Matching bid for observation in path not found in results. Was assigned without updating results."
                        assert len(matching_bids) <= 1, \
                            "There should be at most one matching bid for the current time step."

                        matching_bid: Bid = matching_bids.pop()

                        prev_bids_self = [bid for bid in proposed_bids[task].values()
                                          if bid.t_img < t_start]
                        previous_bids_other = [bid for bid in self._results[task]
                                               if bid.winner != state.agent_name
                                               and bid.t_img < t_start]
                        previous_bids_perf = [bid for bid in self._results[task]
                                              if bid.was_performed() and bid.t_img < t_start]
                        prev_bids = prev_bids_self + previous_bids_other + previous_bids_perf

                        assert matching_bid.n_obs == 0 or prev_bids, \
                            "Previous bids should exist for observation numbers greater than zero."

                        n_obs_candidate[obs_idx][task] = matching_bid.n_obs
                        t_prev_candidate[obs_idx][task] = max(
                            (bid.t_img for bid in prev_bids), default=np.NINF
                        )
                        if matching_bid.n_obs > 0:
                            assert t_prev_candidate[obs_idx][task] >= 0.0, \
                                "Previous observation time is not defined for observation number greater than zero."
                    else:
                        if n_obs > 0:
                            assert t_prev >= 0.0, \
                                "Previous observation time is not defined for observation number greater than zero."

                        new_bid = Bid(task, state.agent_name, n_obs, val, val, state.agent_name,
                                     t_img, t_curr, None, obs_act.instrument_name)
                        new_bids[obs_act.obs_opp][task] = new_bid
                        n_obs_candidate[obs_idx][task] = n_obs
                        t_prev_candidate[obs_idx][task] = t_prev

                elif task in proposed_bids:
                    matching_bids = [bid for bid in proposed_bids[task].values()
                                     if abs(bid.t_img - t_start) <= self.EPS
                                     and bid.owner == state.agent_name]

                    assert matching_bids, \
                        "Matching bid for observation in path not found in results. Was assigned without updating results."
                    assert len(matching_bids) <= 1, \
                        "There should be at most one matching bid for the current time step."

                    matching_bid: Bid = matching_bids.pop()

                    prev_bids_self = [bid for bid in proposed_bids[task].values()
                                      if bid.t_img < t_start]
                    previous_bids_other = [bid for bid in self._results[task]
                                           if bid.winner != state.agent_name
                                           and bid.t_img < t_start]
                    previous_bids_perf = [bid for bid in self._results[task]
                                          if bid.was_performed() and bid.t_img < t_start]
                    prev_bids = prev_bids_self + previous_bids_other + previous_bids_perf

                    assert matching_bid.n_obs == 0 or prev_bids, \
                        "Previous bids should exist for observation numbers greater than zero."

                    n_obs_candidate[obs_idx][task] = matching_bid.n_obs
                    t_prev_candidate[obs_idx][task] = max(
                        (bid.t_img for bid in prev_bids), default=np.NINF
                    )
                    if matching_bid.n_obs > 0:
                        assert t_prev_candidate[obs_idx][task] >= 0.0, \
                            "Previous observation time is not defined for observation number greater than zero."

                elif t_curr in obs_act.obs_opp.accessibility:
                    matching_bids = [bid for bid in self._results[task]
                                     if bid.t_img in obs_act.obs_opp.accessibility
                                     and bid.winner == state.agent_name
                                     and bid.was_performed()]
                    if not matching_bids:
                        raise ValueError(
                            f"No matching performed bids found for task {task} during current "
                            f"observation opportunity accessibility. Cannot assign observation "
                            f"number and previous observation time."
                        )
                    continue

                else:
                    raise ValueError(
                        f"Task {task} in proposed path does not have a best sequence or "
                        f"existing bid to assign observation number and previous observation time from."
                    )

        return n_obs_candidate, t_prev_candidate, new_bids
