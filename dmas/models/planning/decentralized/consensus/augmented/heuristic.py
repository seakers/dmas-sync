from typing import Dict, List, Set
import numpy as np

from execsatm.tasks import EventObservationTask, GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission

from dmas.models.planning.decentralized.consensus.heuristic import HeuristicInsertionConsensusPlanner
from dmas.models.planning.decentralized.consensus.augmented.consensus import AugmentedConsensusPlanner
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

    The only behavioral difference from the base heuristic planner is in
    _estimate_task_value: when this agent evaluates a candidate bid for
    task T at time t, it builds a co_obs dict from already-committed
    partner bids (same event, different instrument, within th0e co-obs
    window) and passes it to the base implementation. mission.calc_task_value
    uses that dict to return a higher value when valid co-observations exist.

    The co-obs bonus accrues sequentially: the first instrument to commit
    for an event gets no bonus; the second sees one committed partner and
    gets a partial boost; the third sees two and gets a larger boost.
    No future or hypothetical bids are considered — only what is already
    in self._results at bid-evaluation time.

    Coalition re-evaluation:
      After each consensus phase, _release_stale_bundle_items scans _bundle for
      tasks that now have new qualifying co-obs partner bids that weren't present
      when _coalition_deps was last built. Those observation opportunities are
      removed from _bundle and _path (without resetting the bid) so the planning
      phase re-evaluates them with the updated co-obs context and can raise the
      bid to capture the bonus. The backward-time asymmetry (only partners strictly
      before t_img qualify) bounds cascade depth to one direction.
    """

    def __init__(self,
                 agent_results_dir: str,
                 heuristic: str = HeuristicInsertionConsensusPlanner.EARLIEST_ACCESS,
                 replan_threshold: int = 1,
                 optimistic_bidding_threshold: int = 1,
                 periodic_overwrite: bool = False,
                 debug: bool = False,
                 logger: bool = None,
                 printouts: bool = True,
                 co_obs_window: float = 300.0,
                 **kwargs
                ):
        """
        Parameters
        ----------
        co_obs_window : float
            Maximum allowable time delta [s] between two observations of
            different instrument types to qualify as a co-observation.
            Default: 300 s (5 min). Set to match mission requirements.
        contested_reset_threshold : int
            Number of times a (task, n_obs) slot may be cascade-reset within
            a single sim timestep before it is frozen. Prevents infinite
            oscillation between agents. Default: 3.
        """
        # co_obs_window passed as keyword so AugmentedConsensusPlanner picks it
        # up via its *args/**kwargs __init__ without disturbing HeuristicInsertionConsensusPlanner's
        # positional argument order.
        super().__init__(agent_results_dir, heuristic, replan_threshold,
                         optimistic_bidding_threshold, periodic_overwrite,
                         debug, logger, printouts,
                         co_obs_window=co_obs_window, **kwargs)

    def _estimate_task_value(self,
                              task: GenericObservationTask,
                              instrument_name: str,
                              th_img: float,
                              t_img: float,
                              d_img: float,
                              specs: object,
                              cross_track_fovs: dict,
                              orbitdata: OrbitData,
                              mission: Mission,
                              n_obs: int = 0,
                              t_prev: float = np.NINF
                             ) -> float:
        """Estimate task value with co-observation context.

        Builds a co_obs dict from already-committed partner bids in
        self._results and passes it to the base implementation, which
        injects it into the measurement performance dict for
        mission.calc_task_value to evaluate.
        """
        co_obs = self._build_co_obs_dict(task, t_img)
        return super()._estimate_task_value(
            task, instrument_name, th_img, t_img, d_img,
            specs, cross_track_fovs, orbitdata, mission,
            n_obs, t_prev, co_obs=co_obs
        )

    def _build_co_obs_dict(self,
                            task: GenericObservationTask,
                            t_obs: float
                           ) -> dict:
        """Build the co_obs dict for _estimate_task_value.

        Implements the coalition definition:
          - Repeat: if any prior committed observation of the same parameter
            type X exists within t_corr (t_prior < t_obs, t_obs - t_prior <=
            t_corr), this observation adds no new coverage → returns {} (n_co=0).
          - Not a repeat: returns {param_Y: t_prior} for each distinct partner
            parameter type Y ≠ X that has at least one qualifying prior committed
            bid.  n_co = len(co_obs) and r_co is computed by the mission.

        Only backward-in-time prior observations qualify (t_prior < t_obs), so
        the first instrument to commit for an event receives no co-obs bonus and
        modifying a later bid never retroactively changes an earlier one.
        The most-recent qualifying t_prior is kept per partner type.
        Returns {} for non-event tasks or tasks with no associated event.
        """
        if not isinstance(task, EventObservationTask) or task.event is None:
            return {}

        co_obs_window = self._co_obs_window_for(task)
        event_tasks = self._event_to_tasks_by_instrument.get(task.event.id, {})

        # Repeat check: if any prior committed observation of the same parameter
        # type exists within the co-obs window, this observation adds no new
        # parameter coverage → n_co = 0, return empty co_obs dict.
        for same_param_task in event_tasks.get(task.parameter, []):
            for bid in self._results.get(same_param_task, []):
                if (bid.has_winner()
                        and not bid.was_performed()
                        and bid.t_img < t_obs
                        and (t_obs - bid.t_img) <= co_obs_window):
                    return {}

        # Not a repeat: one entry per distinct partner parameter type Y ≠ X.
        # Only existence matters; keep the most-recent qualifying t_prior per type.
        co_obs: Dict[str, float] = {}
        for param, partner_tasks in event_tasks.items():
            if param == task.parameter:
                continue
            for partner_task in partner_tasks:
                for bid in self._results.get(partner_task, []):
                    if (bid.has_winner()
                            and not bid.was_performed()
                            and bid.t_img < t_obs
                            and (t_obs - bid.t_img) <= co_obs_window):
                        if param not in co_obs or bid.t_img > co_obs[param]:
                            co_obs[param] = bid.t_img
        return co_obs

    def _release_stale_bundle_items(self, state: SimulationAgentState) -> list:
        """Release bundle items whose co-obs context has improved since last planning.

        After consensus updates _results, scans _bundle for EventObservationTask entries
        won by this agent that now have qualifying co-obs partner bids which weren't
        present when _coalition_deps was last built. Removes their ObservationOpportunity
        from _bundle and _path so the planning phase re-evaluates them with the updated
        co-obs context (via _build_co_obs_dict) and raises the bid to capture the bonus.

        The backward-time asymmetry (only partners strictly before t_img qualify) means
        only the later observer re-evaluates, bounding cascade depth to one direction.
        No bid is reset — the agent retains its current winning bid until the planning
        phase produces a higher one.
        """
        obs_opps_to_release: Set[ObservationOpportunity] = set()

        for obs_opp, task_dict in self._bundle:
            for task, n_obs in task_dict.items():
                if not isinstance(task, EventObservationTask) or task.event is None:
                    continue
                if n_obs >= len(self._results.get(task, [])):
                    continue
                bid = self._results[task][n_obs]
                if not bid.has_winner() or bid.winner != state.agent_name:
                    continue

                t_img = bid.t_img
                co_obs_window = self._co_obs_window_for(task)
                event_tasks = self._event_to_tasks_by_instrument.get(task.event.id, {})

                # Repeat check: same parameter type already committed before t_img
                # within the window → n_co=0 regardless of partner bids; new partners
                # cannot change the value, so skip release.
                is_repeat = any(
                    bid_same.has_winner()
                    and not bid_same.was_performed()
                    and bid_same.t_img < t_img
                    and (t_img - bid_same.t_img) <= co_obs_window
                    for same_param_task in event_tasks.get(task.parameter, [])
                    for bid_same in self._results.get(same_param_task, [])
                )
                if is_repeat:
                    continue

                known_deps = self._coalition_deps.get((task, n_obs), set())

                new_partner_found = False
                for param, partner_tasks in event_tasks.items():
                    if param == task.parameter:
                        continue
                    for partner_task in partner_tasks:
                        for p_n_obs, pbid in enumerate(self._results.get(partner_task, [])):
                            if (pbid.has_winner()
                                    and not pbid.was_performed()
                                    and pbid.t_img < t_img
                                    and (t_img - pbid.t_img) <= co_obs_window
                                    and (partner_task, p_n_obs) not in known_deps):
                                new_partner_found = True
                                break
                        if new_partner_found:
                            break
                    if new_partner_found:
                        break

                if new_partner_found:
                    obs_opps_to_release.add(obs_opp)

        if not obs_opps_to_release:
            return []

        # collect (task, n_obs, bid) triples for the released entries so we can
        # reset them after filtering _bundle; must be gathered before the filter
        # so we still have the original bundle to iterate over.
        released_info = [
            (task, n_obs, self._results[task][n_obs])
            for opp, task_dict in self._bundle
            if opp in obs_opps_to_release
            for task, n_obs in task_dict.items()
            if n_obs < len(self._results.get(task, []))
        ]

        # drop released entries from bundle and path; planning phase re-evaluates them
        self._bundle = [(opp, td) for opp, td in self._bundle
                        if opp not in obs_opps_to_release]
        self._path = [action for action in self._path
                      if action.obs_opp not in obs_opps_to_release]

        # reset released bids in _results so they are no longer orphaned winning bids
        t_curr = state.get_time()
        for task, n_obs, bid in released_info:
            if bid.is_bidder_winning() and not bid.was_performed():
                bid.reset(t_curr)
                self._results[task][n_obs] = bid

        # released bids
        return [bid for _, _, bid in released_info]