from typing import Dict, List, Set
import numpy as np

from execsatm.tasks import EventObservationTask, GenericObservationTask
from execsatm.observations import ObservationOpportunity
from execsatm.mission import Mission

from dmas.models.actions import ObservationAction
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

        # (task, n_obs) → Set[(partner_task, p_n_obs)] of ALL partner bids whose
        # co-obs window overlaps with the task's access window at the last planning
        # phase.  Superset of _coalition_deps: includes partners that were considered
        # but did not shift t_img.  Used by _release_stale_bundle_items to avoid
        # re-triggering re-evaluations for partners the agent already decided to
        # ignore (preventing infinite re-evaluation loops).
        self._reachable_partners: Dict = {}

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
                            t_img: float
                           ) -> dict:
        """Build the co_obs dict for _estimate_task_value.

        Implements the coalition definition:
          - Repeat: if any prior committed observation of the same parameter
            type X exists within t_corr (t_prior < t_img, t_img - t_prior <=
            t_corr), this observation adds no new coverage → returns {} (n_co=0).
          - Not a repeat: returns {param_Y: t_prior} for each distinct partner
            parameter type Y ≠ X that has at least one qualifying prior committed
            bid.  n_co = len(co_obs) and r_co is computed by the mission.

        Only backward-in-time prior observations qualify (t_prior < t_img), so
        the first instrument to commit for an event receives no co-obs bonus and
        modifying a later bid never retroactively changes an earlier one.
        The most-recent qualifying t_prior is kept per partner type.
        Returns {} for non-event tasks or tasks with no associated event.
        """
        if not isinstance(task, EventObservationTask) or task.event is None:
            return {}

        co_obs_window = self._co_obs_window_for(task)
        event_tasks = self._event_to_tasks_by_parameter.get(task.event.id, {})

        # Repeat check: if any prior committed observation of the same parameter
        # type exists within the co-obs window, this observation adds no new
        # parameter coverage → n_co = 0, return empty co_obs dict.
        for same_param_task in event_tasks.get(task.parameter, []):
            for bid in self._results.get(same_param_task, []):
                if (bid.has_winner()
                        and bid.t_img < t_img
                        and (t_img - bid.t_img) <= co_obs_window):
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
                            and bid.t_img < t_img
                            and (t_img - bid.t_img) <= co_obs_window):
                        if param not in co_obs or bid.t_img < co_obs[param]:
                            co_obs[param] = bid.t_img
        return co_obs

    # ------------------------------------------------------------------
    # COALITION DEP REBUILD — extended with reachable-partner tracking
    # ------------------------------------------------------------------

    def _rebuild_coalition_deps(self, state: SimulationAgentState) -> None:
        """Rebuild coalition deps then populate _reachable_partners.

        Calls the base _rebuild_coalition_deps to record actual co-obs deps
        (partners within co_obs_window of the committed t_img), then iterates
        _bundle to record ALL geometrically reachable partners per bundle entry.
        _release_stale_bundle_items checks against the union of both sets so
        that partners already evaluated-but-rejected don't trigger infinite
        re-evaluation loops.
        """
        super()._rebuild_coalition_deps(state)
        self._rebuild_reachable_partners()

    def _rebuild_reachable_partners(self) -> None:
        """Populate _reachable_partners from the current _bundle state.

        For each owned bundle entry (task, n_obs), records all partner bids
        whose co-obs window [t_partner, t_partner + co_obs_window] overlaps
        with the task's access window [t_access_l, t_access_u].  This superset
        of _coalition_deps is used by _release_stale_bundle_items to skip
        partners that were already evaluated at the last planning phase.
        """
        self._reachable_partners.clear()
        for obs_opp, task_dict in self._bundle:
            for task, n_obs in task_dict.items():
                if not isinstance(task, EventObservationTask) or task.event is None:
                    continue
                if n_obs >= len(self._results.get(task, [])):
                    continue
                bid = self._results[task][n_obs]
                if not bid.has_winner():
                    continue

                t_access = obs_opp.task_accessibility.get(task.id)
                if t_access is None:
                    continue
                t_access_l = t_access.left
                t_access_u = t_access.right
                co_obs_window = self._co_obs_window_for(task)
                event_tasks = self._event_to_tasks_by_parameter.get(task.event.id, {})

                reachable: Set = set()
                for param, partner_tasks in event_tasks.items():
                    if param == task.parameter:
                        continue
                    for partner_task in partner_tasks:
                        for p_n_obs, pbid in enumerate(self._results.get(partner_task, [])):
                            if (pbid.has_winner()
                                    # and not pbid.was_performed()
                                    and pbid.t_img + co_obs_window > t_access_l
                                    and pbid.t_img < t_access_u):
                                reachable.add((partner_task, p_n_obs))

                if reachable:
                    self._reachable_partners[(task, n_obs)] = reachable

    def _release_stale_bundle_items(self, state: SimulationAgentState) -> list:
        """Release bundle items whose co-obs context has improved since last planning.

        After consensus updates _results, scans _bundle for EventObservationTask entries
        won by this agent that now have geometrically-reachable co-obs partner bids which
        weren't present when _coalition_deps / _reachable_partners were last built.
        Removes their ObservationOpportunity from _bundle and _path and resets the
        winning bid so the planning phase re-evaluates and may commit a co-obs-adjusted
        time and raised bid value.

        The trigger uses the access window ([t_access_l, t_access_u]) so that partners
        requiring a time shift are detected, not only partners already within co_obs_window
        of the committed t_img.  _reachable_partners prevents infinite re-evaluation loops
        for partners the agent already considered-but-rejected at the last planning phase.
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
                event_tasks = self._event_to_tasks_by_parameter.get(task.event.id, {})

                # Access window for this task within this obs_opp — used to decide
                # whether a partner bid is geometrically reachable for co-obs, independent
                # of where t_img currently sits.
                t_access = obs_opp.task_accessibility.get(task.id)
                if t_access is None:
                    continue
                t_access_l = t_access.left
                t_access_u = t_access.right

                # Repeat check: same parameter type already committed strictly before
                # t_img within the window → n_co=0; new partners cannot help.
                # Repeat check: same parameter type committed/performed strictly before
                # t_img within the window → n_co=0; new partners cannot help.
                is_repeat = any(
                    bid_same.has_winner()
                    and bid_same.t_img < t_img
                    and (t_img - bid_same.t_img) <= co_obs_window
                    for same_param_task in event_tasks.get(task.parameter, [])
                    for bid_same in self._results.get(same_param_task, [])
                )
                if is_repeat:
                    continue

                # Partners already seen at the last planning phase — union of actual
                # coalition deps and all geometrically-reachable partners.  Checking
                # against both prevents re-triggering for partners the agent already
                # evaluated-but-rejected (i.e., co-obs bonus wasn't worth shifting for).
                already_considered = (
                    self._coalition_deps.get((task, n_obs), set())
                    | self._reachable_partners.get((task, n_obs), set())
                )

                new_partner_found = False
                for param, partner_tasks in event_tasks.items():
                    if param == task.parameter:
                        continue
                    for partner_task in partner_tasks:
                        for p_n_obs, pbid in enumerate(self._results.get(partner_task, [])):
                            # A partner bid (committed or performed) qualifies if its
                            # co-obs window overlaps with our access window.  Performed
                            # bids are included: a performed observation is a confirmed
                            # partner and may be a first-time arrival via propagation.
                            if (pbid.has_winner()
                                    and pbid.t_img + co_obs_window > t_access_l
                                    and pbid.t_img < t_access_u
                                    and (partner_task, p_n_obs) not in already_considered):
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
    
    def _find_best_imaging_time(self, task, instrument_name, th_img, t_img_l, t_img_u,
                             duration, specs, cross_track_fovs, orbitdata, mission,
                             n_obs, t_prev, n_grid=7, tol=1.0):
        """Override to inject partner bid times as candidate evaluation points."""
        co_obs_window = self._co_obs_window_for(task)
        is_event_task = co_obs_window > 0 and isinstance(task, EventObservationTask) and task.event is not None

        # collect partner t_img values to avoid in the base search
        partner_t_imgs = set()
        if is_event_task:
            event_tasks = self._event_to_tasks_by_parameter.get(task.event.id, {})
            for param, partner_tasks in event_tasks.items():
                if param == task.parameter:
                    continue
                for partner_task in partner_tasks:
                    for bid in self._results.get(partner_task, []):
                        if bid.has_winner() and t_img_l <= bid.t_img <= t_img_u:
                            partner_t_imgs.add(bid.t_img)

        if not partner_t_imgs:
            # no partners in range — base search covers full interval as before
            return super()._find_best_imaging_time(
                task, instrument_name, th_img, t_img_l, t_img_u,
                duration, specs, cross_track_fovs, orbitdata, mission,
                n_obs, t_prev, n_grid, tol
            )

        # build exclusion zones: [t_partner - tol, t_partner + tol]
        # split [t_img_l, t_img_u] into sub-intervals that avoid those zones
        exclusion_points = sorted(partner_t_imgs)
        intervals = []
        lo = t_img_l
        for t_excl in exclusion_points:
            hi = t_excl - tol
            if hi > lo:
                intervals.append((lo, hi))
            lo = t_excl + tol
        if lo < t_img_u:
            intervals.append((lo, t_img_u))

        # search each sub-interval and take the best non-simultaneous base result
        best_value, best_t = np.NINF, None
        for sub_lo, sub_hi in intervals:
            val, t = super()._find_best_imaging_time(
                task, instrument_name, th_img, sub_lo, sub_hi,
                duration, specs, cross_track_fovs, orbitdata, mission,
                n_obs, t_prev, n_grid, tol=tol
            )
            if val > best_value:
                best_value = val
                best_t = t

        # now search partner-anchored windows for co-obs bonus
        for t_cand in exclusion_points:
            sub_lo = max(t_img_l, t_cand + tol)
            sub_hi = min(t_img_u, t_cand + co_obs_window)
            if sub_hi <= sub_lo:
                continue
            val, t = super()._find_best_imaging_time(
                task, instrument_name, th_img, sub_lo, sub_hi,
                duration, specs, cross_track_fovs, orbitdata, mission,
                n_obs, t_prev, n_grid, tol
            )
            if val > best_value:
                best_value = val
                best_t = t

        return best_value, best_t
    
    # def _find_best_imaging_time(self, task, instrument_name, th_img, t_img_l, t_img_u,
    #                          duration, specs, cross_track_fovs, orbitdata, mission,
    #                          n_obs, t_prev, n_grid=7, tol=1e-3):
    #     """Override to inject partner bid times as candidate evaluation points."""
    #     # Get base result first
    #     base_value, base_t = super()._find_best_imaging_time(
    #         task, instrument_name, th_img, t_img_l, t_img_u,
    #         duration, specs, cross_track_fovs, orbitdata, mission,
    #         n_obs, t_prev, n_grid, tol=1.0
    #     )

    #     # If no co-obs requirement, return base result immediately
    #     co_obs_window = self._co_obs_window_for(task)
    #     if co_obs_window <= 0 or not isinstance(task, EventObservationTask) or task.event is None:
    #         return base_value, base_t

    #     # Collect candidate t_img values just after each partner bid
    #     # (partner_t_prior + epsilon) so t_img is strictly after partner
    #     # and within the co_obs_window
    #     partner_candidates = []
    #     event_tasks = self._event_to_tasks_by_parameter.get(task.event.id, {})
    #     for param, partner_tasks in event_tasks.items():
    #         if param == task.parameter:
    #             continue
    #         for partner_task in partner_tasks:
    #             for bid in self._results.get(partner_task, []):
    #                 # if not bid.has_winner() or bid.was_performed():
    #                 if not bid.has_winner():
    #                     continue
    #                 # candidate: image just after partner, within window
    #                 t_candidate = bid.t_img
    #                 if (t_candidate <= t_img_u 
    #                     and t_img_l <= bid.t_img + co_obs_window):
    #                     partner_candidates.append(t_candidate)

    #     if not partner_candidates:
    #         return base_value, base_t
        
    #     # base_co = self._build_co_obs_dict(task, base_t)

    #     # Evaluate at each partner-anchored candidate
    #     best_value, best_t = base_value, base_t
    #     for t_cand in partner_candidates:
    #         # refine locally around this candidate
    #         lo = max(t_img_l, t_cand + tol) # prevent simultaneous `t_img` with partner
    #         hi = min(t_img_u, t_cand + co_obs_window)
    #         if hi <= lo:
    #             continue
    #         val, t = super()._find_best_imaging_time(
    #             task, instrument_name, th_img, lo, hi,
    #             duration, specs, cross_track_fovs, orbitdata, mission,
    #             n_obs, t_prev, n_grid, tol
    #         )
    #         if val > best_value:
    #             best_value = val
    #             best_t = t

    #     return best_value, best_t

    # def _find_best_imaging_time(self, task, instrument_name, th_img, t_img_l, t_img_u,
    #                          duration, specs, cross_track_fovs, orbitdata, mission,
    #                          n_obs, t_prev, n_grid=7, tol=1.0):
    #     """Override to inject partner bid times as candidate evaluation points."""

    #     # get base result (no co-obs awareness)
    #     base_value, base_t = super()._find_best_imaging_time(
    #         task, instrument_name, th_img, t_img_l, t_img_u,
    #         duration, specs, cross_track_fovs, orbitdata, mission,
    #         n_obs, t_prev, n_grid, tol
    #     )

    #     co_obs_window = self._co_obs_window_for(task)
    #     if co_obs_window <= 0 or not isinstance(task, EventObservationTask) or task.event is None:
    #         return base_value, base_t

    #     # collect partner bid times
    #     partner_t_imgs = []
    #     event_tasks = self._event_to_tasks_by_instrument.get(task.event.id, {})
    #     for param, partner_tasks in event_tasks.items():
    #         if param == task.parameter:
    #             continue
    #         for partner_task in partner_tasks:
    #             for bid in self._results.get(partner_task, []):
    #                 # if bid.has_winner() and not bid.was_performed():
    #                 #     partner_t_imgs.append(bid.t_img)
    #                 # elif bid.has_winner():
    #                 #     x = 1
    #                 if bid.has_winner() and t_prev < bid.t_img <= t_img_u:
    #                     partner_t_imgs.append(bid.t_img)

    #     if not partner_t_imgs:
    #         return base_value, base_t

    #     best_value, best_t = base_value, base_t

    #     for t_partner in partner_t_imgs:
    #         # co-obs window starts strictly after the partner observation
    #         # sample uniformly within [t_partner + tol, t_partner + co_obs_window]
    #         # clamped to the feasible access interval [t_img_l, t_img_u]
    #         window_lo = max(t_img_l, t_partner)
    #         window_hi = min(t_img_u, t_partner + co_obs_window)

    #         if window_hi <= window_lo:
    #             continue

    #         # evaluate on a fine grid over the co-obs window
    #         # use n_grid points so the sample density matches the base search
    #         candidates = [window_lo + i * (window_hi - window_lo) / (n_grid - 1)
    #                     for i in range(n_grid)]

    #         for t_cand in candidates:
    #             val = self._estimate_task_value(
    #                 task, instrument_name, th_img, t_cand, duration,
    #                 specs, cross_track_fovs, orbitdata, mission, n_obs, t_prev
    #             )
    #             if val > best_value:
    #                 best_value = val
    #                 best_t = t_cand

    #     return best_value, best_t
