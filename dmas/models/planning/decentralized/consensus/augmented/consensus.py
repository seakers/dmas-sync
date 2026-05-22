from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

from execsatm.tasks import EventObservationTask, GenericObservationTask
from execsatm.requirements import CoObservationRequirement
from execsatm.mission import Mission

from dmas.models.planning.decentralized.consensus.consensus import ConsensusPlanner
from dmas.models.planning.decentralized.consensus.bids import Bid
from dmas.models.states import SimulationAgentState
from dmas.models.planning.plan import Plan
from dmas.models.science.requests import TaskRequest
from dmas.models.actions import ObservationAction
from dmas.utils.orbitdata import OrbitData
from dmas.models.trackers import TaskObservationTracker


class AugmentedConsensusPlanner(ConsensusPlanner):
    """
    Extends ConsensusPlanner with co-observation awareness.

    New state beyond ConsensusPlanner:
      _co_obs_window : float
          Global fallback maximum time separation [s] between partner
          observations, used when a task has no CoObservationRequirement in
          its objective.  Passed as a keyword-only argument so it threads
          cleanly through Python's MRO without disturbing the positional
          argument order of ConsensusPlanner.

      _co_obs_windows : task_id → float
          Per-task co-observation window derived from each task's
          CoObservationRequirement.decorrelation_time.  Populated
          incrementally as tasks enter the event index and cleaned up when
          they expire.  Look up via _co_obs_window_for(task).

      _event_to_tasks_by_instrument : event_id → parameter → [task]
          Index rebuilt after each consensus round when the task registry
          changes.  Used during bundle-building to find partner tasks and
          during the coalition-dependency check.

      _coalition_deps : (task, n_obs) → set of (partner_task, partner_n_obs)
          Records which partner bids contributed co-obs value to each of this
          agent's committed winning bids.  Rebuilt at the end of every planning
          phase (generate_plan).  Consumed by _check_results_constraints to
          detect when a coalition has been broken by a partner bid being reset.

    Constraint-check extension:
      _check_results_constraints is overridden to run the base ordering check
      first, then append any co-obs coalition violations.  A winning bid is
      invalidated when any of its recorded partner bids has been reset,
      performed, or has drifted outside the co-obs window.  Invalidated bids
      are reset locally and broadcast as violations (same semantics as ordering
      violations), feeding into the existing while-True convergence loop in
      _consensus_phase.

    Inheritance note:
      *args/**kwargs pass-through in __init__ keeps this class transparent in
      multi-inheritance chains (e.g. MRO with HeuristicInsertionConsensusPlanner).
      _co_obs_window is extracted as a keyword-only argument before forwarding.
    """

    def __init__(self, *args, co_obs_window: float = 300.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._co_obs_window: float = co_obs_window

        # task_id -> co-obs window [s] from task's CoObservationRequirement
        self._co_obs_windows: Dict[str, float] = {}

        # event_id -> parameter -> List[EventObservationTask]
        self._event_to_tasks_by_instrument: Dict[str, Dict[str, List[EventObservationTask]]] = \
            defaultdict(lambda: defaultdict(list))

        # (task, n_obs) -> set of (partner_task, partner_n_obs) whose bids
        # contributed co-obs value to this agent's winning bid
        self._coalition_deps: Dict[Tuple[GenericObservationTask, int],
                                    Set[Tuple[GenericObservationTask, int]]] = {}

    # ------------------------------------------------------------------
    # CONSENSUS PHASE
    # ------------------------------------------------------------------

    def _consensus_phase(self,
                         state: SimulationAgentState,
                         incoming_reqs: List[TaskRequest],
                         incoming_bids: List[Union[Bid, dict]],
                         tasks: List[GenericObservationTask],
                         current_plan: Plan,
                         performed_observations: List[ObservationAction]
                        ) -> Tuple[List[Bid], List[Bid], List[Bid], List[ObservationAction]]:
        """Run base consensus phase, then incrementally maintain the co-obs event index.

        Both additions and removals are handled in O(|changed tasks|) using the
        explicit task lists surfaced by the extended _consensus_phase return:
          - task_updates    (index 0): bid dicts for newly announced tasks
          - expired_tasks   (index 4): task objects removed from _results

        The common case (no change) is zero cost.
        """
        consensus_results = super()._consensus_phase(
            state, incoming_reqs, incoming_bids, tasks, current_plan, performed_observations
        )
        # consensus_results <- task_updates, results_updates, bundle_updates, performed_bundle_observations, expired_tasks
        task_updates, *_, expired_tasks = consensus_results

        if expired_tasks:
            self._remove_from_event_index(expired_tasks)
        if task_updates:
            self._update_event_index(task_updates)

        return consensus_results

    # ------------------------------------------------------------------
    # CONSTRAINT CHECK — co-obs coalition validity
    # ------------------------------------------------------------------

    def _check_results_constraints(self,
                                    state: SimulationAgentState
                                   ) -> Tuple[List, List, List[Bid]]:
        """Base ordering constraint check + co-obs coalition validity check.

        After the standard ordering-violation check, inspects _coalition_deps
        for any of this agent's committed winning bids whose recorded partner
        bids have since been reset, performed, or drifted out of the co-obs
        window.  Invalidated bids (and all higher n_obs bids for the same task)
        are reset locally and returned as violations to be broadcast, feeding
        the existing convergence loop.

        The deps entry for an invalidated bid is removed immediately so that
        subsequent iterations of the convergence loop do not re-trigger the
        same violation.
        """
        # perform regular constraint checks for intra-task related constraints
        self._bundle, self._path, bids_in_violation  = super()._check_results_constraints(state)
        
        # update violations with any co-obs coalition constraint violations
        self._bundle, self._path, co_obs_bids_in_violation = self._check_co_obs_constraints(state)

        # combine violation lists (if any) and remove duplicates
        constraint_violations = bids_in_violation + co_obs_bids_in_violation

        # return updated bundle, path, and full violation list
        return self._bundle, self._path, constraint_violations

    def _check_co_obs_constraints(self,
                                   state: SimulationAgentState
                                  ) -> Tuple[List, List, List[dict]]:
        """Detect and reset winning bids whose coalition dependencies are broken.

        Returns a tuple of [bundle, path, list of bid dicts] (same format as base constraint violations)
        with the last conatinig an entry for each bid that was reset due to coalition invalidity.
        """
        # get current simulation time
        t_curr = state.get_time()

        # initialize list of bids to invalidate
        to_invalidate: List[Tuple[GenericObservationTask, int]] = []

        # check every committed winning bid for this agent for broken coalition deps
        for (task, n_obs), deps in list(self._coalition_deps.items()):
            # skip if this agent no longer holds this bid
            if n_obs >= len(self._results[task]):
                self._coalition_deps.pop((task, n_obs), None)
                continue

            # get current bid
            bid = self._results[task][n_obs]

            # check
            if not bid.has_winner() or bid.winner != state.agent_name or bid.was_performed():
                self._coalition_deps.pop((task, n_obs), None)
                continue

            # check every recorded partner
            for (partner_task, partner_n_obs) in deps:
                if partner_n_obs >= len(self._results.get(partner_task, [])):
                    to_invalidate.append((task, n_obs))
                    break
                partner_bid = self._results[partner_task][partner_n_obs]
                still_valid = (partner_bid.has_winner()
                               and not partner_bid.was_performed()
                               and partner_bid.t_img < bid.t_img
                               and (bid.t_img - partner_bid.t_img) <= self._co_obs_window_for(task))
                if not still_valid:
                    to_invalidate.append((task, n_obs))
                    break

        # reset invalidated bids and all higher n_obs slots for that task
        bids_in_violation: List[dict] = []
        for (task, n_obs) in to_invalidate:
            for bid_idx in range(n_obs, len(self._results[task])):
                bid_to_reset: Bid = self._results[task][bid_idx]
                if not bid_to_reset.is_bidder_winning():
                    continue
                bid_to_reset.reset(t_curr)
                self._results[task][bid_idx] = bid_to_reset
                self._coalition_deps.pop((task, bid_idx), None)
                bids_in_violation.append(bid_to_reset.to_dict())

        return self._bundle, self._path, bids_in_violation

    # ------------------------------------------------------------------
    # PLANNING PHASE — rebuild coalition deps after committing bids
    # ------------------------------------------------------------------

    def generate_plan(self,
                      state: SimulationAgentState,
                      specs: object,
                      current_plan: Plan,
                      orbitdata: OrbitData,
                      mission: Mission,
                      tasks: List[GenericObservationTask],
                      observation_history: TaskObservationTracker
                     ) -> Plan:
        """Generate plan (base), then rebuild coalition dependency map.

        _coalition_deps is rebuilt after every planning phase so it reflects
        the co-obs partners that were present in _results when the new bids
        were evaluated.  The rebuilt map is then available for
        _check_co_obs_constraints during the next consensus round.
        """
        plan = super().generate_plan(
            state, specs, current_plan, orbitdata, mission, tasks, observation_history
        )
        self._rebuild_coalition_deps(state)
        return plan

    def _rebuild_coalition_deps(self, state: SimulationAgentState) -> None:
        """Rebuild _coalition_deps from the current committed results.

        For every EventObservationTask bid won by this agent, finds all
        partner bids (same event, different parameter, strictly before this
        bid's t_img, within _co_obs_window) that are currently committed in
        _results.  Records them as dependencies.

        Only bids with at least one qualifying partner are entered into the
        map — non-event tasks and tasks with no in-window prior partners
        are skipped.
        """
        self._coalition_deps.clear()
        for task, bids in self._results.items():
            if not isinstance(task, EventObservationTask) or task.event is None:
                continue
            for n_obs, bid in enumerate(bids):
                if (not bid.has_winner()
                        or bid.winner != state.agent_name
                        or bid.was_performed()):
                    continue
                t_img = bid.t_img
                deps: Set[Tuple[GenericObservationTask, int]] = set()
                for param, partner_tasks in \
                        self._event_to_tasks_by_instrument.get(task.event.id, {}).items():
                    if param == task.parameter:
                        continue
                    for partner_task in partner_tasks:
                        for p_n_obs, pbid in enumerate(self._results.get(partner_task, [])):
                            if (pbid.has_winner()
                                    and not pbid.was_performed()
                                    and pbid.t_img < t_img
                                    and (t_img - pbid.t_img) <= self._co_obs_window_for(task)):
                                deps.add((partner_task, p_n_obs))
                if deps:
                    self._coalition_deps[(task, n_obs)] = deps

    # ------------------------------------------------------------------
    # CO-OBS WINDOW HELPERS
    # ------------------------------------------------------------------

    def _co_obs_window_for(self, task: GenericObservationTask) -> float:
        """Return the co-observation time window [s] for *task*.

        Looks up the cached per-task window in _co_obs_windows first.
        Falls back to the global _co_obs_window if the task has no cached
        entry (non-event tasks, or tasks whose CoObservationRequirement was
        never populated).
        """
        return self._co_obs_windows.get(task.id, self._co_obs_window)

    @staticmethod
    def _extract_co_obs_window(task: GenericObservationTask) -> float:
        """Extract decorrelation_time from a task's CoObservationRequirement, or None."""
        if not isinstance(task, EventObservationTask) or task.event is None:
            return None
        objective = getattr(task, 'objective', None)
        if objective is None:
            return None
        co_req = objective.requirements.get(CoObservationRequirement.ATTRIBUTE)
        if co_req is None:
            return None
        return float(co_req.decorrelation_time)

    # ------------------------------------------------------------------
    # CO-OBS INDEX MANAGEMENT
    # ------------------------------------------------------------------

    def _remove_from_event_index(self,
                                  expired_tasks: List[GenericObservationTask]
                                 ) -> None:
        """Remove expired EventObservationTasks from the event index.

        Called when tasks are removed from _results by __remove_expired_tasks.
        Receives the task objects directly (captured before they are popped
        from _results and _id_to_tasks), so no dict reconstruction or registry
        lookup is needed.
        """
        for task in expired_tasks:
            self._co_obs_windows.pop(task.id, None)
            if not isinstance(task, EventObservationTask) or task.event is None:
                continue
            param_tasks = self._event_to_tasks_by_instrument.get(task.event.id, {})
            bucket = param_tasks.get(task.parameter)
            if bucket is None:
                continue
            try:
                bucket.remove(task)
            except ValueError:
                pass
            if not bucket:
                param_tasks.pop(task.parameter, None)
            if not param_tasks:
                self._event_to_tasks_by_instrument.pop(task.event.id, None)

    def _update_event_index(self, task_updates: list) -> None:
        """Incrementally insert newly announced EventObservationTasks into the index.

        Parses the task object from each bid dict in task_updates using
        _id_to_tasks (the canonical task registry) so no reconstruction from
        dict is needed for already-known tasks.  Skips tasks not yet in the
        registry (they will be caught on the next full rebuild if needed).
        Avoids duplicate entries under the same parameter key.
        """
        for task in task_updates:
            window = self._extract_co_obs_window(task)
            if window is not None:
                self._co_obs_windows[task.id] = window
            if isinstance(task, EventObservationTask) and task.event is not None:
                param_list = self._event_to_tasks_by_instrument[task.event.id][task.parameter]
                if task not in param_list:
                    param_list.append(task)

    def _rebuild_event_index(self) -> None:
        """Rebuild the event-instrument index from the current results table.

        Clears and repopulates _event_to_tasks_by_instrument so it reflects
        exactly the EventObservationTasks that are currently tracked in
        self._results. Called automatically after each consensus round.
        """
        self._event_to_tasks_by_instrument.clear()
        self._co_obs_windows.clear()
        for task in self._results:
            window = self._extract_co_obs_window(task)
            if window is not None:
                self._co_obs_windows[task.id] = window
            if isinstance(task, EventObservationTask) and task.event is not None:
                self._event_to_tasks_by_instrument[task.event.id][task.parameter].append(task)

    # ------------------------------------------------------------------
    # CO-OBS HELPERS (available to subclasses during bundle-building)
    # ------------------------------------------------------------------

    def _get_co_obs_partners(self,
                              task: GenericObservationTask,
                              t_img: float,
                              co_obs_window: float
                             ) -> List[Bid]:
        """Return active winning bids from partner instruments for the same event.

        A partner bid qualifies when it:
        - comes from a different instrument (parameter) observing the same event
        - has an active, non-performed winner
        - has an imaging time strictly before t_img and within co_obs_window

        Returns an empty list for tasks that are not EventObservationTasks or
        have no associated event.
        """
        if not isinstance(task, EventObservationTask) or task.event is None:
            return []
        partners = []
        for param, partner_tasks in self._event_to_tasks_by_instrument.get(task.event.id, {}).items():
            if param == task.parameter:
                continue
            for partner_task in partner_tasks:
                for bid in self._results.get(partner_task, []):
                    if (bid.has_winner()
                            and not bid.was_performed()
                            and bid.t_img < t_img
                            and (t_img - bid.t_img) <= co_obs_window):
                        partners.append(bid)
        return partners