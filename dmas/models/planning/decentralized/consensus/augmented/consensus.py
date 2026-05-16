from collections import defaultdict
from typing import Dict, List, Tuple, Union
import logging

from execsatm.tasks import EventObservationTask, GenericObservationTask

from dmas.models.planning.decentralized.consensus.consensus import ConsensusPlanner
from dmas.models.planning.decentralized.consensus.bids import Bid
from dmas.models.states import SimulationAgentState
from dmas.models.planning.plan import Plan
from dmas.models.science.requests import TaskRequest
from dmas.models.actions import ObservationAction


class AugmentedConsensusPlanner(ConsensusPlanner):
    """
    Extends ConsensusPlanner with co-observation awareness.

    Maintains an event-to-tasks-by-instrument index that is rebuilt after
    each consensus round. Exposes co-obs helper methods that augmented
    subclasses (e.g. AugmentedHeuristicInsertionConsensusPlanner) use
    during bundle-building to compute coalition values and partner bids.

    Inheritance note: this class uses *args/**kwargs in __init__ so that
    it can sit transparently in a multiple-inheritance chain without
    breaking the ConsensusPlanner argument order.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # event_id -> parameter -> List[EventObservationTask]
        self._event_to_tasks_by_instrument: Dict[str, Dict[str, List[EventObservationTask]]] = \
            defaultdict(lambda: defaultdict(list))

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
        """Run base consensus phase, then conditionally refresh the co-obs event index.

        The index is rebuilt only when the task registry changes — i.e., when
        new tasks are announced (task_updates non-empty) or when tasks are
        expired and removed from self._results (detected by a change in the
        result count). Most timesteps neither condition is true, so the index
        is left unchanged at zero cost.
        """
        prev_task_count = len(self._results)
        consensus_results = super()._consensus_phase(
            state, incoming_reqs, incoming_bids, tasks, current_plan, performed_observations
        )
        task_updates = consensus_results[0]
        if task_updates or len(self._results) != prev_task_count:
            self._rebuild_event_index()
        return consensus_results

    # ------------------------------------------------------------------
    # CO-OBS INDEX MANAGEMENT
    # ------------------------------------------------------------------

    def _rebuild_event_index(self) -> None:
        """Rebuild the event-instrument index from the current results table.

        Clears and repopulates _event_to_tasks_by_instrument so it reflects
        exactly the EventObservationTasks that are currently tracked in
        self._results. Called automatically after each consensus round.
        """
        self._event_to_tasks_by_instrument.clear()
        for task in self._results:
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
        - has an imaging time within `co_obs_window` seconds of `t_img`

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
                            and abs(bid.t_img - t_img) <= co_obs_window):
                        partners.append(bid)
        return partners

