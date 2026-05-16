from typing import Dict, List
import numpy as np

from execsatm.tasks import EventObservationTask, GenericObservationTask
from execsatm.mission import Mission

from dmas.models.planning.decentralized.consensus.heuristic import HeuristicInsertionConsensusPlanner
from dmas.models.planning.decentralized.consensus.augmented.consensus import AugmentedConsensusPlanner
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
                 printouts: bool = True,
                 contested_reset_threshold: int = 3
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
        super().__init__(agent_results_dir, heuristic, replan_threshold,
                         optimistic_bidding_threshold, periodic_overwrite,
                         debug, logger, printouts, contested_reset_threshold)
        self._co_obs_window: float = co_obs_window

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

        Returns a mapping of {parameter: committed_t_img} for every partner
        instrument observing the same event whose committed bid falls within
        self._co_obs_window seconds of t_obs. If a partner parameter has
        multiple qualifying bids, the one closest in time to t_obs is kept.

        Returns an empty dict for non-event tasks or tasks with no committed
        partner bids within the window.
        """
        if not isinstance(task, EventObservationTask) or task.event is None:
            return {}

        co_obs: Dict[str, float] = {}
        for param, partner_tasks in self._event_to_tasks_by_instrument.get(task.event.id, {}).items():
            if param == task.parameter:
                continue
            for partner_task in partner_tasks:
                for bid in self._results.get(partner_task, []):
                    if (bid.has_winner()
                            and not bid.was_performed()
                            and abs(bid.t_img - t_obs) <= self._co_obs_window):
                        if (param not in co_obs
                                or abs(bid.t_img - t_obs) < abs(co_obs[param] - t_obs)):
                            co_obs[param] = bid.t_img
        return co_obs
