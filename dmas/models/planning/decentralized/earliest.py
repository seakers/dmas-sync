from typing import Any

from execsatm.observations import ObservationOpportunity

from dmas.models.planning.decentralized.heuristic import HeuristicInsertionPlanner

class EarliestAccessPlanner(HeuristicInsertionPlanner):
    """ Schedules observations based on the earliest feasible access point """
    
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        *_ : Any
                        ) -> tuple:
        """ Heuristic function to sort observation opportunities by their earliest access time. """
        # return to sort using: earliest accessibility time >> longest duration >> highest priority
        return (
                observation_opportunity.accessibility.left, 
                -observation_opportunity.min_duration, 
                observation_opportunity.get_priority()
                )