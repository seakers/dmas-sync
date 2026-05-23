from typing import Any

from execsatm.observations import ObservationOpportunity

from dmas.models.planning.decentralized.heuristic import HeuristicInsertionPeriodicPlanner, HeuristicInsertionReactivePlanner

class EarliestAccessPeriodicPlanner(HeuristicInsertionPeriodicPlanner):
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
    
class EarliestAccessReactivePlanner(HeuristicInsertionReactivePlanner):
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

class EarliestRequestArrivalPeriodicPlanner(HeuristicInsertionPeriodicPlanner):
    """ Schedules observations based on the earliest rqeuest time among its tasks """
    
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        *_ : Any
                        ) -> tuple:
        """ Heuristic function to sort observation opportunities by their earliest request arrival time. """
        # return to sort using: earliest request arrival time >> longest duration >> highest priority
        arrival_time = min([task.req.t_req for task in observation_opportunity.tasks 
                            if hasattr(task, 'req')], default=0.0)
        
        return (
                arrival_time,
                observation_opportunity.accessibility.left, 
                -observation_opportunity.min_duration, 
                observation_opportunity.get_priority()
                )
    
class EarliestRequestArrivalReactivePlanner(HeuristicInsertionReactivePlanner):
    """ Schedules observations based on the earliest rqeuest time among its tasks """
    
    def _calc_heuristic(self,
                        observation_opportunity : ObservationOpportunity, 
                        *_ : Any
                        ) -> tuple:
        """ Heuristic function to sort observation opportunities by their earliest request arrival time. """
        # return to sort using: earliest request arrival time >> longest duration >> highest priority
        arrival_time = min([task.req.t_req for task in observation_opportunity.tasks 
                            if hasattr(task, 'req')], default=0.0)
        
        return (
                arrival_time,
                observation_opportunity.accessibility.left, 
                -observation_opportunity.min_duration, 
                observation_opportunity.get_priority()
                )