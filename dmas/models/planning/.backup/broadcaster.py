from typing import Dict
import numpy as np

from dmas.modules import ClockConfig

from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.orbitdata import IntervalData, OrbitData
from chess3d.agents.planning.plan import *
from chess3d.agents.states import SimulationAgentState
from chess3d.mission.mission import Mission
from chess3d.utils import Interval

class BroadcasterReplanner(AbstractReactivePlanner):
    def update_percepts(self, 
                        state, 
                        current_plan, 
                        incoming_reqs, 
                        relay_messages, 
                        misc_messages, 
                        completed_actions, 
                        aborted_actions, 
                        pending_actions):
        super().update_percepts(state, 
                                current_plan, 
                                incoming_reqs, 
                                relay_messages, 
                                misc_messages, 
                                completed_actions, 
                                aborted_actions, 
                                pending_actions)

class PeriodicBroadcasterReplanner(BroadcasterReplanner):
    def __init__(self, period : float = np.Inf, debug = False, logger = None):
        super().__init__(debug, logger)

        # set parameters
        self.period = period

        # initialize attributes
        self.t_next = 0             # time of next broacast
        
    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitdata : OrbitData
                       ) -> bool:

        # check if a new preplan was just generated
        if isinstance(current_plan, PeriodicPlan): return True
            
        # check if the broadcast time has been reached 
        return state.t >= self.t_next

    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : ObservationHistory,
                    ) -> Plan:
        
        # update next broadcast time
        self.t_next += self.period if state.t >= self.t_next else 0

        # add future broadcast messages to the plan
        broadcasts = [FutureBroadcastMessageAction(FutureBroadcastMessageAction.REWARD,self.t_next),
                        FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS,self.t_next)]
        
        # return modified plan
        return ReactivePlan.from_periodic_plan(current_plan, broadcasts, t=state.t)
 

class OpportunisticBroadcasterReplanner(BroadcasterReplanner):
    def __init__(self, period: float = np.Inf, debug = False, logger = None):
        # initialize the base class
        super().__init__(debug, logger)
        
        # initialize parameters
        self.t_prev = 0             # time of last broadcast in [s]    
        self.period = period        # period of the opportunistic broadcasts in [s]

    def needs_planning(self, 
                       state : SimulationAgentState,
                       specs : object,
                       current_plan : Plan,
                       orbitdata : OrbitData
                       ) -> bool:

        # check if a new preplan was just generated
        return isinstance(current_plan, PeriodicPlan)
    
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        specs : object,
                        current_plan : Plan,
                        clock_config : ClockConfig,
                        orbitdata : OrbitData,
                        mission : Mission,
                        tasks : list,
                        observation_history : ObservationHistory,
                    ) -> Plan:
        # update time of last broadcast
        self.t_prev = state.t

        # determine the next acceptable broadcast times
        t_start = state.t
        t_end = current_plan.t_next if isinstance(current_plan, PeriodicPlan) else np.Inf

        # gather the access opportunities per agent
        access_opportunities : Dict[str, list] = self._calculate_broadcast_opportunities(orbitdata, t_start, t_end)

        # prepare the broadcast messages
        broadcasts : list[FutureBroadcastMessageAction] = []
        for _, intervals in access_opportunities.items():
            # for each agent, prepare a broadcast message for each access opportunity
            for interval in intervals:
                interval : Interval

                if np.isinf(self.period):
                    # create a future broadcast message for the next access opportunity
                    broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastMessageAction.REWARD, interval.left))
                    broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, interval.left))
                else:
                    # create a future broadcast message for every period within the access opportunity
                    n_broadcasts = int(np.ceil((interval.right - interval.left) / self.period))
                    broadcast_times = [interval.left + i * self.period for i in range(n_broadcasts)]
                    
                    for t in broadcast_times:
                        broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastMessageAction.REWARD, t))
                        broadcasts.append(FutureBroadcastMessageAction(FutureBroadcastMessageAction.REQUESTS, t))
        
        # return modified plan
        return ReactivePlan.from_periodic_plan(current_plan, broadcasts, t=state.t)

    def _calculate_broadcast_opportunities(self, orbitdata: OrbitData, t_start : float, t_end : float) -> dict:
        """ calculates future broadcast times based on inter-agent access opportunities """

        # compile all inter-agent access opportunities
        intervals : list[Interval] = { agent_name : sorted([Interval(max(t_start, t_access_start),min(t_end, t_access_end))
                                                    for t_access_start,t_access_end,_ in data.data
                                                    if isinstance(data, IntervalData)
                                                    and (t_start <= t_access_start <= t_end
                                                         or t_start <= t_access_end <= t_end)], key=lambda x: x.left)
                                        for agent_name,data in orbitdata.comms_links.items()}
        

        # TODO include ground station support

        # collect unique start times of every access opportunity
        return intervals
        
        # IDEA aim to broadcasts during intervals where more agents will listen? 
        # merge access intervals
        merged_intervals : list[Interval] = []
        for interval in intervals:
            overlaps = [merged_interval for merged_interval in merged_intervals
                        if interval.has_overlap(merged_interval)]
            
            if not overlaps:
                merged_intervals.append(interval)
            
            for overlap in overlaps:
                overlap.merge(interval)

        return [interval.start for interval in merged_intervals]
