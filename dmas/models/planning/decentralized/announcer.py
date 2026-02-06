
from collections import defaultdict
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

from tqdm import tqdm

from execsatm.tasks import EventObservationTask
from execsatm.events import GeophysicalEvent
from execsatm.mission import Mission
from execsatm.objectives import EventDrivenObjective
from execsatm.utils import Interval

from dmas.models.actions import BroadcastMessageAction, WaitAction
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.plan import PeriodicPlan, Plan
from dmas.models.science.requests import TaskRequest
from dmas.models.states import SimulationAgentState
from dmas.core.messages import BusMessage, MeasurementRequestMessage
from dmas.utils.orbitdata import OrbitData


class EventAnnouncerPlanner(AbstractPeriodicPlanner):
    def __init__(self, 
                 events_path : str,
                 mission : Mission,
                 debug = False, 
                 logger = None,
                 printouts : bool = True
                ) -> None:
        """
        # Event Announcer Planner
        Announces geophysical events to other agents in the mission as they become available.
        
        TODO : expand to announce events at different times, not just when they become available.
        """

        super().__init__(np.Inf, np.Inf, AbstractPeriodicPlanner.OPPORTUNISTIC, debug, logger, printouts)

        # validate inputs
        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')
        if not isinstance(mission, Mission):
            raise ValueError('`mission` must be of type `Mission`.')


        # load predefined events
        self.events : list[GeophysicalEvent] = self.load_events(events_path)
        self.parent_mission : Mission = mission
    
    def load_events(self, events_path : str) -> pd.DataFrame:        
        events_df : pd.DataFrame = pd.read_csv(events_path)

        events = []
        for _,row in events_df.iterrows():
            # convert event to GeophysicalEvent
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row.get('id',None)
            )
            events.append(event)

        return events
    
    
    def generate_plan(  self, 
                        state : SimulationAgentState,
                        _ : object,
                        orbitdata : OrbitData,
                        *__
                    ) -> Plan:
        """ Generates a new plan for the agent """            
        # schedule broadcasts to be perfomed
        broadcasts : list = self._schedule_broadcasts(state, [], orbitdata)
        
        # TODO add maneuvers for pointing-dependent transmissions
        # currently assumes omnidirectional antennas
        
        # wait for next planning period to start
        replan : list = self._schedule_periodic_replan(state, state._t + self.period)

        # generate plan from actions
        self.plan : PeriodicPlan = PeriodicPlan(broadcasts, replan, t=state._t, horizon=self.horizon, t_next=state._t+self.period)    
        
        # return plan and save local copy
        return self.plan.copy()
    
    def _schedule_observations(self, *_) -> list:
        return [] # No scheduling, only announcing events
    
    def _schedule_broadcasts(self, state : SimulationAgentState, _, orbitdata : OrbitData, __ = None) -> List[BroadcastMessageAction]:
        # initialize broadcasts from parent planner
        # broadcasts : List[BroadcastMessageAction] = super()._schedule_broadcasts(state, observations, orbitdata, t)
        broadcasts : List[BroadcastMessageAction] = []

        # get list of future events
        future_events : List[GeophysicalEvent] = [event for event in tqdm(self.events, 
                                                                          desc=f'{state.agent_name}-PREPLANNER: Collecting future events', 
                                                                          leave=False,
                                                                          disable=(len(self.events) < 10) or not self._printouts
                                                                        )
                                                  if event.is_available(state._t)]

        # create requests for each event
        task_requests : List[Tuple[GeophysicalEvent, TaskRequest]] = []
        for event in tqdm(future_events, 
                          desc=f'{state.agent_name}-PREPLANNER: Generating task request from known events',
                          leave=False,
                          disable=(len(future_events) < 10) or not self._printouts
                        ):
            # get event objetives from mission
            objectives  : list[EventDrivenObjective] = [objective for objective in self.parent_mission.objectives
                                                        if isinstance(objective, EventDrivenObjective)
                                                        and objective.event_type == event.event_type]

            for objective in objectives:
                objective : EventDrivenObjective
                # create task
                task = EventObservationTask(objective.parameter, event=event, objective=objective)

                # generate task request 
                task_request = TaskRequest(task,
                                            requester = state.agent_name,
                                            mission_name = self.parent_mission.name,
                                            t_req = event.t_start)
                
                # generate measurement request message
                task_request_msg = MeasurementRequestMessage(state.agent_name, state.agent_name, task_request.to_dict())
                
                # update list of generated requests 
                # task_requests.append((event, task_request, task_request_msg.to_dict()))
                task_requests.append((event, task_request, task_request_msg))

        # initialize set of times when broadcasts are scheduled (sets to avoid duplicates)
        t_broadcasts = set() 

        # create broadcasts for each request
        for event,task_req,task_request_msg in tqdm(task_requests, 
                                                    desc=f'{state.agent_name}/PREPLANNER: Scheduling broadcasts for generated task requests',
                                                    leave=False,
                                                    disable=(len(task_requests) < 10) or not self._printouts
                                                ):
            
            # schedule broadcasts to all available agents
            for target in orbitdata.comms_links.keys():
                # get access intervals with the client agent within the planning horizon
                access_intervals : List[Interval] = orbitdata.get_next_agent_accesses(target, task_req.t_req, include_current=True)

                # create broadcast actions for each access interval
                for next_access in access_intervals:
                    # if no access opportunities in this planning horizon, skip scheduling
                    if next_access.is_empty(): continue

                    # check if the task is available during the given access interval
                    if not event.availability.overlaps(next_access): continue

                    # calculate broadcast time to earliest in this access interval
                    t_broadcast : float = max(next_access.left, task_req.t_req)
                    # t_broadcast : float = max(
                    #                           min(next_access.left + 5*self.EPS,    # give buffer time for access to start
                    #                               next_access.right),               # ensure broadcast is before access ends
                    #                         task_req.t_req)                                # ensure broadcast is not in the past
                    
                    # add broadcast time to set of broadcast times
                    t_broadcasts.add(t_broadcast)

        # iterate through access start times to find active requests
        for t_broadcast in sorted(t_broadcasts):
            # initiate bus messages list 
            task_requests_msgs : List[MeasurementRequestMessage] \
                = [req_msg for event,_,req_msg in task_requests 
                    if event.is_active(t_broadcast)]
            
            # ensure there is at least one active request to broadcast;
            #  should always be true due to previous checks
            assert len(task_requests_msgs) > 0, "No active task requests found for broadcast time."
            
            # compile all requests into single broadcast message
            bus_broadcast = BusMessage(state.agent_name, state.agent_name, task_requests_msgs)

            # create single broadcast action for all requests
            broadcasts.append(BroadcastMessageAction(bus_broadcast.to_dict(), t_broadcast))
           
        return sorted(broadcasts, key=lambda action: action.t_start)