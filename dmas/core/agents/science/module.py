import pandas as pd
from zmq import asyncio as azmq

from instrupy.base import Instrument

from dmas.modules import *

from execsatm.mission import Mission

from chess3d.agents.states import SimulationAgentState
from chess3d.agents.science.processing import DataProcessor
from chess3d.agents.science.requests import *
from chess3d.messages import *


class ScienceModule(InternalModule):
    def __init__(   self, 
                    results_path : str,
                    parent_name : str,
                    parent_network_config: NetworkConfig, 
                    data_processor : DataProcessor,
                    event_mission : Mission,
                    logger: logging.Logger = None
                ) -> None:

        addresses = parent_network_config.get_internal_addresses()        
        sub_addesses = []
        sub_address : str = addresses.get(zmq.PUB)[0]
        sub_addesses.append( sub_address.replace('*', 'localhost') )

        pub_address : str = addresses.get(zmq.SUB)[1]
        pub_address = pub_address.replace('localhost', '*')

        addresses = parent_network_config.get_manager_addresses()
        push_address : str = addresses.get(zmq.PUSH)[0]

        science_network_config =  NetworkConfig(parent_name,
                                        manager_address_map = {
                                        zmq.REQ: [],
                                        zmq.SUB: sub_addesses,
                                        zmq.PUB: [pub_address],
                                        zmq.PUSH: [push_address]})

        super().__init__(   f"{parent_name}-SCIENCE_MODULE", 
                            science_network_config, 
                            parent_network_config, 
                            logging.INFO, 
                            logger)
        
        # initialize
        self.known_reqs : set[TaskRequest] = set()

        # assign parameters
        self.results_path = results_path
        self.parent_name = parent_name
        self.data_processor = data_processor
        self.event_mission = event_mission
    
    async def sim_wait(self, _: float) -> None:
        # is event-driven; no need to support timed delays
        return

    async def setup(self) -> None:
        # setup internal inboxes
        self.onboard_processing_inbox = asyncio.Queue()

    async def live(self) -> None:
        """
        Performs concurrent tasks:
        - Listener: receives messages from the parent agent and checks results
        - Science valuer: determines value of a given measurement
        - Science reasoning: checks data for outliers
        - Onboard processing: converts data of one type into data of another type (e.g. level 0 to level 1)
        """
        try:
            # announce existance to other modules 
            await self.make_announcement()

            # initialize concurrent tasks
            listener_task = asyncio.create_task(self.listener(), name='listener()')
            onboard_processing_task = asyncio.create_task(self.onboard_processing(), name='onboard_processing()')
            tasks : list[asyncio.Task] = [listener_task, onboard_processing_task]
            
            # wait for a task to terminate
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        finally:
            # cancel the remaning tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    await task

    async def make_announcement(self) -> None:
        """ Announces presence of science module to other internal modules """
        # create announcement message
        announcement_msg = SimulationMessage(self.get_element_name(), self.get_parent_name(), 'HANDSHAKE')

        # broadcast announcement
        await self._send_manager_msg(announcement_msg, zmq.PUB)

    async def listener(self) -> None:
        """ Unpacks and classifies any incoming messages """
        try:
            # create poller for all broadcast sockets
            poller = azmq.Poller()
            manager_socket, _ = self._manager_socket_map.get(zmq.SUB)
            poller.register(manager_socket, zmq.POLLIN)

            # listen for broadcasts and place in the appropriate inboxes
            while True:
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                if content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SenseMessage = SenseMessage(**content)

                    senses = []
                    senses.append(senses_msg.state)
                    senses.extend(senses_msg.senses)     

                    for sense in senses:
                        # unpack message
                        msg : SimulationMessage = message_from_dict(**sense)

                        # check type of message being received
                        if isinstance(msg, AgentStateMessage):
                            self.log(f"received agent state message!")
                                                        
                            # update current state
                            state : SimulationAgentState = SimulationAgentState.from_dict(msg.state)
                            await self.update_current_time(state.t)

                        elif isinstance(msg, ObservationResultsMessage):
                            # observation data from another agent was received
                            self.log(f"received observation data from agent!")

                            # send to onboard processor
                            await self.onboard_processing_inbox.put(msg)

                        elif isinstance(msg, MeasurementRequestMessage):
                            # measurement request message received
                            self.log(f"received measurement request message!")

                            # unapack measurement request
                            req : TaskRequest = TaskRequest.from_dict(msg.req)
                            
                            # update list of known measurement requests
                            if req.severity > 0.0: self.known_reqs.add(req)

                # if sim-end message, end agent `live()`
                elif content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    self.log(f"received manager broadcast of type {content['msg_type']}! terminating `live()`...",level=logging.INFO)
                    return

                elif content['msg_type'] == SimulationMessageTypes.OBSERVATION.value:
                    # unpack message
                    self.log(f"received manager broadcast of type {content['msg_type']}!",level=logging.WARN)
                    msg = ObservationResultsMessage(**content)
                    await self.onboard_processing_inbox.put(msg)
        
        except asyncio.CancelledError:
            print("Asyncio cancelled error in science module listener")
            return
                
    async def onboard_processing(self) -> None:
        """ 
        Processes incoming observation data and generates measurement requests if an event is detected.
        
        Always returns a measurement request regardless of the presense of an event following an observation.
        The severity is set depending on the whether there was an event detected or not.
        The planner module will only consider and broadcast measurement requests with a non-zero severity.
        
        """
        try:
            while True:
                # wait for next observation to be performed by parent agent
                msg : ObservationResultsMessage = await self.onboard_processing_inbox.get()

                raise NotImplementedError("No case has been implemented for event detected in onboard processing in real-time processor")   

                # unpack information from observation message
                observation_data = msg.observation_data
                instrument = Instrument.from_dict(msg.instrument)
                
                # process each observation
                reqs_from_observations : list[MeasurementRequestMessage] = []
                for obs in observation_data:
                    # process observation
                    event : GeophysicalEvent = self.data_processor.process_observation(instrument, **obs)

                    # no event in observation; skip
                    if event is None: 
                        raise NotImplementedError("No case has been implemented for no event detected in onboard processing")   

                    # get event objetives from mission
                    objectives = [objective for objective in self.event_mission.objectives
                                if objective.event_type == event.event_type]

                    # generate task request 
                    # measurement_req = MeasurementRequest(self.parent_name,
                    #                                     [lat_event,lon_event,0.0],
                    #                                     severity,
                    #                                     observations_required,
                    #                                     t_start, t_end, t_corr)
                    task_request = TaskRequest(self.parent_name,
                                            event,
                                            objectives,
                                            obs['t_img'])
                                        
                    # check if another request has already been made for this event
                    if any([task_request.same_event(req) for req in self.known_reqs]):
                        # another request has been made for this same event; ignore
                        task_request.severity = 0.0
                    elif event.severity > 0:
                        self.known_reqs.add(task_request)
                    
                    # send request to all internal agent modules
                    req_msg = MeasurementRequestMessage(self.get_module_name(), 
                                                        self.get_parent_name(), 
                                                        task_request.to_dict())
                    reqs_from_observations.append(req_msg)
                
                # create blank request if no requests were discovered in this observation
                if not reqs_from_observations:
                    task_request = TaskRequest(self.get_parent_name(),
                                                         [-1,-1,0.0],
                                                         0.0,
                                                         [],
                                                         -1)
                    req_msg = MeasurementRequestMessage(self.get_module_name(), 
                                                        self.get_parent_name(), 
                                                        task_request.to_dict())
                    reqs_from_observations.append(req_msg)

                # package into bus    
                req_dicts = [req.to_dict() for req in reqs_from_observations]
                req_bus = BusMessage(self.get_module_name(), self.get_parent_name(), req_dicts)
                
                # send to parent agent
                await self._send_manager_msg(req_bus, zmq.PUB)


        except asyncio.CancelledError:
            return
        
    async def teardown(self) -> None:
        # print out events detected by this module
        columns = ['ID','Requester','lat [deg]','lon [deg]','Severity','t start','t end','t corr','Measurment Types']
        data = [(req.id, req.requester, req.target[0], req.target[1], req.severity, req.t_start, req.t_end, req.t_corr, str(req.observation_types))
                for req in self.known_reqs
                if req.requester == self.get_parent_name()]
        
        df = pd.DataFrame(data=data, columns=columns)        
        df.to_csv(f"{self.results_path}/{self.get_parent_name()}/events_detected.csv", index=False)   
    