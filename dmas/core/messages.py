from enum import Enum
import uuid
import json

from dmas.utils.tools import SimulationRoles

class SimulationMessageTypes(Enum):
    MEASUREMENT_REQ = 'MEASUREMENT_REQ'
    AGENT_ACTION = 'AGENT_ACTION'
    AGENT_STATE = 'AGENT_STATE'
    CONNECTIVITY_UPDATE = 'CONNECTIVITY_UPDATE'
    MEASUREMENT_BID = 'MEASUREMENT_BID'
    BID_RESULTS = 'BID_RESULTS'
    PLAN = 'PLAN'
    SENSES = 'SENSES'
    OBSERVATION = 'OBSERVATION'
    OBSERVATION_PERFORMED = 'OBSERVATION_PERFORMED'
    BUS = 'BUS'

class SimulationMessage(object):
    """
    ## Abstract Simulation Message 

    Describes a message to be sent between simulation elements

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
        - path (`list`) : path the message must travel to get to its inteded destination
    """
    def __init__(self, 
                 src : str, 
                 dst : str, 
                 msg_type : str, 
                 id : str = None,
                 path : list = [] 
                ):
        """
        Initiates an instance of a simulation message.
        
        ### Args:
            - src (`str`): name of the simulation element sending this message
            - dst (`str`): name of the intended simulation element to receive this message
            - msg_type (`str`): type of message being sent
            - id (`str`) : Universally Unique IDentifier for this message
            - path (`list`) : path the message must travel to get to its inteded destination
        """
        super().__init__()

        # load attributes from arguments
        self.src = src
        self.dst = dst
        self.msg_type = msg_type
        self.path = [elem for elem in path]
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

        # check types 
        if not isinstance(self.src , str):
            raise TypeError(f'Message sender `src` must be of type `str`. Is of type {type(self.src)}')
        if not isinstance(self.dst , str):
            raise TypeError(f'Message receiver `dst` must be of type `str`. Is of type {type(self.dst)}')
        if not isinstance(self.msg_type , str):
            raise TypeError(f'Message type `msg_type` must be of type `str`. Is of type {type(self.msg_type)}')
        if not isinstance(self.id , str):
            raise TypeError(f'Message id `id` must be of type `str`. Is of type {type(self.id)}')
        
    def __eq__(self, other) -> bool:
        """
        Compares two instances of a simulation message. Returns True if they represent the same message.
        """
        return self.to_dict() == dict(other.__dict__)

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this message object
        """
        return dict(self.__dict__)

    def to_json(self) -> str:
        """
        Creates a json file from this message 
        """
        try:
            return json.dumps(self.to_dict())
        except Exception as e:
            print(f'Failed to create JSON from message. {e}')
            raise e

    def __str__(self) -> str:
        """
        Creates a string representing the contents of this message
        """
        return str(self.to_dict())

    def __repr__(self) -> str:
        return str(self.to_dict())
    
    def __hash__(self) -> int:
        return hash(repr(self))

def message_from_dict(msg_type : str, **kwargs) -> SimulationMessage:
    """
    Creates the appropriate message from a given dictionary in the correct format
    """
    if msg_type == SimulationMessageTypes.MEASUREMENT_REQ.value:
        return MeasurementRequestMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.AGENT_ACTION.value:
        return AgentActionMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.AGENT_STATE.value:
        return AgentStateMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.CONNECTIVITY_UPDATE.value:
        return AgentConnectivityUpdate(**kwargs)
    elif msg_type == SimulationMessageTypes.MEASUREMENT_BID.value:
        return MeasurementBidMessage(**kwargs)
    # elif msg_type == SimulationMessageTypes.BID_RESULTS.value:
    #     return BidResultsMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.PLAN.value:
        return PlanMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.SENSES.value:
        return SenseMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.OBSERVATION.value:
        return ObservationResultsMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.OBSERVATION_PERFORMED.value:
        return ObservationPerformedMessage(**kwargs)
    elif msg_type == SimulationMessageTypes.BUS.value:
        return BusMessage(**kwargs)
    else:
        raise NotImplementedError(f'Action of type {msg_type} not yet implemented.')

class AgentStateMessage(SimulationMessage):
    """
    ## Tic Request Message

    Request from agents indicating that they are waiting for the next time-step advance

    ### Attributes:
        - src (`str`): name of the agent sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
        - state (`dict`): dictionary discribing the state of the agent sending this message
    """
    def __init__(self, 
                src: str, 
                dst: str, 
                state : dict,
                id: str = None, 
                path: list = [],
                **_):
        super().__init__(src, dst, SimulationMessageTypes.AGENT_STATE.value, id, path)
        self.state = state

class AgentConnectivityUpdate(SimulationMessage):
    """
    ## Agent Connectivity Update Message

    Informs an agent that it's connectivity to another agent has changed

    ### Attributes:
        - src (`str`): name of the agent sending this message
        - dst (`str`): name of the intended agent set to receive this message
        - target (`str`): name of the agent that the destination agent will change its connectivity with
        - connected (`bool`): status of the connection between `dst` and `target`
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
        - state (`dict`): dictionary discribing the state of the agent sending this message
    """
    def __init__(self, dst: str, target : str, connected : int, id: str = None, path: list = [], **_):
        super().__init__(SimulationRoles.ENVIRONMENT.value, 
                         dst, 
                         SimulationMessageTypes.CONNECTIVITY_UPDATE.value, 
                         id,
                         path)
        self.target = target
        self.connected = connected

class MeasurementRequestMessage(SimulationMessage):
    """
    ## Measurement Request Message 

    Describes a task request being between simulation elements

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - req (`dict`) : dictionary describing measurement request to be performed
        - id (`str`) : Universally Unique IDentifier for this message
        - path (`list`): sequence of agents meant to relay this mesasge
        - msg_type (`str`): type of message being sent
    """
    def __init__(self, src: str, dst: str, req : dict, id: str = None, path : list = [], **_):
        super().__init__(src, dst, SimulationMessageTypes.MEASUREMENT_REQ.value, id, path)
        
        if not isinstance(req, dict):
            raise AttributeError(f'`req` must be of type `dict`; is of type {type(req)}.')
        self.req = req

class ObservationResultsMessage(SimulationMessage):
    """
    ## Observation Results Request Message 

    Carries information regarding a observation performed on the environment

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - observation_data (`dict`) : observation data being communicated
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
    """
    def __init__(self, 
                 src: str, 
                 dst: str, 
                 agent_state : dict, 
                 observation_action : dict, 
                 instrument : dict,
                 t_start : float,
                 t_end : float,
                 observation_data : list = [],
                 id: str = None, 
                 path : list = [], **_):
        super().__init__(src, dst, SimulationMessageTypes.OBSERVATION.value, id, path)
        
        if not isinstance(observation_action, dict):
            raise AttributeError(f'`observation_action` must be of type `dict`; is of type {type(observation_action)}.')
        if not isinstance(agent_state, dict):
            raise AttributeError(f'`agent_state` must be of type `dict`; is of type {type(agent_state)}.')
        if not isinstance(instrument, dict):
            raise AttributeError(f'`instrument` must be of type `dict`; is of type {type(instrument)}.')
        if not isinstance(observation_data, list):
            raise AttributeError(f'`observation_data` must be of type `list`; is of type {type(observation_data)}.')
        if not all(isinstance(data, dict) for data in observation_data):
            raise AttributeError(f'elements of `observation_data` must be of type `dict`.')

        self.agent_state = agent_state
        self.observation_action = observation_action
        self.instrument = instrument
        self.t_start = t_start
        self.t_end = t_end
        self.observation_data = observation_data

class ObservationPerformedMessage(SimulationMessage):
    def __init__(self, 
                 src: str, 
                 dst: str, 
                 observation_action : dict,
                 id: str = None,
                 path : list = [],
                 **_
                 ):
        """
        ## Observation Perfromed Message

        Informs other agents that a measurement action was performed to satisfy a measurement request
        """
        super().__init__(src, dst, SimulationMessageTypes.OBSERVATION_PERFORMED.value, id, path)
        self.observation_action = observation_action

class MeasurementBidMessage(SimulationMessage):
    """
    ## Measurment Bid Message

    Informs another agents of the bid information held by the sender

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - bid (`dict`): bid information being shared
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
    """
    def __init__(self, 
                src: str, 
                dst: str, 
                bid: dict, 
                id: str = None,
                path : list = [],
                **_):
        """
        Creates an instance of a task bid message

        ### Arguments:
            - src (`str`): name of the simulation element sending this message
            - dst (`str`): name of the intended simulation element to receive this message
            - bid (`dict`): bid information being shared
            - id (`str`) : Universally Unique IDentifier for this message
        """
        super().__init__(src, dst, SimulationMessageTypes.MEASUREMENT_BID.value, id, path)
        self.bid = bid

# class BidResultsMessage(SimulationMessage):
#     """
#     ## Bid Results Message

#     Informs another agents of the bid results information held by the sender

#     ### Attributes:
#         - src (`str`): name of the simulation element sending this message
#         - dst (`str`): name of the intended simulation element to receive this message
#         - bid_results (`dict`): bid results information being shared
#         - msg_type (`str`): type of message being sent
#         - id (`str`) : Universally Unique IDentifier for this message
#     """
#     def __init__(self, 
#                 src: str, 
#                 dst: str, 
#                 results: Dict[GenericObservationTask, List[Bid]], 
#                 id: str = None,
#                 path : list = [],
#                 **_):
#         raise NotImplementedError("BidResultsMessage has been disabled.")
    #     """
    #     Creates an instance of a bid results message

    #     ### Arguments:
    #         - src (`str`): name of the simulation element sending this message
    #         - dst (`str`): name of the intended simulation element to receive this message
    #         - bid_results (`dict`): bid results information being shared
    #         - id (`str`) : Universally Unique IDentifier for this message
    #     """
    #     super().__init__(src, dst, SimulationMessageTypes.BID_RESULTS.value, id, path)
    #     self.results = {
    #         task: [bid for bid in bid_list]
    #         for task, bid_list in results.items()
    #     }

    # def to_dict(self) -> dict:
    #     """
    #     Converts the Bid Results Message to a dictionary format
    #     """
    #     msg_dict = super().to_dict()
    #     msg_dict['results'] = {
    #         task_id: [bid.to_dict() for bid in bid_list]
    #         for task_id, bid_list in self.results.items()
    #     }
    #     return msg_dict

class PlanMessage(SimulationMessage):
    """
    # Plan Message
    
    Informs an agent of a set of tasks to perform. 
    Sent by either an external or internal planner

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - plan (`list`): list of agent actions to perform
        - msg_type (`str`): type of message being sent
        - t_plan (`float`): time at which the plan was created
        - id (`str`) : Universally Unique IDentifier for this message
    """
    def __init__(self, src: str, dst: str, plan : list, t_plan : float, agent_name : str = None, id: str = None, path : list = [], **_):
        """
        Creates an instance of a plan message

        ### Attributes:
            - src (`str`): name of the simulation element sending this message
            - dst (`str`): name of the intended simulation element to receive this message
            - plan (`list`): list of agent actions to perform
            - t_plan (`float`): time at which the plan was created
            - id (`str`) : Universally Unique IDentifier for this message
        """
        super().__init__(src, dst, SimulationMessageTypes.PLAN.value, id, path)
        self.agent_name = agent_name if agent_name else dst
        self.plan = plan
        self.t_plan = t_plan

class SenseMessage(SimulationMessage):
    """
    # Bus Message
    
    Message containing other messages meant to be broadcasted in the same transmission

    ### Attributes:
        - src (`str`): name of the simulation element sending this message
        - dst (`str`): name of the intended simulation element to receive this message
        - senses (`list`): list of senses from the agent
        - msg_type (`str`): type of message being sent
        - id (`str`) : Universally Unique IDentifier for this message
    """
    def __init__(self, src: str, dst: str, state : dict, senses : list, id: str = None, path : list = [], **_):
        """
        Creates an instance of a plan message

        ### Attributes:
            - src (`str`): name of the simulation element sending this message
            - dst (`str`): name of the intended simulation element to receive this message
            - senses (`list`): list of senses from the agent
            - id (`str`) : Universally Unique IDentifier for this message
        """
        super().__init__(src, dst, SimulationMessageTypes.SENSES.value, id, path)
        self.state = state
        self.senses = senses

class AgentActionMessage(SimulationMessage):
    """
    ## Agent Action Message

    Informs the receiver of a action to be performed and its completion status
    """
    def __init__(self, src: str, dst: str, action : dict, status : str=None, id: str = None, path : list = [], **_):
        super().__init__(src, dst, SimulationMessageTypes.AGENT_ACTION.value, id, path)
        self.action = action
        self.status = None
        self.status = status if status is not None else action.get('status', None)

class BusMessage(SimulationMessage):
    """
    ## Bus Message

    A longer message containing a list of other messages to be sent in the same transmission
    """
    def __init__(self, src: str, dst: str, msgs : list, id: str = None, path : list = [], **_):
        super().__init__(src, dst, SimulationMessageTypes.BUS.value, id, path)
        
        if not isinstance(msgs, list):
            raise AttributeError(f'`msgs` must be of type `list`; is of type {type(msgs)}')
        for msg in msgs:
            if not isinstance(msg, (dict, SimulationMessage)):
                raise AttributeError(f'elements of the list `msgs` must be of type `dict` or `SimulationMessage`; contains elements of type {type(msg)}')

        self.msgs = msgs