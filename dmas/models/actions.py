from typing import Any, Union
from abc import ABC
from enum import Enum
import numpy as np
import json
import uuid

from execsatm.observations import ObservationOpportunity

class CoordinateTypes(Enum):
    """
    # Coordinate Type

    Describes the type of coordinate being described by a position vector
    """
    CARTESIAN = 'CARTESIAN'
    KEPLERIAN = 'KEPLERIAN'
    LATLON = 'LATLON'
   
class ActionTypes(Enum):
    IDLE = 'IDLE'
    TRAVEL = 'TRAVEL'
    MANEUVER = 'MANEUVER'
    BROADCAST = 'BROADCAST'
    WAIT = 'WAIT'
    OBSERVE = 'OBSERVE'
    REPLAN = 'REPLAN'

class ActionStatuses(Enum):
    PENDING = 'PENDING'
    COMPLETED = 'COMPLETED'
    ABORTED = 'ABORTED'
    FAILED = 'FAILED'

class AgentAction(ABC):
    """
    ## Agent Action
    
    Describes an action to be performed by an agent

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
    """

    def __init__(   self, 
                    action_type : str,
                    t_start : Union[float, int],
                    t_end : Union[float, int] = np.Inf, 
                    status : str = ActionStatuses.PENDING.value,
                    id : str = None,
                    **_

                ) -> None:
        """
        Creates an instance of an agent action

        ### Arguments:
            - t_start (`float`): start time of the availability of this task in [s] from the beginning of the simulation
            - t_end (`float`): end time of the availability of this task in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__()

        # type and value checks
        if not isinstance(t_start, float) and not isinstance(t_start, int):
            raise AttributeError(f'`t_start` must be of type `float` or type `int`. is of type {type(t_start)}.')
        elif t_start < 0:
            raise ValueError(f'`t_start` must be a value higher than 0. is of value {t_start}.')
        if not isinstance(t_end, float) and not isinstance(t_end, int):
            raise AttributeError(f'`t_end` must be of type `float` or type `int`. is of type {type(t_end)}.')
        elif t_end < 0:
            raise ValueError(f'`t_end` must be a value higher than 0. is of value {t_end}.')
        if t_start > t_end:
            raise ValueError(f'`t_start must be lower or equal than `t_end` (t_start: {t_start}, t_end: {t_end}.')

        # assign values 
        self.action_type = action_type
        self.t_start = t_start
        self.t_end = t_end
        self.status = status
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def __eq__(self, other) -> bool:
        """
        Compares two instances of a task. Returns True if they represent the same task.
        """
        assert isinstance(other, AgentAction), f"Can only compare with another `AgentAction`. is of type {type(other)}."
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this task object
        """
        return dict(self.__dict__)
    
    def to_json(self) -> str:
        """
        Creates a json file from this task 
        """
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        """
        Creates a string representing the contents of this task
        """
        return str(self.to_dict())
    
    def __hash__(self) -> int:
        return hash(repr(self))
    
    def __repr__(self) -> str:
        return f"{self.action_type}_{self.id.split('-')[0]}"
    
class IdleAction(AgentAction):
    """
    ## Idle Action

    Instructs an agent to idle for a given amount of time

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
    """
    def __init__(   self, 
                    t_start : Union[float, int],
                    t_end : Union[float, int], 
                    status : str = 'PENDING',
                    id: str = None, 
                    **_
                ) -> None:
        """
        Creates an isntance of an Idle Action

        ### Arguments:
            - t_start (`float`): start time of this task in [s] from the beginning of the simulation
            - t_end (`float`): end time of this this task in [s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.IDLE.value, t_start, t_end, status, id)

class TravelAction(AgentAction):
    """
    ## Travel Action

    Instructs an agent to travel to a particular position
    
    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
        - final_pos (`list`): coordinates desired destination
        - pos_type (`str`): coordinate basis being used for the desired destination
    """
    def __init__(self,
                final_pos : list, 
                t_start : Union[float, int],
                t_end : Union[float, int] = np.Inf,
                pos_type : str = CoordinateTypes.CARTESIAN.value,
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        """
        Creates an instance of a Travel Action

        ### Arguments:
            - final_pos (`list`): coordinates desired destination
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - pos_type (`str`): coordinate basis being used for the desired destination
            - status (`str`): completion status of the action
            - id (`str`) : identifying number for this action in uuid format
        """
            
        super().__init__(ActionTypes.TRAVEL.value, t_start, t_end, status=status, id=id)
        
        if not isinstance(final_pos, list):
            raise AttributeError(f'`final_pos` must be of type `list`. is of type {type(final_pos)}.')
        
        if pos_type == CoordinateTypes.CARTESIAN.value and len(final_pos) != 3:
            raise ValueError(f'`final_pos` must be a list of 3 values (x, y, z). is of length {len(final_pos)}.')
        elif pos_type == CoordinateTypes.KEPLERIAN.value and len(final_pos) != 5:
            raise ValueError(f'`final_pos` must be a list of 5 values (lat, lon, alt). is of length {len(final_pos)}.')
        elif pos_type == CoordinateTypes.LATLON.value and len(final_pos) != 3:
            raise ValueError(f'`final_pos` must be a list of 3 values (lat, lon, alt). is of length {len(final_pos)}.')
        elif (pos_type != CoordinateTypes.CARTESIAN.value
             and pos_type != CoordinateTypes.KEPLERIAN.value
             and pos_type != CoordinateTypes.LATLON.value):
            raise NotImplemented(f'`pos_type` or type `{pos_type}` not yet supported for `MoveAction`.')

        self.final_pos = final_pos
        self.pos_type = pos_type

class ManeuverAction(AgentAction):
    """
    ## Maneuver Action

    Instructs a satellite agent to perform an attitude maneuver
    
    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of this this action in [s] from the beginning of the simulation
        - status (`str`): completion status of the action
        - id (`str`) : identifying number for this action in uuid format
        - final_attitude (`float`): desired off-nadir angle parallel to velocity vector
    """
    def __init__(self,
                 initial_attitude : list,
                 final_attitude : list, 
                 attitude_rates : list,
                 t_start : Union[float, int],
                 t_end : Union[float, int] = np.Inf,
                 status : str = 'PENDING',
                 id: str = None, 
                 **_) -> None:
        super().__init__(ActionTypes.MANEUVER.value, t_start, t_end, status=status, id=id)
        
        # validate inputs
        assert isinstance(initial_attitude, list), f'`initial_attitude` must be of type `list`. Is of type {type(initial_attitude)}.'
        assert len(initial_attitude) == 3, f'`initial_attitude` must be of type `list` of length 3. Is of length {len(initial_attitude)}.'
        assert isinstance(final_attitude, list), f'`final_attitude` must be of type `list`. Is of type {type(final_attitude)}.'
        assert len(final_attitude) == 3, f'`final_attitude` must be of type `list` of length 3. Is of length {len(final_attitude)}.'
        assert isinstance(attitude_rates, list), f'`attitude_rates` must be of type `list`. Is of type {type(attitude_rates)}.'
        assert len(attitude_rates) == 3, f'`attitude_rates` must be of type `list` of length 3. Is of length {len(attitude_rates)}.'

        assert all(isinstance(th, (float, int)) for th in initial_attitude), f'all values in `initial_attitude` must be numerical values of type `float` or `int`.'
        assert all(isinstance(th, (float, int)) for th in final_attitude), f'all values in `final_attitude` must be numerical values of type `float` or `int`.'
        assert all(isinstance(dth, (float, int)) for dth in attitude_rates), f'all values in `attitude_rates` must be numerical values of type `float` or `int`.'

        for i in range(3):
            dth_req = (final_attitude[i] - initial_attitude[i]) / (t_end - t_start) if t_end > t_start else np.Inf
            assert abs(dth_req - attitude_rates[i]) <= 1e-3, f'Required attitude rate for axis {i} is {dth_req:.3f} deg/s which exceeds the maximum allowed rate of {attitude_rates[i]:.3f} deg/s.'

        # set parameters
        self.initial_attitude = [th for th in initial_attitude]
        self.final_attitude = [th for th in final_attitude]
        self.attitude_rates = [dth for dth in attitude_rates]

class BroadcastMessageAction(AgentAction):
    """
    ## Broadcast Message Action 

    Instructs an agent to broadcast a message to all of its peers

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - msg (`dict`): message to be broadcasted to other agents in the network
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): start time of this actrion in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
    def __init__(self, 
                msg : dict,
                t_start : Union[float, int],
                status : str = 'PENDING',
                id: str = None, 
                **_) -> None:
        """
        Creates an instance of a Broadcast Message Action

        ### Arguments
            - msg (`dict`): message to be broadcasted to other agents in the network
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.BROADCAST.value, t_start, t_start, status=status, id=id)
        
        assert isinstance(msg, dict), f'`msg` must be of type `dict`. Is of type `{type(msg)}`.'
        assert len(msg) > 0, '`msg` cannot be an empty dictionary.'

        self.msg = msg

class FutureBroadcastMessageAction(BroadcastMessageAction):
    """
    ## Future Broadcast Message Action 

    Instructs an agent that a message is to be broadcast to all of its peers but the contents are not yet kown

    ### Attributes:
        - broadcast_type (`str`): type of broadcast to be performed
        - t_start (`float`): start time of this action in [s] from the beginning of the simulation
        - t_end (`float`): start time of this actrion in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
    PLAN = 'PLAN'                           # broadcast latest planner information
    BIDS = 'BIDS'                           # broadcast latest bids for a task
    REQUESTS = 'measurement requests'       # broadcast latest known active measurement requests
    OBSERVATIONS = 'observation results'    # broadcast latest observation info
    REWARD = 'REWARD'                       # broadcast latest reward grid information
    STATE = 'agent state'                   # broadcast latest agent state information
    FUTURE_BROADCAST_TYPES = [PLAN, BIDS, REQUESTS, OBSERVATIONS, REWARD, STATE]
        
    def __init__(self, 
                broadcast_type : str,
                t_start : Union[float, int],
                only_own_info : bool = True,
                desc : Any = None,
                status : str = 'PENDING',
                id: str = None, 
                **_
                ) -> None:
        """
        Creates an instance of a Future Broadcast Message Action

        ### Arguments
            - broadcast_type (`dict`): type of broadcast to be performed
            - t_start (`float`): start time of this action in [s] from the beginning of the simulation
            - own_info (`bool`): whether to only include own agent's information in the broadcast
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        # initialize parent class
        try:
            super().__init__(dict(), t_start, status, id)
        except AssertionError:
            pass  # bypass parent class msg check since msg is not yet known:

        # validate inputs
        assert broadcast_type in self.FUTURE_BROADCAST_TYPES, f'`broadcast_type` must be one of {self.FUTURE_BROADCAST_TYPES}. Is `{broadcast_type}`.'
    
        # set parameters
        self.broadcast_type = broadcast_type
        self.only_own_info = only_own_info
        self.desc = desc

class ObservationAction(AgentAction):
    """
    Describes an observation to be performed by agents in the simulation

    ### Attributes:
        - instrument_name (`str`): name of the instrument_name that will perform this action
        - look_angle (`float`): look angle of the observation in [deg]
        - t_start (`float`): start time of the measurement of this action in [s] from the beginning of the simulation
        - t_end (`float`): end time of the measurement of this action in [s] from the beginning of the simulation
        - duration (`float`): duration of the measurement of this action in [s]
        - obs_opp (`ObservationOpportunity`): the task observation opportunity associated with this action
        - id (`str`) : identifying number for this task in uuid format
    """  
    def __init__(   self,
                    instrument_name : str,
                    look_angle : float, 
                    t_start: Union[float, int], 
                    duration: Union[float, int] = 0.0, 
                    obs_opp : ObservationOpportunity = None,
                    status: str = 'PENDING', 
                    id: str = None, 
                    **_) -> None:
        """
        Creates an instance of an Observation Action
        ### Arguments:
            - instrument_name (`str`): name of the instrument_name that will perform this action
            - look_angle (`float`): look angle of the observation in [deg]
            - t_start (`float`): start time of the measurement of this action in [s] from the beginning of the simulation
            - duration (`float`): duration of the measurement of this action in [s]
            - obs_opp (`ObservationOpportunity`): the task observation opportunity associated with this action
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.OBSERVE.value, t_start, t_start + duration, status, id)
        
        # Concert task from dict if needed
        obs_opp = ObservationOpportunity.from_dict(obs_opp) if isinstance(obs_opp, dict) else obs_opp
        
        # check parameters
        assert isinstance(instrument_name,str), f'`instrument_name` must be of type `str`. Is of type `{type(instrument_name)}`.'
        assert isinstance(look_angle,(int,float)), f'`look_angle` must be a numerical value of type `float` or `int`. Is of type `{type(look_angle)}`'
        assert isinstance(obs_opp,ObservationOpportunity) or obs_opp is None, f'`task` must be of type `SpecificObservationTask` or None. Is of type `{type(obs_opp)}`.'

        # set parameters
        self.instrument_name = instrument_name
        self.look_angle = look_angle
        self.obs_opp : ObservationOpportunity = obs_opp

    def to_dict(self):
        out = super().to_dict()
        out['obs_opp'] = self.obs_opp.to_dict() if self.obs_opp else 'None'
        return out

class WaitAction(AgentAction):
    """
    ## Wait Action

    Instructs an agent to idle until a given time.

    ### Attributes:
        - action_type (`str`): type of action to be performed
        - t_start (`float`): start time of the waiting period in [s] from the beginning of the simulation
        - t_end (`float`): start time of the waiting period in[s] from the beginning of the simulation
        - status (`str`): completion status of the task
        - id (`str`) : identifying number for this task in uuid format
    """
    def __init__(   self, 
                    t_start: Union[float, int], 
                    t_end: Union[float, int] = np.Inf, 
                    status: str = 'PENDING', 
                    id: str = None, 
                    **_
                ) -> None:
        """
        Creates an isntance of a Waif For Message Action

         ### Arguments:
            - t_start (`float`): start time of the waiting period in [s] from the beginning of the simulation
            - t_end (`float`): start time of the waiting period in[s] from the beginning of the simulation
            - status (`str`): completion status of the task
            - id (`str`) : identifying number for this task in uuid format
        """
        super().__init__(ActionTypes.WAIT.value, t_start, t_end, status, id)

class TriggerReplan(AgentAction):
    """
    ## Replan Action

    Instructs the planner to generate a new plan for the agent. Used to schedule future replanning events.
    """
    def __init__(self, 
                t_start: Union[float, int], 
                status: str = 'PENDING', 
                id: str = None, 
                **_
            ) -> None:
        super().__init__(ActionTypes.REPLAN.value, t_start, t_start, status, id)


def action_from_dict(action_type : str, **kwargs) -> AgentAction:
    if action_type == ActionTypes.IDLE.value:
        return IdleAction(**kwargs)
    elif action_type == ActionTypes.TRAVEL.value:
        return TravelAction(**kwargs)
    elif action_type == ActionTypes.MANEUVER.value:
        return ManeuverAction(**kwargs)
    elif action_type == ActionTypes.BROADCAST.value:
        if 'broadcast_type' in kwargs:
            return FutureBroadcastMessageAction(**kwargs)
        else:
            return BroadcastMessageAction(**kwargs)
    elif action_type == ActionTypes.WAIT.value:
        return WaitAction(**kwargs)
    elif action_type == ActionTypes.OBSERVE.value:
        return ObservationAction(**kwargs)
    else:
        raise NotImplementedError(f'Action of type {action_type} not yet implemented.')