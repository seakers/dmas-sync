from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Tuple

import numpy as np

from chess3d.agents.science.requests import TaskRequest # TODO bids from task request method?
from chess3d.agents.planning.tasks import GenericObservationTask


def bid_comparison_input_checks( func : Callable ) -> Callable:
    """ Decorator to validate inputs for bid comparison methods """
    def checker(self, other : object, *args) -> Any:
        # validate inputs
        assert isinstance(self, Bid) and isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'
        assert self.n_obs == other.n_obs, f'cannot compare bids intended for different image numbers (expected image number: {self.n_obs}, given image number: {other.n_obs})'

        # perform comparison
        return func(self, other, *args)
        
    return checker

class Bid(ABC): 

    """
    ## Measurement Task Bid for Consensus Planners

    Describes a bid placed on a task by a given agent
    """
    # Constants
    NONE = 'none'   # NONE value used in various bid attributes
    EPS = 1e-6      # small epsilon value for float comparisons
    
    # Bid comparison results
    UPDATE_TIME = 'update_time'
    UPDATE = 'update'
    LEAVE = 'leave'
    RESET = 'reset'
    COMPLETED = 'completed'
    COMPARISON_RESULTS = [UPDATE_TIME, UPDATE, LEAVE, RESET, COMPLETED]

    # Rebroadcast comparison results
    REBROADCAST_SELF = 'rebroadcast self'
    REBROADCAST_OTHER = 'rebroadcast other'
    REBROADCAST_EMPTY = 'rebroadcast empty'
    NO_REBROADCAST = 'no rebroadcast'
    REBORADCAST_RESULTS = [REBROADCAST_SELF, REBROADCAST_OTHER, REBROADCAST_EMPTY]

    def __init__(self,
                 task : GenericObservationTask,
                 bidder: str,
                 n_obs: int = 0,
                 bid_value: Union[float, int] = 0,
                 winning_bidder: str = NONE,
                 winning_bid: Union[float, int] = 0,
                 t_img: Union[float, int] = np.NINF,
                 t_stamp: Union[float, int] = np.NINF,
                 main_measurement : str = NONE,
                 performed : bool = False,
                 ):
        """
        ## Measurement Task Bid for Consensus Planners

        Creates an instance of a task bid

        ### Attributes:
            - task (`GenericObservationTask`): observation task being bid on
            - bidder (`bidder`): name of the agent keeping track of this bid information
            - n_obs (`int`): image number associated with this bid
            - main_measurement (`str`): name of the main measurement assigned by this subtask bid
            - bid_value (`float` or `int`): latest bid value from bidder
            - winning_bidder (`str`): name of current the winning agent
            - winning_bid (`float` or `int`): current winning bid value
            - t_img (`float` or `int`): time where the task is set to be performed by the winning agent
            - t_stamp (`float` or `int`): latest time-stamp when this bid was updated
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
        """

        # Validate inputs
        assert isinstance(task, GenericObservationTask), f'`task` must be of type `GenericObservationTask`, got `{type(task)}`'
        assert isinstance(bidder, str), f'`bidder` must be of type `str`, got `{type(bidder)}`'
        assert isinstance(n_obs, int) and n_obs >= 0, f'`n_img` must be positive `int`, got `{type(n_obs)}-{n_obs}`'
        assert isinstance(bid_value, (float, int)), f'`bid_value` must be of type `float` or `int`, got `{type(bid_value)}`'
        assert isinstance(winning_bidder, str), f'`winning_bidder` must be of type `str`, got `{type(winning_bidder)}`'
        assert isinstance(winning_bid, (float, int)), f'`winning_bid` must be of type `float` or `int`, got `{type(winning_bid)}`'
        assert isinstance(t_img, (float, int)), f'`t_img` must be of type `float` or `int`, got `{type(t_img)}`'
        assert t_img in task.availability or t_img == np.NINF, f'`t_img` value `{t_img}` not in task availability interval `{task.availability}`'
        assert isinstance(t_stamp, (float, int)), f'`t_update` must be of type `float` or `int`, got `{type(t_stamp)}`'
        assert isinstance(main_measurement, str), f'`main_measurement` must be of type `str`, got `{type(main_measurement)}`'
        assert isinstance(performed, bool), f'`performed` must be of type `bool`, got `{type(performed)}`'

        # Assign attributes
        self.task = task
        self.bidder = bidder
        self.n_obs = n_obs
        self.bid_value = bid_value
        self.winning_bidder = winning_bidder
        self.winning_bid = winning_bid
        self.t_img = t_img
        self.t_stamp = t_stamp
        self.main_measurement = main_measurement
        self.performed = performed

    """
    ---------------------------
    COPY AND OTHER CONSTRUCTORS
    ---------------------------
    """

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this bid
        """
        try:
            bid_dict = {
                'task': self.task.to_dict(),
                'main_measurement': self.main_measurement,
                'bidder': self.bidder,
                'bid_value': self.bid_value,
                'winning_bidder': self.winning_bidder,
                'winning_bid': self.winning_bid,
                't_img': self.t_img,
                'n_img': self.n_obs,
                't_stamp': self.t_stamp,
                'performed': self.performed
            }
            return bid_dict
        finally:
            # check if all required keys are present
            required_keys = ['task', 'main_measurement', 'bidder', 'bid_value', 'winning_bidder', 'winning_bid', 't_img', 'n_img', 't_stamp', 'performed']
            assert all(key in bid_dict for key in required_keys), \
                f'Bid dictionary is missing required keys. Required keys: {required_keys}'
            

    @classmethod
    def from_dict(cls, bid_dict: dict) -> 'Bid':
        """
        Creates a bid class object from a dictionary
        """
        # check if all required keys are present
        required_keys = ['task', 'main_measurement', 'bidder', 'bid_value', 'winning_bidder', 'winning_bid', 't_img', 'n_img', 't_stamp', 'performed']
        assert all(key in bid_dict for key in required_keys), \
            f'Bid dictionary is missing required keys. Required keys: {required_keys}'
        
        # convert task from dict if necessary
        if isinstance(bid_dict['task'], dict):
            task = GenericObservationTask.from_dict(bid_dict['task'])
        elif isinstance(bid_dict['task'], GenericObservationTask):
            task = bid_dict['task']

        # return bid object
        return cls(
            task=task,
            main_measurement=bid_dict['main_measurement'],
            bidder=bid_dict['bidder'],
            bid_value=bid_dict['bid_value'],
            winning_bidder=bid_dict['winning_bidder'],
            winning_bid=bid_dict['winning_bid'],
            t_img=bid_dict['t_img'],
            n_obs=bid_dict.get('n_obs', 0),
            t_stamp=bid_dict['t_stamp'],
            performed=bid_dict['performed']
        )
    
    @abstractmethod
    def copy(self) -> 'Bid':
        """ Creates a deep copy of this bid object """

    """
    ------------------
    BID VALUE COMPARISON METHODS
    ------------------

    Used to compare bids based on their bid values and winning bidders using standard comparison operators.

    """

    @bid_comparison_input_checks
    def __lt__(self, other : 'Bid') -> bool:       
        # check if equal
        if abs(other.bid_value - self.bid_value) < self.EPS:
            # if there's a tie, use tie-breaker
            largest_bid = self.__tie_breaker(self, other)
            return self is not largest_bid

        # compare bids
        return other.bid_value > self.bid_value

    @bid_comparison_input_checks
    def __gt__(self, other : 'Bid') -> bool:
        # check if equal
        if abs(other.bid_value - self.bid_value) < self.EPS:
            # if there's a tie, use tie-breaker
            largest_bid = self.__tie_breaker(self, other)
            return self is largest_bid

        # compare bids
        return other.bid_value < self.bid_value

    @bid_comparison_input_checks
    def __eq__(self, other : 'Bid') -> bool:        
        # compare bids
        return abs(other.winning_bid - self.winning_bid) < self.EPS    # same bid value

    @bid_comparison_input_checks
    def __ne__(self, other : 'Bid') -> bool:        
        # compare bids
        return abs(other.winning_bid - self.winning_bid) > self.EPS    # different bid value

    @bid_comparison_input_checks
    def __le__(self, other : 'Bid') -> bool:
        # compare bids
        return other.bid_value >= self.bid_value or abs(other.bid_value - self.bid_value) < self.EPS
    
    @bid_comparison_input_checks
    def __ge__(self, other : 'Bid') -> bool:
        # compare bids
        return other.bid_value <= self.bid_value or abs(other.bid_value - self.bid_value) < self.EPS

    def __tie_breaker(self, bid1 : 'Bid', bid2 : 'Bid') -> 'Bid':
        """
        Tie-breaking criteria for determining which bid is GREATER in case winning bids are equal
        """
        # validate inputs
        assert isinstance(bid1, Bid) and isinstance(bid2, Bid), f'cannot calculate tie breaker. Both objects must be bids.'

        # compare bids
        ## Check for NONE winning bidders
        if bid2.winning_bidder == self.NONE and bid1.winning_bidder != self.NONE:
            return bid2
        elif bid2.winning_bidder != self.NONE and bid1.winning_bidder == self.NONE:
            return bid1

        ## Compare bidders alphabetically
        return max(bid1, bid2, key=lambda b: b.bidder)

    """
    ------------------
    BID COMPARISON AND UPDATE METHODS
    ------------------

    Compares the values of a bid with another bid and decides to either update, reset, or leave the information contained in this bid
    along with deciding thether to broadcast the updated information to neighboring agents based on the rules specified in:
        - Luke B. Johnson, Sameera S. Ponda, Han-Lim Choi, Jonathan P. How "Asynchronous Decentralized Task Allocation for Dynamic Environments".

    """
    @abstractmethod
    def compare(self, other : 'Bid') -> Tuple[str,str]:
        """ 
        Compares bid with another and indicates whether the bid shouls be updated, left, or reset.
        Also returns whether the bid should be rebroadcasted to neighboring agents.

        ### Arguments:
            - other (`Bid`): bid being compared to

        ### Returns: Tuple(comparison, rebroadcast)
            - comparison (`str`): action to perform to this bid upon comparing this bid to the other bid
            - rebroadcast (`str`): rebroadcast action to perform after comparing this bid to the other bid
        """
        # convert other bid to `Bid` class type if necessary
        other : Bid = Bid.from_dict(other) if isinstance(other, dict) else other

        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'can only compare bids for the same task (expected task id: {self.task.id}, given id: {other.task.id})'
        
        # 1. Sending agent claims itself as winner of this bid.
        if other.believes_i_am_winning():
            return self._case_other_thinks_is_winner(other)

        # 2. Sending agent claims I am the winner of this bid.
        if other.believes_is_other_winning(self):
            return self._case_other_thinks_im_winner(other)

        # 3. Sending agent claims some 3rd party as the winner of this bid.
        if other.believes_third_party_is_winning(self):
            return self._case_other_thinks_third_party_winner(other)
        
        # 4. Sending agent has no winner for this bid.
        if other.believes_no_winner():
            return self._case_other_has_no_winner(other)

        # 5. Fallback (should be unreachable)
        raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.bidder}` and bidder `{other.bidder}`.')

    @abstractmethod
    def _case_other_thinks_is_winner(self, other : 'Bid') -> Tuple[str, str]:
        """ Case: Sending agent claims itself as winner of this bid. """

    @abstractmethod
    def _case_other_thinks_im_winner(self, other : 'Bid') -> Tuple[str, str]:
        """ Case: Sending agent claims I am the winner of this bid. """

    @abstractmethod
    def _case_other_thinks_third_party_winner(self, other : 'Bid') -> Tuple[str, str]:        
        """ Handles the case where the other bid thinks a third party is the winner """

    @abstractmethod
    def _case_other_has_no_winner(self, other : 'Bid') -> Tuple[str, str]:
        """ Handles the case where the other bid has no winner """

    def is_bid_mine(self, other : 'Bid') -> bool:
        return other.bidder == self.bidder

    def believes_i_am_winning(self) -> bool:
        """ Checks if this bid is currently won by the bidder itself """
        return self.winning_bidder == self.bidder

    def believes_is_other_winning(self, other: 'Bid') -> bool:
        """ Checks if this bid is currently won by the other bidder """
        return self.winning_bidder == other.bidder

    def believes_third_party_is_winning(self, other: 'Bid') -> bool:
        """ Checks if this bid is currenly won by a 3rd party (neither self nor other) """
        return self.winning_bidder not in {self.bidder, other.bidder, self.NONE}

    def believes_no_winner(self) -> bool:
        """ Checks if this bid has no winner """
        return self.winning_bidder == self.NONE
    
    def is_same_winner(self, other: 'Bid') -> bool:
        return self.winning_bidder == other.winning_bidder

    def is_tie(self, other: 'Bid') -> bool:
        """ Checks if this bid is tied with another bid """
        return abs(self.winning_bid - other.winning_bid) < self.EPS

    def wins_tie_breaker(self, other: 'Bid') -> bool:
        """ Returns True if, when bids are tied, we should prefer `self` over `other`. """
        larger_name_bid = self.__tie_breaker(self, other)
        return larger_name_bid is not self
    
    def is_same_timestamp(self, other: 'Bid') -> bool:
        """ Checks if this bid has the same timestamp as another bid """
        return abs(self.t_stamp - other.t_stamp) < self.EPS
    
    """
    ---------------------------
    MODIFIERS
    ---------------------------
    """

    def update(self, other : object, t : float) -> 'Bid':
        """ Returns a bid with the updated information after comparing this bid to another bid """
        # compare bids
        comp_result, _ = self.compare(other)
        comp_result : str

        # update accordingly 
        new_bid : Bid = self.copy()
        if comp_result is self.UPDATE_TIME:
            new_bid.__update_time(t)
        elif comp_result is self.UPDATE:
            new_bid._update_info(other, t)
        elif comp_result is self.RESET:
            new_bid.reset(t)
        elif comp_result is self.LEAVE:
            new_bid._leave(t)
        elif comp_result is self.COMPLETED:
            new_bid._perform(t)
        else:
            raise ValueError(f'cannot perform update of type `{comp_result}`')
        
        # return updated bid
        return new_bid
    
    def __update_time(self, t_update : float) -> None:
        """Records the lastest time this bid was updated"""
        self.t_stamp = t_update
    
    def _update_info(self, other : 'Bid', t : float) -> None:
        """
        Updates all of the variable bid information

        ### Arguments:
            - other (`Bid`): equivalent bid being used to update information
        """
        # validate inputs
        assert isinstance(other, Bid), f'can only update bids from other bids.'
        assert self.task == other.task, f'cannot update bid with information from another bid intended for another task (expected task id: {self.task}, given id: {other.task}).'

        # update bid information
        self.winning_bid = other.winning_bid
        self.winning_bidder = other.winning_bidder
        self.main_measurement = other.main_measurement
        self.t_img = other.t_img

        self.t_stamp = t
        self.performed = other.performed if not self.performed else True # Check if this hold true for all values

    def reset(self, t_update : float) -> None:
        """
        Resets the values of this bid while keeping track of lates update time
        """
        assert isinstance(t_update, (float, int)), f'`t_update` must be of type `float` or `int`, got `{type(t_update)}`'
        assert t_update >= 0, f'`t_update` must be non-negative, got `{t_update}`'

        self.winning_bid = 0
        self.winning_bidder = self.NONE
        self.main_measurement = self.NONE
        self.t_img = -1
        self.t_stamp = t_update

    def _leave(self, _, **__) -> None:
        """
        Leaves bid as is (used for code readibility).

        ### Arguments:
            - t_update (`float` or `int`): latest time when this bid was updated
        """
        return
    
    def _perform(self, t_update : float) -> None:
        """ Indicates that this action has been performed """
        self.performed = True
        self.t_stamp = t_update
        
    def set(self, 
            main_instrument : str,
            new_bid : Union[int, float], 
            t_img : Union[int, float], 
            n_img : int,
            t_update : Union[int, float]
        ) -> None:
        """
        Sets new values for this bid

        ### Arguments: 
            - main_instrument (`str`): main measurement set to perform this bid
            - new_bid (`int` or `float`): new bid value
            - t_img (`int` or `float`): new imaging time
            - n_img (`int`): image number
            - t_update (`int` or `float`): update time
        """
        self.main_instrument = main_instrument
        self.winning_bid = new_bid
        self.winning_bidder = self.bidder
        self.t_img = t_img
        self.n_obs = n_img
        self.t_stamp = t_update
    
    def has_winner(self) -> bool:
        """
        Checks if this bid has a winner
        """
        return self.winning_bidder != Bid.NONE

    def set_performed(self, t : float, performed : bool = True, performer : str = None) -> None:
        """
        Sets the performed status of this bid

        ### Arguments:
            - t (`float` or `int`): latest time when this bid was updated
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
            - performer (`str`): name of the agent that performed the task
        """
        self.__update_time(t)
        self.winning_bidder = performer if performer is not None else self.bidder
        self.performed = performed
        self.t_img = t

    def was_performed(self) -> bool:
        """
        Checks if the winner of this bid has performed the measurement request at hand
        """
        return self.performed

    """
    ---------------------------
    STRING REPRESENTATION
    ---------------------------
    """

    def __str__(self) -> str:
        """
        Returns a string representation of this task bid in the following format:
        - `task_id`, `main_measurement`, `target`, `bidder`, `bid`, `winner`, `t_img`, `t_update`
        """
        
        split_id = self.task.id.split('-')
        line_data = {   "task_id" : split_id[0], 
                        "main_measurement" : self.main_measurement, 
                        "target" : self.task.location, 
                        "bidder" : self.bidder, 
                        "bid" : round(self.winning_bid, 3), 
                        "winner" : self.winning_bidder, 
                        "t_img" : round(self.t_img, 3),
                        "t_update" : round(self.t_stamp, 3)
                    }
        out = "Bid("
        for key, value in line_data.items():
            out += f"{key}={value}, "
        out = out[:-2] + ")"

        return out
    
    def __repr__(self):
        task_id = self.task.id.split('-')
        return f'Bid_{task_id[0]}_{self.n_obs}_{self.bidder}_{round(self.winning_bid,1)}'

    def __hash__(self) -> int:
        return hash(repr(self))  


class AsynchronousBid(Bid):
    def __init__(self, task, bidder, n_obs = 0, bid_value = 0, winning_bidder = Bid.NONE, winning_bid = 0, t_img = np.NINF, t_stamp = np.NINF, main_measurement = Bid.NONE, performed = False):
        """ Asynchronous Bid class implementing the bid comparison method according to:
            - Luke B. Johnson, Sameera S. Ponda, Han-Lim Choi, Jonathan P. How "Asynchronous Decentralized Task Allocation for Dynamic Environments".
        """
        super().__init__(task, bidder, n_obs, bid_value, winning_bidder, winning_bid, t_img, t_stamp, main_measurement, performed)

    def copy(self) -> 'Bid':
        """ Creates a deep copy of this bid object """
        return AsynchronousBid.from_dict(self.to_dict())

    def compare(self, other : 'Bid') -> Tuple[str,str]:
        """
        Compares bid with another and either updates, resets, or leaves the information contained in this bid
        depending on the rules specified in:

            - Luke B. Johnson, Sameera S. Ponda, Han-Lim Choi, Jonathan P. How "Asynchronous Decentralized Task Allocation for Dynamic Environments".

        ### Arguments:
            - other (`Bid`): bid being compared to

        ### Returns: Tuple(comparison, rebroadcast)

            - comparison (`self`) : action to perform to this bid upon comparing this bid to the other bid
            - rebroadcast (`self`): rebroadcast action to perform after comparing this bid to the other bid
        """
        return super().compare(other)
        
    def _case_other_thinks_is_winner(self, other : 'Bid') -> Tuple[str, str]:
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if self.is_tie(other):
                if self.wins_tie_breaker(other):
                    # bids tied & won tie-breaker → leave bid & do not rebroadcast
                    # NOTE : rebroadcasting here may cause oscillations
                    return self.LEAVE, self.NO_REBROADCAST
                else:
                    # bids tied & lost tie-breaker → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.winning_bid > self.winning_bid:
                # other bid is higher → update bid & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.winning_bid < self.winning_bid:
                # my bid is higher → update bid time & rebroadcast my bid
                return self.UPDATE_TIME, self.REBROADCAST_SELF            

        # 2. Receiving agent other is the winner already.
        if self.believes_is_other_winning(other):
            if self.is_same_timestamp(other):
                # bids are from the same time → leave bid & do not rebroadcast
                return self.LEAVE, self.NO_REBROADCAST

            elif other.t_stamp > self.t_stamp:
                # other bid is newer → update bid & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.t_stamp < self.t_stamp:
                # my bid is newer → leave bid & do not rebroadcast
                return self.LEAVE, self.NO_REBROADCAST

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if self.is_tie(other):
                # bids tied → leave bid & rebroadcast my bid
                return self.LEAVE, self.REBROADCAST_SELF

            elif other.winning_bid > self.winning_bid:
                if self.is_same_timestamp(other) or other.t_stamp > self.t_stamp:
                    # other bid is higher & same/newer time → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER
                
                elif other.t_stamp < self.t_stamp:
                    # other bid is higher & older time → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER

            elif other.winning_bid < self.winning_bid:
                if self.is_same_timestamp(other) or other.t_stamp < self.t_stamp:
                    # my bid is higher & same/newer time → leave bid & rebroadcast my bid
                    return self.LEAVE, self.REBROADCAST_SELF
                
                elif other.t_stamp > self.t_stamp:
                    # my bid is higher & older time → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER            

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # update bid & rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 5. Fallback (should be unreachable)
        raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.bidder}` and bidder `{other.bidder}`.')

    def _case_other_thinks_im_winner(self, other : 'Bid') -> Tuple[str, str]:
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if self.is_same_timestamp(other):
                # bids are from the same time → leave bid & do not rebroadcast
                return self.LEAVE, self.NO_REBROADCAST 
            
            elif other.t_stamp > self.t_stamp:
                # newer info from myself → adopt it & rebroadcast
                return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.t_stamp < self.t_stamp:
                # we are newer → keep own, no rebroadcast needed
                # NOTE : rebroadcasting here may cause oscillations
                return self.LEAVE, self.NO_REBROADCAST

        # 2. Receiving agent other is the winner already.
        if self.believes_is_other_winning(other):
            # reset & rebroadcast empty bid with current time
            return self.RESET, self.REBROADCAST_EMPTY

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            # leave & rebroadcast own information
            return self.LEAVE, self.REBROADCAST_SELF

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # leave & rebroadcast empty bid with current time
            return self.LEAVE, self.REBROADCAST_EMPTY

        # 5. Fallback (should be unreachable)
        raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.bidder}` and bidder `{other.bidder}`.')
        # # leave & do not rebroadcast
        # return self.LEAVE, self.NO_REBROADCAST

    def _case_other_thinks_third_party_winner(self, other : 'Bid') -> Tuple[str, str]:
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if self.is_tie(other):
                if self.wins_tie_breaker(other):
                    # bids tied & won tie-breaker → leave bid & do not rebroadcast
                    # NOTE : rebroadcasting here may cause oscillations
                    return self.LEAVE, self.NO_REBROADCAST
                else:
                    # bids tied & lost tie-breaker → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER
                
            elif other.winning_bid > self.winning_bid:
                # other bid is higher → update bid & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.winning_bid < self.winning_bid:
                # my bid is higher → update bid time & rebroadcast my bid
                return self.UPDATE_TIME, self.REBROADCAST_SELF

        # 2. Receiving agent other is the winner already.
        if self.believes_is_other_winning(other):
            # update and rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 3. Receiving agent also believes some 3rd party is winner.
        if self.is_same_winner(other):
            if self.is_same_timestamp(other):
                # same time stamp →  leave & do not rebroadcast
                return self.LEAVE, self.NO_REBROADCAST
            
            elif other.t_stamp > self.t_stamp:
                # other bid is newer → update bid & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER

            elif other.t_stamp < self.t_stamp:
                # my bid is newer → update bid time & rebroadcast my bid
                return self.LEAVE, self.REBROADCAST_SELF
        
        # 4. Receiving agent believes some 4th party is winner.
        if self.believes_third_party_is_winning(other):
            if self.is_tie(other):
                # cannot agree on winner → leave & do not rebroadcast
                # NOTE : rebroadcasting here may cause oscillations
                return self.LEAVE, self.NO_REBROADCAST
            
            elif other.winning_bid > self.winning_bid:
                if self.is_same_timestamp(other) or other.t_stamp > self.t_stamp:
                    # other bid is higher & same/newer time → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER

                elif other.t_stamp < self.t_stamp:
                    # my bid is newer → leave & rebroadcast my bid
                    return self.LEAVE, self.REBROADCAST_SELF

            elif other.winning_bid < self.winning_bid:
                if self.is_same_timestamp(other) or other.t_stamp < self.t_stamp:
                    # my bid is higher & same/newer time → leave & rebroadcast my bid
                    return self.LEAVE, self.REBROADCAST_SELF

                elif other.t_stamp > self.t_stamp:
                    # other bid is newer → update bid & rebroadcast other's bid
                    return self.UPDATE, self.REBROADCAST_OTHER

        # 5. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # update & rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 6. Fallback (should be unreachable)
        raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.bidder}` and bidder `{other.bidder}`.')

    def _case_other_has_no_winner(self, other : 'Bid') -> Tuple[str, str]:
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            # leave & rebroadcast own bid
            return self.LEAVE, self.REBROADCAST_SELF

        # 2. Receiving agent other is the winner already.
        if self.believes_is_other_winning(other):
            # update & rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER
            else:
                # my bid is newer → leave bid & rebroadcast my bid 
                # NOTE : rebroadcasting own bid here to ensure other agents get updated info
                return self.LEAVE, self.REBROADCAST_SELF

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # leave & do not rebroadcast
            return self.LEAVE, self.NO_REBROADCAST

        # 5. Fallback (should be unreachable)
        raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.bidder}` and bidder `{other.bidder}`.')
        

class SynchronousBid(Bid):
    def __init__(self, task, bidder, n_img = 0, bid_value = 0, winning_bidder = Bid.NONE, winning_bid = 0, t_img = np.NINF, t_stamp = np.NINF, main_measurement = Bid.NONE, performed = False):
        """ 
        Synchronous Bid class implementing the bid comparison method according to:
            - Li, Guoliang. "Online scheduling of distributed Earth observation satellite system under rigid communication constraints." Advances in Space Research 65.11 (2020): 2475-2496.

        """
        super().__init__(task, bidder, n_img, bid_value, winning_bidder, winning_bid, t_img, t_stamp, main_measurement, performed)

    def copy(self) -> 'Bid':
        """ Creates a deep copy of this bid object """
        return SynchronousBid.from_dict(self.to_dict())

    def compare(self, other : 'Bid') -> Tuple[str,str]:
        """
        Compares bid with another and either updates, resets, or leaves the information contained in this bid
        depending on the rules specified in:

            - Li, Guoliang. "Online scheduling of distributed Earth observation satellite system under rigid communication constraints." Advances in Space Research 65.11 (2020): 2475-2496.

        ### Arguments:
            - other (`Bid`): bid being compared to

        ### Returns: Tuple(comparison, rebroadcast)

            - comparison (`self`) : action to perform to this bid upon comparing this bid to the other bid
            - rebroadcast (`self`): rebroadcast action to perform after comparing this bid to the other bid
        """
        return super().compare(other)
    
    def _case_other_thinks_is_winner(self, other : 'Bid') -> Tuple[str, str]:
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if self.is_tie(other):
                if not self.wins_tie_breaker(other):
                    # bids tied & lost tie-breaker → update bid 
                    return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.winning_bid > self.winning_bid:
                # other bid is higher → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER
            
        # 2. Receiving agent other is the winner already.
        if self.believes_is_other_winning(other):
            # other bid is updated → update bid 
            return self.UPDATE, self.REBROADCAST_OTHER

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER
            
            elif self.is_tie(other):
                if not self.wins_tie_breaker(other):
                    # bids tied & lost tie-breaker → update bid 
                    return self.UPDATE, self.REBROADCAST_OTHER
            
            elif other.winning_bid > self.winning_bid:
                # other bid is higher → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER        
        
        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # update bid & rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 5. Fallback default case 
        return self.LEAVE, self.NO_REBROADCAST
    
    def _case_other_thinks_im_winner(self, other : 'Bid') -> Tuple[str, str]:
        """ Case: Sending agent claims I am the winner of this bid. """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            # it is my bid → leave & do not rebroadcast
            return self.LEAVE, self.NO_REBROADCAST

        # 2. Receiving agent believes other is the winner already.
        if self.believes_is_other_winning(other):
            # conflict → reset & rebroadcast self
            return self.RESET, self.REBROADCAST_SELF

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → reset & rebroadcast self
                return self.RESET, self.REBROADCAST_SELF

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # conflict → leave & do not rebroadcast
            return self.LEAVE, self.NO_REBROADCAST
        
        # 5. Fallback default case
        return self.LEAVE, self.NO_REBROADCAST

    def _case_other_thinks_third_party_winner(self, other : 'Bid') -> Tuple[str, str]:        
        """ Handles the case where the other bid thinks a third party is the winner """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if other.t_stamp > self.t_stamp:
                if self.is_tie(other):
                    if not self.wins_tie_breaker(other):
                        # other bid is newer and wins tie-braker → update bid 
                        return self.UPDATE, self.REBROADCAST_OTHER
                
                elif other.winning_bid > self.winning_bid:
                    # other bid is newer and has higher bid → update bid 
                    return self.UPDATE, self.REBROADCAST_OTHER

        # 2. Receiving agent believes other is the winner already.
        if self.believes_is_other_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER
            else:
                # my bid is newer → reset bid & rebroadcast self
                return self.RESET, self.REBROADCAST_SELF

        # 3. Receiving agent also believes some 3rd party is winner.
        if self.is_same_winner(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER
        
        # 4. Receiving agent believes some 4th party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER

        # 5. Receiving agent bid has no winner.
        if self.believes_no_winner():
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update bid 
                return self.UPDATE, self.REBROADCAST_OTHER
        
        # 6. Fallback default case
        return self.LEAVE, self.NO_REBROADCAST

    def _case_other_has_no_winner(self, other : 'Bid') -> Tuple[str, str]:
        """ Handles the case where the other bid has no winner """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            # it is my bid → leave & do not rebroadcast
            return self.LEAVE, self.NO_REBROADCAST

        # 2. Receiving agent believes other is the winner already.
        if self.believes_is_other_winning(other):
            # they abandoned their bid → update & rebroadcast other's bid
            return self.UPDATE, self.REBROADCAST_OTHER

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamp > self.t_stamp:
                # other bid is newer → update & rebroadcast other's bid
                return self.UPDATE, self.REBROADCAST_OTHER

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # no one is winning → leave & do not rebroadcast
            return self.LEAVE, self.NO_REBROADCAST
        
        # 5. Fallback default case
        return self.LEAVE, self.NO_REBROADCAST

# class BidBuffer(object):
#     """
#     Asynchronous buffer that holds bid information for use by processes within the MACCBBA
#     """
#     def __init__(self) -> None:
#         self.bid_access_lock = asyncio.Lock()
#         self.bid_buffer = {}
#         self.updated = asyncio.Event()             

#     def __len__(self) -> int:
#         l = 0
#         for req_id in self.bid_buffer:
#             for bid in self.bid_buffer[req_id]:
#                 bid : Bid
#                 l += 1 if bid is not None else 0
#         return l

#     async def pop_all(self) -> list:
#         """
#         Returns latest bids for all requests and empties buffer
#         """
#         await self.bid_access_lock.acquire()

#         out = []
#         for req_id in self.bid_buffer:
#             for bid in self.bid_buffer[req_id]:
#                 bid : Bid
#                 if bid is not None:
#                     # place bid in outgoing list
#                     out.append(bid)

#             # reset bids in buffer
#             self.bid_buffer[req_id] = [None for _ in self.bid_buffer[req_id]]

#         self.bid_access_lock.release()

#         return out

#     async def put_bid(self, new_bid : Bid) -> None:
#         """
#         Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
#         """
#         await self.bid_access_lock.acquire()

#         if new_bid.req_id not in self.bid_buffer:
#             req : TaskRequest = TaskRequest.from_dict(new_bid.req)
#             self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

#         current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]
        
#         if (    current_bid is None 
#                 or new_bid.bidder == current_bid.bidder
#                 or new_bid.t_update >= current_bid.t_update
#             ):
#             self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

#         self.bid_access_lock.release()

#         self.updated.set()
#         self.updated.clear()

#     async def put_bids(self, new_bids : list) -> None:
#         """
#         Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
#         """
#         if len(new_bids) == 0:
#             return

#         await self.bid_access_lock.acquire()

#         for new_bid in new_bids:
#             new_bid : Bid

#             if new_bid.req_id not in self.bid_buffer:
#                 req : TaskRequest = TaskRequest.from_dict(new_bid.req)
#                 self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

#             current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]

#             if (    current_bid is None 
#                  or (new_bid.bidder == current_bid.bidder and new_bid.t_update >= current_bid.t_update)
#                  or (new_bid.bidder != new_bid.NONE and current_bid.winning_bidder == new_bid.NONE and new_bid.t_update >= current_bid.t_update)
#                 ):
#                 self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

#         self.bid_access_lock.release()

#         self.updated.set()
#         self.updated.clear()

#     async def wait_for_updates(self, min_len : int = 1) -> list:
#         """
#         Waits for the contents of this buffer to be updated and to contain more updates than the given minimum
#         """
#         while True:
#             await self.updated.wait()

#             if len(self) >= min_len:
#                 break

#         return await self.pop_all()
