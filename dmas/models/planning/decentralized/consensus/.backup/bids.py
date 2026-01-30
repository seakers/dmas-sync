from abc import ABC
import asyncio
from enum import Enum
from typing import Any, Union

import numpy as np

from chess3d.agents.science.requests import TaskRequest

class BidComparisonResults(Enum):
    UPDATE_TIME = 'update_time'
    UPDATE = 'update'
    LEAVE = 'leave'
    RESET = 'reset'
    COMPLETED = 'completed'

class RebroadcastComparisonResults(Enum):
    REBROADCAST_SELF = 'rebroadcast_self'
    REBROADCAST_OTHER = 'rebroadcast_other'
    REBROADCAST_EMPTY = 'rebroadcast_empty'
    NO_REBROADCAST = 'no_rebroadcast'

class Bid(ABC):
    """
    ## Measurement Request Bid for Consensus Planners

    Describes a bid placed on a task by a given agent

    ### Attributes:
        - req (`dict`): measurement request being bid on
        - subtask_index (`int`) : index of the subtask to be bid on
        - main_measurement (`str`): name of the main measurement assigned by this subtask bid
        - bidder (`bidder`): name of the agent keeping track of this bid information
        - own_bid (`float` or `int`): latest bid from bidder
        - winner (`str`): name of current the winning agent
        - winning_bid (`float` or `int`): current winning bid
        - t_img (`float` or `int`): time where the task is set to be performed by the winning agent
        - t_update (`float` or `int`): latest time when this bid was updated
        - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
    """

    NONE = 'none'
    
    def __init__(   self, 
                    req_id : str,
                    main_measurement : str,
                    bidder: str, 
                    bid: Union[float, int] = 0, 
                    winner: str = NONE, 
                    t_img: Union[float, int] = -1, 
                    th_img : Union[float, int] = np.NAN,
                    t_update: Union[float, int] = -1, 
                    performed : bool = False,
                    ) -> object:
        """
        Creates an instance of a task bid

        ### Arguments:
            - req_id (`dict`): id of the measurement request being bid on
            - main_measurement (`str`): name of the main measurement assigned by this subtask bid
            - bidder (`bidder`): name of the agent keeping track of this bid information
            - bid (`float` or `int`): latest bid from bidder
            - winner (`str`): name of the current winning agent
            - t_img (`float` or `int`): time where the task is set to be performed by the winning agent
            - t_update (`float` or `int`): latest time when this bid was updated
            - dt_converge (`float` or `int`): time interval after which local convergence is assumed to have been reached
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
        """
        # constants
        self.req_id = req_id
        self.main_measurement = main_measurement

        # variables
        self.bidder = bidder
        self.bid = bid
        self.winner = winner
        self.t_img = t_img
        self.th_img = th_img
        self.t_update = t_update
        self.performed = performed

    """
    ------------------
    COMPARISON METHODS
    ------------------
    """

    def __lt__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                raise AttributeError(f'cannot compare bids intended for different tasks (expected task id: {self.req_id}, given id: {other.task_id})')
            
            if other.bid == self.bid:
                # if there's a tie, use tie-breaker
                return self != self._tie_breaker(self, other)

            return other.bid > self.bid
        raise ValueError(f'can only compare bids to other bids.')

    def __gt__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                raise AttributeError(f'cannot compare bids intended for different tasks (expected task id: {self.req_id}, given id: {other.task_id})')
            
            if other.bid == self.bid:
                # if there's a tie, use tie-breaker
                return self == self._tie_breaker(self, other)

            return other.bid < self.bid
        raise ValueError(f'can only compare bids to other bids.')

    def __le__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                raise AttributeError(f'cannot compare bids intended for different tasks (expected task id: {self.req_id}, given id: {other.task_id})')
            
            if abs(other.bid - self.bid) < 1e-3:
                return True

            return other.bid >= self.bid
        raise ValueError(f'can only compare bids to other bids.')

    def __ge__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                raise AttributeError(f'cannot compare bids intended for different tasks (expected task id: {self.req_id}, given id: {other.task_id})')
            
            if abs(other.bid - self.bid) < 1e-3:
                return True

            return other.bid <= self.bid
        raise ValueError(f'can only compare bids to other bids.')

    def __eq__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                return False
            
            return abs(other.bid - self.bid) < 1e-3 and other.winner == self.winner
        raise ValueError(f'can only compare bids to other bids.')

    def __ne__(self, other : object) -> bool:
        if isinstance(other, Bid):
            if self.req_id != other.req_id:
                # if update is for a different task, ignore update
                return True
            
            return abs(other.bid - self.bid) > 1e-3 or other.winner != self.winner
        raise ValueError(f'can only compare bids to other bids.')

    def __update_time(self, t_update : float) -> None:
        """Records the lastest time this bid was updated"""
        self.t_update = t_update

    def compare(self, other : Any) -> object:
        """
        Compares bid with another and either updates, resets, or leaves the information contained in this bid
        depending on the rules specified in:
            - Luke B. Johnson, Sameera S. Ponda, Han-Lim Choi, Jonathan P. How "Asynchronous Decentralized Task Allocation for Dynamic Environments".

        ### Arguments:
            - other_dict (`dict`): dictionary representing the bid being compared to

        ### Returns:
            - comparison (`BidComparisonResults`) : action to perform to this bid upon comparing this bid to the other bid
            - rebroadcast (`RebroadcastComparisonResults`): rebroadcast action to perform after comparing this bid to the other bid
        """
        if isinstance(other, dict):
            try:
                other : Bid = Bid.from_dict(other)
            except Exception as e:
                raise ValueError(f'Cannot compare bid. {e}')
        elif isinstance(other, Bid):
            pass
        else:
            raise ValueError(f'Cannot compare bid to an object of type `{type(other)}`')

        if self.req_id != other.req_id:
            # if update is for a different task, ignore update
            raise AttributeError(f'cannot update bid with information from another bid intended for another task (expected task id: {self.task_id}, given id: {other.task_id})')
        
        if other.performed and not self.performed:
            # update and rebroadcast
            return BidComparisonResults.UPDATE, True
        
        if other.bidder == self.bidder:
            if other.t_update > self.t_update:
                # update & rebroadcast other's bid
                return BidComparisonResults.UPDATE, \
                    RebroadcastComparisonResults.REBROADCAST_OTHER
            else:
                # leave & no-rebroadcast
                return BidComparisonResults.UPDATE, \
                    RebroadcastComparisonResults.NO_REBROADCAST
        
        # Total of 31 cases
        elif other.winner == other.bidder:
            if self.winner == self.bidder:
                if other.bid > self.bid:
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.bid == self.bid and self != self._tie_breaker(other, self):
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE, \
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.bid < self.bid:
                    # update time & rebroadcast own bid
                    return BidComparisonResults.UPDATE_TIME,\
                        RebroadcastComparisonResults.REBROADCAST_SELF

            elif self.winner == other.bidder:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.NO_REBROADCAST
                
                elif other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.t_update < self.t_update:
                    # leave & no-rebroadcast
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.NO_REBROADCAST
                
            elif self.winner not in [other.bidder, self.bidder, self.NONE]:
                if other.bid > self.bid:
                    if other.t_update >= self.t_update:
                        # update & rebroadcast other's bid
                        return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                    else:
                        # update & rebroadcast other's bid
                        return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.bid == self.bid:
                    # leave & rebroadcast own information
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.REBROADCAST_SELF
                
                elif other.bid < self.bid:
                    if other.t_update <= self.t_update:
                        # leave & rebroadcast own information
                        return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.REBROADCAST_SELF
                    else:
                        # update & rebroadcast other's bid
                        return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER

            elif self.winner is self.NONE:
                # update & rebroadcast other's bid
                return BidComparisonResults.UPDATE,\
                    RebroadcastComparisonResults.REBROADCAST_OTHER

        elif other.winner == self.bidder:
            if self.winner == self.bidder:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.NO_REBROADCAST

            elif self.winner == other.bidder:
                # reset & rebroadcast empty bid with current time
                return BidComparisonResults.RESET,\
                    RebroadcastComparisonResults.REBROADCAST_EMPTY

            elif self.winner not in [other.bidder, self.bidder, self.NONE]:
                # leave & rebroadcast own information
                return BidComparisonResults.LEAVE,\
                    RebroadcastComparisonResults.REBROADCAST_SELF

            elif self.winner is self.NONE:
                # leave & rebroadcast emtpy bid with current time
                return BidComparisonResults.LEAVE,\
                    RebroadcastComparisonResults.REBROADCAST_EMPTY

        elif other.winner not in [other.bidder, self.bidder, self.NONE]:
            if self.winner == self.bidder:
                if other.bid > self.bid:
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.bid == self.bid and self != self._tie_breaker(other, self): 
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.bid < self.bid:
                    # update time & rebroadcast own bid
                    return BidComparisonResults.UPDATE_TIME,\
                        RebroadcastComparisonResults.REBROADCAST_SELF
                    
            elif self.winner == other.bidder:
                # update & rebroadcast other's bid
                return BidComparisonResults.UPDATE,\
                    RebroadcastComparisonResults.REBROADCAST_OTHER
            
            elif self.winner == other.winner:
                if abs(other.t_update - self.t_update) < 1e-3:
                    # leave & no-rebroadcast
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.NO_REBROADCAST
                
                elif other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
                elif other.t_update < self.t_update:
                    # leave & no-rebroadcast
                    return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.NO_REBROADCAST
                
            elif self.winner not in [other.bidder, self.bidder, other.winner, self.NONE]:
                if other.bid > self.bid:
                    if other.t_update >= self.t_update:
                        # update & rebroadcast other's bid
                        return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER

                    elif other.t_update < self.t_update:
                        # leave & rebroadcast own bid
                        return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.REBROADCAST_SELF

                elif other.bid < self.bid:
                    if other.t_update <= self.t_update:
                        # leave & rebroadcast own bid
                        return BidComparisonResults.LEAVE,\
                        RebroadcastComparisonResults.REBROADCAST_SELF
                    
                    elif other.t_update > self.t_update:
                        # update & rebroadcast other's bid
                        return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER
                
            elif self.winner is self.NONE:
                # update & rebroadcast other's bid
                return BidComparisonResults.UPDATE,\
                    RebroadcastComparisonResults.REBROADCAST_OTHER

        elif other.winner is self.NONE:
            if self.winner == self.bidder:
                # leave & rebroadcast own bid
                return BidComparisonResults.LEAVE,\
                    RebroadcastComparisonResults.REBROADCAST_SELF

            elif self.winner == other.bidder:
                # update & rebroadcast other's bid
                return BidComparisonResults.UPDATE,\
                    RebroadcastComparisonResults.REBROADCAST_OTHER

            elif self.winner not in [other.bidder, self.bidder, self.NONE]:
                if other.t_update > self.t_update:
                    # update & rebroadcast other's bid
                    return BidComparisonResults.UPDATE,\
                        RebroadcastComparisonResults.REBROADCAST_OTHER

            elif self.winner is self.NONE:
                # leave & no-rebroadcast
                return BidComparisonResults.LEAVE,\
                    RebroadcastComparisonResults.NO_REBROADCAST
            
        return BidComparisonResults.LEAVE,\
            RebroadcastComparisonResults.NO_REBROADCAST

    def update(self, other : object, t : float) -> bool :
        """ updates the value of this bid according to the results of comparing """
        # compare bids
        comp_result, _ = self.compare(other)
        comp_result : BidComparisonResults

        # update accordingly 
        new_bid : Bid = self.copy()
        if comp_result is BidComparisonResults.UPDATE_TIME:
            new_bid.__update_time(t)
        elif comp_result is BidComparisonResults.UPDATE:
            new_bid._update_info(other, t)
        elif comp_result is BidComparisonResults.RESET:
            new_bid._reset(t)
        elif comp_result is BidComparisonResults.LEAVE:
            new_bid._leave(t)
        elif comp_result is BidComparisonResults.COMPLETED:
            new_bid._perform(t)
        else:
            raise ValueError(f'cannot perform update of type `{comp_result}`')
        
        return new_bid
        
    def _update_info(self,
                    other, 
                    t : float
                ) -> None:
        """
        Updates all of the variable bid information

        ### Arguments:
            - other (`Bid`): equivalent bid being used to update information
        """
        if self.req_id != other.req_id:
            # if update is for a different task, ignore update
            raise AttributeError(f'cannot update bid with information from another bid intended for another task (expected task id: {self.req_id}, given id: {other.task_id}).')

        if isinstance(other, Bid):
            self.bid = other.bid
            self.winner = other.winner
            self.t_img = other.t_img
            self.th_img = other.th_img

            self.t_update = t
            self.performed = other.performed if not self.performed else True # Check if this hold true for all values

            assert self.t_img == other.t_img
        else:
            raise ValueError(f'can only update bids from other bids.')

    def _reset(self, t_update) -> None:
        """
        Resets the values of this bid while keeping track of lates update time
        """
        self.bid = 0
        self.winner = self.NONE
        self.t_img = -1
        self.th_img = np.NAN
        self.t_update = t_update

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
        self.t_update = t_update

    def _tie_breaker(self, bid1 : object, bid2 : object) -> object:
        """
        Tie-breaking criteria for determining which bid is GREATER in case winning bids are equal
        """
        if not isinstance(bid1, Bid) or not isinstance(bid2, Bid):
            raise ValueError(f'cannot calculate tie breaker. Both objects must be bids.')

        if bid2.winner == self.NONE and bid1.winner != self.NONE:
            return bid2
        elif bid2.winner != self.NONE and bid1.winner == self.NONE:
            return bid1
        elif bid2.winner == self.NONE and bid1.winner == self.NONE:
            return bid1

        elif bid1.bidder == bid2.bidder:
            return bid1
        elif bid1.bidder < bid2.bidder:
            return bid1
        else:
            return bid2

    def set(self, new_bid : Union[int, float], t_img : Union[int, float], th_img : Union[int, float], t_update : Union[int, float]) -> None:
        """
        Sets new values for this bid

        ### Arguments: 
            - new_bid (`int` or `float`): new bid value
            - t_img (`int` or `float`): new imaging time
            - t_update (`int` or `float`): update time
        """
        self.bid = new_bid
        self.winner = self.bidder
        self.t_img = t_img
        self.th_img = th_img
        self.t_update = t_update

    def __str__(self) -> str:
        """
        Returns a string representation of this task bid in the following format:
        - `task_id`, `main_measurement`, `target`, `bidder`, `bid`, `winner`, `t_img`, `t_update`
        """
        req : TaskRequest = TaskRequest.from_dict(self.req)
        split_id = req.id.split('-')
        line_data = [   split_id[0], 
                        self.main_measurement, 
                        req.target, 
                        self.bidder, 
                        round(self.bid, 3), 
                        self.winner, 
                        round(self.t_img, 3),
                        round(self.t_update, 3)
                    ]
        out = ""
        for i in range(len(line_data)):
            line_datum = line_data[i]
            out += str(line_datum)
            if i < len(line_data) - 1:
                out += ','

        return out
    
    def __repr__(self):
        task_id = self.req_id.split('-')
        return f'Bid_{task_id[0]}_{self.bidder}_{self.main_measurement}'

    def __hash__(self) -> int:
        return hash(repr(self))

    def has_winner(self) -> bool:
        """
        Checks if this bid has a winner
        """
        return self.winner != Bid.NONE

    def set_performed(self, t : float, performed : bool = True) -> None:
        self.__update_time(t)
        self.performed = performed
        self.t_img = t
        self.winner = self.bidder
    
    """
    ---------------------------
    COPY AND OTHER CONSTRUCTORS
    ---------------------------
    """
    def copy(self) -> object:
        """
        Returns a deep copy of this bid
        """
        return Bid.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this bid
        """
        return dict(self.__dict__)

    def from_dict(d : dict) -> object:
        """
        Creates a bid class object from a dictionary
        """
        return Bid(**d)
        

class BidBuffer(object):
    """
    Asynchronous buffer that holds bid information for use by processes within the MACCBBA
    """
    def __init__(self) -> None:
        self.bid_access_lock = asyncio.Lock()
        self.bid_buffer = {}
        self.updated = asyncio.Event()             

    def __len__(self) -> int:
        l = 0
        for req_id in self.bid_buffer:
            for bid in self.bid_buffer[req_id]:
                bid : Bid
                l += 1 if bid is not None else 0
        return l

    async def pop_all(self) -> list:
        """
        Returns latest bids for all requests and empties buffer
        """
        await self.bid_access_lock.acquire()

        out = []
        for req_id in self.bid_buffer:
            for bid in self.bid_buffer[req_id]:
                bid : Bid
                if bid is not None:
                    # place bid in outgoing list
                    out.append(bid)

            # reset bids in buffer
            self.bid_buffer[req_id] = [None for _ in self.bid_buffer[req_id]]

        self.bid_access_lock.release()

        return out

    async def put_bid(self, new_bid : Bid) -> None:
        """
        Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
        """
        await self.bid_access_lock.acquire()

        if new_bid.req_id not in self.bid_buffer:
            req : TaskRequest = TaskRequest.from_dict(new_bid.req)
            self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

        current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]
        
        if (    current_bid is None 
                or new_bid.bidder == current_bid.bidder
                or new_bid.t_update >= current_bid.t_update
            ):
            self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

        self.bid_access_lock.release()

        self.updated.set()
        self.updated.clear()

    async def put_bids(self, new_bids : list) -> None:
        """
        Adds bid to the appropriate buffer if it's a more updated bid information than the one at hand
        """
        if len(new_bids) == 0:
            return

        await self.bid_access_lock.acquire()

        for new_bid in new_bids:
            new_bid : Bid

            if new_bid.req_id not in self.bid_buffer:
                req : TaskRequest = TaskRequest.from_dict(new_bid.req)
                self.bid_buffer[new_bid.req_id] = [None for _ in req.dependency_matrix]

            current_bid : Bid = self.bid_buffer[new_bid.req_id][new_bid.subtask_index]

            if (    current_bid is None 
                 or (new_bid.bidder == current_bid.bidder and new_bid.t_update >= current_bid.t_update)
                 or (new_bid.bidder != new_bid.NONE and current_bid.winner == new_bid.NONE and new_bid.t_update >= current_bid.t_update)
                ):
                self.bid_buffer[new_bid.req_id][new_bid.subtask_index] = new_bid.copy()

        self.bid_access_lock.release()

        self.updated.set()
        self.updated.clear()

    async def wait_for_updates(self, min_len : int = 1) -> list:
        """
        Waits for the contents of this buffer to be updated and to contain more updates than the given minimum
        """
        while True:
            await self.updated.wait()

            if len(self) >= min_len:
                break

        return await self.pop_all()
