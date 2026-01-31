from enum import Enum
from typing import Any, Callable, Dict, Union

import numpy as np

from execsatm.tasks import GenericObservationTask


REQUIRED_KEYS = ['task', 'n_obs', 
                 'owner', 'owner_bid', 
                 'winner', 'winning_bid', 't_img', 't_bid', 't_stamps', "main_measurement",
                 'performed']

def bid_comparison_input_checks( func : Callable ) -> Callable:
    """ Decorator to validate inputs for bid comparison methods """
    def checker(self, other : object, *args) -> Any:
        # validate inputs
        assert isinstance(self, Bid) and isinstance(other, (Bid,dict)), \
            f'can only compare bids to other bids or dictionary representation of bids.'
        
        if isinstance(other, dict):
            # ensure all required keys are present
            assert all(key in other for key in REQUIRED_KEYS), \
                f'Bid dictionary is missing required keys. Required keys: {REQUIRED_KEYS}'
            assert self.task.to_dict() == other['task'], \
                f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other["task"]["id"]})'
            assert self.n_obs == other['n_obs'], \
                f'cannot compare bids intended for different image numbers (expected image number: {self.n_obs}, given image number: {other["n_obs"]})'
        else:
            assert self.task == other.task, f'cannot compare bids intended for different tasks (expected task id: {self.task.id}, given id: {other.task.id})'
            assert self.n_obs == other.n_obs, f'cannot compare bids intended for different image numbers (expected image number: {self.n_obs}, given image number: {other.n_obs})'

        # perform comparison
        return func(self, other, *args)
        
    return checker

class BidComparisonResults(Enum):
    """ Bid comparison results """
    UPDATE = 'update'
    LEAVE = 'leave'
    RESET = 'reset'

class Bid: 
    """
    ## Bid for Consensus Planners

    Describes a bid placed on a generic observation task by a given agent
    """
    # Constants
    NONE = 'none'   # NONE value used in various bid attributes
    EPS = 1e-6      # small epsilon value for float comparisons

    def __init__(self,
                 task : GenericObservationTask,
                 owner: str,
                 n_obs: int = 0,
                 owner_bid: float = np.NaN,
                 winning_bid: float = 0,
                 winner: str = NONE,
                 t_img : float = np.NINF,
                 t_bid : float = np.NINF,
                 t_stamps: Dict[str,float] = None,
                 main_measurement : str = NONE,
                 performed : bool = False
                ):
        """
        ## Measurement Task Bid for Consensus Planners

        Creates an instance of a task bid

        ### Attributes:
            - task (`GenericObservationTask`): observation task being bid on
            - owner (`str`): name of the agent keeping track of this bid information
            - n_obs (`int`): task performance number being bid on (e.g., image observation number for the target task)
            - owner_bid (`float` or `int`): latest bid value from bid owner
            - winning_bid (`float` or `int`): current winning bid value
            - winner (`str`): name of current the winning agent
            - t_img (`float` or `int`): time where the task is set to be performed by the winning agent
            - t_bid (`float` or `int`): time where the bid was generated
            - t_stamps (`Dict[str,float]`): latest time-stamps when this bid was updated for each agent
            - main_measurement (`str`): name of the main measurement assigned by this subtask bid
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
        """
        # convert inputs if needed
        t_stamps = {owner: t_bid} if t_stamps is None else t_stamps

        # Validate inputs
        assert isinstance(task, GenericObservationTask), f'`task` must be of type `GenericObservationTask`, got `{type(task)}`'
        assert isinstance(owner, str), f'`bidder` must be of type `str`, got `{type(owner)}`'
        assert isinstance(n_obs, int) and n_obs >= 0, f'`n_img` must be positive `int`, got `{type(n_obs)}-{n_obs}`'
        assert isinstance(owner_bid, (float, int)), f'`owner_bid` must be of type `float` or `int`, got `{type(owner_bid)}`'
        assert isinstance(winning_bid, (float, int)), f'`winning_bid` must be of type `float` or `int`, got `{type(winning_bid)}`'
        assert isinstance(winner, str), f'`winner` must be of type `str`, got `{type(winner)}`'
        assert isinstance(t_img, (float, int)), f'`t_img` must be of type `float` or `int`, got `{type(t_img)}`'
        assert t_img in task.availability or t_img == np.NINF, f'`t_img` value `{t_img}` not in task availability interval `{task.availability}`'
        assert isinstance(t_bid, (float, int)), f'`t_bid` must be of type `float` or `int`, got `{type(t_bid)}`'
        assert t_bid >= 0.0 or t_bid == np.NINF, f'`t_bid` must be non-negative, got `{t_bid}`'
        assert t_bid <= t_img or t_img == np.NINF, \
            f'bid cannot be generated at a time `t_bid` `{t_bid}` later than the desired imaging time `t_img` `{t_img}`'
        assert isinstance(t_stamps, dict), f'`t_stamps` must be of type `dict`, got `{type(t_stamps)}`'
        assert all(isinstance(k, str) and isinstance(v, (float, int)) for k,v in t_stamps.items()), f'`t_stamps` keys must be of type `str` and values of type `float` or `int`'
        assert isinstance(main_measurement, str), f'`main_measurement` must be of type `str`, got `{type(main_measurement)}`'
        assert isinstance(performed, bool), f'`performed` must be of type `bool`, got `{type(performed)}`'

        # Assign task and observation attributes
        self.task = task
        self.n_obs = n_obs

        # Assign bid owner information
        self.owner = owner
        self.owner_bid = owner_bid
        
        # Assign winning bid information
        self.winner = winner
        self.winning_bid = winning_bid
        self.t_img = t_img
        self.t_bid = t_bid
        self.t_stamps = t_stamps
        self.main_measurement = main_measurement

        # Other attributes
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
        out = dict(self.__dict__)
        out['task'] = self.task.to_dict()   
        out['t_stamps'] = {key : val for key, val in self.t_stamps.items()}
        return out            

    @classmethod
    def from_dict(cls, bid_dict: dict) -> 'Bid':
        """
        Creates a bid class object from a dictionary
        """
        # check if all required keys are present
        required_keys = ['task', 'main_measurement', 'owner', 'owner_bid', 
                         'winner', 'winning_bid', 't_img', 'n_obs', 
                         't_bid','t_stamps', 'performed']
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
            n_obs=bid_dict['n_obs'],
            owner=bid_dict['owner'],
            owner_bid=bid_dict['owner_bid'],
            winning_bid=bid_dict['winning_bid'],
            winner=bid_dict['winner'],
            t_img=bid_dict['t_img'],
            t_bid=bid_dict['t_bid'],
            t_stamps=bid_dict['t_stamps'],
            main_measurement=bid_dict['main_measurement'],
            performed=bid_dict['performed']
        )
    
    def copy(self) -> 'Bid':
        """ Creates a deep copy of this bid object """
        return Bid.from_dict(self.to_dict())

    """
    ------------------
    BID VALUE COMPARISON METHODS
    ------------------

    Used to compare bids based on their bid values and winning bidders using standard comparison operators.

    """

    @bid_comparison_input_checks
    def __lt__(self, other : Union['Bid', dict]) -> bool:       
        # check if equal
        if isinstance(other, dict):
            if abs(other['winning_bid'] - self.winning_bid) < self.EPS:
                # if there's a tie, use tie-breaker        
                return not self.__wins_tie_breaker(other)

            # compare bids
            return other['winning_bid'] > self.winning_bid
        
        else:
            if abs(other.winning_bid - self.winning_bid) < self.EPS:
                # if there's a tie, use tie-breaker        
                return not self.__wins_tie_breaker(other)

            # compare bids
            return other.winning_bid > self.winning_bid

    @bid_comparison_input_checks
    def __gt__(self, other : Union['Bid', dict]) -> bool:
        if isinstance(other, dict):
            # check if equal
            if abs(other['winning_bid'] - self.winning_bid) < self.EPS:
                # if there's a tie, use tie-breaker        
                return self.__wins_tie_breaker(other)

            # compare bids
            return other['winning_bid'] < self.winning_bid
        else:
            # check if equal
            if abs(other.winning_bid - self.winning_bid) < self.EPS:
                # if there's a tie, use tie-breaker        
                return self.__wins_tie_breaker(other)

            # compare bids
            return other.winning_bid < self.winning_bid

    @bid_comparison_input_checks
    def __eq__(self, other : Union['Bid', dict]) -> bool:        
        # compare bids
        if isinstance(other, dict):
            return abs(other['winning_bid'] - self.winning_bid) < self.EPS    # same bid value
        else:
            return abs(other.winning_bid - self.winning_bid) < self.EPS    # same bid value

    @bid_comparison_input_checks
    def __ne__(self, other : Union['Bid', dict]) -> bool:        
        if isinstance(other, dict):
            # compare bids
            return abs(other['winning_bid'] - self.winning_bid) > self.EPS    # different bid value
        else:
            # compare bids
            return abs(other.winning_bid - self.winning_bid) > self.EPS    # different bid value

    @bid_comparison_input_checks
    def __le__(self, other : Union['Bid', dict]) -> bool:
        # compare bids
        if isinstance(other, dict):
            return other['winning_bid'] >= self.winning_bid or abs(other['winning_bid'] - self.winning_bid) < self.EPS
        else:
            return other.winning_bid >= self.winning_bid or abs(other.winning_bid - self.winning_bid) < self.EPS
    
    @bid_comparison_input_checks
    def __ge__(self, other : Union['Bid', dict]) -> bool:
        # compare bids
        if isinstance(other, dict):
            return other['winning_bid'] <= self.winning_bid or abs(other['winning_bid'] - self.winning_bid) < self.EPS
        else:
            return other.winning_bid <= self.winning_bid or abs(other.winning_bid - self.winning_bid) < self.EPS

    def __wins_tie_breaker(self, other: Union['Bid', dict]) -> bool:
        """ Returns True if, when bids are tied, we should prefer `self` over `other`. """
        return self is self.__tie_breaker(self, other)

    def __tie_breaker(self, bid1 : 'Bid', bid2 : Union['Bid', dict]) -> 'Bid':
        """
        Tie-breaking criteria for determining which bid is GREATER in case winning bids are equal. 
        Uses winning bidder names to determine winner, goes by alphabetical order of winning bidder names.
        """
        # validate inputs
        assert isinstance(bid1, Bid) and isinstance(bid2, (Bid, dict)), f'cannot calculate tie breaker. Both objects must be bids.'

        if isinstance(bid2, dict):
            # compare bids
            ## Check for NONE winning bidders
            if bid2['winner'] == self.NONE and bid1.winner != self.NONE:
                return bid2
            elif bid2['winner'] != self.NONE and bid1.winner == self.NONE:
                return bid1
            
            ## Compare bidders alphabetically
            name_pairs = {bid1.winner : bid1, bid2['winner'] : bid2}
            min_name = min(name_pairs.keys())
            return name_pairs[min_name]
        
        else:
            # compare bids
            ## Check for NONE winning bidders
            if bid2.winner == self.NONE and bid1.winner != self.NONE:
                return bid2
            elif bid2.winner != self.NONE and bid1.winner == self.NONE:
                return bid1

            ## Compare bidders alphabetically
            return min(bid1, bid2, key=lambda b: b.winner)

    @bid_comparison_input_checks
    def has_different_winner_values(self, other : Union['Bid', dict]) -> bool:
        """ Checks if this bid is different from another bid (i.e., any of the winning bid attributes differ) """
                        
        # Compare winning bid information
        if (
            self.winner != other.winner             # different winning bidder
            or abs(self.winning_bid - other.winning_bid) > self.EPS # different winning bid value
            or abs(self.t_img - other.t_img) > self.EPS             # different imaging time
            # or abs(self.t_bid - other.t_bid) > self.EPS             # different bid time
            # or self.t_stamps != other.t_stamps                      # different time stamps
            ):
            return True

        # # Compare other attributes
        # if self.performed != other.performed:                   # different performed status
        #     return True
        
        # Fallback → bids are the same
        return False

    """
    ------------------
    BID COMPARISON AND UPDATE METHODS
    ------------------

    Compares the values of a bid with another bid and decides to either update, reset, or leave the information contained in this bid
    along with deciding thether to broadcast the updated information to neighboring agents based on the rules specified in:
        - A. Aguilar, D. Fornos, and D. Selva, "Decentralized Task Planning and Scheduling for Multi-Agent Systems under Communication Constraints", IN PREP.

    """
    @bid_comparison_input_checks
    def rule_comparison(self, other : Union['Bid', dict]) -> BidComparisonResults:
        """ 
        Compares bid with another and indicates whether the bid should be updated, left, or reset.
        Also returns whether the bid should be rebroadcasted to neighboring agents.

        ### Arguments:
            - other (`Bid`): bid being compared to

        ### Returns: 
            - comparison (`str`): action to perform to this bid upon comparing this bid to the other bid
        """
        try:
            # convert other bid to `Bid` class type if necessary
            # other : Bid = Bid.from_dict(other) if isinstance(other, dict) else other

            # validate inputs
            assert isinstance(other, Bid), f'can only compare bids to other bids.'
            assert self.task == other.task, f'can only compare bids for the same task (expected task id: {self.task.id}, given id: {other.task.id})'
            
            # 0. Sending agent claims the bid has been performed.
            if other.was_performed():
                comp_result = self._case_other_thinks_bid_was_performed(other)
            # 0.5. Receiving agent has marked bid as performed.
            elif self.was_performed():
                # receiving agent has marked bid as performed → leave info as is
                comp_result = BidComparisonResults.LEAVE

            # 1. Sending agent claims itself as winner of this bid.
            elif other.believes_i_am_winning():
                comp_result = self._case_other_thinks_is_winner(other)

            # 2. Sending agent claims I am the winner of this bid.
            elif other.believes_other_is_winning(self):
                comp_result = self._case_other_thinks_im_winner(other)

            # 3. Sending agent claims some 3rd party as the winner of this bid.
            elif other.believes_third_party_is_winning(self):
                comp_result = self._case_other_thinks_third_party_winner(other)
            
            # 4. Sending agent has no winner for this bid.
            elif other.believes_no_winner():
                comp_result = self._case_other_has_no_winner(other)

            # 5. Fallback (should be unreachable)
            else: 
                raise ValueError(f'could not compare bids. Unknown case encountered between bids from bidder `{self.owner}` and bidder `{other.owner}`.')

            # return comparison result
            return comp_result
        
        except KeyError as e:
            raise e
        
        finally:
            # check output type
            assert isinstance(comp_result, BidComparisonResults), f'comparison result must be of type `BidComparisonResults`, got `{type(comp_result)}`'

    def _case_other_thinks_bid_was_performed(self, other : 'Bid') -> BidComparisonResults:
        """ Case: Sending agent claims the bid has been performed. """
        
        # 1. Receiving agent also believes the bid was performed.
        if self.was_performed():
            if self.winner == other.winner:
                # both agents agree bid was performed and winner → leave info as is
                return BidComparisonResults.LEAVE
            elif abs(self.t_img - other.t_img) <= 1e-6 and not self.__wins_tie_breaker(other):
                # both agents agree bid was performed and at what time but disagree on the winner;
                #  sender wins the tie-breaker → update
                return BidComparisonResults.UPDATE
            elif abs(self.t_img - other.t_img) <= 1e-6 and self.__wins_tie_breaker(other):
                # both agents agree bid was performed and at what time but disagree on the winner;
                #  receiver wins the tie-breaker → leave info as is
                return BidComparisonResults.LEAVE
            elif self.t_img > other.t_img:
                # both agents agree bid was performed but disagree on the winner and imaging time;
                #  sender has an earlier imaging time → update
                return BidComparisonResults.UPDATE
            elif self.t_img < other.t_img:
                # both agents agree bid was performed but disagree on the winner and imaging time;
                #  receiver has an earlier imaging time → leave info as is
                return BidComparisonResults.LEAVE
            # elif self.t_stamps[self.winner] >= other.t_stamps[other.winner]:
            #     # both agents agree bid was performed but disagree on the winner
            #     #  receiver has the more recent bid → leave info as is
            #     return BidComparisonResults.LEAVE
            else:
                # both agents agree bid was performed but disagree on the winner and imaging time;
                #  receiver has an earlier imaging time → leave info as is
                return BidComparisonResults.LEAVE
            
        # 0. Receiving agent does not believe the bid was performed.
        else:
            # receiving agent has not marked bid as performed → update info
            return BidComparisonResults.UPDATE

    def _case_other_thinks_is_winner(self, other : 'Bid') -> BidComparisonResults:
        """ Case: Sending agent claims itself as winner of this bid. """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if other > self:  
                # Sending agent's bid is higher → update info
                return BidComparisonResults.UPDATE
            elif other.t_img <= self.t_img:
                # Sending agent is bidding for an earlier observation → bidder must be optimistic in its bidding; update info
                return BidComparisonResults.UPDATE
            
        # 2. Receiving agent believes other is the winner already.
        if self.believes_other_is_winning(other):
            # Both agents agree the sending agent is the winner → update info
            return BidComparisonResults.UPDATE

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
            # if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner → update info
                return BidComparisonResults.UPDATE
            elif other > self:
                # Sending agent's bid is higher → update info
                return BidComparisonResults.UPDATE
            elif other.t_img <= self.t_img:
                # Sending agent is bidding for an earlier observation → bidder must be optimistic in its bidding; update info
                return BidComparisonResults.UPDATE

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # no known winner → update info
            return BidComparisonResults.UPDATE

        # 5. Fallback → leave info as is
        return BidComparisonResults.LEAVE

    def _case_other_thinks_im_winner(self, other : 'Bid') -> BidComparisonResults:
        """ Case: Sending agent claims I am the winner of this bid. """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            # Receiving agent has more updated information about itself → leave info as is
            return BidComparisonResults.LEAVE
        
        # 2. Receiving agent believes other is the winner already.
        if self.believes_other_is_winning(other):
            # Both agents think the other is winning and are in conflict → reset info
            return BidComparisonResults.RESET

        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
            # if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner → reset info
                return BidComparisonResults.RESET

        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # Receiving agent would know if it was winning a bid → leave info as is
            return BidComparisonResults.LEAVE

        # 5. Fallback → leave info as is
        return BidComparisonResults.LEAVE

    def _case_other_thinks_third_party_winner(self, other : 'Bid') -> BidComparisonResults:        
        """ Handles the case where the other bid thinks a third party is the winner """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            if other.t_stamps[other.winner] > self.t_stamps.get(other.winner, np.NINF):
            # if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner 
                if other > self:  
                    # Sending agent's bid is higher and more updated → update info
                    return BidComparisonResults.UPDATE
                elif other.t_img <= self.t_img:
                    # Sending agent is bidding for an earlier observation and is more updated → bidder must be optimistic in its bidding; update info
                    return BidComparisonResults.UPDATE
        
        # 2. Receiving agent believes other is the winner already.
        if self.believes_other_is_winning(other):
            if other.t_stamps[other.winner] > self.t_stamps.get(other.winner, np.NINF):
            # if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner → update info
                return BidComparisonResults.UPDATE
            else:
                # Receiving agent has more recent info on 3rd party winner → reset info and wait for sender to update
                return BidComparisonResults.RESET
        
        # 3. Receiving agent also believes some 3rd party is the winner.
        if self.has_same_winner(other):
            if other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
            # if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner → update info
                return BidComparisonResults.UPDATE
        
        # 4. Receiving agent believes some 4th party is the winner.
        if self.believes_third_party_is_winning(other):
            if other.t_stamps[other.winner] > self.t_stamps.get(other.winner, np.NINF):
                if other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
                    # Sending agent's bids from all parties are more updated → update info
                    return BidComparisonResults.UPDATE
                
                elif other > self:  
                    # Sending agent's bid is higher and more updated → update info
                    return BidComparisonResults.UPDATE
                
                elif other.t_img <= self.t_img:
                    # Sending agent is bidding for an earlier observation and is more updated → bidder must be optimistic in its bidding; update info
                    return BidComparisonResults.UPDATE

            elif other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
                # Sending agent has older info on 3rd party winner but newer info on receiving agent's winner → reset info
                return BidComparisonResults.RESET
        
        # 5. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # if other.t_bid > self.t_bid:
            #     # Sending agent has more recent info on 3rd party winner → update info
            #     return BidComparisonResults.UPDATE
            try:
                if other.t_stamps[other.winner] > self.t_stamps.get(other.winner, np.NINF):
                    # Sending agent has more recent info on 3rd party winner → update info
                    return BidComparisonResults.UPDATE
            except KeyError as e:
                x = 1
                raise e

        # 6. Fallback → leave info as is
        return BidComparisonResults.LEAVE

    def _case_other_has_no_winner(self, other : 'Bid') -> BidComparisonResults:
        """ Handles the case where the other bid has no winner """
        # 1. Receiving agent believes it is the winner too.
        if self.believes_i_am_winning():
            # Receiving agent would know if it was winning a bid → leave info as is
            return BidComparisonResults.LEAVE
        
        # 2. Receiving agent believes other is the winner already.
        if self.believes_other_is_winning(other):
            # Sending agent would know if it was winning a bid → update info
            return BidComparisonResults.UPDATE
        
        # 3. Receiving agent believes some 3rd party is winner.
        if self.believes_third_party_is_winning(other):
            # if other.t_stamps.get(self.winner, np.NINF) > self.t_stamps[self.winner]:
            if other.t_bid > self.t_bid:
                # Sending agent has more recent info on 3rd party winner → update info
                return BidComparisonResults.UPDATE
        
        # 4. Receiving agent bid has no winner.
        if self.believes_no_winner():
            # Neither bid has a winner → leave info as is
            return BidComparisonResults.LEAVE

        # 5. Fallback → leave info as is
        return BidComparisonResults.LEAVE

    def believes_i_am_winning(self) -> bool:
        """ Checks if this bid is currently won by the bidder itself """
        return self.winner == self.owner

    def believes_other_is_winning(self, other: 'Bid') -> bool:
        """ Checks if this bid is currently won by the other bidder """
        return self.winner == other.owner

    def believes_third_party_is_winning(self, other: 'Bid') -> bool:
        """ Checks if this bid is currenly won by a 3rd party (neither self nor other) """
        return self.winner not in {self.owner, other.owner, self.NONE}

    def believes_no_winner(self) -> bool:
        """ Checks if this bid has no winner """
        return not self.has_winner()
    
    def has_same_winner(self, other: 'Bid') -> bool:
        return self.winner == other.winner and self.has_winner() and other.has_winner()
    
    def has_winner(self) -> bool:    
        """ Checks if this bid has a winner """
        return self.winner != Bid.NONE

    def is_bidder_winning(self) -> bool:
        """ Checks if the bidder of this bid is the current winning bidder """
        return self.owner == self.winner

    def was_performed(self) -> bool:
        """ Checks if the winner of this bid has performed the measurement request at hand """
        return self.performed
            
    def is_mutually_exclusive(self, other : 'Bid') -> bool:
        """ Checks if this bid is mutually exclusive with another bid (i.e., both bids cannot be won by the same agent) """
        raise NotImplementedError("`is_mutually_exclusive` method is not implemented yet.")
        
        # check time overlap?
        if other.n_obs <= self.n_obs:
            return other.t_img <= self.t_img
        
    
    """
    ---------------------------
    MODIFIERS
    ---------------------------
    """
    def set(self, 
            main_measurement : str,
            bid_value : Union[int, float], 
            t_img : Union[int, float],
            t_update : Union[int, float]
        ) -> None:
        """
        Sets new values for this bid. Assumes this bid is winning upon setting new values.

        ### Arguments: 
            - main_instrument (`str`): main measurement set to perform this bid
            - bid_value (`int` or `float`): new bid value
            - t_img (`int` or `float`): new imaging time
            - t_update (`int` or `float`): update time
        """
        # validate inputs
        assert isinstance(bid_value, (float, int)), f'`bid_value` must be of type `float` or `int`, got `{type(bid_value)}`'
        assert bid_value > 0, f'`bid_value` must be positive, got `{bid_value}`'
        assert isinstance(t_img, (float, int)), f'`t_img` must be of type `float` or `int`, got `{type(t_img)}`'
        assert t_img in self.task.availability, f'`t_img` value `{t_img}` not in task availability interval `{self.task.availability}`'
        assert isinstance(t_update, (float, int)), f'`t_update` must be of type `float` or `int`, got `{type(t_update)}`'
        assert t_update >= 0, f'`t_update` must be non-negative, got `{t_update}`'
        assert t_update <= t_img, f'`t_update` time `{t_update}` cannot be later than the proposed imaging time `t_img` `{t_img}`'

        # update bidder information
        self.owner_bid = bid_value
        self.main_measurement = main_measurement

        # update winning bid information
        self.winning_bid = bid_value
        self.winner = self.owner
        self.t_img = t_img

        # update timestamp for this bidder
        self.t_stamps[self.winner] = t_update   

    def set_performed(self, t_update : float, performed : bool = True, performer : str = None) -> None:
        """
        Sets the performed status of this bid

        ### Arguments:
            - t (`float` or `int`): latest time when this bid was updated
            - performed (`bool`): indicates if the winner of this bid has performed the measurement request at hand
            - performer (`str`): name of the agent that performed the task
        """
        # update winning bidder information
        self.winner = performer if performer is not None else self.owner
        
        # update performed status
        self.performed = performed

        # update timestamp for this bidder
        self.t_stamps[self.winner] = t_update

    def update(self, other : 'Bid', t_comp : float) -> 'Bid':
        """ 
        Compares this bid with another and returns a new bid instance with the appropriate updated information.

        ### Arguments:
            - other (`Bid`): bid being compared to
            - t_comp (`float`): time in which the comparison is being made
        """
        # validate inputs
        assert isinstance(other, Bid), f'can only compare bids to other bids.'
        assert self.task == other.task, f'cannot compare bids intended for different tasks (expected {self.task}, got {other.task})'
        assert self.n_obs == other.n_obs, f'cannot compare bids intended for different image numbers (expected {self.n_obs}, got {other.n_obs})'
        assert t_comp >= 0, f'`t_comp` must be non-negative, got `{t_comp}`'
        
        # create a deep copy of this bid
        new_bid : Bid = self.copy()
        
        # check comparison rules
        comp_result : BidComparisonResults = self.rule_comparison(other)

        # update copy according to comparison result
        if comp_result is BidComparisonResults.UPDATE:      new_bid.__update_info(other, t_comp)
        elif comp_result is BidComparisonResults.RESET:     new_bid.reset(t_comp, other=other)
        elif comp_result is BidComparisonResults.LEAVE:     new_bid.__leave(other, t_comp)
        else: raise ValueError(f'cannot perform update of type `{comp_result}`')
        
        # check proper update
        assert abs(new_bid.t_stamps[other.owner] - t_comp) < 1e-6, \
            f'timestamp for bidder `{other.owner}` was not properly updated to `{t_comp}` [s]'
        assert new_bid.t_bid <= new_bid.t_img or new_bid.t_img == np.NINF, \
            f'bid cannot be generated at a time `t_bid` `{new_bid.t_bid}` later than the desired imaging time `t_img` `{new_bid.t_img}`'

        # return updated bid
        return new_bid
    
    def __update_info(self, other : 'Bid', t_comp : float) -> None:
        """
        Updates all of the variable bid information

        ### Arguments:
            - other (`Bid`): equivalent bid being used to update information
            - t_comp (`float` or `int`): latest time when this bid was updated
        """
        # check if other bid has valid values
        # assert other.winner != self.NONE, f'cannot update bid information with a bid that has no winner.'
        # assert other.winning_bid > 0, f'cannot update bid information with a bid that has non-positive winning bid value.'
        assert other.winner == self.NONE or other.t_img in other.task.availability, \
            f'`t_img` value `{other.t_img}` not in task availability interval `{other.task.availability}`'
        assert other.t_bid <= other.t_img or other.t_img == np.NINF, \
            f'bid cannot be generated at a time `t_bid` `{other.t_bid}` later than the desired imaging time `t_img` `{other.t_img}`'

        # update winning bid information
        self.winning_bid = other.winning_bid
        self.winner = other.winner
        
        # update other bid information
        self.t_img = other.t_img
        self.t_bid = other.t_bid
        self.main_measurement = other.main_measurement
        self.performed = other.performed or self.performed

        # check if bid came from the owner agent
        if other.owner == self.owner:
            # bid comes from the owner agent; carryover all timestamps
            for key,t_other in other.t_stamps.items():
                if key in [other.owner, other.winner]: 
                    continue

                # ensure all timestamps are the most recent ones
                self.t_stamps[key] = max(self.t_stamps.get(key, np.NINF), t_other)              

        # update timestamp for the other bidder
        self.t_stamps[other.owner] = max(self.t_stamps.get(other.owner, np.NINF), t_comp)

        # if the winner is different from the owner, update timestamp for the winner as well
        if other.owner != other.winner:
            self.t_stamps[other.winner] = max(self.t_stamps.get(other.winner, np.NINF), other.t_bid)

    def reset(self, t_comp : float, other : 'Bid' = None) -> None:
        """
        Resets the values of this bid while keeping track of lates update time
        
        ### Arguments:
            - t_comp (`float` or `int`): latest time when this bid was updated
            - other (`Bid`): equivalent bid being used to update information
        """
        # if own bid, update internal bid value to 0
        if self.winner == self.owner: self.owner_bid = 0

        # reset winning bid information
        self.winning_bid = 0
        self.winner = self.NONE

        # reset other bid information
        self.t_img = np.NINF
        self.t_bid = np.NINF
        self.main_measurement = self.NONE

        # update timestamp for the other bidder if given
        if other is not None:
            self.t_stamps[other.owner] = t_comp
        else:
            self.t_stamps[self.owner] = t_comp

    def __leave(self, other : 'Bid', t_comp : float) -> None:
        """
        Leaves bid as is.

        ### Arguments:
            - other (`Bid`): equivalent bid being used to update information
            - t_comp (`float` or `int`): latest time when this bid was updated
        """
        # update timestamp for the other bidder       
        self.t_stamps[other.owner] = t_comp

        # update timestamp for self bidder
        self.t_stamps[self.owner] = max(self.t_stamps.get(self.owner, np.NINF), t_comp)
    
    def set_performed(self, t_comp : float, performed : bool = True, performer : str = None) -> None:
        """ Indicates that this action has been performed """
        # validate inputs
        assert isinstance(t_comp, (float, int)), f'`t_comp` must be of type `float` or `int`, got `{type(t_comp)}`'
        assert t_comp >= 0, f'`t_comp` must be non-negative, got `{t_comp}`'
        assert isinstance(performed, bool), f'`performed` must be of type `bool`, got `{type(performed)}`'
        # assert self.is_bidder_winning(), f'only the winning bidder can set the performed status of this bid (current winning bidder: `{self.winning_bidder}`, current bidder: `{self.bidder}`)'

        # update performed status
        self.performed = performed

        # update timestamp for performer
        performer = self.owner if performer is None else performer
        self.t_stamps[performer] = t_comp

        # update timestamp for self bidder
        self.t_stamps[self.owner] = max(self.t_stamps.get(self.owner, np.NINF), t_comp)
    
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
                        "n_obs" : self.n_obs,
                        "bidder" : self.owner, 
                        "bid" : round(self.winning_bid, 3), 
                        "winner" : self.winner, 
                        "t_img" : round(self.t_img, 3),
                        "t_update" : round(max(self.t_stamps.values()), 3),
                        "main_measurement" : self.main_measurement, 
                        "target" : self.task.location, 
                    }
        out = "Bid("
        for key, value in line_data.items():
            out += f"{key}={value}, "
        out = out[:-2] + ")"

        return out
    
    def __repr__(self):
        task_id = self.task.id.split('-')
        return f'Bid({task_id[0]},n={self.n_obs},a={self.owner},w={self.winner},b={round(self.winning_bid,3)},t_img={round(self.t_img,3)})'
        # return f'Bid({task_id[0]},n={self.n_obs},a={self.bidder},w={self.winning_bidder},b={round(self.winning_bid,3)})'

    def __hash__(self) -> int:
        return hash(self.task.id) ^ hash(self.n_obs) ^ hash(self.owner)

