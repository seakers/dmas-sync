from collections import defaultdict
import logging

import numpy as np
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.planning.tasks import GenericObservationTask
from chess3d.agents.states import SimulationAgentState


class BidResultsTracker:
    """ 
    # Bid Results Tracker 

    Object used by agents to track and manage bid results for decentralized consensus planning agents.
    """
    def __init__(self, parent_name : str, optimistic_bidding_threshold : int = 1):
        """ 
        Initializes the `BidResultsTracker` with necessary parameters 
        
        #### Arguments:
        
            - `parent_name` (`str`) : Name of the parent agent using this tracker
            - `optimistic_bidding_threshold` (`int`) : Number of additional bids allowed for optimistic bidding strategy
        """
        # validate inputs
        assert isinstance(parent_name, str), "Parent name must be a string"
        assert isinstance(optimistic_bidding_threshold, int), "Optimistic bidding threshold must be an integer"
        assert optimistic_bidding_threshold >= 0, "Optimistic bidding threshold must be non-negative"

        # set parameters
        self.parent_name : str = parent_name

        # initialize results storage
        self.bids : dict[GenericObservationTask, list[Bid]] = dict()
        self.optimistic_bidding_counters : dict[GenericObservationTask, list[int]] = defaultdict(list)
   
    """
    ---------------------------
    CONSENSUS PHASE
    ---------------------------
    """
    def add_task_to_results(self, task : GenericObservationTask):
        """ Adds a bid to the results for a given task """
        # validate task not already in results
        assert task not in self.bids, "Cannot add task to results. Already exists."

        # initialize empty bid list for task
        self.bids[task] = []

    def compare_bids(self, bid : Bid, t : float) -> None:
        """ Compares an incoming bid with the existing bid for the same task and observation number """
        # validate task already in results
        assert bid.task in self.bids, "Cannot compare bids. Task not found in results."

        # validate bid for observation number exists
        assert bid.n_obs <= len(self.bids[bid.task]), "Cannot compare bids. Bid for observation number not found in results."

        if bid.n_obs == len(self.bids[bid.task]):
            # if bid for new observation number, append to results
            self.bids[bid.task].append(Bid(bid.task, bid.n_obs))

        # get current bid
        current_bid : Bid = self.bids[bid.task][bid.n_obs]

        # compare bids
        updated_bid : Bid = current_bid.update(bid, t)

        # update results with modified bid
        self.bids[bid.task][bid.n_obs] = updated_bid
        

    """
    ---------------------------
    BUNDLE-BUILDING PHASE
    ---------------------------
    """
    def set_bid_for_task(self, bid : Bid):
        """ Sets the bid for a given task in the results """
        # validate task already in results
        assert bid.task in self.bids, "Cannot set bid for task. Task not found in results."

        # update bid in results
        if bid.n_obs < len(self.bids[bid.task]):
            self.bids[bid.task][bid.n_obs] = bid
        elif bid.n_obs == len(self.bids[bid.task]):
            self.bids[bid.task].append(bid)
        else:
            raise IndexError("Cannot set bid for task. `n_obs` index out of range.")

    def can_bid(self, *args) -> bool:
        """ Checks if the agent can place a new bid for the given task based on optimistic bidding strategy """
        raise NotImplementedError("`can_bid` method is not implemented yet.")
    
        # check temporal constraints

        # if failed all constraints, check optimistic bidding counters
        return self.optimistic_bidding_counters[task] > 0 
    
    """
    LOGGING
    """
    def log_results(self, dsc : str, state : SimulationAgentState, level=logging.DEBUG) -> None:
        out = f'\nT{np.round(state.t,3)}[s]:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t n_obs\tins\twinner\tbid\tt_img\tt_stamp  performed\n'
        
        # count characters in line for formatting
        L_LINE = len(line)

        # header
        out += line 

        # divider 
        for _ in range(L_LINE + 25): out += '='
        out += '\n'

        n = 15
        i = 1
        for task, bids in self.bids.items():
            task : GenericObservationTask
            req_id_short = task.id.split('-')[-1]

            # if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            for bid in bids:
                # if i > n: break

                bid : Bid
                # if bid.winner == bid.NONE: continue

                if bid.winner != bid.NONE:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\t{bid.winner[0].lower()}{bid.winner[-1]}\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                else:
                    line = f'{req_id_short} {bid.n_obs}\t{bid.main_measurement}\tn/a\t{np.round(bid.winning_bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_stamp,1)}\t  {(bid.performed)}\n'
                out += line
                i +=1

            for _ in range(L_LINE + 25):
                out += '-'
            out += '\n'

            if i > n:
                out += '\t\t\t...\n'
                for _ in range(L_LINE + 25):
                    out += '-'
                out += '\n'
                break

        print(out)