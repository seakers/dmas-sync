from logging import Logger
import logging
import numpy as np

from orbitpy.util import Spacecraft

from dmas.utils import runtime_tracker

from chess3d.agents.actions import ObservationAction
from chess3d.orbitdata import OrbitData
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.planning.decentralized.consensus.consensus import AbstractConsensusReplanner
from chess3d.agents.science.requests import TaskRequest
from chess3d.agents.states import SatelliteAgentState, SimulationAgentState


class ACBBAPlanner(AbstractConsensusReplanner):
    @runtime_tracker    
    def _generate_bids_from_request(self, 
                                    req: TaskRequest, 
                                    state: SimulationAgentState
                                    ) -> set:
        return [Bid(req.id, observation_type, state.agent_name) 
                for observation_type in req.observation_types]

    @runtime_tracker
    def _compare_bundles(self, bundle_1 : list, bundle_2 : list) -> bool:
        """ Compares two bundles. Returns true if they are equal and false if not. """
        if len(bundle_1) == len(bundle_2):
            for req, subtask, bid in bundle_1:
                if (req, subtask, bid) not in bundle_2:            
                    return False
            return True
        return False
            
    @runtime_tracker
    def log_results(self, dsc : str, state : SimulationAgentState, results : dict, 
                    level=logging.DEBUG) -> None:
        """
        Logs current results at a given time for debugging purposes

        ### Argumnents:
            - dsc (`str`): description of what is to be logged
            - results (`dict`): results to be logged
            - level (`int`): logging level to be used
        """
        out = f'T{state.t}:\t\'{state.agent_name}\'\n{dsc}\n'
        line = 'Req ID\t ins\twinner\tbid\tt_img\tt_update  performed\n'
        L_LINE = len(line)
        out += line 
        for _ in range(L_LINE + 25):
            out += '='
        out += '\n'
        
        n = 15
        i = 1
        for req_id in results:
            req_id : str
            req_id_short = req_id.split('-')[0]

            bids : dict[str,Bid] = results[req_id]
            # if all([bid.winner == bid.NONE for _,bid in bids.items()]): continue

            for _,bid in bids.items():
                # if i > n: break

                bid : Bid
                # if bid.winner == bid.NONE: continue

                if bid.winner != bid.NONE:
                    line = f'{req_id_short} {bid.main_measurement[0]}\t{bid.winner[0]}_{bid.winner[-1]}\t{np.round(bid.bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_update,1)}\t  {int(bid.performed)}\n'
                else:
                    line = f'{req_id_short} {bid.main_measurement[0]}\tn/a\t{np.round(bid.bid,3)}\t{np.round(bid.t_img,3)}\t{np.round(bid.t_update,1)}\t  {int(bid.performed)}\n'
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

    @runtime_tracker
    def _can_bid(self, 
                state : SimulationAgentState, 
                specs : object,
                results : dict,
                req : TaskRequest, 
                main_measurement : str
                ) -> bool:
        """ Checks if an agent has the ability to bid on a measurement task """
        if isinstance(specs, dict):
            payload = specs['instrument']
        elif isinstance(specs, Spacecraft):
            payload = specs.instrument
        else:
            raise ValueError('`specs` type not supported.')

        # check capabilities - TODO: Replace with knowledge graph
        bid : Bid = results[req.id][main_measurement]
        if bid.main_measurement not in [instrument.name for instrument in payload]:
            return False 

        # check time constraints
        ## Constraint 1: task must be able to be performed during or after the current time
        if req.t_end < state.t:
            return False
        
        return True

    @runtime_tracker
    def calc_imaging_time(self, 
                          state : SimulationAgentState, 
                          specs : object, 
                          path : list, 
                          req : TaskRequest, 
                          main_measurement : str, 
                          orbitdata : OrbitData) -> tuple:
        """
        Computes the ideal time when a task in the path would be performed
        
        ### Returns
            - t_img (`float`): earliest valid imaging time
            - th_img (`float`): look angle [deg] at the earliest valid imaging time
        """
        proposed_path = [path_element for path_element in path]
        proposed_path_tasks = [(path_req, path_j) for path_req, path_j,*_ in path]
        j = proposed_path_tasks.index((req,main_measurement))  
        
        # get access times for all available measurements
        if j > 0:
            _,_,t_prev,_,_ = path[j-1]
            t_start = max(t_prev, req.t_start, state.t)

        else:
            t_start = max(state.t, req.t_start)

        accesses = [(t_img*orbitdata.time_step, th_img) 
                        for t_img,_,_,lat,lon,_,th_img,_,_,_,instrument,_ in orbitdata.gp_access_data.values
                        if  abs(req.target[0] - lat) <= 1e-3
                        and abs(req.target[1] - lon) <= 1e-3
                        and t_start <= t_img*orbitdata.time_step <= req.t_end
                        and instrument == main_measurement]
        accesses.sort()
        
        # find earliest time that is allowed
        while accesses:
            earlies_access = accesses.pop(0)
            t_img,th_img = earlies_access
    
            # temporarily update access time and look angle
            proposed_path[j] = (req, main_measurement, t_img, th_img, -1)

            # check if path is valid for the agent
            if self.is_task_path_valid(state, specs, proposed_path, orbitdata):
                # earliest valid path found
                return t_img, th_img
        
        # no valid path was found
        return -1, np.NAN

    @runtime_tracker
    def is_task_path_valid(self, state : SimulationAgentState, specs : object, path : list, orbitdata : OrbitData) -> bool:
        if isinstance(state, SatelliteAgentState):
            # check if no suitable time was found for this observation
            if any([t_img < 0.0 for _,_,t_img,_,_ in path]): return False

            # gather list of observation actions
            observations = [ObservationAction(main_measurement, req.target, th_img, t_img)
                            for req, main_measurement, t_img, th_img, _ in path
                            if isinstance(req,TaskRequest)]

            return self.is_observation_path_valid(state, specs, observations)
        
        else:
            raise NotImplementedError(f'Check for path validity for agents of type {type(state)} not yet supported.')
