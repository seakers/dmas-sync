
from collections import defaultdict
import copy
import unittest

from tqdm import tqdm
from dmas.utils.orbitdata import ConnectivityLevels
from dmas.utils.tools import print_scenario_banner
from tests.connectivity.connectivity_tester import AgentConnectivityTester


class TestGEOSatAgentConnectivity(AgentConnectivityTester, unittest.TestCase):
    """
    ============================================
        MISSION BUILDING UTILITIES
    ============================================
    """
    def setUp(self):
        super().setUp()

        # define ISL access times
        self.isl_access_times = {
            ("sat_1", "sat_2"): [
                (0, 1510),
                (3823, 6376)
            ],
            ("sat_1", "sat_3"): [
                (64, 3413),
                (5667, 6376)
            ],
            ("sat_1", "sat_4"): [
                (1897, 5240)
            ],
            ("sat_2", "sat_3"): [
                (0, 6376)
            ],
            ("sat_2", "sat_4"): [
                (0, 6376)
            ],
            ("sat_3", "sat_4"): [
                (0, 6376)
            ]
        }

        self.gs_access_times = {
            ("gs", "sat_1"): [
                (5168, 5513)
            ],
            ("gs", "sat_2"): [
                (0, 6376)
            ],
            ("gs", "sat_3"): [],
            ("gs", "sat_4"): []
        }

    def build_mission(self, connectivity : str) -> dict:
        # copy mission template
        d = copy.deepcopy(self.mission_template)

        # set connectivity
        d['scenario']['connectivity'] = connectivity
        
        # define sat 1
        sat1 = copy.deepcopy(self.spacecraft_template)
        sat1['name'] = 'sat_1'
        sat1['@id'] = 'sat_1'
        sat1['orbitState']['state']['inc'] = 0.0
        sat1['orbitState']['state']['ta'] = 10.0

        # define sat 2: GEO relay satelliteg
        sat2 = copy.deepcopy(self.spacecraft_template)
        sat2['name'] = 'sat_2'
        sat2['@id'] = 'sat_2'
        sat2['orbitState']['state']['sma'] = 42164.2
        sat2['orbitState']['state']['inc'] = 0.0
        sat2['orbitState']['state']['raan'] = 0.0
        sat2['orbitState']['state']['ta'] = 0.0
        sat2.pop('instrument') 

        # define sat 3: GEO relay satelliteg
        sat3 = copy.deepcopy(self.spacecraft_template)
        sat3['name'] = 'sat_3'
        sat3['@id'] = 'sat_3'
        sat3['orbitState']['state']['sma'] = 42164.2
        sat3['orbitState']['state']['inc'] = 0.0
        sat3['orbitState']['state']['raan'] = 0.0
        sat3['orbitState']['state']['ta'] = 120.0
        sat3.pop('instrument') 

        # define sat 4: GEO relay satelliteg
        sat4 = copy.deepcopy(self.spacecraft_template)
        sat4['name'] = 'sat_4'
        sat4['@id'] = 'sat_4'
        sat4['orbitState']['state']['sma'] = 42164.2
        sat4['orbitState']['state']['inc'] = 0.0
        sat4['orbitState']['state']['raan'] = 0.0
        sat4['orbitState']['state']['ta'] = 240.0
        sat4.pop('instrument') 

        # compile ground stations
        d['groundStation'] = self.compile_ground_stations()

        # configure ground operator 
        d['groundOperator'] = self.define_ground_operators()

        # add spacecraft to mission specifications
        d['spacecraft'] = [sat1, sat2, sat3, sat4]
        
        # return mission specifications
        return d
    
    """
    ============================================
        TEST CASES
    ============================================
    """

    def test_full_connectivity(self):
        # load orbit data
        *_, orbit_data = self.generate_orbit_data(ConnectivityLevels.FULL.value, False)

        # get mission duration 
        mission_duration = self.mission_template['duration'] * 24.0 * 3600 # in seconds

        # check connectivity for each agent
        for _,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying Full Connectivity Case', leave=True):
            agent_accesses = sender_orbitdata.get_next_agent_accesses(0.0)
            access_map = defaultdict(list)
            for interval,target in agent_accesses:
                access_map[target].append(interval)

            for target,intervals in access_map.items():
                # ensure only a single access interval exists between sender and receiver
                self.assertTrue(len(intervals) == 1)

                # ensure access interval spans entire mission duration
                t_start,t_end = intervals[0].left, intervals[0].right
                self.assertTrue(abs(t_start - 0.0) <= sender_orbitdata.time_step)
                self.assertTrue(abs(t_end - mission_duration) <= sender_orbitdata.time_step)
        
    def test_los_connectivity(self):    
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.LOS.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying LOS Connectivity Case', leave=True):
            agent_accesses = sender_orbitdata.get_next_agent_accesses(0.0)
            access_map = defaultdict(list)
            for interval,receiver_name in agent_accesses:
                access_map[receiver_name].append(interval)
            
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(access_map.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # order sender and receiver names to ensure consistent reference access times
                name_tuples = tuple(sorted([sender_name, receiver_name]))
                
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # receiver or sender is ground station, sender must be a satellite; set appropriate reference access times 
                    t_idx_refs = self.gs_access_times[name_tuples]
                
                else:
                    # both sender and receiver are satellites; set reference access times 
                    t_idx_refs = self.isl_access_times[name_tuples]
            
                # convert reference access times to seconds
                t_refs = [(t_start * sender_orbitdata.time_step, t_end * sender_orbitdata.time_step) for t_start,t_end in t_idx_refs]

                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),interval in zip(t_refs, interval_data):
                    self.assertTrue(abs(interval.left - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(interval.right - t_rev_end) <= sender_orbitdata.time_step)
        
        
    def test_isl_connectivity(self):    
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.ISL.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying ISL Connectivity Case', leave=True):
            agent_accesses = sender_orbitdata.get_next_agent_accesses(0.0)
            access_map = defaultdict(list)
            for interval,receiver_name in agent_accesses:
                access_map[receiver_name].append(interval)
            
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(access_map.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # order sender and receiver names to ensure consistent reference access times
                name_tuples = tuple(sorted([sender_name, receiver_name]))
                
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # receiver or sender is ground station, sender must be a satellite; set appropriate reference access times 
                    t_idx_refs = []
                
                else:
                    # both sender and receiver are satellites; set reference access times 
                    t_idx_refs = self.isl_access_times[name_tuples]
            
                # convert reference access times to seconds
                t_refs = [(t_start * sender_orbitdata.time_step, t_end * sender_orbitdata.time_step) for t_start,t_end in t_idx_refs]
                                               
                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),interval in zip(t_refs, interval_data):
                    self.assertTrue(abs(interval.left - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(interval.right - t_rev_end) <= sender_orbitdata.time_step)

    def test_gs_connectivity(self):    
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.GS.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying GS Connectivity Case', leave=True):
            agent_accesses = sender_orbitdata.get_next_agent_accesses(0.0)
            access_map = defaultdict(list)
            for interval,receiver_name in agent_accesses:
                access_map[receiver_name].append(interval)
            
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(access_map.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # order sender and receiver names to ensure consistent reference access times
                name_tuples = tuple(sorted([sender_name, receiver_name]))
                
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # receiver or sender is ground station, sender must be a satellite; set appropriate reference access times 
                    t_idx_refs = self.gs_access_times[name_tuples]
                
                else:
                    # both sender and receiver are satellites; set reference access times 
                    t_idx_refs = []
            
                # convert reference access times to seconds
                t_refs = [(t_start * sender_orbitdata.time_step, t_end * sender_orbitdata.time_step) for t_start,t_end in t_idx_refs]

                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),interval in zip(t_refs, interval_data):
                    self.assertTrue(abs(interval.left - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(interval.right - t_rev_end) <= sender_orbitdata.time_step)

    def test_no_connectivity(self):  
        # load orbit data
        _, _, orbit_data = self.generate_orbit_data(ConnectivityLevels.NONE.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying No Connectivity Case', leave=True):
            agent_accesses = sender_orbitdata.get_next_agent_accesses(0.0)
            access_map = defaultdict(list)
            for interval,receiver_name in agent_accesses:
                access_map[receiver_name].append(interval)
            
            for receiver_name, interval_data in tqdm(access_map.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # ensure no access intervals exist between sender and receiver
                self.assertTrue(len(interval_data) == 0)

if __name__ == '__main__':
    # print banner
    print_scenario_banner("GEO Relay Satellite Test Suite")

    # run tests
    unittest.main()