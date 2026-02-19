from collections import defaultdict
import copy
import unittest

from tqdm import tqdm

from dmas.utils.tools import print_scenario_banner
from tests.connectivity.connectivity_tester import AgentConnectivityTester


class TestPredefAgentConnectivity(AgentConnectivityTester, unittest.TestCase):
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
                (1102,1843),
                (3726,4479),
                (6336,6376)
            ],
            ("sat_1", "sat_3"): [
            ],
            ("sat_1", "relay_sat_1"): [
                (0,1517),
                (3932,6376)
            ],
            ("sat_1", "relay_sat_2"): [
                (89,3459),
                (5804,6376)
            ],

            ("sat_2", "sat_3"): [
                (1529,2270),
                (4155,4909)
            ],
            ("sat_2", "relay_sat_1"): [
                (1438,4272),
                (6254,6376)
            ],
            ("sat_2", "relay_sat_2"): [
                (0,2658),
                (4668,6376)
            ],

            ("sat_3", "relay_sat_1"): [
                (0,2461),
                (4874,6376)
            ],
            ("sat_3", "relay_sat_2"): [
                (1025,4395)
            ],

            ("relay_sat_1", "relay_sat_2"): [
                (0, 6376)
            ]
        }

        self.gs_access_times = {
            ("gs", "sat_1"): [
                (5168, 5513)
            ],
            ("gs", "sat_2"): [
                (2824,3182)
            ],
            ("gs", "sat_3"): [
                (474,817),
                (6087,6376)
            ],
            ("gs", "relay_sat_1"): [
                (0,6376)
            ],
            ("gs", "relay_sat_2"): [

            ]
        }

    def build_mission(self, connectivity : str) -> dict:
        # copy mission template
        d = copy.deepcopy(self.mission_template)

        # define sat 1
        sat1 = copy.deepcopy(self.spacecraft_template)
        sat1['name'] = 'sat_1'
        sat1['@id'] = 'sat_1'
        sat1['orbitState']['state']['inc'] = 0.0
        sat1['orbitState']['state']['ta'] = 10.0

        # define sat 2
        sat2 = copy.deepcopy(self.spacecraft_template)
        sat2['name'] = 'sat_2'
        sat2['@id'] = 'sat_2'
        sat2['orbitState']['state']['inc'] = 180.0
        sat2['orbitState']['state']['ta'] = sat1['orbitState']['state']['ta'] - 60.0 # phase offset by 60.0[deg]

        # define sat 3
        sat3 = copy.deepcopy(self.spacecraft_template)
        sat3['name'] = 'sat_3'
        sat3['@id'] = 'sat_3'
        sat3['orbitState']['state']['inc'] = 0.0
        sat3['orbitState']['state']['ta'] = sat1['orbitState']['state']['ta'] - 60.0 # phase offset by 60.0[deg]

        # define relay sat 1
        r1 = copy.deepcopy(self.spacecraft_template)
        r1['name'] = 'relay_sat_1'
        r1['@id'] = 'r_1'
        r1['orbitState']['state']['sma'] = 35786.0 # geostationary orbit
        r1['orbitState']['state']['inc'] = 0.0
        r1['orbitState']['state']['ta'] = 0.0

        # define relay sat 2
        r2 = copy.deepcopy(self.spacecraft_template)
        r2['name'] = 'relay_sat_2'
        r2['@id'] = 'r_2'
        r2['orbitState']['state']['sma'] = 35786.0 # geostationary orbit
        r2['orbitState']['state']['inc'] = 0.0
        r2['orbitState']['state']['ta'] = 120.0

        # compile ground stations
        d['groundStation'] = self.compile_ground_stations()

        # configure ground operator 
        d['groundOperator'] = self.define_ground_operators()

        # add spacecraft to mission specifications
        d['spacecraft'] = [sat1, sat2, sat3, r1, r2]

        # set connectivity
        d['scenario']['connectivity'] = self.build_predifined_connectivity(connectivity)
        
        # return mission specifications
        return d
    
    def build_predifined_connectivity(self, connectivity : str) -> dict:        
        # initialize connectivity dict with type
        connectivity_dict = {
            "@type": "PREDEF",
            "rulesPath" : f"./tests/connectivity/resources/connectivity/{connectivity}.json",
            "relaysEnabled" : False
        }
        
        return connectivity_dict 

    """
    ============================================
        TEST CASES
    ============================================
    """
    def test_relay_connectivity(self):    
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data('relays', False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying Relays Connectivity Case', leave=True):
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
                    # both sender and receiver are satellites; check if either is a relay satellite
                    if 'relay' in sender_name or 'relay' in receiver_name:
                        # one of the agents is a relay satellite; set appropriate reference access times
                        t_idx_refs = self.isl_access_times[name_tuples]
                    else:
                        # neither agent is a relay satellite; comms should not be allowed
                        t_idx_refs = []
            
                # convert reference access times to seconds
                t_refs = [(t_start * sender_orbitdata.time_step, t_end * sender_orbitdata.time_step) for t_start,t_end in t_idx_refs]

                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),interval in zip(t_refs, interval_data):
                    self.assertTrue(abs(interval.left - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(interval.right - t_rev_end) <= sender_orbitdata.time_step)
        


if __name__ == '__main__':
    # print banner
    print_scenario_banner("Predefined Connectivity Test Suite")

    # run tests
    unittest.main()