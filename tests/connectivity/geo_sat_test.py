
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

        # compile ground stations
        d['groundStation'] = self.compile_ground_stations()

        # configure ground operator 
        d['groundOperator'] = self.define_ground_operators()

        # add spacecraft to mission specifications
        d['spacecraft'] = [sat1, sat2]
        
        # return mission specifications
        return d
    
    """
    ============================================
        TEST CASES
    ============================================
    """

    def test_full_connectivity(self):
        # return # TODO: re-enable when connectivity case is fixed
    
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
        # return # TODO: re-enable when connectivity case is fixed
    
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
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names:
                    # receiver is ground station, sender must be a satellite; set appropriate reference access times 
                    if "1" in sender_name:
                        t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                    
                    elif "2" in sender_name:
                        t_refs = [(0 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)]

                    else:
                        raise ValueError(f'Unknown sender name: {sender_name}')
                else:
                    # receiver is a satellite; check if sender is ground station
                    if sender_name in ground_operator_names:
                        # sender is a ground station; set appropriate reference access times 
                        if "1" in receiver_name:                            
                            t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                        
                        elif "2" in receiver_name:
                            t_refs = [(0 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)]

                        else:
                            raise ValueError(f'Unknown sender name: {sender_name}')
                    else:
                        # both sender and receiver are satellites; set reference access times 
                        t_refs = [
                            (0 * sender_orbitdata.time_step, 1510 * sender_orbitdata.time_step),
                            (3823 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)
                        ]                 

                # debug prints
                # print(f"Checking link {sender_name} -> {receiver_name}")
                # print(f"  Reference access intervals:")
                # for t_ref_start,t_ref_end in t_refs: print(f"    ({t_ref_start}, {t_ref_end})")
                # print(f"  Loaded access intervals:")
                # for t_start,t_end,*_ in interval_data.data: print(f"    ({t_start}, {t_end})")

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
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # eitherreceiver is ground station, sender must be a satellite; no accesses should be available
                    t_refs = []
                    
                else:
                    # both sender and receiver are satellites; set reference access times 
                    t_refs = [
                            (0 * sender_orbitdata.time_step, 1510 * sender_orbitdata.time_step),
                            (3823 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)
                        ]                 

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
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # either receiver is ground station, sender must be a satellite; set appropriate reference access times 
                    if "1" in receiver_name or "1" in sender_name:                            
                        t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                    
                    elif "2" in receiver_name or "2" in sender_name:
                        t_refs = [(0 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)]

                    else:
                        raise ValueError(f'Unknown sender name: {sender_name}')
                    
                else:
                    # both sender and receiver are satellites; no accesses should be available
                    t_refs = []                 

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