import copy
import os
from typing import Dict, Tuple
import unittest

import pandas as pd
from tqdm import tqdm

from chess3d.orbitdata import OrbitData, ConnectivityLevels
from chess3d.utils import print_banner


class TestAgentConnectivity(unittest.TestCase):
    """ Test scenarios where satellites can connect to ground stations but not to eachother. """
    
    # constants
    MISSION_NAME = 'sample_mission'
    GS_NETWORK_NAME = 'gs'

    def setUp(self):
        # Define a basic mission template with connectivity settings
        self.mission_template = {
                        "epoch": {
                            "@type": "GREGORIAN_UT1",
                            "year": 2020,
                            "month": 1,
                            "day": 1,
                            "hour": 0,
                            "minute": 0,
                            "second": 0
                        },
                        "duration": 2.0 / 24.0, # 2 hours in days
                        "propagator": {
                            "@type": "J2 ANALYTICAL PROPAGATOR",
                        },
                        "spacecraft": [],
                        "scenario" : {
                            "connectivity": "LOS",
                            "events": {
                                "@type": "PREDEF",
                                "eventsPath": f"./tests/planners/resources/events.csv"
                            },
                            "clock" : {
                                "@type" : "EVENT"
                            },
                            "scenarioPath" : "./tests/planners/",
                            "name" : "toy",
                            "missionsPath" : f"./tests/connectivity/resources/missions.json"
                        },
                        "settings": {
                            "coverageType": "GRID COVERAGE",
                            "outDir" : "./tests/connectivity/orbit_data"
                        },
                        "grid": [
                            {
                                "@type": "customGrid",
                                "covGridFilePath": "./tests/connectivity/resources/grid.csv"
                            }
                        ],
                    }
        
        self.spacecraft_template = {
                            "@id": "sat_temp",
                            "name": "sat_temp",
                            "spacecraftBus": {
                                "name": "BlueCanyon",
                                "mass": 20,
                                "volume": 0.5,
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                }
                            },
                            "instrument": {
                                "name": "VNIR hyper",
                                "@id" : "vnir_hyp_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 2.5
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Multispectral"
                            },
                            "orbitState": {
                                "date": {
                                    "@type": "GREGORIAN_UT1",
                                    "year": 2020,
                                    "month": 1,
                                    "day": 1,
                                    "hour": 0,
                                    "minute": 0,
                                    "second": 0
                                },
                                "state": {
                                    "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                                    "sma": 7078,
                                    "ecc": 0.01,
                                    "inc": 60.0,
                                    "raan": 0.0,
                                    "aop": 98.0,
                                    "ta": 0.0
                                }
                            },
                            "groundStationNetwork" : self.GS_NETWORK_NAME,
                            "mission" : self.MISSION_NAME
                    }
        
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

        # define sat 2
        sat2 = copy.deepcopy(self.spacecraft_template)
        sat2['name'] = 'sat_2'
        sat2['@id'] = 'sat_2'
        sat2['orbitState']['state']['inc'] = 180.0
        sat2['orbitState']['state']['ta'] = sat1['orbitState']['state']['ta'] - 60.0 # phase offset by 60.0[deg]

        # compile ground stations
        d['groundStation'] = self.compile_ground_stations()

        # configure ground operator 
        d['groundOperator'] = self.define_ground_operators()

        # add spacecraft to mission specifications
        d['spacecraft'] = [sat1, sat2]
        
        # return mission specifications
        return d
    
    def compile_ground_stations(self) -> list:
        grid_path = f"./tests/connectivity/resources/{self.GS_NETWORK_NAME}.csv"
        assert os.path.isfile(grid_path), f"Ground station file not found: {self.GS_NETWORK_NAME}.csv"

        # load ground station network from file
        df = pd.read_csv(grid_path)
        gs_network_df : list[dict] = df.to_dict(orient='records')

        # if no id in file, add index as id
        gs_network = []
        for gs_idx, gs_df in enumerate(gs_network_df):
            gs = {
                "name": gs_df['name'],
                "latitude": gs_df['lat[deg]'],
                "longitude": gs_df['lon[deg]'],
                "altitude": gs_df['alt[km]'],
                "minimumElevation": gs_df['minElevation[deg]'],
                "@id": gs_df['@id'] if '@id' in gs_df else f'{self.GS_NETWORK_NAME}-{gs_idx}',
                "networkName" : self.GS_NETWORK_NAME
            }
            gs_network.append(gs)

        # return ground station network as list of dicts
        return gs_network
    
    def define_ground_operators(self) -> list:
        return [{
                "name" : self.GS_NETWORK_NAME,
                "@id" : self.GS_NETWORK_NAME.lower(),
                "mission" : self.MISSION_NAME,
            }]
        
    def generate_orbit_data(self, connectivity : str, overwrite : bool=False) -> Tuple[set, set, Dict[str, OrbitData]]:
        # create mission specs with given connectivity
        mission_specs = self.build_mission(connectivity=connectivity)

        # precompute orbit data
        orbitdata_dir = OrbitData.precompute(mission_specs, overwrite)

        # extract satellite names
        satellite_names = {sc['name'] for sc in mission_specs['spacecraft']}

        # extract ground operator names
        ground_operator_names = {go['name'] for go in mission_specs['groundOperator']}

        # load and return orbit data
        return satellite_names, ground_operator_names, OrbitData.from_directory(orbitdata_dir)

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
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying Full Connectivity Case', leave=True):
            for interval_data in tqdm(sender_orbitdata.comms_links.values(), desc=f'  Checking links for {sender_name}', leave=False):
                # ensure only a single access interval exists between sender and receiver
                self.assertTrue(len(interval_data) == 1)
                
                # ensure access interval spans entire mission duration
                t_start,t_end,*_ = interval_data.data[0]
                self.assertTrue(abs(t_start - 0.0) <= sender_orbitdata.time_step)
                self.assertTrue(abs(t_end - mission_duration) <= sender_orbitdata.time_step)
        
    def test_los_connectivity(self):
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.LOS.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying LOS Connectivity Case', leave=True):
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(sender_orbitdata.comms_links.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names:
                    # receiver is ground station, sender must be a satellite; set appropriate reference access times 
                    if "1" in sender_name:
                        t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                    
                    elif "2" in sender_name:
                        t_refs = [(2824 * sender_orbitdata.time_step, 3182 * sender_orbitdata.time_step)]

                    else:
                        raise ValueError(f'Unknown sender name: {sender_name}')
                else:
                    # receiver is a satellite; check if sender is ground station
                    if sender_name in ground_operator_names:
                        # sender is a ground station; set appropriate reference access times 
                        if "1" in receiver_name:                            
                            t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                        
                        elif "2" in receiver_name:
                            t_refs = [(2824 * sender_orbitdata.time_step, 3182 * sender_orbitdata.time_step)]

                        else:
                            raise ValueError(f'Unknown sender name: {sender_name}')
                    else:
                        # both sender and receiver are satellites; set reference access times 
                        t_refs = [
                            (1102 * sender_orbitdata.time_step, 1843 * sender_orbitdata.time_step),
                            (3726 * sender_orbitdata.time_step, 4479 * sender_orbitdata.time_step),
                            (6336 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)
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
                for (t_ref_start,t_rev_end),(t_start,t_end, *_) in zip(t_refs, interval_data.data):
                    self.assertTrue(abs(t_start - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(t_end - t_rev_end) <= sender_orbitdata.time_step)
        
        
    def test_isl_connectivity(self):
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.ISL.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying ISL Connectivity Case', leave=True):
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(sender_orbitdata.comms_links.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # eitherreceiver is ground station, sender must be a satellite; no accesses should be available
                    t_refs = []
                    
                else:
                    # both sender and receiver are satellites; set reference access times 
                    t_refs = [
                        (1102 * sender_orbitdata.time_step, 1843 * sender_orbitdata.time_step),
                        (3726 * sender_orbitdata.time_step, 4479 * sender_orbitdata.time_step),
                        (6336 * sender_orbitdata.time_step, 6376 * sender_orbitdata.time_step)
                    ]                 

                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),(t_start,t_end, *_) in zip(t_refs, interval_data.data):
                    self.assertTrue(abs(t_start - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(t_end - t_rev_end) <= sender_orbitdata.time_step)

    def test_gs_connectivity(self):
        # load orbit data
        _, ground_operator_names, orbit_data = self.generate_orbit_data(ConnectivityLevels.GS.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying GS Connectivity Case', leave=True):
            # compare loaded data to printed coverage data
            for receiver_name, interval_data in tqdm(sender_orbitdata.comms_links.items(), desc=f'  Checking links for {sender_name}', leave=False):
                # set reference access times depending on receiver type
                if receiver_name in ground_operator_names or sender_name in ground_operator_names:
                    # either receiver is ground station, sender must be a satellite; set appropriate reference access times 
                    if "1" in sender_name or "1" in receiver_name:
                        t_refs = [(5168 * sender_orbitdata.time_step, 5513 * sender_orbitdata.time_step)]
                    
                    elif "2" in sender_name or "2" in receiver_name:
                        t_refs = [(2824 * sender_orbitdata.time_step, 3182 * sender_orbitdata.time_step)]

                    else:
                        raise ValueError(f'Unknown sender name: {sender_name}')
                    
                else:
                    # both sender and receiver are satellites; no accesses should be available
                    t_refs = []                 

                # ensure number of accesses match reference times
                self.assertTrue(len(interval_data) == len(t_refs))

                # ensure access interval spans entire mission duration
                for (t_ref_start,t_rev_end),(t_start,t_end, *_) in zip(t_refs, interval_data.data):
                    self.assertTrue(abs(t_start - t_ref_start) <= sender_orbitdata.time_step)
                    self.assertTrue(abs(t_end - t_rev_end) <= sender_orbitdata.time_step)

    def test_no_connectivity(self):
        # load orbit data
        _, _, orbit_data = self.generate_orbit_data(ConnectivityLevels.NONE.value, False)

        # check connectivity for each agent
        for sender_name,sender_orbitdata in tqdm(orbit_data.items(), desc=f'Verifying No Connectivity Case', leave=True):
            for interval_data in tqdm(sender_orbitdata.comms_links.values(), desc=f'  Checking links for {sender_name}', leave=False):
                # ensure no access intervals exist between sender and receiver
                self.assertTrue(len(interval_data) == 0)

if __name__ == '__main__':
    # print banner
    print_banner("Connectivity Test Suite")

    # run tests
    unittest.main()