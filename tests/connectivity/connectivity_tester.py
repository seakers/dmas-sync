from abc import ABC, abstractmethod
import copy
import os
from typing import Dict, Tuple
import unittest

import pandas as pd

from dmas.utils.orbitdata import OrbitData
from dmas.utils.tools import print_scenario_banner


class AgentConnectivityTester(ABC):
    """ Test scenarios where satellites can connect to ground stations but not to eachother. """
    
    # constants
    MISSION_NAME = 'sample_mission'
    GS_NETWORK_NAME = 'gs'

    def setUp(self):
        # Define a basic mission template with connectivity settings
        self.duration = 2.0 / 24.0 # 2 hours in days

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
                        "duration": self.duration,
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
                            # "groundStationNetwork" : self.GS_NETWORK_NAME,
                            "mission" : self.MISSION_NAME
                    }
        
    """
    ============================================
        MISSION BUILDING UTILITIES
    ============================================
    """
    @abstractmethod    
    def build_mission(self, connectivity : str) -> dict:
        pass
    
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
        return satellite_names, ground_operator_names, OrbitData.from_directory(orbitdata_dir, simulation_duration=self.duration)

    """
    ============================================
        TEST CASES
    ============================================
    """
    @abstractmethod
    def test_full_connectivity(self):
        pass

    @abstractmethod
    def test_los_connectivity(self):
        pass
        
    @abstractmethod
    def test_isl_connectivity(self):
        pass

    @abstractmethod
    def test_gs_connectivity(self):
        pass

    @abstractmethod
    def test_no_connectivity(self):
        pass
