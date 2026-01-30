import copy
import os
import unittest

import pandas as pd

from dmas.core.orbitdata import OrbitData
from dmas.core.simulation import Simulation
from dmas.models.agent import SimulationAgent
from dmas.models.environment import SimulationEnvironment
from dmas.utils.tools import print_scenario_banner


class TestSimulationLoading(unittest.TestCase):
    # constants
    MISSION_NAME = 'sample_mission'
    GS_NETWORK_NAME = 'gs'

    # Define a basic mission template with connectivity settings
    def setUp(self):
        super().setUp()
        
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
                                "eventsPath": f"./tests/simulation/resources/events.csv"
                            },
                            "clock" : {
                                "@type" : "EVENT"
                            },
                            "scenarioPath" : "./tests/simulation/",
                            "name" : "output_test",
                            "missionsPath" : f"./tests/simulation/resources/missions.json"
                        },
                        "settings": {
                            "coverageType": "GRID COVERAGE",
                            "outDir" : "./tests/simulation/orbit_data"
                        },
                        "grid": [
                            {
                                "@type": "customGrid",
                                "covGridFilePath": "./tests/simulation/resources/grid.csv"
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
                                },
                                "components": {
                                    "adcs" : {
                                        "maxTorque" : 1000,
                                        "maxRate" : 15
                                    }
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
                            "planner" : {
                                # "preplanner": {
                                #     "@type": "heuristic",
                                #     "debug": "False",
                                #     # "period" : 250,
                                # },
                                    "replanner": {
                                        "@type": "consensus",
                                        "model": "heuristicInsertion",
                                        "heuristic" : "taskPriority",
                                        "replanThreshold": 1,
                                        "optimisticBiddingThreshold": 1,
                                        "debug": "False"
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
        grid_path = f"./tests/simulation/resources/{self.GS_NETWORK_NAME}.csv"
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
                "planner" : {
                    "preplanner": {
                        "@type": "eventAnnouncer",
                        "debug": "False",                        
                        "eventsPath" : f"./tests/simulation/resources/events.csv"
                    }
                }
            }]
        
    """
    ============================================
        TEST CASES
    ============================================
    """
    def test_simulation_loads_successfully(self):

       # create mission specs with given connectivity
        mission_specs = self.build_mission(connectivity='LOS')

        # initialize mission
        simulation : Simulation = Simulation.from_dict(mission_specs, overwrite=True)
        
        # check that simulation is initialized
        self.assertIsInstance(simulation, Simulation)
        self.assertIsInstance(simulation._environment, SimulationEnvironment)
        self.assertTrue(all(isinstance(agent, SimulationAgent) for agent in simulation._agents))
        self.assertIsInstance(simulation._orbitdata, dict)
        self.assertTrue(all(isinstance(orbitdata, OrbitData) for orbitdata in simulation._orbitdata.values()))
        self.assertEqual(len(simulation._orbitdata), len(simulation._agents))

    def test_simulation_runs_successfully(self):

       # create mission specs with given connectivity
        mission_specs = self.build_mission(connectivity='LOS')

        # initialize mission
        simulation : Simulation = Simulation.from_dict(mission_specs, overwrite=True)

        # execute simulation
        val = simulation.execute()

        # ensure 
        self.assertTrue(val)
    

if __name__ == '__main__':
    # just ensure it runs without error
    print_scenario_banner('Simulation Loading Tests')

    # run tests
    unittest.main()