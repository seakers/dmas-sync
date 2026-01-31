import unittest

from tqdm import tqdm

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome
# from runtime_plots import plot_scenario_runtime

class TestDynamicProgrammingPlannerBen(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Dynamic Programming Planner Test')
        
        # load scenario json file
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 4.0 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
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
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "thermal",
                        "mass": 10,
                        "volume": 12.45,
                        "dataRate": 40,
                        "bitsPerPixel": 8,
                        "power": 12,
                        "snr": 33,
                        "spatial_res": 50,
                        "spectral_res": 7e-09,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": {
                            "shape": "RECTANGULAR",
                            "angleHeight": 5,
                            "angleWidth": 10
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "@id": "therm1",
                        "@type": "Basic Sensor"
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
                            "inc": 67,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "dynamic",
                            "period" : 500,     # s
                        },
                        # "replanner" : {
                        #     "@type" : "broadcaster"
                        # },
                        "rewardGrid":{
                            "reward_function" : 'event',
                            'initial_reward' : 1.0,
                            'min_reward' : 1.0,
                            'unobserved_reward_rate' : 2.0, # pts/hrs
                            'max_unobserved_reward' : 10.0,
                            'event_reward' : 10.0
                        }
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/preplanners/dynamic/resources/events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/preplanners/dynamic/resources/points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/preplanners/dynamic/resources/events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/preplanners/dynamic/",
                "name" : "dynamic_ben"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/preplanners/dynamic/orbit_data/dynamic_ben"
            }
        }

        # initialize mission
        self.mission : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.mission, Simulation))

    # def test_planner(self) -> None:
    #     # execute mission
    #     self.mission.execute()

    #     # print results
    #     self.mission.print_results()

        # print('DONE')

class TestDynamicProgrammingPlannerRandom(unittest.TestCase):
    def setUp(self) -> None:
        # terminal welcome message
        print_welcome('Dynamic Programming Planner Test')
        
        # load scenario json file
        scenario_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            # "duration": 90/60 / 24.0,
            "duration": 2 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
                    "@id": "thermal_sat_0_0",
                    "name": "thermal_0",
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
                                "maxRate" : 1
                            }
                        }
                    },
                    "instrument": {
                        "name": "thermal",
                        "mass": 10,
                        "volume": 12.45,
                        "dataRate": 40,
                        "bitsPerPixel": 8,
                        "power": 12,
                        "snr": 33,
                        "spatial_res": 50,
                        "spectral_res": 7e-09,
                        "orientation": {
                            "referenceFrame": "NADIR_POINTING",
                            "convention": "REF_FRAME_ALIGNED"
                        },
                        "fieldOfViewGeometry": {
                            "shape": "RECTANGULAR",
                            "angleHeight": 5,
                            "angleWidth": 10
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -50,
                            "A_rollMax": 50
                        },
                        "@id": "therm1",
                        "@type": "Basic Sensor"
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
                            "inc": 67,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "dynamic",
                            "period" : 500,     # s
                            # "sharing" : "True"
                        },
                        # "replanner" : {
                        #     "@type" : "acbba"
                        # },
                        "rewardGrid":{
                            "reward_function" : 'event',
                            'initial_reward' : 1.0,
                            'min_reward' : 1.0,
                            'unobserved_reward_rate' : 2.0, # pts/hrs
                            'max_unobserved_reward' : 10.0,
                            'event_reward' : 10.0
                        }
                    },
                    "science" : {
                        "@type": "lookup", 
                        "eventsPath" : "./tests/preplanners/dynamic/resources/lake_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/preplanners/dynamic/resources/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/preplanners/dynamic/resources/lake_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/preplanners/dynamic/",
                "name" : "dynamic_random"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/preplanners/dynamic/orbit_data/dynamic_random"
            }
        }

        # initialize mission
        self.mission : Simulation = Simulation.from_dict(scenario_specs)

        # check type of mission object
        self.assertTrue(isinstance(self.mission, Simulation))

    def test_planner(self) -> None:
        # execute mission
        self.mission.execute()

        # print results
        self.mission.print_results()

        # # plot runtime
        # scenarios = [
        #     'dynamic_random'
        # ]

        # agents = [
        #     'manager',
        #     'environment',
        #     'thermal_0'
        # ]
            
        # for agent in tqdm(agents, desc='Generating runtime performance plots for agents'):
        #     plot_scenario_runtime(scenarios, agent, False, True, './tests/preplanners/dynamic/results', './tests/preplanners/dynamic')

        # print('DONE')

if __name__ == '__main__':
    unittest.main()