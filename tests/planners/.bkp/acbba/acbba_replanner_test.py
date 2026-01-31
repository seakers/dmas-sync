import copy
import os
import unittest

import numpy as np

from chess3d.simulation import Simulation
from chess3d.utils import print_welcome

class ToyTestACBBAReplanner(unittest.TestCase):
    def setUp(self) -> None:
        # load scenario json file
        mission_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 1 / 24.0,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
                "stepSize": 10
            },
            "spacecraft": [
                {
                    "@id": "img_sat_0",
                    "name": "img_0",
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
                        "name": "visible",
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
                            "angleWidth": 45
                        },
                        "@id": "img1",
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive"
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/toy_events.csv"
                    }
                },
                {
                    "@id": "sar_sat_0",
                    "name": "sar_0",
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
                                "maxRate" : 0.1
                            }
                        }
                    },
                    "instrument": {
                        "name": "sar",
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
                            "angleHeight": 5.0,
                            "angleWidth": 2.5
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -20.0,
                            "A_rollMax": 20.0
                        },
                        "@id": "sar1",
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": -10.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive"
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/toy_events.csv"
                    }
                },
                {
                    "@id": "sar_sat_1",
                    "name": "sar_1",
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
                                "maxRate" : 0.1
                            }
                        }
                    },
                    "instrument": {
                        "name": "sar",
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
                            "angleHeight": 5.0,
                            "angleWidth": 2.5
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -20.0,
                            "A_rollMax": 20.0
                        },
                        "@id": "sar1",
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": -11.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive"
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/toy_events.csv"
                    }
                },
                {
                    "@id": "thermal_sat_0",
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
                                "maxRate" : 0.1
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
                            "angleHeight": 5.0,
                            "angleWidth": 2.5
                        },
                        "maneuver" : {
                            "maneuverType":"SINGLE_ROLL_ONLY",
                            "A_rollMin": -20.0,
                            "A_rollMax": 20.0
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
                            "inc": 0.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": -12.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive"
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/toy_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/replanners/acbba/resources/toy_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/replanners/acbba/resources/toy_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/replanners/acbba/",
                "name" : "toy"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/replanners/acbba/orbit_data/toy"
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/replanners', 'acbba', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy')
        mission_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)


        # initialize mission
        self.toy_mission : Simulation = Simulation.from_dict(mission_specs)

    # def test_toy_mission(self) -> None:
    #     # execute mission
    #     self.toy_mission.execute()

    #     # print results
    #     self.toy_mission.print_results()

class MissionTestACBBAReplanner(unittest.TestCase):
    def setUp(self) -> None:
        # load scenario json file
        mission_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
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
                            "@type" : "heuristic",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                },
                {
                    "@id": "vis_sat_0_0",
                    "name": "vis_0",
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
                        "name": "visual",
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
                            "ta": -5.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "heuristic",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                },
                {
                    "@id": "sar_sat_0_0",
                    "name": "sar_0",
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
                        "name": "sar",
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
                            "ta": -10.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "heuristic",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba",
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/replanners/acbba/resources/lake_event_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/replanners/acbba/",
                "name" : "mission"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/replanners/acbba/orbit_data/mission"
            }
        }

        # set outdir
        orbitdata_dir = os.path.join('./tests/replanners', 'acbba', 'orbit_data')
        scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'mission')
        mission_specs['settings']['outDir'] = scenario_orbitdata_dir
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # initialize mission
        self.mission : Simulation = Simulation.from_dict(mission_specs)

    def test_mission(self) -> None:
        # execute mission
        self.mission.execute()

        # print results
        self.mission.print_results()

class MissionTestDynamicACBBAReplanner(unittest.TestCase):
    def setUp(self) -> None:
        # load scenario json file
        mission_specs : dict = {
            "epoch": {
                "@type": "GREGORIAN_UT1",
                "year": 2020,
                "month": 1,
                "day": 1,
                "hour": 0,
                "minute": 0,
                "second": 0
            },
            "duration": 3 / 24.0,
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
                            "@type" : "naive",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba-dp"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                },
                {
                    "@id": "vis_sat_0_0",
                    "name": "vis_0",
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
                        "name": "visual",
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
                            "ta": -5.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba-dp"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                },
                {
                    "@id": "sar_sat_0_0",
                    "name": "sar_0",
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
                        "name": "sar",
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
                            "ta": -10.0
                        }
                    },
                    "planner" : {
                        "preplanner" : {
                            "@type" : "naive",
                            "period" : 500
                        },
                        "replanner" : {
                            "@type" : "acbba-dp"
                        },
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
                        "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                    }
                }
            ],
            "grid": [
                {
                    "@type": "customGrid",
                    "covGridFilePath": "./tests/replanners/acbba/resources/lake_points.csv"
                }
            ],
            "scenario": {   
                "connectivity" : "FULL", 
                "utility" : "LINEAR",
                "events" : {
                    "@type": "PREDEF", 
                    "eventsPath" : "./tests/replanners/acbba/resources/lake_events.csv"
                },
                "clock" : {
                    "@type" : "EVENT"
                },
                "scenarioPath" : "./tests/replanners/acbba/",
                "name" : "mission_dp"
            },
            "settings": {
                "coverageType": "GRID COVERAGE",
                "outDir" : "./tests/replanners/acbba/orbit_data/mission_dp"
            }
        }

        # initialize mission
        self.mission : Simulation = Simulation.from_dict(mission_specs)

    # def test_mission(self) -> None:
    #     # execute mission
    #     self.toy_mission.execute()

    #     # print results
    #     self.toy_mission.print_results()

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('ACBBA Planner Test')
    
    # run tests
    unittest.main()