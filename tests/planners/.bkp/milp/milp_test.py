import os
import unittest

import numpy as np

from chess3d.mission.mission import *
from chess3d.simulation import Simulation
from chess3d.utils import print_welcome

class TestSolver(unittest.TestCase):
    ...

# class TestMission(unittest.TestCase):
#     def test_events(self):
#         """
#         O4: Observe events “Algal Blooms” (w=10)
#             Main parameter: Chl-A, MR = O1
#             DA: See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)
#             CA: Severity proportional to lake area (as in paper) 
#             CO: Secondary params = Water temperature and water level, MR as in O2 and O3
#             RO: From Ben’s paper, rewards for subsequent observations or something simple like U(n) first increases to guarantee some reobs but then decreases exponentially beyond a certain #obs (e.g., 3)

#         """

#         event = GeophysicalEvent('Algal Bloom', 
#                                  1.0, 
#                                  [
#                                     (0.0,0.0,0,0),
#                                     (1.0,1.0,0,1)
#                                   ], 
#                                  0.5, 
#                                  1.0, 
#                                  0.25)

#         # check initialization
#         self.assertEqual(event.event_type, 'algal bloom')
#         self.assertEqual(event.severity, 1.0)
#         self.assertEqual(event.t_start, 0.5)
#         self.assertEqual(event.t_end, 1.0)
#         self.assertEqual(event.t_corr, 0.25)

#         # check serialization
#         event_dict = event.to_dict()
#         event_reconstructed : GeophysicalEvent = GeophysicalEvent.from_dict(event_dict)
#         self.assertEqual(event.event_type, event_reconstructed.event_type)
#         self.assertEqual(event.severity, event_reconstructed.severity)
#         self.assertEqual(event.t_start, event_reconstructed.t_start)
#         self.assertEqual(event.t_end, event_reconstructed.t_end)
#         self.assertEqual(event.t_corr, event_reconstructed.t_corr)
#         self.assertEqual(event.id, event_reconstructed.id)
#         self.assertEqual(event, event_reconstructed)

#     def test_requirement(self):
#         """
#         O1: Measure Chlorophyll-A (w=1): 
#             Horizontal spatial resolution: Thresholds = [10, 30, 100] m, u = [1.0, 0.7, 0.1]
#             Spectral resolution: Thresholds = [Hyperspectral, Multispectral], u = [1, 0.5]

#         """
#         # Test the MeasurementRequirement class
#         # Create instances of MeasurementRequirement
#         req_1 = MeasurementRequirement('spatial_resolution', 
#                                        [10, 30, 100], 
#                                        [1.0, 0.7, 0.1])
#         req_2 = MeasurementRequirement('spectral_resolution', 
#                                        ["Hyperspectral", "Multispectral"], 
#                                        [1.0, 0.5])

#         # Check initialization
#         self.assertEqual(req_1.attribute, 'spatial_resolution')
#         self.assertEqual(req_1.thresholds, [10, 30, 100])
#         self.assertEqual(req_1.scores, [1.0, 0.7, 0.1])
        
#         self.assertEqual(req_2.attribute, 'spectral_resolution')
#         self.assertEqual(req_2.thresholds, ["hyperspectral", "multispectral"])
#         self.assertEqual(req_2.scores, [1.0, 0.5])
        
#         # Check performance function
#         self.assertAlmostEqual(req_1.calc_preference_value(5), 1.0)
#         self.assertAlmostEqual(req_1.calc_preference_value(20), 0.85)
#         self.assertAlmostEqual(req_1.calc_preference_value(65), 0.4)
#         self.assertAlmostEqual(req_1.calc_preference_value(150), 0.1*np.exp(-50))

#         self.assertAlmostEqual(req_2.calc_preference_value("Hyperspectral"), 1.0)
#         self.assertAlmostEqual(req_2.calc_preference_value("Multispectral"), 0.5)
#         self.assertAlmostEqual(req_2.calc_preference_value("Aspectral"), 0)
        
#     def test_objective(self):
#         """
#         O1: Measure Chlorophyll-A (w=1): 
#             Horizontal spatial resolution: Thresholds = [10, 30, 100] m, u = [1.0, 0.7, 0.1]
#             Spectral resolution: Thresholds = [Hyperspectral, Multispectral], u = [1, 0.5]

#         O2: Measure Water temperature (w=1)
#             Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.3]
        
#         O3: Measure Water level (w=1)
#             Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.5]
#             Accuracy: Thresholds = [10, 50, 100] cm, u = [1.0, 0.5, 0.1]

#         """
#         # Test the Objective class
#         # Create instances of MeasurementRequirement
#         req_1 = MeasurementRequirement('spatial_resolution', 
#                                        [10, 30, 100], 
#                                        [1.0, 0.7, 0.1])
#         req_2 = MeasurementRequirement('spectral_resolution', 
#                                        ["Hyperspectral", "Multispectral"], 
#                                        [1.0, 0.5])
#         req_3 = MeasurementRequirement('accuracy', 
#                                        [10, 50, 100], # [cm] 
#                                        [1.0, 0.5, 0.1])
        
#         # create objectives
#         o_1 = MissionObjective('Chlorophyll-A',
#                         1.0, 
#                         [req_1, req_2],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_2 = MissionObjective('Water Temperature',
#                         1.0, 
#                         [req_1],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_3 = MissionObjective('Water Level',
#                         1.0, 
#                         [req_1, req_3],
#                         ['VNIR'],
#                         'no_change'
#                     )
#         o_4 = MissionObjective('Water Level',
#                         1.0, 
#                         [req_1, req_3],
#                         ['TIR'],
#                         'no_change'
#                     )
        
#         # create example measurement
#         measurement = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 1,
#         }
  
#         self.assertAlmostEqual(o_1.eval_performance(measurement), 0.85*1.0)
#         self.assertAlmostEqual(o_2.eval_performance(measurement), 0.85)
#         self.assertAlmostEqual(o_3.eval_performance(measurement), 0.0)
#         self.assertAlmostEqual(o_4.eval_performance(measurement), 0.0)

#     def test_event_driven_objective(self):
#         """
#         O4: Observe events “Algal Blooms” (w=10)
#             Main parameter: Chl-A, MR = O1
#             DA: See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)
#             CA: Severity proportional to lake area (as in paper) 
#             CO: Secondary params = Water temperature and water level, MR as in O2 and O3
#             RO: From Ben’s paper, rewards for subsequent observations or something simple like U(n) first increases to guarantee some reobs but then decreases exponentially beyond a certain #obs (e.g., 3)
#         """
#         # Create instances of MeasurementRequirement
#         req_1 = MeasurementRequirement('spatial_resolution', 
#                                        [10, 30, 100], 
#                                        [1.0, 0.7, 0.1])
#         req_2 = MeasurementRequirement('spectral_resolution', 
#                                        ["Hyperspectral", "Multispectral"], 
#                                        [1.0, 0.5])
        
#         # create event
#         event = GeophysicalEvent('Algal Bloom',
#                                     1.0, 
#                                     [
#                                         (0.0,0.0,0,0),
#                                         (1.0,1.0,0,1)
#                                     ], 
#                                     0.5, 
#                                     1.0, 
#                                     0.25)

#         # create objectives
#         o_4_1 = EventDrivenObjective('Chlorophyll-A',
#                                      10.0,
#                                      [req_1, req_2],
#                                      event.event_type,
#                                      ['VNIR'],
#                                      'linear_increase'
#                                      )
        
#         # create example measurements
#         measurement_1 = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 1
#         }
#         measurement_2 = {
#             "instrument": "VNIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 2
#         }
#         measurement_3 = {
#             "instrument": "VNIR",
#             "accuracy": 1.0,
#             "n_obs" : 3
#         }
#         measurement_4 = {
#             "instrument": "TIR",
#             "spatial_resolution": 20,
#             "spectral_resolution": "Hyperspectral",
#             "n_obs" : 3
#         }

#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_1), 0.85*1.0*1.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_2), 0.85*1.0*2.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_3), 0.0)
#         self.assertAlmostEqual(o_4_1.eval_performance(measurement_4), 0.0)


# class TestToySatCase(unittest.TestCase):
#     def setUp(self) -> None:
#         # terminal welcome message
#         print_welcome('Simulation Loading Test')
        
#         # load scenario json file
#         scenario_specs : dict = {
#             "epoch": {
#                 "@type": "GREGORIAN_UT1",
#                 "year": 2020,
#                 "month": 1,
#                 "day": 1,
#                 "hour": 0,
#                 "minute": 0,
#                 "second": 0
#             },
#             "duration": 1.0 / 24.0,
#             "propagator": {
#                 "@type": "J2 ANALYTICAL PROPAGATOR",
#             },
#             "spacecraft": [
#                 {
#                     "@id": "thermal_sat_0_0",
#                     "name": "thermal_0",
#                     "spacecraftBus": {
#                         "name": "BlueCanyon",
#                         "mass": 20,
#                         "volume": 0.5,
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "components": {
#                             "adcs" : {
#                                 "maxTorque" : 1000,
#                                 "maxRate" : 1
#                             }
#                         }
#                     },
#                     "instrument": {
#                         # "name": "Altimeter",
#                         # "@id" : "altimeter",
#                         # "@type" : "Altimeter",
#                         # "chirpBandwidth": 150e6,
#                         # "pulseWidth": 50e-6,  
#                         # "orientation": {
#                         #     "referenceFrame": "NADIR_POINTING",
#                         #     "convention": "REF_FRAME_ALIGNED"
#                         # },
#                         # "fieldOfViewGeometry": { 
#                         #     "shape": "RECTANGULAR", 
#                         #     "angleHeight": 2.5, 
#                         #     "angleWidth": 45.0
#                         # },
#                         # "maneuver" : {
#                         #     "maneuverType":"SINGLE_ROLL_ONLY",
#                         #     "A_rollMin": -50,
#                         #     "A_rollMax": 50
#                         # }
#                         "name": "TIR",
#                         "@id" : "vnir_hyp_imager",
#                         "@type" : "VNIR",
#                         "detectorWidth": 6.6e-6,
#                         "focalLength": 3.6,  
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "fieldOfViewGeometry": { 
#                             "shape": "RECTANGULAR", 
#                             "angleHeight": 2.5, 
#                             "angleWidth": 45.0
#                         },
#                         "maneuver" : {
#                             "maneuverType":"SINGLE_ROLL_ONLY",
#                             "A_rollMin": -50,
#                             "A_rollMax": 50
#                         },
#                         "spectral_resolution" : "Multispectral"
#                     },
#                     "orbitState": {
#                         "date": {
#                             "@type": "GREGORIAN_UT1",
#                             "year": 2020,
#                             "month": 1,
#                             "day": 1,
#                             "hour": 0,
#                             "minute": 0,
#                             "second": 0
#                         },
#                         "state": {
#                             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
#                             "sma": 7078,
#                             "ecc": 0.01,
#                             "inc": 0.0,
#                             "raan": 0.0,
#                             "aop": 0.0,
#                             "ta": 95.0
#                         }
#                     },
#                     "planner" : {
#                         "preplanner" : {
#                             "@type" : "heuristic",
#                             "period": 1000,
#                             # "horizon": 500,
#                         },
#                         # "replanner" : {
#                         #     "@type" : "broadcaster",
#                         #     "period" : 400
#                         # },
#                     },
#                     "science" : {
#                         "@type": "lookup", 
#                         "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
#                     },
#                     "mission" : "Algal blooms monitoring"
#                 }
#             ],
#             "grid": [
#                 {
#                     "@type": "customGrid",
#                     "covGridFilePath": "./tests/missions/resources/grids/toy_points.csv"
#                 }
#             ],
#             "scenario": {   
#                 "connectivity" : "FULL", 
#                 "events" : {
#                     "@type": "PREDEF", 
#                     "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
#                 },
#                 "clock" : {
#                     "@type" : "EVENT"
#                 },
#                 "scenarioPath" : "./tests/missions/",
#                 "name" : "toy_sat_case",
#                 "missionsPath" : "./tests/missions/resources/missions/missions.json"
#             },
#             "settings": {
#                 "coverageType": "GRID COVERAGE",
#                 "outDir" : "./tests/missions/orbit_data/toy_sat_case",
#             }
#         }

#         # set outdir
#         orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
#         scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'toy_sat_case')
#         if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
#         if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

#         # initialize mission
#         self.simulation : Simulation = Simulation.from_dict(scenario_specs)

#         # check type of mission object
#         self.assertTrue(isinstance(self.simulation, Simulation))


#     # def test_planner(self) -> None:
#     #     # execute mission
#     #     self.simulation.execute()

#     #     # print results
#     #     self.simulation.print_results()

#     #     print('DONE')

# class TestSingleSatCase(unittest.TestCase):
#     def setUp(self) -> None:
#         # terminal welcome message
#         print_welcome('Simulation Loading Test')
        
#         # load scenario json file
#         scenario_specs : dict = {
#             "epoch": {
#                 "@type": "GREGORIAN_UT1",
#                 "year": 2020,
#                 "month": 1,
#                 "day": 1,
#                 "hour": 0,
#                 "minute": 0,
#                 "second": 0
#             },
#             "duration": 1500 / 3600 / 24.0,
#             "propagator": {
#                 "@type": "J2 ANALYTICAL PROPAGATOR",
#             },
#             "spacecraft": [
#                 {
#                     "@id": "thermal_sat_0_0",
#                     "name": "thermal_0",
#                     "spacecraftBus": {
#                         "name": "BlueCanyon",
#                         "mass": 20,
#                         "volume": 0.5,
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "components": {
#                             "adcs" : {
#                                 "maxTorque" : 1000,
#                                 "maxRate" : 1
#                             }
#                         }
#                     },
#                     "instrument": {
#                         "name": "TIR",
#                         "@id" : "vnir_hyp_imager",
#                         "@type" : "VNIR",
#                         "detectorWidth": 6.6e-6,
#                         "focalLength": 3.6,  
#                         "orientation": {
#                             "referenceFrame": "NADIR_POINTING",
#                             "convention": "REF_FRAME_ALIGNED"
#                         },
#                         "fieldOfViewGeometry": { 
#                             "shape": "RECTANGULAR", 
#                             "angleHeight": 2.5, 
#                             "angleWidth": 5.0
#                         },
#                         "maneuver" : {
#                             "maneuverType":"SINGLE_ROLL_ONLY",
#                             "A_rollMin": -50,
#                             "A_rollMax": 50
#                         },
#                         "spectral_resolution" : "Multispectral"
#                     },
#                     "orbitState": {
#                         "date": {
#                             "@type": "GREGORIAN_UT1",
#                             "year": 2020,
#                             "month": 1,
#                             "day": 1,
#                             "hour": 0,
#                             "minute": 0,
#                             "second": 0
#                         },
#                         "state": {
#                             "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
#                             "sma": 7078,
#                             "ecc": 0.01,
#                             "inc": 60.0,
#                             "raan": 0.0,
#                             "aop": 0.0,
#                             "ta": 95.0
#                         }
#                     },
#                     "planner" : {
#                         "preplanner" : {
#                             "@type" : "heuristic",
#                             "period": 1000,
#                             # "horizon": 500,
#                         },
#                         # "replanner" : {
#                         #     "@type" : "broadcaster",
#                         #     "period" : 400
#                         # },
#                     },
#                     "science" : {
#                         "@type": "lookup", 
#                         "eventsPath" : "./tests/missions/resources/events/toy_events.csv"
#                     },
#                     "mission" : "Algal blooms monitoring"
#                 }
#             ],
#             "grid": [
#                 {
#                     "@type": "customGrid",
#                     "covGridFilePath": "./tests/missions/resources/grids/lake_event_points.csv"
#                 }
#             ],
#             "scenario": {   
#                 "connectivity" : "FULL", 
#                 "events" : {
#                     "@type": "PREDEF", 
#                     "eventsPath" : "./tests/missions/resources/events/lake_events_seed-1000.csv"
#                 },
#                 "clock" : {
#                     "@type" : "EVENT"
#                 },
#                 "scenarioPath" : "./tests/missions/",
#                 "name" : "single_sat_case",
#                 "missionsPath" : "./tests/missions/resources/missions/missions.json"
#             },
#             "settings": {
#                 "coverageType": "GRID COVERAGE",
#                 "outDir" : "./tests/missions/orbit_data/single_sat_case",
#             }
#         }

#         # set outdir
#         orbitdata_dir = os.path.join('./tests/missions', 'orbit_data')
#         scenario_orbitdata_dir = os.path.join(orbitdata_dir, 'single_sat_case')
#         scenario_specs['settings']['outDir'] = scenario_orbitdata_dir
#         if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)
#         if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

#         # initialize mission
#         self.simulation : Simulation = Simulation.from_dict(scenario_specs)

#         # check type of mission object
#         self.assertTrue(isinstance(self.simulation, Simulation))


#     # def test_planner(self) -> None:
#     #     # execute mission
#     #     self.simulation.execute()

#     #     # print results
#     #     self.simulation.print_results()

#     #     print('DONE')

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('MILP Preplanner Test')
    
    # run tests
    unittest.main()