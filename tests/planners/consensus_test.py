import copy
import os
from typing import List
import unittest

import numpy as np
import pandas as pd

from dmas.core.simulation import Simulation
from dmas.utils.tools import print_scenario_banner
from tests.planners.tester import PlannerTester


class TestConsensusPlanner(PlannerTester, unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        # planner output toggle
        self.planner_debug = True

        # test case toggles
        ## common cases
        self.single_sat_toy = False     # NOT IMPLEMENTED YET
        self.multiple_sat_toy = False   # NOT IMPLEMENTED YET
        self.single_sat_lakes = False   
        self.multiple_sat_lakes = False # NOT IMPLEMENTED YET

        ## toy cases
        self.toy_1 = False  # single sat    default mission     single target, no events
        self.toy_2 = False  # single sat    no default mission  one event
        self.toy_3 = False  # two sats      no default mission  one event
        self.toy_4 = False  # two sats      no default mission  one event           optimistic bidding
        self.toy_5 = False  # two sats      no default mission  one event           comm delays
        self.toy_6 = False  # two sats      no default mission  two targets         two events
        self.toy_7 = False  # two sats      no default mission  two targets         two events at different times
        self.toy_8 = False  # single sat    default mission     multiple targets    no events
        self.toy_9 = False  # two sats      default mission     multiple targets    no events
        self.toy_10 = False # single sat    no default mission  two targets         two expiring events 
        self.toy_11 = False # two sat       no default mission  two targets         two expiring events 
        self.toy_12 = False # single sat    default mission     multiple targets    no events           preplan + replan
        self.toy_13 = False # two sats      default mission     multiple targets    no events           preplan + replan
        self.toy_14 = False # single sat    default mission     multiple targets    two events          preplan + replan
        self.toy_15 = False # two sats      default mission     multiple targets    two events          preplan + replan
        self.toy_16 = False # single sat    no default mission  two targets         two expiring events  preplan + replan   not the correct instruments
        self.toy_17 = False # moving relay scenario
        self.toy_18 = False # static relay scenario
        self.toy_19 = False # single sat    default mission     multiple targets    two events           preplan w/short horizon + replan
        self.toy_20 = False # two sats       default mission     multiple targets    two events           preplan w/short horizon + replan
        self.toy_21 = False # single sat    no default mission     multiple targets    two events announced by GS  replan
        self.toy_22 = False # two sats      no default mission     multiple targets    two events announced by GS   replan

        self.toy_23 = False 
        self.toy_24 = False
        self.toy_25 = False
        self.toy_26 = False

        self.toy_27 = False # string of pearls with onboard event detection
        self.toy_28 = True # sequence reset case

    def toy_planner_config(self):
        return {
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskPriority",
                "replanThreshold": 1,
                "optimisticBiddingThreshold": 1,
                "debug": str(self.planner_debug)
            }
        }
    
    def toy_hollistic_planner_config(self):
        return {
            "preplanner": {
                # "@type": "heuristic",
                "@type": "blank",
                "debug": str(self.planner_debug),
                "period" : 250,
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskPriority",
                "replanThreshold": 1,
                "optimisticBiddingThreshold": 1,
                "debug": str(self.planner_debug)
            }
        }
    
    def toy_centralized_planner_config(self):
        return {
            "preplanner": {
                "@type": "worker",
                "dealerName" : -1, # TODO need to define dealer name
                "debug": str(self.planner_debug),
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskPriority",
                "replanThreshold": 1,
                "optimisticBiddingThreshold": 1,
                "periodicOverwrite": "True",
                "debug": str(self.planner_debug)
            }
        }
    
    
    def lakes_planner_config(self):
        return {
            "preplanner": {
                "@type": "heuristic",
                "debug": str(self.planner_debug),
                "period" : 100,
            },
            "replanner": {
                "@type": "consensus",
                "model": "heuristicInsertion",
                "heuristic" : "taskValue",
                "replanThreshold": 1,
                "debug": str(self.planner_debug)
            }
        }
        
    def setup_announcer_config(self, event_name : str = None) -> dict:
        """ Setup announcer planner configuration for the scenario. """

        # default to no planner
        if event_name is None: return {}

        # validate event file exists
        assert isinstance(event_name, str), "event_name must be a string"
        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        
        # return event announcer planner config
        return {
                "preplanner": {
                    "@type": "eventAnnouncer",
                    "debug": str(self.planner_debug),                        
                    "eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                }
            }

    def planner_name(self):
        return "consensus"
    
    def setup_announcer_ground_operators(self, event_name : str, mission_name : str, gs_network_names : List[str]) -> List[dict]:
        """ create ground operator specifications for the scenario. """

        # validate event file exists
        assert isinstance(event_name, str), "`event_name` must be a string"
        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        
        # generate ground operator specifications
        ground_ops = [
            {
                "name" : gs_network_name,
                "@id" : gs_network_name.lower(),
                "planner" : {
                    "preplanner": {
                        "@type": "eventAnnouncer",
                        "debug": str(self.planner_debug),                        
                        "eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                    }
                },
                "mission" : mission_name,
            }

            for gs_network_name in gs_network_names
        ]

        # return ground operator specifications
        return ground_ops
    
    def setup_cbba_ground_operators(self, mission_name : str, gs_network_names : List[str]) -> List[dict]:
        """ create ground operator specifications for the scenario. """
        
        # generate ground operator specifications
        ground_ops = [
            {
                "name" : gs_network_name,
                "@id" : gs_network_name.lower(),
                "planner" : {
                    "replanner": {
                        "@type": "consensus",
                        "model": "heuristicInsertion",
                        "heuristic" : "taskPriority",
                        "replanThreshold": 1,
                        "optimisticBiddingThreshold": 1,
                        "debug": str(self.planner_debug)
                    }
                },
                "mission" : mission_name,
            }

            for gs_network_name in gs_network_names
        ]

        # return ground operator specifications
        return ground_ops
    
    def setup_announcing_cbba_ground_operators(self, event_name : str, mission_name : str, gs_network_names : List[str]) -> List[dict]:
        """ create ground operator specifications for the scenario. """

        # validate event file exists
        assert isinstance(event_name, str), "`event_name` must be a string"
        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        
        # generate ground operator specifications
        ground_ops = [
            {
                "name" : gs_network_name,
                "@id" : gs_network_name.lower(),
                "planner" : {
                    "preplanner": {
                        "@type": "eventAnnouncer",
                        "debug": str(self.planner_debug),                        
                        "eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                    },
                    "replanner": {
                        "@type": "consensus",
                        "model": "heuristicInsertion",
                        "heuristic" : "taskPriority",
                        "replanThreshold": 1,
                        "optimisticBiddingThreshold": 1,
                        "debug": str(self.planner_debug)
                    }
                },
                "mission" : mission_name,
            }

            for gs_network_name in gs_network_names
        ]

        # return ground operator specifications
        return ground_ops
    
    def test_toy_case_1(self):
        """ 
        ## TOY CASE 1
        Test case for single satellite performing multiple observations of a single default mission task.

        ### Goals
        - Validate basic functionality of the bundle-building phase of the consensus planner in a simple scenario. 
        - Ensure that the satellite can plan and execute observations of a single target without any events occurring.
        - Ensure repeated obsevations are being tracked properly by planner.
        
        ### Mission Details
        - Default objectives: continuous observation of a predefined target grid.
        - Event-driven objectives: None

        ### Agents
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 12 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: No events

        ### Expected Outcomes
        - Satellite performs 7 observations, one per each access window.
        - All observations are successfully scheduled and executed without conflicts.
        - The planner effectively tracks the bidding and performance of the observations being scheduled.
        - Final planner results should indicate 7 completed bids and an empty bundle at the end of the simulation.
        - The environment results should reflect the successful completion of all scheduled observations.
        """

        # check for case toggle 
        if not self.toy_1: return

        # setup scenario parameters
        duration = 12.0 / 24.0
        grid_name = 'toy_1'
        scenario_name = f'toy_1-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_1'
        mission_database = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_1"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_database,
                                                   spacecraft=[
                                                       ractive_spacecraft_1
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_2(self):
        """
        ## TOY CASE 2
        Test case for single satellite performing an event-driven task from announcer.

        ### Goals
        - Validate basic functionality of the bundle-building phase of the consensus planner in a reactive scenario. 
        - Ensure that the satellite can plan and execute observations of a single target without any default mission tasks.
        - Ensure event announcements are being processed properly by planner.
        - Ensure repeated obsevations are being tracked properly by planner.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 12 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=1000 s, lasting for 2 hours.
        - Only one observation opportunity should be available during the event duration.

        ### Expected Outcomes
        - Satellite performs 1 observation of the event.
        - The observation is successfully scheduled and executed without conflicts.
        """

        # check for case toggle 
        if not self.toy_2: return

        # setup scenario parameters
        duration = 12.0 / 24.0
        grid_name = 'toy_2'
        scenario_name = f'toy_2-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_2'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_2"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_2"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()
        
        # summarize results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_3(self):
        """
        ## TOY CASE 3
        Test case for two satellite performing an event-driven task from announcer.

        ### Goals
        - Validate the functionality of the consensus planner in a multi-agent reactive scenario.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        
        - SAT2 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 2 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=5 seconds, lasting for 2 hours.
        - Same instruments for both agents
        - Agents offset by 2 degrees in true anomaly
        
        ### Expected Outcomes
        - Both satellites should perform 2 observations of the event each, one per each access window.
        - Agent 1 performs first observation before Agent 2's first observation.
        - Agent 1 performs second observation before Agent 2's second observation but after its first observation.
        - All observations are successfully scheduled and executed without conflicts.
        - The planner effectively tracks the bidding and performance of the observations being scheduled.
        - Final planner results should indicate 2 completed bids per satellite and an empty bundle at the end of the simulation.
        - The environment results should reflect the successful completion of all scheduled observations.
        """
        # check for case toggle 
        if not self.toy_3: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_3'
        scenario_name = f'toy_3-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_3'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_3"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_3"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] \
            = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_3"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_4(self):
        """
        ## TOY CASE 4
        Test case for optimisting bidding between two satellite performing an event-driven task from announcer.

        ### Goals
        - Ensure that the optimistic bidding mechanism in the consensus planner functions correctly in a multi-agent reactive scenario.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        
        - SAT2 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        

        ### Scenario  Description
        - Duration: 2 hours
        - Grid: One target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=5 seconds, lasting for 2 hours.
        - Mission objectives designed such that both agents have conflicting high-priority tasks.
        - Same instruments for both agents
        - Agents offset by 2 degrees in true anomaly

        ### Expected Outcomes
        - Both satellites have 2 observation windows for the event, making a total of 4 possible observations.
        - Agent 1's first observation window is earlier than Agent 2's first window.
        - Agent 1's second observation window is earlier than Agent 2's second window but after Agent 2's first window.
        - Agent 1 is expected to always out-bid Agent 2 for any observation opportunity due to mission objectives.
        - Agent 2 should lose bid to first and second observation in its first bidding round.
        - In the second bidding round, Agent 2 should be able to successfully bid for its second observation opportunity after Agent 1 has secured both of its observations.
        - The optimistic bidding mechanism should allow Agent 2 to plan for its second observation despite initial conflicts.
        - Final planner results should indicate 2 completed bids for Agent 1 and 2 completed bids for Agent 2, with an empty bundle at the end of the simulation.            
        """
        if not self.toy_4: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_4'
        scenario_name = f'toy_4-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_4'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_4"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_4"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_4"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_5(self):
        """
        ## TOY CASE 5
        Test case for two satellite performing an event-driven task from announcer with communication delays.

        ### Goals
        - Validate basic functionality of the bundle-building phase of the consensus planner in a communications-limited scenario.        
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection        
        - SAT2 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection     

        ### Scenario  Description
        - Duration: 6 hours
        - Grid: one target at (lat=0.0°, lon=0.0°)
        - Events: One event occurring at t=,6773.7318734 s, lasting for 200000 s.
        - Same instruments for both agents
        - Agents orbits offset in true anomaly by 60.0 degrees and in inclination by 60.0 degrees.
        - Communication between Agent 1 and announcer is constant.
        - Communication between Agents 1 and 2 only available when in LOS.

        ### Expected Outcomes
        - Agent 1 has earlier observation opportunities for both events due to orbit configuration.
        - Agent 1 is immediately informed of tasks from the announcer due to constant communication link with the announcer.
        - Agent 2 experiences delays in receiving task information due to intermittent communication link with Agent 
        - Agent 1 has already performed an observation before Agent 2 gets notified of the existance of any tasks.
        - Agent 2 should only be able to perform one observation due to observation constraints and communication delays.
        - Both satellites should choose to observe the same task as mission objectives benefit repeated observations.
        """        
        if not self.toy_5: return

        # setup scenario parameters
        duration = 6.0 / 24.0
        grid_name = 'toy_5'
        scenario_name = f'toy_5-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_5'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = -60.0 
        announcer_spacecraft['mission'] = "toy_mission_5"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -60.0 
        ractive_spacecraft_1['mission'] = "toy_mission_5"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 60.0
        ractive_spacecraft_2['mission'] = "toy_mission_5"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_6(self):
        """
        ## TOY CASE 6
        Test case for two satellite agents performing two event-driven tasks from announcer without default mission tasks.

        ### Goals
        - Showcase decision-making between conflicting tasks.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection
        
        ### Scenario Description
        - Duration: 2 hours
        - Grid: two targets at (lat=-5.0°, lon=0.0°) and (lat=5.0°, lon=0.0°)
        - Events: Two event occurring at t=TBD hours, lasting for 2 hours.
        - Same instruments for both agents
        - Agents orbits offset in true anomaly by 2.0 degrees.
        - Communication between agents is constant.

        ### Expected Outcomes
        - Agent 1 has earlier observation opportunities for both events due to orbit configuration.
        - Agents can only perform one task at a time due to observation constraints.
        - Both satellites should choose to observe the same task as mission objectives benefit repeated observations.        
        """
        if not self.toy_6: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_6'
        scenario_name = f'toy_6-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_6'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_6"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_1['mission'] = "toy_mission_6"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_6"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_7(self):
        """
        ## TOY CASE 7
        Test case for two satellite agents performing two event-driven tasks occurring at different times from announcer without default mission tasks.

        ### Goals
        - Showcase decision-making between conflicting tasks.
        
        ### Mission Details
        - Default objectives: None
        - Event-driven objectives: respond to event announcements from an announcer satellite.

        ### Agents
        - SAT0 : 
            - announcer satellite
            - no observation capability
            - onboard event-announcer planner
            - no onboard consensus planner
        - SAT1 : 
            - reactive satellite with narrow swath instrument
            - observation capability
            - onboard consensus planner
            - no onboard event-detection
        
        ### Scenario Description
        - Duration: 2 hours
        - Grid: two targets at (lat=-5.0°, lon=0.0°) and (lat=5.0°, lon=0.0°)
        - Events: Two event occurring at t=TBD hours, lasting for 2 hours.
        - Same instruments for both agents
        - Agents orbits offset in true anomaly by 2.0 degrees.
        - Communication between agents is constant.

        ### Expected Outcomes
        - Agent 1 has earlier observation opportunities for both events due to orbit configuration.
        - Agents 1 and 2 can only perform one task at a time due to observation constraints.
        - Both agents compete for the first event, with Agent 1 expected to perform the observation due to earlier access.
        - Agent 2 wins the bid for the second observation of the first event as it occurs later in time and thus better according to the mission objectives.
        - For the second event, Agent 1 is expected to perform the observation as Agent 2 has already scheduled an observation for the first event and cannot perform both due to observation constraints.
        """

        if not self.toy_7: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_7'
        scenario_name = f'toy_7-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_7'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = -1.0 # phase offset by 2.0[deg]
        announcer_spacecraft['mission'] = "toy_mission_7"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_1['mission'] = "toy_mission_7"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] \
            = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_7"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")
    
    def test_toy_case_8(self):
        """
        ## TOY CASE 8
        Test case for a single satellite performing multiple default mission tasks without events.

        """

        if not self.toy_8: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_8'
        scenario_name = f'toy_8-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_8'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_1['mission'] = "toy_mission_8"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_9(self):
        """
        ## TOY CASE 9
        Test case for multiple satellites performing multiple default mission tasks without events.

        """
        if not self.toy_9: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_9'
        scenario_name = f'toy_9-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_9'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_1['mission'] = "toy_mission_9"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_9"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")
        

    def test_toy_case_10(self):
        """
        ## TOY CASE 10
        Test case for single satellite performing two event-driven tasks with short expiration times from announcer with communication delays.       

        """
        if not self.toy_10: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_10'
        scenario_name = f'toy_10-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_10'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_10"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.0
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_10"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        results_summary = self.simulation.summarize_results()
        
        # verify results
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observable'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observed'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Requested'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Re-observed'].values[0][1], 0)

        # print done
        print(f"{scenario_name}: DONE")

    def test_toy_case_11(self):
        """
        ## TOY CASE 11
        Test case for multiple satellites performing two event-driven tasks with short expiration times from announcer with communication delays.       

        """
        if not self.toy_11: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_11'
        scenario_name = f'toy_11-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_11'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_11"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.0
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_11"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_11"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        results_summary = self.simulation.summarize_results()
        
        # verify results
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observable'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observed'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Requested'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Re-observed'].values[0][1], 2)

        # print done
        print(f"{scenario_name}: DONE")

    def test_toy_case_12(self):
        """
        ## TOY CASE 12
        Test case for a single satellite performing default mission tasks from a pre-planner without any event announcements.

        """
        if not self.toy_12: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_12'
        scenario_name = f'toy_12-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_12'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.0
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_12"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_13(self):
        """
        ## TOY CASE 13
        Test case for multiple satellites performing default mission tasks from a pre-planner without any event announcements.

        """
        if not self.toy_13: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_13'
        scenario_name = f'toy_13-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_13'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_13"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_13"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_14(self):
        """
        ## TOY CASE 14
        Test case for a single satellite reacting to event announcements from an announcer with an existing pre-planned schedule with infinite planning horizon.

        """
        if not self.toy_14: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_14'
        scenario_name = f'toy_14-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_14'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_14"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_14"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_15(self):
        """
        ## TOY CASE 15
        Test case for multiple satellites reacting to event announcements from an announcer with an existing pre-planned schedule with infinite planning horizon.

        """
        if not self.toy_15: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_15'
        scenario_name = f'toy_15-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_15'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_15"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_15"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_15"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_16(self):
        """
        ## TOY CASE 16
        Test case for a single satellite reacting to event announcements from an announcer but has the wrong instrument.

        """
        if not self.toy_16: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_16'
        scenario_name = f'toy_16-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_16'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_16"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = -2.0
        ractive_spacecraft_1['mission'] = "toy_mission_16"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_17(self):
        """
        ## TOY CASE 17
        Test case for multiple satellites with outdated information reacting to event announcements from an announcer.


        Targets:
        1) latitude,longitude = 0.0°, 0.0°

        Contacts:
        - Sat 0 and 1 contact: 0[s] - 7200.0[s]
        - Sat 0 and 2 contact: 0[s] - 7200.0[s]
        - Sat 0 and 3 contact: 0[s] - 7200.0[s]
        - Sat 1 and 2 contact:     None
        - Sat 1 and 3 contact: 2531.02[s] - 7200.0[s]
        - Sat 2 and 3 contact: 0[s] - 5187.22[s]
        - Sat 3 as relay: 2531.02[s] - 5187.22[s]
        
        Event 1 Accesses:
        - Sat 1 target 1 access 1: 33.67[s] - 34.27[s]
        - Sat 1 target 1 access 2: 5914.70[s] - 5915.32[s]
        - Sat 2 target 1 access 1: 755.61[s] - 756.19[s]
        - Sat 2 target 1 access 2: 6636.75[s] - 6637.[s]

        Timeline:
        - T:0[s] Sat 2 and 3 contact starts

        - T:0.001[s] Event 1 starts
        - T:0.001[s] Event 1 announced to Satellites 1, 2, and 3

        - T:0.001[s] Sat 1 determines it is winning n=1 for event 1 for t=33.67[s] 
        - T:0.001[s] Sat 1 determines it is winning n=2 for event 1 for t=5914.70[s] 

        - T:0.001[s] Sat 2 determines it is winning n=1 for event 1 for t=755.61[s]
        - T:0.001[s] Sat 2 determines it is winning n=2 for event 1 for t=6636.75[s]

        - T:0.001[s] Sat 2 informs Sat 3 that it is winning n=1 for event 1 for t=755.61[s]
        - T:0.001[s] Sat 2 informs Sat 3 that it is winning n=2 for event 1 for t=6636.75[s]

        - T:0.001[s] Sat 3 determines Sat 2 is winning n=1 for event 1 for t=755.61[s]
        - T:0.001[s] Sat 3 determines Sat 2 is winning n=2 for event 1 for t=6636.75[s]

        - T:33.67[s] Sat 1 performs observation n=1 of event 1

        - T:755.61[s] Sat 2 performs observation n=1 of event 1
        - T:755.61[s] Sat 2 determines it won n=1 for event 1 for t=755.61[s]

        - T:755.61[s] Sat 2 informs Sat 3 that it performed n=1 for event 1 for t=755.61[s]
        - T:755.61[s] Sat 3 determines that Sat 2 won n=1 for event 1 for t=755.61[s]
        - T:755.61[s] Sat 3 determines Sat 2 is winning n=2 for event 1 for t=6636.75[s]

        - T:2531.02[s] Sat 1 and 3 contact starts
        - T:2531.02[s] Sat 1 informs Sat 3 that it performed n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 informs Sat 1 that Sat 2 performed n=1 for event 1 for t=755.61[s]

        - T:2531.02[s] Sat 1 determines that it won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 1 determines that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 1 determines it is winning n=3 for event 1 for t=5914.70[s]

        - T:2531.02[s] Sat 3 determines that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 determines that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 3 determines that Sat 1 is winning n=3 for event 1 for t=5914.70[s]

        - T:2531.02[s] Sat 3 informs Sat 2 that Sat 1 performed n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 informs Sat 2 that Sat 2 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 informs Sat 2 that Sat 1 is winning n=3 for event 1 for t=5914.70[s]

        - T:2531.02[s] Sat 2 determines that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 2 determines that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 2 abandons n=2 for event 1 for t=6636.75[s]
        - T:2531.02[s] Sat 2 determines that Sat 1 is winning n=3 for event 1 for t=5914.70[s]
        - T:2531.02[s] Sat 2 determines it is winning n=4 for event 1 for t=6636.75[s]

        - T:2531.02[s] Sat 2 informs Sat 3 that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 2 informs Sat 3 that it won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 2 informs Sat 3 that it abandoned n=2 for event 1 for t=6636.75[s]
        - T:2531.02[s] Sat 2 informs Sat 3 that Sat 1 is winning n=3 for event 1 for t=5914.70[s]
        - T:2531.02[s] Sat 2 informs Sat 3 it is winning n=4 for event 1 for t=6636.75[s]

        - T:2531.02[s] Sat 3 determines that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 determines that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 3 determines that Sat 1 is winning n=3 for event 1 for t=5914.70[s]
        - T:2531.02[s] Sat 3 determines that Sat 2 is winning n=4 for event 1 for t=6636.75[s]

        - T:2531.02[s] Sat 3 informs Sat 1 that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 3 informs Sat 1 that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 3 informs Sat 1 that Sat 1 is winning n=3 for event 1 for t=5914.70[s]
        - T:2531.02[s] Sat 3 informs Sat 1 that Sat 2 is winning n=4 for event 1 for t=6636.75[s]

        - T:2531.02[s] Sat 1 determines that Sat 1 won n=1 for event 1 for t=33.67[s]
        - T:2531.02[s] Sat 1 determines that Sat 2 won n=2 for event 1 for t=755.61[s]
        - T:2531.02[s] Sat 1 determines that Sat 1 is winning n=3 for event 1 for t=5914.70[s]
        - T:2531.02[s] Sat 1 determines that Sat 2 is winning n=4 for event 1 for t=6636.75[s]

        - T:5187.22[s] Sat 2 and 3 contact ends

        - T:5914.70[s] Sat 1 performs observation n=3 of event 1
        - T:5914.70[s] Sat 1 informs Sat 3 that it performed n=3 for event 1 for t=5914.70[s]

        - T:5914.70[s] Sat 3 determines that Sat 1 won n=3 for event 1 for t=5914.70[s]
        - T:5914.70[s] Sat 3 informs Sat 1 that Sat 1 performed n=3 for event 1 for t=5914.70[s]
        - T:5914.70[s] Sat 3 informs Sat 1 that Sat 1 won n=3 for event 1 for t=5914.70[s]

        - T:6636.75[s] Sat 2 performs observation n=4 of event 1
        - T:6636.75[s] Sat 2 determines it won n=4 of event 1 at t=6636.75[s]

        - T:7200.00[s] Event 2 ends
        - T:7200.00[s] Sat 1 and 3 contact ends
        - T:7200.00[s] Simulation ends
        """
        if not self.toy_17: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_17'
        scenario_name = f'toy_17-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_17'
        mission_name = 'toy_missions'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        # ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['sma'] = self.R + 400.0 
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0 # outside of LOS of SAT2
        ractive_spacecraft_1['mission'] = "toy_mission_17"

        # SAT0_1 : announcer satellite 1
        announcer_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft_1['@id'] = 'sat0_announcer_1'
        announcer_spacecraft_1['name'] = 'sat0_1'
        announcer_spacecraft_1['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft_1['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft_1['orbitState']['state']['sma'] = self.R + 400.0 
        announcer_spacecraft_1['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft_1['orbitState']['state']['ta'] = 0.0 # in constant LOS of SAT1
        announcer_spacecraft_1['mission'] = "toy_mission_17"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        # ractive_spacecraft_2['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['sma'] = self.R + 400.0 
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = -45.0 # outside of LOS of SAT1
        ractive_spacecraft_2['mission'] = "toy_mission_17"

        # SAT0_2 : announcer satellite 2
        announcer_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft_2['@id'] = 'sat0_announcer_2'
        announcer_spacecraft_2['name'] = 'sat0_2'
        announcer_spacecraft_2['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft_2['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft_2['orbitState']['state']['sma'] = self.R + 400.0 
        announcer_spacecraft_2['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft_2['orbitState']['state']['ta'] = -45.0 # in constant LOS of SAT2
        announcer_spacecraft_2['mission'] = "toy_mission_17"

        # SAT3 : relay satellite with wrong instrument for event tasks
        ractive_spacecraft_3 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_3['@id'] = 'sat3_vnir'
        ractive_spacecraft_3['name'] = 'sat3'
        ractive_spacecraft_3['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_3['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_3['instrument'] = self.instruments['TIR'] # cannot perform event tasks
        ractive_spacecraft_3['orbitState']['state']['sma'] = self.R + 100.0 # lower orbit for faster revisit
        # ractive_spacecraft_3['orbitState']['state']['inc'] = -180.0 # opposite orbital direction so it can talk to SAT2 before SAT1
        ractive_spacecraft_3['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_3['orbitState']['state']['ta'] = -45.0 # start behind SAT2
        ractive_spacecraft_3['mission'] = "toy_mission_17"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft_1,
                                                       announcer_spacecraft_2,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2,
                                                       ractive_spacecraft_3
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_18(self):
        """
        ## TOY CASE 18
        Test case for multiple satellites in a string-of-pearls formation reacting to event 
          anouncements from an announcer with an existing pre-planned schedule with an infinite planning horizon.
          Middle satellite serves as relay between satellites. Does not perform observations

        """
        if not self.toy_18: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_18'
        scenario_name = f'toy_18-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_18'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['sma'] = self.R + 400.0 
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = 22.5 # in constant LOS of SAT1 
        announcer_spacecraft['mission'] = "toy_mission_18"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        # ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['sma'] = self.R + 400.0 
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0 # in constant LOS of SAT0 and SAT3 
        ractive_spacecraft_1['mission'] = "toy_mission_18"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        # ractive_spacecraft_2['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['sma'] = self.R + 400.0 
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = -45.0 # in constant LOS of SAT3
        ractive_spacecraft_2['mission'] = "toy_mission_18"

        # SAT3 : relay satellite with wrong instrument for event tasks
        ractive_spacecraft_3 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_3['@id'] = 'sat3_vnir'
        ractive_spacecraft_3['name'] = 'sat3'
        ractive_spacecraft_3['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_3['spacecraftBus']['components']['adcs']['maxRate'] = 0.8 # slower maneuverability
        ractive_spacecraft_3['instrument'] = self.instruments['TIR'] # cannot perform event tasks
        ractive_spacecraft_3['orbitState']['state']['sma'] = self.R + 400.0 
        ractive_spacecraft_3['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_3['orbitState']['state']['ta'] = -22.5 # in constant LOS of SAT1 and SAT2
        ractive_spacecraft_3['mission'] = "toy_mission_18"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2,
                                                       ractive_spacecraft_3
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_19(self):
        """
        ## TOY CASE 19
        Test case for a single satellite reacting to event announcements from an announcer with an existing pre-planned schedule with a short planning horizon.

        """
        if not self.toy_19: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_19'
        scenario_name = f'toy_19-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_19'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_19"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['planner']['preplanner']['period'] = 50 # fixed replanning period
        ractive_spacecraft_1['planner']['preplanner']['horizon'] = 500 # longer planning horizon
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_19"

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_20(self):
        """
        ## TOY CASE 20
        Test case for multiple satellites reacting to event announcements from an announcer with an existing pre-planned schedule with a short planning horizon.
        """

        if not self.toy_20: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_20'
        scenario_name = f'toy_20-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_20'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['mission'] = "toy_mission_20"

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_1['planner']['preplanner']['period'] = 50 # fixed replanning period
        ractive_spacecraft_1['planner']['preplanner']['horizon'] = 500 # longer planning horizon
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['mission'] = "toy_mission_20"

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_hollistic_planner_config()
        ractive_spacecraft_2['planner']['preplanner']['period'] = 50 # fixed replanning period
        ractive_spacecraft_2['planner']['preplanner']['horizon'] = 500 # longer planning horizon
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = -10.0 # phase offset by 10.0[deg]
        ractive_spacecraft_2['mission'] = "toy_mission_20"


        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_21(self):
        """
        ## TOY CASE 21
        Test case for a single satellite responding to event announcements from a ground station.
        
        Two events present. Both can be observed by the satellite but only one can be announced in time
        due to GS-satellite contact constraints. Only one event should be observed.
        """

        if not self.toy_21: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_21'
        scenario_name = f'toy_21-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_21'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_21'
        gs_network = 'gs_toy_21'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # compile ground stations and operators
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])
        scenario_specs['groundOperator'] = self.setup_announcer_ground_operators(event_name, mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        results_summary : pd.DataFrame = self.simulation.summarize_results()

        # verify results
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observable'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observed'].values[0][1], 1)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Requested'].values[0][1], 0)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Re-observed'].values[0][1], 0)

        # print done
        print(f"{scenario_name}: DONE")

    def test_toy_case_22(self):
        """
        ## TOY CASE 22
        Test case for multiple satellite responding to event announcements from a ground station.

        Two events present. Both can be observed by both satellites but only one can be announced in time
        due to GS-satellite contact constraints. The other is announced but agents cannot reach the task in time
        due to orbital constraints. Only one event should be observed twice.
        """

        if not self.toy_22: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_22'
        scenario_name = f'toy_22-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_22'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_22'
        gs_network = 'gs_toy_22'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = -10.0 # phase offset by 10.0[deg]
        ractive_spacecraft_2['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # compile ground stations and operators
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])
        scenario_specs['groundOperator'] = self.setup_announcer_ground_operators(event_name, mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()
        
        # print results
        results_summary : pd.DataFrame = self.simulation.summarize_results()

        # verify results
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observable'].values[0][1], 2)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Observed'].values[0][1], 1)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Requested'].values[0][1], 0)
        self.assertEqual(results_summary.loc[results_summary['Metric']=='Events Re-observed'].values[0][1], 1)

        # print done
        print(f"{scenario_name}: DONE")

    def test_toy_case_23(self):
        """
        ## TOY CASE 23
        Test case for a single satellite responding to event announcements from a announcer satelite
          while bidding against a ground operator agent.

        Contacts:
        - Sat 0 and 1 contact:     0.00[s] - 7200.00[s]
        - Sat 0 and GS contact: 5796.28[s] - 6262.09[s]
        - Sat 1 and GS contact: 5796.28[s] - 6262.09[s]

        Timeline
        - T:0.00[s] Simulation starts
        - T:0.00[s] Sat 0 and 1 contact starts
        - T:0.001[s] Event 1 starts
        - T:0.001[s] Sat 0 informs Sat 1 of Event 1
        - T:0.001[s] Sat 1 determines it is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] Sat 1 determines it is winning n=2 for Event 1 for t=6373.40[s]
        - T:0.001[s] Sat 1 informs Sat 0 that it is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] Sat 1 informs Sat 0 that it is winning n=2 for Event 1 for t=6373.40[s]

        - T:36.1[s] Sat 1 performs observation n=1 of Event 1
        - T:36.1[s] Sat 1 informs Sat 0 that it performed n=1 for Event 1 for t=36.1[s]
        
        - T:5796.28[s] Sat 0 and GS contact starts
        - T:5796.28[s] Sat 1 and GS contact starts
        - T:5796.28[s] Sat 0 informs GS of Event 1
        - T:5796.28[s] Sat 1 informs GS that it won n=1 for Event 1 for t=36.1[s]
        - T:5796.28[s] Sat 1 informs GS that it is winning n=2 for Event 1 for t=6373.4[s]
        - T:5796.28[s] GS determines Sat 1 won n=1 for Event 1 for t=36.1[s]
        - T:5796.28[s] GS determines Sat 1 is winning n=2 for Event 1 for t=6373.4[s]
        - T:5796.28[s] GS informs Sat 1 that Sat 1 won n=1 for Event 1 for t=36.1[s]
        - T:5796.28[s] GS informs Sat 1 that Sat 1 is winning n=2 for Event 1 for t=6373.4[s]

        - T:6000.0[s] Event 2 starts
        - T:6000.0[s] Sat 0 informs Sat 1 of Event 2
        - T:6000.0[s] Sat 1 determines it is not worth to perform observations for Event 2

        - T:6262.09[s] Sat 0 and GS contact ends
        - T:6262.09[s] Sat 1 and GS contact ends

        - T:6373.40[s] Sat 1 performs observation n=2 of Event 1 for t=6373.40[s]
        - T:6373.40[s] Sat 1 informs Sat 0 that it won n=2 for Event 1 for t=6373.40[s]

        - T:7200.0[s] Event 1 ends
        - T:7200.0[s] Event 2 ends
        - T:7200.0[s] Simulation ends
        """

        if not self.toy_23: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_23'
        scenario_name = f'toy_23-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_23'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_23'
        gs_network = 'gs_toy_23'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['groundStationNetwork'] = gs_network
        announcer_spacecraft['mission'] = mission_name

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # compile ground stations and operators
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])
        scenario_specs['groundOperator'] = self.setup_cbba_ground_operators(mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")
    
    def test_toy_case_24(self):
        """
        ## TOY CASE 24
        Test case for a two satellites responding to event announcements from a announcer satelite
          while bidding against a ground operator agent. Satellites can never see eachother, can only
          share information via the ground station.

        Events
        - Event 1:   0.001[s] - 7200.0[s]
        - Event 2:  6000.0[s] - 7200.0[s]
          
        Agent Contacts:
        - Sat 0 and 2 contact:            NAN
        - Sat 1 and 2 contact:            NAN
        - Sat 0 and 1 contact:     0.00[s] - 7200.00[s]
        - Sat 0 and GS contact:    0.00[s] - 96.41[s]
        - Sat 1 and GS contact:    0.00[s] - 96.41[s]
        - Sat 2 and GS contact:  669.70[s] - 1135.28[s]
        - Sat 0 and GS contact: 5969.01[s] - 6436.63[s]
        - Sat 1 and GS contact: 5969.01[s] - 6436.63[s]
        - Sat 2 and GS contact: 7007.65[s] - 7200.0[s]

        Coverage:
        - Sat 1 over Event 1: 36.17[s] - 37.26[s]
        - Sat 2 over Event 1: 1684.40[s] - 1685.53[s]     
        - Sat 1 over Event 1: 6373.40[s] - 6374.53[s]
        - Sat 1 over Event 2: 6373.40[s] - 6374.53[s]

        Timeline
        - T:0.00[s] Simulation starts
        - T:0.00[s] Sat 0 and 1 contact starts
        - T:0.00[s] Sat 0 and GS contact starts
        - T:0.00[s] Sat 1 and GS contact starts

        - T:0.001[s] Event 1 starts
        - T:0.001[s] Sat 0 informs Sat 1 of Event 1
        - T:0.001[s] Sat 1 determines it is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] Sat 1 determines it is winning n=2 for Event 1 for t=6373.40[s]
        - T:0.001[s] Sat 1 informs Sat 0 that it is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] Sat 1 informs Sat 0 that it is winning n=2 for Event 1 for t=6373.40[s]
        - T:0.001[s] Sat 1 informs GS that it is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] Sat 1 informs GS that it is winning n=2 for Event 1 for t=6373.40[s]
        - T:0.001[s] GS determines Sat 1 is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] GS determines Sat 1 is winning n=2 for Event 1 for t=6373.40[s]
        - T:0.001[s] GS informs Sat 0 that Sat 1 is winning n=1 for Event 1 for t=36.1[s]
        - T:0.001[s] GS informs Sat 1 that Sat 1 is winning n=2 for Event 1 for t=6373.40[s]

        - T:36.1[s] Sat 1 performs observation n=1 of Event 1
        - T:36.1[s] Sat 1 informs Sat 0 that it performed n=1 for Event 1 for t=36.1[s]
        - T:36.1[s] Sat 1 informs GS that it performed n=1 for Event 1 for t=36.1[s]
        - T:36.1[s] GS determines Sat 1 won n=1 for Event 1 for t=36.1[s]
        - T:36.1[s] GS informs Sat 0 that Sat 1 won n=1 for Event 1 for t=36.1[s]
        - T:36.1[s] GS informs Sat 1 that Sat 1 won n=1 for Event 1 for t=36.1[s]

        - T:96.41[s] Sat 0 and GS contact ends
        - T:96.41[s] Sat 1 and GS contact ends

        - T:669.70[s] Sat 2 and GS contact starts
        - T:669.70[s] GS informs Sat 2 that Sat 1 won n=1 for Event 1 for t=36.1[s]
        - T:669.70[s] GS informs Sat 2 that Sat 1 is winning n=2 for Event 1 for t=6373.40[s]
        - T:669.70[s] Sat 2 determines it is winning n=2 for Event 1 for t=1684.40[s]
        - T:669.70[s] Sat 2 informs GS that it is winning n=2 for Event 1 for t=1684.40[s]
        - T:669.70[s] GS determines Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        - T:669.70[s] GS informs Sat 2 that Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        
        - T:1135.28[s] Sat 2 and GS contact ends

        - T:1684.40[s] Sat 2 performs observation n=2 of Event 1 for t=1684.40[s]
        - T:1684.40[s] Sat 2 determines it won n=2 of Event 1 for t=1684.40[s]

        - T:5969.01[s] Sat 0 and GS contact starts
        - T:5969.01[s] Sat 1 and GS contact starts
        - T:5969.01[s] GS informs Sat 0 that Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        - T:5969.01[s] GS informs Sat 1 that Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        - T:5969.01[s] Sat 1 determines Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        - T:5969.01[s] Sat 1 determines it is winning n=3 for Event 1 for t=6373.40[s]
        - T:5969.01[s] Sat 1 informs GS that Sat 2 is winning n=2 for Event 1 for t=1684.40[s]
        - T:5969.01[s] Sat 1 informs GS that it is winning n=3 for Event 1 for t=6373.40[s]
        - T:5969.01[s] GS determines Sat 1 is winning n=3 for Event 1 for t=6373.40[s]
        
        - T:6000.0[s] Event 2 starts
        - T:6000.0[s] Sat 0 informs Sat 1 of Event 2
        - T:6000.0[s] Sat 1 determines it is not worth to perform observations for Event 2

        - T:6373.40[s] Sat 1 performs observation n=3 of Event 1 for t=6373.40[s]
        - T:6373.40[s] Sat 1 determines it won n=3 of Event 1 for t=6373.40[s]
        - T:6373.40[s] Sat 1 informs Sat 0 that it won n=3 for Event 1 for t=6373.40[s]
        - T:6373.40[s] Sat 1 informs GS that it won n=3 for Event 1 for t=6373.40[s]
        - T:6373.40[s] GS determines Sat 1 won n=3 for Event 1 for t=6373.40[s]
        - T:6373.40[s] GS informs Sat 0 that Sat 1 won n=3 for Event 1 for t=6373.40[s]
        - T:6373.40[s] GS informs Sat 1 that Sat 1 won n=3 for Event 1 for t=6373.40[s]

        - T:6436.63[s] Sat 0 and GS contact ends
        - T:6436.63[s] Sat 1 and GS contact ends

        - T:7007.65[s] Sat 2 and GS contact starts
        - T:7007.65[s] GS informs Sat 2 that Sat 1 won n=3 for Event 1 for t=6373.40[s]
        - T:7007.65[s] Sat 2 determines that Sat 1 won n=3 for Event 1 for t=6373.40[s]

        - T:7200.0[s] Event 1 ends
        - T:7200.0[s] Event 2 ends
        - T:7200[s] Sat 0 and 1 contact ends
        - T:7200[s] Sat 2 and GS contact ends
        - T:7200.0[s] Simulation ends       
       
        """

        if not self.toy_24: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_24'
        scenario_name = f'toy_24-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_24'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_24'
        gs_network = 'gs_toy_24'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = 0.0
        announcer_spacecraft['groundStationNetwork'] = gs_network
        announcer_spacecraft['mission'] = mission_name

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta']
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 60.0 # phase offset by 60.0[deg]
        ractive_spacecraft_2['groundStationNetwork'] = gs_network
        ractive_spacecraft_2['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # compile ground stations and operators
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])
        scenario_specs['groundOperator'] = self.setup_cbba_ground_operators(mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_25(self):
        """
        ## TOY CASE 25

        Test case for one satellite responding to event announcements from a ground operator agent
          while bidding against the same ground operator agent. 
        """

        if not self.toy_25: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        n_case = 25
        grid_name = f'toy_{n_case}'
        scenario_name = f'toy_{n_case}-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = f'toy_{n_case}'
        mission_filename = 'toy_missions'
        mission_name = f'toy_mission_{n_case}'
        gs_network = f'gs_toy_{n_case}'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 0.0
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       ractive_spacecraft_1
                                                    ]
                                                   )

        # compile ground stations
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])

        # define ground operator with CBBA + announcer capability
        scenario_specs['groundOperator'] = self.setup_announcing_cbba_ground_operators(event_name, mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_26(self):
        """
        ## TOY CASE 26

        Test case for two satellites responding to event announcements from a ground operator agent
          while bidding against the same ground operator agent. 
        """

        if not self.toy_26: return
        
        # setup scenario parameters
        duration = 2.0 / 24.0
        n_case = 26
        grid_name = f'toy_{n_case}'
        scenario_name = f'toy_{n_case}-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = f'toy_{n_case}'
        mission_filename = 'toy_missions'
        mission_name = f'toy_mission_{n_case}'
        gs_network = f'gs_toy_{n_case}'

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = 10.0
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 60.0 # phase offset by 60.0[deg]
        ractive_spacecraft_2['groundStationNetwork'] = gs_network
        ractive_spacecraft_2['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # compile ground stations
        scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])

        # define ground operator with CBBA + announcer capability
        scenario_specs['groundOperator'] = self.setup_announcing_cbba_ground_operators(event_name, mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_27(self):
        """
        ## TOY CASE 27
        Test case for a two satellites responding to event announcements from a satellite with onboard
        event-detection capabilities. Satellites fly in a string-of-pearls coniguration where only the 
        leading satellite can detect the event, but can share information to the other two satellites 
        via inter-satellite links. The reactive satellites can then bid on performing observations of 
        the event with a ground station operator.

        Events
        - Event 1:   0.001[s] - 7200.0[s]
        - Event 2:  6000.0[s] - 7200.0[s]
        """

        if not self.toy_27: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_27'
        scenario_name = f'toy_27-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_27'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_27'
        gs_network = 'gs_toy_27'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.toy_planner_config()
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = 0.0
        announcer_spacecraft['groundStationNetwork'] = gs_network
        announcer_spacecraft['mission'] = mission_name
        announcer_spacecraft['science'] = self.setup_science_config(event_name)

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 15.0 # phase offset
        ractive_spacecraft_1['groundStationNetwork'] = gs_network
        ractive_spacecraft_1['mission'] = mission_name

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 15.0 # phase offset
        ractive_spacecraft_2['groundStationNetwork'] = gs_network
        ractive_spacecraft_2['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )

        # # compile ground stations and operators
        # scenario_specs['groundStation'] = self.compile_ground_stations([gs_network])
        # scenario_specs['groundOperator'] = self.setup_cbba_ground_operators(mission_name, [gs_network])

        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()
        
        # summarize results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    def test_toy_case_28(self):
        """
        ## TOY CASE 28
        Sequence reset case

        Events
        - Event 1:   0.001[s] - 7200.0[s]
        - Event 2:  6000.0[s] - 7200.0[s]
        """

        if not self.toy_28: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_28'
        scenario_name = f'toy_28-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_28'
        mission_filename = 'toy_missions'
        mission_name = 'toy_mission_28'

        # SAT0 : announcer satellite 
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_announcer'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.setup_announcer_config(event_name)
        announcer_spacecraft.pop('instrument') # no instrument, cannot detect events
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['orbitState']['state']['ta'] = 0.0
        announcer_spacecraft['mission'] = mission_name
        announcer_spacecraft['science'] = self.setup_science_config(event_name)

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_1['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 15.0 # phase offset
        ractive_spacecraft_1['mission'] = mission_name

        # SAT2 : reactive satellite with narrow swath instrument
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config() # no preplan capability
        ractive_spacecraft_2['spacecraftBus']['components']['adcs']['maxRate'] = 1.5
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = ractive_spacecraft_1['orbitState']['state']['ta'] - 15.0 # phase offset
        ractive_spacecraft_2['mission'] = mission_name

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_filename,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                       ractive_spacecraft_1,
                                                       ractive_spacecraft_2
                                                    ]
                                                   )
        
        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # execute mission
        self.simulation.execute()
        
        # summarize results
        self.simulation.summarize_results()

        print(f"{scenario_name}: DONE")

    # def test_toy_case_2X(self):
    #     """
    #     ## TOY CASE 2X
    #     Test case for a single satellite responding to a plan generated from a ground station.
    #     """

    #     if not self.toy_2X: return

    # def test_toy_case_2X(self):
    #     """
    #     ## TOY CASE 2X
    #     Test case for multiple satellites responding to a plan generated from a ground station.
    #     """
    
    #     if not self.toy_2X: return

    # def test_toy_case_2X(self):
    #     """
    #     ## TOY CASE 2X
    #     Single satellite with default mission with event detection and response using toy scenario.
    #     """
    
    #     if not self.toy_2X: return

    # def test_toy_case_2X(self):
    #     """
    #     ## TOY CASE 2X
    #     Multiple satellites with default mission with event detection and response using toy scenario.
    #     """
    
    #     if not self.toy_2X: return

    def test_single_sat_lakes(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # check for case toggle 
        if not self.single_sat_lakes: return

        # setup scenario parameters
        # duration = 2.0 / 24.0
        # duration = 250.0 / 3600.0 / 24.0
        duration = 1.0
        grid_name = 'lake_event_points'
        scenario_name = f'single_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft['planner'] = self.lakes_planner_config()
        spacecraft['planner']['preplanner']['period'] = 250 # fixed replanning period
        # spacecraft['planner']['preplanner']['horizon'] = 500 # longer planning horizon
        spacecraft['mission'] = "Algal bloom comprehensive"
        spacecraft['science'] = self.setup_science_config(event_name)

        # terminal welcome message
        print_scenario_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[spacecraft]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.summarize_results()

        print('DONE')

    # def test_single_sat_announcer_toy(self):
        # # check for case toggle 
        # if not self.single_sat_announcer_toy: return

        # # setup scenario parameters
        # duration = 2.0 / 24.0
        # grid_name = 'toy_points'
        # scenario_name = f'single_sat_announcer_toy_scenario-{self.planner_name()}'
        # connectivity = 'LOS'
        # event_name = 'toy_events'
        # mission_name = 'toy_missions'

        # # SAT0 : announcer satellite 
        # announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        # announcer_spacecraft['@id'] = 'sat0_announcer'
        # announcer_spacecraft['name'] = 'sat0'
        # announcer_spacecraft['planner'] = self.setup_announcer_config(None)
        # announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        # announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        # announcer_spacecraft['mission'] = "Algal bloom response"

        # # SAT1 : reactive satellite with narrow swath instrument
        # ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        # ractive_spacecraft_1['@id'] = 'sat1_vnir'
        # ractive_spacecraft_1['name'] = 'sat1'
        # ractive_spacecraft_1['planner'] = self.toy_planner_config()
        # ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        # ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        # # ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]
        # ractive_spacecraft_1['mission'] = "Algal bloom response"

        # # if 'replanner' in announcer_spacecraft['planner']: announcer_spacecraft["planner"].pop('replanner') # make announcer purely preplanner

        # # terminal welcome message
        # print_welcome(f'`{scenario_name}` PLANNER TEST')

        # # Generate scenario
        # scenario_specs = self.setup_scenario_specs(duration,
        #                                            grid_name, 
        #                                            scenario_name, 
        #                                            connectivity,
        #                                            event_name,
        #                                            mission_name,
        #                                            spacecraft=[
        #                                                announcer_spacecraft,
        #                                                ractive_spacecraft_1
        #                                             ]
        #                                            )


        # # initialize mission
        # self.simulation : Simulation = Simulation.from_dict(scenario_specs, overwrite=True)

        # # execute mission
        # self.simulation.execute()

        # # print results
        # self.simulation.summarize_results()

        # print(f"{scenario_name}: DONE")

if __name__ == '__main__':
    # run tests
    unittest.main()