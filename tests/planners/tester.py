from abc import ABC, abstractmethod
from collections import defaultdict
import os
import copy
from typing import List

import pandas as pd

from chess3d.simulation import Simulation
from chess3d.utils import print_banner

class PlannerTester(ABC):
    R = 6357.0 # Radius of the Earth [km]

    def setUp(self) -> None:        
        # test case toggles
        self.single_sat_toy : bool = False
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = False
        self.multiple_sat_lakes : bool = False

        # load scenario json file
        self.spacecraft_template = {
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
                            "angleWidth": 45.0
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
                            "sma": 7078, # ~700 km altitude
                            "ecc": 0.01,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 98.0,
                            "ta": 0.0
                        }
                    },
                    "planner" : {
                        
                    },
                    # "science" : {
                    #     "@type": "lookup", 
                    #     "eventsPath" : "./tests/planners/resources/events/toy_events.csv"
                    # },
                    "mission" : "Algal bloom comprehensive"
            }
        
        # set outdir
        orbitdata_dir = os.path.join('./tests/planners', 'orbit_data')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)

        # define known list of instruments
        self.instruments = {
                            "VNIR hyp" : {
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
                                    "angleHeight": 0.5, 
                                    "angleWidth": 0.5
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Hyperspectral"
                            },
                            "VNIR multi" : {
                                "name": "VNIR multi",
                                "@id" : "vnir_multi_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 0.5, 
                                    "angleWidth": 0.5
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Multispectral"
                            },
                            "TIR": {
                                "name": "TIR",
                                "@id" : "tir_imager",
                                "@type" : "VNIR",
                                "detectorWidth": 6.6e-6,
                                "focalLength": 3.6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 0.5, 
                                    "angleWidth": 20.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                },
                                "spectral_resolution" : "Multispectral"
                            },
                            "Altimeter": {
                                "name": "Altimeter",
                                "@id" : "altimeter",
                                "@type" : "Altimeter",
                                "chirpBandwidth": 150e6,
                                "pulseWidth": 50e-6,  
                                "orientation": {
                                    "referenceFrame": "NADIR_POINTING",
                                    "convention": "REF_FRAME_ALIGNED"
                                },
                                "fieldOfViewGeometry": { 
                                    "shape": "RECTANGULAR", 
                                    "angleHeight": 2.5, 
                                    "angleWidth": 45.0
                                },
                                "maneuver" : {
                                    "maneuverType":"SINGLE_ROLL_ONLY",
                                    "A_rollMin": -50,
                                    "A_rollMax": 50
                                }
                            }
                        }
        
    def setup_science_config(self, event_name : str) -> dict:
        """ Setup science configuration for the scenario. """

        assert isinstance(event_name, str), "event_name must be a string"

        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        
        return {
                    "@type": "lookup", 
                    f"eventsPath" : f"./tests/planners/resources/events/{event_name}.csv"
                }
        
    def setup_scenario_specs(self, 
                             duration : float, 
                             grid_name : str, 
                             scenario_name : str, 
                             connectivity : str, 
                             event_name : str, 
                             mission_name : str,
                             spacecraft : list = []
                             ) -> dict:
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
            "duration": duration,
            "propagator": {
                "@type": "J2 ANALYTICAL PROPAGATOR",
            }
        }
        scenario_specs['grid'] = self.setup_grid(grid_name)
        scenario_specs['scenario'] = self.setup_scenario(scenario_name, connectivity, event_name, mission_name)
        scenario_specs['settings'] = self.setup_scenario_settings(scenario_name)
        scenario_specs['spacecraft'] = spacecraft

        return scenario_specs

    def setup_grid(self, grid_name : str) -> dict:
        """Setup the grid for the scenario. """

        assert isinstance(grid_name, str), "grid_name must be a string"

        assert os.path.isfile(f"./tests/planners/resources/grids/{grid_name}.csv"), \
            f"Grid file not found: {grid_name}.csv"

        grid = {
            "@type": "customGrid",
            "covGridFilePath": f"./tests/planners/resources/grids/{grid_name}.csv"
        }
        return [grid]

    def setup_scenario(self, scenario_name : str, connectivity : str, event_name : str, mission_name : str) -> dict:
        """Setup the scenario for the simulation. """

        assert isinstance(scenario_name, str), "scenario_name must be a string"
        assert isinstance(connectivity, str), "connectivity must be a string"
        assert isinstance(event_name, str), "event_name must be a string"
        assert isinstance(mission_name, str), "mission_name must be a string"

        assert os.path.isfile(f"./tests/planners/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        assert os.path.isfile(f"./tests/planners/resources/missions/{mission_name}.json"), \
            f"Mission file not found: {mission_name}.json"

        scenario = {
            "connectivity": connectivity,
            "events": {
                "@type": "PREDEF",
                "eventsPath": f"./tests/planners/resources/events/{event_name}.csv"
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "scenarioPath" : "./tests/planners/",
            "name" : scenario_name,
            "missionsPath" : f"./tests/planners/resources/missions/{mission_name}.json"
        }
        return scenario
    
    def setup_scenario_settings(self, full_scenario_name : str) -> dict:
        """ Setup additional scenario settings for orbitpy propagator. """

        assert isinstance(full_scenario_name, str), "scenario_name must be a string"
        assert os.path.isdir(f"./tests/planners/orbit_data"), \
            f"Orbit data directory not found."
        
        # extract relevant scenario name
        scenario_name,*_ = full_scenario_name.split('-')

        # create orbitdata output directory if needed
        scenario_orbitdata_dir = f"./tests/planners/orbit_data/{scenario_name}"
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # create orbitdata settings dictionary
        settings = {
                "coverageType": "GRID COVERAGE",
                "outDir" : f"./tests/planners/orbit_data/{scenario_name}",
            }
        return settings
    
    def compile_ground_stations(self, gs_network_names : List[str] = []) -> List[dict]:
        """Compile ground stations for the scenario. """
        # collect all ground stations from specified networks
        ground_stations = {gs_network_name : self.load_ground_stations(gs_network_name) for gs_network_name in gs_network_names}

        # add network name to each ground station specifications
        for gs_network_name,network in ground_stations.items():
            for gs in network:
                gs['networkName'] = gs_network_name

        # flatten list of lists
        return [ground_station for network in ground_stations.values() for ground_station in network]

    def load_ground_stations(self, gs_network_name : str = None) -> List[dict]:
        if gs_network_name is None: return []

        grid_path = f"./tests/planners/resources/gstations/{gs_network_name}.csv"
        assert os.path.isfile(grid_path), f"Ground station file not found: {gs_network_name}.csv"

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
                "@id": gs_df['@id'] if '@id' in gs_df else f'{gs_network_name}-{gs_idx}'
            }
            gs_network.append(gs)

        # return ground station network as list of dicts
        return gs_network
    
    @abstractmethod
    def toy_planner_config(self) -> dict:
        """ Returns the planner configuration for the toy test cases. """

    @abstractmethod
    def lakes_planner_config(self) -> dict:
        """ Returns the planner configuration for the lakes test cases. """

    @abstractmethod
    def planner_name(self) -> str:
        """ Returns the planner name for the test case. """

    def test_single_sat_toy(self):
        """ Test case for a single satellite with toy events. """
        # check for case toggle 
        if not self.single_sat_toy: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'single_sat_toy_scenario-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite with wide swath instrument
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_tir'
        announcer_spacecraft['name'] = 'SAT0'
        announcer_spacecraft['planner'] = self.toy_planner_config()
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['science'] = self.setup_science_config(event_name)
        # if 'replanner' in announcer_spacecraft['planner']: announcer_spacecraft["planner"].pop('replanner') # make announcer purely preplanner

        # terminal welcome message
        print_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       announcer_spacecraft,
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.process_results()

        print('DONE')

    def test_multiple_sats_toy(self):
        """ Test case for multiple satellites with toy events. """
        # check for case toggle 
        if not self.multiple_sat_toy: return

        # setup scenario parameters
        duration = 1.0 / 24.0
        grid_name = 'toy_points'
        scenario_name = f'multiple_sat_toy_scenario-{self.planner_name()}'
        connectivity = 'LOS'
        event_name = 'toy_events'
        mission_name = 'toy_missions'

        # SAT0 : announcer satellite with wide swath instrument
        announcer_spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        announcer_spacecraft['@id'] = 'sat0_tir'
        announcer_spacecraft['name'] = 'sat0'
        announcer_spacecraft['planner'] = self.toy_planner_config()
        announcer_spacecraft['instrument'] = self.instruments['TIR'] # wide swath instrument
        announcer_spacecraft['orbitState']['state']['inc'] = 0.0
        announcer_spacecraft['science'] = self.setup_science_config(event_name)
        # if 'replanner' in announcer_spacecraft['planner']: announcer_spacecraft["planner"].pop('replanner') # make announcer purely preplanner

        # SAT1 : reactive satellite with narrow swath instrument
        ractive_spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_1['@id'] = 'sat1_vnir'
        ractive_spacecraft_1['name'] = 'sat1'
        ractive_spacecraft_1['planner'] = self.toy_planner_config()
        ractive_spacecraft_1['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_1['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_1['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 2.0 # phase offset by 2.0[deg]

        # SAT2 : reactive satellite with narrow swath instrument and lagging behind announcer
        ractive_spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        ractive_spacecraft_2['@id'] = 'sat2_vnir'
        ractive_spacecraft_2['name'] = 'sat2'
        ractive_spacecraft_2['planner'] = self.toy_planner_config()
        ractive_spacecraft_2['instrument'] = self.instruments['VNIR hyp'] # narrow swath instrument
        ractive_spacecraft_2['orbitState']['state']['inc'] = 0.0
        ractive_spacecraft_2['orbitState']['state']['ta'] = announcer_spacecraft['orbitState']['state']['ta'] - 2.5 # phase offset by 2.5[deg]

        # terminal welcome message
        print_banner(f'`{scenario_name}` PLANNER TEST')

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
                                                    #    ractive_spacecraft_2
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.process_results()

        print('DONE')

    def test_single_sat_lakes(self):
        """ Test case for a single satellite in a lake-monitoring scenario. """
        # check for case toggle 
        if not self.single_sat_lakes: return

        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = f'single_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        spacecraft : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft['planner'] = self.lakes_planner_config()

        # terminal welcome message
        print_banner(f'`{scenario_name}` PLANNER TEST')

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
        self.simulation.process_results()

        print('DONE')

    def test_multiple_sats_lakes(self):
        """ Test case for multiple satellites in a lake-monitoring scenario. """
        # check for case toggle 
        if not self.multiple_sat_lakes: return
        
        # setup scenario parameters
        duration = 2.0 / 24.0
        grid_name = 'lake_event_points'
        scenario_name = f'multiple_sat_lake_scenario-{self.planner_name()}'
        connectivity = 'FULL'
        event_name = 'lake_events_seed-1000'
        mission_name = 'lake_missions'

        spacecraft_1 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_1['planner'] = self.lakes_planner_config()
        spacecraft_1['@id'] = 'sat_1'
        spacecraft_1['name'] = 'sat_1'

        spacecraft_2 : dict = copy.deepcopy(self.spacecraft_template)
        spacecraft_2['planner'] = self.lakes_planner_config()
        spacecraft_2['orbitState']['state']['ta'] = 90.0
        spacecraft_2['@id'] = 'sat_2'
        spacecraft_2['name'] = 'sat_2'

        # terminal welcome message
        print_banner(f'`{scenario_name}` PLANNER TEST')

        # Generate scenario
        scenario_specs = self.setup_scenario_specs(duration,
                                                   grid_name, 
                                                   scenario_name, 
                                                   connectivity,
                                                   event_name,
                                                   mission_name,
                                                   spacecraft=[
                                                       spacecraft_1, 
                                                       spacecraft_2
                                                    ]
                                                   )


        # initialize mission
        self.simulation : Simulation = Simulation.from_dict(scenario_specs)

        # execute mission
        self.simulation.execute()

        # print results
        self.simulation.process_results()

        print('DONE')

