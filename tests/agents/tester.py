from abc import ABC, abstractmethod
import os
from typing import List
import pandas as pd


class AgentTester(ABC):
    def setUp(self):
        # set outdir
        orbitdata_dir = os.path.join('./tests/agents', 'orbit_data')
        if not os.path.isdir(orbitdata_dir): os.mkdir(orbitdata_dir)

        # define template spacecraft
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
                            "sma": 7078,
                            "ecc": 0.0,
                            "inc": 60.0,
                            "raan": 0.0,
                            "aop": 0.0,
                            "ta": 95.0
                        }
                    },
                    "planner" : {
                        # "preplanner" : {
                        #     "@type" : "earliest",
                        #     "period": 500,
                        #     # "horizon": 500,
                        # },
                        # "replanner" : {
                        #     "@type" : "broadcaster",
                        #     "period" : 400
                        # },
                    },
                    # "science" : {
                    #     "@type": "lookup", 
                    #     "eventsPath" : "./tests/agents/resources/events/toy_events.csv"
                    # },
                    "mission" : "Algal blooms monitoring"
            }

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
                                    "angleHeight": 2.5, 
                                    "angleWidth": 5.0
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
        
            
    def setup_scenario_specs(self, 
                             duration : float, 
                             coverage_grid_name : str, 
                             scenario_name : str, 
                             connectivity : str, 
                             event_name : str, 
                             mission_name : str,
                             gs_network_names : List[str] = [],
                             spacecraft : List[dict] = []
                             ) -> dict:
        # validate inputs
        assert all(isinstance(gs_network_name, str) for gs_network_name in gs_network_names), "All gs_network_names must be strings."
        assert all(isinstance(sat, dict) for sat in spacecraft), "All spacecraft must be dictionaries."

        # construct scenario specifications
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
        scenario_specs['grid'] = self.setup_coverage_grid(coverage_grid_name)
        scenario_specs['scenario'] = self.setup_scenario(scenario_name, connectivity, event_name, mission_name)
        scenario_specs['settings'] = self.setup_scenario_settings(scenario_name)
        scenario_specs['spacecraft'] = spacecraft
        scenario_specs['groundStation'] = self.compile_ground_stations(gs_network_names)
                
        if gs_network_names is not None:
            scenario_specs['groundOperator'] = self.setup_ground_operators(gs_network_names, spacecraft)

        # return scenario specifications
        return scenario_specs

    def setup_coverage_grid(self, grid_name : str) -> dict:
        """Setup the grid for the scenario. """

        assert isinstance(grid_name, str), "grid_name must be a string"

        assert os.path.isfile(f"./tests/agents/resources/grids/{grid_name}.csv"), \
            f"Grid file not found: {grid_name}.csv"

        grid = {
            "@type": "customGrid",
            "covGridFilePath": f"./tests/agents/resources/grids/{grid_name}.csv"
        }
        return [grid]

    def setup_scenario(self, scenario_name : str, connectivity : str, event_name : str, mission_name : str) -> dict:
        """Setup the scenario for the simulation. """

        assert isinstance(scenario_name, str), "scenario_name must be a string"
        assert isinstance(connectivity, str), "connectivity must be a string"
        assert isinstance(event_name, str), "event_name must be a string"
        assert isinstance(mission_name, str), "mission_name must be a string"

        assert os.path.isfile(f"./tests/agents/resources/events/{event_name}.csv"), \
            f"Event file not found: {event_name}.csv"
        assert os.path.isfile(f"./tests/agents/resources/missions/{mission_name}.json"), \
            f"Mission file not found: {mission_name}.json"

        scenario = {
            "connectivity": connectivity,
            "events": {
                "@type": "PREDEF",
                "eventsPath": f"./tests/agents/resources/events/{event_name}.csv"
            },
            "clock" : {
                "@type" : "EVENT"
            },
            "scenarioPath" : "./tests/agents/",
            "name" : scenario_name,
            "missionsPath" : f"./tests/agents/resources/missions/{mission_name}.json"
        }
        return scenario
    
    def setup_scenario_settings(self, full_scenario_name : str) -> dict:
        """ Setup additional scenario settings for orbitpy propagator. """

        assert isinstance(full_scenario_name, str), "scenario_name must be a string"
        assert os.path.isdir(f"./tests/agents/orbit_data"), \
            f"Orbit data directory not found."
        
        # extract relevant scenario name
        scenario_name,*_ = full_scenario_name.split('-')

        # create orbitdata output directory if needed
        scenario_orbitdata_dir = f"./tests/agents/orbit_data/{scenario_name}"
        if not os.path.isdir(scenario_orbitdata_dir): os.mkdir(scenario_orbitdata_dir)

        # create orbitdata settings dictionary
        settings = {
                "coverageType": "GRID COVERAGE",
                "outDir" : f"./tests/agents/orbit_data/{scenario_name}",
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

        grid_path = f"./tests/agents/resources/gstations/{gs_network_name}.csv"
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
    def setup_ground_operators(self, gs_network_names : List[str], spacecraft : List[dict]) -> List[dict]:
        """ Setup ground operations for the scenario. """