
from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx

from orbitpy.mission import Mission
from dmas.core.utils import print_scenario_banner

class Constellation(ABC):
    EARTH_RADIUS_KM = 6378 # Earth radius in km    
    GM = 3.986004418e14 * (1/1000)**3 # [km^3/s^2]

    MISSION_TEMPLATE = {
                    "epoch": {
                        "@type": "GREGORIAN_UT1",
                        "year": 2020,
                        "month": 1,
                        "day": 1,
                        "hour": 0,
                        "minute": 0,
                        "second": 0
                    },
                    "duration": 0.75,
                    "propagator": {
                        "@type": "J2 ANALYTICAL PROPAGATOR",
                        # "stepSize": 10
                    },
                    "spacecraft": [],
                    # "grid": [
                    #     {
                    #         "@type": "customGrid",
                    #         "covGridFilePath": "./grids/toy.csv"
                    #     }
                    # ],
                    "scenario" : {
                        "connectivity": "LOS"
                    },
                    "settings": {
                        "coverageType": "GRID COVERAGE",
                        "outDir" : "./orbits"
                    }
                }
    
    SPACECRAFT_TEMPLATE = {
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
                        }
                }
    
    def propagate(self, out_dir : str = None, propagation_period : float = None, debug : bool = False) -> str:
        """ Propagates the constellation and saves results to specified output directory. """

        # define output directory
        out_dir = self.default_out_dir() if out_dir is None else out_dir
        
        # create output directory
        os.makedirs(out_dir, exist_ok=True)

        # define data filename
        data_filename = os.path.join(out_dir, 'MissionSpecs.json')
        
        # convert to orbital parameters
        orbital_params : List[dict] = self.to_orbital_elements(debug)      

        # create satellite specification
        spacecraft = []
        for idx, params in enumerate(orbital_params):
            sat_spec = deepcopy(WalkerConstellation.SPACECRAFT_TEMPLATE)
            sat_spec['@id'] = f'sat_{idx+1}'
            sat_spec['name'] = f'sat_{idx+1}'
            sat_spec['orbitState']['state'] = params
            spacecraft.append(sat_spec)

        # calculate propagation period
        propagation_period = self.get_default_propagation_period() \
            if propagation_period is None else propagation_period

        # create mission specification
        mission_spec : Dict[str, dict] = deepcopy(WalkerConstellation.MISSION_TEMPLATE)
        mission_spec['duration'] = propagation_period 
        mission_spec['spacecraft'] = spacecraft
        mission_spec['settings']['outDir'] = out_dir

        if os.path.exists(data_filename):
            # load existing mission specifications
            existing_mission_spec : dict = json.load(open(data_filename,'r'))
            existing_mission_spec['propagator'].pop('stepSize', None)

            # compare with current specifications
            if existing_mission_spec == mission_spec: 
                print(f"Propagation already exists for `walker_delta_{self.num_sats}sat_{self.num_planes}planes_{self.phasing_param}phasing`.\n")
                return out_dir
            else:
                for key in mission_spec:
                    if existing_mission_spec[key] != mission_spec[key]:
                        print(f"Difference found in key: `{key}`")
                print(f"Existing propagation specifications differ from current specifications for `walker_delta_{self.num_sats}sat_{self.num_planes}planes`.\nRe-propagating...")
        else:
            print("Propagating orbits...")
        
        # create mission for propagation
        mission : Mission = Mission.from_json(mission_spec)  

        # propagate data and save to orbit data directory
        mission.execute(coverage_propagation=False, data_metrics_calculation=False,eclipse_finder=False)                
        print("Propagation done!")

        # delete state propagation from printed data for space savings
        for dir_name in os.listdir(out_dir):
            dir_path = os.path.join(out_dir,dir_name)
            if 'sat' in dir_name.lower() and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)            

        # remove step size from propagator for comparison purposes
        mission_spec['propagator'].pop('stepSize', None)

        # save specifications of propagation in the orbit data directory
        with open(data_filename, 'w') as mission_specs_file:
            mission_specs_file.write(json.dumps(mission_spec, indent=4))
        print(f"Saved mission specifications to \n   `{out_dir}`")

        # return output directory
        return out_dir     
    
    @abstractmethod
    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
    
    @abstractmethod
    def to_orbital_elements(self, debug: bool = False) -> List[dict]:
        """ Converts constellation specifications to orbital parameters."""

    @abstractmethod
    def get_default_propagation_period(self) -> float:
        """ Returns the default propagation period for the constellation in days. """

    def evaluate_connectivity(self, out_dir : str = None, overwrite : bool = False, debug : bool = False) -> Tuple[pd.DataFrame, dict]:
        """ Evaluates the connectivity of the constellation and saves results to specified output directory. """

        # define output directory
        out_dir = self.default_out_dir() if out_dir is None else out_dir

        # check if propagation data exists
        mission_specs_path = os.path.join(out_dir, 'MissionSpecs.json')
        if not os.path.exists(mission_specs_path):
            # propagation data does not exist, propagate constellation
            print(f"No propagation data found for constellation at: `{out_dir}`. Propagating constellation...\n")
            self.propagate(out_dir=out_dir, debug=debug)

        # get time-series conectivity metrics
        metrics_series_df : pd.DataFrame = self.__get_time_series_metrics(out_dir, overwrite, debug)

        ## largest connected component - LCC
        avg_largest_cc_norm = metrics_series_df['lcc [norm]'].mean()
                    
        max_largest_cc_norm = metrics_series_df['lcc [norm]'].max()
        max_largest_cc_norm_series = metrics_series_df[metrics_series_df["lcc [norm]"] == max_largest_cc_norm]

        max_largest_cc_norm_fraction = len(max_largest_cc_norm_series) / len(metrics_series_df)

        # connected components count
        avg_n_components = metrics_series_df['num components'].mean()

        # compile scalar metrics 
        scalar_metrics = {
                    'max lcc [norm]' : max_largest_cc_norm, 
                    'max lcc time-fraction [norm]' : max_largest_cc_norm_fraction, 
                    'avg lcc [norm]' : avg_largest_cc_norm, 
                    'avg num components' : avg_n_components
                }
        
        # return results
        return metrics_series_df, scalar_metrics

    def __get_time_series_metrics(self, data_dir : str, overwrite : bool, debug : bool) -> pd.DataFrame:
        # define path to orbit data directory
        data_dir = self.default_out_dir() if data_dir is None else data_dir

        # define path to save metrics
        metrics_path = os.path.join(data_dir, 'connectivity_series.csv')

        # check if file already exists
        if os.path.exists(metrics_path) and not overwrite:
            # file exists and no overwrite is required, skip computation
            print(f"Connectivity metrics file already exists at:\n   `{metrics_path}`.\nLoading existing metrics...")
            
            # load existing metrics
            metrics_series_df = pd.read_csv(metrics_path)
            
            # print metrics summary
            assert not metrics_series_df.empty, "Loaded connectivity metrics dataframe is empty!"
            if debug:
                print('Connectivity Metrics:')
                print(metrics_series_df.describe().iloc[1:].round(2))
                print(f"\nConnectivity metrics loaded from:\n   `{metrics_path}`\n")
        
        else:# file does not exist or require overwrite, compute metrics
            metrics_series_df = self.__generate_time_series_metrics(data_dir, debug)
        
        # return metrics dataframe
        return metrics_series_df

    def __generate_time_series_metrics(self, data_dir : str, debug : bool) -> pd.DataFrame:
        # define path to save metrics
        metrics_path = os.path.join(data_dir, 'connectivity_series.csv')
        
        # Load temporal graph and event times
        TG, times, T, time_step = self.__generate_access_graph_time_series(data_dir)
        
        # initialize metrics
        n_components_series = []
        n_components_fraction_series = []
        largest_cc_size_series = []
        largest_cc_norm_series = []

        # Evaluate constellation connectivity metrics
        for G in tqdm(TG, desc=f'Evaluating connectivity metrics', unit='time steps', leave=debug):
            # compute connectivity metrics
            connected_components = list(nx.connected_components(G))
            n_cc = nx.number_connected_components(G)
            n_cc_frac = n_cc / G.number_of_nodes()
            largest_cc_size = len(max(connected_components, key=len)) if n_cc > 0 else 0
            largest_cc_norm = largest_cc_size / G.number_of_nodes() 

            # store metrics
            n_components_series.append(n_cc)
            n_components_fraction_series.append(n_cc_frac)
            largest_cc_size_series.append(largest_cc_size)
            largest_cc_norm_series.append(largest_cc_norm)

        # get constellation parameters time series
        metrics_series_dict = self._get_constellation_params_time_series(times)

        # add connectivity metrics to dictionary
        metrics_series_dict.update({
            'time index' : times,
            'time [s]' : [t * time_step for t in times],
            'num components' : n_components_series,
            'n components [norm]' : n_components_fraction_series,
            'lcc' : largest_cc_size_series,
            'lcc [norm]' : largest_cc_norm_series
        })

        # compile to dataframe
        metrics_series_df = pd.DataFrame(metrics_series_dict)

        # save to csv
        metrics_series_df.to_csv(metrics_path, index=False)
        
        # print metrics summary
        if debug:
            print('Connectivity Metrics:')
            print(metrics_series_df.describe().iloc[1:].round(2))
            print(f"\nConnectivity metrics saved to: \n   `{metrics_path}`\n")

        return metrics_series_df
    
    def __generate_access_graph_time_series(self, data_dir : str) -> Tuple[List[nx.Graph],List[int],float,float]:
        # load access events and nodes
        events, nodes, T, time_step = self.__load_access_event_intervals(data_dir)

        # create linear time space
        t_start = 0
        t_end = int(T // time_step)
        times = list(range(t_start, t_end+1))

        # initiate list of graph snapshots    
        snapshot_graphs: list[nx.Graph] = []

        # create snapshot for each time step
        for t in times:
            # get active events at time t
            active = [(u,b,t_start,t_end) for u,b,t_start,t_end in events 
                    if t_start <= t <= t_end]

            # create graph snapshot
            G = nx.Graph()
            
            # specify snapshot time
            G.graph["time"] = float(t)

            # add nodes
            G.add_nodes_from(nodes)

            # add edges
            G.add_edges_from(zip([e[0] for e in active], [e[1] for e in active]))

            # append to snapshots list
            snapshot_graphs.append(G)

        # return list of graph snapshots and corresponding times with time step
        return snapshot_graphs, times, T, time_step

    def __load_access_event_intervals(self, data_dir : str) -> Tuple[list,list,float]:
        # initiale list of events for temporal graph
        events : List[tuple] = []

        # initiate set of nodes
        nodes = set()

        # define mission specification file path
        mission_specs_path = os.path.join(data_dir, 'MissionSpecs.json')

        # get propagation duration from mission specs
        with open(mission_specs_path, 'r') as mission_specs_file:
            mission_specs = json.load(mission_specs_file)
            propagation_duration_days = mission_specs['duration']
            propagation_duration_seconds = propagation_duration_days * 24 * 3600
            T = propagation_duration_seconds            

        # define path to comms data directory
        comms_data_dir = os.path.join(data_dir, 'comm')
        
        # initiate time step variable
        time_step = np.NAN

        # load comms data for every inter-satellite link
        for filename in tqdm(os.listdir(comms_data_dir), desc=f'Loading inter-satellite link data', leave=False):
            # parse filename
            isl_names = filename.split('.')[0]
            sat1, _, sat2 = isl_names.split('_')

            # add nodes to list
            nodes.update( {sat1, sat2} )
            
            # define full path to comms data file
            comms_data_path = os.path.join(comms_data_dir, filename)

            # read propagation time-step
            time_data =  pd.read_csv(comms_data_path, nrows=2)
            _, _, _, _, time_step = time_data.at[1,time_data.axes[1][0]].split(' ')
            time_step = float(time_step)

            # load communications data
            df : pd.DataFrame = pd.read_csv(comms_data_path, skiprows=range(3))

            # skip if dataframe is empty
            if df.empty: continue

            # add edges to temporal graph
            for t_start,t_end in df.values:
                # convert to integer time-steps
                t_start = int(t_start)
                t_end = int(t_end)

                # add contact as an event to list 
                events.append( (sat1, sat2, t_start, t_end) )
        
        # sort events by start time
        events.sort(key=lambda e: e[2])
        
        # return contact events, list of nodes, total propagation time, and time step
        return events, list(nodes), T, time_step 
    
    @abstractmethod
    def _get_constellation_params_time_series(self, times: List[int]) -> Dict[str, List]:
        """ Returns a dictionary of constellation parameters as they evolve over time. """       

class WalkerConstellation(Constellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes an abstract Walker constellation."""
        # validate inputs
        assert isinstance(alt, float) and alt >= 0.0, "Altitude must be a non-negative float"
        assert isinstance(inc, float), "Inclination must be a float"
        assert isinstance(num_sats, int) and num_sats > 0, "Number of satellites must be a positive integer"
        assert isinstance(num_planes, int) and num_planes > 0, "Number of planes must be a positive integer"
        assert isinstance(phasing_param, int) and 0 <= phasing_param, "Phasing parameter must be a positive integer"

        # calculate orbital period
        self.T = 2 * np.pi * np.sqrt( (Constellation.EARTH_RADIUS_KM + alt)**3 / Constellation.GM )

        # Store input parameters
        self.alt = alt
        self.inc = inc
        self.num_sats = num_sats
        self.num_planes = num_planes
        self.phasing_param = phasing_param

    def get_default_propagation_period(self):
        return self.T / 2 / 3600.0 / 24.0 # propagate for half an orbit by default

    def to_orbital_elements(self, debug: bool = False) -> List[dict]:
        """
        Converts constellation specifications to a list of orbital parameters.

        - `i` is the inclination
        - `t` is the total number of satellites;
        - `p` is the number of equally spaced planes; 
        - `f` is the relative spacing between satellites in adjacent planes. 
                The change in true anomaly (in degrees) for equivalent satellites 
                in neighbouring planes is equal to f * 360° / t.
        """

        # initialize list of orbital parameters
        orbital_params = []

        # calculate number of satellites per plane 
        min_sats_per_plane = self.num_sats // self.num_planes
        remaining_sats_to_assign = self.num_sats % self.num_planes

        # calculate RAAN spacing and initial phasing
        raan_spacing = self.calc_raan_spacing()
        aop_phasing = self.phasing_param * (360.0 / self.num_sats)

        # generate orbital parameters for every satellite in every plane
        for plane_idx in range(self.num_planes):
            # calculate plane RAAN and initial aop
            raan = plane_idx * raan_spacing
            aop = plane_idx * aop_phasing 

            # calculate sats per plane (distribute remaining sats)
            sats_per_plane = min_sats_per_plane 
            sats_per_plane += 1 if plane_idx < remaining_sats_to_assign else 0

            # calculate intra-plane true anomaly spacing
            ta_spacing = 360.0 / sats_per_plane

            # generate orbital parameters for each satellite in plane
            for sat_idx in range(sats_per_plane):
                ta = sat_idx * ta_spacing

                params = {
                    "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                    "sma": WalkerConstellation.EARTH_RADIUS_KM + self.alt,
                    "ecc": 0.001, # assume nearly circular orbits
                    "inc": self.inc,
                    "raan": raan,
                    "aop": aop,
                    "ta": ta
                }
                orbital_params.append(params)

        # print results for debugging
        if debug:
            print("="*80 + "\n")
            print(f"Walker Delta Constellation - i:t/p/f = {round(self.inc,2)}°:{self.num_sats}/{self.num_planes}/{self.phasing_param}")
            print("-"*82)
            prev_param = None
            for idx, params in enumerate(orbital_params):
                if prev_param and prev_param['raan'] != params['raan']:
                    print("  " + "-"*80)
                print(f"  Sat-{idx+1}:\tinc={params['inc']}°\traan={round(params['raan'],3)}°\taop={round(params['aop'],3)}°\tta={round(params['ta'],3)}°")
                prev_param = params
            print("="*80 + "\n")     

        # return list of orbital parameters
        return orbital_params
    
    @abstractmethod
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
    
    def _get_constellation_params_time_series(self, times: List[int]) -> Dict[str, List]:
        return {
            "inc [deg]" : [self.inc] * len(times),
            "num sats" : [self.num_sats] * len(times),
            "num planes" : [self.num_planes] * len(times),
            "num planes [norm]" : [self.num_planes / self.num_sats] * len(times),
            "ta phasing param" : [self.phasing_param] * len(times),
            "ta phasing angle [deg]" : [self.phasing_param * (360.0 / self.num_sats)] * len(times)
        }
    
    def __str__(self) -> str:
        return f"Walker Constellation: i:t/p/f = {round(self.inc,2)}°:{self.num_sats}/{self.num_planes}/{self.phasing_param}"
     
class WalkerDeltaConstellation(WalkerConstellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes a Walker Delta constellation."""
        super().__init__(alt, inc, num_sats, num_planes, phasing_param)

    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
        return f'./orbits/walker-delta_{self.inc}inc_{self.num_sats}sat_{self.num_planes}pl_{self.phasing_param}f'
       
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
        return 360.0 / self.num_planes
    
class WalkerStarConstellation(WalkerConstellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes a Walker Star constellation."""
        super().__init__(alt, inc, num_sats, num_planes, phasing_param)
    
    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
        return f'./orbits/walker-star_{self.inc}inc_{self.num_sats}sat_{self.num_planes}pl_{self.phasing_param}f'
       
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
        return 180.0 / self.num_planes

if __name__ == "__main__":

    # terminal welcome message
    print_scenario_banner(f'Walker Delta Constellation Example')
    
    # example 1: Walker Delta Constellation
    
    # define constellation
    walker = WalkerDeltaConstellation(
        alt = 550.0,
        inc = 98.0,
        num_sats = 8,
        num_planes = 3,
        phasing_param = 1
    )

    # propagate constellation 
    walker_data = walker.propagate(debug=True)

    # evaluate connectivity
    walker_metrics_df, walker_scalar_metrics = walker.evaluate_connectivity(debug=True, overwrite=True)

    # print scalar metrics
    print("Scalar Connectivity Metrics:")
    for metric_name, metric_value in walker_scalar_metrics.items():
        print(f"  {metric_name} : {round(metric_value,3)}")
    print("\n")

    # example 2: Walker Star Constellation

    # define constellation
    walker_star = WalkerStarConstellation(
        alt = 550.0,
        inc = 98.0,
        num_sats = 8,
        num_planes = 4,
        phasing_param = 1
    )

    # evauate constellation connectivity without pre-propagation
    walker_star_metrics_df, walker_star_scalar_metrics = walker_star.evaluate_connectivity(debug=True, overwrite=True)

    # print scalar metrics
    print("Scalar Connectivity Metrics:")
    for metric_name, metric_value in walker_star_scalar_metrics.items():
        print(f"  {metric_name} : {round(metric_value,3)}")
    print("\n")