
from datetime import timedelta
from enum import Enum
import gc
import json
import logging
import os
import shutil
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from instrupy.base import Instrument
from orbitpy.util import Spacecraft, dictionary_list_to_object_list

from execsatm.mission import Mission
from execsatm.events import GeophysicalEvent

from dmas.core.messages import SimulationRoles
from dmas.utils.orbitdata import OrbitData
from dmas.models.actions import AgentAction
from dmas.models.agent import SimulationAgent
from dmas.core.environment import SimulationEnvironment
from dmas.models.states import GroundOperatorAgentState, SatelliteAgentState, SimulationAgentState
from dmas.models.planning.centralized.dealer import TestingDealer
from dmas.models.planning.centralized.milp import DealerMILPPlanner
from dmas.models.planning.centralized.worker import WorkerPlanner
from dmas.models.planning.decentralized.announcer import EventAnnouncerPlanner
from dmas.models.planning.decentralized.blank import BlankPlanner
from dmas.models.planning.decentralized.consensus.heuristic import HeuristicInsertionConsensusPlanner
from dmas.models.planning.decentralized.dynamic import DynamicProgrammingPlanner
from dmas.models.planning.decentralized.earliest import EarliestAccessPlanner
from dmas.models.planning.decentralized.heuristic import HeuristicInsertionPeriodicPlanner
from dmas.models.planning.decentralized.nadir import NadirPointingPlanner
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.processing import ObservationDataProcessor, LookupProcessor
from dmas.utils.processing import ResultsProcessor
from dmas.utils.tools import SimulationRoles

class SimulationClockType(Enum):
    # TODO add support for fixed time-step simulations
    FIXED_TIME_STEP = 1
    EVENT_DRIVEN = 2

class Simulation:
    def __init__(self,
                 name : str,
                 duration : float,
                 results_path : str,
                 orbitdata : Dict[str, OrbitData],
                 missions : Dict[str, Mission],
                 events : List[GeophysicalEvent],
                 environment : SimulationEnvironment,
                 agents : List[SimulationAgent],
                 level : int,
                 time_step : float = None,
                 printouts : bool = True
            ) -> None:
        """ 
        Initializes simulation instance 
        
        ### Arguments
        - name : str
            Name of the scenario being simulated
        - duration : float
            Total duration of the simulation in [days]
        - results_path : str
            Path to store simulation results
        - orbitdata : Dict[str, OrbitData]
            Dictionary of OrbitData objects for each agent
        - missions : Dict[str, Mission]
            Dictionary of Mission objects for each agent
        - environment : SimulationEnvironment   
            Simulation environment instance
        - agents : List[SimulationAgent]
            List of SimulationAgent instances
        - level : int
            Logging level
        - time_step : float, optional
            Fixed time step for the simulation (if None, event-driven simulation is used)

        """
        # validate inputs
        assert isinstance(name, str), "Simulation name must be a string."
        assert isinstance(duration, (int, float)) and duration > 0, "Duration must be a positive number."
        assert isinstance(results_path, str), "Results path must be a string."
        assert isinstance(missions, dict), "Missions must be a dictionary."
        assert all(isinstance(mission, Mission) for mission in missions.values()), \
            "All values in missions must be Mission objects."
        assert isinstance(orbitdata, dict), "Orbit data must be a dictionary."
        assert all(isinstance(od, OrbitData) for od in orbitdata.values()), \
            "All values in orbitdata must be OrbitData objects."
        assert isinstance(events, list) and all(isinstance(event, GeophysicalEvent) for event in events), \
            "Events must be a list of GeophysicalEvent objects."
        assert isinstance(environment, SimulationEnvironment), "Environment must be a SimulationEnvironment object."
        assert isinstance(agents, list) and all(isinstance(agent, SimulationAgent) for agent in agents), \
            "Agents must be a list of SimulationAgent objects."
        assert isinstance(level, int), "Logging level must be an integer."
        if time_step is not None: raise NotImplementedError('simulations with fixed time-step not yet implemented.')
        assert isinstance(printouts, bool), "Printouts flag must be a boolean."
        
        # set simulation attributes
        self._name : str = name
        self._duration : float = duration 
        self._results_path : str = results_path         
        self._orbitdata : Dict[str, OrbitData] = orbitdata
        self._missions : Dict[str, Mission] = missions
        self._events : List[GeophysicalEvent] = events
        self._environment : SimulationEnvironment = environment
        self._agents : list[SimulationAgent] = agents
        self._level = level
        self._printouts : bool = printouts

        # initialize simulation runtime trackers
        self._t_0 = np.NAN
        self._t_f = np.NAN

        # initialize execution flag
        self.__executed : bool = False
        self.__processed_results : Tuple = None

    """
    SIMULATION EXECUTION METHODS
    """
    def execute(self, pbar_pos : int = 0, pbar_leave=True) -> None:
        """ executes the simulation """
        try:
            # start simulation start time tracker
            self._t_0 = time.perf_counter()

            # define start and end times in seconds
            t, tf = 0.0, timedelta(days=self._duration).total_seconds()
            
            # initialize state-action pairs
            state_action_pairs = {
                # agents start at their initial state and do nothing
                agent.name : (agent.get_state(), None) 
                for agent in self._agents
            }

            # execute simulation loop
            with tqdm(total=tf, 
                      desc=f'{self._name}: Simulating', 
                      leave=pbar_leave, 
                      mininterval=0.5, 
                      unit=' s', 
                      position=pbar_pos,
                      file=sys.stderr,
                      disable=not self._printouts
                    ) as pbar:
                
                # ----- main event-driven simulation loop -----
                while t < tf:

                    # update simulation states
                    agent_observations : Dict[str, Tuple[SimulationAgentState, AgentAction, str, List, List]] \
                        = self._environment.step(state_action_pairs, t)

                    # validate that all agents' states were updated
                    assert all(agent.name in agent_observations for agent in self._agents), \
                        "Not all agents received senses from the environment."
                    
                    # agent think
                    state_action_pairs : Dict[str, Tuple[SimulationAgentState, AgentAction]] \
                        = {agent.name : agent.decide_action(*agent_observations[agent.name])
                            for agent in self._agents}

                    # validate that all agents generated actions
                    assert all(agent.name in state_action_pairs for agent in self._agents), \
                        "Not all agents generated actions during think phase."
                    assert all(isinstance(state_action_pairs[agent.name][0], SimulationAgentState)
                               for agent in self._agents), \
                        "Not all agents generated valid states during think phase."
                    assert all((state_action_pairs[agent.name][1] is None or isinstance(state_action_pairs[agent.name][1], AgentAction))
                                 for agent in self._agents), \
                        "Not all agents generated valid actions during think phase."
                    
                    # determine next time 
                    t_next_min = min([action.t_end for _,action in state_action_pairs.values()], 
                                     default=tf) # base case for no agents ends simulation

                    # update progress bar 
                    dt_progress = min(t_next_min - t, tf - t)
                    pbar.update(dt_progress)

                    # update current time
                    t += dt_progress

                    # reset agent percepts for next cycle
                    for *_,msgs,obs_data in agent_observations.values():
                        msgs.clear()
                        obs_data.clear()
                    agent_observations.clear()   

                # ----- final simulation step at tf -----
                # Step environment one last time so any actions ending at tf produce results
                agent_observations = self._environment.step(state_action_pairs, tf)

                # Let agents record/consume any final messages/results
                for agent in self._agents:
                    agent.decide_action(*agent_observations[agent.name])

                # cleanup agent percepts
                for *_, msgs, obs_data in agent_observations.values():
                    msgs.clear()
                    obs_data.clear()
                agent_observations.clear()
                    
            # mark simulation as executed
            self.__executed = True

            # start simulation start time tracker
            self._t_f = time.perf_counter()

            # simulation completed; print results for every agent
            for agent in self._agents: agent.print_results()
            
            # print results for the simulation itself
            self.__print_results()

            # return execution status
            return self.__executed

        except Exception as e:
            # mark simulation as false
            self.__executed = False 
            raise e

        finally:
            
            # print results for the simulation environment regardless of the execution outcome
            self._environment.print_results()

    def __print_results(self) -> None:
        """ prints simulation results after execution """
        # TODO add any results printing here

    """
    DATA PROCESSING METHODS
    """
    def process_results(self, 
                        force_process : bool = False,
                        printouts : bool = True) -> List[pd.DataFrame]:
        """ processes simulation results after execution """
        # validate simulation execution
        if not self.is_executed(printouts): 
            raise RuntimeError("Simulation has not been executed yet. Cannot process results.")
        
        # map agent names to their respective missions
        agent_missions : Dict[str, Mission] = {agent.name : agent._mission
                                    for agent in self._agents}
            

        # check if results have already been processed
        if self.__is_processed(printouts) and not force_process:
            if printouts: print("Simulation results have already been processed. Skipping processing step.")
            
            # collect name of processed results files
            self.__processed_results = ResultsProcessor.load_processed_results(self._results_path,
                                                                               self._orbitdata,
                                                                               agent_missions,
                                                                               self._events,
                                                                               printouts
                                                                            )

        else:            
            if printouts: print("Processing simulation results...")

            # process results and store processed 
            self.__processed_results = ResultsProcessor.process_results(self._results_path,
                                                                 self._orbitdata,
                                                                 agent_missions,
                                                                 self._events,
                                                                 printouts=printouts)


        # return processed dataframes
        return self.__processed_results

    def __is_processed(self, printouts : bool) -> bool:
        """ Checks if simulation results have already been processed. """
        # check if this instance of the simnulation has been executed or if results files already exist
        if not self.__executed and printouts:
            print('WARNING: Simulation instance has not been executed yet. Evaluating pre-existing scenario data...\n')
        
        if self.__processed_results:
            if printouts: print("Processed results already stored in this simulation instance.")
            return True

        # check if data processing output files already exist
        existing_processed_files = [f for f in os.listdir(self._results_path) if f.endswith('.parquet')]
        required_processed_files = ['grid_data.parquet', 'events_detected.parquet', 'events_requested.parquet', 'known_tasks.parquet',
                                    'accesses_per_event.parquet', 'accesses_per_task.parquet', 'observations_per_event.parquet', 
                                    'observations_per_task.parquet', 'planned_rewards.parquet', 'execution_costs.parquet']        

        missing_processed_files = set(required_processed_files) - set(existing_processed_files)

        if missing_processed_files:
            # no processed data files found
            if printouts: print(f"Missing processed results files: {missing_processed_files}")
            return False
        
        if printouts: print("All required processed results files found.")
        return True
    
    def summarize_results(self,
                          force_summarize : bool = False,
                          printouts : bool = True,
                          print_to_csv : bool = True,
                          precision : int = 5
                        ) -> pd.DataFrame:
        """ Loads processed data and generates a summary dataframe without reprocessing raw results files. """        
        
        # print divider
        if printouts: print(f"\n\n{'='*30} SIMULATION RESULTS {'='*30}\n")

        # define results summary filename
        summary_path = os.path.join(f"{self._results_path}","summary.csv")

        # check if results summary file exists 
        if os.path.isfile(summary_path) and not force_summarize:
            # file exists and reevaluate is False; skip results summary generation
            print(f"Results summary already exists at: `{summary_path}`")
            results_summary : pd.DataFrame = pd.read_csv(summary_path)

        # results summary does not exist or must be regenerated; check status of processed results
        elif self.__processed_results:
            # processed results are already stored in this simulation instance;
            #   use processed results and generate summary
            processed_results = self.__processed_results

            results_summary : pd.DataFrame \
                = ResultsProcessor.summarize_results(self._results_path,
                                                        self._orbitdata,
                                                        self._events,
                                                        *processed_results, 
                                                        precision=precision, 
                                                        printouts=printouts)
        else:
            # results have not already stored in this simulation instance; 
            #   process results and generate summary
            processed_results = self.process_results(printouts=printouts)

            results_summary : pd.DataFrame \
                = ResultsProcessor.summarize_results(self._results_path,
                                                        self._orbitdata,
                                                        self._events,
                                                        *processed_results, 
                                                        precision=precision, 
                                                        printouts=printouts)
            
        # include runtime in results summary if simulation has been executed by this instance
        if self.__executed:
            runtime_row = {'Metric' : 'Runtime [s]', 'Value' : round(self._t_f - self._t_0, precision)}
            results_summary = pd.concat([results_summary, pd.DataFrame([runtime_row])], ignore_index=True)

        # log results summary
        if printouts:
            print(f"\n\n{'-'*80}\n")
            print(f"\nSIMULATION RESULTS SUMMARY:\n")
            print(results_summary.to_string(index=False))
            print(f"\n{'='*80}\n")

        # save summary to csv if needed
        if print_to_csv: results_summary.to_csv(summary_path, index=False)
        if printouts: print(f" - Results summary saved to: `{summary_path}`")

        # return results summary
        return results_summary
    
    """
    UTILITY METHODS
    """
    def is_executed(self, printouts : bool = True) -> bool:
        """ returns whether the simulation has been executed """
        # return true if this simulation has been executed or if results exist
        return self.__executed or self.__check_if_results_exist(printouts)

    def __check_if_results_exist(self, printouts : bool) -> bool:
        """ checks if the simulation has been executed by looking for result files """
        # check for existence of result files for all agents
        if not self.__executed and printouts:
            print('WARNING: Simulation instance has not been executed yet. Evaluating pre-existing scenario data...\n')

        # ensure all simulation elements have populated their results directories
        results_dirs = [os.path.join(self._results_path, agent.name.lower()) 
                        for agent in self._agents]
        results_dirs += [os.path.join(self._results_path, 'environment')]

        for dir in results_dirs:
            if not os.path.isdir(dir):
                return False
            if len(os.listdir(dir)) < 2:  # assuming at least 2 files indicate results exist
                return False
        
        # all results directories exist and contain files
        return True

    @classmethod
    def from_dict(cls, d : dict, overwrite : bool = False, printouts : bool = True, level=logging.WARNING) -> 'Simulation':
        """ creates simulation instance from dictionary """

        # unpack agent info
        scenario_duration : float = d['duration']  
        spacecraft_dict : List[dict] = d.get('spacecraft', None)
        gstation_dict   : List[dict] = d.get('groundStation', None)
        gops_dict       : List[dict] = d.get('groundOperator', None)
        gsensor_dict    : List[dict] = d.get('groundSensor', None)
        scenario_dict   : dict = d.get('scenario', None)

        if gops_dict is not None: assert gstation_dict is not None, \
            "Both `groundStation` and `groundOperator` must be defined in the input file."
        
        # load agent names
        agent_names = [SimulationRoles.ENVIRONMENT.value]
        agent_ids = []
        if spacecraft_dict: 
            agent_names.extend([spacecraft['name'] for spacecraft in spacecraft_dict])
            agent_ids.extend([spacecraft['@id'] for spacecraft in spacecraft_dict])
        if gops_dict:
            agent_names.extend([ground_operator['name'] for ground_operator in gops_dict])
            agent_ids.extend([ground_operator['@id'] for ground_operator in gops_dict])

        # validate agent names and ids
        assert len(agent_names) > 1, "At least one agent (spacecraft, UAV, or ground station) must be defined in the input file."
        unique_names = set(agent_names)
        assert len(unique_names) == len(agent_names), "All agent names must be unique."
        unique_ids = set(agent_ids)
        assert len(unique_ids) == len(agent_ids), "All agent IDs must be unique."

        # ------------------------------------
        # get scenario name
        scenario_name = scenario_dict.get('name', 'test')
        
        # get scenario path and name
        scenario_path : str = scenario_dict.get('scenarioPath', None)
        if scenario_path is None: raise ValueError(f'`scenarioPath` not contained in input file.')

        # create results directory
        results_path : str = Simulation.__setup_results_directory(scenario_path, scenario_name, agent_names, overwrite)

        # precompute orbit data
        orbitdata_dir = OrbitData.precompute(d, printouts=printouts) if spacecraft_dict is not None else None
        simulation_orbitdata : Dict[str, OrbitData] = OrbitData.from_directory(orbitdata_dir, d, printouts=printouts) if orbitdata_dir is not None else {}
        
        # load missions
        simulation_missions : Dict[str, Mission] = Simulation.load_missions(scenario_dict)

        # load events        
        grid_dict : dict = d.get('grid', None) # TODO for use in random event generation
        events_path : str = Simulation.generate_events(scenario_dict)
        events : List[GeophysicalEvent] = Simulation.__load_events(events_path, scenario_duration, printouts)

        # setup logging
        logger = logging.getLogger(f'Simulation-{scenario_name}')
        if not logger.hasHandlers():
            logger.propagate = False
            logger.setLevel(level)

            c_handler = logging.StreamHandler()
            c_handler.setLevel(level)
            logger.addHandler(c_handler)
            logger.setLevel(level)

        # ------------------------------------
        # create agents 
        agents : list[SimulationAgent] = []
        if isinstance(spacecraft_dict, list):
            for spacecraft in spacecraft_dict:

                # create satellite agent
                agent = Simulation.__spacecraft_agent_factory(spacecraft,
                                                            simulation_orbitdata,
                                                            simulation_missions,
                                                            results_path,
                                                            orbitdata_dir,
                                                            level,
                                                            logger,
                                                            printouts)

                # add to list of agents
                agents.append(agent)

        if isinstance(gops_dict, list):
            for ground_operator in gops_dict:
                
                # create ground operator agent
                agent = Simulation.__ground_operator_agent_factory(ground_operator,
                                                                 simulation_orbitdata,
                                                                 simulation_missions,
                                                                 results_path,
                                                                 orbitdata_dir,
                                                                 level,
                                                                 logger,
                                                                 printouts)
                
                # add to list of agents
                agents.append(agent)
                
        if gsensor_dict is not None:
            # TODO Implement Ground Sensor agents
            raise NotImplementedError('Ground Sensor agents not yet implemented.')
        
        # ------------------------------------
        # create environment
        environment = SimulationEnvironment(results_path, 
                                            simulation_orbitdata,
                                            spacecraft_dict,
                                            gops_dict,
                                            events,
                                            level,
                                            logger,
                                            printouts=printouts)
            
        # return initialized mission
        return Simulation(scenario_name,
                          scenario_duration,
                          results_path,
                          simulation_orbitdata,
                          simulation_missions,
                          events,
                          environment,
                          agents,
                          level,
                          printouts=printouts)

    @staticmethod
    def __setup_results_directory(scenario_path : str, scenario_name : str, agent_names : List[str], overwrite : bool = True) -> str:
        """
        Creates an empty results directory within the current working directory
        """
        # define results paths
        results_path = os.path.join(scenario_path, 'results', scenario_name)
        agents_paths : List[str] = [os.path.join(results_path, agent_name.lower())
                                    for agent_name in agent_names]

        # check if results path exists
        if (not os.path.exists(results_path) 
            and all(os.path.exists(agent_path) for agent_path in agents_paths) 
            and not overwrite):
            # path exists and no overwrite is enabled; return existing results path
            return results_path
        
        if os.path.exists(results_path) and overwrite:
            # path exists and results overwrite is enabled; clear results in case it already exists
            for filename in os.listdir(results_path):
                file_path = os.path.join(results_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    raise e

        # create a results directory for all agents
        for agent_name in agent_names:
            agent_name : str
            agent_results_path : str = os.path.join(results_path, agent_name.lower())
            os.makedirs(agent_results_path, exist_ok=True)

        return results_path
    
    @staticmethod
    def load_missions(scenario_dict : dict) -> dict:
        missions_path = scenario_dict['missionsPath']
        with open(missions_path, 'r') as missions_file:
            missions_dict : dict = json.load(missions_file)
        
        missions = [
            Mission.from_dict(mission_data) 
            for mission_data in missions_dict.get("missions", [])
        ]

        return {mission.name : mission for mission in missions}
    
    @staticmethod
    def generate_events(scenario_dict : dict) -> str:

        # get events configuration dictionary
        events_config_dict : dict = scenario_dict.get('events', None)

        # check if events configuration exists in input file
        if not events_config_dict: 
            raise ValueError('Missing events configuration in Mission Specs input file.')

        # check events configuration format
        events_type : str = events_config_dict.get('@type', None)
        if events_type is None:
            raise ValueError('Event type missing in Mission Specs input file.')
        
        if events_type.lower() == 'predef': # load predefined events
            events_path : str = events_config_dict.get('eventsPath', None) 
            if not events_path: 
                raise ValueError('Path to predefined events not going in Mission Specs input file.')
            else:
                return events_path
            
        # fallback for unimplemented event generation types
        raise NotImplementedError(f'Event generation of type `{events_type}` not yet implemented.')
    
    @staticmethod
    def __load_events(events_path : str, scenario_duration : float, printouts : bool) -> List[GeophysicalEvent]:
        """ Loads events present in the simulation """
        # checks if event path exists
        if events_path is None: return None
        if not os.path.isfile(events_path): raise ValueError(f'List of events not found in `{events_path}`')

        # get simulation duration 
        sim_duration = timedelta(days=scenario_duration).total_seconds()

        # check if events file exists
        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')
        
        # load events as a dataframe
        if printouts: tqdm.write(f' - Loading events from `{events_path}`...')
        events_df : pd.DataFrame = pd.read_csv(events_path)

        # parse events
        events = []
        for _,row in tqdm(events_df.iterrows(), 
                          desc='Loading Events', 
                          total=len(events_df), 
                          leave=False, 
                          mininterval=0.5,
                          disable=len(events_df) < 10 or not printouts,
                          file=sys.stderr,
                        ):
            # convert event to GeophysicalEvent
            if row['start time [s]'] > sim_duration:
                # event is not in the simulation time frame
                continue
            
            # create geophysical event
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row['id']
            )

            # add to list of events
            events.append(event)

        # return list of events
        return events
    
    @staticmethod
    def __agent_factory(agent_dict : dict,
                      agent_specs : object,
                      initial_state : SimulationAgentState,
                      simulation_orbitdata : Dict[str, OrbitData],
                      simulation_missions : Dict[str, Mission],
                      simulation_results_path : str,
                      orbitdata_dir : str,
                      level : int,
                      logger : logging.Logger,
                      printouts : bool
                    ) -> SimulationAgent:
        """ creates a simulation agent from dictionary """
        
        # unpack mission specs
        agent_name = agent_dict.get('name')
        agent_id = agent_dict.get('@id')
        planner_dict = agent_dict.get('planner', None)
        science_dict = agent_dict.get('science', None)
        
        agent_mission_name = agent_dict.get('mission').lower()

        # load orbitdata
        agent_orbitdata : OrbitData = simulation_orbitdata.get(agent_name, None)
        if agent_orbitdata is None:
            raise ValueError(f'OrbitData for spacecraft agent `{agent_name}` not found in precomputed orbit data.')

        # load specific mission assigned to this satellite
        agent_mission : Mission = simulation_missions[agent_mission_name].copy()

        assert agent_mission == simulation_missions[agent_mission_name], \
            f"mission copy failed. {agent_mission} != {simulation_missions[agent_mission_name]}"
        assert agent_mission is not simulation_missions[agent_mission_name], \
            "mission deep copy failed."
        
        # initialize observation data processor 
        processor : ObservationDataProcessor = \
            Simulation.__load_data_processor(science_dict, agent_name, agent_mission)
        
        # load planners
        preplanner, replanner = \
                Simulation.__load_planners(agent_name, 
                                         planner_dict, 
                                         orbitdata_dir, 
                                         agent_mission, 
                                         simulation_missions, 
                                         simulation_orbitdata, 
                                         simulation_results_path,
                                         logger,
                                         printouts
                                        )  

        # build and return agent 
        return SimulationAgent( agent_name, 
                                agent_id,
                                agent_specs,
                                initial_state,
                                agent_mission,
                                simulation_results_path,
                                agent_orbitdata,
                                processor,
                                preplanner,
                                replanner,
                                level,
                                logger,
                                printouts)

    @staticmethod
    def __spacecraft_agent_factory(agent_dict : dict,
                                 simulation_orbitdata : Dict[str, OrbitData],
                                 simulation_missions : Dict[str, Mission],
                                 simulation_results_path : str,
                                 orbitdata_dir : str,
                                 level : int,
                                 logger : logging.Logger,
                                 printouts : bool
                                ) -> SimulationAgent:


        # unpack mission specs
        agent_name = agent_dict.get('name')
        agent_id = agent_dict.get('@id')
        
        # load orbitdata for this spacecraft
        agent_orbitdata : OrbitData = simulation_orbitdata.get(agent_name, None)

        # validate inputs
        if agent_orbitdata is None:
            raise ValueError(f'OrbitData for spacecraft agent `{agent_dict.get("name")}` not found in precomputed orbit data.')

        # load satellite specifications
        agent_specs : Spacecraft = Spacecraft.from_dict(agent_dict)

        # get orbital state and time step
        orbit_state_dict = agent_dict.get('orbitState', None)
        dt = agent_orbitdata.time_step 

        # create initial state
        initial_state = SatelliteAgentState(agent_name,
                                            agent_id,
                                            orbit_state_dict,
                                            time_step=dt) 
        
        # return created agent
        return Simulation.__agent_factory(agent_dict,
                                        agent_specs,
                                        initial_state,
                                        simulation_orbitdata,
                                        simulation_missions,
                                        simulation_results_path,
                                        orbitdata_dir,
                                        level,
                                        logger,
                                        printouts)


    @staticmethod
    def __ground_operator_agent_factory(agent_dict : dict,
                                      simulation_orbitdata : Dict[str, OrbitData],
                                      simulation_missions : Dict[str, Mission],
                                      simulation_results_path : str,
                                      orbitdata_dir : str,
                                      level : int,
                                      logger : logging.Logger,
                                      printouts : bool
                                    ) -> SimulationAgent:
        """ creates a ground operator simulation agent from dictionary """
        
        # unpack mission specs
        agent_name = agent_dict.get('name')
        agent_id = agent_dict.get('@id')
        
        # load payload
        instruments_dict = agent_dict.get('instrument', None)   
        agent_specs : dict = {key: val for key,val in agent_dict.items()}
        agent_specs['payload'] = dictionary_list_to_object_list(instruments_dict, Instrument) \
                                    if instruments_dict else []

        # define initial state
        initial_state = GroundOperatorAgentState(agent_name, agent_id)

        # return created agent
        return Simulation.__agent_factory(agent_dict,
                                        agent_specs,
                                        initial_state,
                                        simulation_orbitdata,
                                        simulation_missions,
                                        simulation_results_path,
                                        orbitdata_dir,
                                        level,
                                        logger,
                                        printouts)

    
    @staticmethod
    def __load_data_processor(science_dict : dict, 
                            agent_name : str,
                            mission : Mission
                        ) -> ObservationDataProcessor:
        if science_dict is not None:
            science_dict : dict

            # load science module type
            science_module_type : str = science_dict.get('@type', None)
            if science_module_type is None: raise ValueError(f'science module type not specified in input file.')

            # create an instance of the science module based on the specified science module type
            if science_module_type.lower() == "lookup":
                # load events path
                events_path : str = science_dict.get('eventsPath', None)

                if events_path is None: raise ValueError(f'predefined events path not specified in input file.')

                # create science module
                processor = LookupProcessor(events_path, agent_name, mission)

            else:
                raise NotImplementedError(f'science module of type `{science_module_type}` not yet supported.')
            
            # return science module
            return processor

        # return nothing
        return None
    
    @staticmethod
    def __load_planners(agent_name : str, 
                      planner_dict : dict, 
                      orbitdata_dir : str, 
                      agent_mission : Mission, 
                      simulation_missions : Dict[str,Mission], 
                      simulation_orbitdata: Dict[str, OrbitData],
                      simulation_results_path : str,
                      logger : logging.Logger,
                      printouts : bool
                    ) -> tuple:
        # check if planner dictionary is empty
        if planner_dict is None: return None, None

        # define path for agent results to be used by planners
        agent_results_dir = os.path.join(simulation_results_path, agent_name.lower())

        # load preplanner
        preplanner = Simulation.__load_preplanner(planner_dict, agent_results_dir, agent_mission, simulation_missions, simulation_orbitdata, orbitdata_dir, agent_name, logger, printouts)

        # load replanner
        replanner = Simulation.__load_replanner(planner_dict, agent_results_dir, logger, printouts)

        # return loaded planners
        return preplanner, replanner
    
    @staticmethod
    def __load_preplanner(planner_dict : dict, 
                          agent_results_dir : str,
                          agent_mission : Mission, 
                          simulation_missions : Dict[str,Mission],
                          simulation_orbitdata : Dict[str, OrbitData],
                          orbitdata_dir : str,
                          agent_name : str,
                          logger : logging.Logger,
                          printouts : bool
                        ) -> AbstractPeriodicPlanner:
        """ loads the preplanner for the agent """
        # get preplanner specs
        preplanner_dict : Dict = planner_dict.get('preplanner', None)
        
        # check if preplanner specs exist
        if preplanner_dict is None: return None

        # get preplanner parameters
        preplanner_type : str = preplanner_dict.get('@type', None)
        if preplanner_type is None: raise ValueError(f'preplanner type within planner module not specified in input file.')

        period = preplanner_dict.get('period', np.Inf)
        horizon = preplanner_dict.get('horizon', period)
        horizon = np.Inf if isinstance(horizon, str) and 'inf' in horizon.lower() else horizon
        debug = bool(preplanner_dict.get('debug', 'false').lower() in ['true', 't'])
        sharing = preplanner_dict.get('sharing', 'none').lower()

        if period > horizon: raise ValueError('replanning period must be greater than planning horizon.')

        # initialize preplanner
        if preplanner_type.lower() in ["heuristic"]:
            return HeuristicInsertionPeriodicPlanner(horizon, period, sharing, debug, logger, printouts)

        elif preplanner_type.lower() in ["naive", "fifo", "earliest"]:
            return EarliestAccessPlanner(horizon, period, sharing, debug, logger, printouts)

        elif preplanner_type.lower() == 'nadir':
            return NadirPointingPlanner(horizon, period, sharing, debug, logger, printouts)

        elif preplanner_type.lower() in ["dynamic", "dp"]:
            model = preplanner_dict.get('model', 'earliest').lower()
            return DynamicProgrammingPlanner(horizon, period, model, sharing, debug, logger, printouts)
        
        elif preplanner_type.lower() in ["eventannouncer", "announcer"]:
            events_path = preplanner_dict.get('eventsPath', None)
            if events_path is None: raise ValueError(f'predefined events path not specified in input file.')
            
            return EventAnnouncerPlanner(events_path, agent_mission, debug, logger, printouts)

        elif preplanner_type.lower() == 'dealer':
            # unpack preplanner parameters
            mode = preplanner_dict.get('@mode', 'test').lower()
            clients = preplanner_dict.get('clients', None)

            # load client specs for all agents (except self)
            client_specs_path = os.path.join(orbitdata_dir, 'MissionSpecs.json')
            with open(client_specs_path, 'r') as clients_file:
                mission_specs : dict = json.load(clients_file)
            client_specs : Dict[str, Spacecraft] = {d['name']: Spacecraft.from_dict(d) 
                                                    for d in mission_specs.get('spacecraft', [])
                                                    if d['name'] != agent_name}

            # load client missions for all agents (except self)
            client_missions : Dict[str, Mission] = {d['name'] : simulation_missions[d['mission'].lower()] 
                                                    for d in mission_specs.get('spacecraft', [])
                                                    if d['name'] != agent_name}

            # load client orbitdata for all agents (except self)
            client_orbitdata = {k:v for k,v in simulation_orbitdata.items() if k != agent_name}

            # remove other clients if specified
            if clients is not None:
                assert len(clients) > 0, '`clients` list is empty.'
                assert all([isinstance(c, str) for c in clients]), '`clients` list must contain strings.'

                for client in list(client_orbitdata.keys()):
                    if client not in clients:
                        client_orbitdata.pop(client, None)
                        client_specs.pop(client, None)
                        client_missions.pop(client, None)
            
            if mode == 'test':                   
                return TestingDealer(client_orbitdata, client_specs, horizon, period, printouts)

            elif mode in ['milp', 'mixed-integer-linear-programming']:
                model = preplanner_dict.get('model', DealerMILPPlanner.STATIC).lower()
                license_path = preplanner_dict.get('licensePath', None)
                max_tasks = preplanner_dict.get('maxTasks', np.Inf)
                max_observations = preplanner_dict.get('maxObservations', 10)

                return DealerMILPPlanner(client_orbitdata, 
                                         client_specs, 
                                         client_missions, 
                                         model, 
                                         license_path, 
                                         horizon, 
                                         period, 
                                         max_tasks=max_tasks, 
                                         max_observations=max_observations, 
                                         debug=debug,
                                         logger=logger,
                                         printouts=printouts
                                        )

        elif preplanner_type.lower() == 'worker':
            dealer_name = preplanner_dict.get('dealerName', None)
            return WorkerPlanner(dealer_name, debug, logger, printouts)

        elif preplanner_type.lower() == 'blank':
            return BlankPlanner(horizon, period, sharing, debug, logger, printouts)

        # elif... # add more preplanners here

        # TODO reactivate MILP preplanner when implemented VVV
        # elif preplanner_type.lower() in ['milp', 'mixed-integer-linear-programming']:
        #     # unpack preplanner parameters
        #     obj = preplanner_dict.get('objective', 'reward').lower()
        #     model = preplanner_dict.get('model', 'earliest').lower()
        #     license_path = preplanner_dict.get('licensePath', None)
        #     max_tasks = preplanner_dict.get('maxTasks', np.Inf)

        #     if license_path is None and not debug: 
        #         raise ValueError('license path for Gurobi MILP preplanner not specified. Set `debug` to true to run with limited functionality or specify a valid license path to `licensePath`.')        

        #     preplanner = SingleSatMILP(obj, model, license_path, horizon, period, max_tasks, debug, logger)
        
        # fallback for unimplemented preplanner types
        raise NotImplementedError(f'preplanner of type `{preplanner_dict}` not yet supported.')
        
    @staticmethod
    def __load_replanner(planner_dict : dict, 
                         agent_results_dir : str, 
                         logger : logging.Logger, 
                         printouts : bool) -> AbstractReactivePlanner:
        """ loads the replanner for the agent """
        # get replanner specs
        replanner_dict = planner_dict.get('replanner', None)

        if replanner_dict is None: return None

        replanner_type : str = replanner_dict.get('@type', None)
        if replanner_type is None: raise ValueError(f'replanner type within planner module not specified in input file.')
        debug = bool(replanner_dict.get('debug', 'false').lower() in ['true', 't'])
        
        if replanner_type.lower() in ['consensus', 'cbba']:
            model = replanner_dict.get('model', 'heuristicInsertion')
            replan_threshold = replanner_dict.get('replanThreshold', 1)
            optimistic_bidding_threshold = replanner_dict.get('optimisticBiddingThreshold', 1)
            periodic_overwrite = bool(replanner_dict.get('periodicOverwrite', 'false').lower() in ['true', 't'])

            if 'heuristic' in model:
                heuristic = replanner_dict.get('heuristic', 'earliestAccess')
                return HeuristicInsertionConsensusPlanner(agent_results_dir, heuristic, replan_threshold, optimistic_bidding_threshold, periodic_overwrite, debug, logger, printouts)
            else:
                raise NotImplementedError(f'replanner model `{model}` not yet supported.')
        
        # fallback for unimplemented replanner types
        raise NotImplementedError(f'replanner of type `{replanner_dict}` not yet supported.')
    