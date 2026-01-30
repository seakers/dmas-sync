import logging
import os
import uuid
from queue import Queue

from execsatm.mission import Mission
from execsatm.tasks import GenericObservationTask, DefaultMissionTask
from execsatm.objectives import DefaultMissionObjective
from execsatm.requirements import SpatialCoverageRequirement, SinglePointSpatialRequirement, MultiPointSpatialRequirement, GridSpatialRequirement

from dmas.core.orbitdata import OrbitData
from dmas.models.actions import AgentAction
from dmas.models.planning.periodic import AbstractPeriodicPlanner
from dmas.models.planning.reactive import AbstractReactivePlanner
from dmas.models.science.processing import DataProcessor
from dmas.models.states import SimulationAgentState
from dmas.models.science.requests import TaskRequest
from dmas.models.planning.plan import PeriodicPlan, Plan


class SimulationAgent(object):
    def __init__(self, 
                 agent_name : str, 
                 agent_id : str,
                 specs : object,
                 initial_state : SimulationAgentState, 
                 mission : Mission,
                 simulation_results_path : str,
                 orbitdata : OrbitData,
                 processor : DataProcessor = None, 
                 preplanner : AbstractPeriodicPlanner = None,
                 replanner : AbstractReactivePlanner = None,
                 level : int = logging.INFO, 
                 logger : logging.Logger = None
                ):
        # validate inputs        
        assert isinstance(agent_name, str), "Agent name must be a string."
        assert isinstance(agent_id, (str, type(None))), "Agent ID must be a string or None."
        assert isinstance(specs, object), "Specs must be an object."
        assert isinstance(initial_state, SimulationAgentState), "Initial state must be a SimulationAgentState object."
        assert isinstance(mission, Mission), "Mission must be an execsatm Mission object."
        assert isinstance(simulation_results_path, str), "Simulation results path must be a string."
        agent_results_path = os.path.join(simulation_results_path, agent_name.lower())
        assert os.path.exists(agent_results_path), f"Agent results path {agent_results_path} does not exist."
        assert isinstance(orbitdata, OrbitData), "Orbit data must be an OrbitData object."
        assert processor is None or isinstance(processor, DataProcessor), "Processor must be a DataProcessor object or None."
        assert preplanner is None or isinstance(preplanner, AbstractPeriodicPlanner), "Preplanner must be an AbstractPeriodicPlanner object or None."
        assert replanner is None or isinstance(replanner, AbstractReactivePlanner), "Replanner must be an AbstractReactivePlanner object or None."
        assert isinstance(level, int), "Logging level must be an integer."
        assert logger is None or isinstance(logger, logging.Logger), "Logger must be a logging.Logger object or None."

        # assign parameters
        self.name : str = agent_name.lower()
        self._id : str = agent_id if agent_id is not None else str(uuid.uuid4())
        self._specs : object = specs
        self._state : SimulationAgentState = initial_state
        self._mission : Mission = mission
        self._simulation_results_path : str = simulation_results_path
        self._results_path : str = agent_results_path
        self._orbitdata : OrbitData = orbitdata
        self._processor : DataProcessor = processor
        self._preplanner : AbstractPeriodicPlanner = preplanner
        self._replanner : AbstractReactivePlanner = replanner

        # initialize logger
        self._logger : logging.Logger = logger if logger is not None \
                                            else logging.getLogger(f"Agent-{self.name}")
        self._logger.setLevel(level)

        # initailize other variables
        self._message_inbox : Queue = Queue([])
        self._message_outbox : Queue = Queue([])
        self._plan : Plan = PeriodicPlan(t=-1.0)
        self._plan_history = []
        self._state_history : list = []
        self._known_tasks : list[GenericObservationTask] \
            = SimulationAgent.__initialize_default_mission_tasks(mission, orbitdata)
        self._known_reqs : set[TaskRequest] = set() # TODO do we need this or is the task list enough?

    @staticmethod
    def __initialize_default_mission_tasks(mission : Mission, orbitdata : OrbitData) -> None:
        """ 
        Creates default observation tasks for each default mission objective
         based on the spatial requirements of each objective.
        """
        # initialize task list
        tasks = []

        # gather targets for each default mission objective
        objective_targets = { objective : [] for objective in mission 
                             # ignore non-default objectives
                             if isinstance(objective, DefaultMissionObjective)
                             }
        
        # iterate through each mission objective
        for objective,targets in objective_targets.items():  
            # collect spatial coverage requirements
            spatial_requirements = [req for req in objective.requirements
                                    if isinstance(req, SpatialCoverageRequirement)]

            # iterate through each spatial requirement
            for req in spatial_requirements:
                if isinstance(req, SinglePointSpatialRequirement):
                    # collect specified target
                    req_targets = [req.target]
                
                elif isinstance(req, MultiPointSpatialRequirement):
                    # collect all specified targets
                    req_targets = [target for target in req.targets]
                
                elif isinstance(req, GridSpatialRequirement):
                    # collect all targets matching this grid requirement
                    req_targets = [
                        (lat, lon, grid_index, gp_index)
                        for grid in orbitdata.grid_data
                        for lat,lon,grid_index,gp_index in grid.values
                        if grid_index == req.grid_index and gp_index < req.grid_size
                    ]
                else: 
                    raise TypeError(f"Unknown spatial requirement type: {type(req)}")
                    
                # add to list of targets for this objective
                targets.extend(req_targets)

            # check if any spatial coverage requirements were found
            if not spatial_requirements:
                # no spatial coverage requirements found; 
                #   collect all targets from all grids known to this agent
                req_targets = list({
                    (lat, lon, grid_index, gp_index)
                    for grid in orbitdata.grid_data
                    for lat,lon,grid_index,gp_index in grid.values
                })
                targets.extend(req_targets)
        
        # iterate through each mission objective
        for objective,targets in objective_targets.items():                           
            # create monitoring tasks from each location in this mission objective
            objective_tasks = [DefaultMissionTask(objective.parameter,
                                        location=(lat, lon, grid_index, gp_index),
                                        mission_duration=orbitdata.duration*24*3600,
                                        objective=objective,
                                        )
                        for lat,lon,grid_index,gp_index in targets
                    ]
            
            # add to list of known tasks
            tasks.extend(objective_tasks)

        # return list of created tasks
        return tasks

    """
    ----------------------
    SIMULATION CYCLE METHODS
    ----------------------
    """
    
    """
    SENSE METHOD
    """
    def sense(self, status : AgentAction) -> list:
        ...

    """
    THINK METHOD
    """
    def think(self, senses : list) -> AgentAction:
        return [] # TODO

    """
    ACT METHOD
    """
    def act(self, action : AgentAction) -> list:
        ...
   
    """
    ----------------------
    UTILITY METHODS
    ----------------------
    """    


    """
    ----------------------
    RESULTS HANDLING METHODS
    ----------------------
    """
    def print_results(self) -> str:
        # TODO 
        ...