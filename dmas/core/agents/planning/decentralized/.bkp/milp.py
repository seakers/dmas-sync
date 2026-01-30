import os
from chess3d.agents.planning.planner import *
from chess3d.agents.planning.periodic import AbstractPeriodicPlanner

import gurobipy as gp
import numpy as np

class SingleSatMILP(AbstractPeriodicPlanner):
    EARLIEST = 'earliest'
    BEST = 'best'
    REOBSERVATION = 'reobservation'
    VALID_MODELS = [EARLIEST, BEST, REOBSERVATION]

    def __init__(self, 
                 objective : str, 
                 model : str, 
                 licence_path : str = None, 
                 horizon = np.Inf, 
                 period = np.Inf, 
                 max_tasks = np.Inf,
                 debug = False, 
                 logger = None
                ):
        """ 
        ## Decentralized Single-Satellite MILP Planner
        Periodic preplanner that generates an observation plan for a single satellite using MILP optimization models.
        
        #### Arguments:
        - `objective`: str - Objective function for the MILP model. Either "reward" or "duration".
        - `model`: str - MILP model to used to schedule observation start times. Either "earliest" or "best".
        - `licence_path`: str - Path to Gurobi licence file. Required if `debug` is False.
        - `horizon`: float - Planning horizon in seconds. Default is infinite.
        - `period`: float - Planning period in seconds. Default is infinite.
        - `max_tasks`: int - Maximum number of tasks to consider in each optimization run. Default is infinite.
        - `debug`: bool - If True, runs in debug mode and does not use optimization. Default is False.
        - `logger`: logging.Logger - Logger for logging messages. Default is None.
        """
        super().__init__(horizon, period, debug, logger)

        if not debug or licence_path is not None:
            # Check for Gurobi license
            assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."

            # Set Gurobi license environment variable
            os.environ['GRB_LICENSE_FILE'] = licence_path

        # Validate inputs
        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."
        assert model in self.VALID_MODELS, f"Model must be either '{self.EARLIEST}', '{self.BEST}' or '{self.REOBSERVATION}'."
        assert (isinstance(max_tasks, int) and max_tasks > 0) or max_tasks == np.Inf, "Max tasks must be a positive integer."

        # Set attributes
        self.objective = objective
        self.model = model
        self.max_tasks = max_tasks

        raise NotImplementedError("This planner is under development and not yet ready for use.")

    @runtime_tracker
    def _schedule_observations(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               _ : ClockConfig, 
                               orbitdata : OrbitData, 
                               schedulable_tasks : list, 
                               mission : Mission, 
                               observation_history : ObservationHistory
                               ) -> list:
        # Check if there are no tasks to schedule
        if not schedulable_tasks: return []
        
        # Set type for `schedulable_tasks` and sort in ascending start time access
        schedulable_tasks : list[ObservationOpportunity] = sorted(schedulable_tasks, key=lambda x: x.accessibility.left)
        
        # validate Inputs
        assert all(isinstance(task, ObservationOpportunity) for task in schedulable_tasks), "All tasks must be of type `SpecificObservationTask`."
        assert all(task.min_duration > 0 for task in schedulable_tasks), "All tasks must have positive duration requirements."
        t_max = max((task.accessibility.right for task in schedulable_tasks if isinstance(task, ObservationOpportunity)))
        assert t_max <= state.t + self.horizon, f"Tasks exceed the planning horizon of {self.horizon} seconds."

        if not isinstance(state, SatelliteAgentState):
            raise NotImplementedError(f'Naive planner not yet implemented for agents of type `{type(state)}.`')
        elif not isinstance(specs, Spacecraft):
            raise ValueError(f'`specs` needs to be of type `{Spacecraft}` for agents with states of type `{SatelliteAgentState}`')
                
        # compile instrument field of view specifications   
        cross_track_fovs : dict = self._collect_fov_specs(specs)
        
        # get pointing agility specifications
        adcs_specs : dict = specs.spacecraftBus.components.get('adcs', None)
        assert adcs_specs, 'ADCS component specifications missing from agent specs object.'

        max_slew_rate = float(adcs_specs['maxRate']) if adcs_specs.get('maxRate', None) is not None else None
        assert max_slew_rate, 'ADCS `maxRate` specification missing from agent specs object.'

        max_torque = float(adcs_specs['maxTorque']) if adcs_specs.get('maxTorque', None) is not None else None
        assert max_torque, 'ADCS `maxTorque` specification missing from agent specs object.'

        # Decide how to split tasks into chunks according to `max_tasks`
        if self.max_tasks < np.Inf:
            task_chunks = [
                schedulable_tasks[i:i + self.max_tasks] 
                for i in range(0, len(schedulable_tasks), self.max_tasks)
            ]
        else:
            task_chunks = [schedulable_tasks]

        # Run optimization on each chunk and flatten results
        observations : list[ObservationAction] = []
        curr_state : SimulationAgentState = state.copy()
        for tasks in task_chunks:
            # Update state
            if observations:
                last_action : ObservationAction = observations[-1]
                curr_state = curr_state.propagate(last_action.t_end)
                curr_state.attitude[0] = last_action.look_angle

            # Filter tasks based on current state
            reduced_tasks = [task for task in tasks
                     if task.accessibility.right - max(task.accessibility.left, curr_state.t) >= task.min_duration
                     and all(not task.is_mutually_exclusive(obs.obs_opp) for obs in observations)]
                     
            for task in reduced_tasks:
                if curr_state.t in task.accessibility:
                    task.accessibility = Interval(curr_state.t, task.accessibility.right)

            if not self._debug:
                # Calculate optimized observation schedule using the selected model
                if self.model == self.EARLIEST:
                    _,x,__,tau,___ = self.__initialize_observations_schedule(curr_state, 
                                                                            specs,
                                                                            cross_track_fovs,
                                                                            max_slew_rate, 
                                                                            max_torque, 
                                                                            orbitdata,
                                                                            reduced_tasks, 
                                                                            mission, 
                                                                            observation_history
                                                                        )
                elif self.model == self.BEST:
                    _,x,__,tau,___ = self.__optimize_observations_schedule(curr_state, 
                                                                                specs,
                                                                                cross_track_fovs,
                                                                                max_slew_rate, 
                                                                                max_torque, 
                                                                                orbitdata,
                                                                                reduced_tasks, 
                                                                                mission, 
                                                                                observation_history
                                                                            )
                                
                elif self.model == self.REOBSERVATION:
                    _,x,__,tau,___ = self.__optimize_observations_schedule_temp(curr_state, 
                                                                                specs,
                                                                                cross_track_fovs,
                                                                                max_slew_rate, 
                                                                                max_torque, 
                                                                                orbitdata,
                                                                                reduced_tasks, 
                                                                                mission, 
                                                                                observation_history,
                                                                                # x_init
                                                                            )
                else:
                    raise ValueError(f'Model `{self.model}` not recognized. Valid models are: {self.VALID_MODELS}')

                # Extract observation sequence
                d = [task.min_duration for task in reduced_tasks if isinstance(task, ObservationOpportunity)]
                th_imgs = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in reduced_tasks if isinstance(task, ObservationOpportunity)]

                # Append to list of observations
                observations.extend(self.__extract_observation_sequence(reduced_tasks, curr_state, x, tau, d, th_imgs))

            else: # DEBUG MODE ON
                # Calculate optimized observation schedule
                t_ref = time.perf_counter()
                _,x_0,__,tau_0,obj_0 = self.__initialize_observations_schedule(curr_state, 
                                                                            specs,
                                                                            cross_track_fovs,
                                                                            max_slew_rate, 
                                                                            max_torque, 
                                                                            orbitdata,
                                                                            reduced_tasks, 
                                                                            mission, 
                                                                            observation_history
                                                                        )
                t_0 = time.perf_counter() - t_ref

                t_ref = time.perf_counter()
                _,x_1,__,tau_1,obj_1 = self.__optimize_observations_schedule(curr_state, 
                                                                            specs,
                                                                            cross_track_fovs,
                                                                            max_slew_rate, 
                                                                            max_torque, 
                                                                            orbitdata,
                                                                            reduced_tasks, 
                                                                            mission, 
                                                                            observation_history
                                                                        )
                t_1 = time.perf_counter() - t_ref

                t_ref = time.perf_counter()
                _,x_2,__,tau_2,obj_2 = self.__optimize_observations_schedule_temp(curr_state, 
                                                                            specs,
                                                                            cross_track_fovs,
                                                                            max_slew_rate, 
                                                                            max_torque, 
                                                                            orbitdata,
                                                                            reduced_tasks, 
                                                                            mission, 
                                                                            observation_history,
                                                                            # x_0
                                                                        )
                t_2 = time.perf_counter() - t_ref

                # Extract observation sequence
                d = [task.min_duration for task in reduced_tasks if isinstance(task, ObservationOpportunity)]
                th_imgs = [np.average((task.slew_angles.left, task.slew_angles.right)) for task in reduced_tasks if isinstance(task, ObservationOpportunity)]

                # Append to list of observations
                # observations.extend(self.__extract_observation_sequence(reduced_tasks, curr_state, x_1, tau_1, d, th_imgs))

                # DEBUG
                obs_0 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_0, tau_0, d, th_imgs)
                obs_1 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_1, tau_1, d, th_imgs)
                obs_2 = self.__extract_observation_sequence(reduced_tasks, curr_state, x_2, tau_2, d, th_imgs)
                
                max_slew_rate, max_torque = self._collect_agility_specs(specs)

                valid_0 : bool = self.is_observation_path_valid(state, obs_0, max_slew_rate, max_torque)
                valid_1 : bool = self.is_observation_path_valid(state, obs_1, max_slew_rate, max_torque)
                valid_2 : bool = self.is_observation_path_valid(state, obs_2, max_slew_rate, max_torque)
                y = 1
            
        return observations
    
    def __initialize_observations_schedule(self, 
                                         state : SimulationAgentState, 
                                         specs : object, 
                                         cross_track_fovs : dict, 
                                         max_slew_rate : float, 
                                         max_torque : float,
                                         orbitdata : OrbitData, 
                                         schedulable_tasks : list, 
                                         mission : Mission, 
                                         observation_history : ObservationHistory
                                         ) -> tuple:
        """ Uses a simplified MILP formulation to generate an initial solution for the full MILP optimization """

        # Check if there are no tasks to schedule
        if not schedulable_tasks: return None, [], [], [], np.NAN

        # Add dummy task to represent initial state
        dummy_task = ObservationOpportunity(set([]),
                                             schedulable_tasks[0].instrument_name,
                                             Interval(state.t,state.t), 
                                             0.0, 
                                             Interval(state.attitude[0],state.attitude[0]))
        tasks : list[ObservationOpportunity] = [dummy_task]
        tasks.extend(schedulable_tasks)

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # List tasks by their index
        indexed_tasks = list(enumerate(tasks))
        task_indices = [j for j,_ in indexed_tasks]

        # Define constants
        rewards = np.array([self.estimate_observation_opportunity_value(task, 
                                                     task.accessibility.left, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,ObservationOpportunity)])
        t_start   = np.array([task.accessibility.left-state.t for task in tasks if isinstance(task, ObservationOpportunity)])
        d         = np.array([task.min_duration for task in tasks if isinstance(task, ObservationOpportunity)])
        th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, ObservationOpportunity)])
        slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
                               for j in task_indices]
                               for j_p in task_indices
                             ])        
        Z = [(j, j_p) 
             for j      in task_indices 
             for j_p    in task_indices
             if t_start[j] + d[j] + slew_time[j, j_p] <= t_start[j_p]   # sequence j->j' is feasible
             and j != j_p                                               # ensure distinct tasks
             ] 
        
        E = [(j,j_p) 
             for j,task_j       in indexed_tasks 
             for j_p,task_j_p   in indexed_tasks
             if ((task_j.is_mutually_exclusive(task_j_p) # mutual exclusivity constraint
                    or (
                        t_start[j] + d[j] + slew_time[j, j_p] > t_start[j_p]    # sequence j->j' is not feasible
                    and  t_start[j_p] + d[j_p] + slew_time[j_p, j] > t_start[j] # sequence j'->j is not feasible
                    ))                                      
                and j < j_p)                               # ensure distinct tasks
             ]
        
        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")

        # Validate constants and decision variables to ensure convergence
        assert all([reward >= 0 for reward in rewards])
        assert all([0 <= t_start[j] <= self.horizon for j in task_indices])
        assert all([slew_time[j][j_p] >= 0 for j in task_indices for j_p in task_indices])
        assert all([j in x.keys() and j_p in x.keys() for j,j_p in Z])

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum( rewards[j] * x[j] for j in task_indices), gp.GRB.MAXIMIZE)
                
        # Add constraints
        ## Always assign first observation
        # with 
        model.addConstr(x[0] == 1)

        ## Enforce exclusivity
        for j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(x[j] + x[j_p] <= 1)

        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)

        # Convert decision variables to arrays
        x_array = np.array([int(x[j].X) for j,_ in indexed_tasks if j > 0])
        z_array = np.array([(j,j_p, int(x[j].X * x[j_p].X)) for j,j_p in Z if j > 0 and j_p > 0])
        tau_array = np.array([t_start[j]+state.t for j,_ in indexed_tasks if j > 0])

        # Return solved model
        return model, x_array, z_array, tau_array, model.getObjective().getValue()
    
    def __optimize_observations_schedule(self, 
                                         state : SimulationAgentState, 
                                         specs : object, 
                                         cross_track_fovs : dict, 
                                         max_slew_rate : float, 
                                         max_torque : float,
                                         orbitdata : OrbitData, 
                                         schedulable_tasks : list, 
                                         mission : Mission, 
                                         observation_history : ObservationHistory,
                                         x_init : np.ndarray = None
                                         ) -> tuple:
        # Check if there are no tasks to schedule
        if not schedulable_tasks: return None, [], [], [], np.NAN

        # Add dummy task to represent initial state
        dummy_task = ObservationOpportunity(set([]),
                                             schedulable_tasks[0].instrument_name,
                                             Interval(state.t,state.t), 
                                             0.0, 
                                             Interval(state.attitude[0],state.attitude[0]))
        tasks : list[ObservationOpportunity] = [dummy_task]
        tasks.extend(schedulable_tasks)

        # Create a new model
        model = gp.Model("single-sat_milp_planner")

        # Set parameter to suppress output
        model.setParam('OutputFlag', int(self._debug))

        # List tasks by their index
        indexed_tasks = list(enumerate(tasks))
        task_indices = [j for j,_ in indexed_tasks]

        # Define constants
        rewards = np.array([self.estimate_observation_opportunity_value(task, 
                                                     task.accessibility.left, 
                                                     task.min_duration, 
                                                     specs, cross_track_fovs, orbitdata, 
                                                     mission, 
                                                     observation_history)
                            for task in tqdm(tasks,leave=False,desc='SATELLITE: Calculating task rewards')
                            if isinstance(task,ObservationOpportunity)])
        t_start   = np.array([task.accessibility.left-state.t for task in tasks if isinstance(task, ObservationOpportunity)])
        t_end     = np.array([task.accessibility.right-state.t for task in tasks if isinstance(task, ObservationOpportunity)])
        d         = np.array([task.min_duration for task in tasks if isinstance(task, ObservationOpportunity)])
        th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, ObservationOpportunity)])
        slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
                               for j in task_indices]
                               for j_p in task_indices
                             ])        
        Z = [(j, j_p) 
             for j      in task_indices 
             for j_p    in task_indices
             if t_start[j] + d[j] + slew_time[j, j_p] <= t_end[j_p] - d[j_p] # sequence j->j' is feasible
             and j != j_p                                                    # ensure distinct tasks
             ] 
        
        E = [(j,j_p) 
             for j,task_j       in indexed_tasks 
             for j_p,task_j_p   in indexed_tasks
             if (task_j.is_mutually_exclusive(task_j_p) # mutual exclusivity constraint
                or (
                    t_start[j] + d[j] + slew_time[j, j_p] > t_end[j_p] - d[j_p]
                and  t_start[j_p] + d[j_p] + slew_time[j_p, j] > t_end[j] - d[j]
                ))                                      # sequence j->j' and j'->j are not feasible
                and j < j_p                             # ensure distinct tasks and non-repeating pairs
             ]
        
        # Create decision variables
        x : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.BINARY, name="x")
        z : gp.tupledict = model.addVars(Z, vtype=gp.GRB.BINARY, name="z")
        tau : gp.tupledict = model.addVars(task_indices, vtype=gp.GRB.CONTINUOUS, name="tau")

        # Validate constants and decision variables to ensure convergence
        assert all([reward >= 0 for reward in rewards])
        assert all([0 <= t_start[j] <= self.horizon for j in task_indices])
        assert all([0 <= t_end[j] <= self.horizon for j in task_indices])
        assert all([t_end[j] - t_start[j] >= d[j] for j in task_indices])
        assert all([t_end[j] - d[j] >= 0.0 for j in task_indices])
        assert all([slew_time[j][j_p] >= 0 for j in task_indices for j_p in task_indices])
        assert all([j in x.keys() and j_p in x.keys() for j,j_p in Z])

        # if initial solution is provided, use it
        if x_init is not None:
            x_init = np.insert(x_init, 0, 1) # compensate for missing initial task
            assert len(x_init) == len(x), "Initial solution length does not match number of tasks."

            for j in task_indices:
                x[j].Start = x_init[j]
                tau[j].Start = float(t_start[j])

            for j,j_p in Z:
                if x_init[j] > 0.5 and x_init[j_p] > 0.5 and j != j_p:
                    z[j, j_p].Start = int(t_start[j] + d[j] + slew_time[j, j_p] <= t_start[j_p])

        # Set objective
        if self.objective == "reward": 
            model.setObjective(gp.quicksum( rewards[j] * x[j] for j in task_indices), gp.GRB.MAXIMIZE)
  
        # Add constraints
        ## Always assign first observation
        model.addConstr(x[0] == 1)
        model.addConstr(tau[0] == 0.0)

        ## Enforce exclusivity
        for j,j_p in tqdm(E, desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
            model.addConstr(x[j] + x[j_p] <= 1)

        ## Observation time and accessibility constraints
        for j in tqdm(x.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
            tau[j].LB = t_start[j]
            tau[j].UB = t_end[j] - d[j]
            # If x[j] == 0, force tau to LB (or any fixed value in the feasible interval)
            model.addGenConstrIndicator(x[j], 0, tau[j] == float(t_start[j]))
           
        ## Slew time constraints for observation task sequence j->j'
        pairs_considered : set = set()
        for j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding slewing constraints", unit='task pairs', leave=False):
           
            # If z[j, j_p] == 1, enforce slew constraint for task sequence j->j'
            model.addGenConstrIndicator(z[j,j_p], 1, tau[j] + d[j] + slew_time[j,j_p] <= tau[j_p])  

            # enforce lower bound for z[j, j_p]
            if (j_p, j) in z.keys(): # j->j' and j'->j sequences are both valid
                if (min(j, j_p),max(j, j_p)) not in pairs_considered:
                    # at least one of z[j, j_p] or z[j_p, j] must be 1 if x[j] and x[j_p] are assigned
                    model.addConstr(x[j] + x[j_p] - 1 <= z[j, j_p] + z[j_p, j])
                    
                    # enforce mutual exclusivity if (j, j_p) and (j_p, j) are both in z
                    model.addConstr(z[j, j_p] + z[j_p, j] <= 1) 

                    # add pair to considered pairs
                    pairs_considered.add((min(j, j_p),max(j, j_p)))

            else: # only j->j' sequence is valid
                # x[j] and x[j_p] are mutually exclusive unless z[j,j_p] is assigned
                model.addConstr(x[j] + x[j_p] <= 1 + z[j, j_p])        

        # Optimize model
        model.optimize()

        # Print results
        self.__print_model_results(model)

        # Convert decision variables to arrays
        x_array = np.array([int(x[j].X) for j,_ in indexed_tasks if j > 0])
        z_array = np.array([(j,j_p,int(z[j,j_p].X)) for j,j_p in Z if j > 0 and j_p > 0])
        tau_array = np.array([float(tau[j].X)+state.t for j,_ in indexed_tasks if j > 0])

        # Return solved model
        return model, x_array, z_array, tau_array, model.getObjective().getValue()


    def __optimize_observations_schedule_temp(self,
                                     state : SimulationAgentState,
                                     specs : object,
                                     cross_track_fovs : dict,
                                     max_slew_rate : float,
                                     max_torque : float,
                                     orbitdata : OrbitData,
                                     schedulable_tasks : list,
                                     mission : Mission,
                                     observation_history : ObservationHistory,
                                     x_init : np.ndarray = None
                                     ) -> tuple:
        raise NotImplementedError("This method is not implemented yet.")

        # # Check if there are no tasks to schedule
        # if not schedulable_tasks: return None, [], [], [], np.NAN

        # # Add dummy task to represent initial state
        # dummy_task = SpecificObservationTask(set([]),
        #                                      schedulable_tasks[0].instrument_name,
        #                                      Interval(state.t,state.t), 
        #                                      0.0, 
        #                                      Interval(state.attitude[0],state.attitude[0]))
        # tasks : list[SpecificObservationTask] = [dummy_task]
        # tasks.extend(schedulable_tasks)

        # # Create a new model
        # model = gp.Model("single-sat_milp_planner")

        # # Set parameter to suppress output
        # model.setParam('OutputFlag', int(self._debug))

        # # List tasks by their index
        # indexed_tasks = list(enumerate(tasks))
        # task_indices = [j for j,_ in indexed_tasks]

        # # Define constants
        # rewards = np.array([self.estimate_task_value(task, 
        #                                              task.accessibility.left, 
        #                                              task.min_duration, 
        #                                              specs, cross_track_fovs, orbitdata, 
        #                                              mission, 
        #                                              observation_history)
        #                     for task in tqdm(tasks,leave=False,desc='SATELLITE: Calculating task rewards')
        #                     if isinstance(task,SpecificObservationTask)])
        # t_start   = np.array([task.accessibility.left-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
        # t_end     = np.array([task.accessibility.right-state.t for task in tasks if isinstance(task, SpecificObservationTask)])
        # d         = np.array([task.min_duration for task in tasks if isinstance(task, SpecificObservationTask)])
        # th_imgs   = np.array([np.average((task.slew_angles.left, task.slew_angles.right)) for task in tasks if isinstance(task, SpecificObservationTask)])
        # slew_time = np.array([[abs(th_imgs[j_p]-th_imgs[j]) / max_slew_rate 
        #                        for j in task_indices]
        #                        for j_p in task_indices
        #                      ])        
        # # Determine reachable transition pairs
        # Z = [(j, j_p) 
        #      for j      in task_indices 
        #      for j_p    in task_indices
        #      if t_start[j] + d[j] + slew_time[j, j_p] <= t_end[j_p] - d[j_p] # sequence j->j' is feasible
        #      and j != j_p                                                    # ensure distinct tasks
        #      ] 
                
        # # Reduce decision space to only include reachable tasks
        # ## Precalculate preemptive pruning
        # succs = defaultdict(list)
        # for j, j_p in Z: succs[j].append(j_p)

        # ## Determine which tasks are reachable
        # reachable_task_indeces = set([0])
        # dq = deque([0])
        # while dq:
        #     # get source tasks
        #     u = dq.popleft()

        #     # get possible successor tasks
        #     unvisited = list({v for v in succs.get(u, []) if v not in reachable_task_indeces})
            
        #     # add unvisited successors to reachable set and queue
        #     reachable_task_indeces.update(unvisited)
        #     dq.extend(unvisited)

        # ## Convert set of reachable tasks to list
        # # reachable_task_indeces = list(reachable_task_indeces)
        # reachable_task_indeces = [j for j in task_indices if (0,j) in Z or j == 0] # Ensure all tasks directly reachable from initial task are included
        # # reachable_task_indeces = [j for j in task_indices] # Do not change reachable tasks for now

        # ## If nothing except dummy reachable, return empty plan
        # if len(reachable_task_indeces) <= 1:
        #     # no feasible tasks
        #     return None, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.NAN

        # ## Build constants for the kept tasks
        # tasks_reachable : list[SpecificObservationTask] = [tasks[old] for old in reachable_task_indeces]

        # ## Remove unreachable tasks from Z
        # Z_reachable = [(i_old,j_old) 
        #                for i_old,j_old in Z 
        #                if i_old in reachable_task_indeces 
        #                and j_old in reachable_task_indeces
        #             ]

        # ## create successor and predecessor mappings for reachable tasks
        # succs_comp = defaultdict(list)
        # preds_comp = defaultdict(list)
        # for j, j_p in Z_reachable:
        #     succs_comp[j].append(j_p)
        #     preds_comp[j_p].append(j)

        # ## Calculate exclusivity
        # E_reachable = [(reachable_task_indeces[j],reachable_task_indeces[j_p]) 
        #                 for j,task_j       in enumerate(tasks_reachable) 
        #                 for j_p,task_j_p   in enumerate(tasks_reachable)
        #                 if (task_j.is_mutually_exclusive(task_j_p)  # mutual exclusivity constraint
        #                     or (
        #                         t_start[j] + d[j] + slew_time[j, j_p] > t_end[j_p] - d[j_p]
        #                     and  t_start[j_p] + d[j_p] + slew_time[j_p, j] > t_end[j] - d[j]
        #                 ))                                          # sequence j->j' and j'->j are not feasible
        #                 and reachable_task_indeces[j] < reachable_task_indeces[j_p]         # ensure distinct tasks and non-repeating pairs
        # ]

        # # Create decision variables
        # x : gp.tupledict = model.addVars(reachable_task_indeces, vtype=gp.GRB.BINARY, name="x")
        # z : gp.tupledict = model.addVars(Z_reachable, vtype=gp.GRB.BINARY, name="z")
        # tau : gp.tupledict = model.addVars(reachable_task_indeces, vtype=gp.GRB.CONTINUOUS, name="tau")
        # assert all([j in x.keys() and j_p in x.keys() for j,j_p in Z_reachable])
        # assert all([j in x.keys() and j_p in x.keys() for j,j_p in E_reachable])

        # ## if initial solution is provided, use it
        # if x_init is not None:
        #     x_init = np.insert(x_init, 0, 1) # compensate for missing initial task
        #     assert len(x_init) == len(x), "Initial solution length does not match number of tasks."

        #     # initialize values of `x` and `tau`
        #     for j in x.keys():
        #         x[j].Start = x_init[j]
        #         tau[j].Start = float(t_start[j])

        #     # initialize values of `z`
        #     for j,j_p in Z_reachable:
        #         if x_init[j] > 0.5 and x_init[j_p] > 0.5 and j != j_p:
        #             z[j, j_p].Start = int(t_start[j] + d[j] + slew_time[j, j_p] <= t_start[j_p])
        
        # # Set objective
        # if self.objective == "reward": 
        #     model.setObjective(gp.quicksum( rewards[j] * x[j] for j in reachable_task_indeces), gp.GRB.MAXIMIZE)
        # # TODO else: add alternative objectives
  
        # # Add constraints
        # ## Always assign first observation
        # model.addConstr(x[0] == 1)
        # model.addConstr(tau[0] == 0.0)
        
        # ## Observation time and accessibility constraints
        # for j in tqdm(x.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding observation time constraints", unit='tasks', leave=False):
        #     tau[j].LB = t_start[j]
        #     tau[j].UB = t_end[j] - d[j]
        #     # If x[j] == 0, force tau to LB (or any fixed value in the feasible interval)
        #     model.addGenConstrIndicator(x[j], 0, tau[j] == float(t_start[j]))

        # ## Enforce exclusivity
        # for j,j_p in tqdm(E_reachable, desc=f"{state.agent_name}/PREPLANNER: Adding exclusivity constraints", unit='task pairs', leave=False):
        #     model.addConstr(x[j] + x[j_p] <= 1)

        # ## Slew time constraints for observation task sequence j->j'
        # pairs_considered : set = set()
        # for j,j_p in tqdm(z.keys(), desc=f"{state.agent_name}/PREPLANNER: Adding slewing constraints", unit='task pairs', leave=False):
           
        #     # If z[j, j_p] == 1, enforce slew constraint for task sequence j->j'
        #     model.addGenConstrIndicator(z[j,j_p], 1, tau[j] + d[j] + slew_time[j,j_p] <= tau[j_p])

        #     # enforce lower bound for z[j, j_p]
        #     if (j_p, j) in z.keys(): # j->j' and j'->j sequences are both valid
        #         if (min(j, j_p),max(j, j_p)) not in pairs_considered:
        #             # at least one of z[j, j_p] or z[j_p, j] must be 1 if x[j] and x[j_p] are assigned
        #             model.addConstr(x[j_p] + x[j] - 1 <= z[j, j_p] + z[j_p, j])
                    
        #             # enforce mutual exclusivity if (j, j_p) and (j_p, j) are both in z
        #             model.addConstr(z[j, j_p] + z[j_p, j] <= 1) 

        #             # add pair to considered pairs
        #             pairs_considered.add((min(j, j_p),max(j, j_p)))

        #     else: # only j->j' sequence is valid
        #         # x[j] and x[j_p] are mutually exclusive unless z[j,j_p] is assigned
        #         model.addConstr(x[j] + x[j_p] <= 1 + z[j, j_p])                

        #     # # enforce upper bound for z[j, j_p]; x[j] and x[j_p] must be assigned if z[j, j_p] is to be assigned
        #     # model.addConstr(z[j, j_p] <= x[j])
        #     # model.addConstr(z[j, j_p] <= x[j_p])

        ## Build clique cover for exclusivity (greedy) to reduce constraints (inside a clique all pairs are mutually exclusive).
        # exclusivity_sets = defaultdict(set)
        # for j,j_p in E_reachable:
        #     exclusivity_sets[j].add(j_p)
        #     exclusivity_sets[j_p].add(j)

        # cliques : list[list[int]] = []
        # for node in reachable_task_indeces:
        #     placed = False
        #     for clique in cliques:
        #         # check if node is exclusive to all members in clique (i.e., node cannot coexist with members)
        #         if all((member in exclusivity_sets[node]) for member in clique):
        #             clique.append(node)
        #             placed = True
        #             break
        #     if not placed and node in exclusivity_sets and exclusivity_sets[node]:
        #         cliques.append([node])

        # ## Observation time and accessibility constraints
        # for j in x.keys():
        #     lb = float(t_start[j])
        #     ub = float(t_end[j] - d[j])
        #     if ub < lb:
        #         # this task cannot be scheduled; force off
        #         model.addConstr(x[j] == 0, name=f"unsched_{j}")
        #         model.addConstr(tau[j] == lb, name=f"tau_fix_{j}")
        #         continue
        #     tau[j].LB = lb
        #     tau[j].UB = ub
        #     # if x[j] == 0, force tau to lb to avoid free floats
        #     model.addGenConstrIndicator(x[j], 0, tau[j] == lb)

        # ## For source (dummy index 0), require exactly one outgoing arc from source (one path start)
        # source_outgoing = [ (j,j_p) for (j,j_p) in Z_reachable if j==0 ]
        # model.addConstr(gp.quicksum(z[j,j_p] for (j,j_p) in source_outgoing) <= 1, name="source_out_deg")
        
        # # For all non-source nodes, incoming == x and outgoing == x
        # for j in x.keys():
        #     if j == 0: continue

        #     incoming = [ (i,j) for i in preds_comp.get(j, []) ]
        #     outgoing = [ (j,o) for o in succs_comp.get(j, []) ]
        #     # equality constraints: sum incoming == x[j] and sum outgoing == x[j]
        #     model.addConstr(gp.quicksum(z[i,j] for (i,j) in incoming) == x[j], name=f"in_deg_{j}")
        #     model.addConstr(gp.quicksum(z[j,k] for (j,k) in outgoing) == x[j], name=f"out_deg_{j}")

        # # indicator arc timing constraints and linking z->x (redundant with degrees, but keep for clarity)
        # for (j, j_p) in Z_reachable:
        #     model.addGenConstrIndicator(z[j, j_p], 1, tau[j] + d[j] + float(slew_time[j, j_p]) <= tau[j_p], name=f"prec_{j}_{j_p}")
        #     # linking (redundant if degrees used): z <= x[i], z <= x[j]
        #     model.addConstr(z[j, j_p] <= x[j], name=f"z_le_x_j_{j}_{j_p}")
        #     model.addConstr(z[j, j_p] <= x[j_p], name=f"z_le_x_j_p_{j}_{j_p}")

        # # exclusivity via clique constraints (if cliques found) else fallback to pairwise
        # if cliques:
        #     for c_idx, clique in enumerate(cliques):
        #         if len(clique) > 1:
        #             model.addConstr(gp.quicksum(x[j] for j in clique) <= 1, name=f"excl_clique_{c_idx}")
        #     # also include any remaining pairwise exclusivity not covered by cliques
        #     covered = set()
        #     for clique in cliques:
        #         for v in clique:
        #             covered.add(v)
        #     # fallback add pairwise for any exclusivity not covered
        #     for (j,j_p) in E_reachable:
        #         if j not in covered or j_p not in covered:
        #             model.addConstr(x[j] + x[j_p] <= 1, name=f"excl_pair_{j}_{j_p}")
        # else:
        #     # no cliques computed; just add pairwise constraints
        #     for (j,j_p) in E_reachable:
        #         model.addConstr(x[j] + x[j_p] <= 1, name=f"excl_pair_{j}_{j_p}")

        # Optimize model
        model.optimize()
    
        # Print results
        self.__print_model_results(model)

        # Convert decision variables to arrays
        x_array = np.array([int(x[j].X) if j in reachable_task_indeces else 0
                            for j,_ in indexed_tasks if j > 0])
        z_array = np.array([(j,j_p,int(z[j,j_p].X)) if (j,j_p) in Z_reachable else (j,j_p,0)
                            for j,j_p in Z if j > 0 and j_p > 0])
        tau_array = np.array([float(tau[j].X)+state.t if j in reachable_task_indeces else t_start[j]+state.t
                              for j,_ in indexed_tasks if j > 0])
           
        # Return solved model
        return model, x_array, z_array, tau_array, model.getObjective().getValue()
        
    def __print_model_results(self, model : gp.Model) -> None:
        """ Prints model results to console and outputs error logs in case of infeasibility """

        # Print model status
        print("\nStatus code:", model.Status)

        # Check model status
        if model.Status == gp.GRB.OPTIMAL:
            print("Optimal solution found.")
            print(f"Obj: {model.ObjVal:g}")

            # delete any `.ilp` files that may have been generated when debugging the model
            class_path = inspect.getfile(type(self))
            class_path = class_path.replace('milp.py', '')
            for filename in os.listdir(class_path):
                if ".ilp" in filename:
                    os.remove(os.path.join(class_path, filename))
        else:
            if model.Status == gp.GRB.INFEASIBLE:
                print("Model is infeasible.")
            
                # diagnose problems with model
                print("Computing IIS...")
                model.computeIIS()

                # print results to file
                filepath = f"./chess3d/agents/planning/preplanners/decentralized/{model.ModelName}.ilp"
                model.write(filepath)
                model.write(f"{filepath}.json")
                print(f"IIS written to `{filepath}`")

            if model.Status == gp.GRB.UNBOUNDED:
                print("Model is unbounded.")

            if model.Status == gp.GRB.INTERRUPTED:
                print("Model was interrupted.")

            raise ValueError(f"Unexpected model status: {model.Status}")
        
        return

    def __extract_observation_sequence(self, schedulable_tasks : list, state : SimulationAgentState, x : gp.tupledict, tau : gp.tupledict, d : np.array, th_imgs : np.array) -> list:
        """ Generates a list of observations from the outputs of MILP solver"""

        # Convert model outputs into observation actions
        observations = [ObservationAction(task.instrument_name,
                              th_imgs[j],
                              tau[j],
                              d[j],
                              task
                              )
                        for j,task in enumerate(schedulable_tasks) 
                        if x[j] > 0.5            # select only assigned tasks
                        ]

        # Return sorted observations in start time
        return sorted(observations, key=lambda obs: obs.t_start)

    def _schedule_broadcasts(self, state, observations, orbitdata, t = None):
        # TODO 
        return super()._schedule_broadcasts(state, observations, orbitdata, t)