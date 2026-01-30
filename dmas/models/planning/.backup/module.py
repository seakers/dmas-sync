import copy
import os
import pandas as pd

from orbitpy.util import Spacecraft

from dmas.modules import *
from dmas.utils import runtime_tracker

from chess3d.agents.planning.plan import Plan, PeriodicPlan
from chess3d.agents.planning.periodic import AbstractPeriodicPlanner
from chess3d.agents.planning.reactive import AbstractReactivePlanner
from chess3d.orbitdata import OrbitData
from chess3d.agents.states import *
from chess3d.agents.science.requests import *
from chess3d.messages import *

class PlanningModule(InternalModule):
    def __init__(self, 
                results_path : str, 
                parent_agent_specs : object,
                parent_network_config: NetworkConfig, 
                preplanner : AbstractPeriodicPlanner = None,
                replanner : AbstractReactivePlanner = None,
                orbitdata : OrbitData = None,
                level: int = logging.INFO, 
                logger: logging.Logger = None
                ) -> None:
        """ Internal Agent Module in charge of planning an scheduling. """
                       
        # intialize module
        if isinstance(parent_agent_specs, dict):
            parent_name : str = parent_agent_specs.get('name', None)
        elif isinstance(parent_agent_specs, Spacecraft):
            parent_name = parent_agent_specs.name
        else:
            raise NotImplementedError('Data type for `parent_agent_specs` not yet supported.')

        # setup network settings
        planner_network_config : NetworkConfig = self._setup_planner_network_config(parent_name, parent_network_config)
        
        super().__init__(f'{parent_name}-PLANNING_MODULE', 
                        planner_network_config, 
                        parent_network_config, 
                        level, 
                        logger)
        
        # initialize default attributes
        self.plan_history = []
        self.stats = {
                    }
        self.agent_state : SimulationAgentState = None
        self.other_modules_exist : bool = False

        # set attributes
        self.results_path = results_path
        self.parent_agent_specs = parent_agent_specs
        self.parent_name = parent_name
        self.preplanner : AbstractPeriodicPlanner = preplanner
        self.replanner : AbstractReactivePlanner = replanner
        self.orbitdata : OrbitData = orbitdata

    def _setup_planner_network_config(self, parent_name : str, parent_network_config : NetworkConfig) -> dict:
        """ Sets up network configuration for intra-agent module communication """
        # get full list of addresses from parent agent
        addresses : dict = parent_network_config.get_internal_addresses()        
        
        # subscribe to parent agent's bradcasts
        sub_address : str = addresses.get(zmq.PUB)[0]
        sub_addesses = [ sub_address.replace('*', 'localhost') ]

        # subscribe to science module's broadcasts
        if len(addresses.get(zmq.SUB)) > 1:
            sub_address : str = addresses.get(zmq.SUB)[1]
            sub_addesses.append( sub_address.replace('*', 'localhost') )

        # obtain address for intra-agent broadcasts
        pub_address : str = addresses.get(zmq.SUB)[0]
        pub_address = pub_address.replace('localhost', '*')

        # obtain results logger address
        addresses = parent_network_config.get_manager_addresses()
        push_address : str = addresses.get(zmq.PUSH)[0]

        # return network configuration
        return  NetworkConfig(  parent_name,
                                manager_address_map = {
                                    zmq.REQ: [],
                                    zmq.SUB: sub_addesses,
                                    zmq.PUB: [pub_address],
                                    zmq.PUSH: [push_address]
                                }
                            )

    async def setup(self) -> None:
        # initialize internal messaging queues
        self.states_inbox = asyncio.Queue()
        self.action_status_inbox = asyncio.Queue()
        self.req_inbox = asyncio.Queue()
        self.observations_inbox = asyncio.Queue()
        self.relay_inbox = asyncio.Queue()
        self.misc_inbox = asyncio.Queue()

        # setup agent state locks
        self.agent_state_lock = asyncio.Lock()
        self.agent_state_updated = asyncio.Event()

    async def sim_wait(self, delay: float) -> None:
        # does nothing; planner module is designed to be event-driven, not time-driven
        return

    async def live(self) -> None:
        """
        Performs two concurrent tasks:
        - Listener: receives messages from the parent agent and checks results
        - Planner: plans and bids according to local information
        """
        try:
            listener_task = asyncio.create_task(self.listener(), name='listener()')
            planner_task = asyncio.create_task(self.planner(), name='planner()')
            
            tasks : list[asyncio.Task] = [listener_task, planner_task]

            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    await task           

    """
    -----------------
        LISTENER
    -----------------
    """
    async def listener(self) -> None:
        """
        Listens for any incoming messages, unpacks them and classifies them into 
        internal inboxes for future processing
        """
        try:
            # listen for broadcasts and place in the appropriate inboxes
            while True:
                # wait for next message from the parent agent
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                # classify mesasge content based of type of message
                if content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    # sim-end message; end agent `live()`
                    self.log(f"received manager broadcast or type {content['msg_type']}! terminating `live()`...")
                    return

                elif content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    # received agent sense message
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SenseMessage = SenseMessage(**content)

                    # sort agent senses to be processed
                    senses = [senses_msg.state]
                    senses.extend(senses_msg.senses)     

                    # process agent senses
                    for sense in senses:
                        # unpack message
                        msg : SimulationMessage = message_from_dict(**sense)

                        # place in correct inbox
                        await self.categorize_messages(msg)

                else:
                    if not self.other_modules_exist:
                        # another type of message was received from another module 
                        self.other_modules_exist = True
                        
                    if content['msg_type'] == 'HANDSHAKE':
                        pass

                    elif content['msg_type'] == SimulationMessageTypes.BUS.value:
                        # received bus message containing other messages
                        self.log(f"received bus message!", level=logging.DEBUG)

                        # unpack bus message 
                        bus_msg : BusMessage = BusMessage(**content)

                        # process agent senses
                        for msg_dict in bus_msg.msgs:
                            # unpack message
                            msg : SimulationMessage = message_from_dict(**msg_dict)

                            # send to planner
                            await self.internal_inbox.put(msg)
                    else:
                        # received an internal message
                        self.log(f"received bus message!", level=logging.DEBUG)

                        # unpack message
                        msg : SimulationMessage = message_from_dict(**content)

                        # send to planner
                        await self.internal_inbox.put(msg)

        except asyncio.CancelledError:
            return
        
    async def categorize_messages(self, msg : SimulationMessage) -> None:
        # check relay path
        if msg.path:
            # message contains relay path; forward message copy
            await self.relay_inbox.put(copy.deepcopy(msg))

        # check type of message being received
        if isinstance(msg, AgentActionMessage):
            # agent action received
            self.log(f"received agent action of status {msg.status}!")
            
            # send to planner
            await self.action_status_inbox.put(msg)

        elif isinstance(msg, AgentStateMessage):
            # agent state received
            self.log(f"received agent state message!")
            
            # unpack state
            state : SimulationAgentState = SimulationAgentState.from_dict(msg.state)                                                          
            
            # send to planner
            await self.states_inbox.put(state)

        elif isinstance(msg, MeasurementRequestMessage):
            # measurement request message received
            self.log(f"received measurement request message!")

            # unapack measurement request
            req : TaskRequest = TaskRequest.from_dict(msg.req)
            
            # send to planner
            await self.req_inbox.put(req)

        elif isinstance(msg, ObservationResultsMessage):
            # observation data from another agent was received
            self.log(f"received observation data from agent!")

            # send to planner
            await self.observations_inbox.put(msg)

        # TODO support down-linked information processing
        ## elif isinstance(msg, DOWNLINKED MESSAGE CONFIRMATION):
        ##     pass

        else:
            # other type of message was received
            self.log(f"received some other kind of message!")

            # send to planner
            await self.misc_inbox.put(msg)
        
    """
    -----------------
         PLANNER
    -----------------
    """
    async def planner(self) -> None:
        """
        Generates a plan for the parent agent to perform
        """
        try:
            # initialize plan
            plan : Plan = PeriodicPlan(t=-1.0)

            # level = logging.WARNING
            level = logging.DEBUG

            while True:
                # wait for agent to update state
                state : SimulationAgentState = await self.states_inbox.get()
                
                t_0 = time.perf_counter()

                # update internal clock
                await self.update_current_time(state.t)
                assert abs(self.get_current_time() - state.t) <= 1e-2

                # --- Check incoming information ---
                # Read incoming messages
                incoming_reqs, relay_messages, misc_messages \
                    = await self._read_incoming_messages()
                
                t_1_5 = time.perf_counter()

                # check action completion
                completed_actions, aborted_actions, pending_actions \
                    = await self.__check_action_completion(level)
                
                # remove aborted or completed actions from plan
                self.update_action_completion(plan, completed_actions, aborted_actions, pending_actions)
                
                # compile measurements performed by myself or other agents
                observations = {action for action in completed_actions
                                if isinstance(action, ObservationAction)}
                observations.update({action_from_dict(**msg.observation_action) 
                                    for msg in misc_messages
                                    if isinstance(msg, ObservationPerformedMessage)})

                t_1 = time.perf_counter()

                # update reward grid
                if self.reward_grid: self.reward_grid.update(self.get_current_time(), observations, incoming_reqs)
                
                t_2 = time.perf_counter()

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_actions(completed_actions, aborted_actions, pending_actions)
                # -------------------------------------
                
                # --- Create plan ---
                if self.preplanner is not None:
                    # there is a preplanner assigned to this planner

                    # update preplanner precepts
                    self.preplanner.update_percepts(state,
                                                    plan, 
                                                    incoming_reqs,
                                                    relay_messages,
                                                    misc_messages,
                                                    completed_actions,
                                                    aborted_actions,
                                                    pending_actions
                                                )
                    
                    # check if there is a need to construct a new plan
                    if self.preplanner.needs_planning(state, 
                                                      self.parent_agent_specs, 
                                                      plan):  
                        # initialize plan      
                        plan : Plan = self.preplanner.generate_plan(state, 
                                                                    self.parent_agent_specs,
                                                                    self.reward_grid,
                                                                    self._clock_config,
                                                                    self.orbitdata
                                                                    )

                        # save copy of plan for post-processing
                        plan_copy = [action for action in plan]
                        self.plan_history.append((state.t, plan_copy))
                        
                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        # self.__log_plan(plan, "PRE-PLAN", logging.WARNING)
                        # x = 1 # breakpoint
                        # -------------------------------------

                t_3 = time.perf_counter()

                # --- Modify plan ---
                # Check if reeplanning is needed
                if self.replanner is not None:
                    # there is a replanner assigned to this planner

                    # update replanner precepts
                    self.replanner.update_percepts( state,
                                                    plan, 
                                                    incoming_reqs,
                                                    relay_messages,
                                                    misc_messages,
                                                    completed_actions,
                                                    aborted_actions,
                                                    pending_actions
                                                )
                    
                    if self.replanner.needs_planning(state, 
                                                     self.parent_agent_specs,
                                                     plan,
                                                     self.orbitdata):    
                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        # self.__log_plan(plan, "ORIGINAL PLAN", logging.WARNING)
                        # x = 1 # breakpoint
                        # -------------------------------------

                        # Modify current Plan      
                        plan : Plan = self.replanner.generate_plan(state, 
                                                                   self.parent_agent_specs,    
                                                                   self.reward_grid,
                                                                   plan,
                                                                   self._clock_config,
                                                                   self.orbitdata
                                                                   )

                        # update last time plan was updated
                        self.t_plan = self.get_current_time()

                        # save copy of plan for post-processing
                        plan_copy = [action for action in plan]
                        self.plan_history.append((self.t_plan, plan_copy))
                    
                        # clear pending actions
                        pending_actions = []

                        # --- FOR DEBUGGING PURPOSES ONLY: ---
                        # self.__log_plan(plan, "REPLAN", logging.WARNING)
                        # x = 1 # breakpoint
                        # -------------------------------------

                t_4 = time.perf_counter()

                # --- Execute plan ---
                # get next action to perform
                plan_out : list = self.get_plan_out(plan)

                # --- FOR DEBUGGING PURPOSES ONLY: ---
                # self.__log_plan(plan_out, "PLAN OUT", logging.WARNING)
                # x = 1 # breakpoint
                # -------------------------------------

                # send plan to parent agent
                self.log(f'sending {len(plan_out)} actions to parent agent...')
                plan_msg = PlanMessage(self.get_element_name(), self.get_network_name(), plan_out, self.get_current_time())
                await self._send_manager_msg(plan_msg, zmq.PUB)

                self.log(f'actions sent!')

                t_5 = time.perf_counter()

                # log runtime
                if 'planner_s1' not in self.stats: self.stats['planner_s1'] = []
                self.stats['planner_s1'].append(t_1_5-t_0)
                if 'planner_s1_5' not in self.stats: self.stats['planner_s1_5'] = []
                self.stats['planner_s1_5'].append(t_1-t_1_5)
                if 'planner_s2' not in self.stats: self.stats['planner_s2'] = []
                self.stats['planner_s2'].append(t_2-t_1)
                if 'planner_s3' not in self.stats: self.stats['planner_s3'] = []
                self.stats['planner_s3'].append(t_3-t_2)
                if 'planner_s4' not in self.stats: self.stats['planner_s4'] = []
                self.stats['planner_s4'].append(t_4-t_3)
                if 'planner_s5' not in self.stats: self.stats['planner_s5'] = []
                self.stats['planner_s5'].append(t_5-t_4)

                dt_0 = t_5 - t_0
                if 'planner' not in self.stats: self.stats['planner'] = []
                self.stats['planner'].append(dt_0)

        except asyncio.CancelledError:
            return
        
        except Exception as e:
            # traceback.format_exc()
            print(e)
            raise e
                
    @runtime_tracker
    async def _read_incoming_messages(self) -> tuple:
        """
        Collects and classifies incoming messages. If a measurement was made by the parent agent,
        it waits until other modules to process the measurement's data and send a response.

        ## Returns:
            incoming_reqs 
            generated_reqs 
            misc_messages
        """
        # compile incoming observations
        incoming_obsevations = []
        while not self.observations_inbox.empty():
            incoming_obsevations.append(await self.observations_inbox.get())

        # compile incoming requests
        requests = []
        while not self.req_inbox.empty():
            requests.append(await self.req_inbox.get())

        # process internal messages
        if self.other_modules_exist:                
            # check if the parent agent just performed an observation
            while len(incoming_obsevations) > 0:    
                # wait for science module to send their assesment of the observation 
                internal_msg = await self.internal_inbox.get()

                # check the type of response from the science module
                if isinstance(internal_msg, MeasurementRequestMessage):
                    # the science module analized out latest observation(s)
                    
                    # check if an outlier was deteced
                    if internal_msg.req['severity'] > 0.0:
                        # event was detected and an observation was requested
                        requests.append(TaskRequest.from_dict(internal_msg.req))

                else:
                    # the science module generated a different response; process later
                    await self.misc_inbox.put(internal_msg)

                # check for if there are more responses from the science module
                if self.internal_inbox.empty():
                    # no more messages from the science module; stop wait
                    break
            
            # process internal messages
            while not self.internal_inbox.empty():
                # get next internal message
                internal_msg = await self.internal_inbox.get()

                # classify message
                if isinstance(internal_msg, MeasurementRequestMessage):
                    # the science module analized out latest observation(s)

                    # check if an outlier was deteced
                    if internal_msg.req['severity'] > 0.0:
                        # event was detected and an observation was requested
                        requests.append(TaskRequest.from_dict(internal_msg.req))
                        
                    # request : MeasurementRequest = MeasurementRequest.from_dict(internal_msg.req)
                    
                    # # check if outlier was deteced
                    # if request.severity > 0.0:
                    #     # event was detected and an observation was requested
                    #     requests.append(request)

                else:
                    # the science module generated a different response; process later
                    await self.misc_inbox.put(internal_msg)

        # compile incoming relay messages
        relay_messages= []
        while not self.relay_inbox.empty():
            relay_messages.append(await self.relay_inbox.get())

        # compile miscellaneous messages
        misc_messages = []
        while not self.misc_inbox.empty():
            misc_messages.append(await self.misc_inbox.get())

        # return classified messages
        return requests, relay_messages, misc_messages

    @runtime_tracker
    async def __check_action_completion(self, level : int = logging.DEBUG) -> tuple:
        """
        Checks incoming messages from agent to check which actions from its plan have been completed, aborted, or are still pending

        ### Arguments:
            - level (`int`): logging level

        ### Returns:
            - `tuple` of action lists `completed_actions, aborted_actions, pending_actions`
        """

        # collect all action statuses from inbox
        actions = []
        while not self.action_status_inbox.empty():
            action_msg : AgentActionMessage = await self.action_status_inbox.get()
            actions.append(action_from_dict(**action_msg.action))

        # classify by action completion
        completed_actions = [action for action in actions
                            if isinstance(action, AgentAction)
                            and action.status == AgentAction.COMPLETED] # planned action completed
                
        aborted_actions = [action for action in actions 
                           if isinstance(action, AgentAction)
                           and action.status == AgentAction.ABORTED]    # planned action aborted
                
        pending_actions = [action for action in actions
                           if isinstance(action, AgentAction)
                           and action.status == AgentAction.PENDING] # planned action wasn't completed

        # return classified lists
        return completed_actions, aborted_actions, pending_actions
    
    @runtime_tracker
    def update_action_completion(self, 
                                 plan : Plan, 
                                 completed_actions : list, 
                                 aborted_actions : list, 
                                 pending_actions : list
                                 ) -> None:
        plan.update_action_completion(  completed_actions, 
                                        aborted_actions, 
                                        pending_actions, 
                                        self.get_current_time()
                                    )             
        
    @runtime_tracker
    def get_plan_out(self, plan : Plan) -> list:
        return plan.get_next_actions(self.get_current_time())
    
    def __log_actions(self, completed_actions : list, aborted_actions : list, pending_actions : list) -> None:
        all_actions = [action for action in completed_actions]
        all_actions.extend([action for action in aborted_actions])
        all_actions.extend([action for action in pending_actions])

        if len(all_actions) > 0:
            out = '\nACTION STATUS:\n'
            for action in all_actions:
                action : AgentAction
                out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
            out += '\n'
            self.log(out, logging.WARNING)

    def __log_plan(self, plan : Plan, title : str, level : int = logging.DEBUG) -> None:
        try:
            out = f'\n{title}\n'

            if isinstance(plan, Plan):
                out += str(plan)
            else:
                for action in plan:
                    if isinstance(action, AgentAction):
                        out += f"{action.id.split('-')[0]}, {action.action_type}, {action.t_start}, {action.t_end}\n"
                    elif isinstance(action, dict):
                        out += f"{action['id'].split('-')[0]}, {action['action_type']}, {action['t_start']}, {action['t_end']}\n"
            

            self.log(out, level)
        except Exception as e:
            print(e)
            raise e
               
    async def teardown(self) -> None:
        # log plan history
        headers = ['plan_index', 't_plan', 'desc', 't_start', 't_end']
        data = []
        
        for i in range(len(self.plan_history)):
            t_plan, plan = self.plan_history[i]
            t_plan : float; plan : list[AgentAction]

            for action in plan:
                desc = f'{action.action_type}'
                if isinstance(action, ObservationAction):
                    desc += f'_{action.instrument_name}'
                    
                line_data = [   i,
                                np.round(t_plan,3),
                                desc,
                                np.round(action.t_start,3 ),
                                np.round(action.t_end,3 )
                            ]
                data.append(line_data)

        df = pd.DataFrame(data, columns=headers)
        # self.log(f'\nPLANNER HISTORY\n{str(df)}\n', level=logging.WARNING)
        df.to_csv(f"{self.results_path}/{self.get_parent_name()}/planner_history.csv", index=False)

        # log reward grid history
        if self.reward_grid is not None:
            headers = ['t_update','grid_index','GP index','lat [deg]', 'log [deg]','instrument','reward','n_observations','n_events']
            data = self.reward_grid.get_history()
            df = pd.DataFrame(data, columns=headers)
            # self.log(f'\nREWARD GRID HISTORY\n{str(df)}\n', level=logging.DEBUG)
            df.to_csv(f"{self.results_path}/{self.get_parent_name()}/reward_grid_history.csv", index=False)

        # log performance stats
        n_decimals = 5
        headers = ['routine','t_avg','t_std','t_med', 't_max', 't_min', 'n', 't_total']
        data = []

        for routine in self.stats:
            n = len(self.stats[routine])
            t_avg = np.round(np.mean(self.stats[routine]),n_decimals) if n > 0 else -1
            t_std = np.round(np.std(self.stats[routine]),n_decimals) if n > 0 else 0.0
            t_median = np.round(np.median(self.stats[routine]),n_decimals) if n > 0 else -1
            t_max = np.round(max(self.stats[routine]),n_decimals) if n > 0 else -1
            t_min = np.round(min(self.stats[routine]),n_decimals) if n > 0 else -1
            t_total = t_avg * n

            line_data = [ 
                            routine,
                            t_avg,
                            t_std,
                            t_median,
                            t_max,
                            t_min,
                            n,
                            t_total
                            ]
            data.append(line_data)

            # save time-series
            time_series = [[v] for v in self.stats[routine]]
            routine_df = pd.DataFrame(data=time_series, columns=['dt'])
            routine_dir = os.path.join(f"{self.results_path}/{self.get_parent_name()}/runtime", f"time_series-planner_{routine}.csv")
            routine_df.to_csv(routine_dir,index=False)

        if isinstance(self.preplanner, AbstractPeriodicPlanner):
            for routine in self.preplanner.stats:
                n = len(self.preplanner.stats[routine])
                t_avg = np.round(np.mean(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.preplanner.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                t_max = np.round(max(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                t_min = np.round(min(self.preplanner.stats[routine]),n_decimals) if n > 0 else -1
                t_total = t_avg * n

                line_data = [ 
                                f"preplanner/{routine}",
                                t_avg,
                                t_std,
                                t_median,
                                t_max,
                                t_min,
                                n,
                                t_total
                                ]
                data.append(line_data)

                # save time-series
                time_series = [[v] for v in self.preplanner.stats[routine]]
                routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                routine_dir = os.path.join(f"{self.results_path}/{self.get_parent_name()}/runtime", f"time_series-preplanner_{routine}.csv")
                routine_df.to_csv(routine_dir,index=False)

        if isinstance(self.replanner, AbstractReactivePlanner):
            for routine in self.replanner.stats:
                n = len(self.replanner.stats[routine])
                t_avg = np.round(np.mean(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.replanner.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                t_max = np.round(max(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                t_min = np.round(min(self.replanner.stats[routine]),n_decimals) if n > 0 else -1
                t_total = t_avg * n

                line_data = [ 
                                f"replanner/{routine}",
                                t_avg,
                                t_std,
                                t_median,
                                t_max,
                                t_min,
                                n,
                                t_total
                                ]
                data.append(line_data)

                # save time-series
                time_series = [[v] for v in self.replanner.stats[routine]]
                routine_df = pd.DataFrame(data=time_series, columns=['dt'])
                routine_dir = os.path.join(f"{self.results_path}/{self.get_parent_name()}/runtime", f"time_series-replanner_{routine}.csv")
                routine_df.to_csv(routine_dir,index=False)

        if self.reward_grid:
            for routine in self.reward_grid.stats:
                n = len(self.reward_grid.stats[routine])
                t_avg = np.round(np.mean(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                t_std = np.round(np.std(self.reward_grid.stats[routine]),n_decimals) if n > 0 else 0.0
                t_median = np.round(np.median(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                t_max = np.round(max(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                t_min = np.round(min(self.reward_grid.stats[routine]),n_decimals) if n > 0 else -1
                t_total = t_avg * n

                line_data = [ 
                                f"reward_grid/{routine}",
                                t_avg,
                                t_std,
                                t_median,
                                t_max,
                                t_min,
                                n,
                                t_total
                                ]
                data.append(line_data)

        stats_df = pd.DataFrame(data, columns=headers)
        # self.log(f'\nPLANNER RUN-TIME STATS\n{str(stats_df)}\n', level=logging.WARNING)
        # self.log(f'total: {sum(stats_df["t_total"])}', level=logging.WARNING)
        stats_df.to_csv(f"{self.results_path}/{self.get_parent_name()}/planner_runtime_stats.csv", index=False)

        await super().teardown()
