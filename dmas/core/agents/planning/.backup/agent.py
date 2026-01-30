# class RealtimeAgent(AbstractAgent):
#     """
#     Implements 
#     """

#     def __init__(self, 
#                  agent_name, 
#                  results_path, 
#                  agent_network_config, 
#                  manager_network_config, 
#                  initial_state, 
#                  specs, 
#                  mission : Mission,
#                  planning_module : InternalModule = None,
#                  science_module : InternalModule = None,
#                  level=logging.INFO, 
#                  logger=None):
        
#         # load agent modules
#         modules = []
#         if planning_module is not None:
#             if not isinstance(planning_module, PlanningModule):
#                 raise AttributeError(f'`planning_module` must be of type `PlanningModule`; is of type {type(planning_module)}')
#             modules.append(planning_module)
#         if science_module is not None:
#             if not isinstance(science_module, ScienceModule):
#                 raise AttributeError(f'`science_module` must be of type `ScienceModule`; is of type {type(science_module)}')
#             modules.append(science_module)

#         super().__init__(agent_name, results_path, agent_network_config, manager_network_config, initial_state, specs, modules, mission, level, logger)

#     @runtime_tracker
#     async def think(self, senses: list) -> list:
#         # send all sensed messages to planner
#         self.log(f'sending {len(senses)} senses to planning module...', level=logging.DEBUG)
#         senses_dict = []
#         state_dict = None
#         for sense in senses:
#             sense : SimulationMessage
#             if isinstance(sense, AgentStateMessage):
#                 state_dict = sense.to_dict()
#             else:
#                 senses_dict.append(sense.to_dict())

#         senses_msg = SenseMessage( self.get_element_name(), 
#                                     self.get_element_name(),
#                                     state_dict, 
#                                     senses_dict)
#         await self.send_internal_message(senses_msg)

#         # wait for planner to send list of tasks to perform
#         self.log(f'senses sent! waiting on response from planner module...')
#         actions = []
        
#         while True:
#             _, _, content = await self.internal_inbox.get()
            
#             if content['msg_type'] == SimulationMessageTypes.PLAN.value:
#                 msg = PlanMessage(**content)

#                 # assert self.get_current_time() - msg.t_plan <= 1e-3

#                 for action_dict in msg.plan:
#                     self.log(f"received an action of type {action_dict['action_type']}", level=logging.DEBUG)
#                     actions.append(action_dict)  
#                 break
        
#         self.log(f"plan of {len(actions)} actions received from planner module!")
#         return actions