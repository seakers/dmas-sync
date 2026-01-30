from typing import Union
import numpy as np
import uuid

from execsatm.tasks import GenericObservationTask
    
class TaskRequest:
    def __init__(self,
                 task : Union[GenericObservationTask, dict],
                 requester : str,
                 mission_name : str,
                 t_req : Union[float, int],
                 id : str = None,
                 **_ 
                 ):
        """
        ### Task Request 
        Indicates the existance of an event of interest at a given target point and requests a series of objectives to be performed by agents.

        ### Arguments:
            - task (`GenericObservationTask` or `dict`): task being requested
            - requester (`str`): name of agent requesting the observations
            - mission_name (`str`): name of the mission
            - t_req (`float`): time at which the request is made in [s] from the beginning of the simulation
            - id (`str`) : identifying number for this task request in uuid format
        
        """
        # convert arguments if needed
        if isinstance(task, dict): task = GenericObservationTask.from_dict(task)
        if isinstance(t_req, str) and t_req.lower() == "inf": t_req = np.Inf

        # check argument types
        assert isinstance(task, GenericObservationTask), f'`task` must be of type `GenericObservationTask`. Is of type {type(task)}.'
        assert isinstance(requester, str), f'`requester` must be of type `str`. Is of type {type(requester)}.'
        assert isinstance(mission_name, str), f'`mission_name` must be of type `str`. Is of type {type(mission_name)}.'
        assert isinstance(t_req, (float, int)), f'`t_req` must be of type `float` or `int`. Is of type {type(t_req)}.'
        assert t_req >= 0, f"`t_req` must have a non-negative value."
        
        # initialize attributes
        self.requester : str = requester
        self.task : GenericObservationTask = task
        self.mission_name : str = mission_name
        self.t_req : float = t_req
        self.id : str = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def __repr__(self):
        task_id = self.id.split('-')
        return f'TaskRequest_{task_id[0]}'

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this measurement request object
        """
        out = dict(self.__dict__)
        out['task'] = self.task.to_dict()
        return out
    
    def __eq__(self, value : 'TaskRequest') -> bool:
        """ compares two TaskRequest objects for equality """
        assert isinstance(value, TaskRequest), f'Cannot compare `TaskRequest` object to an object of type {type(value)}.'
        return self.to_dict() == value.to_dict() 
    
    @classmethod
    def from_dict(cls, d : dict) -> 'TaskRequest':
        return cls(**d)
    
    # def __eq__(self, other_req : 'TaskRequest') -> bool:
    #     """ compares the events being requested for observation between two measurement requests """

    #     # validate inputs
    #     if not isinstance(other_req, TaskRequest):
    #         raise ValueError(f'cannot compare `TaskRequest` object to an object of type {type(other_req)}.')
    #     if not isinstance(self.task, EventObservationTask) or not isinstance(other_req.task, EventObservationTask):
    #         raise ValueError(f'`same_event` can only be used to compare `EventObservationTask` objects. One of the tasks is of type {type(self.task)} and the other is of type {type(other_req.task)}.')

    #     my_task : EventObservationTask = self.task
    #     their_task : EventObservationTask = other_req.task

    #     if my_task.event is None and their_task.event is None:
    #         # both requests have no event associated; compare their task parameters instead
    #         same_type = my_task.task_type == their_task.task_type
    #         same_target = all([abs(my_task.location[i][j]-their_task.location[i][j]) <= 1e-3
    #                         for i in range(len(my_task.location))
    #                         for j in range(len(my_task.location[i]))
    #                         ])
    #         same_severity = True   # no event, so severity is not applicable
    #         same_time = my_task.availability.overlaps(other_req.task.availability)

    #     else:
    #         same_type = my_task.event.event_type == their_task.event.event_type
    #         same_target = all([abs(my_task.event.location[i]-their_task.event.location[i]) <= 1e-3
    #                         for i in range(len(my_task.event.location))])
    #         same_severity = abs(my_task.event.severity - their_task.event.severity) <= 1e-3

    #         my_event_availability = Interval(my_task.event.t_start, my_task.event.t_start+my_task.event.d_exp)
    #         other_event_availability = Interval(their_task.event.t_start, their_task.event.t_start+their_task.event.d_exp)
    #         same_time = my_event_availability.overlaps(other_event_availability)

    #     return (
    #             same_type
    #             and same_target
    #             and same_severity
    #             and same_time
    #             )
    
    def __hash__(self):
        return hash(str(self.to_dict()))