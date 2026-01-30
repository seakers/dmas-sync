import uuid
import numpy as np
from chess3d.utils import Interval

from chess3d.agents.actions import ObservationAction

class ObservationTask:
    def __init__(self, 
                 measurement_type : str,
                 instrument_name : str,
                 availability : Interval,
                 slew_angles : Interval,
                 targets : list,
                 reward : float = np.NINF,
                #  min_duration : float = 0.0,
                 max_duration: float = np.Inf,
                #  t_latest : float = None,
                 id: str = None,
                 ):
        """ Represents an observation task in a planning system. """

        self.measurement_type = measurement_type            # type of measurement being performed
        self.instrument_name = instrument_name              # name of instrument being used
        self.availability = availability                    # time interval during which the task is available
        self.slew_angles = slew_angles                      # feasible slew angles for the instrument at the time of observation
        self.targets = targets                              # list of target ground points to be observed        
        self.reward = reward                                # reward associated with the observation task
        # self.min_duration = min_duration                    # minimum duration of the observation task (0.0 by default)
        self.max_duration = max_duration                    # maximum duration of the observation task
        # self.t_latest = availability.right - self.min_duration \
        #     if t_latest is None else t_latest               # latest start time of the task
        
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1()) # unique identifier for the task

    def copy(self) -> object:
        """ Create a copy of the task WITH DIFFERENT ID. """
        return ObservationTask(
            instrument_name=self.instrument_name,
            availability=self.availability,
            slew_angles=self.slew_angles,
            targets=self.targets,
            reward=self.reward,
            # min_duration=self.min_duration,
            max_duration=self.max_duration,
            # t_latest=self.t_latest,
            id=None
        )

    def can_combine(self, other_task : object) -> bool:
        """ Check if two tasks can be combined based on their time and slew angle. """
        
        # Check if the other task is an instance of ObservationTask
        if not isinstance(other_task, ObservationTask):
            raise ValueError("The other task must be an instance of Task.")
        
        # Check if the tasks have the same instrument name
        if self.instrument_name != other_task.instrument_name:
            raise ValueError("Tasks must have the same instrument name to be clustered.")

        if self.id == other_task.id:
            return False

        # Check if the availability time intervals overlap
        availability_overlap : Interval = self.availability.union(other_task.availability)

        if not availability_overlap.is_empty():
            # Check if the time intervals are within the maximum duration
            if availability_overlap.span() > min(self.max_duration, other_task.max_duration):
                return False
            # elif availability_overlap.span() < max(self.min_duration, other_task.min_duration):
            #     return False

        # Check if the slew angles overlap
        slew_angle_overlap : Interval = self.slew_angles.intersection(other_task.slew_angles) 

        return not (availability_overlap.is_empty() or slew_angle_overlap.is_empty())
        

    def to_observation_action(self) -> ObservationAction:
        """ Convert the task to an observation action. """
        
        # calculate look angle and action duration
        targets = [target for target in self.targets]
        look_angle = (self.slew_angles.start + self.slew_angles.end) / 2
        duration = self.availability.end - self.availability.start
        
        # Create an ObservationAction object
        return ObservationAction(self.instrument_name, targets, look_angle, self.availability.start, duration)
    
    def combine(self, other_task : object) -> None:
        """ Combine two tasks into one. """
        if not isinstance(other_task, ObservationTask):
            raise ValueError("The other task must be an instance of ObservationTask.")
        
        # Combine the time intervals and slew angles
        combined_time_interval : Interval = self.availability.union(other_task.availability)
        combined_slew_angles : Interval  = self.slew_angles.intersection(other_task.slew_angles)
        
        # Check if the combined time interval exceeds the maximum duration
        if combined_time_interval.span() > min(self.max_duration, other_task.max_duration):
            raise ValueError("Combined time interval exceeds maximum duration.")
        
        # Combine the targets
        combined_targets = [target for target in self.targets]
        combined_targets.extend([target for target in other_task.targets 
                                if self.__unique_target(target, combined_targets)])
        
        # Update the task attributes
        self.availability = combined_time_interval
        self.slew_angles = combined_slew_angles
        self.targets = combined_targets
        self.reward += other_task.reward
        # self.min_duration = max(self.min_duration, other_task.min_duration)
        # self.max_duration = min(self.max_duration, other_task.max_duration)
        # self.t_latest = min(self.t_latest, other_task.t_latest)
    
    def __unique_target(self, target : list, known_targets : list) -> bool:
        """ Check if the target is unique in the known targets. """
        for known_target in known_targets:
            if (abs(known_target[0] - target[0]) < 1e-6
                and abs(known_target[1] - target[1]) < 1e-6
                and abs(known_target[2] - target[2]) < 1e-6
            ):
                return False
        return True
    
    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return f"ObservationTask_{str.split(self.id,'-')[0]}"
        # return f"ObservationTask_{str.split(self.id)[0]}"

    def __hash__(self) -> int:
        return hash(repr(self))
    
if __name__ == "__main__":
    # Example usage
    # task = Task("Sample Task", "This is a sample task description.")
    x = 1
    # print(f"Task Name: {task.name}")
    # print(f"Task Description: {task.description}")
    # task.execute()  # This will raise NotImplementedError