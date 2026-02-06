
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from enum import Enum
from dmas.models.actions import ActionStatuses, AgentAction, IdleAction, TravelAction, ManeuverAction
from orbitpy.util import OrbitState
import propcov

class SimulationAgentTypes(Enum):
    SATELLITE = 'SATELLITE'
    UAV = 'UAV'
    GROUND_OPERATOR = 'GROUND_OPERATOR'
    GROUND_SENSOR = 'GROUND_SENSOR'

class AbstractAgentState(ABC):
    """
    Describes the state of an agent
    """
    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Updates the current state
        """
        pass

    @abstractmethod
    def perform_action(self, **kwargs) -> tuple:
        """
        Performs an action that may alter the current state
        """
        pass

    @abstractmethod
    def is_failure(self, **kwargs) -> None:
        """
        Checks if the current state is a failure state
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Creates a string representing the contents of this agent state
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this agent state object
        """

    def __eq__(self, other : object) -> bool:
        """
        Compares two instances of an agent state message. Returns True if they represent the same message.
        """
        assert isinstance(other, AbstractAgentState), f"Cannot compare AbstractAgentState with object of type {type(other)}."
        return self.to_dict() == other.to_dict()

class SimulationAgentState(AbstractAgentState):
    """
    Describes the state of a 3D-CHESS agent
    """
    WAITING = 'WAITING'
    IDLING = 'IDLING'
    TRAVELING = 'TRAVELING'
    MESSAGING = 'MESSAGING'
    MANEUVERING = 'MANEUVERING'
    MEASURING = 'MEASURING'

    def __init__(   self, 
                    agent_name : str,
                    agent_id : str,
                    state_type : str,
                    pos : list,
                    vel : list,
                    attitude : list,
                    attitude_rates : list,
                    status : str = IDLING,
                    t : Union[float, int]=0,
                    **_
                ) -> None:
        """
        Creates an instance of an Abstract Agent State
        """
        super().__init__()
        
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.state_type = state_type
        self.pos : list = pos
        self.vel : list = vel
        self.attitude : list = attitude
        self.attitude_rates : list = attitude_rates
        self.status : str = status
        self._t : float = t

    def update(   self, 
                    t : Union[int, float], 
                    status : str = None, 
                    state : dict = None) -> None:
        # check time-step
        if t - self._t < -1e-5: raise ValueError(f"cannot update agent state with negative time-step of {t - self._t} [s].")
        
        # update position and velocity
        if state is None:
            self.pos, self.vel, self.attitude, self.attitude_rates = self.kinematic_model(t)
        else:
            self.pos = state['pos']
            self.vel = state['vel']
            self.attitude = state['attitude']
            self.attitude_rates = state['attitude_rates']

        assert len(self.pos) == 3, "position vector must be of length 3."
        assert len(self.vel) == 3, "velocity vector must be of length 3."
        assert len(self.attitude) == 3, "attitude vector must be of length 3."
        assert len(self.attitude_rates) == 3, "attitude rates vector must be of length 3."

        # update time and status
        self._t = t 
        self.status = status if status is not None else self.status
        
        
    def propagate(self, tf : Union[int, float]) -> tuple:
        """
        Propagator for the agent's state through time.

        ### Arguments 
            - tf (`int` or `float`) : propagation end time in [s]

        ### Returns:
            - propagated (:obj:`SimulationAgentState`) : propagated state
        """
        propagated : SimulationAgentState = self.copy()
        
        propagated.pos, propagated.vel, \
            propagated.attitude, propagated.attitude_rates = propagated.kinematic_model(tf)

        propagated._t = tf

        return propagated

    @abstractmethod
    def kinematic_model(self, tf : Union[int, float], **kwargs) -> tuple:
        """
        Propagates an agent's dinamics through time

        ### Arguments:
            - tf (`float` or `int`) : propagation end time in [s]

        ### Returns:
            - pos, vel, attitude, atittude_rate (`tuple`) : tuple of updated angular and cartasian position and velocity vectors
        """
        pass

    def perform_action(self, action : AgentAction, t : Union[int, float]) -> tuple:
        """
        Performs an action that may affect the agent's state.

        ### Arguments:
            - action (:obj:`AgentAction`): action to be performed
            - t (`int` or `double`): current simulation time in [s]
        
        ### Returns:
            - status (`str`): action completion status
            - dt (`float`): time to be waited by the agent
        """
        if isinstance(action, IdleAction):
            self.update(t, status=self.IDLING)
            if action.t_end > t:
                dt = action.t_end - t
                status = ActionStatuses.PENDING.value
            else:
                dt = 0.0
                status = ActionStatuses.COMPLETED.value
            return status, dt

        elif isinstance(action, TravelAction):
            return self.perform_travel(action, t)

        elif isinstance(action, ManeuverAction):
            return self.perform_maneuver(action, t)
        
        return ActionStatuses.ABORTED.value, 0.0

    def comp_vectors(self, v1 : list, v2 : list, eps : float = 1e-6):
        """
        compares two vectors
        """
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dz = v1[2] - v2[2]

        dv = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return dv < eps

    @abstractmethod
    def perform_travel(self, action : TravelAction, t : Union[int, float]) -> tuple:
        """
        Performs a travel action

        ### Arguments:
            - action (:obj:`TravelAction`): travel action to be performed
            - t (`int` or `double`): current simulation time in [s]
        
        ### Returns:
            - status (`str`): action completion status
            - dt (`float`): time to be waited by the agent
        """
        pass
    
    @abstractmethod
    def perform_maneuver(self, action : ManeuverAction, t : Union[int, float]) -> tuple:
        """
        Performs a meneuver action

        ### Arguments:
            - action (:obj:`ManeuverAction`): maneuver action to be performed
            - t (`int` or `double`): current simulation time in [s]
        
        ### Returns:
            - status (`str`): action completion status
            - dt (`float`): time to be waited by the agent
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """ Creates a string representing the contents of this agent state """

    def __str__(self):
        return str(dict(self.__dict__))

    def copy(self) -> 'AbstractAgentState':
        d : dict = self.to_dict()
        return SimulationAgentState.from_dict( d )

    @abstractmethod
    def to_dict(self) -> dict:
        return {
            'agent_name' : self.agent_name,
            'agent_id' : self.agent_id,
            'state_type' : self.state_type,
            'pos' : list(self.pos),
            'vel' : list(self.vel),
            'attitude' : list(self.attitude),
            'attitude_rates' : list(self.attitude_rates),
            'status' : self.status,
            't' : self._t
        }

    @abstractmethod
    def from_dict(d : dict) -> object:
        if d['state_type'] == SimulationAgentTypes.GROUND_OPERATOR.value:
            return GroundOperatorAgentState.from_dict(d)
        elif d['state_type'] == SimulationAgentTypes.SATELLITE.value:
            return SatelliteAgentState.from_dict(d)
        else:
            raise NotImplementedError(f"Agent states of type {d['state_type']} not yet supported.")
        
    def get_time(self) -> float:
        """
        Returns the current time of the agent state
        """
        return self._t

class GroundOperatorAgentState(SimulationAgentState):
    """
    Describes the state of a Ground Station Agent
    """
    R = 6.3781363e+003         # radius of the earth [km]
    W = 360 / (24 * 3600)      # angular speed of Earth [deg/s]

    def __init__(self, 
                agent_name : str, 
                agent_id : str,
                status: str = SimulationAgentState.IDLING, 
                pos : list = None,
                vel : list = None,
                t: Union[float, int] = 0, 
                **_) -> None:
        
        # calculate position and velocity if not given
        if pos is None: pos = [self.R, 0, 0]
        if vel is None: vel = np.cross([0, 0, self.W], pos).tolist()
         
        # initialize parent class
        super().__init__(agent_name,
                        agent_id,
                        SimulationAgentTypes.GROUND_OPERATOR.value, 
                        pos, 
                        vel,
                        [0,0,0],
                        [0,0,0],  
                        status, 
                        t)
        
    def kinematic_model(self, _: Union[int, float]) -> tuple:
        # ground operators are fixed on the earth surface
        return self.pos, self.vel, self.attitude, self.attitude_rates

    def is_failure(self) -> None:
        return False # agent never reaches a failure state

    def perform_travel(self, action: TravelAction, _: Union[int, float]) -> tuple:
        # agent cannot travel
        return ActionStatuses.FAILED.value, 0.0 # ground operators cannot displace

    def perform_maneuver(self, action: ManeuverAction, _: Union[int, float]) -> tuple:
        # agent cannot maneuver
        return ActionStatuses.FAILED.value, 0.0 # ground operators cannot perform maneuvers
    
    def __repr__(self):
        return f"GroundOperatorAgentState(agent_name={self.agent_name}, status={self.status}, t={round(self._t,3)})"
    
    def to_dict(self) -> dict:
        return super().to_dict()    

    @staticmethod
    def from_dict(d : dict) -> object:
        return GroundOperatorAgentState(agent_name = d['agent_name'],
                                        agent_id = d['agent_id'],
                                        status = d['status'],
                                        pos = d['pos'],
                                        vel = d['vel'],
                                        t = d['t'])

class GroundSensorAgentState(SimulationAgentState):
    """
    Describes the state of a Ground Sensor Agent
    """
    R = 6.3781363e+003         # radius of the earth [km]
    W = 360 / (24 * 3600)      # angular speed of Earth [deg/s]

    def __init__( self, 
                    agent_name : str,
                    lat : float,
                    lon : float, 
                    alt : float = 0.0,
                    pos : list = None,
                    vel : list = None,
                    t: Union[float, int] = 0, 
                    status : str = SimulationAgentState.IDLING,
                    **_
                ) -> None:

        # calculate position and velocity if not given
        if pos is None: pos = self._rotating_to_inertial([self.R + alt, 0, 0], lat, lon)
        if vel is None: vel = np.cross([0, 0, self.W], pos)

        # initialize parent class
        super().__init__(agent_name,
                        SimulationAgentTypes.GROUND_SENSOR.value,
                        pos,
                        vel,
                        [0,0,0],
                        [0,0,0],
                        status,
                        t)
        
        # assign remaining attributes
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def to_rads(self, th : float) -> float:
        return th * np.pi / 180

    def _inertial_to_rotating(self, v : list, th : float, phi : float) -> list:
        R_i2a = [[ np.cos(self.to_rads(th)), np.sin(self.to_rads(th)), 0],
                 [-np.sin(self.to_rads(th)), np.cos(self.to_rads(th)), 0],
                 [                        0,                        0, 1]]
        R_a2b = [
                 [1, 0, 0],
                 [0, np.cos(self.to_rads(phi)), np.sin(self.to_rads(phi))],
                 [0, -np.sin(self.to_rads(phi)), np.cos(self.to_rads(phi))],
                 ]
        R_i2b = np.dot(R_a2b, R_i2a)
        return np.dot(R_i2b, v)
    
    def _rotating_to_inertial(self, v : list, th : float, phi : float) -> list:
        R_i2a = [[ np.cos(self.to_rads(th)), np.sin(self.to_rads(th)), 0],
                 [-np.sin(self.to_rads(th)), np.cos(self.to_rads(th)), 0],
                 [                        0,                        0, 1]]
        R_a2b = [
                 [1, 0, 0],
                 [0, np.cos(self.to_rads(phi)), np.sin(self.to_rads(phi))],
                 [0, -np.sin(self.to_rads(phi)), np.cos(self.to_rads(phi))],
                 ]
        R_i2b = np.dot(R_a2b, R_i2a)
        R_b2i = np.transpose(R_i2b)
        return np.dot(R_b2i, v)

    def kinematic_model(self, _: Union[int, float]) -> tuple:
        # ground sensors are fixed on the earth surface
        return self.pos, self.vel, self.attitude, self.attitude_rates   

        # TODO depending on reference frame, calculate position and velocity using earth's rotation
        # lon = self.lon * self.W * tf    # longitude "changes" as earth spins 
        # lat = self.lat                  # lattitude stays constant

        # pos = [self.R + self.alt, 0, 0] # in rotating frame
        # pos = GroundOperatorAgentState._rotating_to_inertial(self, pos, lat, lon)
        # vel = np.cross(self.angular_vel, pos)
        
        # return list(pos), list(vel), self.attitude, self.attitude

    def is_failure(self) -> None:
        return False # agent never reaches a failure state

    def perform_travel(self, action: TravelAction, _: Union[int, float]) -> tuple:
        # agent cannot travel
        return ActionStatuses.FAILED.value, 0.0 # ground sensors cannot displace

    def perform_maneuver(self, action: ManeuverAction, _: Union[int, float]) -> tuple:
        # agent cannot maneuver
        return ActionStatuses.FAILED.value, 0.0 # ground sensors cannot perform maneuvers
    
    def __repr__(self):
        return f"GroundSensorAgentState(agent_name={self.agent_name}, status={self.status}, t={round(self._t,3)})"
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d['lat'] = self.lat
        d['lon'] = self.lon
        d['alt'] = self.alt
        return d

    @staticmethod
    def from_dict(d):
        return GroundSensorAgentState(agent_name = d['agent_name'],
                                      lat = d.get('lat', 0.0),
                                      lon = d.get('lon', 0.0),
                                      alt = d.get('alt', 0.0),
                                      pos = d['pos'],
                                      vel = d['vel'],
                                      status = d['status'],
                                      t = d['t'])

class SatelliteAgentState(SimulationAgentState):
    """
    Describes the state of a Satellite Agent
    """
    def __init__( self, 
                    agent_name : str,
                    agent_id : str,
                    orbit_state : dict,
                    time_step : float = None,
                    eps : float = None,
                    pos : list = None,
                    vel : list = None,
                    attitude : list = [0,0,0],
                    attitude_rates : list = [0,0,0],
                    keplerian_state : tuple = None,
                    t: Union[float, int] = 0.0, 
                    eclipse : int = 0,
                    status: str = SimulationAgentState.IDLING, 
                    **_
                ) -> None:
        # initiate orbit state object for future use 
        orbit_state_obj = None

        # calculate position and velocity if both are missing
        if pos is None and vel is None:
            # create orbit state object
            orbit_state_obj : OrbitState = OrbitState.from_dict(orbit_state)
            
            # obtain cartesian state
            cartesian_state = orbit_state_obj.get_cartesian_earth_centered_inertial_state()
            pos = cartesian_state[0:3]
            vel = cartesian_state[3:]
        elif pos is None or vel is None:
            raise ValueError("both position and velocity must be provided, or neither.")
        
        # initialize parent class
        super().__init__(   agent_name,
                            agent_id,
                            SimulationAgentTypes.SATELLITE.value, 
                            pos, 
                            vel, 
                            attitude,
                            attitude_rates,
                            status, 
                            t
                        )
        
        # check if keplerian state is given
        if keplerian_state is None:
            # form orbit state object if not already done
            if orbit_state_obj is None:
                orbit_state_obj : OrbitState = OrbitState.from_dict(orbit_state)
            
            # calculate missing keplerian state
            keplerian_state : tuple = orbit_state_obj.get_keplerian_earth_centered_inertial_state()
            # self.keplerian_state = {"aop" : keplerian_state.aop,
            #                         "ecc" : keplerian_state.ecc,
            #                         "sma" : keplerian_state.sma,
            #                         "inc" : keplerian_state.inc,
            #                         "raan" : keplerian_state.raan,
            #                         "ta" : keplerian_state.ta}
            self.keplerian_state : tuple = (keplerian_state.aop, 
                                            keplerian_state.ecc, 
                                            keplerian_state.sma,
                                            keplerian_state.inc,
                                            keplerian_state.raan,
                                            keplerian_state.ta)
        
        elif keplerian_state is not None:
            # assign given keplerian state
            # self.keplerian_state = dict()
            # self.keplerian_state.update(keplerian_state)        
            self.keplerian_state : tuple  = keplerian_state
        
        # assign remaining attributes
        self.orbit_state = orbit_state
        self.eclipse = eclipse
        self.time_step = time_step
        if eps:
            self.eps = eps
        else:
            self.eps = self.__calc_eps(pos) if self.time_step else 1e-6
        

    def kinematic_model(self, tf: Union[int, float], update_keplerian : bool = True) -> tuple:
        # propagates orbit
        dt = tf - self._t
        if abs(dt) < 1e-6:
            return self.pos, self.vel, self.attitude, self.attitude_rates

        # form the propcov.Spacecraft object
        attitude = propcov.NadirPointingAttitude()
        interp = propcov.LagrangeInterpolator()

        # following snippet is required, because any copy, changes to the propcov objects in the input spacecraft is reflected outside the function.
        spc_date = propcov.AbsoluteDate()
        orbit_state : OrbitState = OrbitState.from_dict(self.orbit_state)
        spc_date.SetJulianDate(orbit_state.date.GetJulianDate())
        spc_orbitstate = orbit_state.state
        
        spc = propcov.Spacecraft(spc_date, spc_orbitstate, attitude, interp, 0, 0, 0, 1, 2, 3) # TODO: initialization to the correct orientation of spacecraft is not necessary for the purpose of orbit-propagation, so ignored for time-being.
        start_date = spc_date

        # following snippet is required, because any copy, changes to the input start_date is reflected outside the function. (Similar to pass by reference in C++.)
        # so instead a separate copy of the start_date is made and is used within this function.
        _start_date = propcov.AbsoluteDate()
        _start_date.SetJulianDate(start_date.GetJulianDate())

        # form the propcov.Propagator object
        prop = propcov.Propagator(spc)

        # propagate to the specified start date since the date at which the orbit-state is defined
        # could be different from the specified start_date (propagation could be either forwards or backwards)
        prop.Propagate(_start_date)
        
        date = _start_date

        if self.time_step:
            # TODO compute dt as a multiple of the registered time-step 
            pass

        date.Advance(tf)
        prop.Propagate(date)
        
        cartesian_state = spc.GetCartesianState().GetRealArray()
        pos = cartesian_state[0:3]
        vel = cartesian_state[3:]

        if update_keplerian:
            keplerian_state = spc.GetKeplerianState().GetRealArray()
            # self.keplerian_state.update({"sma" : keplerian_state[0],
            #                              "ecc" : keplerian_state[1],
            #                              "inc" : keplerian_state[2],
            #                              "raan" : keplerian_state[3],
            #                              "aop" : keplerian_state[4],
            #                              "ta" : keplerian_state[5]})
            self.keplerian_state = tuple(keplerian_state)

        attitude = []
        for i in range(len(self.attitude)):
            th = self.attitude[i] + dt * self.attitude_rates[i]
            attitude.append(th)
       
        return pos, vel, attitude, self.attitude_rates

    def is_failure(self) -> None:
        return False

    def perform_travel(self, action: TravelAction, t: Union[int, float]) -> tuple:
        # update state
        self.update(t, status=self.TRAVELING)

        # check if position was reached
        if self.comp_vectors(self.pos, action.final_pos) or t >= action.t_end - self.eps:
            # if reached, return successful completion status
            return ActionStatuses.COMPLETED.value, 0.0
        else:
            # else, wait until position is reached
            if action.t_end == np.Inf:
                dt = self.time_step if self.time_step else 60.0
            else:
                dt = action.t_end - t
            return ActionStatuses.PENDING.value, dt

    def perform_maneuver(self, action: ManeuverAction, t: Union[int, float]) -> tuple:
        # update state
        self.update(t, status=self.MANEUVERING)
        
        if self.comp_vectors(self.attitude, action.final_attitude, eps = 1e-6):
            # if reached, return successful completion status
            self.attitude_rates = [0,0,0]
            return ActionStatuses.COMPLETED.value, 0.0
        
        elif t > action.t_end + self.eps:
            # could not complete action before action end time
            self.attitude_rates = [0,0,0]
            return ActionStatuses.ABORTED.value, 0.0

        else:
            # update attitude angular rates
            self.attitude_rates = [rate for rate in action.attitude_rates]

            # estimate remaining time for completion
            dts = [action.t_end - t]
            for i in range(len(self.attitude)):
                # estimate completion time 
                dt = (action.final_attitude[i] - self.attitude[i]) / self.attitude_rates[i] if self.attitude_rates[i] > 1e-3 else np.NAN
                dts.append(dt)

            dt_maneuver = min(dts)
            
            assert dt_maneuver >= 0.0, \
                f"negative time-step of {dt_maneuver} [s] for attitude maneuver."

            # return status
            return ActionStatuses.PENDING.value, dt_maneuver
            
    def __calc_eps(self, init_pos : list):
        """
        Calculates tolerance for position vector comparisons
        """

        # form the propcov.Spacecraft object
        attitude = propcov.NadirPointingAttitude()
        interp = propcov.LagrangeInterpolator()

        # following snippet is required, because any copy, changes to the propcov objects in the input spacecraft is reflected outside the function.
        spc_date = propcov.AbsoluteDate()
        orbit_state : OrbitState = OrbitState.from_dict(self.orbit_state)
        spc_date.SetJulianDate(orbit_state.date.GetJulianDate())
        spc_orbitstate = orbit_state.state
        
        spc = propcov.Spacecraft(spc_date, spc_orbitstate, attitude, interp, 0, 0, 0, 1, 2, 3) # TODO: initialization to the correct orientation of spacecraft is not necessary for the purpose of orbit-propagation, so ignored for time-being.
        start_date = spc_date

        # following snippet is required, because any copy, changes to the input start_date is reflected outside the function. (Similar to pass by reference in C++.)
        # so instead a separate copy of the start_date is made and is used within this function.
        _start_date = propcov.AbsoluteDate()
        _start_date.SetJulianDate(start_date.GetJulianDate())

        # form the propcov.Propagator object
        prop = propcov.Propagator(spc)

        # propagate to the specified start date since the date at which the orbit-state is defined
        # could be different from the specified start_date (propagation could be either forwards or backwards)
        prop.Propagate(_start_date)
        
        date = _start_date
        date.Advance(self.time_step)
        prop.Propagate(date)
        
        cartesian_state = spc.GetCartesianState().GetRealArray()
        pos = cartesian_state[0:3]

        dx = init_pos[0] - pos[0]
        dy = init_pos[1] - pos[1]
        dz = init_pos[2] - pos[2]

        return np.sqrt(dx**2 + dy**2 + dz**2) / 2.0
    
    def comp_vectors(self, v1 : list, v2 : list, eps : float = None):
        """
        compares two vectors
        """
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dz = v1[2] - v2[2]

        dv = np.sqrt(dx**2 + dy**2 + dz**2)
        eps = eps if eps is not None else self.eps

        # print( '\n\n', v1, v2, dv, self.eps, dv < self.eps, '\n')

        return dv < eps
    
    def __repr__(self):
        return f"SatelliteAgentState(agent_name={self.agent_name}, status={self.status}, t={round(self._t,3)})"
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d['orbit_state'] = dict(self.orbit_state)
        d['keplerian_state'] = tuple(self.keplerian_state)
        d['time_step'] = self.time_step
        d['eps'] = self.eps
        d['eclipse'] = self.eclipse
        return d
    
    EMPTY_VECTOR = [0,0,0]
    @staticmethod
    def from_dict(d) -> 'SatelliteAgentState':
        return SatelliteAgentState( agent_name = d['agent_name'],
                                    agent_id = d['agent_id'],
                                    orbit_state = d['orbit_state'],
                                    time_step = d.get('time_step', None),
                                    eps = d.get('eps', None),
                                    pos = d.get('pos', None),
                                    vel = d.get('vel', None),
                                    attitude = d.get('attitude', SatelliteAgentState.EMPTY_VECTOR),
                                    attitude_rates = d.get('attitude_rates', SatelliteAgentState.EMPTY_VECTOR),
                                    keplerian_state = d.get('keplerian_state', None),
                                    t = d.get('t', 0.0),
                                    eclipse = d.get('eclipse', 0),
                                    status=d.get('status', SimulationAgentState.IDLING)
                                )
