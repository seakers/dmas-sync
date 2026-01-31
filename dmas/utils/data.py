from typing import Dict
import numpy as np

from execsatm.mission import Mission
from execsatm.utils import Interval
from dmas.core.orbitdata import OrbitData

class DataProcessor:

    @staticmethod
    def collect_results(orbitdata_dir : str, duration : float) -> tuple:
        # collect results
        print('Collecting orbit data...')
        compiled_orbitdata : Dict[str, OrbitData] = OrbitData.from_directory(orbitdata_dir, duration) \
            if orbitdata_dir is not None else None
        # collect missions
        print('Collecting mission data...')
        agent_missions : Dict[str, Mission] = {agent.get_element_name(): agent.mission 
                                                for agent in self.agents}

        # collect observations
        print('Collecting observations performed data...')
        try:
            observations_performed_path = os.path.join(self.environment.results_path, 'measurements.parquet')
            observations_performed = pd.read_parquet(observations_performed_path)
            print('SUCCESS!')

        except pd.errors.EmptyDataError:
            columns = ['observer','t_img','lat','lon','range','look','incidence','zenith','instrument_name']
            observations_performed = pd.DataFrame(data=[],columns=columns)
            print('No observations were performed during the simulation.')

        # load all scenario events
        print('Loading event data...')
        events_df = pd.read_csv(self.environment.events_path)         

        # filter out events that do not occur during this simulation
        events_df = events_df[events_df['start time [s]'] <= self.duration*3600*24] 

        # convert event to dataframe to list of GeophysicalEvent
        events : list[GeophysicalEvent] = []
        for _,row in events_df.iterrows():
            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['start time [s]'],
                row.get('id',None)
            )
            events.append(event)

        # compile events detected
        print('Collecting event detection data...')
        events_detected_df : pd.DataFrame = None
        for agent in self.agents:
            _,agent_name = agent.name.split('/')
            events_detected_path = os.path.join(self.results_path, agent_name.lower(), 'events_detected.parquet')
            if not os.path.isfile(events_detected_path): continue

            # load detected events            
            events_detected_temp = pd.read_parquet(events_detected_path)

            # concatenate to main dataframe
            events_detected_df = pd.concat([events_detected_df, events_detected_temp], axis=0) \
                if events_detected_df is not None else events_detected_temp

        assert events_detected_df is not None, \
            "Couldn't load Event Detection file for any agent."
        
        # remove duplicates
        events_detected_df = events_detected_df.drop_duplicates().reset_index(drop=True)
        
        # convert to list of GeophysicalEvent
        events_detected : list[GeophysicalEvent] = []
        for _,row in events_detected_df.iterrows():

            # TODO test event detection results collection
            raise NotImplementedError('Event detection results collection is imlemented but not yet tested.')

            event = GeophysicalEvent(
                row['event type'],
                (row['lat [deg]'], row['lon [deg]'], row.get('grid index', 0), row['gp_index']),
                row['start time [s]'],
                row['duration [s]'],
                row['severity'],
                row['detection time [s]'],
                row.get('id',None)
            )
            events_detected.append(event)

        # compile measurement requests
        print('Collecting measurement request data...')
        try:
            task_reqs_df = pd.read_parquet((os.path.join(self.environment.results_path, 'requests.parquet')))
        except pd.errors.EmptyDataError:
            columns = ['id','requester','lat [deg]','lon [deg]','severity','t start','t end','t corr','Measurment Types']
            task_reqs_df = pd.DataFrame(data=[],columns=columns)
        # remove duplicates
        task_reqs_df = task_reqs_df.drop_duplicates().reset_index(drop=True)

        # convert to list of TaskRequest
        task_reqs = []
        for _,row in task_reqs_df.iterrows():
            matching_events : list[GeophysicalEvent] = [
                event for event in events
                if event.id == row['event id']
            ]
            assert matching_events, \
                f"No matching event found for measurement request with event id `{row['event id']}`"
            
            # get name of agent requesting the task
            requester = row['requester']
            
            # get matching `EventDrivenObjective`
            relevant_objectives = [objective for objective in agent_missions[requester]
                                    if isinstance(objective, EventDrivenObjective)
                                    and objective.parameter == row['parameter']]
            # ensure exactly one matching objective found
            assert relevant_objectives, \
                f"No matching EventDrivenObjective found in mission for requester `{requester}` and parameter `{row['parameter']}`."
            
            # create event-driven task 
            task = EventObservationTask(
                row['parameter'],
                event=matching_events[0],
                objective=relevant_objectives[0]
            )
            
            # create task request
            req = TaskRequest(
                task,
                row['requester'],
                agent_missions[requester].name,
                row['t_req'],
                row['event id']
            )

            # add to list of task requests
            task_reqs.append(req)

        # compile default tasks from every agent
        default_tasks_df : pd.DataFrame = None
        for agent in self.agents:
            _,agent_name = agent.name.split('/')
            known_tasks_path = os.path.join(self.results_path, agent_name.lower(), 'known_tasks.parquet')
            if not os.path.isfile(known_tasks_path): continue
            
            # load default tasks
            default_tasks_temp = pd.read_parquet(known_tasks_path)

            # concatenate to main dataframe
            default_tasks_df = pd.concat([default_tasks_df, default_tasks_temp], axis=0) \
                if not default_tasks_temp.empty else default_tasks_df

        # remove duplicates
        default_tasks_df = default_tasks_df.drop_duplicates().reset_index(drop=True) \
            if default_tasks_df is not None else pd.DataFrame([])
        
        # convert to list of tasks
        tasks_known : list[GenericObservationTask] = []
        for _,row in default_tasks_df.iterrows():
            # get name of agent requesting the task
            requester = row['requester']
            if row['task type'] != GenericObservationTask.DEFAULT:
                raise ValueError(f"Unknown task type `{row['task type']}` found in known tasks file.")
            
            # get matching `DefaultMissionObjective`
            relevant_objectives = [objective for objective in agent_missions[requester]
                                    if isinstance(objective, DefaultMissionObjective)
                                    and objective.parameter == row['parameter']]
            # ensure exactly one matching objective found
            assert relevant_objectives, \
                f"No matching DefaultMissionObjective found in mission for requester `{requester}` and parameter `{row['parameter']}`."
            
            task = DefaultMissionTask(
                row['parameter'],
                (row['lat [deg]'], row['lon [deg]'], row['grid index'], row['gp index']),
                row['t end'] - row['t start'],
                row['priority'],
                relevant_objectives[0],
                row['id']
            )
            tasks_known.append(task)

        # suplement with event observation tasks
        tasks_known.extend([req.task for req in task_reqs 
                            if req.task not in tasks_known])
        
        # compile broadcast history
        agent_broadcasts_df = pd.read_parquet((os.path.join(self.environment.results_path, 'broadcasts.parquet')))
        # IDEA remove duplicates? 
        # agent_broadcasts_df = agent_broadcasts_df.drop_duplicates().reset_index(drop=True)

        # return collected results
        return compiled_orbitdata, agent_missions, observations_performed, events, events_detected, task_reqs, tasks_known, agent_broadcasts_df

    def _summarize_results(self, 
                            compiled_orbitdata : Dict[str, OrbitData], 
                            agent_missions : Dict[str, Mission],
                            observations_performed : pd.DataFrame, 
                            events : List[GeophysicalEvent], 
                            events_detected : List[GeophysicalEvent], 
                            task_reqs : List[TaskRequest], 
                            tasks_known : List[GenericObservationTask],
                            agent_broadcasts_df : pd.DataFrame,
                            n_decimals : int = 5
                        ) -> pd.DataFrame:
        
        # classify observations
        observations_per_gp, events_per_gp, gps_accessible, \
            events_observable, events_observed, events_detected, events_requested, \
                events_re_observable, events_re_obs, \
                    events_co_observable, events_co_obs, \
                    events_co_observable_fully, events_co_obs_fully, \
                        events_co_observable_partially, events_co_obs_partially, \
                                tasks_observable, tasks_observed \
                                    = self._classify_observations(compiled_orbitdata,
                                                                observations_performed, 
                                                                events, 
                                                                events_detected,
                                                                task_reqs,
                                                                tasks_known)       

        # count observations performed
        # n_events, n_unique_event_obs, n_total_event_obs,
        n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                n_tasks, n_event_tasks, n_default_tasks, \
                                    n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                        n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                            n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved \
                                                    = self._count_observations(  compiled_orbitdata, 
                                                                                observations_performed, 
                                                                                observations_per_gp,
                                                                                events, 
                                                                                events_per_gp,
                                                                                events_observable,
                                                                                events_detected, 
                                                                                events_requested,
                                                                                events_observed, 
                                                                                events_re_observable,
                                                                                events_re_obs, 
                                                                                events_co_observable,
                                                                                events_co_obs, 
                                                                                events_co_observable_fully,
                                                                                events_co_obs_fully, 
                                                                                events_co_observable_partially,
                                                                                events_co_obs_partially,
                                                                                tasks_known,
                                                                                tasks_observable,
                                                                                tasks_observed)
            
        # count probabilities of observations performed
        p_gp_accessible, p_gp_observed, p_gp_observed_if_accessible, p_event_at_gp, p_event_detected, \
            p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                    p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                        p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                            p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected, \
                                    p_task_observable, p_event_task_observable, p_default_task_observable, p_task_observed, p_event_task_observed, p_default_task_observed, \
                                        p_task_observed_if_observable, p_event_task_observed_if_observable, p_default_task_observed_if_observable, \
                                            p_task_reobserved, p_event_task_reobserved, p_default_task_reobserved, \
                                                p_task_reobserved_if_reobservable, p_event_task_reobserved_if_reobservable, p_default_task_reobserved_if_reobservable \
                                                    = self._calc_event_probabilities(compiled_orbitdata, 
                                                                                    gps_accessible,
                                                                                    observations_performed, 
                                                                                    observations_per_gp,
                                                                                    events, 
                                                                                    events_per_gp,
                                                                                    events_observable,
                                                                                    events_detected, 
                                                                                    events_requested,
                                                                                    events_observed, 
                                                                                    events_re_observable,
                                                                                    events_re_obs, 
                                                                                    events_co_observable,
                                                                                    events_co_obs, 
                                                                                    events_co_observable_fully,
                                                                                    events_co_obs_fully, 
                                                                                    events_co_observable_partially,
                                                                                    events_co_obs_partially,
                                                                                    tasks_known,
                                                                                    tasks_observable,
                                                                                    tasks_observed)
        
        # calculate event revisit times
        t_gp_reobservation = self._calc_groundpoint_coverage_metrics(observations_per_gp)
        t_event_reobservation = self._calc_event_coverage_metrics(events_observed)

        # Generate summary
        summary_headers = ['Metric', 'Value']
        summary_data = [
                    # Dates
                    # ['Simulation Start Date', self.environment._clock_config.start_date], 
                    # ['Simulation End Date', self.environment._clock_config.end_date], 

                    # Counters
                    ['Events', n_events],
                    ['Events Observable', n_events_observable],
                    ['Events Observed', n_events_observed],
                    ['Events Detected', n_events_detected],
                    ['Events Requested', n_events_requested],
                    ['Event Observations', n_total_event_obs],
                    ['Events Re-observable', n_events_reobservable],
                    ['Events Re-observed', n_events_reobserved],
                    ['Event Re-observations', n_total_event_re_obs],
                    ['Events Co-observable', n_events_co_observable],
                    ['Events Co-observed', n_events_co_obs],
                    ['Event Co-observations', n_total_event_co_obs],
                    ['Events Fully Co-observable', n_events_co_observable_fully],
                    ['Events Fully Co-observed', n_events_fully_co_obs],
                    ['Event Full Co-observations', n_total_event_fully_co_obs],
                    ['Events Only Partially Co-observable', n_events_co_observable_partially],
                    ['Events Partially Co-observed', n_events_partially_co_obs],
                    ['Event Partial Co-observations', n_total_event_partially_co_obs],
                    ['Tasks Available', n_tasks],
                    ['Event-Driven Tasks Available', n_event_tasks],
                    ['Default Mission Tasks Available', n_default_tasks],
                    ['Tasks Observable', n_tasks_observable],
                    ['Event-Driven Tasks Observable', n_event_tasks_observable],
                    ['Default Mission Tasks Observable', n_default_tasks_observable],
                    ['Tasks Observed', n_tasks_observed],
                    ['Event-Driven Tasks Observed', n_event_tasks_observed],
                    ['Default Mission Tasks Observed', n_default_tasks_observed],

                    # Coverage Metrics #TODO add more
                    ['Ground Points', n_gps],
                    ['Ground Points Accessible', n_gps_accessible],
                    ['Ground Points Observed', n_gps_observed],
                    ['Ground Points Reobserved', n_gps_reobserved],
                    ['Ground Point Observations', n_observations],
                    ['Ground Points with Events', n_gps_with_events],

                    ['Average GP Reobservation Time [s]', t_gp_reobservation['mean']],
                    ['Standard Deviation of GP Reobservation Time [s]', t_gp_reobservation['std']],
                    ['Median GP Reobservation Time [s]', t_gp_reobservation['median']],
                    ['Average Event Reobservation Time [s]', t_event_reobservation['mean']],
                    ['Standard Deviation of Event Reobservation Time [s]', t_event_reobservation['std']],
                    ['Median Event Reobservation Time [s]', t_event_reobservation['median']],

                    # Ground-Point Coverage Probabilities
                    ['P(Ground Point Accessible)', np.round(p_gp_accessible,n_decimals)],
                    ['P(Ground Point Observed)', np.round(p_gp_observed,n_decimals)],
                    ['P(Ground Point Observed | Ground Point Accessible)', np.round(p_gp_observed_if_accessible,n_decimals)],
                    ['P(Event at a GP)', np.round(p_event_at_gp,n_decimals)],

                    # Event Observation Probabilities
                    # TODO add co-observation probabilities
                    ['P(Event Observable)', np.round(p_event_observable,n_decimals)],
                    ['P(Event Re-observable)', np.round(p_event_re_observable,n_decimals)],
                    ['P(Event Co-observable)', np.round(p_event_co_observable,n_decimals)],
                    ['P(Event Fully Co-observable)', np.round(p_event_co_observable_fully,n_decimals)],
                    ['P(Event Partially Co-observable)', np.round(p_event_co_observable_partial,n_decimals)],
                    
                    ['P(Event Detected)', np.round(p_event_detected,n_decimals)],
                    ['P(Event Observed)', np.round(p_event_observed,n_decimals)],
                    ['P(Event Re-observed)', np.round(p_event_re_obs,n_decimals)],
                    ['P(Event Co-observed)', np.round(p_event_co_obs,n_decimals)],
                    ['P(Event Fully Co-observed)', np.round(p_event_co_obs_fully,n_decimals)],
                    ['P(Event Partially Co-observed)', np.round(p_event_co_obs_partial,n_decimals)],

                    ['P(Event Observation | Observation)', np.round(p_event_obs_if_obs,n_decimals)],
                    ['P(Event Re-observation | Observation)', np.round(p_event_re_obs_if_obs,n_decimals)],
                    # ['P(Event Co-observation | Observation)', np.round(p_event_co_obs_if_obs,n_decimals)],
                    # ['P(Event Full Co-observation | Observation)', np.round(p_event_co_obs_partially_if_obs,n_decimals)],
                    # ['P(Event Partial Co-observation | Observation)', np.round(p_event_co_obs_fully_if_obs,n_decimals)],

                    ['P(Event Observed | Observable)', np.round(p_event_observed_if_observable,n_decimals)],
                    ['P(Event Re-observed | Re-observable)', np.round(p_event_re_obs_if_re_observable,n_decimals)],
                    # ['P(Event Co-observed | Co-observable)', np.round(p_event_co_obs_if_co_observable,n_decimals)],
                    # ['P(Event Fully Co-observed | Fully Co-observable)', np.round(p_event_co_obs_fully_if_co_observable_fully,n_decimals)],
                    # ['P(Event Partially Co-observed | Partially Co-observable)', np.round(p_event_co_obs_partial_if_co_observable_partially,n_decimals)],
                    
                    ['P(Event Observed | Event Detected)', np.round(p_event_observed_if_detected,n_decimals)],
                    ['P(Event Re-observed | Event Detected)', np.round(p_event_re_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed | Event Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed Fully | Event Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    # ['P(Event Co-observed Partially | Event Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

                    ['P(Event Observed | Event Observable and Detected)', np.round(p_event_observed_if_detected,n_decimals)],
                    ['P(Event Re-observed | Event Re-observable and Detected)', np.round(p_event_re_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed | Event Co-observable and Detected)', np.round(p_event_co_obs_if_detected,n_decimals)],
                    # ['P(Event Co-observed Fully | Event Fully Co-observable and Detected)', np.round(p_event_co_obs_fully_if_detected,n_decimals)],
                    # ['P(Event Co-observed Partially | Event Partially Co-observable and Detected)', np.round(p_event_co_obs_partial_if_detected,n_decimals)],

                    # Task Observation Probabilities
                    # TODO add co-observation probabilities
                    ['P(Task Observable)', np.round(p_task_observable,n_decimals)],
                    ['P(Task Observed)', np.round(p_task_observed,n_decimals)],
                    ['P(Task Observed | Task Observable)', np.round(p_task_observed_if_observable,n_decimals)],
                    ['P(Task Reobserved)', np.round(p_task_reobserved,n_decimals)],
                    ['P(Task Reobserved | Task Reobservable)', np.round(p_task_reobserved_if_reobservable,n_decimals)],
                    
                    ['P(Event-Driven Task Observable)', np.round(p_event_task_observable,n_decimals)],
                    ['P(Event-Driven Task Observed)', np.round(p_event_task_observed,n_decimals)],
                    ['P(Event-Driven Task Observed | Event-Driven Task Observable)', np.round(p_event_task_observed_if_observable,n_decimals)],
                    ['P(Event-Driven Task Reobserved)', np.round(p_event_task_reobserved,n_decimals)],
                    ['P(Event-Driven Task Reobserved | Event-Driven Task Reobservable)', np.round(p_event_task_reobserved_if_reobservable,n_decimals)],
                    
                    ['P(Default Mission Task Observable)', np.round(p_default_task_observable,n_decimals)],
                    ['P(Default Mission Task Observed)', np.round(p_default_task_observed,n_decimals)],
                    ['P(Default Mission Task Observed | Default Mission Task Observable)', np.round(p_default_task_observed_if_observable,n_decimals)],
                    ['P(Default Mission Task Reobserved)', np.round(p_default_task_reobserved,n_decimals)],
                    ['P(Default Mission Task Reobserved | Default Mission Task Reobservable)', np.round(p_default_task_reobserved_if_reobservable,n_decimals)],
                    
                    # Messaging Statistics
                    ['Total Messages Broadcasted', len(agent_broadcasts_df)],
                    ['P(Message Broadcasted | Bid Message )', len(agent_broadcasts_df[agent_broadcasts_df['message type']=='BUS']) / len(agent_broadcasts_df) if len(agent_broadcasts_df) > 0 else 0.0],
                    ['P(Message Broadcasted | Measurement Request Message )', len(agent_broadcasts_df[agent_broadcasts_df['message type']=='MEASUREMENT_REQ']) / len(agent_broadcasts_df) if len(agent_broadcasts_df) > 0 else 0.0],

                    # TODO Utility Statistics 

                    # Simulation Runtime
                    # ['Total Runtime [s]', round(self.environment.t_f - self.environment.t_0, n_decimals)]
                ]

        return pd.DataFrame(summary_data, columns=summary_headers)
                        
    def _classify_observations(self, 
                                compiled_orbitdata : Dict[str, OrbitData], 
                                observations_performed : pd.DataFrame,
                                events : List[GeophysicalEvent], 
                                events_detected : List[GeophysicalEvent], 
                                task_reqs : List[TaskRequest],
                                known_tasks : List[GenericObservationTask]
                            ) -> tuple:
                
        # classify ground points (GPs) by their accessibility
        gps_accessible = set()
        for agent_orbitdata in tqdm(compiled_orbitdata.values(), desc='Counting total and accessible ground points', leave=False):
            # get set of accessible ground points
            gps_accessible_temp : set = {(row['grid index'], row['GP index']) 
                                            for _,row in agent_orbitdata.gp_access_data}

            # update set of accessible ground points
            gps_accessible.update(gps_accessible_temp)

        # classify observations per GP
        observations_per_gp : Dict[str, pd.DataFrame] \
                                = {group : data
                                for group,data in observations_performed.groupby(['lat [deg]', 'lon [deg]'])} \
                                if not observations_performed.empty else dict() # handle empty observations case

        # collect all event target GP locations
        event_targets = {(round(event.location[0],6), round(event.location[1],6))
                            for event in events}
        
        # classify events per target GP
        events_per_gp = {(lat, lon): [event for event in events
                                    if (abs(round(event.location[0],6) - lat) < 1e-6 
                                    and abs(round(event.location[1],6) - lon) < 1e-6)]
                            for (lat, lon) in event_targets}

        # count event presense, detections, and observations
        events_observable : Dict[GeophysicalEvent, list] = {}
        events_detected : Dict[GeophysicalEvent, list] = {}
        events_requested : Dict[GeophysicalEvent, list] = {}
        events_observed : Dict[GeophysicalEvent, list] = {}

        # for event in tqdm(events.values, 
        for event in tqdm(events, 
                            desc='Classifying event accesses, detections, and observations', 
                            leave=True):
            
            access_intervals, matching_detections, matching_requests, matching_observations \
                = self.__collect_event_observation(event, 
                                            compiled_orbitdata, 
                                            events_detected, 
                                            task_reqs, 
                                            observations_per_gp)

            if access_intervals: events_observable[event] = access_intervals
            if matching_detections: events_detected[event] = matching_detections
            if matching_requests: events_requested[event] = matching_requests
            if matching_observations: events_observed[event] = matching_observations

        # sanity check: ensure all observed events are also observable
        assert all([event in events_observable for event in events_observed])
        

        # find reobserved events
        events_re_observable : Dict[GeophysicalEvent, list] \
                                    = { event: access_intervals 
                                        for event,access_intervals in events_observable.items()
                                        if len(access_intervals) > 1}
        events_re_obs : Dict[GeophysicalEvent, list] \
                            = {event: observations[1:] 
                                for event,observations in events_observed.items()
                                if len(observations) > 1}
        
        # find co-observable events
        events_co_observable : Dict[GeophysicalEvent, list] = {}
        events_co_observable_fully : Dict[GeophysicalEvent, list] = {}
        events_co_observable_partially : Dict[GeophysicalEvent, list] = {}

        # TODO implement co-observation classification
        print('Classifying co-observations... (WARNING: not yet implemented)')
        # for event, access_intervals in tqdm(events_observable.items(), desc='Compiling possible co-observations', leave=False):
        #     # get event characteristics
        #     event_type : str = event.event_type
            
        #     # get types of observations that can be performed for this event
        #     co_observation_params = set()
        #     observations_required = dict()
        #     valid_instruments = set()

        #     for _,mission in self.missions.items():
        #         for objective in mission:
        #             if (isinstance(objective, EventDrivenObjective) 
        #                 and objective.event_type.lower() == event_type.lower()):
                        
        #                 co_observation_params.add(objective.parameter)

        #                 for req in objective:
        #                     if isinstance(req, CapabilityRequirement) and req.attribute == 'instrument':
        #                         valid_instruments.update(set(req.valid_values))
        #                         observations_required[objective.parameter] = set(req.valid_values)
        
        #     raise NotImplementedError("Observation classification is not yet implemented.")
            

        #     # check if there are observations that satisfy the requirements of the request
        #     if len(observations_required) > 1:
        #         # check if valid co-observations match this event
        #         co_observation_opportunities = {(*_, instrument) 
        #                                         for param in observations_required
        #                                         for *_, instrument in access_intervals 
        #                                         if instrument.lower() in observations_required[param]}
        #         co_observable_parameters = {param 
        #                                     for param in observations_required
        #                                     for *_, instrument in access_intervals 
        #                                     if instrument.lower() in observations_required[param]}

        #         if co_observation_params.issubset(co_observable_parameters):
        #             # all required observation types were performed; event was fully co-observed
        #             events_co_observable_fully[event] = co_observation_opportunities
        #         else:
        #             # some required observation types were performed; event was parially co-observed
        #             events_co_observable_partially[event] = co_observation_opportunities

        #         # event is co-observed
        #         events_co_observable[event] = co_observation_opportunities

        # find co-observed events
        events_co_obs : Dict[tuple, list] = {}
        events_co_obs_fully : Dict[tuple, list] = {}
        events_co_obs_partially : Dict[tuple, list] = {}

        # TODO
        # for event, observations in tqdm(events_observed.items(), desc='Compiling co-observations', leave=False):
        #     # get event characteristics
        #     event_type : str = event[3]
            
        #     # get types of observations that can be performed for this event
        #     co_observation_params = set()
        #     observations_required = dict()
        #     valid_instruments = set()

        #     for _,mission in self.missions.items():
        #         for objective in mission:
        #             if isinstance(objective, EventDrivenObjective) and objective.event_type.lower() == event_type.lower():
        #                 co_observation_params.add(objective.parameter)

        #                 for req in objective:
        #                     if isinstance(req, CapabilityRequirement) and req.attribute == 'instrument':
        #                         valid_instruments.update(set(req.valid_values))
        #                         observations_required[objective.parameter] = set(req.valid_values)
            
        #     # get required measurements for a given event
        #     # observations_req = set(str_to_list(event[-1]))
        #     # instruments = {instrument for *_, instrument in observations if instrument in observations_req}

        #     # check if there are observations that satisfy the requirements of the request
        #     if len(observations_required) > 1:
        #         # check if valid co-observations match this even
        #         co_observations = {(*_, instrument) 
        #                            for *_, instrument in observations 
        #                            if any([instrument.lower() in observations_required[param]
        #                                    for param in observations_required])
        #                            }

        #         # TODO find which observations may have triggered co-observations
        #         # if event in events_requested:
        #         #     requesting_observations = {
        #         #         (lat, lon, t_start, duration, severity, observer, t_img, instrument)
        #         #         for lat, lon, t_start, duration, severity, observer, t_img, instrument in co_observations
        #         #         for _, requester, _, _, _, t_start_req, *_ in events_requested[event]
        #         #         if abs(t_start_req - t_img) <= 1e-3 and requester == observer
        #         #     }
        #         #     # remove requesting observations from co-observations (if any)
        #         #     co_observations.difference_update(requesting_observations)

        #         co_observed_parameters = {param 
        #                                     for param in observations_required
        #                                     for *_, instrument in access_intervals 
        #                                     if instrument.lower() in observations_required[param]}

        #         if co_observation_params.issubset(co_observed_parameters):
        #             # all required observation types were performed; event was fully co-observed
        #             events_co_obs_fully[event] = co_observations
        #         else:
        #             # some required observation types were performed; event was parially co-observed
        #             events_co_obs_partially[event] = co_observations

        #         # event is co-observed
                # events_co_obs[event] = co_observations

        assert all([event in events_co_observable for event in events_co_obs])
        assert all([event in events_co_observable and event in events_co_observable_fully for event in events_co_obs_fully])
        assert all([event in events_co_observable for event in events_co_obs_partially])

        # classify observations by tasks
        tasks_observable : Dict[GenericObservationTask, list] = defaultdict(list)
        tasks_observed : Dict[GenericObservationTask, list] = defaultdict(list)

        for task in known_tasks:
            # compile observation requirements for this task
            instrument_capability_reqs : Dict[str, set] = defaultdict(set)

            # group requirements by agents to avoid double counting
            for _,mission in self.missions.items():
                # find objectives matching this task
                if task.objective not in mission: continue # skip if objective not in mission

                # find agents belonging to this mission
                agents = [agent for agent in self.agents if agent.mission == mission]
                    
                # collect instrument capability requirements
                for req in task.objective:
                    # check if requirement is an instrument capability requirement
                    if (isinstance(req, ExplicitCapabilityRequirement) 
                        and req.attribute == 'instrument'):
                        for agent in agents:
                            instrument_capability_reqs[agent.get_element_name()].update({val.lower() for val in req.valid_values})

            # find all accesses and observations that match this task
            task_observations = []
            task_access_windows = []

            # check all task locations
            for location in task.location:
                # unpack location
                task_lat,task_lon,task_grid_idx, task_gp_idx = location
                task_lat = round(task_lat,6)
                task_lon = round(task_lon,6)
                        
                # find access times that overlook a given task's location
                matching_accesses = [
                                (t, row['agent name'], row['instrument'])
                                for agent_name, agent_orbit_data in compiled_orbitdata.items()
                                for t,row in agent_orbit_data.gp_access_data
                                if t in task.availability
                                and abs(task_lat - row['lat [deg]']) < 1e-3 
                                and abs(task_lon - row['lon [deg]']) < 1e-3
                                and row['instrument'].lower() in instrument_capability_reqs[agent_name]
                            ]
                
                # initialize map of compiled access intervals
                access_interval_dict : Dict[tuple,List[Interval]] = defaultdict(list)

                # compile list of access intervals
                for t_access, agent_name, instrument in matching_accesses:
                    # get propagation time step for this agent
                    time_step = compiled_orbitdata[agent_name].time_step 

                    # create unitary interval for this access time
                    access_interval = Interval(t_access, t_access + time_step)

                    # check if this access overlaps with any previous access
                    merged = False
                    for interval in access_interval_dict[(agent_name,instrument)]:
                        if access_interval.overlaps(interval):
                            # if so, join intervals
                            interval.join(access_interval)
                            merged = True
                            break
                    
                    # if merged, continue to next interval
                    if merged: continue
                    
                    # otherwise, create a new access interval
                    access_interval_dict[(agent_name,instrument)].append(access_interval)

                # flatten to list of access intervals
                access_intervals : list = sorted([ (interval,agent_name,instrument) 
                                            for (agent_name,instrument),intervals in access_interval_dict.items()
                                            for interval in intervals ])

                # find observations performed at task location while task was active
                matching_observations = [   (observer, t_obs_start, t_obs_end, img_lat, img_lon, instrument)
                                    # for observer,gp_idx_img,_,img_lat,img_lon,*__,grid_idx_img,instrument,agent_name,___,t_obs_start,t_obs_end in observations_per_gp[(task_lat, task_lon)].values
                                    for gp_idx_img,agent_name,grid_idx_img,_,_,_,instrument,img_lat,img_lon,_,_,observer,_,_,t_obs_end,t_obs_start,_ in observations_per_gp[(task_lat, task_lon)].values
                                                                        
                                    # check if observation time overlaps with event time
                                    if Interval(t_obs_start, t_obs_end).overlaps(task.availability)
                                    # check if instrument matches capability requirements
                                    and (instrument.lower() in instrument_capability_reqs[agent_name]
                                            or not instrument_capability_reqs[agent_name])
                                    
                                    # observation location sanity checks
                                    and grid_idx_img == task_grid_idx
                                    and gp_idx_img == task_gp_idx
                                    and abs(img_lat - task_lat) <= 1e-3 
                                    and abs(img_lon - task_lon) <= 1e-3
                                ] if (task_lat, task_lon) in observations_per_gp else []
                
                # append to task lists
                task_access_windows.extend(access_intervals)
                task_observations.extend(matching_observations)
                            
            if task_observations: tasks_observed[task] = task_observations
            if task_access_windows: tasks_observable[task] = task_access_windows

        return observations_per_gp, events_per_gp, gps_accessible, \
                events_observable, events_observed, events_detected, events_requested, \
                    events_re_observable, events_re_obs, \
                        events_co_observable, events_co_obs, \
                        events_co_observable_fully, events_co_obs_fully, \
                            events_co_observable_partially, events_co_obs_partially, \
                                tasks_observable, tasks_observed

    def __collect_event_observation(self, 
                                event : GeophysicalEvent, 
                                compiled_orbitdata : Dict[str, OrbitData],
                                events_detected : List[GeophysicalEvent], 
                                task_reqs :  List[TaskRequest], 
                                observations_per_gp : Dict[tuple, pd.DataFrame]) -> tuple:
        """ Finds accesses, detections, requests, and observations that match a given event."""

        # unpackage event
        event_lat,event_lon,event_grid_idx,event_gp_idx = event.location
        event_lat = round(event_lat,6)
        event_lon = round(event_lon,6)
        event_type = event.event_type.lower()
        event_id = event.id

        # compile observation requirements for this event type
        instrument_capability_reqs : Dict[str, set] = defaultdict(set)

        # group requirements by agents to avoid double counting
        for _,mission in self.missions.items():
            for objective in mission:
                # find agents belonging to this mission
                agents = [agent for agent in self.agents if agent.mission == mission]

                # check if objective matches event type
                if (isinstance(objective, EventDrivenObjective) 
                    and objective.event_type.lower() == event_type):
                    
                    # collect instrument capability requirements
                    for req in objective:
                        # check if requirement is an instrument capability requirement
                        if (isinstance(req, ExplicitCapabilityRequirement) 
                            and req.attribute == 'instrument'):
                            for agent in agents:
                                instrument_capability_reqs[agent.get_element_name()].update({val.lower() for val in req.valid_values})

        if any([len(instrument_capability_reqs[agent_name]) == 0 
                    for agent_name in instrument_capability_reqs]):
            raise NotImplementedError(f"No instrument capability requirements found for event type `{event_type}`. Case not yet supported.")

        # find access times that overlook a given event's location
        matching_accesses = [
                                (t, row['agent name'], row['instrument'])
                                for agent_name, agent_orbit_data in compiled_orbitdata.items()
                                for t,row in agent_orbit_data.gp_access_data
                                if event.t_start <= t <= event.t_start+event.d_exp
                                and abs(event_lat - row['lat [deg]']) < 1e-3 
                                and abs(event_lon - row['lon [deg]']) < 1e-3
                                and row['instrument'].lower() in instrument_capability_reqs[agent_name]
                            ]
        
        # initialize map of compiled access intervals
        access_interval_dict : Dict[tuple,List[Interval]] = defaultdict(list)

        # compile list of access intervals
        for t_access, agent_name, instrument in matching_accesses:
            # get propagation time step for this agent
            time_step = compiled_orbitdata[agent_name].time_step 

            # create unitary interval for this access time
            access_interval = Interval(t_access, t_access + time_step)

            # check if this access overlaps with any previous access
            merged = False
            for interval in access_interval_dict[(agent_name,instrument)]:
                if access_interval.overlaps(interval):
                    # if so, join intervals
                    interval.join(access_interval)
                    merged = True
                    break
            
            # if merged, continue to next interval
            if merged: continue
            
            # otherwise, create a new access interval
            access_interval_dict[(agent_name,instrument)].append(access_interval)

        # flatten to list of access intervals
        access_intervals : list = sorted([ (interval,agent_name,instrument) 
                                    for (agent_name,instrument),intervals in access_interval_dict.items()
                                    for interval in intervals ])

        # find measurement detections that match this event
        matching_detections = [event_detected for event_detected in events_detected if event_detected.id == event_id]
        matching_detections.sort(key= lambda a : a.t_detect)
        
        # find measurement requests that match this event
        if any(not isinstance(task_req.task, EventObservationTask) for task_req in task_reqs):
            raise NotImplementedError("Non-event observation tasks are not yet supported in event observation classification.")
        else:
            matching_requests = sorted([task_req for task_req in task_reqs if task_req.task.event == event], 
                                        key= lambda a : a.t_req)

        # find observations that overlooked a given event's location
        matching_observations = [   (observer, t_obs_start, t_obs_end, img_lat, img_lon, instrument)
                                    # for observer,gp_idx_img,_,img_lat,img_lon,*__,grid_idx_img,instrument,agent_name,___,t_obs_start,t_obs_end in observations_per_gp[(event_lat, event_lon)].values
                                    for gp_idx_img,agent_name,grid_idx_img,_,_,_,instrument,img_lat,img_lon,_,_,observer,_,_,t_obs_end,t_obs_start,_ in observations_per_gp[(event_lat, event_lon)].values
                                                                        
                                    # check if observation time overlaps with event time
                                    if Interval(t_obs_start, t_obs_end).overlaps(event.availability)
                                    # check if instrument matches capability requirements
                                    and instrument.lower() in instrument_capability_reqs[agent_name]
                                    
                                    # observation location sanity checks
                                    and grid_idx_img == event_grid_idx
                                    and gp_idx_img == event_gp_idx
                                    and abs(img_lat - event_lat) <= 1e-3 
                                    and abs(img_lon - event_lon) <= 1e-3
                                ] if (event_lat, event_lon) in observations_per_gp else []
        matching_observations.sort(key= lambda a : a[1])  # sort by observation start time

        # return classified data
        return access_intervals, matching_detections, matching_requests, matching_observations

    def str2interval(self, s : str) -> Interval:
        s = s.replace(']','')
        s = s.replace('[','')
        s = s.split(',')

        vals = [float(val) for val in s]
        return Interval(vals[0],vals[1])

    def _count_observations(self, 
                            orbitdata : dict, 
                            observations_performed : pd.DataFrame, 
                            observations_per_gp : dict,
                            events : List[GeophysicalEvent],
                            events_per_gp : dict,
                            events_observable : dict,
                            events_detected : dict, 
                            events_requested : dict,
                            events_observed : dict, 
                            events_re_observable : dict,
                            events_re_obs : dict, 
                            events_co_observable : dict,
                            events_co_obs : dict, 
                            events_co_observable_fully : dict,
                            events_co_obs_fully : dict, 
                            events_co_observable_partially : dict,
                            events_co_obs_partially : dict,
                            tasks_known : list,
                            tasks_observable : dict,
                            tasks_observed : dict
                        ) -> tuple:
        
        # count number of groundpoints and their accessibility
        n_gps = None
        gps_accessible_compiled = set()
        for _,agent_orbitdata in tqdm(orbitdata.items(), desc='Counting total and accessible ground points', leave=False):
            agent_orbitdata : OrbitData

            # count number of ground points
            n_gps = sum([len(gps.values) for gps in agent_orbitdata.grid_data]) if n_gps is None else n_gps

            # get set of accessible ground points
            gps_accessible : set = {(row['grid index'], row['GP index']) for _,row in agent_orbitdata.gp_access_data}

            # update set of accessible ground points
            gps_accessible_compiled.update(gps_accessible)

        n_gps_accessible = len(gps_accessible_compiled)
        n_gps_observed = len(observations_per_gp)

        # count number of groun point reobservations
        n_gps_reobserved = len([gp for gp,observations in observations_per_gp.items() 
                                if len(observations) > 1])
        
        # count number of observations performed
        n_observations = len(observations_performed)
        
        # count number of events
        n_events = len(events)

        # count number of groundpoints with events
        n_gps_with_events = len(events_per_gp)

        # count events observable
        n_events_observable = len(events_observable)

        # count event detections
        n_events_detected = len(events_detected)

        # count events with meaurement requests
        n_events_requested = len(events_requested)

        # count event observations
        n_events_observed = len(events_observed)
        n_total_event_obs = sum([len(observations) for _,observations in events_observed.items()])

        assert n_events_detected <= n_events_observed, f"Detected events ({n_events_detected}) should be a subset of observed events ({n_events_observed})."
        assert n_total_event_obs <= n_observations

        # count events reobservable
        n_events_reobservable = len(events_re_observable)

        # count event reobservations
        n_events_reobserved = len(events_re_obs)
        n_total_event_re_obs = sum([len(re_observations) for _,re_observations in events_re_obs.items()])

        assert n_events_reobserved <= n_events_observed
        assert n_total_event_re_obs <= n_observations

        # count events co-observable
        n_events_co_observable = len(events_co_observable)
        n_events_co_observable_fully = len(events_co_observable_fully)
        n_events_co_observable_partially = len(events_co_observable_partially)

        # count event co-observations
        n_events_co_obs = len(events_co_obs)
        n_total_event_co_obs = sum([len(co_observations) for _,co_observations in events_co_obs.items()])        

        assert n_events_co_obs <= n_events_observed
        assert n_total_event_co_obs <= n_observations

        n_events_fully_co_obs = len(events_co_obs_fully)
        n_total_event_fully_co_obs = sum([len(full_co_observations) for _,full_co_observations in events_co_obs_fully.items()])        
        
        assert n_events_fully_co_obs <= n_events_observed
        assert n_total_event_fully_co_obs <= n_total_event_co_obs

        n_events_partially_co_obs = len(events_co_obs_partially)
        n_total_event_partially_co_obs = sum([len(partial_co_observations) for _,partial_co_observations in events_co_obs_partially.items()])        

        assert n_events_partially_co_obs <= n_events_observed
        assert n_total_event_partially_co_obs <= n_total_event_co_obs

        assert n_events_co_obs == n_events_fully_co_obs + n_events_partially_co_obs
        assert n_total_event_co_obs == n_total_event_fully_co_obs + n_total_event_partially_co_obs

        # count observations per task
        n_tasks = len(tasks_known)
        n_event_tasks = len([task for task in tasks_known if isinstance(task, EventObservationTask)])
        n_default_tasks = len([task for task in tasks_known if isinstance(task, DefaultMissionTask)]) 

        assert n_event_tasks + n_default_tasks <= n_tasks

        n_tasks_observable = len(tasks_observable)
        n_event_tasks_observable = len([task for task in tasks_observable if isinstance(task, EventObservationTask)])
        n_default_tasks_observable = len([task for task in tasks_observable if isinstance(task, DefaultMissionTask)])

        assert n_tasks_observable <= n_tasks        
        
        n_tasks_observed = len(tasks_observed)
        n_event_tasks_observed = len([task for task in tasks_observed if isinstance(task, EventObservationTask)])
        n_default_tasks_observed = len([task for task in tasks_observed if isinstance(task, DefaultMissionTask)])
        
        assert n_tasks_observed <= n_tasks_observable
        assert n_event_tasks_observed + n_default_tasks_observed <= n_tasks_observed

        # count reobservations per task
        n_tasks_reobservable = len([task for task,access_intervals in tasks_observable.items() 
                                    if len(access_intervals) > 1])
        n_event_tasks_reobservable = len([task for task in tasks_observable if isinstance(task, EventObservationTask)
                                        and len(tasks_observable[task]) > 1])
        n_default_tasks_reobservable = len([task for task in tasks_observable if isinstance(task, DefaultMissionTask)
                                        and len(tasks_observable[task]) > 1])
        
        assert n_tasks_reobservable <= n_tasks_observable
        assert n_event_tasks_reobservable + n_default_tasks_reobservable <= n_tasks_reobservable

        n_tasks_reobserved = len([task for task,observations in tasks_observed.items() 
                                if len(observations) > 1])
        n_event_tasks_reobserved = len([task for task in tasks_observed if isinstance(task, EventObservationTask)
                                        and len(tasks_observed[task]) > 1])
        n_default_tasks_reobserved = len([task for task in tasks_observed if isinstance(task, DefaultMissionTask)
                                        and len(tasks_observed[task]) > 1])
        
        assert n_tasks_reobserved <= n_tasks_reobservable 
        assert n_tasks_reobserved <= n_tasks_observed
        assert n_event_tasks_reobserved + n_default_tasks_reobserved <= n_tasks_reobserved

        # return values
        return n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
                n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                    n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                        n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                            n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                                n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                    n_tasks, n_event_tasks, n_default_tasks, \
                                        n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                            n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                                n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                    n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved

    def _calc_event_probabilities(self,
                                    orbitdata : dict, 
                                    gps_accessible : dict,
                                    observations_performed : pd.DataFrame, 
                                    observations_per_gp : dict,
                                    events : pd.DataFrame,
                                    events_per_gp : dict,
                                    events_observable : dict,
                                    events_detected : dict, 
                                    events_requested : dict,
                                    events_observed : dict, 
                                    events_re_observable : dict,
                                    events_re_obs : dict, 
                                    events_co_observable : dict,
                                    events_co_obs : dict, 
                                    events_co_observable_fully : dict,
                                    events_co_obs_fully : dict, 
                                    events_co_observable_partially : dict,
                                    events_co_obs_partially : dict,
                                    tasks_known : list,
                                    tasks_observable : dict,
                                    tasks_observed : dict
                                ) -> tuple:

        # count observations by type
        n_observations, n_gps, n_gps_accessible, n_gps_reobserved, n_gps_observed, n_gps_with_events, \
            n_events, n_events_observable, n_events_detected, n_events_requested, n_events_observed, n_total_event_obs, \
                n_events_reobservable, n_events_reobserved, n_total_event_re_obs, \
                    n_events_co_observable, n_events_co_obs, n_total_event_co_obs, \
                        n_events_co_observable_fully, n_events_fully_co_obs, n_total_event_fully_co_obs, \
                            n_events_co_observable_partially, n_events_partially_co_obs, n_total_event_partially_co_obs, \
                                n_tasks, n_event_tasks, n_default_tasks, \
                                    n_tasks_observable, n_event_tasks_observable, n_default_tasks_observable, \
                                        n_tasks_observed, n_event_tasks_observed, n_default_tasks_observed, \
                                            n_tasks_reobservable, n_event_tasks_reobservable, n_default_tasks_reobservable, \
                                                n_tasks_reobserved, n_event_tasks_reobserved, n_default_tasks_reobserved \
                                                    = self._count_observations( orbitdata, 
                                                                                observations_performed, 
                                                                                observations_per_gp,
                                                                                events, 
                                                                                events_per_gp,
                                                                                events_observable,
                                                                                events_detected, 
                                                                                events_requested,
                                                                                events_observed, 
                                                                                events_re_observable,
                                                                                events_re_obs, 
                                                                                events_co_observable,
                                                                                events_co_obs, 
                                                                                events_co_observable_fully,
                                                                                events_co_obs_fully, 
                                                                                events_co_observable_partially,
                                                                                events_co_obs_partially,
                                                                                tasks_known,
                                                                                tasks_observable,
                                                                                tasks_observed)
                    
        # count number of ground points accessible and observed 
        n_gps_observed_and_accessible = len(observations_per_gp)
        n_gps = n_gps if n_gps is not None else 0

        # count number of obseved and detected events
        n_events_observed_and_observable = len([event for event in events_observed
                                                if event in events_observable])
        n_events_observed_and_detected = len([event for event in events_observed
                                                if event in events_detected])
        n_events_observed_and_observable_and_detected = len([event for event in events_observed
                                                            if event in events_observable
                                                            and event in events_detected])
        n_events_observable_and_detected = len([event for event in events_observable
                                                if event in events_detected])
            
        # count number of re-obseved and events
        n_events_re_obs_and_reobservable = len([event for event in events_re_obs
                                                if event in  events_re_observable])
        n_events_re_obs_and_detected = len([event for event in events_re_obs
                                            if event in events_detected])
        n_events_re_obs_and_reobservable_and_detected = len([event for event in events_re_obs
                                                            if event in  events_re_observable
                                                            and event in events_detected])
        n_events_re_observable_and_detected = len([event for event in events_re_observable
                                                    if event in events_detected])

        # count number of co-observed and detected events 
        n_events_co_obs_and_co_observable = len([event for event in events_co_obs
                                                if event in events_co_observable])
        n_events_co_obs_and_detected = len([event for event in events_co_obs
                                                if event in events_detected])
        n_events_co_obs_and_co_observable_and_detected = len([event for event in events_co_obs
                                                                if event in events_co_observable
                                                                and event in events_detected])
        n_events_co_observable_and_detected = len([event for event in events_co_observable
                                                    if event in events_detected])

        
        n_events_fully_co_obs_and_fully_co_observable = len([event for event in events_co_obs_fully
                                                            if event in events_co_observable_fully])
        n_events_fully_co_obs_and_detected = len([event for event in events_co_obs_fully
                                                    if event in events_detected])
        n_events_fully_co_obs_and_fully_co_observable_and_detected = len([event for event in events_co_obs_fully
                                                                            if event in events_co_observable_fully
                                                                            and event in events_detected])
        n_events_fully_co_observable_and_detected = len([event for event in events_co_observable_fully
                                                        if event in events_detected])
        
        n_events_partially_co_obs_and_partially_co_observable = len([event for event in events_co_obs_partially
                                                                    if event in events_co_observable_partially])
        n_events_partially_co_obs_and_detected = len([event for event in events_detected
                                                    if event in events_co_obs_partially])
        n_events_partially_co_obs_and_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                                            if event in events_co_obs_partially
                                                                            and event in events_detected])
        n_events_partially_co_observable_and_detected = len([event for event in events_co_observable_partially
                                                            if event in events_detected])

        # calculate event probabilities
        p_gp_accessible = n_gps_accessible / n_gps if n_gps > 0 else np.NAN
        p_gp_observed = n_gps_observed / n_gps if n_gps > 0 else np.NAN
        p_event_at_gp = n_gps_with_events / n_gps if n_gps > 0 else np.NAN
        p_event_observable = n_events_observable / n_events if n_events > 0 else np.NAN
        p_event_detected = n_events_detected / n_events if n_events > 0 else np.NAN
        p_event_observed = n_events_observed / n_events if n_events > 0 else np.NAN
        p_event_re_observable = n_events_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs = n_events_reobserved / n_events if n_events > 0 else np.NAN
        p_event_co_observable = n_events_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs = n_events_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_fully = n_events_co_observable_fully / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully = n_events_fully_co_obs / n_events if n_events > 0 else np.NAN
        p_event_co_observable_partial = n_events_co_observable_partially / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial = n_events_partially_co_obs / n_events if n_events > 0 else np.NAN    

        # calculate event joint probabilities
        p_gp_observed_and_accessible = p_gp_observed

        p_event_observed_and_observable = n_events_observed_and_observable / n_events if n_events > 0 else np.NAN
        p_event_observed_and_detected = n_events_observed_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observed_and_observable_and_detected = n_events_observed_and_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_observable_and_detected = n_events_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_re_obs_and_reobservable = n_events_re_obs_and_reobservable / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_detected = n_events_re_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_obs_and_reobservable_and_detected = n_events_re_obs_and_reobservable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_re_observable_and_detected = n_events_re_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_and_co_observable = n_events_co_obs_and_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_detected = n_events_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_obs_and_co_observable_and_detected = n_events_co_obs_and_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_co_observable_and_detected = n_events_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_fully_and_co_observable_fully = n_events_fully_co_obs_and_fully_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_fully_and_detected = n_events_fully_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_obs_and_fully_co_observable_and_detected = n_events_fully_co_obs_and_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_fully_co_observable_and_detected = n_events_fully_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        p_event_co_obs_partially_and_co_observable_partially = n_events_partially_co_obs_and_partially_co_observable / n_events if n_events > 0 else np.NAN
        p_event_co_obs_partial_and_detected = n_events_partially_co_obs_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_obs_and_partially_co_observable_and_detected = n_events_partially_co_obs_and_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN
        p_event_partially_co_observable_and_detected = n_events_partially_co_observable_and_detected / n_events if n_events > 0 else np.NAN

        # calculate event conditional probabilities
        p_gp_observed_if_accessible = p_gp_observed_and_accessible / p_gp_accessible if p_gp_accessible > 0.0 else np.NAN

        p_event_obs_if_obs = n_total_event_obs / n_observations if n_observations > 0 else np.NAN
        p_event_re_obs_if_obs = n_total_event_re_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_if_obs = n_total_event_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_fully_if_obs = n_total_event_fully_co_obs / n_observations if n_observations > 0 else np.NAN
        p_event_co_obs_partially_if_obs = n_total_event_partially_co_obs / n_observations if n_observations > 0 else np.NAN
        
        p_event_observed_if_observable = p_event_observed_and_observable / p_event_observable if p_event_observable > 0.0 else np.NAN
        p_event_observed_if_detected = p_event_observed_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_observed_if_observable_and_detected = p_event_observed_and_observable_and_detected / p_event_observable_and_detected if p_event_observable_and_detected > 0.0 else np.NAN
        
        p_event_re_obs_if_re_observable = p_event_re_obs_and_reobservable / p_event_re_observable if p_event_re_observable > 0.0 else np.NAN
        p_event_re_obs_if_detected = p_event_re_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_re_obs_if_reobservable_and_detected = p_event_re_obs_and_reobservable_and_detected / p_event_re_observable_and_detected if p_event_re_observable_and_detected > 0.0 else np.NAN
        
        p_event_co_obs_if_co_observable = p_event_co_obs_and_co_observable / p_event_co_observable if p_event_co_observable > 0.0 else np.NAN
        p_event_co_obs_if_detected = p_event_co_obs_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_if_co_observable_and_detected = p_event_co_obs_and_co_observable_and_detected / p_event_co_observable_and_detected if p_event_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_fully_if_co_observable_fully = p_event_co_obs_fully_and_co_observable_fully / p_event_co_observable_fully if p_event_co_observable_fully > 0.0 else np.NAN
        p_event_co_obs_fully_if_detected = p_event_co_obs_fully_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_fully_if_co_observable_fully_and_detected = p_event_fully_co_obs_and_fully_co_observable_and_detected / p_event_fully_co_observable_and_detected if p_event_fully_co_observable_and_detected > 0 else np.NAN

        p_event_co_obs_partial_if_co_observable_partially = p_event_co_obs_partially_and_co_observable_partially / p_event_co_observable_partial if p_event_co_observable_partial > 0.0 else np.NAN
        p_event_co_obs_partial_if_detected = p_event_co_obs_partial_and_detected / p_event_detected if p_event_detected > 0.0 else np.NAN
        p_event_co_obs_partial_if_co_observable_partially_and_detected = p_event_partially_co_obs_and_partially_co_observable_and_detected / p_event_partially_co_observable_and_detected if p_event_partially_co_observable_and_detected > 0 else np.NAN

        # calculate task propabilites
        p_task_observable = n_tasks_observable / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_observable = n_event_tasks_observable / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_observable = n_default_tasks_observable / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_observed = n_tasks_observed / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_observed = n_event_tasks_observed / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_observed = n_default_tasks_observed / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_observed_if_observable = n_tasks_observed / n_tasks_observable if n_tasks_observable > 0 else np.NAN
        p_event_task_observed_if_observable = n_event_tasks_observed / n_event_tasks_observable if n_event_tasks_observable > 0 else np.NAN
        p_default_task_observed_if_observable = n_default_tasks_observed / n_default_tasks_observable if n_default_tasks_observable > 0 else np.NAN

        # calculate task reobservation probabilities
        p_task_reobserved = n_tasks_reobserved / n_tasks if n_tasks > 0 else np.NAN
        p_event_task_reobserved = n_event_tasks_reobserved / n_event_tasks if n_event_tasks > 0 else np.NAN
        p_default_task_reobserved = n_default_tasks_reobserved / n_default_tasks if n_default_tasks > 0 else np.NAN

        p_task_reobserved_if_reobservable = n_tasks_reobserved / n_tasks_reobservable if n_tasks_reobservable > 0 else np.NAN
        p_event_task_reobserved_if_reobservable = n_event_tasks_reobserved / n_event_tasks_reobservable if n_event_tasks_reobservable > 0 else np.NAN
        p_default_task_reobserved_if_reobservable = n_default_tasks_reobserved / n_default_tasks_reobservable if n_default_tasks_reobservable > 0 else np.NAN

        return p_gp_accessible, p_gp_observed, p_gp_observed_if_accessible, p_event_at_gp, p_event_detected, \
                p_event_obs_if_obs, p_event_re_obs_if_obs, p_event_co_obs_if_obs, p_event_co_obs_fully_if_obs, p_event_co_obs_partially_if_obs, \
                    p_event_observable, p_event_observed, p_event_observed_if_observable, p_event_observed_if_detected, p_event_observed_if_observable_and_detected, \
                        p_event_re_observable, p_event_re_obs, p_event_re_obs_if_re_observable, p_event_re_obs_if_detected, p_event_re_obs_if_reobservable_and_detected, \
                            p_event_co_observable, p_event_co_obs, p_event_co_obs_if_co_observable, p_event_co_obs_if_detected, p_event_co_obs_if_co_observable_and_detected, \
                                p_event_co_observable_fully, p_event_co_obs_fully, p_event_co_obs_fully_if_co_observable_fully, p_event_co_obs_fully_if_detected, p_event_co_obs_fully_if_co_observable_fully_and_detected, \
                                    p_event_co_observable_partial, p_event_co_obs_partial, p_event_co_obs_partial_if_co_observable_partially, p_event_co_obs_partial_if_detected, p_event_co_obs_partial_if_co_observable_partially_and_detected, \
                                        p_task_observable, p_event_task_observable, p_default_task_observable, p_task_observed, p_event_task_observed, p_default_task_observed, p_task_observed_if_observable, p_event_task_observed_if_observable, p_default_task_observed_if_observable, \
                                            p_task_reobserved, p_event_task_reobserved, p_default_task_reobserved, p_task_reobserved_if_reobservable, p_event_task_reobserved_if_reobservable, p_default_task_reobserved_if_reobservable

    def _calc_groundpoint_coverage_metrics(self,
                                    observations_per_gp: Dict[tuple, pd.DataFrame]
                                    ) -> tuple:
        # event reobservation times
        t_reobservations : list = []
        for _,observations in observations_per_gp.items():
            prev_observation = None

            for observation in observations.values:
                if prev_observation is None:
                    prev_observation = observation
                    continue

                # get observation times
                *_,t_start,_ = observation
                *_,t_prev_end,__,___ = prev_observation

                # calculate revisit
                # t : Interval = Interval(t_start, t_end)
                # t_prev : Interval = Interval(t_prev_start, t_prev_end)
                t_reobservation = t_start - t_prev_end

                # add to list
                t_reobservations.append(t_reobservation)

                # update previous observation
                prev_observation = observation
        
        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation

    def _calc_event_coverage_metrics(self, events_observed : dict) -> tuple:
        t_reobservations : list = []
        for event in events_observed:
            prev_observation = None
            for observation in events_observed[event]:
                if prev_observation is None:
                    prev_observation = observation
                    continue

                # get observation times - (lat, lon, t_start, duration, severity, observer, t_img, instrument, observations_req)
                _,_,t_prev_end,*_ = prev_observation
                _,t_start,*_ = observation

                # calculate revisit
                t_reobservation = t_start - t_prev_end
                # add to list
                t_reobservations.append(t_reobservation)

                # update previous observation
                prev_observation = observation

        # compile statistical data
        t_reobservation : dict = {
            'mean' : np.average(t_reobservations) if t_reobservations else np.NAN,
            'std' : np.std(t_reobservations) if t_reobservations else np.NAN,
            'median' : np.median(t_reobservations) if t_reobservations else np.NAN,
            'data' : t_reobservations
        }

        return t_reobservation