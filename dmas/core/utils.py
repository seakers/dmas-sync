from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from enum import Enum
import argparse
import logging
import math
import shutil
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from execsatm.utils import Interval

class CoordinateTypes(Enum):
    """
    # Coordinate Type

    Describes the type of coordinate being described by a position vector
    """
    CARTESIAN = 'CARTESIAN'
    KEPLERIAN = 'KEPLERIAN'
    LATLON = 'LATLON'

class ModuleTypes(Enum):
    """
    # Types of Internal Modules for agents 
    """
    PLANNER = 'PLANNER'
    SCIENCE = 'SCIENCE'
    ENGINEERING = 'ENGINEERING'

def setup_results_directory(scenario_path : str, scenario_name : str, agent_names : list, overwrite : bool = True) -> str:
    """
    Creates an empty results directory within the current working directory
    """
    results_path = os.path.join(scenario_path, 'results', scenario_name)

    if not os.path.exists(results_path):
        # create results directory if it doesn't exist
        os.makedirs(results_path)

    elif overwrite:
        # clear results in case it already exists
        for filename in os.listdir(results_path):
            file_path = os.path.join(results_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        # path exists and no overwrite 
        return results_path

    # create a results directory for all agents
    for agent_name in agent_names:
        agent_name : str
        agent_results_path : str = os.path.join(results_path, agent_name.lower())
        os.makedirs(agent_results_path)

    return results_path

def print_banner(scenario_name = None) -> None:
    # clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # construct banner string
    out = "\n======================================================"
    out += '\n   _____ ____        ________  __________________\n  |__  // __ \      / ____/ / / / ____/ ___/ ___/\n   /_ </ / / /_____/ /   / /_/ / __/  \__ \\__ \ \n ___/ / /_/ /_____/ /___/ __  / /___ ___/ /__/ / \n/____/_____/      \____/_/ /_/_____//____/____/ (v1.1)'
    out += "\n======================================================"
    out += '\n\tTexas A&M University - SEAK Lab ©'
    out += "\n======================================================"
    
    # include scenario name if provided
    if scenario_name is not None: out += f"\nSCENARIO: {scenario_name}"

    # print banner
    print(out)

def arg_parser() -> tuple:
    """
    Parses the input arguments to the command line when starting a simulation
    
    ### Returns:
        `scenario_name`, `plot_results`, `save_plot`, `no_graphic`, `level`
    """
    parser : argparse.ArgumentParser = argparse.ArgumentParser(prog='DMAS for 3D-CHESS',
                                                               description='Simulates an autonomous Earth-Observing satellite mission.',
                                                               epilog='- TAMU')

    parser.add_argument(    '-n',
                            '--scenario-name', 
                            help='name of the scenario being simulated',
                            type=str,
                            required=False,
                            default='none')
    parser.add_argument(    '-p', 
                            '--plot-result',
                            action='store_true',
                            help='creates animated plot of the simulation',
                            required=False,
                            default=False)    
    parser.add_argument(    '-s', 
                            '--save-plot',
                            action='store_true',
                            help='saves animated plot of the simulation as a gif',
                            required=False,
                            default=False) 
    parser.add_argument(    '-d', 
                            '--welcome-graphic',
                            action='store_true',
                            help='draws ascii welcome screen graphic',
                            required=False,
                            default=True)  
    parser.add_argument(    '-l', 
                            '--level',
                            choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'ERROR'],
                            default='WARNING',
                            help='logging level',
                            required=False,
                            type=str)  
                    
    args = parser.parse_args()
    
    scenario_name = args.scenario_name
    plot_results = args.plot_result
    save_plot = args.save_plot
    no_graphic = args.welcome_graphic

    levels = {  'DEBUG' : logging.DEBUG, 
                'INFO' : logging.INFO, 
                'WARNING' : logging.WARNING, 
                'CRITICAL' : logging.CRITICAL, 
                'ERROR' : logging.ERROR
            }
    level = levels.get(args.level)

    return scenario_name, plot_results, save_plot, no_graphic, level

def argmax(values, rel_tol=1e-9, abs_tol=0.0):
        """ returns the index of the highest value in a list of values """
        max_val = max(values)
        for i, val in enumerate(values):
            if math.isclose(val, max_val, rel_tol=rel_tol, abs_tol=abs_tol):
                return i        
            
        raise ValueError("No maximum value found in the list.")

def str_to_list(lst_string : str, list_type : type = str) -> list:
    """ reverts a list that has been printed into a string back into a list """
    
    # remove printed list brackets and quotes
    lst = lst_string.replace('[','')
    lst = lst.replace(']','')
    lst = lst.replace('\'','')
    lst = lst.replace(' ','')

    # convert into a string
    return [list_type(val) for val in lst.split(',')]

LEVELS = {  'DEBUG' : logging.DEBUG, 
            'INFO' : logging.INFO, 
            'WARNING' : logging.WARNING, 
            'CRITICAL' : logging.CRITICAL, 
            'ERROR' : logging.ERROR
        }


def monitoring(kwargs) -> float:
    """
    ### Monitoring Utility Function

    This function calculates the utility of a monitoring observation based on the time since the last observation.
    The utility is calculated as the time since the last observation divided by the total time available for monitoring.

    - :`observation`: The current observation.
    - :`unobserved_reward_rate`: The rate at which the reward decreases for unobserved events.    - :`latest_observation`: The latest observation.
    - :`kwargs`: Additional keyword arguments (not used in this function).
    
    """
    t_img = kwargs['t_start']
    t_prev = kwargs.get('t_prev',0.0)
    unobserved_reward_rate = kwargs.get('unobserved_reward_rate', 1.0)
    n_obs = kwargs.get('n_obs', 0)
        
    assert (t_img - t_prev) >= 0.0 # TODO fix acbba triggering this

    # calculate reward
    reward = (t_img - t_prev) * unobserved_reward_rate / 3600 
    
    # clip reward to [0, 1]
    reward = np.clip(reward, 0.0, 1.0)

    # return reward
    return reward


INTERPOLATION_IGNORED_COLUMNS = [
    'GP index',
    'grid index',
    'agent name',
    'pnt-opt index',
    'instrument'
    'lat [deg]',
    'lon [deg]', 
    ]

class AbstractData(ABC):
    """
    Base class for all database types.
    """
    def __init__(self, name : str, columns : list, data : list):
        self.name = name
        self.columns = columns
        self.data = data

    @abstractmethod
    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param') -> 'AbstractData':
        """ Creates an instance of the class from a pandas DataFrame. """
    
    @abstractmethod
    def update_expired_values(self, t :float) -> None:
        """ Updates the data by removing all values that are older than time `t`. """    

class TimeIndexedData(AbstractData):
    def __init__(self, 
                 name : str,
                 columns : list,
                 t : list,
                 data : Dict[str, np.ndarray],
                 bin_size : float = 3600 # in seconds, default 1 hour
                 ):
        
        # validate inputs
        assert isinstance(name, str), 'name must be a string'
        assert len(columns) == len(data), 'number of columns and data do not match'
        assert all([len(data[col]) == len(t) for col in columns]), 'number of time steps and data do not match'
        assert isinstance(t, (list, np.ndarray)) and all([isinstance(val, (int, float)) for val in t]), 't must be a list or numpy array of numbers'
        
        self.name : str = name
        self.columns : List[str] = columns
        self.t : List[float] = t
        self.data : Dict[str, np.ndarray] = data
        self.bin_size : float = bin_size
                
        # count number of bins
        self.n_bins = int(max(t, default=0) // bin_size) + 1

        # group data indices into bins depending on their time for faster lookup
        grouped_indices = [[] for _ in range(self.n_bins)]
        inv_bs = 1.0 / bin_size
        for i, ti in tqdm(enumerate(t), desc=f'Grouping time-indexed {name} time', unit=' time bins', leave=False):
            b = int(ti * inv_bs)
            if b >= self.n_bins:
                b = self.n_bins - 1
            grouped_indices[b].append(i)

        # assign grouped data
        self.grouped_t = [[t[i] for i in idx] for idx in grouped_indices]
        self.grouped_data = {
            col: [[data[col][i] for i in idx] for idx in grouped_indices]
            for col in columns
        }

        # TEMP original imlementation 
        # # group data into bins depending on their time for faster lookup
        # self.n_bins = ceil(max(t, default=0) / bin_size) 
        # grouped_indices : List[List[float]] = [
        #     [(t_indx, t_i) for t_indx,t_i in enumerate(t)
        #         if (i*self.bin_size) <= t_i < ((i+1)*self.bin_size)]
        #     for i in tqdm(range(self.n_bins), desc=f'Grouping time-indexed {name} time', unit=' time bins', leave=False)
        # ] if self.n_bins > 1 else [ [(t_indx, t_i) for t_indx,t_i in enumerate(t)] ]
        
        # self.grouped_t : List[List[float]] = [
        #     [t_i for _,t_i in grouped_indices[i]] for i in range(self.n_bins)
        # ] if self.n_bins > 1 else [ [t_i for _,t_i in grouped_indices[0]] ]

        # self.grouped_data : Dict[List[List[tuple]]] \
        #     = {col : [
        #             [vals[t_indx] for t_indx,_ in grouped_indices[i] ]
        #             for i in tqdm(range(self.n_bins), desc=f'Grouping `{col}` data', unit=' time bins', leave=False)
        #         ] for col,vals in tqdm(data.items(), desc=f'Grouping time-indexed {name} data', unit=' data columns', leave=False)}
        

    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param') -> 'TimeIndexedData':
        # validate inputs
        assert 'time index' in df.columns or 'time [s]' in df.columns, 'time column not found in dataframe'
        assert time_step > 0.0, 'time step must be greater than 0.0'

        # get data columns 
        columns = list(df.columns.values)
        
        # get appropriate time data
        if 'time index' in columns:
            # sort dataframe by time index
            df = df.sort_values(by=['time index'])

            # get time column index
            time_column_index = columns.index('time index')

            # remove time column from columns list
            columns.remove('time index')

            # get time data in seconds
            t = [val*time_step for val in np.array(df.iloc[:, time_column_index].to_numpy())]

        elif 'time [s]' in columns:
            # sort dataframe by time index
            df = df.sort_values(by=['time [s]'])

            # get time column index
            time_column_index = columns.index('time [s]')
            
            # remove time column from columns list
            columns.remove('time [s]')

            # get time data in seconds
            t = [val for val in np.array(df.iloc[:, time_column_index].to_numpy())]
        else:
            raise ValueError('time column not found in dataframe')

        # get data from dataframe and ignore time column
        data = {col : np.array(df[col]) for col in df.columns.values}
        if 'time index' in data:
            data.pop('time index')
        elif 'time [s]' in data:
            data.pop('time [s]')
                
        # return TimeIndexedData object
        return TimeIndexedData(name, columns, np.array(t), data)

    def lookup_value(self, t : float, columns : list = None) -> dict:
        """
        Returns the value of data at time `t` in seconds
        """
        # get desired columns
        columns = columns if columns is not None else self.columns

        # Choose bin
        if not np.isfinite(t):
            bin_index = len(self.grouped_t) - 1
        else:
            bin_index = int(t // self.bin_size)
            if bin_index >= len(self.grouped_t):
                bin_index = len(self.grouped_t) - 1
            elif bin_index < 0:
                bin_index = 0

        xp = self.grouped_t[bin_index]
        if len(xp) == 0:
            # empty bin; fall back (or return empties)
            return {**{col: np.nan for col in columns}, 'time [s]': t}

        # Clamp to edges like np.interp does
        if t <= xp[0]:
            out = {col: float(self.grouped_data[col][bin_index][0]) for col in columns}
            out['time [s]'] = t
            return out
        if t >= xp[-1]:
            out = {col: float(self.grouped_data[col][bin_index][-1]) for col in columns}
            out['time [s]'] = t
            return out

        # Find right index once
        j = int(np.searchsorted(xp, t, side="right"))
        i = j - 1

        x0 = xp[i]; x1 = xp[j]
        w = (t - x0) / (x1 - x0)

        out = {}
        for col in columns:
            fp = self.grouped_data[col][bin_index]
            y0 = fp[i]; y1 = fp[j]
            out[col] = float(y0 + w * (y1 - y0))
        out['time [s]'] = t
        return out

        # TEMP Original implementation
        # # check if there is any data
        # if not self.grouped_t: 
        #     return {col: [] for col in columns + ['time [s]']}

        # # find the bin to search
        # bin_index = min(int(t // self.bin_size), len(self.grouped_t) - 1) \
        #             if t < np.Inf else len(self.grouped_t) - 1 # ensure index is within bounds

        # # search for exact time in the appropriate bin
        # if bin_index < len(self.grouped_t):            
        #     # if exact time not found, interpolate the data to find the value at time `t` and return the data at the specified columns
        #     out = {col : np.interp(t, self.grouped_t[bin_index], self.grouped_data[col][bin_index]) 
        #             for col in columns}
        #     out['time [s]'] = t
        #     return out
        
        # else:
        #     # if exact time not found, interpolate the data to find the value at time `t` and return the data at the specified columns
        #     out = {col : np.interp(t, self.t, self.data[col]) 
        #             for col in columns}
        #     out['time [s]'] = t
        #     return out
    
    def lookup_interval(self, t_start : float, t_end : float, columns : list = None) -> Dict[str, list]:
        """
        Returns the value of data between the start and end times in seconds
        """
        assert t_start <= t_end, "start time must be less than end time"
        assert t_start >= 0.0, "start time must be greater than 0.0"

        columns = self.columns if columns is None else columns

        if not self.grouped_t:
            return {col: [] for col in (columns + ['time [s]'])}

        eps = 1e-6
        t0 = t_start - eps
        t1 = t_end + eps

        # clamp bins safely
        bin_start = int(t_start // self.bin_size)
        bin_end = len(self.grouped_t) - 1 if not np.isfinite(t_end) else min(int(t_end // self.bin_size), len(self.grouped_t) - 1)

        if bin_start < 0:
            bin_start = 0

        # If start bin beyond range, just use global arrays (or return empty)
        if bin_start >= len(self.grouped_t):
            return {col: [] for col in (columns + ['time [s]'])}

        # Collect per-bin slices, then concatenate once
        t_chunks = []
        idx_slices = []  # store (bin_i, slice(l, r)) so we reuse for each column

        for i in range(bin_start, bin_end + 1):
            t_bin = self.grouped_t[i]
            if len(t_bin) == 0:
                continue

            # t_bin must be sorted for searchsorted
            l = int(np.searchsorted(t_bin, t0, side="left"))
            r = int(np.searchsorted(t_bin, t1, side="right"))
            if r > l:
                idx_slices.append((i, l, r))
                t_chunks.append(t_bin[l:r])

        if not idx_slices:
            return {col: [] for col in (columns + ['time [s]'])}

        t_out = np.concatenate(t_chunks)

        out = {'time [s]': t_out.tolist()}

        for col in columns:
            v_chunks = []
            for i, l, r in idx_slices:
                v_chunks.append(self.grouped_data[col][i][l:r])
            out[col] = np.concatenate(v_chunks).tolist()

        # lengths match by construction
        return out
    
    # TEMP Original implementation
    # def lookup_interval(self, t_start : float, t_end : float, columns : list = None) -> Dict[str, list]:
    #     """
    #     Returns the value of data between the start and end times in seconds
    #     """
    #     # validata imputs
    #     assert t_start <= t_end, 'start time must be less than end time'
    #     assert t_start >= 0.0, 'start time must be greater than 0.0'

    #     # get desired columns
    #     columns = columns if columns is not None else self.columns

    #     # check if there is any data
    #     if not self.grouped_t: 
    #         return {col: [] for col in columns + ['time [s]']}

    #     # find the bin indices of the start and end times
    #     bin_index_start = int(t_start // self.bin_size)
    #     bin_index_end = min(int(t_end // self.bin_size), len(self.grouped_t) - 1) \
    #                     if t_end < np.Inf else len(self.grouped_t) - 1 # ensure end index is within bounds

    #     # search for data in appropriate bins
    #     if bin_index_start < len(self.grouped_t):      
    #         # search for data in the bins
    #         out : dict[np.array] = {col : [val 
    #                                         for i in range(bin_index_start, bin_index_end + 1)
    #                                         for t_i,val in zip(self.grouped_t[i], self.grouped_data[col][i])
    #                                         if t_start-1e-6 <= t_i <= t_end+1e-6]
    #                                 for col in columns}
            
    #         out['time [s]'] = [t_i for i in range(bin_index_start, bin_index_end + 1)
    #                             for t_i in self.grouped_t[i]
    #                             if t_start-1e-6 <= t_i <= t_end+1e-6]

    #     else:
    #         # find the indices of the start and end times
    #         i_start = np.searchsorted(self.t, t_start, side='left')
    #         i_end = np.searchsorted(self.t, t_end, side='right')

    #         # get the data between the start and end times
    #         out : dict[np.array] = {col : self.data[col][i_start:i_end]
    #                             for col in columns}
    #         out['time [s]'] = [t for t in self.t[i_start:i_end]]

    #     # ensure data lengths match
    #     assert all([len(out[col]) == len(out['time [s]']) for col in columns]), 'number of time steps and data do not match'
            
    #     # return the data between the start and end times
    #     return out
        
    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for i in range(len(self.t)):
            row = {col : self.data[col][i] for col in self.columns}
            t = self.t[i]
            yield (t,row)

    def update_expired_values(self, t : float):
        # only keep values that are still active or that haven't expired yet
        unexpired_indeces = [(i,t_i) for i, t_i in enumerate(self.t) 
                            if t_i >= t or abs(t_i - t) <= 1e-6]
        
        # to avoid empty data, keep the last value if all values have expired
        if not unexpired_indeces and self.t.size > 0:
            unexpired_indeces = [(len(self.t)-1, self.t[-1])]
        
        # update internal data
        self.t = np.array([t_i for _, t_i in unexpired_indeces])
        self.data = {col : np.array([self.data[col][i] for i, _ in unexpired_indeces]) 
                    for col in self.columns}

class IntervalData(AbstractData):
    def __init__(self, 
                 name : str,
                 columns : List[str],
                 data : List[tuple],
                 bin_size : float = 3600 # in seconds, default 1 hour
                 ):
        # validate inputs
        assert isinstance(name, str), 'name must be a string'
        assert isinstance(columns, list) and all([isinstance(col, str) for col in columns]), 'columns must be a list of strings'
        assert isinstance(data, list) and all([isinstance(row, tuple) and len(row) >= 2 for row in data]), 'data must be a list of tuples with at least 2 elements (start time, end time)'
        assert isinstance(bin_size, (int, float)) and bin_size > 0, 'bin_size must be a positive number'

        # set attributes
        self.name : str = name
        self.columns : List[str] = columns
        self.data : List[tuple] = data
        self.bin_size : float = bin_size

        # find maximum time to determine number of bins
        max_t_start = max([t_start for t_start,*_ in data], default=0.0)
        max_t_end = max([t_end for _,t_end,*_ in data], default=0.0)
        max_t = max(max_t_start, max_t_end)

        # store raw columns in parallel arrays for speed
        self.n_bins = max(1, int(max_t // self.bin_size) + 1)
        self.bin_to_data_indices = [[] for _ in range(self.n_bins)]

        # group data indices into bins depending on their interval for faster lookup
        for interval_idx, (t_start, t_end, *_) in tqdm(enumerate(data), desc=f'Grouping interval {name} data', unit=' time bins', leave=False):
            b0 = max(0, int(t_start // self.bin_size))
            b1 = min(self.n_bins - 1, int(t_end // self.bin_size))

            assert b1 >= b0, \
                'invalid bin indices computed for interval data'

            for b in range(b0, b1 + 1):
                self.bin_to_data_indices[b].append(interval_idx)

        # sort each bin by start time for early stopping during lookup
        for ids in tqdm(self.bin_to_data_indices, desc=f'Sorting interval {name} data indices', unit=' time bins', leave=False):
            ids.sort(key=lambda idx: data[idx][0])

        # TEMP Original implementation        
        # # group data into bins depending on their start time for faster lookup
        # self.n_bins = ceil(max([t_start for t_start,*_ in data], default=0) / bin_size)
        # self.grouped_data : List[List[tuple]] = [
        #     [(t_start, t_end, *row) for t_start,t_end,*row in self.data
        #      if (i*self.bin_size) <= t_start < ((i+1)*self.bin_size)]
        #     for i in tqdm(range(self.n_bins), desc=f'Grouping interval {name} data', unit=' time bins', leave=False)
        # ] if self.n_bins > 1 else [ self.data ]

    def from_dataframe(df : pd.DataFrame, time_step : float, name : str = 'param') -> 'IntervalData':
        assert time_step > 0.0, 'time step must be greater than 0.0'
        assert 'start index' in df.columns, 'start index column not found in dataframe'
        assert 'end index' in df.columns, 'end index column not found in dataframe'
        
        # sort dataframe by time index
        df.sort_values(by=['start index'])

        # get time column index
        if any(['index' in col for col in df.columns.values]):
            # replace time index with time in seconds
            columns = [col.replace('index', 'time [s]') for col in df.columns.values]
            
            # get time data in Inteval format
            data = [(t_start*time_step, t_end*time_step, *row) 
                    for t_start,t_end,*row in df.values]
        else:
            # get time column index
            columns = [col for col in df.columns.values]
            
            # get time data in Inteval format
            data = [(t_start, t_end, *row) for t_start,t_end,*row in df.values]

        # return IntervalData object
        return IntervalData(name, columns, data)
    
    def lookup(self, t : float) -> Tuple:
        """
        Returns interval that contains time `t`. Returns None if no interval contains time `t`
        """        
        # check if there is any data
        if len(self.data) == 0: return None

        # set tolerance for floating point comparisons
        eps = 1e-6

        # find appropriate bin to search
        bin_idx = int(t // self.bin_size)
        
        # bound bin index
        if bin_idx < 0: 
            bin_idx = 0
        if bin_idx >= self.n_bins: 
            bin_idx = self.n_bins - 1

        # define earliest matching interval
        earliest_interval_idx = None
        t_start_earliest = np.Inf

        # search for interval in appropriate bin
        for data_idx in self.bin_to_data_indices[bin_idx]:
            # get interval data
            t_start_i,t_end_i,*_ = self.data[data_idx]

            # check if interval time starts after `t`
            if t + eps < t_start_i:  
                # early stop because bin list is sorted by start time;
                # no need to check further intervals
                break

            # check if `t` is within interval
            if t_start_i - eps <= t <= t_end_i + eps:
                # choose whatever tie-break you want; here earliest start
                if earliest_interval_idx is None or t_start_i < t_start_earliest:
                    earliest_interval_idx = data_idx
                    t_start_earliest = t_start_i

        # return the earliest matching interval if found
        return tuple(self.data[earliest_interval_idx]) \
            if earliest_interval_idx is not None else None

    # TEMP Original implementation
    # def lookup(self, t : float) -> list:
    #     """
    #     Returns interval that contains time `t`. Returns None if no interval contains time `t`
    #     """
    #     # check if there is any data
    #     if not self.grouped_data: 
    #         return None

    #     # find appropriate bin to search
    #     bin_index = min(int(t // self.bin_size), len(self.grouped_data) - 1) \
    #                 if t < np.Inf else len(self.grouped_data) - 1 # ensure index is within bounds

    #     # search for interval in appropriate bin
    #     if bin_index < len(self.grouped_data):
    #         # get data in the bin to search
    #         search_data = self.grouped_data[bin_index]
            
    #         # search for interval in the bin
    #         intervals = [(t_start,t_end,row)
    #                     for t_start,t_end,*row in search_data
    #                     if t_start-1e-6 <= t <= t_end+1e-6]
    #     else:
    #         # if no bin was found, search all data
    #         intervals = [(t_start,t_end,row) 
    #                     for t_start,t_end,*row in self.data
    #                     if t_start-1e-6 <= t <= t_end+1e-6]
        
    #     # sort intervals by start time
    #     intervals.sort()

    #     # return the first matching interval or None if no interval was found
    #     return intervals[0] if intervals else None

    def lookup_intervals(self, t_start : float, t_end : float) -> List[Interval]:
        """
        Returns all intervals that overlap with the interval [t_start, t_end]
        """
        # validate inputs
        assert isinstance(t_start, (int,float)) and t_start >= 0.0, 'start time must be a positive number'
        assert isinstance(t_end, (int,float)) and t_end >= 0.0, 'end time must be a positive number'
        assert t_start <= t_end, 'start time must be less than end time'

        # check if there is any data
        if len(self.data) == 0: return []
        
        # set tolerance for floating point comparisons
        eps = 1e-6

        # set query interval with tolerance
        q0, q1 = t_start - eps, t_end + eps

        # find appropriate bins to search
        b0 = max(int(t_start // self.bin_size), 0)
        b1 = int(t_end // self.bin_size) if not np.isinf(t_end) else self.n_bins - 1
        
        # check if start bin is beyond range
        if b0 >= self.n_bins: 
            return []

        # bound bin indices
        b0 = min(self.n_bins - 1, b0)
        b1 = min(self.n_bins - 1, b1)

        assert b1 >= b0, \
            'invalid bin indices computed for interval data lookup'

        # store seen interval indices to avoid duplicates
        seen = set()
        
        # search for intervals in appropriate bins
        out = []
        for bin_idx in range(b0, b1 + 1):
            # search for intervals in the bin
            for data_idx in self.bin_to_data_indices[bin_idx]:
                # avoid duplicates
                if data_idx in seen: continue

                # mark interval as seen
                seen.add(data_idx)

                # unpack interval data
                t_start_i,t_end_i,*row = self.data[data_idx]

                # check if interval time starts after `t_end`
                if t_start_i > q1:  
                    # early stop because bin list is sorted by start time;
                    # no need to check further intervals
                    break

                if not (t_end_i < q0 or t_start_i > q1):
                    out.append((t_start_i, t_end_i, *row))

        # sort intervals by start time
        out.sort(key=lambda r: r[0])

        # return the matching intervals
        return [Interval(t_start_i, t_end_i) for t_start_i,t_end_i,*_ in out]
    
        # TEMP Original implementation
        # try:
        #     # check if there is any data
        #     if not self.grouped_data: return []

        #     # find appropriate bin to search
        #     bin_index_start = min(int(t_start // self.bin_size), len(self.grouped_data) - 1)
        #     bin_index_end = min(int(t_end // self.bin_size), len(self.grouped_data) - 1) \
        #                         if t_end < np.Inf else len(self.grouped_data) - 1

        #     # search for intervals in appropriate bins
        #     if bin_index_start < len(self.grouped_data):
        #         # compile list of bins to search
        #         bins_to_search = self.grouped_data[bin_index_start : bin_index_end + 1]
                
        #         # search for intervals in the bins
        #         intervals = [(t_start_i,t_end_i,*_)
        #                     for search_data in bins_to_search
        #                     for t_start_i,t_end_i,*_ in search_data
        #                     if not (t_end_i < t_start - 1e-6 or t_start_i > t_end + 1e-6)]
                
        #     else:
        #         # if no bin was found, search all data
        #         intervals = [(t_start_i,t_end_i) 
        #                     for t_start_i,t_end_i,*_ in self.data
        #                     if not (t_end_i < t_start - 1e-6 or t_start_i > t_end + 1e-6)]
            
        #     # sort intervals by start time
        #     intervals.sort()
            
        #     # return clipped intervals that match the requested interval
        #     return [Interval(max(t_start_i, t_start),(min(t_end_i, t_end))) 
        #             for t_start_i,t_end_i in intervals]
        # except Exception as e:
        #     x = 1 
        #     raise e
    
    def is_active(self, t : float) -> bool:
        """
        Returns True if time `t` is in any of the intervals
        """
        current_interval = self.lookup(t)
        return current_interval is not None
        # return any([t_start-1e-6 <= t <= t_end+1e-6 for t_start,t_end,*_ in self.data])
    
    def update_expired_values(self, t : float) -> None:
        """ 
        Updates the data by removing all intervals that have ended before time `t`. 
        """
        # only keep intervals that are still active or that haven't expired yet
        data = [(t_start,t_end,*row) for t_start,t_end,*row in self.data
                    if t <= t_end or abs(t - t_end) <= 1e-6]
        
        # update internal data if there are any unexpired intervals;
        #  to avoid empty data, keep the last value if all values have expired
        self.data = [self.data[-1]] if not data and self.data else data
        
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        """
        Returns an iterator over the data
        """
        for row in self.data:
            yield row