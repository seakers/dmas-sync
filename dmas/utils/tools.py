
from enum import Enum
import logging
import math
import os

class SimulationRoles(Enum):
    ENVIRONMENT = 'ENVIRONMENT'
    AGENT = 'AGENT'

# class LEVELS(Enum):
#     DEBUG = logging.DEBUG
#     INFO = logging.INFO
#     WARNING = logging.WARNING
#     CRITICAL = logging.CRITICAL
#     ERROR = logging.ERROR


LEVELS = {  'DEBUG' : logging.DEBUG, 
            'INFO' : logging.INFO, 
            'WARNING' : logging.WARNING, 
            'CRITICAL' : logging.CRITICAL, 
            'ERROR' : logging.ERROR
        }

def print_scenario_banner(scenario_name = None, clear : bool = True) -> str:
    # clear the console if specified
    if clear: os.system('cls' if os.name == 'nt' else 'clear')

    # construct banner string
    out = "\n======================================================"
    out += '\n\t    ____  __  ______   _____\n\t   / __ \/  |/  /   | / ___/\n\t  / / / / /|_/ / /| | \__ \ \n\t / /_/ / /  / / ___ |___/ / \n\t/_____/_/  /_/_/  |_/____/  (v2.0.0)'
    # out += '\n______________________________________________________'
    out += '\n'
    out += '\n   Decentralized Multi-Agent System Simulation Tool'
    out += "\n======================================================"
    out += '\n\t Texas A&M University - SEAK Lab ©'
    out += "\n======================================================"
    
    # include scenario name if provided
    if scenario_name is not None: out += f"\nSCENARIO: {scenario_name}"

    # print banner
    print(out)

    # return string if needed
    return out

def argmax(values, rel_tol=1e-9, abs_tol=0.0):
        """ returns the index of the highest value in a list of values """
        max_val = max(values)
        for i, val in enumerate(values):
            if math.isclose(val, max_val, rel_tol=rel_tol, abs_tol=abs_tol):
                return i        
            
        raise ValueError("No maximum value found in the list.")

"""
Improved dual bound calculation.

estimate_task_performance_metrics is called inside the BO objective on every
function evaluation, since measurement_performance depends on t_obs through
d_obs and potentially other fields. Pre-computation is not valid here.
"""







