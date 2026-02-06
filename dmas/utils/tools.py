
from enum import Enum
import gc
import logging
import math
import os
import weakref

class SimulationRoles(Enum):
    ENVIRONMENT = 'ENVIRONMENT'
    AGENT = 'AGENT'

# class LEVELS(Enum):
#     DEBUG = logging.DEBUG
#     INFO = logging.INFO
#     WARNING = logging.WARNING
#     CRITICAL = logging.CRITICAL
#     ERROR = logging.ERROR

class MessageTracker:
    """ Tracks live message objects using weak references. """
    def __init__(self):
        self._refs = weakref.WeakSet()
        self._reffers : dict[int, list] = dict()

    def track(self, msg):
        self._refs.add(msg)
        # self._reffers[id(msg)] = gc.get_referrers(msg)

    def alive(self):
        gc.collect()
        # for msg in list(self._refs):
        #     self._reffers[id(msg)] = gc.get_referrers(msg)
        return len(self._refs)


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
