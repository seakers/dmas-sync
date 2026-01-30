
import os
import shutil
from typing import List

def setup_results_directory(scenario_path : str, scenario_name : str, agent_names : List[str], overwrite : bool = True) -> str:
    """
    Creates an empty results directory within the current working directory
    """
    # define results paths
    results_path = os.path.join(scenario_path, 'results', scenario_name)
    agents_paths : List[str] = [os.path.join(results_path, agent_name.lower())
                                for agent_name in agent_names]

    # check if results path exists
    if (not os.path.exists(results_path) 
        and all(os.path.exists(agent_path) for agent_path in agents_paths) 
        and not overwrite):
        # path exists and no overwrite is enabled; return existing results path
        return results_path
    
    if os.path.exists(results_path) and overwrite:
        # path exists and results overwrite is enabled; clear results in case it already exists
        for filename in os.listdir(results_path):
            file_path = os.path.join(results_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # create a results directory for all agents
    for agent_name in agent_names:
        agent_name : str
        agent_results_path : str = os.path.join(results_path, agent_name.lower())
        os.makedirs(agent_results_path, exist_ok=True)

    return results_path