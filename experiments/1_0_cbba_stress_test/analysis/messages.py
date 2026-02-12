import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def generate_plot(trial_name : str, 
         scenario_id : int,
         bin_width: float = 10.0,         # time bin size (seconds)
         title: str = "Messages over time with task releases",
         base_dir = None,                   
         max_task : int = 300,
         show_plot : bool = True,
         save_plot : bool = False
        ) -> None:
    """ Generates and saves messages vs time plot from experiment results."""

    if base_dir is None:
        base_dir = os.path.join('experiments','1_0_cbba_stress_test','results')

    # define results directory
    results_dir = f'{trial_name}_scenario_{scenario_id}'
    
    # assumes script is being run from root directory
    results_path = os.path.join(base_dir, results_dir)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results directory not found at: `{results_path}`")

    # --- load data ---
    # load events
    events_path = os.path.join('experiments','1_0_cbba_stress_test', 'resources', 'events', f'scenario_{scenario_id}_events.csv')
    events_df = pd.read_csv(events_path)

    # load requests 
    requests_file = os.path.join(results_path, 'environment','requests.parquet')
    if not os.path.exists(requests_file):
        print(f"Requests file not found at: `{requests_file}`. Skipping.")
        return
    try:
        requests_df = pd.read_parquet(requests_file)
    except Exception as e:
        print(f"Error loading requests from `{requests_file}`: {e}. Skipping.")
        return

    if len(requests_df) > max_task:
        print(f"Warning: More than {max_task} requests found in `{requests_file}` ({len(requests_df)}). Skipping.")
        return

    # load broadcasts
    broadcasts_file = os.path.join(results_path, 'environment','broadcasts.parquet')
    broadcasts_df = pd.read_parquet(broadcasts_file)

    # load measurements
    measurements_file = os.path.join(results_path, 'environment','measurements.parquet')
    measurements_df = pd.read_parquet(measurements_file)

    # --- extract times ---
    t_msg : np.ndarray = pd.to_numeric(broadcasts_df['t_broadcast'], errors="coerce").dropna().to_numpy()
    t_task_start : np.ndarray = pd.to_numeric(events_df['start time [s]'], errors="coerce").dropna().to_numpy()
    t_task_requested : np.ndarray = pd.to_numeric(requests_df['t_req'], errors="coerce").dropna().to_numpy()
    

    t_task_start = np.unique(t_task_start)
    t_task_requested = np.unique(t_task_requested)

    if t_msg.size == 0: raise ValueError("No valid message times found.")

    t_min = min(float(t_msg.min()), float(t_task_start.min()), float(t_task_requested.min())) if t_task_start.size else float(t_msg.min())
    t_max = max(float(t_msg.max()), float(t_task_start.max()), float(t_task_requested.max())) if t_task_start.size else float(t_msg.max())

    # --- histogram messages ---
    edges = np.arange(t_min, t_max + bin_width, bin_width)
    counts, _ = np.histogram(t_msg, bins=edges)

    # --- plot ---
    fig, ax = plt.subplots()
    ax.step(edges[:-1], counts, where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Messages per {bin_width:g} s")
    ax.set_title(title + f"\n(`{trial_name}`, Scenario {scenario_id})")
    ax.set_xlim(t_min, t_max)

    # task availability regions
    for _, row in events_df.iterrows():
        ax.axvspan(
            row["start time [s]"],
            row["duration [s]"] + row["start time [s]"],
            alpha=0.15,          # transparency
            color="red"       # region color
        )

    # task request markers
    for x in t_task_requested:
        ax.axvline(x, color="orange", linestyle="--", linewidth=1.5)

    # task observation markers
    for _, row in measurements_df.iterrows():
        # ax.axvspan(
        #     row["t_start"],
        #     row["t_end"],
        #     alpha=0.15,          # transparency
        #     color="green"       # region color
        # )
        ax.axvline(row["t_start"], color="green", linestyle="-", linewidth=1.5)

    # legend
    handles = [plt.Line2D([0], [0], color="blue", label="Messages"),
               plt.Line2D([0], [0], color="red", label="Task Available"),
               plt.Line2D([0], [0], color="orange", linestyle="--", label="Task Requested"),
               plt.Line2D([0], [0], color="green", linestyle="-", label="Observation Performed")]
    ax.legend(handles=handles)

    # --- save plot  ---
    if save_plot:
        # define plots directory and ensure it exists
        plots_dir = os.path.join('experiments','1_0_cbba_stress_test', 'analysis','plots', 'messages_vs_time')
        os.makedirs(plots_dir, exist_ok=True)

        # define plot path and save plot
        plot_path = os.path.join(plots_dir, f'{trial_name}_scenario_{scenario_id}.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: `{plot_path}`")
    
    # --- show plot ---
    if show_plot: 
        plt.show()

    

if __name__ == "__main__":
    # define trial parameters
    base_dir = "/media/aslan15/easystore/Data/1_0_cbba_stress_test/2026_02_11_Grace"
    # base_dir = os.path.expanduser("/media/aslan15/easystore/Data/1_0_cbba_stress_test/2026_02_11_Grace")

    # trial_name = "lhs_trials-2_samples-1000_seed"
    trial_name = "full_factorial_trials"

    for dir_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir_name)) and dir_name.startswith(trial_name):
            # extract scenario ID from directory name
            scenario_id = dir_name.split('_')[-1]  
            
            # generate plot for this scenario
            generate_plot(trial_name, scenario_id, base_dir=base_dir, show_plot=False, save_plot=True)
            x = 1

    print('DONE')