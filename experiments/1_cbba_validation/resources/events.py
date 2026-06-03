import ast
import os
import random
from typing import List
import uuid
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from dmas.utils.tools import print_scenario_banner


def plot_events(events_df: pd.DataFrame, events_path: str, arrival_rate: float, target_distribution: float, scenario_idx: int, overwrite: bool = False) -> None:
    plot_path = os.path.join(os.path.dirname(events_path), 'plots', os.path.basename(events_path).replace('.csv', '.png'))
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    if os.path.isfile(plot_path) and not overwrite:
        return

    lats = events_df['lat [deg]'].tolist()
    lons = events_df['lon [deg]'].tolist()
    start_times_hr = events_df['start time [s]'] / 3600
    durations_min  = events_df['duration [s]'] / 60
    n_meas = events_df['measurements'].apply(lambda x: len(x) if isinstance(x, list) else len(ast.literal_eval(x)))
    unique_counts = sorted(n_meas.unique())

    BG = 'white'
    HIST_COLOR = '#f0c040'
    TIME_COLOR = '#4a90d9'
    TICK_COLOR = 'black'
    cmap = plt.cm.plasma
    meas_colors = {c: cmap(i / max(len(unique_counts) - 1, 1)) for i, c in enumerate(unique_counts)}

    # layout: 3 rows x 6 cols
    #   row 0 (h=1): lon hist (cols 0-4) | empty (col 5)
    #   row 1 (h=4): world map (cols 0-4) | lat hist (col 5)
    #   row 2 (h=2): start time hist (cols 0-2) | duration hist (cols 3-5)
    fig = plt.figure(figsize=(14, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 4, 2], hspace=0.10, wspace=0.08)
    ax_lon  = fig.add_subplot(gs[0, :5], facecolor=BG)
    ax_map  = fig.add_subplot(gs[1, :5])
    ax_lat  = fig.add_subplot(gs[1, 5],  facecolor=BG)
    ax_time = fig.add_subplot(gs[2, :3], facecolor=BG)
    ax_dur  = fig.add_subplot(gs[2, 3:], facecolor=BG, sharey=ax_time)

    # world map
    # m = Basemap(projection='cyl', lon_0=0, resolution='l', ax=ax_map)
    m = Basemap(projection='robin', lon_0=0, resolution='l', ax=ax_map)
    x, y = m(lons, lats)
    m.drawmapboundary(fill_color='#1a2a3a')
    m.fillcontinents(color='#3d3d3d', lake_color='#1a2a3a')
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=7, color='#888888', linewidth=0.4)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=7, color='#888888', linewidth=0.4)
    m.scatter(x, y, 1.5, marker='o', color=HIST_COLOR, alpha=0.5)

    # longitude marginal
    ax_lon.hist(lons, bins=72, color=HIST_COLOR, alpha=0.75, edgecolor='none')
    ax_lon.set_xlim(-180, 180)
    ax_lon.set_ylabel('count', color=TICK_COLOR, fontsize=7)
    ax_lon.xaxis.set_visible(False)
    ax_lon.tick_params(colors=TICK_COLOR, labelsize=7)
    ax_lon.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_lon.set_axisbelow(True)
    ax_lon.grid(True, axis='y', color='#cccccc', linewidth=0.5, alpha=0.7)
    for spine in ax_lon.spines.values():
        spine.set_edgecolor('#cccccc')

    # latitude marginal
    ax_lat.hist(lats, bins=36, color=HIST_COLOR, alpha=0.75, edgecolor='none', orientation='horizontal')
    ax_lat.set_ylim(-90, 90)
    ax_lat.set_xlabel('count', color=TICK_COLOR, fontsize=7)
    ax_lat.xaxis.set_label_position('top')
    ax_lat.yaxis.set_visible(False)
    ax_lat.tick_params(colors=TICK_COLOR, labelsize=7)
    ax_lat.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_lat.set_axisbelow(True)
    ax_lat.grid(True, axis='x', color='#cccccc', linewidth=0.5, alpha=0.7)
    for spine in ax_lat.spines.values():
        spine.set_edgecolor('#cccccc')

    # start time histogram, stacked by measurement count
    stacked_times = [start_times_hr[n_meas == c].values for c in unique_counts]
    ax_time.hist(stacked_times, bins=48, range=(0, 24), stacked=True, alpha=0.85, edgecolor='none',
                 color=[TIME_COLOR] if len(unique_counts) == 1 else [meas_colors[c] for c in unique_counts],
                 label=[f'{c} measurement{"s" if c != 1 else ""}' for c in unique_counts])
    ax_time.set_xlim(0, 24)
    ax_time.set_xticks(np.arange(0, 25, 6))
    ax_time.set_xlabel('start time [hr]', color=TICK_COLOR, fontsize=8)
    ax_time.set_ylabel('count', color=TICK_COLOR, fontsize=8)
    ax_time.tick_params(colors=TICK_COLOR, labelsize=7)
    ax_time.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_time.set_axisbelow(True)
    ax_time.grid(True, axis='y', color='#cccccc', linewidth=0.5, alpha=0.7)
    if len(unique_counts) > 1:
        ax_time.legend(fontsize=7, framealpha=0.5)
    for spine in ax_time.spines.values():
        spine.set_edgecolor('#cccccc')

    # duration histogram, stacked by measurement count (shares y-axis with start time)
    stacked_durs = [durations_min[n_meas == c].values for c in unique_counts]
    ax_dur.hist(stacked_durs, bins=30, range=(5, 15), stacked=True, alpha=0.85, edgecolor='none',
                color=[TIME_COLOR] if len(unique_counts) == 1 else [meas_colors[c] for c in unique_counts],
                label=[f'{c} measurement{"s" if c != 1 else ""}' for c in unique_counts])
    ax_dur.set_xlim(5, 15)
    ax_dur.set_xticks(np.arange(5, 16, 5))
    ax_dur.set_xlabel('duration [min]', color=TICK_COLOR, fontsize=8)
    ax_dur.tick_params(colors=TICK_COLOR, labelsize=7, labelleft=False)
    ax_dur.set_axisbelow(True)
    ax_dur.grid(True, axis='y', color='#cccccc', linewidth=0.5, alpha=0.7)
    if len(unique_counts) > 1:
        ax_dur.legend(fontsize=7, framealpha=0.5)
    for spine in ax_dur.spines.values():
        spine.set_edgecolor('#cccccc')

    title = (f"Events: arrival rate={arrival_rate} tasks/day, "
             f"lat bounds=±{target_distribution}°, "
             f"scenario {scenario_idx} ({len(lats)} events)")
    ax_lon.set_title(title, color=TICK_COLOR, fontsize=10, pad=6)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    # print welcome
    print_scenario_banner('Event generator for Internal Validation Study')

    # set seed
    seed = 1000

    # create random number generator with seed
    rng = np.random.default_rng(seed)

    # load trials
    trials_path = os.path.join('./experiments/1_cbba_validation/resources/trials', f'full_factorial_trials_2026-02-22.csv')
    assert os.path.isfile(trials_path), f'Cannot find trials file at `{trials_path}`'
    trials : pd.DataFrame = pd.read_csv(trials_path)

    # get unique arrival rate and target distribution values
    arrival_rates = trials['Task Arrival Rate'].unique()
    target_distributions = trials['Target Distribution'].unique()
    n_scenarios_per_combination = max(trials['Scenario'].unique()) + 1

    # calculate all combinations of arrival rates and target distributions
    combinations = [(ar, td) for ar in arrival_rates for td in target_distributions]
    print(f"Unique Task Arrival Rates: {arrival_rates}")
    print(f"Unique Target Distributions: {target_distributions}")
    print(f"Total unique combinations of parameters per scenario: {len(combinations)}")
    print(f"Number of scenarios per combination of parameters: {n_scenarios_per_combination}")

    # define and create events output directory
    events_dir = os.path.join('.', 'experiments','1_cbba_validation', 'resources', 'events')
    os.makedirs(events_dir, exist_ok=True)

    # create list of events for each combination of parameters
    for arrival_rate, target_distribution in tqdm(combinations, desc='Generating events for parameter combinations', unit=' combinations'):
        # unpack scenario parameters
        arrival_rate_hz = arrival_rate / 24 / 3600 # [1/s]

        # load matching target grid
        grid_name = f'random_uniform_inland_grid_5000_latbounds--{round(target_distribution,1)}to{round(target_distribution,1)}_seed-{seed}'
        grid_path = os.path.join('./experiments/1_cbba_validation/resources/grids', f'{grid_name}.csv')
        assert os.path.isfile(grid_path), f'Cannot find grid file at `{grid_path}`'
        grid = pd.read_csv(grid_path)

        # sanity check parameters
        if arrival_rate_hz < 0: raise ValueError("`arrival_rate` must be >= 0")

        # generate events for matching scenarios
        for scenario_idx in range(n_scenarios_per_combination):
            T = 3600 * 24   # duration time [s]
            t_start_s = 0.0 # start time [s]
            t_end_s = t_start_s + T     # end time [s]

            # generate event arrival times
            times: List[float] = []

            t = float(t_start_s)
            while True:
                # Exponential inter-arrival time with mean 1/rate_hz
                dt = rng.exponential(scale=1.0 / arrival_rate_hz)
                t += float(dt)
                if t >= t_end_s:
                    break
                times.append(t)

            # randomly select event locations from grid
            n_events = len(times)
            selected_indices = rng.choice(len(grid), size=n_events, replace=True)
            selected_points : pd.DataFrame = grid.iloc[selected_indices]

            # package events into dataframe
            columns = ['gp_index', 'lat [deg]', 'lon [deg]', 'event type', 'start time [s]', 'duration [s]', 'severity', 'measurements', 'id']
            events = []
            for t_start,(gp_idx,gp_row) in tqdm(zip(times, selected_points.iterrows()), desc='Generating events', unit=' events', leave=False):
                # generate event parameters
                duration = rng.uniform(5.0, 15.0) * 60  # [s]
                severity = rng.uniform(5.0, 10.0)       # [norm]
                measurements = ['IMG_A', 'IMG_B', 'IMG_C']
                id = str(uuid.uuid1())

                # save event
                event = [gp_idx, gp_row['lat [deg]'], gp_row['lon [deg]'], 'generic event', t_start, duration, severity, measurements, id]
                events.append(event)

            # compile events into dataframe
            events_df = pd.DataFrame(data=events, columns=columns)

            # sort by start time
            events_df : pd.DataFrame = events_df.sort_values(by='start time [s]').reset_index(drop=True)

            # save events to file
            event_filename = f'events_arrivalrate-{arrival_rate}_targetdist-{target_distribution}_scenario-{scenario_idx}.csv'
            events_path = os.path.join(events_dir, event_filename)
            os.makedirs(os.path.dirname(events_path), exist_ok=True)
            events_df.to_csv(events_path, index=False)
            plot_events(events_df, events_path, arrival_rate, target_distribution, scenario_idx, overwrite=True)
        
    print("All scenario events generated!")