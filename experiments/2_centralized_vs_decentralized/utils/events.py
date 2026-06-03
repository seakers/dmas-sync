import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap

RESOURCES = os.path.join(os.path.dirname(__file__), '..', 'resources')
GRIDS_DIR = os.path.join(RESOURCES, 'grids')
UNPROCESSED_DIR = os.path.join(RESOURCES, 'events', 'unprocessed')
PROCESSED_DIR = os.path.join(RESOURCES, 'events', 'processed')

EVENT_TYPE_TO_GRID_INDEX = {
    'algal_bloom': 0,
    'high_flow_river': 1,
    'wildfire': 2,
}

STATIC_GRIDS = {
    0: pd.read_csv(os.path.join(GRIDS_DIR, 'algal_bloom_grid.csv')),
    # 1: pd.read_csv(os.path.join(GRIDS_DIR, 'river_grid.csv')),
}


def load_river_grid(date_str: str) -> pd.DataFrame:
    path = os.path.join(GRIDS_DIR, f'high_flow_river_grid_{date_str}.csv')
    return pd.read_csv(path)


def load_wildfire_grid(date_str: str) -> pd.DataFrame:
    path = os.path.join(GRIDS_DIR, f'wildfire_grid_{date_str}.csv')
    return pd.read_csv(path)


def find_gp(lat: float, lon: float, grid: pd.DataFrame):
    lats = grid['lat [deg]'].to_numpy()
    lons = grid['lon [deg]'].to_numpy()
    dists = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)
    idx = int(np.argmin(dists))
    return idx, lats[idx], lons[idx]


def process_file(filepath: str, date_str: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    river_grid = load_river_grid(date_str)
    wildfire_grid = load_wildfire_grid(date_str)
    grids = {**STATIC_GRIDS, 1: river_grid, 2: wildfire_grid}

    grid_indices = []
    gp_indices = []
    matched_lats = []
    matched_lons = []

    for _, row in df.iterrows():
        event_type = row['event type']
        grid_index = EVENT_TYPE_TO_GRID_INDEX[event_type]
        gp_index, grid_lat, grid_lon = find_gp(row['lat [deg]'], row['lon [deg]'], grids[grid_index])
        grid_indices.append(grid_index)
        gp_indices.append(gp_index)
        matched_lats.append(grid_lat)
        matched_lons.append(grid_lon)

    df['lat [deg]'] = matched_lats
    df['lon [deg]'] = matched_lons
    df.rename(columns={'uuid': 'id'}, inplace=True)
    df.insert(0, 'grid index', grid_indices)
    df.insert(1, 'gp_index', gp_indices)
    return df


PLOTS_DIR = os.path.join(RESOURCES, 'events', 'plots')

EVENT_COLORS = {
    'algal_bloom':     '#009E73',  # teal
    'high_flow_river': '#56B4E9',  # sky blue
    'wildfire':        '#D55E00',  # vermillion
}


def plot_events(events_df: pd.DataFrame, events_path: str, overwrite: bool = False) -> None:
    plot_path = os.path.join(PLOTS_DIR, os.path.basename(events_path).replace('.csv', '.png'))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if os.path.isfile(plot_path) and not overwrite:
        return

    event_types = [et for et in EVENT_COLORS if et in events_df['event type'].unique()]
    lats = events_df['lat [deg]'].tolist()
    lons = events_df['lon [deg]'].tolist()
    start_times_hr = events_df['start time [s]'] / 3600
    durations_min  = events_df['duration [s]'] / 60

    BG = 'white'
    TICK_COLOR = 'black'

    fig = plt.figure(figsize=(14, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 4, 2], hspace=0.10, wspace=0.08)
    ax_lon  = fig.add_subplot(gs[0, :5], facecolor=BG)
    ax_map  = fig.add_subplot(gs[1, :5])
    ax_lat  = fig.add_subplot(gs[1, 5],  facecolor=BG)
    ax_time = fig.add_subplot(gs[2, :3], facecolor=BG)
    ax_dur  = fig.add_subplot(gs[2, 3:], facecolor=BG, sharey=ax_time)

    # world map — scatter coloured by event type
    m = Basemap(projection='robin', lon_0=0, resolution='l', ax=ax_map)
    m.drawmapboundary(fill_color='#1a2a3a')
    m.fillcontinents(color='#3d3d3d', lake_color='#1a2a3a')
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=7, color='#888888', linewidth=0.4)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=7, color='#888888', linewidth=0.4)
    for et in event_types:
        mask = events_df['event type'] == et
        ex, ey = m(events_df.loc[mask, 'lon [deg]'].tolist(),
                   events_df.loc[mask, 'lat [deg]'].tolist())
        m.scatter(ex, ey, 4, marker='o', color=EVENT_COLORS[et], alpha=0.6, zorder=3,
                  label=et.replace('_', ' ').title())
    ax_map.legend(loc='lower left', fontsize=7, framealpha=0.7)

    def _style(ax, count_axis):
        ax.tick_params(colors=TICK_COLOR, labelsize=7)
        (ax.yaxis if count_axis == 'y' else ax.xaxis).set_major_locator(MaxNLocator(integer=True))
        ax.set_axisbelow(True)
        ax.grid(True, axis=count_axis, color='#cccccc', linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')

    # longitude marginal
    ax_lon.hist([events_df.loc[events_df['event type'] == et, 'lon [deg]'].tolist() for et in event_types],
                bins=72, range=(-180, 180), stacked=True, alpha=0.85, edgecolor='none',
                color=[EVENT_COLORS[et] for et in event_types])
    ax_lon.set_xlim(-180, 180)
    ax_lon.set_ylabel('count', color=TICK_COLOR, fontsize=7)
    ax_lon.xaxis.set_visible(False)
    _style(ax_lon, 'y')

    # latitude marginal
    ax_lat.hist([events_df.loc[events_df['event type'] == et, 'lat [deg]'].tolist() for et in event_types],
                bins=36, range=(-90, 90), stacked=True, alpha=0.85, edgecolor='none',
                orientation='horizontal', color=[EVENT_COLORS[et] for et in event_types])
    ax_lat.set_ylim(-90, 90)
    ax_lat.set_xlabel('count', color=TICK_COLOR, fontsize=7)
    ax_lat.xaxis.set_label_position('top')
    ax_lat.yaxis.set_visible(False)
    _style(ax_lat, 'x')

    # start time histogram
    t_min, t_max = start_times_hr.min(), start_times_hr.max()
    ax_time.hist([start_times_hr[events_df['event type'] == et].values for et in event_types],
                 bins=48, range=(t_min, t_max), stacked=True, alpha=0.85, edgecolor='none',
                 color=[EVENT_COLORS[et] for et in event_types],
                 label=[et.replace('_', ' ').title() for et in event_types])
    ax_time.set_xlim(t_min, t_max)
    ax_time.set_xlabel('start time [hr]', color=TICK_COLOR, fontsize=8)
    ax_time.set_ylabel('count', color=TICK_COLOR, fontsize=8)
    ax_time.legend(fontsize=7, framealpha=0.5)
    _style(ax_time, 'y')

    # duration histogram (shares y-axis with start time)
    d_min, d_max = durations_min.min(), durations_min.max()
    ax_dur.hist([durations_min[events_df['event type'] == et].values for et in event_types],
                bins=30, range=(d_min, d_max), stacked=True, alpha=0.85, edgecolor='none',
                color=[EVENT_COLORS[et] for et in event_types])
    ax_dur.set_xlim(d_min, d_max)
    ax_dur.set_xlabel('duration [min]', color=TICK_COLOR, fontsize=8)
    ax_dur.tick_params(colors=TICK_COLOR, labelsize=7, labelleft=False)
    ax_dur.set_axisbelow(True)
    ax_dur.grid(True, axis='y', color='#cccccc', linewidth=0.5, alpha=0.7)
    for spine in ax_dur.spines.values():
        spine.set_edgecolor('#cccccc')

    case_name = os.path.basename(events_path).replace('.csv', '')
    ax_lon.set_title(f'{case_name}  ({len(lats)} events)', color=TICK_COLOR, fontsize=10, pad=6)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for filename in sorted(os.listdir(UNPROCESSED_DIR)):
        if not filename.endswith('.csv'):
            continue

        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if not match:
            print(f'Skipping {filename}: no date found in filename')
            continue

        date_str = match.group(1)
        in_path = os.path.join(UNPROCESSED_DIR, filename)
        out_path = os.path.join(PROCESSED_DIR, filename)

        print(f'Processing {filename}...')
        df = process_file(in_path, date_str)
        df.to_csv(out_path, index=False)
        print(f'  -> {out_path} ({len(df)} events)')
        plot_events(df, out_path)

        if filename.startswith('comprehensive_case_'):
            for event_type, group in df.groupby('event type'):
                type_path = os.path.join(PROCESSED_DIR, f'{event_type}_{date_str}.csv')
                group.to_csv(type_path, index=False)
                print(f'  -> {type_path} ({len(group)} events)')
                plot_events(group, type_path)


if __name__ == '__main__':
    main()