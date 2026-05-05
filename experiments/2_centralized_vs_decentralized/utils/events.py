import os
import re
import numpy as np
import pandas as pd

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
    0: pd.read_csv(os.path.join(GRIDS_DIR, 'lake_grid.csv')),
    1: pd.read_csv(os.path.join(GRIDS_DIR, 'river_grid.csv')),
}


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
    wildfire_grid = load_wildfire_grid(date_str)
    grids = {**STATIC_GRIDS, 2: wildfire_grid}

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
    df.insert(0, 'grid index', grid_indices)
    df.insert(1, 'GP index', gp_indices)
    return df


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


if __name__ == '__main__':
    main()