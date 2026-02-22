import os
import random
from typing import Tuple
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from shapely import MultiPoint
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

def main(
         n_points : int,
         grid_type : str, 
         rand : bool,
         bounds : Tuple[float,float],
         world : gpd.GeoDataFrame,
         inland : bool = True,
         plot : bool = False,
         seed : int = 1000,
         overwrite : bool = False
         ) -> None:   

    # main(n_points, grid_type, bounds, world, inland=False, plot=True, seed=seed)

    # generate grids
    if grid_type.lower() == 'uniform':
        if rand:
            grid_path = create_random_grid_uniform(n_points, bounds, world, inland, seed, overwrite)
        else:
            grid_path = create_uniform_grid(n_points, bounds, world, inland, overwrite)
        
    elif grid_type.lower() == 'fibonacci':
        if rand: raise ValueError('Random Fibonacci grid not supported.')
        grid_path = create_fibonacci_grid(n_points, bounds, world, inland, overwrite)

    else:
        raise ValueError(f'Cannot generate grid of type `{grid_type}`. Type not supported.')
    
    # plot grids
    if plot: plot_grid(grid_path, grid_type, rand, n_points, inland, overwrite)
        

def create_uniform_grid(n_points : int,
                        bounds : Tuple[float,float],
                        world : gpd.GeoDataFrame, 
                        inland_mask : bool,
                        overwrite : bool
                        ) -> str:
    # set grid filename
    if inland_mask:
        filename = f'uniform_inland_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}.csv'
    else:
        filename = f'uniform_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}.csv'

    # set grid path
    grid_path : str = os.path.join('grids', filename)

    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path
    
    # initialize list for groundpoints
    groundpoints = []

    n_prev = None
    n_curr = n_points

    desc = 'Generating uniform inland coverage grid' if inland_mask else 'Generating uniform grid'
    with tqdm(total=n_points, desc=desc, leave=False) as pbar:
        while len(groundpoints) < n_points:
            n_prev = len(groundpoints)

            # calculate spacing
            k_1 = (1/2) * (1 - np.sqrt( 2*n_curr - 3 ))
            k_2 = (1/2) * (np.sqrt( 2*n_curr - 3 ) + 1)
            k = np.floor(max(k_1,k_2))

            spacing = 180/k # deg / plane

            # generate grid
            groundpoints = [[lat, lon] 
                            for lat in np.linspace(min(bounds), max(bounds), int(180/spacing)+1)
                            for lon in np.linspace(-180, 180, int(360/spacing)+1)
                            if lon < 180
                            ]
            
            # filter inland points        
            groundpoints = [(lat,lon) for lat,lon in tqdm(groundpoints, desc='Filtering inland points', leave=False) 
                            if any(world.contains(Point(lon, lat)))] if inland_mask else groundpoints
                
            # update n_curr to increase number of points
            n_curr = int(n_curr / len(groundpoints) * n_points)

            pbar.update(len(groundpoints) - n_prev)
        
    assert n_points <= len(groundpoints) <= n_points * 1.20 # allow up to 20% more points than requested

    # create dataframe
    df = pd.DataFrame(data=groundpoints[:n_points], columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path
        
def create_random_grid_uniform(n_points : int,
                                bounds : Tuple[float,float],
                                world : gpd.GeoDataFrame, 
                                inland : bool,
                                seed : int,
                                overwrite : bool
                                ) -> str:
    # set grid filename
    if inland:
        filename = f'random_uniform_inland_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}_seed-{seed}.csv'
    else:
        filename = f'random_uniform_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}_seed-{seed}.csv'

    # set grid path
    grid_path : str = os.path.join('grids', filename)

    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path
    
    # set random seed
    random.seed(seed)
    
    # initialize set for unique points
    groundpoints = set()
    
    # collect groundpoints
    desc = 'Generating random uniform inland coverage grid' if inland else 'Generating random uniform grid'
    with tqdm(total=n_points, desc=desc, leave=False) as pbar:
        while len(groundpoints) < n_points:
            # generate random point
            low = [bounds[0], -180.0]
            up = [bounds[1], 180.0]
            n_samples = n_points - len(groundpoints)
            samples = (np.random.uniform(low, up, size=(n_samples,2)))

            new_points = {(round(lat,6), round(lon,6)) for lat,lon in samples}
            new_points = {(lat,lon) for lat,lon in tqdm(new_points, desc='Filtering inland points', leave=False) 
                          if any(world.contains(Point(lon, lat)))} if inland else new_points

            # add rounded points to set to avoid duplicates
            groundpoints.update(new_points)

            # update progress bar
            pbar.update(len(groundpoints) - (n_points - n_samples))

    # create dataframe
    df = pd.DataFrame(data=list(groundpoints), columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def create_fibonacci_grid(n_points : int,
                            bounds : Tuple[float,float],
                            world : gpd.GeoDataFrame, 
                            inland_mask : bool,
                            overwrite : bool
                          ) -> str:
    # set grid filename
    if inland_mask:
        filename = f'fibonacci_inland_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}.csv'
    else:
        filename = f'fibonacci_grid_{n_points}_latbounds-{bounds[0]}to{bounds[1]}.csv'
    
    # set grid path
    grid_path : str = os.path.join('grids', filename)

    # check if grid already exists
    if os.path.isfile(grid_path) and not overwrite: return grid_path
    
    # initialize list for groundpoints
    groundpoints = []

    # set fibonacci ratios
    golden_ratio = (1 + np.sqrt(5)) / 2

    n_curr = n_points
    while len(groundpoints) < n_points:
        groundpoints = set()
        N = int(np.round((n_curr - 1) / 2))
        Ns = [i for i in range(-N,N+1)]

        for i in tqdm(Ns, desc='Generating fibonacci grid', leave=False):
            lat = round(np.arcsin( 2*i / (2*N + 1) ) * 180 / np.pi, 6)
            lon = round((np.mod(i,golden_ratio) * 360 / golden_ratio), 6)

            if lon < -180:
                lon += 360
            if lon > 180:
                lon -= 360

            # filter latitudes outside bounds
            if lat < bounds[0] or lat > bounds[1]:
                continue

            # add point if inland mask not set or point is inland
            if not inland_mask or any(world.contains(Point(lon, lat))):
                groundpoints.add((lat,lon))

        # update n_curr to increase number of points
        n_curr = int(n_curr / len(groundpoints) * n_points)

    # create dataframe
    df = pd.DataFrame(data=list(groundpoints)[:n_points], columns=['lat [deg]','lon [deg]'])

    # save to csv
    df.to_csv(grid_path,index=False)

    # return address
    return grid_path

def plot_grid(grid_path : str, grid_type : str, rand : bool, n_points : int, inland : bool, overwrite : bool) -> None:
    # get plot path
    plot_path = grid_path.replace('.csv', '.png')
    plot_path = plot_path.replace('grids', 'grids/plots')

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # check if plot already exists
    if os.path.isfile(plot_path) and not overwrite: return 

    # load grid data
    df : pd.DataFrame = pd.read_csv(grid_path)
    lons = [lon for _,lon in df.values]
    lats = [lat for lat,_ in df.values]

    # generate plot
    m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
    x, y = m(lons,lats)
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    m.scatter(x,y,3,marker='o',color='k')

    # set title
    title = f"{grid_type} grid of ~{n_points} points" if not inland else f"{grid_type} inland grid of ~{n_points} points"
    if rand: title = "Random " + title
    plt.title(title.capitalize())

    # save plot
    plt.savefig(plot_path)

    # close plot
    plt.close()
    
if __name__ == "__main__":

    # Manually specify the path to Natural Earth shapefile (download if needed)
    WORLD_SHAPEFILE_PATH = "./experiments/1_cbba_validation/resources/grids/ne_110m_land/ne_110m_land.shp"  

    # Load the landmass shapefile
    world : gpd.GeoDataFrame = gpd.read_file(WORLD_SHAPEFILE_PATH)

    # set seed
    seed = 1000

    # set number of points to sample
    points = [1000, 5000, 10000]

    # load trials
    trials_path = os.path.join('trials', f'full_factorial_trials.csv')
    trials : pd.DataFrame = pd.read_csv(trials_path)

    # collect grid types, number of groundpoints and grid distribution
    target_distributions : list = [(-bound, bound) for bound in trials['Target Distribution'].unique()] 
    target_distributions.sort()

    # generate list of all grids to generate
    grids_to_generate = [
        (n_points,grid_type,rand,inland,bounds)
        for n_points in points
        for grid_type in [
                          'uniform', 
                          'fibonacci'
                          ]
        for rand in [True, False]
        for inland in [True, False]
        for bounds in target_distributions
        if not (grid_type=='fibonacci' and rand)
    ]

    # generate grids and plots for all types and number of groundpoints
    for n_points,grid_type,rand,inland,bounds in tqdm(grids_to_generate, desc='Generating coverage grids', unit=' grids'):
        main(n_points, grid_type, rand, bounds, world, inland, plot=True, seed=seed, overwrite=False)

    print("All grids generated!")