import os
import pandas as pd
from tqdm import tqdm
import numpy as np

input_file = './initial_requests.csv'
output_file = './initial_requests_merged.csv'

if __name__ == '__main__':
    """ Reads `initial_requests.csv` file and merges measurement requests """
    
    print('\nMERGING INITIAL REQUESTS')
    reqs : pd.DataFrame = pd.read_csv(input_file)

    if os.path.isfile(output_file):
        print('Requests already merged. Loading file...')
        reqs_merged : pd.DataFrame = pd.read_csv(output_file)
        print('File loaded!')

    else:
        print('Requests not yet merged. Starting merging process...')
        columns = ['lat [deg]', 'lon [deg]', 'start time [s]', 'duration [s]', 'severity', 'measurements']
        data = []

        idx_to_ignore = []
        for idx, row in tqdm(   reqs.iterrows(),
                                desc='Merging requests',
                                unit='rows' ):
        
        
            if idx in idx_to_ignore:
                continue

            lat,lon,t_start,duration,severity = row['lat [deg]'], row['lon [deg]'], row['start time [s]'], row['duration [s]'], row['severity']
            
            same_reqs = reqs.query("`lat [deg]` == @lat & `lon [deg]` == @lon & `start time [s]` == @t_start & `duration [s]` == @duration & `severity` == @severity")
            
            if not same_reqs.empty:
                joint_measurements = []
                for measurements in same_reqs['measurements'].values:
                    measurements : str
                    measurements = measurements.replace('[','')
                    measurements = measurements.replace(']','')
                    measurements = measurements.replace('\'','')
                    measurements = measurements.split(',')

                    for measurement in measurements:
                        if measurement not in joint_measurements:
                            joint_measurements.append(measurement)

                data.append([lat,lon,t_start,duration,severity,joint_measurements])

                indices = same_reqs.index.values
                for i in indices:
                    if i not in idx_to_ignore:
                        idx_to_ignore.append(i)
                
        reqs_merged = pd.DataFrame(data=data, columns=columns)
        print('Requests merged!')
        reqs_merged.to_csv(output_file, index=False)

    n_prev, _ = reqs.shape
    n_new, _ = reqs_merged.shape

    print(f"{n_prev} requests merged into {n_new} requests! Reduced by {np.round((n_prev - n_new)/n_prev*100, 3)}%\n")
    print('\nDONE')