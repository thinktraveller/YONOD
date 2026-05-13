#!/usr/bin/python3

# Author: Linus Lind Jan. 2024
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from argparse import ArgumentParser, RawTextHelpFormatter

def calc_mean_and_std(input_files, mean_output_file, std_output_file):
    random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
    n_states = len(random_state)
    output = pd.DataFrame(np.zeros([6, 7]))
    output.columns = ['Train_sizes', 'Train_MAE', 'Train_R2', 'Train_MSE', 'Test_MAE', 'Test_R2', 'Test_MSE']
    all_data = []

    # Accumulate data from all files
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} does not exist. Skipping this file.")
            continue
        data = pd.read_csv(input_file)
        output = output + data
        all_data.append(data)  # Populate the all_data list

    # Check if we have any valid data
    if not all_data:
        print("No valid data found. Exiting.")
        return None, None

    # Calculate mean
    output_mean = output / len(all_data)
    output_mean.to_csv(mean_output_file, index=None)

    # Calculate standard deviation
    std_data = pd.DataFrame(np.zeros([6, 7]))
    std_data.columns = ['Train_sizes', 'Train_MAE', 'Train_R2', 'Train_MSE', 'Test_MAE', 'Test_R2', 'Test_MSE']
    
    for data in all_data:
        std_data += (data - output_mean) ** 2

    std_data = (std_data / len(all_data)) ** 0.5
    std_data['Train_sizes'] = output_mean['Train_sizes']  # Ensure Train_sizes are the same
    std_data.to_csv(std_output_file, index=None)
    
    return output_mean, std_data

def main(descriptors: list[str], targets: list[str], folder: str):
    random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
    filepath = path.relpath(f'{folder}')
    
    if not os.path.isdir(path.join(filepath, 'results')):
        os.mkdir(path.join(filepath, 'results'))
    if not os.path.isdir(path.join(filepath, 'plots')):
        os.mkdir(path.join(filepath, 'plots'))
    print(filepath)

    for descriptor in descriptors:
        for target in targets:
            if target not in descriptor and 'TopFP' in descriptor:
                continue
            mean_output_file = path.join(filepath, f'mean_{descriptor}_{target}.csv')
            std_output_file = path.join(filepath, f'std_{descriptor}_{target}.csv')
            input_files = []
            for state in random_state:
                output_file = path.join(filepath + '/results/', f'result_{descriptor}_{target}_{state}.csv')
                input_files.append(output_file)

            mean_data, std_data = calc_mean_and_std(input_files, mean_output_file, std_output_file)
            if mean_data is None or std_data is None:
                print(f"No valid data found for {descriptor} and {target}. Skipping these.")
                continue

if __name__ == "__main__":
        
    # Define command-line arguments
    parser = ArgumentParser(description='''Visualize results.''', formatter_class=RawTextHelpFormatter)
    ''''' 
    py src/visualization/plot_processed_results.py -d ATMOMACCS_v1 ATMOMACCS_v2 ATMOMACCS_v3 ATMOMACCS_v4 ATMO_v4 ATMO_DECIMAL_v4 ATMOMACCS_DECIMAL_v4 ATMOMACCS_ALT_v3 MACCS TopFP_log_p_sat TopFP_log_kwiomg TopFP_log_kwg -tars log_p_sat log_kwg log_kwiomg -f data/Wang/KRR_output 
    py src/visualization/plot_processed_results.py -d ATMOMACCS_v1 ATMOMACCS_v2 ATMOMACCS_v3 ATMOMACCS_v4 ATMO_v4 ATMO_DECIMAL_v4 ATMOMACCS_DECIMAL_v4 ATMOMACCS_ALT_v3 MACCS TopFP_log_p_sat -tars log_p_sat -f data/GeckoQ/KRR_output
    py src/visualization/plot_processed_results.py -d ATMOMACCS_v1 ATMOMACCS_v2 ATMOMACCS_v3 ATMOMACCS_v4 ATMO_v4 ATMO_DECIMAL_v4 ATMOMACCS_DECIMAL_v4 ATMOMACCS_ALT_v3 MACCS TopFP_dvap -tars dvap -f data/Ferraz-Caetano/KRR_output
    py src/visualization/plot_processed_results.py -d ATMOMACCS_v1 ATMOMACCS_v2 ATMOMACCS_v3 ATMOMACCS_v4 ATMO_v4 ATMO_DECIMAL_v4 ATMOMACCS_DECIMAL_v4 ATMOMACCS_ALT_v3 MACCS TopFP_tg -tars tg -f data/Li/KRR_output
    '''''

    # Arguments
    parser.add_argument('-d', '--descriptors', nargs='+', help='list of descriptor files (CSV format)')
    parser.add_argument('-tars', '--targets', nargs='+', help='list of target names')
    parser.add_argument('-f', '--folder', help='folder where to find outputs')
    
    args = parser.parse_args()
    targets = args.targets
    descriptors = args.descriptors
    folder = args.folder

    main(descriptors, targets, folder)
