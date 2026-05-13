#!/usr/bin/python3

# Author: Linus Lind Jan. 2024
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from os import path
import argparse

parser = argparse.ArgumentParser(description="Process GECKO-Q data and predict saturation mass concentration.")
parser.add_argument("-d", "--data_folder", help="Path to the folder containing data.")
parser.add_argument("-T", "--temperature", type=float, default=288.15, help="Temperature in Kelvin.")
parser.add_argument("-des","--descriptor", type=str, default='ATMOMACCS_v4', help="descriptor used for predictions.")
parser.add_argument("-t", "--target", type=str, default='log_p_sat', help="name of psat in target folder")

def log_psat_to_c(p_sat: float|int, 
                  molar_mass: float|int, 
                  temp: float|int) -> float: 
    """Apply ideal gas law to convert saturation vapor pressure to saturation mass concentration."""
    R = 8.31446261815324 # 8.3... Pa m^3 / (K mol)
    P = 10**(p_sat) * 1_000 # log10(kPa) -> Pa
    T = temp
    M = molar_mass * 1_000_000
    C = (P * M) / (T * R)
    return C # units micrograms / m^3 

def c_to_volatility_group(c: float|int) -> str:
    """Convert saturation mass concentration to volatility group."""
    volatility_groups = ['ELVOC', 'LVOC', 'SVOC', 'IVOC', 'VOC']
    vol_donahue = np.array([3 * 10**(-4), 0.3, 300, 3 * 10**(6)]) # micro g / m^3
    volatility_ranges = [(vol_donahue[i], vol_donahue[i+1]) \
                         for i in range(len(vol_donahue) - 1)]
    
    if c <= vol_donahue[0]:
        return volatility_groups[0]
    if c >= vol_donahue[3]:
        return volatility_groups[4]
    for i, (_, max_range) in enumerate(volatility_ranges):
        if c <= max_range:
            return volatility_groups[i+1]

def main():
    args = parser.parse_args()
    data_folder = path.relpath(args.data_folder)
    temperature = args.temperature
    descriptor = args.descriptor
    target = args.target
    
    random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
    
    vol_data_path = path.join(data_folder, 'molar_mass.csv')
    if not os.path.exists(vol_data_path):
        print(f"Error: {vol_data_path} does not exist.")
        return

    vol_data = pd.read_csv(vol_data_path, index_col='SMILES')
    print("vol_data:")
    print(vol_data.head())

    vol_data['c'] = vol_data.apply(lambda x: log_psat_to_c(x['log_p_sat'], x['molar_mass'], temperature), axis=1)
    vol_data['volatility'] = vol_data.apply(lambda x: c_to_volatility_group(x['c']), axis=1)
    vol_data.to_csv(path.join(data_folder, 'vol_data.csv'))
    print("vol_data:")
    print(vol_data.head())

    filepath_preds = os.path.join(data_folder, 'KRR_output')
    pred_all = pd.DataFrame()

    for seed in random_state:
        filename_preds = f'output_predictions/output_predictions_{descriptor}_{target}_{seed}.csv'
        pred_path = path.join(filepath_preds, filename_preds)

        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} does not exist. Skipping.")
            continue
        
        pred = pd.read_csv(pred_path, index_col='SMILES')
        print(f"\nRead {pred_path} with shape {pred.shape}")
        print(pred.head())

        pred = pred.join(vol_data, how='inner')
        print(f"After merging, shape of pred: {pred.shape}")
        print(pred.head())

        if pred.empty:
            print(f"Warning: Merging resulted in an empty dataframe for seed {seed}.")
            continue

        pred['c_pred'] = pred.apply(lambda x: log_psat_to_c(x['predictions'], x['molar_mass'], temperature), axis=1)
        pred['volatility_pred'] = pred.apply(lambda x: c_to_volatility_group(x['c_pred']), axis=1)

        # Print the columns of pred to debug
        print(f"Columns before renaming: {pred.columns}")

        # Check if column names match expected names
        expected_columns = ['predictions', 'target_values', 'molar_mass', 'log_p_sat', 'c', 'volatility', 'c_pred', 'volatility_pred']
        if list(pred.columns) != expected_columns:
            print(f"Warning: Column mismatch! Columns are: {pred.columns}")
            continue
        
        reordered = ['molar_mass', 'log_p_sat', 'target_values', 'predictions', 'c', 'c_pred', 'volatility', 'volatility_pred']
        pred = pred.reindex(reordered, axis=1)
        pred_all = pd.concat([pred_all, pred], axis=0)

    if pred_all.empty:
        print("No data to plot. Exiting.")
        return

    print("Final pred_all dataframe:")
    print(pred_all.head())
    print(f"Shape of pred_all: {pred_all.shape}")

    pred_all = pred_all.drop_duplicates()
    pred_all = pred_all.sort_values(by=['c'])
    
    pred_all.to_csv(path.join(data_folder, 'predictions_log_p_sat.csv'))

    wrong_preds = pred_all.query('volatility != volatility_pred')
    volatility_groups = ['ELVOC', 'LVOC', 'SVOC', 'IVOC', 'VOC']

    for group in volatility_groups:
        count_wrong_preds = len(wrong_preds[wrong_preds['volatility'] == group])
        count_total_preds = len(pred_all[pred_all['volatility'] == group])

        def div(x, y):
            return 0 if y == 0 else x / y

        print(f'__________{group}__________\n'
              f'Wrong predictions: {count_wrong_preds}\n'
              f'Total predictions {count_total_preds}\n'
              f'Fraction of right predictions {1 - div(count_wrong_preds, count_total_preds)}\n'
              '__________________________________________\n')

    wrong_preds.to_csv(path.join(data_folder, f'wrong_predictions_{target}.csv'))
    pred_all = pred_all.groupby('SMILES').last()
    vol_donahue = np.log10(np.array([3 * 10**(-4), 0.3, 300, 3 * 10**(6)]))

    plt_pred = np.log10(pred_all['c_pred'].values)
    plt_real = np.log10(pred_all['c'].values)
    plt_wrong_pred = np.log10(wrong_preds['c_pred'].values)
    plt_wrong_real = np.log10(wrong_preds['c'].values)
    fs = 8
    plt.rcParams.update({'font.size': fs})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    #-------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    plt.gcf().set_dpi(1200)
    colors = ['purple', 'magenta', 'crimson', 'brown']
    ax.plot(vol_donahue[0] * np.ones(100), np.linspace(-15, 15, num=100), 'k-', lw=1)
    ax.plot(np.linspace(-15, 15, num=100), vol_donahue[0] * np.ones(100), 'k-', lw=1)
    ax.fill_between(np.linspace(plt_real.min(), vol_donahue[0], num=100),
                    np.ones(100) * 2 * plt_real.min(), np.ones(100) * 2 * plt_real.max(), alpha=0.5, color='purple')
    for i, vol in enumerate(vol_donahue[1:]):
        ax.plot(vol * np.ones(100), np.linspace(-15, 15, num=100), 'k-', lw=1)
        ax.plot(np.linspace(-15, 15, num=100), vol * np.ones(100), 'k-', lw=1)
        ax.fill_between(np.linspace(vol_donahue[i], vol_donahue[i + 1], num=100),
                        np.ones(100) * 2 * plt_real.min(), np.ones(100) * 2 * plt_real.max(), alpha=0.4, color=colors[i])
    ax.fill_between(np.linspace(vol_donahue[3], 2 * plt_real.max(), num=100),
                    np.ones(100) * 2 * plt_real.min(), np.ones(100) * 2 * plt_real.max(), alpha=0.4, color=colors[3])
    ax.scatter(plt_real, plt_pred, s=0.5, color='navy', label='Correctly predicted label')
    ax.scatter(plt_wrong_real, plt_wrong_pred, s=0.5, color='darkred', label='Incorrectly predicted label')
    ax.plot(np.linspace(-15, 15), np.linspace(-15, 15), 'k--', lw=1)
    plt.ylim([plt_real.min() + 0.1, plt_real.max() + 3])
    plt.xlim([plt_real.min(), plt_real.max() + 0.1])
    ax.set_xlabel('Reference C [log$_{10}$(µg m$^{-3}$)]', fontsize=fs)
    ax.set_ylabel('Predicted C [log$_{10}$(µg m$^{-3}$)]', fontsize=fs)
    plt.subplots_adjust(top=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.7, 1.27), 
              borderpad = 0.2, labelspacing = 0.2, fontsize=fs)
    #ax.set_title(f"{descriptor} {target} volatility groups", fontsize=fs)
    
    plot_dir = path.join(data_folder, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.tight_layout()
    fig.savefig(path.join(plot_dir, f'{descriptor}_scatter_plot_donahue_classes.png'),
                dpi=1200)

if __name__ == "__main__":
    main()
    # py src/visualization/plot_volatility_bins.py -d 'data/GeckoQ' -T 298.15
    # py src/visualization/plot_volatility_bins.py -d 'data/Wang' -des 'ATMOMACCS_DECIMAL_v4'
