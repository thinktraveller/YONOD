# Author: Linus Lind, 2025
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import pickle
import numpy as np
import pandas as pd
from typing import NoReturn
from matplotlib import colors
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shap.plots._beeswarm import summary_legacy
from argparse import ArgumentParser, RawTextHelpFormatter
from os import path

############################ parser boilerplate ###############################
parser = ArgumentParser(description='''SHAP ANALYSIS.''', 
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('-d', '--descriptors', nargs='+', 
                    help='list of descriptor files (CSV format)')
parser.add_argument('-tars', '--targets', nargs='+', 
                    help='list of target names')
parser.add_argument('-f', '--folder', help='folder where to find outputs')
parser.add_argument('-e', '--explain', default=False, 
                    help='Toggle explained MACCS patterns')
parser.add_argument('-dup', '--duplicates', default=False, 
                    help='Remove duplicates')
parser.add_argument('-enum', '--enumerate', default=False, 
                    help='Enumerate features')
args = parser.parse_args()
targets = args.targets
descriptors = args.descriptors
folder = args.folder
explain_maccs = args.explain == 'True'
dupes = args.duplicates == 'True'
enumerate_features = args.enumerate == 'True'
filepath = f'{folder}'
random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
###############################################################################

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Truncate a colormap by resampling it between `minval` and `maxval`."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def main() -> NoReturn:
    for descriptor in descriptors:
        for target in targets:
            try: 
                datasetname = args.folder.replace(r'data/', '')\
                                      .replace(r'/shap', '')
                print(datasetname)
                with open(f'data/{datasetname}/descriptors/{descriptor}_feats.txt', 'r') as feats:
                    feat_names = feats.readlines()
                    tmp = []
                    for name in feat_names:
                        tmp.append(name.replace('\n', ''))
                    feat_names = tmp
            except FileNotFoundError:
                print('Descriptor feature names file not found!'\
                      +' Features will be enumerated')
            if explain_maccs:
                def filter_words(question: str):
                    tokens = question.split()
                    tokens = [token.replace('?', '') for token in tokens]
                    stop_words = ['How', 'many', 'there', 'are', 'instances', 'of']
                    return ' '.join([token for token in tokens if token not in stop_words])
                patts = pd.read_csv('src/visualization/explained_patterns_v5.csv', index_col='Key ')
                maccs = patts['Question'].values[1:]
                for i, m in enumerate(maccs):
                    feat_names[i] = filter_words(maccs[i])
                atmo_patts = pd.read_csv('src/visualization/explained_atmo_patterns_v5.csv')
                #print(atmo_patts.shape)
                atmo = atmo_patts['Question'].values
                #print(atmo.shape)
                for i, m in enumerate(atmo):
                    #print(feat_names[i+166], atmo[i])
                    feat_names[i+166] = filter_words(atmo[i])
                #print(feat_names)
            shaps = []
            X_tests = []
            num_missing_files = 0
            for state in random_state:
                try:
                    filename = f'{filepath}/shap_values_{descriptor}_{target}_{state}.pkl'
                    with open(filename, 'rb') as file:
                        shaps.append(pickle.load(file))
                except FileNotFoundError:
                    print(f'File not found, skipping: {filename}')
                    num_missing_files += 1
                    continue
                if datasetname == 'GeckoQ':
                    try:
                        x_text_sample_filename = f'{filepath}/X_test_sample_{descriptor}_{target}_{state}.pkl'
                        with open(x_text_sample_filename, 'rb') as file:
                            X_tests.append(pickle.load(file))
                    except FileNotFoundError:
                        print(f'Files not found, skipping: {x_text_sample_filename}\n {x_text_sample_filename}')
                else:
                    try:
                        x_text_filename = f'{filepath}/X_test_{descriptor}_{state}.pkl'
                        with open(x_text_filename, 'rb') as file:
                            X_tests.append(pickle.load(file))
                    except FileNotFoundError:
                        print(f'Files not found, skipping: {x_text_filename}\n {x_text_filename}')
            if num_missing_files != 0:
                print(f'Number of missing files: {num_missing_files}')
            concatenated_shaps = np.concatenate(shaps, axis=0)
            concatenated_X_tests = np.concatenate(X_tests, axis=0)
            if dupes:
                feat_names = np.array(feat_names)
                _, unique_indices = np.unique(feat_names, return_index=True)
                feat_names_idx = np.sort(unique_indices)
                feat_names = feat_names[feat_names_idx]
                concatenated_shaps = concatenated_shaps[:, feat_names_idx]
                concatenated_X_tests = concatenated_X_tests[:, feat_names_idx]
            if enumerate_features:
                feat_names = [f'{idx + 1}: {feat}' for idx, feat in enumerate(feat_names.flatten())]
            # Initialize arrays to store the average SHAP values
            num_features = concatenated_shaps.shape[1]
            average_shap_when_one = np.zeros(num_features)
            average_shap_when_zero = np.zeros(num_features)
            mean_absolute_shap = np.zeros(num_features)
            # Compute the average SHAP values when each feature is 1 and when it is 0
            for i in range(num_features):
                feature_shap_values = concatenated_shaps[:, i]
                feature_values = concatenated_X_tests[:, i]
                
                # Mask arrays for the feature values
                mask_ones = feature_values == 1
                mask_zeros = feature_values == 0
                
                # Calculate the average SHAP values
                if np.sum(mask_ones) > 0:
                    average_shap_when_one[i] = np.mean(feature_shap_values[mask_ones])
                if np.sum(mask_zeros) > 0:
                    average_shap_when_zero[i] = np.mean(feature_shap_values[mask_zeros])
                mean_absolute_shap[i] = np.mean(np.abs(feature_shap_values))
            shap_df = pd.DataFrame()
            shap_df['Average SHAP when zero'] = average_shap_when_zero
            shap_df['Average SHAP when one'] = average_shap_when_one
            shap_df['ma_shap'] = mean_absolute_shap
            shap_df.index = feat_names
            shap_df = shap_df.loc[(shap_df != 0).any(axis=1), :]
            shap_df = shap_df.sort_values(by='ma_shap', ascending=False)
            shap_df.to_csv(f'{filepath}/shap_averages_{descriptor}_{target}.csv')
            # Get the original Blues colormap
            n_colors = 256
            colors_array = plt.cm.Blues(np.linspace(0, 1, n_colors))

            # Define the purple we want to blend into: RGB = (0.6, 0.2, 0.8)
            target_purple = np.array([0.6, 0.2, 0.8, 1.0])  # RGBA

            # Compute the indices corresponding to 0.65 and 0.85
            start_idx = int(0.56 * n_colors)
            end_idx = int(0.85 * n_colors)

            # Interpolate between original and purple
            blend_range = end_idx - start_idx
            for i, idx in enumerate(range(start_idx, end_idx)):
                alpha = i / blend_range  # linear ramp from 0 to 1
                colors_array[idx] = (1 - alpha) * colors_array[idx] + alpha * target_purple
            colors_array[:, 1] = np.clip(colors_array[:, 1] + 0.1, 0, 1)
            # Create the new colormap
            smooth_purple_blues = LinearSegmentedColormap.from_list("SmoothBlueToPurple", colors_array)

            # Optional truncation (adjust range as needed)
            trunc_cmap = truncate_colormap(smooth_purple_blues, 0.3, 0.68)
            fs=7    
            plt.rcParams['figure.dpi'] = 200
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
            plt.rcParams.update({'font.size': fs})
            summary_legacy(concatenated_shaps, concatenated_X_tests, feature_names = feat_names, max_display = 10,
                         cmap=trunc_cmap, s=5, fs=fs)

if __name__ == '__main__':
    main()

'''''
Example usage (CLI):

Wang dataset, ATMOMACCS v5, log_p_sat, verbatim explanations of patterns, remove duplicate patterns
---------------------------------------------------------------------------------------------------
python src/visualization/shap_analysis.py -d ATMOMACCS_DECIMAL_v4 -tars log_p_sat -f data/Wang/shap -e 'True' -dup 'True'

Li dataset, ATMOMACCS v5, tg, raw pattern names, no removal of duplicate patterns
---------------------------------------------------------------------------------------------------
python src/visualization/shap_analysis.py -d ATMOMACCS_DECIMAL_v4 -tars tg -f data/Li/shap
'''''