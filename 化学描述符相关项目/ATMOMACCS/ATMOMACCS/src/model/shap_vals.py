# Author: Linus Lind 2025
"""
SHAP Analysis for Trained Machine Learning Models.

This script calculates SHAP (SHapley Additive exPlanations) values for a
pre-trained model using test data, optionally sampling a subset of the
test set and/or clustering with k-means to reduce computational cost.
"""

import pickle
from sys import exit
from sys import argv
from shap import KernelExplainer
from shap.utils import sample
from shap import kmeans
from sklearn.model_selection import GridSearchCV
from typing import NoReturn
from os import path

def calculate_shap(descriptor: str, 
                   target: str, 
                   folder: str, 
                   seed: int,*,
                   samples: int | None = None,
                   k_means: int = 0) -> NoReturn:
    '''''
    arguments
    - descriptor: Descriptor name ['ATMOMACCS_v1', 'ATMOMACCS_v2',
                                 'ATMOMACCS_v3', 'ATMOMACCS_v4']
    - target: Target variable ['log_p_sat', 'log_kwg', 'log_kwiomg']      
    - folder: folder path to shap values ['data/Wang', 'data/GeckoQ']
    - seed: Random state random_state = [12, 432, 5, 7543, 12343,
                                       452, 325432435, 326, 436, 2435]
                                                    
    returns: NoReturn
    '''''
    print('#####################################################')
    print(f'shap_vals.py received parameters:\n' + f'descriptor: {descriptor}\n'\
          + f'target: {target}\n' + f'folder: {folder}\n' + f'seed: {seed}\n'\
          +  f'samples: {samples}')
    filepath = path.relpath(f'{folder}')
    if 'GeckoQ' in folder:
        print('The data set is GeckoQ')
        train_size = '27801'
    elif 'Wang' in folder:
        print('The data set is Wang')
        train_size = '3000'
    elif 'Ferraz-Caetano' in folder:
        print('The data set is Ferraz-Caetano')
        train_size = '2118'
    elif 'Li' in folder:
        print('The data set is Li')
        train_size = '1947'
    else:
        print('The data set is Other')
        train_size = input('Provide train_size of pickled model (Check file path): ')

    modelFile = f'trained_model_size_{train_size}_{descriptor}_{target}_{seed}.pkl'
    try:
        filename = f'{filepath}/pickle/{modelFile}'
        with open(filename, 'rb') as file:
            model: GridSearchCV = pickle.load(file)
    except FileNotFoundError:
        print(f'File not found! {filename} \n exiting')
        exit()
    X_test_filename = f'{filepath}/shap/X_test_{descriptor}_{seed}.pkl'
    try: 
        with open(X_test_filename, 'rb') as file:
            X_test = pickle.load(file)
    except FileNotFoundError:
        print(f'File not found! {X_test_filename} \n exiting')
        exit()
    if samples is not None:
        X_test = sample(X_test, int(samples), random_state=int(seed))
        with open(f'{filepath}/shap/X_test_sample_{descriptor}_{target}_{seed}.pkl', 'wb') as outfile:
            pickle.dump(X_test, outfile)
    print(X_test.shape)
    if k_means > 0:
        X_test_clustered = kmeans(X_test, k = k_means)
    feat_names_filename = f'{filepath}/descriptors/{descriptor}_feats.txt'
    try: 
        with open(feat_names_filename, 'r') as feats_file:
            feat_names = feats_file.readlines()
    except FileNotFoundError:
        print(f'File not found! {feat_names_filename} \n exiting')
        exit()
    print('START SHAP ANALYSIS')
    if k_means > 0:
        ex = KernelExplainer(model.predict, X_test_clustered, feature_names=feat_names)
    else:
        ex = KernelExplainer(model.predict, X_test, feature_names=feat_names)
    shap_vals = ex.shap_values(X_test, gc_collect=True)
    shap_value_filename = f'{filepath}/shap/shap_values_{descriptor}_{target}_{seed}.pkl'
    with open(shap_value_filename, 'wb') as outfile:
        print('Pickling shap values')
        pickle.dump(shap_vals, outfile) # pickle the shap values
    print('############\n# Finished #\n############')

if __name__ == '__main__':
    from argparse import ArgumentParser, RawTextHelpFormatter
    ########################## parser boilerplate #############################
    parser = ArgumentParser(description='''SHAP ANALYSIS.''', 
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--descriptor', required=True, help='descriptor name')
    parser.add_argument('-t', '--target', required=True, 
                        help='target name')
    parser.add_argument('-f', '--folder', required=True, 
                        help='folder')
    parser.add_argument('-s', '--seed', required=True, 
                        help='random seed')
    parser.add_argument('-sa', '--samples', default='None', required=False, 
                        help='Sampling subset of test set')
    parser.add_argument('-k', '--kmeans', default=0, required=False, 
                        help='Applying kmeans to samples')
    args = parser.parse_args()

    if args.samples == 'None' and int(args.kmeans) == 0:
        calculate_shap(args.descriptor, 
                    args.target,
                    args.folder,
                    args.seed)
    else:
        calculate_shap(args.descriptor, 
            args.target,
            args.folder,
            args.seed, 
            samples=args.samples,
            k_means=int(args.kmeans))
'''''
Example Usage (CLI):

Full test set, no k-means
---------------------------------------------------------------------------------------------------
python src/model/shap_vals.py -d ATMOMACCS_v4 -t log_p_sat -f data/GeckoQ -s 12

Subset of test set with k-means clustering
---------------------------------------------------------------------------------------------------
python src/model/shap_vals.py -d ATMOMACCS_v4 -t log_p_sat -f data/Wang -s 12 -sa 5 -k 10
'''''