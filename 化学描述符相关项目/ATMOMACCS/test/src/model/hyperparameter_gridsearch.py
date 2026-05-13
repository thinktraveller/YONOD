"""
Grid Search Hyperparameter Optimization for Kernel Ridge Regression (KRR) using Topological Fingerprints (TopFP).

parameter_grid = {
    'fpsize': [4096, 8192],
    'minPath': [1, 2, 3, 4],
    'maxPath': [7, 8, 9, 10],
    'nBitsPerHash': [6, 8, 10]
}

"""
import sys
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from argparse import ArgumentParser
import gc
from generate_topological import smiles_to_mol, generate_topfp
from krr import train_model


VERBOSE = True
def printv(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

parser = ArgumentParser(description='parser for hyperparameter search')
parser.add_argument('-t', '--target', type=str, required=True, help='target value')
parser.add_argument('-ds', '--dataset', type=str, required=True, help='dataset name')
parser.add_argument('-s', '--seed', type=str, required=True, help='seed')
parser.add_argument('-sub', '--subset', type=str, help='Subset of data')
args = parser.parse_args()
dataset = args.dataset
target = args.target
seed = int(args.seed)
subset = float(args.subset)


kernel_used = 'rbf'
cv = 5
alpha = np.logspace(-4, -1, 4)
gamma = np.logspace(-4, -1, 4)
descriptor = 'TopFP'
#random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]

def load_data(smiles_path: str, target_path: str, * , subset=None):
    smiles = np.genfromtxt(smiles_path, comments=None, dtype=str)
    y_data = np.genfromtxt(target_path, comments='#', dtype=np.float32)
    if subset is not None:
        num_samples = int(len(smiles) * subset)
        np.random.seed(seed)
        rand_idx = np.random.choice(len(smiles), size=num_samples, replace=False)
        smiles = smiles[rand_idx] # sample random % of data
        y_data = y_data[rand_idx]
    return smiles, y_data

def custom_train_test_split(X_data, y_data, test_size, seed):
    train_size = len(X_data) - test_size
    X_train, X_test, \
        y_train, y_test = train_test_split(X_data, y_data, train_size=train_size, 
                                        test_size=test_size, shuffle=True, random_state=seed)
    return X_train, X_test, y_train, y_test

parameter_grid = {
    'fpsize': [4096, 8192],
    'minPath': [1, 2, 3, 4],
    'maxPath': [7, 8, 9, 10],
    'nBitsPerHash': [6, 8, 10]
}

printv('Starting hyperparameter optimization: ')
printv(f'kernel_used = {kernel_used}, cv = {cv}, alpha = {alpha}, gamma = {gamma}')
printv(f'Dataset = {dataset}')
printv(f'Target = {target}')
printv(f'seed: {seed}')
results_list = []
printv('-'*20)
data_path = f'data/{dataset}/'
smiles, y_data = load_data(f'{data_path}smiles.txt',f'{data_path}{target}.txt', subset=subset)
printv(len(smiles), len(y_data))
mols = smiles_to_mol('', smiles=smiles)
test_size = int(0.12*len(y_data))
param_iter = ParameterGrid(parameter_grid)
total = len(param_iter)
counter = 0
for idx, params in enumerate(param_iter):
    counter += 1
    printv(f'Gridsearch iter: [{counter}]/[{total}]')
    # Generate fingerprints with the current parameter combination
    X_data = generate_topfp(mols, 
                            fpsize=params['fpsize'], 
                            minPath=params['minPath'], 
                            maxPath=params['maxPath'], 
                            nBitsPerHash=params['nBitsPerHash'])
    X_data = np.array(X_data)
    printv(f'Generated fp')
    X_train, X_test, y_train, y_test = custom_train_test_split(X_data, y_data,
                                                            test_size, seed)
    del X_data
    result = train_model(X_train, y_train, X_test, y_test, kernel_used, alpha, gamma, cv,
                data_path, descriptor, target, seed, calculate_shap=False, pickle_trained_model=False)
    printv(f'Optimized model hyperparams')
    MAE_train, MSE_train, r2_train, \
        MAE, r2, MSE, alpha_opt, \
            gamma_opt, best_score, y_pred = result
    del X_train, X_test
    results_list.append({
        'dataset': dataset,
        'target': target,
        'seed': seed,
        'fpsize': params['fpsize'],
        'minPath': params['minPath'],
        'maxPath': params['maxPath'],
        'nBitsPerHash': params['nBitsPerHash'],
        'MAE_train': MAE_train,
        'MSE_train': MSE_train,
        'r2_train': r2_train,
        'MAE': MAE,
        'MSE': MSE,
        'r2': r2,
        'alpha_opt': alpha_opt,
        'gamma_opt': gamma_opt,
        'best_score': best_score
        })

error_terms = pd.DataFrame(results_list)
error_terms.to_csv(f'data/{dataset}/{target}_{seed}_topfp_hyperparameters.csv')