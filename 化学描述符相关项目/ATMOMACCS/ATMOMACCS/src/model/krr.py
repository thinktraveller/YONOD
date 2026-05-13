# Author: Linus Lind & Hilda Sandström 2025.
# Parts of the script are adapted from  
# from Emma Lumiaro as part of Lumiaro et al. (2021) https://doi.org/10.5194/acp-21-13227-2021
# Changes including, but not limited to:
# filepath organization, code refactoring, adding option to save predictions, plotting,
# CLI argument parsing, KRR hyperparameter search grid, added multiple dataset support
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
# Python 3.12.5 

import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import gc

VERBOSE = True

def main(descriptor, target, data_path, dataset, *, save_predictions=False, 
         random_seeds=None, plotting=False):
    if VERBOSE:
        print(f'Target: {target}')
        print(f'Passed parameters to krr: descriptor={descriptor}, ',
                f'target={target}, ', f'data_path={data_path}, ', 
                f'seeds={random_seeds}, ', f'dataset={dataset}, ', 
                f'save_predictions={save_predictions}, ', 
                f'plotting={plotting}')
    if random_seeds is None:
        random_seeds = 1
    krr_regr(descriptor, target, data_path, random_seeds, 
             dataset, save_predictions, plotting) 

def train_model(X_train, y_train, 
                X_test, y_test, 
                kernel_used, alpha, gamma, cv,
                data_path, descriptor, target, seed, *, 
                pickle_trained_model):
    tuned_parameters = [{'kernel': [kernel_used], 'alpha': alpha, 'gamma': gamma}]
    grid_search = GridSearchCV(KernelRidge(), 
                               tuned_parameters, 
                               cv=cv, 
                               scoring='neg_mean_absolute_error', 
                               n_jobs=1, 
                               verbose=VERBOSE)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    y_pred_train = grid_search.predict(X_train)
    MSE_train = np.mean((y_pred_train - y_train) ** 2)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    y_pred = grid_search.predict(X_test)
    MSE = np.mean((y_pred - y_test) ** 2)  
    MAE = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    gamma_opt = best_params.get("gamma")
    alpha_opt = best_params.get("alpha")
    best_score = grid_search.best_score_
    if pickle_trained_model:
        f_name = f'trained_model_size_{len(X_train)}_{descriptor}_{target}_{seed}.pkl'
        os.makedirs(f'{data_path}/pickle/', exist_ok=True)
        with open(f'{data_path}/pickle/{f_name}', 'wb') as outfile:
            pickle.dump(grid_search, outfile) # pickle the trained model
    del grid_search
    gc.collect()

    return MAE_train, MSE_train, r2_train, MAE, r2, MSE, alpha_opt, gamma_opt, best_score, y_pred

def select_train_test_sizes(datasetname: str) -> tuple:
    
    if datasetname == 'Wang':
        print('Dataset is Wang')
        train_sizes = [500, 1000, 1500, 2000, 2500, 3000]
        test_size = 414
    elif datasetname == 'GeckoQ':
        print('Dataset is GeckoQ')
        train_sizes = [4633, 9267, 13900, 18534, 23167, 27801]
        test_size = 3836
    elif datasetname == 'Ferraz-Caetano':
        print('Dataset is Ferraz-Caetano')
        train_sizes = [353,  706, 1059, 1412, 1765, 2118]
        test_size = 292
    elif datasetname == 'Li':
        print('Dataset is Li')
        train_sizes = [325,  649,  974, 1298, 1623, 1947]
        test_size = 269
    else:
        raise Exception(f'Invalid dataset name: {datasetname}')
    if VERBOSE:
        print(f'Train sizes: {train_sizes}')
        print(f'Test size: {test_size}')
    return train_sizes, test_size

def aavre(pred: np.array, target: np.array):
    assert pred.shape == target.shape
    enum = pred.copy() - target.copy() # need to do copies, otherwise crazy bugs happen
    denom = target.copy()
    denom[denom == 0] = np.inf
    return np.mean(np.abs(enum / denom)) * 100

def krr_regr(descriptor, target, 
             data_path, seed, dataset, 
             save_predictions, plotting):
    name_of_file = f'{descriptor}.txt'
    filepath = os.path.relpath(data_path)

    descriptor_filename = os.path.join(filepath + '/descriptors', name_of_file)
    target_property_filename = os.path.join(filepath, target)

    train_sizes, test_size = select_train_test_sizes(dataset)
    kernel_used = 'rbf'
    cv = 5
    if descriptor == 'MBTR' or descriptor == 'EnsembleFC':
        desc_dtype = np.float32
    else:
        desc_dtype = np.uint8

    targ_dtype = np.float32

    alpha = np.logspace(-4, -1, 4)
    gamma = np.logspace(-4, -1, 4)

    X_data = np.genfromtxt(descriptor_filename, 
                           dtype=desc_dtype)  # MBTR: float, OTHER: uint8
    y_data = np.genfromtxt(target_property_filename, comments='#', 
                           dtype=targ_dtype)  # Use float32
    if dataset == 'GeckoQuirky':
        from numpy import random as r
        r.shuffle(y_data)
        print('TARGET VARIABLE y_data RANDOMIZED')
    X, X_test, \
        y, y_test = train_test_split(X_data, y_data, train_size=len(X_data) - test_size, 
                                     test_size=test_size, shuffle=True, random_state=seed)
    y_test.setflags(write=False)
    
    if VERBOSE:
        print("Descriptor filename:", descriptor_filename)
        print("Target filename:", target_property_filename)
    labs = pd.read_csv(os.path.join(filepath, 'smiles.txt'), header=None)
    labs.columns = ['SMILES']
    _, X_test_lab\
        , _, _ = train_test_split(labs['SMILES'].values, y_data, 
                                  train_size=len(X_data) - test_size, 
                                  test_size=test_size, shuffle=True, 
                                  random_state=seed)

    learning_curve_mae = []
    training_sets = []

    for train_size in reversed(train_sizes):
        if train_size < len(X):
            X_train, _, \
                y_train, _ = train_test_split(X, y, train_size=train_size, 
                                              test_size=1, shuffle=True, 
                                              random_state=seed)
            X = X_train.astype(desc_dtype)
            y = y_train
        else:
            X_train = X.astype(desc_dtype)
            y_train = y
        training_sets.append([X_train, y_train])
    training_sets.reverse()
    figpath = os.path.join(filepath, 'plots')
    krrpath = os.path.join(filepath, 'KRR_output')
    os.makedirs(figpath, exist_ok=True)
    os.makedirs(krrpath, exist_ok=True)
    os.makedirs(os.path.join(krrpath + '/results/'), exist_ok=True)

    output_file = os.path.join(krrpath + '/results/', f'result_{descriptor}_{target.replace(".txt", "")}_{seed}.csv')


    result_list = []
    X_test_shap = X_test

    os.makedirs(f'{data_path}/shap/', exist_ok=True)
    with open(f'{data_path}/shap/X_test_{descriptor}_{seed}.pkl', 'wb') as outfile:
        pickle.dump(X_test_shap, outfile)

    for train_id in range(len(train_sizes)):
        train_size = train_sizes[train_id]
        X_train, y_train = training_sets[train_id]
        if VERBOSE:
            print(f'\nStarting training with train size: {train_size}')
        if train_size == train_sizes[-1]:
            result = train_model(X_train, y_train,
                                 X_test, y_test, 
                                 kernel_used, alpha, gamma, cv, 
                                 data_path, descriptor, target.replace('.txt', ''), 
                                 seed, pickle_trained_model=True)
        else:
            result = train_model(X_train, y_train,
                                 X_test, y_test, 
                                 kernel_used, alpha, gamma, cv, 
                                 data_path, descriptor, target.replace('.txt', ''), 
                                 seed, pickle_trained_model=False)
        MAE_train, MSE_train, r2_train, MAE, r2, \
            MSE, alpha_opt, gamma_opt, best_score, y_pred = result
        learning_curve_mae.append(MAE)
        aavres = aavre(y_pred, y_test)
        result_list.append([train_size, MAE_train, MSE_train, r2_train, 
                            MAE, r2, MSE, alpha_opt, gamma_opt, best_score,
                            aavres])

        # Clear large objects and force garbage collection
        del X_train, y_train, result
        gc.collect()
        if VERBOSE:
            print('--------------------------------')
        
    if save_predictions:
        if VERBOSE:
            print(f'Saving explicit results')
        f_name = f'output_predictions/output_predictions_{descriptor}_{target.replace(".txt", "")}_{seed}.csv'
        os.makedirs(f'{krrpath}/output_predictions', exist_ok=True)
        filename_preds = os.path.join(krrpath, f_name)
        df = pd.DataFrame({'SMILES': X_test_lab, 
                           'predictions': y_pred, 
                           'target_values': y_test})
                           
        df.to_csv(filename_preds, index=False)

    columns = ['Train_sizes', 'Train_MAE', 'Train_R2', 'Train_MSE', 'Test_MAE', 
               'Test_R2', 'Test_MSE', 'opt_alpha', 'opt_gamma', 'best_score',
               'aavre%']
    outdata = pd.DataFrame(result_list, columns=columns)
    outdata.sort_values(by="Train_sizes").to_csv(output_file)

    if plotting:
        if VERBOSE:
            print(f'Saving plots')
        fig, ax = plt.subplots()
        ax.plot(train_sizes, learning_curve_mae, marker='o')
        plt.title('Learning Curve', fontsize=18)
        ax.set_xlabel('Train Size', fontsize=18)
        ax.set_ylabel('MAE', fontsize=18)
        f_name = f'plot_learn_curve_{descriptor}_{target}_{seed}.png'
        fig.savefig(os.path.join(figpath, f_name))

if __name__ == "__main__":
    # Example of how to call the main function
    main("descriptor", "target.txt", "data_path", "dataset", 
         save_predictions=True, random_seeds=[42, 43, 44], plotting=True)
