import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import kendalltau

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

METRICS = ("mae", "rmse", "r2", "medae", "kendall")
DEFAULT_LGBM_PARAMS = dict(n_estimators=500, colsample_bytree=0.3, n_jobs=-1)
DEFAULT_RF_PARAMS = dict(n_estimators=500, max_features=0.3, n_jobs=-1)

# FP parameters
RADIUS = 3
NUM_BITS = 1024
smiles_cache = {}


def eval_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "kendall": float(kendalltau(y_true, y_pred)[0]),
    }


def make_model(model_name: str, random_state: int):
    mn = model_name.lower()
    if mn == "knn":
        n_neighbors = 5
        return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    if mn == "ridge":
        alpha = 1.0
        return Ridge(alpha=alpha, random_state=random_state)
    if mn in ("lgbm", "lightgbm"):
        params = dict(DEFAULT_LGBM_PARAMS)
        return LGBMRegressor(**params, random_state=random_state)
    if mn == "rf" or mn == "randomforest":
        params = dict(DEFAULT_RF_PARAMS)
        return RandomForestRegressor(**params, random_state=random_state)
    raise ValueError(
        "Unknown model. Choose from: KNN, Ridge, LightGBM, RandomForest.")


# ----- code adapted from Src/Featurize/Gen_MFP.py
# Cache unique SMILES
# No need to run all, just unique identities
def smiles_to_mols(smiles_list):
    mols = []
    for s in smiles_list:
        if pd.isna(s) or s == "":
            mol = None
        else:
            mol = smiles_cache.get(s)
            if mol is None:
                mol = Chem.MolFromSmiles(str(s))
                smiles_cache[s] = mol
        mols.append(mol)
    return mols


# Fingerprint Count Vectors
def get_morgan_fp(mol, mfpgen):
    if mol is None:
        return np.zeros(NUM_BITS, dtype=int)
    return np.array(mfpgen.GetCountFingerprintAsNumPy(mol))


# Concatenate component vectors of each non-yield column
def featurize_reaction_dataset(df, smiles_columns):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS,
                                                       fpSize=NUM_BITS)
    all_mol_lists = [smiles_to_mols(df[col]) for col in smiles_columns]
    X = []
    for i in range(len(df)):
        fps = [get_morgan_fp(mols[i], mfpgen) for mols in all_mol_lists]
        X.append(np.concatenate(fps))
    return np.array(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="RF",
        help="Model to train: KNN, Ridge, LightGBM, RandomForest")
    args = parser.parse_args()
    descriptor_type = 'MFP'
    set_nan_to_zero = False
    model_name = args.model

    out_dir = Path(f"../../Results/BH2/Holdout/{descriptor_type}")
    model_in = out_dir / Path(f"model_{descriptor_type}_{args.model}.pkl")
    if not model_in.exists():
        raise FileNotFoundError(f"Model not found: {model_in}")
    with model_in.open("rb") as model_in_f:
        model = pickle.load(model_in_f)

    input_data_path = Path(f"../../Data/HTE_datasets/BH2/Per_Product")

    if not input_data_path.exists():
        raise FileNotFoundError(
            f"Input data directory not found: {input_data_path}")
    for filen in input_data_path.glob("Product_?.csv"):
        print(f"Processing {filen}...")
        df = pd.read_csv(filen)
        mfps = featurize_reaction_dataset(
            df, smiles_columns=[c for c in df.columns[3:]])
        preds = model.predict(mfps)
        df.drop(columns=[c for c in df.columns[3:]], inplace=True)
        df["Predicted_Yield"] = preds
        fn = filen.stem + "_preds.csv"
        df.to_csv(out_dir / Path(fn), index=False)
        print(f"Saved predictions to {out_dir / Path(fn)}")


if __name__ == "__main__":
    main()
