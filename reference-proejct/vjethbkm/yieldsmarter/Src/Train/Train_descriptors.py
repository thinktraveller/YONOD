import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import kendalltau

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

METRICS = ("mae", "rmse", "r2", "medae", "kendall")
DEFAULT_LGBM_PARAMS = dict(n_estimators=500, colsample_bytree=0.3, n_jobs=-1)
DEFAULT_RF_PARAMS = dict(n_estimators=500, max_features=0.3, n_jobs=-1)


def eval_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "kendall": float(kendalltau(y_true, y_pred)[0]),
    }


def make_model(model_name: str, cfg: dict, random_state: int):
    mn = model_name.lower()
    if mn == "knn":
        n_neighbors = int(cfg.get("knn_n_neighbors", 5))
        return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    if mn == "ridge":
        alpha = float(cfg.get("ridge_alpha", 1.0))
        return Ridge(alpha=alpha, random_state=random_state)
    if mn in ("lgbm", "lightgbm"):
        params = dict(DEFAULT_LGBM_PARAMS)
        params.update(cfg.get("lgbm_params", {}))
        return LGBMRegressor(**params, random_state=random_state)
    if mn == "rf" or mn == "randomforest":
        params = dict(DEFAULT_RF_PARAMS)
        params.update(cfg.get("rf_params", {}))
        return RandomForestRegressor(**params, random_state=random_state)
    raise ValueError(
        "Unknown model. Choose from: KNN, Ridge, LightGBM, RandomForest.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="RF",
        help="Model to train: KNN, Ridge, LightGBM, RandomForest")
    parser.add_argument(
        "-d",
        "--descriptor_type",
        default="DFT",
        choices=["DFT", "MFP", "PhysChem", "SOAP"],
        help="Descriptor type to use. Choose from: DFT, MFP, PhysChem, SOAP.")
    parser.add_argument("input_file", help="JSON file with configuration data")
    args = parser.parse_args()

    cfg_path = Path(args.input_file)
    cfg = json.load(cfg_path.open())

    model_name = args.model
    descriptor_type = args.descriptor_type

    input_data_path = Path(
        cfg["descriptors_npz"].format(descr_type=descriptor_type))
    if not input_data_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_data_path}")
    input_data = np.load(input_data_path)

    outer_splits = cfg.get("outer_splits", 5)
    splits = int(cfg.get("splits", 5))
    seed = int(cfg.get("seed", 1000))

    metrics_out = Path(cfg["metrics_out"].format(descr_type=descriptor_type,
                                                 model_name=model_name))
    metrics_summaries_out = Path(cfg["metrics_summaries_out"].format(
        descr_type=descriptor_type, model_name=model_name))
    preds_out = Path(cfg["preds_out"].format(descr_type=descriptor_type,
                                             model_name=model_name))

    X = input_data['X']
    if cfg.get("set_nan_to_zero", False):
        np.nan_to_num(X, copy=False, nan=0.0)
    y = input_data['y']

    train_scores = {m: [] for m in METRICS}
    test_scores = {m: [] for m in METRICS}

    print(
        f"\n=== {descriptor_type} + {model_name} | CV splits={outer_splits}x{splits} | seed={seed} ==="
    )
    y_true_outer, y_pred_outer, fold_id_outer = [], [], []
    for outer_fold in range(0, outer_splits):
        inner_seed = seed + outer_fold
        kf = KFold(n_splits=splits, shuffle=True, random_state=inner_seed)

        y_true_all, y_pred_all, fold_id_all = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
            X_train = X.take(tr_idx, axis=0)
            X_test = X.take(te_idx, axis=0)
            y_train = y.take(tr_idx, axis=0)
            y_test = y.take(te_idx, axis=0)

            model = make_model(model_name, cfg, random_state=inner_seed)
            model.fit(X_train, y_train)

            yhat_train = model.predict(X_train)
            yhat_test = model.predict(X_test)

            for metric, val in eval_metrics(y_train, yhat_train).items():
                train_scores[metric].append(val)
            for metric, val in eval_metrics(y_test, yhat_test).items():
                test_scores[metric].append(val)

            y_true_all.append(y_test)
            y_pred_all.append(yhat_test)
            fold_id_all.append(np.full_like(y_test, fold, dtype=int))

            print(f" Fold {fold} complete.")

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        fold_id_all = np.concatenate(fold_id_all)
        y_true_outer.append(y_true_all)
        y_pred_outer.append(y_pred_all)
        fold_id_outer.append(fold_id_all)
        metrics_summary = {
            "model": model_name,
            "splits": splits,
            "seed": seed,
            "train": {
                m: {
                    "mean": float(np.mean(train_scores[m])),
                    "std": float(np.std(train_scores[m]))
                }
                for m in METRICS
            },
            "test": {
                m: {
                    "mean": float(np.mean(test_scores[m])),
                    "std": float(np.std(test_scores[m]))
                }
                for m in METRICS
            },
            "descriptors_npz": str(input_data_path),
        }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_summaries_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)

    with metrics_out.open("w") as f:
        json.dump({
            "train_metrics": train_scores,
            "test_metrics": test_scores
        },
                  f,
                  indent=2)

    with metrics_summaries_out.open("w") as f:
        json.dump(metrics_summary, f, indent=2)

    np.savez_compressed(preds_out,
                        y_true=np.concatenate(y_true_outer),
                        y_pred=np.concatenate(y_pred_outer),
                        fold_id=np.concatenate(fold_id_outer))

    print(f"\nSaved metrics: {metrics_out}")
    print(f"Saved OOF predictions: {preds_out}")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
