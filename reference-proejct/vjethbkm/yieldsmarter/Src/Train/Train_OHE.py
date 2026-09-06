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
"""
- fits OneHotEncoder ONLY on the training split (no leakage)
"""
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


def feature_splits_from_encoder(enc: OneHotEncoder):
    splits = []
    start = 0
    for cats in enc.categories_:
        end = start + len(cats)
        splits.append((start, end))
        start = end
    return splits


def ohe_fit_transform_fold(train_df,
                           test_df,
                           cols,
                           missing_token="__MISSING__",
                           zero_out_missing=True):
    """
    Fit encoder on train only, transform both train and test.
    """
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(train_df[cols].fillna(missing_token))

    splits = feature_splits_from_encoder(enc)

    Xtr = enc.transform(train_df[cols].fillna(missing_token))
    Xte = enc.transform(test_df[cols].fillna(missing_token))

    if zero_out_missing:
        Xtr = Xtr.copy()
        Xte = Xte.copy()
        for i, c in enumerate(cols):
            s, e = splits[i]
            mtr = train_df[c].isna().to_numpy()
            mte = test_df[c].isna().to_numpy()
            if mtr.any():
                Xtr[mtr, s:e] = 0.0
            if mte.any():
                Xte[mte, s:e] = 0.0

    return Xtr, Xte


def make_model(model_name: str, cfg: dict, random_state: int):
    #mn = model_name.lower()
    if model_name == "KNN":
        n_neighbors = int(cfg.get("knn_n_neighbors", 5))
        return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    elif model_name == "Ridge":
        alpha = float(cfg.get("ridge_alpha", 1.0))
        return Ridge(alpha=alpha, random_state=random_state)
    elif model_name in ("LGBM", "LightGBM"):
        params = dict(DEFAULT_LGBM_PARAMS)
        params.update(cfg.get("lgbm_params", {}))
        return LGBMRegressor(**params, random_state=random_state)
    elif model_name == "RF" or model_name == "RandomForest":
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
    parser.add_argument("input_file",
                        help="CSV file with SMILES + Yield columns")
    args = parser.parse_args()

    cfg_path = Path(args.input_file)
    cfg = json.load(cfg_path.open())

    model_name = args.model

    reaction_csv = Path(cfg["reaction_csv"])
    if not reaction_csv.exists():
        raise FileNotFoundError(f"reaction_csv not found: {reaction_csv}")
    df = pd.read_csv(reaction_csv)
    target = cfg.get("target_col", "Yield")
    cols = [x for x in df.columns if x != target]

    # cols = cfg["categorical_columns"]

    splits = int(cfg.get("splits", 5))
    seed = int(cfg.get("seed", 1000))

    missing_token = cfg.get("missing_token", "__MISSING__")
    zero_out_missing = bool(cfg.get("zero_out_missing", True))

    metrics_out = Path(
        cfg.get("metrics_out", "Results/OHE_metrics_{model_name}.json").format(
            model_name=model_name))
    metrics_summaries_out = Path(
        cfg.get("metrics_summaries_out",
                "Results/OHE_metrics_summaries_{model_name}.json").format(
                    model_name=model_name))
    preds_out = Path(
        cfg.get("preds_out",
                "Results/OHE_oof_predictions_{model_name}.npz").format(
                    model_name=model_name))

    for c in cols:
        if c not in df.columns:
            raise ValueError(
                f"Missing categorical column '{c}' in {reaction_csv}")
    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}' in {reaction_csv}")

    X_df = df[cols]
    y = df[target].to_numpy(dtype=float)

    train_scores = {m: [] for m in METRICS}
    test_scores = {m: [] for m in METRICS}

    outer_splits = cfg.get("outer_splits", 1)
    print(
        f"\n=== OHE + {model_name} | CV splits={outer_splits}x{splits} | seed={seed} ==="
    )
    y_true_outer, y_pred_outer, fold_id_outer = [], [], []
    for outer_fold in range(0, outer_splits):
        inner_seed = seed + outer_fold
        kf = KFold(n_splits=splits, shuffle=True, random_state=inner_seed)

        y_true_all, y_pred_all, fold_id_all = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(kf.split(df), 1):
            tr_df = df.iloc[tr_idx].reset_index(drop=True)
            te_df = df.iloc[te_idx].reset_index(drop=True)

            Xtr, Xte = ohe_fit_transform_fold(
                tr_df,
                te_df,
                cols,
                missing_token=missing_token,
                zero_out_missing=zero_out_missing,
            )

            ytr = tr_df[target].to_numpy(dtype=float)
            yte = te_df[target].to_numpy(dtype=float)

            model = make_model(model_name, cfg, random_state=inner_seed)
            model.fit(Xtr, ytr)

            yhat_tr = model.predict(Xtr)
            yhat_te = model.predict(Xte)

            for metric, val in eval_metrics(ytr, yhat_tr).items():
                train_scores[metric].append(val)
            for metric, val in eval_metrics(yte, yhat_te).items():
                test_scores[metric].append(val)

            y_true_all.append(yte)
            y_pred_all.append(yhat_te)
            fold_id_all.append(np.full_like(yte, fold, dtype=int))

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
            "categorical_columns": cols,
            "target_col": target,
            "reaction_csv": str(reaction_csv),
            "ohe": {
                "missing_token": missing_token,
                "zero_out_missing": zero_out_missing,
                "handle_unknown": "ignore",
                "fit_scope": "train_only_per_fold",
            },
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
