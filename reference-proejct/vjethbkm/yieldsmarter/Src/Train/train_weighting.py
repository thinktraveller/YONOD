import json
from pathlib import Path
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

# weighting utilities
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid


def calculate_focal_weights(y_true, y_pred, alpha=2.0, gamma=2.0):
    loss = np.abs(y_true - y_pred)
    return sigmoid(alpha * loss)**gamma


def calculate_lds_weights(y_true, sigma=1.5):
    kde = gaussian_kde(y_true, bw_method='scott')
    label_density = kde(y_true)
    weights = 1 / (label_density + 1e-6)
    return weights / np.max(weights)


METRICS = ("mae", "rmse", "r2", "medae", "kendall")


def eval_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "kendall": float(kendalltau(y_true, y_pred)[0]),
    }


DEFAULT_LGBM_PARAMS = dict(n_estimators=500, colsample_bytree=0.3, n_jobs=-1)
DEFAULT_RF_PARAMS = dict(n_estimators=500, max_features=0.3, n_jobs=-1)


# Model factory
def make_model(model_type: str, seed: int, cfg: dict = {}):
    mn = model_type.lower()

    if mn == "knn":
        n_neighbors = int(cfg.get("knn_n_neighbors", 5))
        return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    if mn == "ridge":
        alpha = float(cfg.get("ridge_alpha", 1.0))
        return Ridge(alpha=alpha, random_state=seed)
    if mn in ("lgbm", "lightgbm"):
        params = dict(DEFAULT_LGBM_PARAMS)
        params.update(cfg.get("lgbm_params", {}))
        return LGBMRegressor(**params, random_state=seed)
    if mn == "rf" or mn == "randomforest":
        params = dict(DEFAULT_RF_PARAMS)
        params.update(cfg.get("rf_params", {}))
        return RandomForestRegressor(**params, random_state=seed)
    raise ValueError(
        "Unknown model. Choose from: KNN, Ridge, LightGBM, RandomForest.")


# Main training routine
def train_and_save_results(
        X,
        y,
        run_name: str,
        *,
        model_type: str,
        metrics_out: Path,
        metrics_summaries_out: Path,
        preds_out: Path,
        outer_splits: int = 5,
        n_splits: int = 5,
        seed: int = 1000,
        apply_weighting: bool = False,
        weight_mix: float = 0.5,  # mean of focal + lds
):
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_summaries_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)

    train_scores = {m: [] for m in METRICS}
    test_scores = {m: [] for m in METRICS}

    y_true_all, y_pred_all = [], []

    for outer_fold in range(0, outer_splits):
        inner_seed = seed + outer_fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=inner_seed)

        y_true_all, y_pred_all, fold_id_all = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
            print(f"--- Fold {fold} ---")
            X_train = X.take(tr_idx, axis=0)
            X_test = X.take(te_idx, axis=0)
            y_train = y.take(tr_idx, axis=0)
            y_test = y.take(te_idx, axis=0)

            model = make_model(model_type, inner_seed)

            # Optional weighting (TRAINING FOLD ONLY)
            if apply_weighting:
                # initial fit
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)

                w_focal = calculate_focal_weights(y_train, y_train_pred)
                w_lds = calculate_lds_weights(y_train)

                sample_weight = weight_mix * w_focal + (1.0 -
                                                        weight_mix) * w_lds
                sample_weight = sample_weight / np.max(sample_weight)

                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            for m, v in eval_metrics(y_train, y_pred_train).items():
                train_scores[m].append(v)

            for m, v in eval_metrics(y_test, y_pred_test).items():
                test_scores[m].append(v)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred_test)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    metrics_summary = {
        "model_name": run_name,
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
    }

    with open(metrics_out, "w") as f:
        json.dump({
            "train_metrics": train_scores,
            "test_metrics": test_scores
        },
                  f,
                  indent=2)
    with open(metrics_summaries_out, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    np.savez(preds_out, y_true=y_true_all, y_pred=y_pred_all)

    print(f"Saved metrics: {metrics_summaries_out}")
    print(f"Saved predictions: {preds_out}")

    return metrics_summary


def main():
    import argparse
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

    metrics_out = Path(cfg["metrics_out"].replace(
        'Compare_Complexity', 'Reweighting').format(descr_type=descriptor_type,
                                                    model_name=model_name))
    metrics_summaries_out = Path(cfg["metrics_summaries_out"].replace(
        'Compare_Complexity', 'Reweighting').format(descr_type=descriptor_type,
                                                    model_name=model_name))
    preds_out = Path(cfg["preds_out"].replace(
        'Compare_Complexity', 'Reweighting').format(descr_type=descriptor_type,
                                                    model_name=model_name))

    X = input_data['X']
    if cfg.get("set_nan_to_zero", False):
        np.nan_to_num(X, copy=False, nan=0.0)
    y = input_data['y']

    print(
        f"\n=== {descriptor_type} + {model_name} | CV splits={outer_splits}x{splits} | seed={seed} ==="
    )

    # Run WEIGHTED
    train_and_save_results(
        X,
        y,
        run_name=f"{model_name}_{descriptor_type}_weighted",
        model_type=model_name,
        metrics_out=metrics_out,
        metrics_summaries_out=metrics_summaries_out,
        preds_out=preds_out,
        apply_weighting=True,
        outer_splits=outer_splits,
        n_splits=splits,
    )


if __name__ == "__main__":
    main()
