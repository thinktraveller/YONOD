"""Run paper-native train-fold OHE + RF reproductions without modifying YONOD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from run_paper_native_mfp_rf import METRICS, _write_csv, fold_metrics, package_versions, run_rf_5x5, sha256


OHE_SPECS: dict[str, dict[str, str]] = {
    "BH1": {
        "paper_dataset_id": "BH", "csv": "yieldsmarter/Data/HTE_datasets/BH1/BH1.csv",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/OHE/Doyle_OHE_metrics_summaries_RF.json",
        "official_oof": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/OHE/Doyle_OHE_oof_predictions_RF.npz",
        "train_config": "yieldsmarter/Src/Train/BH1_OHE.json", "unit": "yield %",
    },
    "BH2": {
        "paper_dataset_id": "BH2", "csv": "yieldsmarter/Data/HTE_datasets/BH2/BH2.csv",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/OHE/Denmark_OHE_metrics_summaries_RF.json",
        "official_oof": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/OHE/Denmark_OHE_oof_predictions_RF.npz",
        "train_config": "yieldsmarter/Src/Train/BH2_OHE.json", "unit": "yield %",
    },
    "SL1": {
        "paper_dataset_id": "SLAP", "csv": "yieldsmarter/Data/HTE_datasets/SL1/SL1.csv",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Bode_2023/OHE/Bode_OHE_metrics_summaries_RF.json",
        "official_oof": "yieldsmarter/Results/Compare_Complexity/Bode_2023/OHE/Bode_OHE_oof_predictions_RF.npz",
        "train_config": "yieldsmarter/Src/Train/SL1_OHE.json", "unit": "LC-MS product ratio",
    },
    "SM": {
        "paper_dataset_id": "SM", "csv": "yieldsmarter/Data/HTE_datasets/SM/SM.csv",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/OHE/Suzuki_OHE_metrics_summaries_RF.json",
        "official_oof": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/OHE/Suzuki_OHE_oof_predictions_RF.npz",
        "train_config": "yieldsmarter/Src/Train/SM_OHE.json", "unit": "yield %",
    },
}


def feature_splits_from_encoder(encoder: OneHotEncoder) -> list[tuple[int, int]]:
    splits: list[tuple[int, int]] = []
    start = 0
    for categories in encoder.categories_:
        end = start + len(categories)
        splits.append((start, end))
        start = end
    return splits


def ohe_fit_transform_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Mirror the bundled Train_OHE.py train-only OHE transform exactly."""
    missing_token = "__MISSING__"
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[columns].fillna(missing_token))
    splits = feature_splits_from_encoder(encoder)
    train_features = encoder.transform(train_df[columns].fillna(missing_token))
    test_features = encoder.transform(test_df[columns].fillna(missing_token))
    for index, column in enumerate(columns):
        start, end = splits[index]
        train_missing = train_df[column].isna().to_numpy()
        test_missing = test_df[column].isna().to_numpy()
        if train_missing.any():
            train_features[train_missing, start:end] = 0.0
        if test_missing.any():
            test_features[test_missing, start:end] = 0.0
    audit = {
        "ohe_output_dim": int(train_features.shape[1]),
        "train_missing_value_count": int(train_df[columns].isna().sum().sum()),
        "test_missing_value_count": int(test_df[columns].isna().sum().sum()),
        "unseen_test_value_count": int(sum((~test_df[column].isna() & ~test_df[column].isin(encoder.categories_[index])).sum() for index, column in enumerate(columns))),
    }
    return train_features, test_features, audit


def metric_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        metric: {"mean": float(np.mean([record[metric] for record in records])), "std": float(np.std([record[metric] for record in records]))}
        for metric in METRICS
    }


def run_ohe_rf_5x5(dataframe: pd.DataFrame, target: str, columns: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Mirror the 5 repeats × 5 splits OHE/RF control flow of Train_OHE.py."""
    records: list[dict[str, Any]] = []
    y_true_outer: list[np.ndarray] = []
    y_pred_outer: list[np.ndarray] = []
    fold_id_outer: list[np.ndarray] = []
    for repeat in range(5):
        repeat_seed = 1000 + repeat
        splitter = KFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        fold_id_all: list[np.ndarray] = []
        for fold, (train_index, test_index) in enumerate(splitter.split(dataframe), start=1):
            train_df = dataframe.iloc[train_index].reset_index(drop=True)
            test_df = dataframe.iloc[test_index].reset_index(drop=True)
            train_features, test_features, audit = ohe_fit_transform_fold(train_df, test_df, columns)
            model = RandomForestRegressor(
                n_estimators=500, max_features=0.30, n_jobs=-1, random_state=repeat_seed
            )
            model.fit(train_features, train_df[target].to_numpy(dtype=float))
            y_true = test_df[target].to_numpy(dtype=float)
            y_pred = model.predict(test_features)
            record: dict[str, Any] = {
                "repeat": repeat + 1, "fold": fold, "repeat_seed": repeat_seed,
                "train_rows": int(len(train_index)), "test_rows": int(len(test_index)), **audit,
            }
            record.update(fold_metrics(y_true, y_pred))
            records.append(record)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            fold_id_all.append(np.full_like(y_true, fold, dtype=int))
        y_true_outer.append(np.concatenate(y_true_all))
        y_pred_outer.append(np.concatenate(y_pred_all))
        fold_id_outer.append(np.concatenate(fold_id_all))
    return records, metric_summary(records), {
        "y_true": np.concatenate(y_true_outer), "y_pred": np.concatenate(y_pred_outer), "fold_id": np.concatenate(fold_id_outer),
    }


def oof_alignment(local: dict[str, np.ndarray], official_path: Path) -> dict[str, Any]:
    with np.load(official_path) as official:
        prediction_shape_equal = local["y_pred"].shape == official["y_pred"].shape
        max_abs_diff = float(np.abs(local["y_pred"] - official["y_pred"]).max()) if prediction_shape_equal else None
        return {
            "local_oof_rows": int(len(local["y_true"])), "official_oof_rows": int(len(official["y_true"])),
            "y_true_exact": bool(np.array_equal(local["y_true"], official["y_true"])),
            "fold_id_exact": bool(np.array_equal(local["fold_id"], official["fold_id"])),
            "y_pred_exact": bool(np.array_equal(local["y_pred"], official["y_pred"])),
            "y_pred_max_abs_diff": max_abs_diff,
            "y_pred_within_1e_12": bool(max_abs_diff is not None and max_abs_diff <= 1e-12),
        }


def comparison_rows(dataset_id: str, spec: dict[str, str], local: dict[str, dict[str, float]], official: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        official_item = official["test"][metric]
        rows.append({
            "dataset_id": dataset_id, "paper_dataset_id": spec["paper_dataset_id"], "descriptor": "OHE", "model": "RF",
            "metric": "kendall_tau" if metric == "kendall" else metric, "unit": spec["unit"],
            "official_json_mean": official_item["mean"], "official_json_std": official_item["std"],
            "independent_native_mean": local[metric]["mean"], "independent_native_std": local[metric]["std"],
            "native_minus_official_mean": local[metric]["mean"] - official_item["mean"],
            "native_minus_official_std": local[metric]["std"] - official_item["std"], "valid_fold_count": 25,
        })
    return rows


def run_dataset(repro_root: Path, output_dir: Path, dataset_id: str) -> dict[str, Any]:
    spec = OHE_SPECS[dataset_id]
    source_csv = repro_root / spec["csv"]
    official_summary = json.loads((repro_root / spec["official_summary"]).read_text(encoding="utf-8"))
    dataframe = pd.read_csv(source_csv)
    target = "Yield"
    columns = [column for column in dataframe.columns if column != target]
    records, summary, local_oof = run_ohe_rf_5x5(dataframe, target, columns)
    alignment = oof_alignment(local_oof, repro_root / spec["official_oof"])
    rows = comparison_rows(dataset_id, spec, summary, official_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"{dataset_id}_OHE_RF_fold_metrics.csv", records)
    _write_csv(output_dir / f"{dataset_id}_OHE_RF_comparison.csv", rows)
    np.savez_compressed(output_dir / f"{dataset_id}_OHE_RF_oof_predictions.npz", **local_oof)
    return {
        "dataset_id": dataset_id, "paper_dataset_id": spec["paper_dataset_id"], "source_csv": spec["csv"],
        "source_csv_sha256": sha256(source_csv), "source_rows": int(len(dataframe)), "categorical_columns": columns,
        "upstream_training_config": spec["train_config"], "official_summary_json": spec["official_summary"],
        "official_oof_predictions": spec["official_oof"], "oof_alignment": alignment,
        "independent_native_metrics": summary, "comparison_rows": rows,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Four-dataset OHE + RF independent paper-native rerun", "",
        "This mirrors the unmodified `Train_OHE.py` OHE/RF branch: each fold fits `OneHotEncoder(handle_unknown='ignore')` on the training rows only, zeros whole missing-component blocks, then fits RF with the paper's 500 trees, 0.30 max_features and repeat seeds 1000–1004. The upstream file remains unmodified, but its unconditional unavailable LightGBM import requires this RF-only wrapper. YONOD is not used or modified.", "",
        "- The report compares all 25 fold metrics with the bundled official JSON and the concatenated OOF arrays with the bundled official OOF artifact.",
        "- SI records RDKit 2025.03.6 and scikit-learn 1.6.1; actual installed versions are in the JSON. Therefore a differing-stack run is not claimed to be bit-identical across environments.", "",
        "| Dataset | Rows | Metric | Official JSON mean ± std | Independent native mean ± std | Δ mean | Δ std | OOF y/fold/pred exact | max |Δ pred| |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for item in result["datasets"]:
        alignment = item["oof_alignment"]
        exact = f"{alignment['y_true_exact']}/{alignment['fold_id_exact']}/{alignment['y_pred_exact']}"
        for row in item["comparison_rows"]:
            lines.append(f"| {item['dataset_id']} | {item['source_rows']} | {row['metric']} | {row['official_json_mean']:.12g} ± {row['official_json_std']:.12g} | {row['independent_native_mean']:.12g} ± {row['independent_native_std']:.12g} | {row['native_minus_official_mean']:.12g} | {row['native_minus_official_std']:.12g} | {exact} | {alignment['y_pred_max_abs_diff']:.12g} |")
    return "\n".join(lines) + "\n"


def run(repro_root: Path, output_dir: Path, datasets: list[str]) -> dict[str, Any]:
    dataset_results = [run_dataset(repro_root, output_dir, dataset_id) for dataset_id in datasets]
    rows = [row for item in dataset_results for row in item["comparison_rows"]]
    result = {
        "run_type": "independent_paper_native_ohe_rf_matrix", "datasets": dataset_results,
        "actual_versions": package_versions(), "paper_reported_versions": {"rdkit": "2025.03.6", "scikit-learn": "1.6.1"},
        "cv": {"outer_splits": 5, "splits": 5, "seed": 1000, "repeat_seeds": [1000, 1001, 1002, 1003, 1004], "valid_fold_count": 25},
        "rf": {"n_estimators": 500, "max_features": 0.30, "n_jobs": -1},
        "implementation_boundary": "RF-only wrapper; upstream Train_OHE.py files are unmodified and their unconditional LightGBM import is unavailable locally.",
    }
    _write_csv(output_dir / "OHE_RF_matrix_comparison.csv", rows)
    (output_dir / "OHE_RF_matrix_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "OHE_RF_matrix_reproduction.md").write_text(markdown(result), encoding="utf-8")
    tables_dir = repro_root / "outputs" / "tables"
    reports_dir = repro_root / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "native_ohe_rf_matrix_comparison.csv", rows)
    (tables_dir / "native_ohe_rf_matrix_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "native_ohe_rf_matrix_reproduction.md").write_text(markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(OHE_SPECS), default=sorted(OHE_SPECS))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "outputs" / "runs" / "native_ohe_rf_matrix"
    run(root, output_dir, args.datasets)
    print(json.dumps({"datasets": args.datasets, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
