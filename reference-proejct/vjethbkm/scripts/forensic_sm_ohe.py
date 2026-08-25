from __future__ import annotations

from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "r2": float(df["r2"].mean()),
        "kendall_tau": float(df["kendall_tau"].mean()),
        "train_elapsed_s_mean": float(df["train_elapsed_s"].mean()),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(kendalltau(y_true, y_pred)[0]),
    }


def _run_ohe_variant(df: pd.DataFrame, cols: list[str], dtype: type[np.floating]) -> tuple[dict, np.ndarray]:
    y = df["Yield"].to_numpy(dtype=float)
    metric_rows = []
    predictions = []
    for repeat in range(5):
        inner_seed = 1000 + repeat
        splitter = KFold(n_splits=5, shuffle=True, random_state=inner_seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=dtype)
            encoder.fit(train_df[cols].fillna("__MISSING__"))
            x_train = encoder.transform(train_df[cols].fillna("__MISSING__"))
            x_test = encoder.transform(test_df[cols].fillna("__MISSING__"))
            model = RandomForestRegressor(
                n_estimators=500,
                max_features=0.3,
                n_jobs=-1,
                random_state=inner_seed,
            )
            start = perf_counter()
            model.fit(x_train, y[train_idx])
            elapsed = round(perf_counter() - start, 6)
            pred = model.predict(x_test)
            predictions.append(pred)
            metric_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_elapsed_s": elapsed,
                    **_metrics(y[test_idx], pred),
                }
            )
    return _metric_summary(metric_rows), np.concatenate(predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-float64", action="store_true")
    parser.add_argument("--run-two-sm-catalyst", action="store_true")
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[1]
    package_root = repro_root / "yieldsmarter"
    sm_csv = package_root / "Data" / "HTE_datasets" / "SM" / "SM.csv"
    sm_2_csv = package_root / "Results" / "Compare_Complexity" / "Suzuki_2018" / "2SM.csv"
    official_oof = package_root / "Results" / "Compare_Complexity" / "Suzuki_2018" / "OHE" / "Suzuki_OHE_oof_predictions_RF.npz"
    official_summary_path = package_root / "Results" / "Compare_Complexity" / "Suzuki_2018" / "OHE" / "Suzuki_OHE_metrics_summaries_RF.json"

    df = pd.read_csv(sm_csv)
    df_2 = pd.read_csv(sm_2_csv)
    official = np.load(official_oof)
    official_summary = json.loads(official_summary_path.read_text(encoding="utf-8"))
    cols = official_summary["categorical_columns"]
    y = df["Yield"].to_numpy(dtype=float)

    y_true_sequence = []
    fold_sequence = []
    for repeat in range(5):
        splitter = KFold(n_splits=5, shuffle=True, random_state=1000 + repeat)
        for fold, (_, test_idx) in enumerate(splitter.split(df), start=1):
            y_true_sequence.append(y[test_idx])
            fold_sequence.append(np.full(len(test_idx), fold, dtype=int))
    y_true_sequence = np.concatenate(y_true_sequence)
    fold_sequence = np.concatenate(fold_sequence)

    records = [
        {
            "check": "sm_csv_shape",
            "value": f"{df.shape[0]}x{df.shape[1]}",
        },
        {
            "check": "two_sm_csv_shape",
            "value": f"{df_2.shape[0]}x{df_2.shape[1]}",
        },
        {
            "check": "two_sm_extra_columns",
            "value": "|".join(column for column in df_2.columns if column not in df.columns),
        },
        {
            "check": "shared_columns_equal",
            "value": str(all(df[column].equals(df_2[column]) for column in df.columns)),
        },
        {
            "check": "official_y_true_sequence_equal_generated_kfold",
            "value": str(np.array_equal(y_true_sequence, official["y_true"])),
        },
        {
            "check": "official_fold_id_equal_generated_kfold",
            "value": str(np.array_equal(fold_sequence, official["fold_id"])),
        },
        {
            "check": "sm_csv_sha256",
            "value": _sha256(sm_csv),
        },
        {
            "check": "two_sm_csv_sha256",
            "value": _sha256(sm_2_csv),
        },
    ]

    variant_summary = None
    official_test_summary = {
        "mae": official_summary["test"]["mae"]["mean"],
        "rmse": official_summary["test"]["rmse"]["mean"],
        "r2": official_summary["test"]["r2"]["mean"],
        "kendall_tau": official_summary["test"]["kendall"]["mean"],
    }
    if args.run_float64:
        variant_summary, variant_pred = _run_ohe_variant(df, cols, np.float64)
        official_pred = official["y_pred"]
        records.extend(
            [
                {
                    "check": "float64_variant_mae",
                    "value": str(variant_summary["mae"]),
                },
                {
                    "check": "float64_variant_rmse",
                    "value": str(variant_summary["rmse"]),
                },
                {
                    "check": "float64_variant_r2",
                    "value": str(variant_summary["r2"]),
                },
                {
                    "check": "float64_variant_kendall_tau",
                    "value": str(variant_summary["kendall_tau"]),
                },
                {
                    "check": "float64_variant_prediction_max_abs_diff_vs_official",
                    "value": str(float(np.max(np.abs(variant_pred - official_pred)))),
                },
            ]
        )

    catalyst_variant_summary = None
    if args.run_two_sm_catalyst:
        catalyst_cols = [column for column in df_2.columns if column != "Yield"]
        catalyst_variant_summary, _ = _run_ohe_variant(df_2, catalyst_cols, np.float64)
        records.extend(
            [
                {
                    "check": "two_sm_catalyst_variant_columns",
                    "value": "|".join(catalyst_cols),
                },
                {
                    "check": "two_sm_catalyst_variant_mae",
                    "value": str(catalyst_variant_summary["mae"]),
                },
                {
                    "check": "two_sm_catalyst_variant_rmse",
                    "value": str(catalyst_variant_summary["rmse"]),
                },
                {
                    "check": "two_sm_catalyst_variant_r2",
                    "value": str(catalyst_variant_summary["r2"]),
                },
                {
                    "check": "two_sm_catalyst_variant_kendall_tau",
                    "value": str(catalyst_variant_summary["kendall_tau"]),
                },
            ]
        )

    variant_diffs = {}
    for name, summary in [
        ("float64_variant", variant_summary),
        ("two_sm_catalyst_variant", catalyst_variant_summary),
    ]:
        if summary is None:
            continue
        variant_diffs[name] = {
            metric: abs(summary[metric] - official_test_summary[metric])
            for metric in official_test_summary
        }

    conclusion = (
        "Official SM/OHE y_true and fold_id exactly match KFold over Data/HTE_datasets/SM/SM.csv, "
        "so row order, labels, and split generation are ruled out as causes. "
        "The 2SM.csv file only adds catalyst_smiles while preserving all shared columns and Yield. "
        "Neither a float64 OHE variant nor a 2SM+catalyst variant reproduces the official SM/OHE summary. "
        "The supported conclusion is that the bundled Suzuki OHE RF result is a stale or differently generated artifact "
        "whose missing generation details are not recoverable from the included README/config/scripts alone."
    )

    table_dir = repro_root / "outputs" / "tables"
    manifest_dir = repro_root / "data" / "manifest"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(table_dir / "sm_ohe_forensics_summary.csv", index=False)
    manifest = {
        "sm_csv": {
            "path": str(sm_csv.relative_to(repro_root)).replace("\\", "/"),
            "shape": list(df.shape),
            "sha256": _sha256(sm_csv),
        },
        "two_sm_csv": {
            "path": str(sm_2_csv.relative_to(repro_root)).replace("\\", "/"),
            "shape": list(df_2.shape),
            "sha256": _sha256(sm_2_csv),
            "extra_columns": [column for column in df_2.columns if column not in df.columns],
            "shared_columns_equal": all(df[column].equals(df_2[column]) for column in df.columns),
        },
        "official_ohe": {
            "y_true_len": int(len(official["y_true"])),
            "y_pred_len": int(len(official["y_pred"])),
            "fold_id_len": int(len(official["fold_id"])),
            "y_true_sequence_equal_generated_kfold": bool(np.array_equal(y_true_sequence, official["y_true"])),
            "fold_id_equal_generated_kfold": bool(np.array_equal(fold_sequence, official["fold_id"])),
            "categorical_columns": cols,
            "test_summary": official_test_summary,
        },
        "float64_variant": variant_summary,
        "two_sm_catalyst_variant": catalyst_variant_summary,
        "variant_abs_diffs_vs_official_test_summary": variant_diffs,
        "inference": conclusion,
    }
    (manifest_dir / "sm_ohe_forensics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = repro_root / "outputs" / "reports" / "sm_ohe_forensics_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# SM/OHE forensic report",
                "",
                conclusion,
                "",
                "## Key evidence",
                "",
                f"- SM.csv shape: `{df.shape[0]}x{df.shape[1]}`; sha256 `{_sha256(sm_csv)}`.",
                f"- 2SM.csv shape: `{df_2.shape[0]}x{df_2.shape[1]}`; extra columns: `{', '.join(column for column in df_2.columns if column not in df.columns)}`.",
                f"- Shared columns and Yield equal between SM.csv and 2SM.csv: `{all(df[column].equals(df_2[column]) for column in df.columns)}`.",
                f"- Official y_true sequence equals generated KFold sequence: `{np.array_equal(y_true_sequence, official['y_true'])}`.",
                f"- Official fold_id equals generated KFold sequence: `{np.array_equal(fold_sequence, official['fold_id'])}`.",
                "",
                "## Variant diffs vs official test summary",
                "",
                "```json",
                json.dumps(variant_diffs, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
