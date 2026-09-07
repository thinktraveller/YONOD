"""Independently rerun SL1 MFP + RF using paper-native data and tools.

The runner invokes unmodified upstream ``Gen_MFP.py`` on the official CSV and
mirrors only the RF branch of ``Train_descriptors.py``.  The upstream trainer
imports unavailable LightGBM before RF branch selection, so it cannot be used
unchanged in the local offline environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


SPEC = {
    "dataset_id": "SL1",
    "paper_dataset_id": "SLAP",
    "csv": "yieldsmarter/Data/HTE_datasets/SL1/SL1.csv",
    "official_npz": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP.npz",
    "official_summary": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP_metrics_summaries_RF.json",
    "train_config": "yieldsmarter/Src/Train/SL1.json",
    "unit": "LC-MS product ratio",
    "table_s3": {
        "mae": (18.0, 1.8), "rmse": (35.1, 5.1),
        "r2": (0.77, 0.05), "kendall": (0.51, 0.04),
    },
}
METRICS = ("mae", "rmse", "r2", "kendall")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "rdkit"):
        versions[package] = importlib.metadata.version(package)
    from rdkit import Chem
    versions["rdkit_runtime"] = Chem.rdBase.rdkitVersion
    return versions


def generate_mfp(repro_root: Path, source_csv: Path, output_npz: Path) -> None:
    """Call the unmodified bundled MFP generator with the active interpreter."""
    generator = repro_root / "yieldsmarter" / "Src" / "Featurize" / "Gen_MFP.py"
    source_csv = source_csv.resolve()
    output_npz = output_npz.resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, str(generator), str(source_csv), str(output_npz)],
        cwd=generator.parent, check=True, env=environment,
    )


def mfp_alignment(generated_npz: Path, official_npz: Path) -> dict[str, Any]:
    with np.load(generated_npz) as generated, np.load(official_npz) as official:
        same_shape = generated["X"].shape == official["X"].shape
        return {
            "generated_shape": list(generated["X"].shape),
            "official_shape": list(official["X"].shape),
            "x_exact": bool(np.array_equal(generated["X"], official["X"])),
            "x_max_abs_diff": int(np.abs(generated["X"].astype(np.int64) - official["X"].astype(np.int64)).max()) if same_shape else None,
            "y_exact": bool(np.array_equal(generated["y"], official["y"])),
            "smiles_columns_exact": bool(np.array_equal(generated["smiles_columns"], official["smiles_columns"])),
        }


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall": float(kendalltau(y_true, y_pred)[0]),
    }


def run_rf_5x5(X: np.ndarray, y: np.ndarray, seed: int = 1000) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Replicate the RF parameters, seeds and fold loop of Train_descriptors.py."""
    records: list[dict[str, Any]] = []
    for repeat in range(5):
        repeat_seed = seed + repeat
        splitter = KFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
            model = RandomForestRegressor(n_estimators=500, max_features=0.30, n_jobs=-1, random_state=repeat_seed)
            started = time.perf_counter()
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])
            record = {
                "repeat": repeat + 1, "fold": fold, "repeat_seed": repeat_seed,
                "train_rows": int(len(train_index)), "test_rows": int(len(test_index)),
                "fit_predict_seconds": time.perf_counter() - started,
            }
            record.update(fold_metrics(y[test_index], y_pred))
            records.append(record)
    return records, {
        metric: {
            "mean": float(np.mean([record[metric] for record in records])),
            "std": float(np.std([record[metric] for record in records])),
        }
        for metric in METRICS
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(local: dict[str, dict[str, float]], official: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        printed_mean, printed_std = SPEC["table_s3"][metric]
        official_item = official["test"][metric]
        digits = 1 if metric in {"mae", "rmse"} else 2
        rows.append({
            "dataset_id": SPEC["dataset_id"], "paper_dataset_id": SPEC["paper_dataset_id"],
            "descriptor": "MFP", "model": "RF", "metric": "kendall_tau" if metric == "kendall" else metric,
            "unit": SPEC["unit"], "published_table_s3_mean": printed_mean, "published_table_s3_std": printed_std,
            "official_json_mean": official_item["mean"], "official_json_std": official_item["std"],
            "independent_native_mean": local[metric]["mean"], "independent_native_std": local[metric]["std"],
            "native_minus_official_mean": local[metric]["mean"] - official_item["mean"],
            "native_minus_official_std": local[metric]["std"] - official_item["std"],
            "official_json_matches_table_s3_display": round(float(official_item["mean"]), digits) == printed_mean and round(float(official_item["std"]), digits) == printed_std,
            "valid_fold_count": 25,
        })
    return rows


def report(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SL1 MFP + RF independent paper-native rerun", "",
        "This independently regenerates MFP features from the official SL1 CSV using unmodified bundled `Gen_MFP.py`, then mirrors the RF-only branch of bundled `Train_descriptors.py`. It does not train from the official MFP NPZ and does not use or modify YONOD.", "",
        "## Provenance and environment", "",
        f"- Official CSV SHA256: `{result['source_csv_sha256']}`",
        f"- Generated MFP vs bundled NPZ: X exact `{result['mfp_alignment']['x_exact']}`, y exact `{result['mfp_alignment']['y_exact']}`, component columns exact `{result['mfp_alignment']['smiles_columns_exact']}`.",
        f"- Actual versions: `{json.dumps(result['actual_versions'], sort_keys=True)}`.",
        "- Paper SI reports RDKit 2025.03.6 and scikit-learn 1.6.1. This run records version drift and cannot claim bit-identical reproduction on a differing stack.",
        "- The unmodified upstream trainer imports unavailable LightGBM before selecting RF. The wrapper preserves its RF parameters, KFold schedule, random seeds and fold metrics without editing upstream files.", "",
        "## 5 × 5 result comparison", "",
        "| Metric | Table S3 mean ± std | Official JSON mean ± std | Independent native mean ± std | Native − official mean | Native − official std |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        digits = 1 if row["metric"] in {"mae", "rmse"} else 2
        lines.append(f"| {row['metric']} | {row['published_table_s3_mean']:.{digits}f} ± {row['published_table_s3_std']:.{digits}f} | {row['official_json_mean']:.12g} ± {row['official_json_std']:.12g} | {row['independent_native_mean']:.12g} ± {row['independent_native_std']:.12g} | {row['native_minus_official_mean']:.12g} | {row['native_minus_official_std']:.12g} |")
    lines.extend(["", "## Scope boundary", "", "This increment covers SL1/MFP/RF only. It does not validate OHE, BH1, BH2 or SM; in particular, the SM 4620-versus-5760 conflict is neither resolved nor silently selected.", ""])
    return "\n".join(lines)


def publish_summary(repro_root: Path, result: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    """Publish only portable evidence; keep feature cache and fold rows in the run directory."""
    tables_dir = repro_root / "outputs" / "tables"
    reports_dir = repro_root / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "native_sl1_mfp_rf_comparison.csv", comparisons)
    (tables_dir / "native_sl1_mfp_rf_reproduction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "native_sl1_mfp_rf_reproduction.md").write_text(
        report(result, comparisons), encoding="utf-8"
    )


def run(repro_root: Path, output_dir: Path) -> dict[str, Any]:
    source_csv = repro_root / SPEC["csv"]
    official_npz = repro_root / SPEC["official_npz"]
    official_summary = json.loads((repro_root / SPEC["official_summary"]).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_npz = output_dir / "SL1_MFP_generated_from_official_csv.npz"
    generate_mfp(repro_root, source_csv, generated_npz)
    alignment = mfp_alignment(generated_npz, official_npz)
    if not all(alignment[key] for key in ("x_exact", "y_exact", "smiles_columns_exact")):
        raise RuntimeError("Native Gen_MFP output does not exactly match bundled official SL1 MFP")
    with np.load(generated_npz) as generated:
        folds, summary = run_rf_5x5(generated["X"], generated["y"])
    comparisons = metric_rows(summary, official_summary)
    _write_csv(output_dir / "SL1_MFP_RF_fold_metrics.csv", folds)
    _write_csv(output_dir / "SL1_MFP_RF_comparison.csv", comparisons)
    result = {
        "run_type": "independent_paper_native_mfp_rf", "dataset_id": "SL1", "paper_dataset_id": "SLAP",
        "source_csv": SPEC["csv"], "source_csv_sha256": sha256(source_csv),
        "upstream_feature_generator": "yieldsmarter/Src/Featurize/Gen_MFP.py",
        "upstream_training_reference": "yieldsmarter/Src/Train/Train_descriptors.py",
        "upstream_training_config": SPEC["train_config"],
        "implementation_boundary": "RF-only wrapper required because upstream trainer imports unavailable LightGBM before RF branch selection; upstream files were not modified.",
        "actual_versions": package_versions(), "paper_reported_versions": {"rdkit": "2025.03.6", "scikit-learn": "1.6.1"},
        "cv": {"outer_splits": 5, "splits": 5, "seed": 1000, "repeat_seeds": [1000, 1001, 1002, 1003, 1004], "valid_fold_count": 25},
        "rf": {"n_estimators": 500, "max_features": 0.30, "n_jobs": -1},
        "mfp_alignment": alignment, "independent_native_metrics": summary,
        "official_summary_json": SPEC["official_summary"], "comparison_rows": comparisons,
    }
    (output_dir / "SL1_MFP_RF_native_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SL1_MFP_RF_native_reproduction.md").write_text(report(result, comparisons), encoding="utf-8")
    publish_summary(repro_root, result, comparisons)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "outputs" / "runs" / "native_sl1_mfp_rf"
    result = run(root, output_dir)
    print(json.dumps({"dataset": result["dataset_id"], "output_dir": str(output_dir), "metrics": result["independent_native_metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
