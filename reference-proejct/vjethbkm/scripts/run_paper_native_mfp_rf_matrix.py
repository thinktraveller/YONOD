"""Run all four paper-native MFP + RF reproductions without modifying YONOD.

This invokes the bundled MFP generator afresh for each official CSV, checks it
against the bundled MFP artifact, then mirrors the RF-only branch of the
bundled trainer.  The trainer itself cannot import locally because it imports
unavailable LightGBM before choosing the RF branch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from run_paper_native_mfp_rf import METRICS, _write_csv, mfp_alignment, package_versions, run_rf_5x5, sha256


MFP_SPECS: dict[str, dict[str, Any]] = {
    "BH1": {
        "paper_dataset_id": "BH", "csv": "yieldsmarter/Data/HTE_datasets/BH1/BH1.csv",
        "official_npz": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP.npz",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP_metrics_summaries_RF.json",
        "train_config": "yieldsmarter/Src/Train/BH1.json", "unit": "yield %",
        "table_s3": {"mae": (4.5, 0.2), "rmse": (6.9, 0.5), "r2": (0.94, 0.01), "kendall": (0.85, 0.01)},
    },
    "BH2": {
        "paper_dataset_id": "BH2", "csv": "yieldsmarter/Data/HTE_datasets/BH2/BH2.csv",
        "official_npz": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/MFP/Denmark_MFP.npz",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/MFP/Denmark_MFP_metrics_summaries_RF.json",
        "train_config": "yieldsmarter/Src/Train/BH2.json", "unit": "yield %",
        "table_s3": {"mae": (12.1, 0.5), "rmse": (17.9, 0.8), "r2": (0.73, 0.02), "kendall": (0.68, 0.01)},
    },
    "SL1": {
        "paper_dataset_id": "SLAP", "csv": "yieldsmarter/Data/HTE_datasets/SL1/SL1.csv",
        "official_npz": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP.npz",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP_metrics_summaries_RF.json",
        "train_config": "yieldsmarter/Src/Train/SL1.json", "unit": "LC-MS product ratio",
        "table_s3": {"mae": (18.0, 1.8), "rmse": (35.1, 5.1), "r2": (0.77, 0.05), "kendall": (0.51, 0.04)},
    },
    "SM": {
        "paper_dataset_id": "SM", "csv": "yieldsmarter/Data/HTE_datasets/SM/SM.csv",
        "official_npz": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP.npz",
        "official_summary": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP_metrics_summaries_RF.json",
        "train_config": "yieldsmarter/Src/Train/SM.json", "unit": "yield %", "skip_rows_with_missing_values": True,
        "table_s3": {"mae": (7.4, 0.2), "rmse": (11.0, 0.3), "r2": (0.85, 0.01), "kendall": (0.76, 0.01)},
    },
}


def generate_mfp(repro_root: Path, source_csv: Path, output_npz: Path, skip_rows_with_missing_values: bool) -> None:
    generator = repro_root / "yieldsmarter" / "Src" / "Featurize" / "Gen_MFP.py"
    output_npz = output_npz.resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(generator), str(source_csv.resolve()), str(output_npz)]
    if skip_rows_with_missing_values:
        command.append("--skip_rows_with_missing_values")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=generator.parent, check=True, env=environment)


def comparison_rows(dataset_id: str, spec: dict[str, Any], local: dict[str, dict[str, float]], official: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        printed_mean, printed_std = spec["table_s3"][metric]
        official_item = official["test"][metric]
        digits = 1 if metric in {"mae", "rmse"} else 2
        rows.append({
            "dataset_id": dataset_id, "paper_dataset_id": spec["paper_dataset_id"], "descriptor": "MFP", "model": "RF",
            "metric": "kendall_tau" if metric == "kendall" else metric, "unit": spec["unit"],
            "published_table_s3_mean": printed_mean, "published_table_s3_std": printed_std,
            "official_json_mean": official_item["mean"], "official_json_std": official_item["std"],
            "independent_native_mean": local[metric]["mean"], "independent_native_std": local[metric]["std"],
            "native_minus_official_mean": local[metric]["mean"] - official_item["mean"],
            "native_minus_official_std": local[metric]["std"] - official_item["std"],
            "official_json_matches_table_s3_display": round(float(official_item["mean"]), digits) == printed_mean and round(float(official_item["std"]), digits) == printed_std,
            "valid_fold_count": 25,
        })
    return rows


def run_dataset(repro_root: Path, output_dir: Path, dataset_id: str) -> dict[str, Any]:
    spec = MFP_SPECS[dataset_id]
    source_csv = repro_root / spec["csv"]
    official_npz = repro_root / spec["official_npz"]
    official = json.loads((repro_root / spec["official_summary"]).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_npz = output_dir / f"{dataset_id}_MFP_generated_from_official_csv.npz"
    skip_missing = bool(spec.get("skip_rows_with_missing_values", False))
    generate_mfp(repro_root, source_csv, generated_npz, skip_missing)
    alignment = mfp_alignment(generated_npz, official_npz)
    if not all(alignment[key] for key in ("x_exact", "y_exact", "smiles_columns_exact")):
        raise RuntimeError(f"Native {dataset_id} MFP does not exactly match the bundled artifact: {alignment}")
    with np.load(generated_npz) as generated:
        folds, summary = run_rf_5x5(generated["X"], generated["y"])
    rows = comparison_rows(dataset_id, spec, summary, official)
    _write_csv(output_dir / f"{dataset_id}_MFP_RF_fold_metrics.csv", folds)
    _write_csv(output_dir / f"{dataset_id}_MFP_RF_comparison.csv", rows)
    return {
        "dataset_id": dataset_id, "paper_dataset_id": spec["paper_dataset_id"], "source_csv": spec["csv"],
        "source_csv_sha256": sha256(source_csv), "mfp_skip_rows_with_missing_values": skip_missing,
        "official_summary_json": spec["official_summary"], "upstream_training_config": spec["train_config"],
        "mfp_alignment": alignment, "independent_native_metrics": summary, "comparison_rows": rows,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Four-dataset MFP + RF independent paper-native rerun", "",
        "Each MFP is regenerated from the paper's official CSV using unmodified `Gen_MFP.py`; no bundled NPZ is used for training. The RF-only wrapper mirrors the parameters, seeds, KFold schedule and metrics of `Train_descriptors.py`, whose unmodified module cannot import locally because it imports unavailable LightGBM before RF selection. YONOD is not used or modified.", "",
        "- SI versions: RDKit 2025.03.6, scikit-learn 1.6.1. Actual versions are recorded in the JSON, so this differing-stack result is not claimed to be bit-identical across environments.",
        "- SM uniquely passes the bundled `--skip_rows_with_missing_values` flag: the original 5760 CSV rows become the SI-described 4620 complete-component rows. This is an explicit MFP population choice, not an OHE result.", "",
        "| Dataset | Metric | Official JSON mean ± std | Independent native mean ± std | Δ mean | Δ std | MFP X/y/columns exact |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in result["datasets"]:
        alignment = item["mfp_alignment"]
        for row in item["comparison_rows"]:
            lines.append(f"| {item['dataset_id']} | {row['metric']} | {row['official_json_mean']:.12g} ± {row['official_json_std']:.12g} | {row['independent_native_mean']:.12g} ± {row['independent_native_std']:.12g} | {row['native_minus_official_mean']:.12g} | {row['native_minus_official_std']:.12g} | {alignment['x_exact']}/{alignment['y_exact']}/{alignment['smiles_columns_exact']} |")
    return "\n".join(lines) + "\n"


def run(repro_root: Path, output_dir: Path, datasets: list[str]) -> dict[str, Any]:
    dataset_results = [run_dataset(repro_root, output_dir, dataset_id) for dataset_id in datasets]
    rows = [row for item in dataset_results for row in item["comparison_rows"]]
    result = {
        "run_type": "independent_paper_native_mfp_rf_matrix", "datasets": dataset_results,
        "actual_versions": package_versions(), "paper_reported_versions": {"rdkit": "2025.03.6", "scikit-learn": "1.6.1"},
        "cv": {"outer_splits": 5, "splits": 5, "seed": 1000, "repeat_seeds": [1000, 1001, 1002, 1003, 1004], "valid_fold_count": 25},
        "rf": {"n_estimators": 500, "max_features": 0.30, "n_jobs": -1},
        "implementation_boundary": "RF-only wrapper; upstream files are unmodified and their unconditional LightGBM import is unavailable locally.",
    }
    _write_csv(output_dir / "MFP_RF_matrix_comparison.csv", rows)
    (output_dir / "MFP_RF_matrix_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "MFP_RF_matrix_reproduction.md").write_text(markdown(result), encoding="utf-8")
    tables_dir = repro_root / "outputs" / "tables"
    reports_dir = repro_root / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "native_mfp_rf_matrix_comparison.csv", rows)
    (tables_dir / "native_mfp_rf_matrix_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "native_mfp_rf_matrix_reproduction.md").write_text(markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(MFP_SPECS), default=sorted(MFP_SPECS))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "outputs" / "runs" / "native_mfp_rf_matrix"
    result = run(root, output_dir, args.datasets)
    print(json.dumps({"datasets": args.datasets, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
