"""Read-only completion verifier for one numeric-condition paired task."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.layout import resolve_benchmark_output_layout
from yonod.features.numeric_conditions import numeric_input_identity
from yonod.splits.grouping import build_group_ids
from yonod.splits.manifest import validate_split_manifest


PREFIX = "pnecessity_numeric_conditions_"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _state_counts(path: Path) -> dict[str, int]:
    _require(path.is_file(), f"task-state database is missing: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status").fetchall()
    return {str(status): int(count) for status, count in rows}


def verify(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = BenchmarkConfig.from_file(config_path)
    contract = create_benchmark_contract(config)
    task_name = config.outputs_root.name
    _require(task_name.startswith(PREFIX), f"not a numeric-condition product-utility task: {task_name}")
    _require(config.outputs_root == ROOT / "result" / task_name, "result root is not isolated")
    _require(config.artifact_output_dir == config.outputs_root / "feature", "feature root is not isolated")
    _require(config.cv == {"n_splits": 5, "n_repeats": 3, "seed": 20260918}, "unexpected CV contract")
    numeric_contract = config.raw["numeric_contract"]
    _require(bool(numeric_contract["columns"]), "task lacks declared numeric conditions")
    frame = config.validate_dataset()
    numeric_columns = [entry["source"] for entry in numeric_contract["columns"]]
    expected_numeric_identity = numeric_input_identity(frame.loc[:, numeric_columns], frame[config.sample_id_col].astype(str).tolist(), numeric_contract)

    layout = resolve_benchmark_output_layout(contract.run_dir)
    run_manifest_path = layout.manifests / "run_manifest.json"
    _require(run_manifest_path.is_file(), "run manifest is missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for field, expected in {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "dataset_sha256": contract.dataset_sha256,
        "evaluation_protocol": "manifest_outer_cv",
    }.items():
        _require(run_manifest.get(field) == expected, f"run manifest {field!r} differs from current contract")

    split_path = layout.manifests / "split_manifest.parquet"
    _require(split_path.is_file(), "task-local split manifest is missing")
    split = pd.read_parquet(split_path)
    validate_split_manifest(split, n_splits=5, n_repeats=3)
    _require(set(split["run_id"].astype(str)) == {contract.run_id}, "split run ID differs from current contract")
    _require(set(split["dataset_sha256"].astype(str)) == {contract.dataset_sha256}, "split dataset hash differs from current contract")
    expected_groups = build_group_ids(frame, config.sample_id_col, config.grouping, config.smiles_cols)
    by_sample = dict(zip(frame[config.sample_id_col].astype(str), expected_groups.astype(str)))
    observed_groups = split[["sample_id", "group_id"]].drop_duplicates()
    _require(len(observed_groups) == len(by_sample), "split has duplicate or missing sample/group assignments")
    _require(all(by_sample.get(str(row.sample_id)) == str(row.group_id) for row in observed_groups.itertuples(index=False)), "split group assignments differ from non-P molecular contract")

    expected_folds = 15
    statuses = _state_counts(layout.state / "tasks.sqlite")
    _require(statuses == {"succeeded": expected_folds}, f"task-state is incomplete: {statuses}")
    prediction_paths = sorted(layout.predictions.glob("*.parquet"))
    fold_paths = sorted(layout.folds.glob("*.json"))
    _require(len(prediction_paths) == expected_folds and len(fold_paths) == expected_folds, "prediction/fold evidence is incomplete")
    for path in fold_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        numeric = ((metadata.get("feature_transformer") or {}).get("numeric_conditions") or {})
        _require(numeric.get("dimension") == len(numeric_columns), f"{path.name}: numeric feature dimension differs")
        _require(numeric.get("input_identity") == expected_numeric_identity, f"{path.name}: numeric input identity differs")
        state = numeric.get("state") or {}
        _require(state.get("contract") == numeric_contract, f"{path.name}: numeric contract differs")
        _require(int(state.get("fit_rows", 0)) == int(metadata.get("n_train", 0)), f"{path.name}: numeric transformer was not fit solely on the outer training rows")
    completeness_path = layout.metrics / "completeness.parquet"
    _require(completeness_path.is_file(), "metric completeness table is missing")
    completeness = pd.read_parquet(completeness_path)
    _require(not completeness.empty and bool(completeness["is_complete"].all()), "metric completeness failed")
    for report_name in ("benchmark_report.html", "benchmark_report.md"):
        report = layout.report / report_name
        _require(report.is_file() and report.stat().st_size > 0, f"report is missing or empty: {report_name}")
    return {
        "status": "passed",
        "task_name": task_name,
        "run_id": contract.run_id,
        "succeeded_folds": expected_folds,
        "numeric_columns": numeric_columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only numeric-condition product-utility completion verifier.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
