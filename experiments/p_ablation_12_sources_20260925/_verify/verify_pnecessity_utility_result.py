"""Read-only completion verifier for one product-utility benchmark task.

The linear launcher calls this verifier only after ``yonod.py`` exits.  It
binds saved predictions, metrics, reports, task state and splits to the
current schema-2 configuration without importing a descriptor builder or
fitting a model.  A non-zero exit prevents the launcher from starting the
next task.
"""

from __future__ import annotations

import argparse
import hashlib
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
from yonod.benchmark.metrics import rebuild_fold_metrics
from yonod.config.loader import load_run_config
from yonod.splits.manifest import validate_split_manifest


TASK_PREFIX = "pnecessity_utility_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _state_counts(path: Path) -> dict[str, int]:
    _require(path.is_file(), f"task-state database is missing: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
        ).fetchall()
    return {str(status): int(count) for status, count in rows}


def _verify_splits(config: BenchmarkConfig, contract: Any, effective: dict[str, Any]) -> dict[str, int]:
    _require(config.split_manifest_path is not None, "source split manifest is not configured")
    _require(config.split_manifest_path.is_file(), "source split manifest is missing")
    local_path = config.outputs_root / "docs" / "manifests" / "split_manifest.parquet"
    _require(local_path.is_file(), f"task-local split manifest is missing: {local_path}")
    local = pd.read_parquet(local_path)
    validate_split_manifest(local, n_splits=config.cv["n_splits"], n_repeats=config.cv["n_repeats"])
    _require(
        set(local["run_id"].astype(str)) == {contract.run_id},
        "task-local split manifest run_id differs from current contract",
    )
    _require(
        set(local["dataset_sha256"].astype(str)) == {contract.dataset_sha256},
        "task-local split manifest dataset hash differs from current contract",
    )

    population = config.validate_dataset().copy()
    population[config.sample_id_col] = population[config.sample_id_col].astype(str)
    _require(not population[config.sample_id_col].duplicated().any(), "prepared sample IDs are not unique")
    grouping = dict(effective["evaluation"]["grouping"])
    group_column = str(grouping["group_column"])
    _require(group_column in population.columns, f"group column is absent: {group_column}")
    groups = population.set_index(config.sample_id_col, verify_integrity=True)[group_column].astype(str)
    population_ids = set(groups.index)

    observed_folds = 0
    for repeat, repeat_part in local.groupby("repeat", sort=True):
        valid_ids: list[str] = []
        for fold, part in repeat_part.groupby("fold", sort=True):
            observed_folds += 1
            train_ids = set(part.loc[part["role"].eq("train"), "sample_id"].astype(str))
            valid_ids_for_fold = set(part.loc[part["role"].eq("valid"), "sample_id"].astype(str))
            _require(not train_ids.intersection(valid_ids_for_fold), f"repeat={repeat}, fold={fold}: sample overlap")
            _require(train_ids.union(valid_ids_for_fold) == population_ids, f"repeat={repeat}, fold={fold}: membership differs from population")
            _require(
                not set(groups.loc[sorted(train_ids)]).intersection(groups.loc[sorted(valid_ids_for_fold)]),
                f"repeat={repeat}, fold={fold}: non-P input group leakage",
            )
            valid_ids.extend(sorted(valid_ids_for_fold))
        _require(
            len(valid_ids) == len(population_ids) and set(valid_ids) == population_ids,
            f"repeat={repeat}: validation membership is not an exact once-only population cover",
        )
    expected_folds = int(config.cv["n_splits"]) * int(config.cv["n_repeats"])
    _require(observed_folds == expected_folds, f"expected {expected_folds} split folds, found {observed_folds}")
    return {"split_folds": observed_folds, "population_rows": len(population)}


def verify(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    _require(config_path.is_file(), f"configuration is missing: {config_path}")
    effective = load_run_config(config_path).effective
    config = BenchmarkConfig.from_file(config_path)
    contract = create_benchmark_contract(config)
    task_name = config.outputs_root.name
    _require(task_name.startswith(TASK_PREFIX), f"not a product-utility task: {task_name}")
    _require(config.outputs_root == ROOT / "result" / task_name, "result root is not isolated")
    _require(config.artifact_output_dir == config.outputs_root / "feature", "feature path is not isolated")
    _require(config.cv == {"n_splits": 5, "n_repeats": 3, "seed": 20260918}, "unexpected CV contract")
    _require(
        tuple(effective["outputs"]["report_formats"]) == ("html", "markdown"),
        "both HTML and Markdown reports are required",
    )

    run_manifest_path = config.outputs_root / "docs" / "manifests" / "run_manifest.json"
    _require(run_manifest_path.is_file(), "run manifest is missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for field, expected in {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "dataset_sha256": contract.dataset_sha256,
        "evaluation_protocol": "manifest_outer_cv",
    }.items():
        _require(run_manifest.get(field) == expected, f"run manifest {field!r} differs from current contract")
    external = dict(run_manifest.get("external_split_manifest") or {})
    _require(config.split_manifest_path is not None, "source split manifest is missing")
    _require(external.get("sha256") == _sha256(config.split_manifest_path), "source split hash differs from run manifest")
    _require(
        Path(str(external.get("source_path", ""))).resolve() == config.split_manifest_path.resolve(),
        "source split path differs from run manifest",
    )

    expected_folds = int(config.cv["n_splits"]) * int(config.cv["n_repeats"])
    statuses = _state_counts(config.outputs_root / "docs" / "state" / "tasks.sqlite")
    _require(statuses == {"succeeded": expected_folds}, f"task-state is incomplete: {statuses}")
    prediction_paths = sorted((config.outputs_root / "docs" / "predictions").glob("*.parquet"))
    fold_paths = sorted((config.outputs_root / "docs" / "folds").glob("*.json"))
    _require(len(prediction_paths) == expected_folds, f"expected {expected_folds} prediction shards")
    _require(len(fold_paths) == expected_folds, f"expected {expected_folds} fold metadata files")
    for report_name in ("benchmark_report.html", "benchmark_report.md"):
        report = config.outputs_root / "report" / report_name
        _require(report.is_file() and report.stat().st_size > 0, f"report is missing or empty: {report_name}")

    split_summary = _verify_splits(config, contract, effective)
    rebuilt = rebuild_fold_metrics(contract.run_dir)
    _require(len(rebuilt.fold_metrics) == expected_folds, "metric rebuild does not contain all folds")
    _require(rebuilt.exclusions.empty, f"metric rebuild exclusions are present: {len(rebuilt.exclusions)}")
    _require(not rebuilt.completeness.empty and bool(rebuilt.completeness["is_complete"].all()), "metric completeness failed")
    return {
        "status": "passed",
        "task_name": task_name,
        "config_path": str(config_path.relative_to(ROOT)),
        "run_id": contract.run_id,
        "dataset_sha256": contract.dataset_sha256,
        "succeeded_folds": expected_folds,
        "prediction_shards": len(prediction_paths),
        "fold_metadata": len(fold_paths),
        **split_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only product-utility task completion verifier.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
