"""Strict read-only verifier for one ChemRxiv grouped unseen-A formal task.

The verifier deliberately treats a completed task as more than a SQLite
``succeeded`` count. It binds the run to its schema-2 configuration, the
immutable 957-row prepared population, and the immutable 5-fold x 3-repeat
source split plan; it then checks every out-of-fold prediction shard against
the held-out membership. It neither fits models nor writes task outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.splits.manifest import load_external_split_manifest


EXPECTED_TASK_PREFIX = "ablation2_chemrxiv_unseen_amine_"
EXPECTED_TOTAL_FOLDS = 15
EXPECTED_PAIRS = {(repeat, fold) for repeat in range(1, 4) for fold in range(1, 6)}
REQUIRED_REPORTS = ("benchmark_report.html", "benchmark_report.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values_hash(values: Iterable[Any]) -> str:
    payload = [str(value) for value in values]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fail(message: str) -> None:
    raise AssertionError(message)


def _task_statuses(database: Path) -> dict[str, int]:
    if not database.is_file():
        _fail(f"task-state SQLite database is missing: {database}")
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
            ).fetchall()
    except sqlite3.Error as exc:
        _fail(f"cannot read task-state SQLite database {database}: {exc}")
    return {str(status): int(count) for status, count in rows}


def _membership(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split_id", "sample_id", "group_id", "group_strategy", "repeat", "fold", "role",
        "seed", "source_row_index", "dataset_sha256", "grouping_params_json", "population_id",
        "split_hash",
    ]
    missing = set(columns).difference(frame.columns)
    if missing:
        _fail(f"split manifest is missing columns: {sorted(missing)}")
    return frame.loc[:, columns].sort_values(
        ["repeat", "fold", "role", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)


def _expected_ids(population: pd.DataFrame, ids: set[str]) -> list[str]:
    # The executor subselects the validated population in original row order.
    return population.loc[population["sample_id"].astype(str).isin(ids), "sample_id"].astype(str).tolist()


def _metric_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(float(observed), float(expected), rtol=1e-10, atol=1e-12, equal_nan=True):
        _fail(f"metric mismatch for {label}: observed={observed}, recomputed={expected}")


def verify(config_path: Path) -> dict[str, Any]:
    """Verify the completed 5-fold x 3-repeat task at ``config_path``."""
    if not config_path.is_file():
        _fail(f"configuration is missing: {config_path}")
    config = BenchmarkConfig.from_file(config_path)
    contract = create_benchmark_contract(config)
    task_name = config.outputs_root.name
    if not task_name.startswith(EXPECTED_TASK_PREFIX):
        _fail(f"configuration is not an unseen-A formal task: {config_path}")
    if config.outputs_root != ROOT / "result" / task_name:
        _fail(f"{task_name}: result root is not the required isolated formal-task directory")
    if config.cv != {"n_splits": 5, "n_repeats": 3, "seed": 20260918}:
        _fail(f"{task_name}: configuration does not have the required 5 x 3 split contract")
    if config.outputs_root / "feature" != config.artifact_output_dir:
        _fail(f"{task_name}: feature output is not isolated below the task result root")
    if config.split_manifest_path is None or not config.split_manifest_path.is_file():
        _fail(f"{task_name}: immutable source split manifest is missing")

    reports = config.outputs_root / "report"
    missing_reports = [
        name for name in REQUIRED_REPORTS
        if not (reports / name).is_file() or (reports / name).stat().st_size == 0
    ]
    if missing_reports:
        _fail(f"{task_name}: required reports are missing or empty: {', '.join(missing_reports)}")

    statuses = _task_statuses(config.outputs_root / "docs" / "state" / "tasks.sqlite")
    total = sum(statuses.values())
    succeeded = statuses.get("succeeded", 0)
    failed = statuses.get("failed", 0)
    non_success = {name: count for name, count in statuses.items() if name != "succeeded" and count}
    if total != EXPECTED_TOTAL_FOLDS or succeeded != EXPECTED_TOTAL_FOLDS or failed != 0 or non_success:
        _fail(
            f"{task_name}: incomplete task-state counts total={total}, succeeded={succeeded}, "
            f"failed={failed}, all={statuses}"
        )

    run_manifest_path = config.outputs_root / "docs" / "manifests" / "run_manifest.json"
    if not run_manifest_path.is_file():
        _fail(f"{task_name}: run manifest is missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for field, expected in {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "dataset_sha256": contract.dataset_sha256,
        "population_id": "ablation2_chemrxiv_same_source_unseen_amine_v1",
        "evaluation_protocol": "manifest_outer_cv",
    }.items():
        if run_manifest.get(field) != expected:
            _fail(f"{task_name}: run manifest {field!r} does not bind to current contract")
    external = run_manifest.get("external_split_manifest", {})
    manifest_sha = _sha256(config.split_manifest_path)
    if external.get("sha256") != manifest_sha or Path(str(external.get("source_path", ""))).resolve() != config.split_manifest_path.resolve():
        _fail(f"{task_name}: run manifest source split identity differs from configuration")

    population = config.validate_dataset().copy()
    population["sample_id"] = population["sample_id"].astype(str)
    population_index = population.set_index("sample_id", verify_integrity=True)
    if len(population) != 957 or population["amine_structure_key"].astype(str).nunique() != 10:
        _fail(f"{task_name}: expected 957 rows and 10 A/amine groups")
    imported = load_external_split_manifest(
        config.split_manifest_path,
        sample_ids=population["sample_id"].tolist(),
        dataset_sha256=contract.dataset_sha256,
        n_splits=5,
        n_repeats=3,
        run_id=contract.run_id,
    )
    local_manifest_path = config.outputs_root / "docs" / "manifests" / "split_manifest.parquet"
    if not local_manifest_path.is_file():
        _fail(f"{task_name}: task-local split manifest is missing")
    local_manifest = pd.read_parquet(local_manifest_path)
    if not _membership(local_manifest).equals(_membership(imported)):
        _fail(f"{task_name}: task-local split membership differs from immutable source plan")
    if len(local_manifest) != 957 * 3 * 5:
        _fail(f"{task_name}: task-local split manifest has wrong row count")

    expected_folds: dict[tuple[int, int], pd.DataFrame] = {}
    zero_overlap = True
    for (repeat, fold), part in imported.groupby(["repeat", "fold"], sort=True):
        key = (int(repeat), int(fold))
        expected_folds[key] = part.copy()
        train_ids = set(part.loc[part["role"].eq("train"), "sample_id"].astype(str))
        valid_ids = set(part.loc[part["role"].eq("valid"), "sample_id"].astype(str))
        if train_ids.intersection(valid_ids) or len(train_ids) + len(valid_ids) != len(population):
            _fail(f"{task_name}: invalid sample membership in repeat={repeat}, fold={fold}")
        train_groups = set(population_index.loc[sorted(train_ids), "amine_structure_key"].astype(str))
        valid_groups = set(population_index.loc[sorted(valid_ids), "amine_structure_key"].astype(str))
        if train_groups.intersection(valid_groups):
            zero_overlap = False
            _fail(f"{task_name}: A/amine overlap in repeat={repeat}, fold={fold}")
        if len(train_groups) != 8 or len(valid_groups) != 2:
            _fail(f"{task_name}: incorrect held-out A-group count in repeat={repeat}, fold={fold}")
    if set(expected_folds) != EXPECTED_PAIRS:
        _fail(f"{task_name}: source split plan does not contain 15 expected folds")
    for repeat in range(1, 4):
        valid = imported.loc[(imported["repeat"] == repeat) & imported["role"].eq("valid")]
        if valid["sample_id"].astype(str).nunique() != 957 or valid["sample_id"].duplicated().any():
            _fail(f"{task_name}: repeat {repeat} does not cover every sample exactly once")

    descriptor = config.descriptors[0]
    model = config.models[0]
    prediction_dir = config.outputs_root / "docs" / "predictions"
    fold_dir = config.outputs_root / "docs" / "folds"
    prediction_paths = sorted(prediction_dir.glob("*.parquet"))
    metadata_paths = sorted(fold_dir.glob("*.json"))
    if len(prediction_paths) != EXPECTED_TOTAL_FOLDS or len(metadata_paths) != EXPECTED_TOTAL_FOLDS:
        _fail(f"{task_name}: expected 15 prediction and 15 fold-metadata shards")

    shards: list[pd.DataFrame] = []
    seen_pairs: set[tuple[int, int]] = set()
    per_fold_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for prediction_path in prediction_paths:
        shard = pd.read_parquet(prediction_path)
        required = {
            "run_id", "config_hash", "split_id", "sample_id", "group_id", "descriptor", "model",
            "repeat", "fold", "y_true", "y_pred", "feature_schema_hash",
        }
        missing = required.difference(shard.columns)
        if missing:
            _fail(f"{task_name}: prediction shard {prediction_path.name} misses {sorted(missing)}")
        pairs = shard.loc[:, ["repeat", "fold"]].drop_duplicates()
        if len(pairs) != 1:
            _fail(f"{task_name}: prediction shard {prediction_path.name} has multiple fold identities")
        repeat, fold = int(pairs.iloc[0]["repeat"]), int(pairs.iloc[0]["fold"])
        key = (repeat, fold)
        if key not in EXPECTED_PAIRS or key in seen_pairs:
            _fail(f"{task_name}: duplicate or unexpected prediction fold {key}")
        seen_pairs.add(key)
        suffix = f"{descriptor}__{model}__r{repeat:02d}__f{fold:02d}"
        metadata_path = fold_dir / f"{suffix}.json"
        if metadata_path not in metadata_paths:
            _fail(f"{task_name}: prediction {prediction_path.name} has no matching metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = expected_folds[key]
        valid_set = set(expected.loc[expected["role"].eq("valid"), "sample_id"].astype(str))
        train_set = set(expected.loc[expected["role"].eq("train"), "sample_id"].astype(str))
        expected_valid_ids = _expected_ids(population, valid_set)
        expected_train_ids = _expected_ids(population, train_set)
        shard = shard.copy()
        shard["sample_id"] = shard["sample_id"].astype(str)
        if len(shard) != len(valid_set) or shard["sample_id"].duplicated().any() or set(shard["sample_id"]) != valid_set:
            _fail(f"{task_name}: prediction membership differs from source plan in repeat={repeat}, fold={fold}")
        if set(shard["run_id"].astype(str)) != {contract.run_id} or set(shard["config_hash"].astype(str)) != {contract.config_hash}:
            _fail(f"{task_name}: prediction run/config identity mismatch in repeat={repeat}, fold={fold}")
        if set(shard["descriptor"].astype(str)) != {descriptor} or set(shard["model"].astype(str)) != {model}:
            _fail(f"{task_name}: prediction descriptor/model mismatch in repeat={repeat}, fold={fold}")
        if not np.isfinite(shard[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            _fail(f"{task_name}: non-finite OOF prediction in repeat={repeat}, fold={fold}")
        observed_labels = shard.set_index("sample_id")["y_true"].astype(float)
        expected_labels = population_index.loc[observed_labels.index, config.label_col].astype(float)
        if not np.allclose(observed_labels.to_numpy(), expected_labels.to_numpy(), rtol=0.0, atol=1e-12):
            _fail(f"{task_name}: OOF labels differ from immutable population in repeat={repeat}, fold={fold}")
        expected_groups = expected.loc[expected["role"].eq("valid")].set_index("sample_id")["group_id"].astype(str)
        if not shard.set_index("sample_id")["group_id"].astype(str).equals(expected_groups.loc[shard["sample_id"]]):
            _fail(f"{task_name}: OOF group IDs differ from source plan in repeat={repeat}, fold={fold}")
        for field, expected_value in {
            "run_id": contract.run_id,
            "config_hash": contract.config_hash,
            "descriptor": descriptor,
            "model": model,
            "evaluation_protocol": "manifest_outer_cv",
            "repeat": repeat,
            "fold": fold,
            "n_train": len(train_set),
            "n_valid": len(valid_set),
            "split_id": str(expected["split_id"].iloc[0]),
            "split_hash": str(expected["split_hash"].iloc[0]),
        }.items():
            if metadata.get(field) != expected_value:
                _fail(f"{task_name}: metadata {field!r} mismatch in repeat={repeat}, fold={fold}")
        if metadata.get("train_sample_ids_hash") != _values_hash(expected_train_ids):
            _fail(f"{task_name}: train sample hash mismatch in repeat={repeat}, fold={fold}")
        if metadata.get("valid_sample_ids_hash") != _values_hash(expected_valid_ids):
            _fail(f"{task_name}: valid sample hash mismatch in repeat={repeat}, fold={fold}")
        # ``source_row_index`` here is the immutable manifest's population-row
        # position, rather than the ChemRxiv provenance column of the same name.
        manifest_rows = expected.set_index("sample_id")["source_row_index"]
        expected_source_rows = manifest_rows.loc[expected_valid_ids].astype(int).tolist()
        if metadata.get("valid_source_row_index_hash") != _values_hash(expected_source_rows):
            _fail(f"{task_name}: valid source-row hash mismatch in repeat={repeat}, fold={fold}")
        if not isinstance(metadata.get("feature_schema_hash"), str) or not metadata["feature_schema_hash"]:
            _fail(f"{task_name}: feature-schema evidence is missing in repeat={repeat}, fold={fold}")
        abs_error = np.abs(shard["y_true"].to_numpy(dtype=float) - shard["y_pred"].to_numpy(dtype=float))
        per_fold_metrics[key] = {
            "mae": float(abs_error.mean()),
            "rmse": float(mean_squared_error(shard["y_true"], shard["y_pred"]) ** 0.5),
            "r2": float(r2_score(shard["y_true"], shard["y_pred"])),
        }
        shards.append(shard)
    if seen_pairs != EXPECTED_PAIRS:
        _fail(f"{task_name}: OOF prediction shards do not cover all 15 folds")

    oof = pd.concat(shards, ignore_index=True)
    if len(oof) != 3 * len(population) or oof.duplicated(["sample_id", "repeat"]).any():
        _fail(f"{task_name}: OOF predictions are not exactly 3 x 957 unique sample/repeat rows")
    if oof["group_id"].astype(str).nunique() != 10:
        _fail(f"{task_name}: OOF prediction groups do not retain 10 amines")
    for repeat in range(1, 4):
        repeat_ids = set(oof.loc[oof["repeat"].eq(repeat), "sample_id"].astype(str))
        if repeat_ids != set(population["sample_id"]):
            _fail(f"{task_name}: repeat {repeat} OOF coverage differs from 957-row population")

    metrics_path = config.outputs_root / "docs" / "metrics" / "fold_metrics.parquet"
    if not metrics_path.is_file():
        _fail(f"{task_name}: fold metric table is missing")
    metrics = pd.read_parquet(metrics_path)
    metric_pairs = set(map(tuple, metrics[["repeat", "fold"]].astype(int).to_numpy()))
    if len(metrics) != 15 or metric_pairs != EXPECTED_PAIRS:
        _fail(f"{task_name}: fold metric table does not cover all 15 folds")
    for row in metrics.itertuples(index=False):
        key = (int(row.repeat), int(row.fold))
        if str(row.run_id) != contract.run_id or str(row.config_hash) != contract.config_hash:
            _fail(f"{task_name}: fold metric identity mismatch for {key}")
        for metric, expected_value in per_fold_metrics[key].items():
            _metric_close(getattr(row, metric), expected_value, f"{task_name}/{key}/{metric}")

    return {
        "status": "passed",
        "task": task_name,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "result_root": str(config.outputs_root.relative_to(ROOT)),
        "hashes": {
            "config_file_sha256": _sha256(config_path),
            "config_contract_sha256": contract.config_hash,
            "dataset_sha256": contract.dataset_sha256,
            "source_split_manifest_sha256": manifest_sha,
            "task_local_split_manifest_sha256": _sha256(local_manifest_path),
        },
        "task_state": {"total": total, "succeeded": succeeded, "failed": failed, "statuses": statuses},
        "oof": {
            "n_rows": int(len(oof)),
            "n_unique_sample_repeat": int(oof[["sample_id", "repeat"]].drop_duplicates().shape[0]),
            "n_samples_per_repeat": int(len(population)),
            "n_repeats": 3,
            "n_amine_groups": 10,
            "zero_train_valid_amine_overlap_all_15_folds": zero_overlap,
        },
        "reports": [str((reports / name).relative_to(ROOT)) for name in REQUIRED_REPORTS],
        "writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="schema-2 task configuration")
    args = parser.parse_args()
    print(json.dumps(verify(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
