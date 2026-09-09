#!/usr/bin/env python3
"""Prepare, verify, and execute the recoverable step 28.3 static matrix.

The plan is derived only from the frozen step 28.1 inventory and step 28.2
population/split manifests.  Execution is deliberately selector-driven and
serial; invoking the script without --task-id or a filter never starts fits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Set these before importing numpy/scipy/sklearn through the smoke module.
THREAD_ENV = {
    "LOKY_MAX_CPU_COUNT": "19",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _thread_name, _thread_value in THREAD_ENV.items():
    os.environ[_thread_name] = _thread_value

import run_static_descriptor_model_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
MAX_CPUS = 19
MODEL_ORDER = ("rf", "xgboost", "svm", "lightgbm")
DESCRIPTOR_ORDER = ("DFT", "SOAP", "PhysChem", "MFP")
FORBIDDEN_MODELS = {"autogluon"}
FORBIDDEN_DESCRIPTORS = {"ohe"}
STEP28_1_INVENTORY = Path(
    "derived/descriptor_model_effect/step28_1_input_audit/"
    "official_static_descriptor_inventory.json"
)
STEP28_2_DIR = Path("derived/descriptor_model_effect/step28_2_population_splits")
DEFAULT_PLAN_DIR = Path("derived/descriptor_model_effect/step28_3_matrix_plan")
PLAN_JSON_NAME = "task_plan.json"
PLAN_CSV_NAME = "task_plan.csv"

TASK_FIELDS = (
    "schema_version",
    "task_id",
    "status",
    "population_id",
    "dataset_id",
    "descriptor",
    "model",
    "repeat_zero_based",
    "fold_zero_based",
    "manifest_repeat",
    "manifest_fold",
    "seed",
    "n_samples",
    "n_features",
    "n_train",
    "n_valid",
    "source_npz",
    "source_npz_sha256",
    "input_hash",
    "x_key",
    "y_key",
    "feature_hash",
    "y_hash",
    "population_hash",
    "mapping_hash",
    "split_hash",
    "fold_hash",
    "model_config_hash",
    "population_manifest_path",
    "population_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "train_sample_ids_hash",
    "valid_sample_ids_hash",
    "train_source_row_index_hash",
    "valid_source_row_index_hash",
    "train_y_hash",
    "valid_y_hash",
    "output_path",
    "task_contract_hash",
)

CONTRACT_FIELDS = tuple(
    field for field in TASK_FIELDS if field not in {"task_id", "status", "task_contract_hash"}
)

PREDICTION_FIELDS = (
    "prediction_index",
    "task_id",
    "population_id",
    "dataset_id",
    "descriptor",
    "model",
    "repeat_zero_based",
    "fold_zero_based",
    "manifest_repeat",
    "manifest_fold",
    "seed",
    "population_position",
    "sample_id",
    "source_row_index",
    "y_true",
    "y_pred",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_path(value: str | Path) -> Path:
    path = (PROJECT_ROOT / Path(value)).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {value}") from exc
    return path


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _model_config(
    model_id: str,
    seed: int,
    n_features: int,
    n_train: int,
) -> tuple[Any, dict[str, Any]]:
    """Reuse the already accepted step28.3 smoke model factory verbatim."""
    return smoke._build_model(model_id, seed=seed, n_features=n_features, n_train=n_train)


def _task_contract(task: Mapping[str, Any]) -> dict[str, Any]:
    return {field: task[field] for field in CONTRACT_FIELDS}


def _task_id(task: Mapping[str, Any], contract_hash: str) -> str:
    return (
        f"s28-{str(task['dataset_id']).lower()}-{str(task['descriptor']).lower()}-"
        f"{task['model']}-r{int(task['repeat_zero_based']):02d}-"
        f"f{int(task['fold_zero_based']):02d}-{contract_hash[:12]}"
    )


def _plan_paths(plan_dir: Path) -> tuple[Path, Path]:
    return plan_dir / PLAN_JSON_NAME, plan_dir / PLAN_CSV_NAME


def _inventory_map() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(_repo_path(STEP28_1_INVENTORY).read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    result = {(str(row["dataset_id"]), str(row["descriptor"])): dict(row) for row in entries}
    if len(result) != 16:
        raise AssertionError(f"step28.1 inventory must contain 16 unique matrices, got {len(result)}")
    return result


def _included_descriptor_rows() -> list[dict[str, str]]:
    rows = _read_csv(_repo_path(STEP28_2_DIR / "descriptor_availability.csv"))
    included = [
        row
        for row in rows
        if row.get("status") == "included" and _bool(row.get("main_matrix_inclusion"))
    ]
    if len(included) != 15:
        raise AssertionError(f"expected 15 included static matrices, got {len(included)}")
    for row in included:
        if row["descriptor"] not in DESCRIPTOR_ORDER:
            raise AssertionError(f"forbidden/unknown descriptor included: {row['descriptor']}")
        if row["dataset_id"] == "BH1" and row["descriptor"] == "DFT":
            raise AssertionError("BH1 DFT is still blocked and must not enter the plan")
        if row["dataset_id"] == "SM" and row["population_id"] != "SM-static-4620":
            raise AssertionError("SM main matrix must use only SM-static-4620")
    return included


def _source_file_record(path: Path) -> dict[str, str]:
    return {"path": _repo_relative(path), "sha256": smoke._sha256_file(path)}


def _prepare_payload(plan_dir: Path) -> dict[str, Any]:
    smoke._require_yonod_conda_env()
    inventory = _inventory_map()
    descriptor_rows = _included_descriptor_rows()
    fold_manifest_path = _repo_path(STEP28_2_DIR / "fold_manifest.csv")
    fold_rows = _read_csv(fold_manifest_path)
    folds_by_population: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fold_rows:
        folds_by_population[row["population_id"]].append(row)

    model_configs: dict[str, dict[str, Any]] = {}
    model_cache: dict[tuple[str, int, int, int], tuple[str, dict[str, Any]]] = {}
    file_hash_cache: dict[Path, str] = {}
    population_first_cache: dict[str, dict[str, str]] = {}
    tasks: list[dict[str, Any]] = []

    population_rank = {"BH1": 0, "BH2": 1, "SL1": 2, "SM": 3}
    descriptor_rank = {name: index for index, name in enumerate(DESCRIPTOR_ORDER)}
    descriptor_rows.sort(
        key=lambda row: (population_rank[row["dataset_id"]], descriptor_rank[row["descriptor"]])
    )

    for descriptor_row in descriptor_rows:
        population_id = descriptor_row["population_id"]
        dataset_id = descriptor_row["dataset_id"]
        descriptor = descriptor_row["descriptor"]
        inventory_entry = inventory[(dataset_id, descriptor)]
        source_npz = _repo_path(descriptor_row["source_npz"])
        population_manifest = _repo_path(
            STEP28_2_DIR / "population_manifests" / f"{population_id}_population_manifest.csv"
        )
        split_manifest = _repo_path(
            STEP28_2_DIR / "split_manifests" / f"{population_id}_split_manifest.csv"
        )
        for input_path in (source_npz, population_manifest, split_manifest):
            if input_path not in file_hash_cache:
                file_hash_cache[input_path] = smoke._sha256_file(input_path)
        if file_hash_cache[source_npz] != descriptor_row["source_npz_sha256"]:
            raise AssertionError(f"source NPZ hash drift during prepare: {source_npz}")
        if population_id not in population_first_cache:
            population_rows = _read_csv(population_manifest)
            if not population_rows:
                raise AssertionError(f"empty population manifest: {population_manifest}")
            population_first_cache[population_id] = population_rows[0]
        population_first = population_first_cache[population_id]
        x_shape = json.loads(descriptor_row["x_shape_json"])
        if len(x_shape) != 2:
            raise AssertionError(f"invalid X shape for {dataset_id}/{descriptor}: {x_shape}")
        n_samples, n_features = (int(x_shape[0]), int(x_shape[1]))
        population_folds = sorted(
            folds_by_population[population_id], key=lambda row: (int(row["repeat"]), int(row["fold"]))
        )
        if len(population_folds) != 25:
            raise AssertionError(f"{population_id} must have 25 folds")

        for fold_row in population_folds:
            manifest_repeat = int(fold_row["repeat"])
            manifest_fold = int(fold_row["fold"])
            seed = int(fold_row["seed"])
            n_train = int(fold_row["n_train"])
            n_valid = int(fold_row["n_valid"])
            for model_id in MODEL_ORDER:
                config_key = (model_id, seed, n_features, n_train)
                if config_key not in model_cache:
                    _unused_estimator, config = _model_config(
                        model_id, seed=seed, n_features=n_features, n_train=n_train
                    )
                    config_hash = smoke._hash_payload(config)
                    model_cache[config_key] = (config_hash, config)
                    if config_hash in model_configs and model_configs[config_hash] != config:
                        raise AssertionError("model config hash collision")
                    model_configs[config_hash] = config
                model_config_hash, _config = model_cache[config_key]
                output_path = (
                    plan_dir
                    / "runs"
                    / population_id
                    / descriptor
                    / model_id
                    / f"repeat_{manifest_repeat - 1:02d}"
                    / f"fold_{manifest_fold - 1:02d}"
                )
                task: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "pending",
                    "population_id": population_id,
                    "dataset_id": dataset_id,
                    "descriptor": descriptor,
                    "model": model_id,
                    "repeat_zero_based": manifest_repeat - 1,
                    "fold_zero_based": manifest_fold - 1,
                    "manifest_repeat": manifest_repeat,
                    "manifest_fold": manifest_fold,
                    "seed": seed,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "source_npz": _repo_relative(source_npz),
                    "source_npz_sha256": descriptor_row["source_npz_sha256"],
                    "input_hash": descriptor_row["source_npz_sha256"],
                    "x_key": str(inventory_entry.get("x_key", "X")),
                    "y_key": str(inventory_entry.get("y_key", "y")),
                    "feature_hash": descriptor_row["x_sha256"],
                    "y_hash": descriptor_row["y_sha256"],
                    "population_hash": fold_row["population_hash"],
                    "mapping_hash": population_first["mapping_hash"],
                    "split_hash": fold_row["split_hash"],
                    "fold_hash": fold_row["fold_hash"],
                    "model_config_hash": model_config_hash,
                    "population_manifest_path": _repo_relative(population_manifest),
                    "population_manifest_sha256": file_hash_cache[population_manifest],
                    "split_manifest_path": _repo_relative(split_manifest),
                    "split_manifest_sha256": file_hash_cache[split_manifest],
                    "train_sample_ids_hash": fold_row["train_sample_ids_hash"],
                    "valid_sample_ids_hash": fold_row["valid_sample_ids_hash"],
                    "train_source_row_index_hash": fold_row["train_source_row_index_hash"],
                    "valid_source_row_index_hash": fold_row["valid_source_row_index_hash"],
                    "train_y_hash": fold_row["train_y_hash"],
                    "valid_y_hash": fold_row["valid_y_hash"],
                    "output_path": _repo_relative(output_path),
                }
                contract_hash = smoke._hash_payload(_task_contract(task))
                task["task_contract_hash"] = contract_hash
                task["task_id"] = _task_id(task, contract_hash)
                tasks.append(task)

    source_files = {
        "inventory": _source_file_record(_repo_path(STEP28_1_INVENTORY)),
        "descriptor_availability": _source_file_record(
            _repo_path(STEP28_2_DIR / "descriptor_availability.csv")
        ),
        "fold_manifest": _source_file_record(fold_manifest_path),
    }
    plan_hash = smoke._hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "source_files": source_files,
            "model_configs": model_configs,
            "task_contract_hashes": [task["task_contract_hash"] for task in tasks],
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "步骤28.3第二构建单元：默认15静态矩阵×4模型×25外层折的可恢复任务计划。",
        "created_at_utc": _utc_now(),
        "git_commit_at_prepare": _git_commit(),
        "plan_hash": plan_hash,
        "plan_dir": _repo_relative(plan_dir),
        "status_policy": "Plan rows remain immutable pending contracts; runtime status lives under each task output path.",
        "execution_policy": {
            "serial": True,
            "max_cpus": MAX_CPUS,
            "conda_environment": "yonod",
            "thread_environment": THREAD_ENV,
            "external_validation_only": True,
            "no_internal_holdout": True,
        },
        "excluded": {
            "BH1_DFT": "blocked_alignment",
            "OHE": "fold-fitted supplementary baseline, not a static main-matrix descriptor",
            "AutoGluon": "internal-holdout exploratory appendix, not a main-matrix model",
            "SM_OHE_5760": "separate supplementary population; main matrix uses SM-static-4620",
        },
        "source_files": source_files,
        "model_configs": model_configs,
        "task_count": len(tasks),
        "combination_count": len(
            {(task["population_id"], task["descriptor"], task["model"]) for task in tasks}
        ),
        "tasks": tasks,
    }


def prepare_plan(plan_dir: Path) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    payload = _prepare_payload(plan_dir)
    plan_json, plan_csv = _plan_paths(plan_dir)
    smoke._atomic_json(plan_json, payload)
    smoke._atomic_csv(plan_csv, payload["tasks"], TASK_FIELDS)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "task_count": payload["task_count"],
        "combination_count": payload["combination_count"],
        "included_descriptor_matrix_count": len(
            {(task["population_id"], task["descriptor"]) for task in payload["tasks"]}
        ),
        "population_count": len({task["population_id"] for task in payload["tasks"]}),
        "seeds": sorted({task["seed"] for task in payload["tasks"]}),
        "plan_json": _repo_relative(plan_json),
        "plan_json_sha256": smoke._sha256_file(plan_json),
        "plan_csv": _repo_relative(plan_csv),
        "plan_csv_sha256": smoke._sha256_file(plan_csv),
        "full_training_started": False,
    }
    smoke._atomic_json(plan_dir / "preflight_summary.json", summary)
    return summary


def _load_plan(plan_dir: Path) -> dict[str, Any]:
    plan_json, _plan_csv = _plan_paths(plan_dir)
    return json.loads(plan_json.read_text(encoding="utf-8"))


def _compare_task_csv(plan_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    rows = _read_csv(plan_dir / PLAN_CSV_NAME)
    if len(rows) != len(tasks):
        raise AssertionError("task_plan.csv row count differs from task_plan.json")
    for index, (csv_row, json_row) in enumerate(zip(rows, tasks)):
        for field in TASK_FIELDS:
            if str(csv_row[field]) != str(json_row[field]):
                raise AssertionError(f"CSV/JSON mismatch at row {index}, field {field}")


def _recompute_fold_hash(task: Mapping[str, Any], split_rows: Sequence[Mapping[str, str]]) -> str:
    train_rows = [row for row in split_rows if row["role"] == "train"]
    valid_rows = [row for row in split_rows if row["role"] == "valid"]
    payload = {
        "population_id": task["population_id"],
        "dataset_id": task["dataset_id"],
        "population_hash": task["population_hash"],
        "repeat": int(task["manifest_repeat"]),
        "fold": int(task["manifest_fold"]),
        "seed": int(task["seed"]),
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "valid_sample_ids": [row["sample_id"] for row in valid_rows],
        "train_source_row_index": [int(row["source_row_index"]) for row in train_rows],
        "valid_source_row_index": [int(row["source_row_index"]) for row in valid_rows],
    }
    return smoke._hash_payload(payload)


def _split_rows_for_task(task: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = _read_csv(_repo_path(task["split_manifest_path"]))
    selected = [
        row
        for row in rows
        if int(row["repeat"]) == int(task["manifest_repeat"])
        and int(row["fold"]) == int(task["manifest_fold"])
    ]
    selected.sort(key=lambda row: int(row["population_position"]))
    return selected


def _verify_task_inputs(task: Mapping[str, Any], *, verify_feature_array: bool) -> None:
    for field in TASK_FIELDS:
        if field not in task:
            raise AssertionError(f"task missing field: {field}")
    recomputed_contract_hash = smoke._hash_payload(_task_contract(task))
    if task["task_contract_hash"] != recomputed_contract_hash:
        raise AssertionError(f"task contract hash mismatch: {task['task_id']}")
    if task["task_id"] != _task_id(task, recomputed_contract_hash):
        raise AssertionError(f"task_id is not stable/recomputable: {task['task_id']}")
    source_npz = _repo_path(task["source_npz"])
    if smoke._sha256_file(source_npz) != task["source_npz_sha256"]:
        raise AssertionError(f"source NPZ hash drift: {task['task_id']}")
    if task["input_hash"] != task["source_npz_sha256"]:
        raise AssertionError(f"input_hash mismatch: {task['task_id']}")
    population_manifest = _repo_path(task["population_manifest_path"])
    split_manifest = _repo_path(task["split_manifest_path"])
    if smoke._sha256_file(population_manifest) != task["population_manifest_sha256"]:
        raise AssertionError(f"population manifest file drift: {task['task_id']}")
    if smoke._sha256_file(split_manifest) != task["split_manifest_sha256"]:
        raise AssertionError(f"split manifest file drift: {task['task_id']}")
    population_rows = sorted(
        _read_csv(population_manifest), key=lambda row: int(row["population_position"])
    )
    if len(population_rows) != int(task["n_samples"]):
        raise AssertionError(f"population size mismatch: {task['task_id']}")
    if {row["population_hash"] for row in population_rows} != {task["population_hash"]}:
        raise AssertionError(f"population hash mismatch: {task['task_id']}")
    if {row["mapping_hash"] for row in population_rows} != {task["mapping_hash"]}:
        raise AssertionError(f"mapping hash mismatch: {task['task_id']}")
    split_rows = _split_rows_for_task(task)
    if len(split_rows) != int(task["n_samples"]):
        raise AssertionError(f"fold coverage mismatch: {task['task_id']}")
    if {row["split_hash"] for row in split_rows} != {task["split_hash"]}:
        raise AssertionError(f"split hash mismatch: {task['task_id']}")
    if {row["fold_hash"] for row in split_rows} != {task["fold_hash"]}:
        raise AssertionError(f"fold manifest hash mismatch: {task['task_id']}")
    if _recompute_fold_hash(task, split_rows) != task["fold_hash"]:
        raise AssertionError(f"fold hash is not reproducible: {task['task_id']}")
    train_rows = [row for row in split_rows if row["role"] == "train"]
    valid_rows = [row for row in split_rows if row["role"] == "valid"]
    if len(train_rows) != int(task["n_train"]) or len(valid_rows) != int(task["n_valid"]):
        raise AssertionError(f"fold sizes mismatch: {task['task_id']}")
    if smoke._hash_string_sequence(row["sample_id"] for row in train_rows) != task["train_sample_ids_hash"]:
        raise AssertionError(f"train sample hash mismatch: {task['task_id']}")
    if smoke._hash_string_sequence(row["sample_id"] for row in valid_rows) != task["valid_sample_ids_hash"]:
        raise AssertionError(f"valid sample hash mismatch: {task['task_id']}")
    if verify_feature_array:
        with smoke.np.load(source_npz, allow_pickle=False) as archive:
            X = smoke.np.asarray(archive[task["x_key"]])
            y = smoke.np.asarray(archive[task["y_key"]], dtype=smoke.np.float64)
        if list(X.shape) != [int(task["n_samples"]), int(task["n_features"])]:
            raise AssertionError(f"feature shape mismatch: {task['task_id']}")
        if smoke._sha256_array(X) != task["feature_hash"]:
            raise AssertionError(f"feature hash drift: {task['task_id']}")
        if smoke._sha256_array(y) != task["y_hash"]:
            raise AssertionError(f"target hash drift: {task['task_id']}")
        y_population = smoke.np.asarray([float(row["y"]) for row in population_rows], dtype=smoke.np.float64)
        if not smoke.np.array_equal(y, y_population):
            raise AssertionError(f"NPZ y order differs from population: {task['task_id']}")


def verify_plan(plan_dir: Path) -> dict[str, Any]:
    smoke._require_yonod_conda_env()
    plan_dir = _repo_path(plan_dir)
    payload = _load_plan(plan_dir)
    tasks = payload.get("tasks", [])
    if len(tasks) != 1500 or payload.get("task_count") != 1500:
        raise AssertionError(f"expected 1500 tasks, got {len(tasks)}")
    if len({task["task_id"] for task in tasks}) != 1500:
        raise AssertionError("task_id values are not unique")
    combinations = {
        (task["population_id"], task["dataset_id"], task["descriptor"], task["model"])
        for task in tasks
    }
    if len(combinations) != 60 or payload.get("combination_count") != 60:
        raise AssertionError(f"expected 60 combinations, got {len(combinations)}")
    matrix_counts = Counter((task["population_id"], task["descriptor"]) for task in tasks)
    if len(matrix_counts) != 15 or set(matrix_counts.values()) != {100}:
        raise AssertionError("each of 15 descriptor matrices must have exactly 100 tasks")
    if {int(task["seed"]) for task in tasks} != set(range(1000, 1005)):
        raise AssertionError("task seeds must be exactly 1000..1004")
    for task in tasks:
        if int(task["seed"]) != 1000 + int(task["repeat_zero_based"]):
            raise AssertionError(f"repeat/seed mismatch: {task['task_id']}")
        if task["model"].lower() in FORBIDDEN_MODELS:
            raise AssertionError("AutoGluon appeared in main plan")
        if task["descriptor"].lower() in FORBIDDEN_DESCRIPTORS:
            raise AssertionError("OHE appeared in main plan")
        if task["dataset_id"] == "BH1" and task["descriptor"] == "DFT":
            raise AssertionError("BH1 DFT appeared in main plan")
        if task["dataset_id"] == "SM" and task["population_id"] != "SM-static-4620":
            raise AssertionError("non-static SM population appeared in main plan")
        config = payload["model_configs"].get(task["model_config_hash"])
        if config is None or smoke._hash_payload(config) != task["model_config_hash"]:
            raise AssertionError(f"model config hash mismatch: {task['task_id']}")

    per_fold = Counter(
        (
            task["population_id"],
            task["descriptor"],
            int(task["repeat_zero_based"]),
            int(task["fold_zero_based"]),
        )
        for task in tasks
    )
    if set(per_fold.values()) != {4}:
        raise AssertionError("each descriptor fold must contain exactly four models")
    model_sets: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for task in tasks:
        key = (
            task["population_id"],
            task["descriptor"],
            int(task["repeat_zero_based"]),
            int(task["fold_zero_based"]),
        )
        model_sets[key].add(task["model"])
    if any(models != set(MODEL_ORDER) for models in model_sets.values()):
        raise AssertionError("a descriptor fold does not have the required four-model set")

    source_files = payload["source_files"]
    for source in source_files.values():
        if smoke._sha256_file(_repo_path(source["path"])) != source["sha256"]:
            raise AssertionError(f"plan source hash drift: {source['path']}")
    expected_plan_hash = smoke._hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "source_files": source_files,
            "model_configs": payload["model_configs"],
            "task_contract_hashes": [task["task_contract_hash"] for task in tasks],
        }
    )
    if payload["plan_hash"] != expected_plan_hash:
        raise AssertionError("plan_hash is not reproducible")
    _compare_task_csv(plan_dir, tasks)

    # Recompute per-task contracts/folds; expensive NPZ array hashes only once per matrix.
    verified_matrix: set[tuple[str, str]] = set()
    for task in tasks:
        recomputed_contract_hash = smoke._hash_payload(_task_contract(task))
        if task["task_contract_hash"] != recomputed_contract_hash:
            raise AssertionError(f"task contract hash mismatch: {task['task_id']}")
        if task["task_id"] != _task_id(task, recomputed_contract_hash):
            raise AssertionError(f"task_id is not stable/recomputable: {task['task_id']}")
        matrix_key = (task["population_id"], task["descriptor"])
        if matrix_key in verified_matrix:
            continue
        _verify_task_inputs(task, verify_feature_array=True)
        verified_matrix.add(matrix_key)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "task_count": 1500,
        "combination_count": 60,
        "included_descriptor_matrix_count": 15,
        "tasks_per_descriptor_matrix": 100,
        "models_per_fold": 4,
        "population_count": 4,
        "seeds": list(range(1000, 1005)),
        "unique_task_ids": True,
        "all_hashes_recomputed": True,
        "bh1_dft_present": False,
        "ohe_present": False,
        "autogluon_present": False,
        "sm_population_ids": ["SM-static-4620"],
        "full_training_started": False,
    }
    smoke._atomic_json(plan_dir / "verification_summary.json", summary)
    return summary


def _enforce_execution_resources() -> dict[str, Any]:
    smoke._require_yonod_conda_env()
    for name, expected in THREAD_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment drift: {name}={os.environ.get(name)!r}")
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if len(available) < MAX_CPUS:
            raise RuntimeError(f"19 CPUs required, only {len(available)} available")
        if len(available) > MAX_CPUS:
            os.sched_setaffinity(0, set(available[:MAX_CPUS]))
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if len(affinity) != MAX_CPUS:
            raise RuntimeError(f"failed to enforce 19-CPU affinity: {affinity}")
    return {
        "max_cpus": MAX_CPUS,
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity) if affinity is not None else None,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "serial": True,
    }


def _load_execution_inputs(task: Mapping[str, Any]) -> dict[str, Any]:
    _verify_task_inputs(task, verify_feature_array=True)
    population_rows = sorted(
        _read_csv(_repo_path(task["population_manifest_path"])),
        key=lambda row: int(row["population_position"]),
    )
    split_rows = _split_rows_for_task(task)
    train_rows = [row for row in split_rows if row["role"] == "train"]
    valid_rows = [row for row in split_rows if row["role"] == "valid"]
    with smoke.np.load(_repo_path(task["source_npz"]), allow_pickle=False) as archive:
        X = smoke.np.asarray(archive[task["x_key"]])
        y = smoke.np.asarray(archive[task["y_key"]], dtype=smoke.np.float64)
    train_idx = smoke.np.asarray(
        [int(row["population_position"]) for row in train_rows], dtype=smoke.np.int64
    )
    valid_idx = smoke.np.asarray(
        [int(row["population_position"]) for row in valid_rows], dtype=smoke.np.int64
    )
    return {
        "population_rows": population_rows,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
    }


def _completion_is_reusable(task: Mapping[str, Any], output_dir: Path) -> bool:
    completion_path = output_dir / "completion.json"
    if not completion_path.exists():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for field in ("task_id", "task_contract_hash", "plan_hash"):
        if str(completion.get(field)) != str(task.get(field)):
            raise RuntimeError(f"hash drift refuses reuse for {task['task_id']}: {field}")
    metadata_path = _repo_path(completion["metadata_path"])
    predictions_path = _repo_path(completion["predictions_path"])
    if not metadata_path.is_file() or not predictions_path.is_file():
        raise RuntimeError(f"recorded complete artifact is incomplete: {task['task_id']}")
    if smoke._sha256_file(predictions_path) != completion["predictions_sha256"]:
        raise RuntimeError(f"completed prediction hash drift: {task['task_id']}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError(f"completion points to non-complete metadata: {task['task_id']}")
    if metadata.get("task_contract_hash") != task["task_contract_hash"]:
        raise RuntimeError(f"completed metadata contract drift: {task['task_id']}")
    prediction_rows = _read_csv(predictions_path)
    if len(prediction_rows) != int(task["n_valid"]):
        raise RuntimeError(f"completed prediction row count drift: {task['task_id']}")
    y_true = smoke.np.asarray([float(row["y_true"]) for row in prediction_rows])
    y_pred = smoke.np.asarray([float(row["y_pred"]) for row in prediction_rows])
    metrics = smoke._metric_dict(y_true, y_pred)
    for name in ("mae", "rmse", "r2", "kendall_tau"):
        left, right = metrics.get(name), metadata["metrics"].get(name)
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"completed metric drift: {task['task_id']} {name}")
        elif abs(float(left) - float(right)) > 1e-10:
            raise RuntimeError(f"completed metric drift: {task['task_id']} {name}")
    return True


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-p{os.getpid()}"


def _write_prediction_rows(
    path: Path,
    task: Mapping[str, Any],
    valid_rows: Sequence[Mapping[str, str]],
    y_true: Any,
    y_pred: Any,
) -> None:
    rows = []
    for index, (row, truth, prediction) in enumerate(zip(valid_rows, y_true, y_pred)):
        rows.append(
            {
                "prediction_index": index,
                "task_id": task["task_id"],
                "population_id": task["population_id"],
                "dataset_id": task["dataset_id"],
                "descriptor": task["descriptor"],
                "model": task["model"],
                "repeat_zero_based": task["repeat_zero_based"],
                "fold_zero_based": task["fold_zero_based"],
                "manifest_repeat": task["manifest_repeat"],
                "manifest_fold": task["manifest_fold"],
                "seed": task["seed"],
                "population_position": int(row["population_position"]),
                "sample_id": row["sample_id"],
                "source_row_index": int(row["source_row_index"]),
                "y_true": repr(float(truth)),
                "y_pred": repr(float(prediction)),
            }
        )
    smoke._atomic_csv(path, rows, PREDICTION_FIELDS)


def _run_task(task: dict[str, Any], plan_hash: str, resource_policy: Mapping[str, Any]) -> dict[str, Any]:
    task = dict(task)
    task["plan_hash"] = plan_hash
    output_dir = _repo_path(task["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "task_contract.json"
    contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_contract_hash": task["task_contract_hash"],
        "plan_hash": plan_hash,
        "contract": _task_contract(task),
    }
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract_payload:
            raise RuntimeError(f"hash drift refuses task output reuse: {task['task_id']}")
    else:
        smoke._atomic_json(contract_path, contract_payload)
    if _completion_is_reusable(task, output_dir):
        return {"task_id": task["task_id"], "status": "skipped_complete_same_hash"}

    attempt_dir = output_dir / "attempts" / _attempt_id()
    attempt_dir.mkdir(parents=True)
    metadata_path = attempt_dir / "metadata.json"
    predictions_path = attempt_dir / "predictions.csv"
    metadata: dict[str, Any] = {
        **{field: task[field] for field in TASK_FIELDS},
        "plan_hash": plan_hash,
        "status": "started",
        "started_at_utc": _utc_now(),
        "command_line": list(sys.argv),
        "git_commit": _git_commit(),
        "resource_policy": dict(resource_policy),
        "metadata_path": _repo_relative(metadata_path),
        "predictions_path": _repo_relative(predictions_path),
    }
    smoke._atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        inputs = _load_execution_inputs(task)
        estimator, model_config = _model_config(
            task["model"],
            seed=int(task["seed"]),
            n_features=int(task["n_features"]),
            n_train=int(task["n_train"]),
        )
        if smoke._hash_payload(model_config) != task["model_config_hash"]:
            raise RuntimeError("model config hash drift; refusing fit")
        X_train = inputs["X"][inputs["train_idx"]]
        y_train = inputs["y"][inputs["train_idx"]]
        X_valid = inputs["X"][inputs["valid_idx"]]
        y_valid = inputs["y"][inputs["valid_idx"]]
        train_started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        y_pred = smoke.np.asarray(estimator.predict(X_valid), dtype=smoke.np.float64)
        predict_time = time.perf_counter() - predict_started
        if y_pred.shape != y_valid.shape or not smoke.np.isfinite(y_pred).all():
            raise ValueError("prediction shape mismatch or non-finite prediction")
        _write_prediction_rows(predictions_path, task, inputs["valid_rows"], y_valid, y_pred)
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        metadata.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "model_config": model_config,
                "metrics": smoke._metric_dict(y_valid, y_pred),
                "timing": {
                    "train_time_s": float(train_time),
                    "predict_time_s": float(predict_time),
                    "total_time_s": float(time.perf_counter() - started),
                },
                "resource_summary": {
                    "ru_maxrss_kib": int(after_usage.ru_maxrss),
                    "user_cpu_s_delta": float(after_usage.ru_utime - before_usage.ru_utime),
                    "system_cpu_s_delta": float(after_usage.ru_stime - before_usage.ru_stime),
                },
                "hashes": {
                    "input_hash": task["input_hash"],
                    "population_hash": task["population_hash"],
                    "split_hash": task["split_hash"],
                    "fold_hash": task["fold_hash"],
                    "feature_hash": task["feature_hash"],
                    "model_config_hash": task["model_config_hash"],
                    "train_feature_hash": smoke._sha256_array(X_train),
                    "valid_feature_hash": smoke._sha256_array(X_valid),
                    "train_y_hash": smoke._sha256_array(y_train),
                    "valid_y_hash": smoke._sha256_array(y_valid),
                    "predictions_sha256": smoke._sha256_file(predictions_path),
                },
                "external_fold_contract": {
                    "internal_holdout": False,
                    "external_valid_used_for_tuning": False,
                    "external_valid_used_for_early_stopping": False,
                    "row_selection_performed": False,
                },
            }
        )
        smoke._atomic_json(metadata_path, metadata)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": metadata["completed_at_utc"],
            "task_id": task["task_id"],
            "task_contract_hash": task["task_contract_hash"],
            "plan_hash": plan_hash,
            "metadata_path": _repo_relative(metadata_path),
            "predictions_path": _repo_relative(predictions_path),
            "predictions_sha256": smoke._sha256_file(predictions_path),
        }
        smoke._atomic_json(output_dir / "completion.json", completion)
        return {"task_id": task["task_id"], "status": "complete", "metrics": metadata["metrics"]}
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "timing": {"total_time_s": float(time.perf_counter() - started)},
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        smoke._atomic_json(metadata_path, metadata)
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "metadata_path": _repo_relative(metadata_path),
        }


def _select_tasks(tasks: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filters = {
        "population_id": args.population_id,
        "dataset_id": args.dataset_id,
        "descriptor": args.descriptor,
        "model": args.model,
        "repeat_zero_based": args.repeat,
        "fold_zero_based": args.fold,
    }
    if args.task_id:
        if any(value is not None for value in filters.values()):
            raise ValueError("--task-id cannot be combined with task filters")
        selected = [task for task in tasks if task["task_id"] == args.task_id]
    else:
        if not any(value is not None for value in filters.values()):
            raise ValueError("execution requires --task-id or at least one explicit filter")
        selected = []
        for task in tasks:
            matches = True
            for field, value in filters.items():
                if value is None:
                    continue
                if field in {"repeat_zero_based", "fold_zero_based"}:
                    matches = matches and int(task[field]) == int(value)
                else:
                    matches = matches and str(task[field]).lower() == str(value).lower()
            if matches:
                selected.append(task)
    if not selected:
        raise ValueError("task selector matched zero planned tasks")
    if len(selected) > args.max_tasks:
        raise ValueError(
            f"selector matched {len(selected)} tasks, exceeding --max-tasks={args.max_tasks}; "
            "raise the explicit limit to acknowledge serial work"
        )
    return selected


def execute_selected(plan_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    verification = verify_plan(plan_dir)
    if verification["status"] != "passed":
        raise RuntimeError("plan verification did not pass")
    payload = _load_plan(plan_dir)
    selected = _select_tasks(payload["tasks"], args)
    resource_policy = _enforce_execution_resources()
    results = []
    for task in selected:  # Intentionally serial: no job pool and no nested task parallelism.
        results.append(_run_task(task, payload["plan_hash"], resource_policy))
    status = "complete" if all(row["status"] in {"complete", "skipped_complete_same_hash"} for row in results) else "failed"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "completed_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "selected_task_count": len(selected),
        "serial_execution": True,
        "resource_policy": resource_policy,
        "results": results,
    }
    smoke._atomic_json(plan_dir / "last_execution_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", default=DEFAULT_PLAN_DIR.as_posix())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Generate the immutable 1500-task plan.")
    mode.add_argument("--verify-plan", action="store_true", help="Recompute plan/input hashes and gates.")
    parser.add_argument("--task-id")
    parser.add_argument("--population-id")
    parser.add_argument("--dataset-id")
    parser.add_argument("--descriptor")
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--repeat", type=int, choices=range(5), help="Zero-based repeat (0..4).")
    parser.add_argument("--fold", type=int, choices=range(5), help="Zero-based fold (0..4).")
    parser.add_argument("--max-tasks", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    plan_dir = Path(args.plan_dir)
    try:
        if args.prepare:
            result = prepare_plan(plan_dir)
        elif args.verify_plan:
            result = verify_plan(_repo_path(plan_dir))
        else:
            result = execute_selected(plan_dir, args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"prepared", "passed", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
