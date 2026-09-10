#!/usr/bin/env python3
"""Prepare, verify, and run step 29 static-descriptor AutoGluon folds.

This companion runner does not modify YONOD package code.  It consumes the
frozen step 28 population/split manifests and RF fold outputs, then adds only
AutoGluon folds for DFT/SOAP/PhysChem so they can later be paired with RF on
the same external validation samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


THREAD_ENV = {
    "LOKY_MAX_CPU_COUNT": "19",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _thread_name, _thread_value in THREAD_ENV.items():
    os.environ.setdefault(_thread_name, _thread_value)

import numpy as _np

try:  # scipy is available in the yonod env; dry-run can still work without it.
    from scipy.stats import kendalltau as _kendalltau
except Exception:  # pragma: no cover - environment-dependent diagnostic path
    _kendalltau = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: _np.ndarray) -> str:
    contiguous = _np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _sha256_int_sequence(values: Iterable[int]) -> str:
    return _sha256_array(_np.asarray(tuple(int(value) for value in values), dtype=_np.int64))


def _hash_string_sequence(values: Iterable[str]) -> str:
    return _hash_payload([str(value) for value in values])


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _metric_dict(y_true: Any, y_pred: Any) -> dict[str, Any]:
    truth = _np.asarray(y_true, dtype=_np.float64)
    pred = _np.asarray(y_pred, dtype=_np.float64)
    residual = truth - pred
    mae = float(_np.mean(_np.abs(residual)))
    rmse = float(_np.sqrt(_np.mean(residual ** 2)))
    ss_res = float(_np.sum(residual ** 2))
    ss_tot = float(_np.sum((truth - float(_np.mean(truth))) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    result: dict[str, Any] = {"mae": mae, "rmse": rmse, "r2": r2}
    if _kendalltau is None:
        result.update({
            "kendall_tau": None,
            "kendall_tau_pvalue": None,
            "kendall_tau_defined": False,
            "kendall_tau_reason": "scipy_unavailable",
        })
        return result
    tau, pvalue = _kendalltau(truth, pred)
    if _np.isfinite(tau):
        result.update({
            "kendall_tau": float(tau),
            "kendall_tau_pvalue": float(pvalue) if _np.isfinite(pvalue) else None,
            "kendall_tau_defined": True,
            "kendall_tau_reason": "",
        })
    else:
        result.update({
            "kendall_tau": None,
            "kendall_tau_pvalue": None,
            "kendall_tau_defined": False,
            "kendall_tau_reason": "constant_or_invalid_input",
        })
    return result


class _SmokeCompat:
    np = _np
    _hash_payload = staticmethod(_hash_payload)
    _sha256_file = staticmethod(_sha256_file)
    _sha256_array = staticmethod(_sha256_array)
    _sha256_int_sequence = staticmethod(_sha256_int_sequence)
    _hash_string_sequence = staticmethod(_hash_string_sequence)
    _atomic_json = staticmethod(_atomic_json)
    _atomic_csv = staticmethod(_atomic_csv)
    _metric_dict = staticmethod(_metric_dict)


smoke = _SmokeCompat()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
STEP_NAME = "step29_autogluon_static_rf_compare"
EVALUATION_PROTOCOL = "step29_static_descriptor_autogluon_5x5"
MAX_CPUS = 19
SMOKE_CPUS = 2
FORMAL_TIME_LIMIT = 300
SMOKE_TIME_LIMIT = 30
AUTOGUON_PRESETS = "medium_quality"
AUTOGUON_RANDOM_STATE = 42
AUTOGUON_SEED_POLICY = (
    "outer split fixed by step28 split_manifest seeds 1000-1004; "
    "AutoGluon 1.1.1 has no single top-level seed covering all submodels"
)
STEP28_1_INVENTORY = Path(
    "derived/descriptor_model_effect/step28_1_input_audit/"
    "official_static_descriptor_inventory.json"
)
STEP28_2_DIR = Path("derived/descriptor_model_effect/step28_2_population_splits")
STEP28_3_DIR = Path("derived/descriptor_model_effect/step28_3_matrix_plan")
DEFAULT_PLAN_DIR = Path("derived/descriptor_model_effect") / STEP_NAME
TASK_JSON_NAME = "task_manifest.json"
TASK_CSV_NAME = "task_manifest.csv"
DRY_RUN_COMMANDS_NAME = "dry_run_commands.txt"

DESCRIPTOR_ORDER = ("DFT", "SOAP", "PhysChem")
POPULATION_ORDER = ("BH1-static-3955", "BH2-static-3359", "SL1-static-1150", "SM-static-4620")
EXPECTED_MATRIXES = (
    ("BH1-static-3955", "SOAP"),
    ("BH1-static-3955", "PhysChem"),
    ("BH2-static-3359", "DFT"),
    ("BH2-static-3359", "SOAP"),
    ("BH2-static-3359", "PhysChem"),
    ("SL1-static-1150", "DFT"),
    ("SL1-static-1150", "SOAP"),
    ("SL1-static-1150", "PhysChem"),
    ("SM-static-4620", "DFT"),
    ("SM-static-4620", "SOAP"),
    ("SM-static-4620", "PhysChem"),
)
EXPECTED_MATRIX_SET = set(EXPECTED_MATRIXES)

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
    "autogluon_config_hash",
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
    "rf_task_id",
    "rf_task_contract_hash",
    "rf_plan_hash",
    "rf_model_config_hash",
    "rf_output_path",
    "rf_completion_path",
    "rf_metadata_path",
    "rf_predictions_path",
    "rf_predictions_sha256",
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _git_dirty_summary() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    lines = [line for line in output.splitlines() if line.strip()]
    return {"available": True, "dirty": bool(lines), "line_count": len(lines), "lines": lines[:60]}


def _task_manifest_paths(plan_dir: Path) -> tuple[Path, Path]:
    return plan_dir / TASK_JSON_NAME, plan_dir / TASK_CSV_NAME


def _task_contract(task: Mapping[str, Any]) -> dict[str, Any]:
    return {field: task[field] for field in CONTRACT_FIELDS}


def _task_id(task: Mapping[str, Any], contract_hash: str) -> str:
    return (
        f"s29-{str(task['dataset_id']).lower()}-{str(task['descriptor']).lower()}-"
        f"autogluon-r{int(task['repeat_zero_based']):02d}-"
        f"f{int(task['fold_zero_based']):02d}-{contract_hash[:12]}"
    )


def _autogluon_config(*, time_limit: int, num_cpus: int, cleanup: bool = True) -> dict[str, Any]:
    return {
        "adapter": "autogluon.tabular.TabularPredictor direct outer-fold wrapper",
        "display_name": "AutoGluon TabularPredictor",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "external_valid_used_for_tuning": False,
        "external_valid_used_for_early_stopping": False,
        "internal_holdout_scope": "training_fold_only",
        "model_artifact_cleanup": bool(cleanup),
        "model_id": "autogluon",
        "params": {
            "time_limit": int(time_limit),
            "presets": AUTOGUON_PRESETS,
            "num_cpus": int(num_cpus),
            "random_state": AUTOGUON_RANDOM_STATE,
        },
        "resource_policy": {
            "max_cpus": int(num_cpus),
            "serial_outer_folds": True,
        },
        "seed_policy": AUTOGUON_SEED_POLICY,
        "source": "YONOD step29 direct TabularPredictor strict external-fold wrapper",
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_step28_plan() -> dict[str, Any]:
    return _load_json(_repo_path(STEP28_3_DIR / "task_plan.json"))


def _source_file_record(path: Path) -> dict[str, str]:
    return {"path": _repo_relative(path), "sha256": smoke._sha256_file(path)}


def _hash_int_values(values: Iterable[int]) -> str:
    return smoke._sha256_int_sequence(tuple(int(value) for value in values))


def _prediction_hashes(prediction_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    return {
        "valid_sample_ids_hash": smoke._hash_string_sequence(row["sample_id"] for row in prediction_rows),
        "valid_source_row_index_hash": _hash_int_values(int(row["source_row_index"]) for row in prediction_rows),
        "valid_y_hash": smoke._sha256_array(
            smoke.np.asarray([float(row["y_true"]) for row in prediction_rows], dtype=smoke.np.float64)
        ),
    }


def _validate_rf_reference(rf_task: Mapping[str, Any], step28_plan_hash: str) -> dict[str, Any]:
    output_path = _repo_path(rf_task["output_path"])
    completion_path = output_path / "completion.json"
    if not completion_path.is_file():
        raise AssertionError(f"RF completion missing for {rf_task['task_id']}: {completion_path}")
    completion = _load_json(completion_path)
    if completion.get("status") != "complete":
        raise AssertionError(f"RF completion is not complete: {rf_task['task_id']}")
    expected = {
        "task_id": rf_task["task_id"],
        "task_contract_hash": rf_task["task_contract_hash"],
        "plan_hash": step28_plan_hash,
    }
    for key, value in expected.items():
        if str(completion.get(key)) != str(value):
            raise AssertionError(f"RF completion {key} mismatch for {rf_task['task_id']}")
    metadata_path = _repo_path(completion["metadata_path"])
    predictions_path = _repo_path(completion["predictions_path"])
    if not metadata_path.is_file() or not predictions_path.is_file():
        raise AssertionError(f"RF completion artifacts incomplete: {rf_task['task_id']}")
    if smoke._sha256_file(predictions_path) != completion["predictions_sha256"]:
        raise AssertionError(f"RF prediction file hash drift: {rf_task['task_id']}")
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("model") != "rf":
        raise AssertionError(f"RF metadata is not a complete RF fold: {rf_task['task_id']}")
    for field in (
        "population_id",
        "dataset_id",
        "descriptor",
        "repeat_zero_based",
        "fold_zero_based",
        "manifest_repeat",
        "manifest_fold",
        "seed",
        "n_samples",
        "n_features",
        "n_train",
        "n_valid",
        "input_hash",
        "feature_hash",
        "y_hash",
        "population_hash",
        "mapping_hash",
        "split_hash",
        "fold_hash",
        "train_sample_ids_hash",
        "valid_sample_ids_hash",
        "train_source_row_index_hash",
        "valid_source_row_index_hash",
        "train_y_hash",
        "valid_y_hash",
    ):
        if str(metadata.get(field)) != str(rf_task[field]):
            raise AssertionError(f"RF metadata field drift {field}: {rf_task['task_id']}")
    prediction_rows = _read_csv(predictions_path)
    if len(prediction_rows) != int(rf_task["n_valid"]):
        raise AssertionError(f"RF prediction row count mismatch: {rf_task['task_id']}")
    hashes = _prediction_hashes(prediction_rows)
    for field, value in hashes.items():
        if str(value) != str(rf_task[field]):
            raise AssertionError(f"RF prediction {field} mismatch: {rf_task['task_id']}")
    y_true = smoke.np.asarray([float(row["y_true"]) for row in prediction_rows], dtype=smoke.np.float64)
    y_pred = smoke.np.asarray([float(row["y_pred"]) for row in prediction_rows], dtype=smoke.np.float64)
    metrics = smoke._metric_dict(y_true, y_pred)
    for name in ("mae", "rmse", "r2", "kendall_tau"):
        left, right = metrics.get(name), metadata.get("metrics", {}).get(name)
        if name == "kendall_tau" and left is None:
            continue
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"RF metric drift {name}: {rf_task['task_id']}")
        elif abs(float(left) - float(right)) > 1e-10:
            raise AssertionError(f"RF metric drift {name}: {rf_task['task_id']}")
    return {
        "rf_completion_path": _repo_relative(completion_path),
        "rf_metadata_path": _repo_relative(metadata_path),
        "rf_predictions_path": _repo_relative(predictions_path),
        "rf_predictions_sha256": completion["predictions_sha256"],
        "rf_model_config_hash": metadata["model_config_hash"],
    }


def _formal_rf_tasks(step28_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for task in step28_plan.get("tasks", []):
        matrix = (str(task["population_id"]), str(task["descriptor"]))
        if task.get("model") == "rf" and matrix in EXPECTED_MATRIX_SET:
            tasks.append(dict(task))
    rank_population = {value: index for index, value in enumerate(POPULATION_ORDER)}
    rank_descriptor = {value: index for index, value in enumerate(DESCRIPTOR_ORDER)}
    tasks.sort(
        key=lambda row: (
            rank_population[str(row["population_id"])],
            rank_descriptor[str(row["descriptor"])],
            int(row["repeat_zero_based"]),
            int(row["fold_zero_based"]),
        )
    )
    return tasks


def _prepare_payload(plan_dir: Path) -> dict[str, Any]:
    step28_plan = _load_step28_plan()
    step28_plan_hash = str(step28_plan["plan_hash"])
    rf_tasks = _formal_rf_tasks(step28_plan)
    if len(rf_tasks) != 275:
        raise AssertionError(f"expected 275 RF reference tasks, got {len(rf_tasks)}")
    if {
        (task["population_id"], task["descriptor"]) for task in rf_tasks
    } != EXPECTED_MATRIX_SET:
        raise AssertionError("RF reference task matrix is not the step29 11-combo scope")

    model_config = _autogluon_config(time_limit=FORMAL_TIME_LIMIT, num_cpus=MAX_CPUS, cleanup=True)
    model_config_hash = smoke._hash_payload(model_config)
    tasks: list[dict[str, Any]] = []
    for rf_task in rf_tasks:
        rf_reference = _validate_rf_reference(rf_task, step28_plan_hash)
        output_path = (
            plan_dir
            / "runs"
            / str(rf_task["population_id"])
            / str(rf_task["descriptor"])
            / "autogluon"
            / f"repeat_{int(rf_task['repeat_zero_based']):02d}"
            / f"fold_{int(rf_task['fold_zero_based']):02d}"
        )
        task = {
            "schema_version": SCHEMA_VERSION,
            "status": "pending",
            "population_id": rf_task["population_id"],
            "dataset_id": rf_task["dataset_id"],
            "descriptor": rf_task["descriptor"],
            "model": "autogluon",
            "repeat_zero_based": rf_task["repeat_zero_based"],
            "fold_zero_based": rf_task["fold_zero_based"],
            "manifest_repeat": rf_task["manifest_repeat"],
            "manifest_fold": rf_task["manifest_fold"],
            "seed": rf_task["seed"],
            "n_samples": rf_task["n_samples"],
            "n_features": rf_task["n_features"],
            "n_train": rf_task["n_train"],
            "n_valid": rf_task["n_valid"],
            "source_npz": rf_task["source_npz"],
            "source_npz_sha256": rf_task["source_npz_sha256"],
            "input_hash": rf_task["input_hash"],
            "x_key": rf_task["x_key"],
            "y_key": rf_task["y_key"],
            "feature_hash": rf_task["feature_hash"],
            "y_hash": rf_task["y_hash"],
            "population_hash": rf_task["population_hash"],
            "mapping_hash": rf_task["mapping_hash"],
            "split_hash": rf_task["split_hash"],
            "fold_hash": rf_task["fold_hash"],
            "model_config_hash": model_config_hash,
            "autogluon_config_hash": model_config_hash,
            "population_manifest_path": rf_task["population_manifest_path"],
            "population_manifest_sha256": rf_task["population_manifest_sha256"],
            "split_manifest_path": rf_task["split_manifest_path"],
            "split_manifest_sha256": rf_task["split_manifest_sha256"],
            "train_sample_ids_hash": rf_task["train_sample_ids_hash"],
            "valid_sample_ids_hash": rf_task["valid_sample_ids_hash"],
            "train_source_row_index_hash": rf_task["train_source_row_index_hash"],
            "valid_source_row_index_hash": rf_task["valid_source_row_index_hash"],
            "train_y_hash": rf_task["train_y_hash"],
            "valid_y_hash": rf_task["valid_y_hash"],
            "rf_task_id": rf_task["task_id"],
            "rf_task_contract_hash": rf_task["task_contract_hash"],
            "rf_plan_hash": step28_plan_hash,
            "rf_output_path": rf_task["output_path"],
            **rf_reference,
            "output_path": _repo_relative(output_path),
        }
        contract_hash = smoke._hash_payload(_task_contract(task))
        task["task_contract_hash"] = contract_hash
        task["task_id"] = _task_id(task, contract_hash)
        tasks.append(task)

    source_files = {
        "step28_task_plan": _source_file_record(_repo_path(STEP28_3_DIR / "task_plan.json")),
        "step28_descriptor_availability": _source_file_record(
            _repo_path(STEP28_2_DIR / "descriptor_availability.csv")
        ),
        "step28_fold_manifest": _source_file_record(_repo_path(STEP28_2_DIR / "fold_manifest.csv")),
        "step28_input_inventory": _source_file_record(_repo_path(STEP28_1_INVENTORY)),
    }
    plan_hash = smoke._hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "source_files": source_files,
            "rf_plan_hash": step28_plan_hash,
            "model_configs": {model_config_hash: model_config},
            "task_contract_hashes": [task["task_contract_hash"] for task in tasks],
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Step 29: AutoGluon folds for DFT/SOAP/PhysChem paired with step28 RF.",
        "created_at_utc": _utc_now(),
        "git_commit_at_prepare": _git_commit(),
        "plan_hash": plan_hash,
        "plan_dir": _repo_relative(plan_dir),
        "rf_plan_hash": step28_plan_hash,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "excluded": {
            "BH1_DFT": "blocked_alignment retained from step28.1/step28.2",
            "MFP": "handled by step27 bridge material; not rerun in step29",
            "OHE": "not a static precomputed descriptor in the step29 comparison",
            "SVM": "not part of AutoGluon vs RF paired comparison",
        },
        "execution_policy": {
            "serial": True,
            "formal_max_cpus": MAX_CPUS,
            "conda_environment": "yonod",
            "thread_environment": THREAD_ENV,
            "external_valid_used_for_tuning": False,
            "external_valid_used_for_early_stopping": False,
            "autogluon_internal_holdout_scope": "training_fold_only",
        },
        "source_files": source_files,
        "model_configs": {model_config_hash: model_config},
        "task_count": len(tasks),
        "combination_count": len({(task["population_id"], task["descriptor"]) for task in tasks}),
        "tasks": tasks,
    }


def prepare_plan(plan_dir: Path) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    payload = _prepare_payload(plan_dir)
    task_json, task_csv = _task_manifest_paths(plan_dir)
    smoke._atomic_json(task_json, payload)
    smoke._atomic_csv(task_csv, payload["tasks"], TASK_FIELDS)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "formal_nohup_command": (
            "nohup bash scripts/run_static_descriptor_autogluon_19cpu_nohup.sh --run --resume "
            "> logs/step29_static_descriptor_autogluon_19cpu.full.log 2>&1 &"
        ),
        "formal_params": {
            "time_limit": FORMAL_TIME_LIMIT,
            "presets": AUTOGUON_PRESETS,
            "num_cpus": MAX_CPUS,
        },
        "smoke_params": {
            "population_id": "SL1-static-1150",
            "descriptor": "PhysChem",
            "repeat_zero_based": 0,
            "fold_zero_based": 0,
            "time_limit": SMOKE_TIME_LIMIT,
            "num_cpus": SMOKE_CPUS,
        },
        "full_training_started": False,
    }
    smoke._atomic_json(plan_dir / "run_manifest.json", run_manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "task_count": payload["task_count"],
        "combination_count": payload["combination_count"],
        "included_descriptor_matrix_count": payload["combination_count"],
        "population_count": len({task["population_id"] for task in payload["tasks"]}),
        "seeds": sorted({task["seed"] for task in payload["tasks"]}),
        "task_manifest_json": _repo_relative(task_json),
        "task_manifest_json_sha256": smoke._sha256_file(task_json),
        "task_manifest_csv": _repo_relative(task_csv),
        "task_manifest_csv_sha256": smoke._sha256_file(task_csv),
        "full_training_started": False,
    }
    smoke._atomic_json(plan_dir / "preflight_summary.json", summary)
    return summary


def _load_plan(plan_dir: Path) -> dict[str, Any]:
    task_json, _task_csv = _task_manifest_paths(plan_dir)
    return _load_json(task_json)


def _compare_task_csv(plan_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    rows = _read_csv(plan_dir / TASK_CSV_NAME)
    if len(rows) != len(tasks):
        raise AssertionError("task_manifest.csv row count differs from task_manifest.json")
    for index, (csv_row, json_row) in enumerate(zip(rows, tasks)):
        for field in TASK_FIELDS:
            if str(csv_row[field]) != str(json_row[field]):
                raise AssertionError(f"CSV/JSON mismatch at row {index}, field {field}")


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


def _verify_task_inputs(task: Mapping[str, Any], *, verify_feature_array: bool) -> None:
    for field in TASK_FIELDS:
        if field not in task:
            raise AssertionError(f"task missing field: {field}")
    recomputed_contract_hash = smoke._hash_payload(_task_contract(task))
    if task["task_contract_hash"] != recomputed_contract_hash:
        raise AssertionError(f"task contract hash mismatch: {task['task_id']}")
    if task["task_id"] != _task_id(task, recomputed_contract_hash):
        raise AssertionError(f"task_id is not stable/recomputable: {task['task_id']}")
    if task["model"] != "autogluon":
        raise AssertionError(f"non-AutoGluon task in step29: {task['task_id']}")
    if (task["population_id"], task["descriptor"]) not in EXPECTED_MATRIX_SET:
        raise AssertionError(f"task outside step29 matrix scope: {task['task_id']}")

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
    if _hash_int_values(int(row["source_row_index"]) for row in train_rows) != task["train_source_row_index_hash"]:
        raise AssertionError(f"train source row hash mismatch: {task['task_id']}")
    if _hash_int_values(int(row["source_row_index"]) for row in valid_rows) != task["valid_source_row_index_hash"]:
        raise AssertionError(f"valid source row hash mismatch: {task['task_id']}")

    rf_stub = {
        "task_id": task["rf_task_id"],
        "task_contract_hash": task["rf_task_contract_hash"],
        "model_config_hash": task["rf_model_config_hash"],
        "output_path": task["rf_output_path"],
        **{field: task[field] for field in (
            "population_id",
            "dataset_id",
            "descriptor",
            "repeat_zero_based",
            "fold_zero_based",
            "manifest_repeat",
            "manifest_fold",
            "seed",
            "n_samples",
            "n_features",
            "n_train",
            "n_valid",
            "input_hash",
            "feature_hash",
            "y_hash",
            "population_hash",
            "mapping_hash",
            "split_hash",
            "fold_hash",
            "train_sample_ids_hash",
            "valid_sample_ids_hash",
            "train_source_row_index_hash",
            "valid_source_row_index_hash",
            "train_y_hash",
            "valid_y_hash",
        )},
    }
    _validate_rf_reference(rf_stub, str(task["rf_plan_hash"]))

    if verify_feature_array:
        with smoke.np.load(source_npz, allow_pickle=False) as archive:
            X = smoke.np.asarray(archive[task["x_key"]])
            y = smoke.np.asarray(archive[task["y_key"]], dtype=smoke.np.float64)
        if list(X.shape) != [int(task["n_samples"]), int(task["n_features"])]:
            raise AssertionError(f"feature shape mismatch: {task['task_id']}")
        if smoke.np.isinf(X).any():
            raise AssertionError(f"feature matrix contains inf: {task['task_id']}")
        if smoke._sha256_array(X) != task["feature_hash"]:
            raise AssertionError(f"feature hash drift: {task['task_id']}")
        if smoke._sha256_array(y) != task["y_hash"]:
            raise AssertionError(f"target hash drift: {task['task_id']}")
        y_population = smoke.np.asarray([float(row["y"]) for row in population_rows], dtype=smoke.np.float64)
        if not smoke.np.array_equal(y, y_population):
            raise AssertionError(f"NPZ y order differs from population: {task['task_id']}")


def verify_plan(plan_dir: Path) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    payload = _load_plan(plan_dir)
    tasks = payload.get("tasks", [])
    if len(tasks) != 275 or payload.get("task_count") != 275:
        raise AssertionError(f"expected 275 tasks, got {len(tasks)}")
    if len({task["task_id"] for task in tasks}) != 275:
        raise AssertionError("task_id values are not unique")
    combinations = {(task["population_id"], task["descriptor"]) for task in tasks}
    if combinations != EXPECTED_MATRIX_SET:
        raise AssertionError(f"unexpected step29 combination set: {sorted(combinations)}")
    if payload.get("combination_count") != 11:
        raise AssertionError("combination_count must be 11")
    matrix_counts = Counter((task["population_id"], task["descriptor"]) for task in tasks)
    if set(matrix_counts.values()) != {25}:
        raise AssertionError("each step29 descriptor matrix must have exactly 25 tasks")
    if {int(task["seed"]) for task in tasks} != set(range(1000, 1005)):
        raise AssertionError("task seeds must be exactly 1000..1004")
    if any(task["descriptor"] in {"MFP", "OHE"} for task in tasks):
        raise AssertionError("MFP/OHE must not appear in step29")
    if any(task["descriptor"] == "DFT" and task["dataset_id"] == "BH1" for task in tasks):
        raise AssertionError("BH1 DFT must stay excluded")
    if any(task["model"] != "autogluon" for task in tasks):
        raise AssertionError("only AutoGluon tasks are allowed in step29")

    for task in tasks:
        if int(task["seed"]) != 1000 + int(task["repeat_zero_based"]):
            raise AssertionError(f"repeat/seed mismatch: {task['task_id']}")
        config = payload["model_configs"].get(task["model_config_hash"])
        if config is None or smoke._hash_payload(config) != task["model_config_hash"]:
            raise AssertionError(f"AutoGluon config hash mismatch: {task['task_id']}")
        if task["autogluon_config_hash"] != task["model_config_hash"]:
            raise AssertionError(f"AutoGluon config alias mismatch: {task['task_id']}")
    source_files = payload["source_files"]
    for source in source_files.values():
        if smoke._sha256_file(_repo_path(source["path"])) != source["sha256"]:
            raise AssertionError(f"plan source hash drift: {source['path']}")
    expected_plan_hash = smoke._hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "source_files": source_files,
            "rf_plan_hash": payload["rf_plan_hash"],
            "model_configs": payload["model_configs"],
            "task_contract_hashes": [task["task_contract_hash"] for task in tasks],
        }
    )
    if payload["plan_hash"] != expected_plan_hash:
        raise AssertionError("plan_hash is not reproducible")
    _compare_task_csv(plan_dir, tasks)

    verified_matrix: set[tuple[str, str]] = set()
    for task in tasks:
        recomputed_contract_hash = smoke._hash_payload(_task_contract(task))
        if task["task_contract_hash"] != recomputed_contract_hash:
            raise AssertionError(f"task contract hash mismatch: {task['task_id']}")
        if task["task_id"] != _task_id(task, recomputed_contract_hash):
            raise AssertionError(f"task_id is not stable/recomputable: {task['task_id']}")
        matrix_key = (task["population_id"], task["descriptor"])
        if matrix_key not in verified_matrix:
            _verify_task_inputs(task, verify_feature_array=True)
            verified_matrix.add(matrix_key)
        else:
            _verify_task_inputs(task, verify_feature_array=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "rf_plan_hash": payload["rf_plan_hash"],
        "task_count": 275,
        "combination_count": 11,
        "tasks_per_descriptor_matrix": 25,
        "population_count": 4,
        "seeds": list(range(1000, 1005)),
        "unique_task_ids": True,
        "all_hashes_recomputed": True,
        "bh1_dft_present": False,
        "mfp_present": False,
        "ohe_present": False,
        "svm_present": False,
        "rf_reference_complete": True,
        "full_training_started": False,
    }
    smoke._atomic_json(plan_dir / "verification_summary.json", summary)
    return summary


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
    train_idx = smoke.np.asarray([int(row["population_position"]) for row in train_rows], dtype=smoke.np.int64)
    valid_idx = smoke.np.asarray([int(row["population_position"]) for row in valid_rows], dtype=smoke.np.int64)
    return {
        "population_rows": population_rows,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
    }


def _require_yonod_conda_env() -> None:
    executable = Path(sys.executable).as_posix()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "yonod" or "/envs/yonod/" in executable:
        return
    raise RuntimeError(
        "AutoGluon execution must run in conda env 'yonod'; current executable="
        f"{executable!r}, CONDA_DEFAULT_ENV={conda_env!r}"
    )


def _enforce_execution_resources(num_cpus: int) -> dict[str, Any]:
    _require_yonod_conda_env()
    if int(num_cpus) > MAX_CPUS:
        raise RuntimeError(f"num_cpus cannot exceed {MAX_CPUS}")
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if len(available) < int(num_cpus):
            raise RuntimeError(f"{num_cpus} CPUs required, only {len(available)} available")
        if len(available) > int(num_cpus):
            os.sched_setaffinity(0, set(available[: int(num_cpus)]))
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if len(affinity) != int(num_cpus):
            raise RuntimeError(f"failed to enforce CPU affinity for {num_cpus} CPUs: {affinity}")
    return {
        "max_cpus": int(num_cpus),
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity) if affinity is not None else None,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "serial": True,
    }


def _completion_is_reusable(task: Mapping[str, Any], output_dir: Path, plan_hash: str) -> bool:
    completion_path = output_dir / "completion.json"
    if not completion_path.exists():
        return False
    completion = _load_json(completion_path)
    expected = {
        "task_id": task["task_id"],
        "task_contract_hash": task["task_contract_hash"],
        "plan_hash": plan_hash,
        "autogluon_config_hash": task["autogluon_config_hash"],
    }
    for field, value in expected.items():
        if str(completion.get(field)) != str(value):
            raise RuntimeError(f"hash drift refuses reuse for {task['task_id']}: {field}")
    metadata_path = _repo_path(completion["metadata_path"])
    predictions_path = _repo_path(completion["predictions_path"])
    if not metadata_path.is_file() or not predictions_path.is_file():
        raise RuntimeError(f"recorded complete artifact is incomplete: {task['task_id']}")
    if smoke._sha256_file(predictions_path) != completion["predictions_sha256"]:
        raise RuntimeError(f"completed prediction hash drift: {task['task_id']}")
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"completion points to non-complete metadata: {task['task_id']}")
    if metadata.get("task_contract_hash") != task["task_contract_hash"]:
        raise RuntimeError(f"completed metadata contract drift: {task['task_id']}")
    prediction_rows = _read_csv(predictions_path)
    if len(prediction_rows) != int(task["n_valid"]):
        raise RuntimeError(f"completed prediction row count drift: {task['task_id']}")
    hashes = _prediction_hashes(prediction_rows)
    for field, value in hashes.items():
        if str(value) != str(task[field]):
            raise RuntimeError(f"completed prediction {field} drift: {task['task_id']}")
    y_true = smoke.np.asarray([float(row["y_true"]) for row in prediction_rows], dtype=smoke.np.float64)
    y_pred = smoke.np.asarray([float(row["y_pred"]) for row in prediction_rows], dtype=smoke.np.float64)
    if not smoke.np.isfinite(y_pred).all():
        raise RuntimeError(f"completed prediction contains NaN/inf: {task['task_id']}")
    metrics = smoke._metric_dict(y_true, y_pred)
    for name in ("mae", "rmse", "r2", "kendall_tau"):
        left, right = metrics.get(name), metadata.get("metrics", {}).get(name)
        if name == "kendall_tau" and left is None:
            continue
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


def _clone_smoke_task(task: Mapping[str, Any], plan_dir: Path, time_limit: int, num_cpus: int) -> dict[str, Any]:
    cloned = dict(task)
    model_config = _autogluon_config(time_limit=int(time_limit), num_cpus=int(num_cpus), cleanup=True)
    model_config_hash = smoke._hash_payload(model_config)
    cloned["model_config_hash"] = model_config_hash
    cloned["autogluon_config_hash"] = model_config_hash
    cloned["output_path"] = _repo_relative(
        plan_dir
        / "smoke_runs"
        / str(task["population_id"])
        / str(task["descriptor"])
        / "autogluon"
        / f"repeat_{int(task['repeat_zero_based']):02d}"
        / f"fold_{int(task['fold_zero_based']):02d}"
        / f"time_limit_{int(time_limit)}_cpus_{int(num_cpus)}"
    )
    cloned["status"] = "pending"
    contract_hash = smoke._hash_payload(_task_contract(cloned))
    cloned["task_contract_hash"] = contract_hash
    cloned["task_id"] = _task_id(cloned, contract_hash).replace("s29-", "s29-smoke-", 1)
    return cloned


def _autogluon_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from importlib import metadata as importlib_metadata
        for package in ("autogluon", "autogluon.tabular", "autogluon.core", "pandas"):
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                versions[package] = "unavailable"
    except Exception as exc:  # pragma: no cover - diagnostic only
        versions["version_probe_error"] = f"{type(exc).__name__}: {exc}"
    return versions


def _make_autogluon_frame(X: Any, y: Any | None = None) -> Any:
    import pandas as pd

    array = smoke.np.asarray(X)
    columns = [f"f_{index:05d}" for index in range(array.shape[1])]
    frame = pd.DataFrame(array, columns=columns)
    if y is not None:
        frame["yield"] = smoke.np.asarray(y, dtype=smoke.np.float64)
    return frame


def _fit_predict_autogluon_fold(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    *,
    params: Mapping[str, Any],
    artifact_path: Path,
) -> tuple[Any, dict[str, Any]]:
    from autogluon.tabular import TabularPredictor

    train = smoke.np.asarray(X_train)
    valid = smoke.np.asarray(X_valid)
    target = smoke.np.asarray(y_train, dtype=smoke.np.float64)
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("AutoGluon input matrices must be two-dimensional with matching columns")
    if len(train) != len(target):
        raise ValueError("AutoGluon training target length mismatch")
    if smoke.np.isinf(train).any() or smoke.np.isinf(valid).any():
        raise ValueError("AutoGluon feature matrix contains inf")
    if not smoke.np.isfinite(target).all():
        raise ValueError("AutoGluon target contains NaN or inf")

    train_df = _make_autogluon_frame(train, target)
    valid_df = _make_autogluon_frame(valid)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    started_train = time.perf_counter()
    predictor = TabularPredictor(
        label="yield",
        problem_type="regression",
        path=str(artifact_path),
        verbosity=1,
    ).fit(
        train_data=train_df,
        time_limit=int(params["time_limit"]),
        presets=str(params["presets"]),
        num_cpus=int(params["num_cpus"]),
    )
    train_time = time.perf_counter() - started_train
    started_predict = time.perf_counter()
    prediction = smoke.np.asarray(predictor.predict(valid_df), dtype=smoke.np.float64)
    predict_time = time.perf_counter() - started_predict
    leaderboard_rows: int | None = None
    leaderboard_path: str | None = None
    try:
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard_rows = int(len(leaderboard))
        leaderboard_output = artifact_path.parent / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_output, index=False)
        leaderboard_path = _repo_relative(leaderboard_output)
    except Exception as exc:  # pragma: no cover - diagnostic only
        leaderboard_path = f"unavailable: {type(exc).__name__}: {exc}"
    shutil.rmtree(artifact_path, ignore_errors=True)
    return prediction, {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "time_limit": int(params["time_limit"]),
        "presets": str(params["presets"]),
        "num_cpus": int(params["num_cpus"]),
        "random_state": int(params["random_state"]),
        "random_state_policy": AUTOGUON_SEED_POLICY,
        "autogluon_versions": _autogluon_versions(),
        "train_time_s": float(train_time),
        "predict_time_s": float(predict_time),
        "model_artifact_path": str(artifact_path),
        "model_artifact_cleanup": True,
        "leaderboard_rows": leaderboard_rows,
        "leaderboard_path": leaderboard_path,
        "missing_value_policy": "AutoGluon internal train-fold preprocessing; outer valid is predict-only",
    }


def _run_task(
    task: dict[str, Any],
    plan_hash: str,
    resource_policy: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    task = dict(task)
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
        existing = _load_json(contract_path)
        if existing != contract_payload:
            raise RuntimeError(f"hash drift refuses task output reuse: {task['task_id']}")
    else:
        smoke._atomic_json(contract_path, contract_payload)
    if _completion_is_reusable(task, output_dir, plan_hash):
        return {"task_id": task["task_id"], "status": "skipped_complete_same_hash"}

    attempt_dir = output_dir / "attempts" / _attempt_id()
    attempt_dir.mkdir(parents=True)
    metadata_path = attempt_dir / "metadata.json"
    predictions_path = attempt_dir / "predictions.csv"
    artifact_path = attempt_dir / "autogluon_model"
    metadata: dict[str, Any] = {
        **{field: task[field] for field in TASK_FIELDS},
        "plan_hash": plan_hash,
        "status": "started",
        "started_at_utc": _utc_now(),
        "command_line": list(sys.argv),
        "git_commit": _git_commit(),
        "git_dirty_summary": _git_dirty_summary(),
        "resource_policy": dict(resource_policy),
        "metadata_path": _repo_relative(metadata_path),
        "predictions_path": _repo_relative(predictions_path),
    }
    smoke._atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        inputs = _load_execution_inputs(task)
        params = dict(model_config["params"])
        if smoke._hash_payload(model_config) != task["model_config_hash"]:
            raise RuntimeError("AutoGluon config hash drift; refusing fit")
        X_train = inputs["X"][inputs["train_idx"]]
        y_train = inputs["y"][inputs["train_idx"]]
        X_valid = inputs["X"][inputs["valid_idx"]]
        y_valid = inputs["y"][inputs["valid_idx"]]
        prediction, adapter_metadata = _fit_predict_autogluon_fold(
            X_train,
            y_train,
            X_valid,
            params=params,
            artifact_path=artifact_path,
        )
        y_pred = smoke.np.asarray(prediction, dtype=smoke.np.float64)
        if y_pred.ndim != 1:
            y_pred = y_pred.reshape(-1)
        if y_pred.shape != y_valid.shape or not smoke.np.isfinite(y_pred).all():
            raise ValueError("prediction shape mismatch or non-finite prediction")
        _write_prediction_rows(predictions_path, task, inputs["valid_rows"], y_valid, y_pred)
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        versions = dict(adapter_metadata.get("autogluon_versions", {}) or {})
        metrics = smoke._metric_dict(y_valid, y_pred)
        metadata.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "model_config": dict(model_config),
                "autogluon_params": dict(params),
                "autogluon_seed_policy": AUTOGUON_SEED_POLICY,
                "autogluon_versions": versions,
                "autogluon_version": str(versions.get("autogluon.tabular", "unknown")),
                "model_adapter_metadata": adapter_metadata,
                "metrics": metrics,
                "timing": {
                    "train_time_s": float(adapter_metadata.get("train_time_s", 0.0)),
                    "predict_time_s": float(adapter_metadata.get("predict_time_s", 0.0)),
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
                    "autogluon_config_hash": task["autogluon_config_hash"],
                    "train_feature_hash": smoke._sha256_array(X_train),
                    "valid_feature_hash": smoke._sha256_array(X_valid),
                    "train_y_hash": smoke._sha256_array(y_train),
                    "valid_y_hash": smoke._sha256_array(y_valid),
                    "predictions_sha256": smoke._sha256_file(predictions_path),
                },
                "external_fold_contract": {
                    "internal_holdout": True,
                    "internal_holdout_scope": "autogluon_training_fold_only",
                    "external_valid_used_for_tuning": False,
                    "external_valid_used_for_early_stopping": False,
                    "external_valid_used_for_preprocessing_fit": False,
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
            "autogluon_config_hash": task["autogluon_config_hash"],
            "metadata_path": _repo_relative(metadata_path),
            "predictions_path": _repo_relative(predictions_path),
            "predictions_sha256": smoke._sha256_file(predictions_path),
        }
        smoke._atomic_json(output_dir / "completion.json", completion)
        return {"task_id": task["task_id"], "status": "complete", "metrics": metrics}
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
    if args.smoke:
        if len(selected) != 1:
            raise ValueError("--smoke must select exactly one fold")
        selected = [_clone_smoke_task(selected[0], plan_dir, args.time_limit, args.num_cpus)]
        model_config = _autogluon_config(time_limit=args.time_limit, num_cpus=args.num_cpus, cleanup=True)
        run_plan_hash = "smoke-" + smoke._hash_payload(
            {"formal_plan_hash": payload["plan_hash"], "task_contract_hash": selected[0]["task_contract_hash"]}
        )
    else:
        model_config = payload["model_configs"][selected[0]["model_config_hash"]]
        for task in selected:
            if task["model_config_hash"] != selected[0]["model_config_hash"]:
                raise RuntimeError("selected formal tasks have mixed AutoGluon configs")
        run_plan_hash = payload["plan_hash"]
    resource_policy = _enforce_execution_resources(args.num_cpus)
    results = []
    for task in selected:
        results.append(_run_task(task, run_plan_hash, resource_policy, model_config))
    status = (
        "complete"
        if all(row["status"] in {"complete", "skipped_complete_same_hash"} for row in results)
        else "failed"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "completed_at_utc": _utc_now(),
        "plan_hash": run_plan_hash,
        "formal_plan_hash": payload["plan_hash"],
        "smoke": bool(args.smoke),
        "selected_task_count": len(selected),
        "serial_execution": True,
        "resource_policy": resource_policy,
        "results": results,
    }
    summary_name = "last_smoke_execution_summary.json" if args.smoke else "last_execution_summary.json"
    smoke._atomic_json(plan_dir / summary_name, summary)
    return summary


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def dry_run(plan_dir: Path, *, conda_bin: str, conda_env: str, cpu_set: str) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    verification = verify_plan(plan_dir)
    payload = _load_plan(plan_dir)
    commands = []
    runner = "scripts/run_static_descriptor_autogluon_matrix.py"
    for task in payload["tasks"]:
        command = [
            "taskset",
            "-c",
            cpu_set,
            "env",
            *[f"{name}={value}" for name, value in THREAD_ENV.items()],
            conda_bin,
            "run",
            "-n",
            conda_env,
            "python",
            runner,
            "--plan-dir",
            str(Path(payload["plan_dir"])),
            "--task-id",
            task["task_id"],
            "--max-tasks",
            "1",
        ]
        commands.append(" ".join(_shell_quote(str(part)) for part in command))
    command_path = plan_dir / DRY_RUN_COMMANDS_NAME
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "dry_run_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "task_count": len(payload["tasks"]),
        "combination_count": verification["combination_count"],
        "command_count": len(commands),
        "commands_path": _repo_relative(command_path),
        "formal_training_started": False,
    }
    smoke._atomic_json(plan_dir / "dry_run_summary.json", summary)
    return summary


def completion_summary(plan_dir: Path, *, require_complete: bool = False) -> dict[str, Any]:
    plan_dir = _repo_path(plan_dir)
    payload = _load_plan(plan_dir)
    counts = Counter()
    incomplete: list[dict[str, str]] = []
    for task in payload.get("tasks", []):
        output_dir = _repo_path(task["output_path"])
        try:
            reusable = _completion_is_reusable(task, output_dir, payload["plan_hash"])
        except Exception as exc:
            counts["failed_or_drifted"] += 1
            incomplete.append(
                {
                    "task_id": task["task_id"],
                    "population_id": task["population_id"],
                    "descriptor": task["descriptor"],
                    "repeat_zero_based": str(task["repeat_zero_based"]),
                    "fold_zero_based": str(task["fold_zero_based"]),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if reusable:
            counts["complete"] += 1
        else:
            counts["pending"] += 1
            incomplete.append(
                {
                    "task_id": task["task_id"],
                    "population_id": task["population_id"],
                    "descriptor": task["descriptor"],
                    "repeat_zero_based": str(task["repeat_zero_based"]),
                    "fold_zero_based": str(task["fold_zero_based"]),
                    "reason": "completion.json missing",
                }
            )
    status = "complete" if counts["complete"] == 275 else "incomplete"
    if require_complete and status != "complete":
        status = "failed"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checked_at_utc": _utc_now(),
        "plan_hash": payload["plan_hash"],
        "expected_folds": 275,
        "completed_folds": int(counts["complete"]),
        "pending_folds": int(counts["pending"]),
        "failed_or_drifted_folds": int(counts["failed_or_drifted"]),
        "incomplete_preview": incomplete[:100],
    }
    smoke._atomic_json(plan_dir / "completion_summary.json", summary)
    if require_complete and summary["completed_folds"] != 275:
        raise RuntimeError(f"step29 incomplete: {summary['completed_folds']}/275 complete")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", default=DEFAULT_PLAN_DIR.as_posix())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Generate the immutable 275-task manifest.")
    mode.add_argument("--verify-plan", action="store_true", help="Recompute manifest/input/RF hashes.")
    mode.add_argument("--dry-run", action="store_true", help="Write 275 exact fold commands; do not train.")
    mode.add_argument("--completion-summary", action="store_true", help="Summarize formal fold completions.")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--task-id")
    parser.add_argument("--population-id")
    parser.add_argument("--dataset-id")
    parser.add_argument("--descriptor", choices=DESCRIPTOR_ORDER)
    parser.add_argument("--model", choices=("autogluon",), default=None)
    parser.add_argument("--repeat", type=int, choices=range(5), help="Zero-based repeat (0..4).")
    parser.add_argument("--fold", type=int, choices=range(5), help="Zero-based fold (0..4).")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Run the selected fold under smoke_runs/ with override params.")
    parser.add_argument("--time-limit", type=int, default=FORMAL_TIME_LIMIT)
    parser.add_argument("--num-cpus", type=int, default=MAX_CPUS)
    parser.add_argument("--conda-bin", default="/home/wangzh685/miniconda3/bin/conda")
    parser.add_argument("--conda-env", default="yonod")
    parser.add_argument("--cpu-set", default="0-18")
    args = parser.parse_args(argv)
    if args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    if args.time_limit < 1:
        parser.error("--time-limit must be positive")
    if args.num_cpus < 1:
        parser.error("--num-cpus must be positive")
    if args.smoke and (args.prepare or args.verify_plan or args.dry_run or args.completion_summary):
        parser.error("--smoke is an execution modifier and cannot be combined with manifest-only modes")
    plan_dir = Path(args.plan_dir)
    try:
        if args.prepare:
            result = prepare_plan(plan_dir)
        elif args.verify_plan:
            result = verify_plan(plan_dir)
        elif args.dry_run:
            result = dry_run(plan_dir, conda_bin=args.conda_bin, conda_env=args.conda_env, cpu_set=args.cpu_set)
        elif args.completion_summary:
            result = completion_summary(plan_dir, require_complete=args.require_complete)
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
    return 0 if result.get("status") in {"prepared", "passed", "dry_run", "complete", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
