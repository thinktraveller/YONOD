#!/usr/bin/env python3
"""Run the step 28.3 one-fold smoke for official static descriptors.

This script is intentionally outside the YONOD package.  It reads the frozen
step 28.1/28.2 manifests, fits four already-supported model families on one
external fold, and writes isolated audit artifacts under derived/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

THREAD_ENV_DEFAULTS = {
    "LOKY_MAX_CPU_COUNT": "19",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = Path("derived/descriptor_model_effect/step28_3_smoke")
STEP28_1_INVENTORY = Path(
    "derived/descriptor_model_effect/step28_1_input_audit/"
    "official_static_descriptor_inventory.json"
)
STEP28_2_DIR = Path("derived/descriptor_model_effect/step28_2_population_splits")
SMOKE_POPULATION_ID = "SL1-static-1150"
SMOKE_DATASET_ID = "SL1"
SMOKE_DESCRIPTOR = "PhysChem"
SMOKE_REPEAT_ZERO_BASED = 0
SMOKE_FOLD_ZERO_BASED = 0
SMOKE_REPEAT = SMOKE_REPEAT_ZERO_BASED + 1
SMOKE_FOLD = SMOKE_FOLD_ZERO_BASED + 1
SMOKE_SEED = 1000
MAX_CPUS = 19
MODEL_ORDER = ("rf", "xgboost", "svm", "lightgbm")
DESCRIPTOR_ORDER = ("DFT", "SOAP", "PhysChem", "MFP")
FORBIDDEN_MODELS = {"autogluon"}
FORBIDDEN_DESCRIPTORS = {"ohe"}


@dataclass(frozen=True)
class SmokeInputs:
    inventory_entry: Mapping[str, Any]
    population_rows: list[dict[str, str]]
    split_rows: list[dict[str, str]]
    train_rows: list[dict[str, str]]
    valid_rows: list[dict[str, str]]
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    input_hash: str
    feature_hash: str
    y_hash: str
    population_hash: str
    mapping_hash: str
    split_hash: str
    fold_hash: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _sha256_int_sequence(values: Iterable[int]) -> str:
    return _sha256_array(np.asarray(tuple(int(value) for value in values), dtype=np.int64))


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"smoke_sl1_physchem_r00_f00_{stamp}"


def _assert_inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must stay inside repository: {path}") from exc
    return resolved


def _append_log(run_dir: Path, event: str, payload: Mapping[str, Any]) -> None:
    log_path = run_dir / "command_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json({"time_utc": _utc_now(), "event": event, **dict(payload)}))
        handle.write("\n")


def _require_yonod_conda_env() -> None:
    executable = Path(sys.executable).as_posix()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "yonod" or "/envs/yonod/" in executable:
        return
    raise RuntimeError(
        "This smoke must be run in conda env 'yonod'; current executable="
        f"{executable!r}, CONDA_DEFAULT_ENV={conda_env!r}"
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "sklearn", "scipy", "xgboost", "lightgbm"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[module_name] = f"unavailable: {exc}"
    return versions


def _cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return None


def _environment_summary(command_line: Sequence[str]) -> dict[str, Any]:
    affinity = _cpu_affinity()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "package_versions": _package_versions(),
        "command_line": list(command_line),
        "cwd": str(Path.cwd()),
        "cpu_policy": {
            "max_cpus": MAX_CPUS,
            "sequential_models": True,
            "model_thread_caps": {
                "rf": MAX_CPUS,
                "xgboost": MAX_CPUS,
                "svm": 1,
                "lightgbm": MAX_CPUS,
            },
            "affinity_cpus": affinity,
            "affinity_cpu_count": len(affinity) if affinity is not None else None,
            "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_DEFAULTS},
        },
    }


def _load_inventory_entry() -> dict[str, Any]:
    payload = json.loads((PROJECT_ROOT / STEP28_1_INVENTORY).read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in payload.get("entries", [])
        if entry.get("dataset_id") == SMOKE_DATASET_ID and entry.get("descriptor") == SMOKE_DESCRIPTOR
    ]
    if len(entries) != 1:
        raise ValueError(f"expected one inventory entry for {SMOKE_DATASET_ID}/{SMOKE_DESCRIPTOR}")
    entry = entries[0]
    if str(entry.get("descriptor", "")).lower() in FORBIDDEN_DESCRIPTORS:
        raise ValueError("OHE must not enter the static descriptor smoke")
    return entry


def _load_static_npz(entry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    npz_path = PROJECT_ROOT / str(entry["source_npz"])
    with np.load(npz_path, allow_pickle=False) as archive:
        X = np.asarray(archive[str(entry.get("x_key", "X"))])
        y = np.asarray(archive[str(entry.get("y_key", "y"))], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"y must be 1-D and match X rows, got X={X.shape}, y={y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("official feature matrix or y contains NaN/inf")
    return X, y


def _load_inputs() -> SmokeInputs:
    entry = _load_inventory_entry()
    population_path = (
        PROJECT_ROOT
        / STEP28_2_DIR
        / "population_manifests"
        / f"{SMOKE_POPULATION_ID}_population_manifest.csv"
    )
    split_path = (
        PROJECT_ROOT
        / STEP28_2_DIR
        / "split_manifests"
        / f"{SMOKE_POPULATION_ID}_split_manifest.csv"
    )
    population_rows = sorted(_read_csv(population_path), key=lambda row: int(row["population_position"]))
    split_rows_all = _read_csv(split_path)
    split_rows = [
        row
        for row in split_rows_all
        if int(row["repeat"]) == SMOKE_REPEAT and int(row["fold"]) == SMOKE_FOLD
    ]
    split_rows = sorted(split_rows, key=lambda row: int(row["population_position"]))
    if len(population_rows) != 1150 or len(split_rows) != len(population_rows):
        raise ValueError("SL1 smoke population/split row counts are not 1150")
    train_rows = [row for row in split_rows if row["role"] == "train"]
    valid_rows = [row for row in split_rows if row["role"] == "valid"]
    if len(train_rows) != 920 or len(valid_rows) != 230:
        raise ValueError(f"unexpected fold sizes: train={len(train_rows)}, valid={len(valid_rows)}")
    train_ids = {row["sample_id"] for row in train_rows}
    valid_ids = {row["sample_id"] for row in valid_rows}
    if train_ids.intersection(valid_ids):
        raise ValueError("train/valid sample IDs overlap")
    if train_ids.union(valid_ids) != {row["sample_id"] for row in population_rows}:
        raise ValueError("train/valid split does not cover the full population")
    if {int(row["seed"]) for row in split_rows} != {SMOKE_SEED}:
        raise ValueError("smoke fold must use seed 1000")

    X, y = _load_static_npz(entry)
    y_population = np.asarray([float(row["y"]) for row in population_rows], dtype=np.float64)
    if X.shape != tuple(entry["x_shape"]):
        raise ValueError(f"X shape does not match inventory: {X.shape} vs {entry['x_shape']}")
    if not np.array_equal(y, y_population):
        raise ValueError("official NPZ y order does not match population manifest")

    train_idx = np.asarray([int(row["population_position"]) for row in train_rows], dtype=np.int64)
    valid_idx = np.asarray([int(row["population_position"]) for row in valid_rows], dtype=np.int64)
    first_population = population_rows[0]
    first_split = split_rows[0]
    return SmokeInputs(
        inventory_entry=entry,
        population_rows=population_rows,
        split_rows=split_rows,
        train_rows=train_rows,
        valid_rows=valid_rows,
        X=X,
        y=y,
        train_idx=train_idx,
        valid_idx=valid_idx,
        input_hash=_sha256_file(PROJECT_ROOT / str(entry["source_npz"])),
        feature_hash=_sha256_array(X),
        y_hash=_sha256_array(y),
        population_hash=first_population["population_hash"],
        mapping_hash=first_population["mapping_hash"],
        split_hash=first_split["split_hash"],
        fold_hash=first_split["fold_hash"],
    )


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tau = kendalltau(y_true, y_pred)
    tau_value = float(tau.statistic) if np.isfinite(tau.statistic) else None
    tau_pvalue = float(tau.pvalue) if np.isfinite(tau.pvalue) else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": tau_value,
        "kendall_tau_pvalue": tau_pvalue,
        "kendall_tau_defined": tau_value is not None,
        "kendall_tau_reason": "" if tau_value is not None else "undefined_or_constant_input",
    }


def _build_model(model_id: str, seed: int, n_features: int, n_train: int) -> tuple[Any, dict[str, Any]]:
    if model_id == "rf":
        from yonod.models.rf_model import RFYieldModel

        adapter = RFYieldModel(
            n_estimators=500,
            max_features=0.3,
            n_jobs=MAX_CPUS,
            random_state=seed,
        )
        model_config = {
            "model_id": model_id,
            "display_name": "Random Forest",
            "source": "YONOD RFYieldModel with paper-style step28.3 core parameters",
            "params": dict(adapter.params),
            "paper_style_required_params": {
                "n_estimators": 500,
                "max_features": 0.3,
                "random_state": seed,
            },
            "resource_policy": {"n_jobs": MAX_CPUS, "max_cpus": MAX_CPUS},
            "uses_external_valid_for_early_stopping": False,
            "internal_holdout": False,
            "row_selection_performed": False,
        }
        return adapter._build(), model_config
    if model_id == "xgboost":
        from yonod.models.xgb_model import XGBYieldModel

        adapter = XGBYieldModel(random_state=seed, device="cpu")
        params = dict(adapter.params)
        params["n_jobs"] = MAX_CPUS
        model_config = {
            "model_id": model_id,
            "display_name": "XGBoost",
            "source": "YONOD XGBYieldModel adapter defaults plus explicit CPU thread cap",
            "params": params,
            "resource_policy": {"n_jobs": MAX_CPUS, "device": "cpu", "max_cpus": MAX_CPUS},
            "uses_external_valid_for_early_stopping": False,
            "internal_holdout": False,
            "row_selection_performed": False,
        }
        import xgboost as xgb

        return xgb.XGBRegressor(**params), model_config
    if model_id == "svm":
        from yonod.models.svm_model import SVMYieldModel

        adapter = SVMYieldModel(random_state=seed, subsample_n=None)
        model = adapter._build(n_features=n_features, n_train=n_train)
        model_config = {
            "model_id": model_id,
            "display_name": "SVM",
            "source": "YONOD SVMYieldModel adapter defaults; external fold only",
            "params": {
                "kernel": adapter.kernel,
                "C": adapter.C,
                "gamma": adapter.gamma,
                "auto_pca": adapter.auto_pca,
                "pca_threshold": adapter.pca_threshold,
                "pca_n_components": adapter.pca_n_components,
                "pca_active_for_this_feature_matrix": bool(adapter.auto_pca and n_features > adapter.pca_threshold),
                "random_state": seed,
                "subsample_n": adapter.subsample_n,
            },
            "resource_policy": {"n_jobs": 1, "max_cpus": MAX_CPUS},
            "uses_external_valid_for_early_stopping": False,
            "internal_holdout": False,
            "row_selection_performed": False,
        }
        return model, model_config
    if model_id == "lightgbm":
        from yonod.models.lightgbm_model import LightGBMYieldModel

        adapter = LightGBMYieldModel(random_state=seed, n_jobs=MAX_CPUS)
        model_config = {
            "model_id": model_id,
            "display_name": "LightGBM",
            "source": "YONOD LightGBMYieldModel adapter defaults plus explicit CPU thread cap",
            "params": dict(adapter.params),
            "resource_policy": {"n_jobs": MAX_CPUS, "device_type": "cpu", "max_cpus": MAX_CPUS},
            "uses_external_valid_for_early_stopping": False,
            "internal_holdout": False,
            "row_selection_performed": False,
        }
        return adapter._build(), model_config
    raise ValueError(f"unsupported model: {model_id}")


def _task_base(inputs: SmokeInputs) -> dict[str, Any]:
    train_ids = [row["sample_id"] for row in inputs.train_rows]
    valid_ids = [row["sample_id"] for row in inputs.valid_rows]
    train_source_rows = [int(row["source_row_index"]) for row in inputs.train_rows]
    valid_source_rows = [int(row["source_row_index"]) for row in inputs.valid_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "population_id": SMOKE_POPULATION_ID,
        "dataset_id": SMOKE_DATASET_ID,
        "descriptor": SMOKE_DESCRIPTOR,
        "repeat_zero_based": SMOKE_REPEAT_ZERO_BASED,
        "fold_zero_based": SMOKE_FOLD_ZERO_BASED,
        "manifest_repeat": SMOKE_REPEAT,
        "manifest_fold": SMOKE_FOLD,
        "seed": SMOKE_SEED,
        "n_train": len(inputs.train_rows),
        "n_valid": len(inputs.valid_rows),
        "input_hash": inputs.input_hash,
        "source_npz_sha256": inputs.inventory_entry["source_npz_sha256"],
        "source_npz": inputs.inventory_entry["source_npz"],
        "population_hash": inputs.population_hash,
        "mapping_hash": inputs.mapping_hash,
        "split_hash": inputs.split_hash,
        "fold_hash": inputs.fold_hash,
        "feature_hash": inputs.feature_hash,
        "y_hash": inputs.y_hash,
        "train_sample_ids_hash": _hash_string_sequence(train_ids),
        "valid_sample_ids_hash": _hash_string_sequence(valid_ids),
        "train_source_row_index_hash": _sha256_int_sequence(train_source_rows),
        "valid_source_row_index_hash": _sha256_int_sequence(valid_source_rows),
        "target_unit": inputs.inventory_entry["target_unit"],
        "row_selection_performed": False,
        "internal_holdout": False,
        "fit_valid_used_for_early_stopping": False,
    }


def _prediction_rows(inputs: SmokeInputs, model_id: str, y_pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = inputs.y[inputs.valid_idx]
    for position, (split_row, truth, prediction) in enumerate(zip(inputs.valid_rows, y_true, y_pred)):
        rows.append(
            {
                "prediction_index": position,
                "population_id": SMOKE_POPULATION_ID,
                "dataset_id": SMOKE_DATASET_ID,
                "descriptor": SMOKE_DESCRIPTOR,
                "model": model_id,
                "repeat_zero_based": SMOKE_REPEAT_ZERO_BASED,
                "fold_zero_based": SMOKE_FOLD_ZERO_BASED,
                "manifest_repeat": SMOKE_REPEAT,
                "manifest_fold": SMOKE_FOLD,
                "seed": SMOKE_SEED,
                "population_position": int(split_row["population_position"]),
                "sample_id": split_row["sample_id"],
                "source_row_index": int(split_row["source_row_index"]),
                "y_true": repr(float(truth)),
                "y_pred": repr(float(prediction)),
            }
        )
    return rows


def _metric_row(metadata: Mapping[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("metrics", {})
    hashes = metadata.get("hashes", {})
    timing = metadata.get("timing", {})
    return {
        "status": metadata.get("status", ""),
        "population_id": metadata.get("population_id", ""),
        "dataset_id": metadata.get("dataset_id", ""),
        "descriptor": metadata.get("descriptor", ""),
        "model": metadata.get("model", ""),
        "repeat_zero_based": metadata.get("repeat_zero_based", ""),
        "fold_zero_based": metadata.get("fold_zero_based", ""),
        "manifest_repeat": metadata.get("manifest_repeat", ""),
        "manifest_fold": metadata.get("manifest_fold", ""),
        "seed": metadata.get("seed", ""),
        "n_train": metadata.get("n_train", ""),
        "n_valid": metadata.get("n_valid", ""),
        "mae": metrics.get("mae", ""),
        "rmse": metrics.get("rmse", ""),
        "r2": metrics.get("r2", ""),
        "kendall_tau": metrics.get("kendall_tau", ""),
        "kendall_tau_pvalue": metrics.get("kendall_tau_pvalue", ""),
        "kendall_tau_defined": metrics.get("kendall_tau_defined", ""),
        "train_time_s": timing.get("train_time_s", ""),
        "predict_time_s": timing.get("predict_time_s", ""),
        "total_fit_predict_time_s": timing.get("total_fit_predict_time_s", ""),
        "input_hash": hashes.get("input_hash", ""),
        "population_hash": hashes.get("population_hash", ""),
        "split_hash": hashes.get("split_hash", ""),
        "fold_hash": hashes.get("fold_hash", ""),
        "feature_hash": hashes.get("feature_hash", ""),
        "model_config_hash": hashes.get("model_config_hash", ""),
        "predictions_path": metadata.get("predictions_path", ""),
        "metadata_path": metadata.get("metadata_path", ""),
        "error_type": metadata.get("error_type", ""),
        "error_message": metadata.get("error_message", ""),
    }


def _write_task_manifest(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
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
        "n_train",
        "n_valid",
        "input_hash",
        "population_hash",
        "split_hash",
        "fold_hash",
        "feature_hash",
        "model_config_hash",
        "predictions_path",
        "metadata_path",
        "error_message",
    ]
    trimmed_rows = [{field: row.get(field, "") for field in fields} for row in rows]
    _atomic_csv(run_dir / "task_manifest.csv", trimmed_rows, fields)


def _run_one_model(run_dir: Path, inputs: SmokeInputs, model_id: str) -> dict[str, Any]:
    model_dir = run_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_dir / "fold_metadata.json"
    predictions_path = model_dir / "predictions.csv"
    base = _task_base(inputs)
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        **base,
        "model": model_id,
        "status": "started",
        "started_at_utc": _utc_now(),
        "metadata_path": _repo_relative(metadata_path),
        "predictions_path": _repo_relative(predictions_path),
    }
    _atomic_json(metadata_path, metadata)
    _append_log(run_dir, "model_started", {"model": model_id})

    try:
        estimator, model_config = _build_model(
            model_id,
            seed=SMOKE_SEED,
            n_features=int(inputs.X.shape[1]),
            n_train=len(inputs.train_idx),
        )
        model_config_hash = _hash_payload(model_config)
        X_train = inputs.X[inputs.train_idx]
        y_train = inputs.y[inputs.train_idx]
        X_valid = inputs.X[inputs.valid_idx]
        y_valid = inputs.y[inputs.valid_idx]
        train_started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        y_pred = np.asarray(estimator.predict(X_valid), dtype=np.float64)
        predict_time = time.perf_counter() - predict_started
        if y_pred.shape != y_valid.shape:
            raise ValueError(f"prediction shape mismatch: pred={y_pred.shape}, y={y_valid.shape}")
        if not np.isfinite(y_pred).all():
            raise ValueError("predictions contain NaN/inf")

        prediction_rows = _prediction_rows(inputs, model_id, y_pred)
        prediction_fields = [
            "prediction_index",
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
        ]
        _atomic_csv(predictions_path, prediction_rows, prediction_fields)

        metadata.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "model_config": model_config,
                "metrics": _metric_dict(y_valid, y_pred),
                "timing": {
                    "train_time_s": float(train_time),
                    "predict_time_s": float(predict_time),
                    "total_fit_predict_time_s": float(time.perf_counter() - started),
                },
                "hashes": {
                    "input_hash": inputs.input_hash,
                    "source_npz_sha256": inputs.inventory_entry["source_npz_sha256"],
                    "population_hash": inputs.population_hash,
                    "mapping_hash": inputs.mapping_hash,
                    "split_hash": inputs.split_hash,
                    "fold_hash": inputs.fold_hash,
                    "feature_hash": inputs.feature_hash,
                    "y_hash": inputs.y_hash,
                    "model_config_hash": model_config_hash,
                    "train_feature_hash": _sha256_array(X_train),
                    "valid_feature_hash": _sha256_array(X_valid),
                    "train_y_hash": _sha256_array(y_train),
                    "valid_y_hash": _sha256_array(y_valid),
                },
                "external_fold_contract": {
                    "train_valid_source": "step28_2 split_manifest",
                    "fit_valid_used_for_early_stopping": False,
                    "fit_valid_used_for_tuning": False,
                    "internal_holdout": False,
                    "row_selection_performed": False,
                    "ohe_excluded": True,
                    "autogluon_excluded": True,
                },
            }
        )
    except Exception as exc:  # pragma: no cover - exercised only on missing optional deps
        metadata.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "timing": {"total_fit_predict_time_s": float(time.perf_counter() - started)},
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "hashes": {
                    "input_hash": inputs.input_hash,
                    "population_hash": inputs.population_hash,
                    "split_hash": inputs.split_hash,
                    "fold_hash": inputs.fold_hash,
                    "feature_hash": inputs.feature_hash,
                },
            }
        )
    _atomic_json(metadata_path, metadata)
    _append_log(
        run_dir,
        "model_finished",
        {"model": model_id, "status": metadata["status"], "error": metadata.get("error_message", "")},
    )
    return metadata


def _write_run_outputs(run_dir: Path, metadata_rows: Sequence[Mapping[str, Any]], environment: Mapping[str, Any]) -> None:
    metric_rows = [_metric_row(row) for row in metadata_rows]
    metric_fields = list(metric_rows[0]) if metric_rows else []
    if metric_fields:
        _atomic_csv(run_dir / "fold_metrics.csv", metric_rows, metric_fields)

    combined_predictions: list[dict[str, Any]] = []
    for metadata in metadata_rows:
        if metadata.get("status") != "complete":
            continue
        combined_predictions.extend(_read_csv(PROJECT_ROOT / str(metadata["predictions_path"])))
    if combined_predictions:
        _atomic_csv(run_dir / "combined_predictions.csv", combined_predictions, list(combined_predictions[0]))

    complete_count = sum(1 for row in metadata_rows if row.get("status") == "complete")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete_count == len(MODEL_ORDER) else "incomplete",
        "purpose": "步骤28.3第一独立单元：SL1-static-1150 × PhysChem × r00/f00 × 四模型 smoke。",
        "run_dir": _repo_relative(run_dir),
        "created_at_utc": _utc_now(),
        "scope": {
            "population_id": SMOKE_POPULATION_ID,
            "dataset_id": SMOKE_DATASET_ID,
            "descriptor": SMOKE_DESCRIPTOR,
            "repeat_zero_based": SMOKE_REPEAT_ZERO_BASED,
            "fold_zero_based": SMOKE_FOLD_ZERO_BASED,
            "manifest_repeat": SMOKE_REPEAT,
            "manifest_fold": SMOKE_FOLD,
            "seed": SMOKE_SEED,
            "models": list(MODEL_ORDER),
            "excluded_from_scope": {"descriptors": sorted(FORBIDDEN_DESCRIPTORS), "models": sorted(FORBIDDEN_MODELS)},
        },
        "environment": environment,
        "outputs": {
            "task_manifest": _repo_relative(run_dir / "task_manifest.csv"),
            "fold_metrics": _repo_relative(run_dir / "fold_metrics.csv"),
            "combined_predictions": _repo_relative(run_dir / "combined_predictions.csv"),
            "command_log": _repo_relative(run_dir / "command_log.jsonl"),
        },
        "model_metadata": {
            row["model"]: {
                "status": row.get("status"),
                "metadata_path": row.get("metadata_path"),
                "predictions_path": row.get("predictions_path"),
                "model_config_hash": row.get("hashes", {}).get("model_config_hash", ""),
            }
            for row in metadata_rows
        },
        "complete_model_count": complete_count,
        "expected_model_count": len(MODEL_ORDER),
    }
    _atomic_json(run_dir / "environment.json", environment)
    _atomic_json(run_dir / "smoke_run_manifest.json", manifest)
    _write_task_manifest(run_dir, [_metric_row(row) for row in metadata_rows])


def _iter_sklearn_kfold_indices(n_rows: int, n_splits: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows, dtype=np.int64)
    shuffled = indices.copy()
    np.random.RandomState(seed).shuffle(shuffled)
    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=np.int64)
    fold_sizes[: n_rows % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        valid_selection = shuffled[start:stop]
        valid_mask = np.zeros(n_rows, dtype=bool)
        valid_mask[valid_selection] = True
        yield indices[~valid_mask], indices[valid_mask]
        current = stop


def _recompute_split_hash(population_rows: Sequence[Mapping[str, str]]) -> str:
    n_rows = len(population_rows)
    sample_ids = [str(row["sample_id"]) for row in population_rows]
    source_rows = [int(row["source_row_index"]) for row in population_rows]
    y_values = [float(row["y"]) for row in population_rows]
    population_id = str(population_rows[0]["population_id"])
    dataset_id = str(population_rows[0]["dataset_id"])
    population_hash = str(population_rows[0]["population_hash"])
    dataset_sha256 = str(population_rows[0]["candidate_source_csv_sha256"])
    fold_payloads: list[dict[str, Any]] = []
    for repeat in range(1, 6):
        seed = 1000 + repeat - 1
        for fold, (train_idx, valid_idx) in enumerate(_iter_sklearn_kfold_indices(n_rows, 5, seed), start=1):
            train_ids = [sample_ids[int(index)] for index in train_idx]
            valid_ids = [sample_ids[int(index)] for index in valid_idx]
            train_source_rows = [source_rows[int(index)] for index in train_idx]
            valid_source_rows = [source_rows[int(index)] for index in valid_idx]
            fold_payload = {
                "population_id": population_id,
                "dataset_id": dataset_id,
                "population_hash": population_hash,
                "repeat": repeat,
                "fold": fold,
                "seed": seed,
                "train_sample_ids": train_ids,
                "valid_sample_ids": valid_ids,
                "train_source_row_index": train_source_rows,
                "valid_source_row_index": valid_source_rows,
            }
            fold_hash = _hash_payload(fold_payload)
            fold_payloads.append(
                {
                    **fold_payload,
                    "fold_hash": fold_hash,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "train_sample_ids_hash": _hash_string_sequence(train_ids),
                    "valid_sample_ids_hash": _hash_string_sequence(valid_ids),
                    "train_source_row_index_hash": _sha256_int_sequence(train_source_rows),
                    "valid_source_row_index_hash": _sha256_int_sequence(valid_source_rows),
                    "train_y_hash": _sha256_array(np.asarray([y_values[int(index)] for index in train_idx], dtype=np.float64)),
                    "valid_y_hash": _sha256_array(np.asarray([y_values[int(index)] for index in valid_idx], dtype=np.float64)),
                }
            )
    return _hash_payload(
        {
            "schema_version": "1.0",
            "cv_config": {"strategy": "repeated_kfold", "n_repeats": 5, "n_splits": 5, "seed": 1000},
            "population_id": population_id,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "population_hash": population_hash,
            "folds": [
                {
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    "seed": fold["seed"],
                    "fold_hash": fold["fold_hash"],
                    "n_train": fold["n_train"],
                    "n_valid": fold["n_valid"],
                    "train_sample_ids_hash": fold["train_sample_ids_hash"],
                    "valid_sample_ids_hash": fold["valid_sample_ids_hash"],
                    "train_source_row_index_hash": fold["train_source_row_index_hash"],
                    "valid_source_row_index_hash": fold["valid_source_row_index_hash"],
                    "train_y_hash": fold["train_y_hash"],
                    "valid_y_hash": fold["valid_y_hash"],
                }
                for fold in fold_payloads
            ],
        }
    )


def _recompute_population_hash(population_rows: Sequence[Mapping[str, str]]) -> str:
    descriptor_rows = _read_csv(PROJECT_ROOT / STEP28_2_DIR / "descriptor_availability.csv")
    relevant = {
        row["descriptor"]: row
        for row in descriptor_rows
        if row["population_id"] == SMOKE_POPULATION_ID and row["dataset_id"] == SMOKE_DATASET_ID
    }
    if set(relevant) != set(DESCRIPTOR_ORDER):
        raise AssertionError("descriptor availability rows are incomplete for SL1")
    first = population_rows[0]
    row_identity = [
        {
            "population_position": int(row["population_position"]),
            "sample_id": row["sample_id"],
            "source_row_index": int(row["source_row_index"]),
        }
        for row in population_rows
    ]
    return _hash_payload(
        {
            "population_id": SMOKE_POPULATION_ID,
            "dataset_id": SMOKE_DATASET_ID,
            "target_unit": first["target_unit"],
            "row_identity": row_identity,
            "candidate_y_sha256": first["candidate_y_sha256"],
            "included_descriptors": list(DESCRIPTOR_ORDER),
            "excluded_descriptors": {},
            "descriptor_hashes": {
                descriptor: {
                    "source_npz_sha256": relevant[descriptor]["source_npz_sha256"],
                    "x_sha256": relevant[descriptor]["x_sha256"],
                    "y_sha256": relevant[descriptor]["y_sha256"],
                    "status": relevant[descriptor]["status"],
                }
                for descriptor in DESCRIPTOR_ORDER
            },
            "mapping_hash": first["mapping_hash"],
        }
    )


def _float_close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if left in ("", None) or right in ("", None):
        return left in ("", None) and right in ("", None)
    return abs(float(left) - float(right)) <= tolerance


def _verify_prediction_file(path: Path, inputs: SmokeInputs, model_id: str) -> dict[str, Any]:
    rows = _read_csv(path)
    if len(rows) != len(inputs.valid_rows):
        raise AssertionError(f"{model_id} prediction row count mismatch")
    expected_ids = [row["sample_id"] for row in inputs.valid_rows]
    expected_source_rows = [int(row["source_row_index"]) for row in inputs.valid_rows]
    observed_ids = [row["sample_id"] for row in rows]
    observed_source_rows = [int(row["source_row_index"]) for row in rows]
    if observed_ids != expected_ids or observed_source_rows != expected_source_rows:
        raise AssertionError(f"{model_id} valid sample/source rows changed")
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    if not np.array_equal(y_true, inputs.y[inputs.valid_idx]):
        raise AssertionError(f"{model_id} y_true does not match official y")
    if not np.isfinite(y_pred).all():
        raise AssertionError(f"{model_id} predictions contain NaN/inf")
    return {"rows": rows, "metrics": _metric_dict(y_true, y_pred)}


def verify_run(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    del repo_root
    run_dir = _assert_inside_repo(run_dir)
    manifest = json.loads((run_dir / "smoke_run_manifest.json").read_text(encoding="utf-8"))
    inputs = _load_inputs()
    if manifest["scope"]["population_id"] != SMOKE_POPULATION_ID:
        raise AssertionError("manifest population mismatch")
    if manifest["scope"]["descriptor"] != SMOKE_DESCRIPTOR:
        raise AssertionError("manifest descriptor mismatch")
    if set(manifest["scope"]["models"]) != set(MODEL_ORDER):
        raise AssertionError("manifest model set mismatch")
    if any(model.lower() in FORBIDDEN_MODELS for model in manifest["scope"]["models"]):
        raise AssertionError("AutoGluon must be excluded")
    if manifest["scope"]["descriptor"].lower() in FORBIDDEN_DESCRIPTORS:
        raise AssertionError("OHE must be excluded")
    if inputs.input_hash != inputs.inventory_entry["source_npz_sha256"]:
        raise AssertionError("input hash does not match inventory source_npz_sha256")
    descriptor_rows = _read_csv(PROJECT_ROOT / STEP28_2_DIR / "descriptor_availability.csv")
    physchem_rows = [
        row
        for row in descriptor_rows
        if row["population_id"] == SMOKE_POPULATION_ID and row["descriptor"] == SMOKE_DESCRIPTOR
    ]
    if len(physchem_rows) != 1:
        raise AssertionError("missing PhysChem descriptor availability row")
    if inputs.feature_hash != physchem_rows[0]["x_sha256"]:
        raise AssertionError("feature hash does not match step 28.2 descriptor availability")
    if inputs.y_hash != physchem_rows[0]["y_sha256"]:
        raise AssertionError("y hash does not match step 28.2 descriptor availability")
    if inputs.population_hash != _recompute_population_hash(inputs.population_rows):
        raise AssertionError("population_hash is not reproducible")
    if inputs.split_hash != _recompute_split_hash(inputs.population_rows):
        raise AssertionError("split_hash is not reproducible")

    fold_metrics = {row["model"]: row for row in _read_csv(run_dir / "fold_metrics.csv")}
    if set(fold_metrics) != set(MODEL_ORDER):
        raise AssertionError("fold_metrics must contain exactly four models")

    expected_valid_ids: list[str] | None = None
    expected_source_rows: list[int] | None = None
    complete_models: list[str] = []
    verified_metrics: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        metadata_path = run_dir / "models" / model_id / "fold_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise AssertionError(f"{model_id} is not complete")
        if metadata.get("descriptor", "").lower() in FORBIDDEN_DESCRIPTORS:
            raise AssertionError("OHE descriptor appeared in metadata")
        if model_id.lower() in FORBIDDEN_MODELS:
            raise AssertionError("AutoGluon appeared in metadata")
        if metadata["hashes"]["input_hash"] != inputs.input_hash:
            raise AssertionError(f"{model_id} input hash mismatch")
        if metadata["hashes"]["population_hash"] != inputs.population_hash:
            raise AssertionError(f"{model_id} population hash mismatch")
        if metadata["hashes"]["split_hash"] != inputs.split_hash:
            raise AssertionError(f"{model_id} split hash mismatch")
        if metadata["hashes"]["fold_hash"] != inputs.fold_hash:
            raise AssertionError(f"{model_id} fold hash mismatch")
        if metadata["hashes"]["feature_hash"] != inputs.feature_hash:
            raise AssertionError(f"{model_id} feature hash mismatch")
        if metadata["hashes"]["model_config_hash"] != _hash_payload(metadata["model_config"]):
            raise AssertionError(f"{model_id} model_config_hash mismatch")
        contract = metadata.get("external_fold_contract", {})
        if contract.get("internal_holdout") or contract.get("fit_valid_used_for_early_stopping"):
            raise AssertionError(f"{model_id} used internal holdout or external valid for tuning")
        if contract.get("row_selection_performed"):
            raise AssertionError(f"{model_id} performed row selection")
        if model_id == "rf":
            params = metadata["model_config"]["params"]
            if params.get("n_estimators") != 500 or params.get("max_features") != 0.3:
                raise AssertionError("RF paper-style parameters changed")
            if params.get("random_state") != SMOKE_SEED:
                raise AssertionError("RF random_state must be repeat seed 1000")

        prediction_check = _verify_prediction_file(
            run_dir / "models" / model_id / "predictions.csv",
            inputs,
            model_id,
        )
        observed_ids = [row["sample_id"] for row in prediction_check["rows"]]
        observed_source_rows = [int(row["source_row_index"]) for row in prediction_check["rows"]]
        if expected_valid_ids is None:
            expected_valid_ids = observed_ids
            expected_source_rows = observed_source_rows
        elif observed_ids != expected_valid_ids or observed_source_rows != expected_source_rows:
            raise AssertionError("valid samples differ across models")

        for metric_name, metric_value in prediction_check["metrics"].items():
            if metric_name in {"kendall_tau_defined", "kendall_tau_reason"}:
                continue
            if not _float_close(metric_value, metadata["metrics"].get(metric_name)):
                raise AssertionError(f"{model_id} metadata metric mismatch: {metric_name}")
            if not _float_close(metric_value, fold_metrics[model_id].get(metric_name)):
                raise AssertionError(f"{model_id} fold_metrics mismatch: {metric_name}")
        complete_models.append(model_id)
        verified_metrics[model_id] = prediction_check["metrics"]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "run_dir": _repo_relative(run_dir),
        "complete_model_count": len(complete_models),
        "expected_model_count": len(MODEL_ORDER),
        "models": complete_models,
        "valid_sample_count": len(expected_valid_ids or []),
        "train_sample_count": len(inputs.train_rows),
        "train_valid_overlap": False,
        "same_valid_samples_across_models": True,
        "ohe_present": False,
        "autogluon_present": False,
        "hashes_recomputed": {
            "input_hash": inputs.input_hash,
            "population_hash": inputs.population_hash,
            "split_hash": inputs.split_hash,
            "feature_hash": inputs.feature_hash,
        },
        "rf_parameters_verified": True,
        "no_internal_holdout_or_external_valid_tuning": True,
        "metrics_recomputed": verified_metrics,
    }
    _atomic_json(run_dir / "verification_summary.json", summary)
    return summary


def run_smoke(output_dir: Path, run_id: str) -> dict[str, Any]:
    _require_yonod_conda_env()
    output_dir = _assert_inside_repo(PROJECT_ROOT / output_dir)
    run_dir = output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists; choose a new --run-id: {run_dir}")
    run_dir.mkdir(parents=True)
    environment = _environment_summary(sys.argv)
    _atomic_json(run_dir / "environment.json", environment)
    _append_log(run_dir, "run_started", {"run_id": run_id, "command_line": sys.argv})
    inputs = _load_inputs()
    base = _task_base(inputs)
    planned_rows = [
        {
            **base,
            "model": model_id,
            "status": "planned",
            "model_config_hash": "",
            "predictions_path": "",
            "metadata_path": f"{_repo_relative(run_dir)}/models/{model_id}/fold_metadata.json",
            "error_message": "",
        }
        for model_id in MODEL_ORDER
    ]
    _write_task_manifest(run_dir, planned_rows)

    metadata_rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        metadata_rows.append(_run_one_model(run_dir, inputs, model_id))
        _write_run_outputs(run_dir, metadata_rows, environment)

    _write_run_outputs(run_dir, metadata_rows, environment)
    verification: dict[str, Any]
    try:
        verification = verify_run(PROJECT_ROOT, run_dir)
        _append_log(run_dir, "verification_finished", {"status": "passed"})
    except Exception as exc:
        verification = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "verified_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(run_dir / "verification_summary.json", verification)
        _append_log(run_dir, "verification_finished", {"status": "failed", "error": str(exc)})
    manifest = json.loads((run_dir / "smoke_run_manifest.json").read_text(encoding="utf-8"))
    manifest["verification"] = verification
    manifest["status"] = "complete" if verification.get("status") == "passed" else "incomplete"
    _atomic_json(run_dir / "smoke_run_manifest.json", manifest)
    if verification.get("status") != "passed":
        raise RuntimeError(f"smoke verification failed: {verification.get('error_message')}")
    return {"status": "complete", "run_dir": _repo_relative(run_dir), "verification": verification}


def _find_latest_run(output_dir: Path) -> Path:
    candidates = sorted(
        (PROJECT_ROOT / output_dir).glob("smoke_sl1_physchem_r00_f00_*/smoke_run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no smoke run manifests under {PROJECT_ROOT / output_dir}")
    return candidates[-1].parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verify", action="store_true", help="Verify an existing smoke run.")
    parser.add_argument("--run-dir", default=None, help="Repository-relative run directory for --verify.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.verify:
        run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run(output_dir)
        result = verify_run(PROJECT_ROOT, PROJECT_ROOT / run_dir)
    else:
        result = run_smoke(output_dir, args.run_id or _run_id())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
