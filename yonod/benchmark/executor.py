"""由外部 split manifest 驱动的单折 benchmark 执行器。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BenchmarkContract


class FoldExecutionError(RuntimeError):
    """Raised when a requested external fold cannot be executed safely."""


@dataclass(frozen=True)
class FoldExecutionResult:
    prediction_path: Path
    metadata_path: Path
    n_train: int
    n_valid: int
    train_time_s: float
    predict_time_s: float


def _software_versions() -> Dict[str, str]:
    """Return a small, serialisable environment fingerprint for one fold."""
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for distribution in ("numpy", "pandas", "scikit-learn", "pyarrow", "xgboost", "lightgbm"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _feature_schema_hash(X_smiles: np.ndarray, X_numeric: Optional[np.ndarray]) -> str:
    schema = {
        "smiles_shape": list(X_smiles.shape),
        "smiles_dtype": str(X_smiles.dtype),
        "numeric_shape": list(X_numeric.shape) if X_numeric is not None else None,
        "numeric_dtype": str(X_numeric.dtype) if X_numeric is not None else None,
    }
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _build_estimator(model_name: str, model_kwargs: Mapping[str, Any], n_features: int, n_train: int):
    """Return a single-fit estimator; legacy cross_validate remains untouched."""
    if model_name == "rf":
        from ..models.rf_model import RFYieldModel
        return RFYieldModel(**dict(model_kwargs))._build()
    if model_name == "xgb":
        from ..models.xgb_model import XGBYieldModel
        return XGBYieldModel(**dict(model_kwargs))._build()
    if model_name == "svm":
        from ..models.svm_model import SVMYieldModel
        adapter = SVMYieldModel(**dict(model_kwargs))
        return adapter._build(n_features=n_features, n_train=n_train)
    if model_name == "lightgbm":
        from ..models.lightgbm_model import LightGBMYieldModel
        return LightGBMYieldModel(**dict(model_kwargs))._build()
    raise FoldExecutionError(
        "模型 {0!r} 尚未具备严格外部 fold 适配器；AutoGluon 在其内部 holdout "
        "行为被验证前不得进入公平比较矩阵。".format(model_name)
    )


def _prepare_fold_features(
    X_smiles: np.ndarray,
    X_numeric: Optional[np.ndarray],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fold matrices, fitting numeric preprocessing on training rows only."""
    X_train, X_valid = X_smiles[train_idx], X_smiles[valid_idx]
    if X_numeric is None:
        return X_train, X_valid
    scaler = StandardScaler()
    numeric = np.asarray(X_numeric, dtype=float)
    return (
        np.hstack([X_train, scaler.fit_transform(numeric[train_idx])]),
        np.hstack([X_valid, scaler.transform(numeric[valid_idx])]),
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.equals(frame):
            return path
        raise FoldExecutionError("预测分片已存在且内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    verified = pd.read_parquet(temporary)
    if len(verified) != len(frame) or list(verified.columns) != list(frame.columns):
        temporary.unlink(missing_ok=True)
        raise FoldExecutionError("预测分片临时文件校验失败")
    temporary.replace(path)
    return path


def _atomic_write_json(payload: Dict[str, Any], path: Path) -> Path:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous == payload:
            return path
        raise FoldExecutionError("折级元数据已存在且内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _result_from_existing(prediction_path: Path, metadata_path: Path) -> FoldExecutionResult:
    """Return an already completed fold without retraining it."""
    if not prediction_path.exists() and not metadata_path.exists():
        raise FileNotFoundError
    if not prediction_path.exists() or not metadata_path.exists():
        raise FoldExecutionError(
            "发现不完整的折级输出（预测分片与元数据必须同时存在），拒绝静默重跑：{0}".format(
                prediction_path.parent
            )
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prediction = pd.read_parquet(prediction_path)
    except Exception as exc:  # pragma: no cover - exact parquet errors depend on backend
        raise FoldExecutionError("无法读取已有折级输出：{0}".format(exc)) from exc
    required = {"sample_id", "y_true", "y_pred"}
    if prediction.empty or not required.issubset(prediction.columns):
        raise FoldExecutionError("已有预测分片不完整，拒绝将其视为成功：{0}".format(prediction_path))
    try:
        return FoldExecutionResult(
            prediction_path=prediction_path,
            metadata_path=metadata_path,
            n_train=int(metadata["n_train"]),
            n_valid=int(metadata["n_valid"]),
            train_time_s=float(metadata["train_time_s"]),
            predict_time_s=float(metadata["predict_time_s"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FoldExecutionError("已有折级元数据字段不完整：{0}".format(metadata_path)) from exc


def execute_fold(
    contract: BenchmarkContract,
    split_manifest: pd.DataFrame,
    sample_ids: Sequence[str],
    X_smiles: np.ndarray,
    y: Sequence[float],
    descriptor: str,
    model: str,
    repeat: int,
    fold: int,
    X_numeric: Optional[np.ndarray] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> FoldExecutionResult:
    """Fit exactly one manifest-defined fold and write its prediction shard.

    Numeric columns are standardized inside the declared train fold; molecular
    features are passed through untouched because their construction is
    descriptor-specific and already complete before this boundary.
    """
    ids = np.asarray([str(value) for value in sample_ids], dtype=str)
    X_smiles = np.asarray(X_smiles)
    y_array = np.asarray(y, dtype=float)
    if len(ids) != len(X_smiles) or len(ids) != len(y_array):
        raise FoldExecutionError("sample_ids、X_smiles 与 y 的行数必须一致")
    if len(np.unique(ids)) != len(ids):
        raise FoldExecutionError("sample_ids 必须唯一，才能连接 split manifest")
    if X_numeric is not None and len(X_numeric) != len(ids):
        raise FoldExecutionError("X_numeric 与 sample_ids 的行数必须一致")
    if X_smiles.ndim != 2 or X_smiles.shape[1] == 0:
        raise FoldExecutionError("X_smiles 必须是至少包含一列特征的二维数组")
    if not np.isfinite(X_smiles).all() or not np.isfinite(y_array).all():
        raise FoldExecutionError("特征或标签包含 NaN 或 inf")
    if X_numeric is not None:
        numeric_array = np.asarray(X_numeric, dtype=float)
        if numeric_array.ndim != 2 or not np.isfinite(numeric_array).all():
            raise FoldExecutionError("X_numeric 必须是无 NaN/inf 的二维数组")
    part = split_manifest[(split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold)]
    if part.empty:
        raise FoldExecutionError("manifest 中不存在 repeat={0}, fold={1}".format(repeat, fold))
    valid_ids = set(part.loc[part["role"] == "valid", "sample_id"].astype(str))
    train_ids = set(part.loc[part["role"] == "train", "sample_id"].astype(str))
    if valid_ids.intersection(train_ids):
        raise FoldExecutionError("manifest 的 train/valid sample_id 发生重叠")
    if set(ids) != valid_ids.union(train_ids):
        raise FoldExecutionError("输入 sample_ids 与 manifest 当前折的样本集合不一致")
    train_idx = np.flatnonzero(np.isin(ids, list(train_ids)))
    valid_idx = np.flatnonzero(np.isin(ids, list(valid_ids)))
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise FoldExecutionError("manifest 当前折存在空训练集或验证集")
    suffix = "{0}__{1}__r{2:02d}__f{3:02d}".format(descriptor, model, repeat, fold)
    prediction_path = contract.run_dir / "predictions" / (suffix + ".parquet")
    metadata_path = contract.run_dir / "folds" / (suffix + ".json")
    try:
        return _result_from_existing(prediction_path, metadata_path)
    except FileNotFoundError:
        pass
    X_train, X_valid = _prepare_fold_features(X_smiles, X_numeric, train_idx, valid_idx)
    estimator = _build_estimator(model, model_kwargs or {}, X_train.shape[1], len(train_idx))
    started_at_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    estimator.fit(X_train, y_array[train_idx])
    train_time_s = time.perf_counter() - start
    start = time.perf_counter()
    prediction = np.asarray(estimator.predict(X_valid), dtype=float)
    predict_time_s = time.perf_counter() - start
    if not np.isfinite(prediction).all():
        raise FoldExecutionError("模型预测包含 NaN 或 inf")
    group_by_id = part.drop_duplicates("sample_id").set_index("sample_id")["group_id"].astype(str)
    split_id = str(part["split_id"].iloc[0])
    schema_hash = _feature_schema_hash(X_smiles, X_numeric)
    prediction_frame = pd.DataFrame({
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "split_id": split_id,
        "sample_id": ids[valid_idx],
        "group_id": [group_by_id.loc[sample_id] for sample_id in ids[valid_idx]],
        "descriptor": descriptor,
        "model": model,
        "repeat": repeat,
        "fold": fold,
        "y_true": y_array[valid_idx],
        "y_pred": prediction,
        "feature_schema_hash": schema_hash,
        "train_time_s": train_time_s,
        "predict_time_s": predict_time_s,
    })
    expected = set(ids[valid_idx])
    if len(prediction_frame) != len(expected) or set(prediction_frame["sample_id"]) != expected:
        raise FoldExecutionError("预测分片与 manifest 验证集不能一一对应")
    prediction_path = _atomic_write_parquet(prediction_frame, prediction_path)
    metadata = {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "split_id": split_id,
        "descriptor": descriptor,
        "model": model,
        "repeat": repeat,
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "feature_dim": int(X_train.shape[1]),
        "feature_schema_hash": schema_hash,
        "train_time_s": train_time_s,
        "predict_time_s": predict_time_s,
        "model_random_seed": dict(model_kwargs or {}).get("random_state", 42),
        "model_kwargs": dict(model_kwargs or {}),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": _software_versions(),
        "prediction_path": str(prediction_path),
    }
    metadata_path = _atomic_write_json(metadata, metadata_path)
    return FoldExecutionResult(prediction_path, metadata_path, len(train_idx), len(valid_idx), train_time_s, predict_time_s)
