"""Independent schema-2 training and ``all`` orchestration for step 30.7.

This is intentionally a consumer of :mod:`yonod.artifacts.reader`, not of a
descriptor generator.  A new interpreter can load a hash-verified feature
manifest, align labels by ``sample_id``, fit fold-local transformations, and
write a separate modelling run.  ``all`` is only a thin orchestration layer:
it first terminally materialises all selected features, then invokes the same
training routine for each successful package.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.contracts import artifact_content_identity
from yonod.artifacts.reader import ArtifactReadError, FeatureArtifact, load_feature_artifact
from yonod.config.contracts import resolve_config_path
from yonod.config.loader import LoadedRunConfig, load_run_config
from yonod.config.output_layout import combination_name, create_task_layout, task_root
from yonod.model_factory import (
    ModelConfigurationError,
    ResolvedModelConfig,
    construct_autogluon_predictor,
    construct_estimator,
    materialise_fit_parameters,
    resolve_model_config,
    software_versions,
)
from yonod.provenance import feature_dataset_identity
from yonod.features.numeric_conditions import (
    NumericConditionsError, NumericConditionsTransformer, normalise_numeric_contract,
    numeric_input_identity, load_numeric_block,
)


TRAINING_SCHEMA_VERSION = "1.0"


class TrainingServiceError(ValueError):
    """Raised when a train/all request cannot be executed independently."""


@dataclass(frozen=True)
class TrainingRun:
    """A durable, validated training result reference."""

    run_id: str
    artifact_id: str
    feature_id: str
    model: str
    status: str
    run_dir: Path
    manifest_path: Path
    predictions_path: Path
    fold_metrics_path: Path
    reason: str


@dataclass(frozen=True)
class AllRunResult:
    """Terminal references from the feature and training halves of ``all``."""

    feature_status_path: Path
    feature_status: str
    training_runs: tuple[TrainingRun, ...]


FoldRunner = Callable[
    [ResolvedModelConfig, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, Path, str, int],
    tuple[np.ndarray, Dict[str, Any]],
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        # A cheap round-trip catches an interrupted or malformed temporary
        # write before the directory is published.
        verified = pd.read_csv(temporary)
        if list(verified.columns) != list(frame.columns) or len(verified) != len(frame):
            raise TrainingServiceError("训练输出 CSV 临时文件校验失败")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_string_ids(series: pd.Series, field: str) -> np.ndarray:
    if series.isna().any():
        raise TrainingServiceError(f"{field} 不能包含空值")
    ids = series.astype(str).to_numpy(dtype=np.str_)
    if any(not value.strip() for value in ids.tolist()) or len(set(ids.tolist())) != len(ids):
        raise TrainingServiceError(f"{field} 必须是唯一的非空字符串")
    return ids


def _load_label_frame(loaded: LoadedRunConfig) -> tuple[pd.DataFrame, str, str, str]:
    config = loaded.effective
    dataset = config["dataset"]
    path = resolve_config_path(loaded.path, dataset["path"])
    if not path.is_file():
        raise TrainingServiceError(f"train 数据集不存在：{path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise TrainingServiceError(f"无法读取 train 数据集：{exc}") from exc
    id_col = str(dataset["sample_id_col"])
    label_col = str(dataset["column_roles"].get("label", ""))
    if id_col not in frame.columns or label_col not in frame.columns:
        raise TrainingServiceError("train 数据集缺少 sample_id_col 或标签列")
    ids = _as_string_ids(frame[id_col], f"dataset.{id_col}")
    frame = frame.copy()
    frame["__yonod_sample_id__"] = ids
    conditions = list(dataset["column_roles"].get("conditions", []) or [])
    if conditions:
        # pandas' default NA token inference would silently collapse strings
        # such as "NA". Only YAML-declared missing tokens may be filled.
        raw_numeric = pd.read_csv(path, usecols=[id_col, *conditions], keep_default_na=False)
        if len(raw_numeric) != len(frame) or raw_numeric[id_col].astype(str).tolist() != ids.tolist():
            raise TrainingServiceError("数值原始列的 sample_id/行序与训练数据不一致")
        for column in conditions:
            frame[column] = raw_numeric[column].tolist()
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise TrainingServiceError(f"标签列 {label_col!r} 包含不可解析、NaN 或 inf 值")
    frame[label_col] = labels.astype(float)
    try:
        feature_identity = feature_dataset_identity(
            frame,
            sample_id_col=id_col,
            column_roles=dataset["column_roles"],
        )
    except ValueError as exc:
        raise TrainingServiceError(str(exc)) from exc
    return frame, id_col, label_col, feature_identity


def _align_artifact_to_labels(
    artifact: FeatureArtifact,
    frame: pd.DataFrame,
    label_col: str,
    dataset_identity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    source_identity = str(artifact.manifest["sources"]["dataset_identity"])
    if source_identity != dataset_identity:
        raise TrainingServiceError(
            "特征包 dataset_identity 与训练数据内容不一致；请使用产生该 manifest 的数据文件，"
            "或显式重新计算特征"
        )
    valid_ids = artifact.sample_ids[artifact.valid_mask].astype(str)
    lookup = frame.set_index("__yonod_sample_id__", drop=False)
    missing = [sample_id for sample_id in valid_ids.tolist() if sample_id not in lookup.index]
    if missing:
        preview = ", ".join(missing[:5])
        raise TrainingServiceError(f"训练数据缺少特征包中的有效 sample_id：{preview}")
    aligned = lookup.loc[valid_ids.tolist()].copy()
    if len(aligned) != len(valid_ids) or aligned["__yonod_sample_id__"].tolist() != valid_ids.tolist():
        raise TrainingServiceError("标签/辅助表未能按 sample_id 与特征包一对一对齐")
    labels = aligned[label_col].to_numpy(dtype=float)
    matrix = np.asarray(artifact.matrix)
    if len(matrix) != len(labels):
        raise TrainingServiceError("特征矩阵与有效标签行数不一致")
    return valid_ids, matrix, labels, aligned.reset_index(drop=True)


def _evaluation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(config.get("evaluation") or {})
    protocol = str(raw.get("protocol", "outer_kfold"))
    if protocol not in {"outer_kfold", "manifest_outer_cv"}:
        raise TrainingServiceError(
            "普通独立 train 目前仅支持 evaluation.protocol=outer_kfold 或 manifest_outer_cv；"
            "paper_exact 仍必须经其专用协议入口"
        )
    n_splits = raw.get("n_splits", 5)
    n_repeats = raw.get("n_repeats", 1)
    seed = raw.get("seed", 42)
    shuffle = raw.get("shuffle", True)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise TrainingServiceError("evaluation.n_splits 必须为不小于 2 的整数")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise TrainingServiceError("evaluation.n_repeats 必须为不小于 1 的整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingServiceError("evaluation.seed 必须为整数")
    if not isinstance(shuffle, bool):
        raise TrainingServiceError("evaluation.shuffle 必须为 bool")
    return {
        "protocol": protocol,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "shuffle": shuffle,
    }


def _split_rows(sample_ids: np.ndarray, evaluation: Mapping[str, Any]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    if len(sample_ids) < int(evaluation["n_splits"]):
        raise TrainingServiceError("有效特征样本数必须不少于 evaluation.n_splits")
    from sklearn.model_selection import KFold

    rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(1, int(evaluation["n_repeats"]) + 1):
        # A repeat has its own deterministic outer split while the explicit
        # no-shuffle case retains sklearn's required ``random_state=None``.
        random_state = int(evaluation["seed"]) + repeat - 1 if evaluation["shuffle"] else None
        splitter = KFold(n_splits=int(evaluation["n_splits"]), shuffle=bool(evaluation["shuffle"]), random_state=random_state)
        for fold, (train_index, valid_index) in enumerate(splitter.split(sample_ids), start=1):
            rows.append((repeat, fold, train_index.astype(int), valid_index.astype(int)))
    return rows


def _split_identity(sample_ids: np.ndarray, rows: Sequence[tuple[int, int, np.ndarray, np.ndarray]], evaluation: Mapping[str, Any]) -> str:
    payload = {
        "evaluation": dict(evaluation),
        "folds": [
            {
                "repeat": repeat,
                "fold": fold,
                "train_ids": sample_ids[train_index].astype(str).tolist(),
                "valid_ids": sample_ids[valid_index].astype(str).tolist(),
            }
            for repeat, fold, train_index, valid_index in rows
        ],
    }
    return "sha256:" + _sha256_text(_canonical_json(payload))


def _append_fold_numeric(
    config: Mapping[str, Any],
    aligned: pd.DataFrame,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    sample_ids: np.ndarray,
    fold_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    contract = normalise_numeric_contract(config["dataset"])
    if not contract["columns"]:
        return X_train, X_valid, {"columns": [], "fit_scope": "not_used"}
    try:
        transformer = NumericConditionsTransformer(contract)
        train_values = transformer.fit_transform(
            aligned.iloc[train_index], sample_ids[train_index], phase="outer_train"
        )
        valid_values = transformer.transform(
            aligned.iloc[valid_index], sample_ids[valid_index], phase="outer_valid"
        )
    except NumericConditionsError as exc:
        raise TrainingServiceError(str(exc)) from exc
    fold_dir.mkdir(parents=True, exist_ok=True)
    _atomic_yaml(fold_dir / "numeric_conditions.yaml", transformer.state_dict())
    return (
        np.hstack([X_train, train_values]),
        np.hstack([X_valid, valid_values]),
        {
            "columns": [entry["name"] for entry in contract["columns"]],
            "fit_scope": "training_fold_only",
            "state_sha256": _sha256_file(fold_dir / "numeric_conditions.yaml"),
            "fit_sample_ids_sha256": transformer.state_dict()["fit_sample_ids_sha256"],
            "state_path": "numeric_conditions.yaml",
            "dimension": len(contract["columns"]),
        },
    )


def _fold_matrix(
    artifact: FeatureArtifact,
    config: Mapping[str, Any],
    aligned: pd.DataFrame,
    sample_ids: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    fold_dir: Path,
    context: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if artifact.manifest["lifecycle"] != "fold_transform":
        train = np.asarray(artifact.matrix[train_index], dtype=float)
        valid = np.asarray(artifact.matrix[valid_index], dtype=float)
        if not np.isfinite(train).all() or not np.isfinite(valid).all():
            raise TrainingServiceError("静态特征包含 NaN 或 inf")
        train, valid, numeric_audit = _append_fold_numeric(
            config, aligned, train_index, valid_index, train, valid, sample_ids, fold_dir
        )
        return train, valid, {"lifecycle": "static_descriptor", "numeric": numeric_audit}

    transform = artifact.manifest.get("metadata", {}).get("fold_transform", {})
    if transform.get("transform") != "ohe" or transform.get("fit_scope") != "training_fold_only":
        raise TrainingServiceError("fold_transform 产物没有可执行的训练折内 OHE 声明")
    columns = list(transform.get("input_columns", []))
    parameters = dict(transform.get("parameters", {}))
    if not columns or artifact.matrix.shape[1] != len(columns):
        raise TrainingServiceError("fold_transform 原始类别矩阵与列声明不一致")
    # Reconstruct a frame from the verified package, not from whatever order
    # happens to be in the current CSV.  This preserves the artifact's exact
    # sample-ID mapping and prevents a hidden data-column substitution.
    raw = pd.DataFrame(np.asarray(artifact.matrix, dtype=str), columns=columns)
    sentinel = str(transform.get("missing_sentinel", ""))
    if not sentinel:
        raise TrainingServiceError("fold_transform 缺少受保护的缺失值标记")
    raw = raw.mask(raw.eq(sentinel), other=pd.NA)
    try:
        from yonod.descriptors.ohe import OHEFeature
        transformer = OHEFeature(
            columns,
            missing_policy=str(parameters.get("missing_policy", "as_category")),
            dtype=str(parameters.get("dtype", "float32")),
            handle_unknown=str(parameters.get("handle_unknown", "ignore")),
        )
        transformer.fit(raw.iloc[train_index], sample_ids=sample_ids[train_index])
        train = transformer.transform(raw.iloc[train_index], partition="train")
        valid = transformer.transform(raw.iloc[valid_index], partition="valid")
        fold_dir.mkdir(parents=True, exist_ok=True)
        transformer.save(fold_dir / "transform", context=context)
        transform_audit = transformer.metadata()
    except Exception as exc:
        raise TrainingServiceError(f"训练折 OHE 失败：{exc}") from exc
    train, valid, numeric_audit = _append_fold_numeric(
        config, aligned, train_index, valid_index, train, valid, sample_ids, fold_dir
    )
    return train, valid, {
        "lifecycle": "fold_transform",
        "ohe": transform_audit,
        "numeric": numeric_audit,
    }


def _inner_early_stopping_indices(
    n_rows: int,
    declaration: Mapping[str, Any],
    *,
    fold_number: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split an outer training fold for model selection without seeing outer valid.

    This intentionally operates on the already outer-train-only matrix.  The
    resulting inner validation rows can guide tree early stopping, but never
    enter the outer-fold metric or prediction path.
    """
    if n_rows < 3:
        raise TrainingServiceError("启用 inner early stopping 时每个外层训练折至少需要 3 行")
    fraction = float(declaration["validation_fraction"])
    n_valid = max(1, int(np.ceil(n_rows * fraction)))
    if n_valid >= n_rows:
        raise TrainingServiceError("inner early stopping 的 validation_fraction 使内部训练集为空")
    seed = int(declaration["seed"]) + int(fold_number) - 1
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return permutation[n_valid:].astype(int), permutation[:n_valid].astype(int), seed


def _early_stopping_audit(
    *,
    declaration: Mapping[str, Any],
    train_index: np.ndarray,
    inner_train: np.ndarray,
    inner_valid: np.ndarray,
    seed: int,
    best_iteration: Any,
) -> Dict[str, Any]:
    return {
        "fit_scope": "inner_split_of_outer_training_fold_only",
        "outer_training_rows": int(len(train_index)),
        "inner_training_rows": int(len(inner_train)),
        "inner_validation_rows": int(len(inner_valid)),
        "validation_fraction": float(declaration["validation_fraction"]),
        "rounds": int(declaration["rounds"]),
        "random_state": int(seed),
        "inner_training_rows_sha256": _sha256_text(
            _canonical_json(train_index[inner_train].astype(int).tolist())
        ),
        "inner_validation_rows_sha256": _sha256_text(
            _canonical_json(train_index[inner_valid].astype(int).tolist())
        ),
        "best_iteration": None if best_iteration is None else int(best_iteration),
    }


def _fit_with_inner_early_stopping(
    resolved: ResolvedModelConfig,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    fit_params: Mapping[str, Any],
    train_index: np.ndarray,
    *,
    fold_number: int,
) -> tuple[np.ndarray, float, float, Dict[str, Any]]:
    """Fit tree models with an inner split generated from outer-train rows."""
    declaration = resolved.runtime.get("early_stopping")
    if not isinstance(declaration, Mapping):
        raise TrainingServiceError("early_stopping 声明无效")
    inner_train, inner_valid, seed = _inner_early_stopping_indices(
        len(X_train), declaration, fold_number=fold_number
    )
    kwargs = dict(fit_params)
    weight = kwargs.get("sample_weight")
    if isinstance(weight, np.ndarray):
        kwargs["sample_weight"] = weight[inner_train]
    if resolved.model == "xgb":
        kwargs["eval_set"] = [(X_train[inner_valid], y_train[inner_valid])]
        kwargs.setdefault("verbose", False)
        if isinstance(weight, np.ndarray):
            kwargs["sample_weight_eval_set"] = [weight[inner_valid]]
    elif resolved.model == "lightgbm":
        try:
            from lightgbm import early_stopping
        except ImportError as exc:  # constructor already checks, keep the failure clear
            raise TrainingServiceError("LightGBM 未安装，无法使用 inner early stopping") from exc
        kwargs["eval_set"] = [(X_train[inner_valid], y_train[inner_valid])]
        kwargs["callbacks"] = [early_stopping(int(declaration["rounds"]), verbose=False)]
        if isinstance(weight, np.ndarray):
            kwargs["eval_sample_weight"] = [weight[inner_valid]]
    else:  # defensive: runtime validation should have excluded every other model
        raise TrainingServiceError("inner early stopping 仅支持 xgb 或 lightgbm")

    started = time.perf_counter()
    estimator.fit(X_train[inner_train], y_train[inner_train], **kwargs)
    train_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(estimator.predict(X_valid), dtype=float)
    predict_time = time.perf_counter() - started
    best = getattr(estimator, "best_iteration", None)
    if best is None:
        best = getattr(estimator, "best_iteration_", None)
    return prediction, train_time, predict_time, _early_stopping_audit(
        declaration=declaration,
        train_index=train_index,
        inner_train=inner_train,
        inner_valid=inner_valid,
        seed=seed,
        best_iteration=best,
    )


def _default_fold_runner(
    resolved: ResolvedModelConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    aligned: pd.DataFrame,
    train_index: np.ndarray,
    fold_dir: Path,
    label_col: str,
    fold_number: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    fit_params = materialise_fit_parameters(resolved, aligned, train_index)
    if resolved.model != "autogluon":
        estimator, audit = construct_estimator(resolved, n_features=X_train.shape[1], n_train=len(X_train))
        if resolved.model in {"xgb", "lightgbm"} and "early_stopping" in resolved.runtime:
            prediction, train_time, predict_time, early_stopping_audit = _fit_with_inner_early_stopping(
                resolved, estimator, X_train, y_train, X_valid, fit_params, train_index, fold_number=fold_number
            )
            model_file = _persist_fold_estimator(estimator, fold_dir)
            return prediction, {
                **audit,
                **model_file,
                "effective_fit_parameters": {
                    key: ("<training_fold_array>" if isinstance(value, np.ndarray) else value)
                    for key, value in fit_params.items()
                },
                "early_stopping": early_stopping_audit,
                "train_time_s": float(train_time),
                "predict_time_s": float(predict_time),
                "software_versions": software_versions(resolved.model),
            }
        svm_subsample_audit: Dict[str, Any] | None = None
        if resolved.model == "svm":
            subsample = resolved.preprocessing.get("subsample")
            requested = None if subsample is None else subsample.get("n_samples")
            if requested is not None and int(requested) < len(X_train):
                # The historical SVM adapter fit scaler/PCA on every outer
                # training row, then reduced only the expensive SVR fit.
                # Calling Pipeline.fit on the sampled rows would subtly
                # change that policy and let `subsample` alter preprocessing
                # statistics, so fit/transform the prefix explicitly first.
                from sklearn.pipeline import Pipeline

                if not isinstance(estimator, Pipeline):  # defensive guard for a factory regression
                    raise TrainingServiceError("SVM 工厂没有返回 sklearn Pipeline")
                prefix = estimator[:-1]
                transformed_train = (
                    prefix.fit_transform(X_train, y_train)
                    if prefix.steps else np.asarray(X_train, dtype=float)
                )
                seed = int(subsample.get("random_state", 42)) + int(fold_number) - 1
                selection = np.random.default_rng(seed).choice(
                    len(X_train), size=int(requested), replace=False
                )
                final_fit_params: Dict[str, Any] = {}
                for key, value in fit_params.items():
                    if not key.startswith("svr__"):
                        raise TrainingServiceError(f"SVM fit 参数未路由到最终 svr：{key}")
                    final_key = key[len("svr__"):]
                    final_fit_params[final_key] = value[selection] if isinstance(value, np.ndarray) else value
                started = time.perf_counter()
                estimator.named_steps["svr"].fit(transformed_train[selection], y_train[selection], **final_fit_params)
                train_time = time.perf_counter() - started
                started = time.perf_counter()
                prediction = np.asarray(estimator.predict(X_valid), dtype=float)
                predict_time = time.perf_counter() - started
                svm_subsample_audit = {
                    "fit_scope": "svr_training_fold_subset_only",
                    "preprocessing_fit_rows": int(len(X_train)),
                    "svr_fit_rows": int(len(selection)),
                    "requested_n_samples": int(requested),
                    "random_state": seed,
                    "selected_training_rows_sha256": _sha256_text(
                        _canonical_json(train_index[selection].astype(int).tolist())
                    ),
                }
                model_file = _persist_fold_estimator(estimator, fold_dir)
                return prediction, {
                    **audit,
                    **model_file,
                    "effective_fit_parameters": {
                        key: ("<training_fold_array>" if isinstance(value, np.ndarray) else value)
                        for key, value in fit_params.items()
                    },
                    "svm_subsample": svm_subsample_audit,
                    "train_time_s": float(train_time),
                    "predict_time_s": float(predict_time),
                    "software_versions": software_versions(resolved.model),
                }
        started = time.perf_counter()
        estimator.fit(X_train, y_train, **fit_params)
        train_time = time.perf_counter() - started
        started = time.perf_counter()
        prediction = np.asarray(estimator.predict(X_valid), dtype=float)
        predict_time = time.perf_counter() - started
        model_file = _persist_fold_estimator(estimator, fold_dir)
        return prediction, {
            **audit,
            **model_file,
            "effective_fit_parameters": {key: ("<training_fold_array>" if isinstance(value, np.ndarray) else value) for key, value in fit_params.items()},
            **({"svm_subsample": svm_subsample_audit} if svm_subsample_audit is not None else {}),
            "train_time_s": float(train_time),
            "predict_time_s": float(predict_time),
            "software_versions": software_versions(resolved.model),
        }

    predictor_path = Path(resolved.runtime.get("save_path") or fold_dir / "autogluon")
    predictor, audit = construct_autogluon_predictor(resolved, label=label_col, path=predictor_path)
    train_frame = pd.DataFrame(X_train, columns=[f"f{index}" for index in range(X_train.shape[1])])
    valid_frame = pd.DataFrame(X_valid, columns=[f"f{index}" for index in range(X_valid.shape[1])])
    train_frame[label_col] = y_train
    fit_kwargs = dict(fit_params)
    fit_kwargs["train_data"] = train_frame
    started = time.perf_counter()
    predictor.fit(**fit_kwargs)
    train_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(predictor.predict(valid_frame), dtype=float)
    predict_time = time.perf_counter() - started
    dynamic: Dict[str, Any] = {}
    try:
        leaderboard = predictor.leaderboard(silent=True)
        dynamic["leaderboard"] = leaderboard.to_dict(orient="records")
    except Exception as exc:  # data-dependent optional evidence, never fabricated
        dynamic["leaderboard_error"] = f"{type(exc).__name__}: {exc}"
    cleanup = bool(resolved.runtime.get("cleanup", False))
    if cleanup:
        shutil.rmtree(predictor_path, ignore_errors=True)
    return prediction, {
        **audit,
        "effective_fit_parameters": {key: ("<training_fold_array>" if isinstance(value, np.ndarray) else value) for key, value in fit_params.items() if key != "train_data"},
        "dynamic_fit_evidence": dynamic,
        "train_time_s": float(train_time),
        "predict_time_s": float(predict_time),
        "model_artifact_cleanup": cleanup,
        "software_versions": software_versions(resolved.model),
    }


def _persist_fold_estimator(estimator: Any, fold_dir: Path) -> Dict[str, Any]:
    """Bind the fitted estimator to this fold's independently fitted inputs."""
    import joblib

    fold_dir.mkdir(parents=True, exist_ok=True)
    path = fold_dir / "model.joblib"
    joblib.dump(estimator, path, compress=3)
    return {"model_file": "model.joblib", "model_sha256": _sha256_file(path)}


def _run_identity(
    artifact: FeatureArtifact,
    *,
    model: str,
    model_config: ResolvedModelConfig,
    label_identity: str,
    split_identity: str,
    numeric_identity: str | None = None,
    hpo_identity: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "feature_content_identity": artifact_content_identity(artifact.manifest),
        "artifact_id": artifact.manifest["artifact_id"],
        "model": model,
        "model_config": model_config.as_audit_dict(),
        "label_identity": label_identity,
        "split_identity": split_identity,
    }
    if numeric_identity is not None:
        payload["numeric_identity"] = numeric_identity
    if hpo_identity is not None:
        payload["hpo"] = dict(hpo_identity)
    return "train-" + _sha256_text(_canonical_json(payload))[:20]


def _label_identity(sample_ids: np.ndarray, labels: np.ndarray, dataset_identity: str, label_col: str) -> str:
    return "sha256:" + _sha256_text(_canonical_json({
        "dataset_identity": dataset_identity,
        "label_column": label_col,
        "sample_ids": sample_ids.astype(str).tolist(),
        "labels": np.asarray(labels, dtype=float).tolist(),
    }))


def _existing_complete_run(run_dir: Path, expected_run_id: str) -> TrainingRun | None:
    manifest_path = run_dir / "run_manifest.yaml"
    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "fold_metrics.csv"
    effective_config_path = run_dir / "effective_config.yaml"
    source_config_path = run_dir / "source_config.yaml"
    if not run_dir.exists():
        return None
    if not manifest_path.exists() and {p.name for p in run_dir.iterdir()} == {"failed_attempts"}:
        return None
    if not all(path.is_file() for path in (manifest_path, predictions_path, metrics_path, effective_config_path, source_config_path)):
        raise TrainingServiceError(f"已有训练目录不完整，拒绝错误恢复：{run_dir}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
        metrics = pd.read_csv(metrics_path)
    except Exception as exc:
        raise TrainingServiceError(f"无法读取已有训练结果：{exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("run_id") != expected_run_id:
        raise TrainingServiceError("已有训练目录的 run_id 与本次身份不一致，拒绝覆盖")
    hpo_payload = payload.get("hpo")
    if hpo_payload is not None:
        task_root_path = run_dir.parent.parent.resolve()
        hpo_audits = hpo_payload.get("outer_folds", [])
        if not isinstance(hpo_audits, list) or len(hpo_audits) != int(payload.get("n_folds", -1)):
            raise TrainingServiceError("已有 HPO 结果缺少完整逐外层折 study")
        final_info = hpo_payload.get("final_model")
        if hpo_payload.get("declaration", {}).get("final_model", {}).get("enabled") and not isinstance(final_info, Mapping):
            raise TrainingServiceError("已有 HPO 结果缺少显式最终模型审计")
        all_studies = list(hpo_audits)
        if isinstance(final_info, Mapping):
            if final_info.get("purpose") != "final_model" or not isinstance(final_info.get("study"), Mapping):
                raise TrainingServiceError("已有最终模型 study 用途无效")
            all_studies.append(final_info["study"])
            final_bundle = run_dir / str(final_info.get("bundle_path", ""))
            if not final_bundle.is_file() or _sha256_file(final_bundle) != final_info.get("bundle_sha256"):
                raise TrainingServiceError("已有最终模型包缺失或哈希不匹配")
        for fold_audit in all_studies:
            try:
                study_path = (task_root_path / str(fold_audit["study_dir"])).resolve()
                study_path.relative_to(task_root_path)
                expected_files = (
                    (study_path / "study_manifest.json", fold_audit["study_manifest_sha256"]),
                    (study_path / "trials.json", fold_audit["study_trials_sha256"]),
                )
                if any(not path.is_file() or _sha256_file(path) != checksum for path, checksum in expected_files):
                    raise TrainingServiceError("已有 HPO study/试验审计缺失或哈希不匹配")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise TrainingServiceError("已有 HPO 折缺少可核验的持久 study") from exc
    required_prediction = {"sample_id", "y_true", "y_pred", "repeat", "fold"}
    if predictions.empty or not required_prediction.issubset(predictions.columns) or metrics.empty:
        raise TrainingServiceError("已有训练结果缺少完整预测或 fold 指标，拒绝恢复")
    for name in payload.get("model_bundles") or []:
        bundle_path = run_dir / str(name)
        if not bundle_path.is_file():
            raise TrainingServiceError(f"已有模型包缺失，拒绝复用：{bundle_path}")
        bundle_hashes = payload.get("model_bundle_sha256")
        if hpo_payload is not None and (not isinstance(bundle_hashes, Mapping) or
                                        _sha256_file(bundle_path) != bundle_hashes.get(str(name))):
            raise TrainingServiceError(f"已有 HPO 模型包与运行 manifest 哈希不匹配：{bundle_path}")
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(bundle, Mapping) or bundle.get("run_id") != expected_run_id:
            raise TrainingServiceError(f"已有模型包身份不一致，拒绝复用：{bundle_path}")
        model_path = run_dir / str(bundle.get("model_path", ""))
        if not model_path.is_file() or _sha256_file(model_path) != bundle.get("model_sha256"):
            raise TrainingServiceError(f"已有模型文件缺失或篡改，拒绝复用：{model_path}")
        if bundle.get("numeric_state_path"):
            state_path = run_dir / str(bundle["numeric_state_path"])
            if not state_path.is_file() or _sha256_file(state_path) != bundle.get("numeric_state_sha256"):
                raise TrainingServiceError(f"已有数值拟合状态缺失或篡改，拒绝复用：{state_path}")
        if bundle.get("ohe_state_path") and (bundle.get("hpo") or bundle.get("purpose") == "final_model"):
            metadata_path = run_dir / str(bundle["ohe_state_path"]) / "metadata.json"
            if not metadata_path.is_file() or _sha256_file(metadata_path) != bundle.get("ohe_state_metadata_sha256"):
                raise TrainingServiceError(f"已有 OHE 状态元数据缺失或篡改，拒绝复用：{metadata_path}")
    return TrainingRun(
        run_id=expected_run_id,
        artifact_id=str(payload.get("artifact_id")),
        feature_id=str(payload.get("feature_id")),
        model=str(payload.get("model")),
        status="reused",
        run_dir=run_dir,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        fold_metrics_path=metrics_path,
        reason="身份、预测和折指标均已验证，复用完成结果",
    )


def _publish_failed_run(
    loaded: LoadedRunConfig,
    config: Mapping[str, Any],
    artifact: FeatureArtifact,
    *,
    model: str,
    resolved: ResolvedModelConfig,
    run_id: str,
    dataset_identity: str,
    label_identity: str,
    split_identity: str,
    evaluation: Mapping[str, Any],
    n_samples: int,
    n_folds: int,
    cause: Exception,
    hpo_declaration: Mapping[str, Any] | None = None,
    hpo_folds: Sequence[Mapping[str, Any]] = (),
) -> TrainingRun:
    """Publish immutable failure evidence without disturbing other combinations.

    A failed model identity is useful evidence, but it is never a resumable
    complete run. Each attempt receives its own failed_attempts directory, so a
    later retry cannot overwrite this evidence or a successful sibling.
    """
    result_root = _result_root(loaded, config)
    name = combination_name(str(artifact.manifest["feature_id"]), model)
    failure_parent = result_root / "runs" / name / "failed_attempts"
    failure_parent.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex[:12]
    failure_dir = failure_parent / f"{run_id}-{attempt_id}"
    staging = failure_parent / f".{run_id}-{attempt_id}.tmp"
    reason = f"{type(cause).__name__}: {cause}"
    from yonod.hpo.budget import TaskBudgetExpired
    status = "incomplete" if isinstance(cause, TaskBudgetExpired) else "failed"
    try:
        staging.mkdir()
        shutil.copy2(loaded.path, staging / "source_config.yaml")
        _atomic_yaml(staging / "effective_config.yaml", config)
        manifest = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "kind": "yonod_training_run",
            "status": status,
            "run_id": run_id,
            "artifact_id": artifact.manifest["artifact_id"],
            "feature_id": artifact.manifest["feature_id"],
            "feature_content_identity": artifact_content_identity(artifact.manifest),
            "model": model,
            "model_request": resolved.as_audit_dict(),
            **({"hpo": {"declaration": dict(hpo_declaration),
                        "outer_folds": list(hpo_folds),
                        "completed_outer_folds": len(hpo_folds),
                        "expected_outer_folds": int(n_folds)}} if hpo_declaration is not None else {}),
            "dataset_identity": dataset_identity,
            "label_identity": label_identity,
            "label_column": str(config["dataset"]["column_roles"]["label"]),
            "split_identity": split_identity,
            "evaluation": dict(evaluation),
            "n_samples": int(n_samples),
            "n_folds": int(n_folds),
            "source_config_filename": loaded.path.name,
            "source_config_sha256": _sha256_file(staging / "source_config.yaml"),
            "effective_config_sha256": _sha256_file(staging / "effective_config.yaml"),
            "dependency_versions": software_versions(model),
            "failure": {
                "phase": "soft_task_budget" if status == "incomplete" else "fold_execution_or_result_publication",
                "reason": reason,
            },
            "outputs": {
                "predictions": None,
                "fold_metrics": None,
                "source_config": "source_config.yaml",
                "effective_config": "effective_config.yaml",
            },
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_yaml(staging / "run_manifest.yaml", manifest)
        os.replace(staging, failure_dir)
    except Exception as publish_error:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise TrainingServiceError(
            f"训练组合失败后无法原子发布失败证据：{type(publish_error).__name__}: {publish_error}"
        ) from publish_error
    return TrainingRun(
        run_id=run_id,
        artifact_id=str(artifact.manifest["artifact_id"]),
        feature_id=str(artifact.manifest["feature_id"]),
        model=model,
        status=status,
        run_dir=failure_dir,
        manifest_path=failure_dir / "run_manifest.yaml",
        predictions_path=failure_dir / "predictions.csv",
        fold_metrics_path=failure_dir / "fold_metrics.csv",
        reason=reason,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
    )


def _result_root(loaded: LoadedRunConfig, config: Mapping[str, Any]) -> Path:
    return task_root(loaded.path, config)


def _task_budget_context(loaded: LoadedRunConfig, config: Mapping[str, Any]):
    """Share one active HPO task clock across features and all model combinations."""
    from contextlib import nullcontext

    hpo = config.get("hpo") or {}
    if not hpo.get("enabled") or hpo.get("budget", {}).get("task_timeout_s") is None:
        return nullcontext(None)
    from yonod.hpo.budget import ActiveBudgetLedger
    from yonod.hpo.contracts import validate_hpo_declaration

    normalized = validate_hpo_declaration(
        hpo, models=config["models"], stage=config["stage"],
        evaluation=config.get("evaluation"), model_params=config.get("model_params"),
    )
    dataset_path = resolve_config_path(loaded.path, config["dataset"]["path"])
    return ActiveBudgetLedger(
        _result_root(loaded, config),
        {"hpo": normalized, "dataset_sha256": _sha256_file(dataset_path)},
        normalized["budget"]["task_timeout_s"],
    )


def _run_one_artifact(
    loaded: LoadedRunConfig,
    artifact_path: Path,
    *,
    model: str,
    fold_runner: FoldRunner | None = None,
    task_budget: Any = None,
) -> TrainingRun:
    config = loaded.effective
    try:
        artifact = load_feature_artifact(artifact_path)
    except ArtifactReadError as exc:
        raise TrainingServiceError(f"训练输入 manifest 无效：{exc}") from exc
    frame, _id_col, label_col, dataset_identity = _load_label_frame(loaded)
    sample_ids, matrix, labels, aligned = _align_artifact_to_labels(artifact, frame, label_col, dataset_identity)
    evaluation = _evaluation_config(config)
    rows = _split_rows(sample_ids, evaluation)
    split_identity = _split_identity(sample_ids, rows, evaluation)
    label_identity = _label_identity(sample_ids, labels, dataset_identity, label_col)
    numeric_contract = normalise_numeric_contract(config["dataset"])
    numeric_identity = None
    if numeric_contract["columns"]:
        all_ids = frame["__yonod_sample_id__"].astype(str).tolist()
        expected_numeric_identity = numeric_input_identity(frame, all_ids, numeric_contract)
        block_path = (
            artifact.manifest_path.parent.parent.parent / "numeric_blocks" /
            (expected_numeric_identity.removeprefix("sha256:") + ".json")
        )
        try:
            block_frame, block = load_numeric_block(
                block_path, expected_ids=artifact.sample_ids, expected_contract=numeric_contract,
                expected_identity=expected_numeric_identity,
            )
        except NumericConditionsError as exc:
            raise TrainingServiceError(
                "训练需先经 features 阶段发布可核验的原始数值块：" + str(exc)
            ) from exc
        for entry in numeric_contract["columns"]:
            aligned[entry["source"]] = block_frame.loc[artifact.valid_mask, entry["source"]].tolist()
        numeric_identity = block["identity"]
    try:
        resolved = resolve_model_config(model, dict(config.get("model_params") or {}).get(model, {}))
    except ModelConfigurationError as exc:
        raise TrainingServiceError(f"模型参数无效：{exc}") from exc
    if numeric_identity is not None and (
        model == "autogluon" or resolved.runtime.get("early_stopping")
    ):
        raise TrainingServiceError(
            "该模型存在内部验证划分；尚不能保证数值条件仅在内部训练子集拟合，拒绝运行"
        )
    hpo_declaration = None
    if (config.get("hpo") or {}).get("enabled") and model in config["hpo"]["models"]:
        from yonod.hpo.contracts import validate_hpo_declaration
        hpo_declaration = validate_hpo_declaration(
            config["hpo"], models=config["models"], stage=config["stage"],
            evaluation=config.get("evaluation"), model_params=config.get("model_params"),
        )
    run_id = _run_identity(
        artifact, model=model, model_config=resolved, label_identity=label_identity,
        split_identity=split_identity, numeric_identity=numeric_identity,
        hpo_identity=hpo_declaration,
    )
    result_root = _result_root(loaded, config)
    run_dir = result_root / "runs" / combination_name(str(artifact.manifest["feature_id"]), model)
    if run_dir.parent.exists():
        for sibling in run_dir.parent.iterdir():
            if sibling.name.casefold() == run_dir.name.casefold() and sibling.name != run_dir.name:
                raise TrainingServiceError(f"组合目录名称存在大小写冲突：{sibling.name}")
    existing = _existing_complete_run(run_dir, run_id)
    if existing is not None:
        return existing
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_id}.tmp-{uuid.uuid4().hex}"
    runner = fold_runner or _default_fold_runner
    fold_records: list[Dict[str, Any]] = []
    prediction_records: list[Dict[str, Any]] = []
    model_bundles: list[str] = []
    hpo_folds: list[Dict[str, Any]] = []
    owned_task_budget = False
    if task_budget is None and hpo_declaration is not None and hpo_declaration["budget"].get("task_timeout_s") is not None:
        task_budget = _task_budget_context(loaded, config)
        task_budget.__enter__()
        owned_task_budget = True
    try:
        for repeat, fold, train_index, valid_index in rows:
            search_audit = None
            fold_resolved = resolved
            if hpo_declaration is not None:
                from yonod.hpo.ordinary import select_outer_parameters
                search_audit = select_outer_parameters(
                    artifact=artifact, config=config, aligned=aligned,
                    sample_ids=sample_ids, labels=labels, outer_train_index=train_index,
                    repeat=repeat, fold=fold, model=model,
                    numeric_contract=numeric_contract, output_root=result_root,
                    task_budget=task_budget,
                )
                selected_config = copy.deepcopy(dict(config.get("model_params", {}).get(model, {})))
                selected_config["estimator"] = search_audit["constructor_parameters"]
                fold_resolved = resolve_model_config(model, selected_config)
                hpo_folds.append({"repeat": repeat, "fold": fold, **search_audit})
                if task_budget is not None:
                    task_budget.require_start("外层重训")
            fold_context = {
                "run_id": run_id,
                "artifact_id": artifact.manifest["artifact_id"],
                "split_identity": split_identity,
                "repeat": repeat,
                "fold": fold,
                "train_sample_ids_sha256": _sha256_text(_canonical_json(sample_ids[train_index].astype(str).tolist())),
                "valid_sample_ids_sha256": _sha256_text(_canonical_json(sample_ids[valid_index].astype(str).tolist())),
            }
            X_train, X_valid, transform_audit = _fold_matrix(
                artifact, config, aligned, sample_ids, train_index, valid_index,
                staging / "fold_transforms" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}", fold_context,
            )
            started = time.perf_counter()
            prediction, model_audit = runner(
                fold_resolved, X_train, labels[train_index], X_valid, aligned,
                train_index, staging / "models" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}",
                label_col, fold,
            )
            if "model_file" in model_audit:
                model_path = staging / "models" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / str(model_audit["model_file"])
                if not model_path.is_file() or _sha256_file(model_path) != model_audit.get("model_sha256"):
                    raise TrainingServiceError("折级模型文件与模型审计哈希不匹配")
                bundle_path = staging / "model_bundles" / f"repeat-{repeat:02d}-fold-{fold:02d}.yaml"
                bundle = {
                    "schema_version": "yonod_model_bundle/v1",
                    "run_id": run_id,
                    "model": model,
                    "repeat": repeat,
                    "fold": fold,
                    "model_path": model_path.relative_to(staging).as_posix(),
                    "model_sha256": model_audit["model_sha256"],
                    "feature_artifact_id": artifact.manifest["artifact_id"],
                    "feature_content_identity": artifact_content_identity(artifact.manifest),
                    "feature_lifecycle": artifact.manifest["lifecycle"],
                    "feature_spec": artifact.manifest.get("metadata", {}).get("feature_spec"),
                    "descriptor_columns": list(artifact.manifest["matrix"]["columns"].get("names", [])),
                    "descriptor_source_columns": list(artifact.manifest.get("metadata", {}).get("resolved_columns", [])),
                    "descriptor_roles": artifact.manifest.get("metadata", {}).get("resolved_roles"),
                    "descriptor_dim": int(X_train.shape[1] - transform_audit["numeric"].get("dimension", 0)),
                    "feature_dim": int(X_train.shape[1]),
                    "numeric_contract": numeric_contract if numeric_identity is not None else None,
                    "numeric_state_path": (
                        (staging / "fold_transforms" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / "numeric_conditions.yaml").relative_to(staging).as_posix()
                        if numeric_identity is not None else None
                    ),
                    "numeric_state_sha256": transform_audit["numeric"].get("state_sha256"),
                    "ohe_state_path": (
                        (staging / "fold_transforms" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / "transform").relative_to(staging).as_posix()
                        if artifact.manifest["lifecycle"] == "fold_transform" else None
                    ),
                    "ohe_state_metadata_sha256": (
                        _sha256_file(staging / "fold_transforms" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / "transform" / "metadata.json")
                        if artifact.manifest["lifecycle"] == "fold_transform" else None
                    ),
                    "train_sample_ids_sha256": fold_context["train_sample_ids_sha256"],
                    "split_identity": split_identity,
                    "software_versions": software_versions(model),
                    **({"hpo": search_audit} if search_audit is not None else {}),
                }
                _atomic_yaml(bundle_path, bundle)
                model_bundles.append(bundle_path.relative_to(staging).as_posix())
            elapsed = time.perf_counter() - started
            prediction = np.asarray(prediction, dtype=float).reshape(-1)
            if len(prediction) != len(valid_index) or not np.isfinite(prediction).all():
                raise TrainingServiceError("模型预测必须与验证折等长且全部为有限数值")
            r2, rmse, mae = _metrics(labels[valid_index], prediction)
            fold_records.append({
                "repeat": repeat, "fold": fold,
                "n_train": int(len(train_index)), "n_valid": int(len(valid_index)),
                "feature_dim": int(X_train.shape[1]), "r2": r2, "rmse": rmse, "mae": mae,
                "elapsed_s": float(elapsed),
                "model_train_time_s": float(model_audit.get("train_time_s", 0.0)),
                "model_predict_time_s": float(model_audit.get("predict_time_s", 0.0)),
                "transform_audit": _canonical_json(transform_audit),
                "model_audit": _canonical_json(model_audit),
                **({"hpo_study_id": search_audit["study_id"], "hpo_best_trial": search_audit["best_trial_number"],
                    "hpo_best_inner_score": search_audit["best_value"]} if search_audit is not None else {}),
            })
            prediction_records.extend({
                "sample_id": sample_ids[index], "repeat": repeat, "fold": fold,
                "y_true": float(labels[index]), "y_pred": float(value),
            } for index, value in zip(valid_index.tolist(), prediction.tolist()))
        final_info = None
        if hpo_declaration is not None and hpo_declaration["final_model"]["enabled"]:
            from contextlib import nullcontext
            from yonod.hpo.budget import ActiveBudgetLedger
            from yonod.hpo.finalize import fit_final_model
            from yonod.hpo.ordinary import select_outer_parameters
            final_timeout = hpo_declaration["final_model"]["budget"].get("task_timeout_s")
            final_budget_context = (
                ActiveBudgetLedger(
                    result_root, {"run_id": run_id, "purpose": "final_model"}, final_timeout,
                    study_dir=result_root / "hpo" / "final_task_budget" / run_id,
                ) if final_timeout is not None else nullcontext(None)
            )
            with final_budget_context as final_budget:
                if task_budget is not None:
                    task_budget.require_start("最终模型独立搜索")
                final_search = select_outer_parameters(
                    artifact=artifact, config=config, aligned=aligned,
                    sample_ids=sample_ids, labels=labels,
                    outer_train_index=np.arange(len(sample_ids), dtype=int),
                    repeat=0, fold=0, model=model, numeric_contract=numeric_contract,
                    output_root=result_root, task_budget=task_budget,
                    final_budget=final_budget, purpose="final_model",
                )
                if task_budget is not None:
                    task_budget.require_start("完整开发集最终重训")
                if final_budget is not None:
                    final_budget.require_start("完整开发集最终重训")
                final_info = fit_final_model(
                    artifact=artifact, config=config, aligned=aligned,
                    sample_ids=sample_ids, labels=labels, numeric_contract=numeric_contract,
                    model=model, run_id=run_id, staging=staging, search_audit=final_search,
                )
            model_bundles.append(final_info["bundle_path"])
        predictions = pd.DataFrame.from_records(prediction_records)
        metrics = pd.DataFrame.from_records(fold_records)
        if len(predictions) != len(sample_ids) * int(evaluation["n_repeats"]):
            raise TrainingServiceError("OOF 预测数量不完整，拒绝发布训练结果")
        _atomic_csv(staging / "predictions.csv", predictions)
        _atomic_csv(staging / "fold_metrics.csv", metrics)
        try:
            shutil.copy2(loaded.path, staging / "source_config.yaml")
        except OSError as exc:
            raise TrainingServiceError(f"无法保存原始 YAML 配置快照：{exc}") from exc
        _atomic_yaml(staging / "effective_config.yaml", config)
        manifest = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "kind": "yonod_training_run",
            "status": "complete",
            "run_id": run_id,
            "artifact_id": artifact.manifest["artifact_id"],
            "feature_id": artifact.manifest["feature_id"],
            "feature_content_identity": artifact_content_identity(artifact.manifest),
            "model": model,
            "model_request": resolved.as_audit_dict(),
            **({"hpo": {"declaration": hpo_declaration, "outer_folds": hpo_folds,
                        **({"final_model": final_info} if final_info is not None else {})}} if hpo_declaration is not None else {}),
            "dataset_identity": dataset_identity,
            "label_identity": label_identity,
            "label_column": label_col,
            "split_identity": split_identity,
            "numeric_identity": numeric_identity,
            "numeric_contract": numeric_contract if numeric_identity is not None else None,
            "evaluation": evaluation,
            "n_samples": int(len(sample_ids)),
            "n_folds": int(len(rows)),
            "model_bundles": model_bundles,
            **({"model_bundle_sha256": {
                name: _sha256_file(staging / name) for name in model_bundles
            }} if hpo_declaration is not None else {}),
            "source_config_filename": loaded.path.name,
            "source_config_sha256": _sha256_file(staging / "source_config.yaml"),
            "effective_config_sha256": _sha256_file(staging / "effective_config.yaml"),
            "dependency_versions": software_versions(model),
            "outputs": {
                "predictions": "predictions.csv",
                "fold_metrics": "fold_metrics.csv",
                "fold_transform_states": "fold_transforms" if artifact.manifest["lifecycle"] == "fold_transform" or numeric_identity is not None else None,
                "model_bundles": "model_bundles" if model_bundles else None,
                "source_config": "source_config.yaml",
                "effective_config": "effective_config.yaml",
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_yaml(staging / "run_manifest.yaml", manifest)
        # Verify the published content in the staging directory before a
        # single atomic directory rename makes it a resumable result.
        complete = _existing_complete_run(staging, run_id)
        if complete is None:  # defensive, the files above must satisfy it
            raise TrainingServiceError("训练结果 staging 验证失败")
        if run_dir.exists():
            # Only a directory containing failed attempts can reach here;
            # completed or incomplete runs are checked before training.
            os.replace(run_dir / "failed_attempts", staging / "failed_attempts")
            run_dir.rmdir()
        os.replace(staging, run_dir)
    except Exception as exc:
        if (staging / "failed_attempts").is_dir():
            run_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staging / "failed_attempts", run_dir / "failed_attempts")
        if staging.exists():
            shutil.rmtree(staging)
        failed_run = _publish_failed_run(
            loaded, config, artifact,
            model=model, resolved=resolved, run_id=run_id,
            dataset_identity=dataset_identity, label_identity=label_identity,
            split_identity=split_identity, evaluation=evaluation,
            n_samples=len(sample_ids), n_folds=len(rows), cause=exc,
            hpo_declaration=hpo_declaration, hpo_folds=hpo_folds,
        )
        if hpo_declaration is not None:
            try:
                from yonod.pipeline.reporting import rebuild_schema2_hpo_status_report
                rebuild_schema2_hpo_status_report(
                    result_root, formats=config["outputs"].get("report_formats"),
                )
            except Exception as report_exc:
                import sys
                print(f"[hpo] 状态报告生成失败，原始训练错误仍为：{failed_run.reason}；报告错误：{report_exc}", file=sys.stderr)
        return failed_run
    finally:
        if owned_task_budget:
            task_budget.__exit__(None, None, None)
    return TrainingRun(
        run_id=run_id,
        artifact_id=str(artifact.manifest["artifact_id"]),
        feature_id=str(artifact.manifest["feature_id"]),
        model=model,
        status="complete",
        run_dir=run_dir,
        manifest_path=run_dir / "run_manifest.yaml",
        predictions_path=run_dir / "predictions.csv",
        fold_metrics_path=run_dir / "fold_metrics.csv",
        reason="已从独立 manifest 读取、逐折训练并完整发布",
    )


def validate_training_model_parameters(config: Mapping[str, Any]) -> Dict[str, ResolvedModelConfig]:
    """Validate all selected models before a stage can start expensive work."""
    if (config.get("hpo") or {}).get("enabled"):
        from yonod.hpo.contracts import require_search_engine, validate_hpo_declaration
        hpo = validate_hpo_declaration(
            config["hpo"], models=config["models"], stage=config["stage"],
            evaluation=config.get("evaluation"), model_params=config.get("model_params"),
        )
        require_search_engine()
    models = list(config.get("models") or [])
    if not models:
        raise TrainingServiceError("train/all 必须选择至少一个模型")
    raw = dict(config.get("model_params") or {})
    result: Dict[str, ResolvedModelConfig] = {}
    for model in models:
        try:
            result[str(model)] = resolve_model_config(str(model), raw.get(str(model), {}))
        except ModelConfigurationError as exc:
            raise TrainingServiceError(f"model_params.{model} 无效：{exc}") from exc
    return result


def run_train(
    config_path: Path | str,
    *,
    input_manifests: Sequence[Path | str] | None = None,
    fold_runner: FoldRunner | None = None,
    task_budget: Any = None,
) -> tuple[TrainingRun, ...]:
    """Run models from existing artifacts only; it never invokes feature generation.

    Once a feature package and model identity validate, a runtime failure is
    published as that combination's immutable failed run. Sibling combinations
    continue to train.
    """
    loaded = load_run_config(config_path)
    config = loaded.effective
    if config["stage"] not in {"train", "all"}:
        raise TrainingServiceError("run_train 只接受 stage: train 或由 all 编排传入")
    # Validate every selected dependency/parameter before reading the feature
    # matrix or creating a training result directory.
    validate_training_model_parameters(config)
    if input_manifests is None:
        if config["stage"] != "train":
            raise TrainingServiceError("stage: all 必须通过 run_all 编排，不能猜测 input_manifest")
        input_manifests = [resolve_config_path(loaded.path, config["artifacts"]["input_manifest"])]
    paths = [Path(path).resolve() for path in input_manifests]
    if not paths:
        raise TrainingServiceError("没有可训练的 ready feature manifest")
    names: set[str] = set()
    for path in paths:
        artifact = load_feature_artifact(path)
        for model in config["models"]:
            name = combination_name(str(artifact.manifest["feature_id"]), str(model)).casefold()
            if name in names:
                raise TrainingServiceError("组合目录名称冲突；请为特征指定不同的 id（不区分大小写）")
            names.add(name)
    create_task_layout(_result_root(loaded, config))
    from contextlib import nullcontext
    context = _task_budget_context(loaded, config) if task_budget is None else nullcontext(task_budget)
    results: list[TrainingRun] = []
    with context as active_budget:
        for path in paths:
            for model in config["models"]:
                results.append(_run_one_artifact(
                    loaded, path, model=str(model), fold_runner=fold_runner,
                    task_budget=active_budget,
                ))
    return tuple(results)


def run_all(
    config_path: Path | str,
    *,
    feature_computer: Any | None = None,
    fold_runner: FoldRunner | None = None,
) -> AllRunResult:
    """Materialise all features to terminal states, then train ready packages.

    A failed descriptor is isolated.  If every descriptor failed, this raises
    before model construction/training; successful packages from a partial
    feature run remain independently reusable.
    """
    loaded = load_run_config(config_path)
    if loaded.effective["stage"] != "all":
        raise TrainingServiceError("run_all 只接受 stage: all 配置")
    validate_training_model_parameters(loaded.effective)
    from .features import run_features
    with _task_budget_context(loaded, loaded.effective) as task_budget:
        feature_result = run_features(config_path, feature_computer=feature_computer, allow_all=True)
        manifests = [item.manifest_path for item in feature_result.features if item.status in {"ready", "reused"} and item.manifest_path]
        if not manifests:
            raise TrainingServiceError("所有特征候选均失败；不会开始任何模型训练")
        runs = run_train(config_path, input_manifests=manifests, fold_runner=fold_runner,
                         task_budget=task_budget)
    return AllRunResult(feature_result.status_manifest_path, feature_result.status, runs)
