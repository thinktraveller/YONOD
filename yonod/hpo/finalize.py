"""Fit an explicitly requested final model on the full development population."""

from __future__ import annotations

import time
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from yonod.artifacts.contracts import artifact_content_identity
from yonod.artifacts.reader import FeatureArtifact
from yonod.descriptors.ohe import OHEFeature
from yonod.features.numeric_conditions import NumericConditionsTransformer
from yonod.model_factory import construct_estimator, materialise_fit_parameters, resolve_model_config, software_versions
from yonod.pipeline.training import _atomic_yaml, _sha256_file, _sha256_text, _canonical_json
from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
from .search import _audit_value
from .storage import _atomic_json


class FinalModelError(RuntimeError):
    """The final-purpose search or full-population refit cannot be published."""


def fit_final_model(
    *, artifact: FeatureArtifact, config: Mapping[str, Any], aligned: pd.DataFrame,
    sample_ids: np.ndarray, labels: np.ndarray, numeric_contract: Mapping[str, Any],
    model: str, run_id: str, staging: Path, search_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Save one separate final bundle; never reuse a CV fold's state or model."""
    if search_audit.get("purpose") != "final_model":
        raise FinalModelError("最终模型必须使用独立 final-purpose study")
    ids = sample_ids.astype(str).tolist()
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise FinalModelError("完整开发训练集 sample_id/标签无效")
    final_dir = staging / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    lifecycle = artifact.manifest["lifecycle"]
    if lifecycle == "static_descriptor":
        matrix = np.asarray(artifact.matrix, dtype=float)
        ohe_path = None
    elif lifecycle == "fold_transform":
        transform = artifact.manifest.get("metadata", {}).get("fold_transform", {})
        if transform.get("transform") != "ohe" or transform.get("fit_scope") != "training_fold_only":
            raise FinalModelError("最终模型需要未拟合的 OHE 原始特征包")
        columns = list(transform.get("input_columns") or [])
        sentinel = transform.get("missing_sentinel")
        if not columns or len(columns) != artifact.matrix.shape[1] or not isinstance(sentinel, str):
            raise FinalModelError("最终模型 OHE 输入列或缺失标记无效")
        raw = pd.DataFrame(np.asarray(artifact.matrix, dtype=str), columns=columns)
        raw = raw.mask(raw.eq(sentinel), other=pd.NA)
        params = dict(transform.get("parameters") or {})
        ohe = OHEFeature(
            columns, missing_policy=str(params.get("missing_policy", "as_category")),
            dtype=str(params.get("dtype", "float32")),
            handle_unknown=str(params.get("handle_unknown", "ignore")),
        )
        ohe.fit(raw, sample_ids=sample_ids)
        matrix = np.asarray(ohe.transform(raw, partition="train"), dtype=float)
        ohe_path = final_dir / "transform"
        ohe.save(ohe_path, context={"run_id": run_id, "purpose": "final_model"})
    else:
        raise FinalModelError("最终模型不支持未知特征生命周期")
    if matrix.ndim != 2 or len(matrix) != len(ids) or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise FinalModelError("最终模型开发训练集特征矩阵无效")
    descriptor_dim = int(matrix.shape[1])
    numeric_path = None
    numeric_sha = None
    if numeric_contract.get("columns"):
        numeric = NumericConditionsTransformer(numeric_contract)
        values = numeric.fit_transform(aligned, sample_ids, phase="final_full_development")
        matrix = np.hstack([matrix, values])
        numeric_path = final_dir / "numeric_conditions.yaml"
        _atomic_yaml(numeric_path, numeric.state_dict())
        numeric_sha = _sha256_file(numeric_path)
    selected = dict((config.get("model_params") or {}).get(model, {}))
    selected["estimator"] = dict(search_audit["constructor_parameters"])
    resolved = resolve_model_config(model, selected)
    estimator, construction = construct_estimator(resolved, n_features=matrix.shape[1], n_train=len(ids))
    fit_params = materialise_fit_parameters(resolved, aligned, np.arange(len(ids), dtype=int))
    started = time.perf_counter()
    estimator.fit(matrix, np.asarray(labels, dtype=float), **fit_params)
    train_time = time.perf_counter() - started
    model_path = final_dir / "model.joblib"
    joblib.dump(estimator, model_path, compress=3)
    bundle_path = staging / "model_bundles" / "final.yaml"
    bundle = {
        "schema_version": "yonod_final_model_bundle/v1",
        "purpose": "final_model", "run_id": run_id, "model": model,
        "model_path": model_path.relative_to(staging).as_posix(),
        "model_sha256": _sha256_file(model_path),
        "feature_artifact_id": artifact.manifest["artifact_id"],
        "feature_content_identity": artifact_content_identity(artifact.manifest),
        "feature_lifecycle": lifecycle,
        "feature_spec": artifact.manifest.get("metadata", {}).get("feature_spec"),
        "descriptor_columns": list(artifact.manifest["matrix"]["columns"].get("names", [])),
        "descriptor_source_columns": list(artifact.manifest.get("metadata", {}).get("resolved_columns", [])),
        "descriptor_roles": artifact.manifest.get("metadata", {}).get("resolved_roles"),
        "descriptor_dim": descriptor_dim, "feature_dim": int(matrix.shape[1]),
        "numeric_contract": numeric_contract if numeric_path is not None else None,
        "numeric_state_path": numeric_path.relative_to(staging).as_posix() if numeric_path is not None else None,
        "numeric_state_sha256": numeric_sha,
        "ohe_state_path": ohe_path.relative_to(staging).as_posix() if ohe_path is not None else None,
        "ohe_state_metadata_sha256": _sha256_file(ohe_path / "metadata.json") if ohe_path is not None else None,
        "train_sample_ids_sha256": _sha256_text(_canonical_json(ids)),
        "n_development_rows": len(ids),
        "development_population_id": config["hpo"]["final_model"]["development_population_id"],
        "hpo": dict(search_audit),
        "effective_estimator_parameters": construction["effective_estimator_params"],
        "software_versions": software_versions(model),
    }
    _atomic_yaml(bundle_path, bundle)
    return {
        "purpose": "final_model", "study": dict(search_audit),
        "bundle_path": bundle_path.relative_to(staging).as_posix(),
        "bundle_sha256": _sha256_file(bundle_path),
        "model_sha256": _sha256_file(model_path),
        "n_development_rows": len(ids), "train_time_s": float(train_time),
        "independent_test_score": None,
    }


def fit_strict_final_model(
    *, contract: Any, frame: pd.DataFrame, feature_set: Mapping[str, Any],
    feature_matrix: np.ndarray | None, numeric_frame: pd.DataFrame | None,
    numeric_contract: Mapping[str, Any], model: str,
    search_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one strict final model from the full development set."""
    if search_audit.get("purpose") != "final_model":
        raise FinalModelError("strict 最终重训要求独立 final-purpose study")
    root = Path(contract.run_dir).resolve()
    descriptor = str(feature_set["name"])
    destination = root / "final_models" / f"{descriptor}__{model}"
    if destination.exists():
        bundle_path = destination / "bundle.json"
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            model_path = root / bundle["model_path"]
        except (OSError, ValueError, KeyError) as exc:
            raise FinalModelError("已有 strict 最终模型目录不完整，拒绝覆盖") from exc
        if (bundle.get("schema_version") != "yonod_strict_final_model_bundle/v1" or
            bundle.get("run_id") != contract.run_id or
            bundle.get("hpo", {}).get("study_id") != search_audit["study_id"] or
            not model_path.is_file() or _sha256_file(model_path) != bundle.get("model_sha256")):
            raise FinalModelError("已有 strict 最终模型身份或哈希不匹配")
        if bundle.get("ohe_state_path"):
            metadata_path = root / bundle["ohe_state_path"] / "metadata.json"
            if not metadata_path.is_file() or _sha256_file(metadata_path) != bundle.get("ohe_state_metadata_sha256"):
                raise FinalModelError("已有 strict 最终 OHE 状态元数据哈希不匹配")
        return {
            "purpose": "final_model", "study": dict(search_audit),
            "bundle_path": bundle_path.relative_to(root).as_posix(),
            "bundle_sha256": _sha256_file(bundle_path),
            "model_sha256": bundle["model_sha256"],
            "n_development_rows": len(frame), "train_time_s": bundle["train_time_s"],
            "independent_test_score": None, "reused": True,
        }
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        ids = frame[contract.config.sample_id_col].astype(str).to_numpy()
        labels = frame[contract.config.label_col].astype(float).to_numpy()
        if not len(ids) or len(set(ids.tolist())) != len(ids) or not np.isfinite(labels).all():
            raise FinalModelError("strict 最终开发人口 ID 或标签无效")
        ohe_path = None
        if feature_set["kind"] == "fold_transform":
            columns = list(feature_set["component_cols"])
            ohe = ReactionComponentOHE(columns)
            ohe.fit(frame.loc[:, columns], sample_ids=ids)
            matrix = np.asarray(ohe.transform(frame.loc[:, columns], partition="train"), dtype=float)
            ohe_path = staging / "transform"
            ohe.save(ohe_path, context={"run_id": contract.run_id, "purpose": "final_model"})
        else:
            if feature_matrix is None:
                raise FinalModelError("strict 静态最终模型缺少描述符矩阵")
            matrix = np.asarray(feature_matrix, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(ids) or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise FinalModelError("strict 最终模型描述符矩阵无效")
        descriptor_dim = int(matrix.shape[1])
        numeric_path = None
        if numeric_frame is not None:
            transformer = NumericConditionsTransformer(numeric_contract)
            values = transformer.fit_transform(numeric_frame, ids, phase="strict_final_full_development")
            matrix = np.hstack([matrix, values])
            numeric_path = staging / "numeric.json"
            _atomic_json(numeric_path, transformer.state_dict())
        selected = dict(contract.config.model_configs[model])
        selected["estimator"] = dict(search_audit["constructor_parameters"])
        resolved = resolve_model_config(model, selected)
        estimator, construction = construct_estimator(resolved, n_features=matrix.shape[1], n_train=len(ids))
        fit_params = materialise_fit_parameters(resolved, frame, np.arange(len(ids), dtype=int))
        started = time.perf_counter()
        estimator.fit(matrix, labels, **fit_params)
        train_time = time.perf_counter() - started
        model_path = staging / "model.joblib"
        joblib.dump(estimator, model_path, compress=3)
        bundle = {
            "schema_version": "yonod_strict_final_model_bundle/v1",
            "purpose": "final_model", "run_id": contract.run_id,
            "config_hash": contract.config_hash,
            "descriptor": descriptor, "model": model,
            "feature_spec": dict(feature_set),
            "dataset_roles": contract.config.raw.get("dataset_roles"),
            "sample_id_col": contract.config.sample_id_col,
            "feature_dim": int(matrix.shape[1]), "descriptor_dim": descriptor_dim,
            "model_path": (destination / "model.joblib").relative_to(root).as_posix(),
            "model_sha256": _sha256_file(model_path),
            "ohe_state_path": (destination / "transform").relative_to(root).as_posix() if ohe_path else None,
            "ohe_state_metadata_sha256": _sha256_file(ohe_path / "metadata.json") if ohe_path else None,
            "numeric_contract": dict(numeric_contract) if numeric_path else None,
            "numeric_state_path": (destination / "numeric.json").relative_to(root).as_posix() if numeric_path else None,
            "numeric_state_sha256": _sha256_file(numeric_path) if numeric_path else None,
            "train_sample_ids_sha256": _sha256_text(_canonical_json(ids.tolist())),
            "n_development_rows": len(ids), "train_time_s": float(train_time),
            "development_population_id": contract.config.raw["hpo"]["final_model"]["development_population_id"],
            "hpo": dict(search_audit),
            "effective_estimator_parameters": _audit_value(construction["effective_estimator_params"]),
            "software_versions": software_versions(model),
        }
        _atomic_json(staging / "bundle.json", bundle)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    published_bundle = destination / "bundle.json"
    return {
        "purpose": "final_model", "study": dict(search_audit),
        "bundle_path": published_bundle.relative_to(root).as_posix(),
        "bundle_sha256": _sha256_file(published_bundle),
        "model_sha256": bundle["model_sha256"],
        "n_development_rows": len(ids), "train_time_s": float(train_time),
        "independent_test_score": None, "reused": False,
    }
