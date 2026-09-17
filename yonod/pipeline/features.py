"""Independent feature-materialisation service for step 30.4.

The service consumes a schema-2 ``stage: features`` YAML config and produces
only versioned artifact references plus a feature-run status YAML within the
shared task layout. It never imports prediction-model adapters.
Static descriptors are delegated to the existing production
``build_universal_features`` implementation on demand; tests may inject a
small feature computer to prove orchestration without chemistry dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.reader import ArtifactReadError, load_feature_artifact, publish_feature_artifact
from yonod.config.contracts import resolve_config_path
from yonod.config.loader import LoadedRunConfig, load_run_config
from yonod.descriptors.registry import FeatureRegistryError, FeatureSpec, normalise_feature_specs
from yonod.provenance import feature_dataset_identity


FEATURE_SERVICE_SCHEMA_VERSION = "1.0"
_RAW_MISSING_SENTINEL = "\u0000YONOD_MISSING_V1\u0000"


class FeatureServiceError(ValueError):
    """Raised for an invalid feature-only execution request."""


@dataclass(frozen=True)
class FeatureStatus:
    """One persistent feature result; it never retains an in-memory table."""

    feature_id: str
    descriptor: str
    lifecycle: str
    status: str
    artifact_id: str | None
    manifest_path: Path | None
    feature_identity: str | None
    reason: str


@dataclass(frozen=True)
class FeatureRunResult:
    """Terminal state of a features-only invocation."""

    run_id: str
    status: str
    status_manifest_path: Path
    dataset_identity: str | None
    features: tuple[FeatureStatus, ...]


FeatureComputer = Callable[
    [FeatureSpec, pd.DataFrame, Sequence[str], Mapping[str, Sequence[str]]],
    Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Optional[Mapping[str, Any]]]],
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return result or "feature"


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


def _normalise_specs(config: Mapping[str, Any]) -> list[FeatureSpec]:
    try:
        return normalise_feature_specs(config["descriptors"], label_col=None)
    except (FeatureRegistryError, KeyError) as exc:
        raise FeatureServiceError(f"features 描述符配置无效：{exc}") from exc


def _all_smiles_columns(roles: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for role in ("reactants", "products", "others"):
        values.extend(str(value) for value in roles.get(role, []) or [])
    return list(dict.fromkeys(values))


def _expand_column(name: str, available: Sequence[str]) -> list[str]:
    if name in available:
        return [name]
    expression = re.compile(rf"^{re.escape(name)}-(\d+)$")
    matched = [column for column in available if expression.fullmatch(column)]
    return sorted(matched, key=lambda column: int(column.rsplit("-", 1)[1]))


def _resolve_static_inputs(
    spec: FeatureSpec,
    roles: Mapping[str, Any],
) -> tuple[list[str], Dict[str, list[str]]]:
    all_columns = _all_smiles_columns(roles)
    if spec.descriptor == "drfp" or spec.mode == "reaction":
        reactants = list(roles.get("reactants", []) or [])
        products = list(roles.get("products", []) or [])
        if not reactants and not products:
            raise FeatureServiceError("drfp/reaction 特征需要至少一个 reactants 或 products 列")
        extras: list[str] = []
        for declared in spec.extra_reactants:
            resolved = _expand_column(declared, all_columns)
            if not resolved:
                raise FeatureServiceError(f"{spec.id}.extra_reactants 指向不存在的列：{declared}")
            extras.extend(resolved)
        reactants = list(dict.fromkeys([*reactants, *extras]))
        columns = list(dict.fromkeys([*reactants, *products]))
        return columns, {
            "reactant": reactants,
            "product": products,
            "other": [],
        }

    declared = list(spec.columns) or all_columns
    columns: list[str] = []
    for name in declared:
        resolved = _expand_column(name, all_columns)
        if not resolved:
            raise FeatureServiceError(f"{spec.id}.columns 指向不存在的 SMILES 列：{name}")
        columns.extend(resolved)
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise FeatureServiceError(f"{spec.id} 没有可用的 SMILES 输入列")
    return columns, {
        "reactant": [column for column in columns if column in (roles.get("reactants", []) or [])],
        "product": [column for column in columns if column in (roles.get("products", []) or [])],
        "other": [column for column in columns if column in (roles.get("others", []) or [])],
    }


def _production_feature_computer(
    spec: FeatureSpec,
    frame: pd.DataFrame,
    columns: Sequence[str],
    smiles_roles: Mapping[str, Sequence[str]],
    *,
    config_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any] | None]:
    """Delegate static generation to the existing production feature builder."""
    from yonod.universal.feature_builder import build_universal_features

    params = dict(spec.params)
    diagnostic_sink: list[dict[str, Any]] | None = None
    if spec.descriptor == "chemical_vae":
        if config_path is None:  # Defensive: public service always supplies it.
            raise FeatureServiceError("chemical_vae 需要其所属 YAML 路径来解析 model_manifest")
        params["model_manifest"] = str(resolve_config_path(config_path, params["model_manifest"]))
        diagnostic_sink = []
    matrix, _numeric, valid_mask = build_universal_features(
        smiles_cols=list(columns),
        numeric_cols=[],
        df=frame,
        desc_name=spec.descriptor,
        smiles_roles={key: list(value) for key, value in smiles_roles.items()},
        mode=spec.mode,
        descriptor_config={
            "params": params,
            "feature_spec": spec.to_dict(),
            "_chemical_vae_diagnostic_sink": diagnostic_sink,
        },
    )
    diagnostics: Mapping[str, Any] | None = None
    if diagnostic_sink is not None:
        diagnostics = {"columns": diagnostic_sink}
    return np.asarray(matrix), np.asarray(valid_mask), diagnostics


def _validate_feature_frame(config: Mapping[str, Any], dataset_path: Path) -> tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    if not dataset_path.is_file():
        raise FeatureServiceError(f"features 数据集不存在：{dataset_path}")
    try:
        frame = pd.read_csv(dataset_path)
    except Exception as exc:
        raise FeatureServiceError(f"无法读取 features 数据集：{exc}") from exc
    dataset = config["dataset"]
    sample_id_col = dataset["sample_id_col"]
    if sample_id_col not in frame.columns:
        raise FeatureServiceError(f"数据集缺少 sample_id_col：{sample_id_col}")
    raw_ids = frame[sample_id_col]
    if raw_ids.isna().any():
        raise FeatureServiceError("sample_id 不能包含空值")
    sample_ids = raw_ids.astype(str).to_numpy(dtype=np.str_)
    if any(not value.strip() for value in sample_ids.tolist()) or len(set(sample_ids.tolist())) != len(sample_ids):
        raise FeatureServiceError("sample_id 必须是唯一的非空字符串")
    roles = dict(dataset["column_roles"])
    return frame, sample_ids, roles


def _feature_identity(
    spec: FeatureSpec,
    *,
    columns: Sequence[str],
    smiles_roles: Mapping[str, Sequence[str]],
    asset_identity: Mapping[str, Any] | None = None,
) -> str:
    feature_spec = spec.to_dict()
    if asset_identity is not None:
        # A model path is an execution locator, not semantic feature content.
        # The manifest/state hashes below are the cache key and make a copied
        # verified asset reusable while a same-path replacement invalidates it.
        feature_spec["params"] = dict(feature_spec["params"])
        feature_spec["params"]["model_manifest"] = "content_addressed_conversion_manifest"
    payload = {
        "feature_service_schema_version": FEATURE_SERVICE_SCHEMA_VERSION,
        "feature_spec": feature_spec,
        "resolved_columns": list(columns),
        "resolved_roles": {name: list(values) for name, values in sorted(smiles_roles.items())},
        "chemical_vae_asset": dict(asset_identity) if asset_identity is not None else None,
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _chemical_vae_asset_identity(spec: FeatureSpec, config_path: Path) -> Dict[str, Any] | None:
    """Resolve and hash the exact conversion package before cache lookup.

    This intentionally does not import Torch, h5py, or descriptor code.  It
    binds feature identity to both conversion-manifest bytes and the adjacent
    state bytes that the runtime loader will consume.
    """

    if spec.descriptor != "chemical_vae":
        return None
    manifest_reference = spec.params.get("model_manifest")
    if not isinstance(manifest_reference, str):
        raise FeatureServiceError("chemical_vae.model_manifest 必须是字符串")
    manifest_path = resolve_config_path(config_path, manifest_reference)
    if not manifest_path.is_file():
        raise FeatureServiceError(f"chemical_vae model_manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureServiceError(f"无法读取 chemical_vae conversion manifest：{exc}") from exc
    if not isinstance(manifest, Mapping):
        raise FeatureServiceError("chemical_vae conversion manifest 必须是对象")
    asset = manifest.get("asset")
    expected_state = manifest.get("converted_state_sha256")
    source_sha = asset.get("source_encoder_sha256") if isinstance(asset, Mapping) else None
    if not isinstance(expected_state, str) or not isinstance(source_sha, str):
        raise FeatureServiceError("chemical_vae conversion manifest 缺少 state/source 内容哈希")
    state_path = manifest_path.parent / "encoder_state_dict.pt"
    if not state_path.is_file():
        raise FeatureServiceError(f"chemical_vae converted state 不存在：{state_path}")
    actual_state = _sha256_file(state_path)
    if actual_state != expected_state:
        raise FeatureServiceError("chemical_vae converted state hash 与 conversion manifest 不一致")
    return {
        "identity_schema": "chemical_vae_asset_content/v1",
        "conversion_manifest_sha256": _sha256_file(manifest_path),
        "converted_state_sha256": actual_state,
        "source_encoder_sha256": source_sha,
        "converter_version": manifest.get("converter_version"),
        "converter_code_sha256": manifest.get("converter_code_sha256"),
    }


def _chemical_vae_diagnostic_sidecar(
    raw: Mapping[str, Any] | None, sample_ids: np.ndarray
) -> Dict[str, Any] | None:
    """Turn per-column encoder records into an immutable full-row sidecar."""

    if raw is None:
        return None
    columns = raw.get("columns")
    if not isinstance(columns, list) or not columns:
        raise FeatureServiceError("chemical_vae 诊断缺少列记录")
    per_column: Dict[str, Mapping[str, Any]] = {}
    for item in columns:
        if not isinstance(item, Mapping):
            raise FeatureServiceError("chemical_vae 诊断列记录无效")
        column = item.get("column")
        records = item.get("records")
        if not isinstance(column, str) or not column or column in per_column:
            raise FeatureServiceError("chemical_vae 诊断列名无效或重复")
        if not isinstance(records, list) or len(records) != len(sample_ids):
            raise FeatureServiceError("chemical_vae 诊断记录必须覆盖完整 sample_id 行")
        per_column[column] = item
    rows: list[Dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids.tolist()):
        roles: Dict[str, Any] = {}
        for column, item in per_column.items():
            record = item["records"][index]
            if not isinstance(record, Mapping):
                raise FeatureServiceError("chemical_vae 单行诊断无效")
            roles[column] = {"role": item.get("role", "other"), **dict(record)}
        rows.append({"sample_id": str(sample_id), "roles": roles})
    return {
        "schema_version": "chemical_vae_diagnostics/v1",
        "rows": rows,
    }


def _artifact_id(spec: FeatureSpec, dataset_identity: str, feature_identity: str) -> str:
    return "{0}-{1}-{2}".format(
        _safe_identifier(spec.id), dataset_identity.removeprefix("sha256:")[:12],
        feature_identity.removeprefix("sha256:")[:12],
    )


def _relative_to_status(path: Path, status_path: Path) -> str:
    return os.path.relpath(path, start=status_path.parent).replace(os.sep, "/")


def _status_payload(
    *,
    run_id: str,
    status: str,
    source_config: Path,
    dataset_identity: str | None,
    features: Sequence[FeatureStatus],
    status_path: Path,
) -> Dict[str, Any]:
    records = []
    for item in features:
        record: Dict[str, Any] = {
            "feature_id": item.feature_id,
            "descriptor": item.descriptor,
            "lifecycle": item.lifecycle,
            "status": item.status,
            "reason": item.reason,
        }
        if item.artifact_id is not None:
            record["artifact_id"] = item.artifact_id
        if item.manifest_path is not None:
            record["manifest_path"] = _relative_to_status(item.manifest_path, status_path)
        if item.feature_identity is not None:
            record["feature_identity"] = item.feature_identity
        records.append(record)
    return {
        "schema_version": FEATURE_SERVICE_SCHEMA_VERSION,
        "kind": "yonod_feature_run_status",
        "run_id": run_id,
        "status": status,
        "source_config_filename": source_config.name,
        "dataset_identity": dataset_identity,
        "features": records,
    }


def _run_id(dataset_identity: str | None, specs: Sequence[FeatureSpec]) -> str:
    payload = {
        "dataset_identity": dataset_identity,
        "feature_specs": [spec.to_dict() for spec in specs],
        "schema": FEATURE_SERVICE_SCHEMA_VERSION,
    }
    return "features-" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _publish_fold_transform_raw_input(
    *,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    dataset_identity: str,
    feature_identity: str,
    destination: Path,
    artifact_id: str,
) -> Path:
    columns = list(spec.columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureServiceError(f"{spec.id} 的折内类别列不存在：{', '.join(missing)}")
    raw = frame.loc[:, columns].astype("string")
    if raw.eq(_RAW_MISSING_SENTINEL).any().any():
        raise FeatureServiceError(f"{spec.id} 的原始类别值与保留缺失标记冲突")
    raw_matrix = raw.fillna(_RAW_MISSING_SENTINEL).to_numpy(dtype=np.str_)
    return publish_feature_artifact(
        destination,
        artifact_id=artifact_id,
        feature_id=spec.id,
        matrix=raw_matrix,
        sample_ids=sample_ids,
        valid_mask=np.ones(len(sample_ids), dtype=bool),
        dataset_identity=dataset_identity,
        feature_config_identity=feature_identity,
        column_structure={
            "kind": "named",
            "names": columns,
            "semantic": "raw_categorical_columns",
        },
        lifecycle="fold_transform",
        metadata={
            "fold_transform": {
                "transform": "ohe",
                "fit_scope": "training_fold_only",
                "parameters": dict(spec.params),
                "input_columns": columns,
                "missing_sentinel": _RAW_MISSING_SENTINEL,
            }
        },
    )


def _execute_one_feature(
    *,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    roles: Mapping[str, Any],
    dataset_identity: str,
    artifacts_root: Path,
    computer: FeatureComputer,
    config_path: Path,
) -> FeatureStatus:
    try:
        if spec.lifecycle == "fold_transform":
            columns = list(spec.columns)
            feature_identity = _feature_identity(spec, columns=columns, smiles_roles={})
            artifact_id = _artifact_id(spec, dataset_identity, feature_identity)
            destination = artifacts_root / "features" / artifact_id
            if destination.exists():
                artifact = load_feature_artifact(destination / "manifest.yaml", expected_artifact_id=artifact_id)
                if artifact.manifest["sources"].get("dataset_identity") != dataset_identity or artifact.manifest["sources"].get("feature_config_identity") != feature_identity:
                    raise FeatureServiceError("已有 OHE 原始输入包的身份与本次请求不一致，拒绝覆盖")
                return FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "reused", artifact_id,
                                     artifact.manifest_path, feature_identity, "命中已验证的原始类别输入缓存")
            manifest = _publish_fold_transform_raw_input(
                spec=spec, frame=frame, sample_ids=sample_ids, dataset_identity=dataset_identity,
                feature_identity=feature_identity, destination=destination, artifact_id=artifact_id,
            )
            return FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "ready", artifact_id,
                                 manifest, feature_identity, "已保存折内 OHE 的原始类别输入；未执行全量 fit")

        columns, feature_roles = _resolve_static_inputs(spec, roles)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise FeatureServiceError(f"{spec.id} 的 SMILES 输入列不存在：{', '.join(missing)}")
        asset_identity = _chemical_vae_asset_identity(spec, config_path)
        feature_identity = _feature_identity(
            spec, columns=columns, smiles_roles=feature_roles, asset_identity=asset_identity,
        )
        artifact_id = _artifact_id(spec, dataset_identity, feature_identity)
        destination = artifacts_root / "features" / artifact_id
        if destination.exists():
            artifact = load_feature_artifact(destination / "manifest.yaml", expected_artifact_id=artifact_id)
            if artifact.manifest["sources"].get("dataset_identity") != dataset_identity or artifact.manifest["sources"].get("feature_config_identity") != feature_identity:
                raise FeatureServiceError("已有特征包的身份与本次请求不一致，拒绝覆盖")
            return FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "reused", artifact_id,
                                 artifact.manifest_path, feature_identity, "命中已验证的静态特征缓存")

        generated = computer(spec, frame, columns, feature_roles)
        if not isinstance(generated, tuple) or len(generated) not in {2, 3}:
            raise FeatureServiceError("特征计算器必须返回 (matrix, valid_mask) 或加 diagnostics 的三元组")
        matrix, valid_mask = generated[0], generated[1]
        raw_diagnostics = generated[2] if len(generated) == 3 else None
        if raw_diagnostics is not None and not isinstance(raw_diagnostics, Mapping):
            raise FeatureServiceError("特征计算器 diagnostics 必须是 mapping 或 None")
        # Recheck after generation: a same-path asset replacement during
        # inference cannot publish vectors under the pre-cache identity.
        if asset_identity is not None and _chemical_vae_asset_identity(spec, config_path) != asset_identity:
            raise FeatureServiceError("chemical_vae 资产在特征计算期间发生内容变化，拒绝发布")
        diagnostics = _chemical_vae_diagnostic_sidecar(raw_diagnostics, sample_ids)
        matrix = np.asarray(matrix)
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != (len(sample_ids),) or valid_mask.dtype != np.dtype(bool):
            raise FeatureServiceError("特征计算器必须返回与输入行数一致的一维 bool valid_mask")
        if matrix.ndim != 2 or matrix.shape[0] != int(valid_mask.sum()):
            raise FeatureServiceError("特征计算器返回的 matrix 行数必须等于 valid_mask 有效行数")
        names = [f"{spec.id}::{spec.descriptor}_{index}" for index in range(matrix.shape[1])]
        manifest = publish_feature_artifact(
            destination,
            artifact_id=artifact_id,
            feature_id=spec.id,
            matrix=matrix,
            sample_ids=sample_ids,
            valid_mask=valid_mask,
            dataset_identity=dataset_identity,
            feature_config_identity=feature_identity,
            column_structure={"kind": "named", "names": names},
            lifecycle="static_descriptor",
            metadata={
                "feature_spec": spec.to_dict(),
                "resolved_columns": columns,
                "resolved_roles": feature_roles,
                "generation": "yonod.universal.feature_builder",
                **({"chemical_vae_asset": asset_identity} if asset_identity is not None else {}),
            },
            diagnostics=diagnostics,
        )
        return FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "ready", artifact_id,
                             manifest, feature_identity, "静态特征已发布并通过完整读取校验")
    except Exception as exc:
        return FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "failed", None, None, None,
                             f"{type(exc).__name__}: {exc}")


def run_features(
    config_path: Path | str,
    *,
    feature_computer: FeatureComputer | None = None,
    allow_all: bool = False,
) -> FeatureRunResult:
    """Compute/reuse all requested features and return only durable references.

    ``feature_computer`` is a test seam.  When omitted, each static candidate
    calls the established production ``build_universal_features`` path lazily.
    No model factory, model adapter, report builder, or training output path is
    involved in either mode.
    """
    loaded: LoadedRunConfig = load_run_config(config_path)
    config = loaded.effective
    if config["stage"] != "features" and not (allow_all and config["stage"] == "all"):
        raise FeatureServiceError("run_features 只接受 stage: features 配置；stage: all 必须由 run_all 编排")
    specs = _normalise_specs(config)
    artifacts_root = resolve_config_path(loaded.path, config["artifacts"]["output_dir"])
    from yonod.config.output_layout import create_task_layout, task_root
    create_task_layout(task_root(loaded.path, config))
    computer = feature_computer or partial(_production_feature_computer, config_path=loaded.path)
    dataset_path = resolve_config_path(loaded.path, config["dataset"]["path"])
    dataset_identity: str | None = None

    try:
        frame, sample_ids, roles = _validate_feature_frame(config, dataset_path)
        dataset_identity = feature_dataset_identity(
            frame,
            sample_id_col=config["dataset"]["sample_id_col"],
            column_roles=roles,
        )
    except Exception as exc:
        run_id = _run_id(None, specs)
        status_path = artifacts_root / "feature_runs" / f"{run_id}.yaml"
        failures = tuple(
            FeatureStatus(spec.id, spec.descriptor, spec.lifecycle, "failed", None, None, None,
                          f"{type(exc).__name__}: {exc}")
            for spec in specs
        )
        _atomic_yaml(status_path, _status_payload(
            run_id=run_id, status="failed", source_config=loaded.path, dataset_identity=None,
            features=failures, status_path=status_path,
        ))
        return FeatureRunResult(run_id, "failed", status_path, None, failures)

    run_id = _run_id(dataset_identity, specs)
    status_path = artifacts_root / "feature_runs" / f"{run_id}.yaml"
    results = tuple(
        _execute_one_feature(
            spec=spec, frame=frame, sample_ids=sample_ids, roles=roles,
            dataset_identity=dataset_identity, artifacts_root=artifacts_root, computer=computer,
            config_path=loaded.path,
        )
        for spec in specs
    )
    successes = [result for result in results if result.status in {"ready", "reused"}]
    failures = [result for result in results if result.status == "failed"]
    terminal = "ready" if successes and not failures else "partial" if successes else "failed"
    _atomic_yaml(status_path, _status_payload(
        run_id=run_id, status=terminal, source_config=loaded.path, dataset_identity=dataset_identity,
        features=results, status_path=status_path,
    ))
    return FeatureRunResult(run_id, terminal, status_path, dataset_identity, results)
