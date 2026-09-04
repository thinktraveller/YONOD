"""Persistent descriptor artifacts shared by feature generation and modelling.

Each ``.npz`` file is self-describing and contains the descriptor matrix for
valid rows, the complete sample-id vector, the complete validity mask, and a
JSON metadata document.  Cache identity deliberately depends on descriptor
inputs/configuration, but not on model or cross-validation settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1


class DescriptorArtifactError(ValueError):
    """Raised when a descriptor artifact is missing, stale, or malformed."""


@dataclass(frozen=True)
class DescriptorArtifact:
    path: Path
    X_smiles: np.ndarray
    sample_ids: np.ndarray
    valid_mask: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DescriptorPreparation:
    path: Path
    descriptor: str
    status: str
    reason: str
    dataset_fingerprint: str
    descriptor_config: Dict[str, Any]
    n_total: int
    n_valid: int
    feature_dim: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _normalise_config(
    descriptor: str,
    smiles_cols: List[str],
    smiles_roles: Optional[Mapping[str, List[str]]],
    mode: str,
    descriptor_config: Optional[Mapping[str, Any]],
    sample_id_name: str,
) -> Dict[str, Any]:
    roles = smiles_roles or {"reactant": [], "product": [], "other": []}
    return json.loads(_canonical_json({
        "descriptor": descriptor.lower(),
        "mode": mode,
        "sample_id_name": sample_id_name,
        "smiles_cols": list(smiles_cols),
        "smiles_roles": {
            "reactant": list(roles.get("reactant", [])),
            "product": list(roles.get("product", [])),
            "other": list(roles.get("other", [])),
        },
        "options": dict(descriptor_config or {}),
    }))


def _sample_id_array(df: pd.DataFrame, sample_ids: Optional[List[Any]]) -> np.ndarray:
    values = list(df.index) if sample_ids is None else list(sample_ids)
    if len(values) != len(df):
        raise DescriptorArtifactError(
            f"sample_id 数量 {len(values)} 与数据行数 {len(df)} 不一致"
        )
    result = np.asarray([str(value) for value in values], dtype=np.str_)
    if len(set(result.tolist())) != len(result):
        raise DescriptorArtifactError("sample_id 必须唯一，无法建立可靠的描述符样本对齐")
    return result


def dataset_fingerprint(
    df: pd.DataFrame,
    smiles_cols: List[str],
    sample_ids: np.ndarray,
) -> str:
    missing = [column for column in smiles_cols if column not in df.columns]
    if missing:
        raise DescriptorArtifactError(f"描述符输入列不存在：{missing}")
    digest = hashlib.sha256()
    digest.update(_canonical_json(sample_ids.tolist()).encode("utf-8"))
    records = df.loc[:, smiles_cols].where(pd.notna(df.loc[:, smiles_cols]), None)
    digest.update(_canonical_json(records.to_dict(orient="records")).encode("utf-8"))
    return digest.hexdigest()


def descriptor_artifact_path(directory: Path, descriptor: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", descriptor.strip().lower()).strip("-._")
    if not safe:
        raise DescriptorArtifactError("描述符名称不能生成安全文件名")
    return Path(directory) / f"{safe}.npz"


def _feature_structure(
    descriptor: str,
    mode: str,
    smiles_cols: List[str],
    feature_dim: int,
) -> Dict[str, Any]:
    if mode == "concat" and smiles_cols and feature_dim % len(smiles_cols) == 0:
        return {
            "kind": "column_blocks",
            "input_columns": list(smiles_cols),
            "block_dim": feature_dim // len(smiles_cols),
            "feature_name_pattern": f"<input_column>::{descriptor.lower()}_<zero_based_index>",
        }
    return {
        "kind": "single_vector",
        "input_columns": list(smiles_cols),
        "feature_name_pattern": f"{descriptor.lower()}_<zero_based_index>",
    }


def _mfp_component_audit(df: pd.DataFrame, smiles_cols: List[str]) -> Dict[str, Any]:
    """Count zero-block causes without altering paper-protocol raw inputs."""
    from rdkit import Chem
    import rdkit

    blank_count = 0
    invalid_count = 0
    for column in smiles_cols:
        for value in df[column].tolist():
            if pd.isna(value) or not str(value).strip():
                blank_count += 1
            elif Chem.MolFromSmiles(str(value)) is None:
                invalid_count += 1
    return {
        "algorithm": "morgan_count",
        "rdkit_version": str(getattr(rdkit, "__version__", "unknown")),
        "component_order": list(smiles_cols),
        "block_dim": 0,  # populated after feature matrix construction
        "blank_component_count": blank_count,
        "invalid_component_count": invalid_count,
        "zero_block_component_count": blank_count + invalid_count,
        "row_retention_policy": "zero_block_keep_row",
    }


def _write_artifact(
    path: Path,
    X_smiles: np.ndarray,
    sample_ids: np.ndarray,
    valid_mask: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(
            temporary,
            X_smiles=np.asarray(X_smiles),
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            valid_mask=np.asarray(valid_mask, dtype=np.bool_),
            metadata_json=np.asarray(_canonical_json(metadata), dtype=np.str_),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_descriptor_artifact(
    path: Path,
    *,
    expected_descriptor_config: Optional[Mapping[str, Any]] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_sample_ids: Optional[np.ndarray] = None,
) -> DescriptorArtifact:
    path = Path(path)
    if not path.is_file():
        raise DescriptorArtifactError(f"描述符文件不存在：{path}")
    try:
        with np.load(path, allow_pickle=False) as stored:
            required = {"X_smiles", "sample_ids", "valid_mask", "metadata_json"}
            missing = required.difference(stored.files)
            if missing:
                raise DescriptorArtifactError(f"描述符文件缺少字段：{sorted(missing)}")
            X_smiles = np.asarray(stored["X_smiles"])
            sample_ids = np.asarray(stored["sample_ids"], dtype=np.str_)
            valid_mask = np.asarray(stored["valid_mask"], dtype=np.bool_)
            metadata = json.loads(str(stored["metadata_json"].item()))
    except DescriptorArtifactError:
        raise
    except Exception as exc:
        raise DescriptorArtifactError(f"无法读取描述符文件 {path.name}：{exc}") from exc

    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise DescriptorArtifactError("描述符文件 schema 版本不受支持")
    if X_smiles.ndim != 2:
        raise DescriptorArtifactError("X_smiles 必须是二维矩阵")
    if sample_ids.ndim != 1 or valid_mask.ndim != 1 or len(sample_ids) != len(valid_mask):
        raise DescriptorArtifactError("sample_ids 与 valid_mask 必须是一维且长度一致")
    if X_smiles.shape[0] != int(valid_mask.sum()):
        raise DescriptorArtifactError("描述符矩阵行数与 valid_mask 有效行数不一致")
    if X_smiles.shape[1] != int(metadata.get("feature_dim", -1)):
        raise DescriptorArtifactError("描述符矩阵列数与元数据 feature_dim 不一致")
    if int(metadata.get("n_total", -1)) != len(sample_ids):
        raise DescriptorArtifactError("元数据 n_total 与 sample_ids 长度不一致")
    if int(metadata.get("n_valid", -1)) != int(valid_mask.sum()):
        raise DescriptorArtifactError("元数据 n_valid 与 valid_mask 不一致")
    if expected_descriptor_config is not None and metadata.get("descriptor_config") != dict(expected_descriptor_config):
        raise DescriptorArtifactError("描述符配置已变化")
    if expected_dataset_fingerprint is not None and metadata.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise DescriptorArtifactError("描述符输入数据已变化")
    if expected_sample_ids is not None and not np.array_equal(sample_ids, expected_sample_ids.astype(np.str_)):
        raise DescriptorArtifactError("描述符文件 sample_id 与当前数据不一致")
    return DescriptorArtifact(path, X_smiles, sample_ids, valid_mask, metadata)


def prepare_descriptor_artifact(
    directory: Path,
    *,
    descriptor: str,
    smiles_cols: List[str],
    df: pd.DataFrame,
    smiles_roles: Optional[Mapping[str, List[str]]] = None,
    mode: str = "concat",
    descriptor_config: Optional[Mapping[str, Any]] = None,
    sample_ids: Optional[List[Any]] = None,
    sample_id_name: str = "dataframe_index",
    compute: Optional[Callable[..., Any]] = None,
) -> DescriptorPreparation:
    """Reuse a matching artifact or atomically compute and persist a new one."""
    current_sample_ids = _sample_id_array(df, sample_ids)
    config = _normalise_config(
        descriptor, smiles_cols, smiles_roles, mode, descriptor_config, sample_id_name
    )
    fingerprint = dataset_fingerprint(df, smiles_cols, current_sample_ids)
    path = descriptor_artifact_path(directory, descriptor)
    cache_miss_reason = "描述符文件不存在"

    if path.exists():
        try:
            artifact = load_descriptor_artifact(
                path,
                expected_descriptor_config=config,
                expected_dataset_fingerprint=fingerprint,
                expected_sample_ids=current_sample_ids,
            )
            return DescriptorPreparation(
                path, descriptor, "reused", "输入数据与描述符配置一致",
                fingerprint, config, len(artifact.sample_ids), int(artifact.valid_mask.sum()),
                int(artifact.X_smiles.shape[1]),
            )
        except DescriptorArtifactError as exc:
            cache_miss_reason = str(exc)

    if compute is None:
        from .feature_builder import build_universal_features
        compute = build_universal_features
    X_smiles, _X_numeric, valid_mask = compute(
        smiles_cols=smiles_cols,
        numeric_cols=[],
        df=df,
        desc_name=descriptor,
        smiles_roles=dict(smiles_roles or {}),
        mode=mode,
        descriptor_config=dict(descriptor_config or {}),
    )
    X_smiles = np.asarray(X_smiles)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if valid_mask.shape != (len(df),):
        raise DescriptorArtifactError("描述符计算返回的 valid_mask 形状不正确")
    if X_smiles.ndim != 2 or X_smiles.shape[0] != int(valid_mask.sum()):
        raise DescriptorArtifactError("描述符计算返回的矩阵与 valid_mask 无法对齐")
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "descriptor": descriptor.lower(),
        "descriptor_config": config,
        "dataset_fingerprint": fingerprint,
        "feature_dim": int(X_smiles.shape[1]),
        "feature_structure": _feature_structure(descriptor, mode, smiles_cols, int(X_smiles.shape[1])),
        "n_total": int(len(df)),
        "n_valid": int(valid_mask.sum()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if descriptor.lower() == "mfp":
        audit = _mfp_component_audit(df, smiles_cols)
        if len(smiles_cols) == 0 or X_smiles.shape[1] % len(smiles_cols) != 0:
            raise DescriptorArtifactError("MFP 特征维度无法按组件列拆分")
        audit["block_dim"] = int(X_smiles.shape[1] // len(smiles_cols))
        metadata["mfp_protocol"] = audit
    _write_artifact(path, X_smiles, current_sample_ids, valid_mask, metadata)
    status = "recomputed" if path.exists() and cache_miss_reason != "描述符文件不存在" else "computed"
    reason = cache_miss_reason if status == "recomputed" else "首次生成描述符文件"
    return DescriptorPreparation(
        path, descriptor, status, reason, fingerprint, config, len(df),
        int(valid_mask.sum()), int(X_smiles.shape[1]),
    )
