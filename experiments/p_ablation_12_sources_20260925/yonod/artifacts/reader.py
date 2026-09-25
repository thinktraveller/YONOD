"""Dependency-light reader and publisher for schema-2 feature artifacts.

The module deliberately imports only the standard library, NumPy, and PyYAML
(indirectly through the shared safe YAML reader).  In particular it must not
import RDKit, Torch, descriptor registries, or feature generators: a training
process can validate and consume an already materialised feature package in a
fresh interpreter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml

from yonod.config.loader import ConfigLoadError, load_yaml_mapping

from .contracts import ArtifactContractError, resolve_manifest_path, validate_artifact_manifest


class ArtifactReadError(ArtifactContractError):
    """Raised when a published feature package is incomplete or inconsistent."""


@dataclass(frozen=True)
class FeatureArtifact:
    """A verified, read-only feature matrix and its complete sample mapping."""

    manifest_path: Path
    manifest: Dict[str, Any]
    matrix: np.ndarray
    sample_ids: np.ndarray
    valid_mask: np.ndarray
    valid_row_indices: np.ndarray
    diagnostics: Dict[str, Any] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_within_manifest_package(manifest_path: Path, candidate: Path, field: str) -> Path:
    package_root = manifest_path.resolve().parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise ArtifactReadError(f"{field} 经解析后离开 manifest 包目录") from exc
    return resolved


def _verified_files(manifest_path: Path, manifest: Mapping[str, Any]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for name, record in manifest["files"].items():
        field = f"files.{name}"
        try:
            path = resolve_manifest_path(manifest_path, record["path"])
        except ArtifactContractError as exc:
            raise ArtifactReadError(str(exc)) from exc
        path = _ensure_within_manifest_package(manifest_path, path, field)
        if not path.is_file():
            raise ArtifactReadError(f"{field} 指向的文件不存在：{record['path']}")
        size = path.stat().st_size
        if "bytes" in record and size != record["bytes"]:
            raise ArtifactReadError(
                f"{field} 文件大小不匹配：manifest={record['bytes']}，实际={size}"
            )
        actual_digest = _sha256_file(path)
        if actual_digest != record["sha256"]:
            raise ArtifactReadError(f"{field} 内容哈希不匹配")
        resolved[name] = path
    return resolved


def _load_npy(path: Path, field: str) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ArtifactReadError(f"{field} 仅支持 .npy 数组文件，收到 {path.name!r}")
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ArtifactReadError(f"无法读取 {field}：{exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ArtifactReadError(f"{field} 必须为 NumPy 数组")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _normalise_diagnostics_payload(value: Mapping[str, Any], sample_ids: np.ndarray) -> Dict[str, Any]:
    """Validate a versioned full-population diagnostic sidecar in memory."""

    try:
        payload = json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ArtifactReadError("diagnostics 必须是可 JSON 序列化的 mapping") from exc
    schema = payload.get("schema_version")
    rows = payload.get("rows")
    if not isinstance(schema, str) or not schema.strip():
        raise ArtifactReadError("diagnostics.schema_version 必须是非空字符串")
    if not isinstance(rows, list) or len(rows) != len(sample_ids):
        raise ArtifactReadError("diagnostics.rows 必须与完整 sample_ids 等长")
    expected_ids = sample_ids.tolist()
    for index, (row, sample_id) in enumerate(zip(rows, expected_ids)):
        if not isinstance(row, Mapping) or row.get("sample_id") != sample_id:
            raise ArtifactReadError(f"diagnostics.rows[{index}] 必须与完整 sample_ids 同序且匹配")
    return payload


def _load_diagnostics(
    manifest: Mapping[str, Any], file_paths: Mapping[str, Path], sample_ids: np.ndarray
) -> Dict[str, Any] | None:
    declaration = manifest.get("diagnostics")
    if declaration is None:
        return None
    try:
        path = file_paths[declaration["file"]]
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactReadError(f"无法读取 diagnostics sidecar：{exc}") from exc
    validated = _normalise_diagnostics_payload(payload, sample_ids)
    if validated["schema_version"] != declaration["schema_version"]:
        raise ArtifactReadError("diagnostics schema_version 与 manifest 不一致")
    if len(validated["rows"]) != declaration["row_count"]:
        raise ArtifactReadError("diagnostics row_count 与 manifest 不一致")
    return validated


def _validate_physical_arrays(
    manifest: Mapping[str, Any],
    file_paths: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = _load_npy(file_paths[manifest["matrix"]["file"]], "matrix.file")
    sample_ids = _load_npy(file_paths[manifest["samples"]["file"]], "samples.file")
    valid_mask = _load_npy(file_paths[manifest["validity"]["file"]], "validity.file")

    if matrix.ndim != 2:
        raise ArtifactReadError("物理 matrix 必须是二维数组")
    allowed_matrix_kinds = {"U", "S"} if manifest["lifecycle"] == "fold_transform" else {"b", "i", "u", "f"}
    if matrix.dtype.kind not in allowed_matrix_kinds:
        expected = "字符串" if manifest["lifecycle"] == "fold_transform" else "数值"
        raise ArtifactReadError(f"物理 matrix 必须是{expected}数组，收到 {matrix.dtype}")
    try:
        declared_dtype = np.dtype(manifest["matrix"]["dtype"])
    except (TypeError, ValueError) as exc:
        raise ArtifactReadError("matrix.dtype 不是有效 NumPy dtype") from exc
    if matrix.dtype != declared_dtype:
        raise ArtifactReadError(
            f"物理 matrix dtype 与 manifest 不一致：{matrix.dtype} != {declared_dtype}"
        )
    if list(matrix.shape) != list(manifest["matrix"]["shape"]):
        raise ArtifactReadError("物理 matrix.shape 与 manifest 不一致")

    columns = manifest["matrix"]["columns"]
    if columns["kind"] == "named":
        names = columns.get("names")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise ArtifactReadError("matrix.columns.kind=named 时必须提供 names 列表")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ArtifactReadError("matrix.columns.names 必须是非空字符串")
        if len(names) != matrix.shape[1] or len(set(names)) != len(names):
            raise ArtifactReadError("matrix.columns.names 必须与特征维度等长且唯一")

    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ArtifactReadError("物理 sample_ids 必须是一维字符串数组")
    sample_ids = np.asarray(sample_ids, dtype=np.str_)
    if len(sample_ids) != manifest["samples"]["row_count"]:
        raise ArtifactReadError("物理 sample_ids 长度与 samples.row_count 不一致")
    if any(not sample_id.strip() for sample_id in sample_ids.tolist()):
        raise ArtifactReadError("sample_ids 不能包含空值")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ArtifactReadError("sample_ids 必须唯一")

    if valid_mask.ndim != 1 or valid_mask.dtype != np.dtype(bool):
        raise ArtifactReadError("物理 validity_mask 必须是一维 bool 数组")
    if len(valid_mask) != len(sample_ids):
        raise ArtifactReadError("validity_mask 长度必须等于 sample_ids 长度")
    valid_count = int(valid_mask.sum())
    if valid_count != manifest["validity"]["n_valid"]:
        raise ArtifactReadError("物理 validity_mask 有效行数与 manifest 不一致")
    if len(valid_mask) != manifest["validity"]["n_total"]:
        raise ArtifactReadError("物理 validity_mask 总行数与 manifest 不一致")
    if matrix.shape[0] != valid_count:
        raise ArtifactReadError("matrix 行数必须等于 validity_mask 的有效行数")
    if manifest["samples"]["matrix_row_mapping"] != "valid_rows_in_sample_order":
        raise ArtifactReadError(
            "当前读取层仅支持 samples.matrix_row_mapping=valid_rows_in_sample_order"
        )
    return matrix, sample_ids, valid_mask, np.flatnonzero(valid_mask)


def read_feature_manifest(path: Path | str) -> tuple[Path, Dict[str, Any]]:
    """Read a safe YAML manifest and validate its schema without materialising arrays."""
    manifest_path = Path(path).resolve()
    try:
        raw = load_yaml_mapping(manifest_path)
        return manifest_path, validate_artifact_manifest(raw)
    except (ConfigLoadError, ArtifactContractError) as exc:
        raise ArtifactReadError(f"manifest 无效：{exc}") from exc


def load_feature_artifact(
    path: Path | str,
    *,
    expected_artifact_id: str | None = None,
    expected_feature_id: str | None = None,
    allow_partial: bool = False,
) -> FeatureArtifact:
    """Load a hash-verified artifact without importing descriptor generators."""
    manifest_path, manifest = read_feature_manifest(path)
    if manifest["status"] != "ready" and not (allow_partial and manifest["status"] == "partial"):
        raise ArtifactReadError(f"产物状态为 {manifest['status']!r}，不能作为就绪特征读取")
    if expected_artifact_id is not None and manifest["artifact_id"] != expected_artifact_id:
        raise ArtifactReadError("artifact_id 与请求不一致")
    if expected_feature_id is not None and manifest["feature_id"] != expected_feature_id:
        raise ArtifactReadError("feature_id 与请求不一致")
    file_paths = _verified_files(manifest_path, manifest)
    matrix, sample_ids, valid_mask, valid_rows = _validate_physical_arrays(manifest, file_paths)
    diagnostics = _load_diagnostics(manifest, file_paths, sample_ids)
    return FeatureArtifact(
        manifest_path=manifest_path,
        manifest=copy.deepcopy(manifest),
        matrix=_readonly(matrix),
        sample_ids=_readonly(sample_ids),
        valid_mask=_readonly(valid_mask),
        valid_row_indices=_readonly(valid_rows),
        diagnostics=diagnostics,
    )


def _normalise_publish_inputs(
    matrix: Any,
    sample_ids: Any,
    valid_mask: Any,
    *,
    lifecycle: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix_array = np.asarray(matrix)
    allowed_matrix_kinds = {"U", "S"} if lifecycle == "fold_transform" else {"b", "i", "u", "f"}
    if matrix_array.ndim != 2 or matrix_array.dtype.kind not in allowed_matrix_kinds:
        expected = "字符串" if lifecycle == "fold_transform" else "数值"
        raise ArtifactReadError(f"matrix 必须是二维{expected} NumPy 数组")
    sample_array = np.asarray(sample_ids)
    if sample_array.ndim != 1 or sample_array.dtype.kind not in {"U", "S"}:
        raise ArtifactReadError("sample_ids 必须是一维字符串 NumPy 数组")
    sample_array = np.asarray(sample_array, dtype=np.str_)
    if len(sample_array) != len(set(sample_array.tolist())) or any(not value.strip() for value in sample_array.tolist()):
        raise ArtifactReadError("sample_ids 必须为唯一的非空字符串")
    mask_array = np.asarray(valid_mask)
    if mask_array.ndim != 1 or mask_array.dtype != np.dtype(bool):
        raise ArtifactReadError("valid_mask 必须是一维 bool NumPy 数组")
    if len(sample_array) != len(mask_array):
        raise ArtifactReadError("sample_ids 与 valid_mask 长度不一致")
    if matrix_array.shape[0] != int(mask_array.sum()):
        raise ArtifactReadError("matrix 行数必须等于 valid_mask 的有效行数")
    return matrix_array, sample_array, mask_array


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactReadError(f"{field} 必须是非空字符串")
    return value.strip()


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)


def publish_feature_artifact(
    target_dir: Path | str,
    *,
    artifact_id: str,
    feature_id: str,
    matrix: Any,
    sample_ids: Any,
    valid_mask: Any,
    dataset_identity: str,
    feature_config_identity: str,
    column_structure: Mapping[str, Any],
    lifecycle: str = "static_descriptor",
    derived_from: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish a fresh portable package after re-reading it fully.

    The destination itself must not already exist.  All material files and the
    manifest are first written under a sibling temporary directory, validated
    through :func:`load_feature_artifact`, then atomically renamed into place.
    """
    matrix_array, sample_array, mask_array = _normalise_publish_inputs(
        matrix, sample_ids, valid_mask, lifecycle=lifecycle,
    )
    if not isinstance(column_structure, Mapping):
        raise ArtifactReadError("column_structure 必须是 mapping")
    _require_string(dataset_identity, "dataset_identity")
    _require_string(feature_config_identity, "feature_config_identity")
    diagnostics_payload = (
        _normalise_diagnostics_payload(diagnostics, sample_array) if diagnostics is not None else None
    )
    destination = Path(target_dir).resolve()
    if destination.exists():
        raise ArtifactReadError(f"目标产物目录已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"

    try:
        data_dir = staging / "data"
        data_dir.mkdir(parents=True)
        matrix_path = data_dir / "matrix.npy"
        sample_path = data_dir / "sample_ids.npy"
        mask_path = data_dir / "validity_mask.npy"
        np.save(matrix_path, matrix_array, allow_pickle=False)
        np.save(sample_path, sample_array, allow_pickle=False)
        np.save(mask_path, mask_array, allow_pickle=False)
        diagnostics_path = data_dir / "diagnostics.json"
        if diagnostics_payload is not None:
            diagnostics_path.write_text(
                json.dumps(diagnostics_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def record(path: Path) -> Dict[str, Any]:
            return {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }

        manifest: Dict[str, Any] = {
            "schema_version": "2.0",
            "artifact_id": artifact_id,
            "feature_id": feature_id,
            "lifecycle": lifecycle,
            "status": "ready",
            "files": {
                "matrix": record(matrix_path),
                "sample_ids": record(sample_path),
                "validity_mask": record(mask_path),
            },
            "matrix": {
                "file": "matrix",
                # ``dtype.name`` for Unicode arrays (for example ``str640``)
                # is not reliably accepted by ``np.dtype``.  ``dtype.str`` is
                # a stable NumPy round-trip representation for both numeric
                # static matrices and raw fold-transform category matrices.
                "dtype": matrix_array.dtype.str,
                "shape": [int(matrix_array.shape[0]), int(matrix_array.shape[1])],
                "columns": copy.deepcopy(dict(column_structure)),
            },
            "samples": {
                "file": "sample_ids",
                "id_field": "sample_id",
                "row_count": int(len(sample_array)),
                "matrix_row_mapping": "valid_rows_in_sample_order",
            },
            "validity": {
                "file": "validity_mask",
                "n_total": int(len(mask_array)),
                "n_valid": int(mask_array.sum()),
                "semantics": "true iff sample_ids row has the next matrix row in sample order",
            },
            "sources": {
                "dataset_identity": dataset_identity,
                "feature_config_identity": feature_config_identity,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if diagnostics_payload is not None:
            manifest["files"]["diagnostics"] = record(diagnostics_path)
            manifest["diagnostics"] = {
                "file": "diagnostics",
                "schema_version": diagnostics_payload["schema_version"],
                "row_count": int(len(diagnostics_payload["rows"])),
                "sample_id_field": "sample_id",
                "row_mapping": "full_sample_rows_in_sample_order",
            }
        if derived_from is not None:
            manifest["derived_from"] = copy.deepcopy(dict(derived_from))
        if metadata is not None:
            manifest["metadata"] = copy.deepcopy(dict(metadata))
        validate_artifact_manifest(manifest)
        staging_manifest = staging / "manifest.yaml"
        _write_yaml(staging_manifest, manifest)
        # Verify hashes, dimensions, IDs and mask using the same public reader
        # before making the package visible at its final location.
        load_feature_artifact(staging_manifest, expected_artifact_id=artifact_id, expected_feature_id=feature_id)
        os.replace(staging, destination)
        return destination / "manifest.yaml"
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def import_legacy_npz(
    legacy_path: Path | str,
    target_dir: Path | str,
    *,
    artifact_id: str,
    feature_id: str,
    dataset_identity: str,
    feature_config_identity: str,
    column_structure: Mapping[str, Any],
) -> Path:
    """Explicitly convert one known v1 NPZ artifact into a new schema-2 package.

    No labels, population order, or descriptor configuration are inferred.  The
    caller supplies both source identities and column structure, while the v1
    file contributes only its declared matrix, sample IDs, validity mask, and a
    recorded source hash.  The source NPZ is opened read-only and never moved or
    overwritten.
    """
    source = Path(legacy_path).resolve()
    if not source.is_file():
        raise ArtifactReadError(f"旧 NPZ 文件不存在：{source}")
    source_hash = _sha256_file(source)
    try:
        with np.load(source, allow_pickle=False) as stored:
            required = {"X_smiles", "sample_ids", "valid_mask", "metadata_json"}
            missing = required.difference(stored.files)
            if missing:
                raise ArtifactReadError(f"旧 NPZ 缺少字段：{sorted(missing)}")
            matrix = np.asarray(stored["X_smiles"])
            sample_ids = np.asarray(stored["sample_ids"])
            valid_mask = np.asarray(stored["valid_mask"])
            metadata = json.loads(str(stored["metadata_json"].item()))
    except ArtifactReadError:
        raise
    except Exception as exc:
        raise ArtifactReadError(f"无法读取旧 NPZ：{exc}") from exc
    if metadata.get("artifact_schema_version") != 1:
        raise ArtifactReadError("仅支持显式导入 artifact_schema_version=1 的旧 NPZ")
    matrix, sample_ids, valid_mask = _normalise_publish_inputs(
        matrix, sample_ids, valid_mask, lifecycle="static_descriptor",
    )
    if metadata.get("feature_dim") != int(matrix.shape[1]):
        raise ArtifactReadError("旧 NPZ metadata.feature_dim 与矩阵不一致")
    if metadata.get("n_total") != int(len(sample_ids)) or metadata.get("n_valid") != int(valid_mask.sum()):
        raise ArtifactReadError("旧 NPZ 元数据的样本/有效性计数不一致")
    return publish_feature_artifact(
        target_dir,
        artifact_id=artifact_id,
        feature_id=feature_id,
        matrix=matrix,
        sample_ids=sample_ids,
        valid_mask=valid_mask,
        dataset_identity=dataset_identity,
        feature_config_identity=feature_config_identity,
        column_structure=column_structure,
        metadata={
            "legacy_import": {
                "source_filename": source.name,
                "source_sha256": source_hash,
                "source_artifact_schema_version": 1,
            }
        },
    )
