"""Manifest schema contract for portable, versioned feature artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ARTIFACT_SCHEMA_VERSION = "2.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LIFECYCLES = frozenset({"static_descriptor", "fold_transform", "derived"})
_STATUSES = frozenset({"ready", "partial", "failed"})
_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "artifact_id", "feature_id", "lifecycle", "status", "files",
    "matrix", "samples", "validity", "sources", "derived_from", "failure", "created_at",
    "metadata", "diagnostics",
})
_FILE_KEYS = frozenset({"path", "sha256", "bytes"})


class ArtifactContractError(ValueError):
    """Raised when a feature-artifact manifest cannot be trusted."""


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(f"{field} 必须是 mapping")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(f"{field} 必须是非空字符串")
    return value.strip()


def _require_identifier(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if not _IDENTIFIER.fullmatch(result):
        raise ArtifactContractError(f"{field} 只能含字母、数字、点、下划线和连字符")
    return result


def _reject_unknown(raw: Mapping[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ArtifactContractError(f"{field} 包含未知字段：{', '.join(unknown)}")


def _relative_reference(value: Any, field: str) -> str:
    reference = _require_string(value, field)
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactContractError(f"{field} 必须是相对于 manifest 的包内路径")
    return reference


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(f"{field} 必须是非负整数")
    return value


def _validate_file_records(value: Any) -> Mapping[str, Any]:
    files = _as_mapping(value, "files")
    required = {"matrix", "sample_ids", "validity_mask"}
    missing = sorted(required.difference(files))
    if missing:
        raise ArtifactContractError(f"files 缺少必需条目：{', '.join(missing)}")
    for key, record_raw in files.items():
        _require_identifier(key, "files 键")
        record = _as_mapping(record_raw, f"files.{key}")
        _reject_unknown(record, _FILE_KEYS, f"files.{key}")
        _relative_reference(record.get("path"), f"files.{key}.path")
        digest = _require_string(record.get("sha256"), f"files.{key}.sha256")
        if not _SHA256.fullmatch(digest):
            raise ArtifactContractError(f"files.{key}.sha256 必须是小写 SHA-256 十六进制摘要")
        if "bytes" in record:
            _nonnegative_integer(record["bytes"], f"files.{key}.bytes")
    return files


def _require_file_reference(value: Any, files: Mapping[str, Any], field: str) -> None:
    name = _require_string(value, field)
    if name not in files:
        raise ArtifactContractError(f"{field} 必须引用 files 中的条目：{name!r}")


def _validate_ready_payload(raw: Mapping[str, Any], files: Mapping[str, Any]) -> None:
    matrix = _as_mapping(raw.get("matrix"), "matrix")
    _require_file_reference(matrix.get("file"), files, "matrix.file")
    _require_string(matrix.get("dtype"), "matrix.dtype")
    shape = matrix.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 2:
        raise ArtifactContractError("matrix.shape 必须是 [n_rows, n_columns]")
    for index, item in enumerate(shape):
        _nonnegative_integer(item, f"matrix.shape[{index}]")
    columns = _as_mapping(matrix.get("columns"), "matrix.columns")
    _require_string(columns.get("kind"), "matrix.columns.kind")

    samples = _as_mapping(raw.get("samples"), "samples")
    _require_file_reference(samples.get("file"), files, "samples.file")
    _require_string(samples.get("id_field"), "samples.id_field")
    _nonnegative_integer(samples.get("row_count"), "samples.row_count")
    _require_string(samples.get("matrix_row_mapping"), "samples.matrix_row_mapping")

    validity = _as_mapping(raw.get("validity"), "validity")
    _require_file_reference(validity.get("file"), files, "validity.file")
    total = validity.get("n_total")
    valid = validity.get("n_valid")
    _nonnegative_integer(total, "validity.n_total")
    _nonnegative_integer(valid, "validity.n_valid")
    if valid > total:
        raise ArtifactContractError("validity.n_valid 不能大于 validity.n_total")
    if total != samples["row_count"]:
        raise ArtifactContractError("validity.n_total 必须等于 samples.row_count")
    if shape[0] != valid:
        raise ArtifactContractError("matrix.shape[0] 必须等于 validity.n_valid")
    _require_string(validity.get("semantics"), "validity.semantics")

    sources = _as_mapping(raw.get("sources"), "sources")
    _require_string(sources.get("dataset_identity"), "sources.dataset_identity")
    _require_string(sources.get("feature_config_identity"), "sources.feature_config_identity")


def _validate_diagnostics(raw: Mapping[str, Any], files: Mapping[str, Any]) -> None:
    diagnostics = _as_mapping(raw.get("diagnostics"), "diagnostics")
    _reject_unknown(
        diagnostics,
        frozenset({"file", "schema_version", "row_count", "sample_id_field", "row_mapping"}),
        "diagnostics",
    )
    _require_file_reference(diagnostics.get("file"), files, "diagnostics.file")
    _require_string(diagnostics.get("schema_version"), "diagnostics.schema_version")
    if _nonnegative_integer(diagnostics.get("row_count"), "diagnostics.row_count") != raw["samples"]["row_count"]:
        raise ArtifactContractError("diagnostics.row_count 必须等于 samples.row_count")
    if _require_string(diagnostics.get("sample_id_field"), "diagnostics.sample_id_field") != raw["samples"]["id_field"]:
        raise ArtifactContractError("diagnostics.sample_id_field 必须等于 samples.id_field")
    if diagnostics.get("row_mapping") != "full_sample_rows_in_sample_order":
        raise ArtifactContractError("diagnostics.row_mapping 必须为 full_sample_rows_in_sample_order")


def _validate_derived(raw: Mapping[str, Any]) -> None:
    parent = _as_mapping(raw.get("derived_from"), "derived_from")
    _require_identifier(parent.get("artifact_id"), "derived_from.artifact_id")
    _require_string(parent.get("operation"), "derived_from.operation")
    mapping = _as_mapping(parent.get("input_output_mapping"), "derived_from.input_output_mapping")
    if not mapping:
        raise ArtifactContractError("derived_from.input_output_mapping 不能为空")


def validate_artifact_manifest(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the artifact contract without importing any generator library."""
    manifest = _as_mapping(raw, "产物 manifest")
    _reject_unknown(manifest, _TOP_LEVEL_KEYS, "产物 manifest")
    if str(manifest.get("schema_version")) != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactContractError(
            f"schema_version 必须为 {ARTIFACT_SCHEMA_VERSION!r}"
        )
    _require_identifier(manifest.get("artifact_id"), "artifact_id")
    _require_identifier(manifest.get("feature_id"), "feature_id")
    lifecycle = _require_string(manifest.get("lifecycle"), "lifecycle")
    if lifecycle not in _LIFECYCLES:
        raise ArtifactContractError("lifecycle 不受支持")
    status = _require_string(manifest.get("status"), "status")
    if status not in _STATUSES:
        raise ArtifactContractError("status 必须为 ready、partial 或 failed")

    if status == "failed":
        failure = _as_mapping(manifest.get("failure"), "failure")
        _require_string(failure.get("reason"), "failure.reason")
    else:
        files = _validate_file_records(manifest.get("files"))
        _validate_ready_payload(manifest, files)
        if "diagnostics" in manifest:
            _validate_diagnostics(manifest, files)
    if lifecycle == "derived":
        _validate_derived(manifest)
    elif "derived_from" in manifest:
        raise ArtifactContractError("只有 lifecycle=derived 的产物可以包含 derived_from")
    if "metadata" in manifest and not isinstance(manifest["metadata"], Mapping):
        raise ArtifactContractError("metadata 必须是 mapping")
    return copy.deepcopy(dict(manifest))


def resolve_manifest_path(manifest_path: Path | str, reference: str) -> Path:
    """Resolve a validated package-internal reference after artifact relocation."""
    path = Path(manifest_path).resolve()
    relative = _relative_reference(reference, "manifest 路径引用")
    return (path.parent / relative).resolve()


def artifact_content_identity(raw: Mapping[str, Any]) -> str:
    """Hash semantic content while excluding locations, timestamps, and comments.

    Every physical file still contributes its recorded SHA-256.  Therefore a
    copied artifact package has the same identity, while a changed matrix or
    sample mapping cannot silently share an identity.
    """
    manifest = validate_artifact_manifest(raw)
    identity = copy.deepcopy(manifest)
    identity.pop("created_at", None)
    identity.pop("metadata", None)
    for record in identity.get("files", {}).values():
        record.pop("path", None)
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
