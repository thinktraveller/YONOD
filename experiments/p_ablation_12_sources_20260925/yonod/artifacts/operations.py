"""Read-only inspection and immutable schema-2 feature derivations.

This module sits above the dependency-light artifact reader/publisher.  It
does not import descriptor builders, model code, RDKit, Torch, or a runtime
entry point.  Operation YAML is deliberately a separate, narrow interface:
it can select a declared population or named feature columns, and can attach
explicit numeric columns from a one-to-one external CSV table.

Every successful operation publishes a fresh ``lifecycle: derived`` package.
The source package is opened only through :mod:`yonod.artifacts.reader` and is
never modified.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from yonod.config.contracts import resolve_config_path
from yonod.config.loader import ConfigLoadError, load_operation_config

from .contracts import artifact_content_identity
from .reader import ArtifactReadError, FeatureArtifact, load_feature_artifact, publish_feature_artifact


class ArtifactOperationError(ArtifactReadError):
    """Raised when a declared feature inspection or derivation is unsafe."""


@dataclass(frozen=True)
class ArtifactInspection:
    """Small, immutable description returned by :func:`inspect_feature_artifact`."""

    manifest_path: Path
    artifact_id: str
    feature_id: str
    lifecycle: str
    status: str
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    column_names: tuple[str, ...] | None
    sample_count: int
    valid_sample_count: int
    dataset_identity: str
    feature_config_identity: str
    parent_artifact_id: str | None
    content_identity: str


@dataclass(frozen=True)
class DerivedOperation:
    """Persistent reference for one successfully published operation."""

    index: int
    kind: str
    parent_artifact_id: str
    artifact_id: str
    manifest_path: Path
    operation_identity: str


@dataclass(frozen=True)
class DeriveResult:
    """Persistent references for an ordered immutable derivation chain."""

    operation_config_path: Path
    input_manifest_path: Path
    output_root: Path
    operations: tuple[DerivedOperation, ...]

    @property
    def manifest_path(self) -> Path:
        """Return the final manifest path (the config contract requires one step)."""
        return self.operations[-1].manifest_path


_DEFAULT_PROTECTED_COLUMNS = frozenset({"label", "target", "yield", "y"})
_NUMERIC_KINDS = frozenset({"b", "i", "u", "f"})


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactOperationError(f"{field} 必须是非空字符串")
    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ArtifactOperationError(f"{field} 必须是非空字符串列表")
    result = [_require_string(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise ArtifactOperationError(f"{field} 不能包含重复值")
    return result


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactOperationError(f"{field} 必须是 mapping")
    return copy.deepcopy(dict(value))


def _reject_unknown(mapping: Mapping[str, Any], allowed: Iterable[str], field: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in set(allowed))
    if unknown:
        raise ArtifactOperationError(f"{field} 包含未知字段：{', '.join(unknown)}")


def _canonical_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ArtifactOperationError("操作声明必须只包含可序列化的 YAML 标量/列表/mapping") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    """Hash an array with its dtype and shape so a mapping is auditable."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _safe_output_root(output_root: Path, parent_manifest: Path) -> None:
    """Avoid even creating a child directory inside a parent artifact package."""
    parent_package = parent_manifest.resolve().parent
    try:
        output_root.resolve().relative_to(parent_package)
    except ValueError:
        return
    raise ArtifactOperationError("output_dir 不能位于父产物包内；这会修改父产物目录")


def _require_numeric_parent(artifact: FeatureArtifact) -> None:
    if artifact.matrix.dtype.kind not in _NUMERIC_KINDS:
        raise ArtifactOperationError(
            "derive_features 目前只接受数值 static_descriptor/derived 产物；"
            "未拟合 fold_transform 原始类别必须保留给训练折内处理"
        )


def _named_columns(artifact: FeatureArtifact, *, purpose: str) -> list[str]:
    structure = artifact.manifest["matrix"]["columns"]
    if structure.get("kind") != "named":
        raise ArtifactOperationError(f"{purpose} 需要 matrix.columns.kind=named 的稳定列名")
    names = structure.get("names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ArtifactOperationError("产物 matrix.columns.names 无效")
    return [str(value) for value in names]


def _normalise_protected_columns(config: Mapping[str, Any]) -> frozenset[str]:
    raw = config.get("protected_columns", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ArtifactOperationError("protected_columns 必须是字符串列表")
    protected = set(_DEFAULT_PROTECTED_COLUMNS)
    for index, value in enumerate(raw):
        protected.add(_require_string(value, f"protected_columns[{index}]").casefold())
    return frozenset(protected)


def _reject_protected_columns(columns: Iterable[str], protected: frozenset[str], *, field: str) -> None:
    leaking = [str(column) for column in columns if str(column).strip().casefold() in protected]
    if leaking:
        raise ArtifactOperationError(
            f"{field} 包含受保护的标签/目标列，拒绝标签泄漏：{', '.join(leaking)}"
        )


def _validate_parent_for_derivation(artifact: FeatureArtifact, protected: frozenset[str]) -> None:
    _require_numeric_parent(artifact)
    columns = artifact.manifest["matrix"]["columns"]
    if columns.get("kind") == "named":
        _reject_protected_columns(columns.get("names", []), protected, field="父产物特征列")


def _step_kind(raw: Mapping[str, Any], index: int) -> str:
    return _require_string(raw.get("kind"), f"operations[{index}].kind")


def _normalise_select_samples(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    field = f"operations[{index}]"
    step = _mapping(raw, field)
    _reject_unknown(step, {"kind", "sample_ids"}, field)
    if _step_kind(step, index) != "select_samples":
        raise ArtifactOperationError(f"{field}.kind 必须为 select_samples")
    return {"kind": "select_samples", "sample_ids": _string_list(step.get("sample_ids"), f"{field}.sample_ids")}


def _normalise_select_features(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    field = f"operations[{index}]"
    step = _mapping(raw, field)
    _reject_unknown(step, {"kind", "columns"}, field)
    if _step_kind(step, index) != "select_features":
        raise ArtifactOperationError(f"{field}.kind 必须为 select_features")
    return {"kind": "select_features", "columns": _string_list(step.get("columns"), f"{field}.columns")}


def _normalise_join_features(raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    field = f"operations[{index}]"
    step = _mapping(raw, field)
    _reject_unknown(step, {"kind", "table_path", "sample_id_col", "columns", "prefix"}, field)
    if _step_kind(step, index) != "join_features":
        raise ArtifactOperationError(f"{field}.kind 必须为 join_features")
    prefix = _require_string(step.get("prefix"), f"{field}.prefix")
    return {
        "kind": "join_features",
        "table_path": _require_string(step.get("table_path"), f"{field}.table_path"),
        "sample_id_col": _require_string(step.get("sample_id_col"), f"{field}.sample_id_col"),
        "columns": _string_list(step.get("columns"), f"{field}.columns"),
        "prefix": prefix,
    }


def _normalise_step(raw: Any, index: int) -> Dict[str, Any]:
    step = _mapping(raw, f"operations[{index}]")
    kind = _step_kind(step, index)
    if kind == "select_samples":
        return _normalise_select_samples(step, index)
    if kind == "select_features":
        return _normalise_select_features(step, index)
    if kind == "join_features":
        return _normalise_join_features(step, index)
    # Static config validation protects this path too; retaining this error makes
    # the internal API safe when called with a future contract implementation.
    raise ArtifactOperationError(f"operations[{index}].kind 不受支持：{kind!r}")


def _select_samples(artifact: FeatureArtifact, step: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    requested = list(step["sample_ids"])
    input_index = {sample_id: index for index, sample_id in enumerate(artifact.sample_ids.tolist())}
    missing = [sample_id for sample_id in requested if sample_id not in input_index]
    if missing:
        raise ArtifactOperationError(f"select_samples 请求的 sample_id 不在父产物中：{', '.join(missing)}")
    source_indices = np.asarray([input_index[sample_id] for sample_id in requested], dtype=np.int64)
    output_ids = np.asarray(requested, dtype=np.str_)
    output_mask = np.asarray(artifact.valid_mask[source_indices], dtype=bool)
    matrix_row_by_sample_index = {
        int(sample_index): matrix_index
        for matrix_index, sample_index in enumerate(artifact.valid_row_indices.tolist())
    }
    selected_rows = [
        matrix_row_by_sample_index[int(sample_index)]
        for sample_index, is_valid in zip(source_indices.tolist(), output_mask.tolist())
        if is_valid
    ]
    output_matrix = np.asarray(artifact.matrix[selected_rows, :]).copy()
    mapping = {
        "mapping_kind": "sample_selection",
        "input_sample_ids_sha256": _array_hash(artifact.sample_ids),
        "output_sample_ids_sha256": _array_hash(output_ids),
        "input_validity_mask_sha256": _array_hash(artifact.valid_mask),
        "output_validity_mask_sha256": _array_hash(output_mask),
        "output_sample_index_to_input_sample_index": {
            str(output_index): int(input_sample_index)
            for output_index, input_sample_index in enumerate(source_indices.tolist())
        },
        "output_sample_order": "operation sample_ids declaration order",
    }
    declaration = {"kind": "select_samples", "sample_ids": requested}
    return output_matrix, output_ids, output_mask, mapping, declaration


def _select_features(artifact: FeatureArtifact, step: Mapping[str, Any], protected: frozenset[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    names = _named_columns(artifact, purpose="select_features")
    requested = list(step["columns"])
    _reject_protected_columns(requested, protected, field="select_features.columns")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested if name not in positions]
    if missing:
        raise ArtifactOperationError(f"select_features 请求的列不存在：{', '.join(missing)}")
    source_indices = [positions[name] for name in requested]
    output_matrix = np.asarray(artifact.matrix[:, source_indices]).copy()
    mapping = {
        "mapping_kind": "feature_selection",
        "input_column_names_sha256": _canonical_hash(names),
        "output_column_names_sha256": _canonical_hash(requested),
        "output_feature_index_to_input_feature_index": {
            str(output_index): int(input_feature_index)
            for output_index, input_feature_index in enumerate(source_indices)
        },
        "sample_order": "unchanged_from_parent",
    }
    declaration = {"kind": "select_features", "columns": requested}
    structure = {"kind": "named", "names": requested}
    return output_matrix, artifact.sample_ids, artifact.valid_mask, mapping, declaration, structure


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_external_numeric_table(path: Path, *, sample_id_col: str, columns: Sequence[str]) -> tuple[Dict[str, np.ndarray], int]:
    if path.suffix.lower() != ".csv":
        raise ArtifactOperationError("join_features.table_path 目前仅支持 UTF-8 CSV 外部表")
    if not path.is_file():
        raise ArtifactOperationError(f"join_features 外部表不存在：{path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.reader(handle)
            try:
                header = next(rows)
            except StopIteration as exc:
                raise ArtifactOperationError("join_features 外部表不能为空") from exc
            if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                raise ArtifactOperationError("join_features 外部表列名必须非空且唯一")
            required = [sample_id_col, *columns]
            missing = [name for name in required if name not in header]
            if missing:
                raise ArtifactOperationError(f"join_features 外部表缺少显式列：{', '.join(missing)}")
            positions = {name: index for index, name in enumerate(header)}
            result: Dict[str, np.ndarray] = {}
            for row_number, row in enumerate(rows, start=2):
                if len(row) != len(header):
                    raise ArtifactOperationError(f"join_features 外部表第 {row_number} 行列数不匹配表头")
                sample_id = row[positions[sample_id_col]].strip()
                if not sample_id:
                    raise ArtifactOperationError(f"join_features 外部表第 {row_number} 行 sample_id 为空")
                if sample_id in result:
                    raise ArtifactOperationError(f"join_features 外部表 sample_id 重复：{sample_id}")
                try:
                    values = np.asarray([row[positions[column]] for column in columns], dtype=np.float64)
                except ValueError as exc:
                    raise ArtifactOperationError(
                        f"join_features 外部表第 {row_number} 行的显式特征列必须是数值"
                    ) from exc
                if not np.isfinite(values).all():
                    raise ArtifactOperationError(
                        f"join_features 外部表第 {row_number} 行的显式特征列不能为缺失或无穷值"
                    )
                result[sample_id] = values
    except UnicodeDecodeError as exc:
        raise ArtifactOperationError("join_features 外部表必须是 UTF-8 CSV") from exc
    return result, len(result)


def _join_features(
    artifact: FeatureArtifact,
    step: Mapping[str, Any],
    *,
    config_path: Path,
    protected: frozenset[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base_names = _named_columns(artifact, purpose="join_features")
    source_columns = list(step["columns"])
    _reject_protected_columns(source_columns, protected, field="join_features.columns")
    sample_id_col = str(step["sample_id_col"])
    if sample_id_col in source_columns:
        raise ArtifactOperationError("join_features.columns 不能包含 sample_id_col")
    prefix = str(step["prefix"])
    output_names = [f"{prefix}{column}" for column in source_columns]
    if any(not name.strip() for name in output_names):
        raise ArtifactOperationError("join_features 产生了空列名")
    if len(set(output_names)) != len(output_names) or set(output_names).intersection(base_names):
        raise ArtifactOperationError("join_features 输出列与父产物列冲突")

    table_path = resolve_config_path(config_path, str(step["table_path"]))
    external_values, external_row_count = _read_external_numeric_table(
        table_path, sample_id_col=sample_id_col, columns=source_columns,
    )
    parent_ids = artifact.sample_ids.tolist()
    missing = [sample_id for sample_id in parent_ids if sample_id not in external_values]
    if missing:
        raise ArtifactOperationError(
            f"join_features 外部表缺少父产物 sample_id：{', '.join(missing)}"
        )
    parent_id_set = set(parent_ids)
    ignored_count = sum(1 for sample_id in external_values if sample_id not in parent_id_set)
    aligned = np.asarray([external_values[sample_id] for sample_id in parent_ids], dtype=np.float64)
    valid_external = aligned[np.asarray(artifact.valid_mask, dtype=bool)]
    output_matrix = np.concatenate((np.asarray(artifact.matrix), valid_external), axis=1)
    table_hash = _sha256_file(table_path)
    mapping = {
        "mapping_kind": "one_to_one_external_feature_join",
        "input_sample_ids_sha256": _array_hash(artifact.sample_ids),
        "output_sample_ids_sha256": _array_hash(artifact.sample_ids),
        "sample_order": "unchanged_from_parent",
        "external_table_sha256": table_hash,
        "external_sample_id_column": sample_id_col,
        "external_selected_columns": source_columns,
        "external_row_count": int(external_row_count),
        "external_rows_ignored_outside_parent_population": int(ignored_count),
    }
    declaration = {
        "kind": "join_features",
        "external_table_filename": table_path.name,
        "external_table_sha256": table_hash,
        "sample_id_col": sample_id_col,
        "columns": source_columns,
        "prefix": prefix,
    }
    structure = {"kind": "named", "names": [*base_names, *output_names]}
    return output_matrix, artifact.sample_ids, artifact.valid_mask, mapping, declaration, structure


def _operation_identity(parent: FeatureArtifact, declaration: Mapping[str, Any]) -> str:
    return _canonical_hash({
        "parent_content_identity": artifact_content_identity(parent.manifest),
        "operation": declaration,
    })


def _derived_metadata(
    parent: FeatureArtifact,
    declaration: Mapping[str, Any],
    operation_identity: str,
    config_metadata: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "parent_content_identity": artifact_content_identity(parent.manifest),
        "derived_operation": {
            "kind": declaration["kind"],
            "declaration": copy.deepcopy(dict(declaration)),
            "operation_identity": operation_identity,
        },
    }
    if config_metadata:
        metadata["operation_config_metadata"] = copy.deepcopy(dict(config_metadata))
    return metadata


def _synchronise_diagnostics(parent: FeatureArtifact, output_sample_ids: np.ndarray) -> Dict[str, Any] | None:
    """Carry full-row sidecar records through selection/reordering unchanged in meaning."""

    if parent.diagnostics is None:
        return None
    rows = parent.diagnostics.get("rows")
    if not isinstance(rows, list):  # Reader already validates; retain a safe guard for future callers.
        raise ArtifactOperationError("父产物 diagnostics.rows 无效")
    by_sample_id = {str(row["sample_id"]): row for row in rows}
    try:
        selected_rows = [copy.deepcopy(by_sample_id[str(sample_id)]) for sample_id in output_sample_ids.tolist()]
    except KeyError as exc:
        raise ArtifactOperationError("父产物 diagnostics 与 sample_ids 不一致") from exc
    result = copy.deepcopy(parent.diagnostics)
    result["rows"] = selected_rows
    return result


def inspect_feature_artifact(manifest_path: Path | str) -> ArtifactInspection:
    """Return verified artifact facts without creating, changing, or caching files."""
    artifact = load_feature_artifact(manifest_path)
    columns = artifact.manifest["matrix"]["columns"]
    names: tuple[str, ...] | None = None
    if columns.get("kind") == "named":
        names = tuple(str(name) for name in columns["names"])
    parent = artifact.manifest.get("derived_from", {})
    return ArtifactInspection(
        manifest_path=artifact.manifest_path,
        artifact_id=artifact.manifest["artifact_id"],
        feature_id=artifact.manifest["feature_id"],
        lifecycle=artifact.manifest["lifecycle"],
        status=artifact.manifest["status"],
        matrix_shape=tuple(int(value) for value in artifact.matrix.shape),
        matrix_dtype=artifact.matrix.dtype.str,
        column_names=names,
        sample_count=int(len(artifact.sample_ids)),
        valid_sample_count=int(artifact.valid_mask.sum()),
        dataset_identity=artifact.manifest["sources"]["dataset_identity"],
        feature_config_identity=artifact.manifest["sources"]["feature_config_identity"],
        parent_artifact_id=parent.get("artifact_id"),
        content_identity=artifact_content_identity(artifact.manifest),
    )


def derive_features(operation_config_path: Path | str) -> DeriveResult:
    """Apply ordered schema-2 operations and publish a new immutable package per step.

    The operation config is first loaded through the shared safe YAML loader.
    Operations execute in declaration order.  A later failure leaves already
    validated earlier derived packages intact, while never changing their
    parents or overwriting an existing destination.
    """
    config_path = Path(operation_config_path).resolve()
    try:
        config = load_operation_config(config_path)
    except (ConfigLoadError, FileNotFoundError) as exc:
        raise ArtifactOperationError(f"操作配置无效：{exc}") from exc
    input_manifest = resolve_config_path(config_path, config["input_manifest"])
    output_root = resolve_config_path(config_path, config["output_dir"])
    try:
        current = load_feature_artifact(input_manifest)
    except ArtifactReadError as exc:
        raise ArtifactOperationError(f"父产物不可读取：{exc}") from exc
    protected = _normalise_protected_columns(config)
    config_metadata = config.get("metadata")
    if config_metadata is not None and not isinstance(config_metadata, Mapping):
        raise ArtifactOperationError("metadata 必须是 mapping")

    results: list[DerivedOperation] = []
    for index, raw_step in enumerate(config["operations"]):
        _safe_output_root(output_root, current.manifest_path)
        _validate_parent_for_derivation(current, protected)
        step = _normalise_step(raw_step, index)
        kind = step["kind"]
        if kind == "select_samples":
            matrix, sample_ids, valid_mask, mapping, declaration = _select_samples(current, step)
            column_structure = copy.deepcopy(current.manifest["matrix"]["columns"])
        elif kind == "select_features":
            matrix, sample_ids, valid_mask, mapping, declaration, column_structure = _select_features(
                current, step, protected,
            )
        else:
            matrix, sample_ids, valid_mask, mapping, declaration, column_structure = _join_features(
                current, step, config_path=config_path, protected=protected,
            )
        operation_identity = _operation_identity(current, declaration)
        artifact_id = f"{current.manifest['artifact_id']}-derived-{kind}-{operation_identity[:16]}"
        target_dir = output_root / artifact_id
        if target_dir.exists():
            raise ArtifactOperationError(f"派生产物目录已存在，拒绝覆盖：{target_dir}")
        derived_from = {
            "artifact_id": current.manifest["artifact_id"],
            "operation": kind,
            "input_output_mapping": mapping,
        }
        try:
            diagnostics = _synchronise_diagnostics(current, np.asarray(sample_ids, dtype=np.str_))
            manifest = publish_feature_artifact(
                target_dir,
                artifact_id=artifact_id,
                feature_id=current.manifest["feature_id"],
                matrix=matrix,
                sample_ids=sample_ids,
                valid_mask=valid_mask,
                dataset_identity=current.manifest["sources"]["dataset_identity"],
                feature_config_identity=f"sha256:{operation_identity}",
                column_structure=column_structure,
                lifecycle="derived",
                derived_from=derived_from,
                metadata=_derived_metadata(current, declaration, operation_identity, config_metadata),
                diagnostics=diagnostics,
            )
        except ArtifactReadError as exc:
            raise ArtifactOperationError(f"无法发布 {kind} 派生产物：{exc}") from exc
        results.append(DerivedOperation(
            index=index,
            kind=kind,
            parent_artifact_id=current.manifest["artifact_id"],
            artifact_id=artifact_id,
            manifest_path=manifest,
            operation_identity=operation_identity,
        ))
        # Go through the public reader again so the next operation consumes the
        # exact immutable package published by the previous one.
        current = load_feature_artifact(manifest)
    return DeriveResult(
        operation_config_path=config_path,
        input_manifest_path=input_manifest,
        output_root=output_root,
        operations=tuple(results),
    )
