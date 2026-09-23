"""Schema contracts introduced by step 30.1.

These functions validate already-parsed mappings and have no YAML loader on
purpose.  Step 30.2 will provide the single secure YAML loader and call this
module before any data, descriptor, or model dependency is loaded.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


CONFIG_SCHEMA_VERSION = "2.0"
MISSING = object()


class ConfigContractError(ValueError):
    """Raised when a runtime or feature-operation configuration is invalid."""


# This is a routing contract, rather than a permissive ``**kwargs`` promise.
# Parameter-name validation against the installed libraries is intentionally a
# step-30.6 responsibility; callers may only place parameters in these named
# API sections.
MODEL_PARAMETER_ROUTES: Dict[str, frozenset[str]] = {
    "rf": frozenset({"estimator", "fit", "runtime"}),
    "xgb": frozenset({"estimator", "fit", "runtime"}),
    "svm": frozenset({"estimator", "fit", "preprocessing", "runtime"}),
    "lightgbm": frozenset({"estimator", "fit", "runtime"}),
    "autogluon": frozenset({"predictor", "fit", "runtime"}),
}

_STAGES = frozenset({"features", "train", "all", "benchmark"})
_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "project_name", "stage", "dataset", "descriptors",
    "artifacts", "models", "model_params", "evaluation", "outputs", "benchmark", "metadata",
})
_DATASET_KEYS = frozenset({"path", "sample_id_col", "column_roles", "numeric_conditions"})
_COLUMN_ROLE_KEYS = frozenset({
    "label", "reactants", "products", "others", "conditions", "categoricals",
})
_DESCRIPTOR_KEYS = frozenset({
    "id", "descriptor", "lifecycle", "mode", "columns", "extra_reactants", "params",
})
_ARTIFACT_KEYS = frozenset({"output_dir", "input_manifest"})
_OUTPUT_KEYS = frozenset({"root", "report_formats"})
_EVALUATION_KEYS = frozenset({
    "protocol", "n_splits", "n_repeats", "shuffle", "seed", "grouping",
    "split_manifest", "population_manifest", "metrics",
})
_BENCHMARK_KEYS = frozenset({"task_state", "reproduction_protocol", "population_id", "dataset_id"})
_BENCHMARK_TASK_STATE_KEYS = frozenset({"backend", "resumable"})
_OPERATION_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "operation", "input_manifest", "output_dir", "operations", "protected_columns", "metadata",
})
_DERIVE_KINDS = frozenset({"select_samples", "select_features", "join_features"})


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigContractError(f"{field} 必须是 mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ConfigContractError(f"{field} 包含未知字段：{', '.join(unknown)}")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigContractError(f"{field} 必须是非空字符串")
    return value.strip()


def _string_list(value: Any, field: str, *, required: bool = False) -> list[str]:
    if value is None:
        if required:
            raise ConfigContractError(f"{field} 必须是非空字符串列表")
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigContractError(f"{field} 必须是字符串列表")
    result = [_require_string(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if required and not result:
        raise ConfigContractError(f"{field} 必须是非空字符串列表")
    if len(set(result)) != len(result):
        raise ConfigContractError(f"{field} 不能包含重复值")
    return result


def _integer_at_least(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigContractError(f"{field} 必须为不小于 {minimum} 的整数")
    return value


def _validate_schema_version(raw: Mapping[str, Any]) -> None:
    version = raw.get("schema_version")
    if str(version) != CONFIG_SCHEMA_VERSION:
        raise ConfigContractError(
            f"schema_version 必须为 {CONFIG_SCHEMA_VERSION!r}，收到 {version!r}"
        )


def _validate_dataset(raw: Mapping[str, Any], *, stage: str) -> None:
    _reject_unknown(raw, _DATASET_KEYS, "dataset")
    _require_string(raw.get("path"), "dataset.path")
    _require_string(raw.get("sample_id_col"), "dataset.sample_id_col")
    roles = _as_mapping(raw.get("column_roles"), "dataset.column_roles")
    _reject_unknown(roles, _COLUMN_ROLE_KEYS, "dataset.column_roles")

    label = roles.get("label", MISSING)
    if stage in {"train", "all", "benchmark"}:
        _require_string(label, "dataset.column_roles.label")
    elif label is not MISSING and label is not None:
        _require_string(label, "dataset.column_roles.label")

    for name in ("reactants", "products", "others", "conditions", "categoricals"):
        if name in roles:
            _string_list(roles[name], f"dataset.column_roles.{name}")
    conditions = _string_list(roles.get("conditions"), "dataset.column_roles.conditions")
    protected = {str(raw["sample_id_col"]), *[str(value) for value in roles.get("reactants", []) or []],
                 *[str(value) for value in roles.get("products", []) or []],
                 *[str(value) for value in roles.get("others", []) or []],
                 *[str(value) for value in roles.get("categoricals", []) or []]}
    if isinstance(label, str):
        protected.add(label)
    overlaps = sorted(set(conditions) & protected)
    if overlaps:
        raise ConfigContractError("dataset.column_roles.conditions 与 ID、标签或其他输入角色重叠：" + ", ".join(overlaps))
    if "numeric_conditions" in raw or conditions:
        try:
            from yonod.features.numeric_conditions import NumericConditionsError, normalise_numeric_contract
            normalise_numeric_contract(raw)
        except NumericConditionsError as exc:
            raise ConfigContractError(str(exc)) from exc


def _validate_descriptors(value: Any, *, required: bool) -> None:
    if value is None:
        if required:
            raise ConfigContractError("descriptors 必须是非空对象列表")
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or (required and not value):
        raise ConfigContractError("descriptors 必须是非空对象列表" if required else "descriptors 必须是对象列表")
    feature_ids: set[str] = set()
    for index, item in enumerate(value):
        field = f"descriptors[{index}]"
        entry = _as_mapping(item, field)
        _reject_unknown(entry, _DESCRIPTOR_KEYS, field)
        feature_id = _require_string(entry.get("id"), f"{field}.id")
        if feature_id in feature_ids:
            raise ConfigContractError(f"descriptors 的 id 重复：{feature_id!r}")
        feature_ids.add(feature_id)
        _require_string(entry.get("descriptor"), f"{field}.descriptor")
        if "lifecycle" in entry:
            lifecycle = _require_string(entry["lifecycle"], f"{field}.lifecycle")
            if lifecycle not in {"static_descriptor", "fold_transform"}:
                raise ConfigContractError(f"{field}.lifecycle 不受支持：{lifecycle!r}")
        if "mode" in entry:
            _require_string(entry["mode"], f"{field}.mode")
        if "columns" in entry:
            _string_list(entry["columns"], f"{field}.columns")
        if "extra_reactants" in entry:
            _string_list(entry["extra_reactants"], f"{field}.extra_reactants")
        if "params" in entry and not isinstance(entry["params"], Mapping):
            raise ConfigContractError(f"{field}.params 必须是 mapping")


def _validate_artifacts(raw: Mapping[str, Any], *, stage: str) -> None:
    _reject_unknown(raw, _ARTIFACT_KEYS, "artifacts")
    if stage in {"features", "all", "benchmark"}:
        _require_string(raw.get("output_dir"), "artifacts.output_dir")
    elif "output_dir" in raw and raw["output_dir"] is not None:
        _require_string(raw["output_dir"], "artifacts.output_dir")
    if stage == "train":
        _require_string(raw.get("input_manifest"), "artifacts.input_manifest")
    elif "input_manifest" in raw and raw["input_manifest"] is not None:
        _require_string(raw["input_manifest"], "artifacts.input_manifest")


def _validate_models(raw: Mapping[str, Any], *, required: bool) -> list[str]:
    value = raw.get("models", MISSING)
    if value is MISSING:
        if required:
            raise ConfigContractError("models 必须是非空模型标识符列表")
        return []
    models = _string_list(value, "models", required=required)
    unknown = [model for model in models if model not in MODEL_PARAMETER_ROUTES]
    if unknown:
        raise ConfigContractError(f"models 包含不支持的模型：{', '.join(unknown)}")
    return models


def _validate_model_params(raw: Mapping[str, Any], models: Sequence[str]) -> None:
    value = raw.get("model_params", {})
    mapping = _as_mapping(value, "model_params")
    for model_name, sections in mapping.items():
        model = _require_string(model_name, "model_params 模型名")
        if model not in MODEL_PARAMETER_ROUTES:
            raise ConfigContractError(f"model_params.{model} 不属于受支持模型")
        if models and model not in models:
            raise ConfigContractError(f"model_params.{model} 未出现在 models 中")
        section_mapping = _as_mapping(sections, f"model_params.{model}")
        unknown = sorted(str(key) for key in section_mapping if key not in MODEL_PARAMETER_ROUTES[model])
        if unknown:
            allowed = ", ".join(sorted(MODEL_PARAMETER_ROUTES[model]))
            raise ConfigContractError(
                f"model_params.{model} 的参数区段无路由：{', '.join(unknown)}；允许：{allowed}"
            )
        for section, params in section_mapping.items():
            if not isinstance(params, Mapping):
                raise ConfigContractError(f"model_params.{model}.{section} 必须是 mapping")


def _validate_evaluation(value: Any) -> None:
    if value is None:
        return
    mapping = _as_mapping(value, "evaluation")
    _reject_unknown(mapping, _EVALUATION_KEYS, "evaluation")
    for name, minimum in (("n_splits", 2), ("n_repeats", 1)):
        if name in mapping:
            _integer_at_least(mapping[name], f"evaluation.{name}", minimum)
    if "seed" in mapping:
        _integer_at_least(mapping["seed"], "evaluation.seed", -(2 ** 63))
    if "protocol" in mapping:
        _require_string(mapping["protocol"], "evaluation.protocol")
    for name in ("split_manifest", "population_manifest"):
        if name in mapping and mapping[name] is not None:
            _require_string(mapping[name], f"evaluation.{name}")


def _validate_outputs(value: Any) -> None:
    if value is None:
        return
    mapping = _as_mapping(value, "outputs")
    _reject_unknown(mapping, _OUTPUT_KEYS, "outputs")
    if "root" in mapping:
        _require_string(mapping["root"], "outputs.root")
    if "report_formats" in mapping:
        formats = _string_list(mapping["report_formats"], "outputs.report_formats", required=True)
        normalised = [item.strip().lower() for item in formats]
        unknown = sorted(set(normalised) - {"html", "markdown", "md"})
        if unknown:
            raise ConfigContractError(
                "outputs.report_formats 仅支持 html、markdown（或 md）：" + ", ".join(unknown)
            )
        if len(set("markdown" if item == "md" else item for item in normalised)) != len(normalised):
            raise ConfigContractError("outputs.report_formats 不能包含重复格式（markdown 与 md 视为同一格式）")


def _validate_benchmark(value: Any, *, stage: str) -> None:
    """Validate the small schema-2 block unique to strict benchmarks.

    The generic runtime never consumes this block.  Keeping it explicit means
    a normal ``outer_kfold`` task cannot accidentally acquire resumable
    benchmark state or a benchmark cannot silently omit that state contract.
    """
    if stage != "benchmark":
        if value is not None:
            raise ConfigContractError("benchmark 仅可用于 stage: benchmark")
        return
    mapping = _as_mapping(value, "benchmark")
    _reject_unknown(mapping, _BENCHMARK_KEYS, "benchmark")
    task_state = _as_mapping(mapping.get("task_state"), "benchmark.task_state")
    _reject_unknown(task_state, _BENCHMARK_TASK_STATE_KEYS, "benchmark.task_state")
    if task_state.get("backend") != "sqlite":
        raise ConfigContractError("benchmark.task_state.backend 必须为 sqlite")
    if task_state.get("resumable") is not True:
        raise ConfigContractError("benchmark.task_state.resumable 必须为 true")
    protocol = mapping.get("reproduction_protocol", {})
    if not isinstance(protocol, Mapping):
        raise ConfigContractError("benchmark.reproduction_protocol 必须是 mapping")
    for name in ("population_id", "dataset_id"):
        if name in mapping:
            _require_string(mapping[name], f"benchmark.{name}")


def validate_run_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the step-30 runtime schema without reading referenced files.

    The result is a detached deep copy, which permits a future loader to add
    resolved-path views without mutating a caller's parsed YAML mapping.
    """
    mapping = _as_mapping(raw, "运行配置")
    _reject_unknown(mapping, _TOP_LEVEL_KEYS, "运行配置")
    _validate_schema_version(mapping)
    _require_string(mapping.get("project_name"), "project_name")
    stage = _require_string(mapping.get("stage"), "stage")
    if stage not in _STAGES:
        raise ConfigContractError("stage 必须为 features、train、all 或 benchmark")

    _validate_dataset(_as_mapping(mapping.get("dataset"), "dataset"), stage=stage)
    _validate_artifacts(_as_mapping(mapping.get("artifacts"), "artifacts"), stage=stage)
    _validate_descriptors(mapping.get("descriptors"), required=stage in {"features", "all", "benchmark"})
    models = _validate_models(mapping, required=stage in {"train", "all", "benchmark"})
    _validate_model_params(mapping, models)
    _validate_evaluation(mapping.get("evaluation"))
    _validate_outputs(mapping.get("outputs"))
    _validate_benchmark(mapping.get("benchmark"), stage=stage)
    if "metadata" in mapping and not isinstance(mapping["metadata"], Mapping):
        raise ConfigContractError("metadata 必须是 mapping")
    return copy.deepcopy(dict(mapping))


def validate_operation_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a separate, model-free intermediate-feature operation config."""
    mapping = _as_mapping(raw, "操作配置")
    _reject_unknown(mapping, _OPERATION_TOP_LEVEL_KEYS, "操作配置")
    _validate_schema_version(mapping)
    if _require_string(mapping.get("operation"), "operation") != "derive_features":
        raise ConfigContractError("operation 目前仅支持 derive_features")
    _require_string(mapping.get("input_manifest"), "input_manifest")
    _require_string(mapping.get("output_dir"), "output_dir")
    steps = mapping.get("operations")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ConfigContractError("operations 必须是非空操作列表")
    for index, step in enumerate(steps):
        entry = _as_mapping(step, f"operations[{index}]")
        kind = _require_string(entry.get("kind"), f"operations[{index}].kind")
        if kind not in _DERIVE_KINDS:
            raise ConfigContractError(f"operations[{index}].kind 不受支持：{kind!r}")
    if "protected_columns" in mapping:
        _string_list(mapping["protected_columns"], "protected_columns")
    if "metadata" in mapping and not isinstance(mapping["metadata"], Mapping):
        raise ConfigContractError("metadata 必须是 mapping")
    return copy.deepcopy(dict(mapping))


def resolve_config_path(config_path: Path | str, reference: str) -> Path:
    """Resolve a runtime-config reference relative to its owning YAML file."""
    owner = Path(config_path).resolve()
    target = Path(_require_string(reference, "配置路径引用"))
    return target.resolve() if target.is_absolute() else (owner.parent / target).resolve()
