"""Secure YAML loading and explicit configuration layering for step 30.2.

This module is intentionally independent of descriptor/model packages.  It
only reads a small YAML file, normalises documented model aliases, validates
the step-30.1 schema, and combines already-declared values.  Runtime entry
points will switch to it in step 30.8.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .contracts import MISSING, ConfigContractError, validate_operation_config, validate_run_config


class ConfigLoadError(ConfigContractError):
    """Raised for an unsafe, ambiguous, or unsupported YAML configuration."""


@dataclass(frozen=True)
class LoadedRunConfig:
    """A run config retaining the user declaration separate from effective values."""

    path: Path
    explicit_yaml: Dict[str, Any]
    effective: Dict[str, Any]
    explicit_cli: Dict[str, Any]


_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_MODEL_ALIASES = {
    "xgboost": "xgb",
    "random forest": "rf",
    "random_forest": "rf",
    "light gbm": "lightgbm",
    "lgbm": "lightgbm",
    "auto_gluon": "autogluon",
}


def _require_yaml_path(path: Path | str) -> Path:
    source = Path(path).resolve()
    if source.suffix.lower() not in _YAML_SUFFIXES:
        raise ConfigLoadError("运行配置仅接受 .yaml 或 .yml 文件；JSON 请使用一次性迁移器")
    if not source.is_file():
        raise FileNotFoundError(f"配置文件不存在：{source}")
    return source


def _node_key(node: Node, path: str) -> str:
    if not isinstance(node, ScalarNode) or node.tag != "tag:yaml.org,2002:str":
        raise ConfigLoadError(f"{path} 的 mapping 键必须是字符串")
    if not node.value.strip():
        raise ConfigLoadError(f"{path} 的 mapping 键不能为空")
    return node.value


def _check_node(node: Node, path: str = "$") -> None:
    """Reject duplicate mapping keys before PyYAML can silently overwrite one."""
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = _node_key(key_node, path)
            child_path = f"{path}.{key}" if path != "$" else key
            if key in seen:
                raise ConfigLoadError(f"YAML 重复键：{child_path}")
            seen.add(key)
            _check_node(value_node, child_path)
    elif isinstance(node, SequenceNode):
        for index, item in enumerate(node.value):
            _check_node(item, f"{path}[{index}]")


def _reject_alias_events(source: str) -> None:
    """Aliases are rejected so mappings cannot be recursive or shared mutably."""
    try:
        for event in yaml.parse(source, Loader=yaml.SafeLoader):
            if isinstance(event, AliasEvent):
                raise ConfigLoadError("YAML 不支持锚点别名；请在配置中显式写出值")
    except ConfigLoadError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML 语法错误：{exc}") from exc


def load_yaml_mapping(path: Path | str) -> Dict[str, Any]:
    """Load exactly one unambiguous YAML mapping with safe scalar keys only."""
    source_path = _require_yaml_path(path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigLoadError(f"配置必须为 UTF-8 文本：{source_path}") from exc

    _reject_alias_events(source)
    try:
        documents = list(yaml.compose_all(source, Loader=yaml.SafeLoader))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML 语法错误：{exc}") from exc
    if len(documents) != 1 or documents[0] is None:
        raise ConfigLoadError("配置必须恰好包含一个非空 YAML 文档")
    _check_node(documents[0])
    try:
        parsed = yaml.safe_load(source)
    except yaml.YAMLError as exc:  # defensive; compose above already parsed it
        raise ConfigLoadError(f"YAML 语法错误：{exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ConfigLoadError("配置根节点必须是 mapping")
    return copy.deepcopy(dict(parsed))


def _canonical_model_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigLoadError(f"{field} 必须是非空字符串")
    normalized = value.strip().lower()
    return _MODEL_ALIASES.get(normalized, normalized)


def normalise_model_aliases(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return canonical model names and fail if aliases collapse two values.

    The function only recognises a deliberately small, documented compatibility
    list.  It never treats arbitrary spelling variants as a model identifier.
    """
    result = copy.deepcopy(dict(raw))
    models = result.get("models")
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        canonical_models: list[Any] = []
        first_path: Dict[str, str] = {}
        for index, raw_name in enumerate(models):
            field = f"models[{index}]"
            canonical = _canonical_model_name(raw_name, field)
            if canonical in first_path:
                raise ConfigLoadError(
                    f"别名规范化冲突：{field} 与 {first_path[canonical]} 都解析为 {canonical!r}"
                )
            canonical_models.append(canonical)
            first_path[canonical] = field
        result["models"] = canonical_models

    raw_params = result.get("model_params")
    if isinstance(raw_params, Mapping):
        canonical_params: Dict[str, Any] = {}
        first_path = {}
        for raw_name, params in raw_params.items():
            field = f"model_params.{raw_name}"
            canonical = _canonical_model_name(raw_name, field)
            if canonical in canonical_params:
                raise ConfigLoadError(
                    f"别名规范化冲突：{field} 与 {first_path[canonical]} 都解析为 {canonical!r}"
                )
            canonical_params[canonical] = params
            first_path[canonical] = field
        result["model_params"] = canonical_params
    return result


def _merge_value(base: Any, override: Any) -> Any:
    if override is MISSING:
        return copy.deepcopy(base)
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged: Dict[str, Any] = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            merged[key] = _merge_value(merged.get(key, MISSING), value)
        return merged
    return copy.deepcopy(override)


def merge_config_layers(
    library_defaults: Mapping[str, Any] | None,
    explicit_yaml: Mapping[str, Any],
    explicit_cli: Mapping[str, Any] | object = MISSING,
) -> Dict[str, Any]:
    """Apply ``library default → YAML → explicitly requested CLI``.

    Passing ``MISSING`` (the default) means no CLI overlay.  Passing a mapping
    makes every key in that mapping explicit, even when its value is ``None``,
    ``False``, or ``0``.  This is what prevents argparse defaults from silently
    replacing a YAML declaration.
    """
    if library_defaults is not None and not isinstance(library_defaults, Mapping):
        raise ConfigLoadError("library_defaults 必须是 mapping 或 None")
    if not isinstance(explicit_yaml, Mapping):
        raise ConfigLoadError("explicit_yaml 必须是 mapping")
    if explicit_cli is not MISSING and not isinstance(explicit_cli, Mapping):
        raise ConfigLoadError("explicit_cli 必须是 mapping 或 MISSING")
    merged = _merge_value(dict(library_defaults or {}), explicit_yaml)
    return _merge_value(merged, explicit_cli)


def build_explicit_cli_overrides(
    values: Mapping[str, Any] | Any,
    explicit_fields: Mapping[str, str] | Sequence[str],
) -> Dict[str, Any]:
    """Build a nested overlay from arguments the CLI parser observed explicitly.

    ``explicit_fields`` may be a sequence of identical source/destination paths
    or a mapping of destination dotted paths to a source key.  The caller must
    obtain this list from argument-token presence, not from argparse defaults.
    """
    source_values = dict(values) if isinstance(values, Mapping) else vars(values)
    mapping = (
        dict(explicit_fields)
        if isinstance(explicit_fields, Mapping)
        else {str(path): str(path) for path in explicit_fields}
    )
    result: Dict[str, Any] = {}
    for destination, source in mapping.items():
        if source not in source_values:
            raise ConfigLoadError(f"显式 CLI 参数 {source!r} 未提供值")
        parts = str(destination).split(".")
        if not all(part for part in parts):
            raise ConfigLoadError(f"CLI 覆盖路径非法：{destination!r}")
        target = result
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                target[part] = {}
            elif not isinstance(existing, dict):
                raise ConfigLoadError(f"CLI 覆盖路径冲突：{destination!r}")
            target = target[part]
        target[parts[-1]] = copy.deepcopy(source_values[source])
    return result


def load_run_config(
    path: Path | str,
    *,
    library_defaults: Mapping[str, Any] | None = None,
    explicit_cli: Mapping[str, Any] | object = MISSING,
) -> LoadedRunConfig:
    """Load, normalise, and statically validate a standard runtime YAML file."""
    source_path = _require_yaml_path(path)
    explicit_yaml = normalise_model_aliases(load_yaml_mapping(source_path))
    # Validate user YAML before defaults can hide a missing required field.
    try:
        validated_yaml = validate_run_config(explicit_yaml)
    except ConfigContractError as exc:
        raise ConfigLoadError(str(exc)) from exc
    effective = merge_config_layers(library_defaults, validated_yaml, explicit_cli)
    effective = normalise_model_aliases(effective)
    try:
        validated_effective = validate_run_config(effective)
    except ConfigContractError as exc:
        raise ConfigLoadError(f"合并后的运行配置无效：{exc}") from exc
    if validated_effective["stage"] != "benchmark":
        from .output_layout import task_root

        root = task_root(source_path, validated_effective)
        validated_effective["outputs"] = dict(validated_effective.get("outputs") or {})
        validated_effective["outputs"]["root"] = str(root)
        validated_effective["artifacts"]["output_dir"] = str(root / "feature")
    return LoadedRunConfig(
        path=source_path,
        explicit_yaml=validated_yaml,
        effective=validated_effective,
        explicit_cli={} if explicit_cli is MISSING else copy.deepcopy(dict(explicit_cli)),
    )


def load_operation_config(path: Path | str) -> Dict[str, Any]:
    """Load a separate, model-free feature-operation YAML file."""
    source_path = _require_yaml_path(path)
    try:
        return validate_operation_config(load_yaml_mapping(source_path))
    except ConfigContractError as exc:
        raise ConfigLoadError(str(exc)) from exc
