"""Declared numeric inputs with fold-local, serialisable preprocessing.

The transformer never chooses rows or reads a dataset. Callers supply the
current training subset to ``fit`` and use the returned state for validation
and future prediction. A numeric input keeps its column even when constant.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STATE_SCHEMA = "numeric_conditions/v1"
MISSING_STRATEGIES = frozenset({"error", "median", "mean", "constant"})
SCALING_STRATEGIES = frozenset({"none", "standard", "minmax"})


class NumericConditionsError(ValueError):
    """A declared numeric input cannot be parsed or transformed safely."""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise NumericConditionsError(f"{field} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericConditionsError(f"{field} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise NumericConditionsError(f"{field} 必须是有限数值")
    return number


def _missing_rule(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        value = {"strategy": "error"}
    if isinstance(value, str):
        value = {"strategy": value}
    if not isinstance(value, Mapping) or set(value) - {"strategy", "constant"}:
        raise NumericConditionsError(f"{field} 仅允许 strategy 和 constant")
    strategy = value.get("strategy", "error")
    if strategy not in MISSING_STRATEGIES:
        raise NumericConditionsError(f"{field}.strategy 不受支持：{strategy!r}")
    if strategy == "constant":
        if "constant" not in value:
            raise NumericConditionsError(f"{field}.constant 必须显式声明")
        return {"strategy": strategy, "constant": _finite_number(value["constant"], f"{field}.constant")}
    if "constant" in value:
        raise NumericConditionsError(f"{field}.constant 仅能用于 constant 策略")
    return {"strategy": strategy}


def normalise_numeric_contract(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and expand numeric declarations without inspecting data rows."""
    roles = dataset.get("column_roles", {})
    columns = roles.get("conditions", []) or []
    if not isinstance(columns, list) or any(not isinstance(col, str) or not col for col in columns):
        raise NumericConditionsError("dataset.column_roles.conditions 必须是列名列表")
    if len(set(columns)) != len(columns):
        raise NumericConditionsError("dataset.column_roles.conditions 不能重复")
    raw = dataset.get("numeric_conditions") or {}
    if not isinstance(raw, Mapping) or set(raw) - {"defaults", "columns"}:
        raise NumericConditionsError("dataset.numeric_conditions 仅允许 defaults 和 columns")
    defaults = raw.get("defaults") or {}
    overrides = raw.get("columns") or {}
    if not isinstance(defaults, Mapping) or set(defaults) - {"missing", "scaling"}:
        raise NumericConditionsError("dataset.numeric_conditions.defaults 仅允许 missing 和 scaling")
    if not isinstance(overrides, Mapping):
        raise NumericConditionsError("dataset.numeric_conditions.columns 必须是按源列名索引的 mapping")
    unexpected = sorted(str(key) for key in overrides if key not in columns)
    if unexpected:
        raise NumericConditionsError("dataset.numeric_conditions.columns 未在 conditions 声明：" + ", ".join(unexpected))
    default_missing = _missing_rule(defaults.get("missing"), "dataset.numeric_conditions.defaults.missing")
    default_scaling = defaults.get("scaling", "none")
    if default_scaling not in SCALING_STRATEGIES:
        raise NumericConditionsError("dataset.numeric_conditions.defaults.scaling 不受支持")
    entries: list[dict[str, Any]] = []
    output_names: set[str] = set()
    for column in columns:
        field = f"dataset.numeric_conditions.columns.{column}"
        item = overrides.get(column) or {}
        if not isinstance(item, Mapping) or set(item) - {"name", "missing_values", "missing", "scaling", "unit"}:
            raise NumericConditionsError(f"{field} 包含未知字段")
        name = item.get("name", column)
        if not isinstance(name, str) or not name.strip() or name in output_names:
            raise NumericConditionsError(f"{field}.name 必须是唯一的非空名称")
        output_names.add(name)
        markers = item.get("missing_values", [])
        if not isinstance(markers, list) or any(marker is not None and not isinstance(marker, str) for marker in markers):
            raise NumericConditionsError(f"{field}.missing_values 必须是字符串/null 列表")
        if len(set(markers)) != len(markers):
            raise NumericConditionsError(f"{field}.missing_values 不能重复")
        scaling = item.get("scaling", default_scaling)
        if scaling not in SCALING_STRATEGIES:
            raise NumericConditionsError(f"{field}.scaling 不受支持：{scaling!r}")
        unit = item.get("unit") or {}
        if not isinstance(unit, Mapping) or set(unit) - {"source", "target", "scale", "offset"}:
            raise NumericConditionsError(f"{field}.unit 字段无效")
        if unit:
            source, target = unit.get("source"), unit.get("target")
            if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                raise NumericConditionsError(f"{field}.unit 必须声明 source/target")
            scale = _finite_number(unit.get("scale", 1), f"{field}.unit.scale")
            offset = _finite_number(unit.get("offset", 0), f"{field}.unit.offset")
            if scale == 0:
                raise NumericConditionsError(f"{field}.unit.scale 不能为零")
            if source == target and (scale != 1 or offset != 0):
                raise NumericConditionsError(f"{field}.unit 同单位不能声明非恒等换算")
            resolved_unit = {"source": source, "target": target, "scale": scale, "offset": offset}
        else:
            resolved_unit = {"source": "dimensionless", "target": "dimensionless", "scale": 1.0, "offset": 0.0}
        entries.append({
            "source": column, "name": name, "missing_values": markers,
            "missing": _missing_rule(item.get("missing", default_missing), f"{field}.missing"),
            "scaling": scaling, "unit": resolved_unit,
        })
    return {"schema_version": STATE_SCHEMA, "columns": entries}


def parse_numeric_frame(
    frame: pd.DataFrame, sample_ids: Sequence[Any], contract: Mapping[str, Any], *, phase: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Parse source values and apply declared unit conversion, preserving NaN only for declared missing values."""
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(frame):
        raise NumericConditionsError(f"{phase}: sample_id 数量与数值行数不一致")
    names = [entry["source"] for entry in contract["columns"]]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise NumericConditionsError(f"{phase}: 数值输入缺少列：{', '.join(missing)}")
    matrix = np.empty((len(frame), len(names)), dtype=np.float64)
    diagnostics = []
    for col_index, entry in enumerate(contract["columns"]):
        count = 0
        markers = entry["missing_values"]
        for row_index, raw in enumerate(frame[entry["source"]].tolist()):
            is_null = raw is None or raw is pd.NA or (isinstance(raw, float) and math.isnan(raw))
            if (is_null and None in markers) or (not is_null and str(raw) in markers):
                matrix[row_index, col_index] = np.nan
                count += 1
                continue
            label = f"{phase}: 列 {entry['source']!r}, sample_id {ids[row_index]!r}"
            value = _finite_number(raw, label)
            converted = value * entry["unit"]["scale"] + entry["unit"]["offset"]
            if not math.isfinite(converted):
                raise NumericConditionsError(f"{label} 单位换算结果非有限值")
            matrix[row_index, col_index] = converted
        diagnostics.append({"source": entry["source"], "name": entry["name"], "missing_rows": count})
    return matrix, diagnostics


def numeric_input_identity(frame: pd.DataFrame, sample_ids: Sequence[Any], contract: Mapping[str, Any]) -> str:
    """Content identity of declared numeric values and rules, independent of file location."""
    values, _ = parse_numeric_frame(frame, sample_ids, contract, phase="identity")
    records = [[None if np.isnan(value) else float(value) for value in row] for row in values]
    raw = [[_raw_token(frame.iloc[index][entry["source"]]) for entry in contract["columns"]]
           for index in range(len(frame))]
    return _digest({"contract": contract, "sample_ids": [str(value) for value in sample_ids],
                    "raw_values": raw, "parsed_values": records})


def _raw_token(value: Any) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _block_payload(frame: pd.DataFrame, sample_ids: Sequence[Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    ids = [str(value) for value in sample_ids]
    identity = numeric_input_identity(frame, ids, contract)
    raw = [[_raw_token(frame.iloc[index][entry["source"]]) for entry in contract["columns"]]
           for index in range(len(frame))]
    return {"schema_version": STATE_SCHEMA, "identity": identity, "contract": contract,
            "sample_ids": ids, "raw_values": raw}


def load_numeric_block(
    path: Path | str, *, expected_ids: Sequence[Any] | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    expected_identity: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and verify an immutable raw block, including exact row mapping."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NumericConditionsError(f"数值块无法读取：{source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA:
        raise NumericConditionsError("数值块版本不受支持")
    contract = payload.get("contract")
    ids = payload.get("sample_ids")
    raw = payload.get("raw_values")
    if not isinstance(contract, dict) or not isinstance(ids, list) or not isinstance(raw, list):
        raise NumericConditionsError("数值块缺少契约、sample_id 或原始矩阵")
    if len(ids) != len(raw) or len(set(ids)) != len(ids):
        raise NumericConditionsError("数值块 sample_id/行数无效")
    columns = [entry["source"] for entry in contract["columns"]]
    if any(not isinstance(row, list) or len(row) != len(columns) for row in raw):
        raise NumericConditionsError("数值块原始矩阵列数不一致")
    frame = pd.DataFrame(raw, columns=columns)
    actual = numeric_input_identity(frame, ids, contract)
    if actual != payload.get("identity") or source.stem != actual.removeprefix("sha256:"):
        raise NumericConditionsError("数值块内容或文件名身份不匹配")
    if expected_ids is not None and ids != [str(value) for value in expected_ids]:
        raise NumericConditionsError("数值块 sample_id 与特征包/训练表顺序不一致")
    if expected_contract is not None and contract != expected_contract:
        raise NumericConditionsError("数值块预处理契约与当前 YAML 不一致")
    if expected_identity is not None and actual != expected_identity:
        raise NumericConditionsError("数值块内容与当前训练数据不一致")
    return frame, payload


def publish_numeric_block(
    root: Path | str, frame: pd.DataFrame, sample_ids: Sequence[Any], contract: Mapping[str, Any]
) -> Path:
    """Atomically publish a content-addressed, unfitted numeric input block."""
    payload = _block_payload(frame, sample_ids, contract)
    directory = Path(root) / "numeric_blocks"
    destination = directory / (payload["identity"].removeprefix("sha256:") + ".json")
    if destination.exists():
        load_numeric_block(destination, expected_ids=sample_ids, expected_contract=contract,
                           expected_identity=payload["identity"])
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    load_numeric_block(destination, expected_ids=sample_ids, expected_contract=contract,
                       expected_identity=payload["identity"])
    return destination


class NumericConditionsTransformer:
    """Fold-scoped numeric transformer with exact, JSON-safe fitted state."""

    def __init__(self, contract: Mapping[str, Any]):
        if contract.get("schema_version") != STATE_SCHEMA:
            raise NumericConditionsError("数值契约版本不受支持")
        self.contract = json.loads(json.dumps(contract, ensure_ascii=False, allow_nan=False))
        self.state: dict[str, Any] | None = None

    def fit(self, frame: pd.DataFrame, sample_ids: Sequence[Any], *, phase: str = "train") -> "NumericConditionsTransformer":
        ids = [str(value) for value in sample_ids]
        if not ids:
            raise NumericConditionsError(f"{phase}: 训练子集不能为空")
        if len(set(ids)) != len(ids):
            raise NumericConditionsError(f"{phase}: 训练 sample_id 不能重复")
        values, parse_diagnostics = parse_numeric_frame(frame, ids, self.contract, phase=phase)
        states = []
        for index, entry in enumerate(self.contract["columns"]):
            column = values[:, index]
            observed = column[~np.isnan(column)]
            strategy = entry["missing"]["strategy"]
            if strategy == "error" and len(observed) != len(column):
                raise NumericConditionsError(f"{phase}: 列 {entry['source']!r} 含缺失值，sample_id {ids[int(np.flatnonzero(np.isnan(column))[0])]!r}")
            if strategy in {"mean", "median"} and len(observed) == 0:
                raise NumericConditionsError(f"{phase}: 列 {entry['source']!r} 训练子集整列缺失")
            fill = None
            if strategy == "mean":
                fill = float(np.mean(observed))
            elif strategy == "median":
                fill = float(np.median(observed))
            elif strategy == "constant":
                fill = float(entry["missing"]["constant"])
            filled = np.where(np.isnan(column), fill, column) if fill is not None else column
            if not np.isfinite(filled).all():
                raise NumericConditionsError(f"{phase}: 列 {entry['source']!r} 填补后非有限值")
            scaling = entry["scaling"]
            location = float(np.mean(filled)) if scaling == "standard" else float(np.min(filled)) if scaling == "minmax" else 0.0
            raw_scale = float(np.std(filled, ddof=0)) if scaling == "standard" else float(np.max(filled) - np.min(filled)) if scaling == "minmax" else 1.0
            states.append({
                "source": entry["source"], "name": entry["name"], "fill_value": fill,
                "location": location, "scale": raw_scale if raw_scale != 0 else 1.0,
                "constant_train": bool(np.min(filled) == np.max(filled)),
                "train_missing_rows": parse_diagnostics[index]["missing_rows"],
            })
        self.state = {
            "schema_version": STATE_SCHEMA, "contract": self.contract,
            "fit_sample_ids_sha256": _digest(ids), "fit_rows": len(ids), "columns": states,
        }
        return self

    def transform(self, frame: pd.DataFrame, sample_ids: Sequence[Any], *, phase: str = "valid") -> np.ndarray:
        if self.state is None:
            raise NumericConditionsError("数值处理器尚未拟合")
        ids = [str(value) for value in sample_ids]
        values, _ = parse_numeric_frame(frame, ids, self.contract, phase=phase)
        for index, (entry, state) in enumerate(zip(self.contract["columns"], self.state["columns"])):
            column = values[:, index]
            if np.isnan(column).any():
                if entry["missing"]["strategy"] == "error":
                    raise NumericConditionsError(f"{phase}: 列 {entry['source']!r} 含缺失值，sample_id {ids[int(np.flatnonzero(np.isnan(column))[0])]!r}")
                column = np.where(np.isnan(column), state["fill_value"], column)
            values[:, index] = (column - state["location"]) / state["scale"]
        if not np.isfinite(values).all():
            raise NumericConditionsError(f"{phase}: 数值变换结果非有限值")
        return values

    def fit_transform(self, frame: pd.DataFrame, sample_ids: Sequence[Any], *, phase: str = "train") -> np.ndarray:
        return self.fit(frame, sample_ids, phase=phase).transform(frame, sample_ids, phase=phase)

    def state_dict(self) -> dict[str, Any]:
        if self.state is None:
            raise NumericConditionsError("数值处理器尚未拟合")
        return json.loads(json.dumps(self.state, ensure_ascii=False, allow_nan=False))

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "NumericConditionsTransformer":
        if state.get("schema_version") != STATE_SCHEMA or not isinstance(state.get("columns"), list):
            raise NumericConditionsError("数值拟合状态版本或列结构无效")
        result = cls(state["contract"])
        if len(state["columns"]) != len(result.contract["columns"]):
            raise NumericConditionsError("数值拟合状态列数与契约不一致")
        result.state = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
        return result
