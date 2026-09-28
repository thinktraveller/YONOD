"""Pure-data, versioned search spaces for the three supported tree models."""

from __future__ import annotations

import copy
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping


class SpaceContractError(ValueError):
    """A search distribution cannot be sampled safely or routed to a model."""


_SEARCHABLE = {
    "rf": frozenset({"n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "max_samples", "min_impurity_decrease", "ccp_alpha"}),
    "xgb": frozenset({"n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "max_delta_step"}),
    "lightgbm": frozenset({"n_estimators", "learning_rate", "num_leaves", "min_child_samples", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain", "max_depth"}),
}

_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "rf": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 100},
        "max_depth": {"type": "categorical", "choices": [None, 4, 8, 12, 20, 32]},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "float", "low": 0.3, "high": 1.0},
    },
    "xgb": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 100},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 20.0, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 100},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "num_leaves": {"type": "int", "low": 8, "high": 64},
        "min_child_samples": {"type": "int", "low": 5, "high": 50},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    },
}


def _number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SpaceContractError(f"{field} 必须为有限数值，不能是 bool")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise SpaceContractError(f"{field} 必须为有限数值") from exc


def _validate_distribution(raw: Any, field: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SpaceContractError(f"{field} 必须是分布 mapping")
    kind = raw.get("type")
    if kind == "categorical":
        unknown = set(raw) - {"type", "choices"}
        choices = raw.get("choices")
        if unknown or not isinstance(choices, list) or not choices:
            raise SpaceContractError(f"{field} 仅接受 type 与非空 choices")
        for index, choice in enumerate(choices):
            if choice is not None and not isinstance(choice, (str, bool, int, float)):
                raise SpaceContractError(f"{field}.choices[{index}] 必须为安全标量")
            if isinstance(choice, float) and not math.isfinite(choice):
                raise SpaceContractError(f"{field}.choices[{index}] 必须有限")
        if len({(type(choice).__name__, repr(choice)) for choice in choices}) != len(choices):
            raise SpaceContractError(f"{field}.choices 不能重复")
        return {"type": "categorical", "choices": copy.deepcopy(choices)}
    if kind not in {"int", "float"}:
        raise SpaceContractError(f"{field}.type 仅支持 int、float、categorical")
    unknown = set(raw) - {"type", "low", "high", "step", "log"}
    if unknown or "low" not in raw or "high" not in raw:
        raise SpaceContractError(f"{field} 需要 low/high，且不能包含未知键")
    low, high = _number(raw["low"], f"{field}.low"), _number(raw["high"], f"{field}.high")
    if low > high:
        raise SpaceContractError(f"{field}.low 不得大于 high")
    log = raw.get("log", False)
    if not isinstance(log, bool):
        raise SpaceContractError(f"{field}.log 必须为 bool")
    if log and low <= 0:
        raise SpaceContractError(f"{field} 对数分布的 low 必须大于 0")
    result: Dict[str, Any] = {"type": kind}
    if kind == "int":
        if any(isinstance(raw[name], bool) or not isinstance(raw[name], int) for name in ("low", "high")):
            raise SpaceContractError(f"{field} 的整数边界必须为 int")
        step = raw.get("step", 1)
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise SpaceContractError(f"{field}.step 必须为正整数")
        if log and step != 1:
            raise SpaceContractError(f"{field} 的 int log 只能使用 step=1")
        if (raw["high"] - raw["low"]) % step:
            raise SpaceContractError(f"{field} 的整数范围必须可被 step 整除")
        result.update(low=raw["low"], high=raw["high"], step=step)
    else:
        if log and "step" in raw:
            raise SpaceContractError(f"{field} 的 float log 不能同时声明 step")
        if "step" in raw:
            step = _number(raw["step"], f"{field}.step")
            if step <= 0 or (high - low) % step:
                raise SpaceContractError(f"{field} 的浮点 step 必须为正且整除范围")
            result["step"] = float(step)
        result.update(low=float(low), high=float(high))
    result["log"] = log
    return result


def expand_search_space(model: str, declaration: Any, fixed_estimator: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Expand a preset or an explicit space and reject known fixed conflicts."""
    if model not in _SEARCHABLE:
        raise SpaceContractError(f"{model} 不支持 HPO")
    if not isinstance(declaration, Mapping):
        raise SpaceContractError(f"hpo.search_spaces.{model} 必须是 mapping")
    unknown = set(declaration) - {"preset", "parameters"}
    if unknown:
        raise SpaceContractError(f"hpo.search_spaces.{model} 包含未知键：{', '.join(sorted(unknown))}")
    preset = declaration.get("preset")
    if preset is not None and preset != "compact-v1":
        raise SpaceContractError(f"hpo.search_spaces.{model}.preset 仅支持 compact-v1")
    parameters = declaration.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise SpaceContractError(f"hpo.search_spaces.{model}.parameters 必须是 mapping")
    if preset is None and not parameters:
        raise SpaceContractError(f"hpo.search_spaces.{model} 需要 preset 或非空 parameters")
    distributions = copy.deepcopy(_PRESETS[model]) if preset else {}
    distributions.update(copy.deepcopy(dict(parameters)))
    unknown_params = set(distributions) - _SEARCHABLE[model]
    if unknown_params:
        raise SpaceContractError(f"{model} 搜索参数不受支持或属于资源/种子字段：{', '.join(sorted(unknown_params))}")
    expanded = {
        name: _validate_distribution(distribution, f"hpo.search_spaces.{model}.parameters.{name}")
        for name, distribution in sorted(distributions.items())
    }
    fixed = dict(fixed_estimator or {})
    if model == "xgb" and fixed.get("booster", "gbtree") != "gbtree" and preset:
        raise SpaceContractError("xgb compact-v1 仅适用于 booster=gbtree")
    if model == "lightgbm":
        if "subsample" in expanded and fixed.get("subsample_freq", 0) <= 0:
            raise SpaceContractError("lightgbm 搜索 subsample 时必须固定正数 subsample_freq")
        max_depth = fixed.get("max_depth")
        leaves = expanded.get("num_leaves")
        if leaves and leaves["type"] == "categorical":
            if any(isinstance(choice, bool) or not isinstance(choice, int) or choice < 2 for choice in leaves["choices"]):
                raise SpaceContractError("lightgbm num_leaves 类别必须为 >=2 的整数")
            maximum_leaves = max(leaves["choices"])
        else:
            maximum_leaves = leaves.get("high") if leaves else None
        if maximum_leaves is not None and isinstance(max_depth, int) and not isinstance(max_depth, bool) and max_depth > 0:
            if maximum_leaves > 2 ** max_depth:
                raise SpaceContractError("lightgbm compact-v1 的 num_leaves 与固定 max_depth 不兼容；请覆盖空间")
    return {
        "version": preset or "explicit-v1",
        "parameters": expanded,
        "overridden_fixed_parameters": sorted(set(expanded) & set(fixed)),
    }
