"""Schema-2 HPO declarations; validation never imports Optuna or estimators."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from .spaces import SpaceContractError, expand_search_space


class HPOContractError(ValueError):
    """An HPO declaration is incomplete or incompatible with the run."""


SUPPORTED_MODELS = frozenset({"rf", "xgb", "lightgbm"})
_OBJECTIVE_DIRECTIONS = {
    "rmse": "minimize", "mae": "minimize", "r2": "maximize", "kendall_tau": "maximize",
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HPOContractError(f"{field} 必须是 mapping")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise HPOContractError(f"{field} 包含未知字段：{', '.join(unknown)}")


def _integer(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        suffix = f"且 >= {minimum}" if minimum is not None else ""
        raise HPOContractError(f"{field} 必须是整数{suffix}")
    return value


def _budget(value: Any, field: str) -> Dict[str, Any]:
    raw = _mapping(value, field)
    _keys(raw, {"max_trials", "study_timeout_s", "task_timeout_s"}, field)
    if "max_trials" not in raw:
        raise HPOContractError(f"{field}.max_trials 必须显式声明")
    result: Dict[str, Any] = {"max_trials": _integer(raw["max_trials"], f"{field}.max_trials", 1)}
    for name in ("study_timeout_s", "task_timeout_s"):
        if name in raw:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise HPOContractError(f"{field}.{name} 必须为有限正数")
            result[name] = float(value)
    return result


def validate_hpo_declaration(
    value: Any, *, models: Sequence[str], stage: str, evaluation: Any,
    model_params: Any, benchmark: Any = None,
) -> Dict[str, Any]:
    """Validate and expand a complete HPO declaration as detached pure data.

    The disabled form is deliberately cheap: no search engine or model library
    is imported, and omitted ``hpo`` is equivalent to disabled.  Enabled runs
    must specify all behaviour-changing fields rather than taking project
    defaults from a sample YAML.
    """
    if value is None:
        return {"enabled": False}
    raw = _mapping(value, "hpo")
    _keys(raw, {"enabled", "models", "sampler", "seed", "objective", "inner_cv", "budget", "search_spaces", "final_model"}, "hpo")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise HPOContractError("hpo.enabled 必须为 bool")
    if not enabled:
        if set(raw) - {"enabled"}:
            raise HPOContractError("hpo.enabled=false 时只允许 enabled 字段；启用后再填写搜索参数")
        return {"enabled": False}
    if stage not in {"features", "train", "all", "benchmark"}:
        raise HPOContractError(f"stage={stage} 不支持 HPO")
    declared = raw.get("models")
    if not isinstance(declared, list) or not declared or any(not isinstance(name, str) for name in declared):
        raise HPOContractError("hpo.models 必须是非空模型列表")
    if len(set(declared)) != len(declared):
        raise HPOContractError("hpo.models 不能重复")
    unsupported = set(declared) - SUPPORTED_MODELS
    if unsupported:
        raise HPOContractError(f"HPO 首版只支持 rf/xgb/lightgbm：{', '.join(sorted(unsupported))}")
    if not set(declared).issubset(set(models)):
        raise HPOContractError("hpo.models 必须是顶层 models 的子集")
    if raw.get("sampler") != "tpe":
        raise HPOContractError("hpo.sampler 首版仅支持 tpe")
    seed = _integer(raw.get("seed"), "hpo.seed")
    objective = _mapping(raw.get("objective"), "hpo.objective")
    _keys(objective, {"metric", "direction"}, "hpo.objective")
    metric = objective.get("metric")
    if metric not in _OBJECTIVE_DIRECTIONS:
        raise HPOContractError("hpo.objective.metric 仅支持 rmse/mae/r2/kendall_tau")
    if objective.get("direction") != _OBJECTIVE_DIRECTIONS[metric]:
        raise HPOContractError(f"hpo.objective.direction 与 {metric} 的目标方向冲突")
    inner = _mapping(raw.get("inner_cv"), "hpo.inner_cv")
    _keys(inner, {"n_splits", "seed", "shuffle"}, "hpo.inner_cv")
    inner_splits = _integer(inner.get("n_splits"), "hpo.inner_cv.n_splits", 2)
    inner_seed = _integer(inner.get("seed"), "hpo.inner_cv.seed")
    shuffle = inner.get("shuffle")
    if not isinstance(shuffle, bool):
        raise HPOContractError("hpo.inner_cv.shuffle 必须为 bool")
    budget = _budget(raw.get("budget"), "hpo.budget")
    spaces = _mapping(raw.get("search_spaces"), "hpo.search_spaces")
    if set(spaces) != set(declared):
        raise HPOContractError("hpo.search_spaces 必须恰好包含 hpo.models 中的模型")
    fixed_params = _mapping(model_params or {}, "model_params")
    expanded = {}
    for model in declared:
        sections = _mapping(fixed_params.get(model, {}), f"model_params.{model}")
        estimator = _mapping(sections.get("estimator", {}), f"model_params.{model}.estimator")
        runtime = _mapping(sections.get("runtime", {}), f"model_params.{model}.runtime")
        if runtime.get("early_stopping"):
            raise HPOContractError(f"model_params.{model}.runtime.early_stopping 与首版 HPO 不兼容")
        if stage == "benchmark" and sections.get("fit"):
            raise HPOContractError(f"strict HPO 暂不支持 model_params.{model}.fit")
        try:
            expanded[model] = expand_search_space(model, spaces[model], estimator)
        except SpaceContractError as exc:
            raise HPOContractError(str(exc)) from exc
    if stage == "features":
        # Feature-only jobs validate the declaration, then leave all study
        # creation to a later train/all invocation.
        protocol = None
    else:
        evaluation = _mapping(evaluation, "evaluation")
        protocol = evaluation.get("protocol")
    if stage == "benchmark":
        if protocol != "manifest_outer_cv":
            raise HPOContractError("strict HPO 只支持 manifest_outer_cv")
        benchmark = _mapping(benchmark, "benchmark")
        reproduction = _mapping(benchmark.get("reproduction_protocol", {}), "benchmark.reproduction_protocol")
        if reproduction.get("name") in {"vjethbkm_rf_5x5", "paper_exact_5x5"} or reproduction.get("evaluation_protocol") == "paper_exact_5x5":
            raise HPOContractError("固定论文复现协议禁止 HPO；请使用独立研究任务")
    elif stage != "features" and protocol != "outer_kfold":
        raise HPOContractError(
            "普通 HPO 只支持 outer_kfold；manifest_outer_cv 请使用 stage: benchmark 的真实 manifest 路径"
        )
    final = _mapping(raw.get("final_model", {"enabled": False}), "hpo.final_model")
    _keys(final, {"enabled", "development_population_id", "budget"}, "hpo.final_model")
    final_enabled = final.get("enabled", False)
    if not isinstance(final_enabled, bool):
        raise HPOContractError("hpo.final_model.enabled 必须为 bool")
    if final_enabled:
        population_id = final.get("development_population_id")
        if not isinstance(population_id, str) or not population_id.strip():
            raise HPOContractError("hpo.final_model.development_population_id 必须是非空字符串")
        final_result = {
            "enabled": True,
            "development_population_id": population_id.strip(),
            "budget": _budget(final.get("budget"), "hpo.final_model.budget"),
        }
    else:
        if set(final) - {"enabled"}:
            raise HPOContractError("hpo.final_model.enabled=false 时不得声明最终模型预算或人口")
        final_result = {"enabled": False}
    return {
        "enabled": True,
        "models": list(declared),
        "sampler": "tpe",
        "seed": seed,
        "objective": {"metric": metric, "direction": objective["direction"]},
        "inner_cv": {"n_splits": inner_splits, "seed": inner_seed, "shuffle": shuffle},
        "budget": budget,
        "search_spaces": expanded,
        "final_model": final_result,
    }


def require_search_engine() -> None:
    """Fail only for enabled HPO; fixed runs never need Optuna installed."""
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise HPOContractError(
            'HPO 需要 Optuna 4.5.0；请先在 yonod 环境执行 python -m pip install "optuna==4.5.0"'
        ) from exc
