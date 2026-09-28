"""One serial Optuna study over inner folds of one outer-training population."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from yonod.model_factory import construct_estimator, materialise_fit_parameters, resolve_model_config

from .folds import InnerFold, make_fold_matrices


class SearchError(RuntimeError):
    """A study cannot produce a valid winner or its bookkeeping is invalid."""


class CandidateFitError(RuntimeError):
    """One proposed estimator failed for a candidate-specific reason."""


_METRIC_DIRECTIONS = {
    "rmse": "minimize", "mae": "minimize", "r2": "maximize", "kendall_tau": "maximize",
}


def _audit_value(value: Any) -> Any:
    """Make library default snapshots strict JSON without changing fitted models."""
    if isinstance(value, Mapping):
        return {str(key): _audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class SearchPopulation:
    """Already sliced outer-train values; never include outer-validation data."""

    sample_ids: Sequence[str]
    y: np.ndarray
    static_matrix: np.ndarray | None = None
    categorical_frame: pd.DataFrame | None = None
    ohe_factory: Callable[[], Any] | None = None
    numeric_frame: pd.DataFrame | None = None
    numeric_contract: Mapping[str, Any] | None = None
    fit_frame: pd.DataFrame | None = None

    def check(self) -> None:
        n = len(self.sample_ids)
        y = np.asarray(self.y, dtype=float)
        if y.ndim != 1 or len(y) != n or not np.isfinite(y).all():
            raise SearchError("outer-train 标签必须为与 sample_id 等长的有限一维数组")
        if len(set(str(item) for item in self.sample_ids)) != n:
            raise SearchError("outer-train sample_id 必须唯一")
        for name in ("static_matrix", "categorical_frame", "numeric_frame", "fit_frame"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise SearchError(f"outer-train {name} 与 sample_id 行数不一致")


def _score(metric: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) != len(prediction) or not np.isfinite(prediction).all():
        raise CandidateFitError("预测行数不一致或含 NaN/inf")
    if metric == "rmse":
        value = float(np.sqrt(np.mean(np.square(truth - prediction))))
    elif metric == "mae":
        value = float(np.mean(np.abs(truth - prediction)))
    elif metric == "r2":
        if len(truth) < 2 or np.var(truth) == 0:
            raise CandidateFitError("该 inner-fold 的 R² 不可定义：验证标签不足两行或为常量")
        from sklearn.metrics import r2_score
        value = float(r2_score(truth, prediction))
    elif metric == "kendall_tau":
        if len(truth) < 2 or np.var(truth) == 0 or np.var(prediction) == 0:
            raise CandidateFitError("该 inner-fold 的 Kendall τ 不可定义：样本不足或存在常量向量")
        from scipy.stats import kendalltau
        value = float(kendalltau(truth, prediction).statistic)
    else:
        raise SearchError(f"未知优化指标：{metric}")
    if not math.isfinite(value):
        raise CandidateFitError(f"该 inner-fold 的 {metric} 为非有限值")
    return value


def _suggest(trial: Any, space: Mapping[str, Any]) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    for name, distribution in sorted(space["parameters"].items()):
        kind = distribution["type"]
        if kind == "categorical":
            suggestions[name] = trial.suggest_categorical(name, distribution["choices"])
        elif kind == "int":
            suggestions[name] = trial.suggest_int(
                name, distribution["low"], distribution["high"],
                step=distribution.get("step", 1), log=distribution.get("log", False),
            )
        elif kind == "float":
            suggestions[name] = trial.suggest_float(
                name, distribution["low"], distribution["high"],
                **({"step": distribution["step"]} if "step" in distribution else {}),
                log=distribution.get("log", False),
            )
        else:
            raise SearchError(f"未验证的搜索分布：{kind}")
    return suggestions


def run_inner_search(
    *, study: Any, model: str, base_model_config: Mapping[str, Any],
    space: Mapping[str, Any], population: SearchPopulation,
    folds: Sequence[InnerFold], metric: str, budget: Mapping[str, Any],
    elapsed_study_s: float = 0.0, remaining_task_s: float | None = None,
) -> dict[str, Any]:
    """Optimize at most the remaining total attempts, returning a winner audit.

    The persistent study itself is supplied by storage.py.  This function
    never assumes a fresh sampler after restart and never reuses an estimator
    between trials or folds.  Both search and fold execution are serial.
    """
    import optuna  # only reached for an enabled, validated HPO run

    population.check()
    if model not in {"rf", "xgb", "lightgbm"}:
        raise SearchError("首版 HPO 仅支持 rf/xgb/lightgbm")
    if metric not in _METRIC_DIRECTIONS or study.direction.name.lower() != _METRIC_DIRECTIONS[metric]:
        raise SearchError("study direction 与目标指标不一致")
    if not folds or any(not len(fold.train_index) or not len(fold.valid_index) for fold in folds):
        raise SearchError("内层 manifest 为空或含空折")
    max_trials = budget["max_trials"]
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise SearchError("预算 max_trials 必须为正整数")
    existing = len(study.trials)
    if existing > max_trials:
        raise SearchError("现有 trial 数超过配置预算；拒绝跨身份继续")
    remaining_trials = max_trials - existing
    study_limit = budget.get("study_timeout_s")
    remaining_study_s = None if study_limit is None else max(0.0, float(study_limit) - elapsed_study_s)
    if remaining_task_s is not None and remaining_task_s < 0:
        raise SearchError("任务剩余预算不能为负")
    allowed_s = min(
        [value for value in (remaining_study_s, remaining_task_s) if value is not None],
        default=None,
    )
    stop_reason = "max_trials" if remaining_trials == 0 else None
    if allowed_s is not None and allowed_s <= 0:
        stop_reason = "task_timeout" if remaining_task_s is not None and remaining_task_s <= 0 else "study_timeout"
        remaining_trials = 0
    started = time.perf_counter()
    fixed = copy.deepcopy(dict(base_model_config))
    fixed_estimator = copy.deepcopy(dict(fixed.get("estimator", {})))
    labels = np.asarray(population.y, dtype=float)

    def objective(trial: Any) -> float:
        suggestion = _suggest(trial, space)
        model_config = copy.deepcopy(fixed)
        model_config["estimator"] = {**fixed_estimator, **suggestion}
        trial.set_user_attr("suggested_parameters", suggestion)
        trial.set_user_attr("fixed_estimator_parameters", fixed_estimator)
        trial.set_user_attr("constructor_parameters", model_config["estimator"])
        fold_audits = []
        fold_scores = []
        for fold in folds:
            matrices = make_fold_matrices(
                sample_ids=population.sample_ids, train_index=fold.train_index,
                valid_index=fold.valid_index, static_matrix=population.static_matrix,
                categorical_frame=population.categorical_frame, ohe_factory=population.ohe_factory,
                numeric_frame=population.numeric_frame, numeric_contract=population.numeric_contract,
                phase=f"trial={trial.number}/inner={fold.number}",
            )
            resolved = resolve_model_config(model, model_config)
            estimator, construction = construct_estimator(
                resolved, n_features=matrices.X_train.shape[1], n_train=len(fold.train_index)
            )
            effective = _audit_value(construction["effective_estimator_params"])
            if fold_audits and effective != fold_audits[0]["effective_estimator"]:
                raise SearchError("同一 trial 的不同 inner-fold 有不同估计器有效参数")
            trial.set_user_attr("effective_estimator_parameters", effective)
            trial.set_user_attr("parameter_sources", {
                name: ("hpo_trial" if name in suggestion else "yaml" if name in fixed_estimator
                       else "runtime" if name == "device" and "device_policy" in fixed.get("runtime", {})
                       else "library_default")
                for name in effective
            })
            fit_params = materialise_fit_parameters(
                resolved, population.fit_frame if population.fit_frame is not None else pd.DataFrame(index=range(len(labels))),
                np.asarray(fold.train_index, dtype=int),
            )
            try:
                estimator.fit(matrices.X_train, labels[fold.train_index], **fit_params)
                prediction = np.asarray(estimator.predict(matrices.X_valid), dtype=float)
            except Exception as exc:
                # Library training exceptions are candidate-specific; an
                # unrelated programming, I/O or assertion failure must stop
                # the study instead of being hidden among failed trials.
                candidate_error = isinstance(exc, (ValueError, FloatingPointError)) or (
                    type(exc).__module__.startswith(("xgboost", "lightgbm"))
                    and type(exc).__name__ in {"XGBoostError", "LightGBMError"}
                )
                if not candidate_error:
                    raise
                trial.set_user_attr("folds", fold_audits)
                trial.set_user_attr("failure", {"fold": fold.number, "type": type(exc).__name__, "reason": str(exc)})
                raise CandidateFitError(f"inner-fold {fold.number} 训练/预测失败：{exc}") from exc
            try:
                score = _score(metric, labels[fold.valid_index], prediction)
            except CandidateFitError as exc:
                trial.set_user_attr("folds", fold_audits)
                trial.set_user_attr("failure", {"fold": fold.number, "type": type(exc).__name__, "reason": str(exc)})
                raise
            fold_scores.append(score)
            fold_audits.append({
                **fold.audit(), "metric": metric, "score": score,
                "transform": matrices.state, "effective_estimator": effective,
            })
            trial.set_user_attr("folds", fold_audits)
            print(f"[hpo] study={study.study_name} trial={trial.number} inner={fold.number}/{len(folds)} {metric}={score:.6g}", flush=True)
        result = float(np.mean(fold_scores))
        trial.set_user_attr("objective_aggregation", "arithmetic_mean_of_inner_fold_scores")
        trial.set_user_attr("objective_value", result)
        return result

    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, timeout=allowed_s, n_jobs=1, catch=(CandidateFitError,))
    elapsed = time.perf_counter() - started
    trials = study.get_trials(deepcopy=False)
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None and math.isfinite(trial.value)]
    if not complete:
        raise SearchError(f"HPO 的 {len(trials)} 次尝试没有任何有限有效候选；不能回退固定参数")
    if _METRIC_DIRECTIONS[metric] == "minimize":
        best = min(complete, key=lambda trial: (trial.value, trial.number))
    else:
        best = min(complete, key=lambda trial: (-trial.value, trial.number))
    full_params = best.user_attrs.get("effective_estimator_parameters")
    sources = best.user_attrs.get("parameter_sources")
    if not isinstance(full_params, dict) or not isinstance(sources, dict) or set(full_params) != set(sources):
        raise SearchError("最佳 trial 缺少完整有效参数与来源审计")
    if stop_reason is None:
        stop_reason = "max_trials" if len(trials) >= max_trials else "soft_timeout"
    return {
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_value": float(best.value),
        "best_parameters": copy.deepcopy(best.params),
        "constructor_parameters": {**fixed_estimator, **best.params},
        "effective_estimator_parameters": full_params,
        "parameter_sources": sources,
        "attempted_trials": len(trials), "complete_trials": len(complete),
        "failed_trials": sum(trial.state == optuna.trial.TrialState.FAIL for trial in trials),
        "stop_reason": stop_reason,
        "active_search_time_s": elapsed,
        "inner_folds": [fold.audit() for fold in folds],
    }
