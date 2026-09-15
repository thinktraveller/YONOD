"""Step-30 model configuration, construction, and fold-local fitting.

The old adapters deliberately carried convenient project defaults.  That is
useful for a historical command line, but is the wrong boundary for the YAML
interface: an omitted estimator option must reach the installed library as an
omitted option.  This module is the one place where schema-2 model sections
are routed to an estimator/predictor and where their effective values are
recorded.

Imports of optional libraries are deliberately delayed until the selected
model is actually trained.  Validation of the *shape* of a YAML declaration is
therefore cheap, while exact constructor validation happens before the first
fold is fitted.
"""

from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np


class ModelConfigurationError(ValueError):
    """Raised before a model is constructed from an invalid YAML declaration."""


@dataclass(frozen=True)
class ResolvedModelConfig:
    """A model declaration split by its actual destination API."""

    model: str
    estimator: Dict[str, Any]
    fit: Dict[str, Any]
    preprocessing: Dict[str, Any]
    runtime: Dict[str, Any]
    parameter_sources: Dict[str, str]

    def as_audit_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "estimator": copy.deepcopy(self.estimator),
            "fit": copy.deepcopy(self.fit),
            "preprocessing": copy.deepcopy(self.preprocessing),
            "runtime": copy.deepcopy(self.runtime),
            "parameter_sources": copy.deepcopy(self.parameter_sources),
        }


# LightGBM's sklearn constructor accepts ``**kwargs``.  Unlike a normal Python
# signature, that does not mean arbitrary YAML keys are safe.  These names are
# the documented regressor/booster parameters for the supported 4.x family;
# aliases are accepted by LightGBM itself and are represented here so the
# configuration layer does not silently lose a valid library option.
_LIGHTGBM_PARAMETERS = frozenset({
    "boosting_type", "num_leaves", "max_depth", "learning_rate", "n_estimators",
    "subsample_for_bin", "objective", "class_weight", "min_split_gain",
    "min_child_weight", "min_child_samples", "subsample", "subsample_freq",
    "colsample_bytree", "reg_alpha", "reg_lambda", "random_state", "n_jobs",
    "importance_type", "device_type", "verbosity", "max_bin", "min_data_in_bin",
    "bin_construct_sample_cnt", "data_sample_strategy", "extra_trees",
    "path_smooth", "max_delta_step", "linear_tree", "max_cat_to_onehot",
    "cat_l2", "cat_smooth", "max_cat_threshold", "monotone_constraints",
    "monotone_constraints_method", "monotone_penalty", "feature_contri",
    "interaction_constraints", "lambda_l1", "lambda_l2", "bagging_fraction",
    "bagging_freq", "bagging_seed", "feature_fraction", "feature_fraction_seed",
    "drop_rate", "max_drop", "skip_drop", "xgboost_dart_mode", "uniform_drop",
    "dart_seed", "top_rate", "other_rate", "min_data_per_group",
    "max_cat_to_onehot", "deterministic", "force_col_wise", "force_row_wise",
    "num_threads", "seed", "feature_fraction_bynode", "min_gain_to_split",
    "min_sum_hessian_in_leaf", "is_unbalance", "scale_pos_weight",
    "boost_from_average", "first_metric_only", "metric", "eval_at",
    "early_stopping_round", "early_stopping_rounds", "callbacks",
})

_AUTOGLOON_PREDICTOR_PARAMETERS = frozenset({
    "label", "problem_type", "eval_metric", "path", "verbosity", "log_to_file",
    "sample_weight", "weight_evaluation", "groups", "quantile_levels",
    "learner_kwargs", "learner_type", "positive_class",
})
_AUTOGLOON_FIT_PARAMETERS = frozenset({
    "train_data", "time_limit", "presets", "hyperparameters", "feature_metadata",
    "num_cpus", "num_gpus", "fit_weighted_ensemble", "fit_full_last_level_weighted_ensemble",
    "fit_extra", "holdout_frac", "num_bag_folds", "num_bag_sets", "num_stack_levels",
    "auto_stack", "dynamic_stacking", "excluded_model_types", "included_model_types",
    "refit_full", "set_best_to_refit_full", "keep_only_best", "save_space",
    "ag_args", "ag_args_fit", "ag_args_ensemble", "verbosity", "raise_on_no_models_fitted",
    "calibrate", "fit_strategy", "delay_bag_sets", "use_bag_holdout",
})
_AUTOGLOON_RUNTIME_PARAMETERS = frozenset({"save_path", "cleanup"})
_EXTERNAL_FIT_DATA_PARAMETERS = frozenset({
    "eval_set", "sample_weight_eval_set", "base_margin", "base_margin_eval_set",
    "feature_weights", "xgb_model", "init_score", "eval_sample_weight",
    "eval_init_score", "init_model", "callbacks",
})


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigurationError(f"{field} 必须是 mapping")
    return copy.deepcopy(dict(value))


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ModelConfigurationError(f"{field} 必须为 bool")
    return value


def _require_non_negative_integer(value: Any, field: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelConfigurationError(f"{field} 必须为不小于 0 的整数")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str] | frozenset[str], field: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ModelConfigurationError(f"{field} 包含不受支持的参数：{', '.join(unknown)}")


def _constructor_parameters(factory: Any) -> set[str]:
    """Ask a sklearn-compatible estimator for all public constructor options."""
    try:
        return set(factory().get_params(deep=False))
    except Exception as exc:
        raise ModelConfigurationError(f"无法读取所安装库的模型参数目录：{exc}") from exc


def _fit_parameters(method: Any) -> set[str]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError(f"无法读取所安装库的 fit 参数目录：{exc}") from exc
    return {
        name for name, parameter in signature.parameters.items()
        if name not in {"self", "X", "y", "data", "train_data"}
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }


def _library_catalog(model: str) -> tuple[set[str], set[str]]:
    """Return constructor and fit option names for the selected installed model."""
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return _constructor_parameters(RandomForestRegressor), _fit_parameters(RandomForestRegressor.fit)
    if model == "svm":
        from sklearn.svm import SVR
        return _constructor_parameters(SVR), _fit_parameters(SVR.fit)
    if model == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ModelConfigurationError("XGBoost 未安装，无法验证 xgb 参数") from exc
        return _constructor_parameters(XGBRegressor), _fit_parameters(XGBRegressor.fit)
    if model == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ModelConfigurationError("LightGBM 未安装，无法验证 lightgbm 参数") from exc
        return _constructor_parameters(LGBMRegressor) | set(_LIGHTGBM_PARAMETERS), _fit_parameters(LGBMRegressor.fit)
    if model == "autogluon":
        return set(_AUTOGLOON_PREDICTOR_PARAMETERS), set(_AUTOGLOON_FIT_PARAMETERS)
    raise ModelConfigurationError(f"不支持的模型：{model!r}")


def _normalise_sample_weight(value: Any, field: str) -> Dict[str, str]:
    """Force configuration to name a data column rather than embed an array."""
    mapping = _mapping(value, field)
    _reject_unknown(mapping, {"column"}, field)
    column = mapping.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ModelConfigurationError(f"{field}.column 必须是非空字符串")
    return {"column": column.strip()}


def _validate_svm_preprocessing(raw: Mapping[str, Any]) -> Dict[str, Any]:
    _reject_unknown(raw, {"scaler", "pca", "subsample"}, "model_params.svm.preprocessing")
    output: Dict[str, Any] = {}
    if "scaler" in raw:
        scaler = _mapping(raw["scaler"], "model_params.svm.preprocessing.scaler")
        _reject_unknown(scaler, {"enabled", "params"}, "model_params.svm.preprocessing.scaler")
        enabled = _require_bool(scaler.get("enabled"), "model_params.svm.preprocessing.scaler.enabled")
        params = _mapping(scaler.get("params", {}), "model_params.svm.preprocessing.scaler.params")
        from sklearn.preprocessing import StandardScaler
        _reject_unknown(params, _constructor_parameters(StandardScaler), "model_params.svm.preprocessing.scaler.params")
        output["scaler"] = {"enabled": enabled, "params": params}
    if "pca" in raw:
        pca = _mapping(raw["pca"], "model_params.svm.preprocessing.pca")
        _reject_unknown(pca, {"enabled", "feature_threshold", "params"}, "model_params.svm.preprocessing.pca")
        enabled = _require_bool(pca.get("enabled"), "model_params.svm.preprocessing.pca.enabled")
        threshold = pca.get("feature_threshold")
        if threshold is not None:
            _require_non_negative_integer(threshold, "model_params.svm.preprocessing.pca.feature_threshold")
        params = _mapping(pca.get("params", {}), "model_params.svm.preprocessing.pca.params")
        from sklearn.decomposition import PCA
        _reject_unknown(params, _constructor_parameters(PCA), "model_params.svm.preprocessing.pca.params")
        output["pca"] = {"enabled": enabled, "feature_threshold": threshold, "params": params}
    if "subsample" in raw:
        subsample = _mapping(raw["subsample"], "model_params.svm.preprocessing.subsample")
        _reject_unknown(subsample, {"n_samples", "random_state"}, "model_params.svm.preprocessing.subsample")
        count = _require_non_negative_integer(
            subsample.get("n_samples"), "model_params.svm.preprocessing.subsample.n_samples", allow_none=True
        )
        if count == 0:
            raise ModelConfigurationError("model_params.svm.preprocessing.subsample.n_samples 不能为 0")
        if "random_state" in subsample:
            _require_non_negative_integer(subsample["random_state"], "model_params.svm.preprocessing.subsample.random_state")
        output["subsample"] = {"n_samples": count, **({"random_state": subsample["random_state"]} if "random_state" in subsample else {})}
    return output


def _validate_inner_early_stopping(value: Any, field: str) -> Dict[str, Any]:
    """Declare an inner split without accepting YAML-provided data arrays."""
    raw = _mapping(value, field)
    _reject_unknown(raw, {"rounds", "validation_fraction", "seed"}, field)
    rounds = raw.get("rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ModelConfigurationError(f"{field}.rounds 必须为不小于 1 的整数")
    fraction = raw.get("validation_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 < float(fraction) < 1.0:
        raise ModelConfigurationError(f"{field}.validation_fraction 必须在 0 和 1 之间")
    seed = raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ModelConfigurationError(f"{field}.seed 必须为整数")
    return {"rounds": int(rounds), "validation_fraction": float(fraction), "seed": int(seed)}


def _validate_runtime(model: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = _mapping(raw, f"model_params.{model}.runtime")
    if model == "xgb":
        _reject_unknown(runtime, {"device_policy", "early_stopping"}, "model_params.xgb.runtime")
        result: Dict[str, Any] = {}
        if "device_policy" in runtime:
            policy = runtime["device_policy"]
            if policy not in {"auto", "cpu", "cuda"}:
                raise ModelConfigurationError("model_params.xgb.runtime.device_policy 必须为 auto、cpu 或 cuda")
            result["device_policy"] = policy
        if "early_stopping" in runtime:
            result["early_stopping"] = _validate_inner_early_stopping(
                runtime["early_stopping"], "model_params.xgb.runtime.early_stopping"
            )
        return result
    if model == "lightgbm":
        _reject_unknown(runtime, {"early_stopping"}, "model_params.lightgbm.runtime")
        if "early_stopping" in runtime:
            return {"early_stopping": _validate_inner_early_stopping(
                runtime["early_stopping"], "model_params.lightgbm.runtime.early_stopping"
            )}
        return {}
    if model == "autogluon":
        _reject_unknown(runtime, _AUTOGLOON_RUNTIME_PARAMETERS, "model_params.autogluon.runtime")
        if "save_path" in runtime and runtime["save_path"] is not None and not isinstance(runtime["save_path"], str):
            raise ModelConfigurationError("model_params.autogluon.runtime.save_path 必须为字符串或 null")
        if "cleanup" in runtime:
            _require_bool(runtime["cleanup"], "model_params.autogluon.runtime.cleanup")
        return runtime
    if runtime:
        raise ModelConfigurationError(f"model_params.{model}.runtime 当前没有受支持的运行时参数")
    return runtime


def resolve_model_config(model: str, model_params: Mapping[str, Any] | None) -> ResolvedModelConfig:
    """Validate and route a model's YAML section without adding project defaults.

    The return value contains only explicitly declared options.  The factory
    therefore lets the installed library provide omitted defaults naturally.
    """
    if model not in {"rf", "xgb", "svm", "lightgbm", "autogluon"}:
        raise ModelConfigurationError(f"不支持的模型：{model!r}")
    raw = _mapping(model_params or {}, f"model_params.{model}")
    allowed_sections = {
        "rf": {"estimator", "fit", "runtime"},
        "xgb": {"estimator", "fit", "runtime"},
        "svm": {"estimator", "fit", "preprocessing", "runtime"},
        "lightgbm": {"estimator", "fit", "runtime"},
        "autogluon": {"predictor", "fit", "runtime"},
    }[model]
    _reject_unknown(raw, allowed_sections, f"model_params.{model}")
    constructor_allowed, fit_allowed = _library_catalog(model)
    estimator_key = "predictor" if model == "autogluon" else "estimator"
    estimator = _mapping(raw.get(estimator_key, {}), f"model_params.{model}.{estimator_key}")
    _reject_unknown(estimator, constructor_allowed, f"model_params.{model}.{estimator_key}")
    fit = _mapping(raw.get("fit", {}), f"model_params.{model}.fit")
    _reject_unknown(fit, fit_allowed, f"model_params.{model}.fit")
    unsafe_external = sorted(_EXTERNAL_FIT_DATA_PARAMETERS.intersection(fit))
    if unsafe_external:
        raise ModelConfigurationError(
            f"model_params.{model}.fit 不允许直接声明外部数据/回调对象：{', '.join(unsafe_external)}；"
            "如需 XGBoost 或 LightGBM early stopping，请使用 runtime.early_stopping，"
            "它只从当前外层训练折创建 inner validation"
        )
    if "sample_weight" in fit:
        fit["sample_weight"] = _normalise_sample_weight(fit["sample_weight"], f"model_params.{model}.fit.sample_weight")
    if model == "autogluon":
        # The outer validation fold must never enter AutoGluon as a YAML path
        # or object.  Internal holdout is represented by holdout_frac only.
        forbidden = {"train_data", "tuning_data", "unlabeled_data"}.intersection(fit)
        if forbidden:
            raise ModelConfigurationError(
                "model_params.autogluon.fit 不允许声明外部数据对象：" + ", ".join(sorted(forbidden))
            )
        if estimator.get("label") is not None and not isinstance(estimator.get("label"), str):
            raise ModelConfigurationError("model_params.autogluon.predictor.label 必须为字符串")
    if model == "xgb" and "device" in estimator and "runtime" in raw:
        raise ModelConfigurationError("model_params.xgb.estimator.device 与 runtime.device_policy 不能同时声明")
    preprocessing: Dict[str, Any] = {}
    if model == "svm":
        preprocessing = _validate_svm_preprocessing(
            _mapping(raw.get("preprocessing", {}), "model_params.svm.preprocessing")
        )
    runtime = _validate_runtime(model, raw.get("runtime", {}))
    sources = {
        f"{estimator_key}.{key}": "yaml" for key in estimator
    }
    sources.update({f"fit.{key}": "yaml" for key in fit})
    sources.update({f"preprocessing.{key}": "yaml" for key in preprocessing})
    sources.update({f"runtime.{key}": "yaml" for key in runtime})
    return ResolvedModelConfig(model, estimator, fit, preprocessing, runtime, sources)


def _resolve_xgb_device(policy: str) -> str:
    if policy in {"cpu", "cuda"}:
        return policy
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def construct_estimator(config: ResolvedModelConfig, *, n_features: int, n_train: int) -> tuple[Any, Dict[str, Any]]:
    """Create one sklearn-compatible estimator and return an audit snapshot."""
    model = config.model
    params = copy.deepcopy(config.estimator)
    preprocessing_audit: Dict[str, Any] = {}
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(**params)
    elif model == "xgb":
        from xgboost import XGBRegressor
        if "device_policy" in config.runtime:
            params["device"] = _resolve_xgb_device(config.runtime["device_policy"])
        if "early_stopping" in config.runtime:
            # XGBoost 2.x deprecates passing this through `.fit()`.  The
            # data-free stopping budget belongs on the estimator; the inner
            # validation matrix is still supplied by the training service.
            params.setdefault("early_stopping_rounds", int(config.runtime["early_stopping"]["rounds"]))
        estimator = XGBRegressor(**params)
    elif model == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(**params)
    elif model == "svm":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        steps: list[tuple[str, Any]] = []
        scaler = config.preprocessing.get("scaler")
        if scaler and scaler["enabled"]:
            steps.append(("scaler", StandardScaler(**dict(scaler["params"]))))
        pca = config.preprocessing.get("pca")
        if pca and pca["enabled"]:
            threshold = pca.get("feature_threshold")
            if threshold is None or n_features > threshold:
                pca_params = dict(pca["params"])
                if "n_components" in pca_params and isinstance(pca_params["n_components"], int):
                    pca_params["n_components"] = min(pca_params["n_components"], n_features, n_train)
                steps.append(("pca", PCA(**pca_params)))
                preprocessing_audit["pca_effective_n_components"] = pca_params.get("n_components")
        steps.append(("svr", SVR(**params)))
        estimator = Pipeline(steps)
        preprocessing_audit["steps"] = [name for name, _ in steps]
    else:
        raise ModelConfigurationError("AutoGluon 需要 construct_autogluon_predictor")
    try:
        effective = estimator.get_params(deep=True)
    except AttributeError:
        effective = params
    snapshot = json.loads(json.dumps(effective, ensure_ascii=False, sort_keys=True, default=str))
    return estimator, {"effective_estimator_params": snapshot, "preprocessing": preprocessing_audit}


def construct_autogluon_predictor(config: ResolvedModelConfig, *, label: str, path: Path) -> tuple[Any, Dict[str, Any]]:
    """Build an AutoGluon predictor while binding its label to the dataset role."""
    if config.model != "autogluon":
        raise ModelConfigurationError("construct_autogluon_predictor 仅接受 autogluon 配置")
    predictor_params = dict(config.estimator)
    declared_label = predictor_params.get("label", label)
    if declared_label != label:
        raise ModelConfigurationError(
            f"AutoGluon predictor.label={declared_label!r} 与 dataset.column_roles.label={label!r} 不一致"
        )
    predictor_params["label"] = label
    # Artifact directories are execution state, not a path hidden in library
    # defaults.  An explicit predictor.path outranks the runtime path.
    predictor_params.setdefault("path", str(path))
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ModelConfigurationError("AutoGluon 未安装，无法训练 autogluon 模型") from exc
    predictor = TabularPredictor(**predictor_params)
    return predictor, {"effective_predictor_params": copy.deepcopy(predictor_params)}


def materialise_fit_parameters(config: ResolvedModelConfig, frame: Any, train_index: np.ndarray) -> Dict[str, Any]:
    """Resolve data-column fit options on training rows only.

    YAML never stores a free-form array.  At present the common native option
    is ``sample_weight: {column: weight}``; retaining this at the data boundary
    makes it impossible to accidentally feed all-row or validation values to a
    fold fit call.
    """
    fit = copy.deepcopy(config.fit)
    weight = fit.get("sample_weight")
    if weight is not None:
        column = weight["column"]
        if column not in frame.columns:
            raise ModelConfigurationError(f"sample_weight 引用的数据列不存在：{column}")
        values = np.asarray(frame.iloc[train_index][column], dtype=float)
        if values.ndim != 1 or len(values) != len(train_index) or not np.isfinite(values).all():
            raise ModelConfigurationError("sample_weight 训练折值必须为有限的一维数值")
        # sklearn Pipeline routes fit parameters to a named final step.  The
        # same YAML declaration remains model-native and does not expose this
        # implementation detail to users.
        fit.pop("sample_weight")
        fit["svr__sample_weight" if config.model == "svm" else "sample_weight"] = values
    return fit


def software_versions(model: str) -> Dict[str, str]:
    """Report only installed package versions; unavailable optional libs are explicit."""
    distributions = {"rf": ["scikit-learn"], "svm": ["scikit-learn"], "xgb": ["xgboost"], "lightgbm": ["lightgbm"], "autogluon": ["autogluon.tabular"]}[model]
    result: Dict[str, str] = {}
    for name in distributions:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result
