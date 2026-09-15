#!/usr/bin/env python3
"""One-time, auditable conversion of legacy YONOD run configuration files.

This program is intentionally *not* part of the runtime loader.  It reads a
known legacy format, writes a schema-2 YAML only when the old semantics can be
spelled out safely, and always writes a sidecar report.  JSON result files,
NPZ metadata, and reference-project files are outside its scope.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.config.loader import ConfigLoadError, load_run_config


class MigrationError(ValueError):
    """The source configuration has no safe schema-2 equivalent."""


_MODEL_ALIASES = {
    "xgboost": "xgb", "random forest": "rf", "random_forest": "rf",
    "svm": "svm", "autogluon": "autogluon", "auto_gluon": "autogluon",
    "lightgbm": "lightgbm", "light gbm": "lightgbm", "lgbm": "lightgbm",
}
_AG_FIT = {"time_limit", "presets", "hyperparameters", "num_cpus", "num_gpus", "holdout_frac", "num_bag_folds", "num_bag_sets", "num_stack_levels", "auto_stack", "excluded_model_types", "included_model_types", "refit_full", "keep_only_best", "save_space", "ag_args", "ag_args_fit", "ag_args_ensemble", "verbosity"}
_AG_RUNTIME = {"save_path", "cleanup"}
_SVM_PREPROCESSING = {"auto_pca", "pca_threshold", "pca_n_components", "subsample_n"}


def _legacy_json_profile(model: str) -> Dict[str, Any]:
    """Return the values the retired ``main.py --json`` path really used.

    Old wizard JSON was not a general model-parameter surface: apart from a
    small AutoGluon subset, its ``model_params`` field was ignored and values
    came from the legacy adapters/argparse defaults.  Migration must preserve
    those effective values explicitly rather than accidentally activating
    ignored fields in the schema-2 factory.
    """
    profiles: Dict[str, Dict[str, Any]] = {
        "rf": {
            "estimator": {
                "n_estimators": 300, "max_depth": None,
                "min_samples_leaf": 1, "max_features": 1.0,
                "n_jobs": -1, "random_state": 42, "verbose": 0,
            },
        },
        "xgb": {
            "estimator": {
                "n_estimators": 300, "learning_rate": 0.05,
                "max_depth": 6, "tree_method": "hist",
                "random_state": 42, "verbosity": 0,
            },
            "runtime": {"device_policy": "auto"},
        },
        "svm": {
            "estimator": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            "preprocessing": {
                "scaler": {"enabled": True, "params": {"with_mean": False}},
                "pca": {
                    "enabled": True, "feature_threshold": 512,
                    "params": {"n_components": 256, "random_state": 42},
                },
                "subsample": {"n_samples": 8000, "random_state": 42},
            },
        },
        "lightgbm": {
            "estimator": {
                "n_estimators": 500, "learning_rate": 0.05,
                "num_leaves": 31, "min_child_samples": 1,
                "random_state": 42, "n_jobs": 1,
                "device_type": "cpu", "verbosity": -1,
            },
        },
        "autogluon": {
            "predictor": {"problem_type": "regression", "verbosity": 1},
            "fit": {"time_limit": 300, "presets": "medium_quality", "num_cpus": 1},
            "runtime": {"cleanup": True},
        },
    }
    return copy.deepcopy(profiles[model])


def _legacy_json_effective_model_params(raw: Any, models: list[str], report: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the historical wizard's *effective*, not merely stored, values."""
    if raw is None:
        legacy: Mapping[str, Any] = {}
    elif isinstance(raw, Mapping):
        legacy = raw
    else:
        raise MigrationError("旧 model_params 必须为 mapping")

    selected = set(models)
    autogluon_values: Mapping[str, Any] = {}
    for original_model, value in legacy.items():
        model = _normalise_model(original_model)
        if model not in selected:
            report["ignored_legacy_fields"].append(f"model_params.{original_model}（模型不在 models）")
            continue
        if not isinstance(value, Mapping):
            raise MigrationError(f"model_params.{original_model} 必须为 mapping")
        if model != "autogluon":
            report["ignored_legacy_fields"].append(
                f"model_params.{original_model}（旧 main.py --json 未将其传入 {model} 适配器）"
            )
            continue
        autogluon_values = value

    result = {model: _legacy_json_profile(model) for model in models}
    if "autogluon" in selected:
        # config_to_args only read these three top-level keys.  Nested/new
        # sectional declarations were not consumed by the retired JSON path.
        fit = result["autogluon"]["fit"]
        for key in ("time_limit", "presets", "num_cpus"):
            if key in autogluon_values:
                fit[key] = copy.deepcopy(autogluon_values[key])
        ignored = sorted(key for key in autogluon_values if key not in {"time_limit", "presets", "num_cpus"})
        if ignored:
            report["ignored_legacy_fields"].append(
                "model_params.AutoGluon." + ", ".join(ignored) + "（旧 main.py --json 未读取）"
            )
    report["effective_parameter_profile"] = "legacy_main_json_entry_v1"
    return result


def _normalise_model(value: Any) -> str:
    key = str(value).strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        raise MigrationError(f"未知旧模型名称，无法安全转换：{value!r}") from exc


def _relative(reference: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), reference.parent.resolve()).replace(os.sep, "/")


def _report_path(output: Path) -> Path:
    return output.with_name(output.name + ".migration.json")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _descriptors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise MigrationError("旧配置没有非空 descriptors，无法确定特征生命周期")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(value):
        if not isinstance(source, Mapping) or not source.get("descriptor"):
            raise MigrationError(f"descriptors[{index}] 不是有效描述符声明")
        item = {key: copy.deepcopy(source[key]) for key in ("id", "descriptor", "lifecycle", "mode", "columns", "extra_reactants", "params") if key in source}
        item.setdefault("id", str(item["descriptor"]))
        if item["id"] in seen:
            raise MigrationError(f"描述符 id 重复：{item['id']!r}")
        seen.add(str(item["id"]))
        result.append(item)
    return result


def _wrap_legacy_model_params(raw: Any, models: list[str], report: Dict[str, Any]) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise MigrationError("旧 model_params 必须为 mapping")
    result: Dict[str, Any] = {}
    for original_model, value in raw.items():
        model = _normalise_model(original_model)
        if model not in models:
            report["ignored_legacy_fields"].append(f"model_params.{original_model}（模型不在 models）")
            continue
        if not isinstance(value, Mapping):
            raise MigrationError(f"model_params.{original_model} 必须为 mapping")
        # A schema-2 declaration is already sectional; retain it as-is so the
        # authoritative runtime factory can validate detailed library options.
        if any(key in {"estimator", "predictor", "fit", "preprocessing", "runtime"} for key in value):
            result[model] = copy.deepcopy(dict(value))
            continue
        flat = copy.deepcopy(dict(value))
        if model == "autogluon":
            fit = {key: flat.pop(key) for key in list(flat) if key in _AG_FIT}
            runtime = {key: flat.pop(key) for key in list(flat) if key in _AG_RUNTIME}
            section: Dict[str, Any] = {}
            if flat:
                section["predictor"] = flat
            if fit:
                section["fit"] = fit
            if runtime:
                section["runtime"] = runtime
            result[model] = section
        elif model == "svm":
            pre = {key: flat.pop(key) for key in list(flat) if key in _SVM_PREPROCESSING}
            section = {"estimator": flat}
            if pre:
                report["manual_review"].append(
                    "旧 SVM auto_pca/pca_threshold/pca_n_components/subsample_n 没有无歧义的"
                    " schema-2 折内声明；已省略，请按项目协议显式填写 preprocessing。"
                )
            result[model] = section
        else:
            result[model] = {"estimator": flat}
    return result


def _legacy_json_to_schema2(source: Path, payload: Mapping[str, Any], output: Path, sample_id_col: str | None, report: Dict[str, Any]) -> Dict[str, Any]:
    project_name = str(payload.get("project_name", "")).strip()
    roles = payload.get("column_roles")
    if not project_name or not isinstance(roles, Mapping):
        raise MigrationError("旧 JSON 缺少 project_name 或 column_roles")
    if not sample_id_col:
        raise MigrationError("旧 JSON 不包含稳定 sample_id；请显式传入 --sample-id-col，迁移器不会猜测行号")
    models = [_normalise_model(value) for value in payload.get("models", [])]
    if not models:
        raise MigrationError("旧 JSON 没有 models")
    if len(set(models)) != len(models):
        raise MigrationError("旧模型名称规范化后发生冲突")
    normalized = source.parent / f"{project_name}_normalized_dataset.csv"
    dataset_path = normalized
    if not normalized.is_file():
        origin = payload.get("origin_dataset_path")
        if not origin:
            raise MigrationError("找不到旧规范数据集，且 origin_dataset_path 未声明")
        dataset_path = (source.parent / str(origin)).resolve()
        report["manual_review"].append("使用 origin_dataset_path；请确认它是向导过滤后的正确训练 population。")
    if not dataset_path.is_file():
        raise MigrationError(f"迁移后的 dataset.path 不存在：{dataset_path}")
    config: Dict[str, Any] = {
        "schema_version": "2.0",
        "project_name": project_name,
        "stage": "all",
        "dataset": {
            "path": _relative(output, dataset_path),
            "sample_id_col": sample_id_col,
            "column_roles": copy.deepcopy(dict(roles)),
        },
        "descriptors": _descriptors(payload.get("descriptors")),
        "artifacts": {"output_dir": f"artifacts/{project_name}"},
        "models": models,
        "model_params": _legacy_json_effective_model_params(payload.get("model_params"), models, report),
        "evaluation": {"protocol": "outer_kfold", "n_splits": 5, "n_repeats": 1, "shuffle": True, "seed": 42},
        "outputs": {"root": f"results/{project_name}", "report_formats": list(payload.get("report_formats", ["HTML", "Markdown"]))},
        "metadata": {
            "migration": {"source_format": "legacy_yonod_json", "source_filename": source.name},
            "legacy_metadata": copy.deepcopy(dict(payload.get("metadata", {}))),
        },
    }
    report["manual_review"].append("旧 JSON 的普通入口固定使用 5-fold/seed=42；请确认迁移后的 evaluation 与历史运行意图。")
    return config


def _benchmark_yaml_to_schema2(source: Path, payload: Mapping[str, Any], output: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    block = payload.get("benchmark", payload)
    if not isinstance(block, Mapping):
        raise MigrationError("旧 benchmark YAML 根节点/benchmark 节点必须为 mapping")
    grouping = dict(block.get("grouping", {}))
    if grouping.get("strategy") not in {"repeated_kfold", None}:
        raise MigrationError(
            "旧 benchmark grouping.strategy={0!r} 目前不能映射到普通 schema-2 outer_kfold；"
            "请保留原协议入口或先提供专用 split manifest。".format(grouping.get("strategy"))
        )
    for field in ("dataset_path", "sample_id_col", "label_col", "models"):
        if not block.get(field):
            raise MigrationError(f"旧 benchmark 缺少 {field}")
    descriptors = block.get("feature_sets")
    if descriptors is None:
        descriptors = [{"id": name, "descriptor": name, "mode": "concat"} for name in block.get("descriptors", [])]
    converted: list[dict[str, Any]] = []
    for item in descriptors:
        if not isinstance(item, Mapping):
            raise MigrationError("旧 feature_sets/descriptors 包含非 mapping 项")
        if "kind" in item:
            kind = item.get("kind")
            name = str(item.get("name", "")).strip()
            if kind == "precomputed_descriptor":
                converted.append({"id": name, "descriptor": name, "mode": "concat", "columns": list(item.get("component_cols", [])), "params": copy.deepcopy(dict(item.get("params", {})))})
            elif kind == "fold_transform" and name == "ohe":
                converted.append({"id": name, "descriptor": "ohe", "mode": "concat", "columns": list(item.get("component_cols", [])), "params": copy.deepcopy(dict(item.get("params", {})))})
            else:
                raise MigrationError(f"旧 feature_set {name!r} 无安全 schema-2 映射")
        else:
            converted.append(copy.deepcopy(dict(item)))
    models = [_normalise_model(value) for value in block["models"]]
    dataset_path = Path(str(block["dataset_path"]))
    if not dataset_path.is_absolute():
        dataset_path = (source.parent / dataset_path).resolve()
    config = {
        "schema_version": "2.0", "project_name": source.stem, "stage": "all",
        "dataset": {
            "path": _relative(output, dataset_path), "sample_id_col": str(block["sample_id_col"]),
            "column_roles": {"label": str(block["label_col"]), "reactants": list(block.get("smiles_cols", [])), "products": [], "others": [], "categoricals": []},
        },
        "descriptors": _descriptors(converted),
        "artifacts": {"output_dir": f"artifacts/{source.stem}"}, "models": models,
        "model_params": _wrap_legacy_model_params(block.get("model_params"), models, report),
        "evaluation": {"protocol": "outer_kfold", "n_splits": int(block.get("cv", {}).get("n_splits", 5)), "n_repeats": int(block.get("cv", {}).get("n_repeats", 1)), "seed": int(block.get("cv", {}).get("seed", 42)), "shuffle": True},
        "outputs": {"root": _relative(output, (source.parent / str(block.get("outputs", {}).get("root", "results"))).resolve())},
        "metadata": {"migration": {"source_format": "legacy_benchmark_yaml", "source_filename": source.name, "legacy_grouping": grouping}},
    }
    report["manual_review"].append("旧 benchmark 的任务状态、grouping 细节及报告协议未进入普通 train；请核查专用 benchmark/paper_exact 需求。")
    return config


def migrate(source: Path, output: Path, *, sample_id_col: str | None = None) -> Dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    report: Dict[str, Any] = {
        "source": str(source), "output": str(output), "status": "failed",
        "manual_review": [], "ignored_legacy_fields": [],
    }
    created_output = False
    try:
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise MigrationError("旧 JSON 根节点必须为对象")
            config = _legacy_json_to_schema2(source, payload, output, sample_id_col, report)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise MigrationError("旧 YAML 根节点必须为 mapping")
            config = _benchmark_yaml_to_schema2(source, payload, output, report)
        else:
            raise MigrationError("迁移器仅接受旧 JSON 或旧 benchmark YAML")
        if output.exists():
            raise MigrationError(f"输出 YAML 已存在，拒绝覆盖：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        created_output = True
        # The exact executable loader is the final authority; conversion never
        # claims success merely because the YAML serializer accepted a mapping.
        load_run_config(output)
        report.update({"status": "converted", "schema_version": "2.0", "models": config["models"]})
    except Exception as exc:
        if created_output and output.exists() and report["status"] != "converted":
            # We only created this output in the try block and only before
            # validation.  Do not remove a pre-existing user path.
            output.unlink()
        report["error"] = f"{type(exc).__name__}: {exc}"
    _write_report(_report_path(output), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一次性转换旧 YONOD JSON/benchmark YAML 为 schema-2 YAML")
    parser.add_argument("--input", required=True, type=Path, help="旧运行配置（JSON 或旧 benchmark YAML）")
    parser.add_argument("--output", required=True, type=Path, help="新 schema-2 YAML；不得已存在")
    parser.add_argument("--sample-id-col", help="旧 JSON 必填：现有 CSV 中稳定唯一的样本 ID 列")
    args = parser.parse_args(argv)
    report = migrate(args.input, args.output, sample_id_col=args.sample_id_col)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "converted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
