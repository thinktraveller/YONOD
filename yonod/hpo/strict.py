"""Manifest-preserving strict outer-fold selection using the shared search."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from yonod.config.output_layout import combination_name

from .contracts import validate_hpo_declaration
from .folds import make_inner_folds
from .search import SearchPopulation, run_inner_search
from .storage import PersistentStudy, dependency_versions, source_fingerprints


class StrictHPOError(ValueError):
    """A strict manifest fold cannot be searched without preserving grouping."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def select_strict_outer_parameters(
    *, output_root: Path, sample_ids: Sequence[str], labels: Sequence[float],
    split_manifest: pd.DataFrame, repeat: int, fold: int,
    descriptor: str, model: str, model_config: Mapping[str, Any],
    hpo_raw: Mapping[str, Any], models: Sequence[str],
    grouping_strategy: str, feature_matrix: np.ndarray | None,
    categorical_frame: pd.DataFrame | None, ohe_factory: Any | None,
    numeric_frame: pd.DataFrame | None, numeric_contract: Mapping[str, Any] | None,
    source_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a winner from strict outer-train rows and audited group keys."""
    hpo = validate_hpo_declaration(
        hpo_raw, models=models, stage="benchmark",
        evaluation={"protocol": "manifest_outer_cv"},
        model_params={model: model_config}, benchmark={"reproduction_protocol": {}},
    )
    if not hpo["enabled"] or model not in hpo["models"]:
        raise StrictHPOError("当前模型未在 hpo.models 中")
    if hpo["final_model"]["enabled"]:
        raise StrictHPOError("strict 最终模型训练尚未接入；不能复用任一外层折")
    if hpo["budget"].get("study_timeout_s") or hpo["budget"].get("task_timeout_s"):
        raise StrictHPOError("HPO 软时间预算持久账本尚未接入，暂不接受时限字段")
    ids = [str(item) for item in sample_ids]
    if len(set(ids)) != len(ids):
        raise StrictHPOError("strict sample_id 必须唯一")
    part = split_manifest[(split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold)]
    if part.empty or "group_id" not in part:
        raise StrictHPOError("strict 外层 manifest 缺少当前折或 group_id")
    role_by_id = part.set_index("sample_id")["role"].to_dict()
    group_by_id = part.set_index("sample_id")["group_id"].to_dict()
    if len(role_by_id) != len(part) or set(role_by_id) != set(ids):
        raise StrictHPOError("strict manifest 与特征人口不能一对一匹配")
    train = np.asarray([i for i, sample_id in enumerate(ids) if role_by_id[sample_id] == "train"], dtype=int)
    valid = np.asarray([i for i, sample_id in enumerate(ids) if role_by_id[sample_id] == "valid"], dtype=int)
    if not len(train) or not len(valid) or len(train) + len(valid) != len(ids):
        raise StrictHPOError("strict 外层 manifest train/valid 角色无效")
    train_ids = [ids[i] for i in train]
    grouped = grouping_strategy != "repeated_kfold"
    groups = [str(group_by_id[sample_id]) for sample_id in train_ids] if grouped else None
    inner = make_inner_folds(train_ids, groups=groups, **hpo["inner_cv"])
    if feature_matrix is None and categorical_frame is None:
        raise StrictHPOError("strict HPO 需要原始静态矩阵或未拟合 OHE frame")
    static_train = np.asarray(feature_matrix)[train] if feature_matrix is not None else None
    categorical_train = categorical_frame.iloc[train].reset_index(drop=True) if categorical_frame is not None else None
    numeric_train = numeric_frame.iloc[train].reset_index(drop=True) if numeric_frame is not None else None
    source_train = source_frame.iloc[train].reset_index(drop=True) if source_frame is not None else None
    population = SearchPopulation(
        sample_ids=train_ids, y=np.asarray(labels, dtype=float)[train],
        static_matrix=static_train, categorical_frame=categorical_train,
        ohe_factory=ohe_factory, numeric_frame=numeric_train,
        numeric_contract=numeric_contract, fit_frame=source_train,
    )
    if static_train is not None:
        feature_content = {
            "shape": list(static_train.shape), "dtype": str(static_train.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(static_train).tobytes()).hexdigest(),
        }
    else:
        feature_content = {"categorical_rows": categorical_train.astype(str).to_dict(orient="records")}
    identity = {
        "purpose": "outer_fold", "outer_fold": {"repeat": repeat, "fold": fold},
        "feature_identity": feature_content,
        "training_ids": train_ids,
        "training_labels": np.asarray(labels, dtype=float)[train].tolist(),
        "inner_folds": [item.audit() for item in inner],
        "preprocessing": {"grouping_strategy": grouping_strategy,
                          "group_keys": groups, "numeric_contract": numeric_contract,
                          "numeric_rows": numeric_train.to_dict(orient="records") if numeric_train is not None else None},
        "fixed_model_config": dict(model_config),
        "search_space": hpo["search_spaces"][model],
        "objective": hpo["objective"], "seed": hpo["seed"],
        "budget": hpo["budget"], "versions": {
            **dependency_versions(model),
            **source_fingerprints({
                "yonod_hpo_strict_sha256": Path(__file__),
                "yonod_hpo_search_sha256": Path(__file__).with_name("search.py"),
                "yonod_hpo_folds_sha256": Path(__file__).with_name("folds.py"),
                "yonod_model_factory_sha256": Path(__file__).resolve().parents[1] / "model_factory.py",
                "yonod_strict_executor_sha256": Path(__file__).resolve().parents[1] / "benchmark" / "executor.py",
            }),
        },
    }
    store = PersistentStudy(output_root, combination_name(descriptor, model), identity)
    with store.open() as study:
        result = run_inner_search(
            study=study, model=model, base_model_config=model_config,
            space=hpo["search_spaces"][model], population=population,
            folds=inner, metric=hpo["objective"]["metric"], budget=hpo["budget"],
        )
    return {
        **result, "study_id": store.study_id,
        "study_dir": store.root.relative_to(Path(output_root).resolve()).as_posix(),
        "study_manifest_sha256": hashlib.sha256(store.manifest_path.read_bytes()).hexdigest(),
        "study_trials_sha256": hashlib.sha256(store.trials_path.read_bytes()).hexdigest(),
        "outer_training_ids_sha256": _digest(train_ids),
    }
