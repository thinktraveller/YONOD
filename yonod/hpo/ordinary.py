"""Bridge one ordinary outer fold to the isolated nested search service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from yonod.artifacts.contracts import artifact_content_identity
from yonod.artifacts.reader import FeatureArtifact
from yonod.config.output_layout import combination_name
from yonod.descriptors.ohe import OHEFeature

from .contracts import validate_hpo_declaration
from .folds import make_inner_folds
from .search import SearchPopulation, run_inner_search
from .storage import PersistentStudy, dependency_versions, source_fingerprints


class OrdinaryHPOError(ValueError):
    """A selected ordinary model cannot run a nested study safely."""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_outer_parameters(
    *, artifact: FeatureArtifact, config: Mapping[str, Any],
    aligned: pd.DataFrame, sample_ids: np.ndarray, labels: np.ndarray,
    outer_train_index: np.ndarray, repeat: int, fold: int, model: str,
    numeric_contract: Mapping[str, Any], output_root: Path,
) -> dict[str, Any]:
    """Return best parameters for one outer fold, without seeing outer-valid."""
    hpo = validate_hpo_declaration(
        config.get("hpo"), models=config["models"], stage=config["stage"],
        evaluation=config.get("evaluation"), model_params=config.get("model_params"),
    )
    if not hpo["enabled"] or model not in hpo["models"]:
        raise OrdinaryHPOError("仅被 hpo.models 选择的模型可进入搜索")
    if hpo["final_model"]["enabled"]:
        raise OrdinaryHPOError("显式最终模型搜索尚未接入；拒绝伪造 CV 折模型为 final 模型")
    if hpo["budget"].get("study_timeout_s") or hpo["budget"].get("task_timeout_s"):
        raise OrdinaryHPOError("HPO 软时间预算持久账本尚未接入，暂不接受时限字段")
    train = np.asarray(outer_train_index, dtype=int)
    ids = sample_ids[train].astype(str)
    y = np.asarray(labels[train], dtype=float)
    inner = make_inner_folds(ids, **hpo["inner_cv"])
    static_matrix = None
    categorical_frame = None
    ohe_factory = None
    if artifact.manifest["lifecycle"] == "fold_transform":
        transform = artifact.manifest.get("metadata", {}).get("fold_transform", {})
        if transform.get("transform") != "ohe" or transform.get("fit_scope") != "training_fold_only":
            raise OrdinaryHPOError("仅未拟合的原始 OHE 包可进入 HPO")
        columns = list(transform.get("input_columns") or [])
        if not columns or len(columns) != artifact.matrix.shape[1]:
            raise OrdinaryHPOError("OHE 原始类别矩阵与列声明不一致")
        sentinel = transform.get("missing_sentinel")
        if not isinstance(sentinel, str) or not sentinel:
            raise OrdinaryHPOError("OHE 原始缺失标记无效")
        source = pd.DataFrame(np.asarray(artifact.matrix[train], dtype=str), columns=columns)
        categorical_frame = source.mask(source.eq(sentinel), other=pd.NA)
        parameters = dict(transform.get("parameters") or {})
        ohe_factory = lambda: OHEFeature(
            columns, missing_policy=str(parameters.get("missing_policy", "as_category")),
            dtype=str(parameters.get("dtype", "float32")),
            handle_unknown=str(parameters.get("handle_unknown", "ignore")),
        )
    elif artifact.manifest["lifecycle"] == "static_descriptor":
        metadata = artifact.manifest.get("metadata", {})
        if metadata.get("preprocessing") or metadata.get("prefit_transform"):
            raise OrdinaryHPOError("预拟合处理状态无法恢复原始输入，拒绝 HPO")
        static_matrix = np.asarray(artifact.matrix[train])
    else:
        raise OrdinaryHPOError("未知特征生命周期，拒绝 HPO")
    numeric_frame = None
    if numeric_contract["columns"]:
        names = [entry["source"] for entry in numeric_contract["columns"]]
        numeric_frame = aligned.iloc[train].loc[:, names].reset_index(drop=True)
    population = SearchPopulation(
        sample_ids=ids.tolist(), y=y, static_matrix=static_matrix,
        categorical_frame=categorical_frame, ohe_factory=ohe_factory,
        numeric_frame=numeric_frame, numeric_contract=numeric_contract,
        fit_frame=aligned.iloc[train].reset_index(drop=True),
    )
    fixed = dict((config.get("model_params") or {}).get(model, {}))
    versions = dependency_versions(model)
    versions["hpo_contract"] = "2-14/v1"
    versions.update(source_fingerprints({
        "yonod_hpo_ordinary_sha256": Path(__file__),
        "yonod_hpo_search_sha256": Path(__file__).with_name("search.py"),
        "yonod_hpo_folds_sha256": Path(__file__).with_name("folds.py"),
        "yonod_model_factory_sha256": Path(__file__).resolve().parents[1] / "model_factory.py",
        "yonod_training_sha256": Path(__file__).resolve().parents[1] / "pipeline" / "training.py",
    }))
    identity = {
        "purpose": "outer_fold",
        "outer_fold": {"repeat": repeat, "fold": fold},
        "feature_identity": artifact_content_identity(artifact.manifest),
        "training_ids": ids.tolist(),
        "training_labels": y.tolist(),
        "inner_folds": [item.audit() for item in inner],
        "preprocessing": {"feature_lifecycle": artifact.manifest["lifecycle"],
                          "feature_spec": artifact.manifest.get("metadata", {}).get("feature_spec"),
                          "numeric_contract": numeric_contract},
        "fixed_model_config": fixed,
        "search_space": hpo["search_spaces"][model],
        "objective": hpo["objective"],
        "seed": hpo["seed"],
        "budget": hpo["budget"],
        "versions": versions,
    }
    store = PersistentStudy(output_root, combination_name(str(artifact.manifest["feature_id"]), model), identity)
    with store.open() as study:
        result = run_inner_search(
            study=study, model=model, base_model_config=fixed,
            space=hpo["search_spaces"][model], population=population,
            folds=inner, metric=hpo["objective"]["metric"], budget=hpo["budget"],
        )
    return {
        **result,
        "study_id": store.study_id,
        "study_dir": store.root.relative_to(Path(output_root).resolve()).as_posix(),
        "study_manifest_sha256": hashlib.sha256(store.manifest_path.read_bytes()).hexdigest(),
        "study_trials_sha256": hashlib.sha256(store.trials_path.read_bytes()).hexdigest(),
        "outer_training_ids_sha256": _digest(ids.tolist()),
    }
