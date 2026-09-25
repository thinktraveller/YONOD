"""Restore one trusted fold model and transform new raw rows without refitting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml

from yonod.descriptors.ohe import OHEFeature
from yonod.descriptors.registry import normalise_feature_specs
from yonod.features.numeric_conditions import NumericConditionsTransformer
from yonod.pipeline.features import _production_feature_computer


class PredictionContractError(ValueError):
    """A saved fold cannot safely consume the requested prediction rows."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, reference: str, *, file: bool = True) -> Path:
    path = (root / reference).resolve()
    if path == root or root not in path.parents or (file and not path.is_file()) or (not file and not path.is_dir()):
        raise PredictionContractError(f"模型包引用无效：{reference}")
    return path


def predict_saved_fold(
    run_dir: Path | str,
    frame: pd.DataFrame,
    *,
    repeat: int,
    fold: int,
    input_units: Mapping[str, str] | None = None,
    descriptor_config_path: Path | str | None = None,
) -> pd.DataFrame:
    """Predict from raw named columns using one immutable fold's fitted state.

    ``input_units`` must name each dimensioned numeric source column. A
    chemical-VAE descriptor also needs the original YAML path to resolve its
    separately versioned model asset. The caller is responsible for treating
    untrusted joblib files as untrusted code and should load only local runs.
    """
    root = Path(run_dir).resolve()
    bundle_path = _within(root, f"model_bundles/repeat-{repeat:02d}-fold-{fold:02d}.yaml")
    manifest_path = _within(root, "run_manifest.yaml")
    config_path = _within(root, "effective_config.yaml")
    try:
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PredictionContractError(f"无法读取预测契约：{exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "yonod_model_bundle/v1":
        raise PredictionContractError("模型包版本不受支持")
    if not isinstance(manifest, dict) or bundle.get("run_id") != manifest.get("run_id"):
        raise PredictionContractError("模型包与训练结果身份不一致")
    if int(bundle.get("repeat", -1)) != repeat or int(bundle.get("fold", -1)) != fold:
        raise PredictionContractError("模型包 repeat/fold 不匹配")
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise PredictionContractError("预测数据必须是列名唯一的 DataFrame")
    id_col = config["dataset"]["sample_id_col"]
    if id_col not in frame.columns or frame[id_col].isna().any():
        raise PredictionContractError(f"预测数据缺少有效 sample_id 列 {id_col!r}")
    sample_ids = frame[id_col].astype(str).tolist()
    if not sample_ids or any(not value.strip() for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise PredictionContractError("预测 sample_id 必须唯一、非空")

    spec_raw = bundle.get("feature_spec")
    lifecycle = bundle.get("feature_lifecycle")
    if lifecycle == "fold_transform":
        directory = _within(root, str(bundle.get("ohe_state_path")), file=False)
        transformer = OHEFeature.load(directory)
        descriptor = transformer.transform(frame, partition="predict")
    elif lifecycle == "static_descriptor":
        if not isinstance(spec_raw, Mapping):
            raise PredictionContractError("模型包缺少静态描述符声明")
        spec = normalise_feature_specs([spec_raw])[0]
        columns = list(spec_raw.get("columns") or bundle.get("descriptor_source_columns") or [])
        if not columns:
            raise PredictionContractError("模型包缺少描述符源列")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise PredictionContractError("预测数据缺少描述符列：" + ", ".join(missing))
        if spec.descriptor == "chemical_vae" and descriptor_config_path is None:
            raise PredictionContractError("chemical_vae 预测需提供原始 YAML 路径以定位经验证资产")
        source = Path(descriptor_config_path) if descriptor_config_path is not None else config_path
        roles = bundle.get("descriptor_roles") or {"reactant": columns, "product": [], "other": []}
        matrix, valid, _ = _production_feature_computer(spec, frame, columns, roles, config_path=source)
        if not np.asarray(valid, dtype=bool).all():
            bad = [sample_ids[index] for index, flag in enumerate(valid) if not flag]
            raise PredictionContractError("预测分子输入不可编码，sample_id：" + ", ".join(bad[:5]))
        descriptor = np.asarray(matrix, dtype=float)
    else:
        raise PredictionContractError(f"不支持的特征生命周期：{lifecycle!r}")
    descriptor_dim = bundle.get("descriptor_dim")
    if descriptor.shape != (len(frame), descriptor_dim):
        raise PredictionContractError("重新计算的描述符维度与模型包不一致")

    contract = bundle.get("numeric_contract")
    if contract is not None:
        units = dict(input_units or {})
        for entry in contract["columns"]:
            source_unit = entry["unit"]["source"]
            if source_unit != "dimensionless" and units.get(entry["source"]) != source_unit:
                raise PredictionContractError(
                    f"预测列 {entry['source']!r} 必须声明输入单位 {source_unit!r}"
                )
            if entry["source"] in units and units[entry["source"]] != source_unit:
                raise PredictionContractError(f"预测列 {entry['source']!r} 的单位不匹配")
        state_path = _within(root, str(bundle.get("numeric_state_path")))
        if _sha256_file(state_path) != bundle.get("numeric_state_sha256"):
            raise PredictionContractError("数值折状态哈希不匹配")
        state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        numeric = NumericConditionsTransformer.from_state_dict(state)
        if numeric.contract != contract:
            raise PredictionContractError("模型包与数值状态契约不一致")
        descriptor = np.hstack([descriptor, numeric.transform(frame, sample_ids, phase="predict")])
    if descriptor.shape[1] != bundle.get("feature_dim") or not np.isfinite(descriptor).all():
        raise PredictionContractError("模型实际输入维度或有限性与模型包不一致")
    model_path = _within(root, str(bundle.get("model_path")))
    if _sha256_file(model_path) != bundle.get("model_sha256"):
        raise PredictionContractError("模型文件哈希不匹配")
    estimator = joblib.load(model_path)
    prediction = np.asarray(estimator.predict(descriptor), dtype=float).reshape(-1)
    if len(prediction) != len(sample_ids) or not np.isfinite(prediction).all():
        raise PredictionContractError("模型预测数量或有限性无效")
    return pd.DataFrame({"sample_id": sample_ids, "y_pred": prediction})
