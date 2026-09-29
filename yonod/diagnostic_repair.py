"""Explicit, fit-free recovery of one completed fold's training diagnostic.

The source run is immutable. A repair is published into a new directory and
uses only verified saved model/transform state and the original dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from yonod.diagnostics import build_fold_diagnostic, read_diagnostic
from yonod.pipeline.prediction import predict_saved_fold, predict_saved_strict_fold
from yonod.features.numeric_conditions import normalise_numeric_contract, numeric_input_identity
from yonod.provenance import feature_dataset_identity


class DiagnosticRepairError(ValueError):
    """Source evidence cannot support a safe diagnosis without fitting."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _raw_dataset(path: Path, id_column: str, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if id_column not in frame or label_column not in frame:
        raise DiagnosticRepairError("原始数据缺少 sample_id 或标签列")
    ids = frame[id_column].astype(str)
    if ids.empty or ids.duplicated().any() or ids.str.strip().eq("").any():
        raise DiagnosticRepairError("原始数据 sample_id 必须非空且唯一")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise DiagnosticRepairError("原始数据标签包含无效值")
    frame = frame.copy()
    frame[id_column] = ids
    frame[label_column] = labels.astype(float)
    return frame.set_index(id_column, drop=False)


def _source_fold(
    mode: str, root: Path, dataset: Path, feature: str, model: str,
    repeat: int, fold: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    """Return identity, raw train/heldout rows, saved predictions and provenance."""
    if model not in {"rf", "xgb", "lightgbm"}:
        raise DiagnosticRepairError("仅支持已保存完整状态的 RF/XGBoost/LightGBM 折")
    if not feature or feature in {".", ".."} or "/" in feature or "\\" in feature:
        raise DiagnosticRepairError("feature 必须是单个安全的特征 ID")
    if mode == "strict":
        run_manifest_path = root / "docs" / "manifests" / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if _sha(dataset) != run_manifest.get("dataset_sha256"):
            raise DiagnosticRepairError("数据文件哈希与 strict run manifest 不一致")
        metadata_path = root / "docs" / "folds" / f"{feature}__{model}__r{repeat:02d}__f{fold:02d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("run_id"), metadata.get("descriptor"), metadata.get("model"),
                metadata.get("repeat"), metadata.get("fold")) != (
                run_manifest.get("run_id"), feature, model, repeat, fold):
            raise DiagnosticRepairError("strict 折元数据身份不匹配")
        split_path = root / "docs" / "manifests" / "split_manifest.parquet"
        split = pd.read_parquet(split_path)
        part = split.loc[(split.repeat == repeat) & (split.fold == fold)]
        train_ids = part.loc[part.role == "train", "sample_id"].astype(str).tolist()
        heldout_ids = part.loc[part.role == "valid", "sample_id"].astype(str).tolist()
        if (not train_ids or not heldout_ids or len(set(train_ids)) != len(train_ids) or
                len(set(heldout_ids)) != len(heldout_ids) or set(train_ids) & set(heldout_ids)):
            raise DiagnosticRepairError("strict split 人口无效")
        shard = root / "docs" / "predictions" / metadata_path.with_suffix(".parquet").name
        saved = pd.read_parquet(shard)
        if set(saved.sample_id.astype(str)) != set(heldout_ids) or saved.sample_id.duplicated().any():
            raise DiagnosticRepairError("strict 外层预测与 split 人口不匹配")
        bundle_ref = metadata.get("model_adapter_metadata") or {}
        bundle_path = (root / str(bundle_ref.get("bundle_path", ""))).resolve()
        if root not in bundle_path.parents or not bundle_path.is_file() or _sha(bundle_path) != bundle_ref.get("bundle_sha256"):
            raise DiagnosticRepairError("strict 模型包缺失或哈希不匹配")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if (bundle.get("train_sample_ids_hash") != _ids_hash(train_ids) or
                metadata.get("train_sample_ids_hash") != _ids_hash(train_ids)):
            raise DiagnosticRepairError("strict 模型包与 split 外层训练人口不一致")
        id_column = bundle.get("sample_id_col") or run_manifest["benchmark_config"].get("sample_id_col")
        label_column = (bundle.get("dataset_roles") or {}).get("label") or run_manifest["benchmark_config"].get("label_col")
        if not id_column or not label_column:
            raise DiagnosticRepairError("strict 模型包缺少数据角色")
        frame = _raw_dataset(dataset, str(id_column), str(label_column))
        identity = {"run_id": metadata["run_id"], "descriptor": feature, "model": model,
                    "repeat": repeat, "fold": fold, "source_predictions": "source/heldout.parquet"}
        provenance = {"run_manifest": _sha(run_manifest_path), "fold_metadata": _sha(metadata_path),
                      "split_manifest": _sha(split_path), "model_bundle": _sha(bundle_path),
                      "source_predictions": _sha(shard), "dataset": _sha(dataset)}
    elif mode == "ordinary":
        run_dir = root / "runs" / f"run_{feature}-{model}"
        run_manifest_path = run_dir / "run_manifest.yaml"
        run_manifest = yaml.safe_load(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("status") != "complete" or run_manifest.get("feature_id") != feature or run_manifest.get("model") != model:
            raise DiagnosticRepairError("ordinary 运行未完成或组合身份不匹配")
        config_path = run_dir / "effective_config.yaml"
        if _sha(config_path) != run_manifest.get("effective_config_sha256"):
            raise DiagnosticRepairError("ordinary 配置快照哈希不匹配")
        source_config_path = run_dir / "source_config.yaml"
        if _sha(source_config_path) != run_manifest.get("source_config_sha256"):
            raise DiagnosticRepairError("ordinary 源配置快照哈希不匹配")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        roles = config["dataset"]["column_roles"]
        id_column, label_column = config["dataset"]["sample_id_col"], roles["label"]
        frame = _raw_dataset(dataset, id_column, label_column)
        identity_frame = pd.read_csv(dataset)
        for column in roles.get("conditions", []) or []:
            identity_frame[column] = frame[column].tolist()
        identity_frame[id_column] = frame.index.tolist()
        identity_frame[label_column] = frame[label_column].to_numpy(dtype=float)
        dataset_identity = feature_dataset_identity(identity_frame, sample_id_col=id_column,
                                                    column_roles=roles)
        if dataset_identity != run_manifest.get("dataset_identity"):
            raise DiagnosticRepairError("数据身份与 ordinary run manifest 不一致")
        shard = run_dir / "predictions.csv"
        all_predictions = pd.read_csv(shard)
        saved = all_predictions.loc[(all_predictions.repeat == repeat) & (all_predictions.fold == fold)].copy()
        population = set(all_predictions.loc[all_predictions.repeat == repeat, "sample_id"].astype(str))
        ordered_population = [sample_id for sample_id in frame.index.tolist() if sample_id in population]
        label_identity = "sha256:" + hashlib.sha256(json.dumps({
            "dataset_identity": dataset_identity, "label_column": label_column,
            "sample_ids": ordered_population,
            "labels": frame.loc[ordered_population, label_column].astype(float).tolist(),
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if label_identity != run_manifest.get("label_identity"):
            raise DiagnosticRepairError("原始标签与 ordinary run manifest 不一致")
        numeric_contract = normalise_numeric_contract(config["dataset"])
        if numeric_contract["columns"]:
            numeric_identity = numeric_input_identity(frame.reset_index(drop=True), frame.index.tolist(), numeric_contract)
            if numeric_identity != run_manifest.get("numeric_identity"):
                raise DiagnosticRepairError("原始数值输入与 ordinary run manifest 不一致")
        heldout_ids = saved.sample_id.astype(str).tolist()
        train_ids = [sample_id for sample_id in frame.index.tolist() if sample_id in population and sample_id not in set(heldout_ids)]
        if not train_ids or not heldout_ids or len(set(heldout_ids)) != len(heldout_ids):
            raise DiagnosticRepairError("ordinary 折人口无效")
        bundle_path = run_dir / "model_bundles" / f"repeat-{repeat:02d}-fold-{fold:02d}.yaml"
        bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        if (bundle.get("run_id") != run_manifest.get("run_id") or
                bundle.get("train_sample_ids_sha256") != _ids_hash(train_ids)):
            raise DiagnosticRepairError("ordinary 模型包与外层训练人口不一致")
        identity = {"run_id": run_manifest["run_id"], "feature_id": feature, "model": model,
                    "repeat": repeat, "fold": fold, "source_predictions": "source/heldout.csv"}
        provenance = {"run_manifest": _sha(run_manifest_path), "effective_config": _sha(config_path),
                      "model_bundle": _sha(bundle_path), "source_predictions": _sha(shard),
                      "dataset_identity": dataset_identity}
    else:
        raise DiagnosticRepairError("mode 必须为 ordinary 或 strict")
    if not set(train_ids + heldout_ids) <= set(frame.index):
        raise DiagnosticRepairError("原始数据缺少训练或外层样本")
    return (identity, frame.loc[train_ids].copy(), frame.loc[heldout_ids].copy(), saved,
            shard, {"source": provenance, "label_column": str(label_column), "id_column": str(id_column),
                    "run_dir": str(run_dir) if mode == "ordinary" else str(root), "bundle": bundle})


def repair_saved_fold(
    *, mode: str, source_root: Path | str, dataset_path: Path | str,
    destination: Path | str, feature: str, model: str, repeat: int, fold: int,
) -> Path:
    """Predict a saved fold without fit/search and publish a separate repair task."""
    source = Path(source_root).resolve()
    dataset = Path(dataset_path).resolve()
    target = Path(destination).resolve()
    if target.exists() or target == source or source in target.parents or target in source.parents:
        raise DiagnosticRepairError("修复输出必须是尚不存在、与源隔离的目录")
    identity, train, heldout, saved, shard, info = _source_fold(
        mode, source, dataset, feature, model, repeat, fold)
    id_column, label_column = info["id_column"], info["label_column"]
    raw = pd.concat([train, heldout], ignore_index=True)
    if mode == "strict":
        loaded = predict_saved_strict_fold(source, raw, descriptor=feature, model=model,
                                           repeat=repeat, fold=fold)
    else:
        loaded = predict_saved_fold(info["run_dir"], raw, repeat=repeat, fold=fold)
    predictions = loaded.set_index("sample_id")["y_pred"]
    saved = saved.set_index(saved.sample_id.astype(str))
    if set(saved.index) != set(heldout.index):
        raise DiagnosticRepairError("保存的外层预测与折人口不一致")
    np.testing.assert_allclose(predictions.loc[heldout.index].to_numpy(dtype=float),
                               saved.loc[heldout.index, "y_pred"].to_numpy(dtype=float),
                               rtol=1e-6, atol=1e-8, err_msg="保存模型与原外层预测不一致")
    np.testing.assert_allclose(heldout[label_column].to_numpy(dtype=float),
                               saved.loc[heldout.index, "y_true"].to_numpy(dtype=float),
                               rtol=0, atol=1e-12, err_msg="保存标签与原外层预测不一致")
    fit_ids = None
    stop_ids = None
    if mode == "ordinary":
        metrics = pd.read_csv(Path(info["run_dir"]) / "fold_metrics.csv")
        selected = metrics.loc[(metrics.repeat == repeat) & (metrics.fold == fold)]
        if len(selected) != 1:
            raise DiagnosticRepairError("ordinary 折指标无法唯一定位")
        early = (json.loads(selected.iloc[0]["model_audit"]).get("early_stopping") or {})
        if early:
            all_ids = _raw_dataset(dataset, id_column, label_column).index.tolist()
            fit_ids = [all_ids[index] for index in early["inner_training_row_indices"]]
            stop_ids = [all_ids[index] for index in early["inner_validation_row_indices"]]
    diagnostic = build_fold_diagnostic(
        identity=identity, train_ids=train.index.tolist(), heldout_ids=heldout.index.tolist(),
        y_train=train[label_column], pred_train=predictions.loc[train.index],
        y_heldout=heldout[label_column], pred_heldout=saved.loc[heldout.index, "y_pred"],
        fit_ids=fit_ids, stop_ids=stop_ids, preprocessing_fit_ids=train.index.tolist(),
        label_column=label_column,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        (staging / "diagnostics").mkdir(parents=True)
        (staging / "source").mkdir()
        train_path = staging / "diagnostics" / "train_predictions.csv"
        pd.DataFrame({"sample_id": train.index, "population": "outer_train",
                      "y_true": train[label_column].to_numpy(dtype=float),
                      "y_pred": predictions.loc[train.index].to_numpy(dtype=float)}).to_csv(train_path, index=False)
        heldout_path = staging / identity["source_predictions"]
        if mode == "strict":
            saved.reset_index(drop=True).to_parquet(heldout_path, index=False)
        else:
            saved.reset_index(drop=True).to_csv(heldout_path, index=False)
        diagnostic["source_predictions_sha256"] = _sha(heldout_path)
        diagnostic["train_predictions"] = {"path": "diagnostics/train_predictions.csv", "sha256": _sha(train_path)}
        diagnostic_path = staging / "diagnostics" / "fold.json"
        diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        reference = {"status": "complete", "path": "diagnostics/fold.json", "sha256": _sha(diagnostic_path)}
        if read_diagnostic(staging, reference, identity=identity)["status"] != "complete":
            raise DiagnosticRepairError("修复证据自校验失败")
        (staging / "repair_manifest.json").write_text(json.dumps({
            "schema_version": "yonod_diagnostic_repair/v1", "mode": mode,
            "source_root": str(source), "source_fold": identity,
            "source_hashes": info["source"], "dataset_path": str(dataset),
            "dataset_sha256": _sha(dataset), "diagnostic": reference,
            "fit_calls": 0, "search_calls": 0,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair one saved fold diagnostic without fitting")
    parser.add_argument("--mode", choices=("ordinary", "strict"), required=True)
    for key in ("source-root", "dataset-path", "destination", "feature", "model"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    path = repair_saved_fold(mode=args.mode, source_root=args.source_root,
                             dataset_path=args.dataset_path, destination=args.destination,
                             feature=args.feature, model=args.model, repeat=args.repeat, fold=args.fold)
    print(path / "repair_manifest.json")


if __name__ == "__main__":
    main()
