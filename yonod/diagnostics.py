"""Pure fold-level training/heldout diagnostic contract.

This module never fits or transforms a model.  Callers supply predictions from
the already-fitted final outer-fold estimator and the original-label arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "yonod_fold_diagnostic/v1"


class DiagnosticError(ValueError):
    """A fold cannot be paired without changing its population or values."""


def _ids(values: Sequence[Any], name: str) -> list[str]:
    result = [str(value) for value in values]
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise DiagnosticError(f"{name} 必须为非空、非重复的 sample_id 序列")
    return result


def _hash(values: Sequence[str]) -> str:
    source = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _values(values: Sequence[Any], count: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"{name} 必须为数值数组") from exc
    if len(array) != count or not np.isfinite(array).all():
        raise DiagnosticError(f"{name} 长度不匹配或包含非有限数值")
    return array


def score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Return finite, unweighted metrics with per-metric undefined reasons."""
    truth = _values(y_true, len(y_true), "y_true")
    pred = _values(y_pred, len(truth), "y_pred")
    if not len(truth):
        raise DiagnosticError("评分人口不能为空")
    residual = truth - pred
    with np.errstate(over="ignore", invalid="ignore"):
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        denominator = float(np.sum(np.square(truth - np.mean(truth))))
        numerator = float(np.sum(np.square(residual)))
    metrics: dict[str, Any] = {"n_total": len(truth), "n_valid": len(truth), "r2": None, "rmse": None, "mae": None, "reasons": {}}
    for name, value in (("mae", mae), ("rmse", rmse)):
        if np.isfinite(value):
            metrics[name] = value
        else:
            metrics["reasons"][name] = "metric_overflow"
    if len(truth) < 2:
        metrics["reasons"]["r2"] = "fewer_than_two_samples"
    elif not np.isfinite(denominator):
        metrics["reasons"]["r2"] = "metric_overflow"
    elif denominator == 0:
        metrics["reasons"]["r2"] = "constant_target"
    elif np.isfinite(numerator / denominator):
        metrics["r2"] = float(1.0 - numerator / denominator)
    else:
        metrics["reasons"]["r2"] = "metric_overflow"
    return metrics


def build_fold_diagnostic(
    *,
    identity: Mapping[str, Any],
    train_ids: Sequence[Any],
    heldout_ids: Sequence[Any],
    y_train: Sequence[Any],
    pred_train: Sequence[Any],
    y_heldout: Sequence[Any],
    pred_heldout: Sequence[Any],
    fit_ids: Sequence[Any] | None = None,
    stop_ids: Sequence[Any] | None = None,
    label_column: str,
    label_unit: str | None = None,
    preprocessing_fit_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Pair one final fold model's resubstitution and outer-heldout scores."""
    train = _ids(train_ids, "outer_train")
    heldout = _ids(heldout_ids, "outer_heldout")
    if set(train) & set(heldout):
        raise DiagnosticError("outer_train 与 outer_heldout 样本重叠")
    fit = _ids(fit_ids if fit_ids is not None else train, "actual_fit")
    stop = _ids(stop_ids, "early_stop_holdout") if stop_ids is not None else []
    if set(fit) & set(stop) or set(fit) | set(stop) != set(train):
        raise DiagnosticError("actual_fit 与 early_stop_holdout 必须无交集并恰好组成 outer_train")
    preprocess = _ids(preprocessing_fit_ids if preprocessing_fit_ids is not None else train, "preprocessing_fit")
    if not set(preprocess) <= set(train):
        raise DiagnosticError("预处理拟合范围超出 outer_train")
    truth_train = _values(y_train, len(train), "y_train")
    prediction_train = _values(pred_train, len(train), "pred_train")
    truth_heldout = _values(y_heldout, len(heldout), "y_heldout")
    prediction_heldout = _values(pred_heldout, len(heldout), "pred_heldout")
    train_pos = {sample_id: index for index, sample_id in enumerate(train)}
    positions = lambda selected: [train_pos[sample_id] for sample_id in selected]
    mean = float(np.mean(truth_train))
    if not np.isfinite(mean):
        raise DiagnosticError("outer_train 标签均值溢出")
    scores = {
        "outer_train": score(truth_train, prediction_train),
        "actual_fit": score(truth_train[positions(fit)], prediction_train[positions(fit)]),
        "outer_heldout": score(truth_heldout, prediction_heldout),
        "outer_train_mean_on_heldout": score(truth_heldout, np.full(len(heldout), mean)),
    }
    if stop:
        scores["early_stop_holdout"] = score(truth_train[positions(stop)], prediction_train[positions(stop)])
    train_score, outer_score = scores["outer_train"], scores["outer_heldout"]
    gaps = {
        name: None if train_score[name] is None or outer_score[name] is None else (
            train_score[name] - outer_score[name] if name == "r2" else outer_score[name] - train_score[name]
        ) for name in ("r2", "rmse", "mae")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "label": {"column": label_column, "unit": label_unit or "unknown", "scale": "original"},
        "populations": {
            name: {"count": len(ids), "ordered_ids_sha256": _hash(ids), "set_ids_sha256": _hash(sorted(ids))}
            for name, ids in (("outer_train", train), ("actual_fit", fit), ("early_stop_holdout", stop),
                              ("outer_heldout", heldout), ("preprocessing_fit", preprocess))
        },
        "baseline": {"kind": "outer_train_arithmetic_mean", "constant": mean},
        "scores": scores,
        "gaps": gaps,
        "gap_direction": {"r2": "outer_train_minus_outer_heldout", "rmse": "outer_heldout_minus_outer_train", "mae": "outer_heldout_minus_outer_train"},
        "training_score_kind": "resubstitution_diagnostic_only",
    }


def read_diagnostic(root: Path, reference: Mapping[str, Any] | None, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Read a hash-bound report record without model loading or prediction."""
    if reference is None:
        return {"status": "unrecorded", "reason": "历史结果未记录训练诊断"}
    status = str(reference.get("status", "unrecorded"))
    if status != "complete":
        return {"status": status, "reason": str(reference.get("reason", status))}
    try:
        relative = reference["path"]
        if not isinstance(relative, str):
            raise DiagnosticError("诊断引用不是相对路径")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise DiagnosticError("诊断引用越界或文件缺失")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference.get("sha256"):
            raise DiagnosticError("诊断内容哈希不匹配")
        payload = json.loads(data) if path.suffix == ".json" else yaml.safe_load(data)
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise DiagnosticError("诊断 schema 不受支持")
        if not isinstance(payload.get("populations"), Mapping) or not isinstance(payload.get("scores"), Mapping):
            raise DiagnosticError("诊断人口或分数结构缺失")
        for population in ("outer_train", "outer_heldout"):
            if not isinstance(payload["populations"].get(population), Mapping) or not isinstance(payload["populations"][population].get("count"), int):
                raise DiagnosticError(f"诊断人口 {population} 缺失")
        for population in ("outer_train", "outer_heldout", "outer_train_mean_on_heldout"):
            values = payload["scores"].get(population)
            if not isinstance(values, Mapping) or not isinstance(values.get("reasons"), Mapping) or any(metric not in values for metric in ("r2", "rmse", "mae")):
                raise DiagnosticError(f"诊断分数 {population} 缺失")
        if not isinstance(payload.get("gaps"), Mapping) or any(metric not in payload["gaps"] for metric in ("r2", "rmse", "mae")):
            raise DiagnosticError("诊断差距结构缺失")
        if not isinstance(payload.get("baseline"), Mapping) or "constant" not in payload["baseline"]:
            raise DiagnosticError("诊断基线结构缺失")
        recorded = payload.get("identity") or {}
        if any(recorded.get(key) != value for key, value in identity.items()):
            raise DiagnosticError("诊断任务或折身份不匹配")
        source_hash = payload.get("source_predictions_sha256")
        if source_hash is not None:
            source_path = (root / str(recorded.get("source_predictions", ""))).resolve()
            if root.resolve() not in source_path.parents or not source_path.is_file():
                raise DiagnosticError("原外层预测引用越界或缺失")
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
                raise DiagnosticError("原外层预测哈希不匹配")
        train_reference = payload.get("train_predictions")
        if train_reference is not None:
            train_path = (root / str(train_reference.get("path", ""))).resolve()
            if root.resolve() not in train_path.parents or not train_path.is_file():
                raise DiagnosticError("训练逐行预测引用越界或缺失")
            if hashlib.sha256(train_path.read_bytes()).hexdigest() != train_reference.get("sha256"):
                raise DiagnosticError("训练逐行预测哈希不匹配")
        return {"status": "complete", "payload": payload, "path": relative}
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        return {"status": "failed", "reason": f"诊断证据无效：{exc}"}


def diagnostic_table_row(record: Mapping[str, Any], *, feature: str, model: str,
                         repeat: int, fold: int) -> dict[str, Any]:
    """Stable display row; empty metrics remain empty rather than invented."""
    row: dict[str, Any] = {"feature": feature, "model": model, "repeat": repeat,
                           "fold": fold, "status": record.get("status", "unrecorded"),
                           "reason": record.get("reason", "")}
    if record.get("status") == "complete":
        payload = record["payload"]
        scores = payload["scores"]
        row.update({"n_train": payload["populations"]["outer_train"]["count"],
                    "n_heldout": payload["populations"]["outer_heldout"]["count"],
                    "train_mean": payload["baseline"]["constant"],
                    "diagnostic_predict_time_s": payload.get("prediction_time_s")})
        for name, key in (("train", "outer_train"), ("heldout", "outer_heldout"),
                          ("baseline", "outer_train_mean_on_heldout")):
            for metric in ("r2", "rmse", "mae"):
                row[f"{name}_{metric}"] = scores[key][metric]
                if scores[key][metric] is None:
                    row[f"{name}_{metric}_reason"] = scores[key]["reasons"].get(metric)
        for metric in ("r2", "rmse", "mae"):
            row[f"gap_{metric}"] = payload["gaps"][metric]
        row["evidence"] = record["path"]
    return row


def summarize_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize folds within repeat; disclose each metric's own denominator."""
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["feature"]), str(row["model"]), int(row["repeat"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (feature, model, repeat), folds in sorted(groups.items()):
        summary: dict[str, Any] = {"feature": feature, "model": model, "repeat": repeat,
                                   "expected_folds": len(folds),
                                   "complete_folds": sum(row.get("status") == "complete" for row in folds),
                                   "missing_or_failed_folds": sum(row.get("status") != "complete" for row in folds),
                                   "std_ddof": 0}
        for metric in ("train_r2", "train_rmse", "train_mae", "heldout_r2", "heldout_rmse",
                       "heldout_mae", "baseline_r2", "baseline_rmse", "baseline_mae",
                       "gap_r2", "gap_rmse", "gap_mae"):
            values = [float(row[metric]) for row in folds if row.get(metric) is not None
                      and np.isfinite(float(row[metric]))]
            summary[f"{metric}_n"] = len(values)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else None
        summaries.append(summary)
    return summaries
