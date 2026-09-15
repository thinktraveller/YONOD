"""Read-only report post-processing for schema-2 training results.

This module deliberately consumes only a completed ``run_manifest.yaml`` and
the files named by it.  It neither imports descriptor generators nor calls a
model ``fit`` method, so it is also the implementation behind offline rebuild.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from yonod.universal.report import generate_markdown_report, generate_report


class ReportServiceError(ValueError):
    """A persisted training result cannot safely be rendered as a report."""


@dataclass(frozen=True)
class ReportResult:
    run_id: str
    run_dir: Path
    pictures: tuple[Path, ...]
    html_path: Path | None
    markdown_path: Path | None


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportServiceError(f"无法读取 YAML：{path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReportServiceError(f"YAML 顶层必须为 mapping：{path}")
    return value


def _safe_piece(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:72] or "unnamed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _required_file(run_dir: Path, outputs: Mapping[str, Any], key: str) -> Path:
    name = outputs.get(key)
    if not isinstance(name, str) or not name:
        raise ReportServiceError(f"run_manifest.outputs.{key} 缺失")
    candidate = (run_dir / name).resolve()
    if candidate.parent != run_dir.resolve() or not candidate.is_file():
        raise ReportServiceError(f"run_manifest.outputs.{key} 指向不存在或越界文件：{name!r}")
    return candidate


def _validate_predictions(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    needed = {"sample_id", "repeat", "fold", "y_true", "y_pred"}
    if frame.empty or not needed.issubset(frame.columns):
        raise ReportServiceError(f"predictions.csv 必须包含非空列：{sorted(needed)}")
    if frame[list(needed)].isna().any().any():
        raise ReportServiceError("predictions.csv 含缺失的身份或预测字段")
    numeric = frame[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ReportServiceError("predictions.csv 含 NaN/Inf 或非数值预测")
    if frame.duplicated(["sample_id", "repeat", "fold"]).any():
        raise ReportServiceError("predictions.csv 存在重复 sample_id/repeat/fold 点")
    repeats = frame.groupby("repeat", dropna=False)
    expected = int(manifest.get("n_samples", 0))
    for repeat, rows in repeats:
        if expected and len(rows) != expected:
            raise ReportServiceError(f"repeat={repeat!r} 的 OOF 点数为 {len(rows)}，期望 {expected}")
        if rows["sample_id"].nunique() != len(rows):
            raise ReportServiceError(f"repeat={repeat!r} 未覆盖唯一 sample_id")
        if rows.groupby("sample_id")["y_true"].nunique().gt(1).any():
            raise ReportServiceError(f"repeat={repeat!r} 同一样本的真实标签不一致")


def _metrics(predictions: pd.DataFrame, folds: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rows: list[dict[str, Any]] = []
    for repeat, points in predictions.groupby("repeat", sort=True):
        truth = points["y_true"].to_numpy(dtype=float)
        predicted = points["y_pred"].to_numpy(dtype=float)
        r2 = float(r2_score(truth, predicted)) if len(truth) >= 2 and np.unique(truth).size > 1 else float("nan")
        rows.append({"repeat": repeat, "r2": r2, "rmse": float(np.sqrt(mean_squared_error(truth, predicted))), "mae": float(mean_absolute_error(truth, predicted))})
    aggregate = pd.DataFrame(rows)
    model_train_time = pd.to_numeric(folds.get("model_train_time_s"), errors="coerce") if "model_train_time_s" in folds else pd.Series(dtype=float)
    return pd.DataFrame([{
        "descriptor": str(manifest["feature_id"]), "model": str(manifest["model"]),
        "r2": aggregate["r2"].mean(), "r2_std": aggregate["r2"].std(ddof=0),
        "rmse": aggregate["rmse"].mean(), "rmse_std": aggregate["rmse"].std(ddof=0),
        "mae": aggregate["mae"].mean(), "mae_std": aggregate["mae"].std(ddof=0),
        "train_time_s": float(model_train_time.sum()) if not model_train_time.empty else float("nan"),
        "evaluation_protocol": str((manifest.get("evaluation") or {}).get("protocol", "outer_kfold")),
        "expected_folds": int(manifest.get("n_folds", len(folds))), "completed_folds": int(len(folds)),
        "cv": int(manifest.get("n_folds", len(folds))), "oof_complete": True,
        "oof_n_observed": int(len(predictions)), "oof_n_total": int(len(predictions)),
    }])


def _plot_repeat(points: pd.DataFrame, manifest: Mapping[str, Any], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from yonod.universal.report import _configure_chinese_matplotlib
    from sklearn.metrics import mean_squared_error, r2_score
    _configure_chinese_matplotlib(plt)
    true = points["y_true"].to_numpy(dtype=float)
    pred = points["y_pred"].to_numpy(dtype=float)
    r2 = float(r2_score(true, pred)) if len(true) >= 2 and np.unique(true).size > 1 else float("nan")
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
    pad = max((high - low) * 0.05, 1e-9)
    figure, axis = plt.subplots(figsize=(5.4, 5.0), dpi=140)
    axis.scatter(true, pred, s=20, alpha=.65, edgecolor="none", color="#2563eb")
    axis.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#64748b", linewidth=1, label="y = x")
    axis.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), xlabel="True value", ylabel="Predicted value")
    axis.set_title(f"{manifest['feature_id']} × {manifest['model']} | repeat {points['repeat'].iloc[0]}")
    r2_text = "N/A (constant target)" if not math.isfinite(r2) else f"R² = {r2:.3f}"
    axis.text(.04, .96, f"{r2_text}\nRMSE = {rmse:.4g}\nn = {len(points)}", transform=axis.transAxes, va="top", bbox={"boxstyle":"round", "fc":"white", "ec":"#cbd5e1"})
    axis.grid(alpha=.2); axis.legend(loc="lower right", frameon=False); figure.tight_layout()
    figure.savefig(target, format="png", bbox_inches="tight"); plt.close(figure)


def rebuild_schema2_report(run_dir: Path | str, *, output_root: Path | str | None = None, formats: Sequence[str] | None = None) -> ReportResult:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.yaml"
    manifest = _load_yaml(manifest_path)
    if manifest.get("kind") != "yonod_training_run" or manifest.get("schema_version") != "1.0" or manifest.get("status") != "complete":
        raise ReportServiceError("仅可从 status=complete、schema=1.0 的 yonod_training_run 重建报告")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping): raise ReportServiceError("run_manifest.outputs 缺失")
    predictions_path, folds_path = _required_file(run_dir, outputs, "predictions"), _required_file(run_dir, outputs, "fold_metrics")
    source_path, effective_path = _required_file(run_dir, outputs, "source_config"), _required_file(run_dir, outputs, "effective_config")
    if manifest.get("source_config_sha256") != _sha256(source_path) or manifest.get("effective_config_sha256") != _sha256(effective_path):
        raise ReportServiceError("配置快照哈希不匹配，拒绝把已篡改运行渲染为可信报告")
    predictions, folds = pd.read_csv(predictions_path), pd.read_csv(folds_path)
    _validate_predictions(predictions, manifest)
    if folds.empty or not {"repeat", "fold"}.issubset(folds.columns): raise ReportServiceError("fold_metrics.csv 不完整")
    source = _load_yaml(source_path)
    selected = [str(v).strip().lower() for v in (formats if formats is not None else ((source.get("outputs") or {}).get("report_formats", ["html", "markdown"]))) ]
    aliases = {"html": "html", "markdown": "markdown", "md": "markdown"}
    try: selected = [aliases[item] for item in selected]
    except KeyError as exc: raise ReportServiceError(f"不支持 outputs.report_formats：{exc.args[0]!r}") from exc
    if len(set(selected)) != len(selected): raise ReportServiceError("outputs.report_formats 不能重复")
    # Configuration validation rejects this, but this public offline API also
    # guards old/tampered snapshots.  Crucially, do nothing before returning.
    if not selected:
        return ReportResult(str(manifest["run_id"]), run_dir, (), None, None)
    root = Path(output_root).resolve() if output_root is not None else run_dir
    staging = root.parent / f".{root.name}.report-{uuid.uuid4().hex}.tmp" if output_root is not None else run_dir / f".report-{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    try:
        pictures_dir = staging / "pictures"; pictures_dir.mkdir()
        pictures: list[Path] = []
        for repeat, points in predictions.groupby("repeat", sort=True):
            picture = pictures_dir / f"scatter_{_safe_piece(str(manifest['feature_id']))}_{_safe_piece(str(manifest['model']))}_repeat-{_safe_piece(str(repeat))}.png"
            _plot_repeat(points, manifest, picture)
            if picture.stat().st_size < 128: raise ReportServiceError(f"散点图为空：{picture.name}")
            pictures.append(picture)
        metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
        info = {"task_name": source.get("project_name", manifest["run_id"]), "project_folder": str(root), "csv_path": "<stored training result>", "n_samples": manifest.get("n_samples"), "label_col": manifest.get("label_column"), "n_combinations": 1, "run_id": manifest["run_id"], "artifact_id": manifest["artifact_id"], "split_identity": manifest["split_identity"], "model_request": manifest.get("model_request"), "dataset_citation": metadata.get("doi"), "dataset_url": metadata.get("source_url", metadata.get("repo_url")), "dataset_notes": metadata.get("notes", metadata.get("source_notes"))}
        metrics = _metrics(predictions, folds, manifest)
        html_path = generate_report(metrics, info, staging, scatter_paths=pictures) if "html" in selected else None
        markdown_path = generate_markdown_report(metrics, info, staging, scatter_paths=pictures) if "markdown" in selected else None
        if output_root is None:
            for name in ("pictures", "report"):
                destination = run_dir / name
                if destination.exists(): shutil.rmtree(destination)
                os.replace(staging / name, destination)
            shutil.rmtree(staging)
            pictures = [run_dir / "pictures" / p.name for p in pictures]
            html_path = run_dir / "report" / "report.html" if html_path else None
            markdown_path = run_dir / "report" / "report.md" if markdown_path else None
        else:
            if root.exists(): raise ReportServiceError(f"报告输出根已存在，拒绝覆盖：{root}")
            os.replace(staging, root)
            pictures = [root / "pictures" / p.name for p in pictures]
            html_path = root / "report" / "report.html" if html_path else None
            markdown_path = root / "report" / "report.md" if markdown_path else None
        return ReportResult(str(manifest["run_id"]), run_dir, tuple(pictures), html_path, markdown_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def rebuild_schema2_summary_report(
    run_dirs: Sequence[Path | str], *, output_root: Path | str,
    formats: Sequence[str] | None = None,
) -> ReportResult:
    """Atomically render one task-level report spanning completed combinations.

    Unlike :func:`rebuild_schema2_report`, this is the normal schema-2 entry
    point.  It reads every published run independently, then replaces only the
    task root's ``pictures/`` and ``report/`` directories after all rendering
    succeeds.  No per-combination report is emitted.
    """
    directories = [Path(value).resolve() for value in run_dirs]
    if not directories:
        raise ReportServiceError("没有完成的训练组合可用于生成最终报告")
    records: list[tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]] = []
    source: Mapping[str, Any] | None = None
    for run_dir in directories:
        manifest = _load_yaml(run_dir / "run_manifest.yaml")
        if manifest.get("kind") != "yonod_training_run" or manifest.get("schema_version") != "1.0" or manifest.get("status") != "complete":
            raise ReportServiceError(f"仅可汇总已完成训练结果：{run_dir}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ReportServiceError(f"run_manifest.outputs 缺失：{run_dir}")
        predictions_path = _required_file(run_dir, outputs, "predictions")
        folds_path = _required_file(run_dir, outputs, "fold_metrics")
        source_path = _required_file(run_dir, outputs, "source_config")
        effective_path = _required_file(run_dir, outputs, "effective_config")
        if manifest.get("source_config_sha256") != _sha256(source_path) or manifest.get("effective_config_sha256") != _sha256(effective_path):
            raise ReportServiceError(f"配置快照哈希不匹配：{run_dir}")
        predictions, folds = pd.read_csv(predictions_path), pd.read_csv(folds_path)
        _validate_predictions(predictions, manifest)
        if folds.empty or not {"repeat", "fold"}.issubset(folds.columns):
            raise ReportServiceError(f"fold_metrics.csv 不完整：{run_dir}")
        candidate_source = _load_yaml(source_path)
        if source is None:
            source = candidate_source
        records.append((manifest, predictions, folds))
    assert source is not None
    selected = [str(v).strip().lower() for v in (formats if formats is not None else ((source.get("outputs") or {}).get("report_formats", ["html", "markdown"]))) ]
    aliases = {"html": "html", "markdown": "markdown", "md": "markdown"}
    try:
        selected = [aliases[item] for item in selected]
    except KeyError as exc:
        raise ReportServiceError(f"不支持 outputs.report_formats：{exc.args[0]!r}") from exc
    if len(set(selected)) != len(selected):
        raise ReportServiceError("outputs.report_formats 不能重复")
    root = Path(output_root).resolve()
    if not selected:
        return ReportResult("summary", root, (), None, None)
    staging = root / f".report-{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    try:
        pictures_dir = staging / "pictures"
        pictures_dir.mkdir()
        pictures: list[Path] = []
        metric_frames: list[pd.DataFrame] = []
        for manifest, predictions, folds in records:
            metric_frames.append(_metrics(predictions, folds, manifest))
            for repeat, points in predictions.groupby("repeat", sort=True):
                picture = pictures_dir / f"scatter_{_safe_piece(str(manifest['feature_id']))}_{_safe_piece(str(manifest['model']))}_repeat-{_safe_piece(str(repeat))}.png"
                _plot_repeat(points, manifest, picture)
                if picture.stat().st_size < 128:
                    raise ReportServiceError(f"散点图为空：{picture.name}")
                pictures.append(picture)
        metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
        info = {
            "task_name": source.get("project_name", "YONOD"), "project_folder": str(root),
            "csv_path": "<stored training results>", "n_samples": records[0][0].get("n_samples"),
            "label_col": records[0][0].get("label_column"), "n_combinations": len(records),
            "dataset_citation": metadata.get("doi"),
            "dataset_url": metadata.get("source_url", metadata.get("repo_url")),
            "dataset_notes": metadata.get("notes", metadata.get("source_notes")),
        }
        metrics = pd.concat(metric_frames, ignore_index=True)
        html_path = generate_report(metrics, info, staging, scatter_paths=pictures) if "html" in selected else None
        markdown_path = generate_markdown_report(metrics, info, staging, scatter_paths=pictures) if "markdown" in selected else None
        for name in ("pictures", "report"):
            destination = root / name
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging / name, destination)
        shutil.rmtree(staging)
        return ReportResult("summary", root, tuple(root / "pictures" / item.name for item in pictures), root / "report" / "report.html" if html_path else None, root / "report" / "report.md" if markdown_path else None)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
