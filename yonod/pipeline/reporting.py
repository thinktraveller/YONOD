"""Read-only report post-processing for schema-2 training results.

This module deliberately consumes only a completed ``run_manifest.yaml`` and
the files named by it.  It neither imports descriptor generators nor calls a
model ``fit`` method, so it is also the implementation behind offline rebuild.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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


def _numeric_report_rows(run_dir: Path, folds: pd.DataFrame, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read verified per-fold numeric state; never recompute it from source CSV."""
    contract = manifest.get("numeric_contract")
    if not contract:
        return []
    entries = contract.get("columns", [])
    if not entries or "transform_audit" not in folds:
        raise ReportServiceError("数值运行缺少处理契约或折级审计")
    rows: list[dict[str, Any]] = []
    for _, fold_row in folds.iterrows():
        repeat, fold = int(fold_row["repeat"]), int(fold_row["fold"])
        audit = json.loads(fold_row["transform_audit"]).get("numeric", {})
        state_path = run_dir / "fold_transforms" / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / "numeric_conditions.yaml"
        if not state_path.is_file() or audit.get("state_sha256") != _sha256(state_path):
            raise ReportServiceError(f"repeat={repeat}/fold={fold} 数值状态缺失或哈希不匹配")
        state = _load_yaml(state_path)
        if state.get("contract") != contract or len(state.get("columns", [])) != len(entries):
            raise ReportServiceError(f"repeat={repeat}/fold={fold} 数值状态与契约不一致")
        if audit.get("fit_sample_ids_sha256") != state.get("fit_sample_ids_sha256"):
            raise ReportServiceError(f"repeat={repeat}/fold={fold} 数值拟合行身份不一致")
        for entry, fitted in zip(entries, state["columns"]):
            if fitted.get("source") != entry.get("source") or fitted.get("name") != entry.get("name"):
                raise ReportServiceError(f"repeat={repeat}/fold={fold} 数值列顺序不一致")
            rows.append({
                "run": f"{manifest['feature_id']} × {manifest['model']}",
                "repeat": repeat, "fold": fold,
                "source": entry["source"], "name": entry["name"],
                "unit": (entry.get("unit") or {}).get("target", "dimensionless"),
                "missing": entry["missing"]["strategy"], "scaling": entry["scaling"],
                "train_missing_rows": fitted["train_missing_rows"],
                "constant_train": fitted["constant_train"],
                "numeric_dimension": int(audit["dimension"]),
                "input_dimension": int(fold_row["feature_dim"]),
            })
    return rows


def _hpo_report_rows(
    run_dir: Path, predictions: pd.DataFrame, folds: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]] | None:
    """Read verified, exported study evidence without opening its SQLite DB."""
    hpo = manifest.get("hpo")
    if not hpo:
        return None
    audits = hpo.get("outer_folds") if isinstance(hpo, Mapping) else None
    if not isinstance(audits, list) or len(audits) != len(folds):
        raise ReportServiceError("HPO 外层折审计与 fold_metrics 行数不一致")
    task_root = run_dir.parent.parent.resolve()
    studies: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    outer: list[dict[str, Any]] = []
    from scipy.stats import kendalltau
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for audit in audits:
        repeat, fold = int(audit["repeat"]), int(audit["fold"])
        matching_fold = folds.loc[(folds["repeat"] == repeat) & (folds["fold"] == fold)]
        points = predictions.loc[(predictions["repeat"] == repeat) & (predictions["fold"] == fold)]
        if len(matching_fold) != 1 or points.empty:
            raise ReportServiceError(f"HPO repeat={repeat}/fold={fold} 缺少唯一折指标或预测")
        study_dir = (task_root / str(audit["study_dir"])).resolve()
        if task_root not in study_dir.parents or study_dir.parent.parent != task_root / "hpo":
            raise ReportServiceError("HPO study 路径超出任务 hpo 根目录")
        manifest_path, trials_path = study_dir / "study_manifest.json", study_dir / "trials.json"
        if not manifest_path.is_file() or not trials_path.is_file():
            raise ReportServiceError(f"HPO study 证据缺失：{study_dir}")
        if _sha256(manifest_path) != audit.get("study_manifest_sha256") or _sha256(trials_path) != audit.get("study_trials_sha256"):
            raise ReportServiceError(f"HPO study 证据哈希不匹配：{study_dir}")
        try:
            study_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshots = json.loads(trials_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReportServiceError(f"HPO study JSON 无法读取：{study_dir}") from exc
        if study_manifest.get("study_id") != audit.get("study_id") or not isinstance(snapshots, list):
            raise ReportServiceError("HPO study 身份或 trial 导出无效")
        if len(snapshots) != int(audit["attempted_trials"]):
            raise ReportServiceError("HPO trial 尝试数与审计不一致")
        complete = [item for item in snapshots if item.get("state") == "COMPLETE"]
        failed = [item for item in snapshots if item.get("state") == "FAIL"]
        if len(complete) != int(audit["complete_trials"]) or len(failed) != int(audit["failed_trials"]):
            raise ReportServiceError("HPO trial 状态数与审计不一致")
        best = next((item for item in complete if item.get("number") == audit["best_trial_number"]), None)
        if best is None or not math.isclose(float(best["value"]), float(audit["best_value"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ReportServiceError("HPO 最佳 trial 与持久审计不一致")
        fold_row = matching_fold.iloc[0]
        truth, predicted = points["y_true"].to_numpy(dtype=float), points["y_pred"].to_numpy(dtype=float)
        tau = float("nan")
        tau_reason = ""
        if len(truth) < 2 or np.unique(truth).size < 2 or np.unique(predicted).size < 2:
            tau_reason = "样本不足或真实/预测值为常量"
        else:
            tau = float(kendalltau(truth, predicted).statistic)
            if not math.isfinite(tau):
                tau_reason = "Kendall τ 不可定义"
        label = f"{manifest['feature_id']} × {manifest['model']}"
        studies.append({
            "combination": label, "repeat": repeat, "fold": fold,
            "study_id": audit["study_id"], "status": "complete",
            "attempted": len(snapshots), "complete": len(complete), "failed": len(failed),
            "best_trial": audit["best_trial_number"], "best_inner": float(audit["best_value"]),
            "objective": hpo["declaration"]["objective"]["metric"],
            "stop_reason": audit["stop_reason"],
            "search_time_s": float(audit["active_search_time_s"]),
            "best_parameters": audit["effective_estimator_parameters"],
            "parameter_sources": audit["parameter_sources"],
            "trial_export": (Path("hpo") / study_dir.parent.name / study_dir.name / "trials.json").as_posix(),
        })
        best_so_far: float | None = None
        direction = hpo["declaration"]["objective"]["direction"]
        for trial in snapshots:
            attrs = trial.get("user_attrs") or {}
            fold_scores = [item.get("score") for item in attrs.get("folds", [])]
            value = trial.get("value") if trial.get("state") == "COMPLETE" else None
            if value is not None:
                best_so_far = float(value) if best_so_far is None else (
                    min(best_so_far, float(value)) if direction == "minimize" else max(best_so_far, float(value))
                )
            trials.append({
                "combination": label, "repeat": repeat, "fold": fold,
                "trial": trial.get("number"), "state": trial.get("state"),
                "objective": value, "best_so_far": best_so_far,
                "inner_fold_scores": fold_scores,
                "parameters": trial.get("params", {}),
                "failure": (attrs.get("failure") or {}).get("reason", ""),
            })
        outer.append({
            "combination": label, "repeat": repeat, "fold": fold,
            "n_valid": len(points),
            "r2": float(r2_score(truth, predicted)) if len(truth) >= 2 and np.unique(truth).size > 1 else float("nan"),
            "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
            "mae": float(mean_absolute_error(truth, predicted)),
            "kendall_tau": tau, "kendall_tau_reason": tau_reason,
            "outer_train_time_s": float(fold_row.get("model_train_time_s", float("nan"))),
            "outer_predict_time_s": float(fold_row.get("model_predict_time_s", float("nan"))),
        })
    final_rows: list[dict[str, Any]] = []
    final_info = hpo.get("final_model")
    final_enabled = bool(hpo["declaration"]["final_model"]["enabled"])
    if final_enabled:
        if not isinstance(final_info, Mapping) or final_info.get("purpose") != "final_model":
            raise ReportServiceError("启用的 HPO 最终模型缺少独立完成证据")
        final_study = final_info.get("study")
        if not isinstance(final_study, Mapping) or final_study.get("purpose") != "final_model":
            raise ReportServiceError("最终模型 study 用途不匹配")
        study_dir = (task_root / str(final_study["study_dir"])).resolve()
        if study_dir.parent.parent != task_root / "hpo":
            raise ReportServiceError("最终模型 study 路径越界")
        for filename, digest_key in (("study_manifest.json", "study_manifest_sha256"), ("trials.json", "study_trials_sha256")):
            path = study_dir / filename
            if not path.is_file() or _sha256(path) != final_study.get(digest_key):
                raise ReportServiceError("最终模型 study 证据缺失或哈希不匹配")
        bundle_path = (run_dir / str(final_info["bundle_path"])).resolve()
        if run_dir not in bundle_path.parents or not bundle_path.is_file() or _sha256(bundle_path) != final_info.get("bundle_sha256"):
            raise ReportServiceError("最终模型包缺失或哈希不匹配")
        final_rows.append({
            "combination": f"{manifest['feature_id']} × {manifest['model']}",
            "status": "complete",
            "study_id": final_study["study_id"],
            "n_development_rows": final_info["n_development_rows"],
            "attempted": final_study["attempted_trials"],
            "best_trial": final_study["best_trial_number"],
            "best_inner": final_study["best_value"],
            "best_parameters": final_study["effective_estimator_parameters"],
            "search_time_s": final_study["active_search_time_s"],
            "full_refit_time_s": final_info["train_time_s"],
            "independent_test_score": final_info.get("independent_test_score"),
            "bundle_path": final_info["bundle_path"],
        })
    return {"studies": studies, "trials": trials, "outer": outer,
            "final": final_rows, "final_enabled": final_enabled}


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


def _diagnostic_rows(run_dir: Path, predictions: pd.DataFrame, folds: pd.DataFrame, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    from yonod.diagnostics import diagnostic_table_row, read_diagnostic

    references = (manifest.get("diagnostics") or {}).get("folds") or []
    by_fold = {(int(item["repeat"]), int(item["fold"])): item for item in references}
    rows: list[dict[str, Any]] = []
    for _, fold_row in folds.iterrows():
        repeat, fold = int(fold_row["repeat"]), int(fold_row["fold"])
        record = read_diagnostic(run_dir, by_fold.get((repeat, fold)), identity={
            "run_id": manifest["run_id"], "feature_id": manifest["feature_id"],
            "model": manifest["model"], "repeat": repeat, "fold": fold,
        })
        row = diagnostic_table_row(record, feature=str(manifest["feature_id"]),
                                   model=str(manifest["model"]), repeat=repeat, fold=fold)
        points = predictions.loc[(predictions["repeat"] == repeat) & (predictions["fold"] == fold)]
        if not points.empty:
            residual = points["y_true"].to_numpy(dtype=float) - points["y_pred"].to_numpy(dtype=float)
            worst = int(np.argmax(np.abs(residual)))
            row.update({"worst_sample_id": str(points.iloc[worst]["sample_id"]),
                        "worst_residual_y_true_minus_pred": float(residual[worst]),
                        "heldout_target_min": float(points["y_true"].min()),
                        "heldout_target_max": float(points["y_true"].max())})
        rows.append(row)
    return rows


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
        info = {"task_name": source.get("project_name", manifest["run_id"]), "project_folder": str(root), "csv_path": "<stored training result>", "n_samples": manifest.get("n_samples"), "label_col": manifest.get("label_column"), "n_combinations": 1, "run_id": manifest["run_id"], "artifact_id": manifest["artifact_id"], "split_identity": manifest["split_identity"], "model_request": manifest.get("model_request"), "dataset_citation": metadata.get("doi"), "dataset_url": metadata.get("source_url", metadata.get("repo_url")), "dataset_notes": metadata.get("notes", metadata.get("source_notes")), "numeric_audit_rows": _numeric_report_rows(run_dir, folds, manifest), "hpo_report": _hpo_report_rows(run_dir, predictions, folds, manifest), "diagnostic_rows": _diagnostic_rows(run_dir, predictions, folds, manifest)}
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
            "numeric_audit_rows": [row for (manifest, _, folds), run_dir in zip(records, directories) for row in _numeric_report_rows(run_dir, folds, manifest)],
            "diagnostic_rows": [row for (manifest, predictions, folds), run_dir in zip(records, directories) for row in _diagnostic_rows(run_dir, predictions, folds, manifest)],
        }
        hpo_sections = [section for (manifest, predictions, folds), run_dir in zip(records, directories)
                        if (section := _hpo_report_rows(run_dir, predictions, folds, manifest)) is not None]
        if hpo_sections:
            info["hpo_report"] = {
                key: [item for section in hpo_sections for item in section[key]]
                for key in ("studies", "trials", "outer", "final")
            }
            info["hpo_report"]["final_enabled"] = any(section["final_enabled"] for section in hpo_sections)
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


def rebuild_schema2_hpo_status_report(
    task_root: Path | str, failed_manifests: Sequence[Path | str] | None = None,
    formats: Sequence[str] | None = None,
) -> ReportResult:
    """Render incomplete/failed HPO evidence without treating it as OOF.

    This reads failure manifests and their hash-bound study exports. It never
    opens SQLite, fits a model, or changes a source artefact. A completed task
    report, if present, remains separate from these status files.
    """
    root = Path(task_root).resolve()
    paths = ([Path(path).resolve() for path in failed_manifests] if failed_manifests is not None
             else sorted(root.glob("runs/*/failed_attempts/*/run_manifest.yaml")))
    if not paths:
        raise ReportServiceError("没有 HPO incomplete/failed manifest 可用于状态报告")
    aliases = {"html": "html", "markdown": "markdown", "md": "markdown"}
    try:
        selected = [aliases[str(value).strip().lower()] for value in (formats if formats is not None else ("html", "markdown"))]
    except KeyError as exc:
        raise ReportServiceError(f"不支持的 HPO 状态报告格式：{exc.args[0]}") from exc
    if len(set(selected)) != len(selected):
        raise ReportServiceError("HPO 状态报告格式不能重复")
    records: list[dict[str, Any]] = []
    studies: list[dict[str, Any]] = []
    for path in paths:
        if root not in path.parents or path.name != "run_manifest.yaml":
            raise ReportServiceError(f"HPO 失败 manifest 越界：{path}")
        manifest = _load_yaml(path)
        if manifest.get("kind") != "yonod_training_run" or manifest.get("status") not in {"incomplete", "failed"}:
            raise ReportServiceError(f"HPO 状态报告只读取 incomplete/failed 训练 manifest：{path}")
        hpo = manifest.get("hpo")
        if not isinstance(hpo, Mapping) or not (hpo.get("declaration") or {}).get("enabled"):
            continue
        outputs = manifest.get("outputs") or {}
        for key, digest_key in (("source_config", "source_config_sha256"), ("effective_config", "effective_config_sha256")):
            source = _required_file(path.parent, outputs, key)
            if _sha256(source) != manifest.get(digest_key):
                raise ReportServiceError(f"HPO 失败配置快照哈希不匹配：{path}")
        records.append({
            "combination": f"{manifest['feature_id']} × {manifest['model']}",
            "status": manifest["status"],
            "completed": int(hpo.get("completed_outer_folds", 0)),
            "expected": int(hpo.get("expected_outer_folds", manifest.get("n_folds", 0))),
            "reason": (manifest.get("failure") or {}).get("reason", ""),
            "manifest": path.relative_to(root).as_posix(),
        })
        for audit in hpo.get("outer_folds", []):
            study_dir = (root / str(audit["study_dir"])).resolve()
            if study_dir.parent.parent != root / "hpo":
                raise ReportServiceError("HPO 失败 study 路径越界")
            study_manifest = study_dir / "study_manifest.json"
            trials_path = study_dir / "trials.json"
            if not study_manifest.is_file() or not trials_path.is_file():
                raise ReportServiceError("HPO 失败 study 证据缺失")
            if _sha256(study_manifest) != audit.get("study_manifest_sha256") or _sha256(trials_path) != audit.get("study_trials_sha256"):
                raise ReportServiceError("HPO 失败 study 证据哈希不匹配")
            studies.append({
                "combination": records[-1]["combination"],
                "repeat": audit["repeat"], "fold": audit["fold"],
                "study_id": audit["study_id"], "attempted": audit["attempted_trials"],
                "complete": audit["complete_trials"], "failed": audit["failed_trials"],
                "best_inner": audit["best_value"], "stop_reason": audit["stop_reason"],
                "search_time_s": audit["active_search_time_s"],
            })
    if not records:
        raise ReportServiceError("失败 manifest 中没有启用的 HPO 记录")

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    html_path = markdown_path = None
    if "html" in selected:
        def html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
            if not rows:
                return "<p>N/A：尚无已核验的完整外层搜索折。</p>"
            heads = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
            body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, 'N/A')))}</td>" for key, _ in columns) + "</tr>" for row in rows)
            return f"<table border='1'><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>"
        content = (
            "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>YONOD HPO 状态</title></head><body>"
            "<h1>HPO 未完成与失败状态</h1><p>此报告不含完整外层 OOF，也不将内层搜索分数当作外部评估。"
            "仅读取已保存且哈希核验的证据；未发布外层折不计为完成。</p>"
            f"<p>生成时间：{html.escape(timestamp)}</p>"
            + html_table(records, (("combination", "组合"), ("status", "状态"), ("completed", "完成外层折"),
                                   ("expected", "预期外层折"), ("reason", "原因"), ("manifest", "manifest")))
            + "<h2>已核验的搜索折</h2>"
            + html_table(studies, (("combination", "组合"), ("repeat", "repeat"), ("fold", "fold"),
                                   ("study_id", "study"), ("attempted", "尝试"), ("complete", "成功"),
                                   ("failed", "失败"), ("best_inner", "最佳内层均分"),
                                   ("stop_reason", "停止原因"), ("search_time_s", "搜索活跃秒")))
            + "</body></html>"
        )
        html_path = report_dir / "hpo_status.html"
        temporary = report_dir / f".hpo_status-{uuid.uuid4().hex}.html.tmp"
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, html_path)
    if "markdown" in selected:
        lines = ["# YONOD HPO 未完成与失败状态", "", "此报告不含完整外层 OOF；内层搜索分数不是外部评估。", "", f"生成时间：{timestamp}", "",
                 "| 组合 | 状态 | 完成外层折 | 预期外层折 | 原因 | manifest |", "|---|---|---:|---:|---|---|"]
        lines += ["| " + " | ".join(cell(row[key]) for key in ("combination", "status", "completed", "expected", "reason", "manifest")) + " |" for row in records]
        lines += ["", "## 已核验的搜索折", ""]
        if studies:
            lines += ["| 组合 | repeat | fold | study | 尝试 | 成功 | 失败 | 最佳内层均分 | 停止原因 | 搜索活跃秒 |",
                      "|---|---:|---:|---|---:|---:|---:|---:|---|---:|"]
            lines += ["| " + " | ".join(cell(row[key]) for key in ("combination", "repeat", "fold", "study_id", "attempted", "complete", "failed", "best_inner", "stop_reason", "search_time_s")) + " |" for row in studies]
        else:
            lines.append("N/A：尚无已核验的完整外层搜索折。")
        markdown_path = report_dir / "hpo_status.md"
        temporary = report_dir / f".hpo_status-{uuid.uuid4().hex}.md.tmp"
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, markdown_path)
    return ReportResult("hpo_status", root, (), html_path, markdown_path)
