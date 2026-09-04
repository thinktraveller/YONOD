"""Standalone HTML/Markdown reports for an already completed benchmark run."""

from __future__ import annotations

import html
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    COMBINATION_TIME_SUMMARY_COLUMNS,
    DIMENSION_TIME_SUMMARY_COLUMNS,
    summarize_combination_times,
    write_combination_time_summary,
    write_dimension_time_summaries,
)
from .layout import BenchmarkOutputLayout, resolve_benchmark_output_layout


class BenchmarkReportError(RuntimeError):
    """Raised when a report cannot be regenerated from auditable artefacts."""


@dataclass(frozen=True)
class BenchmarkReportResult:
    html_path: Path
    markdown_path: Path
    figures: tuple[Path, ...]


_REQUIRED_TABLES = (
    "fold_metrics", "metric_exclusions", "completeness", "combination_summary",
    "model_comparisons", "descriptor_comparisons", "comparison_exclusions",
    "model_tukey_hsd", "descriptor_tukey_hsd",
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReportError("无法读取报告输入 JSON：{0}".format(path)) from exc
    if not isinstance(value, dict):
        raise BenchmarkReportError("报告输入 JSON 根节点必须是对象：{0}".format(path))
    return value


def _load_tables(layout: BenchmarkOutputLayout) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name in _REQUIRED_TABLES:
        path = layout.metrics / (name + ".parquet")
        if not path.is_file():
            raise BenchmarkReportError(
                "缺少派生指标表 {0}。请先从预测库运行指标重建，而不是重新训练模型。".format(path)
            )
        tables[name] = pd.read_parquet(path)
    return tables


def _load_or_rebuild_time_summaries(
    layout: BenchmarkOutputLayout,
    tables: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Load all timing tables, deriving them from saved fold artefacts if needed.

    Runs generated before the timing feature did not have this derived table.
    Rebuilding it here is intentionally a metrics-only operation: it reads
    existing fold metrics and completeness records and never invokes model
    fitting.
    """
    path = layout.metrics / "combination_time_summary.parquet"
    combination_summary: Optional[pd.DataFrame] = None
    combination_rebuilt = False
    if path.is_file():
        try:
            summary = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise BenchmarkReportError("无法读取组合耗时汇总表：{0}".format(path)) from exc
        if set(COMBINATION_TIME_SUMMARY_COLUMNS).issubset(summary.columns):
            combination_summary = summary.loc[:, COMBINATION_TIME_SUMMARY_COLUMNS]
    if combination_summary is None:
        combination_summary = summarize_combination_times(tables["fold_metrics"], tables["completeness"])
        write_combination_time_summary(layout.run_dir, combination_summary)
        combination_rebuilt = True

    result = {"combination_time_summary": combination_summary}
    expected_columns = set(DIMENSION_TIME_SUMMARY_COLUMNS)
    need_rebuild = combination_rebuilt
    for table_name in ("descriptor_time_summary", "model_time_summary"):
        table_path = layout.metrics / (table_name + ".parquet")
        if not table_path.is_file():
            need_rebuild = True
            continue
        try:
            table = pd.read_parquet(table_path)
        except (OSError, ValueError) as exc:
            raise BenchmarkReportError("无法读取维度耗时汇总表：{0}".format(table_path)) from exc
        if expected_columns.issubset(table.columns):
            result[table_name] = table.loc[:, DIMENSION_TIME_SUMMARY_COLUMNS]
        else:
            need_rebuild = True
    if need_rebuild:
        write_dimension_time_summaries(layout.run_dir, combination_summary)
        for table_name in ("descriptor_time_summary", "model_time_summary"):
            result[table_name] = pd.read_parquet(layout.metrics / (table_name + ".parquet"))
    return result


def _split_audit(split_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"split_id", "repeat", "fold", "role", "sample_id", "group_id"}
    if missing := required.difference(split_manifest.columns):
        raise BenchmarkReportError("split manifest 缺少字段：{0}".format(sorted(missing)))
    rows = []
    for (split_id, repeat, fold), part in split_manifest.groupby(["split_id", "repeat", "fold"], sort=True):
        train = part[part["role"] == "train"]
        valid = part[part["role"] == "valid"]
        train_groups, valid_groups = set(train["group_id"].astype(str)), set(valid["group_id"].astype(str))
        group_sizes = part.groupby("group_id").size()
        rows.append({
            "split_id": str(split_id), "repeat": int(repeat), "fold": int(fold),
            "n_train": int(len(train)), "n_valid": int(len(valid)),
            "n_train_groups": int(len(train_groups)), "n_valid_groups": int(len(valid_groups)),
            "max_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
            "group_leakage": bool(train_groups.intersection(valid_groups)),
        })
    return pd.DataFrame.from_records(rows)


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    if isinstance(value, (float, np.floating)):
        return "{0:.{1}f}".format(float(value), digits)
    return str(value)


def _format_duration(value: Any) -> str:
    """Render a finite duration without changing the raw seconds in tables."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(seconds) or seconds < 0.0:
        return "—"
    if seconds < 60.0:
        return "{0:.2f}s".format(seconds)
    return "{0}m {1:.1f}s".format(int(seconds // 60), seconds % 60)


def _configure_chinese_matplotlib(plt: Any) -> None:
    """Prefer an installed CJK font so generated Chinese labels stay legible."""
    try:
        from matplotlib import font_manager
        installed = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - font discovery is platform-specific
        return
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if candidate in installed:
            current = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = [candidate] + [name for name in current if name != candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _table_html(frame: pd.DataFrame, *, max_rows: int = 150) -> str:
    if frame.empty:
        return "<p class='empty'>无可展示记录；请查看相应排除/完整性表。</p>"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(_format)
    note = ""
    if len(frame) > max_rows:
        note = "<p class='note'>表共有 {0} 行；此处显示前 {1} 行，完整输入表位于 metrics/。</p>".format(len(frame), max_rows)
    return shown.to_html(index=False, escape=True, classes="data-table") + note


def _table_markdown(frame: pd.DataFrame, *, max_rows: int = 150) -> str:
    if frame.empty:
        return "无可展示记录；请查看相应排除/完整性表。"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(_format)
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [cell(column) for column in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for values in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    text = "\n".join(lines)
    if len(frame) > max_rows:
        text += "\n\n> 表共有 {0} 行；此处显示前 {1} 行，完整输入表位于 `metrics/`。".format(len(frame), max_rows)
    return text


def _performance_matrix(summary: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["descriptor", "model", "r2", "rmse", "mae", "complete", "expected_folds", "valid_metric_folds"])
    values = summary.pivot_table(index=["descriptor", "model"], columns="metric", values="mean", aggfunc="first").reset_index()
    complete = completeness[["descriptor", "model", "is_complete", "expected_folds", "valid_metric_folds"]].copy()
    complete = complete.rename(columns={"is_complete": "complete"})
    return values.merge(complete, on=["descriptor", "model"], how="left").sort_values(["descriptor", "model"])


def _protocol_alignment_table(
    config: Mapping[str, Any], split_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Render auditable protocol facts without claiming numerical reproduction."""
    protocol = config.get("reproduction_protocol", {})
    if not isinstance(protocol, Mapping) or not protocol:
        return pd.DataFrame([{
            "protocol": "未声明论文协议",
            "status": "通用 benchmark；不可据此宣称论文流程对齐",
        }])
    grouping = config.get("grouping", {})
    cv = config.get("cv", {})
    feature_sets = config.get("feature_sets", [])
    seeds = sorted(pd.to_numeric(split_manifest.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    return pd.DataFrame([{
        "protocol": protocol.get("name", "未命名"),
        "literature_doi": protocol.get("literature_doi", "未声明"),
        "split_strategy": grouping.get("strategy", "未声明"),
        "cv": json.dumps(cv, ensure_ascii=False, sort_keys=True),
        "manifest_seeds": ", ".join(str(seed) for seed in seeds),
        "source_row_index_recorded": "source_row_index" in split_manifest.columns,
        "feature_sets": json.dumps(feature_sets, ensure_ascii=False, sort_keys=True),
        "status": "流程参数证据；不等同于论文数值复现",
    }])


def _comparison_conclusions(frame: pd.DataFrame, dimension_label: str) -> list[str]:
    if frame.empty:
        return ["没有可比较的完整组合；这不是性能排名结论。"]
    lines = []
    for _, row in frame.head(30).iterrows():
        lines.append(
            "固定{0} `{1}`、指标 {2}（{3}）时，`{4}` 相对 `{5}` 的折级均值差为 {6}，"
            "95% CI [{7}, {8}]，Holm 校正 p={9}；结论：`{10}`。".format(
                dimension_label, row.get("fixed_value", "—"), row.get("metric", "—"),
                row.get("metric_direction", "—"), row.get("left", "—"), row.get("right", "—"),
                _format(row.get("effect_mean_difference")), _format(row.get("difference_ci_low")),
                _format(row.get("difference_ci_high")), _format(row.get("p_value_adjusted")),
                row.get("conclusion", "—"),
            )
        )
    return lines


def _state_summary(layout: BenchmarkOutputLayout) -> pd.DataFrame:
    state_path = layout.state / "tasks.sqlite"
    if not state_path.is_file():
        return pd.DataFrame([{"state_store": "not_created", "count": 0}])
    try:
        with sqlite3.connect(str(state_path)) as connection:
            return pd.read_sql_query("SELECT status AS state_store, COUNT(*) AS count FROM tasks GROUP BY status", connection)
    except sqlite3.Error as exc:
        return pd.DataFrame([{"state_store": "unreadable", "count": 0, "detail": str(exc)}])


def _disk_usage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_figures(layout: BenchmarkOutputLayout, fold_metrics: pd.DataFrame) -> tuple[Path, ...]:
    """Generate stability/OOF figures from saved tables and shards only."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise BenchmarkReportError("需要 matplotlib 才能生成 benchmark 稳定性图") from exc
    pictures_dir = layout.pictures
    pictures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not fold_metrics.empty:
        labels = (fold_metrics["descriptor"].astype(str) + " × " + fold_metrics["model"].astype(str)).unique().tolist()
        figure, axes = plt.subplots(1, 3, figsize=(max(9, len(labels) * 1.6), 4.8), constrained_layout=True)
        for axis, metric in zip(axes, ("r2", "rmse", "mae")):
            data = [fold_metrics.loc[(fold_metrics["descriptor"].astype(str) + " × " + fold_metrics["model"].astype(str)) == label, metric].to_numpy() for label in labels]
            axis.boxplot(data, tick_labels=labels, showmeans=True)
            axis.set_title(metric.upper() + " across folds")
            axis.tick_params(axis="x", rotation=45, labelsize=8)
        stability = pictures_dir / "fold_metric_stability.png"
        figure.savefig(stability, dpi=150)
        plt.close(figure)
        paths.append(stability)

    prediction_frames = []
    for path_text in fold_metrics.get("prediction_path", pd.Series(dtype=str)).dropna().unique():
        path = Path(str(path_text))
        if path.is_file():
            frame = pd.read_parquet(path)
            prediction_frames.append(frame[["descriptor", "model", "y_true", "y_pred"]])
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        axes[0].scatter(predictions["y_true"], predictions["y_pred"], s=12, alpha=0.65)
        lower = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
        upper = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
        axes[0].plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        axes[0].set(xlabel="y_true", ylabel="y_pred", title="Saved fold predictions")
        residuals = predictions["y_pred"] - predictions["y_true"]
        axes[1].hist(residuals, bins=min(30, max(5, int(np.sqrt(len(residuals))))), color="#2563eb", alpha=0.8)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set(xlabel="y_pred − y_true", ylabel="count", title="Residual distribution")
        scatter = pictures_dir / "prediction_residuals.png"
        figure.savefig(scatter, dpi=150)
        plt.close(figure)
        paths.append(scatter)
    return tuple(paths)


def _write_time_figure(
    layout: BenchmarkOutputLayout,
    summary: pd.DataFrame,
    run_id: str,
    *,
    filename: str,
    title: str,
    label_for_row: Any,
    coverage_for_row: Any,
) -> Path:
    """Render one non-overlapping view of the auditable timing summaries."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise BenchmarkReportError("需要 matplotlib 才能生成组合建模耗时柱状图") from exc

    _configure_chinese_matplotlib(plt)

    pictures_dir = layout.pictures
    pictures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = pictures_dir / filename
    comparable_mask = summary["is_time_comparable"].fillna(False).astype(bool)
    comparable = summary.loc[comparable_mask].copy()
    noncomparable = summary.loc[~comparable_mask].copy()
    comparable = comparable.sort_values("total_model_time_s", ascending=False, kind="stable")
    display = pd.concat([comparable, noncomparable], ignore_index=True)
    height = max(3.8, 0.62 * max(1, len(display)) + 1.1)
    # Reserve a footer band for the run/CV comparability note.  Letting the
    # automatic layout use the full canvas makes that note collide with the
    # x-axis label on short smoke charts.
    figure, axis = plt.subplots(figsize=(10.5, height))

    if display.empty:
        axis.set_axis_off()
        axis.text(0.5, 0.5, "没有可用的组合级建模耗时记录", ha="center", va="center", color="#56657a")
    else:
        positions = np.arange(len(display))
        labels = [str(label_for_row(row)) for _, row in display.iterrows()]
        for position, (_, row) in zip(positions, display.iterrows()):
            if bool(row["is_time_comparable"]):
                train = float(row["total_train_time_s"])
                predict = float(row["total_predict_time_s"])
                total = float(row["total_model_time_s"])
                axis.barh(position, train, color="#2563eb", label="累计 CV 训练" if position == 0 else None)
                axis.barh(position, predict, left=train, color="#f59e0b", label="累计 CV 预测" if position == 0 else None)
                axis.annotate(
                    _format_duration(total), xy=(total, position), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8,
                )
            else:
                status = str(row.get("time_status", "时间不可比较"))
                detail = str(row.get("time_status_detail", "")).strip()
                completed = str(coverage_for_row(row))
                text = "不可比较（{0}; {1}{2}）".format(
                    completed, status, "; " + detail if detail else "",
                )
                axis.text(0.01, position, text, transform=axis.get_yaxis_transform(), va="center", color="#6b7280", fontsize=8)
        max_total = float(comparable["total_model_time_s"].max()) if not comparable.empty else 0.0
        # Keep very fast smoke runs legible instead of stretching a 0.09 s
        # comparison across a one-second axis; the small floor still renders
        # an all-zero, but valid, timing fixture safely.
        axis.set_xlim(0.0, max(0.01, max_total * 1.20))
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("累计 CV 建模时间（秒，训练 + 预测）")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        if not comparable.empty:
            axis.legend(loc="lower right")
    figure.text(
        0.01, 0.01,
        "run_id={0}；仅可在相同硬件、线程设置与 CV 配置下横向比较。".format(run_id),
        fontsize=7, color="#56657a",
    )
    figure.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.16)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def _write_combination_time_figure(layout: BenchmarkOutputLayout, summary: pd.DataFrame, run_id: str) -> Path:
    """Render the descriptor × model comparison chart."""
    return _write_time_figure(
        layout, summary, run_id,
        filename="combination_modeling_time.png",
        title="描述符 × 模型：组合级建模耗时",
        label_for_row=lambda row: "{0} × {1}".format(row["descriptor"], row["model"]),
        coverage_for_row=lambda row: "{0}/{1} folds".format(
            row.get("completed_folds", "—"), row.get("expected_folds", "—"),
        ),
    )


def _write_dimension_time_figure(
    layout: BenchmarkOutputLayout,
    summary: pd.DataFrame,
    run_id: str,
    dimension: str,
) -> Path:
    """Render descriptor or model totals from complete combination costs only."""
    if dimension == "descriptor":
        filename = "descriptor_modeling_time.png"
        title = "描述符：各模型组合累计建模耗时"
    elif dimension == "model":
        filename = "model_modeling_time.png"
        title = "建模方法：各描述符组合累计建模耗时"
    else:  # pragma: no cover - internal callers use the two fixed dimensions
        raise BenchmarkReportError("未知的耗时图维度：{0}".format(dimension))
    return _write_time_figure(
        layout, summary, run_id,
        filename=filename,
        title=title,
        label_for_row=lambda row: row["item"],
        coverage_for_row=lambda row: "{0}/{1} combinations".format(
            row.get("comparable_combinations", "—"), row.get("expected_combinations", "—"),
        ),
    )


def _sum_valid_fold_times(fold_metrics: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(fold_metrics.get(column, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 0.0)
    return float(np.sum(values[valid])) if valid.any() else 0.0


def _atomic_write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # ``Path.write_text(newline=...)`` is only available in newer Python
    # releases; YONOD's supported Python 3.9 needs the explicit file handle.
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    temporary.replace(path)
    return path


def generate_benchmark_report(run_dir: Path | str) -> BenchmarkReportResult:
    """Rebuild benchmark HTML and Markdown reports without any model fitting."""
    root = Path(run_dir)
    layout = resolve_benchmark_output_layout(root)
    run_manifest = _read_json(layout.manifests / "run_manifest.json")
    split_manifest = pd.read_parquet(layout.manifests / "split_manifest.parquet")
    tables = _load_tables(layout)
    tables.update(_load_or_rebuild_time_summaries(layout, tables))
    split_audit = _split_audit(split_manifest)
    performance = _performance_matrix(tables["combination_summary"], tables["completeness"])
    state = _state_summary(layout)
    descriptor_status_path = layout.docs / "descriptor_status.csv"
    descriptor_status = (
        pd.read_csv(descriptor_status_path)
        if descriptor_status_path.is_file()
        else pd.DataFrame(columns=[
            "descriptor", "status", "stage", "artifact_path", "n_total",
            "n_valid", "feature_dim", "skipped_model_count", "reason",
        ])
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = str(run_manifest.get("run_id", "—"))
    stability_figures = _write_figures(layout, tables["fold_metrics"])
    combination_time_figure = _write_combination_time_figure(layout, tables["combination_time_summary"], run_id)
    descriptor_time_figure = _write_dimension_time_figure(layout, tables["descriptor_time_summary"], run_id, "descriptor")
    model_time_figure = _write_dimension_time_figure(layout, tables["model_time_summary"], run_id, "model")
    time_figures = (combination_time_figure, descriptor_time_figure, model_time_figure)
    figures = stability_figures + time_figures
    config = run_manifest.get("benchmark_config", {})
    traceability = pd.DataFrame([{
        "run_id": run_id, "config_hash": run_manifest.get("config_hash"),
        "dataset_sha256": run_manifest.get("dataset_sha256"), "code_git_commit": run_manifest.get("code_git_commit"),
        "dataset_path": run_manifest.get("dataset_path"), "grouping": json.dumps(config.get("grouping", {}), ensure_ascii=False),
        "cv": json.dumps(config.get("cv", {}), ensure_ascii=False),
        "derived_from": ", ".join(
            layout.artifact_reference(name) + "/" for name in ("manifests", "predictions", "folds", "metrics")
        ),
    }])
    protocol_alignment = _protocol_alignment_table(config, split_manifest)
    costs = pd.DataFrame([{
        "run_dir": str(root), "disk_bytes": _disk_usage_bytes(root),
        "all_valid_fold_cumulative_train_time_s": _sum_valid_fold_times(tables["fold_metrics"], "train_time_s"),
        "all_valid_fold_cumulative_predict_time_s": _sum_valid_fold_times(tables["fold_metrics"], "predict_time_s"),
        "all_valid_fold_cumulative_model_time_s": _sum_valid_fold_times(tables["fold_metrics"], "train_time_s") + _sum_valid_fold_times(tables["fold_metrics"], "predict_time_s"),
        "time_comparable_combinations": int(tables["combination_time_summary"]["is_time_comparable"].fillna(False).astype(bool).sum()),
        "time_noncomparable_combinations": int((~tables["combination_time_summary"]["is_time_comparable"].fillna(False).astype(bool)).sum()),
        "metric_exclusions": int(len(tables["metric_exclusions"])),
        "comparison_exclusions": int(len(tables["comparison_exclusions"])),
    }])
    model_conclusions = _comparison_conclusions(tables["model_comparisons"], "描述符")
    descriptor_conclusions = _comparison_conclusions(tables["descriptor_comparisons"], "模型")
    css = """
<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1280px;margin:auto;padding:24px;color:#172033;background:#f7f9fc}section{background:white;margin:16px 0;padding:18px;border-radius:8px;box-shadow:0 1px 3px #0002}h1{color:#123b66}h2{border-left:4px solid #2563eb;padding-left:9px}.data-table{border-collapse:collapse;width:100%;font-size:.85rem}.data-table th,.data-table td{border:1px solid #d9e1ec;padding:6px;text-align:left}.data-table th{background:#123b66;color:#fff}.note,.empty{color:#56657a}img{max-width:100%;border:1px solid #d9e1ec;border-radius:5px;margin:8px 0}code{word-break:break-all}</style>
"""
    def section(title: str, body: str) -> str:
        return "<section><h2>{0}</h2>{1}</section>".format(html.escape(title), body)
    figures_html = "".join(
        "<figure><img src='{0}/{1}' alt='{1}'/><figcaption>{1}</figcaption></figure>".format(
            layout.picture_relative_to_report, html.escape(path.name),
        )
        for path in stability_figures
    )
    def time_figure_html(path: Path, caption: str) -> str:
        return "<figure><img src='{0}/{1}' alt='{2}'/><figcaption>{2}（训练与预测堆叠）。</figcaption></figure>".format(
            layout.picture_relative_to_report, html.escape(path.name), html.escape(caption),
        )
    time_note_html = (
        "<p>时间口径：`total_model_time_s = total_train_time_s + total_predict_time_s`，"
        "均为该组合全部有效外部 CV fold 的累计值。描述符特征化与 CLI 端到端墙钟时间不计入柱状图；"
        "不完整、缺失或非法时间的组合保留状态，但不进入耗时排序。描述符和建模方法图是组合成本的两种汇总视图，"
        "不应与组合图相加，也不把共享特征化时间重复归因给模型。</p>"
    )
    time_section_html = (
        time_note_html
        + "<h3>描述符 × 建模方法组合</h3>" + _table_html(tables["combination_time_summary"])
        + time_figure_html(combination_time_figure, "组合级建模耗时柱状图")
        + "<h3>按描述符汇总</h3><p class='note'>每根柱为该描述符下所有完整且时间可比较模型组合的累计时间。</p>"
        + _table_html(tables["descriptor_time_summary"])
        + time_figure_html(descriptor_time_figure, "描述符累计建模耗时柱状图")
        + "<h3>按建模方法汇总</h3><p class='note'>每根柱为该建模方法在所有描述符下完整且时间可比较组合的累计时间。</p>"
        + _table_html(tables["model_time_summary"])
        + time_figure_html(model_time_figure, "建模方法累计建模耗时柱状图")
    )
    conclusion_html = "<ul>" + "".join("<li>{0}</li>".format(html.escape(line)) for line in model_conclusions + descriptor_conclusions) + "</ul>"
    html_content = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>YONOD Benchmark {run}</title>{css}</head><body>
<h1>YONOD 严谨模型比较报告</h1><p>生成时间：{now}；本报告只读取已保存 artefacts，不会重新训练模型。</p>
 {trace}{protocol}{descriptors}{split}{performance}{time}{figures}{statistics}{tukey}{costs}{limits}
</body></html>""".format(
        run=html.escape(run_id), css=css, now=html.escape(now),
        trace=section("实验可追溯性", _table_html(traceability)),
        protocol=section(
            "论文协议对齐状态",
            "<p>此处记录特征、切分与随机种子的运行证据；即使参数匹配，仍需另行核对数据、RDKit/sklearn 版本与指标后才能讨论数值复现。</p>"
            + _table_html(protocol_alignment),
        ),
        descriptors=section(
            "描述符预计算状态",
            "<p>建模任务只读取 <code>descriptors/*.npz</code>；失败描述符的模型任务被隔离跳过。</p>"
            + _table_html(descriptor_status),
        ),
        split=section("切分审计", _table_html(split_audit)),
        performance=section("性能矩阵与完成度", _table_html(performance) + _table_html(tables["completeness"])),
        time=section("建模耗时与成本对比", time_section_html),
        figures=section("预测与稳定性", figures_html or "<p class='empty'>没有可用预测图；请检查 fold_metrics 中的预测路径。</p>"),
        statistics=section("双维统计比较", conclusion_html + "<h3>固定描述符比较模型</h3>" + _table_html(tables["model_comparisons"]) + "<h3>固定模型比较描述符</h3>" + _table_html(tables["descriptor_comparisons"]) + "<h3>不可比较记录</h3>" + _table_html(tables["comparison_exclusions"])),
        tukey=section("Tukey HSD 多重比较", "<h3>模型维度</h3>" + _table_html(tables["model_tukey_hsd"]) + "<h3>描述符维度</h3>" + _table_html(tables["descriptor_tukey_hsd"])),
        costs=section("成本、任务状态与失败", _table_html(costs) + _table_html(state) + "<h3>预测/指标排除记录</h3>" + _table_html(tables["metric_exclusions"])),
        limits=section("统计限制", "<p>比较单位是 CV fold。交叉验证折并非完全独立，因此 p 值不是唯一证据；应结合折级差值、bootstrap CI、稳定性图和任务缺失情况解读。`no_significant_difference` 仅表示当前数据和统计功效下未检出充分证据，不代表两种方法完全相同。Tukey HSD 表基于 fold 值的多重比较，和配对 t 检验并列呈现，不能任选更有利的结果。</p>"),
    )
    markdown = [
        "# YONOD 严谨模型比较报告", "", "生成时间：`{0}`  ".format(now), "run_id：`{0}`".format(run_id),
        "", "> 本报告只读取 `{0}/`、`{1}/`、`{2}/` 和 `{3}/`，不重新训练模型。".format(
            *(layout.artifact_reference(name) for name in ("manifests", "predictions", "folds", "metrics"))
        ), "",
        "## 实验可追溯性", "", _table_markdown(traceability), "", "## 论文协议对齐状态", "",
        "此处记录特征、切分与随机种子的运行证据；流程对齐不等同于论文数值复现，仍需核对数据和软件版本。", "",
        _table_markdown(protocol_alignment), "", "## 切分审计", "", _table_markdown(split_audit),
        "", "## 描述符预计算状态", "",
        "建模任务只读取 `descriptors/*.npz`；失败描述符的模型任务被隔离跳过。", "",
        _table_markdown(descriptor_status),
        "", "## 性能矩阵与完成度", "", _table_markdown(performance), "", _table_markdown(tables["completeness"]),
        "", "## 建模耗时与成本对比", "",
        "时间口径：`total_model_time_s = total_train_time_s + total_predict_time_s`，均为该组合全部有效外部 CV fold 的累计值。描述符特征化与 CLI 端到端墙钟时间不计入柱状图；不完整、缺失或非法时间的组合保留状态，但不进入耗时排序。描述符和建模方法图是组合成本的两种汇总视图，不应与组合图相加，也不把共享特征化时间重复归因给模型。",
        "", "### 描述符 × 建模方法组合", "", _table_markdown(tables["combination_time_summary"]),
        "", "![组合级建模耗时柱状图]({0}/{1})".format(layout.picture_relative_to_report, combination_time_figure.name),
        "", "### 按描述符汇总", "", "每根柱为该描述符下所有完整且时间可比较模型组合的累计时间。",
        "", _table_markdown(tables["descriptor_time_summary"]),
        "", "![描述符累计建模耗时柱状图]({0}/{1})".format(layout.picture_relative_to_report, descriptor_time_figure.name),
        "", "### 按建模方法汇总", "", "每根柱为该建模方法在所有描述符下完整且时间可比较组合的累计时间。",
        "", _table_markdown(tables["model_time_summary"]),
        "", "![建模方法累计建模耗时柱状图]({0}/{1})".format(layout.picture_relative_to_report, model_time_figure.name),
        "", "## 预测与稳定性", "",
    ]
    markdown.extend([
        "- [{0}]({1}/{0})".format(path.name, layout.picture_relative_to_report)
        for path in stability_figures
    ] or ["没有可用预测图。"])
    markdown += [
        "", "## 双维统计比较", "", "### 模型维度统计比较（固定描述符）", "", *["- " + line for line in model_conclusions], "", _table_markdown(tables["model_comparisons"]),
        "", "## 描述符维度统计比较（固定模型）", "", *["- " + line for line in descriptor_conclusions], "", _table_markdown(tables["descriptor_comparisons"]),
        "", "## Tukey HSD 多重比较", "", "### 模型维度", "", _table_markdown(tables["model_tukey_hsd"]), "", "### 描述符维度", "", _table_markdown(tables["descriptor_tukey_hsd"]),
        "", "## 成本、任务状态与失败", "", _table_markdown(costs), "", _table_markdown(state), "", _table_markdown(tables["metric_exclusions"]),
        "", "## 统计限制", "", "比较单位是 CV fold。折之间并非完全独立，p 值不是唯一证据；必须结合差值、bootstrap CI、稳定性图和缺失任务解读。`no_significant_difference` 不代表性能完全相同。Tukey HSD 与配对检验并列呈现，不可任选有利结果。", "",
    ]
    reports_dir = layout.report
    return BenchmarkReportResult(
        html_path=_atomic_write(reports_dir / "benchmark_report.html", html_content),
        markdown_path=_atomic_write(reports_dir / "benchmark_report.md", "\n".join(markdown)),
        figures=figures,
    )
