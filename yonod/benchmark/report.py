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


def _load_tables(root: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name in _REQUIRED_TABLES:
        path = root / "metrics" / (name + ".parquet")
        if not path.is_file():
            raise BenchmarkReportError(
                "缺少派生指标表 {0}。请先从预测库运行指标重建，而不是重新训练模型。".format(path)
            )
        tables[name] = pd.read_parquet(path)
    return tables


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


def _state_summary(root: Path) -> pd.DataFrame:
    state_path = root / "state" / "tasks.sqlite"
    if not state_path.is_file():
        return pd.DataFrame([{"state_store": "not_created", "count": 0}])
    try:
        with sqlite3.connect(str(state_path)) as connection:
            return pd.read_sql_query("SELECT status AS state_store, COUNT(*) AS count FROM tasks GROUP BY status", connection)
    except sqlite3.Error as exc:
        return pd.DataFrame([{"state_store": "unreadable", "count": 0, "detail": str(exc)}])


def _disk_usage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_figures(root: Path, fold_metrics: pd.DataFrame) -> tuple[Path, ...]:
    """Generate stability/OOF figures from saved tables and shards only."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise BenchmarkReportError("需要 matplotlib 才能生成 benchmark 稳定性图") from exc
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not fold_metrics.empty:
        labels = (fold_metrics["descriptor"].astype(str) + " × " + fold_metrics["model"].astype(str)).unique().tolist()
        figure, axes = plt.subplots(1, 3, figsize=(max(9, len(labels) * 1.6), 4.8), constrained_layout=True)
        for axis, metric in zip(axes, ("r2", "rmse", "mae")):
            data = [fold_metrics.loc[(fold_metrics["descriptor"].astype(str) + " × " + fold_metrics["model"].astype(str)) == label, metric].to_numpy() for label in labels]
            axis.boxplot(data, labels=labels, showmeans=True)
            axis.set_title(metric.upper() + " across folds")
            axis.tick_params(axis="x", rotation=45, labelsize=8)
        stability = figures_dir / "fold_metric_stability.png"
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
        scatter = figures_dir / "prediction_residuals.png"
        figure.savefig(scatter, dpi=150)
        plt.close(figure)
        paths.append(scatter)
    return tuple(paths)


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
    run_manifest = _read_json(root / "manifests" / "run_manifest.json")
    split_manifest = pd.read_parquet(root / "manifests" / "split_manifest.parquet")
    tables = _load_tables(root)
    split_audit = _split_audit(split_manifest)
    performance = _performance_matrix(tables["combination_summary"], tables["completeness"])
    state = _state_summary(root)
    figures = _write_figures(root, tables["fold_metrics"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = str(run_manifest.get("run_id", "—"))
    config = run_manifest.get("benchmark_config", {})
    traceability = pd.DataFrame([{
        "run_id": run_id, "config_hash": run_manifest.get("config_hash"),
        "dataset_sha256": run_manifest.get("dataset_sha256"), "code_git_commit": run_manifest.get("code_git_commit"),
        "dataset_path": run_manifest.get("dataset_path"), "grouping": json.dumps(config.get("grouping", {}), ensure_ascii=False),
        "cv": json.dumps(config.get("cv", {}), ensure_ascii=False), "derived_from": "manifests/, predictions/, folds/, metrics/",
    }])
    costs = pd.DataFrame([{
        "run_dir": str(root), "disk_bytes": _disk_usage_bytes(root),
        "total_train_time_s": float(tables["fold_metrics"].get("train_time_s", pd.Series(dtype=float)).sum()),
        "total_predict_time_s": float(tables["fold_metrics"].get("predict_time_s", pd.Series(dtype=float)).sum()),
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
    figures_html = "".join("<figure><img src='../figures/{0}' alt='{0}'/><figcaption>{0}</figcaption></figure>".format(html.escape(path.name)) for path in figures)
    conclusion_html = "<ul>" + "".join("<li>{0}</li>".format(html.escape(line)) for line in model_conclusions + descriptor_conclusions) + "</ul>"
    html_content = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>YONOD Benchmark {run}</title>{css}</head><body>
<h1>YONOD 严谨模型比较报告</h1><p>生成时间：{now}；本报告只读取已保存 artefacts，不会重新训练模型。</p>
{trace}{split}{performance}{figures}{statistics}{tukey}{costs}{limits}
</body></html>""".format(
        run=html.escape(run_id), css=css, now=html.escape(now),
        trace=section("实验可追溯性", _table_html(traceability)),
        split=section("切分审计", _table_html(split_audit)),
        performance=section("性能矩阵与完成度", _table_html(performance) + _table_html(tables["completeness"])),
        figures=section("预测与稳定性", figures_html or "<p class='empty'>没有可用预测图；请检查 fold_metrics 中的预测路径。</p>"),
        statistics=section("双维统计比较", conclusion_html + "<h3>固定描述符比较模型</h3>" + _table_html(tables["model_comparisons"]) + "<h3>固定模型比较描述符</h3>" + _table_html(tables["descriptor_comparisons"]) + "<h3>不可比较记录</h3>" + _table_html(tables["comparison_exclusions"])),
        tukey=section("Tukey HSD 多重比较", "<h3>模型维度</h3>" + _table_html(tables["model_tukey_hsd"]) + "<h3>描述符维度</h3>" + _table_html(tables["descriptor_tukey_hsd"])),
        costs=section("成本、任务状态与失败", _table_html(costs) + _table_html(state) + "<h3>预测/指标排除记录</h3>" + _table_html(tables["metric_exclusions"])),
        limits=section("统计限制", "<p>比较单位是 CV fold。交叉验证折并非完全独立，因此 p 值不是唯一证据；应结合折级差值、bootstrap CI、稳定性图和任务缺失情况解读。`no_significant_difference` 仅表示当前数据和统计功效下未检出充分证据，不代表两种方法完全相同。Tukey HSD 表基于 fold 值的多重比较，和配对 t 检验并列呈现，不能任选更有利的结果。</p>"),
    )
    markdown = [
        "# YONOD 严谨模型比较报告", "", "生成时间：`{0}`  ".format(now), "run_id：`{0}`".format(run_id),
        "", "> 本报告只读取 `manifests/`、`predictions/`、`folds/` 和 `metrics/`，不重新训练模型。", "",
        "## 实验可追溯性", "", _table_markdown(traceability), "", "## 切分审计", "", _table_markdown(split_audit),
        "", "## 性能矩阵与完成度", "", _table_markdown(performance), "", _table_markdown(tables["completeness"]),
        "", "## 预测与稳定性", "",
    ]
    markdown.extend(["- `figures/{0}`".format(path.name) for path in figures] or ["没有可用预测图。"])
    markdown += [
        "", "## 双维统计比较", "", "### 模型维度统计比较（固定描述符）", "", *["- " + line for line in model_conclusions], "", _table_markdown(tables["model_comparisons"]),
        "", "## 描述符维度统计比较（固定模型）", "", *["- " + line for line in descriptor_conclusions], "", _table_markdown(tables["descriptor_comparisons"]),
        "", "## Tukey HSD 多重比较", "", "### 模型维度", "", _table_markdown(tables["model_tukey_hsd"]), "", "### 描述符维度", "", _table_markdown(tables["descriptor_tukey_hsd"]),
        "", "## 成本、任务状态与失败", "", _table_markdown(costs), "", _table_markdown(state), "", _table_markdown(tables["metric_exclusions"]),
        "", "## 统计限制", "", "比较单位是 CV fold。折之间并非完全独立，p 值不是唯一证据；必须结合差值、bootstrap CI、稳定性图和缺失任务解读。`no_significant_difference` 不代表性能完全相同。Tukey HSD 与配对检验并列呈现，不可任选有利结果。", "",
    ]
    reports_dir = root / "reports"
    return BenchmarkReportResult(
        html_path=_atomic_write(reports_dir / "benchmark_report.html", html_content),
        markdown_path=_atomic_write(reports_dir / "benchmark_report.md", "\n".join(markdown)),
        figures=figures,
    )
