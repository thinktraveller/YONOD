"""通用报告生成器（§14.5）。

对外接口
--------
  rank_combinations(metrics_df)  -> pd.DataFrame   加权排名（§14.5.3）
  generate_report(metrics_df, task_info, out_dir)  -> Path  HTML 报告路径

报告结构（§14.5.1）
-------------------
  1. 任务信息头（任务名、CSV、样本量、列情况）
  2. 4×N 结果表格（描述符行 × 模型列，单元格：R²/RMSE/MAE，颜色渐变）
  3. 指标解释（§14.5.2 固定文字段落）
  4. 加权排名推荐（前三名 + 推荐理由，§14.5.3）
  5. 散点图 PNG 索引（如 out_dir 下存在 scatter_*.png）

指标权重（§14.5.2）
-------------------
  综合得分 = R²×0.5 + (1-RMSE/max_RMSE)×0.3 + (1-MAE/max_MAE)×0.2
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ─────────────────────────────── 加权排名 ───────────────────────────────────── #

def rank_combinations(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """按 §14.5.3 公式对可审计外层 CV 组合加权排名。

    当输入表包含协议/折数/OOF 守卫字段时，只有同时满足
    ``evaluation_protocol=outer_kfold``、``expected_folds=completed_folds=cv``、
    ``oof_complete=True`` 且 RMSE/MAE 均值与标准差完整的条目才进入排名。
    旧版无守卫字段的表仍按历史逻辑兼容排名。
    """
    guard = ranking_eligibility_table(metrics_df)
    if guard.empty:
        return pd.DataFrame(columns=_RANK_OUTPUT_COLUMNS)

    df = guard.loc[guard["ranking_eligible"].fillna(False).astype(bool)].copy()
    if df.empty:
        return pd.DataFrame(columns=_RANK_OUTPUT_COLUMNS)

    for column in ("r2", "rmse", "mae"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["r2", "rmse", "mae"]).copy()
    if df.empty:
        return pd.DataFrame(columns=_RANK_OUTPUT_COLUMNS)

    max_rmse = float(df["rmse"].max())
    max_mae = float(df["mae"].max())
    df["score"] = (
        df["r2"] * 0.5
        + (1 - df["rmse"] / max(max_rmse, 1e-9)) * 0.3
        + (1 - df["mae"] / max(max_mae, 1e-9)) * 0.2
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["folds"] = df.apply(_folds_display, axis=1)
    df["reason"] = ""

    for i in range(min(3, len(df))):
        row = df.iloc[i]
        parts: List[str] = []
        if row["r2"] > 0.85:
            parts.append(f"R²={row['r2']:.3f} 解释能力强")
        if row["rmse"] < 0.05:
            parts.append(f"RMSE={row['rmse']:.4f} 误差接近实验重复性")
        if row["mae"] < 0.04:
            parts.append(f"MAE={row['mae']:.4f} 典型偏差极小")
        if not parts:
            parts.append(f"综合评分 {row['score']:.3f} 排名靠前")
        protocol = _scalar_text(row.get("evaluation_protocol"), "legacy_untracked")
        if protocol != "legacy_untracked":
            parts.append(f"协议={protocol}，fold={row.get('folds', '—')}，OOF完整")
        df.at[i, "reason"] = "；".join(parts)

    out_cols = [column for column in _RANK_OUTPUT_COLUMNS if column in df.columns]
    return df.loc[:, out_cols]


# ─────────────────────────────── HTML 辅助 ──────────────────────────────────── #

def _esc(v: Any) -> str:
    return _html.escape(str(v))


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (float, int)):
        number = float(v)
        if not math.isfinite(number):
            return "—"
        return f"{number:.{digits}f}"
    return _esc(str(v))


def _fmt_time(v: Any) -> str:
    """将秒数格式化为人类可读字符串，如 '3.2s' 或 '1m 23s'。"""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(s) or s < 0.0:
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s)//60}m {int(s)%60}s"


def _fmt_metric_with_std(mean: Any, std: Any, digits: int = 4) -> str:
    mean_text = _fmt(mean, digits)
    std_text = _fmt(std, digits)
    if mean_text == "—":
        return "—"
    return mean_text if std_text == "—" else f"{mean_text} ± {std_text}"


def _color_r2(r2: float) -> str:
    """light-blue（低）→ dark-green（高）渐变，负值灰色。"""
    if math.isnan(r2):
        return "#f3f4f6"
    if r2 < 0:
        return "#e5e7eb"
    t = max(0.0, min(1.0, r2))
    r = int(219 + (22  - 219) * t)
    g = int(234 + (101 - 234) * t)
    b = int(254 + (52  - 254) * t)
    return f"rgb({r},{g},{b})"


def _text_color(r2: float) -> str:
    return "white" if (not math.isnan(r2) and r2 > 0.5) else "#111827"


def _b64_png(path: Path) -> Optional[str]:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")
    return None


_UNIVERSAL_TIME_COLUMNS = [
    "descriptor", "model", "reported_rows", "total_train_time_s",
    "is_time_comparable", "time_status",
]
_UNIVERSAL_DIMENSION_TIME_COLUMNS = [
    "aggregation_dimension", "item", "expected_combinations",
    "comparable_combinations", "noncomparable_combinations",
    "is_time_comparable", "time_status", "total_train_time_s",
]

_PROTOCOL_GUARD_FIELDS = {"evaluation_protocol", "expected_folds", "completed_folds", "oof_complete"}
_RANK_OUTPUT_COLUMNS = [
    "rank", "desc_name", "model_name", "evaluation_protocol", "folds", "oof_complete",
    "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std", "score",
    "train_time_s", "autogluon_time_limit", "autogluon_presets", "autogluon_num_cpus",
    "autogluon_seed_policy", "reason",
]
_RANK_ELIGIBILITY_COLUMNS = [
    "desc_name", "model_name", "evaluation_protocol", "cv", "expected_folds", "completed_folds",
    "folds", "oof_complete", "oof_n_observed", "oof_n_total",
    "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std",
    "train_time_s", "autogluon_time_limit", "autogluon_presets", "autogluon_num_cpus",
    "autogluon_seed_policy", "ranking_eligible", "ranking_exclusion_reason",
]


def _first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"期望列之一 {candidates} 不存在于 DataFrame 中。")


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _int_or_none(value: Any) -> Optional[int]:
    number = _float_or_nan(value)
    if math.isnan(number):
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9:
        return None
    return rounded


def _scalar_text(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _truthy_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value) if math.isfinite(float(value)) else default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "完整"}


def _yes_no(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return "是" if _truthy_bool(value) else "否"


def _folds_display(row: Any) -> str:
    completed = _int_or_none(row.get("completed_folds"))
    expected = _int_or_none(row.get("expected_folds"))
    cv = _int_or_none(row.get("cv"))
    if completed is not None and expected is not None and cv is not None:
        return f"{completed}/{expected} (cv={cv})"
    if completed is not None and expected is not None:
        return f"{completed}/{expected}"
    if expected is not None:
        return f"—/{expected}"
    if cv is not None:
        return f"cv={cv}"
    return "—"


def _autogluon_param_summary(row: Any) -> str:
    pairs = []
    labels = {
        "autogluon_time_limit": "time_limit",
        "autogluon_presets": "presets",
        "autogluon_num_cpus": "num_cpus",
        "autogluon_seed_policy": "seed_policy",
    }
    for column, label in labels.items():
        value = row.get(column)
        text = _scalar_text(value, "")
        if text:
            pairs.append(f"{label}={text}")
    return "; ".join(pairs) if pairs else "—"


def _has_protocol_guard_columns(frame: pd.DataFrame) -> bool:
    return any(column in frame.columns for column in _PROTOCOL_GUARD_FIELDS)


def ranking_eligibility_table(metrics_df: pd.DataFrame, expected_cv: Optional[int] = None) -> pd.DataFrame:
    """Return the row-level protocol guard used before recommendation ranking.

    The table is intentionally display-friendly: non-eligible rows retain their
    protocol, fold counts, OOF status, and exclusion reason so holdout results
    remain visible without being blended into outer-CV rankings.
    """
    if metrics_df.empty:
        return pd.DataFrame(columns=_RANK_ELIGIBILITY_COLUMNS)

    df = metrics_df.copy()
    desc_col = _first_existing_column(df, ["desc_name", "descriptor"])
    model_col = _first_existing_column(df, ["model_name", "model"])
    r2_col = _first_existing_column(df, ["r2", "r2_mean"])
    rmse_col = _first_existing_column(df, ["rmse", "rmse_mean"])
    mae_col = _first_existing_column(df, ["mae", "mae_mean"])
    guarded = _has_protocol_guard_columns(df)
    has_rmse_std = "rmse_std" in df.columns
    has_mae_std = "mae_std" in df.columns

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        protocol = _scalar_text(row.get("evaluation_protocol"), "legacy_untracked" if not guarded else "missing")
        cv_value = _int_or_none(row.get("cv", expected_cv))
        if cv_value is None and expected_cv is not None:
            cv_value = int(expected_cv)
        expected = _int_or_none(row.get("expected_folds", cv_value if cv_value is not None else None))
        completed_default = expected if not guarded else None
        completed = _int_or_none(row.get("completed_folds", completed_default))
        oof_default = True if not guarded else False
        oof_complete = _truthy_bool(row.get("oof_complete", oof_default), default=oof_default)
        r2 = _float_or_nan(row.get(r2_col))
        r2_std = _float_or_nan(row.get("r2_std")) if "r2_std" in df.columns else float("nan")
        rmse = _float_or_nan(row.get(rmse_col))
        rmse_std = _float_or_nan(row.get("rmse_std")) if has_rmse_std else float("nan")
        mae = _float_or_nan(row.get(mae_col))
        mae_std = _float_or_nan(row.get("mae_std")) if has_mae_std else float("nan")

        metric_ok = all(not math.isnan(value) for value in (r2, rmse, mae))
        std_ok = True
        if guarded:
            std_ok = has_rmse_std and has_mae_std and not math.isnan(rmse_std) and not math.isnan(mae_std)
        protocol_ok = True if not guarded else protocol == "outer_kfold"
        folds_ok = True
        if guarded:
            folds_ok = expected is not None and completed is not None and completed == expected
            if cv_value is not None:
                folds_ok = folds_ok and expected == cv_value
        oof_ok = True if not guarded else bool(oof_complete)

        reasons: List[str] = []
        if not protocol_ok:
            reasons.append("协议不是 outer_kfold")
        if not folds_ok:
            reasons.append("expected_folds/completed_folds/cv 不一致或缺失")
        if not oof_ok:
            reasons.append("OOF 不完整")
        if not metric_ok:
            reasons.append("R²/RMSE/MAE 均值缺失或非有限")
        if not std_ok:
            reasons.append("RMSE/MAE 标准差缺失或非有限")
        eligible = bool(protocol_ok and folds_ok and oof_ok and metric_ok and std_ok)

        record = {
            "desc_name": _scalar_text(row.get(desc_col)),
            "model_name": _scalar_text(row.get(model_col)),
            "evaluation_protocol": protocol,
            "cv": cv_value,
            "expected_folds": expected,
            "completed_folds": completed,
            "oof_complete": oof_complete,
            "oof_n_observed": _int_or_none(row.get("oof_n_observed")),
            "oof_n_total": _int_or_none(row.get("oof_n_total")),
            "r2": r2, "r2_std": r2_std,
            "rmse": rmse, "rmse_std": rmse_std,
            "mae": mae, "mae_std": mae_std,
            "train_time_s": _float_or_nan(row.get("train_time_s")) if "train_time_s" in df.columns else float("nan"),
            "autogluon_time_limit": row.get("autogluon_time_limit"),
            "autogluon_presets": row.get("autogluon_presets"),
            "autogluon_num_cpus": row.get("autogluon_num_cpus"),
            "autogluon_seed_policy": row.get("autogluon_seed_policy"),
            "ranking_eligible": eligible,
            "ranking_exclusion_reason": "可进入排名" if eligible else "; ".join(reasons),
        }
        record["folds"] = _folds_display(record)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=_RANK_ELIGIBILITY_COLUMNS)


def _universal_time_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create a conservative combination-level training-time summary.

    The universal path has no fold-level prediction time contract.  It only
    reports supplied ``train_time_s`` values and never manufactures a total
    modelling duration from a missing prediction-time column.
    """
    if "train_time_s" not in metrics_df.columns:
        return pd.DataFrame(columns=_UNIVERSAL_TIME_COLUMNS)
    desc_col = "desc_name" if "desc_name" in metrics_df.columns else "descriptor"
    model_col = "model_name" if "model_name" in metrics_df.columns else "model"
    if desc_col not in metrics_df.columns or model_col not in metrics_df.columns:
        return pd.DataFrame(columns=_UNIVERSAL_TIME_COLUMNS)
    frame = metrics_df[[desc_col, model_col, "train_time_s"]].copy()
    frame.columns = ["descriptor", "model", "train_time_s"]
    frame["train_time_s"] = pd.to_numeric(frame["train_time_s"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for (descriptor, model), part in frame.groupby(["descriptor", "model"], sort=True, dropna=False):
        times = part["train_time_s"].to_numpy(dtype=float)
        valid = len(times) > 0 and all(math.isfinite(value) and value >= 0.0 for value in times)
        rows.append({
            "descriptor": str(descriptor),
            "model": str(model),
            "reported_rows": int(len(part)),
            "total_train_time_s": float(times.sum()) if valid else float("nan"),
            "is_time_comparable": bool(valid),
            "time_status": "reported_train_time" if valid else "invalid_or_missing_train_time",
        })
    result = pd.DataFrame.from_records(rows, columns=_UNIVERSAL_TIME_COLUMNS)
    if not result.empty:
        result = result.sort_values(["is_time_comparable", "total_train_time_s"], ascending=[False, False], kind="stable")
    return result


def _universal_dimension_time_summary(
    combination_summary: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """Summarise complete universal-report combinations by one dimension.

    These are two alternative roll-ups of the reported combination training
    times, not extra elapsed stages.  The universal path has no descriptor
    featurisation timing contract, so it is excluded rather than duplicated
    across every model.
    """
    if dimension not in {"descriptor", "model"}:
        raise ValueError("dimension 必须是 descriptor 或 model")
    required = set(_UNIVERSAL_TIME_COLUMNS)
    if required.difference(combination_summary.columns):
        return pd.DataFrame(columns=_UNIVERSAL_DIMENSION_TIME_COLUMNS)
    rows: List[Dict[str, Any]] = []
    for item, part in combination_summary.groupby(dimension, sort=True, dropna=False):
        expected = int(len(part))
        comparable = part["is_time_comparable"].fillna(False).astype(bool)
        comparable_count = int(comparable.sum())
        is_comparable = bool(expected > 0 and comparable_count == expected)
        rows.append({
            "aggregation_dimension": dimension,
            "item": str(item),
            "expected_combinations": expected,
            "comparable_combinations": comparable_count,
            "noncomparable_combinations": expected - comparable_count,
            "is_time_comparable": is_comparable,
            "time_status": "complete_and_comparable" if is_comparable else "contains_noncomparable_combination",
            "total_train_time_s": float(part["total_train_time_s"].sum()) if is_comparable else float("nan"),
        })
    result = pd.DataFrame.from_records(rows, columns=_UNIVERSAL_DIMENSION_TIME_COLUMNS)
    if not result.empty:
        result = result.sort_values(["is_time_comparable", "total_train_time_s"], ascending=[False, False], kind="stable")
    return result


def _configure_chinese_matplotlib(plt: Any) -> Optional[str]:
    """Use an installed CJK font and return its name, or ``None`` if absent.

    Callers which put Chinese text directly into raster figures can use the
    return value to render an explicit ASCII fallback instead of silently
    producing tofu boxes on minimal Linux or Windows installations.
    """
    try:
        from matplotlib import font_manager
        installed = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - font discovery varies by platform
        return None
    # Keep this list deliberately broad: Linux distributions commonly expose
    # Noto or the AR PL families instead of the Windows Chinese fonts.
    for candidate in (
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "Noto Serif CJK SC", "AR PL UMing CN", "AR PL UKai CN",
        "WenQuanYi Zen Hei", "Droid Sans Fallback",
    ):
        if candidate in installed:
            current = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = [candidate] + [name for name in current if name != candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    return None


def _write_universal_time_figure(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    dimension: str,
) -> Optional[Path]:
    """Create one named universal-report training-time comparison PNG."""
    comparable = summary.loc[summary["is_time_comparable"].fillna(False).astype(bool)].copy() if not summary.empty else summary
    if comparable.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    _configure_chinese_matplotlib(plt)
    comparable = comparable.sort_values("total_train_time_s", ascending=False, kind="stable")
    if dimension == "combination":
        labels = (comparable["descriptor"].astype(str) + " × " + comparable["model"].astype(str)).tolist()
        filename = "combination_training_time.png"
        title = "描述符 × 模型：报告提供的训练时间"
    elif dimension == "descriptor":
        labels = comparable["item"].astype(str).tolist()
        filename = "descriptor_training_time.png"
        title = "描述符：各模型组合累计训练时间"
    elif dimension == "model":
        labels = comparable["item"].astype(str).tolist()
        filename = "model_training_time.png"
        title = "建模方法：各描述符组合累计训练时间"
    else:  # pragma: no cover - internal callers use fixed dimensions
        raise ValueError("未知的耗时图维度：{0}".format(dimension))
    figure, axis = plt.subplots(figsize=(10, max(3.6, 0.62 * len(comparable) + 1.0)), constrained_layout=True)
    positions = list(range(len(comparable)))
    totals = comparable["total_train_time_s"].to_numpy(dtype=float)
    axis.barh(positions, totals, color="#2563eb")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("累计训练时间（秒）")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    max_total = float(totals.max()) if len(totals) else 0.0
    axis.set_xlim(0.0, max(0.01, max_total * 1.18))
    for position, total in zip(positions, totals):
        axis.annotate(_fmt_time(total), xy=(total, position), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    figure.text(0.01, 0.01, "仅含 metrics_df.train_time_s；未提供预测时间，不能解释为端到端墙钟时间。", fontsize=7, color="#56657a")
    pictures_dir = out_dir / "pictures"
    pictures_dir.mkdir(parents=True, exist_ok=True)
    path = pictures_dir / filename
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _section_modeling_time(metrics_df: pd.DataFrame, out_dir: Path) -> str:
    """Render a clear no-data state instead of a misleading zero-time chart."""
    summary = _universal_time_summary(metrics_df)
    if "train_time_s" not in metrics_df.columns:
        return """
<section>
<h2>5 · 建模耗时与成本对比</h2>
<p class="note">未提供组合级 <code>train_time_s</code>，无法绘制建模耗时图；报告未使用空值或零值代替耗时。</p>
</section>"""
    dimension_summaries = {
        "descriptor": _universal_dimension_time_summary(summary, "descriptor"),
        "model": _universal_dimension_time_summary(summary, "model"),
    }
    figure_paths = {
        "combination": _write_universal_time_figure(summary, out_dir, dimension="combination"),
        "descriptor": _write_universal_time_figure(dimension_summaries["descriptor"], out_dir, dimension="descriptor"),
        "model": _write_universal_time_figure(dimension_summaries["model"], out_dir, dimension="model"),
    }
    def table_for(frame: pd.DataFrame) -> str:
        shown = frame.copy()
        if not shown.empty:
            shown["total_train_time_s"] = shown["total_train_time_s"].map(_fmt_time)
        return shown.to_html(index=False, escape=True) if not shown.empty else "<p class='note'>没有可用耗时记录。</p>"
    def figure_for(key: str, caption: str) -> str:
        path = figure_paths[key]
        image = _b64_png(path) if path else None
        if not image:
            return "<p class='note'>没有可比较的有限非负训练时间，因此未生成零高柱状图。</p>"
        return "<figure style='text-align:center'><img src='data:image/png;base64,{0}' alt='{1}' style='max-width:100%'/><figcaption class='note'>{1}；同一 PNG 由 Markdown 报告相对引用。</figcaption></figure>".format(image, _esc(caption))
    return """
<section>
<h2>5 · 建模耗时与成本对比</h2>
<p class="note">本路径仅汇总输入 <code>metrics_df.train_time_s</code>，未提供预测时间，故图表仅表示训练时间，不表示训练加预测时间或 CLI 端到端墙钟时间。描述符和建模方法图是组合时间的两种汇总视图，不与组合图相加；共享描述符特征化时间未提供，因此不重复归因。异常或缺失时间标为不可比较，不以零秒显示。</p>
<h3>描述符 × 建模方法组合</h3>{combination_table}{combination_figure}
<h3>按描述符汇总</h3><p class="note">每根柱为该描述符下所有完整且时间可比较模型组合的累计训练时间。</p>{descriptor_table}{descriptor_figure}
<h3>按建模方法汇总</h3><p class="note">每根柱为该建模方法在所有描述符下完整且时间可比较组合的累计训练时间。</p>{model_table}{model_figure}
</section>""".format(
        combination_table=table_for(summary), combination_figure=figure_for("combination", "组合训练时间柱状图"),
        descriptor_table=table_for(dimension_summaries["descriptor"]), descriptor_figure=figure_for("descriptor", "描述符累计训练时间柱状图"),
        model_table=table_for(dimension_summaries["model"]), model_figure=figure_for("model", "建模方法累计训练时间柱状图"),
    )


# ─────────────────────────────── HTML 段落 ──────────────────────────────────── #

_CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", Arial, sans-serif; background: #f9fafb;
       color: #111827; padding: 24px; max-width: 1100px; margin: auto; }
h1 { font-size: 1.6rem; margin-bottom: 6px; color: #1e3a5f; }
h2 { font-size: 1.15rem; margin: 24px 0 10px; color: #1e3a5f;
     border-left: 4px solid #2563eb; padding-left: 10px; }
p  { margin: 6px 0; line-height: 1.6; }
.subtitle { color: #6b7280; font-size: 0.9rem; margin-bottom: 16px; }
section { background: white; border-radius: 8px; padding: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { border: 1px solid #e5e7eb; padding: 7px 10px; text-align: center; }
thead th { background: #1e3a5f; color: white; }
.rank-tbl td:first-child { font-weight: 700; }
.cell-r2 { font-weight: 600; }
.note { font-size: 0.8rem; color: #6b7280; margin-top: 10px; }
.scatter-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.scatter-grid img { max-width: 380px; border-radius: 6px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.12); }
.formula-block { background: #f0f4ff; border-left: 3px solid #2563eb;
                 border-radius: 4px; padding: 12px 20px; margin: 8px 0 12px;
                 color: #1e3a5f; font-size: 1rem; text-align: center; }
</style>
"""


def _section_intro(task_info: Dict[str, Any], now: str) -> str:
    name    = _esc(task_info.get("task_name", "—"))
    csv_p   = _esc(str(task_info.get("csv_path", "—")))
    n_rows  = task_info.get("n_samples", "—")
    smi_c   = _esc(str(task_info.get("smiles_cols", "—")))
    num_c   = _esc(str(task_info.get("numeric_cols", "（无）")))
    lbl     = _esc(task_info.get("label_col", "—"))
    n_combo = task_info.get("n_combinations", "—")

    citation = task_info.get("dataset_citation")
    url      = task_info.get("dataset_url")
    notes    = task_info.get("dataset_notes")
    dataset_section = ""
    if any([citation, url, notes]):
        ds_rows = ""
        if citation:
            ds_rows += (
                f"<tr><th style='text-align:left'>文献引用</th>"
                f"<td style='text-align:left'>{_esc(citation)}</td></tr>"
            )
        if url:
            url_esc = _esc(url)
            ds_rows += (
                f"<tr><th style='text-align:left'>开源地址</th>"
                f"<td style='text-align:left'>"
                f"<a href='{url_esc}' target='_blank' rel='noopener'>{url_esc}</a>"
                f"</td></tr>"
            )
        if notes:
            ds_rows += (
                f"<tr><th style='text-align:left'>备注</th>"
                f"<td style='text-align:left'>{_esc(notes)}</td></tr>"
            )
        dataset_section = (
            f"\n<h2 style='margin-top:16px;font-size:1rem'>数据集来源</h2>"
            f"\n<table>{ds_rows}</table>"
        )

    return f"""
<section>
<h1>YONOD 通用建模报告 — {name}</h1>
<p class="subtitle">生成时间：{now} &nbsp;|&nbsp; 评估组合数：{n_combo}</p>
<table>
<tr><th style="width:160px;text-align:left">任务名称</th><td style="text-align:left">{name}</td></tr>
<tr><th style="text-align:left">CSV 路径</th><td style="text-align:left"><code>{csv_p}</code></td></tr>
<tr><th style="text-align:left">有效样本量</th><td style="text-align:left">{n_rows}</td></tr>
<tr><th style="text-align:left">SMILES 列</th><td style="text-align:left"><code>{smi_c}</code></td></tr>
<tr><th style="text-align:left">数值辅助列</th><td style="text-align:left"><code>{num_c}</code></td></tr>
<tr><th style="text-align:left">标签列</th><td style="text-align:left"><code>{lbl}</code></td></tr>
</table>{dataset_section}
</section>"""


def _section_numeric_audit(task_info: Dict[str, Any]) -> str:
    rows = task_info.get("numeric_audit_rows") or []
    if not rows:
        return ""
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(row.get(key, '—'))}</td>" for key in
        ("run", "repeat", "fold", "source", "name", "unit", "missing", "scaling", "train_missing_rows", "constant_train", "numeric_dimension", "input_dimension")) + "</tr>"
        for row in rows
    )
    return (
        "<section><h2>数值输入与折内处理</h2><p>每行来自已保存且校验的折级处理状态。</p>"
        "<table><tr><th>组合</th><th>repeat</th><th>fold</th><th>源列</th><th>输出名</th>"
        "<th>单位</th><th>缺失策略</th><th>缩放</th><th>训练缺失行</th><th>训练常量</th><th>数值维度</th><th>送模总维度</th></tr>"
        + body + "</table></section>"
    )


def _hpo_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "N/A" if not math.isfinite(value) else f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


_HPO_TABLES = (
    ("studies", "搜索状态与最佳参数", (
        ("combination", "组合"), ("repeat", "repeat"), ("fold", "fold"),
        ("study_id", "study"), ("status", "状态"), ("objective", "目标"),
        ("attempted", "尝试"), ("complete", "成功"), ("failed", "失败"),
        ("best_trial", "最佳 trial"), ("best_inner", "最佳内层均分"),
        ("stop_reason", "停止原因"), ("search_time_s", "搜索活跃秒"),
        ("best_parameters", "实际模型参数"), ("parameter_sources", "参数来源"),
        ("trial_export", "trial 导出"),
    )),
    ("trials", "Trial 轨迹", (
        ("combination", "组合"), ("repeat", "repeat"), ("fold", "fold"),
        ("trial", "trial"), ("state", "状态"), ("objective", "内层目标均分"),
        ("best_so_far", "截至该次最佳"), ("inner_fold_scores", "逐内层折分数"),
        ("parameters", "建议参数"), ("failure", "失败原因"),
    )),
    ("outer", "独立外层折评估与成本", (
        ("combination", "组合"), ("repeat", "repeat"), ("fold", "fold"),
        ("n_valid", "验证行"), ("r2", "R²"), ("rmse", "RMSE"),
        ("mae", "MAE"), ("kendall_tau", "Kendall τ"),
        ("kendall_tau_reason", "τ 缺失原因"),
        ("outer_train_time_s", "外层训练秒"),
        ("outer_predict_time_s", "外层预测秒"),
    )),
    ("final", "显式最终模型（与外层 OOF 分开）", (
        ("combination", "组合"), ("status", "状态"), ("study_id", "独立 study"),
        ("n_development_rows", "开发集行数"), ("attempted", "尝试"),
        ("best_trial", "最佳 trial"), ("best_inner", "最佳内层均分"),
        ("best_parameters", "实际模型参数"), ("search_time_s", "搜索活跃秒"),
        ("full_refit_time_s", "完整开发集重训秒"),
        ("independent_test_score", "独立测试分数"), ("bundle_path", "模型包"),
    )),
)


def _section_hpo(task_info: Dict[str, Any]) -> str:
    report = task_info.get("hpo_report")
    if not report:
        return ""
    parts = [
        "<section><h2>超参数搜索与嵌套评估</h2>",
        "<p>内层目标是各 inner-fold 分数的算术平均；下方外层折指标只使用未参与选参的验证行。"
        "任务其他结果表中的 ordinary 指标是每个 repeat 的 pooled OOF，再跨 repeat 汇总；两种口径不混算。"
        "搜索时间只计活跃执行，外层训练与预测单列；未列出的特征准备和端到端时间不计入这些数字。</p>",
    ]
    for key, title, columns in _HPO_TABLES:
        if key == "final" and not report.get("final_enabled"):
            continue
        rows = report.get(key) or []
        parts.append(f"<h3>{_esc(title)}</h3>")
        if not rows:
            parts.append("<p>N/A：该阶段尚无可核验记录。</p>")
            continue
        head = "".join(f"<th>{_esc(label)}</th>" for _, label in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{_esc(_hpo_cell(row.get(field)))}</td>" for field, _ in columns) + "</tr>"
            for row in rows
        )
        parts.append(f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")
    parts.append("</section>")
    return "".join(parts)


def _markdown_hpo(task_info: Dict[str, Any]) -> List[str]:
    report = task_info.get("hpo_report")
    if not report:
        return []
    lines = [
        "---", "", "## 超参数搜索与嵌套评估", "",
        "内层目标是逐 inner-fold 分数的算术平均；外层折指标只使用未参与选参的验证行。"
        "普通结果表另按每个 repeat 的 pooled OOF 汇总，不与折均值混算。"
        "搜索秒数只含活跃执行；外层训练和预测单列，不包括特征准备或端到端时间。", "",
    ]
    for key, title, columns in _HPO_TABLES:
        if key == "final" and not report.get("final_enabled"):
            continue
        rows = report.get(key) or []
        lines += [f"### {title}", ""]
        if not rows:
            lines += ["N/A：该阶段尚无可核验记录。", ""]
            continue
        lines.append("| " + " | ".join(label for _, label in columns) + " |")
        lines.append("|" + "---|" * len(columns))
        for row in rows:
            cells = [_hpo_cell(row.get(field)).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
                     for field, _ in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _section_grid(df: pd.DataFrame) -> str:
    """4×N 描述符 × 模型矩阵，单元格显示 R²/RMSE/MAE 及协议。"""
    if df.empty:
        return """
<section>
<h2>1 · 描述符 × 模型结果矩阵</h2>
<p class="note">没有有效模型结果；请查看“描述符预计算状态”中的失败原因。</p>
</section>"""
    # 统一列名
    desc_col  = "desc_name"  if "desc_name"  in df.columns else "descriptor"
    model_col = "model_name" if "model_name" in df.columns else "model"
    r2_col    = "r2_mean"    if "r2_mean"    in df.columns else "r2"
    rmse_col  = "rmse_mean"  if "rmse_mean"  in df.columns else "rmse"
    mae_col   = "mae_mean"   if "mae_mean"   in df.columns else "mae"

    descs  = df[desc_col].unique().tolist()
    models = df[model_col].unique().tolist()

    # 表头
    head = "".join(f"<th>{_esc(m)}</th>" for m in models)
    rows_html = ""
    for d in descs:
        cells = f"<th style='text-align:left'>{_esc(d)}</th>"
        for m in models:
            sub = df[(df[desc_col] == d) & (df[model_col] == m)]
            if sub.empty:
                cells += "<td>—</td>"
            else:
                row = sub.iloc[0]
                r2 = _float_or_nan(row.get(r2_col))
                rmse = row.get(rmse_col)
                mae = row.get(mae_col)
                bg = _color_r2(r2)
                fg = _text_color(r2)
                protocol = _scalar_text(row.get("evaluation_protocol"), "legacy_untracked")
                folds = _folds_display(row)
                oof = _yes_no(row.get("oof_complete")) if "oof_complete" in df.columns else "—"
                cells += (
                    f"<td style='background:{bg};color:{fg}'>"
                    f"<span class='cell-r2'>R²={_fmt(r2,3)}</span><br>"
                    f"<small>RMSE={_fmt_metric_with_std(rmse, row.get('rmse_std'), 4)}<br>"
                    f"MAE={_fmt_metric_with_std(mae, row.get('mae_std'), 4)}<br>"
                    f"evaluation_protocol={_esc(protocol)}<br>folds={_esc(folds)}；OOF={_esc(oof)}</small>"
                    f"</td>"
                )
        rows_html += f"<tr>{cells}</tr>\n"

    return f"""
<section>
<h2>1 · 描述符 × 模型结果矩阵</h2>
<p>单元格颜色：浅蓝（低 R²）→ 深绿（高 R²）；负 R² 为灰色。RMSE/MAE 以均值 ± 标准差显示；协议、fold 完成度与 OOF 完整性直接列入单元格。</p>
<table>
<thead><tr><th></th>{head}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>"""


def _section_glossary() -> str:
    """§14.5.2 固定指标解释文字（含 MathJax LaTeX 公式）。"""
    return r"""
<section>
<h2>2 · 指标解读与计算公式</h2>

<p><b>R²（决定系数）</b>：衡量模型预测方差占真实方差的比例，取值上限为 1。</p>
<div class="formula-block">
  \[R^2 = 1 - \frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})^2}\]
  <span style="font-size:0.82rem;color:#4b5563">\(y_i\)：真实值 &ensp; \(\hat{y}_i\)：预测值 &ensp; \(\bar{y}\)：真实值均值</span>
</div>
<ul style="margin:4px 0 12px 20px;line-height:1.8">
  <li>R² &gt; 0.85：预测可靠性较强，可用于辅助筛选实验条件</li>
  <li>R² 0.7～0.85：中等预测能力，趋势判断可参考，具体数值需谨慎</li>
  <li>R² &lt; 0.7 或负值：拟合效果弱；样本量极少时（如 5 折 CV 每折仅 2 个测试样本）负值属正常现象</li>
</ul>

<p><b>RMSE（均方根误差）</b>：对大误差样本更敏感，平方项会放大异常值。标签归一化到 [0,1] 时，单位等同于产率百分点。</p>
<div class="formula-block">
  \[\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}\]
  <span style="font-size:0.82rem;color:#4b5563">\(n\)：样本量</span>
</div>
<ul style="margin:4px 0 12px 20px;line-height:1.8">
  <li>RMSE &lt; 0.05：平均误差约 5 个百分点，接近实验重复性误差范围</li>
  <li>RMSE 0.05～0.10：中等误差，可区分高产率和低产率区间</li>
  <li>RMSE &gt; 0.10：误差偏大，不建议用于定量预测</li>
</ul>

<p><b>MAE（平均绝对误差）</b>：对每个样本的预测偏差取绝对值后平均，不受极端样本干扰，反映"典型单次预测"误差。</p>
<div class="formula-block">
  \[\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|\]
</div>
<ul style="margin:4px 0 12px 20px;line-height:1.8">
  <li>MAE &lt; 0.04：典型偏差极小，预测稳定性好</li>
  <li>MAE 0.04～0.08：中等偏差，结合 R² 综合评估</li>
  <li>MAE &gt; 0.08：典型偏差较大，预测结果存在系统性偏移风险</li>
</ul>

<p class="note">综合排名权重：
  \(S = R^2 \times 0.5 \;+\; \left(1 - \dfrac{\mathrm{RMSE}}{\mathrm{RMSE}_{\max}}\right) \times 0.3 \;+\; \left(1 - \dfrac{\mathrm{MAE}}{\mathrm{MAE}_{\max}}\right) \times 0.2\)
</p>
</section>"""


def _section_protocol_guard(metrics_df: pd.DataFrame) -> str:
    guard = ranking_eligibility_table(metrics_df)
    if guard.empty:
        return """
<section>
<h2>3 · 协议与排名守卫</h2>
<p class="note">没有可审计指标记录；推荐排名为空。</p>
</section>"""
    shown_columns = [
        "desc_name", "model_name", "evaluation_protocol", "cv", "expected_folds",
        "completed_folds", "folds", "oof_complete", "oof_n_observed", "oof_n_total",
        "rmse", "rmse_std", "mae", "mae_std",
        "autogluon_time_limit", "autogluon_presets", "autogluon_num_cpus",
        "autogluon_seed_policy", "ranking_eligible", "ranking_exclusion_reason",
    ]
    shown = guard.loc[:, [column for column in shown_columns if column in guard.columns]].copy()
    for column in ("oof_complete", "ranking_eligible"):
        if column in shown.columns:
            shown[column] = shown[column].map(lambda value: "是" if _truthy_bool(value) else "否")
    for column in ("rmse", "rmse_std", "mae", "mae_std"):
        if column in shown.columns:
            shown[column] = shown[column].map(lambda value: _fmt(value, 4))
    table_html = shown.to_html(index=False, escape=True)
    return f"""
<section>
<h2>3 · 协议与排名守卫</h2>
<p>推荐排名只纳入 <code>evaluation_protocol=outer_kfold</code>、<code>expected_folds=completed_folds=cv</code>、<code>oof_complete=True</code>，且 RMSE/MAE 均值与标准差均完整的条目。<code>autogluon_internal_holdout</code> 等旧 internal holdout 结果只展示，不参与排名。</p>
{table_html}
</section>"""


def _section_ranking(ranked: pd.DataFrame) -> str:
    if ranked.empty:
        return "<section><h2>4 · 推荐组合</h2><p>没有满足严格排名守卫的外层 CV 结果；请查看上一节的排除原因。</p></section>"

    has_time = "train_time_s" in ranked.columns and not pd.to_numeric(ranked["train_time_s"], errors="coerce").dropna().empty

    rows_html = ""
    for _, r in ranked.head(3).iterrows():
        medal = ["🥇", "🥈", "🥉"][int(r["rank"]) - 1]
        reason_text = _esc(r["reason"]) if r["reason"] else "综合指标较优"
        time_cell = f"<td>{_fmt_time(r['train_time_s'])}</td>" if has_time else ""
        rows_html += (
            f"<tr>"
            f"<td>{medal}</td>"
            f"<td><b>{_esc(r['desc_name'])}</b></td>"
            f"<td><b>{_esc(r['model_name'])}</b></td>"
            f"<td>{_esc(_scalar_text(r.get('evaluation_protocol'), '—'))}</td>"
            f"<td>{_esc(_scalar_text(r.get('folds'), '—'))}</td>"
            f"<td>{_yes_no(r.get('oof_complete'))}</td>"
            f"<td>{_fmt(r['r2'], 3)}</td>"
            f"<td>{_fmt_metric_with_std(r.get('rmse'), r.get('rmse_std'), 4)}</td>"
            f"<td>{_fmt_metric_with_std(r.get('mae'), r.get('mae_std'), 4)}</td>"
            f"<td>{_fmt(r['score'], 3)}</td>"
            f"{time_cell}"
            f"<td style='text-align:left'>{reason_text}</td>"
            f"</tr>\n"
        )

    all_rows = ""
    for _, r in ranked.iterrows():
        time_cell = f"<td>{_fmt_time(r['train_time_s'])}</td>" if has_time else ""
        all_rows += (
            f"<tr>"
            f"<td>{int(r['rank'])}</td>"
            f"<td>{_esc(r['desc_name'])}</td>"
            f"<td>{_esc(r['model_name'])}</td>"
            f"<td>{_esc(_scalar_text(r.get('evaluation_protocol'), '—'))}</td>"
            f"<td>{_esc(_scalar_text(r.get('folds'), '—'))}</td>"
            f"<td>{_yes_no(r.get('oof_complete'))}</td>"
            f"<td>{_fmt(r['r2'], 3)}</td>"
            f"<td>{_fmt_metric_with_std(r.get('rmse'), r.get('rmse_std'), 4)}</td>"
            f"<td>{_fmt_metric_with_std(r.get('mae'), r.get('mae_std'), 4)}</td>"
            f"<td>{_fmt(r['score'], 3)}</td>"
            f"{time_cell}"
            f"</tr>\n"
        )

    time_th = "<th>用时</th>" if has_time else ""

    return f"""
<section>
<h2>4 · 推荐组合（严格外层 CV 加权排名前三）</h2>
<table class="rank-tbl">
<thead><tr><th>名次</th><th>描述符</th><th>模型</th><th>evaluation_protocol</th><th>folds</th><th>OOF</th>
<th>R²</th><th>RMSE 均值±标准差</th><th>MAE 均值±标准差</th><th>综合分</th>{time_th}<th>推荐理由</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<details style="margin-top:12px">
  <summary style="cursor:pointer;color:#2563eb">展开完整严格排名</summary>
  <table style="margin-top:8px">
  <thead><tr><th>名次</th><th>描述符</th><th>模型</th><th>evaluation_protocol</th><th>folds</th><th>OOF</th>
  <th>R²</th><th>RMSE 均值±标准差</th><th>MAE 均值±标准差</th><th>综合分</th>{time_th}</tr></thead>
  <tbody>{all_rows}</tbody>
  </table>
</details>
</section>"""


def _section_column_mapping(task_info: Dict[str, Any]) -> str:
    """列映射与分类信息段落。"""
    column_mapping = task_info.get("column_mapping", [])
    if not column_mapping:
        return ""

    rows_html = ""
    for item in column_mapping:
        origin = _esc(str(item.get("origin_name", "—")))
        role = _esc(str(item.get("role", "—")))
        new_name = _esc(str(item.get("new_name", "—")))
        # 角色中文映射
        role_cn = {
            "label": "标签列",
            "reactant": "反应物",
            "product": "产物",
            "others": "其他分子",
            "condition": "反应条件"
        }.get(role, role)
        rows_html += (
            f"<tr>"
            f"<td style='text-align:left'>{origin}</td>"
            f"<td>{role_cn}</td>"
            f"<td style='text-align:left'><code>{new_name}</code></td>"
            f"</tr>\n"
        )

    return f"""
<section>
<h2>4 · 列映射与分类</h2>
<p style="font-size:0.85rem;color:#6b7280;margin-bottom:10px">
  原始数据集列名到规范名称的映射关系，以及每列的角色分类。
</p>
<table>
<thead><tr><th style="text-align:left">原始列名</th><th>角色</th><th style="text-align:left">规范名称</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>"""


def _section_descriptor_config(task_info: Dict[str, Any]) -> str:
    """描述符配置信息段落。"""
    descriptors = task_info.get("descriptors", [])
    if not descriptors:
        return ""

    rows_html = ""
    for desc in descriptors:
        desc_name = _esc(str(desc.get("descriptor", "—")))
        mode = _esc(str(desc.get("mode", "concat")))
        columns = desc.get("columns", [])
        columns_str = ", ".join(columns) if columns else "（全部SMILES列）"
        # 模式中文映射
        mode_cn = {
            "concat": "拼接",
            "sum": "求和",
            "reaction": "反应差分"
        }.get(mode, mode)
        rows_html += (
            f"<tr>"
            f"<td><b>{desc_name}</b></td>"
            f"<td>{mode_cn}</td>"
            f"<td style='text-align:left'><code>{_esc(columns_str)}</code></td>"
            f"</tr>\n"
        )

    return f"""
<section>
<h2>4.1 · 描述符配置</h2>
<p style="font-size:0.85rem;color:#6b7280;margin-bottom:10px">
  描述符向量化配置：每个描述符应用于哪些列，以及采用的拼接模式。
</p>
<table>
<thead><tr><th>描述符</th><th>模式</th><th style="text-align:left">应用列</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>"""


def _section_descriptor_precomputation(task_info: Dict[str, Any]) -> str:
    """Show descriptor artifact reuse/recompute/failure outcomes."""
    statuses = task_info.get("descriptor_statuses", [])
    if not statuses:
        return ""
    labels = {
        "computed": "新生成",
        "recomputed": "已重算",
        "reused": "已复用",
        "deferred": "按折拟合",
        "completed": "已完成",
        "failed": "失败",
    }
    rows_html = ""
    for item in statuses:
        status = str(item.get("status", "—"))
        rows_html += (
            "<tr>"
            f"<td><b>{_esc(item.get('feature_id', item.get('descriptor', '—')))}</b></td>"
            f"<td>{_esc(item.get('descriptor', '—'))}</td>"
            f"<td>{_esc(item.get('lifecycle', 'static_descriptor'))}</td>"
            f"<td>{_esc(labels.get(status, status))}</td>"
            f"<td>{_esc(item.get('stage', '—'))}</td>"
            f"<td>{_fmt(item.get('n_valid'))}/{_fmt(item.get('n_total'))}</td>"
            f"<td>{_fmt(item.get('feature_dim_min', item.get('feature_dim')))}–{_fmt(item.get('feature_dim_max', item.get('feature_dim')))}</td>"
            f"<td>{_fmt(item.get('completed_folds'))}</td>"
            f"<td>{_esc(item.get('skipped_model_count', 0))}</td>"
            f"<td style='text-align:left'>{_esc(item.get('reason', ''))}</td>"
            f"<td style='text-align:left'><code>{_esc(item.get('artifact_path', '—'))}</code></td>"
            "</tr>\n"
        )
    status_path = task_info.get("descriptor_status_path")
    status_note = (
        f"<p class='note'>完整机器可读状态：<code>{_esc(status_path)}</code></p>"
        if status_path else ""
    )
    return f"""
<section>
<h2>4.2 · 特征准备状态（描述符预计算状态）</h2>
<p class="note">静态描述符从持久化文件读取；fold_transform 只在训练折拟合并保存折级状态。单个特征失败时仅跳过其模型任务。</p>
<table>
<thead><tr><th>特征 ID</th><th>实现</th><th>生命周期</th><th>状态</th><th>阶段</th><th>有效/总样本</th><th>维度范围</th><th>完成折数</th><th>跳过模型数</th><th>原因</th><th>文件</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
{status_note}
</section>"""


def _section_data_paths(task_info: Dict[str, Any]) -> str:
    """数据路径信息段落。"""
    origin_path = task_info.get("origin_dataset_path")
    dataset_path = task_info.get("dataset_path") or task_info.get("csv_path")
    config_path = task_info.get("config_path")
    project_folder = task_info.get("project_folder")

    # 如果没有任何路径信息则跳过
    if not any([origin_path, config_path, project_folder]):
        return ""

    rows_html = ""

    # 项目文件夹（优先显示）
    if project_folder:
        rows_html += (
            f"<tr><th style='text-align:left'>项目文件夹</th>"
            f"<td style='text-align:left'><code>{_esc(str(project_folder))}</code></td></tr>"
        )

    # 原始数据集
    if origin_path:
        rows_html += (
            f"<tr><th style='text-align:left'>原始数据集</th>"
            f"<td style='text-align:left'><code>{_esc(str(origin_path))}</code></td></tr>"
        )

    # 规范化数据集
    if dataset_path and dataset_path != origin_path:
        rows_html += (
            f"<tr><th style='text-align:left'>规范化数据集</th>"
            f"<td style='text-align:left'><code>{_esc(str(dataset_path))}</code></td></tr>"
        )

    # 配置文件
    if config_path:
        rows_html += (
            f"<tr><th style='text-align:left'>配置文件</th>"
            f"<td style='text-align:left'><code>{_esc(str(config_path))}</code></td></tr>"
        )

    return f"""
<section>
<h2>4.3 · 数据路径</h2>
<table>
{rows_html}
</table>
</section>"""


def _section_scatter(out_dir: Path, scatter_paths: Optional[List[Path]] = None) -> str:
    """Render scatter images from an explicit run-scoped inventory when given.

    The legacy directory scan remains for the direct-CSV interface only.  The
    schema-2 post-processing service always supplies its own inventory so an
    old image under a shared output root cannot leak into a new report.
    """
    pics_dir = out_dir / "pictures"
    pngs = list(scatter_paths) if scatter_paths is not None else (sorted(pics_dir.glob("scatter_*.png")) if pics_dir.exists() else [])
    if not pngs:
        return ""
    imgs = ""
    for p in pngs:
        b64 = _b64_png(p)
        if b64:
            label = _esc(p.stem.replace("scatter_", "").replace("_", " × "))
            imgs += (
                f'<figure style="text-align:center">'
                f'<img src="data:image/png;base64,{b64}" alt="{label}"/>'
                f'<figcaption style="font-size:.8rem;color:#6b7280">{label}</figcaption>'
                f'</figure>\n'
            )
    return f"""
<section>
<h2>6 · 散点图画廊（真实值 vs 预测值）</h2>
<p style="font-size:0.85rem;color:#6b7280;margin-bottom:10px">
  每张图对应一个（描述符 × 模型）组合，横轴为真实标签，纵轴为模型预测值，虚线为 y = x 理想参考线。
  点越靠近虚线表示预测越准确；图内标注了该组合的 R²、RMSE 及样本量。
</p>
<div class="scatter-grid">{imgs}</div>
</section>"""


# ──────────────────────────── 主入口 ─────────────────────────────────────────── #

def generate_report(
    metrics_df: pd.DataFrame,
    task_info: Dict[str, Any],
    out_dir: Path,
    filename: str = "report.html",
    scatter_paths: Optional[List[Path]] = None,
) -> Path:
    """生成自包含 HTML 报告。

    Args:
        metrics_df: 包含各 (描述符, 模型) 评估结果的 DataFrame。
                    必须含 descriptor/desc_name, model/model_name,
                    r2_mean/r2, rmse_mean/rmse, mae_mean/mae 列。
        task_info:  任务元数据字典，键包括：
                    task_name, csv_path, n_samples,
                    smiles_cols, numeric_cols, label_col, n_combinations。
        out_dir:    输出根目录；图片位于 pictures/，报告位于 report/。
        filename:   输出 HTML 文件名（默认 report.html）。

    Returns:
        生成的 HTML 文件路径。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pictures").mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ranked = rank_combinations(metrics_df)

    from yonod.diagnostics import summarize_diagnostic_rows
    diagnostic_rows = task_info.get("diagnostic_rows") or []
    diagnostic_summary = summarize_diagnostic_rows(diagnostic_rows)
    diagnostic_section = "<section><h2>训练与外层评估诊断</h2><p>训练分数是已见样本上的拟合诊断，不代表泛化；基线常数取每折 outer-train 标签均值。R² 差距为训练减外层，RMSE/MAE 差距为外层减训练。缺失状态不会参与外层排名。</p>"
    diagnostic_section += (pd.DataFrame(diagnostic_rows).to_html(index=False, escape=True, classes="data-table")
                           if diagnostic_rows else "<p>未记录训练诊断。</p>") + "</section>"
    if diagnostic_summary:
        diagnostic_section += "<section><h2>训练诊断折间汇总</h2><p>按 repeat 单列；标准差使用 ddof=0，各指标分别披露有效折数。原普通 OOF 仍按每个 repeat 汇总。</p>" + pd.DataFrame(diagnostic_summary).to_html(index=False, escape=True, classes="data-table") + "</section>"
    body = (
        _section_intro(task_info, now)
        + _section_numeric_audit(task_info)
        + _section_hpo(task_info)
        + diagnostic_section
        + _section_grid(metrics_df)
        + _section_glossary()
        + _section_protocol_guard(metrics_df)
        + _section_ranking(ranked)
        + _section_column_mapping(task_info)
        + _section_descriptor_config(task_info)
        + _section_descriptor_precomputation(task_info)
        + _section_data_paths(task_info)
        + _section_modeling_time(metrics_df, out_dir)
        + _section_scatter(out_dir, scatter_paths)
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>YONOD 报告 — {_esc(str(task_info.get('task_name', '')))} </title>
{_CSS}
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
{body}
<p class="note" style="text-align:center;margin-top:24px">
由 YONOD report.py 自动生成 · {now}
</p>
</body>
</html>"""

    out_path = report_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_markdown_report(
    metrics_df: pd.DataFrame,
    task_info: Dict[str, Any],
    out_dir: Path,
    filename: str = "report.md",
    scatter_paths: Optional[List[Path]] = None,
) -> Path:
    """生成 Markdown 格式报告，适合二次编辑和版本管理。

    包含任务信息、数据集来源、结果矩阵、详细指标、推荐排名和指标说明。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pictures").mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    now       = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = str(task_info.get("task_name", "—"))

    # 统一列名
    desc_col  = "desc_name"  if "desc_name"  in metrics_df.columns else "descriptor"
    model_col = "model_name" if "model_name" in metrics_df.columns else "model"
    r2_col    = "r2_mean"    if "r2_mean"    in metrics_df.columns else "r2"
    rmse_col  = "rmse_mean"  if "rmse_mean"  in metrics_df.columns else "rmse"
    mae_col   = "mae_mean"   if "mae_mean"   in metrics_df.columns else "mae"

    def _fv(v: Any, d: int = 4) -> str:
        try:
            number = float(v)
        except (TypeError, ValueError):
            return "—"
        if not math.isfinite(number):
            return "—"
        return f"{number:.{d}f}"

    def _md(v: Any) -> str:
        if v is None:
            return "—"
        try:
            if pd.isna(v):
                return "—"
        except (TypeError, ValueError):
            pass
        return str(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def _md_metric_with_std(row: Any, mean_col: str, std_col: str, digits: int = 4) -> str:
        mean_text = _fv(row.get(mean_col), digits)
        std_text = _fv(row.get(std_col), digits)
        if mean_text == "—":
            return "—"
        return mean_text if std_text == "—" else f"{mean_text} ± {std_text}"

    def _md_autogluon_params(row: Any) -> str:
        return _md(_autogluon_param_summary(row))

    lines: List[str] = []

    # ── 标题 ─────────────────────────────────────────────────────────────────
    lines += [
        f"# YONOD 建模报告 — {task_name}",
        "",
        f"> 生成时间：{now}  ",
        f"> 评估组合数：{task_info.get('n_combinations', '—')}",
        "",
        "---",
        "",
    ]

    numeric_rows = task_info.get("numeric_audit_rows") or []
    if numeric_rows:
        lines += [
            "## 数值输入与折内处理", "",
            "以下记录来自已保存且校验的折级状态。", "",
            "| 组合 | repeat | fold | 源列 | 输出名 | 单位 | 缺失策略 | 缩放 | 训练缺失行 | 训练常量 | 数值维度 | 送模总维度 |",
            "|---|---:|---:|---|---|---|---|---|---:|---|---:|---:|",
        ]
        for row in numeric_rows:
            fields = [row.get(key, "—") for key in ("run", "repeat", "fold", "source", "name", "unit", "missing", "scaling", "train_missing_rows", "constant_train", "numeric_dimension", "input_dimension")]
            lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in fields) + " |")
        lines.append("")

    lines.extend(_markdown_hpo(task_info))
    diagnostic_rows = task_info.get("diagnostic_rows") or []
    lines += ["## 训练与外层评估诊断", "",
              "训练分数是已见样本上的拟合诊断，不代表泛化。基线常数取每折 outer-train 标签均值；R² 差距为训练减外层，RMSE/MAE 差距为外层减训练。缺失状态不进入外层排名。", ""]
    if diagnostic_rows:
        columns = list(dict.fromkeys(key for row in diagnostic_rows for key in row))
        lines += ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
        for row in diagnostic_rows:
            lines.append("| " + " | ".join(_md(row.get(key)) for key in columns) + " |")
        lines.append("")
        from yonod.diagnostics import summarize_diagnostic_rows
        summaries = summarize_diagnostic_rows(diagnostic_rows)
        summary_columns = list(summaries[0])
        lines += ["### 折间汇总", "", "按 repeat 单列；标准差使用 ddof=0，各指标分别披露有效折数。原普通 OOF 仍按每个 repeat 汇总。", "",
                  "| " + " | ".join(summary_columns) + " |", "|" + "|".join("---" for _ in summary_columns) + "|"]
        for row in summaries:
            lines.append("| " + " | ".join(_md(row.get(key)) for key in summary_columns) + " |")
        lines.append("")
    else:
        lines += ["未记录训练诊断。", ""]

    # ── 任务信息 ─────────────────────────────────────────────────────────────
    lines += [
        "## 任务信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 任务名称 | {task_name} |",
        f"| 项目文件夹 | `{task_info.get('project_folder', '—')}` |",
        f"| CSV 路径 | `{task_info.get('csv_path', '—')}` |",
        f"| 有效样本量 | {task_info.get('n_samples', '—')} |",
        f"| SMILES 列 | `{task_info.get('smiles_cols', '—')}` |",
        f"| 数值辅助列 | `{task_info.get('numeric_cols', '（无）')}` |",
        f"| 标签列 | `{task_info.get('label_col', '—')}` |",
        "",
    ]

    # ── 数据集来源（可选）───────────────────────────────────────────────────
    citation = task_info.get("dataset_citation")
    url      = task_info.get("dataset_url")
    notes    = task_info.get("dataset_notes")
    if any([citation, url, notes]):
        lines += ["---", "", "## 数据集来源", "", "| 项目 | 信息 |", "|---|---|"]
        if citation:
            lines.append(f"| 文献引用 | {citation} |")
        if url:
            lines.append(f"| 开源地址 | {url} |")
        if notes:
            lines.append(f"| 备注 | {notes} |")
        lines.append("")

    # ── 结果矩阵（R²）────────────────────────────────────────────────────────
    lines += ["---", "", "## 结果矩阵（R²）", ""]
    if metrics_df.empty:
        lines += ["没有有效模型结果；请查看下方“描述符预计算状态”中的失败原因。"]
    else:
        descs  = metrics_df[desc_col].unique().tolist()
        models = metrics_df[model_col].unique().tolist()
        header = "| 描述符 \\ 模型 | " + " | ".join(models) + " |"
        sep    = "|---|" + "---|" * len(models)
        lines += [header, sep]
        for d in descs:
            cells = []
            for m in models:
                sub = metrics_df[(metrics_df[desc_col] == d) & (metrics_df[model_col] == m)]
                if sub.empty:
                    cells.append("—")
                else:
                    cells.append(_fv(sub.iloc[0].get(r2_col, float("nan"))))
            lines.append("| " + d + " | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 详细指标 ─────────────────────────────────────────────────────────────
    has_time = "train_time_s" in metrics_df.columns
    time_th = " 用时 |" if has_time else ""
    time_sep = "---|" if has_time else ""

    lines += ["---", "", "## 详细指标", ""]
    if metrics_df.empty:
        lines.append("无有效指标记录。")
    else:
        lines.append(f"| 描述符 | 模型 | evaluation_protocol | folds | oof_complete | R² 均值 | R² 标准差 | RMSE 均值 | RMSE 标准差 | MAE 均值 | MAE 标准差 | AutoGluon关键参数 | seed policy |{time_th}")
        lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|{time_sep}")
        for _, row in metrics_df.iterrows():
            protocol = _md(row.get("evaluation_protocol", "legacy_untracked"))
            folds = _md(_folds_display(row))
            oof = _yes_no(row.get("oof_complete")) if "oof_complete" in metrics_df.columns else "—"
            seed_policy = _md(row.get("autogluon_seed_policy", "—"))
            t_cell = f" {_fmt_time(row.get('train_time_s'))} |" if has_time else ""
            lines.append(
                f"| {_md(row[desc_col])} | {_md(row[model_col])}"
                f" | {protocol} | {folds} | {oof}"
                f" | {_fv(row.get(r2_col))} | {_fv(row.get('r2_std'))}"
                f" | {_fv(row.get(rmse_col))} | {_fv(row.get('rmse_std'))}"
                f" | {_fv(row.get(mae_col))} | {_fv(row.get('mae_std'))}"
                f" | {_md_autogluon_params(row)} | {seed_policy} |{t_cell}"
            )
    lines.append("")

    # ── 协议与排名守卫 ───────────────────────────────────────────────────────
    guard = ranking_eligibility_table(metrics_df)
    lines += [
        "---", "", "## 协议与排名守卫", "",
        "推荐排名只纳入 `evaluation_protocol=outer_kfold`、`expected_folds=completed_folds=cv`、`oof_complete=True`，且 RMSE/MAE 均值与标准差均完整的条目；`autogluon_internal_holdout` 只展示，不参与排名。",
        "",
    ]
    if guard.empty:
        lines.append("无可审计指标记录。")
    else:
        lines += [
            "| 描述符 | 模型 | evaluation_protocol | cv | expected_folds | completed_folds | folds | oof_complete | OOF观测/总数 | RMSE | RMSE标准差 | MAE | MAE标准差 | autogluon_time_limit | autogluon_presets | autogluon_num_cpus | autogluon_seed_policy | 排名资格 | 排除原因 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, item in guard.iterrows():
            observed = item.get("oof_n_observed")
            total = item.get("oof_n_total")
            oof_counts = "—" if pd.isna(observed) or pd.isna(total) else f"{int(observed)}/{int(total)}"
            lines.append(
                f"| {_md(item.get('desc_name'))} | {_md(item.get('model_name'))}"
                f" | {_md(item.get('evaluation_protocol'))} | {_md(item.get('cv'))}"
                f" | {_md(item.get('expected_folds'))} | {_md(item.get('completed_folds'))}"
                f" | {_md(item.get('folds'))} | {_yes_no(item.get('oof_complete'))}"
                f" | {oof_counts} | {_fv(item.get('rmse'))} | {_fv(item.get('rmse_std'))}"
                f" | {_fv(item.get('mae'))} | {_fv(item.get('mae_std'))}"
                f" | {_md(item.get('autogluon_time_limit'))} | {_md(item.get('autogluon_presets'))}"
                f" | {_md(item.get('autogluon_num_cpus'))} | {_md(item.get('autogluon_seed_policy'))}"
                f" | {_yes_no(item.get('ranking_eligible'))} | {_md(item.get('ranking_exclusion_reason'))} |"
            )
    lines.append("")

    # ── 推荐组合 ─────────────────────────────────────────────────────────────
    ranked = rank_combinations(metrics_df)
    lines += ["---", "", "## 推荐组合（严格外层 CV 加权排名）", ""]
    if not ranked.empty:
        has_rk_time = "train_time_s" in ranked.columns and not pd.to_numeric(ranked["train_time_s"], errors="coerce").dropna().empty
        rk_time_th = " 用时 |" if has_rk_time else ""
        rk_time_sep = "---|" if has_rk_time else ""
        rk_header = f"| 名次 | 描述符 | 模型 | evaluation_protocol | folds | OOF | R² | RMSE 均值±标准差 | MAE 均值±标准差 | 综合分 |{rk_time_th} 推荐理由 |"
        rk_sep = f"|---|---|---|---|---|---|---|---|---|---|{rk_time_sep}---|"
        lines += [rk_header, rk_sep]
        for _, r in ranked.head(3).iterrows():
            medal = ["🥇", "🥈", "🥉"][int(r["rank"]) - 1]
            t_cell = f" {_fmt_time(r['train_time_s'])} |" if has_rk_time else ""
            reason = r.get("reason") or "综合指标较优"
            lines.append(
                f"| {medal} {int(r['rank'])} | **{_md(r['desc_name'])}** | **{_md(r['model_name'])}**"
                f" | {_md(r.get('evaluation_protocol'))} | {_md(r.get('folds'))} | {_yes_no(r.get('oof_complete'))}"
                f" | {_fv(r['r2'], 3)} | {_md_metric_with_std(r, 'rmse', 'rmse_std', 4)} | {_md_metric_with_std(r, 'mae', 'mae_std', 4)}"
                f" | {_fv(r['score'], 3)} |{t_cell} {_md(reason)} |"
            )
        lines += ["", "<details>", "<summary>展开完整严格排名</summary>", "", rk_header, rk_sep]
        for _, r in ranked.iterrows():
            t_cell = f" {_fmt_time(r['train_time_s'])} |" if has_rk_time else ""
            lines.append(
                f"| {int(r['rank'])} | {_md(r['desc_name'])} | {_md(r['model_name'])}"
                f" | {_md(r.get('evaluation_protocol'))} | {_md(r.get('folds'))} | {_yes_no(r.get('oof_complete'))}"
                f" | {_fv(r['r2'], 3)} | {_md_metric_with_std(r, 'rmse', 'rmse_std', 4)} | {_md_metric_with_std(r, 'mae', 'mae_std', 4)}"
                f" | {_fv(r['score'], 3)} |{t_cell}|"
            )
        lines += ["", "</details>", ""]
    else:
        lines += ["没有满足严格排名守卫的外层 CV 结果；请查看上一节的排除原因。", ""]

    # ── 建模耗时与成本对比 ───────────────────────────────────────────────────
    time_summary = _universal_time_summary(metrics_df)
    descriptor_time_summary = _universal_dimension_time_summary(time_summary, "descriptor")
    model_time_summary = _universal_dimension_time_summary(time_summary, "model")
    time_figures = {
        "combination": _write_universal_time_figure(time_summary, out_dir, dimension="combination") if "train_time_s" in metrics_df.columns else None,
        "descriptor": _write_universal_time_figure(descriptor_time_summary, out_dir, dimension="descriptor") if "train_time_s" in metrics_df.columns else None,
        "model": _write_universal_time_figure(model_time_summary, out_dir, dimension="model") if "train_time_s" in metrics_df.columns else None,
    }
    lines += ["---", "", "## 建模耗时与成本对比", ""]
    if "train_time_s" not in metrics_df.columns:
        lines += ["未提供组合级 `train_time_s`，无法绘制建模耗时图；报告未使用空值或零值代替耗时。", ""]
    else:
        lines += [
            "本路径仅汇总输入 `metrics_df.train_time_s`，未提供预测时间，故图表仅表示训练时间，不表示训练加预测时间或 CLI 端到端墙钟时间。描述符和建模方法图是组合时间的两种汇总视图，不与组合图相加；共享描述符特征化时间未提供，因此不重复归因。异常或缺失时间标为不可比较，不以零秒显示。",
            "",
            "### 描述符 × 建模方法组合",
            "",
            "| 描述符 | 模型 | 报告记录数 | 组合训练时间 | 可比较 | 时间状态 |",
            "|---|---|---|---|---|---|",
        ]
        if time_summary.empty:
            lines += ["| — | — | — | — | 否 | 无可用记录 |"]
        else:
            for _, row in time_summary.iterrows():
                lines.append(
                    "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                        row["descriptor"], row["model"], int(row["reported_rows"]),
                        _fmt_time(row["total_train_time_s"]), "是" if bool(row["is_time_comparable"]) else "否",
                        row["time_status"],
                    )
                )
        if time_figures["combination"]:
            lines += ["", "![组合训练时间柱状图](../pictures/{0})".format(time_figures["combination"].name)]
        else:
            lines += ["", "> 没有可比较的有限非负训练时间，因此未生成零高柱状图。"]
        for title, frame, figure, label in (
            ("按描述符汇总", descriptor_time_summary, time_figures["descriptor"], "描述符累计训练时间柱状图"),
            ("按建模方法汇总", model_time_summary, time_figures["model"], "建模方法累计训练时间柱状图"),
        ):
            lines += [
                "", "### {0}".format(title), "",
                "每根柱为该维度下所有完整且时间可比较组合的累计训练时间。",
                "",
                "| 项目 | 预期组合数 | 可比较组合数 | 不可比较组合数 | 累计训练时间 | 可比较 | 时间状态 |",
                "|---|---|---|---|---|---|---|",
            ]
            if frame.empty:
                lines += ["| — | — | — | — | — | 否 | 无可用记录 |"]
            else:
                for _, row in frame.iterrows():
                    lines.append(
                        "| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
                            row["item"], int(row["expected_combinations"]), int(row["comparable_combinations"]),
                            int(row["noncomparable_combinations"]), _fmt_time(row["total_train_time_s"]),
                            "是" if bool(row["is_time_comparable"]) else "否", row["time_status"],
                        )
                    )
            if figure:
                lines += ["", "![{0}](../pictures/{1})".format(label, figure.name)]
            else:
                lines += ["", "> 没有可比较的有限非负训练时间，因此未生成零高柱状图。"]
        lines.append("")

    # ── 列映射与分类 ─────────────────────────────────────────────────────────
    column_mapping = task_info.get("column_mapping", [])
    if column_mapping:
        role_cn_map = {
            "label": "标签列",
            "reactant": "反应物",
            "product": "产物",
            "others": "其他分子",
            "condition": "反应条件"
        }
        lines += ["---", "", "## 列映射与分类", ""]
        lines += ["| 原始列名 | 角色 | 规范名称 |", "|---|---|---|"]
        for item in column_mapping:
            origin = str(item.get("origin_name", "—"))
            role = str(item.get("role", "—"))
            new_name = str(item.get("new_name", "—"))
            role_cn = role_cn_map.get(role, role)
            lines.append(f"| {origin} | {role_cn} | `{new_name}` |")
        lines.append("")

    # ── 描述符配置 ───────────────────────────────────────────────────────────
    descriptors = task_info.get("descriptors", [])
    if descriptors:
        mode_cn_map = {
            "concat": "拼接",
            "sum": "求和",
            "reaction": "反应差分"
        }
        lines += ["---", "", "## 描述符配置", ""]
        lines += ["| 描述符 | 模式 | 应用列 |", "|---|---|---|"]
        for desc in descriptors:
            desc_name = str(desc.get("descriptor", "—"))
            mode = str(desc.get("mode", "concat"))
            columns = desc.get("columns", [])
            columns_str = ", ".join(columns) if columns else "（全部SMILES列）"
            mode_cn = mode_cn_map.get(mode, mode)
            lines.append(f"| **{desc_name}** | {mode_cn} | `{columns_str}` |")
        lines.append("")

    # ── 描述符预计算状态 ───────────────────────────────────────────────────
    descriptor_statuses = task_info.get("descriptor_statuses", [])
    if descriptor_statuses:
        status_labels = {
            "computed": "新生成",
            "recomputed": "已重算",
            "reused": "已复用",
            "deferred": "按折拟合",
            "completed": "已完成",
            "failed": "失败",
        }
        lines += [
            "---", "", "## 特征准备状态（描述符预计算状态）", "",
            "静态描述符从持久化文件读取；fold_transform 只在训练折拟合并保存折级状态。单个特征失败时仅跳过其模型任务。", "",
            "| 特征 ID | 实现 | 生命周期 | 状态 | 阶段 | 有效/总样本 | 维度范围 | 完成折数 | 跳过模型数 | 原因 | 文件 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for item in descriptor_statuses:
            status = str(item.get("status", "—"))
            lines.append(
                "| {feature_id} | {descriptor} | {lifecycle} | {status} | {stage} | {n_valid}/{n_total} | {feature_dim_min}–{feature_dim_max} | {completed_folds} | {skipped} | {reason} | `{path}` |".format(
                    feature_id=_md(item.get("feature_id", item.get("descriptor", "—"))),
                    descriptor=_md(item.get("descriptor", "—")),
                    lifecycle=_md(item.get("lifecycle", "static_descriptor")),
                    status=_md(status_labels.get(status, status)),
                    stage=_md(item.get("stage", "—")),
                    n_valid=_md(item.get("n_valid", "—")),
                    n_total=_md(item.get("n_total", "—")),
                    feature_dim_min=_md(item.get("feature_dim_min", item.get("feature_dim", "—"))),
                    feature_dim_max=_md(item.get("feature_dim_max", item.get("feature_dim", "—"))),
                    completed_folds=_md(item.get("completed_folds", "—")),
                    skipped=_md(item.get("skipped_model_count", 0)),
                    reason=_md(item.get("reason", "")),
                    path=_md(item.get("artifact_path", "—")),
                )
            )
        status_path = task_info.get("descriptor_status_path")
        if status_path:
            lines += ["", f"完整机器可读状态：`{_md(status_path)}`"]
        lines.append("")

    # ── 数据路径 ─────────────────────────────────────────────────────────────
    origin_path = task_info.get("origin_dataset_path")
    dataset_path = task_info.get("dataset_path") or task_info.get("csv_path")
    config_path = task_info.get("config_path")
    if any([origin_path, config_path]):
        lines += ["---", "", "## 数据路径", ""]
        lines += ["| 项目 | 路径 |", "|---|---|"]
        if origin_path:
            lines.append(f"| 原始数据集 | `{origin_path}` |")
        if dataset_path and dataset_path != origin_path:
            lines.append(f"| 规范化数据集 | `{dataset_path}` |")
        if config_path:
            lines.append(f"| 配置文件 | `{config_path}` |")
        lines.append("")

    # ── OOF 散点图（schema-2 supplies an explicit inventory) ───────────────
    markdown_scatters = list(scatter_paths) if scatter_paths is not None else sorted((out_dir / "pictures").glob("scatter_*.png"))
    if markdown_scatters:
        lines += ["---", "", "## OOF 预测散点图", ""]
        for picture in markdown_scatters:
            # Reports live in ``root/report`` and images in ``root/pictures``.
            # Use that stable sibling layout instead of an absolute path, so a
            # copied report bundle remains valid on Windows and Linux.
            relative = Path("..") / "pictures" / picture.name
            label = _md(picture.stem.replace("scatter_", "").replace("_", " × "))
            lines += [f"### {label}", "", f"![{label}]({relative.as_posix()})", ""]

    # ── 指标说明 ─────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 指标说明",
        "",
        "**R²（决定系数）**：衡量模型预测方差占真实方差的比例，取值上限为 1。",
        "",
        "- R² > 0.85：预测可靠性较强，可用于辅助筛选实验条件",
        "- R² 0.7～0.85：中等预测能力，趋势判断可参考，具体数值需谨慎",
        "- R² < 0.7 或负值：拟合效果弱；样本量极少时负值属正常现象",
        "",
        "**RMSE（均方根误差）**：对大误差样本更敏感。标签归一化到 [0,1] 时，单位等同于产率百分点。",
        "",
        "- RMSE < 0.05：平均误差约 5 个百分点，接近实验重复性误差范围",
        "- RMSE 0.05～0.10：中等误差，可区分高产率和低产率区间",
        "- RMSE > 0.10：误差偏大，不建议用于定量预测",
        "",
        "**MAE（平均绝对误差）**：对每个样本的预测偏差取绝对值后平均。",
        "",
        "- MAE < 0.04：典型偏差极小，预测稳定性好",
        "- MAE 0.04～0.08：中等偏差，结合 R² 综合评估",
        "- MAE > 0.08：典型偏差较大",
        "",
        "**综合评分**：S = R² × 0.5 + (1 − RMSE / RMSE_max) × 0.3 + (1 − MAE / MAE_max) × 0.2",
        "",
        "---",
        "",
        f"*由 YONOD report.py 自动生成 · {now}*",
    ]

    out_path = report_dir / filename
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
