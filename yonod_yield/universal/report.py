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
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ─────────────────────────────── 加权排名 ───────────────────────────────────── #

def rank_combinations(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """按 §14.5.3 公式对 (描述符, 模型) 组合加权排名。

    Args:
        metrics_df: 含列 desc_name / descriptor, model_name / model,
                    r2 / r2_mean, rmse / rmse_mean, mae / mae_mean 的 DataFrame。

    Returns:
        含 rank / desc_name / model_name / r2 / rmse / mae / score / reason 的 DataFrame，
        按 score 降序排列。
    """
    df = metrics_df.copy()

    # 列名兼容：run_yonod.py 输出 "descriptor"/"model"/"r2_mean" 等
    def _col(candidates: List[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"期望列之一 {candidates} 不存在于 DataFrame 中。")

    desc_col  = _col(["desc_name", "descriptor"])
    model_col = _col(["model_name", "model"])
    r2_col    = _col(["r2", "r2_mean"])
    rmse_col  = _col(["rmse", "rmse_mean"])
    mae_col   = _col(["mae", "mae_mean"])

    df = df.rename(columns={
        desc_col: "desc_name", model_col: "model_name",
        r2_col: "r2", rmse_col: "rmse", mae_col: "mae",
    })

    # 过滤掉 NaN 行
    df = df.dropna(subset=["r2", "rmse", "mae"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["rank", "desc_name", "model_name",
                                     "r2", "rmse", "mae", "score", "reason"])

    max_rmse = df["rmse"].max()
    max_mae  = df["mae"].max()

    df["score"] = (
        df["r2"]                            * 0.5
        + (1 - df["rmse"] / max(max_rmse, 1e-9)) * 0.3
        + (1 - df["mae"]  / max(max_mae,  1e-9)) * 0.2
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
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
        df.at[i, "reason"] = "；".join(parts)

    return df[["rank", "desc_name", "model_name", "r2", "rmse", "mae", "score", "reason"]]


# ─────────────────────────────── HTML 辅助 ──────────────────────────────────── #

def _esc(v: Any) -> str:
    return _html.escape(str(v))


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return _esc(str(v))


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
</table>
</section>"""


def _section_grid(df: pd.DataFrame) -> str:
    """4×N 描述符 × 模型矩阵，单元格显示 R²/RMSE/MAE。"""
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
                r2   = float(row.get(r2_col,   float("nan")))
                rmse = float(row.get(rmse_col, float("nan")))
                mae  = float(row.get(mae_col,  float("nan")))
                bg   = _color_r2(r2)
                fg   = _text_color(r2)
                cells += (
                    f"<td style='background:{bg};color:{fg}'>"
                    f"<span class='cell-r2'>R²={_fmt(r2,3)}</span><br>"
                    f"<small>RMSE={_fmt(rmse,4)}<br>MAE={_fmt(mae,4)}</small>"
                    f"</td>"
                )
        rows_html += f"<tr>{cells}</tr>\n"

    return f"""
<section>
<h2>1 · 描述符 × 模型结果矩阵</h2>
<p>单元格颜色：浅蓝（低 R²）→ 深绿（高 R²）；负 R² 为灰色。</p>
<table>
<thead><tr><th></th>{head}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>"""


def _section_glossary() -> str:
    """§14.5.2 固定指标解释文字。"""
    return """
<section>
<h2>2 · 指标解读（化学实验直觉版）</h2>
<p><b>R²（决定系数）</b>：取值 0～1，越接近 1 表示模型对标签变化的解释能力越强。</p>
<ul style="margin:8px 0 8px 20px;line-height:1.8">
  <li>R² &gt; 0.85：模型具有较强的预测可靠性，可用于辅助实验设计</li>
  <li>R² 0.7～0.85：中等预测能力，趋势判断可参考，具体数值需谨慎</li>
  <li>R² &lt; 0.7：模型对该描述符/模型组合的拟合效果较弱</li>
</ul>
<p><b>RMSE（均方根误差）</b>：与标签量纲相同（yield 为 0~1 浮点时单位为"产率"）。</p>
<ul style="margin:8px 0 8px 20px;line-height:1.8">
  <li>RMSE &lt; 0.05：平均预测误差约 5 个百分点，接近实验重复性误差范围</li>
  <li>RMSE 0.05～0.10：中等误差，可区分高产率和低产率区间</li>
  <li>RMSE &gt; 0.10：误差较大，不建议用于定量预测</li>
</ul>
<p><b>MAE（平均绝对误差）</b>：比 RMSE 对异常值更鲁棒，反映典型单样本的预测偏差。</p>
<p class="note">综合排名权重：R²×0.5 + (1−RMSE/max_RMSE)×0.3 + (1−MAE/max_MAE)×0.2</p>
</section>"""


def _section_ranking(ranked: pd.DataFrame) -> str:
    if ranked.empty:
        return "<section><h2>3 · 推荐组合</h2><p>无有效结果。</p></section>"

    rows_html = ""
    for _, r in ranked.head(3).iterrows():
        medal = ["🥇", "🥈", "🥉"][int(r["rank"]) - 1]
        reason_text = _esc(r["reason"]) if r["reason"] else "综合指标较优"
        rows_html += (
            f"<tr>"
            f"<td>{medal}</td>"
            f"<td><b>{_esc(r['desc_name'])}</b></td>"
            f"<td><b>{_esc(r['model_name'])}</b></td>"
            f"<td>{_fmt(r['r2'], 3)}</td>"
            f"<td>{_fmt(r['rmse'], 4)}</td>"
            f"<td>{_fmt(r['mae'], 4)}</td>"
            f"<td>{_fmt(r['score'], 3)}</td>"
            f"<td style='text-align:left'>{reason_text}</td>"
            f"</tr>\n"
        )

    # 完整排名（可折叠）
    all_rows = ""
    for _, r in ranked.iterrows():
        all_rows += (
            f"<tr>"
            f"<td>{int(r['rank'])}</td>"
            f"<td>{_esc(r['desc_name'])}</td>"
            f"<td>{_esc(r['model_name'])}</td>"
            f"<td>{_fmt(r['r2'], 3)}</td>"
            f"<td>{_fmt(r['rmse'], 4)}</td>"
            f"<td>{_fmt(r['mae'], 4)}</td>"
            f"<td>{_fmt(r['score'], 3)}</td>"
            f"</tr>\n"
        )

    return f"""
<section>
<h2>3 · 推荐组合（加权排名前三）</h2>
<table class="rank-tbl">
<thead><tr><th>名次</th><th>描述符</th><th>模型</th>
<th>R²</th><th>RMSE</th><th>MAE</th><th>综合分</th><th>推荐理由</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<details style="margin-top:12px">
  <summary style="cursor:pointer;color:#2563eb">展开完整排名</summary>
  <table style="margin-top:8px">
  <thead><tr><th>名次</th><th>描述符</th><th>模型</th>
  <th>R²</th><th>RMSE</th><th>MAE</th><th>综合分</th></tr></thead>
  <tbody>{all_rows}</tbody>
  </table>
</details>
</section>"""


def _section_scatter(out_dir: Path) -> str:
    pngs = sorted(out_dir.glob("scatter_*.png"))
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
<h2>4 · 散点图画廊</h2>
<div class="scatter-grid">{imgs}</div>
</section>"""


# ──────────────────────────── 主入口 ─────────────────────────────────────────── #

def generate_report(
    metrics_df: pd.DataFrame,
    task_info: Dict[str, Any],
    out_dir: Path,
    filename: str = "report.html",
) -> Path:
    """生成自包含 HTML 报告。

    Args:
        metrics_df: 包含各 (描述符, 模型) 评估结果的 DataFrame。
                    必须含 descriptor/desc_name, model/model_name,
                    r2_mean/r2, rmse_mean/rmse, mae_mean/mae 列。
        task_info:  任务元数据字典，键包括：
                    task_name, csv_path, n_samples,
                    smiles_cols, numeric_cols, label_col, n_combinations。
        out_dir:    输出目录（同时扫描其中的 scatter_*.png）。
        filename:   输出 HTML 文件名（默认 report.html）。

    Returns:
        生成的 HTML 文件路径。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ranked = rank_combinations(metrics_df)

    body = (
        _section_intro(task_info, now)
        + _section_grid(metrics_df)
        + _section_glossary()
        + _section_ranking(ranked)
        + _section_scatter(out_dir)
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>YONOD 报告 — {_esc(str(task_info.get('task_name', '')))} </title>
{_CSS}
</head>
<body>
{body}
<p class="note" style="text-align:center;margin-top:24px">
由 YONOD report.py 自动生成 · {now}
</p>
</body>
</html>"""

    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path
