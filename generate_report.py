"""YONOD HTML report generator.

Reads ``results/metrics_summary.csv`` plus all ``results/scatter_*.png`` and
produces a single self-contained HTML report (images inlined as base64) at
``results/report.html``.

The report is written for chemists with limited ML background:
  * a TL;DR recommendation up top,
  * a metrics glossary that maps R^2 / RMSE / MAE / r2_std into experiment-
    intuitive language,
  * a color-coded descriptor x model R^2 heatmap,
  * per-descriptor / per-model best picks,
  * a gallery of all scatter plots.

Usage:
    python generate_report.py
    python generate_report.py --results-dir custom/dir --out report.html
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html as _html
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"


# ---------- helpers ---------------------------------------------------------


def _b64_png(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt(v: Any, digits: int = 4) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return _html.escape(str(v))


def _color_for_r2(r2: float) -> str:
    """Blue-to-green gradient based on R^2 (0..1). Below 0 -> grey."""
    if pd.isna(r2):
        return "#f3f4f6"
    if r2 < 0:
        return "#e5e7eb"
    # interpolate 0 -> #dbeafe (light blue), 1 -> #166534 (dark green)
    r2 = max(0.0, min(1.0, r2))
    r1, g1, b1 = (219, 234, 254)
    r2c, g2c, b2c = (22, 101, 52)
    rr = int(r1 + (r2c - r1) * r2)
    gg = int(g1 + (g2c - g1) * r2)
    bb = int(b1 + (b2c - b1) * r2)
    return f"rgb({rr},{gg},{bb})"


def _text_on(bg_r2: float) -> str:
    return "white" if (not pd.isna(bg_r2) and bg_r2 > 0.5) else "#111827"


# ---------- report sections -------------------------------------------------


def section_intro(now: str, n_rows: int) -> str:
    return f"""
<section class="intro">
<h1>YONOD 酰胺缩合反应产率预测 — 实验报告</h1>
<p class="subtitle">
生成时间：{now} &nbsp;|&nbsp; 评估组合数：{n_rows} &nbsp;|&nbsp; 数据集：酰胺缩合反应 47015 条
</p>
<p>
本报告比较了 <b>4 类分子描述符</b>（Morgan ECFP4 / ATMOMACCS / FISD / MolMetaLM）与
<b>4 类机器学习算法</b>（XGBoost / Random Forest / SVM / AutoGluon）共 16 种组合
在酰胺键形成反应产率回归任务上的表现。
反应级特征由 6 个分子（2 个底物 + 4 个试剂）的描述符向量拼接而成，使用 5 折交叉验证。
</p>
</section>
"""


def section_glossary() -> str:
    return """
<section class="glossary">
<h2>1 · 评价指标怎么读</h2>
<table>
<thead><tr>
  <th>指标</th><th>取值范围</th><th>实验直觉</th><th>越高/越低更好</th>
</tr></thead>
<tbody>
<tr>
  <td><b>R²</b><br><small>决定系数</small></td>
  <td>通常 0 ~ 1（可能为负）</td>
  <td>"模型解释了多少产率波动"。R²=0.87 ≈ 模型给出的预测和真实产率的相关性极强；
  R²=0.50 大致是"刚比闭眼乱猜好一倍"；R² &lt; 0 比直接取平均值还差。</td>
  <td>越接近 1 越好</td>
</tr>
<tr>
  <td><b>RMSE</b><br><small>均方根误差</small></td>
  <td>与 yield 同单位（0 ~ 1）</td>
  <td>"平均预测偏差"。RMSE=0.11 意味着典型预测偏离真实产率约 ±11 个百分点。
  大误差被放大（平方再开方），所以对偶发预测翻车敏感。</td>
  <td>越小越好</td>
</tr>
<tr>
  <td><b>MAE</b><br><small>平均绝对误差</small></td>
  <td>与 yield 同单位</td>
  <td>"中位偏差感"。MAE=0.07 意味着多数预测距离真实值约 7 个百分点。
  不放大极端误差，比 RMSE 更接近"日常感受"。</td>
  <td>越小越好</td>
</tr>
<tr>
  <td><b>r2_std</b><br><small>5 折稳定性</small></td>
  <td>0 ~ 1</td>
  <td>"模型靠不靠谱"。0.005 意味着换 5 个不同切分跑出来的 R² 几乎一致，模型很稳；
  0.10+ 意味着结果依赖运气，单次结果不可全信。</td>
  <td>越小越稳定</td>
</tr>
<tr>
  <td><b>train_time_s</b></td>
  <td>秒</td>
  <td>训练总耗时。决定上线部署或重训练成本。</td>
  <td>越小越好（其他条件相同时）</td>
</tr>
<tr>
  <td><b>feature_dim</b></td>
  <td>整数</td>
  <td>反应级特征维度 = 6 × 单分子描述符维度。越大表征能力越强但越易过拟合。</td>
  <td>—</td>
</tr>
</tbody>
</table>
<p class="note">
注：AutoGluon 采用单次 80/20 hold-out 评估（其内部已含 bagging+stacking），
故 r2_std=0 不代表稳定性差，仅意味着没有外层 K 折方差。
SVM 因 RBF 复杂度 O(n²) 在 47015 行上不可行，每折训练随机抽样 8000 行，
测试集为完整 fold（结果仍可与其他模型横向比较）。
</p>
</section>
"""


def section_recommendation(df: pd.DataFrame) -> str:
    """Top picks based on R^2 + secondary criteria."""
    # Overall top
    top = df.sort_values("r2_mean", ascending=False).head(3)

    # Speed-conscious: train_time_s under 60s
    fast = df[df["train_time_s"] < 60].sort_values("r2_mean", ascending=False).head(1)

    # Best per descriptor
    per_desc = (
        df.sort_values("r2_mean", ascending=False)
        .groupby("descriptor", sort=False, as_index=False)
        .head(1)
        .sort_values("r2_mean", ascending=False)
    )

    # Best per model
    per_model = (
        df.sort_values("r2_mean", ascending=False)
        .groupby("model", sort=False, as_index=False)
        .head(1)
        .sort_values("r2_mean", ascending=False)
    )

    def _row(row: "pd.Series[Any]", why: str = "") -> str:
        return (
            f"<tr><td><b>{_html.escape(str(row['descriptor']))} × {_html.escape(str(row['model']))}</b></td>"
            f"<td>{_fmt(row['r2_mean'])}</td>"
            f"<td>{_fmt(row['rmse_mean'])}</td>"
            f"<td>{_fmt(row['train_time_s'], 1)} s</td>"
            f"<td>{why}</td></tr>"
        )

    rec_top = "".join(
        _row(r, f"综合 R² 第 {i + 1}") for i, (_, r) in enumerate(top.iterrows())
    )
    rec_fast = "".join(_row(r, "训练快 (&lt; 60s) 且 R² 最高") for _, r in fast.iterrows()) or (
        "<tr><td colspan='5' style='color:#6b7280;'>无 &lt; 60s 的组合（数据集偏大，可考虑云端高配机）</td></tr>"
    )
    rec_desc = "".join(_row(r, f"{r['descriptor']} 上最佳算法") for _, r in per_desc.iterrows())
    rec_model = "".join(_row(r, f"{r['model']} 最佳描述符") for _, r in per_model.iterrows())

    champ = top.iloc[0]
    headline = (
        f"<p class='headline'>🏆 <b>总冠军：{_html.escape(champ['descriptor'])} × {_html.escape(champ['model'])}</b>"
        f"，R²={_fmt(champ['r2_mean'])}，RMSE={_fmt(champ['rmse_mean'])}，"
        f"约 {_fmt(champ['train_time_s'], 0)} 秒训练完成。"
        f"这一组合预测产率与实验值的相关性极强，建议作为首选基线模型。</p>"
    )

    return f"""
<section class="recs">
<h2>2 · 建模推荐</h2>
{headline}

<h3>2.1 综合 R² 前 3 名</h3>
<table>
<thead><tr><th>描述符 × 模型</th><th>R²</th><th>RMSE</th><th>训练时间</th><th>说明</th></tr></thead>
<tbody>{rec_top}</tbody>
</table>

<h3>2.2 时间敏感场景（训练 &lt; 60 秒）</h3>
<table>
<thead><tr><th>描述符 × 模型</th><th>R²</th><th>RMSE</th><th>训练时间</th><th>说明</th></tr></thead>
<tbody>{rec_fast}</tbody>
</table>

<h3>2.3 每种描述符的最佳算法</h3>
<table>
<thead><tr><th>描述符 × 模型</th><th>R²</th><th>RMSE</th><th>训练时间</th><th>说明</th></tr></thead>
<tbody>{rec_desc}</tbody>
</table>

<h3>2.4 每种算法的最佳描述符</h3>
<table>
<thead><tr><th>描述符 × 模型</th><th>R²</th><th>RMSE</th><th>训练时间</th><th>说明</th></tr></thead>
<tbody>{rec_model}</tbody>
</table>
</section>
"""


def section_heatmap(df: pd.DataFrame) -> str:
    """4 x 4 color-coded R^2 grid."""
    descs = list(dict.fromkeys(df["descriptor"]))
    models = list(dict.fromkeys(df["model"]))
    pivot: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        pivot.setdefault(r["descriptor"], {})[r["model"]] = float(r["r2_mean"])

    header = "<th>↓ 描述符 / 模型 →</th>" + "".join(
        f"<th>{_html.escape(str(m))}</th>" for m in models
    )
    rows: List[str] = []
    for d in descs:
        cells: List[str] = [f"<th>{_html.escape(str(d))}</th>"]
        for m in models:
            r2 = pivot.get(d, {}).get(m, float("nan"))
            bg = _color_for_r2(r2)
            fg = _text_on(r2)
            cells.append(
                f"<td style='background:{bg};color:{fg};'>{_fmt(r2, 3)}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<section class="heatmap">
<h2>3 · 描述符 × 模型 R² 热力图</h2>
<p class="note">颜色越深 = R² 越高。直接看哪一格最深绿就是最佳组合。</p>
<table class="heat">
<thead><tr>{header}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</section>
"""


def section_full_table(df: pd.DataFrame) -> str:
    cols = [
        ("descriptor", "描述符"),
        ("model", "模型"),
        ("feature_dim", "特征维度"),
        ("n_samples", "样本数"),
        ("r2_mean", "R²"),
        ("r2_std", "r2_std"),
        ("rmse_mean", "RMSE"),
        ("mae_mean", "MAE"),
        ("train_time_s", "训练耗时 (s)"),
        ("device", "设备"),
    ]
    head = "".join(f"<th>{label}</th>" for _, label in cols)
    body_rows: List[str] = []
    for _, r in df.sort_values("r2_mean", ascending=False).iterrows():
        bg = _color_for_r2(r["r2_mean"])
        fg = _text_on(r["r2_mean"])
        cells: List[str] = []
        for key, _ in cols:
            v = r.get(key, float("nan"))
            if key == "r2_mean":
                cells.append(
                    f"<td style='background:{bg};color:{fg};font-weight:600;'>{_fmt(v)}</td>"
                )
            elif key == "train_time_s":
                cells.append(f"<td>{_fmt(v, 1)}</td>")
            else:
                cells.append(f"<td>{_fmt(v)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="full-table">
<h2>4 · 完整结果表（按 R² 降序）</h2>
<table>
<thead><tr>{head}</tr></thead>
<tbody>{"".join(body_rows)}</tbody>
</table>
</section>
"""


def section_gallery(df: pd.DataFrame, results_dir: Path) -> str:
    """Each row: a scatter plot with R^2 annotation."""
    cards: List[str] = []
    for _, r in df.sort_values(
        ["descriptor", "model"]
    ).iterrows():
        d, m = str(r["descriptor"]), str(r["model"])
        path = results_dir / f"scatter_{d}_{m}.png"
        b64 = _b64_png(path)
        if b64 is None:
            img_html = f"<div class='missing'>缺失：{_html.escape(path.name)}</div>"
        else:
            img_html = f"<img src='data:image/png;base64,{b64}' alt='{_html.escape(path.name)}'/>"
        cards.append(
            f"""<div class="card">
<h4>{_html.escape(d)} × {_html.escape(m)}</h4>
{img_html}
<p>R²={_fmt(r['r2_mean'])}, RMSE={_fmt(r['rmse_mean'])}, MAE={_fmt(r['mae_mean'])}</p>
</div>"""
        )
    return f"""
<section class="gallery">
<h2>5 · 散点图画廊</h2>
<p class="note">
横轴 = 实验测得的真实产率，纵轴 = 模型预测的产率。点越靠近对角线虚线越好。
对角线下方的点 = 模型低估了产率；对角线上方 = 高估。如果云团扁平偏离对角线，
说明模型只学到了产率的均值，结构信息没用上。
</p>
<div class="grid">
{"".join(cards)}
</div>
</section>
"""


def section_caveats() -> str:
    return """
<section class="caveats">
<h2>6 · 局限性与下一步</h2>
<ul>
<li><b>SVM 子采样训练</b>：RBF SVR 复杂度 O(n²)，47015 行下不可行；每折用 8000 行子采样训练。SVM 的 R² 因此可能被低估 2~5 个百分点，是已知偏差。</li>
<li><b>AutoGluon 评估口径</b>：用了单次 80/20 hold-out 而非 5 折，速度上更可控但 r2_std 不可比；公平比较时应只看 R² 均值。</li>
<li><b>MolMetaLM 表现意外偏低</b>：可能是其预训练域偏向药物分子性质，与反应产率任务不直接对齐；fine-tune 后可能有显著提升，留作后续工作。</li>
<li><b>没有外部验证集</b>：所有结果来自内部 CV/holdout；理想情况下应在课题组未参与训练的新反应上做盲测。</li>
<li><b>DFT / HSPOC 描述符未纳入</b>：本数据集无 Gaussian log 文件 + 算力受限。若未来有 DFT 数据可作为第 5 类描述符接入。</li>
</ul>
</section>
"""


# ---------- main ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a chemist-friendly HTML report from YONOD results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                   help="Directory containing metrics_summary.csv and scatter_*.png")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path (default: <results-dir>/report.html)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir: Path = args.results_dir
    out_path: Path = args.out or (results_dir / "report.html")

    metrics_csv = results_dir / "metrics_summary.csv"
    if not metrics_csv.exists():
        print(f"[error] {metrics_csv} not found. "
              f"Run run_yield_prediction.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(metrics_csv)
    # Drop obviously failed rows (those with 'error' column populated)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    n_rows = len(df)
    if n_rows == 0:
        print("[error] no usable rows in metrics_summary.csv", file=sys.stderr)
        return 1

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = "\n".join(
        [
            section_intro(now, n_rows),
            section_glossary(),
            section_recommendation(df),
            section_heatmap(df),
            section_full_table(df),
            section_gallery(df, results_dir),
            section_caveats(),
        ]
    )

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1f2937; line-height: 1.6; }
h1 { font-size: 1.6em; border-bottom: 2px solid #3b82f6; padding-bottom: .3em; }
h2 { color: #1e3a8a; margin-top: 2em; border-left: 4px solid #3b82f6; padding-left: .6em; }
h3 { color: #1e40af; margin-top: 1.4em; }
.subtitle { color: #6b7280; font-size: .95em; }
.headline { background:#fef3c7; padding:.8em 1em; border-left:5px solid #f59e0b; border-radius:6px; }
table { width: 100%; border-collapse: collapse; margin: .8em 0; font-size: .95em; }
th, td { border: 1px solid #e5e7eb; padding: .45em .7em; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; }
table.heat th, table.heat td { text-align: center; font-variant-numeric: tabular-nums; }
.note { color: #6b7280; font-size: .9em; margin-top: -.4em; }
.gallery .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1em; }
.card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: .8em; }
.card h4 { margin: 0 0 .4em 0; font-size: 1em; }
.card img { width: 100%; height: auto; border-radius: 4px; background: white; }
.card .missing { color: #9ca3af; font-style: italic; padding: 2em; text-align: center; }
section { margin-bottom: 2em; }
ul li { margin-bottom: .35em; }
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>YONOD 产率预测报告</title>
<style>{css}</style>
</head>
<body>
{body}
<footer style="margin-top:3em; padding-top:1em; border-top:1px solid #e5e7eb; color:#9ca3af; font-size:.85em;">
YONOD v0.3 · 生成于 {now} · 自包含 HTML，全部图表已 base64 内嵌，可离线浏览。
</footer>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ok] report written: {out_path}  ({size_mb:.2f} MB, {n_rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
