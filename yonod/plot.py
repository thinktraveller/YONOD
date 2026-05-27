"""Prediction-vs-truth scatter plot generator.

One function: ``plot_scatter(y_true, y_pred, desc, model, save_dir)``.
Saves a PNG to ``{save_dir}/scatter_{desc}_{model}.png`` with R^2 / RMSE
annotation and a y=x diagonal reference line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")  # headless backend; avoids GUI / Tk dependency on Windows.

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ── Chinese font setup (Windows: Microsoft YaHei; Linux: WenQuanYi; fallback: DejaVu) ──
_CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "微软雅黑",
    "SimHei", "黑体",
    "STHeiti", "华文黑体",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_chosen = next((f for f in _CHINESE_FONT_CANDIDATES if f in _available_fonts), None)
if _chosen:
    matplotlib.rcParams["font.sans-serif"] = [_chosen] + matplotlib.rcParams.get("font.sans-serif", [])
matplotlib.rcParams["axes.unicode_minus"] = False  # prevent minus sign from becoming a box


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    desc: str,
    model: str,
    save_dir: Union[str, Path],
    *,
    figsize: tuple = (5, 5),
    dpi: int = 120,
    x_label: str = "真实值",
    y_label: str = "预测值",
) -> Path:
    """Render and save a prediction-vs-truth scatter plot.

    X-axis range is determined by y_true min/max;
    Y-axis range is determined by y_pred min/max.
    The y=x reference line spans the combined range of both axes.

    Args:
        y_true: ground-truth labels, shape (n,).
        y_pred: model predictions, shape (n,).
        desc: descriptor name (used in title and filename).
        model: model name (used in title and filename).
        save_dir: directory to save the PNG into; created if missing.
        x_label / y_label: axis labels (default: Chinese).

    Returns:
        Path to the saved PNG file.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(y_true, y_pred, s=18, alpha=0.5, edgecolor="none", color="#3b82f6")

    # Independent axis ranges: x follows y_true, y follows y_pred.
    x_lo, x_hi = float(y_true.min()), float(y_true.max())
    y_lo, y_hi = float(y_pred.min()), float(y_pred.max())
    x_pad = 0.05 * (x_hi - x_lo + 1e-9)
    y_pad = 0.05 * (y_hi - y_lo + 1e-9)
    x_lims = (x_lo - x_pad, x_hi + x_pad)
    y_lims = (y_lo - y_pad, y_hi + y_pad)

    # y=x reference line spans the combined visible range; matplotlib clips to axes.
    diag_lo = min(x_lims[0], y_lims[0])
    diag_hi = max(x_lims[1], y_lims[1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            "--", color="#666666", linewidth=1, label="y = x")

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{desc}  ×  {model}", fontsize=11)
    ax.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.4f}\nn = {len(y_true)}",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"scatter_{desc}_{model}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
