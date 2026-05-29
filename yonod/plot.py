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
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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
    x_label: str = "True Value",
    y_label: str = "Predicted Value",
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
        x_label / y_label: axis labels.

    Returns:
        Path to the saved PNG file.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(y_true, y_pred, s=18, alpha=0.5, edgecolor="none", color="#3b82f6")

    # Shared axis range: both axes span the same [min, max] so the y=x line
    # sits exactly on the diagonal and the visual aspect ratio is meaningful.
    all_lo = min(float(y_true.min()), float(y_pred.min()))
    all_hi = max(float(y_true.max()), float(y_pred.max()))
    pad = 0.05 * (all_hi - all_lo + 1e-9)
    lims = (all_lo - pad, all_hi + pad)

    ax.plot(lims, lims, "--", color="#666666", linewidth=1, label="y = x")

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{desc} x {model}", fontsize=11)
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
