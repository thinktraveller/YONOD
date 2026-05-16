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


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    desc: str,
    model: str,
    save_dir: Union[str, Path],
    *,
    figsize: tuple = (5, 5),
    dpi: int = 120,
) -> Path:
    """Render and save a prediction-vs-truth scatter plot.

    Args:
        y_true: ground-truth yields, shape (n,).
        y_pred: model predictions, shape (n,).
        desc: descriptor name (used in title and filename).
        model: model name (used in title and filename).
        save_dir: directory to save the PNG into; created if missing.

    Returns:
        Path to the saved PNG file.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, edgecolor="none", color="#3b82f6")

    # Axis range follows observed values (incl. any negative predictions) so
    # outliers stay visible; cropping was rejected because it would hide
    # negative-prediction samples and mislead the chemist reading the plot.
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "--", color="#666666", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("True yield")
    ax.set_ylabel("Predicted yield")
    # Use mathtext so R^2 renders as R-squared (not the literal caret).
    ax.set_title(f"{desc} x {model}")
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_true)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"scatter_{desc}_{model}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
