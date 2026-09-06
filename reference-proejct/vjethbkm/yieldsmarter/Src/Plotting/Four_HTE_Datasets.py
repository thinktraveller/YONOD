import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BASE_DIR = "../../Data/HTE_datasets"
DATASETS = ["BH1", "BH2", "SM", "SL1"]
COLOR = "#506483"

fig, axes = plt.subplots(len(DATASETS), 1, figsize=(7.2, 14), sharex=False)
fig.subplots_adjust(hspace=0.8)

for ax, ds in zip(axes, DATASETS):
    path = os.path.join(BASE_DIR, ds, f"{ds}.csv")
    if not os.path.isfile(path):
        ax.axis("off"); continue
    df = pd.read_csv(path)
    if df.empty:
        ax.axis("off"); continue

    last = df.columns[-1]
    name = last.strip().lower()
    x_label = "LC-MS ratio" if ds == "SL1" or any(k in name for k in ["lc-ms","lcms","ratio"]) \
              else ("Yield (%)" if "yield" in name else last)

    x = pd.to_numeric(df[last], errors="coerce").dropna().clip(lower=0).to_numpy()
    if x.size < 2:
        ax.axis("off"); continue

    xmax = float(np.nanmax(x))
    pad = 0.03 * max(xmax, 1e-9)
    x_grid = np.linspace(0.0, xmax + pad, 512)
    density = gaussian_kde(x)(x_grid)

    ax.fill_between(x_grid, density, 0, facecolor=COLOR, edgecolor=COLOR, linewidth=1.5)
    ax.set_xlim(left=0)

    ax.set_xlabel(x_label, labelpad=12, fontsize=14)
    ax.set_ylabel("Density", labelpad=12, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)

    ax.text(0.98, 0.98, rf"$\bf{{{ds}}}$: {len(df)} reactions",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            bbox=dict(facecolor="lightgray", alpha=0.5, boxstyle="round,pad=0.3"))

    ax.annotate("", xy=(0, 1.02), xytext=(0, 1.0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=COLOR),
                annotation_clip=False, zorder=5)

plt.savefig("../../Results/Plots/AllDatasetsDensity.png", dpi=300, bbox_inches="tight")
#plt.show()
