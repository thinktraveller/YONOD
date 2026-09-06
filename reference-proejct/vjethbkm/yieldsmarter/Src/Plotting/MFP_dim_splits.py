import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

base_root = os.path.join("..", "..", "Results", "Dimensionality_Splits")
plots_dir = os.path.join("..", "..", "Results", "Plots")
os.makedirs(plots_dir, exist_ok=True)

dims = ["0D", "1D", "2D"]
datasets = ["BH1", "BH2", "SM", "SL1"]

plot_config = {
    "0D": {"BH1": "BH1_0D", "BH2": "BH2_0D", "SM": "SM_0D", "SL1": "SL1_0D"},
    "1D": {"BH1": "BH1_1D_additives", "BH2": "BH2_1D_catalyst",
           "SM": "SM_1D_reactant_1", "SL1": "SL1_1D_Aldehyde_1"},
    "2D": {"BH1": "BH1_2D", "BH2": "BH2_2D", "SM": "SM_2D", "SL1": "SL1_2D"},
}

# --- Base fill color + hatches ---
dim_colors = {dim: "#A5A1C8" for dim in dims}
dim_hatches = {"0D": "///", "1D": "xx", "2D": ".."}
EDGE = "black"

# Collect MAE data
mae_data = {ds: [] for ds in datasets}
for dim in dims:
    for ds in datasets:
        folder = plot_config[dim][ds]
        path = os.path.join(base_root, ds, folder, f"{folder}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
                mae = metrics["test"]["mae"]
                if isinstance(mae, dict):
                    mae = mae.get("mean", 0.0)
                mae_data[ds].append(mae)
        else:
            mae_data[ds].append(0.0)

# --- Styling function ---
SPINE_W = 2.0
def style_axes(ax, fontsize=16):
    ax.tick_params(axis="x", which="both", length=0, width=0, labelsize=fontsize)
    ax.tick_params(axis="y", which="both", length=0, width=0, labelsize=fontsize)
    sns.despine(ax=ax, right=True, top=True, left=False, offset=5)
    ax.spines["bottom"].set_linewidth(SPINE_W)
    ax.spines["left"].set_linewidth(1.6)

# --- Plot: 2x2 subplots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
axes = axes.flatten()
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

bars_for_legend = []

for i, ds in enumerate(datasets):
    ax = axes[i]
    vals = mae_data[ds]
    x = np.arange(len(dims))

    for j, dim in enumerate(dims):
        bar = ax.bar(
            x[j], vals[j], width=0.7,
            color=dim_colors[dim], edgecolor=EDGE,
            linewidth=2.0, alpha=1.0, zorder=1,
            hatch=dim_hatches[dim], label=dim
        )
        if i == 0:  # only collect legend handles once
            bars_for_legend.append(bar)

    ax.set_title(ds, fontsize=16, pad=10,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="lightgray", alpha=0.25,
                           edgecolor="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=12)
    ax.set_ylim(0, 30)
    style_axes(ax, fontsize=16)

    if ds == "SL1":
        ax.set_ylabel("Test MAE (LC-MS product ratio)", fontsize=16)
    else:
        ax.set_ylabel("Test MAE (%)", fontsize=16)

    ax.text(-0.10, 1.08, subplot_labels[i],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            va="bottom", ha="right", clip_on=False)

# Add shared legend (0D/1D/2D with hatches)
fig.legend(bars_for_legend, dims, loc="lower center", ncol=3,
           fontsize=14, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
out_path = os.path.join(plots_dir, "MFP_dim_splits_2x2.pdf")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved figure to: {out_path}")

