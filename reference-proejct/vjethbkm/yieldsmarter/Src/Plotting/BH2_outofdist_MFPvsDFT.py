from glob import glob
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import kendalltau

cmap = mpl.colormaps["tab20c"]
DFT_EDGE, DFT_FILL = cmap(4), cmap(5)
MFP_EDGE, MFP_FILL = cmap(12), cmap(13)
DASHED_GREY = "#6b7280"
LETTERS = list("abcdefghijklm")

AXIS_LABEL_FSIZE, TICK_LABEL_FSIZE, TITLE_FSIZE = 12, 11, 12


def load_triplet(true_file, dft_file):
    """Load and align true, DFT, and MFP predictions."""
    if not (os.path.exists(true_file) and os.path.exists(dft_file)):
        return None, None, None, 0
    df_true, df_dft = pd.read_csv(true_file), pd.read_csv(dft_file)
    n_true = df_true["y_true"].notna().sum()
    if "y_pred" not in df_dft:
        return None, None, None, n_true
    n = min(len(df_true), len(df_dft))
    x = pd.to_numeric(df_true["y_true"][:n], errors="coerce")
    y_dft = pd.to_numeric(df_dft["y_true"][:n], errors="coerce")
    y_mfp = pd.to_numeric(df_dft["y_pred"][:n], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y_dft) & np.isfinite(y_mfp)
    return x[mask], y_dft[mask], y_mfp[mask], int(n_true)


def plot_bar(ax, summary_csvs):
    """Top barplot with MAE ± std."""
    accum = []
    for fn in summary_csvs:
        df = pd.read_csv(fn)
        d = {
            'Product': fn.split("/")[-1].split("_")[1]
        }  # extract product letter
        y_true = df['True_Yield']
        y_dft = df['DFT_Yield']
        y_mfp = df['Predicted_Yield']
        d["Mean_DFT"] = mean_absolute_error(y_true, y_dft)
        d["Mean_MFP"] = mean_absolute_error(y_true, y_mfp)
        accum.append(d)

    s = pd.DataFrame(accum)
    print(s)

    x = np.arange(len(s))
    width = 0.4
    ax.bar(
        x - width / 2,
        s["Mean_MFP"],
        width,
        #    yerr=s["Mean_MFP_stdev"],
        color=MFP_FILL,
        edgecolor=MFP_EDGE,
        error_kw=dict(ecolor="black", lw=1.5, capsize=3))
    ax.bar(
        x + width / 2,
        s["Mean_DFT"],
        width,
        #    yerr=s["Mean_DFT_stdev"],
        color=DFT_FILL,
        edgecolor=DFT_EDGE,
        error_kw=dict(ecolor="black", lw=1.5, capsize=3))
    ax.set_ylabel("Avg. MAE", fontsize=AXIS_LABEL_FSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(s["Product"].str.upper(), fontsize=TICK_LABEL_FSIZE)
    ax.set_ylim(0, 60)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_LABEL_FSIZE)


def plot_scatter_grid(fig, summary_csvs, bar_ax):
    """Add scatter subplots under barplot, aligned horizontally."""
    left, right = bar_ax.get_position().x0, bar_ax.get_position().x1
    bar_bottom = bar_ax.get_position().y0

    accum = {}
    for fn in summary_csvs:
        df = pd.read_csv(fn)
        key = fn.split("/")[-1].split("_")[1]
        accum[key] = df

    # Layout parameters
    ncols, nrows = 4, 4

    wgap, hgap = 0.006, 0.035
    bar_to_scatter_gap = 0.20
    lim = 110
    pad = 10

    total_w = right - left - (ncols - 1) * wgap
    total_h = 0.93 - (nrows - 1) * hgap

    w, h = total_w / ncols, total_h / nrows

    # Bottom of first scatter row sits below barplot
    bottom = bar_bottom - total_h - bar_to_scatter_gap

    # Explicit 4×4 grid layout so 'M' sits directly below 'K'
    grid_positions = {
        'a': (0, 0),
        'b': (0, 1),
        'c': (0, 2),
        'd': (0, 3),
        'e': (1, 0),
        'f': (1, 1),
        'g': (1, 2),
        'h': (1, 3),
        'i': (2, 0),
        'j': (2, 1),
        'k': (2, 2),
        'l': (2, 3),
        'm': (3, 1)
    }

    for letter, (r, c) in grid_positions.items():
        y = bottom + (nrows - 1 - r) * (h + hgap)
        x = left + c * (w + wgap)
        if letter == 'm':
            x += w * 0.5 + 2 * wgap

        ax = fig.add_axes([x, y, w, h])

        x_true = accum[letter]['True_Yield']
        y_dft = accum[letter]['DFT_Yield']
        y_mfp = accum[letter]['Predicted_Yield']
        n_true = x_true.shape[0]

        if x_true is None:
            ax.axis("off")
            continue

        ax.scatter(x_true,
                   y_mfp,
                   s=42,
                   alpha=0.9,
                   facecolors=MFP_FILL,
                   edgecolors=MFP_EDGE,
                   linewidths=0.8)
        ax.scatter(x_true,
                   y_dft,
                   s=42,
                   alpha=0.9,
                   facecolors=DFT_FILL,
                   edgecolors=DFT_EDGE,
                   linewidths=0.8)
        ax.plot([0, 100], [0, 100], ls="--", color=DASHED_GREY, lw=1)

        ax.set_xlim(-pad, lim)
        ax.set_ylim(-pad, lim)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_aspect("equal", adjustable="box")

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.spines["bottom"].set_bounds(0, 100)  # start slightly after 0
        ax.spines["left"].set_bounds(0, 100)  # start slightly above 0

        ax.text(0.5,
                0.84,
                f"({letter.upper()})",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold")

        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=TICK_LABEL_FSIZE)


def main(preds_dir, out, dpi):
    preds_dir = Path(preds_dir)

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    plt.close("all")
    fig = plt.figure(figsize=(10, 8))

    preds_files = glob(str(preds_dir / "*.csv"))

    ax_bar = fig.add_axes([0.08, 0.85, 0.88,
                           0.15])  # moved higher to make space below
    plot_bar(ax_bar, preds_files)

    handles = [
        mpl.patches.Patch(facecolor=MFP_FILL, edgecolor=MFP_EDGE, label="MFP"),
        mpl.patches.Patch(facecolor=DFT_FILL,
                          edgecolor=DFT_EDGE,
                          label="DFT-derived (*)")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),  # right below the barplot
        ncol=2,
        frameon=False,
        fontsize=11)

    plot_scatter_grid(fig, preds_files, bar_ax=ax_bar)

    # Dynamically place x-axis label just below the bottom (M) subplot
    bottom_m = fig.axes[-1].get_position().y0
    fig.text(0.53,
             bottom_m - 0.05,
             "True Yield (%)",
             ha="center",
             va="center",
             fontsize=AXIS_LABEL_FSIZE)

    fig.text(0.03,
             0.45,
             "Predicted Yield (%)",
             va="center",
             rotation="vertical",
             fontsize=AXIS_LABEL_FSIZE)

    fig.text(0.000, 0.99, "a", fontsize=TITLE_FSIZE + 5, fontweight="bold")
    fig.text(0.00,
             0.82,
             "b",
             fontsize=TITLE_FSIZE + 5,
             fontweight="bold",
             ha="left",
             va="top")

    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="../../Results/BH2/Holdout/MFP")
    ap.add_argument("--out", default="../../Results/Fig_predictions.pdf")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    main(args.preds_dir, args.out, args.dpi)
