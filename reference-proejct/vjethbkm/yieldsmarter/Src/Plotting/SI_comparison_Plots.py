import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json, os

sns.set(style='white', context='talk')

DATASETS = ['Doyle', 'Denmark', 'Suzuki', 'Bode']
MODELS   = ['LightGBM', 'KNN', 'Ridge']

COLORS = ['#D62828', '#2A9D8F', '#F4A261']


def load_predictions(dataset, model):
    """Load .npz (preds) and handle key names."""
    npz_file  = f"{dataset}_MFP_predictions_{model}.npz"
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Missing predictions file: {npz_file}")
    data = np.load(npz_file)
    if "y_true" in data and "y_pred" in data:
        x, y = data["y_true"], data["y_pred"]
    elif "test_true" in data and "test_pred" in data:
        x, y = data["test_true"], data["test_pred"]
    else:
        raise KeyError(f"Unexpected keys in {npz_file}: {data.files}")
    return x, y


def load_metrics(dataset, model):
    """Load metrics from the JSON file (MAE and R² test means)."""
    json_file = f"{dataset}_MFP_metrics_{model}.json"

    if not os.path.exists(json_file):
        print(f"Warning: Missing metrics file for {dataset} - {model}")
        return None, None
    with open(json_file, "r") as f:
        metrics = json.load(f)
    try:
        mae = metrics["test"]["mae"]["mean"]
        r2 = metrics["test"]["r2"]["mean"]
    except KeyError:
        print(f" Unexpected JSON structure in {json_file}")
        mae, r2 = None, None
    return mae, r2


def plot_dataset(dataset):
    """One row × three columns (LightGBM, KNN, Ridge) with shared axis labels."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        constrained_layout=True,
        sharex=True,
        sharey=True
    )

    # Axis settings per dataset
    if dataset == "Bode":
        lim = 660
        tick_vals = np.arange(0, 601, 200)
        x_label = "True LC–MS Ratio"
        y_label = "Predicted LC–MS Ratio"
        x_lo = y_lo = -20
    else:
        lim = 100
        tick_vals = np.arange(0, 110, 20)
        x_label = "True Yield (%)"
        y_label = "Predicted Yield (%)"
        x_lo = y_lo = -40

    for i, model in enumerate(MODELS):
        ax = axes[i]
        color = COLORS[i]

        x, y = load_predictions(dataset, model)
        mae, r2 = load_metrics(dataset, model)

        sns.scatterplot(
            x=x, y=y, ax=ax, s=50,
            edgecolor='black', facecolor=color, linewidth=1.2
        )

        ax.plot([0, lim], [0, lim], ls='--', c='black', lw=1.2)

        ax.set_xlim(x_lo, lim)
        ax.set_ylim(y_lo, lim)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)

        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(bottom=True, labelbottom=True,
                       left=True, labelleft=True,
                       direction='out', length=5)

        # Add R² and MAE values (top-left)
        if mae is not None and r2 is not None:
            ax.text(0.04, 0.94,
                    f"$R^2$ = {r2:.2f}\nMAE = {mae:.2f}",
                    transform=ax.transAxes,
                    fontsize=16,
                    color='black',
                    va='top',
                    ha='left')

        ax.set_title(model, fontsize=16, pad=4,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightgray", alpha=0.25,
                               edgecolor="none"))

    # Shared axis labels
    fig.text(0.52, -0.05, x_label, ha='center', fontsize=18)
    fig.text(-0.015, 0.5, y_label, va='center',
             rotation='vertical', fontsize=18)

    outname = f"{dataset}_comparison.pdf"
    plt.savefig(outname, bbox_inches='tight')
    plt.close(fig)
    print(f" Saved {outname}")


def main():
    for ds in DATASETS:
        plot_dataset(ds)


if __name__ == "__main__":
    main()
