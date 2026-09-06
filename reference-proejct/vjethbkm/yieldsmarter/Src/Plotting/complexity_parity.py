import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json, os
from matplotlib.lines import Line2D

sns.set(style='white', context='talk')

DESCRIPTORS = ['DFT', 'SOAP', 'PhysChem', 'MFP', 'OHE']  # rows
DATASETS = ['Doyle', 'Denmark', 'Suzuki', 'Bode']  # columns (keep for paths)

DISPLAY_TITLES = {
    'Doyle': 'BH1',
    'Denmark': 'BH2',
    'Suzuki': 'SM',
    'Bode': 'SL1'
}

COLORS = ['#fd8d3c', '#6baed6', '#74c476', '#9e9ac8', '#969696']

ROOTS = {
    'Doyle': '../../Results/Compare_Complexity/Doyle_2018/',
    'Denmark': '../../Results/Compare_Complexity/Denmark_2023/',
    'Suzuki': '../../Results/Compare_Complexity/Suzuki_2018/',
    'Bode': '../../Results/Compare_Complexity/Bode_2023/'
}

DESCRIPTORS_displayed = ['DFT-derived', 'SOAP', 'PhysChem', 'MFP',
                         'OHE']  # rows

n_rows, n_cols = len(DESCRIPTORS), len(DATASETS)

fig, axes = plt.subplots(n_rows,
                         n_cols,
                         figsize=(18, 12),
                         squeeze=False,
                         layout='constrained')

fig.text(0.52, -0.02, 'True Yield (%) / LC-MS Ratio', ha='center', fontsize=20)
fig.text(-0.01,
         0.50,
         'Predicted Yield (%) / LC-MS Ratio',
         va='center',
         rotation='vertical',
         fontsize=20)

row_letters = ['(a)', '(b)', '(c)', '(d)', '(e)']

for row_idx, desc in enumerate(DESCRIPTORS):
    for col_idx, dataset in enumerate(DATASETS):
        ax = axes[row_idx, col_idx]

        # Axis limits & ticks per dataset
        if dataset == "Bode":
            xylim = 660
            tick_vals = np.arange(0, 601, 200)
            ax.set_xlim(-20, xylim)
            ax.set_ylim(-20, xylim)
        else:
            xylim = 105
            tick_vals = np.arange(0, 110, 50)
            ax.set_xlim(-5, xylim)
            ax.set_ylim(-5, xylim)

        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.tick_params(bottom=True,
                       labelbottom=True,
                       left=True,
                       labelleft=True,
                       direction='out',
                       length=5)

        gap = 5  # ok, for this, just adjust this for more or less separation (in data units)

        # Shift spines inward slightly to create the gap
        ax.spines['bottom'].set_position(('data', 0 - gap))
        ax.spines['left'].set_position(('data', 0 - gap))

        if dataset == "Bode":
            ax.spines['bottom'].set_bounds(0, 600)
            ax.spines['left'].set_bounds(0, 600)
        else:
            ax.spines['bottom'].set_bounds(0, 100)
            ax.spines['left'].set_bounds(0, 100)

        # Line styling
        ax.spines['bottom'].set_linewidth(1.3)
        ax.spines['left'].set_linewidth(1.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        base_root = ROOTS.get(dataset, None)
        if base_root is None:
            ax.axis("off")
            continue

        base_path = os.path.join(base_root, desc)
        npz_file = os.path.join(base_path, f"{dataset}_{desc}_oof_predictions_RF.npz")
        json_file = os.path.join(base_path, f"{dataset}_{desc}_metrics_RF.json")

        print(npz_file, json_file)
        # Plot if files exist
        if os.path.exists(npz_file) and os.path.exists(json_file):
            data = np.load(npz_file)
            metrics = json.load(open(json_file))
            #            x, y = data['test_true'], data['test_pred']
            # --- handle different npz key names ---
            if 'test_true' in data and 'test_pred' in data:
                x, y = data['test_true'], data['test_pred']
            elif 'y_true' in data and 'y_pred' in data:
                x, y = data['y_true'], data['y_pred']
            else:
                raise KeyError(
                    f"Expected keys not found in {npz_file}. Found: {data.files}"
                )

            # color by descriptor (row)
            if 0:
                sns.scatterplot(x=x,
                                y=y,
                                ax=ax,
                                s=30,
                                edgecolor='white',
                                facecolor=COLORS[row_idx],
                                alpha=0.3,
                                linewidth=0.7)
                # Column titles (top row) with new names
                if row_idx == 0:
                    ax.set_title(DISPLAY_TITLES[dataset],
                                 fontsize=16,
                                 pad=3,
                                 bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="lightgray",
                                           alpha=0.25,
                                           edgecolor="none"))
            else:
                # Create inset axes for marginal histograms
                #from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
                divider_size = "20%"
                gap_frac = 0.04

                # Main scatter (shrink the existing ax)
                ax.scatter(x,
                           y,
                           s=20,
                           color=COLORS[row_idx],
                           alpha=0.4,
                           edgecolor='white',
                           linewidth=0.5)

                # Top histogram (x distribution)
                ax_histx = ax.inset_axes([0, 1 + gap_frac, 1, 0.25])
                ax_histx.hist(x,
                              bins=40,
                              color=COLORS[row_idx],
                              alpha=0.6,
                              edgecolor='none')
                ax_histx.set_xlim(ax.get_xlim())
                ax_histx.axis('off')

                # Right histogram (y distribution)
                ax_histy = ax.inset_axes([1 + gap_frac, 0, 0.25, 1])
                ax_histy.hist(y,
                              bins=20,
                              color=COLORS[row_idx],
                              alpha=0.6,
                              edgecolor='none',
                              orientation='horizontal')
                ax_histy.set_ylim(ax.get_ylim())
                ax_histy.axis('off')

                # Column titles (top row) with new names
                if row_idx == 0:
                    ax_histx.set_title(DISPLAY_TITLES[dataset],
                                       fontsize=20,
                                       pad=20,
                                       bbox=dict(boxstyle="round,pad=0.3",
                                                 facecolor="lightgray",
                                                 alpha=0.25,
                                                 edgecolor="none"))

            # y = x diagonal
            ax.plot([0, xylim], [0, xylim], ls='--', c='black', lw=1.2)

legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='none',
           markerfacecolor=color,
           markersize=14,
           label=desc) for desc, color in zip(DESCRIPTORS_displayed, COLORS)
]

fig.legend(handles=legend_elements,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=len(DESCRIPTORS_displayed),
           frameon=False,
           handletextpad=1.2,
           fontsize=20)

fig.set_constrained_layout_pads(
    w_pad=0.1,  # padding between figure edge and subplots (width)
    h_pad=0.1,  # padding between figure edge and subplots (height)
    wspace=0,  # space between columns
    hspace=0  # this can be increased this for more vertical space between rows
)

plt.savefig('../../Results/full_grid_parity.png',
            bbox_inches='tight',
            pad_inches=0.1)
#plt.show()
