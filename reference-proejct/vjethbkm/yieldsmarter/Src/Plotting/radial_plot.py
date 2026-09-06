import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

colors = {
    "DFT": "#EE934F",
    "SOAP": "#7BACD2",
    "PhysChem": "#87C27E",
    "MFP": "#9E9AC4",
    "OHE": "#969696",
}

criteria = [
    "Chemical\nMeaningfulness",
    "Computational\nCost",
    "Generalizability",
    "Data\nAvailability",
    "Predictive\nPower",
    "Interpretability",
]

# Scores 1–5 mapped to 0–4 
scores_15 = {
    "DFT":      [5, 5, 4, 2, 5, 3],
    "SOAP":     [4, 4, 4, 2, 4, 4],
    "PhysChem": [5, 3, 3, 4, 4, 4],
    "MFP":      [3, 2, 3, 4, 4, 4],
    "OHE":      [1, 1, 1, 5, 3, 2],
}


scores = {k: [v-1 for v in vals] for k, vals in scores_15.items()}

num_vars = len(criteria)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))

plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.12)

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)

ax.set_thetagrids(np.degrees(angles[:-1]), labels=criteria)
ax.tick_params(axis="x", pad=28)
for t in ax.get_xticklabels():
    t.set_fontsize(18)
    t.set_clip_on(False)
    t.set_bbox(dict(boxstyle="round,pad=0.4",
                   facecolor="0.8", alpha=0.7, edgecolor="none"))

ax.set_ylim(0, 4.3)

ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels([])

ax.grid(True, zorder=0)
ax.spines['polar'].set_visible(False)

ax.plot([0, 0], [0, 4.1], color="black", lw=2.5, zorder=40)
ax.plot(0, 4.1, marker=(3, 0, 0), markersize=22,
        color="black", zorder=50, clip_on=False)

tick_half_angle = np.deg2rad(2.0)
for r in range(0, 5):
    ax.plot([-tick_half_angle, tick_half_angle], [r, r],
            color="black", lw=2.5, zorder=45)
    ax.text(np.deg2rad(7), r, str(r),
            ha="left", va="center",
            fontsize=18, fontweight="bold",
            clip_on=False, zorder=60,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

for label, vals in scores.items():
    data = vals + vals[:1]
    ax.fill(angles, data, alpha=0.15, color=colors[label], zorder=10)
    ax.plot(angles, data, color=colors[label],
            linewidth=2.4, zorder=20, label=label)

# Legend
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),frameon=False, fontsize=18)

#plt.show()
fig.savefig("../../Results/Plots/radial_plot.png",dpi=200, bbox_inches="tight")
