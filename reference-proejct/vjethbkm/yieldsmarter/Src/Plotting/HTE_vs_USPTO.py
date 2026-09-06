import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import re

# The USPTO dataset has a lot of non-numeric strings to parse and interpret
def interpret_yield_value(val):
    val = str(val).strip()
    try:
        if val.startswith(('~', '≈')):
            val = val[1:]
        if val.startswith('<'):
            return float(val[1:]) / 2.0
        if val.startswith('>'):
            return float(val[1:]) + 0.5
        if ' to ' in val:
            a, b = val.split(' to ')
            return (float(a) + float(b)) / 2.0
        return float(val)
    except Exception:
        m = re.search(r'[\d.]+', val)
        return float(m.group()) if m else None

bh_df = pd.read_csv('../../Data/HTE_datasets/BH1/BH1.csv')
# Download USPTO data from: https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
uspto_df = pd.read_csv('../../Data/HTE_datasets/BH1/uspto_1976_2016_grants_yielddata.csv')

bh_df['numeric_yield'] = bh_df['Yield'].apply(interpret_yield_value)
uspto_df['numeric_yield'] = uspto_df['Yield'].apply(interpret_yield_value)

bh = bh_df.dropna(subset=['numeric_yield'])['numeric_yield'].clip(0, 100)
uspto = uspto_df.dropna(subset=['numeric_yield'])['numeric_yield'].clip(0, 100)

bins = np.arange(0, 105, 5)          
xticks = np.arange(0, 101, 20)      

sns.set(style='white', context='talk')
COLOR_BH = '#C56637'
COLOR_USPTO = '#769FC7'
BAR_EDGE = 'black'

MAJOR_TICK = dict(direction='out', length=8, width=1.6, color='black')
MINOR_LEN = 4
SPINE_W = 1.6

def style_axes(ax, add_minor=True):
    ax.tick_params(**MAJOR_TICK)
    if add_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', which='minor', direction='out', length=MINOR_LEN, width=1.2, color='black')
    sns.despine(ax=ax, offset=5)
    ax.spines['left'].set_linewidth(SPINE_W)
    ax.spines['bottom'].set_linewidth(SPINE_W)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
ax_bh, ax_uspto = axes

# (a) BH (HTE) histogram
ax_bh.hist(bh, bins=bins, color=COLOR_BH, edgecolor=BAR_EDGE, alpha=0.85)
ax_bh.set_xlim(0, 100)
ax_bh.set_xticks(xticks)
style_axes(ax_bh)
ax_bh.set_xlabel('Yield (%)')
ax_bh.set_ylabel('Number of reactions')
ax_bh.set_title('HTE yield distribution',
                fontsize=18, pad=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.25, edgecolor="none"))
ax_bh.text(-0.10, 1.08, "(a)", transform=ax_bh.transAxes,
           fontsize=18, fontweight="bold", va="bottom", ha="right", clip_on=False)

# (b) USPTO histogram
ax_uspto.hist(uspto, bins=bins, color=COLOR_USPTO, edgecolor=BAR_EDGE, alpha=0.85)
ax_uspto.set_xlim(0, 100)
ax_uspto.set_xticks(xticks)
style_axes(ax_uspto)
ax_uspto.set_xlabel('Yield (%)')
ax_uspto.set_ylabel('')  
ax_uspto.set_title('Patent yield distribution',
                   fontsize=18, pad=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.25, edgecolor="none"))
ax_uspto.text(-0.10, 1.08, "(b)", transform=ax_uspto.transAxes,
              fontsize=18, fontweight="bold", va="bottom", ha="right", clip_on=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.25, bottom=0.10)
plt.savefig('../../Results/Plots/Low_High.png', dpi=200, bbox_inches='tight')
plt.show()
