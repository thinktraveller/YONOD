import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os import path
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']


datasets = ['Wang', 'GeckoQ']
savepath = path.relpath('src/validation/1006/wsimpol')

targets_latex = {
    'log_p_sat': r'$\log_{10}(P_\text{sat})$',
    'log_kwiomg': r'$\log_{10}(K_\text{WIOM/G})$' ,
    'log_kwg': r'$\log_{10}(K_\text{W/G})$',
    'dvap': r'$\Delta H_\text{vap}$',
    'tg' : r'$T_g$'
}

ylabels_latex  ={
    'log_p_sat': r'Test MAE ($\log_{10}$(kPa))',
    'log_kwiomg': r'Test MAE ($\log_{10}$(1))',
    'log_kwg': r'Test MAE ($\log_{10}$(1))',
    'dvap': r'Test MAE (kJ mol$^{-1}$)',
    'tg' : 'Test MAE (K)'
}

ylims = {
    'Wanglog_p_sat': (0.24, 0.62),
    'Wanglog_kwg': (0.33, 1.0),
    'Wanglog_kwiomg': (0.23, 0.71),
    'GeckoQlog_p_sat': (0.67, 0.92),
    'Litg' : (15, 35),
    'Ferraz-Caetanodvap' : (1.5, 19.5)
}

xticks = {
    'Wang': [500, 1000, 1500, 2000, 2500, 3000],
    'GeckoQ': [4650, 9250, 13900, 18500, 23200, 27800],
    'Ferraz-Caetano': [350, 700, 1050, 1400, 1750, 2100],
    'Li': [325,  650,  975, 1300, 1625, 1950]
}

yticks = {
    'Wanglog_p_sat': np.linspace(0.25, 0.60, num=8),
    'Wanglog_kwg': np.linspace(0.35, 0.85, num=6),
    'Wanglog_kwiomg': np.linspace(0.25, 0.65, num=9),
    'GeckoQlog_p_sat': np.linspace(0.7,0.9, num=5),
    'Litg' : np.linspace(16, 32, num=9),
    'Ferraz-Caetanodvap' : np.linspace(2,16, num=8)
}

dataset_to_desc_leg_tar = {
    'Wang': (['ATMO_DECIMAL_v4', 'ATMOMACCS_DECIMAL_v4'],
             ['ATMO v5','ATMOMACCS v5'],
             ['log_p_sat']
             ),
    'GeckoQ': (['ATMO_DECIMAL_v4', 'ATMOMACCS_DECIMAL_v4'],
             ['ATMO v5','ATMOMACCS v5'],
             ['log_p_sat']),

}

ds_topfp = {
    'log_p_sat' : 'TopFP_log_p_sat',
    'log_kwg' : 'TopFP_log_kwg',
    'log_kwiomg' : 'TopFP_log_kwiomg',
    'dvap' : 'TopFP_dvap',
    'tg' : 'TopFP_tg'
} 

ds_to_simpol_res = {
    'Wang' : 1.2869750402858329,
    'GeckoQ': 2.300181766009167
}

export_scaling = 1
export_scaling = 12
fs = 7.5
plt.rcParams.update({'font.size': fs})
lw = 1.2
ms = 4.5
colors = ['#53555e',  # dark gray / charcoal: MACCS
          '#f5bad3',   # soft pink / baby pink: ATMO 
          '#e1aee4',  # pastel pink-purple / light orchid: ATMOMACCS v1
          '#bdabbf',  # muted lavender / dusty lilac: ATMOMACCS v2
          '#949494',  # light gray '#89bebf': TopFP
          '#b9e4e4',  # light aqua / pale cyan: : ATMOMACCS v3
          '#89bebf',  # muted teal / soft cyan: ATMOMACCS v4
          '#8995BF'  # desaturated periwinkle / dusty indigo: : ATMOMACCS v5
         ]

import matplotlib.pyplot as plt
import numpy as np

for dataset in datasets:
    lw = 1.2
    ms = 4
    descriptors, legends, targets = dataset_to_desc_leg_tar[dataset]

    for target in ['log_p_sat']:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        simpol_val = ds_to_simpol_res[dataset]
        ylim_main = ylims[dataset + target]

        fig, (ax_cut, ax_main) = plt.subplots(2, 1, sharex=True,
                                            figsize=(3.5, 2.625),
                                            gridspec_kw={'height_ratios':[1,4]})
        fig.subplots_adjust(hspace=0.01)

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

        filepath = path.relpath(f'data/{dataset}/KRR_output')

        for i, descriptor in enumerate(descriptors):
            if 'TopFP' in descriptor:
                descriptor = ds_topfp[target]
            try: 
                desc_mean = pd.read_csv(path.join(filepath, f'mean_{descriptor}_{target}.csv'))
                desc_std = pd.read_csv(path.join(filepath, f'std_{descriptor}_{target}.csv'))
            except FileNotFoundError:
                continue

            std = desc_std['Test_MAE'].values
            mae = desc_mean['Test_MAE'].values
            t_size = desc_mean['Train_sizes'].values

            # Plot curves on both axes
            for ax in [ax_cut, ax_main]:
                ax.errorbar(t_size, mae, yerr=std, marker='o',
                            linewidth=lw, markersize=ms, color=colors[i],
                            capsize=1.5, capthick=0.9, elinewidth=lw-0.4)

        # SIMPol reference (draw on top axis; span full x-range)
        x_min, x_max = np.inf, -np.inf
        x_min = min(x_min, np.min(t_size))
        x_max = max(x_max, np.max(t_size))
        ax_cut.hlines(ds_to_simpol_res[dataset], xmin=x_min, xmax=x_max,
                    linestyle='--', color='red', alpha=0.7)

        # Y limits
        ax_cut.set_ylim(ds_to_simpol_res[dataset]*0.97, ds_to_simpol_res[dataset]*1.03)  # top panel
        ax_main.set_ylim(*ylims[dataset + target])                                        # bottom panel

        # Hide spines BETWEEN axes (correct sides)
        ax_cut.spines['bottom'].set_visible(False)   # bottom edge of top axis
        ax_main.spines['top'].set_visible(False)     # top edge of bottom axis

        # X ticks only on bottom axis
        ax_cut.tick_params(labeltop=False, bottom=False)           # no labels on top axis
        ax_main.xaxis.tick_bottom()                  # labels on bottom axis

        # Diagonal break marks (corrected for top/bottom order)
        d = .5
        kwargs = dict(marker=[(-1,-d), (1,d)], markersize=6,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_cut.plot([0,1],[0,0], transform=ax_cut.transAxes, **kwargs)   # bottom edge of top axis
        ax_main.plot([0,1],[1,1], transform=ax_main.transAxes, **kwargs) # top edge of bottom axis
        ax_cut.set_yticks([round(ds_to_simpol_res[dataset], 2)])
        ax_cut.set_xticks([])

        # Labels/ticks/grid
        ax_main.set_xlabel('Train size', fontsize=fs)
        ax_main.set_ylabel(ylabels_latex[target], fontsize=fs)
        ax_cut.set_title(f'{dataset} {targets_latex[target]} learning curve', fontsize=fs)
        ax_main.set_xticks(xticks[dataset])
        ax_main.set_yticks(yticks[dataset + target])
        ax_main.grid(True)
        ax_cut.grid(True)

        legends  += ['SIMPOL']
        #ax_main.legend(legends, fontsize=fs-0.5, ncol=2, borderpad=0.2, labelspacing=0.25)
        outfilename = f'src/visualization/figs/lcurves/manuscript/wsimpol/{dataset}_{target}_l_curve_w_simpol_cutoff.png'
        fig.tight_layout()
        plt.savefig(outfilename, dpi=100 * export_scaling)
        plt.close()




for dataset in datasets:
    descriptors, legends, targets = dataset_to_desc_leg_tar[dataset]

    for target in ['log_p_sat']:
        fig = plt.figure(figsize=(3.5, 2.625))

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

        plt.title(f'{dataset} {targets_latex[target]} learning curve', fontsize=fs)
        filepath = path.relpath(f'data/{dataset}/KRR_output')

        random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]

        for i, descriptor in enumerate(descriptors):
            if 'TopFP' in descriptor:
                descriptor = ds_topfp[target]

            try:
                desc_mean = pd.read_csv(path.join(filepath, f'mean_{descriptor}_{target}.csv'))
            except FileNotFoundError:
                print(f'WARNING!!! File not found: mean_{descriptor}_{target}.csv, path: {filepath} SKIPPING')
                continue

            try:
                desc_std = pd.read_csv(path.join(filepath, f'std_{descriptor}_{target}.csv'))
            except FileNotFoundError:
                print(f'WARNING!!! File not found: std_{descriptor}_{target}.csv, path: {filepath} SKIPPING')
                continue

            std = desc_std['Test_MAE'].values
            mae = desc_mean['Test_MAE'].values
            t_size = desc_mean['Train_sizes'].values

            labels = {'ATMO_DECIMAL_v4': 'ATMO v5',
                      'ATMOMACCS_DECIMAL_v4' : 'ATMOMACCS v5'}
            # Errorbar curves
            plt.errorbar(
                t_size, mae, yerr=std, marker='o',
                linewidth=lw, markersize=ms, color=colors[i],
                capsize=1.5, capthick=0.9, elinewidth=lw-0.4, label=labels[descriptor]
            )

        # Simpol line (still included, but inside normal axis)
        plt.plot(t_size, np.full_like(t_size, ds_to_simpol_res[dataset]), linestyle="--", color="red", label='SIMPOL', alpha=0.7)

        # Labels
        plt.xlabel('Train size', fontsize=fs)
        plt.ylabel(ylabels_latex[target], fontsize=fs)
        plt.ylim(ylims[dataset + target])
        plt.xticks(xticks[dataset], fontsize=fs)
        plt.yticks(yticks[dataset + target], fontsize=fs)
        plt.grid(True)

        # Place legend outside
        plt.legend(fontsize=fs-0.5, ncol=1,
            borderpad=0.2, labelspacing=0.25,
            loc="upper center", bbox_to_anchor=(0.5, -0.3)  # outside below plot
        )

        outfilename = f'src/visualization/figs/lcurves/manuscript/wsimpol/{dataset}_{target}_l_curve_nocutoff_legends.png'
        fig.tight_layout()
        plt.savefig(outfilename, dpi=100 * export_scaling, bbox_inches="tight")
        plt.close()