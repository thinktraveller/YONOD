import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os import path

datasets = ['Wang', 'GeckoQ', 'Ferraz-Caetano', 'Li']
savepath = path.relpath('src/validation/1006')

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
    'Wanglog_p_sat': (0.24, 0.75),
    'Wanglog_kwg': (0.33, 1.0),
    'Wanglog_kwiomg': (0.23, 0.71),
    'GeckoQlog_p_sat': (0.67, 1.38),
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
    'Wanglog_p_sat': np.linspace(0.25, 0.65, num=9),
    'Wanglog_kwg': np.linspace(0.35, 0.85, num=6),
    'Wanglog_kwiomg': np.linspace(0.25, 0.65, num=9),
    'GeckoQlog_p_sat': np.linspace(0.7,1.2, num=6),
    'Litg' : np.linspace(16, 32, num=9),
    'Ferraz-Caetanodvap' : np.linspace(2,16, num=8)
}

dataset_to_desc_leg_tar = {
    'Wang': (['MACCS', 'ATMO_DECIMAL_v4', 'ATMOMACCS_v1', 'ATMOMACCS_v2', 
              'TopFP', 'ATMOMACCS_v3', 'ATMOMACCS_v4', 'ATMOMACCS_DECIMAL_v4'],
             ['MACCS', 'ATMO v5', 'ATMOMACCS v1', 'ATMOMACCS v2', 
              'TopFP', 'ATMOMACCS v3', 'ATMOMACCS v4', 'ATMOMACCS v5'],
             ['log_p_sat','log_kwiomg', 'log_kwg']
             ),
    'GeckoQ': (['MACCS', 'ATMO_DECIMAL_v4', 'ATMOMACCS_v1', 'ATMOMACCS_v2', 
              'TopFP', 'ATMOMACCS_v3', 'ATMOMACCS_v4', 'ATMOMACCS_DECIMAL_v4'],
             ['MACCS', 'ATMO v5', 'ATMOMACCS v1', 'ATMOMACCS v2', 
              'TopFP', 'ATMOMACCS v3', 'ATMOMACCS v4', 'ATMOMACCS v5'],
             ['log_p_sat']),
    'Ferraz-Caetano': (['MACCS', 'ATMO_DECIMAL_v4', 'ATMOMACCS_v1', 'ATMOMACCS_v2', 
            'TopFP', 'ATMOMACCS_v3', 'ATMOMACCS_v4', 'ATMOMACCS_DECIMAL_v4'],
            #'Ensemble', 
            ['MACCS', 'ATMO v5', 'ATMOMACCS v1', 'ATMOMACCS v2', 
            'TopFP', 'ATMOMACCS v3', 'ATMOMACCS v4', 'ATMOMACCS v5'],
            ['dvap']),
    'Li': (['MACCS', 'ATMO_DECIMAL_v4', 'ATMOMACCS_v1', 'ATMOMACCS_v2', 
              'TopFP', 'ATMOMACCS_v3', 'ATMOMACCS_v4', 'ATMOMACCS_DECIMAL_v4'],
             ['MACCS', 'ATMO v5', 'ATMOMACCS v1', 'ATMOMACCS v2', 
              'TopFP', 'ATMOMACCS v3', 'ATMOMACCS v4', 'ATMOMACCS v5'],
             ['tg']),

}

ds_topfp = {
    'log_p_sat' : 'TopFP_log_p_sat',
    'log_kwg' : 'TopFP_log_kwg',
    'log_kwiomg' : 'TopFP_log_kwiomg',
    'dvap' : 'TopFP_dvap',
    'tg' : 'TopFP_tg'
} 

export_scaling = 1
export_scaling = 12
fs = 7.5
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
for dataset in datasets:
    if dataset == 'Ferraz-Caetano' or dataset == 'Li' or dataset == 'GeckoQ':
        lw = 1
        ms = 3.3
    else:
        lw = 1.2
        ms = 4
    descriptors, \
        legends, \
            targets = dataset_to_desc_leg_tar[dataset]
    for target in targets:
            fig = plt.figure(figsize=(3.5, 2.625))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
            plt.title(f'{dataset} {targets_latex[target]} learning curve', fontsize=fs)
            filepath = path.relpath(f'data/{dataset}/KRR_output')
            random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
            '''
            if dataset == 'Ferraz-Caetano':
                plt.axhline(y=3.02, color='red', linestyle='--')
            '''
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
                #plt.plot(t_size, mae, '-o')
                plt.errorbar(t_size, mae, yerr = std, marker = 'o',
                             linewidth = lw, markersize = ms, color=colors[i],
                             capsize=1.5, capthick=0.9, elinewidth=lw-0.4)
                plt.xlabel('Train size', fontsize=fs)
                plt.ylabel(ylabels_latex[target], fontsize = fs)
                plt.ylim(ylims[dataset + target])
            plt.legend(legends, fontsize=fs-0.5, ncol=2, borderpad = 0.2, labelspacing = 0.25)
            plt.xticks(xticks[dataset], fontsize=fs)
            plt.yticks(yticks[dataset + target],fontsize=fs)
            plt.grid(True)
            outfilename = f'src/visualization/figs/lcurves/{dataset}_{target}_l_curve.png'
            fig.tight_layout()
            plt.savefig(outfilename, dpi=100 * export_scaling)
            plt.close()
            '''
            im = Image.open(f'src/visualization/1006/{dataset}_{target}_l_curve.png')
            im_resize = im.resize((1050, 788))
            im_resize.save(outfilename, optimize=True, quality=50, dpi=(300, 300))
            '''