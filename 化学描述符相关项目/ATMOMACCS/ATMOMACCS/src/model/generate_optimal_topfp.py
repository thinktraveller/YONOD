'''Static hard-coded script to generate topfp with optimal hyperparameters'''
from generate_topological import smiles_to_mol, generate_topfp, save_to_file
from typing import NoReturn
def generate_opt_topfp() -> NoReturn:
    '''
    Generates optimized TopFP for each dataset 
    '''
    opt_hyperparams = {
    'Wang_log_p_sat': {'fpsize': 8192,
                    'minPath': 1,
                    'maxPath': 7,
                    'nBitsPerHash': 7},

    'Wang_log_kwiomg': {'fpsize': 4096,
                        'minPath': 1,
                        'maxPath': 7,
                        'nBitsPerHash': 7},

    'Wang_log_kwg': {'fpsize': 8192,
                    'minPath': 1,
                    'maxPath': 7,
                    'nBitsPerHash': 6},

    'Li_tg': {'fpsize': 8192,
            'minPath': 1,
            'maxPath': 8,
            'nBitsPerHash': 16},

    'Ferraz-Caetano_dvap': {'fpsize': 8192,
                            'minPath': 1,
                            'maxPath': 18,
                            'nBitsPerHash': 13},

    'GeckoQ_log_p_sat': {'fpsize': 8192,
                            'minPath': 2,
                            'maxPath': 8,
                            'nBitsPerHash': 6}
    }

    datasets = ['Wang', 'Li', 'Ferraz-Caetano', 'GeckoQ']
    targets = {
        'Wang' : ['log_p_sat', 'log_kwg', 'log_kwiomg'],
        'Li': ['tg'],
        'Ferraz-Caetano': ['dvap'],
        'GeckoQ': ['log_p_sat']
    }

    for dataset in datasets:
        for target in targets[dataset]:
            smiles_file = f'data/{dataset}/smiles.txt'
            mols = smiles_to_mol(smiles_file)
            params = opt_hyperparams[f'{dataset}_{target}']
            topfp = generate_topfp(mols, **params)
            save_to_file(f'data/{dataset}/descriptors/TopFP_{target}.txt', topfp)

if __name__ == '__main__':
    generate_opt_topfp()

'''
Example usage (CLI):

python src/model/generate_optimal_topfp.py
'''