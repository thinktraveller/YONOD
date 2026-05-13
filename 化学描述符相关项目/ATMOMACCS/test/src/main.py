""" Make descriptor file and run KRR training and prediction"""

# Author: Linus Lind, 2025
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
# Modified by Hilda Sandström: CLI interfacing 
from model import krr
from argparse import ArgumentParser
from typing import NoReturn
from model.generate_ATMOMACCS import generate_all
from model.generate_optimal_topfp import generate_opt_topfp
from model.generate_MACCS import main as generate_maccs

# Create a new ArgumentParser object
parser = ArgumentParser(description='parser to bespe')
parser.add_argument('-v', '--version', type=str, required=True, help='Path to SMARTS patterns CSV file')
parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the data directory')
parser.add_argument('-b', '--desc', default='ATMOMACCS', type=str, help='descriptor')
parser.add_argument('-t', '--target', type=str, help='target value filename')
parser.add_argument('-ds', '--dataset', type=str, required=True, help='type of dataset: gecko, wang or example')
parser.add_argument('-s', '--seed', type=str, required=True, help='seed')
parser.add_argument('-sh', '--shap', type=str, default='False', help='Whether to do SHAP')
parser.add_argument('-re', '--generate_descriptors', type=str, default='False', help='Generate descriptors? Recommended to run on the first run, after which descriptors are saved in the data directory')

# the main for executing the whole code, Generates the chosen descriptors and
# runs the KRR script. Generates a summary of the results.

def main() -> NoReturn:
    # generate descriptors
    args = parser.parse_args()
    data_path: str = args.data_path
    ver: int = int(args.version)
    desc: str = args.desc
    seed: str = args.seed
    if args.shap == 'True':
        shap = True
        print('WARNING: SHAP is very expensive, consider running instead shap_vals.py')
    else:
        shap = False
    target = args.target

    if int(ver) == 0:
        descriptor = desc
    else:
        descriptor = f'{desc}_v{ver}'
    print(f'Descriptor: {descriptor}')
    #random_state = [12, 432, 5, 7543, 12343, 452, 325432435, 326, 436, 2435]
    if 'maccs' in descriptor.lower():
        generate_maccs(data_path)

    if 'atmo' in descriptor.lower():
        generate_all() # generates all ATMO, and ATMOMACCS descriptors
    elif 'topfp' in descriptor.lower():
        generate_opt_topfp() # Generates TopFP for all datasets
        print('WARNING: TopFP is expensive to generate for Li and Ferraz-Caetano datasets. ' \
                'Consider running generate_optimal_topfp.py beforehand')
    else:
        raise NotImplementedError(f'Descriptor: {descriptor} not implemented!') 

    
    # run KRR model
    print(f'Starting KRR script')
    krr.main(descriptor, target, data_path, args.dataset, save_predictions=True, random_seeds=int(seed), plotting=False, calculate_shap=shap)
    print('Finished')

if __name__ == "__main__":
    main()
    #py -3.12 src/main.py -v 4 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s 2435
    #py -3.12 src/main.py -v 4 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s 2435
    #py -3.12 src/main.py -v 4 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s 2435
    #py -3.12 src/main.py -v 4 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s 2435
    #py -3.12 src/main.py -v 4 -d 'data/GeckoQ' -t 'log_p_sat.txt' -ds 'Gecko' -s 2435