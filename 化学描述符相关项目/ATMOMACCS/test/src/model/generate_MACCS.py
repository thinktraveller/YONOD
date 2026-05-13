#Author: Linus Lind
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import sys
from rdkit import Chem
from os import path
import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
GenMACCSKeys = GetMACCSKeysFingerprint

def generate_MACCS(data_path):
    filepath = path.relpath(data_path)
    name_of_file = f'smiles.txt'
    filename= path.join(filepath, name_of_file)
    all_smi = open(filename,'r')
    mol_train = [Chem.MolFromSmiles(x.strip()) for x in all_smi]
    fin_train = [GenMACCSKeys(x) for x in mol_train]
    matrix = []
    for on in fin_train:
        # Exclude the first column from each MACCS key fingerprint
        s = [on[i] for i in range(1, len(on))]
        matrix.append(s)
    return matrix

def main(data_path, filename= 'MACCS.txt'):
    matrix = generate_MACCS(data_path)
    filepath = path.relpath(data_path)
    fileoutname = path.join(filepath, filename)
    np.savetxt(fileoutname, matrix, fmt="%s")

if __name__ == "__main__":
    main(sys.argv[1])
    #py src/model/generate_MACCS.py 'data/Ferraz-Caetano'
