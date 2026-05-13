from rdkit import Chem
from rdkit.Chem.rdmolops import RDKFingerprint as TopFP
from argparse import ArgumentParser
import numpy as np
import gc


def to_bool(string: str|bool):
	if type(string) == bool:
		return string
	if string == 'True':
		return True
	elif string == 'False':
		return False
	else:
		raise Exception(f'Cannot cast string to bool for string : {string}')

def smiles_to_mol(filename: str, *, smiles: list[str]|None = None):
	if smiles is None:
		all_smi = open(filename,'r')
	else:
		all_smi = smiles
	mols = [Chem.MolFromSmiles(x.strip()) for x in all_smi]
	return mols

def generate_topfp_mem(mol, *, fpsize=8192, minPath=1, maxPath=8, nBitsPerHash=16):
	return TopFP(mol, fpSize=fpsize, minPath=minPath, maxPath=maxPath, nBitsPerHash=nBitsPerHash)

def generate_topfp(mols: list, *, fpsize=8192, minPath=1, maxPath=8, nBitsPerHash=16):
	fin_train=[]
	for mol in mols:
		fin_train.append(TopFP(mol, fpSize=fpsize, minPath=minPath, 
							   maxPath=maxPath, 
							   nBitsPerHash=nBitsPerHash))
	matrix = []

	i = 0
	for on in fin_train:
		s = [on[i] for i in range(len(on))]
		if(i == 1):
			print(np.array(s).shape)
		i += 1
		matrix.append(s)
	return matrix

def save_to_file(filename: str, matrix: list):
	np.savetxt(filename, matrix, fmt = "%s")


if __name__ == '__main__':
	parser = ArgumentParser(description='TopFP parser')
	parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to smiles.txt')
	parser.add_argument('-s', '--save', type=str, default=False, help='Save to file?')
	args = parser.parse_args()
	filepath = args.data_path
	enable_save = to_bool(args.save)

	mols = smiles_to_mol(filepath)
	topfp = generate_topfp(mols)
	if enable_save:
		filename = input('Input filename with path: ')
		assert filename != '', 'Valid path required'
		save_to_file(filename, topfp)

