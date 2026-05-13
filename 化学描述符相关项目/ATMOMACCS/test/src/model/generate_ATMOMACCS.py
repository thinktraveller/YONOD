import pandas as pd
import numpy as np
from rdkit import Chem
from ATMOMACCS_no_binary import pyGenATMOMACCS as pyGenATMOMACCS_DECIMAL
from ATMOMACCS_no_binary import pyGenATMOKeys as pyGenATMO_DECIMAL
from ATMOMACCS import pyGenATMOMACCS
from ATMOMACCS import pyGenATMOKeys
from generate_MACCS import generate_MACCS as GenMACCS

def generate_ATMOMACCS(dataset: str, ver: int, decimal: bool = False, bits: int = 6):
  smiles = np.loadtxt(f'data/{dataset}/smiles.txt', dtype=np.str_, comments=None)
  if decimal:
    fp = [pyGenATMOMACCS_DECIMAL(Chem.MolFromSmiles(smi), version=ver)\
          for smi in smiles]
  else:
    fp = [pyGenATMOMACCS(Chem.MolFromSmiles(smi), version=ver, bit_width=bits)\
          for smi in smiles]
  fp = np.matrix(fp, dtype=int)
  fp = fp[:, 1:] # skip the "empty key" on index 0
  return fp

def generate_ATMO(dataset: str, ver: int, decimal: bool = False, bits: int = 6):
  smiles = np.loadtxt(f'data/{dataset}/smiles.txt', dtype=np.str_, comments=None)
  if decimal:
    atmo = [pyGenATMO_DECIMAL(Chem.MolFromSmiles(smi), version=ver)\
            for smi in smiles]
  else:
    atmo = [pyGenATMOKeys(Chem.MolFromSmiles(smi), version=ver, bit_width=bits)\
            for smi in smiles]
    atmo  = np.matrix(atmo, dtype=int)
  return atmo

def generate_all():
  versions = [1,2,3,4]
  datasets = ['Wang', 'Li', 'Ferraz-Caetano', 'GeckoQ']

  # generate ATMOMACCS versions 1-4:
  for ver in versions: 
    print(f'ATMOMACCS version: {ver}')
    for dataset in datasets:
      print(f'Data set: {dataset}')
      desc_fpath = f'data/{dataset}/descriptors'
      if dataset == 'Ferraz-Caetano':
        bits = 7
      else:
        bits = 6

      atmomaccs = generate_ATMOMACCS(dataset, ver, decimal=False, bits=bits)
      np.savetxt(f'{desc_fpath}/ATMOMACCS_v{ver}.txt', atmomaccs, fmt="%s")
      if ver == 4:
        atmo = generate_ATMO(dataset, ver, decimal=False, bits=bits)
        np.savetxt(f'{desc_fpath}/ATMO_v{ver}.txt', atmo, fmt="%s")
  # generate ATMO v5 and ATMOMACCS v5 
  for dataset in datasets:
    print(f'Data set: {dataset}')
    ver = 4

    atmomaccs = generate_ATMOMACCS(dataset, ver, decimal=True)
    np.savetxt(f'{desc_fpath}/ATMOMACCS_DECIMAL_v{ver}.txt', atmomaccs, fmt="%s")

    atmo = generate_ATMO(dataset, ver, decimal=True)
    np.savetxt(f'{desc_fpath}/ATMO_DECIMAL_v{ver}.txt', atmo, fmt="%s")

  # generate MACCS
  for dataset in datasets:
    desc_fpath = f'data/{dataset}/descriptors'
    maccs = np.array(GenMACCS(f'data/{dataset}')).astype(np.uint8)
    np.savetxt(f'{desc_fpath}/MACCS.txt', maccs, fmt="%s")

  # generate ATMOMACCS_ALT_v3
  for dataset in datasets:
    print(f'Data set: {dataset}')
    desc_fpath = f'data/{dataset}/descriptors'
    if dataset == 'Ferraz-Caetano':
      bits = 7
    else:
      bits = 6
    atmomaccs_v4 = generate_ATMOMACCS(dataset, 4, decimal=False, bits=bits)
    oxygens = atmomaccs_v4[:, -bits:]
    print(f'oxygens.shape: {oxygens.shape}')
    atmomaccs_v2 = generate_ATMOMACCS(dataset, 2, decimal=False, bits=bits)
    atmomaccs_alt_v3 = np.hstack([atmomaccs_v2, oxygens])
    print(f'atmomaccs_alt_v3.shape: {atmomaccs_alt_v3.shape}')
    np.savetxt(f'{desc_fpath}/ATMOMACCS_ALT_v3.txt', atmomaccs_alt_v3, fmt="%s")


if __name__ == '__main__':
  generate_all()


'''
python src/model/generate_ATMOMACCS.py
'''