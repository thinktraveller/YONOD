import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit import ForceField
from rdkit.Chem import rdForceFieldHelpers

cfg = json.load(open(sys.argv[1]))

xyz_dir = Path(cfg.get("xyz_dir", "."))


def generate_conformers(smiles, num_confs=500):
    ps = rdDistGeom.srETKDGv3()
    ps.numThreads = 12
    ps.randomSeed = 0xf00d
    ps.pruneRmsThresh = 0.5
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=ps)
    tmol = Chem.Mol(mol)
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(tmol):
        energies = [
            e for _, e in rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                tmol, numThreads=12)
        ]
    elif rdForceFieldHelpers.UFFHasAllMoleculeParams(tmol):
        energies = [
            e for _, e in rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                tmol, numThreads=12)
        ]
    else:
        print(
            f"Warning: no force field for {smiles}, picking a random conformer"
        )
        energies = [(0, cid) for cid in cids]
    energies = list(zip(energies, cids))
    energies.sort()
    best_cid = energies[0][1]
    return tmol, best_cid


def gen_conf(row, comp):
    print(f"Generating conformer for {comp['name']} ID {row['ID']}...")
    tmol, best_cid = generate_conformers(row[comp['reaction_col']])
    pn = str(xyz_dir / comp['xyz_template'].format(id=row['ID']))
    Chem.MolToXYZFile(tmol, pn, confId=best_cid)


for comp in cfg["components"]:
    # print(f"Processing {comp['name']}...")
    df = pd.read_csv(comp["lookup_csv"])
    df["ID"] = df[comp["lookup_key_col"]].astype(str).str.strip()
    df.apply(lambda row: gen_conf(row, comp), axis=1)
