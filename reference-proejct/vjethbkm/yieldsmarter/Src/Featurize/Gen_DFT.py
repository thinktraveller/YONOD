import json, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

cfg = json.load(open(sys.argv[1]))

reaction_csv = cfg["reaction_csv"]
target_col = cfg.get("target_col", "Yield")
out = Path(cfg.get("output_npz", "reaction_dft_dataset.npz"))
out.parent.mkdir(parents=True, exist_ok=True)

components_order = cfg["components_order"]
reaction_smiles_cols = cfg["reaction_smiles_cols"]
props_paths = cfg["props_paths"]
smiles_map_paths = cfg["smiles_map_paths"]

# Load reaction table
rxn = pd.read_csv(reaction_csv)

# Load props + maps
props = {k: pd.read_csv(v) for k, v in props_paths.items()}
maps = {k: pd.read_csv(v) for k, v in smiles_map_paths.items()}

# Normalize DFT IDs: "Amine_12" -> "12"
for key, df in props.items():
    df["ID"] = df["ID"].astype(str).apply(
        lambda x: re.sub(r"^[A-Za-z_]+", "", x)).str.strip()
    props[key] = df

# Merge SMILES map ↔ props on numeric ID (inner keeps only those with DFT)
for key in props:
    maps[key]["ID"] = maps[key]["ID"].astype(str).str.strip()
    props[key]["ID"] = props[key]["ID"].astype(str).str.strip()
    props[key] = pd.merge(maps[key], props[key], on="ID", how="inner")


def get_vector(component, smiles):
    smiles = str(smiles).strip().replace('"', '')
    df = props[component]
    # assumes column 1 is SMILES in the merged table: [ID, <smiles>, <properties...>]
    df_smiles = df.iloc[:, 1].astype(str).str.strip().str.replace('"', '')
    row = df[df_smiles.str.lower() == smiles.lower()]
    if row.empty:
        raise ValueError(f"No DFT found for {component} SMILES: {smiles}")
    return row.iloc[0, 2:].to_numpy(dtype=float)


X, y = [], []
for i, r in rxn.iterrows():
    try:
        vecs = []
        for comp in components_order:
            col = reaction_smiles_cols[comp]
            vecs.append(get_vector(comp, r[col]))
        X.append(np.concatenate(vecs))
        y.append(r[target_col])
    except Exception as e:
        import traceback
        traceback.print_exc()
        assert (0)
        print(f"Skipping row {i}: {e}")

if X:
    X = np.vstack(X)
    y = np.array(y, dtype=float)
    np.savez(out, X=X, y=y)
    print(f"Saved {out} (X: {X.shape}, y: {y.shape})")
else:
    print("No valid rows were found — check missing SMILES or DFT data.")
