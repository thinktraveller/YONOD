#!/usr/bin/env python3
"""
Gen_PhysChem.py (aka configurable 3rdkit)

Reads a JSON config (like SL1_PhysChem.json) and generates RDKit PhysChem descriptors
per SMILES column, concatenates into X, aligns y, and saves:
- unscaled .npz
- optionally scaled .npz + scaler.pkl
- a log file

For example run:
  python Gen_PhysChem.py SM_PhysChem.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def compute_descriptors_cached(smiles_series,
                               log,
                               num_descriptors=20,
                               min_frac_variance=None):
    """
    Compute descriptors once per unique (canonical) SMILES then map back to rows.
    Returns a Series of lists (len = n_descriptors) aligned to the original series.
    """
    # Canonicalize and get uniques
    osmiles = smiles_series.dropna()
    canonical_series = osmiles.map(Chem.CanonSmiles)
    smiles_map = dict(zip(osmiles, canonical_series))
    unique_smiles = canonical_series.unique().tolist()

    log(f"     Computing descriptors for {len(unique_smiles)} unique SMILES")

    tm = Chem.MolFromSmiles('CCCC')
    dvs = Descriptors.CalcMolDescriptors(tm)
    num_descriptors = len(dvs)

    descrs = []
    for smi in unique_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            log(f"       Warning: invalid SMILES '{smi}' will get zero descriptors"
                )
            descrs.append([0.0] * num_descriptors)
            continue
        descrs.append(list(Descriptors.CalcMolDescriptors(m).values()))
    descrs = np.array(descrs, dtype=np.float64)

    # z-scale the descriptors
    stds = np.std(descrs, axis=0)
    ok = [i for i, s in enumerate(stds) if not np.isnan(s) and s != 0.0]
    filtered_descrs = descrs[:, ok]
    means = np.mean(filtered_descrs, axis=0)
    stds = np.std(filtered_descrs, axis=0)
    scaled_descrs = (filtered_descrs - means) / stds

    n_components = num_descriptors
    if min_frac_variance is not None:
        n_components = min_frac_variance
    pca = PCA(n_components=n_components, svd_solver='full')
    final_descrs = pca.fit_transform(scaled_descrs)

    res = []
    for smi in smiles_series:
        csmi = smiles_map.get(smi, None)
        if csmi is None:
            # log(f"       Warning: SMILES '{smi}' not found in map, assigning zero descriptors"
            #     )
            res.append([0.0] * pca.n_components_)
            continue
        idx = unique_smiles.index(csmi)
        res.append(final_descrs[idx])

    return res


def main():
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python Gen_PhysChem.py <config.json>")

    cfg_path = Path(sys.argv[1])
    cfg = json.load(open(cfg_path, "r"))

    # Ensure output dirs exist
    out_npz = Path(cfg["output_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(
        cfg.get("log_file",
                out_npz.with_suffix("").as_posix() + "_log.txt"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "w")

    def log(msg):
        lf.write(f"{ts()} - {msg}\n")
        lf.flush()
        print(msg)

    reaction_csv = Path(cfg["reaction_csv"])
    if not reaction_csv.exists():
        log(f"ERROR: reaction_csv not found: {reaction_csv}")
        lf.close()
        raise FileNotFoundError(reaction_csv)
    target_col = cfg.get("target_col", "Yield")
    df = pd.read_csv(reaction_csv)
    if target_col not in df.columns:
        log(f"ERROR: target column '{target_col}' missing from CSV columns.")
        lf.close()
        raise ValueError(f"Target column '{target_col}' missing.")
    if cfg.get("skip_rows_with_missing_values", False):
        initial_len = len(df)
        df = df.dropna(subset=cfg["components"] + [target_col])
        log(f"Dropped {initial_len - len(df)} rows with missing values in components or target."
            )
    min_frac_variance = cfg.get("min_frac_variance", None)

    log(f"Config: {cfg_path}")
    log(f"Reaction CSV: {reaction_csv}")
    log(f"Target column: {target_col}")
    log(f"Output (unscaled): {out_npz}")
    log(f"Min frac variance for PCA: {min_frac_variance}")
    all_descrs = None
    for i, component in enumerate(cfg["components"]):
        descrs = compute_descriptors_cached(
            df[component], log, min_frac_variance=min_frac_variance)
        print(
            f"Component '{component}': computed descriptors shape: {np.array(descrs).shape}"
        )
        if not i:
            all_descrs = list(descrs)
        else:
            for i, (d, new_d) in enumerate(zip(all_descrs, descrs)):
                all_descrs[i] = np.concatenate((d, new_d))
    X = np.array(all_descrs, dtype=np.float64)
    y = df[target_col].values.astype(float)

    log("\nDone.")
    lf.close()

    np.savez(out_npz, X=X, y=y)
    print(out_npz, X.shape, y.shape)


if __name__ == "__main__":
    main()
