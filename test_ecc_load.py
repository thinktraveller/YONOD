# -*- coding: utf-8 -*-
"""Smoke test: load ECC dataset and verify ee conversion."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yonod_yield.features.ecc_dataset import load_ecc_dataset
from yonod_yield.metrics.ee_metrics import ddG_to_ee, ee_mae

xlsx = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "数据集",
    "Enantioselective-Cross-Coupling-Prediction",
    "Data",
    "csv",
    "Raw_Dataset.csv",
)

print(f"Path: {xlsx}")
print(f"Exists: {os.path.exists(xlsx)}")

lig, prod, temps, ddG = load_ecc_dataset(xlsx)
print(f"Loaded: {len(lig)} valid rows")
print(f"Temperature range: {temps.min():.1f} - {temps.max():.1f} K")
print(f"ddG range: {ddG.min():.3f} - {ddG.max():.3f} kcal/mol")
print(f"Sample ligand SMILES: {lig[0][:70]}")
print(f"Sample product SMILES: {prod[0][:70]}")

import numpy as np
ee_sample = ddG_to_ee(ddG[:5], temps[:5])
print(f"ee% for first 5 rows: {ee_sample.round(1)}")

# Test ee_mae with dummy prediction
dummy_pred = ddG[:10] + 0.5
mae_val = ee_mae(ddG[:10], dummy_pred, temps[:10])
print(f"ee_MAE test (ddG+0.5 error): {mae_val:.2f}%")
print("ALL OK")
