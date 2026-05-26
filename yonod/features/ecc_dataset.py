"""ECC dataset loader for Enantioselective Cross-Coupling reactions.

Reads Raw_Dataset.xlsx and returns three parallel arrays:
  - ligand_smiles / product_smiles (str lists)
  - temperatures in Kelvin (float ndarray)
  - ddG values in kcal/mol (float ndarray)

The ΔΔG column name contains Unicode characters (△△G), so we use fuzzy
matching on any column whose name contains 'G' and 'Kcal'.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _find_ddg_column(df: pd.DataFrame) -> str:
    """Return the column name for ΔΔG, tolerating Unicode variations."""
    for col in df.columns:
        col_lower = col.lower()
        if ("g" in col_lower or "△" in col) and ("kcal" in col_lower or "kj" in col_lower):
            return col
    # Fallback: any column with 'G' and numeric-ish content
    for col in df.columns:
        if col.strip().upper() in ("△△G", "DDG", "DELTADELTAG", "ΔΔG"):
            return col
    raise KeyError(
        f"Cannot find ΔΔG column. Available columns: {list(df.columns)}"
    )


def _is_valid_smiles(smi: str) -> bool:
    if not _RDKIT_AVAILABLE:
        return bool(smi and str(smi).strip() not in ("", "nan", "None"))
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol is not None
    except Exception:
        return False


def load_ecc_dataset(
    xlsx_path: str | Path,
    celsius_to_kelvin: bool = True,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Load Raw_Dataset.xlsx and return cleaned arrays.

    Args:
        xlsx_path: path to Raw_Dataset.xlsx
        celsius_to_kelvin: if True, add 273.15 to temperature column

    Returns:
        ligand_smiles:  list[str]
        product_smiles: list[str]
        temperatures:   float ndarray, Kelvin
        ddG:            float ndarray, kcal/mol
    """
    xlsx_path = Path(xlsx_path)
    if xlsx_path.suffix.lower() == ".csv":
        df = pd.read_csv(xlsx_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(xlsx_path, engine="openpyxl")

    # --- locate required columns ---
    # SMILES columns
    lig_col = next(
        (c for c in df.columns if "ligand" in c.lower() and "smiles" in c.lower()),
        None,
    )
    prod_col = next(
        (c for c in df.columns if "product" in c.lower() and "smiles" in c.lower()),
        None,
    )
    temp_col = next(
        (c for c in df.columns if "temp" in c.lower()), None
    )
    ddg_col = _find_ddg_column(df)

    if lig_col is None:
        raise KeyError(f"Cannot find Ligand_SMILES column. Columns: {list(df.columns)}")
    if prod_col is None:
        raise KeyError(f"Cannot find Product_SMILES column. Columns: {list(df.columns)}")
    if temp_col is None:
        raise KeyError(f"Cannot find Temperature column. Columns: {list(df.columns)}")

    df = df[[lig_col, prod_col, temp_col, ddg_col]].copy()
    df.columns = ["ligand_smiles", "product_smiles", "temperature", "ddG"]

    # --- coerce numerics ---
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["ddG"] = pd.to_numeric(df["ddG"], errors="coerce")

    # --- drop rows missing any field ---
    df = df.dropna()

    # --- filter invalid SMILES ---
    lig_valid = df["ligand_smiles"].apply(_is_valid_smiles)
    prod_valid = df["product_smiles"].apply(_is_valid_smiles)
    df = df[lig_valid & prod_valid].reset_index(drop=True)

    # --- temperature unit conversion ---
    if celsius_to_kelvin:
        df["temperature"] = df["temperature"] + 273.15

    ligand_smiles = df["ligand_smiles"].tolist()
    product_smiles = df["product_smiles"].tolist()
    temperatures = df["temperature"].to_numpy(dtype=np.float64)
    ddG = df["ddG"].to_numpy(dtype=np.float64)

    return ligand_smiles, product_smiles, temperatures, ddG
