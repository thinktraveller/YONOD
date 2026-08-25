"""Low-cost reaction descriptors for the Stage A smoke benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureResult:
    name: str
    train: np.ndarray
    test: np.ndarray
    dim: int
    elapsed_s: float
    invalid_smiles: int = 0


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 6)


def build_ohe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    component_cols: list[str],
) -> FeatureResult:
    start = perf_counter()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train = encoder.fit_transform(train_df[component_cols].fillna("<missing>"))
    test = encoder.transform(test_df[component_cols].fillna("<missing>"))
    return FeatureResult("ohe", train, test, train.shape[1], _elapsed(start))


def _mol_from_smiles(smiles: object) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _morgan_counts(smiles: object, radius: int, n_bits: int) -> tuple[np.ndarray, int]:
    mol = _mol_from_smiles(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return arr, 1
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetCountFingerprint(mol)
    for idx, value in fp.GetNonzeroElements().items():
        arr[int(idx)] = float(value)
    return arr, 0


def _physchem(smiles: object, descriptor_names: list[str]) -> tuple[np.ndarray, int]:
    mol = _mol_from_smiles(smiles)
    arr = np.zeros(len(descriptor_names), dtype=np.float32)
    if mol is None:
        return arr, 1
    descriptor_map = dict(Descriptors.descList)
    for index, name in enumerate(descriptor_names):
        try:
            value = float(descriptor_map[name](mol))
        except Exception:
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        arr[index] = value
    return arr, 0


def _concat_component_features(
    df: pd.DataFrame,
    component_cols: list[str],
    featurize_one,
) -> tuple[np.ndarray, int]:
    rows: list[np.ndarray] = []
    invalid = 0
    for _, row in df.iterrows():
        blocks = []
        for column in component_cols:
            block, bad = featurize_one(row[column])
            blocks.append(block)
            invalid += bad
        rows.append(np.concatenate(blocks).astype(np.float32))
    return np.vstack(rows), invalid


def build_morgan(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    component_cols: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> FeatureResult:
    start = perf_counter()
    train, bad_train = _concat_component_features(
        train_df,
        component_cols,
        lambda smiles: _morgan_counts(smiles, radius, n_bits),
    )
    test, bad_test = _concat_component_features(
        test_df,
        component_cols,
        lambda smiles: _morgan_counts(smiles, radius, n_bits),
    )
    return FeatureResult("morgan", train, test, train.shape[1], _elapsed(start), bad_train + bad_test)


def build_physchem(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    component_cols: list[str],
) -> FeatureResult:
    start = perf_counter()
    descriptor_names = [name for name, _ in Descriptors.descList]
    train, bad_train = _concat_component_features(
        train_df,
        component_cols,
        lambda smiles: _physchem(smiles, descriptor_names),
    )
    test, bad_test = _concat_component_features(
        test_df,
        component_cols,
        lambda smiles: _physchem(smiles, descriptor_names),
    )
    return FeatureResult("physchem", train, test, train.shape[1], _elapsed(start), bad_train + bad_test)


def build_descriptor(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    component_cols: list[str],
) -> FeatureResult:
    if name == "ohe":
        return build_ohe(train_df, test_df, component_cols)
    if name == "morgan":
        return build_morgan(train_df, test_df, component_cols)
    if name == "physchem":
        return build_physchem(train_df, test_df, component_cols)
    raise KeyError(f"Unsupported descriptor for Stage A smoke benchmark: {name}")
