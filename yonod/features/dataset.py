"""Generic CSV dataset loader and single-molecule featurizer.

Contract (v1.1+):
    Any input CSV must have at least two columns:
      * column 0 -> SMILES string
      * column 1 -> float label (yield, solubility, logP, ...; any continuous target)
    Header row is required (used for column naming in the report only).
    Additional columns are ignored.

This replaces the amide-condensation-specific 9-column schema used in v1.0
(see ``reaction_featurizer.py`` for the legacy reaction-level path).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..descriptors.base import BaseDescriptor


def load_dataset(
    csv_path: str | Path,
) -> Tuple[List[str], np.ndarray, str, str]:
    """Read a 2-column CSV.

    Returns:
        smiles:        list of SMILES strings (column 0)
        labels:        1-D float ndarray (column 1)
        smiles_header: name of column 0 (for plot axis / report)
        label_header:  name of column 1 (for plot axis / report)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(
            f"{csv_path} has {df.shape[1]} column(s); expected at least 2 "
            "(column 0 = SMILES, column 1 = float label)."
        )
    smiles_header = str(df.columns[0])
    label_header = str(df.columns[1])
    smiles = df.iloc[:, 0].astype(str).tolist()
    labels = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    # Drop rows where label became NaN (non-numeric or empty cells).
    if np.isnan(labels).any():
        keep = ~np.isnan(labels)
        smiles = [s for s, k in zip(smiles, keep) if k]
        labels = labels[keep]
    return smiles, labels, smiles_header, label_header


class MoleculeFeaturizer:
    """Single-molecule featurizer: SMILES list -> (X_valid, mask).

    Thin wrapper that delegates to the chosen ``BaseDescriptor`` and exposes
    the same ``transform()`` contract as the legacy ``ReactionFeaturizer``.
    """

    def __init__(self, descriptor: BaseDescriptor) -> None:
        self.descriptor = descriptor

    @property
    def output_dim(self) -> int:
        return self.descriptor.output_dim

    def transform(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Featurize and drop rows that failed parsing.

        Returns (X_valid, mask) -- mask has the original list length so the
        caller can filter the label vector consistently.
        """
        features, mask = self.descriptor.featurize(smiles_list)
        return features[mask], mask
