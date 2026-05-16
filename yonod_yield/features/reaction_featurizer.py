"""Reaction-level feature builder.

For each row of the amide-condensation CSV, assembles the descriptor vectors
of 6 molecules (2 substrates + 4 reagents) into a single concatenated feature
of length ``6 * descriptor.output_dim``.

Substrates (``sub_1_smiles`` / ``sub_2_smiles``) are featurized directly;
reagent columns are looked up in the precomputed cache built by
``reagent_cache.build_reagent_feat_cache``.

Only ``sub_1`` / ``sub_2`` featurization failures invalidate a row -- reagent
slots always resolve (``(无)`` -> zero vector). The returned mask refers to
the original df length so the caller can filter ``y`` consistently.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..descriptors.base import BaseDescriptor
from .reagent_cache import (
    REAGENT_COLS,
    build_reagent_feat_cache,
    get_reagent_feat,
)


class ReactionFeaturizer:
    """Combine 6 molecular descriptor vectors into one reaction-level feature."""

    def __init__(self, descriptor: BaseDescriptor) -> None:
        self.descriptor = descriptor

    @property
    def output_dim(self) -> int:
        return 6 * self.descriptor.output_dim

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X_valid, mask)``.

        Args:
            df: DataFrame with ``sub_1_smiles``, ``sub_2_smiles`` and the four
                reagent columns (``activation_id``, ``additive_id``,
                ``base_id``, ``solvent_id``).

        Returns:
            X_valid: ndarray of shape ``(n_valid, 6 * d)`` -- rows with failed
                substrate featurization are dropped.
            mask: bool ndarray of shape ``(len(df),)`` indicating which input
                rows survived (caller can do ``y = df['yield'].values[mask]``).
        """
        for col in ["sub_1_smiles", "sub_2_smiles", *REAGENT_COLS]:
            if col not in df.columns:
                raise KeyError(f"Missing expected column {col!r}")

        cache = build_reagent_feat_cache(self.descriptor, df)
        d = self.descriptor.output_dim
        n = len(df)

        sub1, sub1_mask = self.descriptor.featurize(df["sub_1_smiles"].tolist())
        sub2, sub2_mask = self.descriptor.featurize(df["sub_2_smiles"].tolist())

        reagent_blocks = []
        for col in REAGENT_COLS:
            block = np.zeros((n, d), dtype=np.float32)
            for i, raw in enumerate(df[col].to_numpy()):
                block[i] = get_reagent_feat(raw, cache)
            reagent_blocks.append(block)

        X_full = np.concatenate([sub1, sub2, *reagent_blocks], axis=1)
        mask = sub1_mask & sub2_mask
        return X_full[mask], mask
