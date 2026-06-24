"""BaseDescriptor abstract class for YONOD yield prediction.

Every concrete descriptor (Morgan, ATMOMACCS, FISD, MolMetaLM) inherits from
BaseDescriptor and must:
  * set class attributes ``name`` and ``output_dim``;
  * implement ``featurize(smiles_list) -> (features, mask)``.

The returned ``features`` matrix has shape (n, output_dim); failed SMILES
(un-parseable, network error, etc.) are filled with zero rows and flagged
``False`` in ``mask``.
"""

from __future__ import annotations

import abc
from typing import List, Tuple

import numpy as np


def split_multi_smiles(smiles: str) -> List[str]:
    """Split a SMILES string containing multiple components.

    Supports multiple separators with priority:
    1. Comma (,) - highest priority
    2. Semicolon (;) - second priority
    3. Period (.) - lowest priority (used for salts/complexes in SMILES)

    Args:
        smiles: Input SMILES string that may contain multiple components
            separated by comma, semicolon, or period.

    Returns:
        List of individual SMILES components (stripped of whitespace).
        Returns [smiles] if no separator is found.

    Examples:
        >>> split_multi_smiles("CCO,CC")
        ['CCO', 'CC']
        >>> split_multi_smiles("CCO;CC;C")
        ['CCO', 'CC', 'C']
        >>> split_multi_smiles("CCO.CC")  # period as fallback
        ['CCO', 'CC']
        >>> split_multi_smiles("[Na+].[Cl-]")  # ionic compound
        ['[Na+]', '[Cl-]']
    """
    if not smiles:
        return []

    # Priority 1: Comma
    if ',' in smiles:
        return [s.strip() for s in smiles.split(',')]

    # Priority 2: Semicolon
    if ';' in smiles:
        return [s.strip() for s in smiles.split(';')]

    # Priority 3: Period (fallback, used in SMILES for salts)
    if '.' in smiles:
        return [s.strip() for s in smiles.split('.')]

    # No separator found
    return [smiles.strip()]


class BaseDescriptor(abc.ABC):
    """Abstract base for molecular descriptors used in YONOD."""

    name: str = "base"
    output_dim: int = 0

    @abc.abstractmethod
    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute descriptor vectors for a batch of SMILES.

        Args:
            smiles_list: list of SMILES strings, length n. None / empty values
                should be handled by the caller before reaching here (reagent
                ``(无)`` is mapped to a zero vector in ``features.reagent_cache``).

        Returns:
            features: ndarray of shape ``(n, output_dim)``. Failed rows are 0.
            mask: bool ndarray of shape ``(n,)``; True where featurization
                succeeded.
        """
        raise NotImplementedError

    def _empty_outputs(
        self, n: int, dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-allocate a zero feature matrix and an all-False mask.

        Subclasses populate the rows that succeed and flip mask entries to True.
        """
        features = np.zeros((n, self.output_dim), dtype=dtype)
        mask = np.zeros(n, dtype=bool)
        return features, mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, output_dim={self.output_dim})"
        )
