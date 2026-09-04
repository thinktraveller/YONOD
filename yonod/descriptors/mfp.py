"""Paper-aligned Morgan count fingerprint descriptor.

``mfp`` is intentionally separate from :mod:`yonod.descriptors.morgan`:
the latter is YONOD's established binary ECFP4 implementation, while this
descriptor emits count vectors using the YieldSmarter paper's radius-3,
1024-bin Morgan generator.  Reaction-level component concatenation and the
"zero block but keep row" rule live in ``feature_builder``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .base import BaseDescriptor


class MFPDescriptor(BaseDescriptor):
    """A single-component Morgan count fingerprint (default: r=3, 1024 bins)."""

    name = "mfp"
    output_dim = 1024

    def __init__(self, radius: int = 3, fp_size: int = 1024) -> None:
        if int(radius) < 0:
            raise ValueError("MFP radius 必须 >= 0")
        if int(fp_size) < 8:
            raise ValueError("MFP fp_size 必须 >= 8")
        self.radius = int(radius)
        self.output_dim = int(fp_size)
        self._generator = GetMorganGenerator(radius=self.radius, fpSize=self.output_dim)

    def featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute count fingerprints without binary coercion.

        Invalid/empty entries are represented by zero blocks and reported as
        invalid in the returned mask.  The paper-specific reaction builder
        deliberately keeps those rows; this lower-level interface follows the
        common descriptor convention so it remains safe to reuse elsewhere.
        """
        features, mask = self._empty_outputs(len(smiles_list), dtype=np.int32)
        for index, smiles in enumerate(smiles_list):
            if smiles is None or not str(smiles).strip():
                continue
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                continue
            features[index] = np.asarray(
                self._generator.GetCountFingerprintAsNumPy(molecule), dtype=np.int32
            )
            mask[index] = True
        return features, mask
