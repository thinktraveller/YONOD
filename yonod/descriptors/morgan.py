"""Morgan ECFP4 fingerprint descriptor (1024-bit, radius=2)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import BaseDescriptor


class MorganDescriptor(BaseDescriptor):
    """Morgan / ECFP4 binary fingerprint.

    radius=2, nBits=1024, returned as float32 ndarray for downstream ML
    compatibility (XGBoost / sklearn handle the 0/1 floats identically to
    uint8 bits).
    """

    name = "morgan"
    output_dim = 1024

    def __init__(self, radius: int = 2, n_bits: int = 1024) -> None:
        self.radius = radius
        self.output_dim = n_bits

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.output_dim
            )
            arr = np.zeros(self.output_dim, dtype=np.float32)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            features[i] = arr
            mask[i] = True

        return features, mask
