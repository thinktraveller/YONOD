"""ATMOMACCS / MACCS keys descriptor (166-d, RDKit standard keys).

Mirrors what the upstream project's ``generate_MACCS.py`` does:

    fp = GetMACCSKeysFingerprint(mol)        # length 167
    out = [fp[i] for i in range(1, len(fp))] # drop bit 0 -> length 166

Bit 0 of RDKit's MACCS is reserved and always 0, so dropping it is lossless
and standard practice (the ATMOMACCS reference implementation does this too).

We deliberately do NOT import from the external ``化学描述符相关项目/ATMOMACCS/``
package because:
  * its public entrypoint reads SMILES from disk (``data_path/smiles.txt``),
    which doesn't match our in-memory list interface;
  * the underlying call is just RDKit, with zero added value to wrap.

If a future version wants the 287-d ``pyGenATMOMACCS`` variant, swap the
implementation here -- the class name and registry key stay the same.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.DataStructs import ConvertToNumpyArray

from .base import BaseDescriptor


class ATMOMACCSDescriptor(BaseDescriptor):
    """166-bit MACCS keys descriptor (bit 0 dropped, matches upstream convention)."""

    name = "atmomaccs"
    output_dim = 166

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
            fp = GetMACCSKeysFingerprint(mol)  # length 167
            full = np.zeros(167, dtype=np.float32)
            ConvertToNumpyArray(fp, full)
            features[i] = full[1:]  # drop bit 0
            mask[i] = True

        return features, mask
