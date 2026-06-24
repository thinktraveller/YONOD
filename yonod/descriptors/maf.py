"""MAF (Molecular Additive Fingerprint) descriptor.

MAF is designed for multi-component reactions (e.g., amide coupling with
6 molecules: 2 substrates + 4 reagents). It computes ECFP for each component
and performs element-wise addition to produce a single fingerprint vector
that encodes "how many components contain each substructure".

Input format:
    Each SMILES in `smiles_list` should be a dot-separated multi-molecule
    string, e.g., "CCO.CC(=O)O.c1ccccc1". Components marked as "(无)"
    contribute zero vectors (automatically handled by RDKit parsing failure).

Output:
    128-dimensional integer vector, where each element is in [0, n_components].
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from .base import BaseDescriptor, split_multi_smiles


class MAFDescriptor(BaseDescriptor):
    """Molecular Additive Fingerprint (MAF) for multi-component reactions.

    Generates ECFP (radius=2, 128 bits) for each component, then sums them
    element-wise to capture collective substructure information.
    """

    name = "maf"
    output_dim = 128  # 用户需求指定的维度

    def __init__(self, radius: int = 2, n_bits: int = 128) -> None:
        """
        Args:
            radius: ECFP radius (default 2, same as Morgan ECFP4).
            n_bits: Fingerprint length (default 128 per user spec).
        """
        self.radius = radius
        self.output_dim = n_bits

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MAF for a batch of multi-molecule SMILES strings.

        Args:
            smiles_list: Each entry is a dot-separated SMILES string
                representing a complete reaction (e.g., "mol1.mol2.mol3").
                Empty strings or unparsable entries are marked as failed.

        Returns:
            features: ndarray shape (n, output_dim), dtype=int32.
                Each row is the element-wise sum of ECFP vectors across
                all valid components in that reaction.
            mask: bool array shape (n,), True where at least one component
                was successfully parsed.
        """
        n = len(smiles_list)
        # 使用 int32 存储加和后的整数 (最多 6 个分子, 每位最大值 6, 无溢出风险)
        features = np.zeros((n, self.output_dim), dtype=np.int32)
        mask = np.zeros(n, dtype=bool)

        for i, multi_smi in enumerate(smiles_list):
            if not multi_smi or multi_smi.strip() == "":
                continue

            # 分割多分子 SMILES (支持逗号、分号、点号分隔符)
            component_smiles = split_multi_smiles(multi_smi)

            # 累加各组分的 ECFP
            accumulated_fp = np.zeros(self.output_dim, dtype=np.int32)
            valid_count = 0

            for comp_smi in component_smiles:
                comp_smi = comp_smi.strip()
                if not comp_smi or comp_smi == "(无)":
                    # 空组分或显式标记为"无"的试剂 → 跳过
                    continue

                mol = Chem.MolFromSmiles(comp_smi)
                if mol is None:
                    # RDKit 解析失败 → 跳过该组分 (但不标记整个反应失败)
                    continue

                # 生成 ECFP (二进制指纹)
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.radius, nBits=self.output_dim
                )

                # 转换为 NumPy 数组 (float32 → 0.0/1.0)
                arr = np.zeros(self.output_dim, dtype=np.float32)
                ConvertToNumpyArray(fp, arr)

                # 累加到整数数组 (0/1 → int32)
                accumulated_fp += arr.astype(np.int32)
                valid_count += 1

            # 只要有至少一个组分成功解析, 就认为该反应有效
            if valid_count > 0:
                features[i] = accumulated_fp
                mask[i] = True

        return features, mask
