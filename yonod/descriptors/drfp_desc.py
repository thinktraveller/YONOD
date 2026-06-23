"""DRFP (Differential Reaction Fingerprint) descriptor.

DRFP captures the structural changes during a chemical reaction by
computing the symmetric difference of reactant and product fingerprints.

Reference:
    Probst, D., Schwaller, P., & Reymond, J. L. (2022).
    Reaction classification and yield prediction using the differential
    reaction fingerprint DRFP. Digital Discovery, 1(2), 91-97.

Input format:
    Reaction SMARTS: "reactant1.reactant2>>product"
    Example: "CCO.CC(=O)O>>CCOC(C)=O"
"""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np

from .base import BaseDescriptor

# 延迟导入 drfp（可能未安装）
_DRFP_AVAILABLE = False
_DrfpEncoder = None


def _ensure_drfp():
    """Lazily import drfp to avoid import errors if not installed."""
    global _DRFP_AVAILABLE, _DrfpEncoder
    if _DrfpEncoder is not None:
        return True
    try:
        from drfp import DrfpEncoder
        _DrfpEncoder = DrfpEncoder
        _DRFP_AVAILABLE = True
        return True
    except ImportError:
        return False


class DRFPDescriptor(BaseDescriptor):
    """Differential Reaction Fingerprint (DRFP) descriptor.

    Computes reaction fingerprints that capture structural changes
    between reactants and products.

    Input format:
        Each SMILES in smiles_list should be a reaction SMARTS:
        "reactant1.reactant2>>product"

    Attributes:
        name: "drfp"
        output_dim: 2048 (default) or custom n_bits
    """

    name = "drfp"
    output_dim = 2048

    def __init__(self, n_bits: int = 2048) -> None:
        """Initialize DRFP encoder.

        Args:
            n_bits: Fingerprint length (default 2048, must be 2048 for drfp 0.3.x).
        """
        if n_bits != 2048:
            raise ValueError("drfp 0.3.x 仅支持 2048 维指纹")
        self.output_dim = n_bits
        self._encoder = None

    def _get_encoder(self):
        """Get or create the DRFP encoder."""
        if self._encoder is not None:
            return self._encoder

        if not _ensure_drfp():
            raise ImportError(
                "drfp 库未安装。请运行: pip install drfp -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

        # drfp 0.3.x 使用无参数初始化
        self._encoder = _DrfpEncoder()
        return self._encoder

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute DRFP for a batch of reaction SMARTS.

        Args:
            smiles_list: List of reaction SMARTS strings.
                Format: "reactant1.reactant2>>product"

        Returns:
            features: ndarray of shape (n, output_dim), dtype=float32.
            mask: bool array of shape (n,), True where encoding succeeded.
        """
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)

        encoder = self._get_encoder()

        for i, rxn_smarts in enumerate(smiles_list):
            if not rxn_smarts or not rxn_smarts.strip():
                continue

            # 验证反应 SMARTS 格式
            if ">>" not in rxn_smarts:
                continue

            try:
                # DRFP encoder.encode() 返回 list of numpy arrays
                fp_list = encoder.encode([rxn_smarts])

                if fp_list is not None and len(fp_list) > 0:
                    # 取第一个元素（对应输入的第一个反应）
                    arr = np.array(fp_list[0], dtype=np.float32)

                    # 确保维度匹配
                    if len(arr) == self.output_dim:
                        features[i] = arr
                        mask[i] = True

            except Exception:
                continue

        return features, mask


def build_reaction_smarts_from_df(
    df: pd.DataFrame,
    reactant_cols: List[str],
    product_cols: List[str],
) -> pd.Series:
    """从 DataFrame 构建反应 SMARTS 列。

    Args:
        df: 数据集 DataFrame
        reactant_cols: 反应物列名列表
        product_cols: 产物列名列表

    Returns:
        Series of reaction SMARTS strings (format: "R1.R2>>P1.P2")

    Example:
        reactant_cols = ['sub_1_smiles', 'sub_2_smiles']
        product_cols = ['product_smiles']
        → "CCO.CC(=O)O>>CCOC(C)=O"
    """
    def _build_smarts(row):
        # 提取反应物
        reactants = []
        for col in reactant_cols:
            smi = str(row[col]).strip()
            if smi and smi != "(无)" and smi.lower() != "nan":
                reactants.append(smi)

        # 提取产物
        products = []
        for col in product_cols:
            smi = str(row[col]).strip()
            if smi and smi != "(无)" and smi.lower() != "nan":
                products.append(smi)

        # 构建反应 SMARTS
        if not reactants or not products:
            return ""

        reactant_part = ".".join(reactants)
        product_part = ".".join(products)
        return f"{reactant_part}>>{product_part}"

    return df.apply(_build_smarts, axis=1)


def build_reaction_smarts(
    sub1_smiles: str,
    sub2_smiles: str,
    product_smiles: str,
) -> str:
    """Construct reaction SMARTS from individual SMILES.

    Args:
        sub1_smiles: First substrate SMILES.
        sub2_smiles: Second substrate SMILES.
        product_smiles: Product SMILES.

    Returns:
        Reaction SMARTS in format "sub1.sub2>>product".
        Returns empty string if any input is invalid.
    """
    # 清理输入
    sub1 = (sub1_smiles or "").strip()
    sub2 = (sub2_smiles or "").strip()
    prod = (product_smiles or "").strip()

    # 检查有效性
    if not sub1 or not prod:
        return ""

    # 构造反应 SMARTS
    if sub2 and sub2 != "(无)":
        return f"{sub1}.{sub2}>>{prod}"
    else:
        return f"{sub1}>>{prod}"
