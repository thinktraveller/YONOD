"""通用特征构建器。

将 LoadedDataset（由 csv_loader 产出）中的 smiles_cols 和 numeric_cols
转换为下游 ML 使用的两块特征矩阵：

  X_smiles  : np.ndarray, shape (n, n_smiles_cols * desc_dim)
               多 SMILES 列的描述符向量按列拼接，顺序与 smiles_cols 一致。
               失败行（RDKit 解析失败）被 mask 过滤掉。
  X_numeric : np.ndarray, shape (n, n_numeric_cols) 或 None（无数值列时）
               原始数值，**不做归一化**；归一化在 CV 循环内每折独立完成。

设计约束（见构建计划书 §14.2、§14.7）：
  - SMILES 描述符原样保留，不做 z-score / min-max。
  - 数值列归一化时机：KFold 循环内，train_idx 切分之后，model.fit 之前。
  - 本函数只负责"原始特征提取"，不触及 StandardScaler。

支持的 desc_name 值：
  "morgan"     → MorganDescriptor（1024 维 ECFP4）
  "maccs"      → ATMOMACCSDescriptor（166 维 MACCS keys）
  "fisd"       → FISDDescriptor（50 维 QM9 GCN）
  "molmetalm"  → MolMetaLMDescriptor（768 维 LLM mean-pool）
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..descriptors.base import BaseDescriptor


# --------------------------------------------------------------------------- #
# 描述符工厂                                                                    #
# --------------------------------------------------------------------------- #

_DESCRIPTOR_REGISTRY: dict[str, type] = {}


_DESCRIPTOR_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "morgan":    ("..descriptors.morgan",    "MorganDescriptor"),
    "maccs":     ("..descriptors.atmomaccs", "ATMOMACCSDescriptor"),
    "fisd":      ("..descriptors.fisd",      "FISDDescriptor"),
    "molmetalm": ("..descriptors.molmetalm", "MolMetaLMDescriptor"),
}


def _get_descriptor(desc_name: str) -> BaseDescriptor:
    """按名称实例化描述符，按需懒加载对应模块（避免 torch_geometric 等重依赖预先导入）。"""
    name_lower = desc_name.lower()
    if name_lower in _DESCRIPTOR_REGISTRY:
        return _DESCRIPTOR_REGISTRY[name_lower]()

    if name_lower not in _DESCRIPTOR_IMPORT_MAP:
        available = list(_DESCRIPTOR_IMPORT_MAP.keys())
        raise ValueError(
            f"未知描述符名称 {desc_name!r}。可用值：{available}"
        )

    module_path, class_name = _DESCRIPTOR_IMPORT_MAP[name_lower]
    import importlib
    mod = importlib.import_module(module_path, package=__package__)
    cls = getattr(mod, class_name)
    _DESCRIPTOR_REGISTRY[name_lower] = cls
    return cls()


# --------------------------------------------------------------------------- #
# 主函数                                                                        #
# --------------------------------------------------------------------------- #

def build_universal_features(
    smiles_cols: List[str],
    numeric_cols: List[str],
    df: pd.DataFrame,
    desc_name: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """计算 SMILES 描述符矩阵与原始数值矩阵。

    Args:
        smiles_cols:  SMILES 列名列表，顺序决定拼接顺序。
        numeric_cols: 数值辅助列名列表（如温度）；空列表则返回 None。
        df:           已通过 csv_loader 加载的 DataFrame（含所有角色列）。
        desc_name:    描述符名称，见模块文档。

    Returns:
        X_smiles  : ndarray shape (n_valid, n_smiles_cols * desc_dim)
        X_numeric : ndarray shape (n_valid, len(numeric_cols)) 或 None
        valid_mask: bool ndarray shape (len(df),)，True 表示该行通过描述符过滤
    """
    if not smiles_cols:
        raise ValueError("smiles_cols 不能为空")

    descriptor = _get_descriptor(desc_name)
    n = len(df)

    # --- 逐列计算 SMILES 描述符 ---
    col_blocks: list[np.ndarray] = []
    row_mask = np.zeros(n, dtype=bool)  # 至少一列有效则行有效（空列用零向量填充）

    for col in smiles_cols:
        if col not in df.columns:
            raise KeyError(f"DataFrame 中未找到 SMILES 列 {col!r}")
        # 规范化为 RDKit 标准点分隔形式（'.'）：
        #   逗号（','）→ 阴阳离子对，如 'CCN=C=NCCCN(C)C,Cl'
        #   星号（'*'）→ 反应步骤分隔符，如反应 SMILES 'A.B*C.D*E'
        #   波浪线（'~'）→ 组分替代表示，如 'CC(=O)O~[Pd]'
        smiles_list = [
            s.replace(",", ".").replace("*", ".").replace("~", ".")
            if isinstance(s, str) else ""
            for s in df[col].fillna("").tolist()
        ]
        feats, mask = descriptor.featurize(smiles_list)
        col_blocks.append(feats)
        row_mask |= mask  # 任一列有效则保留该行；无效列贡献零向量

    # 拼接多列描述符，只保留 valid 行
    X_smiles_full = np.concatenate(col_blocks, axis=1)  # (n, n_cols * d)
    X_smiles = X_smiles_full[row_mask]

    # --- 数值辅助列 ---
    X_numeric: Optional[np.ndarray] = None
    if numeric_cols:
        missing = [c for c in numeric_cols if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame 中未找到数值列：{missing}")
        X_numeric_full = df[numeric_cols].to_numpy(dtype=np.float32)
        X_numeric = X_numeric_full[row_mask]

    n_valid = int(row_mask.sum())
    n_dropped = n - n_valid
    if n_dropped:
        _warn(
            f"[feature_builder] 描述符计算失败，过滤掉 {n_dropped} 行 "
            f"（共 {n} 行，保留 {n_valid} 行）"
        )

    return X_smiles, X_numeric, row_mask


# --------------------------------------------------------------------------- #
# 辅助                                                                          #
# --------------------------------------------------------------------------- #

def _warn(msg: str) -> None:
    """输出警告到 stderr，兼容 Windows GBK 终端。"""
    try:
        print(msg, file=sys.stderr)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), file=sys.stderr)
