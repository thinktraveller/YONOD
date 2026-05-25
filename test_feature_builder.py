"""验证 feature_builder.build_universal_features()。

用例：
  1. 单 SMILES 列 + 无数值列 → X_smiles.shape[1] == desc_dim
  2. 多 SMILES 列（2 列）+ 无数值列 → X_smiles.shape[1] == 2 * desc_dim
  3. 单 SMILES 列 + 1 个数值列 → X_numeric.shape == (n, 1)
  4. 含无效 SMILES 行时 valid_mask 正确过滤
  5. 未知 desc_name 抛出 ValueError
"""

import numpy as np
import pandas as pd
import pytest

from yonod_yield.universal.feature_builder import build_universal_features

# 已知有效 SMILES（乙醇 / 苯 / 丙酮）
SMILES_A = ["CCO", "c1ccccc1"]
SMILES_B = ["CC(=O)C", "CC(=O)C"]
INVALID = "NOT_A_SMILES"

DESC = "morgan"  # 最轻量，1024 维，无需 GPU


def _make_df(smi_a, smi_b=None, temp=None, label=None):
    data = {"smi_a": smi_a}
    if smi_b is not None:
        data["smi_b"] = smi_b
    if temp is not None:
        data["Temp"] = temp
    if label is not None:
        data["label"] = label
    return pd.DataFrame(data)


# ------------------------------------------------------------------ #
# 用例 1：单 SMILES 列，无数值列                                        #
# ------------------------------------------------------------------ #
def test_single_smiles_col_no_numeric():
    df = _make_df(SMILES_A)
    X_smi, X_num, mask = build_universal_features(["smi_a"], [], df, DESC)
    assert X_smi.shape == (2, 1024), f"期望 (2, 1024)，实际 {X_smi.shape}"
    assert X_num is None
    assert mask.sum() == 2
    print(f"[PASS] 用例1  X_smiles.shape={X_smi.shape}")


# ------------------------------------------------------------------ #
# 用例 2：双 SMILES 列，维度 = 2 × desc_dim                            #
# ------------------------------------------------------------------ #
def test_double_smiles_cols():
    df = _make_df(SMILES_A, smi_b=SMILES_B)
    X_smi, X_num, mask = build_universal_features(["smi_a", "smi_b"], [], df, DESC)
    assert X_smi.shape == (2, 2 * 1024), f"期望 (2, 2048)，实际 {X_smi.shape}"
    assert X_num is None
    print(f"[PASS] 用例2  X_smiles.shape={X_smi.shape}")


# ------------------------------------------------------------------ #
# 用例 3：单 SMILES 列 + 1 个数值列                                     #
# ------------------------------------------------------------------ #
def test_with_numeric_col():
    df = _make_df(SMILES_A, temp=[298.0, 310.0])
    X_smi, X_num, mask = build_universal_features(["smi_a"], ["Temp"], df, DESC)
    assert X_smi.shape == (2, 1024)
    assert X_num is not None and X_num.shape == (2, 1)
    assert np.allclose(X_num[:, 0], [298.0, 310.0])
    print(f"[PASS] 用例3  X_numeric.shape={X_num.shape}, 数值={X_num[:,0].tolist()}")


# ------------------------------------------------------------------ #
# 用例 4：含无效 SMILES，valid_mask 正确过滤                            #
# ------------------------------------------------------------------ #
def test_invalid_smiles_mask():
    df = _make_df(["CCO", INVALID, "c1ccccc1"])
    X_smi, X_num, mask = build_universal_features(["smi_a"], [], df, DESC)
    assert mask.sum() == 2, f"期望保留 2 行，实际 {mask.sum()}"
    assert X_smi.shape[0] == 2
    print(f"[PASS] 用例4  mask={mask.tolist()}, 保留行={mask.sum()}")


# ------------------------------------------------------------------ #
# 用例 5：未知 desc_name 抛出 ValueError                               #
# ------------------------------------------------------------------ #
def test_unknown_desc_raises():
    df = _make_df(SMILES_A)
    with pytest.raises(ValueError, match="未知描述符名称"):
        build_universal_features(["smi_a"], [], df, "nonexistent_desc")
    print("[PASS] 用例5  未知描述符名称正确抛出 ValueError")


if __name__ == "__main__":
    test_single_smiles_col_no_numeric()
    test_double_smiles_cols()
    test_with_numeric_col()
    test_invalid_smiles_mask()
    test_unknown_desc_raises()
    print("\n[OK] 全部 5 个用例通过")
