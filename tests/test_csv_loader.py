"""验证 csv_loader 对酰胺缩合 CSV 和 ECC CSV 的列探测结果（第1步单元测试）。"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from yonod_yield.universal.csv_loader import load_csv_with_roles

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def check(name, actual, expected):
    ok = set(actual) == set(expected)
    status = PASS if ok else FAIL
    print(f"  {status} {name}")
    print(f"         实际: {sorted(actual)}")
    if not ok:
        print(f"         期望: {sorted(expected)}")
        errors.append(f"{name}: 期望 {sorted(expected)}，实际 {sorted(actual)}")


# ══════════════════════════════════════════
# 测试 1：酰胺缩合数据集（完全自动探测）
# ══════════════════════════════════════════
# 该数据集中 base_id / solvent_id 列实际存储的是 SMILES 字符串，
# activation_id / additive_id 含多片段 SMILES（逗号分隔）和 (无)，
# 有效率低于 0.5 阈值，正确地被跳过。
print("\n" + "=" * 60)
print("测试 1：酰胺缩合数据集 — 完全自动探测")
print("=" * 60)

ds1 = load_csv_with_roles(
    "数据集/酰胺缩合数据集.csv",
    nrows=200,
)

# 期望：3 个 _smiles 列 + base_id + solvent_id（均为真实 SMILES）
EXPECTED_SMILES_AMIDE = ["sub_1_smiles", "sub_2_smiles", "product_smiles", "base_id", "solvent_id"]
check("SMILES 列（自动）", ds1.smiles_cols, EXPECTED_SMILES_AMIDE)
check("数值辅助列", ds1.numeric_cols, [])
check("标签列", [ds1.label_col], ["yield"])
assert len(ds1.df) > 0, "df 不应为空"
print(f"  df 形状: {ds1.df.shape}  （期望列数 = {len(EXPECTED_SMILES_AMIDE) + 1}）")


# ══════════════════════════════════════════
# 测试 2：酰胺缩合数据集（显式指定 3 SMILES + label）
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("测试 2：酰胺缩合数据集 — 显式指定 3 SMILES 列")
print("=" * 60)

ds2 = load_csv_with_roles(
    "数据集/酰胺缩合数据集.csv",
    label_col="yield",
    smiles_cols=["sub_1_smiles", "sub_2_smiles", "product_smiles"],
    nrows=200,
)

check("SMILES 列（显式）", ds2.smiles_cols, ["sub_1_smiles", "sub_2_smiles", "product_smiles"])
check("数值辅助列", ds2.numeric_cols, [])
check("标签列", [ds2.label_col], ["yield"])
assert ds2.df.shape[1] == 4, f"df 应有 4 列（3 SMILES + 1 标签），实际 {ds2.df.shape[1]}"
print(f"  df 形状: {ds2.df.shape}")


# ══════════════════════════════════════════
# 测试 3：ECC 数据集（显式指定 SMILES + 数值辅助列）
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("测试 3：ECC 数据集 — 显式指定含温度辅助列")
print("=" * 60)

ds3 = load_csv_with_roles(
    "数据集/镍催化偶联数据集/Raw_Dataset.csv",
    label_col="∆∆G (Kcal/mol)",
    smiles_cols=["Ligand_SMILES", "Product_SMILES"],
    numeric_cols=["Temp (K)"],
    nrows=200,
)

check("SMILES 列（显式）", ds3.smiles_cols, ["Ligand_SMILES", "Product_SMILES"])
check("数值辅助列（显式）", ds3.numeric_cols, ["Temp (K)"])
check("标签列", [ds3.label_col], ["∆∆G (Kcal/mol)"])
assert ds3.df.shape[1] == 4, f"df 应有 4 列（2 SMILES + 1 温度 + 1 标签），实际 {ds3.df.shape[1]}"
print(f"  df 形状: {ds3.df.shape}")


# ══════════════════════════════════════════
# 测试 4：ECC 数据集（自动探测 SMILES，显式标签）
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("测试 4：ECC 数据集 — 自动探测 SMILES 列")
print("=" * 60)

ds4 = load_csv_with_roles(
    "数据集/镍催化偶联数据集/Raw_Dataset.csv",
    label_col="∆∆G (Kcal/mol)",
    nrows=200,
)

# ECC 中 Ligand_SMILES / Product_SMILES 是唯一的 object 类型 SMILES 列
check("SMILES 列（自动）", ds4.smiles_cols, ["Ligand_SMILES", "Product_SMILES"])
check("标签列", [ds4.label_col], ["∆∆G (Kcal/mol)"])
print(f"  df 形状: {ds4.df.shape}")


# ══════════════════════════════════════════
# 测试 5：错误处理 — 指定不存在的列名
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("测试 5：错误处理 — 指定不存在的列名")
print("=" * 60)

try:
    load_csv_with_roles(
        "数据集/酰胺缩合数据集.csv",
        smiles_cols=["nonexistent_col"],
        nrows=10,
    )
    errors.append("测试5：应抛出 ValueError 但未抛出")
    print(f"  {FAIL} 应抛出 ValueError 但未抛出")
except ValueError as e:
    print(f"  {PASS} 正确抛出 ValueError: {str(e)[:80]}...")


# ══════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════
print("\n" + "=" * 60)
if errors:
    print(f"X {len(errors)} 项测试失败：")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("全部测试通过！csv_loader 第1步验证完成。")
    print()
    print("说明：酰胺缩合数据集中 base_id/solvent_id 存储真实 SMILES，")
    print("auto_detect 正确将其纳入；activation_id/additive_id 含多片段 SMILES")
    print("（逗号分隔）和 (无) 占位符，有效率<50% 被正确排除。")
