"""通用 CSV 数据集加载器。

列角色声明优先级：
  1. 用户通过 load_csv_with_roles() 参数显式指定
  2. 自动探测：SMILES 列用 RDKit 解析率 > threshold 判定；
              标签列取最后一列浮点型（显式打印警告）

返回值契约（LoadedDataset）：
  smiles_cols   : SMILES 列名列表（按原始列顺序排列）
  numeric_cols  : 数值辅助列名列表（温度、压力等；可为空列表）
  label_col     : 标签列名
  smiles_roles  : 三分类角色映射 {'reactant': [...], 'product': [...], 'other': [...]}
  df            : 仅保留 smiles_cols + numeric_cols + label_col 的 DataFrame
                  已过滤掉标签列中有 NaN 的行
"""

from __future__ import annotations

import random
import sys
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

# Windows GBK 终端无法编码部分 Unicode 字符（如 ∆∆G），强制使用 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    smiles_cols: List[str]
    numeric_cols: List[str]
    label_col: str
    smiles_roles: Dict[str, List[str]] = field(default_factory=lambda: {
        'reactant': [],
        'product': [],
        'other': []
    })


# ──────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────

def _infer_label_col(df: pd.DataFrame) -> str:
    """取 DataFrame 最后一列浮点型列作为标签列。"""
    float_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32")]
    if not float_cols:
        raise ValueError(
            "CSV 中没有找到浮点型列，无法自动推断标签列。"
            "请用 --label-col 手动指定。"
        )
    last_float = float_cols[-1]
    print(f"[警告] 自动推断标签列为 '{last_float}'，如有误请用 --label-col 显式指定。")
    return last_float


def auto_detect_smiles_cols(
    df: pd.DataFrame,
    label_col: str,
    threshold: float = 0.5,
    sample_size: int = 50,
    seed: int = 42,
) -> List[str]:
    """对 df 中除 label_col 之外每列随机抽样，用 RDKit 判断是否为 SMILES 列。

    Args:
        df: 输入 DataFrame（已加载 CSV）
        label_col: 标签列名，探测时跳过
        threshold: SMILES 有效率阈值，超过此值则认定为 SMILES 列
        sample_size: 每列随机抽样行数
        seed: 随机种子（保证可复现）

    Returns:
        检测到的 SMILES 列名列表（按 df 列顺序）

    Raises:
        ValueError: 未找到任何 SMILES 列时
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # 静默 RDKit 的 SMILES 解析警告
    except ImportError as exc:
        raise ImportError("auto_detect_smiles_cols 依赖 RDKit，请先 conda install -c conda-forge rdkit") from exc

    rng = random.Random(seed)
    smiles_cols: List[str] = []

    for col in df.columns:
        if col == label_col:
            continue
        # 只对 object 类型列尝试（int/float 列必然不是 SMILES）
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna().astype(str).tolist()
        if not non_null:
            continue
        sample = rng.sample(non_null, min(sample_size, len(non_null)))
        valid_count = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None)
        valid_rate = valid_count / len(sample)
        if valid_rate > threshold:
            smiles_cols.append(col)
            print(f"[自动探测] 列 '{col}' SMILES 有效率 {valid_rate:.1%} > {threshold:.0%}，纳入 SMILES 列")
        else:
            print(f"[自动探测] 列 '{col}' SMILES 有效率 {valid_rate:.1%} ≤ {threshold:.0%}，跳过")

    if not smiles_cols:
        raise ValueError(
            "自动探测未找到任何 SMILES 列（所有 object 列的有效率均低于阈值）。\n"
            f"请用 --smiles-cols 手动指定，或用 --smiles-threshold 调低阈值（当前 {threshold}）。"
        )
    return smiles_cols


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def load_csv_with_roles(
    csv_path: str | Path,
    label_col: Optional[str] = None,
    smiles_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    reactant_cols: Optional[List[str]] = None,
    product_cols: Optional[List[str]] = None,
    other_cols: Optional[List[str]] = None,
    smiles_threshold: float = 0.5,
    encoding: str = "utf-8-sig",
    nrows: Optional[int] = None,
) -> LoadedDataset:
    """加载 CSV 并确定各列角色（支持三分类模式）。

    Args:
        csv_path: CSV 文件路径
        label_col: 标签列名；None → 自动推断最后一列浮点型
        smiles_cols: SMILES 列名列表（传统模式）；None → 自动探测
        numeric_cols: 数值辅助列名列表（温度等）；None → 空列表
        reactant_cols: 反应物列名列表（三分类模式）
        product_cols: 产物列名列表（三分类模式）
        other_cols: 其他参与者列名列表（三分类模式，可选）
        smiles_threshold: 自动探测 SMILES 列的有效率阈值
        encoding: CSV 编码（默认 utf-8-sig，兼容 BOM）
        nrows: 仅读取前 N 行（调试用）

    Returns:
        LoadedDataset（df 仅含有效列，已过滤标签列 NaN 行）
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

    df_raw = pd.read_csv(csv_path, encoding=encoding, nrows=nrows)
    print(f"[加载] {csv_path.name}  {len(df_raw)} 行 × {len(df_raw.columns)} 列")

    # ── 确定标签列 ──
    if label_col is None:
        label_col = _infer_label_col(df_raw)
    elif label_col not in df_raw.columns:
        raise ValueError(
            f"指定的标签列 '{label_col}' 不存在于 CSV 中。\n"
            f"可用列：{list(df_raw.columns)}"
        )
    if df_raw[label_col].dtype not in ("float64", "float32", "int64", "int32"):
        # 尝试强制转换
        try:
            df_raw[label_col] = pd.to_numeric(df_raw[label_col], errors="raise")
            print(f"[标签列] '{label_col}' 强制转换为数值成功")
        except Exception:
            raise ValueError(
                f"标签列 '{label_col}' 包含非数值数据，无法用于回归预测。"
            )

    # ── 确定 SMILES 列及其角色分类 ──
    # 检测是否启用三分类模式
    use_role_classification = (reactant_cols is not None or product_cols is not None)

    if use_role_classification:
        # 三分类模式
        reactant_cols = reactant_cols or []
        product_cols = product_cols or []

        # 验证列名有效性
        all_specified = reactant_cols + product_cols + (other_cols or [])
        missing = [c for c in all_specified if c not in df_raw.columns]
        if missing:
            raise ValueError(f"指定的 SMILES 列不存在：{missing}\n可用列：{list(df_raw.columns)}")

        # 验证反应物和产物列不能重叠
        overlap = set(reactant_cols) & set(product_cols)
        if overlap:
            raise ValueError(f"反应物列和产物列不能重叠：{overlap}")

        # 验证至少需要反应物或产物
        if not reactant_cols and not product_cols:
            raise ValueError("三分类模式至少需要指定 --reactant-cols 或 --product-cols")

        # 自动推断 other_cols（如果未指定）
        if other_cols is None:
            # 自动探测所有 SMILES 列
            detected_smiles = auto_detect_smiles_cols(df_raw, label_col, smiles_threshold)
            specified = set(reactant_cols + product_cols)
            other_cols = [c for c in detected_smiles if c not in specified]
            if other_cols:
                print(f"[自动推断] 其他参与者列：{other_cols}")

        # 构建角色映射
        smiles_roles = {
            'reactant': reactant_cols,
            'product': product_cols,
            'other': other_cols or [],
        }

        # 所有 SMILES 列（按角色顺序拼接）
        smiles_cols = reactant_cols + product_cols + (other_cols or [])

        print(f"[三分类模式] 反应物列：{reactant_cols}")
        print(f"[三分类模式] 产物列：{product_cols}")
        if other_cols:
            print(f"[三分类模式] 其他参与者列：{other_cols}")

    else:
        # 传统模式（不区分角色）
        if smiles_cols is None:
            print("[自动探测] 未指定 --smiles-cols，开始自动探测 SMILES 列 ...")
            smiles_cols = auto_detect_smiles_cols(df_raw, label_col, smiles_threshold)
        else:
            missing = [c for c in smiles_cols if c not in df_raw.columns]
            if missing:
                raise ValueError(f"指定的 SMILES 列不存在：{missing}\n可用列：{list(df_raw.columns)}")
            print(f"[SMILES 列] 使用显式指定：{smiles_cols}")

        # 传统模式：所有列归入 'other'
        smiles_roles = {
            'reactant': [],
            'product': [],
            'other': smiles_cols,
        }
        print("[传统模式] 所有 SMILES 列等价处理")

    # ── 确定数值辅助列 ──
    if numeric_cols is None:
        numeric_cols = []
    else:
        missing = [c for c in numeric_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(f"指定的数值辅助列不存在：{missing}\n可用列：{list(df_raw.columns)}")
        print(f"[数值辅助列] {numeric_cols}")

    # ── 列重叠检查 ──
    all_role_cols = set(smiles_cols) | set(numeric_cols) | {label_col}
    if len(all_role_cols) != len(smiles_cols) + len(numeric_cols) + 1:
        raise ValueError(
            "smiles_cols / numeric_cols / label_col 之间存在重叠列，请检查列名是否有误。"
        )

    # ── 构建精简 DataFrame ──
    keep_cols = smiles_cols + numeric_cols + [label_col]
    df = df_raw[keep_cols].copy()

    # 过滤标签 NaN 行
    before = len(df)
    df = df.dropna(subset=[label_col])
    dropped = before - len(df)
    if dropped:
        print(f"[过滤] 丢弃标签列含 NaN 的 {dropped} 行，剩余 {len(df)} 行")

    print(
        f"[完成] 角色确认：SMILES={smiles_cols}，"
        f"数值辅助={numeric_cols if numeric_cols else '（无）'}，"
        f"标签='{label_col}'，有效行数={len(df)}"
    )
    return LoadedDataset(
        df=df,
        smiles_cols=smiles_cols,
        numeric_cols=numeric_cols,
        label_col=label_col,
        smiles_roles=smiles_roles,
    )
