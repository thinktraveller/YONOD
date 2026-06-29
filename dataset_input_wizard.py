"""
YONOD 数据集输入向导 (Dataset Input Wizard)

完全重构的数据集输入流程,采用向导式、逐列声明的交互方式。

核心功能:
1. 逐列声明列角色(标签、反应物SMILES、产物SMILES、其他组分SMILES、条件数值)
2. 合法性检验(SMILES合法性、数值合法性)
3. 生成规范数据集(固定列顺序)
4. 生成列映射文件(记录原始列名→角色→新列名的对应关系)
5. 生成非法输入报告(Markdown格式)

作者: YONOD构建专家
版本: v1.0
日期: 2026-06-29
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from rdkit import Chem
from rdkit import RDLogger

# 静默RDKit警告信息
RDLogger.DisableLog('rdApp.*')


def step1_collect_basic_info() -> Dict:
    """
    步骤1: 收集基本信息

    收集四项基本信息:
    1. 初始数据集路径(必选)
    2. 列映射文件路径(可选,如提供则跳过步骤2)
    3. 项目名称(必选,用于生成输出文件前缀)
    4. 项目文件夹位置(必选,所有输出文件保存位置)

    Returns:
        dict: {
            'dataset_path': str,
            'mapping_path': str | None,
            'project_name': str,
            'project_folder': str
        }
    """
    print("=" * 60)
    print("步骤1: 指定初始数据集、列映射文件、项目名称和项目文件夹")
    print("=" * 60)

    # 1. 输入初始数据集路径
    while True:
        dataset_path = input("\n请输入初始数据集路径(CSV格式): ").strip()
        if os.path.exists(dataset_path) and dataset_path.endswith('.csv'):
            print(f"✓ 数据集文件存在: {dataset_path}")
            break
        else:
            print("✗ 文件不存在或不是CSV格式,请重新输入")

    # 2. 可选: 输入列映射文件路径
    mapping_path = input("\n请输入列映射文件路径(可选,直接回车跳过): ").strip()
    if mapping_path:
        if os.path.exists(mapping_path) and mapping_path.endswith('.csv'):
            print(f"✓ 列映射文件存在: {mapping_path}")
            print("  将跳过步骤2,直接使用此映射进行合法性检验")
        else:
            print("✗ 文件不存在或不是CSV格式,将忽略此输入,进入正常流程")
            mapping_path = None
    else:
        mapping_path = None

    # 3. 输入项目名称
    while True:
        project_name = input("\n请输入项目名称(仅英文字母、数字、下划线): ").strip()
        if re.match(r'^[a-zA-Z0-9_]+$', project_name):
            print(f"✓ 项目名称: {project_name}")
            break
        else:
            print("✗ 项目名称只能包含英文字母、数字、下划线")

    # 4. 输入项目文件夹位置
    while True:
        project_folder = input("\n请输入项目文件夹位置(将在此创建输出文件): ").strip()
        if os.path.isdir(project_folder) or not os.path.exists(project_folder):
            os.makedirs(project_folder, exist_ok=True)
            print(f"✓ 项目文件夹: {project_folder}")
            break
        else:
            print("✗ 路径无效或不是文件夹")

    return {
        'dataset_path': dataset_path,
        'mapping_path': mapping_path,
        'project_name': project_name,
        'project_folder': project_folder
    }


def load_and_preview_dataset(dataset_path: str) -> pd.DataFrame:
    """
    加载数据集并展示基本信息

    Args:
        dataset_path: 数据集文件路径

    Returns:
        pd.DataFrame: 原始数据集

    Raises:
        UnicodeDecodeError: 如果CSV文件编码不是UTF-8
        pd.errors.ParserError: 如果CSV格式错误
    """
    try:
        # 首先尝试UTF-8编码
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8失败,尝试GBK编码(中文Windows常见编码)
        print("⚠ UTF-8编码读取失败,尝试使用GBK编码...")
        try:
            df = pd.read_csv(dataset_path, encoding='gbk')
            print("✓ 使用GBK编码成功读取")
        except UnicodeDecodeError:
            # 如果GBK也失败,尝试latin1(几乎不会失败)
            print("⚠ GBK编码读取失败,尝试使用latin1编码...")
            df = pd.read_csv(dataset_path, encoding='latin1')
            print("✓ 使用latin1编码成功读取")

    print(f"\n数据集基本信息:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")

    # 检查是否有重复列名
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"\n⚠ 警告: 数据集存在重复列名,Pandas已自动添加后缀: {duplicate_cols}")

    print(f"\n列序号和列名称:")
    for idx, col in enumerate(df.columns):
        print(f"  [{idx}] {col}")

    return df


# ==================== 步骤2: 逐列声明列角色和名称 ====================

def step2_select_columns(df: pd.DataFrame) -> List[int]:
    """
    步骤2.1: 从原始数据集中选择需要的列

    Args:
        df: 原始数据集DataFrame

    Returns:
        list: 选中的列索引列表
    """
    print("\n" + "=" * 60)
    print("步骤2.1: 选择需要的列")
    print("=" * 60)
    print("当前所有列:")
    for idx, col in enumerate(df.columns):
        print(f"  [{idx}] {col}")

    print("\n请输入需要的列序号,用逗号分隔(例如: 0,1,3,5)")
    print("或输入'all'选择全部列")

    while True:
        user_input = input("列序号: ").strip()

        if user_input.lower() == 'all':
            selected_indices = list(range(len(df.columns)))
            break

        try:
            selected_indices = [int(x.strip()) for x in user_input.split(',')]
            # 验证索引有效性
            if all(0 <= idx < len(df.columns) for idx in selected_indices):
                break
            else:
                print("[X] 存在无效的列序号,请重新输入")
        except ValueError:
            print("[X] 输入格式错误,请使用逗号分隔的数字")

    selected_columns = [df.columns[idx] for idx in selected_indices]
    print(f"\n[OK] 已选择 {len(selected_columns)} 列:")
    for idx, col in zip(selected_indices, selected_columns):
        print(f"  [{idx}] {col}")

    return selected_indices


def validate_numeric_column(df: pd.DataFrame, col_idx: int) -> Set[int]:
    """
    验证数值列,返回非法行索引集合

    非法情况:
    1. 非数值值
    2. 空值(NaN)

    Returns:
        set: 非法行索引集合
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value):
            invalid_rows.add(idx)
            continue

        # 检查是否为数值
        try:
            float(value)
        except (ValueError, TypeError):
            invalid_rows.add(idx)

    return invalid_rows


def validate_numeric_column_allow_empty(df: pd.DataFrame, col_idx: int) -> Set[int]:
    """
    验证数值列(允许空值),返回非法行索引集合

    非法情况: 非数值值(但空值允许)
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        if pd.isna(value):
            continue  # 空值允许

        try:
            float(value)
        except (ValueError, TypeError):
            invalid_rows.add(idx)

    return invalid_rows


def validate_smiles_column(df: pd.DataFrame, col_idx: int, allow_empty: bool = False) -> Set[int]:
    """
    验证SMILES列,返回非法行索引集合

    Args:
        df: 数据集
        col_idx: 列索引
        allow_empty: 是否允许空值(others列允许,reactant/product列不允许)

    非法情况:
    1. 非法SMILES(RDKit无法解析)
    2. 空值(如果allow_empty=False)

    Returns:
        set: 非法行索引集合
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value) or str(value).strip() == '':
            if not allow_empty:
                invalid_rows.add(idx)
            continue

        # 将值转为字符串
        smiles_str = str(value).strip()

        # 尝试解析SMILES(可能包含多个分子,用.或空格或;或,分隔)
        # 分隔后逐个验证
        separators = ['.', ' ', ';', ',']
        molecules = [smiles_str]  # 默认当作单个分子

        for sep in separators:
            if sep in smiles_str:
                molecules = [s.strip() for s in smiles_str.split(sep) if s.strip()]
                break

        # 验证每个分子
        for mol_smiles in molecules:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                invalid_rows.add(idx)
                break

    return invalid_rows


def validate_product_column(df: pd.DataFrame, col_idx: int) -> Set[int]:
    """
    验证产物列,返回非法行索引集合

    产物列特殊要求:
    1. 不能为空
    2. 必须是合法SMILES
    3. 不能含有分隔符(.;, 空格)

    注: 此处采用严格定义,可在文档中说明宽松定义的可能性
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value) or str(value).strip() == '':
            invalid_rows.add(idx)
            continue

        smiles_str = str(value).strip()

        # 检查是否含有分隔符
        if any(sep in smiles_str for sep in ['.', ';', ',', ' ']):
            invalid_rows.add(idx)
            continue

        # 验证SMILES
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            invalid_rows.add(idx)

    return invalid_rows


def step2_2_declare_label_column(df: pd.DataFrame, available_columns: List[int], used_names: Set[str]) -> Dict:
    """
    步骤2.2: 声明标签列(label)

    Args:
        df: 原始数据集
        available_columns: 可用列索引列表
        used_names: 已使用的列名称集合

    Returns:
        dict: {'origin_idx': int, 'origin_name': str, 'role': 'label', 'name': str, 'invalid_rows': set}
    """
    print("\n" + "=" * 60)
    print("步骤2.2: 声明标签列(label)")
    print("=" * 60)
    print("可选列:")
    for idx in available_columns:
        print(f"  [{idx}] {df.columns[idx]}")

    # 选择列
    while True:
        try:
            col_idx = int(input("请输入标签列序号(只能选择1列): ").strip())
            if col_idx in available_columns:
                break
            else:
                print("[X] 该列不在可选列表中")
        except ValueError:
            print("[X] 请输入有效的数字")

    origin_name = df.columns[col_idx]

    # 输入列名称
    default_name = 'yield'
    print(f"\n请为此列指定名称(默认: {default_name})")
    print("列名称只能包含英文字母、数字、下划线、连字符")

    while True:
        name = input(f"列名称[{default_name}]: ").strip() or default_name

        # 验证名称格式
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            print("[X] 列名称只能包含英文、数字、下划线、连字符")
            continue

        # 验证名称唯一性
        if name in used_names:
            print(f"[X] 列名称'{name}'已被使用,请使用其他名称")
            continue

        break

    # 合法性检验: 排除非数值和空值
    invalid_rows = validate_numeric_column(df, col_idx)

    print(f"\n[OK] 标签列配置完成:")
    print(f"  原始列名: {origin_name}")
    print(f"  新列名: {name}")
    print(f"  非法行数: {len(invalid_rows)}")

    used_names.add(name)
    available_columns.remove(col_idx)

    return {
        'origin_idx': col_idx,
        'origin_name': origin_name,
        'role': 'label',
        'name': name,
        'invalid_rows': invalid_rows
    }


def step2_3_declare_reactant_columns(df: pd.DataFrame, available_columns: List[int], used_names: Set[str]) -> List[Dict]:
    """
    步骤2.3: 声明反应物SMILES列(可多列)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='reactant', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.3: 声明反应物SMILES列")
    print("=" * 60)

    reactant_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入反应物列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            # 验证有效性
            if not all(idx in available_columns for idx in col_indices):
                print("[X] 存在无效的列序号")
                continue

            # 为每列声明名称
            for col_idx in col_indices:
                origin_name = df.columns[col_idx]
                default_name = 'reactant'

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称(默认: {default_name})")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("[X] 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"[X] 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非法SMILES和空值
                invalid_rows = validate_smiles_column(df, col_idx, allow_empty=False)

                print(f"[OK] 非法行数: {len(invalid_rows)}")

                reactant_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'reactant',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 共声明 {len(reactant_columns)} 个反应物列")
    return reactant_columns


def step2_4_declare_product_column(df: pd.DataFrame, available_columns: List[int], used_names: Set[str]) -> Dict:
    """
    步骤2.4: 声明产物SMILES列(必须且只能1列)

    关键要求:
    - 每个单元格只能包含一个独立的SMILES(不含分隔符)

    Returns:
        dict: origin_idx, origin_name, role='product', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.4: 声明产物SMILES列")
    print("=" * 60)
    print("[!] 产物列要求: 每个单元格只能包含一个独立的SMILES")
    print("    不允许含有分隔符(点号.、分号;、逗号,)")
    print()

    print("可选列:")
    for idx in available_columns:
        print(f"  [{idx}] {df.columns[idx]}")

    # 选择列
    while True:
        try:
            col_idx = int(input("请输入产物列序号(只能选择1列): ").strip())
            if col_idx in available_columns:
                break
            else:
                print("[X] 该列不在可选列表中")
        except ValueError:
            print("[X] 请输入有效的数字")

    origin_name = df.columns[col_idx]
    default_name = 'product'

    print(f"\n请为此列指定名称(默认: {default_name})")

    while True:
        name = input(f"列名称[{default_name}]: ").strip() or default_name

        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            print("[X] 列名称只能包含英文、数字、下划线、连字符")
            continue

        if name in used_names:
            print(f"[X] 列名称'{name}'已被使用")
            continue

        break

    # 合法性检验: 排除非法SMILES、空值、含分隔符的SMILES
    invalid_rows = validate_product_column(df, col_idx)

    print(f"\n[OK] 产物列配置完成:")
    print(f"  原始列名: {origin_name}")
    print(f"  新列名: {name}")
    print(f"  非法行数: {len(invalid_rows)}")

    used_names.add(name)
    available_columns.remove(col_idx)

    return {
        'origin_idx': col_idx,
        'origin_name': origin_name,
        'role': 'product',
        'name': name,
        'invalid_rows': invalid_rows
    }


def suggest_others_name(origin_name: str) -> str:
    """
    根据原始列名推荐智能命名

    规则:
    - 包含'溶剂'/'solvent' → 'solvent'
    - 包含'催化'/'catalyst' → 'catalyst'
    - 包含'试剂'/'reagent' → 'reagent'
    - 包含'碱'/'base' → 'base'
    - 否则返回原始列名
    """
    origin_lower = origin_name.lower()

    if '溶剂' in origin_name or 'solvent' in origin_lower:
        return 'solvent'
    elif '催化' in origin_name or 'catalyst' in origin_lower:
        return 'catalyst'
    elif '试剂' in origin_name or 'reagent' in origin_lower:
        return 'reagent'
    elif '碱' in origin_name or 'base' in origin_lower:
        return 'base'
    else:
        return origin_name


def step2_5_declare_others_columns(df: pd.DataFrame, available_columns: List[int], used_names: Set[str]) -> List[Dict]:
    """
    步骤2.5: 声明其他组分SMILES列(可多列)

    特点:
    - 允许空值
    - 提供智能命名建议(solvent、catalyst、reagent、base)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='others', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.5: 声明其他组分SMILES列")
    print("=" * 60)
    print("常用名称建议: solvent(溶剂)、catalyst(催化剂)、reagent(试剂)、base(碱)")
    print()

    others_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入其他组分列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            if not all(idx in available_columns for idx in col_indices):
                print("[X] 存在无效的列序号")
                continue

            for col_idx in col_indices:
                origin_name = df.columns[col_idx]

                # 智能命名建议
                suggested_name = suggest_others_name(origin_name)
                default_name = origin_name  # 默认依然是原始列名

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称")
                print(f"  默认: {default_name}")
                if suggested_name != default_name:
                    print(f"  建议: {suggested_name} (可复制)")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("[X] 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"[X] 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非法SMILES,但允许空值
                invalid_rows = validate_smiles_column(df, col_idx, allow_empty=True)

                print(f"[OK] 非法行数: {len(invalid_rows)}")

                others_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'others',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 共声明 {len(others_columns)} 个其他组分列")
    return others_columns


def suggest_condition_name(origin_name: str) -> str:
    """根据原始列名推荐智能命名"""
    origin_lower = origin_name.lower()

    if '温度' in origin_name or 'temp' in origin_lower:
        return 'temperature'
    elif '压力' in origin_name or 'pressure' in origin_lower:
        return 'pressure'
    elif '时间' in origin_name or 'time' in origin_lower:
        return 'time'
    else:
        return origin_name


def step2_6_declare_condition_columns(df: pd.DataFrame, available_columns: List[int], used_names: Set[str]) -> List[Dict]:
    """
    步骤2.6: 声明条件数值列(可多列)

    特点:
    - 允许空值
    - 提供智能命名建议(temperature、pressure、time)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='condition', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.6: 声明条件数值列")
    print("=" * 60)
    print("常用名称建议: temperature(温度)、pressure(压力)、time(时间)")
    print()

    condition_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入条件数值列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            if not all(idx in available_columns for idx in col_indices):
                print("[X] 存在无效的列序号")
                continue

            for col_idx in col_indices:
                origin_name = df.columns[col_idx]

                # 智能命名建议
                suggested_name = suggest_condition_name(origin_name)
                default_name = origin_name

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称")
                print(f"  默认: {default_name}")
                if suggested_name != default_name:
                    print(f"  建议: {suggested_name} (可复制)")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("[X] 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"[X] 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非数值,但允许空值
                invalid_rows = validate_numeric_column_allow_empty(df, col_idx)

                print(f"[OK] 非法行数: {len(invalid_rows)}")

                condition_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'condition',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 共声明 {len(condition_columns)} 个条件数值列")
    return condition_columns


def step2_orchestrate(df: pd.DataFrame) -> List[Dict]:
    """
    步骤2总调度函数: 逐列声明列角色和名称

    Returns:
        list[dict]: 所有列的配置列表
    """
    # 2.1 选择需要的列
    selected_indices = step2_select_columns(df)
    available_columns = selected_indices.copy()
    used_names = set()

    all_configs = []

    # 2.2 声明标签列(必须)
    label_config = step2_2_declare_label_column(df, available_columns, used_names)
    all_configs.append(label_config)

    # 2.3 声明反应物SMILES列(可选)
    reactant_configs = step2_3_declare_reactant_columns(df, available_columns, used_names)
    all_configs.extend(reactant_configs)

    # 2.4 声明产物SMILES列(必须)
    product_config = step2_4_declare_product_column(df, available_columns, used_names)
    all_configs.append(product_config)

    # 2.5 声明其他组分SMILES列(可选)
    others_configs = step2_5_declare_others_columns(df, available_columns, used_names)
    all_configs.extend(others_configs)

    # 2.6 声明条件数值列(可选)
    condition_configs = step2_6_declare_condition_columns(df, available_columns, used_names)
    all_configs.extend(condition_configs)

    # 汇总统计
    print("\n" + "=" * 60)
    print("步骤2完成! 列声明汇总:")
    print("=" * 60)
    print(f"  标签列: 1")
    print(f"  反应物SMILES列: {len(reactant_configs)}")
    print(f"  产物SMILES列: 1")
    print(f"  其他组分SMILES列: {len(others_configs)}")
    print(f"  条件数值列: {len(condition_configs)}")
    print(f"  总计: {len(all_configs)} 列")

    # 计算总非法行数
    all_invalid_rows = set()
    for config in all_configs:
        all_invalid_rows.update(config['invalid_rows'])

    print(f"\n  合法性检验: 共发现 {len(all_invalid_rows)} 个非法行")

    return all_configs


def main():
    """
    主函数:运行数据集输入向导
    """
    print("\n" + "=" * 60)
    print("YONOD 数据集输入向导 v1.0")
    print("=" * 60)
    print("\n欢迎使用YONOD数据集输入向导!")
    print("本向导将引导您完成数据集的配置和规范化。")

    # 步骤1: 收集基本信息
    basic_info = step1_collect_basic_info()

    # 加载并展示数据集
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    df = load_and_preview_dataset(basic_info['dataset_path'])

    print("\n" + "=" * 60)
    print("步骤1完成!")
    print("=" * 60)

    # 检查是否跳过步骤2
    if basic_info['mapping_path']:
        print("\n[!] 检测到列映射文件,将跳过步骤2")
        print("(步骤3功能尚未实现)")
        return

    # 步骤2: 逐列声明列角色和名称
    print("\n接下来将进入步骤2: 逐列声明列角色和名称")
    input("按回车键继续...")

    all_configs = step2_orchestrate(df)

    print("\n" + "=" * 60)
    print("步骤2完成!")
    print("=" * 60)
    print("\n接下来将进入步骤3: 生成规范数据集与非法输入排除报告")
    print("(步骤3功能尚未实现)")


if __name__ == "__main__":
    main()
