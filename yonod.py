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
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from datetime import datetime
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
            print(f"[OK] 数据集文件存在: {dataset_path}")
            break
        else:
            print("[X] 文件不存在或不是CSV格式,请重新输入")

    # 2. 可选: 输入列映射文件路径
    mapping_path = input("\n请输入列映射文件路径(可选,直接回车跳过): ").strip()
    if mapping_path:
        if os.path.exists(mapping_path) and mapping_path.endswith('.csv'):
            print(f"[OK] 列映射文件存在: {mapping_path}")
            print("  将跳过步骤2,直接使用此映射进行合法性检验")
        else:
            print("[X] 文件不存在或不是CSV格式,将忽略此输入,进入正常流程")
            mapping_path = None
    else:
        mapping_path = None

    # 3. 输入项目名称
    while True:
        project_name = input("\n请输入项目名称(仅英文字母、数字、下划线): ").strip()
        if re.match(r'^[a-zA-Z0-9_]+$', project_name):
            print(f"[OK] 项目名称: {project_name}")
            break
        else:
            print("[X] 项目名称只能包含英文字母、数字、下划线")

    # 4. 输入项目文件夹位置
    while True:
        project_folder = input("\n请输入项目文件夹位置(将在此创建输出文件): ").strip()
        if os.path.isdir(project_folder) or not os.path.exists(project_folder):
            os.makedirs(project_folder, exist_ok=True)
            print(f"[OK] 项目文件夹: {project_folder}")
            break
        else:
            print("[X] 路径无效或不是文件夹")

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
            print("[OK] 使用GBK编码成功读取")
        except UnicodeDecodeError:
            # 如果GBK也失败,尝试latin1(几乎不会失败)
            print("⚠ GBK编码读取失败,尝试使用latin1编码...")
            df = pd.read_csv(dataset_path, encoding='latin1')
            print("[OK] 使用latin1编码成功读取")

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


# ==================== 步骤3: 生成规范数据集与非法输入排除报告 ====================

def split_smiles(smiles_str: str) -> List[str]:
    """
    拆分包含多个SMILES的字符串

    支持的分隔符: , ; 空格 .
    优先级：逗号 > 分号 > 空格 > 点号

    Args:
        smiles_str: SMILES字符串（可能包含多个分子）

    Returns:
        list: SMILES列表
    """
    # 按优先级尝试分隔符
    separators = [',', ';', ' ', '.']

    for sep in separators:
        if sep in smiles_str:
            parts = [s.strip() for s in smiles_str.split(sep) if s.strip()]
            # 过滤掉纯分隔符的残留（如空格分隔时可能有'.', ',', ';'残留）
            parts = [p for p in parts if p not in ['.', ',', ';', ' ']]
            if parts:  # 确保有有效部分
                return parts

    # 无分隔符,单个SMILES
    return [smiles_str.strip()]


def step3_1_generate_invalid_report(
    df: pd.DataFrame,
    all_column_configs: List[Dict],
    project_folder: str,
    project_name: str
) -> Optional[str]:
    """
    步骤3.1: 生成非法输入排除报告(Markdown格式)

    Args:
        df: 原始数据集
        all_column_configs: 所有列的配置列表(每个元素是dict,包含invalid_rows)
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 报告文件路径(如果生成),否则返回None
    """
    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_column_configs:
        all_invalid_rows.update(config['invalid_rows'])

    if not all_invalid_rows:
        print("\n[OK] 无非法输入,跳过报告生成")
        return None

    print(f"\n生成非法输入排除报告... 共 {len(all_invalid_rows)} 行")

    report_path = os.path.join(project_folder, f"{project_name}_invalid_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {project_name} 非法输入排除报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 第一部分: 汇总信息
        f.write("## 一、排除行汇总\n\n")
        f.write(f"**总计排除行数:** {len(all_invalid_rows)}\n\n")

        # 按行索引排序
        sorted_invalid_rows = sorted(all_invalid_rows)

        # 为每行标注在哪些步骤出现非法值
        row_error_map = {}  # {row_idx: [column_configs]}
        for row_idx in sorted_invalid_rows:
            row_error_map[row_idx] = []
            for config in all_column_configs:
                if row_idx in config['invalid_rows']:
                    row_error_map[row_idx].append(config)

        f.write("| 行号 | 出现非法值的列 |\n")
        f.write("|------|----------------|\n")
        for row_idx in sorted_invalid_rows:
            error_cols = [cfg['origin_name'] for cfg in row_error_map[row_idx]]
            f.write(f"| {row_idx} | {', '.join(error_cols)} |\n")

        f.write("\n---\n\n")

        # 第二部分: 逐行详细信息
        f.write("## 二、排除行详细信息\n\n")

        for row_idx in sorted_invalid_rows:
            f.write(f"### 行 {row_idx}\n\n")

            # 提取该行数据
            row_data = df.iloc[row_idx]

            # 生成表格
            f.write("| 列名 | 值 |\n")
            f.write("|------|----|\n")

            for config in all_column_configs:
                col_name = config['origin_name']
                col_value = row_data[col_name]

                # 如果该列在此行非法,加粗
                if row_idx in config['invalid_rows']:
                    col_value_str = f"**{col_value}**"
                else:
                    col_value_str = str(col_value)

                f.write(f"| {col_name} | {col_value_str} |\n")

            # 标注非法值位置
            error_cols = [cfg['origin_name'] for cfg in row_error_map[row_idx]]
            f.write(f"\n**非法值所在列:** {', '.join(error_cols)}\n\n")
            f.write("---\n\n")

    print(f"[OK] 报告已生成: {report_path}")
    return report_path


def step3_2_generate_column_mapping(
    all_column_configs: List[Dict],
    project_folder: str,
    project_name: str
) -> str:
    """
    步骤3.2: 生成列映射说明表(CSV格式)

    注意: 此CSV文件仅供人类参考查阅,不参与main.py的执行流程。
    main.py使用JSON配置文件中的column_roles字段来识别列角色。

    Args:
        all_column_configs: 所有列的配置列表
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 列映射文件路径
    """
    print("\n生成列映射说明表...")

    mapping_path = os.path.join(project_folder, f"{project_name}_column_mapping.csv")

    mapping_data = []
    for config in all_column_configs:
        mapping_data.append({
            'origin_name': config['origin_name'],
            'role': config['role'],
            'name': config['name']
        })

    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(mapping_path, index=False, encoding='utf-8')

    print(f"[OK] 列映射表已生成: {mapping_path}")
    print("    (注: 此文件仅供人类参考,不参与建模执行流程)")
    print("\n列映射内容预览:")
    print(mapping_df.to_string(index=False))

    return mapping_path


def step3_3_generate_normalized_dataset(
    df: pd.DataFrame,
    all_column_configs: List[Dict],
    project_folder: str,
    project_name: str
) -> str:
    """
    步骤3.3: 生成规范数据集(改进版: 两次扫描)

    第一次扫描: 确定每类SMILES的最大列数
    第二次扫描: 填充数据

    Args:
        df: 原始数据集
        all_column_configs: 所有列的配置列表
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 规范数据集文件路径
    """
    print("\n生成规范数据集...")

    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_column_configs:
        all_invalid_rows.update(config['invalid_rows'])

    valid_row_indices = [i for i in range(len(df)) if i not in all_invalid_rows]

    if all_invalid_rows:
        print(f"  跳过 {len(all_invalid_rows)} 个非法行")

    # 按角色分组
    role_groups = {
        'reactant': [],
        'others': [],
        'condition': [],
        'product': [],
        'label': []
    }

    for config in all_column_configs:
        role_groups[config['role']].append(config)

    # 第一次扫描: 确定最大列数
    max_reactant_count = 0
    max_others_count = {}  # {others_name: max_count}

    for row_idx in valid_row_indices:
        row_data = df.iloc[row_idx]

        # 统计reactant
        reactant_count = 0
        for config in role_groups['reactant']:
            col_name = config['origin_name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                reactant_count += len(smiles_parts)
        max_reactant_count = max(max_reactant_count, reactant_count)

        # 统计others
        for config in role_groups['others']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                if new_name not in max_others_count:
                    max_others_count[new_name] = 0
                max_others_count[new_name] = max(max_others_count[new_name], len(smiles_parts))

    print(f"  最大reactant数量: {max_reactant_count}")
    for name, count in max_others_count.items():
        print(f"  最大{name}数量: {count}")

    # 构建最终列名列表(按顺序)
    final_columns = []

    # reactant列
    for i in range(max_reactant_count):
        final_columns.append(f'reactant-{i+1}')

    # others列
    for config in role_groups['others']:
        name = config['name']
        if name in max_others_count:
            count = max_others_count[name]
            if count == 1:
                final_columns.append(name)
            else:
                for i in range(count):
                    final_columns.append(f'{name}-{i+1}')
        else:
            # 如果所有行都为空,仍然保留一列
            final_columns.append(name)

    # condition列
    for config in role_groups['condition']:
        final_columns.append(config['name'])

    # product列
    for config in role_groups['product']:
        final_columns.append(config['name'])

    # label列
    for config in role_groups['label']:
        final_columns.append(config['name'])

    # 第二次扫描: 填充数据
    normalized_rows = []

    for row_idx in valid_row_indices:
        row_data = df.iloc[row_idx]
        normalized_row = {col: '' for col in final_columns}  # 初始化为空字符串

        # 填充reactant
        reactant_smiles_list = []
        for config in role_groups['reactant']:
            col_name = config['origin_name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                reactant_smiles_list.extend(smiles_parts)

        for i, smiles in enumerate(reactant_smiles_list):
            if i < max_reactant_count:
                normalized_row[f'reactant-{i+1}'] = smiles

        # 填充others
        for config in role_groups['others']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]

            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                if len(smiles_parts) == 1:
                    if new_name in normalized_row:
                        normalized_row[new_name] = smiles_parts[0]
                else:
                    for i, smiles in enumerate(smiles_parts):
                        col_name_indexed = f'{new_name}-{i+1}'
                        if col_name_indexed in normalized_row:
                            normalized_row[col_name_indexed] = smiles

        # 填充condition
        for config in role_groups['condition']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        # 填充product
        for config in role_groups['product']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = str(value).strip() if pd.notna(value) else ''

        # 填充label
        for config in role_groups['label']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        normalized_rows.append(normalized_row)

    # 转换为DataFrame(按final_columns顺序)
    normalized_df = pd.DataFrame(normalized_rows, columns=final_columns)

    # 将NaN替换为空字符串（确保空值显示为空字符串而非NaN）
    normalized_df = normalized_df.fillna('')

    # 保存
    normalized_path = os.path.join(project_folder, f"{project_name}_normalized_dataset.csv")
    normalized_df.to_csv(normalized_path, index=False, encoding='utf-8')

    print(f"[OK] 规范数据集已生成: {normalized_path}")
    print(f"  有效行数: {len(normalized_df)}")
    print(f"  总列数: {len(normalized_df.columns)}")
    print("\n列名预览:")
    if len(final_columns) <= 10:
        print(f"  {', '.join(final_columns)}")
    else:
        print(f"  {', '.join(final_columns[:10])}...")

    return normalized_path


def step3_orchestrate(
    df: pd.DataFrame,
    all_configs: List[Dict],
    project_folder: str,
    project_name: str
) -> Dict[str, Optional[str]]:
    """
    步骤3总调度函数: 生成规范数据集与非法输入排除报告

    Returns:
        dict: {
            'invalid_report': str | None,
            'column_mapping': str,
            'normalized_dataset': str | None
        }
    """
    print("\n" + "=" * 60)
    print("步骤3: 生成规范数据集与非法输入排除报告")
    print("=" * 60)

    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_configs:
        all_invalid_rows.update(config['invalid_rows'])

    # 3.1 生成非法输入报告(如果有非法输入)
    invalid_report_path = step3_1_generate_invalid_report(
        df, all_configs, project_folder, project_name
    )

    # 3.2 生成列映射说明表(总是生成)
    mapping_path = step3_2_generate_column_mapping(
        all_configs, project_folder, project_name
    )

    # 3.3 生成规范数据集
    if all_invalid_rows:
        # 有非法行,生成规范数据集
        normalized_path = step3_3_generate_normalized_dataset(
            df, all_configs, project_folder, project_name
        )
    else:
        # 无非法行,跳过规范数据集生成
        print("\n[OK] 无非法输入,跳过规范数据集生成")
        print("    原始数据集已是规范格式,可直接使用")
        normalized_path = None

    print("\n" + "=" * 60)
    print("步骤3完成!")
    print("=" * 60)

    return {
        'invalid_report': invalid_report_path,
        'column_mapping': mapping_path,
        'normalized_dataset': normalized_path
    }


# ============================================================================
# 步骤4: 指定描述符
# ============================================================================

def step4_select_descriptors():
    """
    展示所有可用描述符,让用户选择

    Returns:
        list: 选中的描述符名称列表
    """
    print("\n" + "=" * 60)
    print("步骤4: 指定描述符")
    print("=" * 60)

    descriptors = {
        'morgan': '横向拼接,可编辑参与列和拼接顺序',
        'atmomaccs': '横向拼接,可编辑参与列和拼接顺序',
        'rdkit2d': '横向拼接,可编辑参与列和拼接顺序',
        'fisd': '横向拼接,可编辑参与列和拼接顺序',
        'molmetalm': '横向拼接,可编辑参与列和拼接顺序',
        'maf': '逐点加和,可编辑参与列',
        'drfp': '固定反应模式(reactant→product),可编辑额外加入反应物的others列'
    }

    print("\n可用描述符:")
    for idx, (name, desc) in enumerate(descriptors.items(), 1):
        print(f"  [{idx}] {name}: {desc}")

    print("\n请选择描述符(输入序号,用逗号分隔,如: 1,2,5)")
    print("或输入'all'选择全部描述符")

    while True:
        user_input = input("描述符序号: ").strip()

        if user_input.lower() == 'all':
            selected = list(descriptors.keys())
            break

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(descriptors) for idx in indices):
                selected = [list(descriptors.keys())[i-1] for i in indices]
                break
            else:
                print("[X] 存在无效的序号")
        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 已选择 {len(selected)} 个描述符: {', '.join(selected)}")
    return selected


def step4_configure_descriptor(descriptor_name, all_column_configs):
    """
    配置单个描述符的嵌入方式

    Args:
        descriptor_name: 描述符名称
        all_column_configs: 所有列的配置列表(用于展示可选列)

    Returns:
        dict: {
            'descriptor': str,
            'mode': 'concat' | 'sum' | 'reaction',
            'columns': list,  # 参与嵌入的列(按顺序)
            'extra_reactants': list  # 仅DRFP使用,额外加入反应物的others列
        }
    """
    print(f"\n配置描述符: {descriptor_name}")

    # 获取所有SMILES列(reactant, others, product)
    smiles_columns = [
        cfg for cfg in all_column_configs
        if cfg['role'] in ['reactant', 'others', 'product']
    ]

    print("\n当前所有SMILES列:")
    for idx, cfg in enumerate(smiles_columns):
        print(f"  [{idx}] {cfg['name']} (角色: {cfg['role']})")

    if descriptor_name in ['morgan', 'atmomaccs', 'rdkit2d', 'fisd', 'molmetalm']:
        # 横向拼接模式
        print("\n该描述符为横向拼接模式")
        print("默认顺序: reactant -> others -> product")
        print("您可以:")
        print("  1. 使用默认顺序")
        print("  2. 自定义参与列和顺序(输入列序号,用逗号分隔)")

        choice = input("选择(1/2): ").strip()

        if choice == '1':
            # 使用默认顺序
            columns = []
            # 按角色顺序添加
            for role in ['reactant', 'others', 'product']:
                for cfg in all_column_configs:
                    if cfg['role'] == role:
                        columns.append(cfg['name'])
        else:
            # 自定义顺序
            print("\n请输入列序号(用逗号分隔,顺序即为拼接顺序):")
            while True:
                user_input = input("列序号: ").strip()
                try:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                    if all(0 <= idx < len(smiles_columns) for idx in indices):
                        columns = [smiles_columns[i]['name'] for i in indices]
                        break
                    else:
                        print("[X] 存在无效的序号")
                except ValueError:
                    print("[X] 输入格式错误")

        print(f"[OK] 拼接顺序: {' -> '.join(columns)}")

        return {
            'descriptor': descriptor_name,
            'mode': 'concat',
            'columns': columns
        }

    elif descriptor_name == 'maf':
        # 逐点加和模式
        print("\n该描述符为逐点加和模式,不存在拼接顺序")
        print("请选择参与嵌入的列(输入列序号,用逗号分隔):")

        while True:
            user_input = input("列序号: ").strip()
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                if all(0 <= idx < len(smiles_columns) for idx in indices):
                    columns = [smiles_columns[i]['name'] for i in indices]
                    break
                else:
                    print("[X] 存在无效的序号")
            except ValueError:
                print("[X] 输入格式错误")

        print(f"[OK] 参与加和的列: {', '.join(columns)}")

        return {
            'descriptor': descriptor_name,
            'mode': 'sum',
            'columns': columns
        }

    elif descriptor_name == 'drfp':
        # 固定反应模式
        print("\n该描述符为固定反应模式: reactant -> product")
        print("您可以选择将某些others列加入到reactant中(如催化剂)")

        # 列出所有others列
        others_columns = [cfg for cfg in all_column_configs if cfg['role'] == 'others']

        if not others_columns:
            print("  无可选的others列")
            extra_reactants = []
        else:
            print("\n可选的others列:")
            for idx, cfg in enumerate(others_columns):
                print(f"  [{idx}] {cfg['name']}")

            user_input = input("请输入要加入reactant的others列序号(用逗号分隔,直接回车跳过): ").strip()

            if user_input:
                try:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                    extra_reactants = [others_columns[i]['name'] for i in indices if 0 <= i < len(others_columns)]
                except ValueError:
                    print("[X] 输入格式错误,跳过")
                    extra_reactants = []
            else:
                extra_reactants = []

        if extra_reactants:
            print(f"[OK] 额外加入反应物的列: {', '.join(extra_reactants)}")
        else:
            print("[OK] 使用默认reactant列")

        return {
            'descriptor': descriptor_name,
            'mode': 'reaction',
            'extra_reactants': extra_reactants
        }


def generate_default_descriptor_config(descriptor_name, all_column_configs):
    """
    为单个描述符生成默认配置

    Args:
        descriptor_name: 描述符名称
        all_column_configs: 所有列的配置列表

    Returns:
        dict: 描述符配置字典
    """
    if descriptor_name in ['morgan', 'atmomaccs', 'rdkit2d', 'fisd', 'molmetalm']:
        # concat模式: 按 reactant → others → product 顺序
        columns = []
        for role in ['reactant', 'others', 'product']:
            for cfg in all_column_configs:
                if cfg['role'] == role:
                    columns.append(cfg['name'])

        return {
            'descriptor': descriptor_name,
            'mode': 'concat',
            'columns': columns
        }

    elif descriptor_name == 'maf':
        # sum模式: 包含所有SMILES列
        columns = [
            cfg['name'] for cfg in all_column_configs
            if cfg['role'] in ['reactant', 'others', 'product']
        ]

        return {
            'descriptor': descriptor_name,
            'mode': 'sum',
            'columns': columns
        }

    elif descriptor_name == 'drfp':
        # reaction模式: extra_reactants为空
        return {
            'descriptor': descriptor_name,
            'mode': 'reaction',
            'extra_reactants': []
        }

    else:
        raise ValueError(f"未知的描述符: {descriptor_name}")


def display_descriptor_configs_summary(descriptor_configs):
    """
    以表格形式展示所有描述符的配置摘要

    Args:
        descriptor_configs: 描述符配置列表
    """
    print("\n" + "=" * 80)
    print("描述符配置摘要")
    print("=" * 80)

    # 表头
    print(f"{'序号':<6}{'描述符':<15}{'模式':<12}{'配置详情':<45}")
    print("-" * 80)

    # 表内容
    for idx, config in enumerate(descriptor_configs, 1):
        desc_name = config['descriptor']
        mode = config['mode']

        if mode == 'concat':
            # 横向拼接模式: 显示列顺序
            columns_str = ' → '.join(config['columns'])
            if len(columns_str) > 45:
                columns_str = columns_str[:42] + '...'
        elif mode == 'sum':
            # 加和模式: 显示参与列数量
            columns_str = f"加和 {len(config['columns'])} 列: {', '.join(config['columns'])}"
            if len(columns_str) > 45:
                columns_str = f"加和 {len(config['columns'])} 列"
        elif mode == 'reaction':
            # 反应模式: 显示额外反应物
            extra = config.get('extra_reactants', [])
            if extra:
                columns_str = f"reactant+{', '.join(extra)} → product"
            else:
                columns_str = "reactant → product"
        else:
            columns_str = "未知模式"

        print(f"{idx:<6}{desc_name:<15}{mode:<12}{columns_str:<45}")

    print("=" * 80)


def select_descriptor_to_edit(descriptor_configs):
    """
    让用户选择要编辑的描述符

    Args:
        descriptor_configs: 描述符配置列表

    Returns:
        int or None: 要编辑的描述符索引(0-based),若用户选择完成则返回None
    """
    print("\n请选择要编辑的描述符(输入序号),或输入 'done' 使用当前配置:")

    while True:
        user_input = input("序号 (或 'done'): ").strip()

        if user_input.lower() == 'done':
            return None

        try:
            idx = int(user_input)
            if 1 <= idx <= len(descriptor_configs):
                return idx - 1  # 返回0-based索引
            else:
                print(f"[X] 无效的序号,请输入 1-{len(descriptor_configs)}")
        except ValueError:
            print("[X] 输入格式错误,请输入数字或 'done'")


def step4_orchestrate(all_column_configs):
    """
    步骤4总控函数(默认配置 + 可选编辑模式)

    Args:
        all_column_configs: 所有列的配置列表

    Returns:
        list: 所有描述符的配置列表
    """
    # 4.1 选择描述符
    selected_descriptors = step4_select_descriptors()

    # 4.2 自动生成所有描述符的默认配置
    descriptor_configs = []
    for desc_name in selected_descriptors:
        config = generate_default_descriptor_config(desc_name, all_column_configs)
        descriptor_configs.append(config)

    # 4.3 显示配置摘要
    display_descriptor_configs_summary(descriptor_configs)

    # 4.4 询问是否编辑
    print("\n是否需要编辑某个描述符的配置? (Y/n)")
    choice = input("选择: ").strip().lower()

    if choice not in ['y', 'yes', '']:
        print("\n[OK] 使用默认配置")
        return descriptor_configs

    # 4.5 循环编辑流程
    while True:
        # 选择要编辑的描述符
        edit_idx = select_descriptor_to_edit(descriptor_configs)

        if edit_idx is None:
            # 用户选择完成编辑
            print("\n[OK] 配置完成!")
            break

        # 编辑选中的描述符
        desc_name = descriptor_configs[edit_idx]['descriptor']
        print(f"\n正在编辑描述符: {desc_name}")

        # 调用原有的配置函数
        new_config = step4_configure_descriptor(desc_name, all_column_configs)
        descriptor_configs[edit_idx] = new_config

        # 显示更新后的配置摘要
        display_descriptor_configs_summary(descriptor_configs)

    return descriptor_configs


# ============================================================================
# 步骤5-8: 其他配置项
# ============================================================================

def step5_select_models():
    """
    选择建模模型

    Returns:
        list: 选中的模型名称列表
    """
    print("\n" + "=" * 60)
    print("步骤5: 指定建模模型")
    print("=" * 60)

    models = ['XGBoost', 'Random Forest', 'SVM', 'AutoGluon', 'Neural Network']

    print("\n可用模型:")
    for idx, model in enumerate(models, 1):
        print(f"  [{idx}] {model}")

    print("\n请选择模型(输入序号,用逗号分隔,如: 1,2,4)")
    print("或输入'all'选择全部模型")

    while True:
        user_input = input("模型序号: ").strip()

        if user_input.lower() == 'all':
            selected = models
            break

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(models) for idx in indices):
                selected = [models[i-1] for i in indices]
                break
            else:
                print("[X] 存在无效的序号")
        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 已选择 {len(selected)} 个模型: {', '.join(selected)}")
    return selected


def step6_dataset_metadata():
    """
    收集数据集元信息

    Returns:
        dict: {
            'repo_url': str,
            'doi': str,
            'notes': str
        }
    """
    print("\n" + "=" * 60)
    print("步骤6: 补充数据集信息")
    print("=" * 60)

    repo_url = input("项目地址(可选): ").strip()
    doi = input("文献DOI(可选): ").strip()
    notes = input("备注(可选): ").strip()

    metadata = {
        'repo_url': repo_url,
        'doi': doi,
        'notes': notes
    }

    print("\n[OK] 元信息已记录")
    return metadata


def step7_select_report_format():
    """
    选择报告输出格式

    Returns:
        list: 选中的格式列表
    """
    print("\n" + "=" * 60)
    print("步骤7: 选择报告输出格式")
    print("=" * 60)

    formats = ['Markdown', 'HTML', 'PDF', 'JSON']

    print("\n可用格式:")
    for idx, fmt in enumerate(formats, 1):
        print(f"  [{idx}] {fmt}")

    print("\n请选择输出格式(输入序号,用逗号分隔,如: 1,3)")

    while True:
        user_input = input("格式序号: ").strip()

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(formats) for idx in indices):
                selected = [formats[i-1] for i in indices]
                break
            else:
                print("[X] 存在无效的序号")
        except ValueError:
            print("[X] 输入格式错误")

    print(f"\n[OK] 已选择输出格式: {', '.join(selected)}")
    return selected


def step8_confirm_and_generate(df, all_invalid_rows, project_folder, project_name):
    """
    展示最终配置,生成修复后数据集(可选)

    Args:
        df: 原始数据集
        all_invalid_rows: 所有非法行的索引集合
        project_folder: 项目文件夹
        project_name: 项目名称

    Returns:
        str: 修复后数据集文件路径(如果生成),否则返回None
    """
    print("\n" + "=" * 60)
    print("步骤8: 确认配置并生成修复后数据集")
    print("=" * 60)

    print("\n是否生成修复后数据集(删除非法行)?")
    print("  1. 是,生成修复后数据集")
    print("  2. 否,仅保留原始数据集")

    choice = input("选择(1/2): ").strip()

    if choice == '1':
        print("\n[OK] 将生成修复后数据集(删除非法行)")
        fixed_path = generate_fixed_dataset(df, all_invalid_rows, project_folder, project_name)
        return fixed_path
    else:
        print("\n[OK] 不生成修复后数据集")
        return None


def generate_fixed_dataset(df, all_invalid_rows, project_folder, project_name):
    """
    生成修复后数据集(删除非法行)

    Args:
        df: 原始数据集
        all_invalid_rows: 所有非法行的索引集合
        project_folder: 项目文件夹
        project_name: 项目名称

    Returns:
        str: 修复后数据集文件路径
    """
    print("\n生成修复后数据集...")

    valid_row_indices = [i for i in range(len(df)) if i not in all_invalid_rows]
    fixed_df = df.iloc[valid_row_indices]

    fixed_path = os.path.join(project_folder, f"{project_name}_修复后数据集.csv")
    fixed_df.to_csv(fixed_path, index=False, encoding='utf-8')

    print(f"[OK] 修复后数据集已生成: {fixed_path}")
    print(f"  原始行数: {len(df)}")
    print(f"  删除行数: {len(all_invalid_rows)}")
    print(f"  剩余行数: {len(fixed_df)}")

    return fixed_path


def save_config_file(
    project_folder: str,
    project_name: str,
    dataset_path: str,
    column_mapping_path: str,
    normalized_dataset_path: Optional[str],
    all_column_configs: List[Dict],
    descriptor_configs: List[Dict],
    selected_models: List[str],
    metadata: Dict,
    report_formats: List[str]
) -> str:
    """
    保存配置文件 (yonod_config.json)

    Args:
        project_folder: 项目文件夹路径
        project_name: 项目名称
        dataset_path: 原始数据集路径
        column_mapping_path: 列映射文件路径
        normalized_dataset_path: 规范数据集路径（可能为None）
        all_column_configs: 所有列的配置列表
        descriptor_configs: 描述符配置列表
        selected_models: 选中的模型列表
        metadata: 元信息字典
        report_formats: 报告格式列表

    Returns:
        str: 配置文件路径
    """
    # 构建列角色信息
    column_roles = {
        'label': None,
        'reactants': [],
        'products': [],
        'others': [],
        'conditions': []
    }

    for cfg in all_column_configs:
        role = cfg['role']
        name = cfg['name']
        if role == 'label':
            column_roles['label'] = name
        elif role == 'reactant':
            column_roles['reactants'].append(name)
        elif role == 'product':
            column_roles['products'].append(name)
        elif role == 'others':
            column_roles['others'].append(name)
        elif role == 'condition':
            column_roles['conditions'].append(name)

    # 确定要使用的数据集路径（优先使用规范数据集）
    if normalized_dataset_path:
        effective_dataset_path = normalized_dataset_path
    else:
        effective_dataset_path = dataset_path

    # 构建配置字典
    config = {
        'version': '1.0',
        'project_name': project_name,
        'dataset_path': effective_dataset_path,
        'column_mapping_path': column_mapping_path,  # 注: 仅供人类参考,不参与main.py执行流程
        'descriptors': descriptor_configs,
        'models': selected_models,
        'metadata': metadata,
        'report_formats': report_formats,
        'column_roles': column_roles  # main.py使用此字段识别列角色
    }

    # 保存配置文件
    config_path = os.path.join(project_folder, f"{project_name}_yonod_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[OK] 配置文件已生成: {config_path}")
    return config_path


def step9_auto_launch_modeling(config_path: str) -> bool:
    """
    步骤9: 询问是否自动启动建模

    Args:
        config_path: 配置文件路径

    Returns:
        bool: 是否成功启动建模
    """
    print("\n" + "=" * 60)
    print("步骤9: 启动建模")
    print("=" * 60)

    print("\n是否立即启动建模?")
    print("  Y - 是，立即启动")
    print("  n - 否，稍后手动执行")

    user_input = input("\n选择 [Y/n]: ").strip().lower()

    if user_input in ('', 'y', 'yes'):
        print("\n[启动] 正在调用 yonod.py 进行建模...")
        print("=" * 60)

        # 构建命令
        main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
        cmd = [sys.executable, main_script, '--config', config_path]

        try:
            # 使用 subprocess 调用，实时输出
            result = subprocess.run(
                cmd,
                check=False,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if result.returncode == 0:
                print("\n[完成] 建模任务已成功完成!")
                return True
            else:
                print(f"\n[警告] 建模任务返回码: {result.returncode}")
                return False

        except Exception as e:
            print(f"\n[错误] 启动建模失败: {e}")
            print(f"\n您可以稍后手动执行以下命令:")
            print(f"  python main.py --config \"{config_path}\"")
            return False
    else:
        print("\n[跳过] 已跳过自动建模")
        print(f"\n您可以稍后手动执行以下命令:")
        print(f"  python main.py --config \"{config_path}\"")
        return False


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

    # 步骤3: 生成规范数据集与非法输入排除报告
    print("\n接下来将进入步骤3: 生成规范数据集与非法输入排除报告")
    input("按回车键继续...")

    # 收集所有非法行索引(在步骤3之前需要完成)
    all_invalid_rows = set()
    for cfg in all_configs:
        if 'invalid_rows' in cfg:
            all_invalid_rows.update(cfg['invalid_rows'])

    output_paths = step3_orchestrate(
        df,
        all_configs,
        basic_info['project_folder'],
        basic_info['project_name']
    )

    print("\n" + "=" * 60)
    print("步骤3完成!")
    print("=" * 60)

    # 步骤4: 指定描述符
    print("\n接下来将进入步骤4: 指定描述符")
    input("按回车键继续...")

    descriptor_configs = step4_orchestrate(all_configs)

    print("\n" + "=" * 60)
    print("步骤4完成!")
    print("=" * 60)

    # 步骤5: 选择建模模型
    print("\n接下来将进入步骤5: 选择建模模型")
    input("按回车键继续...")

    selected_models = step5_select_models()

    print("\n" + "=" * 60)
    print("步骤5完成!")
    print("=" * 60)

    # 步骤6: 补充数据集元信息
    print("\n接下来将进入步骤6: 补充数据集元信息")
    input("按回车键继续...")

    metadata = step6_dataset_metadata()

    print("\n" + "=" * 60)
    print("步骤6完成!")
    print("=" * 60)

    # 步骤7: 选择报告输出格式
    print("\n接下来将进入步骤7: 选择报告输出格式")
    input("按回车键继续...")

    report_formats = step7_select_report_format()

    print("\n" + "=" * 60)
    print("步骤7完成!")
    print("=" * 60)

    # 步骤8: 确认配置并生成修复后数据集
    print("\n接下来将进入步骤8: 确认配置并生成修复后数据集")
    input("按回车键继续...")

    fixed_dataset_path = step8_confirm_and_generate(
        df,
        all_invalid_rows,
        basic_info['project_folder'],
        basic_info['project_name']
    )

    # 保存配置文件
    config_path = save_config_file(
        project_folder=basic_info['project_folder'],
        project_name=basic_info['project_name'],
        dataset_path=basic_info['dataset_path'],
        column_mapping_path=output_paths['column_mapping'],
        normalized_dataset_path=output_paths['normalized_dataset'],
        all_column_configs=all_configs,
        descriptor_configs=descriptor_configs,
        selected_models=selected_models,
        metadata=metadata,
        report_formats=report_formats
    )

    # 输出最终总结
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print("\n生成的文件:")
    if output_paths['invalid_report']:
        print(f"  - 非法输入报告: {output_paths['invalid_report']}")
    print(f"  - 列映射表: {output_paths['column_mapping']}")
    if output_paths['normalized_dataset']:
        print(f"  - 规范数据集: {output_paths['normalized_dataset']}")
    else:
        print("  - 规范数据集: 无需生成（原始数据集无非法值）")
    if fixed_dataset_path:
        print(f"  - 修复后数据集: {fixed_dataset_path}")
    print(f"  - 配置文件: {config_path}")

    print("\n配置汇总:")
    print(f"  - 描述符: {', '.join([cfg['descriptor'] for cfg in descriptor_configs])}")
    print(f"  - 模型: {', '.join(selected_models)}")
    print(f"  - 报告格式: {', '.join(report_formats)}")
    if metadata['repo_url']:
        print(f"  - 项目地址: {metadata['repo_url']}")
    if metadata['doi']:
        print(f"  - DOI: {metadata['doi']}")
    if metadata['notes']:
        print(f"  - 备注: {metadata['notes']}")

    # 步骤9: 询问是否自动启动建模
    step9_auto_launch_modeling(config_path)


if __name__ == "__main__":
    main()
