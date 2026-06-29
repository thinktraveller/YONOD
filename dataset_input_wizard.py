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
    print("\n接下来将进入步骤2: 逐列声明列角色和名称")
    print("(步骤2功能尚未实现)")


if __name__ == "__main__":
    main()
