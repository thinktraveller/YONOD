"""YONOD 交互式向导。

双击或在终端运行 `python yonod.py`，按提示填写参数后自动调用 run_yonod.main()。
列可通过列名、字母（A/B/C…）或序号（1/2/3…）三种方式指定。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


# ── 正则常量 ─────────────────────────────────────────────────────────────────

_RE_LETTER  = re.compile(r'^[A-Za-z]$')          # 单个字母（列代号）
_RE_NUMBER  = re.compile(r'^\d+$')               # 纯数字（1-based 序号）
_RE_TASK    = re.compile(r'^[A-Za-z0-9_\-]+$')   # 任务名：仅英文字母/数字/下划线/连字符


# ── 基础输入 ─────────────────────────────────────────────────────────────────

def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[中断] 用户取消。")
        sys.exit(0)


def _ask_required(prompt: str) -> str:
    """循环询问直到用户输入非空内容。"""
    while True:
        val = _input(prompt + ": ")
        if val:
            return val
        print("  [错误] 此项为必填，不能为空。")


def _ask_optional(prompt: str, default: str = "") -> str:
    """可选输入；直接回车返回 default。"""
    hint = f" [默认: {default}]" if default else " [留空跳过]"
    val = _input(prompt + hint + ": ")
    return val if val else default


# ── 列解析 ───────────────────────────────────────────────────────────────────

def _show_columns(columns: list) -> None:
    """打印列名及其字母/序号索引。"""
    print()
    print("  列序号  列名")
    print("  ------  ----")
    for i, col in enumerate(columns):
        letter = chr(ord('A') + i) if i < 26 else f"({i+1})"
        print(f"  {letter}({i+1:>2})   {col}")
    print()


def _resolve_one(token: str, columns: list) -> str | None:
    """将单个 token 解析为列名。支持：列名 / 字母 A-Z / 1-based 数字。"""
    # 精确列名匹配
    if token in columns:
        return token
    # 单字母 → 列索引
    if _RE_LETTER.match(token):
        idx = ord(token.upper()) - ord('A')
        if 0 <= idx < len(columns):
            return columns[idx]
        return None
    # 纯数字 → 1-based 索引
    if _RE_NUMBER.match(token):
        idx = int(token) - 1
        if 0 <= idx < len(columns):
            return columns[idx]
        return None
    return None


def _ask_single_col(prompt: str, columns: list, required: bool = True) -> str | None:
    """询问单列；支持列名/字母/数字。required=True 时不允许留空。"""
    while True:
        if required:
            raw = _ask_required(prompt)
        else:
            raw = _ask_optional(prompt)
            if not raw:
                return None

        token = raw.strip()
        if not token:
            if not required:
                return None
            print("  [错误] 此项为必填。")
            continue

        result = _resolve_one(token, columns)
        if result is None:
            print(f"  [错误] 无法识别 '{token}'，请输入列名、字母（如 B）或序号（如 2）。")
            continue
        return result


def _ask_multi_cols(prompt: str, columns: list, required: bool = True) -> list:
    """询问多列（空格分隔）；required=True 时至少需要一列。"""
    while True:
        if required:
            raw = _ask_required(prompt + "（空格分隔多列）")
        else:
            raw = _ask_optional(prompt + "（空格分隔多列）")
            if not raw:
                return []

        tokens = raw.split()
        resolved = []
        bad = []
        for t in tokens:
            r = _resolve_one(t, columns)
            if r is None:
                bad.append(t)
            elif r not in resolved:
                resolved.append(r)

        if bad:
            print(f"  [错误] 以下标识无法解析：{bad}。请用列名、字母或序号。")
            continue
        if required and not resolved:
            print("  [错误] 至少需要指定一列。")
            continue
        return resolved


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("  YONOD - Your One-stop Notebook Of Descriptors")
    print("  通用交互向导  v2")
    print("=" * 60)
    print()

    # ── 1. CSV 路径 ──────────────────────────────────────────────────────────
    print("[1/8] 输入 CSV 文件路径（可直接拖拽文件到终端）")
    while True:
        raw = _ask_required("  路径")
        csv_path = Path(raw.strip('"').strip("'"))
        if not csv_path.exists():
            print(f"  [错误] 文件不存在: {csv_path}")
            continue
        break

    # 加载 CSV 仅读取表头
    try:
        df_head = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
    except Exception:
        try:
            df_head = pd.read_csv(csv_path, encoding="gbk", nrows=0)
        except Exception as exc:
            print(f"  [错误] 无法读取 CSV：{exc}")
            sys.exit(1)

    columns: list = list(df_head.columns)
    print(f"  已检测到 {len(columns)} 列：")
    _show_columns(columns)

    # ── 2. 标签列（单列，必填）──────────────────────────────────────────────
    print("[2/8] 标签列（预测目标，如 yield / ee / ddG）")
    print("      支持：列名  /  字母（如 B）  /  序号（如 2）")
    label_col = _ask_single_col("  标签列", columns, required=True)
    print(f"  → 标签列确认：'{label_col}'")
    print()

    # ── 3. SMILES 列（多列，必填）───────────────────────────────────────────
    print("[3/8] SMILES 列（分子结构列，至少指定一列）")
    print("      支持：列名 / 字母 / 序号，多列用空格分隔")
    smiles_cols = _ask_multi_cols("  SMILES 列", columns, required=True)
    # 不允许与标签列重叠
    if label_col in smiles_cols:
        smiles_cols.remove(label_col)
        print(f"  [警告] 标签列 '{label_col}' 已从 SMILES 列中移除。")
    if not smiles_cols:
        print("  [错误] SMILES 列不能全为标签列，请重新运行并指定正确的列。")
        sys.exit(1)
    print(f"  → SMILES 列确认：{smiles_cols}")
    print()

    # ── 4. 数值辅助列（多列，可选）──────────────────────────────────────────
    print("[4/8] 数值辅助列（温度/压力等，可选）")
    print("      支持：列名 / 字母 / 序号，多列用空格分隔")
    numeric_cols = _ask_multi_cols("  数值辅助列", columns, required=False)
    # 去除与已使用列的重叠
    bad_numeric = [c for c in numeric_cols if c in smiles_cols or c == label_col]
    if bad_numeric:
        numeric_cols = [c for c in numeric_cols if c not in bad_numeric]
        print(f"  [警告] 以下列与标签列或 SMILES 列重叠，已忽略：{bad_numeric}")
    print(f"  → 数值辅助列确认：{numeric_cols if numeric_cols else '（无）'}")
    print()

    # ── 5. 任务名称（英文，必填）────────────────────────────────────────────
    print("[5/8] 任务名称（用于输出目录和报告标题）")
    print("      规则：仅允许英文字母、数字、下划线、连字符，如 amide_coupling")
    while True:
        task_name = _ask_required("  任务名称")
        if not _RE_TASK.match(task_name):
            print("  [错误] 任务名称只能包含英文字母、数字、下划线（_）、连字符（-）。")
            continue
        break
    print(f"  → 任务名称确认：'{task_name}'")
    print()

    # ── 6. 输出目录（可选，默认在脚本同级 result/<task_name>）───────────────
    default_out = str(Path(__file__).resolve().parent / "result" / task_name)
    print(f"[6/8] 输出目录（默认：{default_out}）")
    output_dir_raw = _ask_optional("  输出目录", default=default_out)
    output_dir_raw = output_dir_raw.strip('"').strip("'")
    print(f"  → 输出目录确认：'{output_dir_raw}'")
    print()

    # ── 7. 描述符（可选）────────────────────────────────────────────────────
    print("[7/8] 描述符选择（可选）")
    print("      可选值: morgan  maccs  fisd  molmetalm")
    descs_raw = _ask_optional("  描述符（空格分隔，留空=全选）")
    descs = descs_raw.split() if descs_raw else []
    if descs:
        _valid_descs = {"morgan", "maccs", "fisd", "molmetalm"}
        bad_descs = [d for d in descs if d not in _valid_descs]
        if bad_descs:
            print(f"  [警告] 未知描述符已忽略：{bad_descs}")
            descs = [d for d in descs if d in _valid_descs]
    print(f"  → 描述符确认：{descs if descs else '（全选）'}")
    print()

    # ── 8. 模型（可选）──────────────────────────────────────────────────────
    print("[8/8] 模型选择（可选）")
    print("      可选值: xgb  rf  svm  autogluon")
    models_raw = _ask_optional("  模型（空格分隔，留空=全选）")
    models = models_raw.split() if models_raw else []
    if models:
        _valid_models = {"xgb", "rf", "svm", "autogluon"}
        bad_models = [m for m in models if m not in _valid_models]
        if bad_models:
            print(f"  [警告] 未知模型已忽略：{bad_models}")
            models = [m for m in models if m in _valid_models]
    print(f"  → 模型确认：{models if models else '（全选）'}")
    print()

    # ── 命令预览 ─────────────────────────────────────────────────────────────
    cmd_parts = [
        "python", "run_yonod.py",
        "--csv", f'"{csv_path}"',
        "--label-col", label_col,
        "--smiles-cols", *smiles_cols,
    ]
    if numeric_cols:
        cmd_parts += ["--numeric-cols", *numeric_cols]
    cmd_parts += ["--task-name", task_name, "--output-dir", f'"{output_dir_raw}"']
    if descs:
        cmd_parts += ["--descriptors", *descs]
    if models:
        cmd_parts += ["--models", *models]

    print("=" * 60)
    print("  即将执行（等效命令）:")
    print("  " + " ".join(cmd_parts))
    print("=" * 60)
    print()
    _input("按 Enter 确认执行，Ctrl+C 取消... ")

    # ── 调用 run_yonod.main() ────────────────────────────────────────────────
    argv = [
        "run_yonod.py",
        "--csv", str(csv_path),
        "--label-col", label_col,
        "--smiles-cols", *smiles_cols,
    ]
    if numeric_cols:
        argv += ["--numeric-cols", *numeric_cols]
    argv += ["--task-name", task_name, "--output-dir", output_dir_raw]
    if descs:
        argv += ["--descriptors", *descs]
    if models:
        argv += ["--models", *models]

    sys.argv = argv
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_yonod import main as _run_main
    sys.exit(_run_main())


if __name__ == "__main__":
    main()
