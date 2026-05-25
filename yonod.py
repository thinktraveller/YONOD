"""YONOD 交互式向导。

双击或在终端运行 `python yonod.py`，按提示填写参数后自动调用 run_yonod.main()。
无需 bat / cmd，所有交互均在 Python 内完成，不受 cmd.exe 特殊字符限制。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ask(prompt: str, default: str = "") -> str:
    hint = f"（留空{f'={default}' if default else '跳过'}）" if True else ""
    suffix = f" [{default}]" if default else " [留空跳过]"
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[中断] 用户取消。")
        sys.exit(0)
    return val if val else default


def main() -> None:
    print()
    print("=" * 55)
    print("  YONOD - Your One-stop Notebook Of Descriptors")
    print("  通用交互向导")
    print("=" * 55)
    print()

    # ── 1. CSV 路径 ──────────────────────────────────────────
    csv_raw = _ask("[1/8] 输入 CSV 路径（可直接拖拽文件到终端）", "")
    csv_raw = csv_raw.strip('"').strip("'")
    if not csv_raw:
        print("[错误] CSV 路径不能为空。")
        sys.exit(1)
    csv_path = Path(csv_raw)
    if not csv_path.exists():
        print(f"[错误] 文件不存在: {csv_path}")
        sys.exit(1)

    # ── 2. 标签列 ────────────────────────────────────────────
    label_col = _ask("[2/8] 标签列名称（如 yield / ee / delta_G）", "")
    if not label_col:
        print("[错误] 标签列名称不能为空。")
        sys.exit(1)

    # ── 3. SMILES 列（可选，空格分隔） ───────────────────────
    smiles_raw = _ask("[3/8] SMILES 列名，多列用空格分隔", "")
    smiles_cols = smiles_raw.split() if smiles_raw else []

    # ── 4. 数值辅助列（可选） ────────────────────────────────
    numeric_raw = _ask("[4/8] 数值辅助列名，多列用空格分隔", "")
    numeric_cols = numeric_raw.split() if numeric_raw else []

    # ── 5. 输出目录（可选） ──────────────────────────────────
    output_dir_raw = _ask("[5/8] 输出目录路径", "")
    output_dir_raw = output_dir_raw.strip('"').strip("'")

    # ── 6. 任务名称（可选） ──────────────────────────────────
    task_name = _ask("[6/8] 任务名称（用于报告标题）", "")

    # ── 7. 描述符（可选） ────────────────────────────────────
    print()
    print("  [7/8] 可选描述符: morgan  maccs  fisd  molmetalm")
    descs_raw = _ask("       请选，空格分隔", "")
    descs = descs_raw.split() if descs_raw else []

    # ── 8. 模型（可选） ──────────────────────────────────────
    print()
    print("  [8/8] 可选模型: xgb  rf  svm  autogluon")
    models_raw = _ask("       请选，空格分隔", "")
    models = models_raw.split() if models_raw else []

    # ── 构建等效命令（仅用于预览） ───────────────────────────
    cmd_parts = ["python", "run_yonod.py", "--csv", f'"{csv_path}"', "--label-col", f'"{label_col}"']
    if smiles_cols:
        cmd_parts += ["--smiles-cols"] + smiles_cols
    if numeric_cols:
        cmd_parts += ["--numeric-cols"] + numeric_cols
    if output_dir_raw:
        cmd_parts += ["--output-dir", f'"{output_dir_raw}"']
    if task_name:
        cmd_parts += ["--task-name", f'"{task_name}"']
    if descs:
        cmd_parts += ["--descriptors"] + descs
    if models:
        cmd_parts += ["--models"] + models

    print()
    print("=" * 55)
    print("  即将执行（等效命令）:")
    print("  " + " ".join(cmd_parts))
    print("=" * 55)
    print()
    try:
        input("按 Enter 确认执行，Ctrl+C 取消... ")
    except (EOFError, KeyboardInterrupt):
        print("\n[中断] 用户取消。")
        sys.exit(0)

    # ── 调用 run_yonod.main() ────────────────────────────────
    argv = ["run_yonod.py", "--csv", str(csv_path), "--label-col", label_col]
    if smiles_cols:
        argv += ["--smiles-cols"] + smiles_cols
    if numeric_cols:
        argv += ["--numeric-cols"] + numeric_cols
    if output_dir_raw:
        argv += ["--output-dir", output_dir_raw]
    if task_name:
        argv += ["--task-name", task_name]
    if descs:
        argv += ["--descriptors"] + descs
    if models:
        argv += ["--models"] + models

    sys.argv = argv

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_yonod import main as _run_main
    sys.exit(_run_main())


if __name__ == "__main__":
    main()
