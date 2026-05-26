"""YONOD 通用入口（向导 + CLI 合并版）。

双击或不带参数运行时启动交互向导；带 --csv 等参数直接进入 CLI pipeline。

交互向导用法
------------
    python yonod.py

CLI 用法示例
------------
    # 显式指定标签列和 SMILES 列：
    python yonod.py --csv 数据集/酰胺缩合数据集.csv --label-col yield --smiles-cols Reactant1 Reactant2

    # ECC 数据集（2 SMILES 列 + 温度辅助列）：
    python yonod.py --csv 数据集/镍催化偶联数据集/Raw_Dataset.csv \\
        --smiles-cols Ligand_SMILES Product_SMILES \\
        --numeric-cols "Temp (K)" \\
        --label-col "ddG" --task-name ECC

    # 仅跑 morgan × rf（快速冒烟）：
    python yonod.py --csv ... --smiles-cols ... --label-col ... --descriptors morgan --models rf

输出（默认到 results/<task-name>/）
-------------------------------------
  metrics_summary.csv   所有 (描述符, 模型) 组合的指标
  run_<timestamp>.log   控制台镜像日志
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yonod_yield.universal.csv_loader import load_csv_with_roles
from yonod_yield.universal.feature_builder import build_universal_features

_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm"]
_MODEL_NAMES = ["xgb", "rf", "svm", "autogluon"]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

_CSV_COLUMNS = [
    "task_name", "descriptor", "model",
    "n_smiles_cols", "n_numeric_cols",
    "n_samples", "n_total", "coverage", "feature_dim",
    "cv", "r2_mean", "r2_std", "rmse_mean", "mae_mean",
    "train_time_s", "device",
]

# ── 向导正则常量 ──────────────────────────────────────────────────────────────

_RE_LETTER = re.compile(r'^[A-Za-z]$')
_RE_NUMBER = re.compile(r'^\d+$')
_RE_TASK   = re.compile(r'^[A-Za-z0-9_\-]+$')


# ─────────────────────────────────────── 日志 ──────────────────────────────── #

class _Tee:
    """同时写入 stdout 和日志文件，每行前缀时间戳。"""

    def __init__(self, fh: TextIO) -> None:
        self.fh = fh
        self._real_stdout = sys.__stdout__
        self._bol = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        chunks: List[str] = []
        for ch in data:
            if self._bol and ch != "\n":
                chunks.append(_dt.datetime.now().strftime("[%H:%M:%S] "))
                self._bol = False
            chunks.append(ch)
            if ch == "\n":
                self._bol = True
        out = "".join(chunks)
        self._real_stdout.write(out)
        self.fh.write(out)
        self.fh.flush()
        return len(data)

    def flush(self) -> None:
        self._real_stdout.flush()
        self.fh.flush()


class _Heartbeat:
    def __init__(self, interval: float, label: str) -> None:
        self.interval = interval
        self.label = label
        self._stop = threading.Event()
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.time()
        self._stop.clear()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            print(f"  [hb] {self.label} still running, {time.time()-self._t0:.0f}s", flush=True)


# ─────────────────────────────────── 模型工厂 ──────────────────────────────── #

def _make_model(model_name: str, args: argparse.Namespace) -> Any:
    if model_name == "xgb":
        from yonod_yield.models.xgb_model import XGBYieldModel
        return XGBYieldModel()
    if model_name == "rf":
        from yonod_yield.models.rf_model import RFYieldModel
        return RFYieldModel(
            n_jobs=args.rf_n_jobs,
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
        )
    if model_name == "svm":
        from yonod_yield.models.svm_model import SVMYieldModel
        sub = args.svm_subsample if args.svm_subsample > 0 else None
        return SVMYieldModel(subsample_n=sub)
    if model_name == "autogluon":
        from yonod_yield.models.autogluon_model import AutoGluonYieldModel
        return AutoGluonYieldModel()
    raise ValueError(f"未知模型名称 {model_name!r}")


# ──────────────────────────── 带数值列的 CV 循环 ───────────────────────────── #

def _cv_with_numeric(
    model_name: str,
    model: Any,
    X_smiles: np.ndarray,
    X_numeric: np.ndarray,
    y: np.ndarray,
    cv: int,
    svm_subsample: Optional[int],
) -> Dict[str, Any]:
    """KFold CV，在每折 train 集上 fit StandardScaler（防数据泄露）。

    AutoGluon 使用内部 holdout，无法嵌入 KFold；对其使用全局 scaler 并打印警告。
    """
    if model_name == "autogluon":
        print(
            "[warning] AutoGluon + numeric cols: 使用全局 StandardScaler（非折内归一化）",
            file=sys.stderr,
        )
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_numeric)
        X_combined = np.hstack([X_smiles, X_num_scaled])
        return model.cross_validate(X_combined, y, cv=cv)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2s, rmses, maes = [], [], []
    oof = np.full(len(y), np.nan, dtype=np.float64)
    t0 = time.time()

    for fold_idx, (tr, te) in enumerate(kf.split(X_smiles)):
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(X_numeric[tr])
        X_num_te = scaler.transform(X_numeric[te])
        X_tr = np.hstack([X_smiles[tr], X_num_tr])
        X_te = np.hstack([X_smiles[te], X_num_te])
        y_tr, y_te = y[tr], y[te]

        if model_name == "rf":
            est = model._build()
            est.fit(X_tr, y_tr)
        elif model_name == "xgb":
            est = model._build()
            est.fit(X_tr, y_tr)
        elif model_name == "svm":
            n_features = X_tr.shape[1]
            n_train_fold = len(X_tr)
            if svm_subsample and svm_subsample < n_train_fold:
                rng = np.random.default_rng(seed=fold_idx)
                sub_idx = rng.choice(n_train_fold, size=svm_subsample, replace=False)
                X_tr_fit = X_tr[sub_idx]
                y_tr_fit = y_tr[sub_idx]
            else:
                X_tr_fit = X_tr
                y_tr_fit = y_tr
            est = model._build(n_features=n_features, n_train=len(X_tr_fit))
            est.fit(X_tr_fit, y_tr_fit)
        else:
            raise ValueError(f"_cv_with_numeric: 未处理的模型 {model_name!r}")

        pred = est.predict(X_te)
        oof[te] = pred
        r2s.append(r2_score(y_te, pred))
        rmses.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        maes.append(float(mean_absolute_error(y_te, pred)))

    elapsed = time.time() - t0
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "train_time_s": float(elapsed),
        "device": "cpu",
        "oof_pred": oof,
        "oof_y_true": y,
    }


# ────────────────────────────── argparse ──────────────────────────────────── #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YONOD 通用化入口：CSV → 特征 → 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True, help="输入数据集 CSV 路径")
    p.add_argument("--label-col", required=True, help="标签列名（必填）")
    p.add_argument("--smiles-cols", nargs="+", required=True, help="SMILES 列名，多列空格分隔（必填）")
    p.add_argument("--numeric-cols", nargs="+", default=None, help="数值辅助列名（可选）")
    p.add_argument("--task-name", default=None, help="任务名称（报告标题，默认 CSV 文件名去后缀）")
    p.add_argument(
        "--descriptors", nargs="+", default=_DESCRIPTOR_NAMES,
        choices=_DESCRIPTOR_NAMES, help="选用描述符"
    )
    p.add_argument(
        "--models", nargs="+", default=_MODEL_NAMES,
        choices=_MODEL_NAMES, help="选用模型"
    )
    p.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    p.add_argument("--cv", type=int, default=5, help="K-Fold 的 K 值")
    p.add_argument("--nrows", type=int, default=None, help="仅读取前 N 行（调试用）")
    p.add_argument("--append", action="store_true", help="追加写入 metrics_summary.csv")
    p.add_argument("--log-file", type=Path, default=None, help="日志文件路径")
    p.add_argument("--heartbeat", type=float, default=30.0, help="心跳打印间隔（秒，0 禁用）")
    p.add_argument("--svm-subsample", type=int, default=8000, help="SVM 子采样数量")
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=None)
    return p.parse_args(argv)


# ────────────────────────────── 辅助函数 ──────────────────────────────────── #

def _resolve_output_dir(csv_path: Path, task_name: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    return DEFAULT_RESULTS_ROOT / f"{task_name}建模报告"


def _print_table(rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [c for c in ["descriptor", "model", "feature_dim", "n_samples",
                        "r2_mean", "r2_std", "rmse_mean", "mae_mean", "train_time_s"]
            if c in df.columns]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print()
    print(df[cols].to_string(index=False))


def _save_metrics(rows: List[dict], out_dir: Path, append: bool) -> Path:
    csv_path = out_dir / "metrics_summary.csv"
    df = pd.DataFrame(rows)
    existing_cols = [c for c in _CSV_COLUMNS if c in df.columns]
    df = df[existing_cols]
    if append and csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


# ─────────────────────────────────── main ─────────────────────────────────── #

def main(argv: Optional[List[str]] = None) -> int:
    """CLI pipeline 入口：解析参数 → 加载数据 → 描述符×模型 grid → 输出报告。"""
    args = parse_args(argv)

    if not args.csv.exists():
        print(f"[error] CSV 不存在: {args.csv}", file=sys.stderr)
        return 1

    task_name = args.task_name or args.csv.stem
    out_dir = _resolve_output_dir(args.csv, task_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log_file or (
        out_dir / f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log_fh)

    print(f"[init] 任务：{task_name}")
    print(f"[init] CSV：{args.csv}  nrows={args.nrows or 'all'}")
    print(f"[init] 输出目录：{out_dir}")

    dataset = load_csv_with_roles(
        csv_path=args.csv,
        smiles_cols=args.smiles_cols,
        numeric_cols=args.numeric_cols or [],
        label_col=args.label_col,
        nrows=args.nrows,
    )
    df = dataset.df
    smiles_cols = dataset.smiles_cols
    numeric_cols = dataset.numeric_cols
    label_col = dataset.label_col

    print(
        f"[load] n_rows={len(df)}  "
        f"SMILES列={smiles_cols}  "
        f"数值列={numeric_cols or '无'}  "
        f"标签列='{label_col}'"
    )
    print(f"[grid] 描述符={args.descriptors}  模型={args.models}  cv={args.cv}")

    svm_sub = args.svm_subsample if args.svm_subsample > 0 else None

    rows: List[dict] = []

    total = len(args.descriptors) * len(args.models)
    done = 0

    for desc_name in args.descriptors:
        print(f"\n[desc] 计算描述符: {desc_name} ...")
        X_smiles, X_numeric, mask = build_universal_features(
            smiles_cols=smiles_cols,
            numeric_cols=numeric_cols,
            df=df,
            desc_name=desc_name,
        )
        y = df[label_col].values[mask].astype(np.float64)

        print(
            f"[desc] {desc_name}: n_valid={mask.sum()}  "
            f"X_smiles.shape={X_smiles.shape}"
            + (f"  X_numeric.shape={X_numeric.shape}" if X_numeric is not None else "")
        )

        for model_name in args.models:
            done += 1
            label = f"{desc_name} x {model_name} ({done}/{total})"
            print(f"[eval] 开始: {label}")

            hb = _Heartbeat(interval=args.heartbeat, label=label)
            if args.heartbeat > 0:
                hb.start()

            try:
                model = _make_model(model_name, args)

                if X_numeric is None:
                    metrics = model.cross_validate(X_smiles, y, cv=args.cv)
                else:
                    metrics = _cv_with_numeric(
                        model_name, model,
                        X_smiles, X_numeric, y,
                        cv=args.cv,
                        svm_subsample=svm_sub,
                    )
            except Exception as exc:
                print(f"[error] {label} 失败: {exc}", file=sys.stderr)
                hb.stop()
                continue
            finally:
                hb.stop()

            row: dict = {
                "task_name":     task_name,
                "descriptor":    desc_name,
                "model":         model_name,
                "n_smiles_cols": len(smiles_cols),
                "n_numeric_cols":len(numeric_cols),
                "n_samples":     int(mask.sum()),
                "n_total":       int(len(mask)),
                "coverage":      float(mask.sum() / max(len(mask), 1)),
                "feature_dim":   int(X_smiles.shape[1]) + (X_numeric.shape[1] if X_numeric is not None else 0),
                "cv":            args.cv,
            }
            for k, v in metrics.items():
                if k not in ("oof_pred", "oof_y_true"):
                    row[k] = v

            rows.append(row)
            print(
                f"[done] {label}  "
                f"R²={row.get('r2_mean', float('nan')):.4f}  "
                f"RMSE={row.get('rmse_mean', float('nan')):.4f}  "
                f"t={row.get('train_time_s', 0):.1f}s"
            )

    _print_table(rows)

    if rows:
        csv_out = _save_metrics(rows, out_dir, args.append)
        print(f"\n[save] 指标已保存: {csv_out}")

        try:
            from yonod_yield.universal.report import generate_report
            task_info = {
                "task_name":     task_name,
                "csv_path":      args.csv,
                "n_samples":     rows[0].get("n_samples", "—") if rows else "—",
                "smiles_cols":   smiles_cols,
                "numeric_cols":  numeric_cols or "（无）",
                "label_col":     label_col,
                "n_combinations":len(rows),
            }
            report_path = generate_report(
                pd.DataFrame(rows), task_info, out_dir
            )
            print(f"[report] HTML 报告已生成: {report_path}")
        except Exception as exc:
            print(f"[warn] HTML 报告生成失败（不影响指标 CSV）: {exc}", file=sys.stderr)
    else:
        print("\n[warn] 无有效结果，metrics_summary.csv 未写入", file=sys.stderr)

    print(f"[done] 全部完成。日志: {log_path}")
    return 0


# ─────────────────────────── 交互向导（wizard） ───────────────────────────── #

def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[中断] 用户取消。")
        sys.exit(0)


def _ask_required(prompt: str) -> str:
    while True:
        val = _input(prompt + ": ")
        if val:
            return val
        print("  [错误] 此项为必填，不能为空。")


def _ask_optional(prompt: str, default: str = "") -> str:
    hint = f" [默认: {default}]" if default else " [留空跳过]"
    val = _input(prompt + hint + ": ")
    return val if val else default


def _show_columns(columns: list) -> None:
    print()
    print("  列序号  列名")
    print("  ------  ----")
    for i, col in enumerate(columns):
        letter = chr(ord('A') + i) if i < 26 else f"({i+1})"
        print(f"  {letter}({i+1:>2})   {col}")
    print()


def _resolve_one(token: str, columns: list) -> str | None:
    if token in columns:
        return token
    if _RE_LETTER.match(token):
        idx = ord(token.upper()) - ord('A')
        if 0 <= idx < len(columns):
            return columns[idx]
        return None
    if _RE_NUMBER.match(token):
        idx = int(token) - 1
        if 0 <= idx < len(columns):
            return columns[idx]
        return None
    return None


def _ask_single_col(prompt: str, columns: list, required: bool = True) -> str | None:
    while True:
        raw = _ask_required(prompt) if required else _ask_optional(prompt)
        if not raw:
            return None
        result = _resolve_one(raw.strip(), columns)
        if result is None:
            print(f"  [错误] 无法识别 '{raw}'，请输入列名、字母（如 B）或序号（如 2）。")
            continue
        return result


def _ask_multi_cols(prompt: str, columns: list, required: bool = True) -> list:
    while True:
        raw = _ask_required(prompt + "（空格分隔多列）") if required \
              else _ask_optional(prompt + "（空格分隔多列）")
        if not raw:
            return []
        tokens = raw.split()
        resolved, bad = [], []
        for t in tokens:
            r = _resolve_one(t, columns)
            (bad if r is None else resolved).append(r if r else t)
        if bad:
            print(f"  [错误] 以下标识无法解析：{bad}。请用列名、字母或序号。")
            continue
        resolved = list(dict.fromkeys(resolved))
        if required and not resolved:
            print("  [错误] 至少需要指定一列。")
            continue
        return resolved


def _col_display(col: str, columns: list) -> str:
    idx = columns.index(col) + 1
    return f"第 {idx} 列（列名：'{col}'）"


def _validate_label(df: pd.DataFrame, label_col: str, columns: list) -> None:
    print(f"  [验证] 检查标签列 '{label_col}' 的数值合法性 ...")
    series = df[label_col]
    numeric = pd.to_numeric(series, errors="coerce")
    bad_mask = numeric.isna() & series.notna()
    if bad_mask.any():
        raw_idx = int(bad_mask.idxmax())
        display_row = raw_idx + 2
        bad_val = series.iloc[raw_idx]
        col_desc = _col_display(label_col, columns)
        print(f"\n  [错误] {col_desc} 第 {display_row} 行的值 '{bad_val}' 不是数值。")
        print("         标签列必须为整数或浮点数，请检查数据后重新运行。")
        sys.exit(1)
    print(f"  [验证] 标签列 '{label_col}' 全量数值验证通过（共 {len(df)} 行）。")


def _normalize_smiles(smi: str) -> str:
    """将各类非标准分隔符规范化为 RDKit 标准的点分隔形式。

    处理三类情况：
    - 逗号（','）：阴阳离子对，如 'CCN=C=NCCCN(C)C,Cl'
    - 星号（'*'）：反应步骤分隔符，如 'A.B*C.D*E'（反应 SMILES 格式）
    - 波浪线（'~'）：组分替代表示，如 'CC(=O)O~CC(=O)O~[Pd]'
    连续分隔符（如 '**' 空步骤）会产生 '..'，一并压缩为单个 '.'。
    """
    s = smi.replace(",", ".").replace("*", ".").replace("~", ".")
    s = re.sub(r'\.{2,}', '.', s)
    return s.strip('.')


def _validate_smiles(df: pd.DataFrame, smiles_cols: list, columns: list) -> None:
    """用 RDKit 逐行检查 SMILES 列；发现第一个无效 SMILES 则退出。

    验证前先将逗号分隔的离子对规范化为点分隔（RDKit 标准），
    因此 'CCN=C=NCCCN(C)C,Cl' 这类试剂 SMILES 会被正确接受。
    RDKit 未安装时仅打印警告，不中断流程。
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        print("  [警告] RDKit 未安装，跳过 SMILES 格式验证。")
        return

    n = len(df)
    for col in smiles_cols:
        col_desc = _col_display(col, columns)
        print(f"  [验证] 检查 SMILES 列 '{col}'（{n} 行）...")
        for raw_idx, val in enumerate(df[col]):
            if pd.isna(val):
                continue
            normalized = _normalize_smiles(str(val))
            if Chem.MolFromSmiles(normalized) is None:
                display_row = raw_idx + 2
                print(f"\n  [错误] {col_desc} 第 {display_row} 行的值 '{val}' 不是有效的 SMILES。")
                print("         请检查数据（是否有乱码、截断或占位符）后重新运行。")
                sys.exit(1)
        print(f"  [验证] SMILES 列 '{col}' 全量验证通过。")


def wizard() -> None:
    """交互向导：逐步收集参数，确认后调用 main() 执行 pipeline。"""
    print()
    print("=" * 60)
    print("  YONOD - Your One-stop Notebook Of Descriptors")
    print("  通用交互向导  v3")
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

    print("  正在读取 CSV ...")
    df: pd.DataFrame | None = None
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        print("  [错误] 无法以 UTF-8 或 GBK 编码读取该 CSV，请检查文件格式。")
        sys.exit(1)

    columns: list = list(df.columns)
    print(f"  已读取 {len(df)} 行 × {len(columns)} 列")
    _show_columns(columns)

    # ── 2. 标签列 ────────────────────────────────────────────────────────────
    print("[2/8] 标签列（预测目标，如 yield / ee / ddG）")
    print("      支持：列名  /  字母（如 B）  /  序号（如 2）")
    label_col = _ask_single_col("  标签列", columns, required=True)
    print(f"  → 标签列确认：'{label_col}'")
    _validate_label(df, label_col, columns)
    print()

    # ── 3. SMILES 列 ─────────────────────────────────────────────────────────
    print("[3/8] SMILES 列（分子结构列，至少指定一列）")
    print("      支持：列名 / 字母 / 序号，多列用空格分隔")
    smiles_cols = _ask_multi_cols("  SMILES 列", columns, required=True)
    if label_col in smiles_cols:
        smiles_cols.remove(label_col)
        print(f"  [警告] 标签列 '{label_col}' 已从 SMILES 列中移除。")
    if not smiles_cols:
        print("  [错误] SMILES 列不能全为标签列，请重新运行并指定正确的列。")
        sys.exit(1)
    print(f"  → SMILES 列确认：{smiles_cols}")
    _validate_smiles(df, smiles_cols, columns)
    print()

    # ── 4. 数值辅助列 ────────────────────────────────────────────────────────
    print("[4/8] 数值辅助列（温度/压力等，可选）")
    print("      支持：列名 / 字母 / 序号，多列用空格分隔")
    numeric_cols = _ask_multi_cols("  数值辅助列", columns, required=False)
    bad_numeric = [c for c in numeric_cols if c in smiles_cols or c == label_col]
    if bad_numeric:
        numeric_cols = [c for c in numeric_cols if c not in bad_numeric]
        print(f"  [警告] 以下列与标签列或 SMILES 列重叠，已忽略：{bad_numeric}")
    print(f"  → 数值辅助列确认：{numeric_cols if numeric_cols else '（无）'}")
    print()

    # ── 5. 任务名称 ──────────────────────────────────────────────────────────
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

    # ── 6. 输出目录 ──────────────────────────────────────────────────────────
    default_out = str(Path(__file__).resolve().parent / "result" / task_name)
    print(f"[6/8] 输出目录（默认：{default_out}）")
    output_dir_raw = _ask_optional("  输出目录", default=default_out)
    output_dir_raw = output_dir_raw.strip('"').strip("'")
    print(f"  → 输出目录确认：'{output_dir_raw}'")
    print()

    # ── 7. 描述符 ────────────────────────────────────────────────────────────
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

    # ── 8. 模型 ──────────────────────────────────────────────────────────────
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
        "python", "yonod.py",
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

    # ── 构建 argv 并调用 main() ──────────────────────────────────────────────
    argv = [
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

    sys.exit(main(argv))


# ─────────────────────────────── 入口 ────────────────────────────────────── #

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        wizard()
