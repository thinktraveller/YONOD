"""YONOD 通用化入口。

支持任意 CSV 数据集（多 SMILES 列 + 可选数值辅助列），通过 argparse 驱动
csv_loader → feature_builder → 模型评估 → 结果汇总 的完整 pipeline。

用法示例
--------
# 自动探测 SMILES 列，显式指定标签列：
python run_yonod.py --csv 数据集/酰胺缩合数据集.csv --label-col yield

# ECC 数据集（2 SMILES 列 + 温度辅助列）：
python run_yonod.py --csv 数据集/镍催化偶联数据集/Raw_Dataset.csv \\
    --smiles-cols Ligand_SMILES Product_SMILES \\
    --numeric-cols "Temp (K)" \\
    --label-col "ddG" --task-name ECC

# 仅跑 morgan × rf（快速冒烟）：
python run_yonod.py --csv ... --descriptors morgan --models rf

输出（默认到 results/<task-name>/）
-------------------------------------
  metrics_summary.csv   所有 (描述符, 模型) 组合的指标
  run_<timestamp>.log   控制台镜像日志
"""

from __future__ import annotations

import argparse
import datetime as _dt
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

# 仅导入注册表（避免立即导入所有模型/描述符的重依赖）
_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm"]
_MODEL_NAMES = ["xgb", "rf", "svm", "autogluon"]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# 指标 CSV 的列顺序
_CSV_COLUMNS = [
    "task_name", "descriptor", "model",
    "n_smiles_cols", "n_numeric_cols",
    "n_samples", "n_total", "coverage", "feature_dim",
    "cv", "r2_mean", "r2_std", "rmse_mean", "mae_mean",
    "train_time_s", "device",
]


# ─────────────────────────────────────── 日志 ──────────────────────────────── #

class _Tee:
    """同时写入 stdout 和日志文件，每行前缀时间戳。"""

    def __init__(self, fh: TextIO) -> None:
        self.fh = fh
        self._real_stdout = sys.__stdout__
        self._bol = True  # beginning of line

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
    """按名称实例化模型适配器（懒加载避免 torch / autogluon 重依赖）。"""
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
        # 1. 折内归一化数值列（只在训练折上 fit）
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(X_numeric[tr])
        X_num_te = scaler.transform(X_numeric[te])
        X_tr = np.hstack([X_smiles[tr], X_num_tr])
        X_te = np.hstack([X_smiles[te], X_num_te])
        y_tr, y_te = y[tr], y[te]

        # 2. 构建并训练该折的估计器
        if model_name == "rf":
            est = model._build()
            est.fit(X_tr, y_tr)
        elif model_name == "xgb":
            est = model._build()
            est.fit(X_tr, y_tr)
        elif model_name == "svm":
            # SVM Pipeline 内部再做一次 scaler + 可选 PCA，对 X_tr 无害
            # （数值列已经 z-score 过，SVM 内部 scaler 会再处理，但量纲已统一）
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YONOD 通用化入口：CSV → 特征 → 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 必选
    p.add_argument("--csv", type=Path, required=True, help="输入数据集 CSV 路径")
    # 列角色（必须显式指定，不再自动探测）
    p.add_argument("--label-col", required=True, help="标签列名（必填）")
    p.add_argument("--smiles-cols", nargs="+", required=True, help="SMILES 列名，多列空格分隔（必填）")
    p.add_argument("--numeric-cols", nargs="+", default=None, help="数值辅助列名（可选）")
    # 任务控制
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
    # 与 Track A 兼容的参数
    p.add_argument("--append", action="store_true", help="追加写入 metrics_summary.csv")
    p.add_argument("--log-file", type=Path, default=None, help="日志文件路径")
    p.add_argument("--heartbeat", type=float, default=30.0, help="心跳打印间隔（秒，0 禁用）")
    p.add_argument("--svm-subsample", type=int, default=8000, help="SVM 子采样数量")
    # RF 参数
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=None)
    return p.parse_args()


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

def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        print(f"[error] CSV 不存在: {args.csv}", file=sys.stderr)
        return 1

    task_name = args.task_name or args.csv.stem
    out_dir = _resolve_output_dir(args.csv, task_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_path = args.log_file or (
        out_dir / f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log_fh)

    print(f"[init] 任务：{task_name}")
    print(f"[init] CSV：{args.csv}  nrows={args.nrows or 'all'}")
    print(f"[init] 输出目录：{out_dir}")

    # ── 1. 加载数据集 ────────────────────────────────────────────────────────
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

    # ── 2. 描述符 × 模型 grid ─────────────────────────────────────────────────
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
                    # 无数值列：直接调用模型的 cross_validate
                    metrics = model.cross_validate(X_smiles, y, cv=args.cv)
                else:
                    # 有数值列：折内归一化 CV
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
            # 合并模型返回的指标（去掉 oof 预测向量）
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

    # ── 3. 汇总输出 ───────────────────────────────────────────────────────────
    _print_table(rows)

    if rows:
        csv_out = _save_metrics(rows, out_dir, args.append)
        print(f"\n[save] 指标已保存: {csv_out}")

        # ── 4. 生成 HTML 报告 ──────────────────────────────────────────────────
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


if __name__ == "__main__":
    sys.exit(main())
