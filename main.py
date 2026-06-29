"""YONOD 通用入口（向导 + CLI 合并版）。

双击或不带参数运行时启动交互向导；带 --csv 等参数直接进入 CLI pipeline。

交互向导用法
------------
    python main.py

CLI 用法示例
------------
    # 显式指定标签列和 SMILES 列：
    python main.py --csv 数据集/酰胺缩合数据集.csv --label-col yield --smiles-cols Reactant1 Reactant2

    # ECC 数据集（2 SMILES 列 + 温度辅助列）：
    python main.py --csv 数据集/镍催化偶联数据集/Raw_Dataset.csv \\
        --smiles-cols Ligand_SMILES Product_SMILES \\
        --numeric-cols "Temp (K)" \\
        --label-col "ddG" --task-name ECC

    # 仅跑 morgan × rf（快速冒烟）：
    python main.py --csv ... --smiles-cols ... --label-col ... --descriptors morgan --models rf

输出（默认到 results/<task-name>/）
-------------------------------------
  metrics_summary.csv   所有 (描述符, 模型) 组合的指标
  run_<timestamp>.log   控制台镜像日志
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

from yonod.universal.csv_loader import load_csv_with_roles
from yonod.universal.feature_builder import build_universal_features
from yonod.descriptors.base import split_multi_smiles

_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm", "maf", "rdkit2d", "drfp"]
_MODEL_NAMES = ["xgb", "rf", "svm", "autogluon"]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

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
        self._real_stdout: TextIO = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
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
        from yonod.models.xgb_model import XGBYieldModel
        return XGBYieldModel()
    if model_name == "rf":
        from yonod.models.rf_model import RFYieldModel
        return RFYieldModel(
            n_jobs=args.rf_n_jobs,
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
        )
    if model_name == "svm":
        from yonod.models.svm_model import SVMYieldModel
        sub = args.svm_subsample if args.svm_subsample > 0 else None
        return SVMYieldModel(subsample_n=sub)
    if model_name == "autogluon":
        from yonod.models.autogluon_model import AutoGluonYieldModel
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


# ────────────────────────── 配置文件读取 ──────────────────────────────────── #

_MODEL_NAME_MAP = {
    'XGBoost': 'xgb',
    'Random Forest': 'rf',
    'SVM': 'svm',
    'AutoGluon': 'autogluon',
    'Neural Network': 'nn',
    # 小写版本（向后兼容）
    'xgboost': 'xgb',
    'random forest': 'rf',
    'svm': 'svm',
    'autogluon': 'autogluon',
    'neural network': 'nn',
}


def _map_model_name(display_name: str) -> str:
    """将显示名称映射为内部模型标识符。"""
    return _MODEL_NAME_MAP.get(display_name, display_name.lower())


def load_config_from_json(config_path: Path) -> Dict[str, Any]:
    """
    从 JSON 配置文件加载配置。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON 格式错误
        ValueError: 配置文件缺少必要字段
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 验证必要字段
    required_fields = ['dataset_path', 'column_roles', 'descriptors', 'models']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"配置文件缺少必要字段: {missing}")

    # 验证 column_roles 结构
    column_roles = config.get('column_roles', {})
    if 'label' not in column_roles or not column_roles['label']:
        raise ValueError("配置文件中 column_roles.label 不能为空")

    return config


def config_to_args(config: Dict[str, Any], config_path: Path) -> argparse.Namespace:
    """
    将配置字典转换为 argparse.Namespace 对象。

    Args:
        config: 从 JSON 加载的配置字典
        config_path: 配置文件路径（用于解析相对路径）

    Returns:
        argparse.Namespace 对象，与 parse_args() 返回值兼容
    """
    column_roles = config['column_roles']

    # 构建 SMILES 列列表
    smiles_cols = []
    reactant_cols = column_roles.get('reactants', [])
    product_cols = column_roles.get('products', [])
    other_cols = column_roles.get('others', [])

    # 按角色顺序拼接
    smiles_cols = reactant_cols + product_cols + other_cols

    # 解析数据集路径（支持相对路径）
    dataset_path_str = config['dataset_path']
    dataset_path = Path(dataset_path_str)
    if not dataset_path.is_absolute():
        dataset_path = config_path.parent / dataset_path

    # 解析描述符配置
    descriptor_configs = config.get('descriptors', [])
    descriptors = [cfg['descriptor'] for cfg in descriptor_configs]

    # 解析模型列表（映射显示名称）
    models_raw = config.get('models', [])
    models = [_map_model_name(m) for m in models_raw]

    # 过滤无效模型
    valid_models = {'xgb', 'rf', 'svm', 'autogluon'}
    models = [m for m in models if m in valid_models]

    # 解析元信息
    metadata = config.get('metadata', {})

    # 解析输出格式
    report_formats = config.get('report_formats', ['HTML', 'Markdown'])
    output_format = 'both'
    if 'HTML' in report_formats and 'Markdown' not in report_formats:
        output_format = 'html'
    elif 'Markdown' in report_formats and 'HTML' not in report_formats:
        output_format = 'md'

    # 解析输出目录
    project_name = config.get('project_name', dataset_path.stem)
    output_dir = config_path.parent  # 默认输出到配置文件所在目录

    args = argparse.Namespace(
        csv=dataset_path,
        label_col=column_roles['label'],
        smiles_cols=smiles_cols if smiles_cols else None,
        numeric_cols=column_roles.get('conditions', []) or None,
        reactant_cols=reactant_cols if reactant_cols else None,
        product_cols=product_cols if product_cols else None,
        other_cols=other_cols if other_cols else None,
        task_name=project_name,
        descriptors=descriptors if descriptors else _DESCRIPTOR_NAMES,
        models=models if models else _MODEL_NAMES,
        output_dir=output_dir,
        cv=5,
        nrows=None,
        append=False,
        log_file=None,
        heartbeat=30.0,
        svm_subsample=8000,
        rf_n_jobs=-1,
        rf_n_estimators=300,
        rf_max_depth=None,
        dataset_citation=metadata.get('doi'),
        dataset_url=metadata.get('repo_url'),
        dataset_notes=metadata.get('notes'),
        output_format=output_format,
        # 新增：保存原始配置供后续使用
        _config=config,
        _config_path=config_path,
    )

    return args


# ────────────────────────────── argparse ──────────────────────────────────── #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YONOD 通用化入口：CSV → 特征 → 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 配置文件模式（优先级最高）
    p.add_argument("--config", type=Path, default=None,
                   help="配置文件路径（JSON 格式，由 yonod.py 生成）")

    # 传统 CLI 参数（当不使用 --config 时必填）
    p.add_argument("--csv", type=Path, default=None, help="输入数据集 CSV 路径")
    p.add_argument("--label-col", default=None, help="标签列名")
    p.add_argument("--smiles-cols", nargs="+", default=None, help="SMILES 列名（传统模式必填，三分类模式可省略）")
    p.add_argument("--numeric-cols", nargs="+", default=None, help="数值辅助列名（可选）")

    # 新增：三分类模式参数
    p.add_argument("--reactant-cols", nargs="+", default=None, help="反应物 SMILES 列名（三分类模式）")
    p.add_argument("--product-cols", nargs="+", default=None, help="产物 SMILES 列名（三分类模式）")
    p.add_argument("--other-cols", nargs="+", default=None, help="其他参与者 SMILES 列名（三分类模式，可选）")

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
    p.add_argument("--dataset-citation", default=None, metavar="TEXT",
                   help="数据集文献来源（可选，显示在报告中）")
    p.add_argument("--dataset-url", default=None, metavar="URL",
                   help="数据集开源地址（可选，显示在报告中）")
    p.add_argument("--dataset-notes", default=None, metavar="TEXT",
                   help="数据集备注（可选，显示在报告中）")
    p.add_argument(
        "--output-format", choices=["html", "md", "both"], default="both",
        help="报告输出格式：html / md / both（默认 both，同时生成 HTML 和 Markdown）",
    )
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

    # 处理配置文件模式
    if args.config is not None:
        try:
            config = load_config_from_json(args.config)
            args = config_to_args(config, args.config)
            print(f"[config] 已从配置文件加载: {args._config_path}")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"[error] 配置文件错误: {e}", file=sys.stderr)
            return 1

    # 验证必要参数
    if args.csv is None:
        print("[error] 请指定 --csv 参数或使用 --config 配置文件。", file=sys.stderr)
        return 1

    if args.label_col is None:
        print("[error] 请指定 --label-col 参数或使用 --config 配置文件。", file=sys.stderr)
        return 1

    if not args.csv.exists():
        print(f"[error] CSV 不存在: {args.csv}", file=sys.stderr)
        return 1

    # 验证 SMILES 列参数：必须指定 --smiles-cols 或三分类参数
    if not args.smiles_cols and not (args.reactant_cols or args.product_cols):
        print(
            "[error] 请指定 --smiles-cols 参数，或使用三分类模式（--reactant-cols + --product-cols）。",
            file=sys.stderr
        )
        return 1

    # 验证 DRFP 描述符与列角色分类的依赖关系
    if "drfp" in args.descriptors:
        if not args.reactant_cols or not args.product_cols:
            print(
                "[error] DRFP 描述符需要列角色分类，请同时指定 --reactant-cols 和 --product-cols 参数。",
                file=sys.stderr
            )
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
        reactant_cols=args.reactant_cols,
        product_cols=args.product_cols,
        other_cols=args.other_cols,
        nrows=args.nrows,
    )
    df = dataset.df
    smiles_cols = dataset.smiles_cols
    numeric_cols = dataset.numeric_cols
    label_col = dataset.label_col
    smiles_roles = dataset.smiles_roles

    # 打印角色信息
    if smiles_roles['reactant'] or smiles_roles['product']:
        print("\n[INFO] SMILES 列角色分类：")
        print(f"  反应物列: {', '.join(smiles_roles['reactant'])}")
        print(f"  产物列: {', '.join(smiles_roles['product'])}")
        if smiles_roles['other']:
            print(f"  其他参与者列: {', '.join(smiles_roles['other'])}")
    else:
        print("\n[INFO] 使用传统模式（所有 SMILES 列等价）")

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

        # 根据描述符类型显示使用的列
        if desc_name.lower() == "drfp":
            cols_used = smiles_roles['reactant'] + smiles_roles['product']
            print(f"  使用列（反应物+产物）: {', '.join(cols_used)}")
        else:
            print(f"  使用列（全部）: {', '.join(smiles_cols)}")

        X_smiles, X_numeric, mask = build_universal_features(
            smiles_cols=smiles_cols,
            numeric_cols=numeric_cols,
            df=df,
            desc_name=desc_name,
            smiles_roles=smiles_roles,
        )
        y = df[label_col].to_numpy(dtype=np.float64)[mask]

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
            oof_pred   = metrics.get("oof_pred")
            oof_y_true = metrics.get("oof_y_true")
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

            if oof_pred is not None and oof_y_true is not None:
                try:
                    from yonod.plot import plot_scatter
                    valid = ~np.isnan(oof_pred)
                    if valid.sum() >= 2:
                        plot_scatter(
                            oof_y_true[valid], oof_pred[valid],
                            desc_name, model_name,
                            out_dir / "pictures",
                        )
                except Exception as _exc:
                    print(f"[warn] 散点图生成失败 ({label}): {_exc}", file=sys.stderr)

    _print_table(rows)

    if rows:
        csv_out = _save_metrics(rows, out_dir, args.append)
        print(f"\n[save] 指标已保存: {csv_out}")

        task_info = {
            "task_name":        task_name,
            "csv_path":         args.csv,
            "n_samples":        rows[0].get("n_samples", "—") if rows else "—",
            "smiles_cols":      smiles_cols,
            "numeric_cols":     numeric_cols or "（无）",
            "label_col":        label_col,
            "n_combinations":   len(rows),
            "dataset_citation": args.dataset_citation,
            "dataset_url":      args.dataset_url,
            "dataset_notes":    args.dataset_notes,
        }
        metrics_df = pd.DataFrame(rows)
        fmt = args.output_format

        if fmt in ("html", "both"):
            try:
                from yonod.universal.report import generate_report
                report_path = generate_report(metrics_df, task_info, out_dir)
                print(f"[report] HTML 报告已生成: {report_path}")
            except Exception as exc:
                print(f"[warn] HTML 报告生成失败（不影响指标 CSV）: {exc}", file=sys.stderr)

        if fmt in ("md", "both"):
            try:
                from yonod.universal.report import generate_markdown_report
                md_path = generate_markdown_report(metrics_df, task_info, out_dir)
                print(f"[report] Markdown 报告已生成: {md_path}")
            except Exception as exc:
                print(f"[warn] Markdown 报告生成失败（不影响指标 CSV）: {exc}", file=sys.stderr)
    else:
        print("\n[warn] 无有效结果，metrics_summary.csv 未写入", file=sys.stderr)

    print(f"[done] 全部完成。日志: {log_path}")
    return 0


# ─────────────────────────────── 入口 ────────────────────────────────────── #

def _print_usage() -> None:
    """无参数运行时打印友好的使用提示。"""
    print()
    print("=" * 60)
    print("  YONOD - You Only Need Outstanding Descriptors")
    print("  有机合成反应产率预测平台")
    print("=" * 60)
    print()
    print("使用方法:")
    print()
    print("  场景A: 完整向导流程（首次使用推荐）")
    print("    python yonod.py")
    print()
    print("  场景B: 使用配置文件建模（向导生成的配置）")
    print("    python main.py --config path/to/yonod_config.json")
    print()
    print("  场景C: 传统CLI模式（直接指定参数）")
    print("    python main.py --csv data.csv --label-col yield \\")
    print("        --smiles-cols R1 R2 --descriptors morgan --models xgb")
    print()
    print("更多帮助:")
    print("    python main.py --help")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        _print_usage()
        sys.exit(0)
