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

输出（默认到 results/<task-name>建模报告/）
----------------------------------------------
  descriptors/<name>.npz   可复用的描述符矩阵、样本映射与配置元数据
  docs/metrics_summary.csv   所有 (描述符, 模型) 组合的指标
  docs/descriptor_status.csv 描述符生成、复用与失败状态
  docs/run_<timestamp>.log   控制台镜像日志
  pictures/*.png             散点图与训练时间图
  report/report.{html,md}    最终报告
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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
from yonod.universal.descriptor_artifact import (
    descriptor_artifact_path,
    load_descriptor_artifact,
    prepare_descriptor_artifact,
)
from yonod.descriptors.base import split_multi_smiles
from yonod.descriptors.ohe import OHEFeature
from yonod.descriptors.registry import (
    FeatureRegistryError,
    FeatureSpec,
    available_feature_names,
    normalise_feature_specs,
)

# New features are selectable but are intentionally not silently added to an
# existing task when the user accepts the legacy default grid.
_DEFAULT_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm", "maf", "rdkit2d", "drfp"]
_DESCRIPTOR_NAMES = list(available_feature_names())
_MODEL_NAMES = ["xgb", "rf", "svm", "autogluon", "lightgbm"]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

_CSV_COLUMNS = [
    "task_name", "feature_id", "descriptor", "lifecycle", "model",
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
    if model_name == "lightgbm":
        from yonod.models.lightgbm_model import LightGBMYieldModel
        return LightGBMYieldModel()
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
        elif model_name in {"xgb", "lightgbm"}:
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
    'LightGBM': 'lightgbm',
    # 小写版本（向后兼容）
    'xgboost': 'xgb',
    'random forest': 'rf',
    'svm': 'svm',
    'autogluon': 'autogluon',
    'lightgbm': 'lightgbm',
}


def _map_model_name(display_name: str) -> str:
    """将显示名称映射为内部模型标识符。"""
    return _MODEL_NAME_MAP.get(display_name, display_name.lower())


def load_config_from_json(config_path: Path, csv_override: Optional[Path] = None) -> Dict[str, Any]:
    """
    从 JSON 配置文件加载配置。

    Args:
        config_path: 配置文件路径
        csv_override: 可选的CSV数据集路径，用于覆盖默认路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON 格式错误
        ValueError: 配置文件缺少必要字段或数据集不存在
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 验证必要字段（移除 dataset_path 检查）
    required_fields = ['project_name', 'column_roles', 'descriptors', 'models']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"配置文件缺少必要字段: {missing}")

    # 验证 column_roles 结构
    column_roles = config.get('column_roles', {})
    if 'label' not in column_roles or not column_roles['label']:
        raise ValueError("配置文件中 column_roles.label 不能为空")

    # 构造数据集路径
    project_name = config['project_name']
    config_dir = config_path.parent

    if csv_override is not None:
        # 使用用户指定的数据集路径
        dataset_path = csv_override
    else:
        # 使用命名约定: <project_name>_normalized_dataset.csv
        dataset_path = config_dir / f"{project_name}_normalized_dataset.csv"

    # 检查数据集是否存在
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {dataset_path}\n"
            f"请确保数据集文件位于配置文件同级目录下，且文件名为: {project_name}_normalized_dataset.csv\n"
            f"或使用 --csv 参数指定数据集路径"
        )

    # 将数据集路径添加到配置中（供后续使用）
    config['_dataset_path'] = dataset_path

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

    # 使用 load_config_from_json 中设置的数据集路径
    dataset_path = config['_dataset_path']

    # 规范化公开 feature declaration。旧的静态 descriptor JSON 仍可直接
    # 使用；OHE 会被标记为 fold_transform，不能落入全局 artifact 阶段。
    descriptor_configs = config.get('descriptors', [])
    try:
        feature_specs = normalise_feature_specs(
            descriptor_configs,
            label_col=str(column_roles['label']),
        )
    except FeatureRegistryError as exc:
        raise ValueError(f"descriptors 配置无效：{exc}") from exc
    descriptors = [spec.descriptor for spec in feature_specs]

    configured_categoricals = list(column_roles.get('categoricals', []) or [])
    for spec in feature_specs:
        if spec.lifecycle == "fold_transform":
            configured_categoricals.extend(spec.columns)
    categorical_cols = list(dict.fromkeys(configured_categoricals)) or None

    # 解析模型列表（映射显示名称）
    models_raw = config.get('models', [])
    models = [_map_model_name(m) for m in models_raw]

    # 过滤无效模型
    valid_models = {'xgb', 'rf', 'svm', 'autogluon', 'lightgbm'}
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
    # 向导将 JSON 与规范化 CSV 放在 <项目目录>/docs/；运行产物则应回到
    # 项目根目录下的 docs/、pictures/、report/ 三个固定子目录。旧版根级
    # JSON 保持以其所在目录为输出根，避免移动已有运行结果。
    output_dir = config_path.parent.parent if config_path.parent.name == "docs" else config_path.parent

    args = argparse.Namespace(
        csv=dataset_path,
        label_col=column_roles['label'],
        smiles_cols=smiles_cols if smiles_cols else None,
        numeric_cols=column_roles.get('conditions', []) or None,
        categorical_cols=categorical_cols,
        reactant_cols=reactant_cols if reactant_cols else None,
        product_cols=product_cols if product_cols else None,
        other_cols=other_cols if other_cols else None,
        task_name=project_name,
        descriptors=descriptors if descriptors else _DEFAULT_DESCRIPTOR_NAMES,
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
        mfp_radius=3,
        mfp_fp_size=1024,
        ohe_cols=None,
        ohe_missing_policy="as_category",
        dataset_citation=metadata.get('doi'),
        dataset_url=metadata.get('repo_url'),
        dataset_notes=metadata.get('notes'),
        output_format=output_format,
        # 新增：保存原始配置供后续使用
        _config=config,
        _config_path=config_path,
        json=config_path,
        config=config_path,
        # 新增：保留完整的描述符配置列表（包含 columns 和 mode）
        _descriptor_configs=[spec.to_dict() for spec in feature_specs],
        _feature_specs=feature_specs,
    )

    return args


# ────────────────────────── 描述符列解析 ──────────────────────────────────── #

def _resolve_descriptor_columns(
    desc_config: Dict[str, Any],
    all_smiles_cols: List[str],
    smiles_roles: Dict[str, List[str]],
) -> List[str]:
    """
    将描述符配置中的列名解析为规范化 CSV 中的实际列名。

    配置中的列名可能是：
      - 用户声明的基础名称（如 'reactant', 'catalyst'）
      - 规范化后的实际列名（如 'reactant-1', 'catalyst-1'）

    本函数负责将基础名称展开为所有匹配的实际列名。

    Args:
        desc_config: 单个描述符配置字典，包含 'descriptor', 'mode', 'columns' 等
        all_smiles_cols: 规范化 CSV 中的所有 SMILES 列名
        smiles_roles: 列角色映射 {'reactant': [...], 'product': [...], 'other': [...]}

    Returns:
        解析后的实际列名列表
    """
    mode = desc_config.get('mode', 'concat')
    desc_name = desc_config.get('descriptor', '')

    # DRFP 使用 reaction 模式，固定使用 reactant + product
    if mode == 'reaction' or desc_name.lower() == 'drfp':
        # 基础列 = reactant + product
        base_cols = smiles_roles.get('reactant', []) + smiles_roles.get('product', [])
        # 额外加入的 others 列
        extra_reactants = desc_config.get('extra_reactants', [])
        if extra_reactants:
            # 展开 extra_reactants 中的基础名称
            for base_name in extra_reactants:
                matched = _expand_base_name(base_name, all_smiles_cols)
                base_cols.extend(matched)
        return base_cols

    # concat 或 sum 模式：从配置的 columns 中解析
    config_columns = desc_config.get('columns', [])

    if not config_columns:
        # 如果没有指定 columns，使用默认的全部列
        return all_smiles_cols

    # 展开每个配置的列名
    resolved = []
    for col_name in config_columns:
        matched = _expand_base_name(col_name, all_smiles_cols)
        if matched:
            resolved.extend(matched)
        else:
            # 如果没有匹配到，可能是精确列名，直接添加（如果存在）
            if col_name in all_smiles_cols:
                resolved.append(col_name)
            else:
                print(f"[warn] 描述符配置中的列名 '{col_name}' 未在数据集中找到，已跳过", file=sys.stderr)

    return resolved


def _expand_base_name(base_name: str, all_cols: List[str]) -> List[str]:
    """
    将基础名称展开为所有匹配的实际列名。

    例如：
      - 'reactant' -> ['reactant-1', 'reactant-2', ...]
      - 'catalyst' -> ['catalyst'] 或 ['catalyst-1', 'catalyst-2', ...]
      - 'reactant-1' -> ['reactant-1']（精确匹配）

    Args:
        base_name: 基础名称或精确列名
        all_cols: 所有可用的列名

    Returns:
        匹配的列名列表
    """
    matched = []

    # 精确匹配
    if base_name in all_cols:
        return [base_name]

    # 模式匹配：base_name-N
    for col in all_cols:
        if col == base_name:
            matched.append(col)
        elif col.startswith(f"{base_name}-"):
            # 验证后缀是数字
            suffix = col[len(base_name) + 1:]
            if suffix.isdigit():
                matched.append(col)

    # 按数字后缀排序
    def sort_key(c: str) -> int:
        if '-' in c:
            suffix = c.rsplit('-', 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return 0

    matched.sort(key=sort_key)
    return matched


def _get_descriptor_config(
    desc_name: str,
    descriptor_configs: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    根据描述符名称查找对应的配置。

    Args:
        desc_name: 描述符名称
        descriptor_configs: 描述符配置列表

    Returns:
        匹配的配置字典，或 None
    """
    if not descriptor_configs:
        return None

    for cfg in descriptor_configs:
        if cfg.get('descriptor', '').lower() == desc_name.lower():
            return cfg

    return None


def _feature_specs_for_args(args: argparse.Namespace, *, label_col: str) -> list[FeatureSpec]:
    """Return one validated public feature declaration per requested candidate.

    JSON configuration reaches this function through ``_feature_specs``.  CLI
    flags are converted to the same schema so neither MFP parameters nor OHE
    lifecycle rules can bypass configuration/artifact auditing.
    """
    configured = getattr(args, "_feature_specs", None)
    if configured is not None:
        return list(configured)

    declarations: list[dict[str, Any]] = []
    for descriptor in args.descriptors:
        item: dict[str, Any] = {"id": descriptor, "descriptor": descriptor}
        if descriptor == "drfp":
            item["mode"] = "reaction"
        elif descriptor == "maf":
            item["mode"] = "sum"
        else:
            item["mode"] = "concat"
        if descriptor == "mfp":
            item["params"] = {
                "radius": int(args.mfp_radius),
                "fp_size": int(args.mfp_fp_size),
                "profile": "standard",
            }
        elif descriptor == "ohe":
            if not args.ohe_cols:
                raise FeatureRegistryError(
                    "选择 ohe 时必须提供 --ohe-cols COL [COL ...]；"
                    "OHE 不会自动猜测类别列。"
                )
            item["columns"] = list(args.ohe_cols)
            item["params"] = {
                "missing_policy": args.ohe_missing_policy,
                "dtype": "float32",
                "handle_unknown": "ignore",
            }
        declarations.append(item)
    return normalise_feature_specs(declarations, label_col=label_col)


def _resolve_static_feature_columns(
    spec: FeatureSpec,
    all_smiles_cols: List[str],
    smiles_roles: Dict[str, List[str]],
) -> List[str]:
    """Resolve old base-name shorthand while preserving declared order."""
    if spec.lifecycle != "static_descriptor":
        raise ValueError("只有静态特征可以解析 SMILES 列")
    return _resolve_descriptor_columns(spec.to_dict(), all_smiles_cols, smiles_roles)


# ────────────────────────────── argparse ──────────────────────────────────── #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YONOD 通用化入口：CSV → 特征 → 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 配置文件模式（优先级最高）
    p.add_argument("--json", type=Path, default=None,
                   help="配置文件路径（JSON 格式，由 yonod.py 生成）")
    p.add_argument("--config", type=Path, default=None,
                   help="配置文件路径（向后兼容，等同于 --json）")

    # 数据集路径（可选，用于覆盖 JSON 中的默认路径）
    p.add_argument("--csv", type=Path, default=None,
                   help="输入数据集 CSV 路径（可选，覆盖 JSON 中的默认路径）")
    p.add_argument("--label-col", default=None, help="标签列名")
    p.add_argument("--smiles-cols", nargs="+", default=None, help="SMILES 列名（传统模式必填，三分类模式可省略）")
    p.add_argument("--numeric-cols", nargs="+", default=None, help="数值辅助列名（可选）")
    p.add_argument(
        "--categorical-cols", nargs="+", default=None,
        help="显式保留的类别列（供 OHE 使用；可与 SMILES 列重叠）",
    )

    # 新增：三分类模式参数
    p.add_argument("--reactant-cols", nargs="+", default=None, help="反应物 SMILES 列名（三分类模式）")
    p.add_argument("--product-cols", nargs="+", default=None, help="产物 SMILES 列名（三分类模式）")
    p.add_argument("--other-cols", nargs="+", default=None, help="其他参与者 SMILES 列名（三分类模式，可选）")

    p.add_argument("--task-name", default=None, help="任务名称（报告标题，默认 CSV 文件名去后缀）")
    p.add_argument(
        "--descriptors", nargs="+", default=_DEFAULT_DESCRIPTOR_NAMES,
        choices=_DESCRIPTOR_NAMES,
        help="选用特征（mfp 为计数 Morgan；ohe 在每个 CV 训练折拟合）",
    )
    p.add_argument(
        "--models", nargs="+", default=_MODEL_NAMES,
        choices=_MODEL_NAMES, help="选用模型"
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="结果根目录（内含 docs/、pictures/、report/）",
    )
    p.add_argument("--cv", type=int, default=5, help="K-Fold 的 K 值")
    p.add_argument("--nrows", type=int, default=None, help="仅读取前 N 行（调试用）")
    p.add_argument("--append", action="store_true", help="追加写入 docs/metrics_summary.csv")
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="日志文件名（始终写入 <output-dir>/docs/）",
    )
    p.add_argument("--heartbeat", type=float, default=30.0, help="心跳打印间隔（秒，0 禁用）")
    p.add_argument("--svm-subsample", type=int, default=8000, help="SVM 子采样数量")
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--mfp-radius", type=int, default=3, help="MFP 的 Morgan 半径")
    p.add_argument("--mfp-fp-size", type=int, default=1024, help="MFP 每组分的 count 指纹维度")
    p.add_argument(
        "--ohe-cols", nargs="+", default=None,
        help="OHE 的有序类别列；选择 ohe 时必填",
    )
    p.add_argument(
        "--ohe-missing-policy", choices=["as_category", "zero_block", "error"],
        default="as_category", help="OHE 对缺失类别的处理策略",
    )
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
    cols = [c for c in ["feature_id", "descriptor", "lifecycle", "model", "feature_dim", "n_samples",
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


def _save_descriptor_status(rows: List[dict], docs_dir: Path) -> Path:
    """Persist the complete descriptor-stage audit table for this invocation."""
    columns = [
        "feature_id", "descriptor", "lifecycle", "status", "stage", "artifact_path", "n_total",
        "n_valid", "feature_dim", "feature_dim_min", "feature_dim_max", "completed_folds",
        "skipped_model_count", "reason",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    path = docs_dir / "descriptor_status.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _create_output_layout(out_dir: Path) -> tuple[Path, Path, Path]:
    """Create the user-facing output contract without touching old artefacts."""
    docs_dir = out_dir / "docs"
    pictures_dir = out_dir / "pictures"
    report_dir = out_dir / "report"
    for directory in (docs_dir, pictures_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return docs_dir, pictures_dir, report_dir


def _log_path_in_docs(log_file: Optional[Path], docs_dir: Path) -> Path:
    """Keep generated logs in docs/, including when a legacy path is supplied."""
    if log_file is None:
        return docs_dir / f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return docs_dir / log_file.name


# ─────────────────────────────────── main ─────────────────────────────────── #

def main(argv: Optional[List[str]] = None) -> int:
    """CLI pipeline 入口：解析参数 → 加载数据 → 描述符×模型 grid → 输出报告。"""
    args = parse_args(argv)

    # 兼容处理：--config 参数映射到 --json
    if args.config is not None and args.json is None:
        args.json = args.config

    # 处理配置文件模式
    if args.json is not None:
        try:
            # 将 --csv 参数传递给 load_config_from_json
            config = load_config_from_json(args.json, csv_override=args.csv)
            args = config_to_args(config, args.json)
            print(f"[config] 已从配置文件加载: {args._config_path}")
            if config.get('_dataset_path'):
                print(f"[config] 数据集路径: {config['_dataset_path']}")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"[error] 配置文件错误: {e}", file=sys.stderr)
            return 1

    # 验证必要参数（仅在非配置文件模式下检查）
    if args.json is None:
        if args.csv is None:
            print("[error] 请指定 --json 参数（配置文件模式）或 --csv 参数（传统CLI模式）。", file=sys.stderr)
            return 1

        if args.label_col is None:
            print("[error] 请指定 --label-col 参数或使用 --json 配置文件。", file=sys.stderr)
            return 1

    try:
        feature_specs = _feature_specs_for_args(args, label_col=str(args.label_col))
    except FeatureRegistryError as exc:
        print(f"[error] 特征配置错误: {exc}", file=sys.stderr)
        return 1
    args._feature_specs = feature_specs
    if args.ohe_cols:
        args.categorical_cols = list(dict.fromkeys((args.categorical_cols or []) + list(args.ohe_cols)))

    # 检查数据集文件是否存在
    if args.csv is not None and not args.csv.exists():
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
    if any(spec.descriptor == "drfp" for spec in feature_specs):
        if not args.reactant_cols or not args.product_cols:
            print(
                "[error] DRFP 描述符需要列角色分类，请同时指定 --reactant-cols 和 --product-cols 参数。",
                file=sys.stderr
            )
            return 1

    task_name = args.task_name or args.csv.stem
    out_dir = _resolve_output_dir(args.csv, task_name, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir, pictures_dir, report_dir = _create_output_layout(out_dir)

    log_path = _log_path_in_docs(args.log_file, docs_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous_stdout = sys.stdout
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log_fh)

    print(f"[init] 任务：{task_name}")
    print(f"[init] CSV：{args.csv}  nrows={args.nrows or 'all'}")
    print(f"[init] 输出目录：{out_dir}")
    print(f"[init] 文档：{docs_dir}  图片：{pictures_dir}  报告：{report_dir}")
    if args.log_file is not None and args.log_file != Path(args.log_file.name):
        print(f"[init] 已将 --log-file 规范为 docs 内文件名：{log_path.name}")

    dataset = load_csv_with_roles(
        csv_path=args.csv,
        smiles_cols=args.smiles_cols,
        numeric_cols=args.numeric_cols or [],
        categorical_cols=args.categorical_cols or [],
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
    print(
        "[grid] 特征={0}  模型={1}  cv={2}".format(
            ["{0}:{1}".format(spec.id, spec.lifecycle) for spec in feature_specs],
            args.models, args.cv,
        )
    )

    svm_sub = args.svm_subsample if args.svm_subsample > 0 else None

    rows: List[dict] = []
    descriptor_statuses: List[dict] = []
    descriptors_dir = out_dir / "descriptors"
    descriptors_dir.mkdir(parents=True, exist_ok=True)
    total = len(feature_specs) * len(args.models)
    done = 0
    sample_ids = [str(value) for value in df.index]
    sample_id_array = np.asarray(sample_ids, dtype=str)
    prepared_features: Dict[str, Dict[str, Any]] = {}
    status_by_id: Dict[str, Dict[str, Any]] = {}

    print("\n[phase 1/2] 预计算静态特征；按折特征将在建模阶段拟合")
    for spec in feature_specs:
        if spec.lifecycle == "fold_transform":
            missing = [column for column in spec.columns if column not in df.columns]
            state_root = out_dir / "fold_transformers" / spec.id
            status = {
                "feature_id": spec.id,
                "descriptor": spec.descriptor,
                "lifecycle": spec.lifecycle,
                "status": "deferred" if not missing else "failed",
                "stage": "fold_transform" if not missing else "validate",
                "artifact_path": str(state_root),
                "n_total": len(df), "n_valid": len(df), "feature_dim": None,
                "feature_dim_min": None, "feature_dim_max": None, "completed_folds": 0,
                "skipped_model_count": len(args.models) if missing else 0,
                "reason": (
                    "每个 CV 训练折单独 fit；不会生成全数据 OHE artifact"
                    if not missing else "OHE 类别列不存在：{0}".format(", ".join(missing))
                ),
            }
            descriptor_statuses.append(status)
            status_by_id[spec.id] = status
            if not missing:
                prepared_features[spec.id] = {"spec": spec, "state_root": state_root}
                print("[feature:deferred] {0}: OHE 将在每个训练折拟合，列={1}".format(
                    spec.id, ", ".join(spec.columns)
                ))
            else:
                print("[feature:failed] {0}: {1}".format(spec.id, status["reason"]), file=sys.stderr)
            continue

        print("\n[feature] 准备静态特征: {0} ({1}) ...".format(spec.id, spec.descriptor))
        desc_smiles_cols = _resolve_static_feature_columns(spec, smiles_cols, smiles_roles)
        desc_smiles_roles = {
            'reactant': [c for c in desc_smiles_cols if c in smiles_roles.get('reactant', [])],
            'product': [c for c in desc_smiles_cols if c in smiles_roles.get('product', [])],
            'other': [c for c in desc_smiles_cols if c in smiles_roles.get('other', [])],
        }
        feature_config = {"params": dict(spec.params), "feature_spec": spec.to_dict()}
        artifact_path = descriptor_artifact_path(descriptors_dir, spec.id)
        try:
            preparation = prepare_descriptor_artifact(
                descriptors_dir,
                descriptor=spec.descriptor,
                feature_id=spec.id,
                smiles_cols=list(desc_smiles_cols),
                df=df,
                smiles_roles=desc_smiles_roles,
                mode=spec.mode,
                descriptor_config=feature_config,
                sample_ids=sample_ids,
                sample_id_name="dataframe_index_after_label_filter",
                compute=build_universal_features,
            )
            prepared_features[spec.id] = {
                "spec": spec, "preparation": preparation,
                "smiles_cols": list(desc_smiles_cols), "mode": spec.mode,
            }
            status = {
                "feature_id": spec.id, "descriptor": spec.descriptor, "lifecycle": spec.lifecycle,
                "status": preparation.status, "stage": "precompute",
                "artifact_path": str(preparation.path), "n_total": preparation.n_total,
                "n_valid": preparation.n_valid, "feature_dim": preparation.feature_dim,
                "feature_dim_min": preparation.feature_dim, "feature_dim_max": preparation.feature_dim,
                "completed_folds": None, "skipped_model_count": 0, "reason": preparation.reason,
            }
            descriptor_statuses.append(status)
            status_by_id[spec.id] = status
            print("[feature:{0}] {1}: n_valid={2}/{3}  feature_dim={4}  file={5}".format(
                preparation.status, spec.id, preparation.n_valid, preparation.n_total,
                preparation.feature_dim, preparation.path,
            ))
        except Exception as exc:
            status = {
                "feature_id": spec.id, "descriptor": spec.descriptor, "lifecycle": spec.lifecycle,
                "status": "failed", "stage": "precompute", "artifact_path": str(artifact_path),
                "n_total": len(df), "n_valid": None, "feature_dim": None,
                "feature_dim_min": None, "feature_dim_max": None, "completed_folds": 0,
                "skipped_model_count": len(args.models), "reason": f"{type(exc).__name__}: {exc}",
            }
            descriptor_statuses.append(status)
            status_by_id[spec.id] = status
            print("[feature:failed] {0}: {1}".format(spec.id, status["reason"]), file=sys.stderr)

    status_path = _save_descriptor_status(descriptor_statuses, docs_dir)
    print(f"\n[save] 描述符状态已保存: {status_path}")
    print("\n[phase 2/2] 从 descriptors/ 文件读取特征并开始建模")

    for spec in feature_specs:
        context = prepared_features.get(spec.id)
        if context is None:
            done += len(args.models)
            print("[skip] {0}: 特征阶段失败，跳过 {1} 个模型".format(spec.id, len(args.models)))
            continue
        if spec.lifecycle == "static_descriptor":
            preparation = context["preparation"]
            desc_smiles_cols = context["smiles_cols"]
            try:
                artifact = load_descriptor_artifact(
                    preparation.path,
                    expected_descriptor_config=preparation.descriptor_config,
                    expected_dataset_fingerprint=preparation.dataset_fingerprint,
                    expected_sample_ids=np.asarray(sample_ids, dtype=np.str_),
                )
            except Exception as exc:
                status = status_by_id[spec.id]
                status.update({"status": "failed", "stage": "load", "skipped_model_count": len(args.models),
                               "reason": f"{type(exc).__name__}: {exc}"})
                done += len(args.models)
                print("[feature:failed] {0} 建模前读取失败: {1}".format(spec.id, status["reason"]), file=sys.stderr)
                continue
            X_smiles = artifact.X_smiles
            mask = artifact.valid_mask
            X_numeric = df[numeric_cols].to_numpy(dtype=np.float32)[mask] if numeric_cols else None
            y = df[label_col].to_numpy(dtype=np.float64)[mask]
            input_columns = desc_smiles_cols
            print("[feature:loaded] {0}: n_valid={1}  X.shape={2}".format(
                spec.id, int(mask.sum()), X_smiles.shape
            ))
        else:
            X_smiles = None
            X_numeric = None
            mask = np.ones(len(df), dtype=bool)
            y = df[label_col].to_numpy(dtype=np.float64)
            input_columns = list(spec.columns)

        for model_name in args.models:
            done += 1
            label = f"{spec.id} x {model_name} ({done}/{total})"
            print(f"[eval] 开始: {label}")

            hb = _Heartbeat(interval=args.heartbeat, label=label)
            if args.heartbeat > 0:
                hb.start()

            try:
                if spec.lifecycle == "fold_transform":
                    metrics = _cv_with_ohe_feature(
                        spec=spec, frame=df, sample_ids=sample_id_array, numeric_cols=numeric_cols,
                        y=y, model_name=model_name, args=args,
                        state_root=context["state_root"], svm_subsample=svm_sub,
                    )
                else:
                    model = _make_model(model_name, args)
                    if X_numeric is None:
                        metrics = model.cross_validate(X_smiles, y, cv=args.cv)
                    else:
                        metrics = _cv_with_numeric(
                            model_name, model, X_smiles, X_numeric, y,
                            cv=args.cv, svm_subsample=svm_sub,
                        )
            except Exception as exc:
                print(f"[error] {label} 失败: {exc}", file=sys.stderr)
                hb.stop()
                continue
            finally:
                hb.stop()

            row: dict = {
                "task_name":     task_name,
                "feature_id":    spec.id,
                "descriptor":    spec.descriptor,
                "lifecycle":     spec.lifecycle,
                "model":         model_name,
                "n_smiles_cols": len(input_columns),
                "n_numeric_cols":len(numeric_cols),
                "n_samples":     int(mask.sum()),
                "n_total":       int(len(mask)),
                "coverage":      float(mask.sum() / max(len(mask), 1)),
                "feature_dim":   (
                    int(X_smiles.shape[1]) + (X_numeric.shape[1] if X_numeric is not None else 0)
                    if X_smiles is not None else int(metrics["fold_transform_summary"]["output_dim_max"]) + len(numeric_cols)
                ),
                "cv":            args.cv,
            }
            oof_pred   = metrics.get("oof_pred")
            oof_y_true = metrics.get("oof_y_true")
            for k, v in metrics.items():
                if k not in ("oof_pred", "oof_y_true"):
                    row[k] = v

            rows.append(row)
            if spec.lifecycle == "fold_transform":
                summary = metrics["fold_transform_summary"]
                status_by_id[spec.id].update({
                    "status": "completed", "stage": "fold_transform",
                    "feature_dim": summary["output_dim_max"],
                    "feature_dim_min": summary["output_dim_min"],
                    "feature_dim_max": summary["output_dim_max"],
                    "completed_folds": summary["completed_folds"],
                    "reason": "每个 CV 训练折独立拟合；状态已保存并完成加载校验",
                })
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
                            spec.id, model_name,
                            pictures_dir,
                        )
                except Exception as _exc:
                    print(f"[warn] 散点图生成失败 ({label}): {_exc}", file=sys.stderr)

    _print_table(rows)
    status_path = _save_descriptor_status(descriptor_statuses, docs_dir)
    if rows:
        csv_out = _save_metrics(rows, docs_dir, args.append)
        print(f"\n[save] 指标已保存: {csv_out}")
    else:
        print("\n[warn] 无有效结果，docs/metrics_summary.csv 未写入", file=sys.stderr)

    task_info = {
        "task_name":        task_name,
        "csv_path":         args.csv,
        "project_folder":   str(out_dir),
        "n_samples":        rows[0].get("n_samples", "—") if rows else len(df),
        "smiles_cols":      smiles_cols,
        "numeric_cols":     numeric_cols or "（无）",
        "label_col":        label_col,
        "n_combinations":   len(rows),
        "dataset_citation": args.dataset_citation,
        "dataset_url":      args.dataset_url,
        "dataset_notes":    args.dataset_notes,
        "descriptors":      [spec.to_dict() for spec in feature_specs],
        "descriptor_statuses": descriptor_statuses,
        "descriptor_status_path": str(status_path),
    }
    if hasattr(args, '_config') and args._config:
        task_info["column_mapping"] = args._config.get("column_mapping", [])
        task_info["origin_dataset_path"] = args._config.get("origin_dataset_path")
        task_info["dataset_path"] = str(args.csv)
    if hasattr(args, '_config_path') and args._config_path:
        task_info["config_path"] = str(args._config_path)
    metrics_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_CSV_COLUMNS)
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

    print(f"[done] 全部完成。日志: {log_path}")
    sys.stdout.flush()
    sys.stdout = previous_stdout
    log_fh.close()
    return 0


def _stable_hash(value: Any) -> str:
    """Hash an audit payload without relying on process-local object IDs."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_predict_explicit_fold(
    model_name: str,
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    fold_index: int,
    svm_subsample: Optional[int],
) -> tuple[np.ndarray, str]:
    """Build a fresh supported estimator for an externally defined CV fold."""
    if model_name == "autogluon":
        raise ValueError(
            "OHE 需要显式的训练折/验证折边界；AutoGluon 当前仅提供内部 holdout，"
            "因此不能用于 OHE 的无泄漏 CV。请选择 rf、xgb、svm 或 lightgbm。"
        )
    model = _make_model(model_name, args)
    if model_name in {"rf", "xgb", "lightgbm"}:
        estimator = model._build()
        estimator.fit(X_train, y_train)
    elif model_name == "svm":
        train_index = np.arange(len(X_train))
        if svm_subsample and svm_subsample < len(train_index):
            rng = np.random.default_rng(seed=fold_index)
            train_index = rng.choice(train_index, size=svm_subsample, replace=False)
        estimator = model._build(n_features=X_train.shape[1], n_train=len(train_index))
        estimator.fit(X_train[train_index], y_train[train_index])
    else:  # Keep this exhaustive if a future CLI model is added.
        raise ValueError(f"OHE 显式 CV 尚未支持模型 {model_name!r}")
    return np.asarray(estimator.predict(X_valid), dtype=np.float64), str(getattr(model, "device", "cpu"))


def _cv_with_ohe_feature(
    *,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    sample_ids: np.ndarray,
    numeric_cols: List[str],
    y: np.ndarray,
    model_name: str,
    args: argparse.Namespace,
    state_root: Path,
    svm_subsample: Optional[int],
) -> Dict[str, Any]:
    """Run ordinary KFold CV with a strictly train-fold-local OHE encoder.

    This intentionally does not call a model adapter's convenience
    ``cross_validate`` method: those methods own their own split loop and
    would make it impossible to prove which samples taught the OHE vocabulary.
    """
    if spec.lifecycle != "fold_transform" or spec.descriptor != "ohe":
        raise ValueError("_cv_with_ohe_feature 仅接受 OHE fold_transform")
    if len(y) < args.cv:
        raise ValueError("样本数必须不少于 cv 折数")
    component_frame = frame.loc[:, list(spec.columns)]
    numeric = frame.loc[:, numeric_cols].to_numpy(dtype=np.float32) if numeric_cols else None
    if numeric is not None and not np.isfinite(numeric).all():
        raise ValueError("数值辅助列包含 NaN 或 inf，无法在 OHE CV 中安全缩放")

    state_root.mkdir(parents=True, exist_ok=True)
    dataset_hash = _stable_hash({
        "sample_ids": sample_ids.tolist(),
        "columns": list(spec.columns),
        "records": component_frame.where(pd.notna(component_frame), None).to_dict(orient="records"),
    })
    splitter = KFold(n_splits=args.cv, shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    r2s: List[float] = []
    rmses: List[float] = []
    maes: List[float] = []
    audits: List[Dict[str, Any]] = []
    started = time.time()
    device = "cpu"

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(component_frame), start=1):
        transformer = OHEFeature(
            spec.columns,
            missing_policy=str(spec.params["missing_policy"]),
            dtype=str(spec.params["dtype"]),
            handle_unknown=str(spec.params["handle_unknown"]),
        )
        transformer.fit(component_frame.iloc[train_idx], sample_ids=sample_ids[train_idx])
        X_train = transformer.transform(component_frame.iloc[train_idx], partition="train")
        X_valid = transformer.transform(component_frame.iloc[valid_idx], partition="valid")
        if numeric is not None:
            scaler = StandardScaler()
            X_train = np.hstack([X_train, scaler.fit_transform(numeric[train_idx])])
            X_valid = np.hstack([X_valid, scaler.transform(numeric[valid_idx])])

        metadata = transformer.metadata()
        context = {
            "feature_id": spec.id,
            "feature_spec_hash": spec.schema_hash,
            "dataset_hash": dataset_hash,
            "split_hash": _stable_hash({
                "cv": int(args.cv), "seed": 42, "repeat": 1, "fold": fold_index,
                "train_sample_ids": sample_ids[train_idx].tolist(),
                "valid_sample_ids": sample_ids[valid_idx].tolist(),
            }),
            "repeat": 1,
            "fold": fold_index,
            "train_sample_ids_hash": metadata["train_sample_ids_hash"],
        }
        fold_dir = state_root / "repeat-01" / f"fold-{fold_index:02d}"
        transformer.save(fold_dir, context=context)
        # Read the freshly persisted state back immediately. This checks the
        # atomic state/metadata contract without refitting on validation rows.
        OHEFeature.load(fold_dir, expected_context=context)

        prediction, device = _fit_predict_explicit_fold(
            model_name, args, X_train, y[train_idx], X_valid,
            fold_index=fold_index, svm_subsample=svm_subsample,
        )
        if not np.isfinite(prediction).all():
            raise ValueError(f"OHE 第 {fold_index} 折模型预测包含 NaN 或 inf")
        oof[valid_idx] = prediction
        r2s.append(float(r2_score(y[valid_idx], prediction)))
        rmses.append(float(np.sqrt(mean_squared_error(y[valid_idx], prediction))))
        maes.append(float(mean_absolute_error(y[valid_idx], prediction)))
        audits.append({
            "fold": fold_index,
            "state_directory": str(fold_dir),
            "output_dim": int(metadata["output_dim"]),
            "categories_hash": metadata["categories_hash"],
            "train_sample_ids_hash": metadata["train_sample_ids_hash"],
            "valid_unseen_count": int(metadata["valid_unseen_count"]),
            "valid_missing_count": int(metadata["valid_missing_count"]),
        })

    summary = {
        "feature_id": spec.id,
        "descriptor": spec.descriptor,
        "lifecycle": spec.lifecycle,
        "feature_spec_hash": spec.schema_hash,
        "dataset_hash": dataset_hash,
        "fit_scope": "train_only_per_fold",
        "completed_folds": len(audits),
        "output_dim_min": min(item["output_dim"] for item in audits),
        "output_dim_max": max(item["output_dim"] for item in audits),
        "folds": audits,
    }
    (state_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "train_time_s": float(time.time() - started),
        "device": device,
        "oof_pred": oof,
        "oof_y_true": y,
        "fold_transform_summary": summary,
    }


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
    print("    python main.py --json path/to/project_yonod_config.json")
    print()
    print("    可选：覆盖默认数据集路径")
    print("    python main.py --json config.json --csv custom_dataset.csv")
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
