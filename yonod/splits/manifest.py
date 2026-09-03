"""分组重复交叉验证的 split manifest 生成、审计与落盘。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..benchmark.config import BenchmarkConfigError, BenchmarkContract
from ..benchmark.layout import resolve_benchmark_output_layout
from .grouping import build_group_ids


class SplitManifestError(BenchmarkConfigError):
    """Raised when a group split is infeasible or fails leakage checks."""


MANIFEST_COLUMNS = [
    "run_id", "split_id", "sample_id", "group_id", "group_strategy",
    "repeat", "fold", "role", "seed", "dataset_sha256", "grouping_params_json",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assign_groups_to_folds(group_sizes: pd.Series, n_splits: int, seed: int) -> Dict[str, int]:
    """Greedily balance complete groups while retaining deterministic seeded repeats."""
    rng = np.random.default_rng(seed)
    order = pd.DataFrame({"group_id": group_sizes.index.astype(str), "size": group_sizes.values})
    order["tie"] = rng.random(len(order))
    order = order.sort_values(["size", "tie", "group_id"], ascending=[False, True, True], kind="mergesort")
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    assignments: Dict[str, int] = {}
    for _, row in order.iterrows():
        fold = int(np.argmin(fold_sizes))
        assignments[str(row["group_id"])] = fold + 1
        fold_sizes[fold] += int(row["size"])
    return assignments


def create_split_manifest(contract: BenchmarkContract) -> pd.DataFrame:
    """Create the sole reusable split plan for every candidate in one run."""
    config = contract.config
    frame = config.validate_dataset().copy()
    group_ids = build_group_ids(frame, config.sample_id_col, config.grouping, config.smiles_cols)
    sizes = group_ids.value_counts(sort=False)
    n_samples = len(frame)
    n_splits = int(config.cv["n_splits"])
    n_repeats = int(config.cv["n_repeats"])
    seed_start = int(config.cv["seed"])
    if len(sizes) < n_splits:
        raise SplitManifestError(
            "分组数量不足：只有 {0} 个 group，但 n_splits={1}。请降低折数或调整分组参数。"
            .format(len(sizes), n_splits)
        )
    max_fraction = float(config.grouping.get("max_group_fraction", 0.8))
    largest = int(sizes.max())
    if not 0 < max_fraction <= 1:
        raise SplitManifestError("grouping.max_group_fraction 必须处于 (0, 1]")
    if largest / max(n_samples, 1) > max_fraction:
        raise SplitManifestError(
            "最大 group 有 {0}/{1} 个样本（{2:.1%}），超过 max_group_fraction={3:.1%}；"
            "请调整分组阈值、选择其他策略或降低分组粒度。"
            .format(largest, n_samples, largest / max(n_samples, 1), max_fraction)
        )

    grouping_params_json = _canonical_json(config.grouping)
    split_payload = {
        "run_id": contract.run_id,
        "dataset_sha256": contract.dataset_sha256,
        "grouping": config.grouping,
        "cv": config.cv,
        "sample_group_pairs": sorted(
            zip(frame[config.sample_id_col].astype(str).tolist(), group_ids.astype(str).tolist())
        ),
    }
    split_id = "split-" + hashlib.sha256(_canonical_json(split_payload).encode("utf-8")).hexdigest()[:12]
    records: List[Dict[str, Any]] = []
    sample_ids = frame[config.sample_id_col].astype(str)
    strategy = str(config.grouping["strategy"])
    for repeat in range(1, n_repeats + 1):
        seed = seed_start + repeat - 1
        assignments = _assign_groups_to_folds(sizes, n_splits, seed)
        fold_for_row = group_ids.astype(str).map(assignments)
        for fold in range(1, n_splits + 1):
            valid_mask = fold_for_row.eq(fold)
            for sample_id, group_id, is_valid in zip(sample_ids, group_ids, valid_mask):
                records.append({
                    "run_id": contract.run_id,
                    "split_id": split_id,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "group_strategy": strategy,
                    "repeat": repeat,
                    "fold": fold,
                    "role": "valid" if bool(is_valid) else "train",
                    "seed": seed,
                    "dataset_sha256": contract.dataset_sha256,
                    "grouping_params_json": grouping_params_json,
                })
    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    validate_split_manifest(manifest, n_splits=n_splits, n_repeats=n_repeats)
    return manifest.sort_values(["repeat", "fold", "role", "sample_id"], kind="mergesort").reset_index(drop=True)


def validate_split_manifest(manifest: pd.DataFrame, n_splits: int, n_repeats: int) -> None:
    """Fail loudly if the manifest would leak a group or omit validation samples."""
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise SplitManifestError("split manifest 缺少列：{0}".format(", ".join(missing)))
    if manifest.empty:
        raise SplitManifestError("split manifest 不能为空")
    expected_pairs = n_splits * n_repeats
    actual_pairs = manifest[["repeat", "fold"]].drop_duplicates()
    if len(actual_pairs) != expected_pairs:
        raise SplitManifestError("split manifest 的 repeat/fold 数量错误：{0}，预期 {1}".format(len(actual_pairs), expected_pairs))
    for (repeat, fold), part in manifest.groupby(["repeat", "fold"], sort=True):
        valid = part[part["role"] == "valid"]
        train = part[part["role"] == "train"]
        if len(valid) == 0 or len(train) == 0:
            raise SplitManifestError("repeat={0}, fold={1} 出现空训练集或验证集".format(repeat, fold))
        if valid["sample_id"].duplicated().any() or train["sample_id"].duplicated().any():
            raise SplitManifestError("repeat={0}, fold={1} 中 sample_id 重复".format(repeat, fold))
        leaked = set(valid["group_id"]).intersection(train["group_id"])
        if leaked:
            raise SplitManifestError(
                "repeat={0}, fold={1} 检测到 group 泄漏，示例：{2}".format(repeat, fold, sorted(leaked)[:3])
            )
    for repeat, part in manifest.groupby("repeat", sort=True):
        valid_counts = part.loc[part["role"] == "valid", "sample_id"].value_counts()
        all_ids = part["sample_id"].drop_duplicates()
        if not valid_counts.reindex(all_ids, fill_value=0).eq(1).all():
            raise SplitManifestError("repeat={0} 中每个 sample_id 必须恰好一次进入验证集".format(repeat))


def write_split_manifest(contract: BenchmarkContract, manifest: pd.DataFrame) -> Path:
    """Write a checked Parquet manifest atomically and reject incompatible reuse."""
    if manifest.empty:
        raise SplitManifestError("不能写入空 split manifest")
    split_ids = manifest["split_id"].drop_duplicates().tolist()
    if len(split_ids) != 1:
        raise SplitManifestError("一个运行只能写入一个 split_id")
    path = resolve_benchmark_output_layout(contract.run_dir).manifests / "split_manifest.parquet"
    if path.exists():
        previous = pd.read_parquet(path)
        if previous.equals(manifest):
            return path
        raise SplitManifestError("已有 split manifest 与当前内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    manifest.to_parquet(temporary, index=False)
    checked = pd.read_parquet(temporary)
    if len(checked) != len(manifest) or list(checked.columns) != list(manifest.columns):
        temporary.unlink(missing_ok=True)
        raise SplitManifestError("split manifest 临时文件校验失败")
    temporary.replace(path)
    return path
