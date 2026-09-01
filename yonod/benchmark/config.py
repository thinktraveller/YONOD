"""Benchmark 配置、数据版本与运行清单契约。

本模块刻意不训练模型。它负责在任何特征或模型计算之前固化一次
benchmark 的数据、列角色和评价配置，使后续 split manifest、预测分片和
统计结果能够指向同一份不可歧义的运行元数据。
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark cannot be audited safely before training."""


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically for hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "benchmark"


def _require_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise BenchmarkConfigError(f"{field} 必须是非空字符串列表")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise BenchmarkConfigError(f"{field} 不能包含空值")
    if len(set(values)) != len(values):
        raise BenchmarkConfigError(f"{field} 不能包含重复列名")
    return values


@dataclass(frozen=True)
class BenchmarkConfig:
    """经 schema 校验且路径已解析的 benchmark 配置。"""

    source_path: Path
    dataset_path: Path
    sample_id_col: str
    label_col: str
    smiles_cols: tuple[str, ...]
    descriptors: tuple[str, ...]
    models: tuple[str, ...]
    grouping: Dict[str, Any]
    cv: Dict[str, Any]
    outputs_root: Path
    raw: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path | str) -> "BenchmarkConfig":
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"benchmark 配置文件不存在：{source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise BenchmarkConfigError("配置根节点必须是 mapping，且包含 benchmark 节点")
        block = raw.get("benchmark", raw)
        if not isinstance(block, Mapping):
            raise BenchmarkConfigError("benchmark 节点必须是 mapping")

        required = ("dataset_path", "sample_id_col", "label_col", "smiles_cols", "descriptors", "models")
        missing = [key for key in required if not block.get(key)]
        if missing:
            raise BenchmarkConfigError(f"配置缺少必填字段：{', '.join(missing)}")

        dataset_path = Path(str(block["dataset_path"]))
        if not dataset_path.is_absolute():
            dataset_path = (source_path.parent / dataset_path).resolve()
        outputs = block.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise BenchmarkConfigError("outputs 必须是 mapping")
        outputs_root = Path(str(outputs.get("root", "results")))
        if not outputs_root.is_absolute():
            outputs_root = (source_path.parent / outputs_root).resolve()

        grouping = dict(block.get("grouping", {}))
        cv = dict(block.get("cv", {}))
        if not grouping.get("strategy"):
            raise BenchmarkConfigError("grouping.strategy 不能为空")
        for field in ("n_repeats", "n_splits", "seed"):
            if field not in cv:
                raise BenchmarkConfigError(f"cv.{field} 不能为空")
        if int(cv["n_repeats"]) < 1 or int(cv["n_splits"]) < 2:
            raise BenchmarkConfigError("cv.n_repeats 必须 >= 1，cv.n_splits 必须 >= 2")

        return cls(
            source_path=source_path,
            dataset_path=dataset_path,
            sample_id_col=str(block["sample_id_col"]),
            label_col=str(block["label_col"]),
            smiles_cols=_require_string_list(block["smiles_cols"], "smiles_cols"),
            descriptors=_require_string_list(block["descriptors"], "descriptors"),
            models=_require_string_list(block["models"], "models"),
            grouping=grouping,
            cv=cv,
            outputs_root=outputs_root,
            raw=dict(block),
        )

    def validate_dataset(self) -> pd.DataFrame:
        """Read and validate the data contract before any training begins."""
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"数据集不存在：{self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        required = [self.sample_id_col, self.label_col, *self.smiles_cols]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise BenchmarkConfigError(f"数据集缺少列：{', '.join(missing)}")
        sample_ids = frame[self.sample_id_col]
        if sample_ids.isna().any() or sample_ids.astype(str).str.strip().eq("").any():
            raise BenchmarkConfigError(f"sample_id 列 {self.sample_id_col!r} 存在空值")
        if sample_ids.duplicated().any():
            duplicates = sample_ids[sample_ids.duplicated()].head(5).tolist()
            raise BenchmarkConfigError(
                f"sample_id 列 {self.sample_id_col!r} 不唯一，示例重复值：{duplicates}"
            )
        labels = pd.to_numeric(frame[self.label_col], errors="coerce")
        if labels.isna().any():
            bad_count = int(labels.isna().sum())
            raise BenchmarkConfigError(f"标签列 {self.label_col!r} 含 {bad_count} 个不可解析或空值")
        return frame

    def normalized_for_hash(self, dataset_sha256: str) -> Dict[str, Any]:
        """Return every result-affecting setting in a deterministic structure."""
        return {
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": dataset_sha256,
            "sample_id_col": self.sample_id_col,
            "label_col": self.label_col,
            "smiles_cols": list(self.smiles_cols),
            "descriptors": list(self.descriptors),
            "models": list(self.models),
            "model_params": self.raw.get("model_params", {}),
            "grouping": self.grouping,
            "cv": self.cv,
            "outputs_root": str(self.outputs_root),
        }


@dataclass(frozen=True)
class BenchmarkContract:
    """A run-ready, reproducible identity for one benchmark execution."""

    config: BenchmarkConfig
    dataset_sha256: str
    config_hash: str
    run_id: str
    run_dir: Path
    manifest: Dict[str, Any]


def create_benchmark_contract(config: BenchmarkConfig) -> BenchmarkContract:
    """Validate data and build a deterministic run identity without writing files."""
    config.validate_dataset()
    dataset_sha256 = _sha256_file(config.dataset_path)
    normalized = config.normalized_for_hash(dataset_sha256)
    config_hash = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    run_id = f"{_slug(config.dataset_path.stem)}-{config_hash[:12]}"
    project_root = config.source_path.parent
    for parent in config.source_path.parents:
        if (parent / ".git").exists():
            project_root = parent
            break
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset_sha256": dataset_sha256,
        "dataset_path": str(config.dataset_path),
        "benchmark_config": normalized,
        "source_config_path": str(config.source_path),
        "code_git_commit": _git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return BenchmarkContract(
        config=config,
        dataset_sha256=dataset_sha256,
        config_hash=config_hash,
        run_id=run_id,
        run_dir=config.outputs_root / run_id,
        manifest=manifest,
    )


def write_run_manifest(contract: BenchmarkContract) -> Path:
    """Persist the run manifest without allowing an incompatible overwrite.

    A repeated invocation for the same config is safe and supports later resume;
    a conflicting file at the deterministic run path is treated as corruption.
    """
    manifest_path = contract.run_dir / "manifests" / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_hash") != contract.config_hash:
            raise BenchmarkConfigError(
                f"运行目录已被不同配置占用：{contract.run_dir}；请使用新的 outputs.root"
            )
        return manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(contract.manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(manifest_path)
    return manifest_path
