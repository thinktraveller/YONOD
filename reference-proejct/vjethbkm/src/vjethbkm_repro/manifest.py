"""Manifest and data-audit helpers for the VJETHBKM reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .paths import REPRO_ROOT


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    path: Path
    label_col: str
    component_cols: list[str]
    nrows: int | None = None


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_repro_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPRO_ROOT / path).resolve()


def dataset_config_from_sources(dataset_id: str) -> DatasetConfig:
    sources = load_yaml(REPRO_ROOT / "configs" / "10_data_sources.yaml")
    try:
        cfg = sources["datasets"][dataset_id]
    except KeyError as exc:
        raise KeyError(f"Dataset not configured: {dataset_id}") from exc

    return DatasetConfig(
        dataset_id=dataset_id,
        path=resolve_repro_path(cfg["path"]),
        label_col=cfg["label_col"],
        component_cols=list(cfg["component_cols"]),
        nrows=cfg.get("nrows"),
    )


def read_dataset(config: DatasetConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(f"Missing dataset: {config.path}")
    required = [*config.component_cols, config.label_col]
    df = pd.read_csv(config.path, nrows=config.nrows)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {config.dataset_id}: {missing}")
    if df.empty:
        raise ValueError(f"Dataset is empty: {config.dataset_id}")
    return df


def dataset_stats(df: pd.DataFrame, config: DatasetConfig) -> dict:
    target = pd.to_numeric(df[config.label_col], errors="coerce")
    stats = {
        "dataset_id": config.dataset_id,
        "path": str(config.path),
        "sha256": sha256_file(config.path),
        "n_rows_read": int(len(df)),
        "n_columns": int(len(df.columns)),
        "target_column": config.label_col,
        "target_missing": int(target.isna().sum()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }
    for column in config.component_cols:
        stats[f"unique_{column}"] = int(df[column].nunique(dropna=True))
        stats[f"missing_{column}"] = int(df[column].isna().sum())
    return stats


def write_stats_csv(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(records)).to_csv(output_path, index=False)

