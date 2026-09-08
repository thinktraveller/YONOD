"""Paper-exact YieldMaster material generation and RF alignment.

This module is deliberately separate from the ordinary ``main.py`` path.  It
builds immutable, hash-audited inputs for the YieldSmarter paper protocol and
reruns only the RF baseline against the accepted paper-native reference files.
The long-running AutoGluon stage can later consume the same population and
split manifests, but this module never launches AutoGluon.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import BenchmarkConfig, create_benchmark_contract
from .paper_exact import (
    PAPER_EXACT_EVALUATION_PROTOCOL,
    PAPER_EXACT_POPULATIONS,
    PaperExactError,
    PaperExactPopulationSpec,
    build_population_manifest,
    compute_fold_metrics,
    get_population_spec,
    sha256_file,
    validate_paper_exact_split_manifest,
)
from ..splits.manifest import create_split_manifest
from .fold_preprocessors import ReactionComponentOHE
from ..universal.feature_builder import build_universal_features


PAPER_EXACT_STAMP = "20260908"
PAPER_EXACT_SEEDS = tuple(range(1000, 1005))
PAPER_EXACT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_features": 0.30,
    "n_jobs": -1,
}
MFP_DESCRIPTOR_CONFIG: dict[str, Any] = {
    "profile": "vjethbkm",
    "algorithm": "morgan_count",
    "radius": 3,
    "fp_size": 1024,
    "input_normalization": "raw_csv_value",
    "blank_or_invalid_component": "zero_block_keep_row",
}


POPULATION_DIR_NAMES = {
    "bh1_paper_exact": "yieldmaster_bh1_mfp_ohe_paper_exact_5x5_{stamp}",
    "bh2_paper_exact": "yieldmaster_bh2_mfp_ohe_paper_exact_5x5_{stamp}",
    "sl1_paper_exact": "yieldmaster_sl1_mfp_ohe_paper_exact_5x5_{stamp}",
    "sm_ohe_paper_exact_5760": "yieldmaster_sm_ohe_paper_exact_5x5_5760_{stamp}",
    "sm_mfp_paper_exact_4620": "yieldmaster_sm_mfp_paper_exact_5x5_4620_{stamp}",
}


DATASET_SOURCES = {
    "BH1": {
        "paper_dataset_id": "BH",
        "source_csv": "yieldsmarter/Data/HTE_datasets/BH1/BH1.csv",
        "unit": "yield %",
        "official_mfp_npz": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP.npz",
        "official_mfp_summary": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP_metrics_summaries_RF.json",
        "official_ohe_summary": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/OHE/Doyle_OHE_metrics_summaries_RF.json",
        "official_ohe_oof": "yieldsmarter/Results/Compare_Complexity/Doyle_2018/OHE/Doyle_OHE_oof_predictions_RF.npz",
    },
    "BH2": {
        "paper_dataset_id": "BH2",
        "source_csv": "yieldsmarter/Data/HTE_datasets/BH2/BH2.csv",
        "unit": "yield %",
        "official_mfp_npz": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/MFP/Denmark_MFP.npz",
        "official_mfp_summary": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/MFP/Denmark_MFP_metrics_summaries_RF.json",
        "official_ohe_summary": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/OHE/Denmark_OHE_metrics_summaries_RF.json",
        "official_ohe_oof": "yieldsmarter/Results/Compare_Complexity/Denmark_2023/OHE/Denmark_OHE_oof_predictions_RF.npz",
    },
    "SL1": {
        "paper_dataset_id": "SLAP",
        "source_csv": "yieldsmarter/Data/HTE_datasets/SL1/SL1.csv",
        "unit": "LC-MS product ratio",
        "official_mfp_npz": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP.npz",
        "official_mfp_summary": "yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP_metrics_summaries_RF.json",
        "official_ohe_summary": "yieldsmarter/Results/Compare_Complexity/Bode_2023/OHE/Bode_OHE_metrics_summaries_RF.json",
        "official_ohe_oof": "yieldsmarter/Results/Compare_Complexity/Bode_2023/OHE/Bode_OHE_oof_predictions_RF.npz",
    },
    "SM": {
        "paper_dataset_id": "SM",
        "source_csv": "yieldsmarter/Data/HTE_datasets/SM/SM.csv",
        "unit": "yield %",
        "official_mfp_npz": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP.npz",
        "official_mfp_summary": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP_metrics_summaries_RF.json",
        "official_ohe_summary": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/OHE/Suzuki_OHE_metrics_summaries_RF.json",
        "official_ohe_oof": "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/OHE/Suzuki_OHE_oof_predictions_RF.npz",
    },
}


REFERENCE_RF_RUNS = {
    "mfp": Path("outputs/runs/native_mfp_rf_matrix_20260907"),
    "ohe": Path("outputs/runs/native_ohe_rf_matrix_20260907"),
}


@dataclass(frozen=True)
class PaperExactMaterial:
    population_id: str
    dataset_id: str
    population_dir: Path
    config_path: Path
    population_csv_path: Path
    population_manifest_path: Path
    split_manifest_path: Path
    descriptor_tasks_path: Path
    run_manifest_path: Path
    row_count: int
    source_csv_path: Path
    source_csv_sha256: str
    population_sha256: str
    population_hash: str
    split_hash: str
    feature_paths: Mapping[str, Path]
    feature_hashes: Mapping[str, str]
    mfp_alignment: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class PaperExactRFRun:
    output_root: Path
    fold_metrics_path: Path
    predictions_path: Path
    summary_path: Path
    alignment_path: Path
    manifest_path: Path
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    alignment: pd.DataFrame
    max_abs_metric_diff: float


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_array_payload(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(_canonical_json(list(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _fixed_unicode_array(values: Sequence[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype="<U{0}".format(width))


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists() and not overwrite and path.read_text(encoding="utf-8") != text:
        raise PaperExactError("拒绝覆盖已有不同 JSON：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _write_csv(path: Path, frame: pd.DataFrame, *, overwrite: bool) -> Path:
    text = frame.to_csv(index=False, encoding=None, lineterminator="\n")
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") != text:
            raise PaperExactError("拒绝覆盖已有不同 CSV：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> Path:
    frame = pd.DataFrame.from_records(rows)
    return _write_csv(path, frame, overwrite=overwrite)


def paper_exact_population_dir(output_root: Path | str, population_id: str, *, stamp: str = PAPER_EXACT_STAMP) -> Path:
    if population_id not in POPULATION_DIR_NAMES:
        raise PaperExactError("未知 paper_exact population_id：{0}".format(population_id))
    return Path(output_root) / POPULATION_DIR_NAMES[population_id].format(stamp=stamp)


def paper_exact_batch_dir(output_root: Path | str, *, stamp: str = PAPER_EXACT_STAMP) -> Path:
    return Path(output_root) / "yieldmaster_paper_exact_5x5_{0}".format(stamp)


def _source_csv_path(reference_root: Path, spec: PaperExactPopulationSpec) -> Path:
    try:
        relative = DATASET_SOURCES[spec.dataset_id]["source_csv"]
    except KeyError as exc:
        raise PaperExactError("缺少数据源映射：{0}".format(spec.dataset_id)) from exc
    path = reference_root / str(relative)
    if not path.is_file():
        raise FileNotFoundError("paper_exact 源 CSV 不存在：{0}".format(path))
    return path


def _official_mfp_npz_path(reference_root: Path, spec: PaperExactPopulationSpec) -> Path:
    path = reference_root / str(DATASET_SOURCES[spec.dataset_id]["official_mfp_npz"])
    if not path.is_file():
        raise FileNotFoundError("官方 MFP NPZ 不存在：{0}".format(path))
    return path


def _population_filter_name(spec: PaperExactPopulationSpec) -> str:
    if spec.population_id == "sm_mfp_paper_exact_4620":
        return "dropna_component_and_label_for_mfp"
    return "raw_csv_all_rows"


def load_paper_exact_population(
    reference_root: Path | str,
    population_id: str,
) -> tuple[pd.DataFrame, np.ndarray, Path, str]:
    """Load the exact paper population and retain original source row indices."""
    reference_root = Path(reference_root)
    spec = get_population_spec(population_id)
    source_csv = _source_csv_path(reference_root, spec)
    source_sha = sha256_file(source_csv)
    raw = pd.read_csv(source_csv)
    missing = [column for column in [*spec.component_cols, spec.label_col] if column not in raw.columns]
    if missing:
        raise PaperExactError("{0} 源 CSV 缺少列：{1}".format(population_id, ", ".join(missing)))
    source_row_index = np.arange(len(raw), dtype=np.int64)
    frame = raw.loc[:, [*spec.component_cols, spec.label_col]].copy()
    if population_id == "sm_mfp_paper_exact_4620":
        keep = frame.loc[:, [*spec.component_cols, spec.label_col]].notna().all(axis=1).to_numpy()
        frame = frame.loc[keep].reset_index(drop=True)
        source_row_index = source_row_index[keep]
    else:
        frame = frame.reset_index(drop=True)
    if len(frame) != spec.expected_rows:
        raise PaperExactError(
            "{0} material 行数为 {1}，预期 {2}".format(population_id, len(frame), spec.expected_rows)
        )
    material = frame.copy()
    material.insert(0, "source_row_index", source_row_index.astype(int))
    material.insert(
        0,
        "sample_id",
        ["{0}:source-row-{1:06d}".format(population_id, int(index)) for index in source_row_index],
    )
    return material, source_row_index.astype(int), source_csv, source_sha




def _paper_exact_split_hash(contract: Any, population_frame: pd.DataFrame, population_id: str) -> str:
    payload = {
        "run_id": contract.run_id,
        "population_id": population_id,
        "dataset_sha256": contract.dataset_sha256,
        "grouping": contract.config.grouping,
        "cv": contract.config.cv,
        "source_rows": [
            {"source_row_index": int(row.source_row_index), "sample_id": str(row.sample_id)}
            for row in population_frame[["sample_id", "source_row_index"]].itertuples(index=False)
        ],
    }
    return _hash_payload(payload)


def _remap_split_manifest_to_source_rows(contract: Any, split_manifest: pd.DataFrame, population_frame: pd.DataFrame, population_id: str) -> pd.DataFrame:
    source_by_sample = population_frame.set_index("sample_id")["source_row_index"].astype(int).to_dict()
    result = split_manifest.copy()
    result["source_row_index"] = result["sample_id"].astype(str).map(source_by_sample).astype(int)
    split_hash = _paper_exact_split_hash(contract, population_frame, population_id)
    result["split_hash"] = split_hash
    result["split_id"] = "split-" + split_hash[:12]
    return result

def _feature_set_contract(spec: PaperExactPopulationSpec) -> list[dict[str, Any]]:
    feature_sets: list[dict[str, Any]] = []
    for feature_id in spec.feature_ids:
        if feature_id == "mfp":
            feature_sets.append({
                "name": "mfp",
                "kind": "precomputed_descriptor",
                "component_cols": list(spec.component_cols),
                "params": dict(MFP_DESCRIPTOR_CONFIG),
            })
        elif feature_id == "ohe":
            feature_sets.append({
                "name": "ohe",
                "kind": "fold_transform",
                "component_cols": list(spec.component_cols),
                "params": {
                    "encoder": "OneHotEncoder",
                    "handle_unknown": "ignore",
                    "missing_token": "__MISSING__",
                    "missing_policy": "zero_block",
                    "zero_out_missing": True,
                    "fit_scope": "train_only_per_fold",
                },
            })
        else:
            raise PaperExactError("paper_exact 暂不支持特征：{0}".format(feature_id))
    return feature_sets


def build_paper_exact_config(
    spec: PaperExactPopulationSpec,
    *,
    dataset_path: str = "data/population.csv",
    outputs_root: str = "runs",
) -> dict[str, Any]:
    source_info = DATASET_SOURCES[spec.dataset_id]
    return {
        "benchmark": {
            "dataset_path": dataset_path,
            "sample_id_col": "sample_id",
            "label_col": spec.label_col,
            "smiles_cols": list(spec.component_cols),
            "feature_sets": _feature_set_contract(spec),
            "models": ["rf"],
            "model_params": {"rf": dict(PAPER_EXACT_RF_PARAMS)},
            "grouping": {"strategy": "repeated_kfold", "source_order": "raw_csv"},
            "cv": {"n_repeats": 5, "n_splits": 5, "seed": 1000},
            "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
            "reproduction_protocol": {
                "name": PAPER_EXACT_EVALUATION_PROTOCOL,
                "paper": "Yield Smarter, Not Harder",
                "split": "5 repeats x 5 folds, sklearn KFold shuffle=True, seeds 1000-1004",
                "rf_params": dict(PAPER_EXACT_RF_PARAMS),
            },
            "population_id": spec.population_id,
            "dataset_id": spec.dataset_id,
            "paper_exact": {
                "population_id": spec.population_id,
                "dataset_id": spec.dataset_id,
                "paper_dataset_id": source_info["paper_dataset_id"],
                "expected_rows": spec.expected_rows,
                "component_cols": list(spec.component_cols),
                "feature_ids": list(spec.feature_ids),
                "source_csv": source_info["source_csv"],
                "source_row_filter": _population_filter_name(spec),
                "unit": source_info["unit"],
                "notes": spec.notes,
            },
            "outputs": {"root": outputs_root},
        }
    }


def _mfp_alignment(X: np.ndarray, y: np.ndarray, spec: PaperExactPopulationSpec, reference_root: Path) -> dict[str, Any]:
    official_path = _official_mfp_npz_path(reference_root, spec)
    with np.load(official_path) as official:
        same_shape = X.shape == official["X"].shape
        max_abs_diff = (
            int(np.abs(X.astype(np.int64) - official["X"].astype(np.int64)).max())
            if same_shape else None
        )
        return {
            "official_npz": str(official_path),
            "generated_shape": list(X.shape),
            "official_shape": list(official["X"].shape),
            "x_exact": bool(np.array_equal(X, official["X"])),
            "x_max_abs_diff": max_abs_diff,
            "y_exact": bool(np.array_equal(y, official["y"])),
            "smiles_columns_exact": bool(np.array_equal(np.asarray(spec.component_cols), official["smiles_columns"])),
        }


def prepare_paper_exact_material(
    reference_root: Path | str,
    output_root: Path | str,
    population_id: str,
    *,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
) -> PaperExactMaterial:
    """Create immutable CSV/NPZ/manifest material for one population."""
    reference_root = Path(reference_root).resolve()
    output_root = Path(output_root).resolve()
    spec = get_population_spec(population_id)
    population_dir = paper_exact_population_dir(output_root, population_id, stamp=stamp)
    data_dir = population_dir / "data"
    manifest_dir = population_dir / "manifests"
    feature_dir = population_dir / "features"
    task_dir = population_dir / "tasks"

    population_frame, source_rows, source_csv, source_sha = load_paper_exact_population(reference_root, population_id)
    population_csv_path = data_dir / "population.csv"
    _write_csv(population_csv_path, population_frame, overwrite=overwrite)
    population_sha = sha256_file(population_csv_path)

    config = build_paper_exact_config(spec)
    config_path = population_dir / "paper_exact_rf_config.json"
    _write_json(config_path, config, overwrite=overwrite)
    contract = create_benchmark_contract(BenchmarkConfig.from_file(config_path))
    split_manifest = _remap_split_manifest_to_source_rows(
        contract, create_split_manifest(contract), population_frame, population_id
    )
    population_manifest = build_population_manifest(
        population_frame,
        spec,
        dataset_sha256=contract.dataset_sha256,
        sample_id_col="sample_id",
        source_row_indices=population_frame["source_row_index"].astype(int).tolist(),
    )
    validate_paper_exact_split_manifest(split_manifest, population_manifest)
    population_hash = str(population_manifest["population_hash"].iloc[0])
    split_hash = str(split_manifest["split_hash"].iloc[0])
    population_manifest_path = data_dir / "population_manifest.csv"
    split_manifest_path = manifest_dir / "split_manifest.csv"
    run_manifest_path = manifest_dir / "run_manifest.json"
    _write_csv(population_manifest_path, population_manifest, overwrite=overwrite)
    _write_csv(split_manifest_path, split_manifest, overwrite=overwrite)
    _write_json(run_manifest_path, contract.manifest, overwrite=overwrite)

    descriptor_rows = [
        {
            "population_id": population_id,
            "dataset_id": spec.dataset_id,
            "feature_id": feature_id,
            "model": "rf",
            "expected_folds": 25,
            "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
            "split_hash": split_hash,
            "population_hash": population_hash,
        }
        for feature_id in spec.feature_ids
    ]
    descriptor_tasks_path = task_dir / "descriptor_tasks.csv"
    _write_rows_csv(descriptor_tasks_path, descriptor_rows, overwrite=overwrite)

    feature_paths: dict[str, Path] = {}
    feature_hashes: dict[str, str] = {}
    mfp_alignment: Optional[Mapping[str, Any]] = None
    y = population_frame[spec.label_col].to_numpy(dtype=float)
    if "mfp" in spec.feature_ids:
        X, _numeric, valid_mask = build_universal_features(
            list(spec.component_cols),
            [],
            population_frame,
            "mfp",
            mode="concat",
            descriptor_config=dict(MFP_DESCRIPTOR_CONFIG),
        )
        if not bool(np.asarray(valid_mask).all()):
            raise PaperExactError("{0} MFP unexpectedly dropped rows".format(population_id))
        alignment = _mfp_alignment(X, y, spec, reference_root)
        if not all(alignment[key] for key in ("x_exact", "y_exact", "smiles_columns_exact")):
            raise PaperExactError("{0} MFP 与官方 NPZ 不一致：{1}".format(population_id, alignment))
        feature_hash = _hash_array_payload(X=X, y=y, source_row_index=source_rows)
        feature_path = feature_dir / "mfp.npz"
        if feature_path.exists() and not overwrite:
            with np.load(feature_path) as previous:
                if "feature_hash" not in previous.files or str(previous["feature_hash"]) != feature_hash:
                    raise PaperExactError("拒绝覆盖已有不同 MFP 特征：{0}".format(feature_path))
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_path,
            X=X,
            y=y,
            sample_id=_fixed_unicode_array(population_frame["sample_id"].astype(str).tolist()),
            source_row_index=source_rows,
            smiles_columns=_fixed_unicode_array(spec.component_cols),
            feature_id="mfp",
            feature_hash=feature_hash,
            population_hash=population_hash,
            split_hash=split_hash,
        )
        feature_paths["mfp"] = feature_path
        feature_hashes["mfp"] = feature_hash
        mfp_alignment = alignment
        _write_json(feature_dir / "mfp_alignment.json", alignment, overwrite=overwrite)
    if "ohe" in spec.feature_ids:
        ohe_contract = {
            "feature_id": "ohe",
            "fit_scope": "train_only_per_fold",
            "encoder": "OneHotEncoder",
            "handle_unknown": "ignore",
            "missing_token": "__MISSING__",
            "missing_policy": "zero_block",
            "component_cols": list(spec.component_cols),
            "global_matrix": False,
            "population_hash": population_hash,
            "split_hash": split_hash,
        }
        feature_hash = _hash_payload(ohe_contract)
        ohe_path = feature_dir / "ohe_contract.json"
        _write_json(ohe_path, {**ohe_contract, "feature_hash": feature_hash}, overwrite=overwrite)
        feature_paths["ohe"] = ohe_path
        feature_hashes["ohe"] = feature_hash

    material_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "population_id": population_id,
        "dataset_id": spec.dataset_id,
        "paper_dataset_id": DATASET_SOURCES[spec.dataset_id]["paper_dataset_id"],
        "row_count": len(population_frame),
        "expected_rows": spec.expected_rows,
        "source_csv": str(source_csv),
        "source_csv_sha256": source_sha,
        "source_row_filter": _population_filter_name(spec),
        "population_csv": str(population_csv_path),
        "population_sha256": population_sha,
        "population_hash": population_hash,
        "split_manifest": str(split_manifest_path),
        "split_hash": split_hash,
        "feature_paths": {key: str(value) for key, value in feature_paths.items()},
        "feature_hashes": feature_hashes,
        "mfp_alignment": dict(mfp_alignment or {}),
    }
    _write_json(manifest_dir / "paper_exact_material_manifest.json", material_manifest, overwrite=overwrite)

    return PaperExactMaterial(
        population_id=population_id,
        dataset_id=spec.dataset_id,
        population_dir=population_dir,
        config_path=config_path,
        population_csv_path=population_csv_path,
        population_manifest_path=population_manifest_path,
        split_manifest_path=split_manifest_path,
        descriptor_tasks_path=descriptor_tasks_path,
        run_manifest_path=run_manifest_path,
        row_count=len(population_frame),
        source_csv_path=source_csv,
        source_csv_sha256=source_sha,
        population_sha256=population_sha,
        population_hash=population_hash,
        split_hash=split_hash,
        feature_paths=feature_paths,
        feature_hashes=feature_hashes,
        mfp_alignment=mfp_alignment,
    )


def prepare_all_paper_exact_materials(
    reference_root: Path | str,
    output_root: Path | str,
    *,
    population_ids: Optional[Sequence[str]] = None,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
) -> list[PaperExactMaterial]:
    ids = tuple(population_ids or PAPER_EXACT_POPULATIONS.keys())
    return [
        prepare_paper_exact_material(reference_root, output_root, population_id, stamp=stamp, overwrite=overwrite)
        for population_id in ids
    ]


def _indices_for_fold(split_manifest: pd.DataFrame, sample_ids: Sequence[str], repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, int]:
    part = split_manifest[(split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold)]
    if part.empty:
        raise PaperExactError("split manifest 缺少 repeat={0}, fold={1}".format(repeat, fold))
    pos_by_id = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
    train_ids = part.loc[part["role"] == "train"].sort_values("source_row_index")["sample_id"].astype(str).tolist()
    valid_ids = part.loc[part["role"] == "valid"].sort_values("source_row_index")["sample_id"].astype(str).tolist()
    seed_values = pd.to_numeric(part["seed"], errors="raise").astype(int).unique().tolist()
    if len(seed_values) != 1:
        raise PaperExactError("split manifest 当前折 seed 不唯一")
    return (
        np.asarray([pos_by_id[sample_id] for sample_id in train_ids], dtype=np.int64),
        np.asarray([pos_by_id[sample_id] for sample_id in valid_ids], dtype=np.int64),
        int(seed_values[0]),
    )


def _paper_metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = compute_fold_metrics(y_true, y_pred)
    try:
        from scipy.stats import kendalltau
        metrics["kendall_tau"] = float(kendalltau(y_true, y_pred)[0])
    except Exception:
        pass
    return metrics


def _summary_from_folds(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (population_id, dataset_id, feature_id, model), part in frame.groupby(
        ["population_id", "dataset_id", "feature_id", "model"], sort=True
    ):
        for metric in ("mae", "rmse", "r2", "kendall_tau"):
            values = pd.to_numeric(part[metric], errors="raise").to_numpy(dtype=float)
            rows.append({
                "population_id": population_id,
                "dataset_id": dataset_id,
                "feature_id": feature_id,
                "model": model,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "valid_fold_count": int(len(values)),
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
            })
    return pd.DataFrame.from_records(rows)


def _reference_fold_metrics(reference_root: Path, dataset_id: str, feature_id: str) -> pd.DataFrame:
    descriptor = feature_id.upper()
    root = reference_root / REFERENCE_RF_RUNS[feature_id]
    path = root / "{0}_{1}_RF_fold_metrics.csv".format(dataset_id, descriptor)
    if not path.is_file():
        raise FileNotFoundError("缺少已验收 native RF 折级基准：{0}".format(path))
    frame = pd.read_csv(path)
    if "kendall" in frame.columns and "kendall_tau" not in frame.columns:
        frame = frame.rename(columns={"kendall": "kendall_tau"})
    return frame


def _reference_summary_metrics(reference_root: Path, dataset_id: str, feature_id: str) -> pd.DataFrame:
    descriptor = feature_id.upper()
    root = reference_root / REFERENCE_RF_RUNS[feature_id]
    path = root / "{0}_RF_matrix_comparison.csv".format(descriptor)
    if not path.is_file():
        raise FileNotFoundError("缺少已验收 native RF 汇总基准：{0}".format(path))
    frame = pd.read_csv(path)
    frame = frame[
        (frame["dataset_id"].astype(str) == dataset_id)
        & (frame["descriptor"].astype(str).str.lower() == feature_id)
        & (frame["model"].astype(str).str.upper() == "RF")
    ].copy()
    if frame.empty:
        raise PaperExactError("汇总基准中缺少 {0}/{1}".format(dataset_id, feature_id))
    return frame


def compare_rf_to_native_reference(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    reference_root: Path | str,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    if fold_metrics.empty:
        raise PaperExactError("RF 对齐输入没有任何折级指标")
    reference_root = Path(reference_root).resolve()
    rows: list[dict[str, Any]] = []
    for (population_id, dataset_id, feature_id), part in fold_metrics.groupby(
        ["population_id", "dataset_id", "feature_id"], sort=True
    ):
        reference_folds = _reference_fold_metrics(reference_root, str(dataset_id), str(feature_id))
        merged = part.merge(
            reference_folds,
            left_on=["repeat", "fold"],
            right_on=["repeat", "fold"],
            suffixes=("_yonod", "_native"),
            how="outer",
            indicator=True,
        )
        if not merged["_merge"].eq("both").all() or len(merged) != 25:
            raise PaperExactError("{0}/{1} RF 折集合与已验收 native 基准不一致".format(population_id, feature_id))
        native_summary = _reference_summary_metrics(reference_root, str(dataset_id), str(feature_id))
        for metric in ("mae", "rmse", "r2", "kendall_tau"):
            fold_diff = np.abs(
                pd.to_numeric(merged[f"{metric}_yonod"], errors="raise").to_numpy(dtype=float)
                - pd.to_numeric(merged[f"{metric}_native"], errors="raise").to_numpy(dtype=float)
            )
            max_fold_diff = float(fold_diff.max())
            summary_row = summary[
                (summary["population_id"].astype(str) == str(population_id))
                & (summary["feature_id"].astype(str) == str(feature_id))
                & (summary["metric"].astype(str) == metric)
            ].iloc[0]
            native_row = native_summary[native_summary["metric"].astype(str) == metric].iloc[0]
            mean_diff = float(summary_row["mean"] - float(native_row["independent_native_mean"]))
            std_diff = float(summary_row["std"] - float(native_row["independent_native_std"]))
            row_count_equal = bool(
                pd.to_numeric(merged["train_rows_yonod"], errors="raise").equals(
                    pd.to_numeric(merged["train_rows_native"], errors="raise")
                )
                and pd.to_numeric(merged["test_rows_yonod"], errors="raise").equals(
                    pd.to_numeric(merged["test_rows_native"], errors="raise")
                )
            )
            seed_equal = bool(
                pd.to_numeric(merged["seed"], errors="raise").equals(
                    pd.to_numeric(merged["repeat_seed"], errors="raise")
                )
            )
            rows.append({
                "population_id": population_id,
                "dataset_id": dataset_id,
                "feature_id": feature_id,
                "model": "rf",
                "metric": metric,
                "valid_fold_count": 25,
                "yonod_paper_exact_mean": float(summary_row["mean"]),
                "yonod_paper_exact_std": float(summary_row["std"]),
                "native_reference_mean": float(native_row["independent_native_mean"]),
                "native_reference_std": float(native_row["independent_native_std"]),
                "delta_mean": mean_diff,
                "delta_std": std_diff,
                "max_abs_fold_metric_diff": max_fold_diff,
                "row_counts_equal": row_count_equal,
                "seeds_equal": seed_equal,
                "passes_tolerance": bool(
                    max_fold_diff <= tolerance
                    and abs(mean_diff) <= tolerance
                    and abs(std_diff) <= tolerance
                    and row_count_equal
                    and seed_equal
                ),
            })
    return pd.DataFrame.from_records(rows)


def run_rf_for_material(
    material: PaperExactMaterial,
    *,
    n_estimators: int = 500,
    max_features: float = 0.30,
    n_jobs: int = -1,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = get_population_spec(material.population_id)
    population = pd.read_csv(material.population_csv_path)
    split_manifest = pd.read_csv(material.split_manifest_path)
    sample_ids = population["sample_id"].astype(str).tolist()
    y = population[spec.label_col].to_numpy(dtype=float)
    source_row_index = population["source_row_index"].astype(int).to_numpy()
    fold_pairs = split_manifest[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for feature_id in spec.feature_ids:
        X: Optional[np.ndarray] = None
        feature_hash = str(material.feature_hashes[feature_id])
        if feature_id == "mfp":
            with np.load(material.feature_paths["mfp"]) as feature:
                X = np.asarray(feature["X"])
                if not np.array_equal(feature["y"], y):
                    raise PaperExactError("{0} MFP y 与 population.csv 不一致".format(material.population_id))
        elif feature_id != "ohe":
            raise PaperExactError("paper_exact RF 暂不支持特征：{0}".format(feature_id))
        for repeat, fold in fold_pairs.itertuples(index=False, name=None):
            train_idx, valid_idx, seed = _indices_for_fold(split_manifest, sample_ids, int(repeat), int(fold))
            if feature_id == "mfp":
                assert X is not None
                X_train = X[train_idx]
                X_valid = X[valid_idx]
                feature_dim = int(X_train.shape[1])
                feature_audit: dict[str, Any] = {
                    "feature_id": "mfp",
                    "feature_hash": feature_hash,
                    "fit_scope": "precomputed_from_paper_csv",
                }
            else:
                transformer = ReactionComponentOHE(spec.component_cols, missing_token="__MISSING__")
                component_frame = population.loc[:, list(spec.component_cols)]
                transformer.fit(component_frame.iloc[train_idx], sample_ids=[sample_ids[index] for index in train_idx])
                X_train = transformer.transform(component_frame.iloc[train_idx], partition="train")
                X_valid = transformer.transform(component_frame.iloc[valid_idx], partition="valid")
                feature_dim = int(X_train.shape[1])
                feature_audit = transformer.metadata()
                feature_audit["feature_hash"] = feature_hash
            model = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_features=float(max_features),
                n_jobs=int(n_jobs),
                random_state=seed,
            )
            started = time.perf_counter()
            model.fit(X_train, y[train_idx])
            y_pred = model.predict(X_valid)
            elapsed = time.perf_counter() - started
            metrics = _paper_metric_values(y[valid_idx], y_pred)
            split_hash = str(split_manifest["split_hash"].iloc[0])
            fold_rows.append({
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "rf",
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "repeat": int(repeat),
                "fold": int(fold),
                "seed": int(seed),
                "split_hash": split_hash,
                "population_hash": material.population_hash,
                "feature_hash": feature_hash,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(valid_idx)),
                "feature_dim": feature_dim,
                "fit_predict_seconds": float(elapsed),
                **metrics,
            })
            for index, pred in zip(valid_idx, y_pred):
                prediction_rows.append({
                    "population_id": material.population_id,
                    "dataset_id": spec.dataset_id,
                    "feature_id": feature_id,
                    "model": "rf",
                    "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "seed": int(seed),
                    "split_hash": split_hash,
                    "population_hash": material.population_hash,
                    "feature_hash": feature_hash,
                    "sample_id": sample_ids[int(index)],
                    "source_row_index": int(source_row_index[int(index)]),
                    "y_true": float(y[int(index)]),
                    "y_pred": float(pred),
                })
            fold_metadata = {
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "rf",
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "repeat": int(repeat),
                "fold": int(fold),
                "seed": int(seed),
                "rf_params": {
                    "n_estimators": int(n_estimators),
                    "max_features": float(max_features),
                    "n_jobs": int(n_jobs),
                    "random_state": int(seed),
                },
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(valid_idx)),
                "feature_dim": feature_dim,
                "train_sample_ids_hash": _hash_payload([sample_ids[index] for index in train_idx]),
                "valid_sample_ids_hash": _hash_payload([sample_ids[index] for index in valid_idx]),
                "valid_source_row_index_hash": _hash_payload([int(source_row_index[index]) for index in valid_idx]),
                "feature_audit": feature_audit,
                "metrics": metrics,
            }
            fold_path = material.population_dir / "rf" / "folds" / "{0}__rf__r{1:02d}__f{2:02d}.json".format(feature_id, int(repeat), int(fold))
            _write_json(fold_path, fold_metadata, overwrite=overwrite)

    fold_frame = pd.DataFrame.from_records(fold_rows)
    prediction_frame = pd.DataFrame.from_records(prediction_rows)
    rf_dir = material.population_dir / "rf"
    _write_csv(rf_dir / "fold_metrics.csv", fold_frame, overwrite=overwrite)
    _write_csv(rf_dir / "predictions.csv", prediction_frame, overwrite=overwrite)
    summary = _summary_from_folds(fold_frame)
    _write_csv(rf_dir / "summary.csv", summary, overwrite=overwrite)
    return fold_frame, prediction_frame


def run_paper_exact_rf_matrix(
    reference_root: Path | str,
    output_root: Path | str,
    *,
    population_ids: Optional[Sequence[str]] = None,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
    tolerance: float = 1e-9,
    n_estimators: int = 500,
    max_features: float = 0.30,
    n_jobs: int = -1,
) -> PaperExactRFRun:
    reference_root = Path(reference_root).resolve()
    output_root = Path(output_root).resolve()
    materials = prepare_all_paper_exact_materials(
        reference_root,
        output_root,
        population_ids=population_ids,
        stamp=stamp,
        overwrite=overwrite,
    )
    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for material in materials:
        folds, predictions = run_rf_for_material(
            material,
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=n_jobs,
            overwrite=overwrite,
        )
        all_fold_metrics.append(folds)
        all_predictions.append(predictions)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = _summary_from_folds(fold_metrics)
    alignment = compare_rf_to_native_reference(fold_metrics, summary, reference_root, tolerance=tolerance)
    max_abs_diff = float(alignment["max_abs_fold_metric_diff"].max()) if not alignment.empty else float("nan")
    if not bool(alignment["passes_tolerance"].all()):
        failed = alignment.loc[~alignment["passes_tolerance"]].copy()
        raise PaperExactError("YONOD paper_exact RF 未与已验收 native 基准对齐：\n{0}".format(failed.to_string(index=False)))

    batch_dir = paper_exact_batch_dir(output_root, stamp=stamp)
    fold_metrics_path = batch_dir / "rf_fold_metrics.csv"
    predictions_path = batch_dir / "rf_predictions.csv"
    summary_path = batch_dir / "rf_summary.csv"
    alignment_path = batch_dir / "rf_alignment.csv"
    manifest_path = batch_dir / "paper_exact_rf_run_manifest.json"
    _write_csv(fold_metrics_path, fold_metrics, overwrite=overwrite)
    _write_csv(predictions_path, predictions, overwrite=overwrite)
    _write_csv(summary_path, summary, overwrite=overwrite)
    _write_csv(alignment_path, alignment, overwrite=overwrite)
    descriptor_tasks = pd.concat(
        [pd.read_csv(material.descriptor_tasks_path) for material in materials],
        ignore_index=True,
    )
    _write_csv(batch_dir / "descriptor_tasks.csv", descriptor_tasks, overwrite=overwrite)
    command_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "python": sys.version,
        "platform": platform.platform(),
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "population_ids": [material.population_id for material in materials],
        "descriptor_task_count": int(len(descriptor_tasks)),
        "fold_task_count": int(len(fold_metrics)),
        "seeds": list(PAPER_EXACT_SEEDS),
        "rf_params": {
            "n_estimators": int(n_estimators),
            "max_features": float(max_features),
            "n_jobs": int(n_jobs),
        },
        "max_abs_metric_diff_vs_native_reference": max_abs_diff,
        "alignment_passed": bool(alignment["passes_tolerance"].all()),
        "population_dirs": {material.population_id: str(material.population_dir) for material in materials},
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "alignment": str(alignment_path),
        },
    }
    _write_json(manifest_path, command_manifest, overwrite=overwrite)
    return PaperExactRFRun(
        output_root=batch_dir,
        fold_metrics_path=fold_metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        alignment_path=alignment_path,
        manifest_path=manifest_path,
        fold_metrics=fold_metrics,
        summary=summary,
        alignment=alignment,
        max_abs_metric_diff=max_abs_diff,
    )


__all__ = [
    "DATASET_SOURCES",
    "MFP_DESCRIPTOR_CONFIG",
    "PAPER_EXACT_RF_PARAMS",
    "PAPER_EXACT_SEEDS",
    "PAPER_EXACT_STAMP",
    "POPULATION_DIR_NAMES",
    "PaperExactMaterial",
    "PaperExactRFRun",
    "build_paper_exact_config",
    "compare_rf_to_native_reference",
    "load_paper_exact_population",
    "paper_exact_batch_dir",
    "paper_exact_population_dir",
    "prepare_all_paper_exact_materials",
    "prepare_paper_exact_material",
    "run_paper_exact_rf_matrix",
]
