
"""AutoGluon runner for YieldMaster paper-exact 5x5 manifests.

This module intentionally consumes the immutable materials produced by
``paper_exact_pipeline`` instead of the ordinary ``main.py`` path.  The goal is
not a faster shortcut; it is an auditable runner where RF and AutoGluon share
one population manifest and one 5 repeats x 5 folds split manifest.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .fold_preprocessors import ReactionComponentOHE
from .paper_exact import (
    PAPER_EXACT_EVALUATION_PROTOCOL,
    PAPER_EXACT_POPULATIONS,
    PaperExactError,
    get_population_spec,
    validate_paper_exact_split_manifest,
)
from .paper_exact_pipeline import (
    DATASET_SOURCES,
    PAPER_EXACT_SEEDS,
    PAPER_EXACT_STAMP,
    PaperExactMaterial,
    _git_commit,
    _hash_payload,
    _indices_for_fold,
    _paper_metric_values,
    _write_csv,
    _write_json,
    _fixed_unicode_array,
    _hash_array_payload,
    build_paper_exact_config,
    paper_exact_batch_dir,
    paper_exact_population_dir,
    prepare_paper_exact_material,
)
from ..models.autogluon_model import AutoGluonYieldModel


PAPER_EXACT_AUTOGLOON_PARAMS: dict[str, Any] = {
    "time_limit": 300,
    "presets": "medium_quality",
    "num_cpus": 19,
    "random_state": 42,
}
PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR = 25
PAPER_EXACT_AUTOGLOON_SEED_POLICY = (
    "outer split fixed by paper_exact split_manifest seeds 1000-1004; "
    "AutoGluon 1.1.1 has no single top-level seed covering all submodels"
)


@dataclass(frozen=True)
class PaperExactAutoGluonRun:
    output_root: Path
    fold_metrics_path: Path
    predictions_path: Path
    summary_path: Path
    paired_input_path: Optional[Path]
    manifest_path: Path
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    paired_input: Optional[pd.DataFrame]
    expected_fold_count: int
    completed_fold_count: int
    status: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PaperExactError("JSON 根节点必须为对象：{0}".format(path))
    return payload


def load_existing_paper_exact_material(
    output_root: Path | str,
    population_id: str,
    *,
    stamp: str = PAPER_EXACT_STAMP,
) -> PaperExactMaterial:
    """Load an already prepared immutable paper_exact material without rewriting it."""
    output_root = Path(output_root).resolve()
    population_dir = paper_exact_population_dir(output_root, population_id, stamp=stamp)
    manifest_path = population_dir / "manifests" / "paper_exact_material_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("paper_exact material manifest 不存在：{0}".format(manifest_path))
    payload = _read_json(manifest_path)
    feature_paths = {str(key): Path(value) for key, value in dict(payload.get("feature_paths", {})).items()}
    feature_hashes = {str(key): str(value) for key, value in dict(payload.get("feature_hashes", {})).items()}
    required_paths = [
        Path(payload["population_csv"]),
        population_dir / "data" / "population_manifest.csv",
        Path(payload["split_manifest"]),
    ]
    required_paths.extend(feature_paths.values())
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("paper_exact material 文件缺失：{0}".format(", ".join(missing)))

    population_manifest_path = population_dir / "data" / "population_manifest.csv"
    population_manifest = pd.read_csv(population_manifest_path)
    split_manifest = pd.read_csv(Path(payload["split_manifest"]))
    validate_paper_exact_split_manifest(split_manifest, population_manifest)

    return PaperExactMaterial(
        population_id=population_id,
        dataset_id=str(payload["dataset_id"]),
        population_dir=population_dir,
        config_path=population_dir / "paper_exact_rf_config.json",
        population_csv_path=Path(payload["population_csv"]),
        population_manifest_path=population_manifest_path,
        split_manifest_path=Path(payload["split_manifest"]),
        descriptor_tasks_path=population_dir / "tasks" / "descriptor_tasks.csv",
        run_manifest_path=population_dir / "manifests" / "run_manifest.json",
        row_count=int(payload["row_count"]),
        source_csv_path=Path(payload["source_csv"]),
        source_csv_sha256=str(payload["source_csv_sha256"]),
        population_sha256=str(payload["population_sha256"]),
        population_hash=str(payload["population_hash"]),
        split_hash=str(payload["split_hash"]),
        feature_paths=feature_paths,
        feature_hashes=feature_hashes,
        mfp_alignment=payload.get("mfp_alignment") or None,
    )


def load_or_prepare_paper_exact_material(
    reference_root: Path | str,
    output_root: Path | str,
    population_id: str,
    *,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
) -> PaperExactMaterial:
    """Load existing material, or prepare it if missing/explicitly overwritten."""
    if not overwrite:
        try:
            return load_existing_paper_exact_material(output_root, population_id, stamp=stamp)
        except FileNotFoundError:
            pass
    return prepare_paper_exact_material(
        reference_root,
        output_root,
        population_id,
        stamp=stamp,
        overwrite=overwrite,
    )


def load_or_prepare_all_paper_exact_materials(
    reference_root: Path | str,
    output_root: Path | str,
    *,
    population_ids: Optional[Sequence[str]] = None,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
) -> list[PaperExactMaterial]:
    ids = tuple(population_ids or PAPER_EXACT_POPULATIONS.keys())
    return [
        load_or_prepare_paper_exact_material(
            reference_root,
            output_root,
            population_id,
            stamp=stamp,
            overwrite=overwrite,
        )
        for population_id in ids
    ]



def _npz_member_dtype(path: Path, member_name: str) -> np.dtype:
    """Read an NPY member dtype from an NPZ header without loading array data."""
    npy_name = member_name if member_name.endswith(".npy") else member_name + ".npy"
    with zipfile.ZipFile(path, "r") as archive:
        if npy_name not in archive.namelist():
            raise PaperExactError("NPZ 缺少成员 {0}：{1}".format(member_name, path))
        with archive.open(npy_name, "r") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    if len(shape) == 0:
        return np.dtype(dtype)
    return np.dtype(dtype)


def _require_npz_member_string_dtype(path: Path, member_name: str) -> np.dtype:
    dtype = _npz_member_dtype(path, member_name)
    if dtype.hasobject:
        raise PaperExactError(
            "不安全的 object dtype NPZ 成员 {0}：{1}；请先运行显式迁移 "
            "--migrate-trusted-mfp-sample-ids，或用修复后的材料生成器重建".format(member_name, path)
        )
    if dtype.kind not in {"U", "S"}:
        raise PaperExactError("NPZ 成员 {0} 必须为固定字符串 dtype，当前为 {1}".format(member_name, dtype))
    return dtype


def validate_mfp_npz_safe_string_dtype(material: PaperExactMaterial) -> dict[str, Any]:
    """Validate MFP NPZ string members without enabling pickle loading."""
    if "mfp" not in material.feature_paths:
        return {"population_id": material.population_id, "feature_id": "mfp", "present": False}
    path = Path(material.feature_paths["mfp"])
    sample_dtype = _require_npz_member_string_dtype(path, "sample_id")
    smiles_dtype = _require_npz_member_string_dtype(path, "smiles_columns")
    with np.load(path, allow_pickle=False) as feature:
        sample_ids = feature["sample_id"].astype(str).tolist()
        source_rows = np.asarray(feature["source_row_index"])
        X = np.asarray(feature["X"])
        y = np.asarray(feature["y"])
        stored_hash = str(feature["feature_hash"])
    population = pd.read_csv(material.population_csv_path)
    expected_ids = population["sample_id"].astype(str).tolist()
    if sample_ids != expected_ids:
        raise PaperExactError("MFP NPZ sample_id 与 population.csv 不一致：{0}".format(path))
    recomputed = _hash_array_payload(X=X, y=y, source_row_index=source_rows)
    if recomputed != stored_hash or stored_hash != str(material.feature_hashes["mfp"]):
        raise PaperExactError("MFP NPZ feature_hash 重新审计失败：{0}".format(path))
    return {
        "population_id": material.population_id,
        "feature_id": "mfp",
        "present": True,
        "path": str(path),
        "sample_id_dtype": str(sample_dtype),
        "smiles_columns_dtype": str(smiles_dtype),
        "feature_hash": stored_hash,
        "row_count": int(len(sample_ids)),
    }


def validate_paper_exact_autogluon_prerequisites(materials: Sequence[PaperExactMaterial]) -> dict[str, Any]:
    """Fail fast before training if any material requires unsafe pickle loading."""
    validations = [validate_mfp_npz_safe_string_dtype(material) for material in materials]
    return {
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "material_count": int(len(materials)),
        "mfp_npz_validations": validations,
        "status": "ready",
    }


def migrate_trusted_mfp_npz_sample_ids_to_unicode(
    material: PaperExactMaterial,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rewrite trusted paper_exact MFP NPZ string members to fixed Unicode dtype.

    This is intentionally explicit because it uses ``allow_pickle=True`` only to
    read NPZ files that were generated by this project's earlier stage.  It
    re-audits X/y/source-row hashes before and after rewriting and refuses to
    alter unknown feature content.
    """
    if "mfp" not in material.feature_paths:
        return {"population_id": material.population_id, "feature_id": "mfp", "present": False, "changed": False}
    path = Path(material.feature_paths["mfp"])
    sample_dtype = _npz_member_dtype(path, "sample_id")
    smiles_dtype = _npz_member_dtype(path, "smiles_columns")
    if not sample_dtype.hasobject and not smiles_dtype.hasobject:
        validation = validate_mfp_npz_safe_string_dtype(material)
        validation["changed"] = False
        return validation
    if not overwrite:
        raise PaperExactError(
            "MFP NPZ 含 object dtype；迁移会重写受信材料文件，请显式传入 overwrite=True：{0}".format(path)
        )
    with np.load(path, allow_pickle=True) as feature:
        arrays = {name: feature[name] for name in feature.files}
    required = {"X", "y", "sample_id", "source_row_index", "smiles_columns", "feature_hash"}
    missing = required.difference(arrays)
    if missing:
        raise PaperExactError("MFP NPZ 缺少成员，拒绝迁移：{0}".format(", ".join(sorted(missing))))
    stored_hash = str(arrays["feature_hash"])
    before_hash = _hash_array_payload(X=arrays["X"], y=arrays["y"], source_row_index=arrays["source_row_index"])
    if before_hash != stored_hash or stored_hash != str(material.feature_hashes["mfp"]):
        raise PaperExactError("迁移前 feature_hash 重新审计失败：{0}".format(path))
    population = pd.read_csv(material.population_csv_path)
    sample_ids = [str(value) for value in arrays["sample_id"].tolist()]
    expected_ids = population["sample_id"].astype(str).tolist()
    if sample_ids != expected_ids:
        raise PaperExactError("迁移前 sample_id 与 population.csv 不一致：{0}".format(path))
    arrays["sample_id"] = _fixed_unicode_array(sample_ids)
    arrays["smiles_columns"] = _fixed_unicode_array([str(value) for value in arrays["smiles_columns"].tolist()])
    after_hash = _hash_array_payload(X=arrays["X"], y=arrays["y"], source_row_index=arrays["source_row_index"])
    if after_hash != before_hash:
        raise PaperExactError("迁移改变了特征 hash，拒绝写回：{0}".format(path))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    validation = validate_mfp_npz_safe_string_dtype(material)
    validation.update({
        "changed": True,
        "before_sample_id_dtype": str(sample_dtype),
        "before_smiles_columns_dtype": str(smiles_dtype),
        "feature_hash_before": before_hash,
        "feature_hash_after": after_hash,
    })
    return validation


def migrate_all_trusted_mfp_npz_sample_ids_to_unicode(
    materials: Sequence[PaperExactMaterial],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    results = [migrate_trusted_mfp_npz_sample_ids_to_unicode(material, overwrite=overwrite) for material in materials]
    return {
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "material_count": int(len(materials)),
        "changed_count": int(sum(1 for item in results if item.get("changed"))),
        "results": results,
        "status": "migrated" if any(item.get("changed") for item in results) else "already_ready",
    }


def _selected_feature_ids(population_id: str, feature_ids: Optional[Sequence[str]]) -> tuple[str, ...]:
    spec = get_population_spec(population_id)
    if feature_ids is None:
        return tuple(spec.feature_ids)
    selected = tuple(str(item).lower() for item in feature_ids)
    invalid = [item for item in selected if item not in spec.feature_ids]
    if invalid:
        raise PaperExactError("{0} 不支持特征：{1}".format(population_id, ", ".join(invalid)))
    if not selected:
        raise PaperExactError("feature_ids 不能为空")
    return selected


def _fold_pairs(split_manifest: pd.DataFrame) -> pd.DataFrame:
    pairs = split_manifest[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    if len(pairs) != PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR:
        raise PaperExactError("paper_exact split manifest 必须有 25 个 repeat/fold")
    return pairs


def _fold_feature_matrices(
    material: PaperExactMaterial,
    population: pd.DataFrame,
    sample_ids: Sequence[str],
    source_row_index: np.ndarray,
    feature_id: str,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, str, dict[str, Any]]:
    spec = get_population_spec(material.population_id)
    feature_hash = str(material.feature_hashes[feature_id])
    if feature_id == "mfp":
        with np.load(material.feature_paths["mfp"], allow_pickle=False) as feature:
            X = np.asarray(feature["X"])
            feature_sample_ids = feature["sample_id"].astype(str).tolist()
        if feature_sample_ids != [str(item) for item in sample_ids]:
            raise PaperExactError("{0} MFP sample_id 与 population.csv 不一致".format(material.population_id))
        X_train = X[train_idx]
        X_valid = X[valid_idx]
        feature_audit: dict[str, Any] = {
            "feature_id": "mfp",
            "fit_scope": "precomputed_from_paper_csv",
            "global_feature_hash": feature_hash,
            "component_cols": list(spec.component_cols),
        }
    elif feature_id == "ohe":
        transformer = ReactionComponentOHE(spec.component_cols, missing_token="__MISSING__")
        component_frame = population.loc[:, list(spec.component_cols)]
        transformer.fit(component_frame.iloc[train_idx], sample_ids=[sample_ids[index] for index in train_idx])
        X_train = transformer.transform(component_frame.iloc[train_idx], partition="train")
        X_valid = transformer.transform(component_frame.iloc[valid_idx], partition="valid")
        feature_audit = transformer.metadata()
        feature_audit["global_feature_hash"] = feature_hash
    else:
        raise PaperExactError("paper_exact AutoGluon 暂不支持特征：{0}".format(feature_id))
    if X_train.ndim != 2 or X_valid.ndim != 2 or X_train.shape[1] != X_valid.shape[1]:
        raise PaperExactError("{0}/{1} 折级特征矩阵维度非法".format(material.population_id, feature_id))
    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise PaperExactError("{0}/{1} 折级特征矩阵包含 NaN 或 inf".format(material.population_id, feature_id))
    feature_dim = int(X_train.shape[1])
    fold_feature_hash = _hash_payload({
        "feature_id": feature_id,
        "global_feature_hash": feature_hash,
        "feature_dim": feature_dim,
        "train_source_row_index": [int(source_row_index[index]) for index in train_idx],
        "valid_source_row_index": [int(source_row_index[index]) for index in valid_idx],
        "feature_audit": feature_audit,
    })
    feature_audit["fold_feature_hash"] = fold_feature_hash
    feature_audit["feature_dim"] = feature_dim
    return X_train, X_valid, feature_dim, fold_feature_hash, feature_audit



def _quarantine_incomplete_artifact(artifact_path: Path, ag_dir: Path, suffix: str) -> Optional[Path]:
    """Move an unregistered AutoGluon model artifact aside without deleting it."""
    if not artifact_path.exists():
        return None
    quarantine_root = ag_dir / "incomplete_artifacts"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = quarantine_root / "{0}__quarantine_{1}".format(suffix, stamp)
    counter = 1
    while target.exists():
        target = quarantine_root / "{0}__quarantine_{1}_{2}".format(suffix, stamp, counter)
        counter += 1
    shutil.move(str(artifact_path), str(target))
    return target


def _read_existing_autogluon_fold(
    metadata_path: Path,
    prediction_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not metadata_path.exists() and not prediction_path.exists():
        raise FileNotFoundError
    if not metadata_path.exists() or not prediction_path.exists():
        raise PaperExactError("发现 AutoGluon 不完整折输出，拒绝静默重跑：{0}".format(metadata_path.parent))
    metadata = _read_json(metadata_path)
    prediction = pd.read_csv(prediction_path)
    row = metadata.get("fold_metric_row")
    if not isinstance(row, dict):
        raise PaperExactError("已有 AutoGluon fold metadata 缺少 fold_metric_row：{0}".format(metadata_path))
    required_prediction_cols = {
        "population_id", "dataset_id", "feature_id", "model", "evaluation_protocol",
        "repeat", "fold", "seed", "sample_id", "source_row_index", "y_true", "y_pred",
    }
    if prediction.empty or not required_prediction_cols.issubset(prediction.columns):
        raise PaperExactError("已有 AutoGluon 预测分片字段不完整：{0}".format(prediction_path))
    if set(prediction["evaluation_protocol"].astype(str)) != {PAPER_EXACT_EVALUATION_PROTOCOL}:
        raise PaperExactError("已有 AutoGluon 预测分片协议不是 paper_exact_5x5：{0}".format(prediction_path))
    return row, prediction


def run_autogluon_for_material(
    material: PaperExactMaterial,
    *,
    feature_ids: Optional[Sequence[str]] = None,
    time_limit: int = 300,
    presets: str = "medium_quality",
    num_cpus: int = 19,
    random_state: int = 42,
    cleanup: bool = True,
    overwrite: bool = False,
    max_folds_per_task: Optional[int] = None,
    quarantine_incomplete_artifacts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run AutoGluon for one population over manifest-defined paper_exact folds."""
    if max_folds_per_task is not None and int(max_folds_per_task) < 1:
        raise PaperExactError("max_folds_per_task 必须 >= 1")
    spec = get_population_spec(material.population_id)
    selected_features = _selected_feature_ids(material.population_id, feature_ids)
    population = pd.read_csv(material.population_csv_path)
    split_manifest = pd.read_csv(material.split_manifest_path)
    sample_ids = population["sample_id"].astype(str).tolist()
    y = population[spec.label_col].to_numpy(dtype=float)
    source_row_index = population["source_row_index"].astype(int).to_numpy()
    fold_pairs = _fold_pairs(split_manifest)
    if max_folds_per_task is not None:
        fold_pairs = fold_pairs.iloc[: int(max_folds_per_task)].copy()

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    ag_dir = material.population_dir / "autogluon"

    for feature_id in selected_features:
        feature_hash = str(material.feature_hashes[feature_id])
        for repeat, fold in fold_pairs.itertuples(index=False, name=None):
            repeat = int(repeat)
            fold = int(fold)
            suffix = "{0}__autogluon__r{1:02d}__f{2:02d}".format(feature_id, repeat, fold)
            metadata_path = ag_dir / "folds" / (suffix + ".json")
            prediction_path = ag_dir / "predictions" / (suffix + ".csv")
            if not overwrite:
                try:
                    row, prediction_frame = _read_existing_autogluon_fold(metadata_path, prediction_path)
                    fold_rows.append(row)
                    prediction_frames.append(prediction_frame)
                    continue
                except FileNotFoundError:
                    pass

            train_idx, valid_idx, seed = _indices_for_fold(split_manifest, sample_ids, repeat, fold)
            X_train, X_valid, feature_dim, fold_feature_hash, feature_audit = _fold_feature_matrices(
                material,
                population,
                sample_ids,
                source_row_index,
                feature_id,
                train_idx,
                valid_idx,
            )
            artifact_path = ag_dir / "models" / suffix
            quarantined_artifact_path: Optional[Path] = None
            if artifact_path.exists() and not overwrite and not metadata_path.exists():
                if quarantine_incomplete_artifacts:
                    quarantined_artifact_path = _quarantine_incomplete_artifact(artifact_path, ag_dir, suffix)
                else:
                    raise PaperExactError("发现未登记的 AutoGluon artifact，拒绝覆盖：{0}".format(artifact_path))

            adapter = AutoGluonYieldModel(
                time_limit=int(time_limit),
                presets=str(presets),
                num_cpus=int(num_cpus),
                random_state=int(random_state),
                cleanup=bool(cleanup),
            )
            started = time.perf_counter()
            prediction, adapter_metadata = adapter.fit_predict_fold(
                X_train,
                y[train_idx],
                X_valid,
                fold_index=(repeat - 1) * 5 + fold,
                context={
                    "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                    "protocol_family": "manifest_outer_cv",
                    "outer_seed": int(seed),
                    "model_artifact_path": str(artifact_path),
                    "cleanup": bool(cleanup),
                },
            )
            elapsed = time.perf_counter() - started
            prediction = np.asarray(prediction, dtype=float)
            if prediction.ndim != 1:
                prediction = prediction.reshape(-1)
            if len(prediction) != len(valid_idx) or not np.isfinite(prediction).all():
                raise PaperExactError("AutoGluon 预测长度或数值非法：{0}".format(suffix))
            metrics = _paper_metric_values(y[valid_idx], prediction)
            split_hash = str(split_manifest["split_hash"].iloc[0])
            versions = dict(adapter_metadata.get("autogluon_versions", {}) or {})
            train_ids = [sample_ids[int(index)] for index in train_idx]
            valid_ids = [sample_ids[int(index)] for index in valid_idx]
            train_source_rows = [int(source_row_index[int(index)]) for index in train_idx]
            valid_source_rows = [int(source_row_index[int(index)]) for index in valid_idx]
            row: dict[str, Any] = {
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "autogluon",
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "repeat": repeat,
                "fold": fold,
                "seed": int(seed),
                "split_hash": split_hash,
                "population_hash": material.population_hash,
                "feature_hash": feature_hash,
                "fold_feature_hash": fold_feature_hash,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(valid_idx)),
                "feature_dim": feature_dim,
                "fit_predict_seconds": float(elapsed),
                "train_time_s": float(adapter_metadata.get("train_time_s", 0.0)),
                "predict_time_s": float(adapter_metadata.get("predict_time_s", 0.0)),
                "autogluon_time_limit": int(time_limit),
                "autogluon_presets": str(presets),
                "autogluon_num_cpus": int(num_cpus),
                "autogluon_random_state": int(random_state),
                "autogluon_version": str(versions.get("autogluon.tabular", "unknown")),
                "autogluon_seed_policy": PAPER_EXACT_AUTOGLOON_SEED_POLICY,
                **metrics,
            }
            prediction_frame = pd.DataFrame({
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "autogluon",
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "repeat": repeat,
                "fold": fold,
                "seed": int(seed),
                "split_hash": split_hash,
                "population_hash": material.population_hash,
                "feature_hash": feature_hash,
                "fold_feature_hash": fold_feature_hash,
                "sample_id": valid_ids,
                "source_row_index": valid_source_rows,
                "y_true": y[valid_idx].astype(float),
                "y_pred": prediction.astype(float),
            })
            metadata = {
                "schema_version": 1,
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "protocol_family": "manifest_outer_cv",
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "autogluon",
                "repeat": repeat,
                "fold": fold,
                "seed": int(seed),
                "manifest_seed": int(seed),
                "outer_seed": int(seed),
                "split_hash": split_hash,
                "population_hash": material.population_hash,
                "feature_hash": feature_hash,
                "fold_feature_hash": fold_feature_hash,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(valid_idx)),
                "feature_dim": feature_dim,
                "train_sample_ids_hash": _hash_payload(train_ids),
                "valid_sample_ids_hash": _hash_payload(valid_ids),
                "train_source_row_index_hash": _hash_payload(train_source_rows),
                "valid_source_row_index_hash": _hash_payload(valid_source_rows),
                "autogluon_params": {
                    "time_limit": int(time_limit),
                    "presets": str(presets),
                    "num_cpus": int(num_cpus),
                    "random_state": int(random_state),
                    "cleanup": bool(cleanup),
                },
                "autogluon_seed_policy": PAPER_EXACT_AUTOGLOON_SEED_POLICY,
                "autogluon_versions": versions,
                "autogluon_version": str(versions.get("autogluon.tabular", "unknown")),
                "model_artifact_path": str(artifact_path),
                "model_artifact_cleanup": bool(cleanup),
                "quarantined_incomplete_artifact_path": str(quarantined_artifact_path) if quarantined_artifact_path else None,
                "feature_audit": feature_audit,
                "feature_transformer": feature_audit if feature_id == "ohe" else None,
                "metrics": metrics,
                "fold_metric_row": row,
                "prediction_path": str(prediction_path),
                "model_adapter_metadata": adapter_metadata,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_csv(prediction_path, prediction_frame, overwrite=overwrite)
            _write_json(metadata_path, metadata, overwrite=overwrite)
            fold_rows.append(row)
            prediction_frames.append(prediction_frame)

    if not fold_rows:
        raise PaperExactError("AutoGluon 未产生任何折级结果")
    fold_frame = pd.DataFrame.from_records(fold_rows).sort_values(
        ["population_id", "feature_id", "repeat", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["population_id", "feature_id", "repeat", "fold", "source_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    suffix = "_partial" if max_folds_per_task is not None else ""
    _write_csv(ag_dir / ("fold_metrics{0}.csv".format(suffix)), fold_frame, overwrite=overwrite)
    _write_csv(ag_dir / ("predictions{0}.csv".format(suffix)), prediction_frame, overwrite=overwrite)
    summary = _summary_from_model_folds(fold_frame)
    _write_csv(ag_dir / ("summary{0}.csv".format(suffix)), summary, overwrite=overwrite)
    return fold_frame, prediction_frame


def _expected_fold_records(
    materials: Sequence[PaperExactMaterial],
    *,
    model: str,
    feature_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for material in materials:
        selected = _selected_feature_ids(material.population_id, feature_ids)
        split_manifest = pd.read_csv(material.split_manifest_path)
        pairs = split_manifest[["repeat", "fold", "seed", "split_hash"]].drop_duplicates().sort_values(["repeat", "fold"])
        if len(pairs) != PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR:
            raise PaperExactError("{0} split manifest 不完整".format(material.population_id))
        spec = get_population_spec(material.population_id)
        for feature_id in selected:
            for row in pairs.itertuples(index=False):
                records.append({
                    "population_id": material.population_id,
                    "dataset_id": spec.dataset_id,
                    "feature_id": feature_id,
                    "model": model,
                    "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                    "repeat": int(row.repeat),
                    "fold": int(row.fold),
                    "seed": int(row.seed),
                    "split_hash": str(row.split_hash),
                })
    return pd.DataFrame.from_records(records)


def validate_complete_paper_exact_model_folds(
    fold_metrics: pd.DataFrame,
    materials: Sequence[PaperExactMaterial],
    *,
    model: str,
    feature_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Strictly require every model/feature task to contain all 25 folds."""
    expected = _expected_fold_records(materials, model=model, feature_ids=feature_ids)
    if expected.empty:
        raise PaperExactError("paper_exact 完整性校验没有期望折")
    required_cols = set(expected.columns)
    missing_cols = required_cols.difference(fold_metrics.columns)
    if missing_cols:
        raise PaperExactError("{0} 折级指标缺少列：{1}".format(model, ", ".join(sorted(missing_cols))))
    actual = fold_metrics.loc[
        (fold_metrics["model"].astype(str) == model)
        & (fold_metrics["evaluation_protocol"].astype(str) == PAPER_EXACT_EVALUATION_PROTOCOL),
        list(expected.columns),
    ].copy()
    keys = ["population_id", "feature_id", "model", "repeat", "fold"]
    duplicate_count = int(actual.duplicated(keys).sum())
    expected_keys = {tuple(row) for row in expected[keys].itertuples(index=False, name=None)}
    actual_keys = {tuple(row) for row in actual[keys].itertuples(index=False, name=None)}
    missing = sorted(expected_keys.difference(actual_keys))
    unexpected = sorted(actual_keys.difference(expected_keys))
    if duplicate_count or missing or unexpected or len(actual) != len(expected):
        raise PaperExactError(
            "paper_exact {0} fold 不完整：expected={1}, completed={2}, duplicates={3}, "
            "missing_head={4}, unexpected_head={5}".format(
                model,
                len(expected),
                len(actual),
                duplicate_count,
                missing[:5],
                unexpected[:5],
            )
        )
    merged = expected.merge(actual, how="left", on=list(expected.columns), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise PaperExactError("paper_exact {0} seed/split_hash 与 manifest 不一致".format(model))
    return {
        "model": model,
        "expected_fold_count": int(len(expected)),
        "completed_fold_count": int(len(actual)),
        "descriptor_task_count": int(len(expected) // PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR),
        "expected_folds_per_descriptor": PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
        "status": "complete",
    }


def _summary_from_model_folds(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (population_id, dataset_id, feature_id, model), part in frame.groupby(
        ["population_id", "dataset_id", "feature_id", "model"], sort=True
    ):
        completed = int(len(part))
        status = "complete" if completed == PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR else "incomplete"
        for metric in ("mae", "rmse", "r2", "kendall_tau"):
            values = pd.to_numeric(part[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise PaperExactError("{0}/{1}/{2} summary 指标包含 NaN 或 inf".format(population_id, feature_id, metric))
            rows.append({
                "population_id": population_id,
                "dataset_id": dataset_id,
                "feature_id": feature_id,
                "model": model,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "valid_fold_count": completed,
                "expected_folds": PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
                "completed_folds": completed,
                "paper_exact_status": status,
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
            })
    return pd.DataFrame.from_records(rows)


def load_validated_rf_fold_metrics(
    reference_root: Path | str,
    output_root: Path | str,
    materials: Sequence[PaperExactMaterial],
    *,
    stamp: str = PAPER_EXACT_STAMP,
    feature_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load the accepted RF gate outputs before allowing formal AG pairing."""
    batch_dir = paper_exact_batch_dir(output_root, stamp=stamp)
    alignment_path = batch_dir / "rf_alignment.csv"
    fold_metrics_path = batch_dir / "rf_fold_metrics.csv"
    if not alignment_path.is_file() or not fold_metrics_path.is_file():
        raise FileNotFoundError("缺少 RF gate 输出，请先运行 paper_exact RF 对齐：{0}".format(batch_dir))
    alignment = pd.read_csv(alignment_path)
    if alignment.empty or "passes_tolerance" not in alignment.columns:
        raise PaperExactError("RF alignment 文件不完整：{0}".format(alignment_path))
    passes = alignment["passes_tolerance"]
    if passes.dtype != bool:
        passes = passes.astype(str).str.lower().isin({"true", "1", "yes"})
    if not bool(passes.all()):
        failed = alignment.loc[~passes]
        raise PaperExactError("RF gate 未全部通过，禁止启动 AutoGluon：\n{0}".format(failed.to_string(index=False)))
    rf = pd.read_csv(fold_metrics_path)
    validate_complete_paper_exact_model_folds(rf, materials, model="rf", feature_ids=feature_ids)
    return rf


def build_rf_autogluon_paired_input(
    rf_fold_metrics: pd.DataFrame,
    autogluon_fold_metrics: pd.DataFrame,
    materials: Sequence[PaperExactMaterial],
    *,
    feature_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Create fold-level paired-comparison input after strict completeness checks."""
    validate_complete_paper_exact_model_folds(rf_fold_metrics, materials, model="rf", feature_ids=feature_ids)
    validate_complete_paper_exact_model_folds(autogluon_fold_metrics, materials, model="autogluon", feature_ids=feature_ids)
    key_cols = [
        "population_id", "dataset_id", "feature_id", "evaluation_protocol",
        "repeat", "fold", "seed", "split_hash", "population_hash",
    ]
    rf = rf_fold_metrics.loc[rf_fold_metrics["model"].astype(str) == "rf"].copy()
    ag = autogluon_fold_metrics.loc[autogluon_fold_metrics["model"].astype(str) == "autogluon"].copy()
    merged = rf.merge(
        ag,
        on=key_cols,
        suffixes=("_rf", "_autogluon"),
        how="outer",
        indicator=True,
    )
    expected = _expected_fold_records(materials, model="autogluon", feature_ids=feature_ids)
    if len(merged) != len(expected) or not merged["_merge"].eq("both").all():
        raise PaperExactError("RF 与 AutoGluon 不能形成完整 fold-level 配对")
    rows: list[dict[str, Any]] = []
    for row in merged.sort_values(["population_id", "feature_id", "repeat", "fold"]).itertuples(index=False):
        payload = {column: getattr(row, column) for column in key_cols}
        payload.update({
            "rf_feature_hash": getattr(row, "feature_hash_rf", None),
            "autogluon_feature_hash": getattr(row, "feature_hash_autogluon", None),
            "autogluon_fold_feature_hash": getattr(row, "fold_feature_hash", None),
            "rf_train_rows": int(getattr(row, "train_rows_rf")),
            "rf_test_rows": int(getattr(row, "test_rows_rf")),
            "autogluon_train_rows": int(getattr(row, "train_rows_autogluon")),
            "autogluon_test_rows": int(getattr(row, "test_rows_autogluon")),
            "rf_rmse": float(getattr(row, "rmse_rf")),
            "autogluon_rmse": float(getattr(row, "rmse_autogluon")),
            "delta_rmse_autogluon_minus_rf": float(getattr(row, "rmse_autogluon") - getattr(row, "rmse_rf")),
            "rf_mae": float(getattr(row, "mae_rf")),
            "autogluon_mae": float(getattr(row, "mae_autogluon")),
            "delta_mae_autogluon_minus_rf": float(getattr(row, "mae_autogluon") - getattr(row, "mae_rf")),
            "rf_r2": float(getattr(row, "r2_rf")),
            "autogluon_r2": float(getattr(row, "r2_autogluon")),
            "delta_r2_autogluon_minus_rf": float(getattr(row, "r2_autogluon") - getattr(row, "r2_rf")),
            "rf_kendall_tau": float(getattr(row, "kendall_tau_rf")),
            "autogluon_kendall_tau": float(getattr(row, "kendall_tau_autogluon")),
            "delta_kendall_tau_autogluon_minus_rf": float(
                getattr(row, "kendall_tau_autogluon") - getattr(row, "kendall_tau_rf")
            ),
            "paired_comparison_ready": True,
            "expected_paired_folds_per_descriptor": PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
        })
        rows.append(payload)
    return pd.DataFrame.from_records(rows)


def _autogluon_config_for_spec(
    population_id: str,
    *,
    time_limit: int,
    presets: str,
    num_cpus: int,
    random_state: int,
    cleanup: bool,
) -> dict[str, Any]:
    spec = get_population_spec(population_id)
    config = build_paper_exact_config(spec, outputs_root="autogluon")
    block = config["benchmark"]
    block["models"] = ["autogluon"]
    block["model_params"] = {
        "autogluon": {
            "time_limit": int(time_limit),
            "presets": str(presets),
            "num_cpus": int(num_cpus),
            "random_state": int(random_state),
            "cleanup": bool(cleanup),
            "keep_autogluon_artifacts": bool(not cleanup),
        }
    }
    block["reproduction_protocol"]["autogluon_params"] = block["model_params"]["autogluon"]
    block["reproduction_protocol"]["autogluon_seed_policy"] = PAPER_EXACT_AUTOGLOON_SEED_POLICY
    block["paper_exact"]["rf_alignment_required_before_formal_ag"] = True
    return config


def write_paper_exact_autogluon_launch_materials(
    reference_root: Path | str,
    output_root: Path | str,
    *,
    population_ids: Optional[Sequence[str]] = None,
    stamp: str = PAPER_EXACT_STAMP,
    time_limit: int = 300,
    presets: str = "medium_quality",
    num_cpus: int = 19,
    random_state: int = 42,
    cleanup: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write AG configs and launch manifests without starting training."""
    output_root = Path(output_root).resolve()
    materials = load_or_prepare_all_paper_exact_materials(
        reference_root,
        output_root,
        population_ids=population_ids,
        stamp=stamp,
        overwrite=overwrite,
    )
    rows: list[dict[str, Any]] = []
    config_paths: dict[str, str] = {}
    for material in materials:
        spec = get_population_spec(material.population_id)
        config = _autogluon_config_for_spec(
            material.population_id,
            time_limit=time_limit,
            presets=presets,
            num_cpus=num_cpus,
            random_state=random_state,
            cleanup=cleanup,
        )
        config_path = material.population_dir / "paper_exact_autogluon_config.json"
        _write_json(config_path, config, overwrite=overwrite)
        config_paths[material.population_id] = str(config_path)
        task_rows = []
        for feature_id in spec.feature_ids:
            row = {
                "population_id": material.population_id,
                "dataset_id": spec.dataset_id,
                "feature_id": feature_id,
                "model": "autogluon",
                "expected_folds": PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "split_hash": material.split_hash,
                "population_hash": material.population_hash,
                "feature_hash": material.feature_hashes[feature_id],
                "autogluon_time_limit": int(time_limit),
                "autogluon_presets": str(presets),
                "autogluon_num_cpus": int(num_cpus),
                "autogluon_seed_policy": PAPER_EXACT_AUTOGLOON_SEED_POLICY,
            }
            rows.append(row)
            task_rows.append(row)
        _write_csv(material.population_dir / "tasks" / "autogluon_descriptor_tasks.csv", pd.DataFrame.from_records(task_rows), overwrite=overwrite)

    batch_dir = paper_exact_batch_dir(output_root, stamp=stamp)
    tasks = pd.DataFrame.from_records(rows)
    tasks_path = batch_dir / "autogluon_descriptor_tasks.csv"
    _write_csv(tasks_path, tasks, overwrite=overwrite)
    launch_manifest = {
        "schema_version": 1,
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "population_ids": [material.population_id for material in materials],
        "descriptor_task_count": int(len(tasks)),
        "expected_autogluon_fold_fits": int(len(tasks) * PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR),
        "seeds": list(PAPER_EXACT_SEEDS),
        "rf_gate_required": True,
        "rf_gate_files": {
            "alignment": str(batch_dir / "rf_alignment.csv"),
            "fold_metrics": str(batch_dir / "rf_fold_metrics.csv"),
        },
        "autogluon_params": {
            "time_limit": int(time_limit),
            "presets": str(presets),
            "num_cpus": int(num_cpus),
            "random_state": int(random_state),
            "cleanup": bool(cleanup),
        },
        "autogluon_seed_policy": PAPER_EXACT_AUTOGLOON_SEED_POLICY,
        "population_dirs": {material.population_id: str(material.population_dir) for material in materials},
        "autogluon_config_paths": config_paths,
        "formal_nohup_launcher": "bash result/YSNH文献复现/yieldmaster_paper_exact_5x5_{0}/run_autogluon_paper_exact_nohup.sh".format(stamp),
        "notes": "本 manifest 只描述正式启动材料；真实运行时 git commit、开始/结束时间和耗时由 runner 写入 run_manifest。",
    }
    manifest_path = batch_dir / "autogluon_launch_manifest.json"
    _write_json(manifest_path, launch_manifest, overwrite=overwrite)
    return {
        "batch_dir": batch_dir,
        "tasks_path": tasks_path,
        "manifest_path": manifest_path,
        "config_paths": config_paths,
        "descriptor_task_count": int(len(tasks)),
        "expected_autogluon_fold_fits": int(len(tasks) * PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR),
    }


def run_paper_exact_autogluon_matrix(
    reference_root: Path | str,
    output_root: Path | str,
    *,
    population_ids: Optional[Sequence[str]] = None,
    feature_ids: Optional[Sequence[str]] = None,
    stamp: str = PAPER_EXACT_STAMP,
    overwrite: bool = False,
    time_limit: int = 300,
    presets: str = "medium_quality",
    num_cpus: int = 19,
    random_state: int = 42,
    cleanup: bool = True,
    require_rf_alignment: bool = True,
    allow_partial: bool = False,
    max_folds_per_task: Optional[int] = None,
    quarantine_incomplete_artifacts: bool = False,
) -> PaperExactAutoGluonRun:
    """Run the formal AG matrix, blocking strict pairing unless all folds complete."""
    reference_root = Path(reference_root).resolve()
    output_root = Path(output_root).resolve()
    materials = load_or_prepare_all_paper_exact_materials(
        reference_root,
        output_root,
        population_ids=population_ids,
        stamp=stamp,
        overwrite=overwrite,
    )
    validate_paper_exact_autogluon_prerequisites(materials)
    write_paper_exact_autogluon_launch_materials(
        reference_root,
        output_root,
        population_ids=[material.population_id for material in materials],
        stamp=stamp,
        time_limit=time_limit,
        presets=presets,
        num_cpus=num_cpus,
        random_state=random_state,
        cleanup=cleanup,
        overwrite=overwrite,
    )
    rf_fold_metrics: Optional[pd.DataFrame] = None
    if require_rf_alignment:
        rf_fold_metrics = load_validated_rf_fold_metrics(
            reference_root,
            output_root,
            materials,
            stamp=stamp,
            feature_ids=feature_ids,
        )

    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    started = time.perf_counter()
    for material in materials:
        folds, predictions = run_autogluon_for_material(
            material,
            feature_ids=feature_ids,
            time_limit=time_limit,
            presets=presets,
            num_cpus=num_cpus,
            random_state=random_state,
            cleanup=cleanup,
            overwrite=overwrite,
            max_folds_per_task=max_folds_per_task,
            quarantine_incomplete_artifacts=quarantine_incomplete_artifacts,
        )
        all_fold_metrics.append(folds)
        all_predictions.append(predictions)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    expected = _expected_fold_records(materials, model="autogluon", feature_ids=feature_ids)
    completion: dict[str, Any]
    status = "complete"
    try:
        completion = validate_complete_paper_exact_model_folds(
            fold_metrics,
            materials,
            model="autogluon",
            feature_ids=feature_ids,
        )
    except PaperExactError:
        if not allow_partial:
            raise
        status = "incomplete"
        completion = {
            "model": "autogluon",
            "expected_fold_count": int(len(expected)),
            "completed_fold_count": int(len(fold_metrics)),
            "descriptor_task_count": int(len(expected) // PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR),
            "expected_folds_per_descriptor": PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
            "status": status,
        }
    summary = _summary_from_model_folds(fold_metrics)
    batch_dir = paper_exact_batch_dir(output_root, stamp=stamp)
    suffix = "_partial" if status != "complete" else ""
    fold_metrics_path = batch_dir / ("autogluon_fold_metrics{0}.csv".format(suffix))
    predictions_path = batch_dir / ("autogluon_predictions{0}.csv".format(suffix))
    summary_path = batch_dir / ("autogluon_summary{0}.csv".format(suffix))
    manifest_path = batch_dir / ("paper_exact_autogluon_run_manifest{0}.json".format(suffix))
    _write_csv(fold_metrics_path, fold_metrics, overwrite=overwrite)
    _write_csv(predictions_path, predictions, overwrite=overwrite)
    _write_csv(summary_path, summary, overwrite=overwrite)

    paired_input: Optional[pd.DataFrame] = None
    paired_input_path: Optional[Path] = None
    if status == "complete" and require_rf_alignment:
        if rf_fold_metrics is None:
            rf_fold_metrics = load_validated_rf_fold_metrics(
                reference_root,
                output_root,
                materials,
                stamp=stamp,
                feature_ids=feature_ids,
            )
        paired_input = build_rf_autogluon_paired_input(
            rf_fold_metrics,
            fold_metrics,
            materials,
            feature_ids=feature_ids,
        )
        paired_input_path = batch_dir / "rf_vs_autogluon_paired_fold_input.csv"
        _write_csv(paired_input_path, paired_input, overwrite=overwrite)

    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "python": sys.version,
        "platform": platform.platform(),
        "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
        "status": status,
        "population_ids": [material.population_id for material in materials],
        "feature_ids_filter": list(feature_ids) if feature_ids is not None else None,
        "expected_fold_count": int(completion["expected_fold_count"]),
        "completed_fold_count": int(completion["completed_fold_count"]),
        "descriptor_task_count": int(completion["descriptor_task_count"]),
        "seeds": list(PAPER_EXACT_SEEDS),
        "rf_alignment_required": bool(require_rf_alignment),
        "allow_partial": bool(allow_partial),
        "max_folds_per_task": max_folds_per_task,
        "quarantine_incomplete_artifacts": bool(quarantine_incomplete_artifacts),
        "autogluon_params": {
            "time_limit": int(time_limit),
            "presets": str(presets),
            "num_cpus": int(num_cpus),
            "random_state": int(random_state),
            "cleanup": bool(cleanup),
        },
        "autogluon_seed_policy": PAPER_EXACT_AUTOGLOON_SEED_POLICY,
        "fit_predict_wall_seconds": float(time.perf_counter() - started),
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "paired_input": str(paired_input_path) if paired_input_path else None,
        },
    }
    _write_json(manifest_path, run_manifest, overwrite=overwrite)
    return PaperExactAutoGluonRun(
        output_root=batch_dir,
        fold_metrics_path=fold_metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        paired_input_path=paired_input_path,
        manifest_path=manifest_path,
        fold_metrics=fold_metrics,
        summary=summary,
        paired_input=paired_input,
        expected_fold_count=int(completion["expected_fold_count"]),
        completed_fold_count=int(completion["completed_fold_count"]),
        status=status,
    )


__all__ = [
    "PAPER_EXACT_AUTOGLOON_PARAMS",
    "PAPER_EXACT_AUTOGLOON_SEED_POLICY",
    "PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR",
    "PaperExactAutoGluonRun",
    "build_rf_autogluon_paired_input",
    "load_existing_paper_exact_material",
    "load_or_prepare_all_paper_exact_materials",
    "load_or_prepare_paper_exact_material",
    "migrate_all_trusted_mfp_npz_sample_ids_to_unicode",
    "migrate_trusted_mfp_npz_sample_ids_to_unicode",
    "load_validated_rf_fold_metrics",
    "run_autogluon_for_material",
    "run_paper_exact_autogluon_matrix",
    "validate_complete_paper_exact_model_folds",
    "validate_mfp_npz_safe_string_dtype",
    "validate_paper_exact_autogluon_prerequisites",
    "write_paper_exact_autogluon_launch_materials",
]
