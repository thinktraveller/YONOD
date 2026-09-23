"""Audit paired model runs using their saved inputs and fold states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.reader import load_feature_artifact
from yonod.features.numeric_conditions import load_numeric_block


class PairVerificationError(ValueError):
    """Two runs cannot be interpreted as differing only by allowed blocks."""


def _run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise PairVerificationError(f"训练 manifest 不存在：{manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise PairVerificationError("配对仅接受完整训练结果")
    predictions = pd.read_csv(run_dir / manifest["outputs"]["predictions"])
    metrics = pd.read_csv(run_dir / manifest["outputs"]["fold_metrics"])
    return manifest, predictions, metrics


def _numeric_block(artifact_path: Path, identity: str, sample_ids: np.ndarray):
    block_path = artifact_path.parent.parent.parent / "numeric_blocks" / (identity.removeprefix("sha256:") + ".json")
    return load_numeric_block(block_path, expected_ids=sample_ids, expected_identity=identity)


def verify_paired_runs(
    full_run_dir: Path | str,
    reduced_run_dir: Path | str,
    full_feature_manifest: Path | str,
    reduced_feature_manifest: Path | str,
    *,
    allowed_extra_columns: Sequence[str],
    required_shared_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Check actual shared molecular/numeric blocks and all saved fold states.

    The current comparable static form is one ordered ``concat`` descriptor
    per arm. Any other lifecycle or unequal block width is rejected, never
    interpreted as a passing comparison merely because dimensions match.
    """
    full_dir, reduced_dir = Path(full_run_dir), Path(reduced_run_dir)
    full, full_predictions, full_metrics = _run(full_dir)
    reduced, reduced_predictions, reduced_metrics = _run(reduced_dir)
    for key in ("model", "model_request", "dataset_identity", "label_identity", "split_identity", "evaluation", "n_samples", "n_folds"):
        if full.get(key) != reduced.get(key):
            raise PairVerificationError(f"配对运行 {key} 不一致")
    full_artifact = load_feature_artifact(full_feature_manifest)
    reduced_artifact = load_feature_artifact(reduced_feature_manifest)
    for run, artifact in ((full, full_artifact), (reduced, reduced_artifact)):
        if run["artifact_id"] != artifact.manifest["artifact_id"]:
            raise PairVerificationError("配对运行引用的特征产物不一致")
    if not np.array_equal(full_artifact.sample_ids, reduced_artifact.sample_ids) or not np.array_equal(
        full_artifact.valid_mask, reduced_artifact.valid_mask
    ):
        raise PairVerificationError("配对特征包 sample_id/有效行不一致")
    if full_artifact.manifest["lifecycle"] != "static_descriptor" or reduced_artifact.manifest["lifecycle"] != "static_descriptor":
        raise PairVerificationError("当前配对核验仅支持静态 concat 描述符")
    full_meta, reduced_meta = full_artifact.manifest["metadata"], reduced_artifact.manifest["metadata"]
    full_spec, reduced_spec = full_meta.get("feature_spec"), reduced_meta.get("feature_spec")
    if not isinstance(full_spec, Mapping) or not isinstance(reduced_spec, Mapping):
        raise PairVerificationError("特征包缺少描述符配置")
    for field in ("descriptor", "mode", "params"):
        if full_spec.get(field) != reduced_spec.get(field):
            raise PairVerificationError(f"配对描述符 {field} 不一致")
    if full_spec.get("mode") != "concat":
        raise PairVerificationError("配对描述符必须是可逐列核验的 concat 模式")
    full_columns = list(full_meta.get("resolved_columns") or [])
    reduced_columns = list(reduced_meta.get("resolved_columns") or [])
    allowed = set(allowed_extra_columns)
    if not allowed or len(allowed) != len(allowed_extra_columns):
        raise PairVerificationError("allowed_extra_columns 必须是非空且无重复的列名列表")
    shared = [column for column in full_columns if column not in allowed]
    if shared != reduced_columns or set(full_columns).difference(reduced_columns) != allowed:
        raise PairVerificationError("配对分子输入存在声明外差异或共享列顺序变化")
    if len(set(full_columns)) != len(full_columns) or len(set(reduced_columns)) != len(reduced_columns):
        raise PairVerificationError("描述符源列重复")
    required = set(required_shared_columns)
    actual_shared = set(shared)
    numeric_columns = [entry["source"] for entry in (full.get("numeric_contract") or {}).get("columns", [])]
    actual_shared.update(numeric_columns)
    if not required.issubset(actual_shared):
        raise PairVerificationError("必需共享条件/组分未进入模型：" + ", ".join(sorted(required - actual_shared)))
    full_width = full_artifact.matrix.shape[1] / len(full_columns)
    reduced_width = reduced_artifact.matrix.shape[1] / len(reduced_columns)
    if not full_width.is_integer() or full_width != reduced_width:
        raise PairVerificationError("配对描述符无法分解为相同宽度的列块")
    width = int(full_width)
    for index, column in enumerate(reduced_columns):
        full_index = full_columns.index(column)
        if not np.array_equal(
            full_artifact.matrix[:, full_index * width:(full_index + 1) * width],
            reduced_artifact.matrix[:, index * width:(index + 1) * width],
        ):
            raise PairVerificationError(f"共享分子列 {column!r} 的实际描述符不同")
    if full.get("numeric_identity") != reduced.get("numeric_identity") or full.get("numeric_contract") != reduced.get("numeric_contract"):
        raise PairVerificationError("配对数值块身份或处理契约不一致")
    if full.get("numeric_identity"):
        full_raw, _ = _numeric_block(Path(full_feature_manifest), full["numeric_identity"], full_artifact.sample_ids)
        reduced_raw, _ = _numeric_block(Path(reduced_feature_manifest), reduced["numeric_identity"], reduced_artifact.sample_ids)
        if not full_raw.equals(reduced_raw):
            raise PairVerificationError("配对原始数值块不一致")
    key = ["repeat", "fold", "sample_id"]
    for column in [*key, "y_true"]:
        if column not in full_predictions or column not in reduced_predictions:
            raise PairVerificationError("配对预测缺少样本/标签键")
    left = full_predictions.sort_values(key).reset_index(drop=True)
    right = reduced_predictions.sort_values(key).reset_index(drop=True)
    if not left.loc[:, [*key, "y_true"]].equals(right.loc[:, [*key, "y_true"]]):
        raise PairVerificationError("配对样本、折或真实标签不一致")
    fold_key = ["repeat", "fold"]
    full_metrics = full_metrics.sort_values(fold_key).reset_index(drop=True)
    reduced_metrics = reduced_metrics.sort_values(fold_key).reset_index(drop=True)
    if not full_metrics.loc[:, fold_key].equals(reduced_metrics.loc[:, fold_key]):
        raise PairVerificationError("配对折清单不一致")
    for index, (full_row, reduced_row) in enumerate(zip(full_metrics.itertuples(), reduced_metrics.itertuples())):
        full_audit = json.loads(full_row.transform_audit)
        reduced_audit = json.loads(reduced_row.transform_audit)
        full_numeric, reduced_numeric = full_audit.get("numeric"), reduced_audit.get("numeric")
        if full_numeric != reduced_numeric:
            raise PairVerificationError(f"第 {index + 1} 个配对折的数值处理状态不同")
        if full_numeric and full_numeric.get("columns"):
            repeat, fold = int(full_row.repeat), int(full_row.fold)
            relative = Path("fold_transforms") / f"repeat-{repeat:02d}" / f"fold-{fold:02d}" / "numeric_conditions.yaml"
            full_state = yaml.safe_load((full_dir / relative).read_text(encoding="utf-8"))
            reduced_state = yaml.safe_load((reduced_dir / relative).read_text(encoding="utf-8"))
            if full_state != reduced_state:
                raise PairVerificationError(f"repeat={repeat}/fold={fold} 数值拟合参数不一致")
    return {
        "status": "passed", "shared_molecular_columns": shared,
        "allowed_extra_columns": list(allowed_extra_columns), "shared_numeric_columns": numeric_columns,
        "n_folds": len(full_metrics), "n_predictions": len(left),
        "split_identity": full["split_identity"], "numeric_identity": full.get("numeric_identity"),
    }


def verified_paired_delta_p(
    full_run_dir: Path | str,
    reduced_run_dir: Path | str,
    full_feature_manifest: Path | str,
    reduced_feature_manifest: Path | str,
    *,
    allowed_extra_columns: Sequence[str],
    required_shared_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Compute descriptive paired ΔP only after the saved-input gate passes.

    The comparison is row-weighted OOF MAE, not a significance test or a
    chemical causal claim. A caller that needs grouped inference must apply
    its own predeclared grouping protocol *after* this gate succeeds.
    """
    verified = verify_paired_runs(
        full_run_dir, reduced_run_dir, full_feature_manifest,
        reduced_feature_manifest, allowed_extra_columns=allowed_extra_columns,
        required_shared_columns=required_shared_columns,
    )
    _, full, _ = _run(Path(full_run_dir))
    _, reduced, _ = _run(Path(reduced_run_dir))
    keys = ["repeat", "fold", "sample_id"]
    full = full.sort_values(keys).reset_index(drop=True)
    reduced = reduced.sort_values(keys).reset_index(drop=True)
    full_error = np.abs(full["y_true"].to_numpy(dtype=float) - full["y_pred"].to_numpy(dtype=float))
    reduced_error = np.abs(reduced["y_true"].to_numpy(dtype=float) - reduced["y_pred"].to_numpy(dtype=float))
    if not np.isfinite(full_error).all() or not np.isfinite(reduced_error).all():
        raise PairVerificationError("配对预测包含非有限误差")
    return {
        "verification": verified,
        "estimand": "MAE(reduced) - MAE(full)",
        "full_mae": float(full_error.mean()),
        "reduced_mae": float(reduced_error.mean()),
        "delta_p": float(reduced_error.mean() - full_error.mean()),
        "n_oof_predictions": int(len(full_error)),
        "inference_status": "descriptive_only",
    }
