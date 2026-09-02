"""Rebuild auditable benchmark metrics and comparisons from saved fold outputs.

Nothing here trains a model.  All inputs are the immutable run/split manifests,
prediction shards, and fold metadata emitted by :mod:`yonod.benchmark.executor`.
This boundary lets reports be regenerated in a new Python process and prevents
an in-memory training result from becoming an untraceable conclusion.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricRebuildError(RuntimeError):
    """Raised when saved benchmark artefacts cannot be audited safely."""


FOLD_METRIC_COLUMNS = [
    "run_id", "config_hash", "split_id", "descriptor", "model", "repeat", "fold",
    "n_valid", "r2", "rmse", "mae", "train_time_s", "predict_time_s",
    "prediction_path", "metadata_path",
]
EXCLUSION_COLUMNS = [
    "run_id", "config_hash", "split_id", "descriptor", "model", "repeat", "fold",
    "reason_code", "reason_detail", "prediction_path",
]
COMPLETENESS_COLUMNS = [
    "run_id", "config_hash", "split_id", "descriptor", "model", "expected_folds",
    "available_folds", "valid_metric_folds", "is_complete", "missing_or_excluded_reason",
]
COMBINATION_TIME_SUMMARY_COLUMNS = [
    "run_id", "config_hash", "split_id", "descriptor", "model",
    "expected_folds", "completed_folds", "is_complete", "time_status", "time_status_detail",
    "is_time_comparable", "total_train_time_s", "mean_train_time_s", "median_train_time_s",
    "max_train_time_s", "total_predict_time_s", "total_model_time_s",
]


@dataclass(frozen=True)
class MetricRebuildResult:
    fold_metrics: pd.DataFrame
    exclusions: pd.DataFrame
    completeness: pd.DataFrame


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricRebuildError("无法读取 JSON：{0}: {1}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise MetricRebuildError("JSON 根节点必须是对象：{0}".format(path))
    return payload


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _fold_time_value(metadata: Mapping[str, Any], field: str) -> float:
    """Return a fold time when it is a valid duration, otherwise ``NaN``.

    A bad timing value must not invalidate otherwise auditable prediction
    metrics.  It is carried forward as ``NaN`` and made explicit in the
    combination-level timing table instead of being silently summed as zero.
    """
    try:
        value = float(metadata.get(field))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) and value >= 0.0 else float("nan")


def _expected_tasks(run_manifest: Mapping[str, Any], split_manifest: pd.DataFrame) -> pd.DataFrame:
    config = run_manifest.get("benchmark_config", {})
    descriptors = config.get("descriptors", [])
    models = config.get("models", [])
    if not descriptors or not models:
        raise MetricRebuildError("run manifest 缺少 descriptors 或 models，无法审计理论任务数")
    fold_pairs = split_manifest[["split_id", "repeat", "fold"]].drop_duplicates()
    records = []
    for descriptor, model in itertools.product(descriptors, models):
        for _, pair in fold_pairs.iterrows():
            records.append({
                "descriptor": str(descriptor), "model": str(model),
                "split_id": str(pair["split_id"]), "repeat": int(pair["repeat"]), "fold": int(pair["fold"]),
            })
    return pd.DataFrame.from_records(records)


def _prediction_key(frame: pd.DataFrame, path: Path) -> Tuple[str, str, str, str, int, int]:
    required = {"run_id", "config_hash", "split_id", "descriptor", "model", "repeat", "fold", "sample_id", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise MetricRebuildError("预测分片缺少字段 {0}: {1}".format(sorted(missing), path))
    static = ["run_id", "config_hash", "split_id", "descriptor", "model", "repeat", "fold"]
    if any(frame[column].nunique(dropna=False) != 1 for column in static):
        raise MetricRebuildError("预测分片包含多个 fold 任务键：{0}".format(path))
    row = frame.iloc[0]
    return (
        str(row["run_id"]), str(row["config_hash"]), str(row["split_id"]),
        str(row["descriptor"]), str(row["model"]), int(row["repeat"]), int(row["fold"]),
    )


def rebuild_fold_metrics(run_dir: Path | str) -> MetricRebuildResult:
    """Read saved artefacts and calculate fold metrics without model retraining.

    Invalid shards become explicit exclusions.  They never enter aggregate or
    paired statistics, and a combination with any missing/excluded fold is
    marked incomplete so it cannot be compared with complete combinations.
    """
    root = Path(run_dir)
    run_manifest = _read_json(root / "manifests" / "run_manifest.json")
    split_path = root / "manifests" / "split_manifest.parquet"
    if not split_path.is_file():
        raise MetricRebuildError("缺少 split manifest：{0}".format(split_path))
    split_manifest = pd.read_parquet(split_path)
    # The split manifest is intentionally data/split-centric: its immutable
    # identity is run_id + dataset_sha256 + split_id.  config_hash belongs to
    # the run manifest and prediction shards, not to every split row.
    required_split = {"run_id", "dataset_sha256", "split_id", "sample_id", "role", "repeat", "fold"}
    if required_split.difference(split_manifest.columns):
        raise MetricRebuildError("split manifest 缺少审计字段")
    run_id = str(run_manifest.get("run_id", ""))
    config_hash = str(run_manifest.get("config_hash", ""))
    if not run_id or not config_hash:
        raise MetricRebuildError("run manifest 缺少 run_id 或 config_hash")
    if set(split_manifest["run_id"].astype(str)) != {run_id}:
        raise MetricRebuildError("split manifest 的 run_id 与 run manifest 不一致")
    expected_dataset_sha = str(run_manifest.get("dataset_sha256", ""))
    if not expected_dataset_sha or set(split_manifest["dataset_sha256"].astype(str)) != {expected_dataset_sha}:
        raise MetricRebuildError("split manifest 的 dataset_sha256 与 run manifest 不一致")

    expected = _expected_tasks(run_manifest, split_manifest)
    expected_keys = {
        (str(row["split_id"]), str(row["descriptor"]), str(row["model"]), int(row["repeat"]), int(row["fold"]))
        for _, row in expected.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, int, int]] = set()
    prediction_dir = root / "predictions"
    for prediction_path in sorted(prediction_dir.glob("*.parquet")) if prediction_dir.exists() else []:
        try:
            prediction = pd.read_parquet(prediction_path)
            key = _prediction_key(prediction, prediction_path)
            shard_run_id, shard_hash, split_id, descriptor, model, repeat, fold = key
            compact_key = (split_id, descriptor, model, repeat, fold)
            metadata_path = root / "folds" / (prediction_path.stem + ".json")
            if compact_key not in expected_keys:
                raise MetricRebuildError("预测分片不属于本运行配置声明的任务")
            if key in seen:
                raise MetricRebuildError("发现同一任务键的重复预测分片")
            seen.add(key)
            if shard_run_id != run_id or shard_hash != config_hash:
                raise MetricRebuildError("预测分片的 run_id/config_hash 与运行清单不一致")
            metadata = _read_json(metadata_path)
            for field, value in {
                "run_id": run_id, "config_hash": config_hash, "split_id": split_id,
                "descriptor": descriptor, "model": model, "repeat": repeat, "fold": fold,
            }.items():
                if metadata.get(field) != value:
                    raise MetricRebuildError("折级元数据 {0!r} 与预测分片不一致".format(field))
            valid_ids = set(split_manifest.loc[
                (split_manifest["split_id"].astype(str) == split_id)
                & (split_manifest["repeat"] == repeat)
                & (split_manifest["fold"] == fold)
                & (split_manifest["role"] == "valid"), "sample_id"
            ].astype(str))
            prediction_ids = prediction["sample_id"].astype(str)
            if prediction_ids.duplicated().any() or set(prediction_ids) != valid_ids:
                raise MetricRebuildError("预测 sample_id 与 manifest 验证集不能一一对应")
            y_true = pd.to_numeric(prediction["y_true"], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(prediction["y_pred"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                raise ValueError("non_finite_prediction")
            if len(y_true) < 2:
                raise ValueError("insufficient_validation_rows")
            if np.ptp(y_true) == 0:
                raise ValueError("constant_validation_label")
            rows.append({
                "run_id": run_id, "config_hash": config_hash, "split_id": split_id,
                "descriptor": descriptor, "model": model, "repeat": repeat, "fold": fold,
                "n_valid": int(len(y_true)), "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "train_time_s": _fold_time_value(metadata, "train_time_s"),
                "predict_time_s": _fold_time_value(metadata, "predict_time_s"),
                "prediction_path": str(prediction_path), "metadata_path": str(metadata_path),
            })
        except ValueError as exc:
            # Deliberately retain bad/undefined statistical cases as audit rows.
            if str(exc) not in {"non_finite_prediction", "insufficient_validation_rows", "constant_validation_label"}:
                raise
            identity = {"run_id": run_id, "config_hash": config_hash, "split_id": "", "descriptor": "", "model": "", "repeat": None, "fold": None}
            try:
                _, _, identity["split_id"], identity["descriptor"], identity["model"], identity["repeat"], identity["fold"] = key
            except UnboundLocalError:
                pass
            exclusions.append({**identity, "reason_code": str(exc), "reason_detail": str(exc), "prediction_path": str(prediction_path)})
        except (MetricRebuildError, OSError, KeyError) as exc:
            exclusions.append({
                "run_id": run_id, "config_hash": config_hash, "split_id": "", "descriptor": "", "model": "",
                "repeat": None, "fold": None, "reason_code": "invalid_prediction_artifact",
                "reason_detail": str(exc), "prediction_path": str(prediction_path),
            })

    fold_metrics = pd.DataFrame.from_records(rows, columns=FOLD_METRIC_COLUMNS)
    exclusion_frame = pd.DataFrame.from_records(exclusions, columns=EXCLUSION_COLUMNS)
    complete_rows = []
    for (split_id, descriptor, model), part in expected.groupby(["split_id", "descriptor", "model"], sort=True):
        expected_pairs = {(int(row["repeat"]), int(row["fold"])) for _, row in part.iterrows()}
        actual = fold_metrics[
            (fold_metrics["split_id"] == split_id) & (fold_metrics["descriptor"] == descriptor) & (fold_metrics["model"] == model)
        ] if not fold_metrics.empty else _empty(FOLD_METRIC_COLUMNS)
        actual_pairs = set(zip(actual.get("repeat", []), actual.get("fold", [])))
        reasons = []
        missing_pairs = expected_pairs.difference(actual_pairs)
        if missing_pairs:
            reasons.append("missing_or_excluded_folds={0}".format(sorted(missing_pairs)))
        complete_rows.append({
            "run_id": run_id, "config_hash": config_hash, "split_id": split_id,
            "descriptor": descriptor, "model": model, "expected_folds": len(expected_pairs),
            "available_folds": len(seen.intersection({(run_id, config_hash, split_id, descriptor, model, repeat, fold) for repeat, fold in expected_pairs})),
            "valid_metric_folds": len(actual_pairs), "is_complete": not reasons,
            "missing_or_excluded_reason": "; ".join(reasons),
        })
    completeness = pd.DataFrame.from_records(complete_rows, columns=COMPLETENESS_COLUMNS)
    return MetricRebuildResult(fold_metrics, exclusion_frame, completeness)


def _bootstrap_mean_ci(values: Sequence[float], n_bootstrap: int, seed: int, confidence: float = 0.95) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        raise MetricRebuildError("bootstrap 输入必须为非空有限数值")
    if n_bootstrap < 100:
        raise MetricRebuildError("n_bootstrap 必须 >= 100")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def summarize_combinations(fold_metrics: pd.DataFrame, *, n_bootstrap: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Summarise complete fold metrics with bootstrap CIs over folds."""
    if fold_metrics.empty:
        return _empty(["run_id", "config_hash", "split_id", "descriptor", "model", "metric"])
    rows: List[Dict[str, Any]] = []
    for keys, part in fold_metrics.groupby(["run_id", "config_hash", "split_id", "descriptor", "model"], sort=True):
        for metric in ("r2", "rmse", "mae"):
            values = part[metric].to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(values, n_bootstrap, seed + len(rows))
            rows.append({
                "run_id": keys[0], "config_hash": keys[1], "split_id": keys[2], "descriptor": keys[3], "model": keys[4],
                "metric": metric, "n_folds": int(len(values)), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "std": float(np.std(values, ddof=0)),
                "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "bootstrap_n": int(n_bootstrap), "bootstrap_seed": int(seed + len(rows) - 1),
            })
    return pd.DataFrame.from_records(rows)


def summarize_combination_times(
    fold_metrics: pd.DataFrame,
    completeness: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate auditable fold durations by descriptor/model combination.

    The output deliberately includes incomplete and invalid-time combinations.
    They retain partial diagnostic values when possible, but
    ``is_time_comparable`` is false so neither report charts nor rankings can
    mistake a partially executed job for a faster complete one.
    """
    required_completeness = {
        "run_id", "config_hash", "split_id", "descriptor", "model",
        "expected_folds", "valid_metric_folds", "is_complete", "missing_or_excluded_reason",
    }
    if missing := required_completeness.difference(completeness.columns):
        raise MetricRebuildError("完整性表缺少组合耗时汇总字段：{0}".format(sorted(missing)))
    required_metrics = {
        "run_id", "config_hash", "split_id", "descriptor", "model",
        "train_time_s", "predict_time_s",
    }
    if not fold_metrics.empty and (missing := required_metrics.difference(fold_metrics.columns)):
        raise MetricRebuildError("折级指标表缺少组合耗时字段：{0}".format(sorted(missing)))

    rows: List[Dict[str, Any]] = []
    group_columns = ["run_id", "config_hash", "split_id", "descriptor", "model"]
    for _, complete_row in completeness.sort_values(group_columns).iterrows():
        keys = {column: complete_row[column] for column in group_columns}
        if fold_metrics.empty:
            part = _empty(FOLD_METRIC_COLUMNS)
        else:
            mask = np.ones(len(fold_metrics), dtype=bool)
            for column, value in keys.items():
                mask &= fold_metrics[column].astype(str).to_numpy() == str(value)
            part = fold_metrics.loc[mask]

        expected_folds = int(complete_row["expected_folds"])
        completed_folds = int(len(part))
        is_complete = bool(complete_row["is_complete"])
        train_times = pd.to_numeric(part.get("train_time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        predict_times = pd.to_numeric(part.get("predict_time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        valid_times = (
            completed_folds > 0
            and np.isfinite(train_times).all()
            and np.isfinite(predict_times).all()
            and (train_times >= 0.0).all()
            and (predict_times >= 0.0).all()
        )
        is_time_comparable = bool(is_complete and completed_folds == expected_folds and valid_times)
        status_parts: List[str] = []
        detail_parts: List[str] = []
        if not is_complete or completed_folds != expected_folds:
            status_parts.append("incomplete_folds")
            reason = str(complete_row.get("missing_or_excluded_reason", "")).strip()
            detail_parts.append(reason or "completed_folds={0}/{1}".format(completed_folds, expected_folds))
        if completed_folds == 0:
            status_parts.append("no_valid_metric_folds")
        elif not valid_times:
            status_parts.append("invalid_or_missing_fold_time")
            detail_parts.append("train_time_s/predict_time_s 必须均为有限且非负的秒数")
        if not status_parts:
            status_parts.append("complete_and_comparable")

        # Never let pandas' default ``sum(skipna=True)`` turn a missing or
        # invalid duration into an apparently valid total.
        if valid_times:
            total_train = float(np.sum(train_times))
            mean_train = float(np.mean(train_times))
            median_train = float(np.median(train_times))
            max_train = float(np.max(train_times))
            total_predict = float(np.sum(predict_times))
            total_model = float(total_train + total_predict)
        else:
            total_train = mean_train = median_train = max_train = float("nan")
            total_predict = total_model = float("nan")
        rows.append({
            **keys,
            "expected_folds": expected_folds,
            "completed_folds": completed_folds,
            "is_complete": is_complete,
            "time_status": "; ".join(status_parts),
            "time_status_detail": "; ".join(part for part in detail_parts if part),
            "is_time_comparable": is_time_comparable,
            "total_train_time_s": total_train,
            "mean_train_time_s": mean_train,
            "median_train_time_s": median_train,
            "max_train_time_s": max_train,
            "total_predict_time_s": total_predict,
            "total_model_time_s": total_model,
        })
    return pd.DataFrame.from_records(rows, columns=COMBINATION_TIME_SUMMARY_COLUMNS)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (total - rank)))
        adjusted[index] = running
    return adjusted


def _metric_direction(metric: str) -> str:
    return "higher_is_better" if metric == "r2" else "lower_is_better"


def _label_difference(metric: str, difference: float, p_adjusted: float, alpha: float) -> str:
    if p_adjusted >= alpha:
        return "no_significant_difference"
    better = difference > 0 if metric == "r2" else difference < 0
    return "significantly_better" if better else "significantly_worse"


def paired_comparisons(
    fold_metrics: pd.DataFrame,
    completeness: pd.DataFrame,
    *,
    dimension: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models per descriptor or descriptors per model on matched folds.

    The first returned table contains only complete comparisons.  The second
    explains every omitted comparison so a report cannot quietly rank an
    incomplete candidate beside a complete one.
    """
    if dimension not in {"model", "descriptor"}:
        raise MetricRebuildError("dimension 必须是 model 或 descriptor")
    try:
        from scipy.stats import ttest_rel
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise MetricRebuildError("需要 scipy >= 1.11 才能进行配对统计比较") from exc
    compare_col = dimension
    fixed_col = "descriptor" if dimension == "model" else "model"
    records: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    complete_keys = set(
        tuple(row) for row in completeness.loc[completeness["is_complete"], ["split_id", "descriptor", "model"]].itertuples(index=False, name=None)
    )
    for (run_id, config_hash, split_id, fixed_value), fixed_part in fold_metrics.groupby(["run_id", "config_hash", "split_id", fixed_col], sort=True):
        candidates = sorted(fixed_part[compare_col].unique())
        for metric in ("r2", "rmse", "mae"):
            pending: List[Dict[str, Any]] = []
            for left, right in itertools.combinations(candidates, 2):
                left_key = (split_id, fixed_value, left) if dimension == "model" else (split_id, left, fixed_value)
                right_key = (split_id, fixed_value, right) if dimension == "model" else (split_id, right, fixed_value)
                if left_key not in complete_keys or right_key not in complete_keys:
                    exclusions.append({
                        "comparison_dimension": dimension, "fixed_value": fixed_value, "metric": metric,
                        "left": left, "right": right, "reason_code": "incomplete_fold_set",
                    })
                    continue
                left_values = fixed_part[fixed_part[compare_col] == left][["repeat", "fold", metric]].rename(columns={metric: "left_value"})
                right_values = fixed_part[fixed_part[compare_col] == right][["repeat", "fold", metric]].rename(columns={metric: "right_value"})
                paired = left_values.merge(right_values, on=["repeat", "fold"], how="inner", validate="one_to_one")
                if len(paired) != len(left_values) or len(paired) != len(right_values):
                    exclusions.append({
                        "comparison_dimension": dimension, "fixed_value": fixed_value, "metric": metric,
                        "left": left, "right": right, "reason_code": "unmatched_fold_keys",
                    })
                    continue
                difference = paired["left_value"].to_numpy() - paired["right_value"].to_numpy()
                if len(difference) < 2:
                    raw_p = 1.0
                elif np.ptp(difference) == 0:
                    # Equal non-zero paired differences are the clearest
                    # possible directional fixture; scipy reports an infinite
                    # t statistic here, so encode its limiting p value.
                    raw_p = 0.0 if difference[0] != 0 else 1.0
                else:
                    raw_p = float(ttest_rel(paired["left_value"], paired["right_value"]).pvalue)
                    if not np.isfinite(raw_p):
                        raw_p = 1.0
                ci_low, ci_high = _bootstrap_mean_ci(difference, n_bootstrap, seed + len(records) + len(pending))
                pending.append({
                    "run_id": run_id, "config_hash": config_hash, "split_id": split_id,
                    "comparison_dimension": dimension, "fixed_value": fixed_value, "metric": metric,
                    "metric_direction": _metric_direction(metric), "left": left, "right": right,
                    "n_folds": int(len(difference)), "effect_mean_difference": float(np.mean(difference)),
                    "difference_ci_low": ci_low, "difference_ci_high": ci_high,
                    "p_value_raw": raw_p, "statistical_test": "paired_ttest", "multiple_testing": "holm_within_fixed_metric",
                })
            if pending:
                adjusted = _holm_adjust([record["p_value_raw"] for record in pending])
                for record, p_adjusted in zip(pending, adjusted):
                    record["p_value_adjusted"] = float(p_adjusted)
                    record["conclusion"] = _label_difference(record["metric"], record["effect_mean_difference"], float(p_adjusted), alpha)
                    records.append(record)
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(exclusions)


def tukey_hsd_comparisons(fold_metrics: pd.DataFrame, *, dimension: str, alpha: float = 0.05) -> pd.DataFrame:
    """Provide Tukey-HSD multiple-comparison output alongside paired tests."""
    if dimension not in {"model", "descriptor"}:
        raise MetricRebuildError("dimension 必须是 model 或 descriptor")
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise MetricRebuildError("需要 statsmodels >= 0.14 才能生成 Tukey HSD 结果") from exc
    compare_col = dimension
    fixed_col = "descriptor" if dimension == "model" else "model"
    records: List[Dict[str, Any]] = []
    for (run_id, config_hash, split_id, fixed_value), part in fold_metrics.groupby(["run_id", "config_hash", "split_id", fixed_col], sort=True):
        if part[compare_col].nunique() < 2:
            continue
        for metric in ("r2", "rmse", "mae"):
            result = pairwise_tukeyhsd(endog=part[metric].to_numpy(dtype=float), groups=part[compare_col].astype(str), alpha=alpha)
            groups = result.groupsunique
            first, second = result._multicomp.pairindices
            for index, (left_index, right_index) in enumerate(zip(first, second)):
                records.append({
                    "run_id": run_id, "config_hash": config_hash, "split_id": split_id,
                    "comparison_dimension": dimension, "fixed_value": fixed_value, "metric": metric,
                    "left": str(groups[left_index]), "right": str(groups[right_index]),
                    "mean_difference_right_minus_left": float(result.meandiffs[index]),
                    "tukey_ci_low": float(result.confint[index, 0]), "tukey_ci_high": float(result.confint[index, 1]),
                    "tukey_p_value_adjusted": float(result.pvalues[index]), "tukey_reject": bool(result.reject[index]),
                    "statistical_test": "tukey_hsd_unpaired_fold_values",
                })
    return pd.DataFrame.from_records(records)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame, table_name: str) -> Path:
    """Write and read back a derived table before atomically publishing it."""
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    if len(pd.read_parquet(temporary)) != len(frame):
        temporary.unlink(missing_ok=True)
        raise MetricRebuildError("派生指标表临时文件校验失败：{0}".format(table_name))
    temporary.replace(path)
    return path


def write_combination_time_summary(run_dir: Path | str, summary: pd.DataFrame) -> Path:
    """Atomically persist the descriptor/model timing summary for a run."""
    missing = set(COMBINATION_TIME_SUMMARY_COLUMNS).difference(summary.columns)
    if missing:
        raise MetricRebuildError("组合耗时汇总表缺少字段：{0}".format(sorted(missing)))
    root = Path(run_dir) / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    ordered = summary.loc[:, COMBINATION_TIME_SUMMARY_COLUMNS]
    return _atomic_write_parquet(root / "combination_time_summary.parquet", ordered, "combination_time_summary")


def write_metric_tables(
    run_dir: Path | str,
    rebuilt: MetricRebuildResult,
    summary: pd.DataFrame,
    model_comparisons: pd.DataFrame,
    descriptor_comparisons: pd.DataFrame,
    comparison_exclusions: pd.DataFrame,
    model_tukey: pd.DataFrame,
    descriptor_tukey: pd.DataFrame,
) -> Dict[str, Path]:
    """Atomically persist all derived tables under the run's ``metrics`` directory."""
    root = Path(run_dir) / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "fold_metrics": rebuilt.fold_metrics, "metric_exclusions": rebuilt.exclusions,
        "completeness": rebuilt.completeness, "combination_summary": summary,
        "model_comparisons": model_comparisons, "descriptor_comparisons": descriptor_comparisons,
        "comparison_exclusions": comparison_exclusions,
        "model_tukey_hsd": model_tukey, "descriptor_tukey_hsd": descriptor_tukey,
    }
    paths: Dict[str, Path] = {}
    for name, frame in tables.items():
        path = root / (name + ".parquet")
        paths[name] = _atomic_write_parquet(path, frame, name)
    combination_time_summary = summarize_combination_times(rebuilt.fold_metrics, rebuilt.completeness)
    paths["combination_time_summary"] = write_combination_time_summary(run_dir, combination_time_summary)
    return paths
