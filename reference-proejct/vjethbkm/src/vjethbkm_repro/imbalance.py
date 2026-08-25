"""Yield imbalance and high-yield analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
DEFAULT_LABELS = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


def assign_yield_bucket(values, bins: list[float] | None = None, labels: list[str] | None = None) -> pd.Series:
    bins = bins or DEFAULT_BINS
    labels = labels or DEFAULT_LABELS
    return pd.cut(pd.Series(values, dtype=float), bins=bins, labels=labels, include_lowest=True, right=False)


def yield_bucket_summary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    target = pd.to_numeric(df[label_col], errors="raise")
    buckets = assign_yield_bucket(target)
    summary = (
        pd.DataFrame({"bucket": buckets, "yield": target})
        .groupby("bucket", observed=False)
        .agg(count=("yield", "size"), mean_yield=("yield", "mean"), min_yield=("yield", "min"), max_yield=("yield", "max"))
        .reset_index()
    )
    summary["fraction"] = summary["count"] / max(float(len(df)), 1.0)
    return summary


def bucket_sample_weights(y_true) -> np.ndarray:
    buckets = assign_yield_bucket(y_true)
    counts = buckets.value_counts()
    weights = np.array(
        [
            1.0 / float(counts[bucket]) if pd.notna(bucket) and float(counts[bucket]) > 0 else 0.0
            for bucket in buckets
        ],
        dtype=float,
    )
    mean = weights[weights > 0].mean() if np.any(weights > 0) else 1.0
    return weights / mean


def prediction_bucket_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["bucket"] = assign_yield_bucket(df["y_true"])
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    return (
        df.groupby(["descriptor", "model", "bucket"], observed=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        )
        .reset_index()
    )


def high_yield_summary(predictions: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    records: list[dict] = []
    for (descriptor, model), group in predictions.groupby(["descriptor", "model"]):
        truth = group["y_true"] >= threshold
        predicted = group["y_pred"] >= threshold
        true_positive = int((truth & predicted).sum())
        false_positive = int((~truth & predicted).sum())
        false_negative = int((truth & ~predicted).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        records.append(
            {
                "descriptor": descriptor,
                "model": model,
                "threshold": threshold,
                "actual_high_yield": int(truth.sum()),
                "predicted_high_yield": int(predicted.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame.from_records(records)
