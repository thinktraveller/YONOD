"""Reproducible, group-aware synthesis of the 12 ChemRxiv frozen tests.

The formal runs remain read-only inputs.  This script first invokes the
existing per-task verifier for every fixed YAML, then uses only the saved
external prediction parquet files to produce an immutable analysis bundle.
It never imports a descriptor builder or a model training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_ablation2_frozen_external_result import verify


OUTPUT = ROOT / "project-docs" / "docs" / "ablation2" / "chemrxiv_frozen_analysis_v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260922
EXTERNAL_ROWS = 957
EXTERNAL_GROUPS = 10

LINES: Tuple[Dict[str, str], ...] = (
    {
        "id": "morgan_rf",
        "descriptor": "morgan",
        "model": "rf",
        "label": "Morgan ECFP4 × RF",
    },
    {
        "id": "mfp_lightgbm",
        "descriptor": "mfp",
        "model": "lightgbm",
        "label": "MFP × LightGBM",
    },
)

COMBOS: Tuple[Tuple[str, str], ...] = (
    ("full", "A+B+P+C"),
    ("minus_a", "B+P+C"),
    ("minus_p", "A+B+C"),
    ("minus_a_p", "B+C"),
    ("product_conditions", "P+C"),
    ("conditions_only", "C only"),
)

# ``variant - reference`` is positive when the reference has lower MAE.
# The first two are exactly the pre-registered marginal A comparisons.
CONTRASTS: Tuple[Dict[str, str], ...] = (
    {
        "id": "a_with_product",
        "reference": "full",
        "variant": "minus_a",
        "label": "ΔA|P = L(B+P+C) − L(A+B+P+C)",
        "family": "primary_a",
        "scope": "pre-registered primary marginal A contrast",
    },
    {
        "id": "a_without_product",
        "reference": "minus_p",
        "variant": "minus_a_p",
        "label": "ΔA|¬P = L(B+C) − L(A+B+C)",
        "family": "primary_a",
        "scope": "pre-registered primary marginal A contrast",
    },
    {
        "id": "full_vs_minus_p",
        "reference": "full",
        "variant": "minus_p",
        "label": "L(A+B+C) − L(A+B+P+C)",
        "family": "exploratory_full",
        "scope": "fixed-matrix secondary full-input contrast",
    },
    {
        "id": "full_vs_minus_a_p",
        "reference": "full",
        "variant": "minus_a_p",
        "label": "L(B+C) − L(A+B+P+C)",
        "family": "exploratory_full",
        "scope": "fixed-matrix secondary full-input contrast",
    },
    {
        "id": "full_vs_product_conditions",
        "reference": "full",
        "variant": "product_conditions",
        "label": "L(P+C) − L(A+B+P+C)",
        "family": "exploratory_full",
        "scope": "fixed-matrix secondary full-input contrast",
    },
    {
        "id": "full_vs_conditions_only",
        "reference": "full",
        "variant": "conditions_only",
        "label": "L(C) − L(A+B+P+C)",
        "family": "exploratory_full",
        "scope": "fixed-matrix secondary full-input contrast",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_text_values(values: Iterable[Any]) -> str:
    return hash_json([str(value) for value in values])


def task_name(line: Dict[str, str], combo: str) -> str:
    return "ablation2_chemrxiv_frozen_{0}_{1}_v1".format(line["id"], combo)


def config_path(line: Dict[str, str], combo: str) -> Path:
    return ROOT / "config" / (task_name(line, combo) + ".yaml")


def task_root(line: Dict[str, str], combo: str) -> Path:
    return ROOT / "result" / task_name(line, combo)


def prediction_path(line: Dict[str, str], combo: str) -> Path:
    expected = task_root(line, combo) / "docs" / "predictions" / "{0}_{1}__{2}.parquet".format(
        line["descriptor"], combo, line["model"]
    )
    if not expected.is_file():
        raise RuntimeError("Missing expected prediction file: {0}".format(expected))
    return expected


def metric_file(line: Dict[str, str], combo: str) -> Path:
    expected = task_root(line, combo) / "docs" / "metrics" / "external_test_metrics.csv"
    if not expected.is_file():
        raise RuntimeError("Missing expected metrics file: {0}".format(expected))
    return expected


def holm_adjust(values: Sequence[float]) -> List[float]:
    """Return Holm adjusted p-values in the original order."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def rank_biserial(values: np.ndarray) -> float:
    """Signed-rank rank-biserial correlation; zero differences are excluded."""
    nonzero = values[~np.isclose(values, 0.0)]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def bootstrap_ci(group_differences: np.ndarray, seed_offset: int) -> Tuple[float, float]:
    """Equal-cluster percentile interval for the frozen ten amine structures."""
    if len(group_differences) != EXTERNAL_GROUPS:
        raise RuntimeError("Expected exactly 10 external amine groups for bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = group_differences[
        rng.integers(0, EXTERNAL_GROUPS, size=(BOOTSTRAP_DRAWS, EXTERNAL_GROUPS))
    ]
    low, high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def metrics(frame: pd.DataFrame, prediction_column: str) -> Dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)
    tau = stats.kendalltau(y_true, y_pred).statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(tau) if tau is not None else float("nan"),
    }


def read_and_verify(line: Dict[str, str], combo: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the task verifier, then load a normalized external prediction table."""
    config = config_path(line, combo)
    if not config.is_file():
        raise RuntimeError("Missing formal YAML: {0}".format(config))
    integrity = verify(config)
    if integrity.get("status") != "passed":
        raise RuntimeError("Task verifier did not pass: {0}".format(config))

    prediction_file = prediction_path(line, combo)
    frame = pd.read_parquet(prediction_file)
    required = {
        "sample_id",
        "external_source_id",
        "external_group_id",
        "y_true",
        "y_pred",
        "evaluation_protocol",
        "partition",
    }
    absent = sorted(required.difference(frame.columns))
    if absent:
        raise RuntimeError("Prediction lacks required columns {0}: {1}".format(absent, prediction_file))
    if len(frame) != EXTERNAL_ROWS or frame["sample_id"].astype(str).nunique() != EXTERNAL_ROWS:
        raise RuntimeError("Prediction is not a unique 957-row external population: {0}".format(prediction_file))
    if frame["external_group_id"].astype(str).nunique() != EXTERNAL_GROUPS:
        raise RuntimeError("Prediction does not expose the fixed 10 amine groups: {0}".format(prediction_file))
    if not frame["evaluation_protocol"].eq("frozen_train_external_test").all():
        raise RuntimeError("Unexpected evaluation protocol in {0}".format(prediction_file))
    if not frame["partition"].eq("external_test").all():
        raise RuntimeError("Unexpected partition in {0}".format(prediction_file))
    for column in ("y_true", "y_pred"):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise RuntimeError("Non-finite {0} in {1}".format(column, prediction_file))

    columns = ["sample_id", "external_source_id", "external_group_id", "y_true", "y_pred"]
    frame = frame.loc[:, columns].copy()
    for column in ("sample_id", "external_source_id", "external_group_id"):
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    provenance = {
        "task_name": task_name(line, combo),
        "line_id": line["id"],
        "line_label": line["label"],
        "combo": combo,
        "config_path": str(config.relative_to(ROOT)),
        "config_sha256": sha256_file(config),
        "prediction_path": str(prediction_file.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_file),
        "metrics_path": str(metric_file(line, combo).relative_to(ROOT)),
        "metrics_sha256": sha256_file(metric_file(line, combo)),
        "sample_id_sha256": hash_text_values(frame["sample_id"]),
        "external_source_id_sha256": hash_text_values(frame["external_source_id"]),
        "external_group_id_sha256": hash_text_values(frame["external_group_id"]),
        "integrity": integrity,
    }
    return frame, provenance


def join_combos(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    expected = [combo for combo, _ in COMBOS]
    if sorted(predictions) != sorted(expected):
        raise RuntimeError("A required input combination is missing")
    identity = ["sample_id", "external_source_id", "external_group_id", "y_true"]
    merged = predictions["full"].rename(columns={"y_pred": "pred_full"})
    for combo in expected:
        if combo == "full":
            continue
        right = predictions[combo].rename(columns={"y_pred": "pred_" + combo})
        merged = merged.merge(
            right.loc[:, identity + ["pred_" + combo]],
            how="inner",
            on=identity,
            validate="one_to_one",
        )
    if len(merged) != EXTERNAL_ROWS or merged["external_group_id"].nunique() != EXTERNAL_GROUPS:
        raise RuntimeError("Input predictions do not have a one-to-one external join")
    return merged.sort_values("sample_id", kind="stable").reset_index(drop=True)


def compute_line(line: Dict[str, str], joined: pd.DataFrame, line_number: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    group_mae_rows: List[pd.DataFrame] = []
    error = pd.DataFrame({"external_group_id": joined["external_group_id"]})
    for combo, inputs in COMBOS:
        prediction_column = "pred_" + combo
        item = {
            "line_id": line["id"],
            "line_label": line["label"],
            "combo": combo,
            "input_blocks": inputs,
            "n_external": len(joined),
        }
        item.update(metrics(joined, prediction_column))
        metric_rows.append(item)
        error[combo] = (joined[prediction_column] - joined["y_true"]).abs()
        group = error.loc[:, ["external_group_id", combo]].groupby("external_group_id", sort=True).mean().reset_index()
        group = group.rename(columns={combo: "group_mae"})
        group.insert(0, "n_rows", joined.groupby("external_group_id").size().reindex(group["external_group_id"]).to_numpy())
        group.insert(0, "input_blocks", inputs)
        group.insert(0, "combo", combo)
        group.insert(0, "line_label", line["label"])
        group.insert(0, "line_id", line["id"])
        group_mae_rows.append(group)

    per_group = error.groupby("external_group_id", sort=True).mean()
    rows: List[Dict[str, Any]] = []
    for contrast_number, specification in enumerate(CONTRASTS):
        difference = (per_group[specification["variant"]] - per_group[specification["reference"]]).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(difference, line_number * 100 + contrast_number)
        nonzero = difference[~np.isclose(difference, 0.0)]
        test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto") if len(nonzero) else None
        weighted = float((error[specification["variant"]] - error[specification["reference"]]).mean())
        rows.append(
            {
                "line_id": line["id"],
                "line_label": line["label"],
                "contrast": specification["id"],
                "contrast_label": specification["label"],
                "reference": specification["reference"],
                "variant": specification["variant"],
                "scope": specification["scope"],
                "test_family": specification["family"],
                "direction": "positive: reference has lower group MAE",
                "effect_group_mean": float(difference.mean()),
                "effect_group_median": float(np.median(difference)),
                "effect_row_weighted": weighted,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED + line_number * 100 + contrast_number,
                "n_amine_groups": EXTERNAL_GROUPS,
                "n_nonzero_groups": int(len(nonzero)),
                "wilcoxon_statistic": float(test.statistic) if test is not None else float("nan"),
                "p_raw": float(test.pvalue) if test is not None else float("nan"),
                "rank_biserial": rank_biserial(difference),
                "practical_mae_threshold": 0.01,
            }
        )

    # The derived substitution contrast was pre-specified, but its inference is
    # bootstrap-only: it is not added to a post hoc p-value family.
    difference = (
        (per_group["minus_a_p"] - per_group["minus_p"])
        - (per_group["minus_a"] - per_group["full"])
    ).to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(difference, line_number * 100 + len(CONTRASTS))
    rows.append(
        {
            "line_id": line["id"],
            "line_label": line["label"],
            "contrast": "a_product_substitution",
            "contrast_label": "Δsub,A = ΔA|¬P − ΔA|P",
            "reference": "derived",
            "variant": "derived",
            "scope": "pre-registered derived substitution estimand; bootstrap-only",
            "test_family": "derived_bootstrap_only",
            "direction": "positive: P reduces A conditional predictive contribution",
            "effect_group_mean": float(difference.mean()),
            "effect_group_median": float(np.median(difference)),
            "effect_row_weighted": float(
                ((error["minus_a_p"] - error["minus_p"]) - (error["minus_a"] - error["full"])).mean()
            ),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED + line_number * 100 + len(CONTRASTS),
            "n_amine_groups": EXTERNAL_GROUPS,
            "n_nonzero_groups": int((~np.isclose(difference, 0.0)).sum()),
            "wilcoxon_statistic": float("nan"),
            "p_raw": float("nan"),
            "rank_biserial": rank_biserial(difference),
            "practical_mae_threshold": 0.01,
        }
    )
    return pd.DataFrame(metric_rows), pd.concat(group_mae_rows, ignore_index=True), pd.DataFrame(rows)


def add_adjustments_and_interpretation(contrasts: pd.DataFrame) -> pd.DataFrame:
    contrasts = contrasts.copy()
    contrasts["p_holm"] = np.nan
    contrasts["holm_scope"] = "not assigned"
    for line in contrasts["line_id"].unique():
        primary = (contrasts["line_id"] == line) & (contrasts["test_family"] == "primary_a")
        values = contrasts.loc[primary, "p_raw"].astype(float).tolist()
        contrasts.loc[primary, "p_holm"] = holm_adjust(values)
        contrasts.loc[primary, "holm_scope"] = "two pre-registered marginal A contrasts within model line"

        exploratory = (contrasts["line_id"] == line) & (contrasts["test_family"] == "exploratory_full")
        values = contrasts.loc[exploratory, "p_raw"].astype(float).tolist()
        contrasts.loc[exploratory, "p_holm"] = holm_adjust(values)
        contrasts.loc[exploratory, "holm_scope"] = "four fixed-matrix secondary full-input contrasts within model line (exploratory)"

    conclusions: List[str] = []
    for _, row in contrasts.iterrows():
        if row["contrast"] == "a_product_substitution":
            if row["ci_low"] > 0.01:
                conclusions.append("supports the pre-specified conditional substitution pattern in this external collection")
            elif row["ci_high"] < 0:
                conclusions.append("direction is inconsistent with the pre-specified substitution pattern in this external collection")
            else:
                conclusions.append("uncertain at the pre-specified 0.01 practical-MAE boundary")
        elif row["test_family"] == "primary_a":
            conclusions.append("primary marginal A result; use Holm-adjusted p with bootstrap interval")
        else:
            conclusions.append("secondary fixed-matrix comparison; descriptive/exploratory, not a causal claim")
    contrasts["conclusion"] = conclusions
    return contrasts


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows: List[str] = []
    for _, row in frame.iterrows():
        values: List[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else ("{0:.%df}" % digits).format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + rows)


def write_figures(metrics_table: pd.DataFrame, contrasts: pd.DataFrame, figures: Path) -> None:
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#777777"]
    combo_order = [combo for combo, _ in COMBOS]
    combo_labels = ["Full", "−A", "−P", "−A−P", "P+C", "C only"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, line in zip(axes, LINES):
        table = metrics_table[metrics_table["line_id"] == line["id"]].set_index("combo").loc[combo_order]
        positions = np.arange(len(combo_order))
        axis.bar(positions, table["mae"], color=palette, edgecolor="white", linewidth=0.7)
        axis.set_xticks(positions, combo_labels, rotation=25, ha="right")
        axis.set_title(line["label"])
        axis.set_ylabel("External MAE (yield fraction)")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen development→external test: aggregate MAE (957 reactions)")
    figure.tight_layout()
    figure.savefig(figures / "figure-01-external-mae-by-input.pdf", bbox_inches="tight")
    figure.savefig(figures / "figure-01-external-mae-by-input.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    selected = contrasts[contrasts["contrast"].isin([item["id"] for item in CONTRASTS])].copy()
    label_map = {
        "a_with_product": "Full vs −A",
        "full_vs_minus_p": "Full vs −P",
        "full_vs_minus_a_p": "Full vs −A−P",
        "full_vs_product_conditions": "Full vs P+C",
        "full_vs_conditions_only": "Full vs C only",
        "a_without_product": "−P vs −A−P",
    }
    selected["short_label"] = selected["contrast"].map(label_map)
    order = ["a_with_product", "full_vs_minus_p", "full_vs_minus_a_p", "full_vs_product_conditions", "full_vs_conditions_only", "a_without_product"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for axis, line in zip(axes, LINES):
        table = selected[selected["line_id"] == line["id"]].set_index("contrast").loc[order].reset_index()
        estimate = table["effect_group_mean"].to_numpy(float)
        error = np.vstack([estimate - table["ci_low"].to_numpy(float), table["ci_high"].to_numpy(float) - estimate])
        positions = np.arange(len(table))
        axis.errorbar(estimate, positions, xerr=error, fmt="o", color="#0072B2", capsize=3)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(positions, table["short_label"].tolist(), fontsize=8)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(line["label"])
        axis.set_xlabel("Group-mean ΔMAE; positive favors reference")
    figure.suptitle("Paired external amine-group contrasts; 95% cluster-bootstrap intervals")
    figure.tight_layout()
    figure.savefig(figures / "figure-02-group-paired-ablation-contrasts.pdf", bbox_inches="tight")
    figure.savefig(figures / "figure-02-group-paired-ablation-contrasts.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    substitution = contrasts[contrasts["contrast"] == "a_product_substitution"].copy()
    figure, axis = plt.subplots(figsize=(7.5, 3.7))
    y = np.arange(len(substitution))
    estimate = substitution["effect_group_mean"].to_numpy(float)
    error = np.vstack([estimate - substitution["ci_low"].to_numpy(float), substitution["ci_high"].to_numpy(float) - estimate])
    axis.errorbar(estimate, y, xerr=error, fmt="o", color="#D55E00", capsize=4)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.axvline(0.01, color="#333333", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, substitution["line_label"].tolist())
    axis.set_xlabel("Δsub,A group-mean MAE; 95% cluster-bootstrap interval")
    axis.set_title("Pre-registered product substitution of amine A")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "figure-03-a-product-substitution.pdf", bbox_inches="tight")
    figure.savefig(figures / "figure-03-a-product-substitution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_analysis_report(metrics_table: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    metric_display = metrics_table.loc[:, ["line_label", "combo", "input_blocks", "mae", "rmse", "r2", "kendall_tau"]]
    contrast_display = contrasts.loc[:, ["line_label", "contrast", "effect_group_mean", "ci_low", "ci_high", "p_raw", "p_holm", "rank_biserial", "conclusion"]]
    substitution = contrasts[contrasts["contrast"] == "a_product_substitution"].set_index("line_id")
    morgan = substitution.loc["morgan_rf"]
    mfp = substitution.loc["mfp_lightgbm"]
    return """# ChemRxiv frozen external-test analysis v1

## Analysis question and fixed evidence

Does the pre-registered product-substitution pattern for amine A appear when a model fitted once on the 47,015-row development population is evaluated on the independent 957-row ChemRxiv collection? All 12 fixed input/model tasks were re-verified before this synthesis. Each result is one frozen development fit and one external prediction—not CV, not a model replicate, and not eligible for strict-CV ranking.

The input task outputs share the same 957 unique `sample_id` values, `y_true`, `external_source_id`, and 10 `external_group_id` amine structures. Exact development/external sample-ID intersection is zero, and the task verifier checks source hashes, label isolation, finite predictions, recomputed metrics, and HTML/Markdown task reports.

## Descriptive external performance

{metrics}

All external R² values are negative and Kendall τ values are low. These aggregate metrics document a difficult cross-source shift; they do not license a cross-protocol numerical comparison against the development CV metrics.

## Paired group-level ablations

Each point estimate is the equally weighted mean of 10 within-amine-group MAE differences. Positive values favor the reference input (lower MAE); intervals are 10,000-draw percentile cluster bootstraps that resample whole amine groups. The two marginal A rows are the pre-registered primary p-value family. The remaining Full contrasts were fixed input rows but are marked secondary/exploratory; their Holm adjustment is separate. The derived substitution effect retains its pre-registered bootstrap-only treatment.

{contrasts}

## Claim candidates

- Claim:
  - Source evidence: `tables/contrast_summary.csv`, `figures/figure-03-a-product-substitution.pdf`.
  - Allowed wording: “For Morgan ECFP4 × RF on this 957-reaction, one-acid external collection, Δsub,A was {morgan_effect:.4f} MAE (95% cluster-bootstrap CI {morgan_low:.4f} to {morgan_high:.4f}), exceeding the pre-specified 0.01 practical boundary.”
  - Forbidden stronger wording: “Product structure chemically makes amine identity redundant” or “the result proves general external generalization.”
  - Uncertainty: one frozen fit, ten amine groups, a single external collection and one acid; the second model line must not be treated as a chemical replicate.
  - Next check: a separately designed multi-acid external replication or preregistered within-source unseen-A validation.
  - Decision: retain as source-bounded support only.

- Claim:
  - Source evidence: `tables/contrast_summary.csv`, `figures/figure-03-a-product-substitution.pdf`.
  - Allowed wording: “The MFP × LightGBM substitution estimate was {mfp_effect:.4f} MAE (95% cluster-bootstrap CI {mfp_low:.4f} to {mfp_high:.4f}); its interval did not clear the pre-specified practical boundary.”
  - Forbidden stronger wording: “The hypothesis is disproved” or “the two model lines disagree conclusively.”
  - Uncertainty: ten groups provide limited precision and there is no repeated model seed.
  - Next check: retain the line as a pre-specified robustness result and seek an independent external collection.
  - Decision: weaken to uncertain robustness evidence.

## What this does and does not compare

The development results and this external collection answer different questions: the former used audited repeated CV protocols within the development population, while this analysis uses one frozen development fit and a zero-overlap external collection. The external C mapping intentionally excludes development `additive_smiles`, which lacks a one-to-one external counterpart. Thus MAE/R² values must not be pooled, ranked together, or interpreted as a direct estimate of a development-to-external performance drop.

This analysis supports only a source-bounded predictive-information pattern, not chemical causality, mutual information, unseen-acid generalization, or equivalence. A non-significant group-level contrast is not evidence of no effect.

## Reproduction

Run `python _verify/analyze_ablation2_chemrxiv_frozen.py` from the repository root with the `yonod` environment active. The script invokes the 12 existing verifiers and reads saved prediction parquet files only; it never starts a model or writes beneath any formal `result/ablation2_chemrxiv_frozen_*_v1/` task root.
""".format(
        metrics=markdown_table(metric_display, list(metric_display.columns)),
        contrasts=markdown_table(contrast_display, list(contrast_display.columns)),
        morgan_effect=morgan["effect_group_mean"],
        morgan_low=morgan["ci_low"],
        morgan_high=morgan["ci_high"],
        mfp_effect=mfp["effect_group_mean"],
        mfp_low=mfp["ci_low"],
        mfp_high=mfp["ci_high"],
    )


def build_stats_appendix(contrasts: pd.DataFrame) -> str:
    primary = contrasts[contrasts["test_family"] == "primary_a"].loc[
        :, ["line_label", "contrast", "n_amine_groups", "n_nonzero_groups", "effect_group_mean", "effect_group_median", "effect_row_weighted", "ci_low", "ci_high", "wilcoxon_statistic", "p_raw", "p_holm", "rank_biserial"]
    ]
    return """# Statistical appendix

## Unit, estimand, and assumptions

- **Inference unit:** the 10 unique external `external_group_id` amine structures, not the 957 reaction rows, task files, model lines, or CV folds.
- **Primary metric:** MAE in yield fraction; lower is better. RMSE, R² and Kendall τ are descriptive auxiliaries.
- **Point estimate:** for a contrast `variant − reference`, calculate each group’s mean absolute-error difference and take the equally weighted mean of the ten values. Positive means the reference input has lower group MAE.
- **Uncertainty:** 10,000 fixed-seed percentile cluster-bootstrap draws resample all rows belonging to an amine structure by resampling its group mean with replacement. Intervals are conditional on this one external reaction collection.
- **Tests:** two-sided SciPy Wilcoxon signed-rank tests (`method='auto'`) use the nonzero group differences. Rank-biserial correlation is `(W+ − W−)/(W+ + W−)`. With only 10 groups, exact/small-sample limitations remain material; zero differences are excluded from the test and effect-size ranks.
- **Multiplicity:** Holm within each model line over exactly the two pre-registered marginal A contrasts. Four additional fixed-matrix Full-input comparisons receive a separately labelled exploratory Holm family. Δsub,A is pre-specified but reported bootstrap-only, without an added p-value family.

## Pre-registered marginal A results

{primary}

## Design limits and excluded claims

1. This is one fixed development fit per model/input, not an independent seed replicate or CV estimate. There are no fold-level data for an external-test inference.
2. The external source has 957 reactions but only 10 amine groups and one acid. Treating rows as 957 independent repetitions would understate uncertainty; the group bootstrap does not create new chemical-source replication.
3. The predictor labels were not used for fit, scaling, model selection, or prediction, as checked by the task verifier. This prevents the particular leakage mode but cannot remove distribution shift or all dataset-design limitations.
4. The two pre-specified model lines are robustness implementations, not independent chemical experiments. Agreement is useful; disagreement cannot by itself refute the hypothesis.
5. No conclusion about B/acid ablation is possible because the external source has no within-source acid variation; no conclusion about unseen-acid generalization is valid.
6. The non-CV external test and development CV cannot be pooled for a common standard error, model rank, p-value, or direct performance-drop estimate.
""".format(primary=markdown_table(primary, list(primary.columns)))


def build_figure_catalog() -> str:
    return """# Figure catalog

## Figure 01 — external MAE by input

- **Files:** `figures/figure-01-external-mae-by-input.pdf` and `.png`
- **Purpose:** show the six fixed input configurations separately for each pre-specified model line.
- **Data:** 957-row aggregate MAE reconstructed from saved external predictions.
- **Caption requirement:** state that bars are one frozen development→external fit, not CV means and not uncertainty intervals.
- **Reader should notice:** `minus_a` and `P+C` have lower aggregate MAE than Full in both lines, while C-only and minus-A-minus-P are worse.
- **Interpretation boundary:** aggregate differences alone are not an inferential result; see Figures 02–03 and group-aware tables.

## Figure 02 — paired external amine-group contrasts

- **Files:** `figures/figure-02-group-paired-ablation-contrasts.pdf` and `.png`
- **Purpose:** display Full-based ablations plus the no-product marginal A contrast with ten-group cluster-bootstrap intervals.
- **Data:** equally weighted mean of within-amine-group MAE differences; 95% percentile cluster-bootstrap intervals, 10,000 draws.
- **Caption requirement:** define positive ΔMAE as lower MAE for the reference input, and distinguish pre-registered marginal A rows from secondary Full-input rows.
- **Reader should notice:** intervals are substantially wider than a row-wise analysis would imply, because only ten groups are inferential units.
- **Interpretation boundary:** it does not establish causality, a multi-source result, or model equivalence.

## Figure 03 — pre-registered product substitution of A

- **Files:** `figures/figure-03-a-product-substitution.pdf` and `.png`
- **Purpose:** directly show Δsub,A relative to zero and the pre-registered 0.01 practical-MAE boundary.
- **Data:** `ΔA|¬P − ΔA|P`, with ten-group percentile cluster-bootstrap intervals.
- **Caption requirement:** name the 957-reaction, one-acid external collection and make clear that the dashed line is the practical boundary, not a hypothesis-test threshold.
- **Reader should notice:** Morgan × RF clears the practical boundary while MFP × LightGBM does not; the result is therefore source-bounded and only partially robust across the two model lines.
- **Interpretation boundary:** no causal or general external-generalization claim follows.
"""


def build_html(markdown: str) -> str:
    return """<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>ChemRxiv frozen analysis v1</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;line-height:1.45;padding:0 1rem}}pre{{white-space:pre-wrap;font:0.91rem/1.45 ui-monospace,monospace}}</style>
<body><pre>{}</pre></body></html>
""".format(html.escape(markdown))


def file_hashes(directory: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def write_bundle(stage: Path, metrics_table: pd.DataFrame, group_mae: pd.DataFrame, contrasts: pd.DataFrame, provenance: List[Dict[str, Any]]) -> None:
    (stage / "figures").mkdir(parents=True)
    (stage / "tables").mkdir()
    (stage / "report").mkdir()
    report = build_analysis_report(metrics_table, contrasts)
    (stage / "analysis-report.md").write_text(report, encoding="utf-8")
    (stage / "stats-appendix.md").write_text(build_stats_appendix(contrasts), encoding="utf-8")
    (stage / "figure-catalog.md").write_text(build_figure_catalog(), encoding="utf-8")
    (stage / "report" / "analysis_report.html").write_text(build_html(report), encoding="utf-8")
    metrics_table.to_csv(stage / "tables" / "external_metric_summary.csv", index=False)
    group_mae.to_csv(stage / "tables" / "group_mae.csv", index=False)
    contrasts.to_csv(stage / "tables" / "contrast_summary.csv", index=False)
    (stage / "tables" / "task_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_figures(metrics_table, contrasts, stage / "figures")
    manifest = {
        "analysis_name": "chemrxiv_frozen_analysis_v1",
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "formal_task_count": len(provenance),
        "formal_task_ids": [entry["task_name"] for entry in provenance],
        "input_provenance_sha256": hash_json(provenance),
        "formal_result_policy": "read-only; saved prediction parquet only; no model training",
        "external_rows": EXTERNAL_ROWS,
        "external_amine_groups": EXTERNAL_GROUPS,
        "unit_of_inference": "external_group_id / amine structure",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "type": "percentile cluster bootstrap"},
        "statistics": {
            "primary": "two-sided group-level Wilcoxon signed-rank, Holm within two marginal A contrasts per line",
            "secondary": "separate exploratory Holm family for four Full-input comparisons per line",
            "substitution": "pre-registered bootstrap-only derived estimand",
        },
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "output_file_sha256": file_hashes(stage),
    }
    (stage / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_existing(output: Path) -> None:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Existing output lacks its run manifest: {0}".format(output))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("output_file_sha256")
    observed = file_hashes(output)
    if not isinstance(expected, dict) or expected != observed:
        raise RuntimeError("Existing output hash manifest does not match its files: {0}".format(output))
    if manifest.get("analysis_script_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Existing output was produced by a different analysis-script revision")
    print("[analysis] existing bundle verified: {0}".format(output.relative_to(ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true", help="verify an existing immutable output bundle only")
    args = parser.parse_args()
    if OUTPUT.exists():
        if args.check_existing:
            check_existing(OUTPUT)
            return
        raise RuntimeError("Output already exists; use --check-existing or choose a new analysis version: {0}".format(OUTPUT))
    if args.check_existing:
        raise RuntimeError("No existing output bundle to check: {0}".format(OUTPUT))

    all_metrics: List[pd.DataFrame] = []
    all_group_mae: List[pd.DataFrame] = []
    all_contrasts: List[pd.DataFrame] = []
    provenance: List[Dict[str, Any]] = []
    for line_number, line in enumerate(LINES, start=1):
        predictions: Dict[str, pd.DataFrame] = {}
        for combo, _ in COMBOS:
            prediction, evidence = read_and_verify(line, combo)
            predictions[combo] = prediction
            provenance.append(evidence)
        metric_table, group_table, contrast_table = compute_line(line, join_combos(predictions), line_number)
        all_metrics.append(metric_table)
        all_group_mae.append(group_table)
        all_contrasts.append(contrast_table)
    metrics_table = pd.concat(all_metrics, ignore_index=True)
    group_mae = pd.concat(all_group_mae, ignore_index=True)
    contrasts = add_adjustments_and_interpretation(pd.concat(all_contrasts, ignore_index=True))

    stage = Path(tempfile.mkdtemp(prefix=".chemrxiv_frozen_analysis_v1.", dir=str(OUTPUT.parent)))
    try:
        write_bundle(stage, metrics_table, group_mae, contrasts, provenance)
        os.replace(stage, OUTPUT)
        stage = None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
    print("[analysis] 12/12 formal tasks verified; wrote {0}".format(OUTPUT.relative_to(ROOT)))
    print(contrasts.loc[:, ["line_label", "contrast", "effect_group_mean", "ci_low", "ci_high", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
