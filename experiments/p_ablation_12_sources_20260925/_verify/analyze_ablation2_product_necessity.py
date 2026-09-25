"""Reanalyse Ablation2 around whether product P materially improves MAE.

This is a read-only, post-score reanalysis.  It consumes the three completed
strict-analysis bundles and writes a new, isolated evidence bundle under
``project-docs/docs/ablation2/product_necessity_reanalysis_v1``.  It never trains a model,
changes an existing result, or reuses rows/folds/repeats as independent
chemical observations.

The operational estimand is

    Delta_P = MAE(A+B+C) - MAE(A+B+P+C).

Positive values mean that knowing P improves prediction.  The question is
one-sided: can P be omitted without a material *loss* in MAE?  A conservative
decision gate is ``upper 95% cluster-bootstrap CI <= 0.01``.  This is not a
symmetric equivalence claim and a non-significant two-sided test is never used
as evidence that P is unnecessary.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project-docs" / "docs" / "ablation2" / "product_necessity_reanalysis_v1"
SCRIPT = Path(__file__).resolve()
PRACTICAL_THRESHOLD = 0.01
DEV_BOOTSTRAP_DRAWS = 4_000
CHEMRXIV_BOOTSTRAP_DRAWS = 10_000

LINES = (
    ("morgan", "rf", "morgan_rf", "Morgan ECFP4 × RF"),
    ("mfp", "lightgbm", "mfp_lightgbm", "MFP × LightGBM"),
)
DEV_PROTOCOLS = {
    "random_repeat_group": "repeat_group_id",
    "unseen_amine": "amine_group_id",
    "unseen_acid": "acid_group_id",
    "unseen_substrate_pair": "substrate_pair_group_id",
}
PROTOCOL_LABELS = {
    "random_repeat_group": "随机重复组",
    "unseen_amine": "未见胺",
    "unseen_acid": "未见酸",
    "unseen_substrate_pair": "未见底物对",
}
SOURCE_BUNDLES = {
    "development": ROOT / "result" / "ablation2_statistics_v1",
    "frozen_external": ROOT / "project-docs" / "docs" / "ablation2" / "chemrxiv_frozen_analysis_v1",
    "same_source_unseen_amine": ROOT / "project-docs" / "docs" / "ablation2" / "chemrxiv_unseen_amine_analysis_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*values: Any) -> int:
    encoded = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**32)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    shown = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
    headers = [column.replace("_", " ") for column in columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(map(str, values)) + " |" for values in shown.itertuples(index=False, name=None))
    return "\n".join(rows)


def holm(values: pd.Series) -> pd.Series:
    """Holm adjustment, preserving null values and original order."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid.items(), start=1):
        running = max(running, min(1.0, (total - rank + 1) * float(value)))
        result.loc[index] = running
    return result


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def wilcoxon_summary(differences: np.ndarray) -> dict[str, Any]:
    """Descriptive paired test against zero; it is not an equivalence test."""
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return {"wilcoxon_statistic": 0.0, "p_raw": 1.0, "wilcoxon_method": "all_zero", "n_nonzero": 0}
    try:
        method = "exact" if len(nonzero) <= 50 else "auto"
        result = stats.wilcoxon(nonzero, alternative="two-sided", method=method)
    except ValueError:
        result = stats.wilcoxon(nonzero, alternative="two-sided", method="approx")
        method = "normal_approximation_due_to_ties"
    return {
        "wilcoxon_statistic": float(result.statistic),
        "p_raw": float(result.pvalue),
        "wilcoxon_method": method,
        "n_nonzero": int(len(nonzero)),
    }


def shapiro_summary(differences: np.ndarray) -> dict[str, float]:
    # A Shapiro p-value at tens of thousands of clusters is neither stable nor
    # decision-relevant; Wilcoxon is used regardless.  Retain it only where it
    # is a meaningful small-sample diagnostic.
    if not 3 <= len(differences) <= 5_000:
        return {"shapiro_w": float("nan"), "shapiro_p": float("nan")}
    result = stats.shapiro(differences)
    return {"shapiro_w": float(result.statistic), "shapiro_p": float(result.pvalue)}


def bootstrap_group_mean(differences: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    """Percentile interval resampling equal-weighted chemical groups."""
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), size=(draws, len(differences)))]
    return tuple(float(value) for value in np.quantile(sampled.mean(axis=1), [0.025, 0.975]))


def bootstrap_development(
    repeat_group_rows: dict[int, pd.DataFrame], draws: int, seed: int
) -> tuple[float, float]:
    """Match the completed development analysis's nested split/group design.

    First resample effective outer partitions; inside every selected partition
    resample its complete holdout-group table.  Group MAEs are weighted by the
    number of OOF rows in the group, so this remains an interval for the
    row-weighted MAE estimand while preserving the clustered sampling unit.
    Batching avoids allocating a 46k-group by 4k-draw dense array.
    """
    repeat_ids = sorted(repeat_group_rows)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    batch_size = 24
    for start in range(0, draws, batch_size):
        batch = min(batch_size, draws - start)
        selected_repeats = rng.integers(0, len(repeat_ids), size=(batch, len(repeat_ids)))
        numerators = np.zeros(batch, dtype=float)
        denominators = np.zeros(batch, dtype=float)
        for position in range(len(repeat_ids)):
            for repeat_position, repeat in enumerate(repeat_ids):
                target = np.flatnonzero(selected_repeats[:, position] == repeat_position)
                if not len(target):
                    continue
                group = repeat_group_rows[repeat]
                choices = rng.integers(0, len(group), size=(len(target), len(group)))
                weights = group["n"].to_numpy(dtype=float)[choices]
                delta = group["delta"].to_numpy(dtype=float)[choices]
                numerators[target] += (weights * delta).sum(axis=1)
                denominators[target] += weights.sum(axis=1)
        values[start:start + batch] = numerators / denominators
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def assert_bundle(name: str, paths: Iterable[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{name} strict analysis bundle is incomplete: {missing}")


def validate_sources() -> dict[str, str]:
    """Validate prior strict bundles before consuming their derived sources."""
    dev = SOURCE_BUNDLES["development"]
    frozen = SOURCE_BUNDLES["frozen_external"]
    unseen = SOURCE_BUNDLES["same_source_unseen_amine"]
    assert_bundle("development", [
        dev / "analysis-report.md", dev / "stats-appendix.md", dev / "figure-catalog.md", dev / "run_manifest.json",
        dev / "tables/task_provenance.parquet", dev / "tables/repeat_partition_signatures.csv",
        dev / "tables/paired_effects.csv",
    ])
    assert_bundle("frozen external", [
        frozen / "analysis-report.md", frozen / "stats-appendix.md", frozen / "figure-catalog.md", frozen / "run_manifest.json",
        frozen / "tables/task_provenance.json", frozen / "tables/group_mae.csv", frozen / "tables/contrast_summary.csv",
    ])
    assert_bundle("same-source unseen amine", [
        unseen / "analysis-report.md", unseen / "stats-appendix.md", unseen / "figure-catalog.md", unseen / "run_manifest.json",
        unseen / "tables/task_integrity.json", unseen / "tables/group_mean_mae.csv", unseen / "tables/a_contrasts.csv",
    ])
    dev_manifest = json.loads((dev / "run_manifest.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads((frozen / "run_manifest.json").read_text(encoding="utf-8"))
    unseen_manifest = json.loads((unseen / "run_manifest.json").read_text(encoding="utf-8"))
    if dev_manifest.get("formal_task_count") != 64 or dev_manifest.get("prediction_shard_count") != 960:
        raise RuntimeError("development analysis manifest does not prove the 64-task / 960-shard contract")
    if frozen_manifest.get("formal_task_count") != 12 or frozen_manifest.get("external_rows") != 957 or frozen_manifest.get("external_amine_groups") != 10:
        raise RuntimeError("frozen external manifest does not prove the 12-task / 957-row / 10-group contract")
    if unseen_manifest.get("bootstrap", {}).get("n_groups") != 10:
        raise RuntimeError("same-source manifest does not prove the 10-group inference contract")
    provenance = pd.read_parquet(dev / "tables/task_provenance.parquet")
    if len(provenance) != 64 or not provenance["state"].map(lambda value: value == {"succeeded": 15}).all():
        raise RuntimeError("development task provenance is incomplete")
    frozen_tasks = json.loads((frozen / "tables/task_provenance.json").read_text(encoding="utf-8"))
    unseen_tasks = json.loads((unseen / "tables/task_integrity.json").read_text(encoding="utf-8"))
    if len(frozen_tasks) != 12 or len(unseen_tasks) != 12:
        raise RuntimeError("a ChemRxiv task integrity table does not contain exactly 12 tasks")
    if any(item.get("integrity", {}).get("status") != "passed" for item in frozen_tasks):
        raise RuntimeError("frozen external task integrity did not pass")
    if any(item.get("status") != "passed" for item in unseen_tasks):
        raise RuntimeError("same-source task integrity did not pass")
    source_paths = [
        dev / "run_manifest.json", dev / "tables/task_provenance.parquet", dev / "tables/repeat_partition_signatures.csv",
        dev / "tables/paired_effects.csv", frozen / "run_manifest.json", frozen / "tables/task_provenance.json",
        frozen / "tables/group_mae.csv", frozen / "tables/contrast_summary.csv", unseen / "run_manifest.json",
        unseen / "tables/task_integrity.json", unseen / "tables/group_mean_mae.csv", unseen / "tables/a_contrasts.csv",
    ]
    for descriptor, model, _, _ in LINES:
        for protocol in DEV_PROTOCOLS:
            paired = dev / "tables" / f"paired_errors_{descriptor}_{model}_{protocol}.parquet"
            if not paired.is_file():
                raise RuntimeError(f"missing paired development errors: {paired}")
            source_paths.append(paired)
    source_paths.append(ROOT / "result" / "ablation2_amide_data_prep_v1" / "data" / "standardized_population.csv")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}


def summarize(
    *, track: str, line_id: str, line_label: str, protocol: str, group_unit: str,
    differences: np.ndarray, ci_low: float, ci_high: float, n_rows: int,
    nominal_repeats: int, distinct_partitions: int, row_weighted_effect: float,
    bootstrap_draws: int, bootstrap_seed: int,
) -> dict[str, Any]:
    test = wilcoxon_summary(differences)
    normality = shapiro_summary(differences)
    if ci_high <= PRACTICAL_THRESHOLD:
        classification = "supports_no_material_product_benefit_within_scope"
    elif ci_low > PRACTICAL_THRESHOLD:
        classification = "supports_material_product_benefit_within_scope"
    else:
        classification = "inconclusive_for_product_nonnecessity"
    return {
        "track": track,
        "line_id": line_id,
        "line_label": line_label,
        "protocol": protocol,
        "protocol_label": PROTOCOL_LABELS.get(protocol, protocol),
        "group_unit": group_unit,
        "formula": "MAE(A+B+C) − MAE(A+B+P+C)",
        "direction": "positive = product P lowers MAE",
        "n_rows": int(n_rows),
        "n_groups": int(len(differences)),
        "n_nominal_repeats": int(nominal_repeats),
        "n_distinct_partitions": int(distinct_partitions),
        "effect_group_mean": float(differences.mean()),
        "effect_group_sd": float(differences.std(ddof=1)),
        "effect_group_median": float(np.median(differences)),
        "effect_row_weighted": float(row_weighted_effect),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "practical_mae_threshold": PRACTICAL_THRESHOLD,
        "ci_upper_le_practical_threshold": bool(ci_high <= PRACTICAL_THRESHOLD),
        "n_positive_groups": int((differences > 0).sum()),
        "n_negative_groups": int((differences < 0).sum()),
        "n_zero_groups": int(np.isclose(differences, 0.0).sum()),
        "rank_biserial": rank_biserial(differences),
        **normality,
        **test,
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
        "classification": classification,
    }


def analyze_development() -> pd.DataFrame:
    """Recompute Delta_P from paired OOF error tables and original group keys."""
    base = SOURCE_BUNDLES["development"]
    population = pd.read_csv(
        ROOT / "result" / "ablation2_amide_data_prep_v1" / "data" / "standardized_population.csv",
        usecols=["sample_id", *DEV_PROTOCOLS.values()],
    )
    if len(population) != 47_015 or population["sample_id"].duplicated().any():
        raise RuntimeError("development population no longer has its fixed unique sample identity")
    partitions = pd.read_csv(base / "tables" / "repeat_partition_signatures.csv")
    if len(partitions) != 12:
        raise RuntimeError("development partition-signature audit is incomplete")
    rows: list[dict[str, Any]] = []
    for descriptor, model, line_id, line_label in LINES:
        for protocol, group_unit in DEV_PROTOCOLS.items():
            source = pd.read_parquet(base / "tables" / f"paired_errors_{descriptor}_{model}_{protocol}.parquet")
            required = {"repeat", "sample_id", "full", "minus_p"}
            if required.difference(source.columns):
                raise RuntimeError(f"{source}: missing a paired full/minus_p error column")
            if source.duplicated(["repeat", "sample_id"]).any() or len(source) != 3 * 47_015:
                raise RuntimeError(f"{source}: invalid paired OOF coverage")
            joined = source.loc[:, ["repeat", "sample_id", "full", "minus_p"]].merge(
                population.loc[:, ["sample_id", group_unit]], on="sample_id", how="left", validate="many_to_one"
            )
            if joined[group_unit].isna().any() or not np.isfinite(joined[["full", "minus_p"]].to_numpy(dtype=float)).all():
                raise RuntimeError(f"{source}: cannot align finite paired errors to {group_unit}")
            relevant = partitions.loc[partitions["protocol"].eq(protocol)].copy()
            if len(relevant) != 3:
                raise RuntimeError(f"{protocol}: expected three nominal partition records")
            effective = relevant.loc[relevant["is_representative"].astype(bool), "repeat"].astype(int).tolist()
            if not effective:
                raise RuntimeError(f"{protocol}: no effective outer partition")
            effective_joined = joined.loc[joined["repeat"].isin(effective)].copy()
            effective_joined["delta"] = effective_joined["minus_p"] - effective_joined["full"]
            group_rows: dict[int, pd.DataFrame] = {}
            for repeat, part in effective_joined.groupby("repeat", sort=True):
                grouped = part.groupby(group_unit, sort=False).agg(n=("sample_id", "size"), delta=("delta", "mean")).reset_index()
                group_rows[int(repeat)] = grouped
            per_group = pd.concat(group_rows.values(), ignore_index=True).groupby(group_unit, sort=False)["delta"].mean().to_numpy(dtype=float)
            seed = stable_seed("product-necessity", "development", descriptor, model, protocol)
            ci_low, ci_high = bootstrap_development(group_rows, DEV_BOOTSTRAP_DRAWS, seed)
            rows.append(summarize(
                track="development", line_id=line_id, line_label=line_label, protocol=protocol,
                group_unit=group_unit, differences=per_group, ci_low=ci_low, ci_high=ci_high,
                n_rows=len(effective_joined), nominal_repeats=3, distinct_partitions=len(effective),
                row_weighted_effect=float(effective_joined["delta"].mean()), bootstrap_draws=DEV_BOOTSTRAP_DRAWS,
                bootstrap_seed=seed,
            ))
    return pd.DataFrame(rows)


def analyze_frozen_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the frozen ten-amine group table; no frozen model is re-evaluated."""
    source = pd.read_csv(SOURCE_BUNDLES["frozen_external"] / "tables" / "group_mae.csv")
    expected = {"line_id", "line_label", "combo", "external_group_id", "group_mae", "n_rows"}
    if expected.difference(source.columns) or source.shape[0] != 120:
        raise RuntimeError("frozen group-MAE input does not have its 2 x 6 x 10 contract")
    rows: list[dict[str, Any]] = []
    group_outputs: list[pd.DataFrame] = []
    for _, _, line_id, line_label in LINES:
        subset = source.loc[(source["line_id"] == line_id) & source["combo"].isin(["full", "minus_p"])].copy()
        wide = subset.pivot(index="external_group_id", columns="combo", values="group_mae").sort_index()
        weights = subset.loc[subset["combo"].eq("full")].set_index("external_group_id")["n_rows"].reindex(wide.index)
        if wide.shape != (10, 2) or weights.isna().any():
            raise RuntimeError(f"frozen {line_id}: full/minus_p is not a paired ten-group matrix")
        difference = (wide["minus_p"] - wide["full"]).to_numpy(dtype=float)
        seed = stable_seed("product-necessity", "frozen", line_id)
        ci_low, ci_high = bootstrap_group_mean(difference, CHEMRXIV_BOOTSTRAP_DRAWS, seed)
        rows.append(summarize(
            track="frozen_external", line_id=line_id, line_label=line_label, protocol="frozen_development_to_external",
            group_unit="external_group_id / amine structure", differences=difference, ci_low=ci_low, ci_high=ci_high,
            n_rows=int(weights.sum()), nominal_repeats=1, distinct_partitions=1,
            row_weighted_effect=float(np.average(difference, weights=weights.to_numpy(dtype=float))),
            bootstrap_draws=CHEMRXIV_BOOTSTRAP_DRAWS, bootstrap_seed=seed,
        ))
        group_outputs.append(pd.DataFrame({
            "track": "frozen_external", "line_id": line_id, "line_label": line_label,
            "group_id": wide.index.astype(str), "delta_p": difference, "n_rows": weights.to_numpy(dtype=int),
        }))
    return pd.DataFrame(rows), pd.concat(group_outputs, ignore_index=True)


def analyze_same_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the repeat-averaged ten-group same-source OOF table."""
    source = pd.read_csv(SOURCE_BUNDLES["same_source_unseen_amine"] / "tables" / "group_mean_mae.csv")
    expected = {"line_id", "line_label", "combo", "group_id", "mae_group_mean", "n_rows_per_repeat", "n_repeats"}
    if expected.difference(source.columns) or source.shape[0] != 120:
        raise RuntimeError("same-source group-MAE input does not have its 2 x 6 x 10 contract")
    rows: list[dict[str, Any]] = []
    group_outputs: list[pd.DataFrame] = []
    for _, _, line_id, line_label in LINES:
        subset = source.loc[(source["line_id"] == line_id) & source["combo"].isin(["full", "minus_p"])].copy()
        wide = subset.pivot(index="group_id", columns="combo", values="mae_group_mean").sort_index()
        full_meta = subset.loc[subset["combo"].eq("full")].set_index("group_id").reindex(wide.index)
        if wide.shape != (10, 2) or not full_meta["n_repeats"].eq(3).all():
            raise RuntimeError(f"same-source {line_id}: full/minus_p is not a paired 10-group, 3-repeat matrix")
        difference = (wide["minus_p"] - wide["full"]).to_numpy(dtype=float)
        weights = full_meta["n_rows_per_repeat"].to_numpy(dtype=float)
        seed = stable_seed("product-necessity", "same-source", line_id)
        ci_low, ci_high = bootstrap_group_mean(difference, CHEMRXIV_BOOTSTRAP_DRAWS, seed)
        rows.append(summarize(
            track="same_source_unseen_amine", line_id=line_id, line_label=line_label, protocol="ChemRxiv_grouped_unseen_amine",
            group_unit="amine structure after three-repeat averaging", differences=difference, ci_low=ci_low, ci_high=ci_high,
            n_rows=int(weights.sum() * 3), nominal_repeats=3, distinct_partitions=3,
            row_weighted_effect=float(np.average(difference, weights=weights)),
            bootstrap_draws=CHEMRXIV_BOOTSTRAP_DRAWS, bootstrap_seed=seed,
        ))
        group_outputs.append(pd.DataFrame({
            "track": "same_source_unseen_amine", "line_id": line_id, "line_label": line_label,
            "group_id": wide.index.astype(str), "delta_p": difference, "n_rows_per_repeat": weights.astype(int),
        }))
    return pd.DataFrame(rows), pd.concat(group_outputs, ignore_index=True)


def add_descriptive_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    adjusted = summary.copy()
    adjusted["p_holm_descriptive"] = np.nan
    adjusted["holm_scope"] = "not assigned"
    # This family is intentionally descriptive: the corrected question was
    # selected after all scores existed.  It prevents raw zero-difference
    # p-values from being silently presented as many independent confirmations.
    for (track, line_id), indices in adjusted.groupby(["track", "line_id"]).groups.items():
        index = list(indices)
        adjusted.loc[index, "p_holm_descriptive"] = holm(adjusted.loc[index, "p_raw"])
        adjusted.loc[index, "holm_scope"] = f"descriptive Delta_P contrasts within {track}/{line_id}"
    return adjusted


def write_figures(summary: pd.DataFrame, groups: pd.DataFrame, directory: Path) -> list[dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10, "pdf.fonttype": 42})
    colors = {"morgan_rf": "#0072B2", "mfp_lightgbm": "#D55E00"}
    # Keep plot text ASCII: this runtime's default Matplotlib font has no CJK
    # glyph coverage.  The surrounding Chinese report/caption is authoritative.
    track_titles = {
        "development": "Development: four holdout protocols",
        "frozen_external": "Frozen development-to-ChemRxiv",
        "same_source_unseen_amine": "ChemRxiv grouped unseen-amine",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=False)
    for axis, track in zip(axes, ["development", "frozen_external", "same_source_unseen_amine"]):
        data = summary.loc[summary["track"].eq(track)].copy()
        if track == "development":
            data["plot_label"] = data["line_label"] + " / " + data["protocol"]
        else:
            data["plot_label"] = data["line_label"]
        data = data.iloc[::-1].reset_index(drop=True)
        for y, row in data.iterrows():
            axis.errorbar(
                row["effect_group_mean"], y,
                xerr=[[row["effect_group_mean"] - row["ci95_low"]], [row["ci95_high"] - row["effect_group_mean"]]],
                fmt="o", color=colors[row["line_id"]], capsize=3, ms=5, zorder=3,
            )
        axis.axvline(0, color="#444444", lw=0.8)
        axis.axvline(PRACTICAL_THRESHOLD, color="#666666", lw=0.9, ls="--")
        axis.set_yticks(range(len(data)), data["plot_label"], fontsize=7)
        axis.set_title(track_titles[track])
        axis.set_xlabel("Delta P = MAE(without P) - MAE(with P)")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Product-P predictive benefit: positive lowers MAE; dashed line = 0.01 MAE", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / "figure-01-product-necessity-forest.pdf", bbox_inches="tight")
    fig.savefig(directory / "figure-01-product-necessity-forest.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.5), sharex=True)
    for row_index, track in enumerate(["frozen_external", "same_source_unseen_amine"]):
        for column, (_, _, line_id, line_label) in enumerate(LINES):
            axis = axes[row_index, column]
            values = groups.loc[(groups["track"] == track) & (groups["line_id"] == line_id)].sort_values("delta_p")
            y = np.arange(1, len(values) + 1)
            axis.scatter(values["delta_p"], y, c=colors[line_id], s=28, alpha=0.85)
            axis.axvline(0, color="#444444", lw=0.8)
            axis.axvline(PRACTICAL_THRESHOLD, color="#666666", lw=0.9, ls="--")
            axis.set_ylim(0.3, 10.7)
            axis.set_yticks([1, 5, 10], ["group 1", "group 5", "group 10"] if column == 0 else [])
            axis.set_title(f"{track_titles[track]}\n{line_label}")
            axis.grid(axis="x", alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("group-level Delta P")
    fig.suptitle("ChemRxiv: 10 amine-group paired differences (sorted, anonymous order)", y=0.99, fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / "figure-02-chemrxiv-group-heterogeneity.pdf", bbox_inches="tight")
    fig.savefig(directory / "figure-02-chemrxiv-group-heterogeneity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return [
        {
            "file": "figures/figure-01-product-necessity-forest.pdf",
            "purpose": "在三条不可合并证据轨道内展示 ΔP、95% 簇 bootstrap 区间与 0.01 MAE 门槛。",
        },
        {
            "file": "figures/figure-02-chemrxiv-group-heterogeneity.pdf",
            "purpose": "展示两条 ChemRxiv 轨道中十个胺结构的 ΔP 异质性，避免将 957 行误作 957 个独立观察。",
        },
    ]


def result_sentence(row: pd.Series) -> str:
    return (
        f"{row['line_label']} / {row['protocol_label']}: ΔP={row['effect_group_mean']:.4f} "
        f"(95% CI {row['ci95_low']:.4f}–{row['ci95_high']:.4f}; n={int(row['n_groups'])} 组)"
    )


def render_analysis_report(summary: pd.DataFrame) -> str:
    display = summary.copy()
    display["decision"] = display["classification"].map({
        "supports_no_material_product_benefit_within_scope": "CI 上界 ≤ 0.01：轨道内支持可省略 P",
        "supports_material_product_benefit_within_scope": "CI 下界 > 0.01：轨道内支持 P 有实质增益",
        "inconclusive_for_product_nonnecessity": "区间跨 0.01：对可省略 P 不确定",
    })
    table = markdown_table(display, [
        "track", "line_label", "protocol_label", "n_groups", "effect_group_mean", "ci95_low", "ci95_high", "p_raw",
        "rank_biserial", "decision",
    ])
    no_material = summary.loc[summary["classification"].eq("supports_no_material_product_benefit_within_scope")]
    uncertain = summary.loc[summary["classification"].eq("inconclusive_for_product_nonnecessity")]
    material = summary.loc[summary["classification"].eq("supports_material_product_benefit_within_scope")]
    bullets = []
    if len(no_material):
        bullets.append("在下列轨道内，95% CI 上界不超过 0.01：" + "；".join(result_sentence(row) for _, row in no_material.iterrows()) + "。")
    if len(uncertain):
        bullets.append("以下结果的上界仍超过 0.01，故不能把‘未发现差异’写成 P 不重要：" + "；".join(result_sentence(row) for _, row in uncertain.iterrows()) + "。")
    if len(material):
        bullets.append("以下轨道的区间完全超过 0.01，显示 P 的预测增益在该限定设置中有实际意义：" + "；".join(result_sentence(row) for _, row in material.iterrows()) + "。")
    if not bullets:
        bullets.append("所有轨道均对 P 是否可省略不确定。")
    return f"""# Ablation2：产物 P 是否可省略的严格重分析

## 分析问题

在 A=胺、B=酸、C=记录条件均已经知道的设定下，加入产物结构 P 是否带来**实质**的产率预测收益？本分析的唯一核心量为：

`ΔP = MAE(A+B+C) − MAE(A+B+P+C)`。

正值表示 P 使 MAE 下降。实践问题是单侧的：若 P 被删除造成的最坏合理 MAE 损失不超过 0.01，P 在该设定内可被视为不必提供。因而判定门槛是 ΔP 的 95% 簇 bootstrap 区间**上界** `≤ 0.01`；双侧 p 值不用于证明“没有重要性”。

## 结论摘要

这是一项**纠正后的事后重分析**。此前实验、冻结矩阵和统计产生时，主要问题是产物是否改变 A 的条件预测贡献（Δsub,A），并非本文件的 ΔP 问题。因此，本文件提供可信的已观察证据与下一步决策，但不把任一结果称为对新核心假设的预注册确认。

{chr(10).join('- ' + item for item in bullets)}

三条轨道绝不合并：开发集是同一 47,015 行人群内的交叉验证；冻结轨道是 47,015 行训练到 957 行 ChemRxiv 的一次外测；同源轨道是在该 957 行内做胺组留出。两条模型线是稳健性实现，不是独立化学重复。

## 精确数值

{table}

`p_raw` 是以组均 ΔP 为单位的双侧 Wilcoxon 对零检验，仅描述方向；`p_holm_descriptive`（见统计附录）在同一轨道/模型线的 ΔP 对照内校正。它们既不能证明非劣，也不能把事后问题变成确认性检验。

## 解释边界

- 删除 P 只比较模型在一个信息可得性设定下的条件预测表现；它不是化学因果干预、分子互信息估计，也不表示所有实际工作流都预先知道目标产物。
- 0.01 是“产品是否值得额外提供”的操作性 MAE 容忍度，不是化学效应阈值。其上界判定回答“可否省略 P”，而非“两种模型绝对等价”。
- 对开发集未见酸协议，只有一个有效外层划分；其区间条件于该划分。ChemRxiv 两条轨道都只有 10 个胺组、一个酸，不能外推为未见酸或跨反应家族结论。
- 当 CI 跨过 0.01 时，正确结果是“尚不能确认 P 不重要”，不是“P 很重要”，也不是“P 没有作用”。

## Claim Candidates

- Claim: 当前证据能够回答“P 是否带来超过 0.01 MAE 的额外预测收益”，但结论必须逐轨道、逐模型线给出。
  - Source evidence: `tables/product_effects.csv` 与两张实际图。
  - Allowed wording: “在 [指定轨道/模型] 中，ΔP 的 95% CI [未/已] 将 0.01 排除在可省略范围之外。”
  - Forbidden stronger wording: “产物在酰胺反应中普遍不重要”或“已证明知道反应物后产物无信息”。
  - Uncertainty: 新核心问题是在评分后更正；ChemRxiv 只有 10 个 A 组且单酸。
  - Next check: 一个独立、多酸、角色兼容的数据源上的预注册冻结 ΔP 验证。
  - Decision: revise.

- Claim: 双侧不显著不是 P 不重要的证据。
  - Source evidence: 所有 ΔP 的 CI 与 0.01 门槛列。
  - Allowed wording: “未达到零差异显著性不等于满足实际可省略界限。”
  - Forbidden stronger wording: “p>0.05 因而 P 无用。”
  - Uncertainty: 小组数限制区间精度。
  - Next check: 事先锁定同一单侧实用界限的外部复制。
  - Decision: keep.
"""


def render_stats_appendix(summary: pd.DataFrame) -> str:
    detail = markdown_table(summary, [
        "track", "line_id", "protocol", "group_unit", "n_rows", "n_groups", "n_nominal_repeats",
        "n_distinct_partitions", "effect_group_mean", "effect_group_sd", "effect_group_median",
        "effect_row_weighted", "ci95_low", "ci95_high", "n_positive_groups", "n_negative_groups",
        "rank_biserial", "shapiro_w", "shapiro_p", "wilcoxon_statistic", "p_raw", "p_holm_descriptive",
        "bootstrap_draws", "bootstrap_seed", "classification",
    ])
    return f"""# 统计附录：ΔP 产物必要性重分析

## 比较、方向与独立单位

估计量固定为 `ΔP = MAE(A+B+C) − MAE(A+B+P+C)`。正值意味着 P 降低 MAE。开发集直接从配对 OOF 绝对误差重算；同一 `repeat, sample_id` 的 Full 和 minus-P 误差一一配对，再映射到预先声明的留出组。其推断单位是协议相应的留出组，且只有不同的外层划分被计作重复。ChemRxiv 冻结外测与同源未见胺轨道均以十个胺结构为单位；后者每个胺组先跨三次 OOF repeat 平均。

## 区间与实用判定

开发集使用 {DEV_BOOTSTRAP_DRAWS:,} 次嵌套 percentile bootstrap：先重采样有效外层划分、再在每个所选划分内重采样完整留出组，并按组中的 OOF 行数加权得到行加权 MAE 差。两条 ChemRxiv 轨道用 {CHEMRXIV_BOOTSTRAP_DRAWS:,} 次等权胺组 percentile bootstrap。每项固定 seed 已记录在表中。

实际意义门槛为 `δ=0.01` MAE。因为决策是“删掉 P 会不会造成足以影响采用的性能损失”，采用保守单侧门槛：仅当 95% CI 上界 `≤δ` 才把该轨道标记为“轨道内支持 P 无实质收益”。这不是双侧等效检验，更不是关于真实化学机制的结论。

## 描述性零差异检验

对每个组均 ΔP 进行了双侧配对 Wilcoxon signed-rank；小样本无零差异时使用 exact 方法。Shapiro-Wilk 仅在 3–5,000 组时报告；无论其值如何，因簇化、偏态和小样本风险，均不以参数检验替代 Wilcoxon。Holm 调整只覆盖同一轨道/模型线内报告的 ΔP 零差异对照，且整体只是事后描述，不是确认性显著性家族。

## 完整数值

{detail}

## 统计限制

1. 这不是新的独立实验：核心问题的更正发生在既有分数之后。区间量化已观察数据下的不确定性，不恢复预注册地位。
2. 模型线共享相同化学样本和划分，不能把两行结果合并或当作两次独立化学复制。
3. 开发集的未见酸仅一个有效划分；ChemRxiv 只有十个胺组和一个酸。重复 CV、折和行数均未被当成额外独立组。
4. 双侧 `p_raw` 的零假设与“ΔP ≤ 0.01”不同，所以 p 值不能替代上界判定。
"""


def render_figure_catalog(specs: list[dict[str, str]]) -> str:
    return f"""# Figure catalog：ΔP 产物必要性重分析

## Figure 01 — product-necessity forest

- Filename: `{specs[0]['file']}`（同名 PNG 供快速查看）
- Purpose: {specs[0]['purpose']}
- Data source: `tables/product_effects.csv`，来自三个已完成严格分析包的配对误差或组均 MAE。
- Caption requirements: 定义 `ΔP=MAE(A+B+C)-MAE(A+B+P+C)`；正值含义；误差线为相应簇 bootstrap 95% CI；虚线为 0.01 MAE；三条轨道不可合并。
- Key observation: 读者应看区间上界相对 0.01 的位置，而不是仅看点估计或 p 值。
- Interpretation checklist: (1) 图为何存在？比较 P 的实际预测收益；(2) 注意什么？是否所有合理 ΔP 都不超过 0.01；(3) 决策变化？只有该条件成立的限定轨道才允许“可省略 P”的操作性表述。
- Known caveat: 核心问题为事后纠正；各轨道的训练/测试人群和重采样设计不同。

## Figure 02 — ChemRxiv group heterogeneity

- Filename: `{specs[1]['file']}`（同名 PNG 供快速查看）
- Purpose: {specs[1]['purpose']}
- Data source: `tables/chemrxiv_group_product_effects.csv`。
- Caption requirements: 每点是一个胺结构的组均 ΔP；十点不是按化学身份排序；同源轨道已先跨 3 个 repeat 平均；虚线为 0.01 MAE。
- Key observation: 组间差异直接展示了为什么 957 行不能被误用为 957 个独立推断单位。
- Interpretation checklist: (1) 图为何存在？检查组级异质性；(2) 注意什么？差异跨越零和实用界限的程度；(3) 决策变化？不从聚合 MAE 或单点估计推出普遍必要性。
- Known caveat: 两条 ChemRxiv 轨道只有一个酸；冻结与同源设计不能合并。
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def write_bundle(summary: pd.DataFrame, groups: pd.DataFrame, source_hashes: dict[str, str]) -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis bundle: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".product_necessity_reanalysis_v1.", dir=OUT.parent))
    try:
        tables = stage / "tables"
        tables.mkdir()
        summary.to_csv(tables / "product_effects.csv", index=False)
        groups.to_csv(tables / "chemrxiv_group_product_effects.csv", index=False)
        decisions = summary.loc[:, [
            "track", "line_id", "protocol", "effect_group_mean", "ci95_low", "ci95_high",
            "practical_mae_threshold", "ci_upper_le_practical_threshold", "classification",
        ]]
        decisions.to_csv(tables / "product_necessity_decisions.csv", index=False)
        figures = write_figures(summary, groups, stage / "figures")
        analysis = render_analysis_report(summary)
        appendix = render_stats_appendix(summary)
        catalog = render_figure_catalog(figures)
        write_text(stage / "analysis-report.md", analysis)
        write_text(stage / "stats-appendix.md", appendix)
        write_text(stage / "figure-catalog.md", catalog)
        write_text(
            stage / "report" / "analysis-report.html",
            "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>Ablation2 product necessity reanalysis</title><body><pre>"
            + html.escape(analysis) + "</pre></body></html>\n",
        )
        output_hashes = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        manifest = {
            "analysis_id": "product_necessity_reanalysis_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_script": str(SCRIPT.relative_to(ROOT)),
            "analysis_script_sha256": sha256_file(SCRIPT),
            "analysis_mode": "read_only_post_score_reanalysis",
            "model_fits_started": 0,
            "corrected_core_question": "Given A+B+C, does adding P materially improve yield prediction?",
            "estimand": "Delta_P = MAE(A+B+C) - MAE(A+B+P+C); positive means P lowers MAE",
            "decision_rule": "within-track operational non-necessity only if upper 95% cluster-bootstrap CI <= 0.01 MAE",
            "post_hoc_status": "The central question was corrected after the prior plans and scores; no result is a preregistered confirmation of this endpoint.",
            "source_bundle_hashes": source_hashes,
            "statistics": {
                "development": {"draws": DEV_BOOTSTRAP_DRAWS, "method": "nested effective-outer-partition and holdout-group percentile bootstrap"},
                "chemrxiv": {"draws": CHEMRXIV_BOOTSTRAP_DRAWS, "method": "equal-amine-group percentile bootstrap"},
                "zero_difference_test": "two-sided paired Wilcoxon, descriptive only; Holm within track/model line",
            },
            "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "output_file_sha256_excluding_manifest": output_hashes,
        }
        write_text(stage / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, OUT)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return OUT


def check_existing() -> None:
    required = [
        OUT / "analysis-report.md", OUT / "stats-appendix.md", OUT / "figure-catalog.md", OUT / "run_manifest.json",
        OUT / "tables/product_effects.csv", OUT / "tables/chemrxiv_group_product_effects.csv",
        OUT / "figures/figure-01-product-necessity-forest.pdf", OUT / "figures/figure-02-chemrxiv-group-heterogeneity.pdf",
    ]
    assert_bundle("product necessity reanalysis", required)
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != "product_necessity_reanalysis_v1" or manifest.get("model_fits_started") != 0:
        raise RuntimeError("product necessity run manifest has an invalid identity")
    expected_sources = validate_sources()
    if manifest.get("source_bundle_hashes") != expected_sources:
        raise RuntimeError("source artifacts have changed since the product-necessity bundle was created")
    summary = pd.read_csv(OUT / "tables/product_effects.csv")
    if len(summary) != 12 or not summary["ci_upper_le_practical_threshold"].isin([True, False]).all():
        raise RuntimeError("product necessity summary is incomplete")
    print(json.dumps({"status": "passed", "output": str(OUT.relative_to(ROOT)), "rows": len(summary)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true", help="validate, but never overwrite, the existing bundle")
    args = parser.parse_args()
    if args.check_existing:
        check_existing()
        return 0
    source_hashes = validate_sources()
    development = analyze_development()
    frozen, frozen_groups = analyze_frozen_external()
    same_source, same_source_groups = analyze_same_source()
    summary = add_descriptive_adjustments(pd.concat([development, frozen, same_source], ignore_index=True))
    if len(summary) != 12 or not np.isfinite(summary[["effect_group_mean", "ci95_low", "ci95_high"]].to_numpy(dtype=float)).all():
        raise RuntimeError("reanalysis did not produce 12 finite product contrasts")
    output = write_bundle(summary, pd.concat([frozen_groups, same_source_groups], ignore_index=True), source_hashes)
    print(json.dumps({
        "status": "passed", "output": str(output.relative_to(ROOT)),
        "product_effects": summary.loc[:, ["track", "line_id", "protocol", "effect_group_mean", "ci95_low", "ci95_high", "classification"]].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
