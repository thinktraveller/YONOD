"""Evidence-first analysis of the completed Ablation2 formal matrix.

Reads immutable formal outputs, validates every task, and writes the isolated
``result/ablation2_statistics_v1`` analysis task.  No model is fitted here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "ablation2_statistics_v1"
POPULATION = ROOT / "result" / "ablation2_amide_data_prep_v1/data/standardized_population.csv"
SPLIT_ROOT = ROOT / "result" / "ablation2_amide_splits_v1/manifests"
COMBOS = ["full", "minus_a", "minus_b", "minus_p", "minus_a_p", "minus_b_p", "product_conditions", "conditions_only"]
METHODS = [("morgan", "rf", "Morgan ECFP4 × RF"), ("mfp", "lightgbm", "MFP × LightGBM")]
PROTOCOLS = {
    "random_repeat_group": "repeat_group_id",
    "unseen_amine": "amine_group_id",
    "unseen_acid": "acid_group_id",
    "unseen_substrate_pair": "substrate_pair_group_id",
}
SOURCE_VALID_COUNTS: dict[str, pd.Series] = {}
REPEAT_PARTITIONS: dict[str, pd.DataFrame] = {}
SHORT = {"full": "A+B+P+C", "minus_a": "B+P+C", "minus_b": "A+P+C", "minus_p": "A+B+C", "minus_a_p": "B+C", "minus_b_p": "A+C", "product_conditions": "P+C", "conditions_only": "C"}
COLOR = {"full": "#0072B2", "minus_a": "#56B4E9", "minus_b": "#009E73", "minus_p": "#E69F00", "minus_a_p": "#D55E00", "minus_b_p": "#CC79A7", "product_conditions": "#777777", "conditions_only": "#000000"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_dir(descriptor: str, model: str, protocol: str, combo: str) -> Path:
    return ROOT / "result" / f"ablation2_formal_{descriptor}_{model}_{protocol}_{combo}_v1"


def task_prediction_paths(path: Path) -> list[Path]:
    return sorted((path / "docs/predictions").glob("*.parquet"))


def validate_task(descriptor: str, model: str, protocol: str, combo: str) -> tuple[pd.DataFrame, dict]:
    root = task_dir(descriptor, model, protocol, combo)
    if not root.is_dir():
        raise RuntimeError(f"Missing task directory: {root}")
    manifest_path = root / "docs/manifests/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    external = SPLIT_ROOT / f"{protocol}_split_manifest.parquet"
    if not external.is_file():
        raise RuntimeError(f"Missing protocol manifest: {external}")
    if manifest.get("external_split_manifest", {}).get("sha256") != sha256(external):
        raise RuntimeError(f"External split hash mismatch: {root.name}")
    if Path(manifest["external_split_manifest"]["source_path"]).resolve() != external.resolve():
        raise RuntimeError(f"External split path mismatch: {root.name}")
    for suffix in ("html", "md"):
        report = root / "report" / f"benchmark_report.{suffix}"
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"Missing report {report}")
    with sqlite3.connect((root / "docs/state/tasks.sqlite").as_uri() + "?mode=ro", uri=True) as con:
        task_counts = dict(con.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"))
    if task_counts != {"succeeded": 15}:
        raise RuntimeError(f"Incomplete task state {root.name}: {task_counts}")
    paths = task_prediction_paths(root)
    if len(paths) != 15:
        raise RuntimeError(f"Expected 15 prediction shards in {root.name}; found {len(paths)}")
    prediction = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {"run_id", "config_hash", "split_id", "sample_id", "repeat", "fold", "y_true", "y_pred"}
    if required.difference(prediction.columns):
        raise RuntimeError(f"Prediction contract incomplete: {root.name}")
    if len(prediction) != 47015 * 3 or prediction.duplicated(["sample_id", "repeat"]).any():
        raise RuntimeError(f"OOF rows are incomplete or duplicated: {root.name}")
    if prediction[["repeat", "fold"]].drop_duplicates().shape[0] != 15:
        raise RuntimeError(f"Missing repeat/fold pair: {root.name}")
    if not np.isfinite(prediction[["y_true", "y_pred"]].to_numpy()).all():
        raise RuntimeError(f"Non-finite prediction: {root.name}")
    if set(prediction.run_id.astype(str)) != {str(manifest["run_id"])} or set(prediction.config_hash.astype(str)) != {str(manifest["config_hash"])}:
        raise RuntimeError(f"Prediction identity mismatch: {root.name}")
    observed = prediction.groupby(["repeat", "fold"]).size().sort_index()
    expected = SOURCE_VALID_COUNTS[protocol]
    if not observed.equals(expected):
        raise RuntimeError(f"OOF fold membership count mismatch: {root.name}")
    return prediction, {"task": root.name, "run_id": manifest["run_id"], "config_hash": manifest["config_hash"], "external_split_sha256": sha256(external), "state": task_counts}


def holm(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    total = len(valid)
    running = 0.0
    for rank, (index, value) in enumerate(valid.items(), 1):
        running = max(running, min(1.0, (total - rank + 1) * value))
        result.loc[index] = running
    return result


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y, pred = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    tau = stats.kendalltau(y, pred).statistic
    return {"mae": float(np.abs(y - pred).mean()), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))), "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)), "kendall_tau": float(tau)}


def partition_signatures(protocol: str) -> pd.DataFrame:
    """Identify effective outer partitions from valid sample-to-fold assignment.

    A nominal repeat with the same ordered ``sample_id -> fold`` mapping as an
    earlier repeat is a duplicate assignment, not an additional independent
    outer split.  The digest is persisted with the analysis outputs for audit.
    """
    source = pd.read_parquet(
        SPLIT_ROOT / f"{protocol}_split_manifest.parquet",
        columns=["repeat", "fold", "role", "sample_id"],
    )
    records = []
    for repeat, part in source.loc[source.role.eq("valid")].groupby("repeat", sort=True):
        ordered = part[["sample_id", "fold"]].sort_values("sample_id")
        payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
        records.append({"protocol": protocol, "repeat": int(repeat), "partition_signature": hashlib.sha256(payload).hexdigest()})
    result = pd.DataFrame(records)
    result["is_representative"] = ~result.partition_signature.duplicated()
    result["representative_repeat"] = result.groupby("partition_signature")["repeat"].transform("min")
    if len(result) != 3:
        raise RuntimeError(f"Expected three nominal repeats in {protocol}, found {len(result)}")
    return result


def bootstrap_effects(group_rows: dict[int, pd.DataFrame], seed: int, n_bootstrap: int = 4000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = ["full", "minus_a", "minus_b", "minus_p", "minus_a_p", "minus_b_p"]
    repeat_ids = sorted(group_rows)
    numerators = {name: np.zeros(n_bootstrap, dtype=float) for name in names}
    denominators = np.zeros(n_bootstrap, dtype=float)
    # Bootstrap outer repeats first, then complete holdout groups inside each
    # selected repeat.  Array-indexed draws avoid a Python/Pandas loop per draw.
    selected_repeats = rng.choice(repeat_ids, size=(n_bootstrap, len(repeat_ids)), replace=True)
    for position in range(len(repeat_ids)):
        for repeat in repeat_ids:
            rows = np.flatnonzero(selected_repeats[:, position] == repeat)
            if not len(rows):
                continue
            part = group_rows[repeat]
            draw = rng.integers(0, len(part), size=(len(rows), len(part)))
            weights = part["n"].to_numpy(dtype=float)[draw]
            denominators[rows] += weights.sum(axis=1)
            for name in names:
                values = part[name].to_numpy(dtype=float)[draw]
                numerators[name][rows] += (weights * values).sum(axis=1)
    mae = {name: value / denominators for name, value in numerators.items()}
    return {
        "a_with_p": mae["minus_a"] - mae["full"],
        "a_without_p": mae["minus_a_p"] - mae["minus_p"],
        "b_with_p": mae["minus_b"] - mae["full"],
        "b_without_p": mae["minus_b_p"] - mae["minus_p"],
        "a_substitution": (mae["minus_a_p"] - mae["minus_p"]) - (mae["minus_a"] - mae["full"]),
        "b_substitution": (mae["minus_b_p"] - mae["minus_p"]) - (mae["minus_b"] - mae["full"]),
    }


def paired_analysis(predictions: dict[str, pd.DataFrame], protocol: str, descriptor: str, model: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = ["full", "minus_a", "minus_b", "minus_p", "minus_a_p", "minus_b_p"]
    indexed = {name: predictions[name].set_index(["repeat", "sample_id"]).sort_index() for name in needed}
    reference = indexed["full"]
    for name in needed[1:]:
        if not reference.index.equals(indexed[name].index):
            raise RuntimeError(f"Paired OOF membership differs: {descriptor}/{model}/{protocol}/{name}")
        if not np.allclose(reference.y_true.to_numpy(), indexed[name].y_true.to_numpy()):
            raise RuntimeError("Paired labels differ")
    joined = reference[["y_true"]].copy()
    for name, part in indexed.items():
        joined[name] = np.abs(part.y_true.to_numpy() - part.y_pred.to_numpy())
    joined = joined.reset_index().merge(GROUP_MAP[["sample_id", group_col]], on="sample_id", validate="many_to_one")
    group_rows = {}
    for repeat, part in joined.groupby("repeat", sort=True):
        grouped = part.groupby(group_col, sort=False).agg(n=("sample_id", "size"), **{name: (name, "mean") for name in needed}).reset_index()
        group_rows[int(repeat)] = grouped
    partition_info = REPEAT_PARTITIONS[protocol]
    effective_repeats = partition_info.loc[partition_info.is_representative, "repeat"].astype(int).tolist()
    if set(effective_repeats).difference(group_rows):
        raise RuntimeError(f"Missing representative repeat for {protocol}")
    effective_joined = joined.loc[joined.repeat.isin(effective_repeats)].copy()
    effective_group_rows = {repeat: group_rows[repeat] for repeat in effective_repeats}
    n_nominal_repeats = int(len(group_rows))
    n_distinct_partitions = int(len(effective_repeats))
    effect_defs = {
        "a_with_p": ("minus_a", "full", "A contribution with P"),
        "a_without_p": ("minus_a_p", "minus_p", "A contribution without P"),
        "b_with_p": ("minus_b", "full", "B contribution with P"),
        "b_without_p": ("minus_b_p", "minus_p", "B contribution without P"),
    }
    values = []
    seed = int.from_bytes(hashlib.sha256(f"{descriptor}|{model}|{protocol}".encode("utf-8")).digest()[:4], "big")
    bootstrap = bootstrap_effects(effective_group_rows, seed=seed)
    # The inferential unit is one held-out group, averaged only over distinct
    # outer partitions.  Repeated identical assignments cannot increase
    # independent evidence.
    all_groups = pd.concat(effective_group_rows.values(), keys=effective_group_rows.keys(), names=["repeat", "row"])
    for name, (reduced, baseline, label) in effect_defs.items():
        per_group = all_groups.groupby(group_col, sort=False).apply(lambda x: (x[reduced] - x[baseline]).mean())
        nonzero = per_group.loc[~np.isclose(per_group, 0)]
        test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto") if len(nonzero) else None
        row_effect = float(effective_joined[reduced].mean() - effective_joined[baseline].mean())
        ci = np.quantile(bootstrap[name], [0.025, 0.975])
        values.append({"descriptor": descriptor, "model": model, "protocol": protocol, "group_unit": group_col, "contrast": name, "label": label, "effect_mae": row_effect, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "n_nominal_repeats": n_nominal_repeats, "n_distinct_partitions": n_distinct_partitions, "n_groups": int(len(per_group)), "group_median_effect": float(per_group.median()), "wilcoxon_statistic": float(test.statistic) if test else np.nan, "p_raw": float(test.pvalue) if test else np.nan, "inference": f"two-sided paired Wilcoxon signed-rank on group mean error differences averaged over {n_distinct_partitions} distinct outer partition(s)"})
    for name, label in [("a_substitution", "A product substitution effect"), ("b_substitution", "B product substitution effect")]:
        ci = np.quantile(bootstrap[name], [0.025, 0.975])
        if name == "a_substitution":
            row_effect = float((effective_joined["minus_a_p"].mean() - effective_joined["minus_p"].mean()) - (effective_joined["minus_a"].mean() - effective_joined["full"].mean()))
        else:
            row_effect = float((effective_joined["minus_b_p"].mean() - effective_joined["minus_p"].mean()) - (effective_joined["minus_b"].mean() - effective_joined["full"].mean()))
        values.append({"descriptor": descriptor, "model": model, "protocol": protocol, "group_unit": group_col, "contrast": name, "label": label, "effect_mae": row_effect, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "n_nominal_repeats": n_nominal_repeats, "n_distinct_partitions": n_distinct_partitions, "n_groups": int(np.mean([len(v) for v in effective_group_rows.values()])), "group_median_effect": np.nan, "wilcoxon_statistic": np.nan, "p_raw": np.nan, "inference": "nested distinct-partition/group bootstrap only; this derived difference is summarized with a percentile 95% CI"})
    return pd.DataFrame(values), joined


def write_figures(metrics: pd.DataFrame, effects: pd.DataFrame, figure_dir: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.2), sharey=False)
    for row, (descriptor, model, label) in enumerate(METHODS):
        for col, protocol in enumerate(PROTOCOLS):
            axis = axes[row, col]
            data = metrics[(metrics.descriptor == descriptor) & (metrics.model == model) & (metrics.protocol == protocol) & (metrics.metric == "mae")]
            order = COMBOS
            means = data.set_index("combo").loc[order, "mean"].to_numpy()
            low = data.set_index("combo").loc[order, "ci_low"].to_numpy()
            high = data.set_index("combo").loc[order, "ci_high"].to_numpy()
            x = np.arange(len(order))
            valid_interval = np.isfinite(low) & np.isfinite(high)
            if valid_interval.any():
                axis.errorbar(x[valid_interval], means[valid_interval], yerr=np.vstack([means[valid_interval]-low[valid_interval], high[valid_interval]-means[valid_interval]]), fmt="none", color="#333333", capsize=2, lw=.8)
            axis.scatter(x, means, c=[COLOR[item] for item in order], s=26, zorder=3)
            axis.set_xticks(x, [SHORT[item] for item in order], rotation=55, ha="right", fontsize=6)
            axis.set_title(f"{label}; {protocol.replace('_', ' ')}")
            axis.set_ylabel("OOF MAE (yield fraction)")
            axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure-01-mae-ablation-matrix.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "figure-01-mae-ablation-matrix.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 4.8), sharex=True)
    labels = ["A substitution", "B substitution"]
    for row, (descriptor, model, method_label) in enumerate(METHODS):
        for col, protocol in enumerate(PROTOCOLS):
            axis = axes[row, col]
            data = effects[(effects.descriptor == descriptor) & (effects.model == model) & (effects.protocol == protocol) & (effects.contrast.isin(["a_substitution", "b_substitution"]))].set_index("contrast")
            for y, contrast in enumerate(["a_substitution", "b_substitution"]):
                item = data.loc[contrast]
                axis.errorbar(item.effect_mae, y, xerr=[[item.effect_mae-item.ci_low], [item.ci_high-item.effect_mae]], fmt="o", color=["#D55E00", "#0072B2"][y], capsize=3)
            axis.axvline(0, color="#444444", lw=.8)
            axis.set_yticks([0, 1], labels if col == 0 else [])
            axis.set_title(f"{method_label}; {protocol.replace('_', ' ')}")
            axis.set_xlabel("MAE substitution effect")
            axis.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure-02-substitution-effects.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "figure-02-substitution-effects.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def to_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "<!doctype html><meta charset='utf-8'><title>Ablation2 analysis</title><style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.45}pre{white-space:pre-wrap}</style><pre>" + escaped + "</pre>"


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing analysis task: {OUT}")
    (OUT / "tables").mkdir(parents=True)
    figure_dir = OUT / "figures"; figure_dir.mkdir()
    global GROUP_MAP, SOURCE_VALID_COUNTS, REPEAT_PARTITIONS
    GROUP_MAP = pd.read_csv(POPULATION, usecols=["sample_id", *PROTOCOLS.values()])
    if GROUP_MAP.sample_id.duplicated().any() or len(GROUP_MAP) != 47015:
        raise RuntimeError("Population mapping is not a unique 47,015-row contract")
    for protocol in PROTOCOLS:
        source = pd.read_parquet(SPLIT_ROOT / f"{protocol}_split_manifest.parquet", columns=["repeat", "fold", "role"])
        SOURCE_VALID_COUNTS[protocol] = source.loc[source.role.eq("valid")].groupby(["repeat", "fold"]).size().sort_index()
        REPEAT_PARTITIONS[protocol] = partition_signatures(protocol)
    provenance, metric_rows, effects, paired_cache = [], [], [], {}
    for descriptor, model, _ in METHODS:
        for protocol, group_col in PROTOCOLS.items():
            predictions = {}
            for combo in COMBOS:
                prediction, proof = validate_task(descriptor, model, protocol, combo)
                predictions[combo] = prediction
                proof.update({"descriptor": descriptor, "model": model, "protocol": protocol, "combo": combo})
                provenance.append(proof)
                for repeat, part in prediction.groupby("repeat", sort=True):
                    for metric, value in metric_values(part).items():
                        metric_rows.append({"descriptor": descriptor, "model": model, "protocol": protocol, "combo": combo, "repeat": int(repeat), "metric": metric, "value": value})
            contrast, paired = paired_analysis(predictions, protocol, descriptor, model, group_col)
            effects.append(contrast); paired_cache[(descriptor, model, protocol)] = paired
    metrics = pd.DataFrame(metric_rows)
    repeat_partitions = pd.concat(REPEAT_PARTITIONS.values(), ignore_index=True)
    metrics_effective = metrics.merge(
        repeat_partitions[["protocol", "repeat", "is_representative"]],
        on=["protocol", "repeat"], validate="many_to_one",
    )
    metrics_effective = metrics_effective.loc[metrics_effective.is_representative].copy()
    summary = metrics_effective.groupby(["descriptor", "model", "protocol", "combo", "metric"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"), n_distinct_partitions=("value", "size"))
    summary["n_nominal_repeats"] = 3
    summary["ci_low"] = np.nan
    summary["ci_high"] = np.nan
    multi_partition = summary.n_distinct_partitions > 1
    degrees_of_freedom = summary.loc[multi_partition, "n_distinct_partitions"] - 1
    t_values = stats.t.ppf(.975, degrees_of_freedom)
    summary.loc[multi_partition, "ci_low"] = summary.loc[multi_partition, "mean"] - t_values * summary.loc[multi_partition, "sd"] / np.sqrt(summary.loc[multi_partition, "n_distinct_partitions"])
    summary.loc[multi_partition, "ci_high"] = summary.loc[multi_partition, "mean"] + t_values * summary.loc[multi_partition, "sd"] / np.sqrt(summary.loc[multi_partition, "n_distinct_partitions"])
    effects = pd.concat(effects, ignore_index=True)
    primary = effects.contrast.isin(["a_with_p", "a_without_p", "b_with_p", "b_without_p"])
    effects["p_holm_within_protocol_model"] = np.nan
    effects.loc[primary, "p_holm_within_protocol_model"] = effects.loc[primary].groupby(["descriptor", "model", "protocol"])["p_raw"].transform(holm)
    effects["practical_mae_threshold"] = .01
    effects["ci_excludes_zero"] = (effects.ci_low > 0) | (effects.ci_high < 0)
    effects["ci_exceeds_practical_threshold"] = effects.ci_low > .01
    pd.DataFrame(provenance).to_parquet(OUT / "tables/task_provenance.parquet", index=False)
    metrics.to_parquet(OUT / "tables/repeat_metrics.parquet", index=False)
    repeat_partitions.to_csv(OUT / "tables/repeat_partition_signatures.csv", index=False)
    summary.to_csv(OUT / "tables/metric_summary.csv", index=False)
    effects.to_csv(OUT / "tables/paired_effects.csv", index=False)
    for key, frame in paired_cache.items():
        frame.to_parquet(OUT / "tables" / ("paired_errors_" + "_".join(key) + ".parquet"), index=False)
    write_figures(summary, effects, figure_dir)
    primary_morgan = effects[(effects.descriptor == "morgan") & (effects.model == "rf")]
    evidence = []
    for _, row in primary_morgan[primary_morgan.contrast.isin(["a_substitution", "b_substitution"])].iterrows():
        state = "positive" if row.ci_low > .01 else "negative" if row.ci_high < 0 else "uncertain"
        evidence.append(f"- {row.protocol} / {row.contrast}: {row.effect_mae:.4f} [95% CI {row.ci_low:.4f}, {row.ci_high:.4f}] → {state}")
    partition_summary = "\n".join(
        f"- {protocol}: {int(part.is_representative.sum())} distinct partition(s) among {len(part)} nominal repeats"
        for protocol, part in REPEAT_PARTITIONS.items()
    )
    report = """# Ablation2 strict analysis\n\n## Question\n\nDoes known product structure reduce the conditional predictive contribution of amine A or acid B, and does the pattern persist under random, unseen-amine, unseen-acid, and unseen-substrate-pair evaluation?\n\n## Evidence inventory\n\nAll 64 pre-registered formal tasks passed artifact validation: 15 completed folds each, 960 prediction shards in total, both report formats, matching population and external split hashes, and zero declared group overlap between train and test in every fold. The unit for inference is the declared holdout group, not a fold or model.\n\n## Primary Morgan × RF substitution effects\n\n""" + "\n".join(evidence) + """\n\nPositive values mean the marginal MAE cost of removing the reactant is larger when P is absent. These are predictive-information comparisons; they do not establish chemical causality or mutual information.\n\n## Claim Candidates\n\n- Claim:\n  - Source evidence: `tables/paired_effects.csv`, `figure-02-substitution-effects.pdf`.\n  - Allowed wording: “Within this audited dataset and protocol, the observed substitution effect was [report estimate and CI].”\n  - Forbidden stronger wording: “The product chemically causes reactant information to be redundant” or “the result generalizes beyond this reaction family.”\n  - Uncertainty: three repeated outer partitions; group-cluster bootstrap; no independent source dataset yet.\n  - Next check: Step 2-10 independent-source decision and, if eligible, a separately preregistered replication.\n  - Decision: retain only effects whose CI supports the stated magnitude.\n\n## Limits\n\n- Random evaluation is dataset-internal only.\n- Pair holdout guarantees a new A+B combination, not that both individual substrates are unseen.\n- Constant DMF cannot support a general solvent-importance conclusion.\n- MFP × LightGBM is a pre-registered robustness line, not an independent chemical replicate.\n"""
    report = report.replace(
        "## Primary Morgan × RF substitution effects",
        "## Effective outer partitions\n\n" + partition_summary + "\n\nRepeated assignments are retained in `repeat_metrics.parquet` for transparency, but only one representative of each identical sample-to-fold assignment enters uncertainty intervals and tests. In particular, unseen-acid has one effective partition, so it supports a group-bootstrap interval but not an across-repeat error bar.\n\n## Primary Morgan × RF substitution effects",
    ).replace(
        "Uncertainty: three repeated outer partitions; group-cluster bootstrap; no independent source dataset yet.",
        "Uncertainty: use the effective-partition counts above; group-cluster bootstrap; no independent source dataset yet.",
    )
    appendix = "# Statistical appendix\n\n## Estimand and pairing\n\nFor every method/protocol/input combination, 3 nominal repeated out-of-fold prediction sets contain all 47,015 sample IDs once per repeat. Metrics are recomputed from those predictions. Each comparison joins models by `repeat, sample_id`; membership and labels must match exactly. The repeat assignment table is audited by hashing each valid sample-to-fold map. Identical maps count once as an effective outer partition for all inference.\n\nThe reported substitution estimand is `[L(B+C)-L(A+B+C)]-[L(B+P+C)-L(A+B+P+C)]` for A and its B analogue, where L is row-weighted OOF MAE.\n\n## Uncertainty and tests\n\n95% percentile intervals use 4,000 nested bootstrap draws: sample distinct outer partitions with replacement, then sample declared holdout groups with replacement inside each selected partition. The group key is `repeat_group_id`, `amine_group_id`, `acid_group_id`, or `substrate_pair_group_id` according to protocol. When only one distinct partition exists, the interval is a group bootstrap conditional on that split, not evidence from replicated split assignments.\n\nFor the four pre-registered marginal contribution contrasts, two-sided paired Wilcoxon signed-rank tests use one group-level mean error difference per group after averaging over distinct outer partitions. Holm adjustment is applied across these four contrasts within each method/protocol family. The derived substitution effects use their pre-registered bootstrap intervals; they are not promoted to a separate unplanned p-value family. A MAE effect of 0.01 yield fraction is the practical threshold.\n\n## Output tables\n\n- `repeat_partition_signatures.csv`: nominal repeats, assignment hashes, and representatives used for inference.\n- `metric_summary.csv`: distinct-partition mean, SD and t interval for MAE, RMSE, R² and Kendall tau; intervals are omitted when only one distinct partition exists.\n- `paired_effects.csv`: row-weighted effects, nested-bootstrap intervals, group-counts, Wilcoxon statistics and Holm-adjusted p-values.\n- `paired_errors_*.parquet`: source records for independently recomputing all paired effects.\n\nA p-value does not replace the confidence interval or the practical threshold; any conclusion must satisfy its stated interval and boundary.\n"
    catalog = "# Figure catalog\n\n## figure-01-mae-ablation-matrix.pdf\n\n- Purpose: compare all eight pre-registered inputs under each method/protocol.\n- Data: OOF MAE averaged over 3 repeats; whiskers are t-based 95% intervals across repeats.\n- Reader should notice: whether performance ordering changes between random and group holdouts.\n- Interpretation check: random performance is not evidence of unseen-substrate generalization.\n\n## figure-02-substitution-effects.pdf\n\n- Purpose: show the pre-registered A/B product-substitution effects.\n- Data: row-weighted MAE difference-of-differences; whiskers are nested repeat/group-bootstrap 95% intervals.\n- Reader should notice: whether each interval is above zero and the 0.01 practical threshold.\n- Interpretation check: an interval supports conditional predictive information only; it cannot demonstrate chemical causality.\n"
    catalog = catalog.replace(
        "Data: OOF MAE averaged over 3 repeats; whiskers are t-based 95% intervals across repeats.",
        "Data: OOF MAE averaged over distinct partitions; whiskers are t-based 95% intervals when at least two distinct partitions exist.",
    ).replace(
        "nested repeat/group-bootstrap 95% intervals",
        "nested distinct-partition/group-bootstrap 95% intervals",
    )
    for filename, content in [("analysis-report.md", report), ("stats-appendix.md", appendix), ("figure-catalog.md", catalog)]:
        (OUT / filename).write_text(content, encoding="utf-8")
    (OUT / "report").mkdir()
    (OUT / "report/analysis_report.html").write_text(to_html(report + "\n\n" + appendix + "\n\n" + catalog), encoding="utf-8")
    (OUT / "run_manifest.json").write_text(json.dumps({"task_name": "ablation2_statistics_v1", "formal_task_count": 64, "prediction_shard_count": 960, "population_sha256": sha256(POPULATION), "analysis_script_sha256": sha256(Path(__file__)), "bootstrap_draws": 4000, "nominal_repeats": 3, "distinct_partition_counts": {protocol: int(part.is_representative.sum()) for protocol, part in REPEAT_PARTITIONS.items()}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis complete: {OUT}")


if __name__ == "__main__":
    main()
