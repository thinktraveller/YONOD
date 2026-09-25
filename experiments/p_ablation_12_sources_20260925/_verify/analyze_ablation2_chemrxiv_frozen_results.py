"""Read-only statistical synthesis for the completed ChemRxiv frozen external matrix.

This script never trains a model or changes a frozen task directory.  It verifies
each of the 12 completed tasks, then analyses their saved external predictions
using the ten pre-registered external amine groups as paired cluster units.
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
from typing import Any, Dict, Iterable, List, Tuple

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


TASK_PREFIX = "ablation2_chemrxiv_frozen"
OUTPUT_NAME = "ablation2_chemrxiv_frozen_statistics_v1"
OUTPUT_ROOT = ROOT / "result" / OUTPUT_NAME
SUMMARY_DOCUMENT = ROOT / "project-docs" / "docs" / "ablation2" / "chemrxiv_frozen_statistics_v1.md"
BOOTSTRAP_DRAWS = 10_000
RANDOM_SEED = 20260922

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

PRIMARY_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("a_with_product", "ΔA|P = L(B+P+C) − L(A+B+P+C)"),
    ("a_without_product", "ΔA|¬P = L(B+C) − L(A+B+C)"),
)
SUBSTITUTION_CONTRAST = (
    "a_product_substitution",
    "Δsub,A = ΔA|¬P − ΔA|P",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_values_sha256(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_name(line: Dict[str, str], combo: str) -> str:
    return "{0}_{1}_{2}_v1".format(TASK_PREFIX, line["id"], combo)


def config_path(line: Dict[str, str], combo: str) -> Path:
    return ROOT / "config" / (task_name(line, combo) + ".yaml")


def prediction_path(line: Dict[str, str], combo: str) -> Path:
    task_root = ROOT / "result" / task_name(line, combo) / "docs" / "predictions"
    candidates = sorted(task_root.glob("*.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one prediction parquet for {0}; found {1}".format(
                task_name(line, combo), len(candidates)
            )
        )
    return candidates[0]


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm adjustment, returned in the source order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [float("nan")] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def rank_biserial(values: np.ndarray) -> float:
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> Tuple[float, float]:
    if len(values) != 10:
        raise RuntimeError("Cluster bootstrap requires exactly the pre-registered 10 amine groups")
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    estimates = sampled.mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def overall_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    tau = stats.kendalltau(y_true, y_pred).statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(tau) if tau is not None else float("nan"),
    }


def validate_and_read(line: Dict[str, str], combo: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config_path(line, combo)
    if not config.is_file():
        raise RuntimeError("Missing frozen configuration: {0}".format(config))
    integrity = verify(config)
    if integrity.get("status") != "passed":
        raise RuntimeError("Frozen task did not pass integrity verification: {0}".format(config))
    prediction = pd.read_parquet(prediction_path(line, combo))
    required = {
        "sample_id",
        "external_source_id",
        "external_group_id",
        "y_true",
        "y_pred",
        "evaluation_protocol",
        "partition",
    }
    missing = required.difference(prediction.columns)
    if missing:
        raise RuntimeError("Prediction is missing required columns {0}: {1}".format(sorted(missing), config))
    if len(prediction) != 957 or prediction["sample_id"].astype(str).nunique() != 957:
        raise RuntimeError("Prediction is not the frozen 957-row external population: {0}".format(config))
    if prediction["external_group_id"].astype(str).nunique() != 10:
        raise RuntimeError("Prediction does not have the pre-registered 10 amine groups: {0}".format(config))
    if not prediction["evaluation_protocol"].eq("frozen_train_external_test").all():
        raise RuntimeError("Unexpected evaluation protocol in {0}".format(config))
    if not prediction["partition"].eq("external_test").all():
        raise RuntimeError("Unexpected partition in {0}".format(config))
    for column in ("y_true", "y_pred"):
        if not np.isfinite(prediction[column].to_numpy(dtype=float)).all():
            raise RuntimeError("Non-finite {0} in {1}".format(column, config))
    prediction = prediction.loc[:, sorted(required.union({"run_id", "config_hash", "descriptor", "model"}).intersection(prediction.columns))].copy()
    prediction["sample_id"] = prediction["sample_id"].astype(str)
    prediction["external_source_id"] = prediction["external_source_id"].astype(str)
    prediction["external_group_id"] = prediction["external_group_id"].astype(str)
    provenance = {
        "task_name": task_name(line, combo),
        "line_id": line["id"],
        "line_label": line["label"],
        "combo": combo,
        "config_path": str(config.relative_to(ROOT)),
        "config_sha256": sha256_file(config),
        "prediction_path": str(prediction_path(line, combo).relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path(line, combo)),
        "sample_id_sha256": stable_values_sha256(prediction["sample_id"].tolist()),
        "external_group_id_sha256": stable_values_sha256(prediction["external_group_id"].tolist()),
        "integrity": integrity,
    }
    return prediction, provenance


def join_line_predictions(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    expected_combos = [item[0] for item in COMBOS]
    if sorted(predictions) != sorted(expected_combos):
        raise RuntimeError("A frozen input combination is missing from the analysis matrix")
    full = predictions["full"].loc[
        :, ["sample_id", "external_source_id", "external_group_id", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "pred_full"})
    result = full
    identity_columns = ["sample_id", "external_source_id", "external_group_id", "y_true"]
    for combo, _ in COMBOS:
        if combo == "full":
            continue
        right = predictions[combo].loc[:, identity_columns + ["y_pred"]].rename(
            columns={"y_pred": "pred_" + combo}
        )
        result = result.merge(right, on=identity_columns, how="inner", validate="one_to_one")
    if len(result) != 957:
        raise RuntimeError("Input predictions do not join one-to-one on the frozen external population")
    if result["external_group_id"].nunique() != 10:
        raise RuntimeError("Joined prediction table lost external amine groups")
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def compute_line_results(
    line: Dict[str, str],
    joined: pd.DataFrame,
    rng: np.random.Generator,
    draws: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict[str, Any]] = []
    group_rows: List[pd.DataFrame] = []
    for combo, input_blocks in COMBOS:
        prediction_column = "pred_" + combo
        abs_error = (joined[prediction_column] - joined["y_true"]).abs()
        row = {"line_id": line["id"], "line_label": line["label"], "combo": combo, "input_blocks": input_blocks}
        row.update(overall_metrics(joined.rename(columns={prediction_column: "y_pred"})))
        overall_rows.append(row)
        group = joined.assign(abs_error=abs_error).groupby("external_group_id", sort=True).agg(
            n_rows=("sample_id", "size"),
            mae=("abs_error", "mean"),
        ).reset_index()
        group.insert(0, "input_blocks", input_blocks)
        group.insert(0, "combo", combo)
        group.insert(0, "line_label", line["label"])
        group.insert(0, "line_id", line["id"])
        group_rows.append(group)

    errors = pd.DataFrame({"external_group_id": joined["external_group_id"]})
    for combo, _ in COMBOS:
        errors[combo] = (joined["pred_" + combo] - joined["y_true"]).abs()
    group_errors = errors.groupby("external_group_id", sort=True).mean()
    group_errors["a_with_product"] = group_errors["minus_a"] - group_errors["full"]
    group_errors["a_without_product"] = group_errors["minus_a_p"] - group_errors["minus_p"]
    group_errors["a_product_substitution"] = group_errors["a_without_product"] - group_errors["a_with_product"]
    group_contrasts = group_errors.loc[:, [item[0] for item in PRIMARY_CONTRASTS] + [SUBSTITUTION_CONTRAST[0]]].reset_index()
    group_contrasts.insert(0, "line_label", line["label"])
    group_contrasts.insert(0, "line_id", line["id"])

    contrast_rows: List[Dict[str, Any]] = []
    for contrast, label in PRIMARY_CONTRASTS + (SUBSTITUTION_CONTRAST,):
        group_values = group_errors[contrast].to_numpy(dtype=float)
        row_effect = float(
            (errors["minus_a"] - errors["full"]).mean()
            if contrast == "a_with_product"
            else (errors["minus_a_p"] - errors["minus_p"]).mean()
            if contrast == "a_without_product"
            else ((errors["minus_a_p"] - errors["minus_p"]) - (errors["minus_a"] - errors["full"])).mean()
        )
        ci_low, ci_high = bootstrap_interval(group_values, rng, draws)
        row: Dict[str, Any] = {
            "line_id": line["id"],
            "line_label": line["label"],
            "contrast": contrast,
            "label": label,
            "effect_group_mean": float(group_values.mean()),
            "effect_group_median": float(np.median(group_values)),
            "effect_row_weighted": row_effect,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_draws": draws,
            "bootstrap_seed": RANDOM_SEED,
            "n_amine_groups": int(len(group_values)),
            "practical_mae_threshold": 0.01,
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "ci_exceeds_practical_threshold": bool(ci_low > 0.01),
        }
        if contrast in {item[0] for item in PRIMARY_CONTRASTS}:
            nonzero = group_values[~np.isclose(group_values, 0.0)]
            test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto") if len(nonzero) else None
            row.update(
                {
                    "test_family": "two primary marginal A contrasts within line/external track",
                    "wilcoxon_statistic": float(test.statistic) if test is not None else float("nan"),
                    "p_raw": float(test.pvalue) if test is not None else float("nan"),
                    "n_nonzero_groups": int(len(nonzero)),
                    "rank_biserial": rank_biserial(group_values),
                }
            )
        else:
            row.update(
                {
                    "test_family": "bootstrap-only derived substitution estimand (pre-specified; no extra p-value family)",
                    "wilcoxon_statistic": float("nan"),
                    "p_raw": float("nan"),
                    "n_nonzero_groups": int((~np.isclose(group_values, 0.0)).sum()),
                    "rank_biserial": float("nan"),
                }
            )
        contrast_rows.append(row)
    return (
        pd.DataFrame(overall_rows),
        pd.concat(group_rows, ignore_index=True),
        group_contrasts,
        pd.DataFrame(contrast_rows),
    )


def conclusion(row: pd.Series) -> str:
    if row["contrast"] != "a_product_substitution":
        return "marginal A information comparison; interpret with its interval and Holm-adjusted test"
    if row["ci_low"] > 0.01:
        return "supports the pre-specified conditional predictive substitution pattern in this external collection"
    if row["ci_high"] < 0:
        return "direction is inconsistent with the pre-specified substitution pattern in this external collection"
    return "uncertain at the pre-specified practical MAE boundary"


def markdown_table(frame: pd.DataFrame, columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else "{0:.4f}".format(value))
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + body)


def write_figures(overall: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#999999"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, line in zip(axes, LINES):
        data = overall[overall["line_id"] == line["id"]].set_index("combo").loc[[item[0] for item in COMBOS]]
        axis.bar(np.arange(len(data)), data["mae"], color=colors)
        axis.set_xticks(np.arange(len(data)), [item[0] for item in COMBOS], rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("External MAE (yield fraction)")
        axis.set_title(line["label"])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Frozen external-test descriptive MAE; no CV interpretation")
    fig.tight_layout()
    fig.savefig(output / "figure-01-external-mae-by-input.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "figure-01-external-mae-by-input.pdf", bbox_inches="tight")
    plt.close(fig)

    plotted = contrasts[contrasts["contrast"] == "a_product_substitution"]
    fig, axis = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(plotted))
    values = plotted["effect_group_mean"].to_numpy(dtype=float)
    lower = values - plotted["ci_low"].to_numpy(dtype=float)
    upper = plotted["ci_high"].to_numpy(dtype=float) - values
    axis.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=4, color="#D55E00")
    axis.axvline(0, color="#444444", lw=0.8)
    axis.axvline(0.01, color="#444444", lw=0.8, ls="--")
    axis.set_yticks(y, plotted["line_label"].tolist())
    axis.set_xlabel("Group-mean Δsub,A (MAE); 95% cluster-bootstrap CI")
    axis.set_title("Product substitution of amine A (ten external amine groups)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "figure-02-a-product-substitution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "figure-02-a-product-substitution.pdf", bbox_inches="tight")
    plt.close(fig)


def build_report(overall: pd.DataFrame, contrasts: pd.DataFrame, provenance: List[Dict[str, Any]], draws: int) -> str:
    display_overall = overall.loc[:, ["line_label", "combo", "input_blocks", "mae", "rmse", "r2", "kendall_tau"]].copy()
    display_contrasts = contrasts.loc[
        :, ["line_label", "contrast", "effect_group_mean", "ci_low", "ci_high", "p_raw", "p_holm", "rank_biserial", "conclusion"]
    ].copy()
    return """# ChemRxiv 冻结外测统计综合 v1

## 输入与方法

- 已对 12/12 个冻结 development→external 任务再次运行逐项完整性核验；统计只读取其保存的 prediction parquet，未训练、覆盖或新增模型。
- 每项任务均为 47,015 行 development population 单次拟合后，对同一 957 行 ChemRxiv external population 的一次预测；它不是交叉验证，逐任务分数也不是模型重复。
- 推断单位是预注册的 10 个 `external_group_id` 胺结构。每个胺组先计算组内平均绝对误差差，再按组等权重汇总。95% 区间为固定随机种子、{draws} 次 percentile cluster bootstrap（重采样 10 个胺组）。
- 两个预注册的 A 边际对比使用双侧组级 Wilcoxon signed-rank，并在每一条模型线的 frozen-external track 内对这两个 p 值作 Holm 校正。派生的 Δsub,A 只报告预注册 bootstrap 区间，不额外扩增 p 值家族。

## 描述性外测指标

{overall_table}

## A 组分的配对统计

正值表示移除 A 会提高 MAE；Δsub,A 为“无 P 时移除 A 的代价”减去“有 P 时移除 A 的代价”。其为条件预测信息比较，不能解释为化学因果或互信息。

{contrast_table}

## 受限解释

- 只有当 Δsub,A 的 95% 区间下界超过预注册的 0.01 MAE 阈值，才可在该外部集合中称为支持“P 减少 A 的条件预测贡献”的模式；所有其他情形均保留为不确定或方向相反。
- Morgan ECFP4 × RF 和 MFP × LightGBM 是预注册的两条模型线，用于稳健性检查，而不是相互独立的化学重复。
- 外部来源只有一种酸；结果不能声称未见酸泛化、跨反应家族普适性、因果替代或模型等效。非显著结果也不能证明没有效应。
- 这份综合不启动同源 grouped unseen-A 验证；该后续研究需单独决策，且不能与本冻结外测混为一谈。

## 可复算产物

- `tables/task_provenance.json`：12 项输入配置、预测哈希及完整性核验摘要。
- `tables/overall_external_metrics.csv`：从保存预测重算的 957 行描述性指标。
- `tables/group_mae.csv` 与 `tables/group_contrasts.csv`：十个胺组层面的配对源数据。
- `tables/statistical_summary.csv`：bootstrap、Wilcoxon 与 Holm 输出。
- `figures/`：外测 MAE 与 Δsub,A 图（PNG/PDF）。
""".format(
        draws=draws,
        overall_table=markdown_table(display_overall, list(display_overall.columns)),
        contrast_table=markdown_table(display_contrasts, list(display_contrasts.columns)),
    )


def write_text_outputs(
    output: Path,
    overall: pd.DataFrame,
    group_metrics: pd.DataFrame,
    group_contrasts: pd.DataFrame,
    contrasts: pd.DataFrame,
    provenance: List[Dict[str, Any]],
    draws: int,
) -> str:
    tables = output / "tables"
    figures = output / "figures"
    report_dir = output / "report"
    tables.mkdir(parents=True)
    figures.mkdir()
    report_dir.mkdir()
    report = build_report(overall, contrasts, provenance, draws)
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    (report_dir / "analysis_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>ChemRxiv frozen external statistics</title>"
        "<style>body{font-family:system-ui;max-width:1120px;margin:2rem auto;line-height:1.45}pre{white-space:pre-wrap}</style>"
        "<pre>" + html.escape(report) + "</pre>",
        encoding="utf-8",
    )
    appendix = """# Statistical appendix

The analysis joins every input condition by `sample_id`, `external_source_id`, `external_group_id`, and `y_true`; a mismatch blocks output. The source prediction files are first checked by the existing frozen-external integrity verifier, including population hashes, train/external ID disjointness, single-fit/CV=0 identity, label isolation, finite predictions, recomputed task metrics, and both report formats.

For each amine group g, each marginal contrast is its within-group mean absolute-error difference. The estimand is the equally weighted mean of the ten group values. Bootstrap draws sample those ten groups with replacement; the resulting percentile interval is conditional on this particular external collection and its sole acid. Wilcoxon tests are two-sided on the nonzero group values. Holm is limited to the two pre-registered marginal A contrasts per model line; the product-substitution contrast remains bootstrap-only as pre-specified.
"""
    (output / "statistical-appendix.md").write_text(appendix, encoding="utf-8")
    overall.to_csv(tables / "overall_external_metrics.csv", index=False)
    group_metrics.to_csv(tables / "group_mae.csv", index=False)
    group_contrasts.to_csv(tables / "group_contrasts.csv", index=False)
    contrasts.to_csv(tables / "statistical_summary.csv", index=False)
    (tables / "task_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_figures(overall, contrasts, figures)
    manifest = {
        "analysis_task": OUTPUT_NAME,
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "formal_task_count": len(provenance),
        "external_row_count": 957,
        "external_amine_group_count": 10,
        "bootstrap_draws": draws,
        "bootstrap_seed": RANDOM_SEED,
        "unit_of_inference": "external_group_id (external amine structure)",
        "models": [line["label"] for line in LINES],
        "source_data_policy": "saved prediction parquet only; no model fitting or frozen-result modification",
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.bootstrap_draws < 1_000:
        raise ValueError("At least 1,000 bootstrap draws are required")
    if args.seed != RANDOM_SEED:
        raise ValueError("The frozen analysis requires seed {0}".format(RANDOM_SEED))
    if OUTPUT_ROOT.exists() or SUMMARY_DOCUMENT.exists():
        raise RuntimeError("Refusing to overwrite an existing frozen-statistics output or summary document")

    rng = np.random.default_rng(args.seed)
    all_provenance: List[Dict[str, Any]] = []
    overall_frames: List[pd.DataFrame] = []
    group_metric_frames: List[pd.DataFrame] = []
    group_contrast_frames: List[pd.DataFrame] = []
    contrast_frames: List[pd.DataFrame] = []
    for line in LINES:
        predictions: Dict[str, pd.DataFrame] = {}
        for combo, _ in COMBOS:
            prediction, provenance = validate_and_read(line, combo)
            predictions[combo] = prediction
            all_provenance.append(provenance)
        joined = join_line_predictions(predictions)
        overall, group_metrics, group_contrasts, contrasts = compute_line_results(
            line, joined, rng, args.bootstrap_draws
        )
        overall_frames.append(overall)
        group_metric_frames.append(group_metrics)
        group_contrast_frames.append(group_contrasts)
        contrast_frames.append(contrasts)

    overall = pd.concat(overall_frames, ignore_index=True)
    group_metrics = pd.concat(group_metric_frames, ignore_index=True)
    group_contrasts = pd.concat(group_contrast_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    contrasts["p_holm"] = np.nan
    for line_id in contrasts["line_id"].unique():
        mask = (contrasts["line_id"] == line_id) & contrasts["contrast"].isin([item[0] for item in PRIMARY_CONTRASTS])
        raw = contrasts.loc[mask, "p_raw"].tolist()
        contrasts.loc[mask, "p_holm"] = holm_adjust([float(value) for value in raw])
    contrasts["conclusion"] = contrasts.apply(conclusion, axis=1)

    temporary = Path(tempfile.mkdtemp(prefix="." + OUTPUT_NAME + ".", dir=str(OUTPUT_ROOT.parent)))
    try:
        report = write_text_outputs(
            temporary,
            overall,
            group_metrics,
            group_contrasts,
            contrasts,
            all_provenance,
            args.bootstrap_draws,
        )
        os.replace(temporary, OUTPUT_ROOT)
        temporary = None
        SUMMARY_DOCUMENT.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_DOCUMENT.write_text(report, encoding="utf-8")
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    result = contrasts.loc[:, ["line_label", "contrast", "effect_group_mean", "ci_low", "ci_high", "p_holm", "conclusion"]]
    print(result.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
