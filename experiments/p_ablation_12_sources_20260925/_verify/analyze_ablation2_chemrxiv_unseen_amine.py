"""Strict group-aware analysis of the ChemRxiv same-source unseen-A matrix.

This analysis reads the immutable 12 formal task outputs, reruns the strict
read-only verifier for each task, and writes a separate evidence bundle below
``project-docs/docs/ablation2``.  It never fits a model.  The inferential unit is one of
the ten unique amine structures: each group's loss is first averaged over its
three held-out repeats, so rows, folds, and repeats are never treated as
independent chemical replicates.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_ablation2_chemrxiv_unseen_amine_result import verify


OUTPUT_NAME = "chemrxiv_unseen_amine_analysis_v1"
OUTPUT_ROOT = ROOT / "project-docs" / "docs" / "ablation2" / OUTPUT_NAME
TASK_PREFIX = "ablation2_chemrxiv_unseen_amine"
BOOTSTRAP_DRAWS = 10_000
RANDOM_SEED = 20260922

LINES = (
    {"id": "morgan_rf", "descriptor": "morgan", "model": "rf", "label": "Morgan ECFP4 × RF"},
    {"id": "mfp_lightgbm", "descriptor": "mfp", "model": "lightgbm", "label": "MFP × LightGBM"},
)
COMBOS = (
    ("full", "A+B+P+C"),
    ("minus_a", "B+P+C"),
    ("minus_p", "A+B+C"),
    ("minus_a_p", "B+C"),
    ("product_conditions", "P+C"),
    ("conditions_only", "C only"),
)
CONTRASTS = (
    ("a_with_product", "ΔA|P", "MAE(B+P+C) − MAE(A+B+P+C)", "minus_a", "full", True),
    ("a_without_product", "ΔA|¬P", "MAE(B+C) − MAE(A+B+C)", "minus_a_p", "minus_p", True),
)
SUBSTITUTION_ID = "a_product_substitution"
SUBSTITUTION_LABEL = "Δsub,A"
SUBSTITUTION_FORMULA = "ΔA|¬P − ΔA|P"

OKABE_ITO = {"blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00", "black": "#000000", "grey": "#777777"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        json.dumps([str(value) for value in values], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def task_name(line: dict[str, str], combo: str) -> str:
    return f"{TASK_PREFIX}_{line['id']}_{combo}_v1"


def config_path(line: dict[str, str], combo: str) -> Path:
    return ROOT / "config" / f"{task_name(line, combo)}.yaml"


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    tau = stats.kendalltau(y_true, y_pred).statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(tau) if tau is not None else float("nan"),
    }


def holm_adjust(values: list[float]) -> list[float]:
    """Holm-adjust two-sided p-values and preserve source order."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def bootstrap_ci(differences: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile CI from a 10-amine cluster bootstrap, not rows/folds."""
    if len(differences) != 10:
        raise RuntimeError("The pre-registered cluster bootstrap requires exactly ten amine groups")
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def wilcoxon_summary(differences: np.ndarray) -> dict[str, Any]:
    """Two-sided paired Wilcoxon; retain its small-n/zero caveat explicitly."""
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return {"wilcoxon_statistic": 0.0, "p_wilcoxon_raw": 1.0, "wilcoxon_method": "all_zero", "n_nonzero": 0}
    try:
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="exact")
        method = "exact"
    except ValueError:
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="approx")
        method = "normal_approximation_due_to_zeros_or_ties"
    return {
        "wilcoxon_statistic": float(result.statistic),
        "p_wilcoxon_raw": float(result.pvalue),
        "wilcoxon_method": method,
        "n_nonzero": int(len(nonzero)),
    }


def read_task(line: dict[str, str], combo: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config_path(line, combo)
    integrity = verify(config)
    if integrity.get("status") != "passed":
        raise RuntimeError(f"Strict verifier did not pass: {config}")
    task_root = ROOT / integrity["result_root"]
    paths = sorted((task_root / "docs" / "predictions").glob("*.parquet"))
    if len(paths) != 15:
        raise RuntimeError(f"{task_root.name}: strict verifier passed without exactly 15 prediction shards")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {"sample_id", "group_id", "repeat", "fold", "y_true", "y_pred", "run_id", "config_hash"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{task_root.name}: prediction columns missing {sorted(missing)}")
    frame = frame.loc[:, sorted(required.union({"descriptor", "model"}).intersection(frame.columns))].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["repeat"] = frame["repeat"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    if len(frame) != 2871 or frame.duplicated(["sample_id", "repeat"]).any():
        raise RuntimeError(f"{task_root.name}: expected exactly 3×957 OOF sample/repeat predictions")
    if frame["group_id"].nunique() != 10 or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{task_root.name}: group or finite prediction contract failed")
    provenance = {
        "task": task_root.name,
        "line_id": line["id"],
        "combo": combo,
        "config": str(config.relative_to(ROOT)),
        "prediction_paths": [str(path.relative_to(ROOT)) for path in paths],
        "prediction_shas": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "integrity": integrity,
    }
    return frame, provenance


def join_line_predictions(frames: dict[str, pd.DataFrame], line: dict[str, str]) -> pd.DataFrame:
    """Pair inputs only on the same sample and repeat, with fold identity checked."""
    required_combos = [combo for combo, _ in COMBOS]
    if sorted(frames) != sorted(required_combos):
        raise RuntimeError(f"{line['id']}: incomplete six-input matrix")
    identity = ["sample_id", "repeat", "fold", "group_id", "y_true"]
    first = frames["full"].loc[:, identity + ["y_pred"]].rename(columns={"y_pred": "pred_full"})
    joined = first
    for combo in required_combos[1:]:
        right = frames[combo].loc[:, identity + ["y_pred"]].rename(columns={"y_pred": f"pred_{combo}"})
        joined = joined.merge(right, on=identity, how="inner", validate="one_to_one")
    if len(joined) != 2871 or joined.duplicated(["sample_id", "repeat"]).any():
        raise RuntimeError(f"{line['id']}: input predictions are not one-to-one paired by sample/repeat")
    for repeat in (1, 2, 3):
        subset = joined.loc[joined["repeat"].eq(repeat)]
        if len(subset) != 957 or subset["sample_id"].nunique() != 957:
            raise RuntimeError(f"{line['id']}: repeat {repeat} does not cover 957 paired OOF samples")
    return joined.sort_values(["repeat", "sample_id"], kind="mergesort").reset_index(drop=True)


def compute_tables(line: dict[str, str], joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-repeat metrics, coverage, group/repeat MAE, group means, contrasts."""
    per_repeat_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    group_repeat_parts: list[pd.DataFrame] = []
    group_mean_parts: list[pd.DataFrame] = []
    for combo, input_blocks in COMBOS:
        prediction_column = f"pred_{combo}"
        for repeat, part in joined.groupby("repeat", sort=True):
            metrics = metric_values(part.rename(columns={prediction_column: "y_pred"})[["y_true", "y_pred"]])
            per_repeat_rows.append({
                "line_id": line["id"], "line_label": line["label"], "combo": combo,
                "input_blocks": input_blocks, "repeat": int(repeat), "n_rows": int(len(part)), **metrics,
            })
            coverage_rows.append({
                "line_id": line["id"], "combo": combo, "repeat": int(repeat), "n_rows": int(len(part)),
                "n_unique_sample_ids": int(part["sample_id"].nunique()),
                "sample_id_sha256": stable_sha256(part.sort_values("sample_id")["sample_id"]),
                "sample_fold_sha256": stable_sha256(
                    part.loc[:, ["sample_id", "fold"]].sort_values("sample_id").astype(str).agg(":".join, axis=1)
                ),
            })
        working = joined.loc[:, ["sample_id", "group_id", "repeat", "fold", "y_true", prediction_column]].copy()
        working["abs_error"] = (working[prediction_column] - working["y_true"]).abs()
        groups = working.groupby(["group_id", "repeat", "fold"], sort=True).agg(
            n_rows=("sample_id", "size"), mae=("abs_error", "mean")
        ).reset_index()
        if len(groups) != 30 or groups.groupby(["group_id", "repeat"]).size().ne(1).any():
            raise RuntimeError(f"{line['id']}/{combo}: group/repeat/fold aggregation is incomplete")
        groups.insert(0, "input_blocks", input_blocks)
        groups.insert(0, "combo", combo)
        groups.insert(0, "line_label", line["label"])
        groups.insert(0, "line_id", line["id"])
        group_repeat_parts.append(groups)
        means = groups.groupby("group_id", sort=True).agg(
            n_rows_per_repeat=("n_rows", "first"), n_repeats=("repeat", "nunique"), mae_group_mean=("mae", "mean"),
            mae_group_sd_over_repeats=("mae", "std"),
        ).reset_index()
        if not means["n_repeats"].eq(3).all():
            raise RuntimeError(f"{line['id']}/{combo}: not every amine was averaged across all three repeats")
        means.insert(0, "input_blocks", input_blocks)
        means.insert(0, "combo", combo)
        means.insert(0, "line_label", line["label"])
        means.insert(0, "line_id", line["id"])
        group_mean_parts.append(means)

    group_repeat = pd.concat(group_repeat_parts, ignore_index=True)
    group_mean = pd.concat(group_mean_parts, ignore_index=True)
    pivot = group_mean.pivot(index="group_id", columns="combo", values="mae_group_mean").sort_index()
    if pivot.shape != (10, 6):
        raise RuntimeError(f"{line['id']}: expected a complete 10 amine × 6 input group-mean matrix")
    contrast_rows: list[dict[str, Any]] = []
    two_primary_p: list[float] = []
    primary_positions: list[int] = []
    for contrast_id, short_label, formula, numerator, denominator, primary in CONTRASTS:
        differences = (pivot[numerator] - pivot[denominator]).to_numpy(dtype=float)
        stats_row = _contrast_row(line, contrast_id, short_label, formula, differences, primary)
        contrast_rows.append(stats_row)
        if primary:
            two_primary_p.append(stats_row["p_wilcoxon_raw"])
            primary_positions.append(len(contrast_rows) - 1)
    substitution = (pivot["minus_a_p"] - pivot["minus_p"]) - (pivot["minus_a"] - pivot["full"])
    contrast_rows.append(_contrast_row(
        line, SUBSTITUTION_ID, SUBSTITUTION_LABEL, SUBSTITUTION_FORMULA,
        substitution.to_numpy(dtype=float), False,
    ))
    adjusted = holm_adjust(two_primary_p)
    for index, value in zip(primary_positions, adjusted):
        contrast_rows[index]["p_wilcoxon_holm_within_line_two_primary_A_tests"] = value
    contrasts = pd.DataFrame(contrast_rows)
    return (
        pd.DataFrame(per_repeat_rows), pd.DataFrame(coverage_rows), group_repeat,
        group_mean, contrasts,
    )


def _contrast_row(
    line: dict[str, str], contrast_id: str, short_label: str, formula: str, differences: np.ndarray, primary: bool
) -> dict[str, Any]:
    low, high = bootstrap_ci(differences, RANDOM_SEED + (17 if line["id"] == "mfp_lightgbm" else 0) + len(contrast_id))
    wilcoxon = wilcoxon_summary(differences)
    shapiro = stats.shapiro(differences)
    return {
        "line_id": line["id"], "line_label": line["label"], "contrast_id": contrast_id,
        "contrast_label": short_label, "formula": formula, "preregistered_primary_A_marginal": primary,
        "n_amine_groups": 10, "mean_delta_mae": float(differences.mean()), "sd_delta_mae": float(differences.std(ddof=1)),
        "median_delta_mae": float(np.median(differences)),
        "iqr_delta_mae": float(np.quantile(differences, 0.75) - np.quantile(differences, 0.25)),
        "ci95_group_bootstrap_low": low, "ci95_group_bootstrap_high": high,
        "n_positive": int((differences > 0).sum()), "n_negative": int((differences < 0).sum()), "n_zero": int(np.isclose(differences, 0).sum()),
        "rank_biserial": rank_biserial(differences), "shapiro_w": float(shapiro.statistic), "shapiro_p": float(shapiro.pvalue),
        "p_wilcoxon_holm_within_line_two_primary_A_tests": float("nan"), **wilcoxon,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}" if isinstance(value, (float, np.floating, int, np.integer)) else str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    printable = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: _fmt(value, digits))
    headers = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in printable.itertuples(index=False, name=None))
    return "\n".join(lines)


def make_figures(group_mean: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> list[dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    figure_specs: list[dict[str, str]] = []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    contrast_order = [item[0] for item in CONTRASTS] + [SUBSTITUTION_ID]
    labels = [item[1] for item in CONTRASTS] + [SUBSTITUTION_LABEL]
    for axis, line in zip(axes, LINES):
        values = group_mean.loc[group_mean["line_id"].eq(line["id"])].pivot(
            index="group_id", columns="combo", values="mae_group_mean"
        ).sort_index()
        deltas = [
            values["minus_a"] - values["full"], values["minus_a_p"] - values["minus_p"],
            (values["minus_a_p"] - values["minus_p"]) - (values["minus_a"] - values["full"]),
        ]
        for position, difference in enumerate(deltas):
            axis.plot([position] * len(difference), difference, "o", color=OKABE_ITO["grey"], alpha=0.65, ms=4, zorder=2)
            row = contrasts.loc[(contrasts["line_id"].eq(line["id"])) & (contrasts["contrast_id"].eq(contrast_order[position]))].iloc[0]
            mean = float(row["mean_delta_mae"])
            low, high = float(row["ci95_group_bootstrap_low"]), float(row["ci95_group_bootstrap_high"])
            axis.errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt="D", color=OKABE_ITO["blue"],
                          capsize=4, ms=6, zorder=4, label="group mean ± 95% CI" if position == 0 else None)
        axis.axhline(0, color=OKABE_ITO["black"], lw=0.9, ls="--")
        axis.set_title(line["label"], fontsize=11)
        axis.set_xticks(range(3), labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Group-mean ΔMAE\n(positive = removing A increases loss)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    pdf = output / "figure-01-a-marginal-contrasts.pdf"
    png = output / "figure-01-a-marginal-contrasts.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    figure_specs.append({"file": f"figures/{pdf.name}", "purpose": "Pre-registered A marginal contrasts and derived product-substitution contrast at the 10-amine-group unit."})

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharex=True)
    for axis, line in zip(axes, LINES):
        values = group_mean.loc[group_mean["line_id"].eq(line["id"])].pivot(
            index="group_id", columns="combo", values="mae_group_mean"
        ).sort_index()
        substitution = ((values["minus_a_p"] - values["minus_p"]) - (values["minus_a"] - values["full"])).sort_values()
        y = np.arange(len(substitution))
        axis.hlines(y, 0, substitution.to_numpy(), color=OKABE_ITO["grey"], lw=1.2)
        axis.plot(substitution.to_numpy(), y, "o", color=OKABE_ITO["orange"], ms=5)
        axis.axvline(0, color=OKABE_ITO["black"], lw=0.9, ls="--")
        axis.set_yticks(y, [f"amine {index + 1}" for index in y])
        axis.set_title(line["label"], fontsize=11)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Amine group (sorted within model line)")
    for axis in axes:
        axis.set_xlabel("Δsub,A = ΔA|¬P − ΔA|P (group mean)")
    fig.tight_layout()
    pdf = output / "figure-02-group-product-substitution.pdf"
    png = output / "figure-02-group-product-substitution.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    figure_specs.append({"file": f"figures/{pdf.name}", "purpose": "Heterogeneity of the derived substitution contrast across the ten held-out amine groups."})
    return figure_specs


def render_analysis_report(contrasts: pd.DataFrame, coverage: pd.DataFrame) -> str:
    primary = contrasts.loc[contrasts["contrast_id"].isin([item[0] for item in CONTRASTS])].copy()
    display = contrasts.loc[:, [
        "line_label", "contrast_label", "mean_delta_mae", "ci95_group_bootstrap_low", "ci95_group_bootstrap_high",
        "p_wilcoxon_raw", "p_wilcoxon_holm_within_line_two_primary_A_tests", "rank_biserial", "n_positive", "n_negative",
    ]]
    lines = [
        "# ChemRxiv 同源 grouped unseen-A 严格分析（v1）",
        "",
        "## Analysis question",
        "",
        "在固定的同源 ChemRxiv 957 行总体中，使用从训练中完全留出的 A/胺结构组，检验删去 A 后的条件预测损失是否在有 P 与无 P 情景不同。两个预注册模型线（Morgan ECFP4 × RF、MFP × LightGBM）各有 6 个输入组合、5 folds × 3 repeats。",
        "",
        "## Integrity and comparison unit",
        "",
        "12/12 任务通过严格只读验证：每项 SQLite 均为 15 succeeded / 0 failed；每项有 3×957 个 OOF 行，15 个 fold 的 A 组训练/验证交集均为零，配置—数据—source split—task-local split 身份及 HTML/Markdown 报告均一致。输入组合只按相同 `sample_id × repeat` 配对。",
        "",
        "推断单位是 **10 个独特 amine groups**。对每组，先计算每个 repeat 的组内 MAE（该组恰在一个 fold 留出），再对 3 个 repeat 平均；因此 957 行、15 folds 和 3 repeats 都不被作为独立化学重复。误差区间是 10 组的非参数 cluster bootstrap（10,000 draws）；Wilcoxon 是双侧配对检验。每模型线的两个预注册 A 边际比较做 Holm 校正；Δsub,A 只作为由它们构造的估计，不作为新增显著性筛选。",
        "",
        "## Pre-registered A contrasts",
        "",
        _markdown_table(display, list(display.columns)),
        "",
        "正的 ΔMAE 表示删除 A 使该组的预测 MAE 上升。表中的方向和区间是模型线内、来源内 A 外推的描述；两条模型线使用相同总体，并非独立化学复制。",
        "",
        "## Coverage",
        "",
        "每条模型线、每个输入、每个 repeat 均保留 957 个独特 OOF sample IDs；`tables/oof_coverage.csv` 同时保存 sample 和 sample→fold 指纹。",
        "",
        "## Claim candidates",
        "",
        "- Claim: 本来源内的 held-out A 组可用于比较 A 删除前后的条件预测损失。",
        "  - Source evidence: 12 项完整性验证、共享 source split 和保存的 OOF 预测。",
        "  - Allowed wording: “在该 ChemRxiv 来源内的未见 A 组留出下，组级 ΔMAE 为……。”",
        "  - Forbidden stronger wording: “这证明 A/P 的因果或互信息关系，或对未见酸/独立来源普适。”",
        "  - Uncertainty: 只有 10 个 A 结构组、单酸总体、重复划分相关。",
        "  - Next check: 在角色多样性更高的独立来源上复制。",
        "  - Decision: keep with boundary.",
        "",
        "- Claim: Δsub,A 描述有 P 与无 P 情景下 A 删除损失的差异。",
        "  - Source evidence: 两个预注册 A 边际差的组级配对构造及 bootstrap 区间。",
        "  - Allowed wording: “Δsub,A 的方向/区间在该来源内为……，并在两条预注册模型线上作稳健性检查。”",
        "  - Forbidden stronger wording: “P 造成 A 信息冗余”或“分子成分贡献具有因果效应”。",
        "  - Uncertainty: 派生量、10 组小样本以及模型线非独立。",
        "  - Next check: 预先定义的外部/多酸复制。",
        "  - Decision: retain only as bounded predictive evidence.",
        "",
        "## Boundary against frozen development→external tests",
        "",
        "本 bundle 与之前的 frozen development→ChemRxiv external 12 项测试不可直接比较或合并：前者是在同一个 ChemRxiv 来源内、对 10 个 A 组做 5×3 grouped OOF；后者是 development 的 47,015 行单次训练后对 ChemRxiv 957 行的冻结外测。两者训练总体、评估机制、可用信息与不确定性单位都不同。这里不提供跨轨道的胜负、p 值或效应量比较。",
        "",
        "## Files",
        "",
        "- `stats-appendix.md`: 可复算方法、检验与限制。",
        "- `tables/`: 逐任务完整性、逐 repeat 指标、coverage、逐组 MAE 与主对比。",
        "- `figures/`: 两张真实矢量图和 600-DPI PNG 对应物。",
        "- `run_manifest.json`: 输入及代码哈希、环境与完整性记录。",
    ]
    return "\n".join(lines) + "\n"


def render_stats_appendix(per_repeat: pd.DataFrame, group_repeat: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    per_repeat_summary = per_repeat.groupby(["line_label", "combo", "input_blocks"], sort=True).agg(
        n_repeats=("repeat", "nunique"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"), tau_mean=("kendall_tau", "mean"),
    ).reset_index()
    display = contrasts.loc[:, [
        "line_label", "contrast_label", "formula", "n_amine_groups", "mean_delta_mae", "sd_delta_mae",
        "median_delta_mae", "iqr_delta_mae", "ci95_group_bootstrap_low", "ci95_group_bootstrap_high",
        "wilcoxon_statistic", "p_wilcoxon_raw", "p_wilcoxon_holm_within_line_two_primary_A_tests",
        "rank_biserial", "shapiro_w", "shapiro_p", "wilcoxon_method", "n_positive", "n_negative", "n_zero",
    ]]
    return "\n".join([
        "# Statistical appendix — ChemRxiv same-source grouped unseen-A (v1)",
        "",
        "## Design and valid unit",
        "",
        "The fixed population has 957 rows, 10 amine structure groups and one acid structure. Every task uses the identical 5-fold × 3-repeat precomputed amine-group manifest. In each repeat an amine group is held out exactly once. The valid inferential unit is therefore the unique amine group (n=10), after within-group averaging of its three held-out-repeat MAEs. Fold-level, repeat-level, and row-level entries are retained for audit and descriptive reporting only.",
        "",
        "## Metrics",
        "",
        "Lower MAE/RMSE is better; higher R²/Kendall τ is better. Per-repeat scores below are calculated from the 957 OOF rows in that repeat. They are not used as 3 independent samples for a significance test.",
        "",
        _markdown_table(per_repeat_summary, list(per_repeat_summary.columns)),
        "",
        "## Main paired statistics",
        "",
        "For ΔA|P and ΔA|¬P, a positive value means that removing A increased group-level MAE. The 95% interval is a percentile bootstrap that resamples the ten group means with replacement (10,000 draws; seed 20260922 plus deterministic contrast offset). Wilcoxon is two-sided paired Wilcoxon on those ten group means. Holm correction is only across the two pre-registered A marginal tests within each model line. Δsub,A is a planned derived estimate, but its raw Wilcoxon output is descriptive and is not an added significance-selection rule.",
        "",
        _markdown_table(display, list(display.columns)),
        "",
        "## Assumptions and limitations",
        "",
        "- Independence is not assumed for rows, folds, repeats, or the two model lines. Pairing is maintained by `sample_id × repeat`; grouped means are the only inferential observations.",
        "- Shapiro–Wilk values are recorded for transparency, but n=10 gives low power. The analysis uses the pre-specified non-parametric paired Wilcoxon rather than selecting a parametric test after inspection.",
        "- Bootstrap intervals quantify variation under resampling these ten observed groups; they do not establish sampling from a broader chemistry population.",
        "- The sole acid structure and a single source preclude claims for unseen acids, independent sources, causal effects, component mutual information, or model equivalence.",
        "- The two model lines are robustness checks on the same data, not independent replications. Their agreement/disagreement must not be converted into another p-value.",
        "",
        "## Audit table locations",
        "",
        "`tables/per_repeat_metrics.csv` has all 36 descriptive repeat rows; `tables/group_mae_by_repeat.csv` has the 360 group×repeat×input rows; `tables/group_mean_mae.csv` shows the 120 group means used as the base for pairing; `tables/a_contrasts.csv` contains the exact numeric rows above.",
    ]) + "\n"


def render_figure_catalog(specs: list[dict[str, str]]) -> str:
    return "\n".join([
        "# Figure catalog — ChemRxiv same-source grouped unseen-A (v1)",
        "",
        "## Figure 1 — A marginal contrasts",
        "",
        f"- File: `{specs[0]['file']}` (matching 600-DPI PNG is also present).",
        "- Purpose: display the two pre-registered A marginal contrasts and their derived Δsub,A for both model lines at the correct 10-group unit.",
        "- Data: group MAE first averaged across all three repeats; grey circles are individual amine groups; blue diamond/error bar is the unweighted group mean and 95% group bootstrap CI.",
        "- Caption requirements: define positive ΔMAE, n=10 groups, bootstrap procedure, and that the two model lines are not independent chemistry replications.",
        "- Observation checklist: inspect sign consistency, interval location relative to zero, and between-group spread; do not count the dots as independent reaction rows.",
        "- Interpretation/decision: only changes a bounded source-internal predictive claim; it cannot establish independent-source, unseen-acid, or causal evidence.",
        "",
        "## Figure 2 — Product-substitution heterogeneity",
        "",
        f"- File: `{specs[1]['file']}` (matching 600-DPI PNG is also present).",
        "- Purpose: show whether the derived Δsub,A is concentrated or heterogeneous across held-out amine groups.",
        "- Data: ten group means after three-repeat averaging, sorted within each model line; the vertical dashed line is zero.",
        "- Caption requirements: give the Δsub,A formula and disclose sorting within panels; group labels are anonymous ordinal labels, not chemical identities.",
        "- Observation checklist: distinguish a shared direction from a uniform group effect; use the table for exact group identities and values.",
        "- Interpretation/decision: motivates cautious replication design; does not identify a molecular mechanism or a causal interaction.",
    ]) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def write_bundle(
    integrity_rows: list[dict[str, Any]], per_repeat: pd.DataFrame, coverage: pd.DataFrame,
    group_repeat: pd.DataFrame, group_mean: pd.DataFrame, contrasts: pd.DataFrame,
    provenance: list[dict[str, Any]], script_path: Path,
) -> Path:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis bundle: {OUTPUT_ROOT}")
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.", dir=OUTPUT_ROOT.parent))
    try:
        tables = staging / "tables"
        tables.mkdir()
        integrity_table = pd.DataFrame([{
            "task": row["task"], "status": row["status"], "succeeded": row["task_state"]["succeeded"],
            "failed": row["task_state"]["failed"], "oof_rows": row["oof"]["n_rows"],
            "n_amine_groups": row["oof"]["n_amine_groups"], "config_contract_sha256": row["hashes"]["config_contract_sha256"],
            "dataset_sha256": row["hashes"]["dataset_sha256"], "source_split_manifest_sha256": row["hashes"]["source_split_manifest_sha256"],
        } for row in integrity_rows])
        integrity_table.to_csv(tables / "task_integrity.csv", index=False)
        write_text(tables / "task_integrity.json", json.dumps(integrity_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        per_repeat.to_csv(tables / "per_repeat_metrics.csv", index=False)
        coverage.to_csv(tables / "oof_coverage.csv", index=False)
        group_repeat.to_csv(tables / "group_mae_by_repeat.csv", index=False)
        group_mean.to_csv(tables / "group_mean_mae.csv", index=False)
        contrasts.to_csv(tables / "a_contrasts.csv", index=False)
        specs = make_figures(group_mean, contrasts, staging / "figures")
        analysis = render_analysis_report(contrasts, coverage)
        appendix = render_stats_appendix(per_repeat, group_repeat, contrasts)
        catalog = render_figure_catalog(specs)
        write_text(staging / "analysis-report.md", analysis)
        write_text(staging / "stats-appendix.md", appendix)
        write_text(staging / "figure-catalog.md", catalog)
        html_report = "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>ChemRxiv same-source unseen-A analysis v1</title><body><pre>" + html.escape(analysis) + "</pre></body></html>\n"
        write_text(staging / "report" / "analysis-report.html", html_report)
        manifest = {
            "analysis_id": OUTPUT_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_script": str(script_path.relative_to(ROOT)),
            "analysis_script_sha256": sha256_file(script_path),
            "writes_only_analysis_bundle": True,
            "model_fits_started": 0,
            "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "random_seed": RANDOM_SEED,
            "bootstrap": {"unit": "unique amine group after three-repeat averaging", "n_groups": 10, "draws": BOOTSTRAP_DRAWS, "method": "percentile"},
            "primary_tests": {"test": "two-sided paired Wilcoxon", "p_adjustment": "Holm within each model line across two pre-registered A marginal contrasts"},
            "integrity": integrity_rows,
            "input_provenance": provenance,
            "output_tables": sorted(str(path.relative_to(staging)) for path in tables.glob("*")),
            "figures": specs,
            "boundary": "same-source grouped unseen-A only; not combinable with frozen development-to-external test results",
        }
        write_text(staging / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return OUTPUT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    all_integrity: list[dict[str, Any]] = []
    all_provenance: list[dict[str, Any]] = []
    all_repeat: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    all_group_repeat: list[pd.DataFrame] = []
    all_group_mean: list[pd.DataFrame] = []
    all_contrasts: list[pd.DataFrame] = []
    for line in LINES:
        frames: dict[str, pd.DataFrame] = {}
        for combo, _ in COMBOS:
            frame, provenance = read_task(line, combo)
            frames[combo] = frame
            all_integrity.append(provenance["integrity"])
            all_provenance.append(provenance)
        joined = join_line_predictions(frames, line)
        repeat, coverage, group_repeat, group_mean, contrasts = compute_tables(line, joined)
        all_repeat.append(repeat)
        all_coverage.append(coverage)
        all_group_repeat.append(group_repeat)
        all_group_mean.append(group_mean)
        all_contrasts.append(contrasts)
    output = write_bundle(
        all_integrity, pd.concat(all_repeat, ignore_index=True), pd.concat(all_coverage, ignore_index=True),
        pd.concat(all_group_repeat, ignore_index=True), pd.concat(all_group_mean, ignore_index=True),
        pd.concat(all_contrasts, ignore_index=True), all_provenance, Path(__file__).resolve(),
    )
    summary = pd.concat(all_contrasts, ignore_index=True).loc[:, [
        "line_id", "contrast_id", "mean_delta_mae", "ci95_group_bootstrap_low", "ci95_group_bootstrap_high",
        "p_wilcoxon_raw", "p_wilcoxon_holm_within_line_two_primary_A_tests", "rank_biserial",
    ]]
    print(json.dumps({"status": "passed", "output": str(output.relative_to(ROOT)), "contrasts": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
