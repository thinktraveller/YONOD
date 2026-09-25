"""Read-only inference audit for completed within-source product-utility tasks.

The audit never fits a model and never writes an analysis bundle.  It verifies
that paired Full/minus-P OOF predictions have identical source/sample/repeat/
fold membership, collapses repeated OOF errors to the preregistered unique
``non_p_input_group_id`` unit, then reports the within-source effect:

    Delta_P = MAE(minus_p) - MAE(full)

Positive values mean that adding P lowered MAE.  Its practical decision rule
is intentionally asymmetric: the 95% paired group-bootstrap upper bound must
be <= 0.01 MAE before the result supports omitting P in that source/model line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, shapiro, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_pnecessity_utility_result import verify as verify_task_result


PREP_ROOT = ROOT / "result" / "pnecessity_utility_prepare_v1"
MATRIX_PATH = PREP_ROOT / "static_config_matrix.csv"
SOURCES_PATH = PREP_ROOT / "source_manifest.json"
EXPECTED_REPEATS = 3
EXPECTED_FOLDS = 15
PRACTICAL_THRESHOLD = 0.01


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _stable_seed(*parts: str) -> int:
    material = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**32)


def _load_predictions(task_name: str) -> pd.DataFrame:
    directory = ROOT / "result" / task_name / "docs" / "predictions"
    paths = sorted(directory.glob("*.parquet"))
    _require(len(paths) == EXPECTED_FOLDS, f"{task_name}: expected {EXPECTED_FOLDS} prediction shards")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {"sample_id", "group_id", "repeat", "fold", "y_true", "y_pred"}
    _require(not required.difference(frame.columns), f"{task_name}: prediction fields are incomplete")
    frame = frame.loc[:, ["sample_id", "group_id", "repeat", "fold", "y_true", "y_pred"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    _require(
        not frame.duplicated(["sample_id", "repeat", "fold"]).any(),
        f"{task_name}: duplicate sample/repeat/fold OOF prediction",
    )
    _require(
        np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all(),
        f"{task_name}: non-finite OOF prediction or label",
    )
    return frame


def _validate_coverage(frame: pd.DataFrame, *, task_name: str, population: pd.DataFrame) -> None:
    expected_ids = set(population["sample_id"])
    _require(len(frame) == EXPECTED_REPEATS * len(population), f"{task_name}: OOF row count differs from 3× population")
    for repeat, part in frame.groupby("repeat", sort=True):
        _require(int(repeat) in {1, 2, 3}, f"{task_name}: unexpected repeat {repeat}")
        _require(len(part) == len(population), f"{task_name}: repeat {repeat} row count differs from population")
        _require(set(part["sample_id"]) == expected_ids, f"{task_name}: repeat {repeat} lacks exact population coverage")
    merged = frame.merge(population, on="sample_id", how="left", validate="many_to_one")
    _require(not merged["non_p_input_group_id"].isna().any(), f"{task_name}: prediction sample is absent from prepared data")
    # The precomputed-column splitter serializes its runtime group ID as
    # ``precomputed:<prepared non-P key>``.  Preserve that prefix in the
    # audit rather than stripping it, but prove it is a one-to-one encoding of
    # the prepared data's product-free grouping key.
    expected_runtime_group = "precomputed:" + merged["non_p_input_group_id"]
    _require(
        (merged["group_id"] == expected_runtime_group).all(),
        f"{task_name}: prediction group identity is not the expected precomputed non-P encoding",
    )


def _bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    _require(len(values) > 0 and np.isfinite(values).all(), "bootstrap values are invalid")
    generator = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    batch_size = 128
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sampled = generator.integers(0, len(values), size=(stop - start, len(values)))
        result[start:stop] = values[sampled].mean(axis=1)
    low, high = np.percentile(result, [2.5, 97.5])
    return float(low), float(high)


def _wilcoxon_summary(values: np.ndarray) -> dict[str, float | int | str | None]:
    nonzero = values[~np.isclose(values, 0.0, rtol=0.0, atol=1e-15)]
    if len(nonzero) == 0:
        return {"wilcoxon_statistic": 0.0, "p_raw": 1.0, "rank_biserial": 0.0, "n_nonzero_groups": 0, "wilcoxon_method": "all_zero"}
    statistic, p_value = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
    ranks = rankdata(np.abs(nonzero), method="average")
    rank_biserial = float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))
    return {
        "wilcoxon_statistic": float(statistic),
        "p_raw": float(p_value),
        "rank_biserial": rank_biserial,
        "n_nonzero_groups": int(len(nonzero)),
        "wilcoxon_method": "scipy_auto_two_sided_paired",
    }


def _normality_summary(values: np.ndarray) -> dict[str, float | str | None]:
    if not 3 <= len(values) <= 5_000:
        return {"shapiro_w": None, "shapiro_p": None, "normality_status": "not_assessed_n_outside_3_to_5000"}
    statistic, p_value = shapiro(values)
    return {"shapiro_w": float(statistic), "shapiro_p": float(p_value), "normality_status": "assessed_descriptive_only"}


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    count = len(p_values)
    order = p_values.sort_values(kind="mergesort")
    adjusted = pd.Series(index=order.index, dtype=float)
    running = 0.0
    for rank, (index, value) in enumerate(order.items()):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted.loc[index] = running
    return adjusted.reindex(p_values.index)


def _classification(ci_low: float, ci_high: float) -> str:
    if ci_low > PRACTICAL_THRESHOLD:
        return "supports_material_P_benefit_within_source_line"
    if ci_high <= PRACTICAL_THRESHOLD:
        return "supports_no_material_P_benefit_within_source_line"
    return "inconclusive_for_practical_P_benefit_within_source_line"


def _source_decision(lines: pd.DataFrame) -> str:
    classes = set(lines["classification"])
    if classes == {"supports_no_material_P_benefit_within_source_line"}:
        return "both_lines_support_no_material_P_benefit_within_source"
    if "supports_material_P_benefit_within_source_line" in classes:
        return "at_least_one_line_supports_material_P_benefit_source_level_nonnecessity_not_supported"
    return "inconclusive_across_frozen_robustness_lines"


def audit(*, bootstrap_draws: int, verify_all: bool) -> dict[str, Any]:
    matrix = pd.read_csv(MATRIX_PATH)
    _require(matrix.shape[0] == 36, "static configuration matrix must contain 36 tasks")
    _require(set(matrix["arm"]) == {"full", "minus_p"}, "matrix arms differ from paired design")
    source_payload = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    source_by_id = {str(item["source_id"]): item for item in source_payload["sources"]}

    verified_tasks = 0
    if verify_all:
        for config_path in matrix["config_path"].sort_values():
            verify_task_result(ROOT / str(config_path))
            verified_tasks += 1

    records: list[dict[str, Any]] = []
    for (source_id, line), pair in matrix.groupby(["source_id", "line"], sort=True):
        _require(len(pair) == 2 and set(pair["arm"]) == {"full", "minus_p"}, f"{source_id}/{line}: task pair incomplete")
        source = source_by_id[str(source_id)]
        population = pd.read_csv(ROOT / str(source["prepared_path"]), usecols=["sample_id", "non_p_input_group_id"])
        population["sample_id"] = population["sample_id"].astype(str)
        population["non_p_input_group_id"] = population["non_p_input_group_id"].astype(str)
        _require(not population["sample_id"].duplicated().any(), f"{source_id}: prepared sample identity is not unique")

        task_by_arm = pair.set_index("arm")["task_name"].to_dict()
        full = _load_predictions(str(task_by_arm["full"]))
        minus = _load_predictions(str(task_by_arm["minus_p"]))
        _validate_coverage(full, task_name=str(task_by_arm["full"]), population=population)
        _validate_coverage(minus, task_name=str(task_by_arm["minus_p"]), population=population)
        keys = ["sample_id", "repeat", "fold"]
        merged = full.merge(minus, on=keys, how="outer", validate="one_to_one", suffixes=("_full", "_minus"), indicator=True)
        _require((merged["_merge"] == "both").all(), f"{source_id}/{line}: Full/minus-P OOF membership differs")
        _require(
            np.allclose(merged["y_true_full"], merged["y_true_minus"], rtol=0.0, atol=0.0),
            f"{source_id}/{line}: paired OOF labels differ",
        )
        _require(
            (merged["group_id_full"] == merged["group_id_minus"]).all(),
            f"{source_id}/{line}: paired OOF group identity differs",
        )
        merged["full_abs_error"] = np.abs(merged["y_true_full"] - merged["y_pred_full"])
        merged["minus_abs_error"] = np.abs(merged["y_true_minus"] - merged["y_pred_minus"])
        grouped = (
            merged.groupby(["repeat", "group_id_full"], sort=False)
            .agg(full_abs_error=("full_abs_error", "mean"), minus_abs_error=("minus_abs_error", "mean"), n_rows=("sample_id", "size"))
            .reset_index()
        )
        expected_groups = int(population["non_p_input_group_id"].nunique())
        _require(len(grouped) == EXPECTED_REPEATS * expected_groups, f"{source_id}/{line}: incomplete repeat/group coverage")
        per_group = (
            grouped.groupby("group_id_full", sort=False)
            .agg(full_abs_error=("full_abs_error", "mean"), minus_abs_error=("minus_abs_error", "mean"), repeats=("repeat", "nunique"), rows_per_repeat=("n_rows", "mean"))
            .reset_index()
        )
        _require((per_group["repeats"] == EXPECTED_REPEATS).all(), f"{source_id}/{line}: group is not present in all repeats")
        differences = (per_group["minus_abs_error"] - per_group["full_abs_error"]).to_numpy(dtype=float)
        seed = _stable_seed("pnecessity-utility", str(source_id), str(line))
        ci_low, ci_high = _bootstrap_mean(differences, draws=bootstrap_draws, seed=seed)
        full_mae = float(merged["full_abs_error"].mean())
        minus_mae = float(merged["minus_abs_error"].mean())
        records.append({
            "source_id": str(source_id),
            "line": str(line),
            "full_task": str(task_by_arm["full"]),
            "minus_p_task": str(task_by_arm["minus_p"]),
            "n_rows": int(len(population)),
            "n_non_p_groups": expected_groups,
            "n_repeats": EXPECTED_REPEATS,
            "full_mae_row_weighted": full_mae,
            "minus_p_mae_row_weighted": minus_mae,
            "delta_p_row_weighted": minus_mae - full_mae,
            "delta_p_group_mean": float(differences.mean()),
            "delta_p_group_sd": float(differences.std(ddof=1)),
            "delta_p_group_median": float(np.median(differences)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": seed,
            "n_positive_groups": int((differences > 0).sum()),
            "n_negative_groups": int((differences < 0).sum()),
            "n_zero_groups": int(np.isclose(differences, 0.0, rtol=0.0, atol=1e-15).sum()),
            **_normality_summary(differences),
            **_wilcoxon_summary(differences),
            "practical_threshold_mae": PRACTICAL_THRESHOLD,
            "classification": _classification(ci_low, ci_high),
        })

    summary = pd.DataFrame.from_records(records).sort_values(["source_id", "line"], kind="mergesort").reset_index(drop=True)
    summary["p_holm_within_source_two_lines"] = summary.groupby("source_id", sort=False)["p_raw"].transform(_holm_adjust)
    decisions = pd.DataFrame.from_records([
        {"source_id": str(source_id), "source_decision": _source_decision(lines)}
        for source_id, lines in summary.groupby("source_id", sort=True)
    ])
    return {
        "status": "passed_legacy_alignment_only",
        "analysis_mode": "read_only_legacy_result_audit",
        "input_block_verification": "legacy_unverified",
        "full_condition_claim_permitted": False,
        "input_block_limit": "Historical strict outputs predate saved molecular/numeric block and fold-state identities; OOF pairing alone cannot certify all known C reached both models.",
        "model_fits_started": 0,
        "estimand": "Delta_P = MAE(minus_p) - MAE(full); positive means P lowers MAE",
        "independent_unit": "unique non_p_input_group_id; each group's absolute errors averaged over three repeats before inference",
        "practical_decision_rule": "support no material P benefit only if 95% paired group-bootstrap CI upper bound <= 0.01 MAE",
        "integrity_checked_tasks": verified_tasks,
        "comparisons": summary.to_dict(orient="records"),
        "source_decisions": decisions.to_dict(orient="records"),
    }


def _compact_report(payload: dict[str, Any]) -> str:
    summary = pd.DataFrame(payload["comparisons"])
    decisions = pd.DataFrame(payload["source_decisions"])
    lines = [
        "read_only_result_audit=passed_legacy_alignment_only",
        "input_block_verification=legacy_unverified; full_condition_claim_permitted=false",
        f"integrity_checked_tasks={payload['integrity_checked_tasks']}",
        f"bootstrap_draws={int(summary['bootstrap_draws'].iloc[0])}",
        "source_id\tline\tn_groups\tfull_mae\tminus_p_mae\tdelta_p\tci95\tp_holm\tclassification",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['source_id']}\t{row['line']}\t{int(row['n_non_p_groups'])}\t"
            f"{row['full_mae_row_weighted']:.6f}\t{row['minus_p_mae_row_weighted']:.6f}\t"
            f"{row['delta_p_group_mean']:.6f}\t[{row['ci95_low']:.6f},{row['ci95_high']:.6f}]\t"
            f"{row['p_holm_within_source_two_lines']:.6g}\t{row['classification']}"
        )
    lines.append("source_id\tsource_decision")
    for _, row in decisions.iterrows():
        lines.append(f"{row['source_id']}\t{row['source_decision']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--verify-all", action="store_true", help="also run the task-level read-only integrity verifier on all 36 inputs")
    parser.add_argument("--compact", action="store_true", help="print a compact tab-separated audit summary instead of JSON")
    args = parser.parse_args()
    _require(args.bootstrap_draws >= 1_000, "bootstrap draws must be at least 1,000")
    payload = audit(bootstrap_draws=args.bootstrap_draws, verify_all=args.verify_all)
    print(_compact_report(payload) if args.compact else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
