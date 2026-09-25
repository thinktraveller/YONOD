"""Strict, read-only analysis for the completed numeric-condition P ablation.

This program never calls a training entry point and never writes into a task's
``result/`` directory.  It re-runs the task-level *read-only* verifier, then
uses saved OOF predictions to estimate, separately for each source and frozen
model line,

    Delta_P = MAE(minus_p) - MAE(full).

The statistical unit is the unique non-product molecular component group.  A
group's absolute errors are averaged within repeat and then over the three
repeats before a paired percentile group bootstrap or Wilcoxon test is run.
Rows, folds, repeats, and the two model lines are deliberately not treated as
independent chemical replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, shapiro, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_pnecessity_numeric_conditions_result import verify as verify_task_result
from yonod.benchmark.config import BenchmarkConfig
from yonod.splits.grouping import build_group_ids


PREP_ROOT = ROOT / "result" / "pnecessity_numeric_conditions_prepare_v1"
MATRIX_PATH = PREP_ROOT / "generated_config_manifest.json"
DEFAULT_OUTPUT = ROOT / "project-docs" / "docs" / "product_necessity_multidataset" / "numeric_conditions_analysis_v1"
EXPECTED_SOURCES = {
    "ord_roche_borylation",
    "ord_nicolit",
    "ord_shields",
    "ord_c_n",
}
EXPECTED_LINES = {"morgan_rf", "mfp_lightgbm"}
EXPECTED_ARMS = {"full", "minus_p"}
EXPECTED_FOLDS = 15
EXPECTED_REPEATS = 3
PRACTICAL_THRESHOLD = 0.01
BOOTSTRAP_DRAWS = 10_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    material = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**32)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.loc[:, columns].copy()
    for column in columns:
        ordered[column] = ordered[column].astype(str)
    ordered = ordered.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _format_number(value: float | int | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "not assessed"
    return f"{float(value):.{digits}f}"


def _format_p(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "not assessed"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _safe_label(source: str, line: str) -> str:
    return source.removeprefix("ord_").replace("_", " ") + " | " + line.replace("_", " × ")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"{path}: YAML root is not a mapping")
    return value


def _get_task_run_dir(task_name: str) -> Path:
    task_root = ROOT / "result" / task_name
    candidates = sorted(task_root.glob("*/docs/manifests/run_manifest.json"))
    legacy = sorted(task_root.glob("manifests/run_manifest.json"))
    candidates.extend(legacy)
    _require(len(candidates) == 1, f"{task_name}: expected exactly one run manifest, found {len(candidates)}")
    manifest = candidates[0]
    if manifest.parent.parent.name == "docs":
        return manifest.parents[2]
    return manifest.parents[1]


def _load_predictions(run_dir: Path, task_name: str) -> pd.DataFrame:
    paths = sorted((run_dir / "docs" / "predictions").glob("*.parquet"))
    _require(len(paths) == EXPECTED_FOLDS, f"{task_name}: expected {EXPECTED_FOLDS} prediction shards, found {len(paths)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {"sample_id", "group_id", "repeat", "fold", "y_true", "y_pred"}
    _require(not required.difference(frame.columns), f"{task_name}: prediction fields are incomplete")
    frame = frame.loc[:, ["sample_id", "group_id", "repeat", "fold", "y_true", "y_pred"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    _require(not frame.duplicated(["sample_id", "repeat", "fold"]).any(), f"{task_name}: duplicate OOF prediction")
    _require(np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all(), f"{task_name}: non-finite prediction")
    return frame


def _load_folds(run_dir: Path, task_name: str) -> dict[tuple[int, int], dict[str, Any]]:
    paths = sorted((run_dir / "docs" / "folds").glob("*.json"))
    _require(len(paths) == EXPECTED_FOLDS, f"{task_name}: expected {EXPECTED_FOLDS} fold metadata files")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload["repeat"]), int(payload["fold"]))
        _require(key not in result, f"{task_name}: duplicate fold metadata for {key}")
        result[key] = payload
    _require(set(result) == {(repeat, fold) for repeat in range(1, 4) for fold in range(1, 6)}, f"{task_name}: incomplete repeat/fold grid")
    return result


def _load_split(run_dir: Path, task_name: str) -> pd.DataFrame:
    path = run_dir / "docs" / "manifests" / "split_manifest.parquet"
    _require(path.is_file(), f"{task_name}: split manifest is missing")
    frame = pd.read_parquet(path)
    needed = ["sample_id", "group_id", "repeat", "fold", "role"]
    _require(not set(needed).difference(frame.columns), f"{task_name}: split manifest fields are incomplete")
    return frame.loc[:, needed].copy()


def _validate_coverage(predictions: pd.DataFrame, population: pd.DataFrame, task_name: str) -> dict[str, Any]:
    expected_ids = set(population["sample_id"])
    _require(len(predictions) == EXPECTED_REPEATS * len(population), f"{task_name}: OOF row count is not 3× population")
    expected_group_by_id = population.set_index("sample_id")["expected_group_id"].to_dict()
    for repeat in range(1, EXPECTED_REPEATS + 1):
        part = predictions.loc[predictions["repeat"] == repeat]
        _require(len(part) == len(population), f"{task_name}: repeat {repeat} lacks population coverage")
        _require(set(part["sample_id"]) == expected_ids, f"{task_name}: repeat {repeat} has wrong sample membership")
        observed = part[["sample_id", "group_id"]].drop_duplicates("sample_id")
        _require(len(observed) == len(population), f"{task_name}: repeat {repeat} has non-unique sample/group identity")
        _require(
            all(expected_group_by_id.get(str(row.sample_id)) == str(row.group_id) for row in observed.itertuples(index=False)),
            f"{task_name}: runtime group IDs are not the expected non-P component groups",
        )
    return {
        "oof_rows": int(len(predictions)),
        "oof_rows_expected": int(EXPECTED_REPEATS * len(population)),
        "repeat_population_complete": True,
    }


def _bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    _require(len(values) > 0 and np.isfinite(values).all(), "invalid bootstrap values")
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    # Bound the index matrix to roughly 64 MB so a source with many groups stays safe.
    batch_size = max(1, min(256, 8_000_000 // max(1, len(values))))
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
        result[start:stop] = values[sampled].mean(axis=1)
    low, high = np.percentile(result, [2.5, 97.5])
    return float(low), float(high)


def _cluster_bootstrap_row_weighted(
    delta_error_sums: np.ndarray,
    row_counts: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap whole groups while retaining the row-MAE estimand.

    The repeated OOF prediction for each sample has already been averaged
    before these group totals are constructed.  A draw resamples groups (the
    independent clusters), then recomputes total delta absolute error divided
    by total sampled rows.  It is therefore not a row bootstrap.
    """
    _require(len(delta_error_sums) == len(row_counts) and len(row_counts) > 0, "invalid cluster bootstrap inputs")
    _require(np.isfinite(delta_error_sums).all() and np.isfinite(row_counts).all() and (row_counts > 0).all(), "invalid cluster bootstrap values")
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    batch_size = max(1, min(256, 8_000_000 // max(1, len(row_counts))))
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sampled = rng.integers(0, len(row_counts), size=(stop - start, len(row_counts)))
        result[start:stop] = delta_error_sums[sampled].sum(axis=1) / row_counts[sampled].sum(axis=1)
    low, high = np.percentile(result, [2.5, 97.5])
    return float(low), float(high)


def _wilcoxon_summary(values: np.ndarray) -> dict[str, Any]:
    nonzero = values[~np.isclose(values, 0.0, rtol=0.0, atol=1e-15)]
    if len(nonzero) == 0:
        return {
            "wilcoxon_statistic": 0.0,
            "p_raw": 1.0,
            "rank_biserial": 0.0,
            "n_nonzero_groups": 0,
            "wilcoxon_method": "all_zero",
        }
    statistic, p_value = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
    ranks = rankdata(np.abs(nonzero), method="average")
    return {
        "wilcoxon_statistic": float(statistic),
        "p_raw": float(p_value),
        "rank_biserial": float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks)),
        "n_nonzero_groups": int(len(nonzero)),
        "wilcoxon_method": "scipy_auto_two_sided_paired",
    }


def _normality_summary(values: np.ndarray) -> dict[str, Any]:
    if not 3 <= len(values) <= 5000:
        return {"shapiro_w": None, "shapiro_p": None, "normality_status": "not_assessed_n_outside_3_to_5000"}
    statistic, p_value = shapiro(values)
    return {"shapiro_w": float(statistic), "shapiro_p": float(p_value), "normality_status": "assessed_descriptive_only"}


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    count = len(p_values)
    ordered = p_values.sort_values(kind="mergesort")
    adjusted = pd.Series(index=ordered.index, dtype=float)
    running = 0.0
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted.loc[index] = running
    return adjusted.reindex(p_values.index)


def _classification(ci_low: float, ci_high: float, n_groups: int) -> str:
    if n_groups < 5:
        return "inconclusive_insufficient_unique_groups"
    if ci_low > PRACTICAL_THRESHOLD:
        return "supports_material_P_benefit_within_source_line"
    if ci_high <= PRACTICAL_THRESHOLD:
        return "supports_no_material_P_benefit_within_source_line"
    return "inconclusive_for_practical_P_benefit_within_source_line"


def _source_decision(lines: pd.DataFrame) -> str:
    classes = set(lines["classification"])
    if classes == {"supports_no_material_P_benefit_within_source_line"}:
        return "both_frozen_lines_support_no_material_P_benefit_within_source"
    if "supports_material_P_benefit_within_source_line" in classes:
        return "at_least_one_line_supports_material_P_benefit;_source_level_omission_not_supported"
    return "inconclusive_across_frozen_robustness_lines"


def _no_p_leakage(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_no_p_leakage(key) and _no_p_leakage(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_no_p_leakage(item) for item in value)
    return str(value) != "p_smiles"


def _descriptor_without_p(raw: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = raw.get("descriptors")
    _require(isinstance(descriptors, list) and len(descriptors) == 1, "expected exactly one descriptor declaration")
    result: list[dict[str, Any]] = []
    for descriptor in descriptors:
        current = dict(descriptor)
        current.pop("id", None)
        current["columns"] = [column for column in current.get("columns", []) if column != "p_smiles"]
        result.append(current)
    return result


def _source_population(source: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ROOT / str(source["prepared_path"])
    frame = pd.read_csv(path)
    required = {"sample_id", "yield", "non_p_input_group_id"}
    numeric_columns = [str(column) for column in source["numeric_c"]]
    _require(not required.difference(frame.columns), f"{source['source_id']}: prepared fields are incomplete")
    _require(not set(numeric_columns).difference(frame.columns), f"{source['source_id']}: declared numeric C is absent from prepared CSV")
    _require(not frame["sample_id"].astype(str).duplicated().any(), f"{source['source_id']}: sample IDs are not unique")
    population = frame.loc[:, ["sample_id", "non_p_input_group_id"]].copy()
    population["sample_id"] = population["sample_id"].astype(str)
    numeric_evidence: dict[str, Any] = {"source_id": str(source["source_id"]), "prepared_path": str(source["prepared_path"]), "prepared_sha256_current": _sha256_file(path), "prepared_sha256_declared": str(source["prepared_sha256"]), "n_rows": int(len(frame)), "numeric_columns": ";".join(numeric_columns)}
    _require(numeric_evidence["prepared_sha256_current"] == numeric_evidence["prepared_sha256_declared"], f"{source['source_id']}: prepared CSV SHA-256 drift")
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="raise")
        _require(np.isfinite(series.to_numpy(dtype=float)).all(), f"{source['source_id']}: numeric C has non-finite values")
        numeric_evidence[f"{column}_n_unique"] = int(series.nunique(dropna=False))
        numeric_evidence[f"{column}_min"] = float(series.min())
        numeric_evidence[f"{column}_max"] = float(series.max())
    return population, numeric_evidence


def _runtime_group_population(config_path: Path) -> pd.DataFrame:
    """Rebuild the grouping contract from this YAML's molecular component columns.

    ``non_p_input_group_id`` in the prepared table deliberately includes
    numeric-C values for historical identity purposes.  The numeric-C task's
    component-holdout grouping deliberately does *not*: a numeric variant of
    the same molecular tuple stays in one molecular group.  This reconstruction
    avoids accidentally treating those two different contracts as equivalent.
    """
    config = BenchmarkConfig.from_file(config_path)
    frame = config.validate_dataset()
    groups = build_group_ids(frame, config.sample_id_col, config.grouping, config.smiles_cols)
    output = pd.DataFrame({"sample_id": frame[config.sample_id_col].astype(str), "expected_group_id": groups.astype(str)})
    _require(not output["sample_id"].duplicated().any(), f"{config_path}: duplicate sample IDs in runtime grouping")
    return output


def _validate_pair_config(
    full_raw: dict[str, Any],
    minus_raw: dict[str, Any],
    full_config_path: Path,
    minus_config_path: Path,
    source_id: str,
    line: str,
) -> dict[str, Any]:
    full_dataset = full_raw.get("dataset", {})
    minus_dataset = minus_raw.get("dataset", {})
    _require(full_dataset.get("path") == minus_dataset.get("path"), f"{source_id}/{line}: paired data path differs")
    _require(full_dataset.get("sample_id_col") == minus_dataset.get("sample_id_col") == "sample_id", f"{source_id}/{line}: paired sample ID differs")
    full_roles = full_dataset.get("column_roles", {})
    minus_roles = minus_dataset.get("column_roles", {})
    _require(full_roles.get("label") == minus_roles.get("label") == "yield", f"{source_id}/{line}: labels differ")
    _require(full_roles.get("products") == ["p_smiles"], f"{source_id}/{line}: Full product role is invalid")
    _require(minus_roles.get("products") == [], f"{source_id}/{line}: minus-P product role is invalid")
    _require("p_smiles" in full_roles.get("reactants", []), f"{source_id}/{line}: Full lacks P molecular input")
    _require("p_smiles" not in minus_roles.get("reactants", []), f"{source_id}/{line}: minus-P reactants contain P")
    _require(_no_p_leakage(minus_roles), f"{source_id}/{line}: minus-P roles leak p_smiles")
    _require(
        [item for item in full_roles.get("reactants", []) if item != "p_smiles"] == minus_roles.get("reactants", []),
        f"{source_id}/{line}: non-P molecular inputs differ",
    )
    _require(full_roles.get("conditions") == minus_roles.get("conditions"), f"{source_id}/{line}: numeric C list differs")
    _require(full_dataset.get("numeric_conditions") == minus_dataset.get("numeric_conditions"), f"{source_id}/{line}: numeric C YAML declaration differs")
    full_numeric_contract = BenchmarkConfig.from_file(full_config_path).raw["numeric_contract"]
    minus_numeric_contract = BenchmarkConfig.from_file(minus_config_path).raw["numeric_contract"]
    _require(full_numeric_contract == minus_numeric_contract, f"{source_id}/{line}: normalized numeric C contract differs")
    _require(_descriptor_without_p(full_raw) == _descriptor_without_p(minus_raw), f"{source_id}/{line}: descriptor differs beyond P")
    for field in ("models", "model_params", "evaluation"):
        _require(full_raw.get(field) == minus_raw.get(field), f"{source_id}/{line}: paired {field} differs")
    for field in ("prepared_data_sha256", "raw_data_sha256", "comparison"):
        _require(
            full_raw.get("metadata", {}).get(field) == minus_raw.get("metadata", {}).get(field),
            f"{source_id}/{line}: paired metadata {field} differs",
        )
    return {
        "paired_config_identical_except_p": True,
        "minus_p_static_p_leakage_check": "passed",
        "numeric_contract": full_numeric_contract,
        "numeric_columns": list(full_roles.get("conditions", [])),
    }


def _validate_pair_runtime(
    source_id: str,
    line: str,
    full_task: str,
    minus_task: str,
    full_predictions: pd.DataFrame,
    minus_predictions: pd.DataFrame,
    full_split: pd.DataFrame,
    minus_split: pd.DataFrame,
    full_folds: dict[tuple[int, int], dict[str, Any]],
    minus_folds: dict[tuple[int, int], dict[str, Any]],
    numeric_contract: Any,
    numeric_columns: list[str],
) -> dict[str, Any]:
    split_columns = ["sample_id", "group_id", "repeat", "fold", "role"]
    full_digest = _mapping_digest(full_split, split_columns)
    minus_digest = _mapping_digest(minus_split, split_columns)
    _require(full_digest == minus_digest, f"{source_id}/{line}: Full/minus-P split membership differs")
    for task_name, predictions, split in (
        (full_task, full_predictions, full_split),
        (minus_task, minus_predictions, minus_split),
    ):
        valid = split.loc[split["role"].astype(str) == "valid", ["sample_id", "group_id", "repeat", "fold"]].copy()
        valid["sample_id"] = valid["sample_id"].astype(str)
        valid["group_id"] = valid["group_id"].astype(str)
        _require(not valid.duplicated(["sample_id", "repeat", "fold"]).any(), f"{task_name}: duplicate validation split member")
        observed = predictions.loc[:, ["sample_id", "group_id", "repeat", "fold"]].copy()
        observed["sample_id"] = observed["sample_id"].astype(str)
        observed["group_id"] = observed["group_id"].astype(str)
        _require(
            _mapping_digest(valid, ["sample_id", "group_id", "repeat", "fold"])
            == _mapping_digest(observed, ["sample_id", "group_id", "repeat", "fold"]),
            f"{task_name}: OOF predictions do not equal manifest validation membership",
        )
    keys = ["sample_id", "repeat", "fold"]
    merged = full_predictions.merge(minus_predictions, on=keys, how="outer", suffixes=("_full", "_minus"), indicator=True, validate="one_to_one")
    _require((merged["_merge"] == "both").all(), f"{source_id}/{line}: Full/minus-P OOF membership differs")
    _require(np.array_equal(merged["y_true_full"].to_numpy(), merged["y_true_minus"].to_numpy()), f"{source_id}/{line}: paired labels differ")
    _require((merged["group_id_full"] == merged["group_id_minus"]).all(), f"{source_id}/{line}: paired OOF groups differ")
    state_differences = 0
    constant_fold_counts = {column: 0 for column in numeric_columns}
    for key in sorted(full_folds):
        full_meta = full_folds[key]
        minus_meta = minus_folds.get(key)
        _require(minus_meta is not None, f"{source_id}/{line}: paired fold metadata absent for {key}")
        # ``split_hash`` intentionally includes the arm-specific run_id in the
        # manifest payload.  Pairing is instead proved above from the complete
        # split membership digest and here from each fold's train/valid IDs.
        for field in ("n_train", "n_valid", "train_sample_ids_hash", "valid_sample_ids_hash"):
            _require(full_meta.get(field) == minus_meta.get(field), f"{source_id}/{line}: {key} differs in {field}")
        full_numeric = ((full_meta.get("feature_transformer") or {}).get("numeric_conditions") or {})
        minus_numeric = ((minus_meta.get("feature_transformer") or {}).get("numeric_conditions") or {})
        _require(full_numeric.get("dimension") == len(numeric_columns), f"{source_id}/{line}: {key} Full numeric dimension is wrong")
        _require(minus_numeric.get("dimension") == len(numeric_columns), f"{source_id}/{line}: {key} minus-P numeric dimension is wrong")
        _require(full_numeric.get("input_identity") == minus_numeric.get("input_identity"), f"{source_id}/{line}: {key} numeric identity differs")
        _require((full_numeric.get("state") or {}).get("contract") == numeric_contract, f"{source_id}/{line}: {key} Full numeric contract drift")
        _require((minus_numeric.get("state") or {}).get("contract") == numeric_contract, f"{source_id}/{line}: {key} minus-P numeric contract drift")
        _require((full_numeric.get("state") or {}) == (minus_numeric.get("state") or {}), f"{source_id}/{line}: {key} fold-local numeric state differs")
        _require(int((full_numeric.get("state") or {}).get("fit_rows", -1)) == int(full_meta.get("n_train", -2)), f"{source_id}/{line}: {key} numeric fit is not fold-local")
        state_differences += int((full_numeric.get("state") or {}) != (minus_numeric.get("state") or {}))
        state_columns = (full_numeric.get("state") or {}).get("columns") or []
        _require([item.get("source") for item in state_columns] == numeric_columns, f"{source_id}/{line}: {key} numeric column order differs")
        for item in state_columns:
            if bool(item.get("constant_train")):
                constant_fold_counts[str(item.get("source"))] += 1
    _require(state_differences == 0, f"{source_id}/{line}: shared numeric fold state differs")
    return {
        "split_membership_sha256": full_digest,
        "oof_membership_paired": True,
        "labels_paired": True,
        "runtime_groups_paired": True,
        "fold_train_valid_membership_paired": True,
        "numeric_conditions_entered_all_15_folds": True,
        "numeric_states_identical_across_arms": True,
        "numeric_constant_train_folds": json.dumps(constant_fold_counts, sort_keys=True),
    }


@dataclass
class AnalysisResult:
    task_verification: pd.DataFrame
    pair_integrity: pd.DataFrame
    repeat_metrics: pd.DataFrame
    group_effects: pd.DataFrame
    line_statistics: pd.DataFrame
    source_summary: pd.DataFrame
    numeric_evidence: pd.DataFrame
    provenance: dict[str, Any]


def analyze(bootstrap_draws: int = BOOTSTRAP_DRAWS) -> AnalysisResult:
    _require(MATRIX_PATH.is_file(), f"missing generated task manifest: {MATRIX_PATH}")
    manifest = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    generated = manifest.get("generated")
    sources = manifest.get("selected_sources")
    _require(isinstance(generated, list) and len(generated) == 16, "task manifest does not contain exactly 16 generated tasks")
    _require(isinstance(sources, list) and {str(item["source_id"]) for item in sources} == EXPECTED_SOURCES, "task manifest sources differ")
    source_by_id = {str(item["source_id"]): item for item in sources}
    matrix = pd.DataFrame(generated)
    _require(set(matrix["source_id"]) == EXPECTED_SOURCES, "task matrix sources differ")
    _require(set(matrix["line_id"]) == EXPECTED_LINES and set(matrix["arm"]) == EXPECTED_ARMS, "task matrix arms or lines differ")
    _require(len(matrix.groupby(["source_id", "line_id", "arm"])) == 16, "task matrix has duplicate source/line/arm entries")

    task_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    line_records: list[dict[str, Any]] = []
    numeric_records: list[dict[str, Any]] = []

    for (source_id, line), pair in matrix.groupby(["source_id", "line_id"], sort=True):
        source_id = str(source_id)
        line = str(line)
        _require(len(pair) == 2 and set(pair["arm"]) == EXPECTED_ARMS, f"{source_id}/{line}: task pair incomplete")
        task_by_arm = pair.set_index("arm").to_dict(orient="index")
        arm_payload: dict[str, dict[str, Any]] = {}
        for arm in sorted(EXPECTED_ARMS):
            item = task_by_arm[arm]
            config_path = ROOT / str(item["config_path"])
            task_name = str(item["task_name"])
            verification = verify_task_result(config_path)
            _require(verification["status"] == "passed", f"{task_name}: task verifier did not pass")
            run_dir = _get_task_run_dir(task_name)
            raw = _load_yaml(config_path)
            arm_payload[arm] = {
                "task_name": task_name,
                "config_path": config_path,
                "raw": raw,
                "run_dir": run_dir,
                "verification": verification,
                "predictions": _load_predictions(run_dir, task_name),
                "split": _load_split(run_dir, task_name),
                "folds": _load_folds(run_dir, task_name),
                "runtime_population": _runtime_group_population(config_path),
            }
            task_records.append({
                "source_id": source_id,
                "line": line,
                "arm": arm,
                "task_name": task_name,
                "config_path": str(item["config_path"]),
                "config_sha256_current": _sha256_file(config_path),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "run_id": verification["run_id"],
                "task_verifier": verification["status"],
                "sqlite_succeeded_folds": int(verification["succeeded_folds"]),
                "numeric_columns": ";".join(verification["numeric_columns"]),
                "prediction_shards": EXPECTED_FOLDS,
                "fold_metadata_shards": EXPECTED_FOLDS,
                "html_markdown_reports": "present",
            })

        config_evidence = _validate_pair_config(
            arm_payload["full"]["raw"],
            arm_payload["minus_p"]["raw"],
            arm_payload["full"]["config_path"],
            arm_payload["minus_p"]["config_path"],
            source_id,
            line,
        )
        population, numeric_evidence = _source_population(source_by_id[source_id])
        bound_population = arm_payload["full"]["runtime_population"]
        minus_population = arm_payload["minus_p"]["runtime_population"]
        _require(
            _mapping_digest(bound_population, ["sample_id", "expected_group_id"]) == _mapping_digest(minus_population, ["sample_id", "expected_group_id"]),
            f"{source_id}/{line}: arms rebuild different non-P molecular runtime groups",
        )
        full_coverage = _validate_coverage(arm_payload["full"]["predictions"], bound_population, arm_payload["full"]["task_name"])
        minus_coverage = _validate_coverage(arm_payload["minus_p"]["predictions"], bound_population, arm_payload["minus_p"]["task_name"])
        runtime_evidence = _validate_pair_runtime(
            source_id,
            line,
            arm_payload["full"]["task_name"],
            arm_payload["minus_p"]["task_name"],
            arm_payload["full"]["predictions"],
            arm_payload["minus_p"]["predictions"],
            arm_payload["full"]["split"],
            arm_payload["minus_p"]["split"],
            arm_payload["full"]["folds"],
            arm_payload["minus_p"]["folds"],
            config_evidence["numeric_contract"],
            config_evidence["numeric_columns"],
        )
        numeric_evidence.update({
            "line": line,
            "prepared_non_p_plus_numeric_identity_groups": int(population["non_p_input_group_id"].nunique()),
            "runtime_non_p_groups": int(bound_population["expected_group_id"].nunique()),
            "numeric_conditions_entered_all_15_folds": True,
            "numeric_constant_train_folds": runtime_evidence["numeric_constant_train_folds"],
        })
        numeric_records.append(numeric_evidence)

        full = arm_payload["full"]["predictions"]
        minus = arm_payload["minus_p"]["predictions"]
        keys = ["sample_id", "repeat", "fold"]
        joined = full.merge(minus, on=keys, how="inner", validate="one_to_one", suffixes=("_full", "_minus"))
        joined["full_abs_error"] = np.abs(joined["y_true_full"] - joined["y_pred_full"])
        joined["minus_p_abs_error"] = np.abs(joined["y_true_minus"] - joined["y_pred_minus"])
        joined["delta_abs_error"] = joined["minus_p_abs_error"] - joined["full_abs_error"]

        for repeat in range(1, EXPECTED_REPEATS + 1):
            part = joined.loc[joined["repeat"] == repeat]
            repeat_records.append({
                "source_id": source_id,
                "line": line,
                "repeat": repeat,
                "n_rows": int(len(part)),
                "full_mae_row_weighted": float(part["full_abs_error"].mean()),
                "minus_p_mae_row_weighted": float(part["minus_p_abs_error"].mean()),
                "delta_p_row_weighted": float(part["delta_abs_error"].mean()),
            })

        by_repeat_group = (
            joined.groupby(["repeat", "group_id_full"], sort=True)
            .agg(full_abs_error=("full_abs_error", "mean"), minus_p_abs_error=("minus_p_abs_error", "mean"), rows_per_repeat=("sample_id", "size"))
            .reset_index()
        )
        n_groups = int(bound_population["expected_group_id"].nunique())
        _require(len(by_repeat_group) == EXPECTED_REPEATS * n_groups, f"{source_id}/{line}: repeat/group OOF coverage incomplete")
        per_group = (
            by_repeat_group.groupby("group_id_full", sort=True)
            .agg(full_abs_error=("full_abs_error", "mean"), minus_p_abs_error=("minus_p_abs_error", "mean"), repeats=("repeat", "nunique"), rows_per_repeat=("rows_per_repeat", "mean"))
            .reset_index()
        )
        _require((per_group["repeats"] == EXPECTED_REPEATS).all(), f"{source_id}/{line}: group not present in all repeats")
        per_group["delta_p_group"] = per_group["minus_p_abs_error"] - per_group["full_abs_error"]
        # The practical boundary is in row-weighted MAE units.  Collapse each sample's
        # three OOF errors first, then retain complete molecular groups as bootstrap
        # clusters.  This preserves the MAE estimand while respecting cluster dependence.
        per_sample = (
            joined.groupby(["sample_id", "group_id_full"], sort=True)
            .agg(full_abs_error=("full_abs_error", "mean"), minus_p_abs_error=("minus_p_abs_error", "mean"), repeats=("repeat", "nunique"))
            .reset_index()
        )
        _require((per_sample["repeats"] == EXPECTED_REPEATS).all(), f"{source_id}/{line}: sample not observed in all repeats")
        per_sample["delta_abs_error"] = per_sample["minus_p_abs_error"] - per_sample["full_abs_error"]
        row_weighted_groups = (
            per_sample.groupby("group_id_full", sort=True)
            .agg(delta_error_sum=("delta_abs_error", "sum"), row_count=("sample_id", "size"))
            .reset_index()
        )
        _require(
            _mapping_digest(per_group, ["group_id_full"]) == _mapping_digest(row_weighted_groups, ["group_id_full"]),
            f"{source_id}/{line}: repeat-averaged samples do not cover the inference groups",
        )
        per_group = per_group.merge(row_weighted_groups, on="group_id_full", how="inner", validate="one_to_one")
        per_group.insert(0, "line", line)
        per_group.insert(0, "source_id", source_id)
        group_frames.append(per_group.rename(columns={"group_id_full": "group_id"}))
        values = per_group["delta_p_group"].to_numpy(dtype=float)
        seed = _stable_seed("pnecessity-numeric-conditions-v1", source_id, line)
        ci_low, ci_high = _cluster_bootstrap_row_weighted(
            per_group["delta_error_sum"].to_numpy(dtype=float),
            per_group["row_count"].to_numpy(dtype=float),
            draws=bootstrap_draws,
            seed=seed,
        )
        line_records.append({
            "source_id": source_id,
            "line": line,
            "full_task": arm_payload["full"]["task_name"],
            "minus_p_task": arm_payload["minus_p"]["task_name"],
            "n_rows": int(len(population)),
            "n_unique_non_p_groups": n_groups,
            "max_group_rows_per_repeat": int(per_group["rows_per_repeat"].max()),
            "n_repeats": EXPECTED_REPEATS,
            "full_mae_row_weighted": float(joined["full_abs_error"].mean()),
            "minus_p_mae_row_weighted": float(joined["minus_p_abs_error"].mean()),
            "delta_p_row_weighted": float(per_sample["delta_abs_error"].mean()),
            "delta_p_group_mean": float(values.mean()),
            "delta_p_group_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "delta_p_group_median": float(np.median(values)),
            "delta_p_group_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": seed,
            "n_positive_groups": int((values > 0).sum()),
            "n_negative_groups": int((values < 0).sum()),
            "n_zero_groups": int(np.isclose(values, 0.0, rtol=0.0, atol=1e-15).sum()),
            "practical_threshold_mae": PRACTICAL_THRESHOLD,
            **_normality_summary(values),
            **_wilcoxon_summary(values),
            "classification": _classification(ci_low, ci_high, n_groups),
        })
        pair_records.append({
            "source_id": source_id,
            "line": line,
            "full_task": arm_payload["full"]["task_name"],
            "minus_p_task": arm_payload["minus_p"]["task_name"],
            **config_evidence,
            **runtime_evidence,
            "full_oof_rows": full_coverage["oof_rows"],
            "minus_p_oof_rows": minus_coverage["oof_rows"],
            "full_repeat_population_complete": full_coverage["repeat_population_complete"],
            "minus_p_repeat_population_complete": minus_coverage["repeat_population_complete"],
        })

    line_statistics = pd.DataFrame(line_records).sort_values(["source_id", "line"], kind="mergesort").reset_index(drop=True)
    line_statistics["p_holm_within_source_two_lines"] = line_statistics.groupby("source_id", sort=False)["p_raw"].transform(_holm_adjust)
    source_summary = pd.DataFrame([
        {
            "source_id": source_id,
            "n_rows": int(lines["n_rows"].iloc[0]),
            "n_unique_non_p_groups": int(lines["n_unique_non_p_groups"].iloc[0]),
            "morgan_rf_classification": str(lines.loc[lines["line"] == "morgan_rf", "classification"].iloc[0]),
            "mfp_lightgbm_classification": str(lines.loc[lines["line"] == "mfp_lightgbm", "classification"].iloc[0]),
            "source_decision": _source_decision(lines),
        }
        for source_id, lines in line_statistics.groupby("source_id", sort=True)
    ])
    provenance = {
        "analysis_status": "passed",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": _sha256_file(Path(__file__)),
        "generated_task_manifest": str(MATRIX_PATH.relative_to(ROOT)),
        "generated_task_manifest_sha256": _sha256_file(MATRIX_PATH),
        "n_tasks": int(len(task_records)),
        "n_pairs": int(len(pair_records)),
        "n_sources": int(len(source_summary)),
        "model_fits_started_by_analysis": 0,
        "metric": "Delta_P = MAE(minus_p) - MAE(full); positive values favour Full/P",
        "independent_unit": "unique runtime non-P molecular component group; sample errors averaged across 3 repeats before clustered inference",
        "bootstrap": {"draws": bootstrap_draws, "ci": "paired percentile 95%", "resampling_unit": "complete unique groups", "primary_estimand": "row-weighted delta MAE recomputed as sum(delta error)/sum(rows) per cluster draw"},
        "inference": "two-sided paired Wilcoxon signed-rank on equal-group effects with rank-biserial effect; Holm correction within each source across its two frozen model lines",
        "practical_rule": "support no material P benefit only when the 95% group-bootstrap CI upper bound is <= 0.01 MAE",
        "scope": "within-source engineering utility; no cross-source pooling or general chemical claim",
    }
    return AnalysisResult(
        task_verification=pd.DataFrame(task_records).sort_values(["source_id", "line", "arm"], kind="mergesort"),
        pair_integrity=pd.DataFrame(pair_records).sort_values(["source_id", "line"], kind="mergesort"),
        repeat_metrics=pd.DataFrame(repeat_records).sort_values(["source_id", "line", "repeat"], kind="mergesort"),
        group_effects=pd.concat(group_frames, ignore_index=True).sort_values(["source_id", "line", "group_id"], kind="mergesort"),
        line_statistics=line_statistics,
        source_summary=source_summary.sort_values("source_id", kind="mergesort"),
        numeric_evidence=pd.DataFrame(numeric_records).sort_values(["source_id", "line"], kind="mergesort"),
        provenance=provenance,
    )


def _write_markdown_table(frame: pd.DataFrame, path: Path, columns: list[str], rename: dict[str, str] | None = None) -> None:
    display = frame.loc[:, columns].copy()
    if rename:
        display = display.rename(columns=rename)
    path.write_text(_markdown_table(display) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without optional pandas/tabulate extras."""
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "not assessed"
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    headers = [cell(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _write_figures(result: AnalysisResult, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    summary = result.line_statistics.copy().sort_values(["source_id", "line"], kind="mergesort").reset_index(drop=True)
    colors = {
        "supports_no_material_P_benefit_within_source_line": "#0072B2",
        "supports_material_P_benefit_within_source_line": "#D55E00",
        "inconclusive_for_practical_P_benefit_within_source_line": "#CC79A7",
        "inconclusive_insufficient_unique_groups": "#CC79A7",
    }
    height = max(4.4, 0.58 * len(summary) + 1.8)
    fig, axis = plt.subplots(figsize=(8.0, height))
    positions = np.arange(len(summary))[::-1]
    for position, (_, row) in zip(positions, summary.iterrows()):
        color = colors[str(row["classification"])]
        mean = float(row["delta_p_row_weighted"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="o", color=color, ecolor=color, capsize=3.5, markersize=6, linewidth=1.7)
    axis.axvline(0.0, color="#000000", linewidth=1.0, zorder=0)
    axis.axvline(PRACTICAL_THRESHOLD, color="#D55E00", linestyle="--", linewidth=1.3, zorder=0)
    bound = max(abs(float(summary["ci95_low"].min())), abs(float(summary["ci95_high"].max())), PRACTICAL_THRESHOLD) + 0.002
    axis.set_xlim(min(-0.002, -bound), max(0.012, bound))
    axis.set_yticks(positions, [_safe_label(str(row.source_id), str(row.line)) for row in summary.itertuples(index=False)])
    axis.set_xlabel("Clustered row-MAE ΔP = MAE(minus-P) − MAE(Full); positive favours P")
    axis.grid(axis="x", color="#999999", alpha=0.25, linewidth=0.7)
    axis.text(
        PRACTICAL_THRESHOLD - 0.00015,
        positions.max() - 0.12,
        "0.01 practical threshold",
        color="#D55E00",
        ha="right",
        va="top",
        rotation=90,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.30, bottom=0.18, right=0.96, top=0.96)
    fig.savefig(figures / "figure-01-delta-p-forest.pdf", bbox_inches="tight")
    fig.savefig(figures / "figure-01-delta-p-forest.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    groups = result.group_effects.copy().sort_values(["source_id", "line", "group_id"], kind="mergesort")
    labels = [_safe_label(str(row.source_id), str(row.line)) for row in summary.itertuples(index=False)]
    values = [
        groups.loc[(groups["source_id"] == row.source_id) & (groups["line"] == row.line), "delta_p_group"].to_numpy(dtype=float)
        for row in summary.itertuples(index=False)
    ]
    fig, axis = plt.subplots(figsize=(10.0, 5.3))
    # C-N has >22k molecular groups.  Drawing every outlier into a 600-DPI
    # raster and vector PDF is both unreadable and impractically slow; the
    # box/whisker summary still conveys the distribution while the complete
    # unaggregated group values remain in tables/group_effects.csv.
    box = axis.boxplot(values, labels=labels, patch_artist=True, showfliers=False, whis=1.5)
    for patch, (_, row) in zip(box["boxes"], summary.iterrows()):
        patch.set_facecolor(colors[str(row["classification"])])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")
    for component in ("medians", "whiskers", "caps"):
        for artist in box[component]:
            artist.set_color("#333333")
            artist.set_linewidth(1.0)
    axis.axhline(0.0, color="#000000", linewidth=1.0, zorder=0)
    axis.axhline(PRACTICAL_THRESHOLD, color="#D55E00", linestyle="--", linewidth=1.3, zorder=0)
    axis.set_ylabel("Per-group ΔP after averaging three repeats")
    axis.tick_params(axis="x", labelrotation=35, labelsize=9)
    axis.grid(axis="y", color="#999999", alpha=0.25, linewidth=0.7)
    ymax = max(float(groups["delta_p_group"].max()), PRACTICAL_THRESHOLD)
    ymin = min(float(groups["delta_p_group"].min()), 0.0)
    margin = max(0.003, 0.1 * (ymax - ymin if ymax > ymin else 0.01))
    axis.set_ylim(ymin - margin, ymax + margin)
    for index, row in summary.iterrows():
        axis.text(index + 1, axis.get_ylim()[1] - margin * 0.3, f"n={int(row['n_unique_non_p_groups'])}", ha="center", va="top", fontsize=8)
    fig.subplots_adjust(left=0.11, bottom=0.30, right=0.98, top=0.96)
    fig.savefig(figures / "figure-02-group-delta-distributions.pdf", bbox_inches="tight")
    fig.savefig(figures / "figure-02-group-delta-distributions.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _analysis_report(result: AnalysisResult) -> str:
    summary = result.line_statistics
    decisions = result.source_summary
    no_material = int((summary["classification"] == "supports_no_material_P_benefit_within_source_line").sum())
    material = int((summary["classification"] == "supports_material_P_benefit_within_source_line").sum())
    inconclusive = len(summary) - no_material - material
    table = summary.loc[:, ["source_id", "line", "n_rows", "n_unique_non_p_groups", "full_mae_row_weighted", "minus_p_mae_row_weighted", "delta_p_row_weighted", "ci95_low", "ci95_high", "classification"]].copy()
    for column in ["full_mae_row_weighted", "minus_p_mae_row_weighted", "delta_p_row_weighted", "ci95_low", "ci95_high"]:
        table[column] = table[column].map(lambda value: _format_number(value, 6))
    return f"""# Numeric-condition P utility: strict result analysis

## Question and estimand

For each of four local ORD sources and two frozen model lines, does adding product structure P to the already recorded non-product molecular inputs and variable numeric conditions reduce holdout MAE by a practically meaningful amount? The fixed estimand is `Delta_P = MAE(minus_p) - MAE(full)`, so positive values favour Full/P. Labels are 0–1 yield and the practical boundary is `0.01` MAE (about one percentage point yield).

## Integrity gate

All **16/16** tasks passed the task-level read-only verifier: exactly 15 SQLite `succeeded` folds, 15 OOF prediction shards, 15 fold metadata shards, complete metrics, and non-empty HTML/Markdown reports. Every pair also passed an additional analysis-level gate: identical prepared CSV, labels, non-P molecular inputs, numeric-C contract, model settings, 5×3 split membership, OOF `(sample_id, repeat, fold)` membership, labels, and runtime non-P group identity. Static inspection confirms `minus_p` contains no `p_smiles` in its roles; only Full has P in its molecular descriptor inputs.

For all 8 pairs, each declared numeric C column is present as a feature block of the recorded dimension in all 15 fold metadata records, fitted on that fold's train rows only, and has the same input identity and fitted state in Full and minus-P. The exact evidence is in [`tables/numeric_condition_evidence.csv`](tables/numeric_condition_evidence.csv) and [`tables/pair_integrity.csv`](tables/pair_integrity.csv).

## Statistical unit and rule

The independent resampling unit is one **unique non-P molecular component group** within a source. Each sample's absolute errors are first averaged across its three OOF repeats. The primary practical estimand remains conventional row-weighted MAE: bootstrap draws resample complete molecular groups, then recompute `sum(delta absolute error) / sum(rows)` within the draw. Thus the 0.01 row-MAE threshold and its group-cluster CI use the same estimand. Rows, folds, repeats, and frozen model lines were not used as independent observations. A separate equal-group mean is retained only as a sensitivity distribution for Wilcoxon/rank-biserial reporting. The practical decision is based on the cluster CI—not a zero-difference p value: only an upper bound `<= 0.01` supports "no observed practical P benefit".

## Main result

Of 8 source × model-line contrasts, **{no_material}** meet the predeclared practical no-benefit rule, **{material}** support a material P benefit, and **{inconclusive}** are inconclusive. This count is a set of source-contained results, not eight independent chemical replications.

{_markdown_table(table)}

The source-level combinations are:

{_markdown_table(decisions)}

Figure 1 is the primary practical-threshold result; Figure 2 displays the distribution of the actual group-level paired effects rather than only a mean.

![Forest plot of ΔP and 95% group-bootstrap CIs](figures/figure-01-delta-p-forest.png)

![Distribution of three-repeat group-level ΔP values](figures/figure-02-group-delta-distributions.png)

## Allowed conclusion and boundaries

The permitted conclusion is source-specific: for any line with CI upper bound `<= 0.01`, **under that local dataset's recorded A/B/non-P molecular C/numeric C representation and component-holdout protocol, embedding P did not show a benefit reaching the 0.01-MAE practical threshold.** Agreement between two frozen model lines is a robustness check, not independent experimental replication.

This bundle does **not** pool rows, folds, repeats, model lines, or sources. It does not establish chemical causality, product availability in prospective use, unseen-acid generalization, cross-reaction-family generality, source independence, or equivalence. `ord_shields` and `ord_c_n` have only 1 and 4 distinct A+B reactant pairs, respectively, even though adding non-P molecular C produces more runtime groups; numerical condition entry repairs a previous capability block but does not repair that reactant-pair diversity limit. The older pre-registration document was explicitly a draft and did not freeze these later numeric-C runs before scoring; therefore this is a rigorously specified secondary extension, not a confirmatory preregistered replication.

## Claim candidates

- Claim:
  - Source evidence: each line in the main table with CI upper bound `<= 0.01` and the paired integrity tables.
  - Allowed wording: "In this source and frozen model line, P did not show a holdout-MAE improvement of at least 0.01 after the recorded non-P inputs, including numeric C, were supplied."
  - Forbidden stronger wording: "P is generally unnecessary", "P is chemically irrelevant", or any cross-source pooled claim.
  - Uncertainty: local data representation, source lineage, group diversity, and only two model lines.
  - Next check: independently sourced, multi-acid data under a frozen protocol.
  - Decision: keep only with its source and model-line boundary.

- Claim:
  - Source evidence: all 16 result verifier entries and fold-local numeric feature metadata.
  - Allowed wording: "The tested numeric C columns entered both arms fold-locally and were paired across Full/minus-P."
  - Forbidden stronger wording: "All possible condition information was captured".
  - Uncertainty: only declared numeric columns and this strict runtime path were assessed.
  - Next check: configuration-level audit for any future source.
  - Decision: keep as execution-integrity evidence.
"""


def _stats_appendix(result: AnalysisResult) -> str:
    summary = result.line_statistics.copy()
    display = summary.loc[:, ["source_id", "line", "n_unique_non_p_groups", "delta_p_row_weighted", "ci95_low", "ci95_high", "delta_p_group_mean", "delta_p_group_sd", "delta_p_group_median", "delta_p_group_iqr", "n_positive_groups", "n_negative_groups", "n_zero_groups", "wilcoxon_statistic", "p_raw", "p_holm_within_source_two_lines", "rank_biserial", "shapiro_w", "shapiro_p", "normality_status"]].copy()
    for column in display.columns:
        if column not in {"source_id", "line", "n_unique_non_p_groups", "n_positive_groups", "n_negative_groups", "n_zero_groups", "normality_status"}:
            display[column] = display[column].map(lambda value: _format_number(value, 6))
    return f"""# Statistical appendix: numeric-condition P utility

## Inputs and comparison family

This appendix analyses four sources × two frozen robustness lines = 8 comparisons. One comparison family contains the two model lines within one source; Holm adjustment is applied only within that predeclared two-test family. There is no pooled p value across sources, and no cross-source meta-analysis.

## Estimand and unit

`Delta_P = MAE(minus_p) - MAE(full)`. Positive ΔP favours P. For each `(source, model line)`, OOF predictions are exactly paired by sample ID, repeat, and fold; sample-level absolute errors are averaged over the three repeats. The primary ΔP is the normal row-weighted MAE difference. Complete runtime non-P molecular groups are the resampling clusters, so no row, fold, repeat, or model line is counted as an independent observation.

## Intervals and tests

The primary 95% intervals are percentile CIs from 10,000 paired bootstrap resamples of complete groups. In each draw, group-level sums of repeat-averaged delta errors and their row counts are resampled together, and `sum(delta)/sum(rows)` is recomputed. This aligns the CI with row-weighted MAE and the 0.01 threshold without treating rows as independently resampled. A deterministic source/line-specific seed is recorded in [`tables/line_statistics.csv`](tables/line_statistics.csv). The equal-group mean is a descriptive sensitivity summary; it is not used for the practical decision. Two-sided Wilcoxon signed-rank tests use nonzero equal-group effects (`scipy` automatic exact/asymptotic selection), and rank-biserial correlation is signed so positive values favour P. Shapiro–Wilk is descriptive only where 3≤n≤5,000; the paired non-parametric test was selected because group effects can be non-normal, tied, and strongly uneven in row size. No parametric test or independent-fold test was used.

The practical decision is separate from the zero-difference test: a line supports no observed practical P benefit only when `CI_upper <= 0.01`, while `CI_lower > 0.01` would support a material P benefit. Other outcomes are inconclusive.

## Exact group-level inference table

{_markdown_table(display)}

## Descriptive repeat metrics

Each repeat contains the full source population once, but repeats share data and their CV training sets overlap; these values are descriptive checks, not 24 independent observations. Exact values are in [`tables/repeat_metrics.csv`](tables/repeat_metrics.csv).

## Failure and assumption checks

- No task failure, missing fold, incomplete OOF population, label mismatch, split mismatch, or paired runtime-group mismatch was accepted.
- The group-resampling design acknowledges within-group dependence and repeated OOF predictions. It does not establish that groups themselves are chemically independent sources.
- Small group counts and imbalanced group sizes directly limit interval stability. See `max_group_rows_per_repeat` in [`tables/line_statistics.csv`](tables/line_statistics.csv); the figures show the unpooled group-effect distributions.
- This numeric-C extension reused the pre-stated estimand, threshold, bootstrap, Wilcoxon, and Holm procedure, but its earlier general pre-registration was marked draft and did not enumerate these late capability-enabled runs. Treat its outcome as rigorously audited secondary evidence, not confirmation.
"""


def _figure_catalog() -> str:
    return """# Figure catalog: numeric-condition P utility

## Figure 1 — `figure-01-delta-p-forest.pdf`

- Purpose: make the practical `0.01` MAE decision transparent for every source/model comparison.
- Data source: `tables/line_statistics.csv`, calculated from saved paired OOF predictions after group-within-repeat averaging.
- What is plotted: row-MAE ΔP point estimates and 10,000-draw paired cluster-bootstrap 95% CIs. Each draw resamples complete non-P molecular groups and recomputes the row-weighted MAE difference; black vertical line is zero and dashed red line is the 0.01 practical boundary.
- Caption requirements: define ΔP direction, CI construction, group unit, source/model labels, and state that each line is analysed separately.
- Observation to check: whether the full 95% CI lies at/below the practical boundary, rather than whether a p value crosses 0.05.
- Interpretation and decision: a CI upper bound at/below 0.01 permits only the within-source/model statement that no practical P gain was observed; it never supports cross-source pooling or product-causality claims.
- Caveat: two frozen model lines are robustness implementations of one source, not independent chemical repetitions.

## Figure 2 — `figure-02-group-delta-distributions.pdf`

- Purpose: reveal heterogeneity hidden by a single ΔP mean and show the number of independent group units.
- Data source: `tables/group_effects.csv`; each box summarizes actual group effects after averaging three repeats.
- What is plotted: boxplot of per-group ΔP for each source/model line, with `n` printed as unique group count; zero and 0.01 reference lines are shown.
- Caption requirements: state group construction, repeat averaging order, boxplot convention, and that groups—not points/folds—are inference units.
- Observation to check: whether the point estimate/CI in Figure 1 masks wide or skewed group effects, especially for low-diversity sources.
- Interpretation and decision: heterogeneous effects constrain generality and motivate an independent, more diverse source; the plot does not identify a causal mechanism.
- Caveat: group sizes differ and local molecular-component grouping is not a claim of cross-source independence.
"""


def _load_existing_validated_tables(output: Path) -> AnalysisResult:
    """Render a bundle from a completed analysis table set without re-reading OOF.

    This recovery path is intentionally narrow: it accepts only the exact
    default derived directory and requires all tables emitted *after* the
    full 16-task verifier, pairing checks, and bootstrap completed.  It is for
    recovering from report/figure rendering failures, not for skipping an
    integrity audit.
    """
    output = output.resolve()
    _require(output.name == "numeric_conditions_analysis_v1", "render recovery accepts only the dedicated derived bundle")
    tables = output / "tables"
    names = {
        "task_verification": "task_verification.csv",
        "pair_integrity": "pair_integrity.csv",
        "repeat_metrics": "repeat_metrics.csv",
        "group_effects": "group_effects.csv",
        "line_statistics": "line_statistics.csv",
        "source_summary": "source_summary.csv",
        "numeric_evidence": "numeric_condition_evidence.csv",
    }
    paths = {key: tables / value for key, value in names.items()}
    _require(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()), "render recovery requires all completed analysis CSV tables")
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    _require(len(frames["task_verification"]) == 16 and (frames["task_verification"]["task_verifier"] == "passed").all(), "recovery task-verification table is not 16/16 passed")
    _require(len(frames["pair_integrity"]) == 8, "recovery pair-integrity table is not 8 pairs")
    required_pair_flags = [
        "paired_config_identical_except_p",
        "oof_membership_paired",
        "labels_paired",
        "runtime_groups_paired",
        "fold_train_valid_membership_paired",
        "numeric_conditions_entered_all_15_folds",
        "numeric_states_identical_across_arms",
    ]
    def is_true(value: Any) -> bool:
        return value is True or str(value).strip().lower() == "true"

    _require(all(frames["pair_integrity"][field].map(is_true).all() for field in required_pair_flags), "recovery pair-integrity table has a failed gate")
    _require(len(frames["line_statistics"]) == 8 and len(frames["source_summary"]) == 4, "recovery statistics table counts differ")
    provenance = {
        "analysis_status": "passed;_render_recovery_from_completed_validation_tables",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": _sha256_file(Path(__file__)),
        "render_recovery": True,
        "render_recovery_reason": "prior full verification/statistics completed; only report/figure rendering was interrupted",
        "source_table_sha256_before_render": {name: _sha256_file(path) for name, path in paths.items()},
        "n_tasks": 16,
        "n_pairs": 8,
        "n_sources": 4,
        "model_fits_started_by_analysis": 0,
        "metric": "Delta_P = MAE(minus_p) - MAE(full); positive values favour Full/P",
        "independent_unit": "unique runtime non-P molecular component group; sample errors averaged across 3 repeats before clustered inference",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "ci": "paired percentile 95%", "resampling_unit": "complete unique groups", "primary_estimand": "row-weighted delta MAE recomputed as sum(delta error)/sum(rows) per cluster draw"},
        "inference": "two-sided paired Wilcoxon signed-rank on equal-group effects with rank-biserial effect; Holm correction within each source across its two frozen model lines",
        "practical_rule": "support no material P benefit only when the 95% group-bootstrap CI upper bound is <= 0.01 MAE",
        "scope": "within-source engineering utility; no cross-source pooling or general chemical claim",
    }
    return AnalysisResult(provenance=provenance, **frames)


def write_bundle(result: AnalysisResult, output: Path, overwrite: bool) -> None:
    output = output.resolve()
    if output.exists():
        _require(overwrite, f"analysis output already exists: {output}; use --overwrite only to replace this derived bundle")
        _require(output.name == "numeric_conditions_analysis_v1", "refusing to overwrite an unexpected directory")
        shutil.rmtree(output)
    tables = output / "tables"
    report = output / "report"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "task_verification.csv": result.task_verification,
        "pair_integrity.csv": result.pair_integrity,
        "repeat_metrics.csv": result.repeat_metrics,
        "group_effects.csv": result.group_effects,
        "line_statistics.csv": result.line_statistics,
        "source_summary.csv": result.source_summary,
        "numeric_condition_evidence.csv": result.numeric_evidence,
    }.items():
        frame.to_csv(tables / name, index=False)
    _write_markdown_table(
        result.line_statistics,
        tables / "line_statistics.md",
        ["source_id", "line", "n_rows", "n_unique_non_p_groups", "full_mae_row_weighted", "minus_p_mae_row_weighted", "delta_p_row_weighted", "ci95_low", "ci95_high", "delta_p_group_mean", "p_raw", "p_holm_within_source_two_lines", "rank_biserial", "classification"],
    )
    _write_markdown_table(result.source_summary, tables / "source_summary.md", list(result.source_summary.columns))
    _write_figures(result, figures)
    (output / "analysis-report.md").write_text(_analysis_report(result), encoding="utf-8")
    (output / "stats-appendix.md").write_text(_stats_appendix(result), encoding="utf-8")
    (output / "figure-catalog.md").write_text(_figure_catalog(), encoding="utf-8")
    (report / "run_manifest.json").write_text(json.dumps(result.provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report / "reproducibility.md").write_text(
        "# Reproducibility\n\n"
        "Run from repository root in the `yonod` Conda environment. This command only reads completed task outputs; it does not launch models or descriptors.\n\n"
        "```bash\n"
        "source /home/wangzh685/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate yonod\n"
        "python _verify/analyze_pnecessity_numeric_conditions_results.py\n"
        "```\n\n"
        "The command re-runs the read-only completion verifier for all 16 YAMLs and replaces no experiment result. It writes this derived bundle under `docs/`.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="derived docs bundle; never a task result directory")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--overwrite", action="store_true", help="replace only the default-named derived analysis bundle")
    parser.add_argument("--render-existing", action="store_true", help="recover reports/figures from a completed validated table set without re-reading OOF")
    args = parser.parse_args()
    _require(args.bootstrap_draws >= 1000, "bootstrap draws must be at least 1,000")
    _require(not args.render_existing or args.overwrite, "--render-existing requires --overwrite for the dedicated derived bundle")
    result = _load_existing_validated_tables(args.output) if args.render_existing else analyze(bootstrap_draws=args.bootstrap_draws)
    write_bundle(result, args.output, overwrite=args.overwrite)
    compact = result.line_statistics.loc[:, ["source_id", "line", "n_unique_non_p_groups", "delta_p_row_weighted", "ci95_low", "ci95_high", "classification"]]
    print("numeric-condition analysis: passed")
    print(compact.to_csv(index=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
