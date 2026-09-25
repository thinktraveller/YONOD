"""Read-only verifier for the ChemRxiv same-source grouped unseen-A plan.

It checks the immutable prepared population and source split plan, then loads
all twelve schema-2 configurations.  No model, descriptor, result directory,
or task-local copied manifest is written by this verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.splits.manifest import load_external_split_manifest, validate_split_manifest


TASK_NAME = "ablation2_chemrxiv_unseen_amine_splits_v1"
PREPARED = ROOT / "result" / TASK_NAME / "data" / "standardized_chemrxiv_unseen_amine_population.csv"
SOURCE = (
    ROOT
    / "result"
    / "ablation2_chemrxiv_external_data_audit_v1"
    / "data"
    / "standardized_external_population.csv"
)
PREP_REPORT = ROOT / "result" / TASK_NAME / "run_manifest.json"
MANIFEST = ROOT / "result" / TASK_NAME / "manifests" / "unseen_amine_split_manifest.parquet"
INPUTS = {
    "full": ("reactant-2", "reactant-1", "product", "reagent-1", "reagent-2", "solvent-1"),
    "minus_a": ("reactant-1", "product", "reagent-1", "reagent-2", "solvent-1"),
    "minus_p": ("reactant-2", "reactant-1", "reagent-1", "reagent-2", "solvent-1"),
    "minus_a_p": ("reactant-1", "reagent-1", "reagent-2", "solvent-1"),
    "product_conditions": ("product", "reagent-1", "reagent-2", "solvent-1"),
    "conditions_only": ("reagent-1", "reagent-2", "solvent-1"),
}
LINES = {"morgan_rf": ("morgan", "rf"), "mfp_lightgbm": ("mfp", "lightgbm")}
EXPECTED_SOURCE_SHA = "540686d342ed5811e1981decb0cf7091b26092a1339786919e39373b6e9633d2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_name(line: str, input_name: str) -> str:
    return f"ablation2_chemrxiv_unseen_amine_{line}_{input_name}_v1"


def _normalised_membership(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["split_id", "sample_id", "group_id", "group_strategy", "repeat", "fold", "role", "seed", "source_row_index", "dataset_sha256", "grouping_params_json", "population_id", "split_hash"]
    return frame.loc[:, columns].sort_values(["repeat", "fold", "role", "sample_id"], kind="mergesort").reset_index(drop=True)


def _fail(message: str) -> None:
    raise AssertionError(message)


def _verify_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in (PREPARED, SOURCE, PREP_REPORT, MANIFEST):
        if not path.is_file():
            _fail(f"required preparation artifact is missing: {path}")
    source_sha = _sha256(SOURCE)
    if source_sha != EXPECTED_SOURCE_SHA:
        _fail("the standardized ChemRxiv source SHA-256 differs from the preregistered frozen input")
    report = json.loads(PREP_REPORT.read_text(encoding="utf-8"))
    prepared = pd.read_csv(PREPARED, keep_default_na=False, encoding="utf-8")
    source = pd.read_csv(SOURCE, keep_default_na=False, encoding="utf-8")
    if len(prepared) != 957 or prepared["sample_id"].duplicated().any():
        _fail("prepared population is not 957 rows with unique sample IDs")
    required = ["sample_id", "yield", "amine_structure_key", "acid_structure_key", "reactant-1", "reactant-2", "product", "reagent-1", "reagent-2", "solvent-1"]
    if [column for column in required if column not in prepared.columns]:
        _fail("prepared population is missing a declared model or grouping column")
    source_lookup = source.set_index("sample_id", verify_integrity=True)
    prepared_lookup = prepared.set_index("sample_id", verify_integrity=True)
    if set(prepared_lookup.index) != set(source_lookup.index):
        _fail("prepared population sample IDs differ from frozen standardized ChemRxiv source")
    preserved = ["yield", "amine_structure_key", "acid_structure_key", "reactant-1", "reactant-2", "product", "reagent-1", "reagent-2", "solvent-1"]
    for column in preserved:
        if not prepared_lookup.loc[source_lookup.index, column].astype(str).equals(source_lookup[column].astype(str)):
            _fail(f"prepared population changed source column: {column}")
    sizes = prepared["amine_structure_key"].astype(str).value_counts()
    if len(sizes) != 10 or sorted(sizes.tolist()) != [94, 95] + [96] * 8:
        _fail(f"unexpected amine group cardinalities: {sizes.to_dict()}")
    if prepared["acid_structure_key"].astype(str).nunique() != 1:
        _fail("prepared population no longer has the stated one-acid limitation")
    prepared_sha = _sha256(PREPARED)
    if report["prepared_population"]["sha256"] != prepared_sha:
        _fail("prepared population SHA-256 does not match preparation report")
    if report["source_population"]["sha256"] != source_sha:
        _fail("source population SHA-256 does not match preparation report")
    return prepared, report


def _verify_manifest(prepared: pd.DataFrame, report: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(MANIFEST)
    validate_split_manifest(manifest, n_splits=5, n_repeats=3)
    prepared_sha = _sha256(PREPARED)
    if set(manifest["dataset_sha256"].astype(str)) != {prepared_sha}:
        _fail("source split manifest does not bind to the prepared population SHA-256")
    if _sha256(MANIFEST) != report["split_manifest"]["sha256"]:
        _fail("source split manifest SHA-256 does not match preparation report")
    if len(manifest) != 957 * 5 * 3 or len(manifest[["repeat", "fold"]].drop_duplicates()) != 15:
        _fail("source split manifest does not contain 957 x 5 x 3 membership rows")
    lookup = prepared.set_index("sample_id", verify_integrity=True)
    fold_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for repeat, repeat_part in manifest.groupby("repeat", sort=True):
        per_group_folds: dict[str, list[int]] = {group: [] for group in lookup["amine_structure_key"].astype(str).unique()}
        for fold, part in repeat_part.groupby("fold", sort=True):
            train_ids = part.loc[part["role"].eq("train"), "sample_id"].astype(str).tolist()
            valid_ids = part.loc[part["role"].eq("valid"), "sample_id"].astype(str).tolist()
            train_groups = set(lookup.loc[train_ids, "amine_structure_key"].astype(str))
            valid_groups = set(lookup.loc[valid_ids, "amine_structure_key"].astype(str))
            overlap = train_groups.intersection(valid_groups)
            if overlap:
                _fail(f"A/amine leakage in repeat={repeat}, fold={fold}: {sorted(overlap)}")
            if len(valid_groups) != 2 or len(train_groups) != 8:
                _fail(f"unexpected A-group count in repeat={repeat}, fold={fold}")
            direct_manifest_groups = set(part.loc[part["role"].eq("valid"), "group_id"].astype(str))
            expected_manifest_groups = {f"precomputed:{group}" for group in valid_groups}
            if direct_manifest_groups != expected_manifest_groups:
                _fail(f"manifest group IDs do not encode the held-out amine groups in repeat={repeat}, fold={fold}")
            for group in valid_groups:
                per_group_folds[group].append(int(fold))
            fold_rows.append(
                {"repeat": int(repeat), "fold": int(fold), "n_train": len(train_ids), "n_valid": len(valid_ids), "n_valid_groups": len(valid_groups)}
            )
        if any(len(folds) != 1 for folds in per_group_folds.values()):
            _fail(f"not every A/amine group appears in validation exactly once in repeat={repeat}")
        valid_counts = repeat_part.loc[repeat_part["role"].eq("valid"), "sample_id"].value_counts()
        if len(valid_counts) != 957 or not valid_counts.eq(1).all():
            _fail(f"not every sample appears in validation exactly once in repeat={repeat}")
        repeat_rows.append({"repeat": int(repeat), "n_groups": len(per_group_folds), "valid_once_groups": sum(len(folds) == 1 for folds in per_group_folds.values())})
    return manifest, {"folds": fold_rows, "repeats": repeat_rows}


def _verify_configurations(prepared: pd.DataFrame, source_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    expected_dataset = PREPARED.resolve()
    expected_manifest = MANIFEST.resolve()
    expected_source_membership = _normalised_membership(source_manifest)
    observed_roots: set[Path] = set()
    config_rows: list[dict[str, Any]] = []
    for line, (descriptor_name, model_name) in LINES.items():
        for input_name, expected_columns in INPUTS.items():
            task_name = _task_name(line, input_name)
            path = ROOT / "config" / f"{task_name}.yaml"
            if not path.is_file():
                _fail(f"missing required schema-2 run configuration: {path}")
            config = BenchmarkConfig.from_file(path)
            contract = create_benchmark_contract(config)
            if config.dataset_path != expected_dataset or config.split_manifest_path != expected_manifest:
                _fail(f"{task_name}: dataset or external manifest path deviates from the common plan")
            if config.models != (model_name,) or config.descriptors != (f"{descriptor_name}_{input_name}",):
                _fail(f"{task_name}: descriptor/model identity is incorrect")
            feature = config.feature_sets[0]
            if tuple(feature["component_cols"]) != expected_columns:
                _fail(f"{task_name}: ablation input columns differ from preregistration")
            if config.cv != {"n_splits": 5, "n_repeats": 3, "seed": 20260918}:
                _fail(f"{task_name}: CV contract differs from 5-fold x 3-repeat seed 20260918")
            if config.grouping.get("strategy") != "precomputed_column" or config.grouping.get("group_column") != "amine_structure_key":
                _fail(f"{task_name}: not bound to precomputed A/amine groups")
            if config.outputs_root.name != task_name or config.artifact_output_dir != config.outputs_root / "feature":
                _fail(f"{task_name}: isolated output-root/feature contract is broken")
            if config.outputs_root in observed_roots:
                _fail(f"{task_name}: output root is shared")
            observed_roots.add(config.outputs_root)
            if config.model_configs[model_name]["estimator"].get("n_jobs") != 19:
                _fail(f"{task_name}: model n_jobs is not 19")
            imported = load_external_split_manifest(
                expected_manifest,
                sample_ids=prepared["sample_id"].astype(str).tolist(),
                dataset_sha256=contract.dataset_sha256,
                n_splits=5,
                n_repeats=3,
                run_id=contract.run_id,
            )
            if not _normalised_membership(imported).equals(expected_source_membership):
                _fail(f"{task_name}: imported manifest membership differs from paired source plan")
            config_rows.append(
                {
                    "task": task_name,
                    "config": str(path.relative_to(ROOT)),
                    "config_sha256": _sha256(path),
                    "run_id": contract.run_id,
                    "config_hash": contract.config_hash,
                    "model": model_name,
                    "input": input_name,
                }
            )
    if len(config_rows) != 12 or len({row["config_hash"] for row in config_rows}) != 12:
        _fail("expected twelve unique configuration identities")
    return config_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepared, report = _verify_population()
    source_manifest, split_audit = _verify_manifest(prepared, report)
    configs = _verify_configurations(prepared, source_manifest)
    summary = {
        "status": "passed",
        "writes": 0,
        "prepared_population": {
            "n_samples": int(len(prepared)),
            "n_amine_groups": int(prepared["amine_structure_key"].astype(str).nunique()),
            "n_acid_groups": int(prepared["acid_structure_key"].astype(str).nunique()),
            "sha256": _sha256(PREPARED),
        },
        "source_manifest": {
            "n_rows": int(len(source_manifest)),
            "n_folds": len(split_audit["folds"]),
            "n_repeats": len(split_audit["repeats"]),
            "sha256": _sha256(MANIFEST),
            "folds": split_audit["folds"],
            "repeats": split_audit["repeats"],
        },
        "configurations": configs,
        "static_contracts": {
            "zero_within_fold_amine_overlap": True,
            "every_amine_group_once_in_validation_per_repeat": True,
            "paired_sample_fold_membership_identical_across_12_configs": True,
            "isolated_result_roots": True,
            "html_and_markdown_declared": True,
            "n_jobs_per_model": 19,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
