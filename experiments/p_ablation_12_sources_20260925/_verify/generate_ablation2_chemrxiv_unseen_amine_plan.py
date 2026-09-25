"""Prepare the locked ChemRxiv grouped-unseen-amine benchmark plan.

This is deliberately a preparation-only utility: it copies the already
standardized ChemRxiv population into an immutable task-local snapshot,
constructs one 5-fold x 3-repeat amine-group manifest, and writes the 12
schema-2 run configurations.  It never constructs descriptors or fits a
model.  Existing non-identical artifacts are rejected rather than replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.splits.manifest import create_split_manifest, validate_split_manifest


SOURCE_POPULATION = (
    ROOT
    / "result"
    / "ablation2_chemrxiv_external_data_audit_v1"
    / "data"
    / "standardized_external_population.csv"
)
SOURCE_AUDIT = ROOT / "result" / "ablation2_chemrxiv_external_data_audit_v1" / "run_manifest.json"
TASK_NAME = "ablation2_chemrxiv_unseen_amine_splits_v1"
PREPARED_COLUMNS = (
    "sample_id",
    "source_reaction_key",
    "source_row_index",
    "reactant-1",
    "reactant-2",
    "reagent-1",
    "reagent-2",
    "solvent-1",
    "product",
    "yield",
    "amine_structure_key",
    "acid_structure_key",
    "product_structure_key",
)
INPUTS = {
    "full": ("reactant-2", "reactant-1", "product", "reagent-1", "reagent-2", "solvent-1"),
    "minus_a": ("reactant-1", "product", "reagent-1", "reagent-2", "solvent-1"),
    "minus_p": ("reactant-2", "reactant-1", "reagent-1", "reagent-2", "solvent-1"),
    "minus_a_p": ("reactant-1", "reagent-1", "reagent-2", "solvent-1"),
    "product_conditions": ("product", "reagent-1", "reagent-2", "solvent-1"),
    "conditions_only": ("reagent-1", "reagent-2", "solvent-1"),
}
MODEL_LINES = {
    "morgan_rf": {
        "descriptor": "morgan",
        "model": "rf",
        "params": {
            "rf": {
                "estimator": {
                    "n_estimators": 300,
                    "max_features": 1.0,
                    "min_samples_leaf": 1,
                    "random_state": 20260918,
                    "n_jobs": 19,
                }
            }
        },
    },
    "mfp_lightgbm": {
        "descriptor": "mfp",
        "model": "lightgbm",
        "params": {
            "lightgbm": {
                "estimator": {
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 1,
                    "random_state": 20260918,
                    "n_jobs": 19,
                }
            }
        },
    },
}


@dataclass(frozen=True)
class _SplitConfig:
    """The minimal immutable input consumed by ``create_split_manifest``."""

    frame: pd.DataFrame
    sample_id_col: str
    grouping: dict[str, Any]
    cv: dict[str, int]
    raw: dict[str, str]
    smiles_cols: tuple[str, ...]

    def validate_dataset(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass(frozen=True)
class _SplitContract:
    config: _SplitConfig
    run_id: str
    dataset_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Write only a new artifact, or accept exactly identical existing bytes."""
    if path.exists():
        if path.read_bytes() == data:
            return
        raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.equals(frame):
            return
        raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        checked = pd.read_parquet(temporary)
        if len(checked) != len(frame) or list(checked.columns) != list(frame.columns):
            raise RuntimeError(f"parquet staging validation failed: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, encoding="utf-8", lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _task_name(line: str, input_name: str) -> str:
    return f"ablation2_chemrxiv_unseen_amine_{line}_{input_name}_v1"


def _config_payload(line: str, input_name: str) -> dict[str, Any]:
    definition = MODEL_LINES[line]
    task_name = _task_name(line, input_name)
    descriptor: dict[str, Any] = {
        "id": f"{definition['descriptor']}_{input_name}",
        "descriptor": definition["descriptor"],
        "lifecycle": "static_descriptor",
        "mode": "concat",
        "columns": list(INPUTS[input_name]),
    }
    if definition["descriptor"] == "mfp":
        descriptor["params"] = {"radius": 3, "fp_size": 1024, "profile": "standard"}
    return {
        "schema_version": "2.0",
        "project_name": task_name,
        "stage": "benchmark",
        "dataset": {
            "path": f"./result/{TASK_NAME}/data/standardized_chemrxiv_unseen_amine_population.csv",
            "sample_id_col": "sample_id",
            "column_roles": {
                "label": "yield",
                "reactants": ["reactant-2", "reactant-1"],
                "products": ["product"],
                "others": ["reagent-1", "reagent-2", "solvent-1"],
                "conditions": [],
                "categoricals": [],
            },
        },
        "descriptors": [descriptor],
        "artifacts": {"output_dir": f"./result/{task_name}/feature"},
        "models": [definition["model"]],
        "model_params": definition["params"],
        "evaluation": {
            "protocol": "manifest_outer_cv",
            "n_splits": 5,
            "n_repeats": 3,
            "seed": 20260918,
            "grouping": {
                "strategy": "precomputed_column",
                "group_column": "amine_structure_key",
                "protocol_label": "chemrxiv_same_source_grouped_unseen_amine",
                "max_group_fraction": 0.8,
            },
            "split_manifest": f"./result/{TASK_NAME}/manifests/unseen_amine_split_manifest.parquet",
        },
        "outputs": {"root": f"./result/{task_name}", "report_formats": ["html", "markdown"]},
        "benchmark": {
            "task_state": {"backend": "sqlite", "resumable": True},
            "population_id": "ablation2_chemrxiv_same_source_unseen_amine_v1",
            "dataset_id": "ablation2_chemrxiv_standardized_external_population",
        },
        "metadata": {
            "notes": (
                "Preregistered same-source grouped unseen-A validation. The source is the "
                "fixed ChemRxiv standardized external population; reactant-2 is A, reactant-1 "
                "is B, product is P, and C is reagent-1/reagent-2/solvent-1. No model has "
                "been started by plan generation. This is a within-source A extrapolation "
                "test and cannot establish unseen-acid or cross-source generalization."
            )
        },
    }


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False).encode("utf-8")


def _audit_folds(frame: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = frame.set_index("sample_id", verify_integrity=True)
    fold_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for repeat, repeat_part in manifest.groupby("repeat", sort=True):
        assignments: list[dict[str, Any]] = []
        for fold, part in repeat_part.groupby("fold", sort=True):
            train_ids = part.loc[part["role"].eq("train"), "sample_id"].astype(str).tolist()
            valid_ids = part.loc[part["role"].eq("valid"), "sample_id"].astype(str).tolist()
            train_groups = set(index.loc[train_ids, "amine_structure_key"].astype(str))
            valid_groups = set(index.loc[valid_ids, "amine_structure_key"].astype(str))
            if train_groups.intersection(valid_groups):
                raise RuntimeError(f"group leakage in repeat={repeat}, fold={fold}")
            if len(valid_groups) != 2:
                raise RuntimeError(f"expected two held-out amines in repeat={repeat}, fold={fold}")
            fold_rows.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "n_train": len(train_ids),
                    "n_valid": len(valid_ids),
                    "n_train_amine_groups": len(train_groups),
                    "n_valid_amine_groups": len(valid_groups),
                    "amine_group_intersection": 0,
                    "valid_sample_ids_sha256": _sha256_bytes(
                        _canonical_json(sorted(valid_ids)).encode("utf-8")
                    ),
                    "valid_amine_groups_sha256": _sha256_bytes(
                        _canonical_json(sorted(valid_groups)).encode("utf-8")
                    ),
                }
            )
            for group in sorted(valid_groups):
                group_n = int(index["amine_structure_key"].astype(str).eq(group).sum())
                schedule_rows.append(
                    {"repeat": int(repeat), "fold": int(fold), "amine_structure_key": group, "n_valid": group_n}
                )
                assignments.append({"amine_structure_key": group, "fold": int(fold)})
        schedule = pd.DataFrame.from_records(assignments)
        if schedule["amine_structure_key"].nunique() != 10 or not schedule["amine_structure_key"].value_counts().eq(1).all():
            raise RuntimeError(f"repeat={repeat} does not hold out every amine group exactly once")
        signature_rows.append(
            {
                "repeat": int(repeat),
                "assignment_sha256": _sha256_bytes(_canonical_json(assignments).encode("utf-8")),
            }
        )
    return (
        pd.DataFrame.from_records(fold_rows),
        pd.DataFrame.from_records(schedule_rows),
        pd.DataFrame.from_records(signature_rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_POPULATION)
    parser.add_argument("--task-name", default=TASK_NAME)
    args = parser.parse_args()
    if args.task_name != TASK_NAME:
        raise ValueError(f"this preregistered generator only permits task name {TASK_NAME!r}")
    source = args.source.resolve()
    if source != SOURCE_POPULATION.resolve():
        raise ValueError(f"this preregistered generator only permits source {SOURCE_POPULATION}")
    if not source.is_file() or not SOURCE_AUDIT.is_file():
        raise FileNotFoundError("standardized ChemRxiv population or its audit manifest is missing")

    source_frame = pd.read_csv(source, keep_default_na=False, encoding="utf-8")
    missing = [column for column in PREPARED_COLUMNS if column not in source_frame.columns]
    if missing:
        raise ValueError(f"source standardized population lacks required columns: {missing}")
    prepared = source_frame.loc[:, PREPARED_COLUMNS].copy()
    if len(prepared) != 957 or prepared["sample_id"].duplicated().any():
        raise ValueError("expected exactly 957 unique standardized ChemRxiv rows")
    group_sizes = prepared["amine_structure_key"].astype(str).value_counts().sort_index()
    if len(group_sizes) != 10 or sorted(group_sizes.tolist()) != [94, 95] + [96] * 8:
        raise ValueError(f"unexpected amine-group cardinalities: {group_sizes.to_dict()}")
    if prepared["acid_structure_key"].astype(str).nunique() != 1:
        raise ValueError("same-source grouped unseen-A plan requires the preregistered single-acid population")
    if prepared.loc[:, ["reactant-1", "reactant-2", "reagent-1", "reagent-2", "solvent-1", "product"]].eq("").any().any():
        raise ValueError("required molecular columns contain blank values")

    output = ROOT / "result" / TASK_NAME
    data_path = output / "data" / "standardized_chemrxiv_unseen_amine_population.csv"
    _atomic_bytes(data_path, _csv_bytes(prepared))
    dataset_sha = _sha256(data_path)
    source_sha = _sha256(source)
    audit_sha = _sha256(SOURCE_AUDIT)
    if source_sha != "540686d342ed5811e1981decb0cf7091b26092a1339786919e39373b6e9633d2":
        raise ValueError("the frozen standardized ChemRxiv source checksum is not the preregistered value")

    split_config = _SplitConfig(
        frame=prepared,
        sample_id_col="sample_id",
        grouping={
            "strategy": "precomputed_column",
            "group_column": "amine_structure_key",
            "protocol_label": "chemrxiv_same_source_grouped_unseen_amine",
            "max_group_fraction": 0.8,
        },
        cv={"n_splits": 5, "n_repeats": 3, "seed": 20260918},
        raw={"population_id": "ablation2_chemrxiv_same_source_unseen_amine_v1"},
        smiles_cols=("reactant-2", "reactant-1", "product", "reagent-1", "reagent-2", "solvent-1"),
    )
    split_contract = _SplitContract(
        config=split_config,
        run_id="ablation2-chemrxiv-unseen-amine-split-v1",
        dataset_sha256=dataset_sha,
    )
    manifest = create_split_manifest(split_contract)
    validate_split_manifest(manifest, n_splits=5, n_repeats=3)
    manifest_path = output / "manifests" / "unseen_amine_split_manifest.parquet"
    _atomic_parquet(manifest, manifest_path)
    fold_audit, group_schedule, repeat_signatures = _audit_folds(prepared, manifest)
    _atomic_bytes(output / "audits" / "fold_membership_audit.csv", _csv_bytes(fold_audit))
    _atomic_bytes(output / "audits" / "group_validation_schedule.csv", _csv_bytes(group_schedule))
    _atomic_bytes(output / "audits" / "repeat_partition_signatures.csv", _csv_bytes(repeat_signatures))

    config_hashes: dict[str, str] = {}
    for line in MODEL_LINES:
        for input_name in INPUTS:
            task_name = _task_name(line, input_name)
            config_path = ROOT / "config" / f"{task_name}.yaml"
            config_bytes = _yaml_bytes(_config_payload(line, input_name))
            _atomic_bytes(config_path, config_bytes)
            config_hashes[task_name] = _sha256(config_path)

    report = {
        "task_kind": "same-source grouped unseen-A split preparation; no descriptor or model fitting",
        "source_population": {
            "path": str(source.relative_to(ROOT)),
            "sha256": source_sha,
            "audit_manifest": str(SOURCE_AUDIT.relative_to(ROOT)),
            "audit_manifest_sha256": audit_sha,
        },
        "prepared_population": {
            "path": str(data_path.relative_to(ROOT)),
            "sha256": dataset_sha,
            "n_samples": int(len(prepared)),
            "n_amine_groups": int(len(group_sizes)),
            "amine_group_sizes": {key: int(value) for key, value in group_sizes.items()},
            "n_acid_groups": int(prepared["acid_structure_key"].astype(str).nunique()),
        },
        "split_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _sha256(manifest_path),
            "split_id": str(manifest["split_id"].iloc[0]),
            "source_run_id": str(manifest["run_id"].iloc[0]),
            "n_rows": int(len(manifest)),
            "n_folds": int(len(fold_audit)),
            "n_splits": 5,
            "n_repeats": 3,
            "seed": 20260918,
            "repeat_assignment_hashes": repeat_signatures.to_dict(orient="records"),
        },
        "formal_tasks": {
            "count": len(config_hashes),
            "config_sha256": config_hashes,
            "models_per_task": 1,
            "reports": ["html", "markdown"],
            "cpu_budget_per_task": 19,
        },
        "static_contracts": [
            "each fold has zero A/amine-group overlap between train and validation",
            "each repeat holds every one of ten A/amine groups out exactly once",
            "all twelve configurations use the same prepared sample population and source split manifest",
            "each configuration has an isolated result root and an outputs.root/feature artifact directory",
        ],
    }
    _atomic_bytes(
        output / "run_manifest.json",
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
