"""Generate strict paired YAML for the four former numeric-C blockers.

This generator is intentionally model-free.  It binds its declarations to the
already prepared, immutable source CSVs and never mutates their historical
manifest.  The strict runner consumes molecular columns only when they appear
in ``column_roles.reactants``; therefore every non-product molecular component
(A, B and molecular C) is explicitly listed there.  Numeric C remains in the
declared fold-local numeric-condition block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARENT_PREP_ROOT = ROOT / "result" / "pnecessity_utility_prepare_v1"
PREP_ROOT = ROOT / "result" / "pnecessity_numeric_conditions_prepare_v1"
PREFIX = "pnecessity_numeric_conditions_"
SOURCE_ORDER = (
    "ord_roche_borylation",
    "ord_nicolit",
    "ord_shields",
    "ord_c_n",
)
ARMS = ("full", "minus_p")
FROZEN_LINES: dict[str, dict[str, Any]] = {
    "morgan_rf": {
        "model": "rf",
        "descriptor": "morgan",
        "descriptor_params": {},
        "model_params": {
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
        "model": "lightgbm",
        "descriptor": "mfp",
        "descriptor_params": {"radius": 3, "fp_size": 1024, "profile": "standard"},
        "model_params": {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_yaml(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False), path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", path)


def _task_name(source_id: str, line_id: str, arm: str) -> str:
    return f"{PREFIX}{source_id}_{line_id}_{arm}_v1"


def _numeric_entry(column: str) -> dict[str, Any]:
    if column == "c_temperature_c":
        unit = {"source": "degC", "target": "degC", "scale": 1.0, "offset": 0.0}
    elif column == "c_reaction_time_s":
        unit = {"source": "s", "target": "s", "scale": 1.0, "offset": 0.0}
    else:
        unit = {"source": "dimensionless", "target": "dimensionless", "scale": 1.0, "offset": 0.0}
    return {
        "name": column,
        "missing": {"strategy": "error"},
        "scaling": "standard",
        "unit": unit,
    }


def _build_config(source: Mapping[str, Any], line_id: str, arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    line = FROZEN_LINES[line_id]
    prepared = source["prepared_columns"]
    molecular_inputs = ["a_smiles", "b_smiles", *prepared["molecular_c"]]
    numeric_inputs = list(prepared["numeric_c"])
    products = ["p_smiles"] if arm == "full" else []
    model_molecules = [*molecular_inputs, *products]
    task = _task_name(str(source["source_id"]), line_id, arm)
    descriptor: dict[str, Any] = {
        "id": f"{line_id}_{arm}",
        "descriptor": line["descriptor"],
        "lifecycle": "static_descriptor",
        "mode": "concat",
        "columns": list(model_molecules),
    }
    if line["descriptor_params"]:
        descriptor["params"] = dict(line["descriptor_params"])
    grouping = {
        "strategy": "component_holdout",
        "component_cols": molecular_inputs,
        "protocol_label": "pnecessity_non_p_molecular_component_holdout_numeric_v1",
        "max_group_fraction": 0.8,
    }
    return {
        "schema_version": "2.0",
        "project_name": task,
        "stage": "benchmark",
        "dataset": {
            "path": "./" + str(source["prepared_path"]),
            "sample_id_col": "sample_id",
            "column_roles": {
                # The strict adapter materialises only reactant-role molecular
                # columns.  This role is an execution mapping, not a claim that
                # a catalyst or solvent is chemically a reactant.
                "label": "yield",
                # ``BenchmarkConfig`` materialises descriptor columns only
                # from reactants.  In the Full arm p_smiles is deliberately
                # duplicated here and in products: the former is an explicit
                # strict-execution mapping, the latter preserves its source
                # role for audit and reporting.
                "reactants": list(model_molecules),
                "products": list(products),
                "others": [],
                "conditions": numeric_inputs,
                "categoricals": [],
            },
            "numeric_conditions": {
                "defaults": {"missing": {"strategy": "error"}, "scaling": "standard"},
                "columns": {column: _numeric_entry(column) for column in numeric_inputs},
            },
        },
        "descriptors": [descriptor],
        "artifacts": {"output_dir": f"./result/{task}/feature"},
        "models": [line["model"]],
        "model_params": line["model_params"],
        "evaluation": {
            "protocol": "manifest_outer_cv",
            "n_splits": 5,
            "n_repeats": 3,
            "seed": 20260918,
            "grouping": {
                **grouping,
                "component_cols": list(molecular_inputs),
            },
        },
        "outputs": {"root": f"./result/{task}", "report_formats": ["html", "markdown"]},
        "benchmark": {
            "task_state": {"backend": "sqlite", "resumable": True},
            "population_id": f"pnecessity-numeric-{source['source_id']}-{source['prepared_sha256'][:16]}",
            "dataset_id": str(source["source_id"]),
        },
        "metadata": {
            "comparison": "within_source_paired_utility",
            "input_arm": arm,
            "parent_preparation_manifest": "./result/pnecessity_utility_prepare_v1/source_manifest.json",
            "prepared_data_sha256": source["prepared_sha256"],
            "raw_data_sha256": source["raw_sha256"],
            "molecular_c_execution_policy": "A/B and every non-product molecular C are materialised as strict molecular inputs; their chemical source roles remain in the parent preparation manifest",
            "numeric_c_execution_policy": "all recorded numeric C are fold-locally parsed, unit-audited, and standard-scaled; full and minus-P use the same numeric contract",
            "split_policy": "strict runner deterministically creates the same non-product molecular component-holdout membership for each paired arm; numeric variants within an identical molecular tuple stay together conservatively",
            "limitation": "within-source result only; do not pool rows, folds, repeats, or effect estimates across sources",
            **({"product_embedding_execution_mapping": "p_smiles remains the declared product and is additionally materialised as a strict molecular input only in the Full arm"} if arm == "full" else {}),
        },
    }


def generate(*, parent_prep_root: Path = PARENT_PREP_ROOT, prep_root: Path = PREP_ROOT, config_dir: Path = ROOT / "config") -> dict[str, Any]:
    source_manifest_path = parent_prep_root / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing parent preparation manifest: {source_manifest_path}")
    parent = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_by_id = {str(source["source_id"]): source for source in parent.get("sources", [])}
    missing = [source_id for source_id in SOURCE_ORDER if source_id not in source_by_id]
    if missing:
        raise ValueError(f"parent manifest is missing selected numeric-C sources: {missing}")

    selected = []
    generated = []
    for source_id in SOURCE_ORDER:
        source = source_by_id[source_id]
        numeric = list(source["prepared_columns"]["numeric_c"])
        if not numeric or not source["conditions"]["variable_numeric_c"]:
            raise ValueError(f"{source_id}: selected source no longer has the expected variable numeric C")
        prepared_path = ROOT / str(source["prepared_path"])
        raw_path = ROOT / str(source["raw_path"])
        if not prepared_path.is_file() or _sha256(prepared_path) != source["prepared_sha256"]:
            raise ValueError(f"{source_id}: prepared CSV is missing or hash-diverged")
        if not raw_path.is_file() or _sha256(raw_path) != source["raw_sha256"]:
            raise ValueError(f"{source_id}: raw source is missing or hash-diverged")
        selected.append({
            "source_id": source_id,
            "parent_historical_status": source["config_status"],
            "prepared_path": source["prepared_path"],
            "prepared_sha256": source["prepared_sha256"],
            "raw_path": source["raw_path"],
            "raw_sha256": source["raw_sha256"],
            "prepared_rows": source["prepared_rows"],
            "molecular_c": source["prepared_columns"]["molecular_c"],
            "numeric_c": numeric,
            "variable_numeric_c": source["conditions"]["variable_numeric_c"],
        })
        for line_id in FROZEN_LINES:
            for arm in ARMS:
                task = _task_name(source_id, line_id, arm)
                path = config_dir / f"{task}.yaml"
                _atomic_yaml(_build_config(source, line_id, arm), path)
                generated.append({
                    "source_id": source_id,
                    "line_id": line_id,
                    "arm": arm,
                    "task_name": task,
                    "config_path": str(path.relative_to(ROOT)),
                })

    result = {
        "generator": "_verify/generate_pnecessity_numeric_conditions_configs.py",
        "parent_preparation_manifest": str(source_manifest_path.relative_to(ROOT)),
        "parent_preparation_manifest_sha256": _sha256(source_manifest_path),
        "protocol": {
            "n_splits": 5,
            "n_repeats": 3,
            "seed": 20260918,
            "grouping": "component_holdout over all non-product molecular inputs",
            "numeric_conditions": "fold-local standard scaling with declared units",
        },
        "n_selected_sources": len(selected),
        "n_generated_configs": len(generated),
        "expected_generated_configs": len(SOURCE_ORDER) * len(FROZEN_LINES) * len(ARMS),
        "selected_sources": selected,
        "generated": generated,
        "excluded_capability_gap": {
            "source_id": "liao_ir_oh_insertion",
            "reason": "strict runtime cannot faithfully represent its mandatory published 412/235 model/external fixed split",
        },
    }
    _atomic_json(result, prep_root / "generated_config_manifest.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the four-source numeric-condition paired YAML matrix; never starts a model.")
    parser.add_argument("--parent-prep-root", type=Path, default=PARENT_PREP_ROOT)
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args()
    report = generate(
        parent_prep_root=args.parent_prep_root.resolve(),
        prep_root=args.prep_root.resolve(),
        config_dir=args.config_dir.resolve(),
    )
    print(json.dumps({key: report[key] for key in ("n_selected_sources", "n_generated_configs", "expected_generated_configs", "excluded_capability_gap")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
