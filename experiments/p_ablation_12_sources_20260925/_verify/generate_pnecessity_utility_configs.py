"""Generate only schema-2 product-utility benchmark declarations.

This generator does not invoke YONOD, descriptors, training, a queue, or a
runner.  It turns the prepared-source manifest into one Full/minus-P pair per
frozen model line only when every variable recorded C field is representable by
the current strict benchmark feature path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREP_ROOT = ROOT / "result" / "pnecessity_utility_prepare_v1"
CONFIG_PREFIX = "pnecessity_utility_"

FROZEN_LINES: dict[str, dict[str, Any]] = {
    "morgan_rf": {
        "model": "rf",
        "descriptor": "morgan",
        "descriptor_params": {},
        "model_params": {
            "rf": {"estimator": {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1, "random_state": 20260918, "n_jobs": 19}},
        },
    },
    "mfp_lightgbm": {
        "model": "lightgbm",
        "descriptor": "mfp",
        "descriptor_params": {"radius": 3, "fp_size": 1024, "profile": "standard"},
        "model_params": {
            "lightgbm": {"estimator": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 1, "random_state": 20260918, "n_jobs": 19}},
        },
    },
}


def _atomic_yaml(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _task_name(source_id: str, line_id: str, arm: str) -> str:
    return f"{CONFIG_PREFIX}{source_id}_{line_id}_{arm}_v1"


def _build_config(source: Mapping[str, Any], line_id: str, arm: str) -> dict[str, Any]:
    if arm not in {"full", "minus_p"}:
        raise ValueError(f"unknown arm: {arm}")
    line = FROZEN_LINES[line_id]
    prepared = source["prepared_columns"]
    non_p_molecules = ["a_smiles", "b_smiles", *prepared["molecular_c"]]
    products = ["p_smiles"] if arm == "full" else []
    descriptor_columns = [*non_p_molecules, *products]
    task = _task_name(str(source["source_id"]), line_id, arm)
    descriptor: dict[str, Any] = {
        "id": f"{line_id}_{arm}", "descriptor": line["descriptor"], "lifecycle": "static_descriptor", "mode": "concat", "columns": descriptor_columns,
    }
    if line["descriptor_params"]:
        descriptor["params"] = dict(line["descriptor_params"])
    fixed_numeric = list(source["conditions"]["fixed_or_unrecorded_numeric_c"])
    numeric_cols = list(prepared["numeric_c"])
    fixed_numeric_cols = ["c_" + item.replace("-", "_").lower() for item in fixed_numeric]
    if sorted(fixed_numeric_cols) != sorted(numeric_cols):
        raise ValueError(
            f"{source['source_id']}: config generator received non-fixed numeric fields; blocked source must not reach config generation"
        )
    config: dict[str, Any] = {
        "schema_version": "2.0",
        "project_name": task,
        "stage": "benchmark",
        "dataset": {
            "path": "./" + str(source["prepared_path"]),
            "sample_id_col": "sample_id",
            "column_roles": {
                "label": "yield", "reactants": ["a_smiles", "b_smiles"], "products": products,
                "others": list(prepared["molecular_c"]), "conditions": numeric_cols, "categoricals": [],
            },
        },
        "descriptors": [descriptor],
        "artifacts": {"output_dir": f"./result/{task}/feature"},
        "models": [line["model"]],
        "model_params": line["model_params"],
        "evaluation": {
            "protocol": "manifest_outer_cv", "n_splits": 5, "n_repeats": 3, "seed": 20260918,
            "grouping": {"strategy": "precomputed_column", "group_column": "non_p_input_group_id", "protocol_label": "pnecessity_within_source_non_p_input", "max_group_fraction": 0.8},
            "split_manifest": "./" + str(source["split_manifest"]["path"]),
        },
        "outputs": {"root": f"./result/{task}", "report_formats": ["html", "markdown"]},
        "benchmark": {
            "task_state": {"backend": "sqlite", "resumable": True},
            "population_id": f"pnecessity-{source['source_id']}-{source['prepared_sha256'][:16]}",
            "dataset_id": str(source["source_id"]),
        },
        "metadata": {
            "comparison": "within_source_paired_utility", "input_arm": arm,
            "source_manifest": "./result/pnecessity_utility_prepare_v1/source_manifest.json",
            "prepared_data_sha256": source["prepared_sha256"],
            "raw_data_sha256": source["raw_sha256"],
            "fixed_or_unrecorded_numeric_c": fixed_numeric,
            "c_field_policy": "all recorded non-product molecular C fields are descriptor components; fixed numeric C fields are retained in the paired table and declared as conditions",
            "limitation": "within-source result only; do not pool rows, folds, repeats, or effect estimates across sources",
        },
    }
    return config


def generate(*, prep_root: Path = PREP_ROOT, config_dir: Path = ROOT / "config") -> dict[str, Any]:
    source_manifest = prep_root / "source_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing prepared source manifest: {source_manifest}")
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    generated: list[dict[str, str]] = []
    blocked: list[dict[str, Any]] = []
    for source in payload["sources"]:
        if source["config_status"] != "configurable":
            blocked.append({"source_id": source["source_id"], "reason": source["config_status_reason"], "variable_numeric_c": source["conditions"]["variable_numeric_c"]})
            # A previous revision may have emitted YAML before a newly
            # discovered source-specific fidelity constraint.  Remove only
            # the exact, generated task identities after checking their
            # project_name, never arbitrary config files.
            for line_id in FROZEN_LINES:
                for arm in ("full", "minus_p"):
                    stale_task = _task_name(str(source["source_id"]), line_id, arm)
                    stale_path = config_dir / f"{stale_task}.yaml"
                    if stale_path.exists():
                        observed = yaml.safe_load(stale_path.read_text(encoding="utf-8"))
                        if not isinstance(observed, Mapping) or observed.get("project_name") != stale_task:
                            raise RuntimeError(f"refusing to remove non-generated YAML: {stale_path}")
                        stale_path.unlink()
            continue
        for line_id in FROZEN_LINES:
            for arm in ("full", "minus_p"):
                task = _task_name(str(source["source_id"]), line_id, arm)
                path = config_dir / f"{task}.yaml"
                _atomic_yaml(_build_config(source, line_id, arm), path)
                generated.append({"source_id": str(source["source_id"]), "line_id": line_id, "arm": arm, "task_name": task, "config_path": str(path.relative_to(ROOT))})
    result = {
        "generator": "_verify/generate_pnecessity_utility_configs.py",
        "source_manifest": str(source_manifest.relative_to(ROOT)),
        "n_requested_sources": len(payload["sources"]), "n_configurable_sources": len({item["source_id"] for item in generated}),
        "n_generated_configs": len(generated), "expected_generated_configs": len(FROZEN_LINES) * 2 * len({item["source_id"] for item in generated}),
        "generated": generated, "capability_blocked": blocked,
    }
    result_path = prep_root / "generated_config_manifest.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static paired product-utility YAML; never starts a model.")
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args()
    report = generate(prep_root=args.prep_root.resolve(), config_dir=args.config_dir.resolve())
    print(json.dumps({key: report[key] for key in ("n_requested_sources", "n_configurable_sources", "n_generated_configs", "expected_generated_configs", "capability_blocked")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
