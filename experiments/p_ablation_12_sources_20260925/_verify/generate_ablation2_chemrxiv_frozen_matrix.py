"""Generate and statically audit the preregistered ChemRxiv frozen-test matrix.

This utility never imports a descriptor implementation, fits a model, or
reads external labels.  It only freezes the 12 schema-2 task declarations and
proves that their two populations, column slots, checksums and result roots
meet the dedicated ``frozen_train_external_test`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.frozen_external import EXTERNAL_PROTOCOL, FrozenExternalTestConfig


DEVELOPMENT = ROOT / "result" / "ablation2_amide_data_prep_v1" / "data" / "standardized_population.csv"
EXTERNAL = ROOT / "result" / "ablation2_chemrxiv_external_data_audit_v1" / "data" / "standardized_external_population.csv"
AUDIT_MANIFEST = ROOT / "result" / "ablation2_chemrxiv_external_data_audit_v1" / "run_manifest.json"
CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "project-docs" / "docs" / "ablation2"
AUDIT_JSON = DOCS_DIR / "chemrxiv_frozen_matrix_audit_v1.json"
AUDIT_REPORT = DOCS_DIR / "chemrxiv_frozen_matrix_audit_v1.md"

# The preregistration fixes external C as reagent-1/reagent-2/solvent-1.
# Development has four historical C slots; additive has no distinct external
# counterpart.  The formal external matrix therefore fixes its comparable C
# slots before any score is read: activation/base/solvent.
COMMON_C_COLUMNS = ("activation_smiles", "base_smiles", "solvent_smiles")
EXTERNAL_COLUMN_MAP = {
    "amine_smiles": "reactant-2",
    "acid_smiles": "reactant-1",
    "product_smiles": "product",
    "activation_smiles": "reagent-1",
    "base_smiles": "reagent-2",
    "solvent_smiles": "solvent-1",
}
INPUTS = {
    "full": ("amine_smiles", "acid_smiles", "product_smiles", *COMMON_C_COLUMNS),
    "minus_a": ("acid_smiles", "product_smiles", *COMMON_C_COLUMNS),
    "minus_p": ("amine_smiles", "acid_smiles", *COMMON_C_COLUMNS),
    "minus_a_p": ("acid_smiles", *COMMON_C_COLUMNS),
    "product_conditions": ("product_smiles", *COMMON_C_COLUMNS),
    "conditions_only": COMMON_C_COLUMNS,
}
METHODS: dict[str, dict[str, Any]] = {
    "morgan_rf": {
        "descriptor": "morgan",
        "model": "rf",
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
        "descriptor_params": {},
    },
    "mfp_lightgbm": {
        "descriptor": "mfp",
        "model": "lightgbm",
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
        "descriptor_params": {"radius": 3, "fp_size": 1024, "profile": "standard"},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_name(method: str, input_id: str) -> str:
    return "ablation2_chemrxiv_frozen_{0}_{1}_v1".format(method, input_id)


def task_config(method: str, input_id: str) -> dict[str, Any]:
    spec = METHODS[method]
    name = task_name(method, input_id)
    columns = list(INPUTS[input_id])
    descriptor = {
        "id": "{0}_{1}".format(spec["descriptor"], input_id),
        "descriptor": spec["descriptor"],
        "lifecycle": "static_descriptor",
        "mode": "concat",
        "columns": columns,
    }
    if spec["descriptor_params"]:
        descriptor["params"] = dict(spec["descriptor_params"])
    return {
        "schema_version": "2.0",
        "project_name": name,
        "stage": "benchmark",
        "dataset": {
            "path": "./result/ablation2_amide_data_prep_v1/data/standardized_population.csv",
            "sample_id_col": "sample_id",
            "column_roles": {
                "label": "yield",
                "reactants": ["amine_smiles", "acid_smiles"],
                "products": ["product_smiles"],
                "others": list(COMMON_C_COLUMNS),
                "conditions": [],
                "categoricals": [],
            },
        },
        "descriptors": [descriptor],
        "artifacts": {"output_dir": "./result/{0}/feature".format(name)},
        "models": [spec["model"]],
        "model_params": spec["model_params"],
        "evaluation": {
            "protocol": EXTERNAL_PROTOCOL,
            "seed": 20260918,
            "external_test": {
                "path": "./result/ablation2_chemrxiv_external_data_audit_v1/data/standardized_external_population.csv",
                "sample_id_col": "sample_id",
                "label_col": "yield",
                "source_id_col": "source_reaction_key",
                "group_col": "amine_structure_key",
                "column_map": dict(EXTERNAL_COLUMN_MAP),
                "expected_sha256": sha256(EXTERNAL),
                "audit_manifest": "./result/ablation2_chemrxiv_external_data_audit_v1/run_manifest.json",
                "audit_manifest_sha256": sha256(AUDIT_MANIFEST),
            },
        },
        "outputs": {"root": "./result/{0}".format(name), "report_formats": ["html", "markdown"]},
        "benchmark": {
            "task_state": {"backend": "sqlite", "resumable": True},
            "population_id": "ablation2_amide_standardized_v1",
            "dataset_id": "ablation2_chemrxiv_amide_external",
        },
        "metadata": {
            "notes": (
                "Pre-registered frozen-development external test; no external row or label may enter fitting, scaling, "
                "model selection, tuning, or early stopping. Cross-source C slots are fixed before scoring: development "
                "activation_smiles/base_smiles/solvent_smiles map one-to-one to external reagent-1/reagent-2/solvent-1. "
                "Development additive_smiles is excluded because the preregistered external C has no distinct counterpart."
            )
        },
    }


def write_configs() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for method in METHODS:
        for input_id in INPUTS:
            path = CONFIG_DIR / (task_name(method, input_id) + ".yaml")
            text = yaml.safe_dump(task_config(method, input_id), allow_unicode=True, sort_keys=False)
            if path.exists() and path.read_text(encoding="utf-8") != text:
                raise RuntimeError("Refusing to overwrite a divergent formal YAML: {0}".format(path))
            if not path.exists():
                path.write_text(text, encoding="utf-8")
            paths.append(path)
    return paths


def static_audit(paths: list[Path]) -> dict[str, Any]:
    expected_names = {task_name(method, input_id) for method in METHODS for input_id in INPUTS}
    if len(paths) != 12 or {path.stem for path in paths} != expected_names:
        raise RuntimeError("Formal matrix must contain exactly 12 named YAML configurations")
    seen_outputs: set[Path] = set()
    records = []
    for path in paths:
        config = FrozenExternalTestConfig.from_file(path)
        if config.dataset_path != DEVELOPMENT.resolve() or config.external_path != EXTERNAL.resolve():
            raise RuntimeError("Unexpected development/external dataset in {0}".format(path.name))
        if config.audit_manifest_path != AUDIT_MANIFEST.resolve():
            raise RuntimeError("Unexpected audit manifest in {0}".format(path.name))
        if config.external_column_map != EXTERNAL_COLUMN_MAP:
            raise RuntimeError("External role mapping changed in {0}".format(path.name))
        if config.external_group_col != "amine_structure_key":
            raise RuntimeError("External bootstrap group changed in {0}".format(path.name))
        if config.outputs_root in seen_outputs or config.outputs_root.exists():
            raise RuntimeError("Formal result root is duplicate or already occupied: {0}".format(config.outputs_root))
        seen_outputs.add(config.outputs_root)
        expected_columns = INPUTS[next(input_id for input_id in INPUTS if path.stem.endswith("_" + input_id + "_v1"))]
        actual_columns = config.feature_sets[0].columns
        if actual_columns != expected_columns:
            raise RuntimeError("Input columns differ from frozen matrix in {0}".format(path.name))
        n_jobs = config.model_configs[config.models[0]].estimator.get("n_jobs")
        if n_jobs != 19:
            raise RuntimeError("19-CPU model declaration missing in {0}".format(path.name))
        records.append({
            "config": str(path.relative_to(ROOT)),
            "task_name": path.stem,
            "descriptor": config.feature_sets[0].algorithm,
            "model": config.models[0],
            "input_columns": list(actual_columns),
            "output_root": str(config.outputs_root.relative_to(ROOT)),
            "external_sha256": config.external_expected_sha256,
            "audit_manifest_sha256": config.audit_manifest_sha256,
        })
    return {
        "matrix_version": "ablation2_chemrxiv_frozen_matrix_v1",
        "formal_task_count": len(records),
        "formal_model_tasks_started": 0,
        "development_dataset": str(DEVELOPMENT.relative_to(ROOT)),
        "development_sha256": sha256(DEVELOPMENT),
        "external_dataset": str(EXTERNAL.relative_to(ROOT)),
        "external_sha256": sha256(EXTERNAL),
        "audit_manifest": str(AUDIT_MANIFEST.relative_to(ROOT)),
        "audit_manifest_sha256": sha256(AUDIT_MANIFEST),
        "external_group_column": "amine_structure_key",
        "cross_source_c_slot_mapping": {
            "development": list(COMMON_C_COLUMNS),
            "external": ["reagent-1", "reagent-2", "solvent-1"],
            "mapping": dict(EXTERNAL_COLUMN_MAP),
            "excluded_development_condition_slot": "additive_smiles",
        },
        "records": records,
    }


def write_audit(audit: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    AUDIT_JSON.write_text(text, encoding="utf-8")
    report = """# ChemRxiv frozen external matrix static audit v1

## Outcome

- Formal frozen-test YAML files: {count}; formal model tasks started: **0**.
- Each configuration loaded through `FrozenExternalTestConfig.from_file()` without importing descriptors or fitting/scoring a model.
- All output roots are distinct, under `result/`, and were absent at audit time.
- Each task declares HTML and Markdown reports plus 19 CPU for its selected estimator.

## Fixed cross-source roles

The pre-registered external roles are A=`reactant-2`, B=`reactant-1`, P=`product`, C=`reagent-1`, `reagent-2`, `solvent-1`. To preserve a one-to-one feature-slot alignment, development C is fixed for this extension to `activation_smiles`, `base_smiles`, `solvent_smiles`, respectively. `additive_smiles` is excluded from every formal external configuration because it has no distinct pre-registered external counterpart. This is an execution mapping fixed before any 957-row prediction; it is not a claim that the two source taxonomies are chemically identical.

| Source | SHA-256 |
| --- | --- |
| Development population | `{development_sha}` |
| Standardized external population | `{external_sha}` |
| External audit manifest | `{audit_sha}` |

## Matrix

| Method | Input sets | Tasks |
| --- | --- | --- |
| Morgan ECFP4 × RF | full, minus_a, minus_p, minus_a_p, product_conditions, conditions_only | 6 |
| MFP × LightGBM | full, minus_a, minus_p, minus_a_p, product_conditions, conditions_only | 6 |

The frozen external protocol has one development fit and one external prediction; it is not CV, contains no external split manifest, and cannot enter strict-CV rankings. The next authorised operation is linear task launch via `yonod.py`, one YAML at a time; this audit did not launch any task.
""".format(
        count=audit["formal_task_count"],
        development_sha=audit["development_sha256"],
        external_sha=audit["external_sha256"],
        audit_sha=audit["audit_manifest_sha256"],
    )
    AUDIT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create the twelve YAML declarations if absent")
    args = parser.parse_args()
    paths = [CONFIG_DIR / (task_name(method, input_id) + ".yaml") for method in METHODS for input_id in INPUTS]
    if args.write:
        paths = write_configs()
    if missing := [path for path in paths if not path.is_file()]:
        raise RuntimeError("Missing formal YAML configurations: " + ", ".join(str(path) for path in missing))
    audit = static_audit(paths)
    write_audit(audit)
    print("[static-audit] formal_yaml={0} models_started=0 audit={1}".format(len(paths), AUDIT_REPORT))


if __name__ == "__main__":
    main()
