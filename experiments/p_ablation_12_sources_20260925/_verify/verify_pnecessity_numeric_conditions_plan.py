"""Static launch gate for the numeric-condition Full/minus-P YAML matrix.

It neither computes descriptors nor fits estimators.  It verifies source and
YAML identities, exercises the strict adapter's data/numeric preflight, and
compares the generated Full/minus-P fold memberships after removing task-only
manifest identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.config.loader import load_run_config
from yonod.splits.manifest import create_split_manifest, validate_split_manifest


PREP_ROOT = ROOT / "result" / "pnecessity_numeric_conditions_prepare_v1"
PREFIX = "pnecessity_numeric_conditions_"
LINES = {"morgan_rf": "rf", "mfp_lightgbm": "lightgbm"}
ARMS = ("full", "minus_p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _task_name(source_id: str, line: str, arm: str) -> str:
    return f"{PREFIX}{source_id}_{line}_{arm}_v1"


def _flatten(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{path}[{index}]")
    else:
        yield path, value


def _assignment_digest(manifest: pd.DataFrame) -> str:
    columns = ["sample_id", "group_id", "repeat", "fold", "role"]
    records = manifest.loc[:, columns].sort_values(columns, kind="mergesort").to_dict(orient="records")
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sources = manifest.get("selected_sources")
    _require(isinstance(sources, list) and len(sources) == 4, "expected four selected numeric-condition sources")
    result = {str(source["source_id"]): source for source in sources}
    _require(len(result) == 4, "selected source IDs are not unique")
    return result


def _check_arm(task: str, path: Path, source: Mapping[str, Any], line: str, arm: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file(), f"missing YAML: {path.relative_to(ROOT)}")
    loaded = load_run_config(path)
    raw = loaded.effective
    molecules = ["a_smiles", "b_smiles", *list(source["molecular_c"])]
    numeric = list(source["numeric_c"])
    products = ["p_smiles"] if arm == "full" else []
    model_molecules = [*molecules, *products]
    _require(raw["project_name"] == task and raw["stage"] == "benchmark", f"{task}: identity/stage mismatch")
    _require(raw["models"] == [LINES[line]], f"{task}: frozen model line mismatch")
    roles = raw["dataset"]["column_roles"]
    _require(roles["label"] == "yield", f"{task}: label must be normalized yield")
    _require(roles["reactants"] == model_molecules, f"{task}: strict molecular input mapping is incomplete or reordered")
    _require(roles["others"] == [] and roles["categoricals"] == [], f"{task}: unsupported unmaterialized C role")
    _require(roles["conditions"] == numeric, f"{task}: numeric C differs from prepared source")
    _require(roles["products"] == products, f"{task}: product arm mismatch")
    contract_columns = raw["dataset"]["numeric_conditions"]["columns"]
    _require(list(contract_columns) == numeric, f"{task}: numeric contract source order differs")
    for column in numeric:
        _require(contract_columns[column]["name"] == column, f"{task}: numeric output name differs for {column}")
        _require(contract_columns[column]["missing"] == {"strategy": "error"}, f"{task}: missing rule differs for {column}")
        _require(contract_columns[column]["scaling"] == "standard", f"{task}: scaling differs for {column}")
    descriptor = raw["descriptors"]
    _require(len(descriptor) == 1, f"{task}: exactly one descriptor is required")
    _require(descriptor[0]["columns"] == model_molecules, f"{task}: descriptor components differ")
    _require(descriptor[0]["lifecycle"] == "static_descriptor", f"{task}: unexpected descriptor lifecycle")
    grouping = raw["evaluation"]["grouping"]
    _require(
        grouping == {
            "strategy": "component_holdout",
            "component_cols": molecules,
            "protocol_label": "pnecessity_non_p_molecular_component_holdout_numeric_v1",
            "max_group_fraction": 0.8,
        },
        f"{task}: grouping does not bind every non-P molecular component",
    )
    _require("split_manifest" not in raw["evaluation"], f"{task}: current strict adapter must generate its own deterministic manifest")
    _require(raw["evaluation"]["n_splits"] == 5 and raw["evaluation"]["n_repeats"] == 3 and raw["evaluation"]["seed"] == 20260918, f"{task}: CV contract differs")
    _require(raw["outputs"]["report_formats"] == ["html", "markdown"], f"{task}: dual reports absent")
    _require(raw["outputs"]["root"] == f"./result/{task}", f"{task}: result root not isolated")
    _require(raw["artifacts"]["output_dir"] == f"./result/{task}/feature", f"{task}: feature root not isolated")
    model = LINES[line]
    _require(list(raw["model_params"]) == [model], f"{task}: non-selected model parameters declared")
    _require(raw["model_params"][model]["estimator"]["n_jobs"] == 19, f"{task}: 19-CPU declaration absent")
    _require(raw["metadata"]["prepared_data_sha256"] == source["prepared_sha256"], f"{task}: prepared hash mismatch")
    _require(raw["metadata"]["raw_data_sha256"] == source["raw_sha256"], f"{task}: raw hash mismatch")
    _require(raw["metadata"]["input_arm"] == arm, f"{task}: arm metadata mismatch")
    if arm == "full":
        _require("product_embedding_execution_mapping" in raw["metadata"], f"{task}: missing explicit strict product mapping")
    if arm == "minus_p":
        protected = {"roles": roles, "descriptor": descriptor, "grouping": grouping, "numeric": contract_columns, "metadata": raw["metadata"]}
        leaks = [(key, value) for key, value in _flatten(protected) if "p_smiles" in str(key) or "p_smiles" in str(value)]
        _require(not leaks, f"{task}: minus-P product leak: {leaks[:3]}")

    strict = BenchmarkConfig.from_file(path)
    benchmark_contract = create_benchmark_contract(strict)
    prepared_path = ROOT / str(source["prepared_path"])
    raw_path = ROOT / str(source["raw_path"])
    _require(strict.dataset_path == prepared_path.resolve(), f"{task}: resolved prepared path mismatch")
    _require(_sha256(prepared_path) == source["prepared_sha256"], f"{task}: prepared data hash diverged")
    _require(_sha256(raw_path) == source["raw_sha256"], f"{task}: raw data hash diverged")
    _require(benchmark_contract.dataset_sha256 == source["prepared_sha256"], f"{task}: strict dataset identity diverged")
    generated_manifest = create_split_manifest(benchmark_contract)
    validate_split_manifest(generated_manifest, n_splits=5, n_repeats=3)
    return ({
        "task_name": task,
        "config_path": str(path.relative_to(ROOT)),
        "config_sha256": _sha256(path),
        "source_id": str(source["source_id"]),
        "line": line,
        "arm": arm,
        "prepared_sha256": benchmark_contract.dataset_sha256,
        "contract_run_id": benchmark_contract.run_id,
        "split_assignment_sha256": _assignment_digest(generated_manifest),
        "n_split_rows": int(len(generated_manifest)),
    }, _assignment_digest(generated_manifest))


def verify(prep_root: Path = PREP_ROOT) -> dict[str, Any]:
    generated_path = prep_root / "generated_config_manifest.json"
    _require(generated_path.is_file(), f"missing generated manifest: {generated_path}")
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    sources = _source_map(generated)
    _require(generated.get("n_generated_configs") == 16 and generated.get("expected_generated_configs") == 16, "generated matrix is not 16 configs")
    expected = {
        _task_name(source_id, line, arm): ROOT / "config" / f"{_task_name(source_id, line, arm)}.yaml"
        for source_id in sources for line in LINES for arm in ARMS
    }
    actual = {path.stem: path for path in (ROOT / "config").glob(PREFIX + "*.yaml")}
    _require(set(actual) == set(expected), f"numeric-condition YAML set mismatch; unexpected={sorted(set(actual)-set(expected))}, missing={sorted(set(expected)-set(actual))}")

    records = []
    for source_id, source in sorted(sources.items()):
        for line in LINES:
            pair = []
            for arm in ARMS:
                task = _task_name(source_id, line, arm)
                record, split_digest = _check_arm(task, expected[task], source, line, arm)
                records.append(record)
                pair.append((arm, record, split_digest))
            _require(pair[0][2] == pair[1][2], f"{source_id}/{line}: Full/minus-P generated split membership differs")
            _require(pair[0][1]["prepared_sha256"] == pair[1][1]["prepared_sha256"], f"{source_id}/{line}: paired data differs")

    result = {
        "status": "passed",
        "checker": "_verify/verify_pnecessity_numeric_conditions_plan.py",
        "generated_manifest": str(generated_path.relative_to(ROOT)),
        "n_sources": len(sources),
        "n_configs": len(records),
        "records": records,
        "excluded_capability_gap": generated["excluded_capability_gap"],
    }
    _atomic_json(result, prep_root / "static_validation.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Statically verify numeric-condition paired YAML and deterministic paired splits.")
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT)
    args = parser.parse_args()
    result = verify(args.prep_root.resolve())
    print(json.dumps({key: result[key] for key in ("status", "n_sources", "n_configs", "excluded_capability_gap")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
