"""Static, non-model verification for the product-utility YAML matrix.

This checker is intentionally prohibited from importing the benchmark runner or
any descriptor builder.  It parses schema-2 YAML, validates its strict
benchmark contract and source/split identities, and proves that each minus-P
arm has no product field in descriptor, categorical, metadata, or grouping
paths.  It writes an auditable pass/fail record below the dedicated preparation
result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.config.loader import load_run_config
from yonod.splits.manifest import validate_split_manifest


PREP_ROOT = ROOT / "result" / "pnecessity_utility_prepare_v1"
PREFIX = "pnecessity_utility_"
LINES = {"morgan_rf": "rf", "mfp_lightgbm": "lightgbm"}
ARMS = ("full", "minus_p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{path}[{index}]")
    else:
        yield path, value


def _task_name(source_id: str, line: str, arm: str) -> str:
    return f"{PREFIX}{source_id}_{line}_{arm}_v1"


def _expected_task_paths(sources: list[Mapping[str, Any]]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    expected: dict[str, Path] = {}
    blocked: list[dict[str, Any]] = []
    for source in sources:
        if source["config_status"] != "configurable":
            blocked.append(dict(source))
            continue
        for line in LINES:
            for arm in ARMS:
                task = _task_name(str(source["source_id"]), line, arm)
                expected[task] = ROOT / "config" / f"{task}.yaml"
    return expected, blocked


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_arm(
    *, task: str, path: Path, source: Mapping[str, Any], line: str, arm: str
) -> dict[str, Any]:
    _require(path.is_file(), f"missing YAML: {path.relative_to(ROOT)}")
    loaded = load_run_config(path)
    raw = loaded.effective
    _require(raw["project_name"] == task, f"{task}: project_name mismatch")
    _require(raw["stage"] == "benchmark", f"{task}: must use benchmark stage")
    _require(raw["models"] == [LINES[line]], f"{task}: model line mismatch")
    roles = raw["dataset"]["column_roles"]
    expected_molecular_c = list(source["prepared_columns"]["molecular_c"])
    expected_numeric_c = list(source["prepared_columns"]["numeric_c"])
    _require(roles["reactants"] == ["a_smiles", "b_smiles"], f"{task}: reactants must be canonical A/B")
    _require(roles["others"] == expected_molecular_c, f"{task}: non-P molecular C roles differ from prepared source")
    _require(roles["conditions"] == expected_numeric_c, f"{task}: non-P numeric C roles differ from prepared source")
    _require(roles["categoricals"] == [], f"{task}: unexpected categorical field")
    descriptor = raw["descriptors"]
    _require(len(descriptor) == 1, f"{task}: exactly one frozen descriptor is required")
    descriptor_columns = descriptor[0]["columns"]
    non_p_columns = ["a_smiles", "b_smiles", *expected_molecular_c]
    expected_descriptor_columns = [*non_p_columns, *( ["p_smiles"] if arm == "full" else [])]
    _require(descriptor_columns == expected_descriptor_columns, f"{task}: descriptor arm differs from strict paired definition")
    _require(raw["outputs"]["report_formats"] == ["html", "markdown"], f"{task}: required report formats absent")
    _require(raw["artifacts"]["output_dir"] == f"./result/{task}/feature", f"{task}: feature output not isolated")
    _require(raw["outputs"]["root"] == f"./result/{task}", f"{task}: result root not isolated")
    _require(raw["evaluation"]["n_splits"] == 5 and raw["evaluation"]["n_repeats"] == 3 and raw["evaluation"]["seed"] == 20260918, f"{task}: CV protocol differs")
    _require(raw["evaluation"]["grouping"] == {"strategy": "precomputed_column", "group_column": "non_p_input_group_id", "protocol_label": "pnecessity_within_source_non_p_input", "max_group_fraction": 0.8}, f"{task}: grouping differs")
    _require(raw["evaluation"]["split_manifest"] == "./" + str(source["split_manifest"]["path"]), f"{task}: split source differs")
    _require(raw["metadata"]["prepared_data_sha256"] == source["prepared_sha256"], f"{task}: prepared hash mismatch")
    _require(raw["metadata"]["raw_data_sha256"] == source["raw_sha256"], f"{task}: raw hash mismatch")
    _require(raw["metadata"]["input_arm"] == arm, f"{task}: arm metadata mismatch")
    model_params = raw["model_params"]
    model_name = LINES[line]
    _require(list(model_params) == [model_name], f"{task}: declares parameters for non-selected model")
    _require(model_params[model_name]["estimator"]["n_jobs"] == 19, f"{task}: n_jobs must be 19")
    if arm == "full":
        _require(roles["products"] == ["p_smiles"], f"{task}: Full must expose exactly p_smiles as product")
    else:
        _require(roles["products"] == [], f"{task}: minus-P must have no product role")
        forbidden_section = {
            "descriptor": descriptor_columns,
            "categoricals": roles["categoricals"],
            "conditions": roles["conditions"],
            "grouping": raw["evaluation"]["grouping"],
            "metadata": raw["metadata"],
        }
        leakage = [(key_path, value) for key_path, value in _flatten(forbidden_section) if "p_smiles" in str(key_path) or "p_smiles" in str(value)]
        _require(not leakage, f"{task}: minus-P has a p_smiles leak in protected configuration sections: {leakage[:3]}")
        role_leak = [(key_path, value) for key_path, value in _flatten(roles) if "p_smiles" in str(key_path) or "p_smiles" in str(value)]
        _require(not role_leak, f"{task}: minus-P roles contain p_smiles: {role_leak[:3]}")

    # BenchmarkConfig parses schema and records a contract without invoking a
    # runner, descriptors, or an estimator.  This also reads the prepared CSV
    # solely to validate ID/label presence and validates all path isolation.
    strict = BenchmarkConfig.from_file(path)
    contract = create_benchmark_contract(strict)
    prepared_path = ROOT / str(source["prepared_path"])
    _require(strict.dataset_path == prepared_path.resolve(), f"{task}: resolved dataset path mismatch")
    _require(_sha256(prepared_path) == source["prepared_sha256"], f"{task}: current prepared CSV hash diverged")
    raw_path = ROOT / str(source["raw_path"])
    _require(_sha256(raw_path) == source["raw_sha256"], f"{task}: source raw hash diverged")
    _require(contract.dataset_sha256 == source["prepared_sha256"], f"{task}: contract data hash diverged")
    return {
        "task_name": task, "config_path": str(path.relative_to(ROOT)), "source_id": source["source_id"], "line": line, "arm": arm,
        "config_sha256": _sha256(path), "prepared_sha256": contract.dataset_sha256,
        "split_manifest_sha256": _sha256(ROOT / str(source["split_manifest"]["path"])), "contract_run_id": contract.run_id,
    }


def verify(prep_root: Path = PREP_ROOT) -> dict[str, Any]:
    payload_path = prep_root / "source_manifest.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    sources = payload["sources"]
    _require(len(sources) == 14, "source manifest must retain all 14 requested sources")
    expected, blocked = _expected_task_paths(sources)
    _require(len(blocked) == 5 and len(expected) == 36, "matrix must be 9 configurable sources / 36 YAML plus 5 blocks")
    existing = {path.stem: path for path in (ROOT / "config").glob(PREFIX + "*.yaml")}
    _require(set(existing) == set(expected), f"generated YAML set mismatch; unexpected={sorted(set(existing).difference(expected))}, missing={sorted(set(expected).difference(existing))}")
    for source in blocked:
        published_split = source.get("published_fixed_split")
        if published_split is not None:
            _require(published_split.get("protocol") == "published_model_external_split", f"{source['source_id']}: unknown published-split block")
            for path_key, sha_key, row_key in (("model_path", "model_sha256", "model_rows"), ("external_path", "external_sha256", "external_rows")):
                split_path = ROOT / str(published_split[path_key])
                _require(split_path.is_file() and _sha256(split_path) == published_split[sha_key], f"{source['source_id']}: published split asset hash mismatch")
                _require(int(published_split[row_key]) > 0, f"{source['source_id']}: published split row count absent")
        else:
            _require(source["conditions"]["variable_numeric_c"], f"{source['source_id']}: blocked source lacks stated variable C")
        generated_for_blocked = [name for name in existing if f"_{source['source_id']}_" in name]
        _require(not generated_for_blocked, f"{source['source_id']}: capability-blocked source has YAML: {generated_for_blocked}")

    records: list[dict[str, Any]] = []
    source_by_id = {str(item["source_id"]): item for item in sources}
    for task, path in sorted(expected.items()):
        matches = [(source_id, line, arm) for source_id in source_by_id for line in LINES for arm in ARMS if task == _task_name(source_id, line, arm)]
        _require(len(matches) == 1, f"cannot decode task identity {task}")
        source_id, line, arm = matches[0]
        records.append(_check_arm(task=task, path=path, source=source_by_id[source_id], line=line, arm=arm))
    record_frame = pd.DataFrame.from_records(records)

    pairing_rows: list[dict[str, Any]] = []
    for (source_id, line), pair in record_frame.groupby(["source_id", "line"], sort=True):
        _require(set(pair["arm"]) == set(ARMS) and len(pair) == 2, f"{source_id}/{line}: Full/minus-P pair incomplete")
        _require(pair["prepared_sha256"].nunique() == 1, f"{source_id}/{line}: arms do not read identical prepared CSV")
        _require(pair["split_manifest_sha256"].nunique() == 1, f"{source_id}/{line}: arms do not share actual split membership")
        pairing_rows.append({"source_id": source_id, "line": line, "prepared_sha256": pair["prepared_sha256"].iloc[0], "split_manifest_sha256": pair["split_manifest_sha256"].iloc[0], "paired_data_identity": True, "paired_split_identity": True})
    pairing_frame = pd.DataFrame.from_records(pairing_rows)

    split_rows: list[dict[str, Any]] = []
    for source in sources:
        if source["config_status"] != "configurable":
            continue
        manifest_path = ROOT / str(source["split_manifest"]["path"])
        manifest = pd.read_parquet(manifest_path)
        validate_split_manifest(manifest, n_splits=5, n_repeats=3)
        ids = pd.read_csv(ROOT / str(source["prepared_path"]), usecols=["sample_id", "non_p_input_group_id"]).set_index("sample_id")
        _require(set(manifest["sample_id"].astype(str)) == set(ids.index.astype(str)), f"{source['source_id']}: manifest data identity fails")
        _require(set(manifest["dataset_sha256"].astype(str)) == {source["prepared_sha256"]}, f"{source['source_id']}: manifest CSV hash fails")
        for (repeat, fold), part in manifest.groupby(["repeat", "fold"], sort=True):
            train = set(ids.loc[part.loc[part["role"].eq("train"), "sample_id"], "non_p_input_group_id"])
            valid = set(ids.loc[part.loc[part["role"].eq("valid"), "sample_id"], "non_p_input_group_id"])
            _require(not train.intersection(valid), f"{source['source_id']}: non-P grouping leakage at repeat={repeat}, fold={fold}")
        split_rows.append({"source_id": source["source_id"], "n_manifest_rows": len(manifest), "n_folds": int(manifest[["repeat", "fold"]].drop_duplicates().shape[0]), "non_p_group_leakage": 0, "manifest_sha256": _sha256(manifest_path)})

    prep_root.mkdir(parents=True, exist_ok=True)
    record_frame.to_csv(prep_root / "static_config_matrix.csv", index=False, encoding="utf-8")
    pairing_frame.to_csv(prep_root / "static_pairing_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame.from_records(split_rows).to_csv(prep_root / "static_split_audit.csv", index=False, encoding="utf-8")
    result = {
        "status": "passed", "validator": "_verify/verify_pnecessity_utility_plan.py", "validation_kind": "static only; no descriptor/model/runner invocation",
        "n_requested_sources": 14, "n_configurable_sources": 9, "n_capability_blocked_sources": 5,
        "n_expected_yaml": 36, "n_verified_yaml": len(records), "n_paired_source_model_comparisons": len(pairing_rows),
        "checks": ["schema YAML parse and strict BenchmarkConfig contract", "isolated output path and required HTML/Markdown reports", "selected-model-only parameters and n_jobs=19", "current raw/prepared hashes", "paired Full/minus-P CSV and split-manifest identity", "15-fold non-P input-group zero overlap", "minus-P no p_smiles role/descriptor/categorical/metadata/grouping leak"],
        "capability_blocked": [{"source_id": item["source_id"], "variable_numeric_c": item["conditions"]["variable_numeric_c"], "reason": item["config_status_reason"]} for item in blocked],
    }
    (prep_root / "static_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Statically verify paired product-utility configurations without any model work.")
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT)
    args = parser.parse_args()
    result = verify(args.prep_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
