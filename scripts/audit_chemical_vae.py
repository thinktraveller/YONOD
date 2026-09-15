#!/usr/bin/env python3
"""Create reproducible pre-integration evidence for Chemical VAE.

This utility intentionally does *not* load TensorFlow/Keras or run model
inference.  It performs the two gates that must precede a descriptor
implementation:

* inspect the immutable Chemical VAE metadata/HDF5 encoder assets and the
  active environment; and
* scan representative YONOD inputs against the exact, frozen ZINC character
  table without canonicalising, splitting salts, truncating, or expanding the
  table.

The output is evidence rather than a feature artifact.  A successful run only
means that the preconditions are auditable; numerical parity remains a step
31.3 requirement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = REPOSITORY_ROOT / "reference-proejct" / "chemical_vae"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "derived" / "chemical_vae"
SUPPORTED_VARIANT = "zinc"

@dataclass(frozen=True)
class CoverageDataset:
    """One explicitly scoped CSV input scan."""

    name: str
    path: Path
    sample_id_column: Optional[str]
    columns: Tuple[Tuple[str, str], ...]


def default_coverage_datasets(repository_root: Path) -> Tuple[CoverageDataset, ...]:
    """Return the fixed representative input scope for work package 31.2."""

    return (
        CoverageDataset(
            name="benchmark_smoke_fixture",
            path=repository_root / "dataset" / "benchmark_smoke_fixture.csv",
            sample_id_column="reaction_id",
            columns=(
                ("reactant", "reactant_1_smiles"),
                ("reactant", "reactant_2_smiles"),
            ),
        ),
        CoverageDataset(
            name="chemical_vae_contract_boundaries",
            path=repository_root / "tests" / "fixtures" / "chemical_vae_boundary_smiles.csv",
            sample_id_column="case_id",
            columns=(("contract_test", "smiles"),),
        ),
        CoverageDataset(
            name="amide_coupling",
            path=repository_root / "dataset" / "amide-coupling.csv",
            sample_id_column="row_id",
            columns=(
                ("reactant", "sub_1_smiles"),
                ("reactant", "sub_2_smiles"),
                ("condition", "activation"),
                ("condition", "additive"),
                ("condition", "base"),
                ("condition", "solvent"),
            ),
        ),
        CoverageDataset(
            name="ord_suzuki_metal_components",
            path=(
                repository_root
                / "dataset"
                / "ORD"
                / "suzuki_yield_structure_v1_normalized_dataset.csv"
            ),
            sample_id_column=None,
            columns=(
                ("reactant", "reactant-1"),
                ("reactant", "reactant-2"),
                ("condition", "reagent-1"),
                ("catalyst", "catalyst-1"),
                ("catalyst", "catalyst-2"),
                ("solvent", "solvent-1"),
                ("solvent", "solvent-2"),
                ("solvent", "solvent-3"),
                ("solvent", "solvent-4"),
                ("solvent", "solvent-5"),
                ("solvent", "solvent-6"),
                ("solvent", "solvent-7"),
            ),
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    """Return a content hash without loading an asset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_command(args: Sequence[str], cwd: Path) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"args": list(args), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "args": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot(repository_root: Path) -> Dict[str, Any]:
    """Capture installed versions and declared dependency differences."""

    package_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "rdkit": "rdkit",
        "scikit_learn": "scikit-learn",
        "autogluon_tabular": "autogluon.tabular",
        "h5py": "h5py",
        "tensorflow": "tensorflow",
        "keras": "keras",
    }
    installed = {key: _package_version(value) for key, value in package_names.items()}
    declared_requirements = (repository_root / "requirements.txt").read_text(encoding="utf-8")
    return {
        "captured_at": _utc_now(),
        "python": {
            "version": sys.version.replace("\n", " "),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": installed,
        "requirements_txt_sha256": sha256_file(repository_root / "requirements.txt"),
        "requirements_mentions_h5py": bool(re.search(r"(?m)^h5py(?:[<=>!~]|$)", declared_requirements)),
        "pip_check": _run_command([sys.executable, "-m", "pip", "check"], repository_root),
        "dependency_interpretation": {
            "h5py": (
                "installed only to inspect/convert Chemical VAE HDF5 assets; "
                "it does not install TensorFlow or Keras"
            ),
            "tensorflow_keras": (
                "not installed and intentionally not selected for the current "
                "encoder migration path"
            ),
            "torch": "retained; no version was downgraded for this audit",
        },
    }


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def inspect_encoder_h5(path: Path) -> Dict[str, Any]:
    """Extract only encoder architecture and weight metadata from legacy HDF5."""

    try:
        h5py = importlib.import_module("h5py")
    except ModuleNotFoundError:
        return {
            "status": "not_inspected",
            "reason": "h5py is unavailable",
            "path": str(path),
        }

    with h5py.File(path, "r") as handle:
        attributes = {str(key): _decode_h5_value(value) for key, value in handle.attrs.items()}
        raw_model_config = attributes.get("model_config")
        if not isinstance(raw_model_config, str):
            raise ValueError(f"{path} has no JSON model_config attribute")
        model_config = json.loads(raw_model_config)
        datasets: List[Dict[str, Any]] = []

        def collect_dataset(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                datasets.append({"name": name, "shape": list(item.shape), "dtype": str(item.dtype)})

        handle.visititems(collect_dataset)

    config = model_config.get("config", {})
    layers = config.get("layers", [])
    input_layers = config.get("input_layers", [])
    output_layers = config.get("output_layers", [])
    return {
        "status": "inspected",
        "path": str(path),
        "keras_backend": attributes.get("backend"),
        "keras_version": attributes.get("keras_version"),
        "model_name": config.get("name"),
        "input_layers": input_layers,
        "output_layers": output_layers,
        "layers": [
            {
                "name": layer.get("name"),
                "class_name": layer.get("class_name"),
                "config": layer.get("config", {}),
            }
            for layer in layers
        ],
        "weight_datasets": sorted(datasets, key=lambda item: item["name"]),
    }


def _encoder_contract(h5_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn inspected Keras metadata into the migration-critical contract."""

    if h5_metadata.get("status") != "inspected":
        return {"status": "not_verified", "reason": h5_metadata.get("reason")}
    layers = list(h5_metadata["layers"])
    input_layer = next((layer for layer in layers if layer["class_name"] == "InputLayer"), None)
    z_mean_layer = next((layer for layer in layers if layer["name"] == "z_mean_sample"), None)
    if input_layer is None or z_mean_layer is None:
        raise ValueError("encoder model config lacks required input_molecule_smi or z_mean_sample layer")
    return {
        "status": "verified_from_hdf5_metadata",
        "input_shape": input_layer["config"].get("batch_input_shape"),
        "input_dtype": input_layer["config"].get("dtype"),
        "z_mean_layer": z_mean_layer["name"],
        "z_mean_units": z_mean_layer["config"].get("units"),
        "z_mean_activation": z_mean_layer["config"].get("activation"),
        "output_layers": h5_metadata["output_layers"],
        "inference_relevant_layers": [
            {
                "name": layer["name"],
                "class_name": layer["class_name"],
                "activation": layer["config"].get("activation"),
                "padding": layer["config"].get("padding"),
                "kernel_size": layer["config"].get("kernel_size"),
                "filters_or_units": layer["config"].get("filters", layer["config"].get("units")),
                "epsilon": layer["config"].get("epsilon"),
                "dropout_rate": layer["config"].get("rate"),
            }
            for layer in layers
            if layer["class_name"] in {"Conv1D", "BatchNormalization", "Dense", "Flatten", "Dropout"}
        ],
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_reference(reference_root: Path) -> Dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], reference_root)
    commit_date = _run_command(["git", "log", "-1", "--format=%aI"], reference_root)
    status = _run_command(["git", "status", "--short"], reference_root)
    return {
        "commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "commit_date": commit_date.get("stdout") if commit_date.get("returncode") == 0 else None,
        "worktree_status": status.get("stdout") if status.get("returncode") == 0 else None,
        "commands": {"rev_parse": commit, "commit_date": commit_date, "status": status},
    }


def asset_audit(reference_root: Path, repository_root: Path) -> Dict[str, Any]:
    """Audit both bundled variants without copying or mutating their assets."""

    reference = _git_reference(reference_root)
    variants: Dict[str, Any] = {}
    for variant_dir in sorted((reference_root / "models").iterdir()):
        if not variant_dir.is_dir():
            continue
        exp_path = variant_dir / "exp.json"
        if not exp_path.is_file():
            continue
        exp = _read_json(exp_path)
        char_path = variant_dir / str(exp["char_file"])
        encoder_path = variant_dir / str(exp["encoder_weights_file"])
        decoder_path = variant_dir / str(exp["decoder_weights_file"])
        property_path = variant_dir / str(exp["prop_pred_weights_file"]) if exp.get("do_prop_pred") else None
        char_table = _read_json(char_path)
        encoder_h5 = inspect_encoder_h5(encoder_path)
        contract = _encoder_contract(encoder_h5)
        variants[variant_dir.name] = {
            "metadata": {
                "exp_json": _safe_relative(exp_path, repository_root),
                "exp_json_sha256": sha256_file(exp_path),
                "name": exp.get("name"),
                "max_len": exp.get("MAX_LEN"),
                "padding": exp.get("PADDING"),
                "declared_hidden_dim": exp.get("hidden_dim"),
                "property_head_enabled": bool(exp.get("do_prop_pred")),
            },
            "charset": {
                "path": _safe_relative(char_path, repository_root),
                "sha256": sha256_file(char_path),
                "characters": char_table,
                "character_count": len(char_table),
                "duplicates": sorted({char for char in char_table if char_table.count(char) > 1}),
            },
            "assets": {
                "encoder_h5": {
                    "path": _safe_relative(encoder_path, repository_root),
                    "sha256": sha256_file(encoder_path),
                    "size_bytes": encoder_path.stat().st_size,
                },
                "decoder_h5": {
                    "path": _safe_relative(decoder_path, repository_root),
                    "sha256": sha256_file(decoder_path),
                    "size_bytes": decoder_path.stat().st_size,
                    "inspection_scope": "listed_and_hashed_only; decoder is outside the step-31 descriptor boundary",
                },
                "property_predictor_h5": (
                    {
                        "path": _safe_relative(property_path, repository_root),
                        "sha256": sha256_file(property_path),
                        "size_bytes": property_path.stat().st_size,
                        "inspection_scope": "listed_and_hashed_only; property head is out of scope",
                    }
                    if property_path is not None and property_path.is_file()
                    else None
                ),
            },
            "encoder_hdf5": encoder_h5,
            "encoder_contract": contract,
        }
        if contract.get("status") == "verified_from_hdf5_metadata":
            expected_input = [None, exp.get("MAX_LEN"), len(char_table)]
            if contract.get("input_shape") != expected_input:
                raise ValueError(
                    f"{variant_dir.name}: exp/charset input {expected_input!r} conflicts with HDF5 "
                    f"input {contract.get('input_shape')!r}"
                )
            if contract.get("z_mean_units") != exp.get("hidden_dim"):
                raise ValueError(
                    f"{variant_dir.name}: exp hidden_dim {exp.get('hidden_dim')!r} conflicts with HDF5 "
                    f"z_mean units {contract.get('z_mean_units')!r}"
                )

    zinc = variants.get(SUPPORTED_VARIANT)
    if zinc is None:
        raise FileNotFoundError(f"reference asset variant {SUPPORTED_VARIANT!r} is missing")
    properties = variants.get("zinc_properties")
    same_encoder = (
        properties is not None
        and zinc["assets"]["encoder_h5"]["sha256"] == properties["assets"]["encoder_h5"]["sha256"]
    )
    return {
        "audited_at": _utc_now(),
        "reference_repository": reference,
        "variants": variants,
        "variant_status": {
            "zinc": "asset_and_structure_audited; numerical_parity_not_yet_verified",
            "zinc_properties": (
                "asset_and_structure_audited_but_not_selectable; property-head/variant parity remains unverified"
            ),
        },
        "cross_variant_checks": {
            "zinc_and_zinc_properties_encoder_sha256_equal": same_encoder,
            "note": (
                "matching encoder bytes do not validate the properties variant or make its property head in scope"
            ),
        },
    }


def _rdkit_parser() -> Any:
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:
        raise RuntimeError("RDKit is required for the step-31.2 parse audit") from exc
    RDLogger.DisableLog("rdApp.error")
    return Chem


def assess_smiles(raw_value: Any, *, charset: Sequence[str], max_len: int, chem: Any) -> Dict[str, Any]:
    """Assess one raw CSV cell using the frozen, identity preprocessing contract."""

    raw = "" if raw_value is None else str(raw_value)
    # Input identity is intentional.  No strip/canonicalisation/salt splitting
    # occurs before the checks, so a later descriptor must adopt this exact
    # contract or create a new audit identity.
    processed = raw
    missing = processed == ""
    unknown_characters = sorted(set(processed).difference(charset)) if not missing else []
    length_ok = not missing and len(processed) <= max_len
    rdkit_parse_ok = False
    if not missing:
        try:
            rdkit_parse_ok = chem.MolFromSmiles(processed) is not None
        except Exception:
            rdkit_parse_ok = False
    charset_ok = not missing and not unknown_characters
    reasons: List[str] = []
    if missing:
        reasons.append("missing")
    else:
        if not rdkit_parse_ok:
            reasons.append("rdkit_invalid")
        if not length_ok:
            reasons.append("too_long")
        if not charset_ok:
            reasons.append("unsupported_character")
    preflight_encodable = not reasons
    return {
        "raw_input": raw,
        "processed_input": processed,
        "preprocessing": "identity",
        "is_missing": missing,
        "length": len(processed),
        "length_ok": length_ok,
        "rdkit_parse_ok": rdkit_parse_ok,
        "charset_ok": charset_ok,
        "unknown_characters": unknown_characters,
        "failure_reasons": reasons,
        "preflight_encodable": preflight_encodable,
        "contains_dot": "." in processed,
        "contains_comma": "," in processed,
        "contains_semicolon": ";" in processed,
        "contains_ni": "[Ni]" in processed or "[Ni+" in processed,
        "contains_pd": "[Pd]" in processed or "[Pd+" in processed,
    }


def _sample_id(dataset: CoverageDataset, row_index: int, value: Any) -> str:
    if dataset.sample_id_column is None:
        return f"{dataset.name}:row:{row_index + 1:06d}"
    return str(value)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})


def coverage_audit(
    datasets: Iterable[CoverageDataset],
    *,
    charset: Sequence[str],
    max_len: int,
    repository_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Write full row/role diagnostics and aggregate coverage counts."""

    chem = _rdkit_parser()
    diagnostics: List[Dict[str, Any]] = []
    reactions: List[Dict[str, Any]] = []
    dataset_manifest: List[Dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.path.is_file():
            raise FileNotFoundError(f"coverage input does not exist: {dataset.path}")
        frame = pd.read_csv(dataset.path, dtype=str, keep_default_na=False)
        required = [column for _, column in dataset.columns]
        if dataset.sample_id_column is not None:
            required.append(dataset.sample_id_column)
        missing_columns = sorted(set(required).difference(frame.columns))
        if missing_columns:
            raise KeyError(f"{dataset.path}: missing requested audit columns {missing_columns}")
        dataset_manifest.append(
            {
                "name": dataset.name,
                "path": _safe_relative(dataset.path, repository_root),
                "sha256": sha256_file(dataset.path),
                "row_count": len(frame),
                "sample_id_column": dataset.sample_id_column,
                "sample_id_source": dataset.sample_id_column or "one_based_csv_row_number",
                "columns": [{"role": role, "column": column} for role, column in dataset.columns],
            }
        )
        for row_index, row in frame.iterrows():
            identifier = _sample_id(
                dataset,
                int(row_index),
                row[dataset.sample_id_column] if dataset.sample_id_column is not None else None,
            )
            row_assessments: List[Dict[str, Any]] = []
            for role, column in dataset.columns:
                assessment = assess_smiles(row[column], charset=charset, max_len=max_len, chem=chem)
                record = {
                    "dataset": dataset.name,
                    "sample_id": identifier,
                    "source_row_number": int(row_index) + 1,
                    "role": role,
                    "column": column,
                    **assessment,
                }
                record["unknown_characters"] = json.dumps(record["unknown_characters"], ensure_ascii=False)
                record["failure_reasons"] = json.dumps(record["failure_reasons"], ensure_ascii=False)
                diagnostics.append(record)
                row_assessments.append(assessment)
            success_count = sum(item["preflight_encodable"] for item in row_assessments)
            reactions.append(
                {
                    "dataset": dataset.name,
                    "sample_id": identifier,
                    "source_row_number": int(row_index) + 1,
                    "role_count": len(row_assessments),
                    "preflight_successful_role_count": success_count,
                    "concat_row_kept": bool(success_count),
                    "strict_all_roles_encodable": success_count == len(row_assessments),
                    "final_inference_status": "not_run_step_31_2_preflight_only",
                }
            )

    role_coverage: List[Dict[str, Any]] = []
    for (dataset_name, role, column), group in pd.DataFrame(diagnostics).groupby(["dataset", "role", "column"], sort=True):
        failures = group["failure_reasons"].map(json.loads)
        role_coverage.append(
            {
                "dataset": dataset_name,
                "role": role,
                "column": column,
                "cell_count": int(len(group)),
                "missing_count": int(group["is_missing"].sum()),
                "rdkit_parse_ok_count": int(group["rdkit_parse_ok"].sum()),
                "charset_ok_count": int(group["charset_ok"].sum()),
                "length_ok_count": int(group["length_ok"].sum()),
                "preflight_encodable_count": int(group["preflight_encodable"].sum()),
                "rdkit_invalid_count": int(sum("rdkit_invalid" in item for item in failures)),
                "unsupported_character_count": int(sum("unsupported_character" in item for item in failures)),
                "too_long_count": int(sum("too_long" in item for item in failures)),
                "contains_dot_count": int(group["contains_dot"].sum()),
                "contains_comma_count": int(group["contains_comma"].sum()),
                "contains_semicolon_count": int(group["contains_semicolon"].sum()),
                "contains_ni_count": int(group["contains_ni"].sum()),
                "contains_pd_count": int(group["contains_pd"].sum()),
            }
        )

    reaction_coverage: List[Dict[str, Any]] = []
    for dataset_name, group in pd.DataFrame(reactions).groupby("dataset", sort=True):
        reaction_coverage.append(
            {
                "dataset": dataset_name,
                "reaction_count": int(len(group)),
                "concat_kept_count": int(group["concat_row_kept"].sum()),
                "concat_dropped_all_roles_failed_count": int((~group["concat_row_kept"]).sum()),
                "strict_all_roles_encodable_count": int(group["strict_all_roles_encodable"].sum()),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_columns = [
        "dataset", "sample_id", "source_row_number", "role", "column", "raw_input", "processed_input",
        "preprocessing", "is_missing", "length", "length_ok", "rdkit_parse_ok", "charset_ok",
        "unknown_characters", "failure_reasons", "preflight_encodable", "contains_dot", "contains_comma",
        "contains_semicolon", "contains_ni", "contains_pd",
    ]
    _write_csv(output_dir / "sample_role_diagnostics.csv", diagnostics, diagnostic_columns)
    _write_csv(
        output_dir / "role_coverage.csv",
        role_coverage,
        [
            "dataset", "role", "column", "cell_count", "missing_count", "rdkit_parse_ok_count",
            "charset_ok_count", "length_ok_count", "preflight_encodable_count", "rdkit_invalid_count",
            "unsupported_character_count", "too_long_count", "contains_dot_count", "contains_comma_count",
            "contains_semicolon_count", "contains_ni_count", "contains_pd_count",
        ],
    )
    _write_csv(
        output_dir / "reaction_coverage.csv",
        reactions,
        [
            "dataset", "sample_id", "source_row_number", "role_count", "preflight_successful_role_count",
            "concat_row_kept", "strict_all_roles_encodable", "final_inference_status",
        ],
    )
    _write_csv(
        output_dir / "reaction_coverage_summary.csv",
        reaction_coverage,
        [
            "dataset", "reaction_count", "concat_kept_count", "concat_dropped_all_roles_failed_count",
            "strict_all_roles_encodable_count",
        ],
    )
    manifest = {
        "generated_at": _utc_now(),
        "scope": "step_31_2_preflight_only",
        "inference_performed": False,
        "model_variant": SUPPORTED_VARIANT,
        "preprocessing_contract": {
            "name": "identity",
            "description": (
                "Use raw CSV cell text exactly. Do not trim, canonicalise, split salts, replace delimiters, truncate, "
                "or expand the frozen character table. Only exactly empty cells are missing."
            ),
        },
        "charset": {"characters": list(charset), "character_count": len(charset), "max_len": max_len},
        "datasets": dataset_manifest,
        "reaction_coverage": reaction_coverage,
        "artifacts": {
            "sample_role_diagnostics": "sample_role_diagnostics.csv",
            "role_coverage": "role_coverage.csv",
            "reaction_coverage": "reaction_coverage.csv",
            "reaction_coverage_summary": "reaction_coverage_summary.csv",
        },
    }
    _write_json(output_dir / "coverage_manifest.json", manifest)
    return manifest


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _backend_decision_markdown(environment: Mapping[str, Any], assets: Mapping[str, Any]) -> str:
    zinc = assets["variants"][SUPPORTED_VARIANT]
    contract = zinc["encoder_contract"]
    return "\n".join(
        [
            "# Step 31.1 backend decision",
            "",
            "## Decision",
            "",
            "Proceed to step 31.3 with a **PyTorch encoder-only reconstruction read from the audited HDF5 asset**. "
            "This is a provisional engineering path, not a numerical-parity acceptance. TensorFlow/Keras remain uninstalled.",
            "",
            "## Evidence",
            "",
            f"- Current PyTorch: `{environment['packages']['torch']}`; retained without downgrade.",
            f"- HDF5 reader: `h5py {environment['packages']['h5py']}`; `pip check` returned "
            f"`{environment['pip_check'].get('returncode')}`.",
            f"- Audited `{SUPPORTED_VARIANT}` input: `{contract.get('input_shape')}`, dtype "
            f"`{contract.get('input_dtype')}`.",
            f"- Audited output: `{contract.get('z_mean_layer')}`, {contract.get('z_mean_units')} linear units.",
            f"- Original encoder SHA-256: `{zinc['assets']['encoder_h5']['sha256']}`.",
            "",
            "## Non-decisions / gates",
            "",
            "- No TensorFlow/Keras or old upstream dependency stack was installed.",
            "- Decoder, TerminalGRU, property head, model-training CSV and latent standardisation are outside this path.",
            "- `zinc_properties` is not selectable. Its encoder was inventoried, but variant/property-head numerical parity is unverified.",
            "- Step 31.3 must still produce a trusted independent reference and frozen tolerances before `chemical_vae` can be exposed as usable.",
            "",
        ]
    )


def _dependency_resolution_markdown(environment: Mapping[str, Any]) -> str:
    """Document the concrete, non-downgrading dependency outcome."""

    return "\n".join(
        [
            "# Step 31.1 dependency resolution",
            "",
            "## Resolved dependency",
            "",
            "- Chemical VAE HDF5 inspection/conversion dependency: `h5py==3.14.0`.",
            f"- Active environment reports: `h5py {environment['packages']['h5py']}` with "
            f"`numpy {environment['packages']['numpy']}` on Python {platform.python_version()}.",
            f"- Post-install `python -m pip check` exit status: `{environment['pip_check'].get('returncode')}`.",
            "- The package is declared in `requirements.txt`; it is used for asset inspection/conversion, not by "
            "ordinary Morgan/RF execution.",
            "",
            "## Compatibility decision",
            "",
            "- `h5py` was selected because its CPython 3.9 wheel resolved against the already-installed NumPy 1.26.4.",
            "- TensorFlow and Keras were not installed.  The obsolete upstream Keras/TensorFlow stack is neither a "
            "project dependency nor a fallback runtime for this path.",
            "- PyTorch remains at its pre-audit installed version; no core package was downgraded.",
            "",
            "The 2026-09-14 dry-run/install command transcript and the no-conflict result are recorded in "
            "`project-docs/buildlog.md`; this snapshot records the reproducible resulting state.",
            "",
        ]
    )


def run_audit(repository_root: Path, reference_root: Path, output_root: Path) -> Dict[str, Any]:
    """Perform the complete 31.1 + 31.2 pre-integration audit."""

    repository_root = repository_root.resolve()
    reference_root = reference_root.resolve()
    output_root = output_root.resolve()
    audit_dir = output_root / "step31_1_audit"
    coverage_dir = output_root / "step31_2_coverage"
    environment = environment_snapshot(repository_root)
    assets = asset_audit(reference_root, repository_root)
    _write_json(audit_dir / "environment_snapshot.json", environment)
    _write_json(audit_dir / "asset_manifest.json", assets)
    (audit_dir / "dependency_resolution.md").write_text(
        _dependency_resolution_markdown(environment), encoding="utf-8"
    )
    (audit_dir / "backend_decision.md").write_text(
        _backend_decision_markdown(environment, assets), encoding="utf-8"
    )
    zinc_charset = assets["variants"][SUPPORTED_VARIANT]["charset"]["characters"]
    zinc_max_len = int(assets["variants"][SUPPORTED_VARIANT]["metadata"]["max_len"])
    coverage = coverage_audit(
        default_coverage_datasets(repository_root),
        charset=zinc_charset,
        max_len=zinc_max_len,
        repository_root=repository_root,
        output_dir=coverage_dir,
    )
    return {
        "audit_dir": str(audit_dir),
        "coverage_dir": str(coverage_dir),
        "environment": environment,
        "assets": assets,
        "coverage": coverage,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="YONOD repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="read-only local chemical_vae reference repository",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory that receives derived/chemical_vae evidence",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_audit(args.repository_root, args.reference_root, args.output_root)
    reaction_summaries = result["coverage"]["reaction_coverage"]
    print(
        json.dumps(
            {
                "status": "complete_preflight_only",
                "audit_dir": result["audit_dir"],
                "coverage_dir": result["coverage_dir"],
                "reaction_coverage": reaction_summaries,
                "numerical_parity": "not_verified",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
