#!/usr/bin/env python3
"""Build shared populations and repeated-KFold splits for step 28.2.

The script reads the step 28.1 inventory and the official precomputed static
descriptor matrices in VJETHBKM.  It writes an isolated, reproducible manifest
package under ``derived/descriptor_model_effect/step28_2_population_splits``.

It deliberately treats CSV row order as candidate evidence only.  The official
NPZ files do not contain row-level identifiers, so the generated
``source_row_index`` values are marked as ``candidate_order_mapping`` rather
than author-embedded sample IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from audit_official_static_descriptors import (  # noqa: E402
    DATASETS,
    DESCRIPTORS,
    sha256_array,
    sha256_file,
    sha256_int_sequence,
)


SCHEMA_VERSION = "1.0"
CV_CONFIG = {"strategy": "repeated_kfold", "n_repeats": 5, "n_splits": 5, "seed": 1000}
DEFAULT_OUTPUT_DIR = "derived/descriptor_model_effect/step28_2_population_splits"
STEP28_1_INVENTORY = (
    "derived/descriptor_model_effect/step28_1_input_audit/"
    "official_static_descriptor_inventory.json"
)


@dataclass(frozen=True)
class StaticPopulationSpec:
    population_id: str
    dataset_id: str
    expected_rows: int
    included_descriptors: tuple[str, ...]
    excluded_descriptors: Mapping[str, str]
    note: str


POPULATIONS = (
    StaticPopulationSpec(
        population_id="BH1-static-3955",
        dataset_id="BH1",
        expected_rows=3955,
        included_descriptors=("SOAP", "PhysChem", "MFP"),
        excluded_descriptors={
            "DFT": "blocked_alignment: BH1/DFT has 3960 rows and no author row-level mapping for the extra five rows",
        },
        note="BH1 DFT remains blocked; SOAP/PhysChem/MFP share the 3955-row candidate CSV order.",
    ),
    StaticPopulationSpec(
        population_id="BH2-static-3359",
        dataset_id="BH2",
        expected_rows=3359,
        included_descriptors=DESCRIPTORS,
        excluded_descriptors={},
        note="All four official static descriptors share the 3359-row candidate CSV order.",
    ),
    StaticPopulationSpec(
        population_id="SL1-static-1150",
        dataset_id="SL1",
        expected_rows=1150,
        included_descriptors=DESCRIPTORS,
        excluded_descriptors={},
        note="All four official static descriptors share the 1150-row candidate CSV order.",
    ),
    StaticPopulationSpec(
        population_id="SM-static-4620",
        dataset_id="SM",
        expected_rows=4620,
        included_descriptors=DESCRIPTORS,
        excluded_descriptors={},
        note="Static SM population only; it must not be mixed with the separate SM-OHE-5760 population.",
    ),
)


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonempty(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def _spec_by_dataset() -> dict[str, Any]:
    return {spec.dataset_id: spec for spec in DATASETS}


def _source_candidate(repo_root: Path, dataset_id: str) -> dict[str, Any]:
    spec = _spec_by_dataset()[dataset_id]
    path = repo_root / spec.source_csv
    selected_rows: list[int] = []
    selected_y: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = (spec.source_target_column, *spec.source_required_columns)
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        source_rows = 0
        for source_row_index, row in enumerate(reader):
            source_rows += 1
            if all(_nonempty(row[column]) for column in required):
                selected_rows.append(source_row_index)
                selected_y.append(float(row[spec.source_target_column]))

    y = np.asarray(selected_y, dtype=np.float64)
    return {
        "source_csv": spec.source_csv,
        "source_csv_sha256": sha256_file(path),
        "source_csv_row_count": source_rows,
        "source_target_column": spec.source_target_column,
        "source_selection_rule": spec.source_selection_rule,
        "target_unit": spec.target_unit,
        "source_row_indices": selected_rows,
        "y": y,
        "candidate_source_row_index_sha256": sha256_int_sequence(selected_rows),
        "candidate_y_sha256": sha256_array(y),
    }


def _load_inventory(repo_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = repo_root / STEP28_1_INVENTORY
    if not path.is_file():
        raise FileNotFoundError(f"Missing step 28.1 inventory: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    by_key = {(entry["dataset_id"], entry["descriptor"]): entry for entry in entries}
    expected = {(spec.dataset_id, descriptor) for spec in DATASETS for descriptor in DESCRIPTORS}
    if set(by_key) != expected:
        raise ValueError("Step 28.1 inventory does not cover exactly 4 datasets x 4 descriptors")
    return by_key


def _load_npz_arrays(repo_root: Path, entry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = repo_root / str(entry["source_npz"])
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["X"]), np.asarray(archive["y"])


def _descriptor_summary(
    repo_root: Path,
    population: StaticPopulationSpec,
    inventory: Mapping[tuple[str, str], Mapping[str, Any]],
    source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    descriptor_rows: list[dict[str, Any]] = []
    descriptor_summary: dict[str, Any] = {}
    descriptor_y: dict[str, np.ndarray] = {}
    source_y = np.asarray(source["y"], dtype=np.float64)
    for descriptor in DESCRIPTORS:
        entry = inventory[(population.dataset_id, descriptor)]
        status = "included" if descriptor in population.included_descriptors else "excluded"
        exclusion_reason = population.excluded_descriptors.get(descriptor, "")
        x_hash = ""
        y_hash = str(entry["y_sha256"])
        y_equal_population = False
        if status == "included":
            X, y = _load_npz_arrays(repo_root, entry)
            x_hash = sha256_array(X)
            y_hash = sha256_array(y)
            descriptor_y[descriptor] = y
            y_equal_population = bool(y.shape == source_y.shape and np.array_equal(y, source_y))
            if X.shape[0] != population.expected_rows or y.shape[0] != population.expected_rows:
                raise ValueError(
                    f"{population.population_id}/{descriptor} rows do not match population: "
                    f"X={X.shape}, y={y.shape}"
                )
            if not y_equal_population:
                raise ValueError(f"{population.population_id}/{descriptor} y is not equal to candidate source y")
            if entry["sample_mapping"].get("npz_embedded_identifier_keys"):
                raise ValueError(f"{population.population_id}/{descriptor} unexpectedly has embedded row IDs")
            if not entry["sample_mapping"].get("source_candidate_y_exact_match"):
                raise ValueError(f"{population.population_id}/{descriptor} lacks exact candidate-y evidence")
        else:
            y_equal_population = bool(
                str(entry.get("y_sha256")) == str(source["candidate_y_sha256"])
                and int(entry.get("y_shape", [0])[0]) == population.expected_rows
            )

        row = {
            "population_id": population.population_id,
            "dataset_id": population.dataset_id,
            "descriptor": descriptor,
            "status": status,
            "main_matrix_inclusion": status == "included",
            "source_npz": entry["source_npz"],
            "source_npz_sha256": entry["source_npz_sha256"],
            "x_shape_json": _json_cell(entry["x_shape"]),
            "x_dtype": entry["x_dtype"],
            "x_sha256": x_hash,
            "y_shape_json": _json_cell(entry["y_shape"]),
            "y_dtype": entry["y_dtype"],
            "y_sha256": y_hash,
            "y_equal_population_y": y_equal_population,
            "mapping_method": "candidate_order_mapping" if status == "included" else "excluded_no_mapping",
            "mapping_evidence_grade": (
                "csv_y_exact_plus_generator_order_evidence_no_npz_row_ids"
                if status == "included"
                else "excluded_descriptor_not_mapped"
            ),
            "npz_embedded_identifier_keys_json": _json_cell(
                entry["sample_mapping"].get("npz_embedded_identifier_keys", [])
            ),
            "exclusion_reason": exclusion_reason,
        }
        descriptor_rows.append(row)
        descriptor_summary[descriptor] = dict(row)
    return descriptor_rows, descriptor_summary, descriptor_y


def _population_rows(
    population: StaticPopulationSpec,
    source: Mapping[str, Any],
    descriptor_summary: Mapping[str, Any],
    population_hash: str,
    mapping_hash: str,
) -> list[dict[str, Any]]:
    availability = {
        descriptor: descriptor in population.included_descriptors for descriptor in DESCRIPTORS
    }
    x_hashes = {
        descriptor: descriptor_summary[descriptor]["x_sha256"]
        for descriptor in DESCRIPTORS
        if descriptor in population.included_descriptors
    }
    y_hashes = {
        descriptor: descriptor_summary[descriptor]["y_sha256"]
        for descriptor in DESCRIPTORS
        if descriptor in population.included_descriptors
    }
    rows: list[dict[str, Any]] = []
    for position, (source_row_index, y_value) in enumerate(
        zip(source["source_row_indices"], np.asarray(source["y"], dtype=np.float64))
    ):
        sample_id = f"{population.population_id}:source-row-{int(source_row_index):06d}"
        rows.append(
            {
                "population_id": population.population_id,
                "dataset_id": population.dataset_id,
                "population_position": position,
                "sample_id": sample_id,
                "source_row_index": int(source_row_index),
                "y": repr(float(y_value)),
                "target_unit": source["target_unit"],
                "candidate_source_csv": source["source_csv"],
                "candidate_source_csv_sha256": source["source_csv_sha256"],
                "candidate_source_row_index_sha256": source["candidate_source_row_index_sha256"],
                "candidate_y_sha256": source["candidate_y_sha256"],
                "mapping_method": "candidate_order_mapping",
                "mapping_evidence_grade": "csv_y_exact_plus_generator_order_evidence_no_npz_row_ids",
                "source_row_identity_status": "candidate_csv_order_not_author_embedded_id",
                "descriptor_availability_json": _json_cell(availability),
                "descriptor_x_sha256_json": _json_cell(x_hashes),
                "descriptor_y_sha256_json": _json_cell(y_hashes),
                "mapping_hash": mapping_hash,
                "population_hash": population_hash,
                "row_exclusion_reason": "",
            }
        )
    return rows


def _build_population(
    repo_root: Path,
    population: StaticPopulationSpec,
    inventory: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    source = _source_candidate(repo_root, population.dataset_id)
    if len(source["source_row_indices"]) != population.expected_rows:
        raise ValueError(
            f"{population.population_id} candidate rows={len(source['source_row_indices'])}, "
            f"expected={population.expected_rows}"
        )
    descriptor_rows, descriptor_summary, descriptor_y = _descriptor_summary(
        repo_root, population, inventory, source
    )
    row_identity = [
        {
            "population_position": int(position),
            "sample_id": f"{population.population_id}:source-row-{int(source_row):06d}",
            "source_row_index": int(source_row),
        }
        for position, source_row in enumerate(source["source_row_indices"])
    ]
    mapping_hash = _hash_payload(
        {
            "population_id": population.population_id,
            "dataset_id": population.dataset_id,
            "source_csv_sha256": source["source_csv_sha256"],
            "source_row_indices": source["source_row_indices"],
            "candidate_y_sha256": source["candidate_y_sha256"],
            "mapping_method": "candidate_order_mapping",
            "mapping_evidence_grade": "csv_y_exact_plus_generator_order_evidence_no_npz_row_ids",
        }
    )
    population_hash = _hash_payload(
        {
            "population_id": population.population_id,
            "dataset_id": population.dataset_id,
            "target_unit": source["target_unit"],
            "row_identity": row_identity,
            "candidate_y_sha256": source["candidate_y_sha256"],
            "included_descriptors": list(population.included_descriptors),
            "excluded_descriptors": dict(population.excluded_descriptors),
            "descriptor_hashes": {
                descriptor: {
                    "source_npz_sha256": descriptor_summary[descriptor]["source_npz_sha256"],
                    "x_sha256": descriptor_summary[descriptor]["x_sha256"],
                    "y_sha256": descriptor_summary[descriptor]["y_sha256"],
                    "status": descriptor_summary[descriptor]["status"],
                }
                for descriptor in DESCRIPTORS
            },
            "mapping_hash": mapping_hash,
        }
    )
    rows = _population_rows(population, source, descriptor_summary, population_hash, mapping_hash)
    source_y = np.asarray(source["y"], dtype=np.float64)
    for descriptor, y in descriptor_y.items():
        if not np.array_equal(y, source_y):
            raise ValueError(f"{population.population_id}/{descriptor} does not share y order")

    return {
        "population": population,
        "source": source,
        "descriptor_rows": descriptor_rows,
        "descriptor_summary": descriptor_summary,
        "population_rows": rows,
        "population_hash": population_hash,
        "mapping_hash": mapping_hash,
    }


def _hash_string_sequence(values: Iterable[str]) -> str:
    return _hash_payload([str(value) for value in values])


def _hash_int_sequence(values: Iterable[int]) -> str:
    return sha256_int_sequence([int(value) for value in values])


def _iter_sklearn_kfold_indices(n_rows: int, n_splits: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield indices matching sklearn KFold(shuffle=True, random_state=seed)."""
    indices = np.arange(n_rows, dtype=np.int64)
    shuffled = indices.copy()
    np.random.RandomState(seed).shuffle(shuffled)
    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=np.int64)
    fold_sizes[: n_rows % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        valid_selection = shuffled[start:stop]
        valid_mask = np.zeros(n_rows, dtype=bool)
        valid_mask[valid_selection] = True
        yield indices[~valid_mask], indices[valid_mask]
        current = stop


def _split_assignments(population_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    n_rows = len(population_rows)
    sample_ids = [str(row["sample_id"]) for row in population_rows]
    source_rows = [int(row["source_row_index"]) for row in population_rows]
    y_values = [float(row["y"]) for row in population_rows]
    population_id = str(population_rows[0]["population_id"])
    dataset_id = str(population_rows[0]["dataset_id"])
    population_hash = str(population_rows[0]["population_hash"])
    dataset_sha256 = str(population_rows[0]["candidate_source_csv_sha256"])

    fold_payloads: list[dict[str, Any]] = []
    for repeat in range(1, CV_CONFIG["n_repeats"] + 1):
        seed = CV_CONFIG["seed"] + repeat - 1
        splitter = _iter_sklearn_kfold_indices(n_rows, CV_CONFIG["n_splits"], seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter, start=1):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            valid_idx = np.asarray(valid_idx, dtype=np.int64)
            train_ids = [sample_ids[int(index)] for index in train_idx]
            valid_ids = [sample_ids[int(index)] for index in valid_idx]
            train_source_rows = [source_rows[int(index)] for index in train_idx]
            valid_source_rows = [source_rows[int(index)] for index in valid_idx]
            fold_payload = {
                "population_id": population_id,
                "dataset_id": dataset_id,
                "population_hash": population_hash,
                "repeat": repeat,
                "fold": fold,
                "seed": seed,
                "train_sample_ids": train_ids,
                "valid_sample_ids": valid_ids,
                "train_source_row_index": train_source_rows,
                "valid_source_row_index": valid_source_rows,
            }
            fold_hash = _hash_payload(fold_payload)
            fold_payloads.append(
                {
                    **fold_payload,
                    "fold_hash": fold_hash,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "train_sample_ids_hash": _hash_string_sequence(train_ids),
                    "valid_sample_ids_hash": _hash_string_sequence(valid_ids),
                    "train_source_row_index_hash": _hash_int_sequence(train_source_rows),
                    "valid_source_row_index_hash": _hash_int_sequence(valid_source_rows),
                    "train_y_hash": sha256_array(np.asarray([y_values[int(index)] for index in train_idx], dtype=np.float64)),
                    "valid_y_hash": sha256_array(np.asarray([y_values[int(index)] for index in valid_idx], dtype=np.float64)),
                }
            )

    split_hash = _hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "cv_config": CV_CONFIG,
            "population_id": population_id,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "population_hash": population_hash,
            "folds": [
                {
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    "seed": fold["seed"],
                    "fold_hash": fold["fold_hash"],
                    "n_train": fold["n_train"],
                    "n_valid": fold["n_valid"],
                    "train_sample_ids_hash": fold["train_sample_ids_hash"],
                    "valid_sample_ids_hash": fold["valid_sample_ids_hash"],
                    "train_source_row_index_hash": fold["train_source_row_index_hash"],
                    "valid_source_row_index_hash": fold["valid_source_row_index_hash"],
                    "train_y_hash": fold["train_y_hash"],
                    "valid_y_hash": fold["valid_y_hash"],
                }
                for fold in fold_payloads
            ],
        }
    )
    split_id = "split-" + split_hash[:12]

    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    role_by_pair: dict[tuple[int, int], tuple[set[int], set[int], dict[str, Any]]] = {}
    for fold in fold_payloads:
        train_positions = {
            int(str(sample_id).rsplit("-", 1)[1])
            for sample_id in fold["train_sample_ids"]
            if str(sample_id).startswith(population_id + ":source-row-")
        }
        valid_positions = {
            int(str(sample_id).rsplit("-", 1)[1])
            for sample_id in fold["valid_sample_ids"]
            if str(sample_id).startswith(population_id + ":source-row-")
        }
        # The sets above are source row numbers.  They are used only for O(1)
        # role lookup while writing the row-level manifest.
        role_by_pair[(int(fold["repeat"]), int(fold["fold"]))] = (
            train_positions,
            valid_positions,
            fold,
        )
        fold_rows.append(
            {
                "population_id": population_id,
                "dataset_id": dataset_id,
                "split_id": split_id,
                "split_hash": split_hash,
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "seed": fold["seed"],
                "n_train": fold["n_train"],
                "n_valid": fold["n_valid"],
                "train_sample_ids_hash": fold["train_sample_ids_hash"],
                "valid_sample_ids_hash": fold["valid_sample_ids_hash"],
                "train_source_row_index_hash": fold["train_source_row_index_hash"],
                "valid_source_row_index_hash": fold["valid_source_row_index_hash"],
                "train_y_hash": fold["train_y_hash"],
                "valid_y_hash": fold["valid_y_hash"],
                "fold_hash": fold["fold_hash"],
                "population_hash": population_hash,
                "dataset_sha256": dataset_sha256,
                "split_protocol": "sklearn.KFold(n_splits=5, shuffle=True, random_state=1000..1004)",
            }
        )

    for repeat in range(1, CV_CONFIG["n_repeats"] + 1):
        for fold in range(1, CV_CONFIG["n_splits"] + 1):
            train_rows, valid_rows, fold_meta = role_by_pair[(repeat, fold)]
            for position, row in enumerate(population_rows):
                source_row_index = int(row["source_row_index"])
                role = "valid" if source_row_index in valid_rows else "train"
                if source_row_index not in train_rows and source_row_index not in valid_rows:
                    raise ValueError(f"{population_id} source row missing from split: {source_row_index}")
                split_rows.append(
                    {
                        "population_id": population_id,
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "split_hash": split_hash,
                        "population_hash": population_hash,
                        "dataset_sha256": dataset_sha256,
                        "repeat": repeat,
                        "fold": fold,
                        "seed": fold_meta["seed"],
                        "role": role,
                        "population_position": position,
                        "sample_id": row["sample_id"],
                        "source_row_index": source_row_index,
                        "y": row["y"],
                        "fold_hash": fold_meta["fold_hash"],
                        "train_sample_ids_hash": fold_meta["train_sample_ids_hash"],
                        "valid_sample_ids_hash": fold_meta["valid_sample_ids_hash"],
                        "train_source_row_index_hash": fold_meta["train_source_row_index_hash"],
                        "valid_source_row_index_hash": fold_meta["valid_source_row_index_hash"],
                        "split_protocol": "repeated independent sklearn.KFold",
                    }
                )
    return split_rows, fold_rows, split_hash


def _write_outputs(repo_root: Path, output_dir: Path, built: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    population_dir = output_dir / "population_manifests"
    split_dir = output_dir / "split_manifests"
    all_descriptor_rows: list[dict[str, Any]] = []
    all_exclusion_rows: list[dict[str, Any]] = []
    population_index_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    split_index_rows: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {
        "population_manifests": {},
        "split_manifests": {},
    }

    population_fields = [
        "population_id",
        "dataset_id",
        "population_position",
        "sample_id",
        "source_row_index",
        "y",
        "target_unit",
        "candidate_source_csv",
        "candidate_source_csv_sha256",
        "candidate_source_row_index_sha256",
        "candidate_y_sha256",
        "mapping_method",
        "mapping_evidence_grade",
        "source_row_identity_status",
        "descriptor_availability_json",
        "descriptor_x_sha256_json",
        "descriptor_y_sha256_json",
        "mapping_hash",
        "population_hash",
        "row_exclusion_reason",
    ]
    descriptor_fields = [
        "population_id",
        "dataset_id",
        "descriptor",
        "status",
        "main_matrix_inclusion",
        "source_npz",
        "source_npz_sha256",
        "x_shape_json",
        "x_dtype",
        "x_sha256",
        "y_shape_json",
        "y_dtype",
        "y_sha256",
        "y_equal_population_y",
        "mapping_method",
        "mapping_evidence_grade",
        "npz_embedded_identifier_keys_json",
        "exclusion_reason",
    ]
    split_fields = [
        "population_id",
        "dataset_id",
        "split_id",
        "split_hash",
        "population_hash",
        "dataset_sha256",
        "repeat",
        "fold",
        "seed",
        "role",
        "population_position",
        "sample_id",
        "source_row_index",
        "y",
        "fold_hash",
        "train_sample_ids_hash",
        "valid_sample_ids_hash",
        "train_source_row_index_hash",
        "valid_source_row_index_hash",
        "split_protocol",
    ]
    fold_fields = [
        "population_id",
        "dataset_id",
        "split_id",
        "split_hash",
        "repeat",
        "fold",
        "seed",
        "n_train",
        "n_valid",
        "train_sample_ids_hash",
        "valid_sample_ids_hash",
        "train_source_row_index_hash",
        "valid_source_row_index_hash",
        "train_y_hash",
        "valid_y_hash",
        "fold_hash",
        "population_hash",
        "dataset_sha256",
        "split_protocol",
    ]

    for item in built:
        population = item["population"]
        source = item["source"]
        rows = item["population_rows"]
        population_path = population_dir / f"{population.population_id}_population_manifest.csv"
        _atomic_csv(population_path, rows, population_fields)
        all_descriptor_rows.extend(item["descriptor_rows"])
        for descriptor, reason in population.excluded_descriptors.items():
            all_exclusion_rows.append(
                {
                    "population_id": population.population_id,
                    "dataset_id": population.dataset_id,
                    "descriptor": descriptor,
                    "status": "excluded",
                    "exclusion_reason": reason,
                    "main_matrix_inclusion": False,
                }
            )
        population_index_rows.append(
            {
                "population_id": population.population_id,
                "dataset_id": population.dataset_id,
                "row_count": len(rows),
                "target_unit": source["target_unit"],
                "included_descriptors_json": _json_cell(list(population.included_descriptors)),
                "excluded_descriptors_json": _json_cell(dict(population.excluded_descriptors)),
                "candidate_source_csv": source["source_csv"],
                "candidate_source_csv_sha256": source["source_csv_sha256"],
                "candidate_source_row_index_sha256": source["candidate_source_row_index_sha256"],
                "candidate_y_sha256": source["candidate_y_sha256"],
                "mapping_method": "candidate_order_mapping",
                "mapping_evidence_grade": "csv_y_exact_plus_generator_order_evidence_no_npz_row_ids",
                "mapping_hash": item["mapping_hash"],
                "population_hash": item["population_hash"],
                "note": population.note,
            }
        )

        split_rows, fold_rows, split_hash = _split_assignments(rows)
        split_path = split_dir / f"{population.population_id}_split_manifest.csv"
        _atomic_csv(split_path, split_rows, split_fields)
        outputs["population_manifests"][population.population_id] = _relative(repo_root, population_path)
        outputs["split_manifests"][population.population_id] = _relative(repo_root, split_path)
        all_fold_rows.extend(fold_rows)
        split_index_rows.append(
            {
                "population_id": population.population_id,
                "dataset_id": population.dataset_id,
                "row_count": len(rows),
                "fold_count": len(fold_rows),
                "split_row_count": len(split_rows),
                "seeds_json": _json_cell(list(range(1000, 1005))),
                "split_hash": split_hash,
                "split_id": "split-" + split_hash[:12],
                "population_hash": item["population_hash"],
            }
        )

    _atomic_csv(output_dir / "population_index.csv", population_index_rows, list(population_index_rows[0]))
    _atomic_csv(output_dir / "descriptor_availability.csv", all_descriptor_rows, descriptor_fields)
    _atomic_csv(
        output_dir / "descriptor_exclusion_manifest.csv",
        all_exclusion_rows,
        ["population_id", "dataset_id", "descriptor", "status", "exclusion_reason", "main_matrix_inclusion"],
    )
    _atomic_csv(output_dir / "fold_manifest.csv", all_fold_rows, fold_fields)
    _atomic_csv(output_dir / "split_index.csv", split_index_rows, list(split_index_rows[0]))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "步骤28.2共享静态描述符population与唯一5x5 split manifest；不执行建模。",
        "source_inventory": STEP28_1_INVENTORY,
        "cv_config": CV_CONFIG,
        "population_count": len(population_index_rows),
        "included_descriptor_matrix_count": sum(len(item["population"].included_descriptors) for item in built),
        "excluded_descriptor_matrix_count": len(all_exclusion_rows),
        "total_fold_count": len(all_fold_rows),
        "total_split_rows": sum(int(row["split_row_count"]) for row in split_index_rows),
        "mapping_policy": (
            "All source_row_index values are candidate CSV-order mappings.  "
            "They are not author-embedded row identifiers because the official NPZ files do not contain row-level IDs."
        ),
        "ohe_policy": "OHE is not part of this static descriptor main manifest; SM-OHE-5760 remains separate.",
        "bh1_dft_policy": "BH1/DFT is excluded until an author row-level mapping resolves the 3960-vs-3955 mismatch.",
        "outputs": {
            **outputs,
            "population_index": _relative(repo_root, output_dir / "population_index.csv"),
            "descriptor_availability": _relative(repo_root, output_dir / "descriptor_availability.csv"),
            "descriptor_exclusion_manifest": _relative(repo_root, output_dir / "descriptor_exclusion_manifest.csv"),
            "fold_manifest": _relative(repo_root, output_dir / "fold_manifest.csv"),
            "split_index": _relative(repo_root, output_dir / "split_index.csv"),
        },
    }
    _atomic_json(output_dir / "manifest_summary.json", summary)
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ensure_split_valid(split_rows: Sequence[Mapping[str, str]], population_rows: Sequence[Mapping[str, str]]) -> None:
    expected_ids = [str(row["sample_id"]) for row in population_rows]
    expected_sources = {str(row["sample_id"]): int(row["source_row_index"]) for row in population_rows}
    expected_set = set(expected_ids)
    repeats = sorted({int(row["repeat"]) for row in split_rows})
    if repeats != list(range(1, 6)):
        raise AssertionError(f"unexpected repeats: {repeats}")
    seeds = sorted({int(row["seed"]) for row in split_rows})
    if seeds != list(range(1000, 1005)):
        raise AssertionError(f"unexpected seeds: {seeds}")
    for repeat in repeats:
        repeat_rows = [row for row in split_rows if int(row["repeat"]) == repeat]
        folds = sorted({int(row["fold"]) for row in repeat_rows})
        if folds != list(range(1, 6)):
            raise AssertionError(f"repeat {repeat} does not have five folds")
        valid_seen: list[str] = []
        for fold in folds:
            part = [row for row in repeat_rows if int(row["fold"]) == fold]
            train = {row["sample_id"] for row in part if row["role"] == "train"}
            valid = {row["sample_id"] for row in part if row["role"] == "valid"}
            if train.intersection(valid):
                raise AssertionError(f"train/valid overlap at repeat={repeat}, fold={fold}")
            if train.union(valid) != expected_set:
                raise AssertionError(f"train/valid union mismatch at repeat={repeat}, fold={fold}")
            valid_seen.extend(sorted(valid))
            for row in part:
                if int(row["source_row_index"]) != expected_sources[row["sample_id"]]:
                    raise AssertionError("split source_row_index does not match population manifest")
        counts = {sample_id: 0 for sample_id in expected_ids}
        for sample_id in valid_seen:
            counts[sample_id] += 1
        if set(counts.values()) != {1}:
            raise AssertionError(f"repeat {repeat} does not validate every sample exactly once")


def verify_outputs(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "manifest_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("population_count") != 4:
        raise AssertionError("step 28.2 must have four static populations")
    if summary.get("included_descriptor_matrix_count") != 15:
        raise AssertionError("step 28.2 must include exactly 15 usable static descriptor matrices")
    if summary.get("excluded_descriptor_matrix_count") != 1:
        raise AssertionError("step 28.2 must exclude only BH1/DFT")

    descriptor_rows = _read_csv(output_dir / "descriptor_availability.csv")
    included = [row for row in descriptor_rows if row["status"] == "included"]
    excluded = [row for row in descriptor_rows if row["status"] == "excluded"]
    if len(included) != 15 or len(excluded) != 1:
        raise AssertionError("descriptor availability inclusion/exclusion count mismatch")
    if not any(row["dataset_id"] == "BH1" and row["descriptor"] == "DFT" for row in excluded):
        raise AssertionError("BH1/DFT must be the excluded descriptor")
    if any(row["descriptor"] == "OHE" for row in descriptor_rows):
        raise AssertionError("OHE must not enter the static descriptor manifest")
    if not all(row["mapping_method"] == "candidate_order_mapping" for row in included):
        raise AssertionError("included descriptors must use candidate_order_mapping")

    population_index = _read_csv(output_dir / "population_index.csv")
    split_index = _read_csv(output_dir / "split_index.csv")
    split_index_by_population = {row["population_id"]: row for row in split_index}
    total_folds = 0
    total_split_rows = 0
    for row in population_index:
        population_id = row["population_id"]
        population_rows = _read_csv(output_dir / "population_manifests" / f"{population_id}_population_manifest.csv")
        if len(population_rows) != int(row["row_count"]):
            raise AssertionError(f"{population_id} population row count mismatch")
        if len({item["sample_id"] for item in population_rows}) != len(population_rows):
            raise AssertionError(f"{population_id} sample_id is not unique")
        if not all(item["source_row_identity_status"] == "candidate_csv_order_not_author_embedded_id" for item in population_rows):
            raise AssertionError(f"{population_id} overclaims source-row identity")
        split_rows = _read_csv(output_dir / "split_manifests" / f"{population_id}_split_manifest.csv")
        _ensure_split_valid(split_rows, population_rows)
        _expected_split_rows, _expected_fold_rows, expected_split_hash = _split_assignments(population_rows)
        split_hashes = {item["split_hash"] for item in split_rows}
        if split_hashes != {split_index_by_population[population_id]["split_hash"]}:
            raise AssertionError(f"{population_id} split_hash mismatch")
        if expected_split_hash != split_index_by_population[population_id]["split_hash"]:
            raise AssertionError(f"{population_id} split_hash is not reproducible")
        fold_pairs = {(item["repeat"], item["fold"]) for item in split_rows}
        if len(fold_pairs) != 25:
            raise AssertionError(f"{population_id} must have 25 repeat/fold pairs")
        total_folds += 25
        total_split_rows += len(split_rows)

    if total_folds != int(summary["total_fold_count"]):
        raise AssertionError("fold total mismatch")
    if total_split_rows != int(summary["total_split_rows"]):
        raise AssertionError("split row total mismatch")

    # Smoke-check the first fold against every included official matrix.  Since
    # the NPZ files do not carry IDs, fold membership is applied by manifest
    # population position only after the candidate y order has been verified.
    inventory = _load_inventory(repo_root)
    for row in population_index:
        population_id = row["population_id"]
        dataset_id = row["dataset_id"]
        population_rows = _read_csv(output_dir / "population_manifests" / f"{population_id}_population_manifest.csv")
        y_population = np.asarray([float(item["y"]) for item in population_rows], dtype=np.float64)
        split_rows = _read_csv(output_dir / "split_manifests" / f"{population_id}_split_manifest.csv")
        fold_one = [
            item for item in split_rows if int(item["repeat"]) == 1 and int(item["fold"]) == 1 and item["role"] == "valid"
        ]
        valid_positions = [int(item["population_position"]) for item in fold_one]
        for descriptor_row in included:
            if descriptor_row["population_id"] != population_id:
                continue
            X, y = _load_npz_arrays(repo_root, inventory[(dataset_id, descriptor_row["descriptor"])])
            if X.shape[0] != len(population_rows):
                raise AssertionError(f"{population_id}/{descriptor_row['descriptor']} X rows mismatch")
            if not np.array_equal(y, y_population):
                raise AssertionError(f"{population_id}/{descriptor_row['descriptor']} y order mismatch")
            if not np.array_equal(y[valid_positions], y_population[valid_positions]):
                raise AssertionError(f"{population_id}/{descriptor_row['descriptor']} fold smoke y mismatch")

    return {
        "status": "passed",
        "population_count": 4,
        "included_descriptor_matrix_count": 15,
        "excluded_descriptor_matrix_count": 1,
        "total_fold_count": total_folds,
        "total_split_rows": total_split_rows,
        "seeds": list(range(1000, 1005)),
        "bh1_dft_status": "excluded",
        "ohe_in_static_manifest": False,
    }


def build(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    inventory = _load_inventory(repo_root)
    built = [_build_population(repo_root, population, inventory) for population in POPULATIONS]
    summary = _write_outputs(repo_root, output_dir, built)
    verification = verify_outputs(repo_root, output_dir)
    _atomic_json(output_dir / "verification_summary.json", verification)
    return {"status": "built", "summary": summary, "verification": verification}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Repository-relative output directory for step 28.2 artifacts.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing outputs without rebuilding.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    try:
        output_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("--output-dir must stay inside the repository") from exc

    result = verify_outputs(repo_root, output_dir) if args.verify else build(repo_root, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
