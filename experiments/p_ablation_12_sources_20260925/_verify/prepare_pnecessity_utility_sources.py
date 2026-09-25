"""Prepare the paired, within-source product-utility populations.

This utility is deliberately upstream of descriptors and models.  It reads the
14 user-selected local source tables, canonicalizes explicitly mapped molecular
roles, normalizes yield to [0, 1], and writes immutable source-specific CSVs.
The same CSV is the only permitted input for the source's Full and minus-P
arms.  Sources whose variable numeric C fields cannot be consumed by the
current strict benchmark feature path are retained and explicitly marked
``capability_blocked``; no condition field is silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.splits.manifest import create_split_manifest, validate_split_manifest


TASK_NAME = "pnecessity_utility_prepare_v1"
SEED = 20260918
N_SPLITS = 5
N_REPEATS = 3

# Existing strict benchmarking consumes only molecular descriptor columns.  It
# validates schema ``conditions`` but does not pass them into the feature matrix.
# Keep this explicit, so a future feature-path extension changes one audited
# capability declaration rather than accidentally reclassifying a source.
STRICT_VARIABLE_NUMERIC_C_SUPPORTED = False


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    a: str
    b: str
    p: str
    label: str
    label_scale: float
    molecular_c: tuple[str, ...] = ()
    numeric_c: tuple[str, ...] = ()
    source_key: str | None = None
    family: str = "ORD"
    published_fixed_split: Mapping[str, Any] | None = None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("ord_ahneman", "dataset/ORD/ahneman_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "reagent-2", "catalyst-1", "solvent-1", "solvent-2", "solvent-3", "solvent-4", "solvent-5"), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_c_n", "dataset/ORD/c_n_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "catalyst-1", "catalyst-2", "solvent-1", "solvent-2", "solvent-3", "solvent-4", "solvent-5", "solvent-6", "solvent-7", "solvent-8", "solvent-9", "solvent-10"), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_chan_lam", "dataset/ORD/chan_lam_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "catalyst-1", "solvent-1", "solvent-2"), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_chemrxiv_ch_arylation", "dataset/ORD/chemrxiv_ch_arylation_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "reagent-2", "catalyst-1", "catalyst-2", "solvent-1", "solvent-2", "solvent-3"), ("temperature_c", "reaction_time_s", "reflux_value", "conditions_are_dynamic_value")),
    SourceSpec("ord_nano", "dataset/ORD/nano_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("catalyst-1", "catalyst-2", "catalyst-3", "catalyst-4", "solvent-1", "solvent-2", "solvent-3"), ("temperature_c", "reaction_time_s", "illumination_wavelength_nm", "reflux_value")),
    SourceSpec("ord_nicolit", "dataset/ORD/nicolit_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               (), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_roche_borylation", "dataset/ORD/roche_borylation_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reagent-1", "product", "yield_percent", 100.0,
               ("reagent-2", "reagent-3", "reagent-4", "reagent-5", "catalyst-1", "catalyst-2", "solvent-1", "solvent-2"), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_shields", "dataset/ORD/shields_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "catalyst-1", "catalyst-2", "solvent-1"), ("temperature_c", "reaction_time_s")),
    SourceSpec("ord_suzuki", "dataset/ORD/suzuki_yield_structure_v1_normalized_dataset.csv", "reactant-1", "reactant-2", "product", "yield_percent", 100.0,
               ("reagent-1", "catalyst-1", "catalyst-2", "solvent-1", "solvent-2", "solvent-3", "solvent-4", "solvent-5", "solvent-6", "solvent-7"), ("temperature_c", "pressure_kpa", "reaction_time_s", "conditions_are_dynamic_value")),
    SourceSpec("liao_dhp", "dataset/廖矿标课题组/DHP 衍生物催化羧酸脱羧硒化/DHP_catalyzed_decarboxylative_selenation.csv", "Substrate_1", "Substrate_2", "Product", "Yield", 1.0,
               ("Catalyst", "Solvent"), (), "Entry", "Liao"),
    SourceSpec("liao_ir_oh_insertion", "dataset/廖矿标课题组/Ir 催化羧酸与亚砜叶立德选择性 O–H 插入/Ir_catalyzed_OH_insertion.csv", "acid", "ylide", "product", "yield (%)", 100.0,
               (), (), "Well", "Liao", {
                   "protocol": "published_model_external_split",
                   "model_path": "dataset/廖矿标课题组/Ir 催化羧酸与亚砜叶立德选择性 O–H 插入/model_set (412).xlsx",
                   "model_sha256": "862d226639b3f103a1a01d1268106e30ea4df6cc181de1c9b266f6beecf51cbd",
                   "model_rows": 412,
                   "external_path": "dataset/廖矿标课题组/Ir 催化羧酸与亚砜叶立德选择性 O–H 插入/external_set (235).xlsx",
                   "external_sha256": "2f112eba355684927e4f429f0f8409abea5d046d455c06d59be026b3963e355c",
                   "external_rows": 235,
                   "raw_rows_outside_published_split": 6,
                   "canonical_role_overlap_train_external": {"acid": 80, "ylide": 5, "product": 2},
                   "full_reaction_overlap_train_external": 0,
               }),
    SourceSpec("liao_ru_oh_insertion", "dataset/廖矿标课题组/Ru（钌）催化亚砜叶立德对膦酸 O–H 键插入/Ru_catalyzed_insertion.csv", "sulfoxonium ylide", "P(O)OH", "product", "yield (%)", 100.0,
               (), (), "Well", "Liao"),
    SourceSpec("liao_pd_ch_functionalization", "dataset/廖矿标课题组/光诱导钯催化 C–H 官能团化三步串联/Pd_catalyzed_CH_functionalization.csv", "reactant1", "reactant2", "product", "yield", 1.0,
               (), (), "Unnamed: 0", "Liao"),
    SourceSpec("liao_rh_ch_amidation", "dataset/廖矿标课题组/铑催化的芳醛、腙的邻位C-H酰胺化反应/all_HTE_set.csv", "hydrazone", "dioxazolone", "product", "yield (%)", 100.0,
               (), (), "Unnamed: 0", "Liao"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    if not name:
        raise ValueError(f"cannot normalize blank column name: {value!r}")
    return name


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        checked = pd.read_parquet(temporary)
        if len(checked) != len(frame) or list(checked.columns) != list(frame.columns):
            raise RuntimeError("staged parquet verification failed")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


class _Canonicalizer:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, str]] = {}

    def convert(self, raw_value: Any) -> tuple[str, str]:
        raw = _text(raw_value)
        if raw in self._cache:
            return self._cache[raw]
        if not raw:
            result = ("", "blank")
        else:
            molecule = Chem.MolFromSmiles(raw.replace(",", "."))
            if molecule is None:
                result = ("", "invalid")
            else:
                result = (Chem.MolToSmiles(molecule, canonical=True), "valid")
        self._cache[raw] = result
        return result


@dataclass(frozen=True)
class _SplitConfig:
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


def _assert_known_schema(raw: pd.DataFrame, spec: SourceSpec) -> None:
    required = {spec.a, spec.b, spec.p, spec.label, *spec.molecular_c, *spec.numeric_c}
    if spec.source_key:
        required.add(spec.source_key)
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{spec.source_id}: mapped source columns absent: {missing}")
    identifiers = {spec.source_key} if spec.source_key else set()
    mapped = {spec.a, spec.b, spec.p, spec.label, *spec.molecular_c, *spec.numeric_c, *identifiers}
    extras = sorted(set(raw.columns).difference(mapped))
    if extras:
        raise ValueError(
            f"{spec.source_id}: unmapped source fields would violate all-non-P preservation: {extras}"
        )


def _prepare_source(spec: SourceSpec, output: Path) -> tuple[dict[str, Any], pd.DataFrame | None]:
    source = ROOT / spec.path
    if not source.is_file():
        raise FileNotFoundError(f"{spec.source_id}: source does not exist: {source}")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")
    _assert_known_schema(raw, spec)
    canonicalizer = _Canonicalizer()
    records: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    status_counts: dict[str, dict[str, int]] = {role: {"valid": 0, "blank": 0, "invalid": 0} for role in ("a", "b", "p", *spec.molecular_c)}
    numeric_values: dict[str, set[str]] = {column: set() for column in spec.numeric_c}
    numeric_missing: dict[str, int] = {column: 0 for column in spec.numeric_c}

    for source_row_index, (_, row) in enumerate(raw.iterrows()):
        molecule_values: dict[str, tuple[str, str, str]] = {}
        for role, column in (("a", spec.a), ("b", spec.b), ("p", spec.p)):
            raw_value = _text(row[column])
            canonical, status = canonicalizer.convert(raw_value)
            molecule_values[role] = (raw_value, canonical, status)
            status_counts[role][status] += 1
        for column in spec.molecular_c:
            raw_value = _text(row[column])
            canonical, status = canonicalizer.convert(raw_value)
            molecule_values[column] = (raw_value, canonical, status)
            status_counts[column][status] += 1

        normalized_yield = pd.to_numeric(pd.Series([row[spec.label]]), errors="coerce").iloc[0]
        normalized_yield = float(normalized_yield) / spec.label_scale if pd.notna(normalized_yield) else float("nan")
        core_failures = [role for role in ("a", "b", "p") if molecule_values[role][2] != "valid"]
        label_ok = pd.notna(normalized_yield) and 0.0 <= normalized_yield <= 1.0
        if core_failures or not label_ok:
            rejection_rows.append({
                "source_id": spec.source_id, "source_row_index": source_row_index,
                "reason": "invalid_core:" + ",".join(core_failures) if core_failures else "invalid_label",
            })
            continue

        source_key = _text(row[spec.source_key]) if spec.source_key else str(source_row_index)
        item: dict[str, Any] = {
            "sample_id": f"{spec.source_id}:{source_row_index + 1:06d}",
            "source_id": spec.source_id,
            "source_row_index": source_row_index,
            "source_row_key": source_key,
            "yield": normalized_yield,
            "a_smiles": molecule_values["a"][1],
            "b_smiles": molecule_values["b"][1],
            "p_smiles": molecule_values["p"][1],
            "a_source_smiles": molecule_values["a"][0],
            "b_source_smiles": molecule_values["b"][0],
            "p_source_smiles": molecule_values["p"][0],
        }
        non_p_payload: dict[str, str] = {
            "a_smiles": item["a_smiles"], "b_smiles": item["b_smiles"],
        }
        for column in spec.molecular_c:
            name = "c_" + _safe_name(column)
            raw_value, canonical, status = molecule_values[column]
            item[name + "_smiles"] = canonical
            item[name + "_source"] = raw_value
            item[name + "_status"] = status
            non_p_payload[name + "_canonical"] = canonical
            non_p_payload[name + "_source"] = raw_value
        for column in spec.numeric_c:
            name = "c_" + _safe_name(column)
            value = _text(row[column])
            item[name] = value
            non_p_payload[name] = value
            if value:
                numeric_values[column].add(value)
            else:
                numeric_missing[column] += 1
        item["non_p_input_group_id"] = "nonp-" + _stable_hash(non_p_payload)[:24]
        records.append(item)

    prepared = pd.DataFrame.from_records(records)
    if prepared.empty:
        raise ValueError(f"{spec.source_id}: all source rows failed label/core validation")
    if prepared["sample_id"].duplicated().any():
        raise ValueError(f"{spec.source_id}: prepared sample_id is not unique")
    csv_path = output / "data" / f"{spec.source_id}.csv"
    _atomic_csv(prepared, csv_path)
    _atomic_csv(pd.DataFrame.from_records(rejection_rows, columns=["source_id", "source_row_index", "reason"]), output / "rejections" / f"{spec.source_id}.csv")

    dynamic_numeric = [column for column in spec.numeric_c if len(numeric_values[column]) > 1]
    fixed_numeric = [column for column in spec.numeric_c if len(numeric_values[column]) <= 1]
    usable_molecular_c = ["c_" + _safe_name(column) + "_smiles" for column in spec.molecular_c]
    c_invalid = {column: status_counts[column]["invalid"] for column in spec.molecular_c}
    c_blank = {column: status_counts[column]["blank"] for column in spec.molecular_c}
    status = "configurable"
    reason = "all varying recorded fields are molecular; numeric C is fixed or unrecorded"
    if spec.published_fixed_split is not None:
        status = "capability_blocked"
        reason = (
            "the source requires its published 412/235 model/external split; manifest_outer_cv cannot encode one fixed holdout, "
            "and frozen_train_external_test rejects the published train/external role overlap required by its independent-source contract"
        )
    elif dynamic_numeric and not STRICT_VARIABLE_NUMERIC_C_SUPPORTED:
        status = "capability_blocked"
        reason = (
            "current strict benchmark runtime validates but does not consume "
            "dataset.column_roles.conditions; variable numeric C cannot be kept in the model matrix"
        )
    report = {
        "source_id": spec.source_id,
        "source_family": spec.family,
        "raw_path": spec.path,
        "raw_sha256": _sha256(source),
        "raw_rows": int(len(raw)),
        "prepared_path": str(csv_path.relative_to(ROOT)),
        "prepared_sha256": _sha256(csv_path),
        "prepared_rows": int(len(prepared)),
        "dropped_rows": int(len(rejection_rows)),
        "normalization": {"yield_source_column": spec.label, "yield_scale_divisor": spec.label_scale, "yield_unit": "fraction_0_to_1", "molecular": "RDKit canonical isomeric SMILES; source text retained"},
        "role_mapping": {
            "a": spec.a, "b": spec.b, "p": spec.p,
            "molecular_c": list(spec.molecular_c), "numeric_c": list(spec.numeric_c),
            "source_row_key": spec.source_key or "source_row_index",
        },
        "prepared_columns": {"a": "a_smiles", "b": "b_smiles", "p": "p_smiles", "molecular_c": usable_molecular_c, "numeric_c": ["c_" + _safe_name(column) for column in spec.numeric_c], "group": "non_p_input_group_id"},
        "non_p_group_definition": "SHA-256 of canonical A/B plus every recorded non-P molecular C canonical/source text and numeric C text; never P",
        "quality": {
            "core_status_counts": {role: status_counts[role] for role in ("a", "b", "p")},
            "molecular_c_invalid_rows": c_invalid, "molecular_c_blank_rows": c_blank,
            "n_non_p_input_groups": int(prepared["non_p_input_group_id"].nunique()),
            "largest_non_p_input_group_rows": int(prepared["non_p_input_group_id"].value_counts().max()),
            "duplicate_non_p_input_rows": int(prepared.duplicated(subset=["non_p_input_group_id"], keep=False).sum()),
            "n_unique_a": int(prepared["a_smiles"].nunique()), "n_unique_b": int(prepared["b_smiles"].nunique()), "n_unique_p": int(prepared["p_smiles"].nunique()),
        },
        "conditions": {
            "fixed_or_unrecorded_numeric_c": fixed_numeric,
            "variable_numeric_c": dynamic_numeric,
            "numeric_unique_nonblank": {column: len(numeric_values[column]) for column in spec.numeric_c},
            "numeric_missing_rows": numeric_missing,
            "runtime_capability": "variable numeric conditions unsupported by current strict descriptor feature path" if dynamic_numeric else "no variable numeric condition needs feature-path support",
        },
        "config_status": status,
        "config_status_reason": reason,
        "published_fixed_split": dict(spec.published_fixed_split) if spec.published_fixed_split is not None else None,
        "limitations": [
            "within-source engineering utility only; neither rows nor fold metrics may be pooled across sources",
            "source uses the locally available snapshot; local file hash does not establish upstream provenance",
            "low molecular diversity and duplicate non-P inputs remain source-specific limits",
        ],
    }
    return report, prepared if status == "configurable" else None


def _make_manifest(source_id: str, frame: pd.DataFrame, prepared_sha: str, output: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config = _SplitConfig(
        frame=frame,
        sample_id_col="sample_id",
        grouping={"strategy": "precomputed_column", "group_column": "non_p_input_group_id", "protocol_label": "pnecessity_within_source_non_p_input", "max_group_fraction": 0.8},
        cv={"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED},
        raw={"population_id": f"pnecessity-{source_id}-{prepared_sha[:16]}"},
        smiles_cols=tuple(column for column in frame.columns if column.endswith("_smiles")),
    )
    contract = _SplitContract(config=config, run_id=f"pnecessity-{source_id}-nonp-split-v1", dataset_sha256=prepared_sha)
    manifest = create_split_manifest(contract)
    validate_split_manifest(manifest, n_splits=N_SPLITS, n_repeats=N_REPEATS)
    path = output / "manifests" / f"{source_id}_non_p_input_split_manifest.parquet"
    _atomic_parquet(manifest, path)
    audit_rows: list[dict[str, Any]] = []
    indexed = frame.set_index("sample_id")
    for (repeat, fold), part in manifest.groupby(["repeat", "fold"], sort=True):
        train_ids = part.loc[part["role"].eq("train"), "sample_id"].tolist()
        valid_ids = part.loc[part["role"].eq("valid"), "sample_id"].tolist()
        overlap = set(indexed.loc[train_ids, "non_p_input_group_id"]).intersection(indexed.loc[valid_ids, "non_p_input_group_id"])
        if overlap:
            raise RuntimeError(f"{source_id}: non-P group leakage in repeat={repeat}, fold={fold}")
        audit_rows.append({"source_id": source_id, "repeat": int(repeat), "fold": int(fold), "n_train": len(train_ids), "n_valid": len(valid_ids), "non_p_group_overlap": len(overlap)})
    return {
        "path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "split_id": str(manifest["split_id"].iloc[0]),
        "n_rows": int(len(manifest)), "n_folds": N_SPLITS * N_REPEATS,
        "group_column": "non_p_input_group_id", "group_definition_has_p": False,
    }, pd.DataFrame.from_records(audit_rows)


def prepare(output: Path) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    reports: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    for spec in SOURCES:
        report, configurable = _prepare_source(spec, output)
        if configurable is not None:
            manifest, audit = _make_manifest(spec.source_id, configurable, report["prepared_sha256"], output)
            report["split_manifest"] = manifest
            audit_rows.append(audit)
        else:
            report["split_manifest"] = None
        reports.append(report)
    summary = {
        "task_kind": "standardized paired population preparation; no descriptors or model fitting",
        "task_name": TASK_NAME,
        "protocol": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED, "grouping": "non_p_input_group_id"},
        "current_runtime_capability": {"variable_numeric_conditions_consumed": STRICT_VARIABLE_NUMERIC_C_SUPPORTED, "evidence": "_verify/run_benchmark.py passes X_numeric=None to fold execution; only molecular descriptor component columns are materialized"},
        "sources": reports,
        "summary": {"n_requested_sources": len(SOURCES), "n_configurable_sources": sum(item["config_status"] == "configurable" for item in reports), "n_capability_blocked_sources": sum(item["config_status"] == "capability_blocked" for item in reports), "n_expected_configs": sum(item["config_status"] == "configurable" for item in reports) * 4},
    }
    _atomic_json(summary, output / "source_manifest.json")
    manifest_rows = []
    for item in reports:
        manifest_rows.append({
            "source_id": item["source_id"], "config_status": item["config_status"], "config_status_reason": item["config_status_reason"],
            "raw_path": item["raw_path"], "raw_sha256": item["raw_sha256"], "raw_rows": item["raw_rows"],
            "prepared_path": item["prepared_path"], "prepared_sha256": item["prepared_sha256"], "prepared_rows": item["prepared_rows"], "dropped_rows": item["dropped_rows"],
            "variable_numeric_c": json.dumps(item["conditions"]["variable_numeric_c"], ensure_ascii=False),
            "fixed_or_unrecorded_numeric_c": json.dumps(item["conditions"]["fixed_or_unrecorded_numeric_c"], ensure_ascii=False),
            "n_non_p_input_groups": item["quality"]["n_non_p_input_groups"],
            "split_manifest_path": item["split_manifest"]["path"] if item["split_manifest"] else "",
        })
    _atomic_csv(pd.DataFrame.from_records(manifest_rows), output / "source_manifest.csv")
    _atomic_csv(pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame(), output / "audits" / "non_p_group_fold_audit.csv")
    _atomic_json({"all_source_csv_sha256": {item["source_id"]: item["prepared_sha256"] for item in reports}, "source_manifest_sha256": _sha256(output / "source_manifest.csv")}, output / "run_manifest.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare paired product-utility source populations without model work.")
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--overwrite", action="store_true", help="replace only this generated task directory")
    args = parser.parse_args()
    output = ROOT / "result" / str(args.task_name)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}; pass --overwrite only for this generated task")
    if output.exists() and args.overwrite:
        # This command never touches inputs.  Replace individual files atomically;
        # retained stale files are removed only within its named result directory.
        import shutil
        shutil.rmtree(output)
    report = prepare(output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
