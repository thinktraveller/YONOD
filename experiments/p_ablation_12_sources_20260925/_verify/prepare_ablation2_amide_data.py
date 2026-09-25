"""Prepare an immutable, audited input table for ablation2 amide research.

This is a data-preparation utility, not a modelling entry point.  It never
changes the source CSV and writes every generated file below one explicitly
named ``result/<task_name>/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "廖矿标课题组" / "酰胺缩合产率" / "amide-coupling.csv"
ROLE_COLUMNS = {
    "amine": "sub_1_smiles",
    "acid": "sub_2_smiles",
    "product": "product_smiles",
    "activation": "activation",
    "additive": "additive",
    "base": "base",
    "solvent": "solvent",
}
CORE_ROLES = ("amine", "acid", "product")
CONDITION_ROLES = ("activation", "additive", "base", "solvent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(*parts: str) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_smiles(value: Any) -> tuple[str, str, str]:
    """Return ``(status, normalized_source, audit_key)`` without data loss."""
    if value is None or pd.isna(value) or not str(value).strip():
        return "blank_source_value", "", "missing"
    raw = str(value).strip()
    normalized = raw.replace(",", ".")
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return "invalid_smiles", normalized, "invalid:" + _stable_key(normalized)
    return "valid", normalized, "smiles:" + Chem.MolToSmiles(molecule, canonical=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _role_heuristics(frame: pd.DataFrame) -> dict[str, int]:
    """Audit chemistry-role evidence without making it a model filter."""
    amine_with_n = 0
    acid_with_carboxyl = 0
    product_with_amide = 0
    carboxyl = Chem.MolFromSmarts("C(=O)[O;H1,-1]")
    amide = Chem.MolFromSmarts("C(=O)N")
    for row in frame.itertuples(index=False):
        amine = Chem.MolFromSmiles(str(row.amine_smiles).replace(",", "."))
        acid = Chem.MolFromSmiles(str(row.acid_smiles).replace(",", "."))
        product = Chem.MolFromSmiles(str(row.product_smiles).replace(",", "."))
        amine_with_n += int(amine is not None and any(atom.GetAtomicNum() == 7 for atom in amine.GetAtoms()))
        acid_with_carboxyl += int(acid is not None and carboxyl is not None and acid.HasSubstructMatch(carboxyl))
        product_with_amide += int(product is not None and amide is not None and product.HasSubstructMatch(amide))
    return {
        "amine_rows_with_nitrogen": amine_with_n,
        "acid_rows_with_carboxyl_pattern": acid_with_carboxyl,
        "product_rows_with_amide_pattern": product_with_amide,
    }


def prepare(source: Path, output_root: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"source CSV does not exist: {source}")
    raw = pd.read_csv(source, keep_default_na=False, encoding="utf-8")
    required = {"row_id", "yield", *ROLE_COLUMNS.values()}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"source CSV missing required columns: {', '.join(missing)}")
    if raw["row_id"].isna().any() or raw["row_id"].astype(str).duplicated().any():
        raise ValueError("row_id must be a unique non-null source identifier")

    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for _, row in raw.iterrows():
        source_id = str(row["row_id"])
        target = pd.to_numeric(pd.Series([row["yield"]]), errors="coerce").iloc[0]
        item: dict[str, Any] = {
            "sample_id": f"amide2:{source_id}",
            "source_row_id": source_id,
            "yield": float(target) if pd.notna(target) else None,
        }
        for role, column in ROLE_COLUMNS.items():
            raw_value = row[column]
            raw_text = "" if pd.isna(raw_value) else str(raw_value)
            status, normalized, key = _normalise_smiles(raw_text)
            item[f"{role}_smiles"] = raw_text
            item[f"{role}_normalized_smiles"] = normalized
            item[f"{role}_structure_status"] = status
            item[f"{role}_structure_key"] = key
            if role in CONDITION_ROLES:
                item[f"{role}_source_presence"] = "blank_source_value" if status == "blank_source_value" else "reported_value"
            if status == "invalid_smiles":
                issues.append({
                    "sample_id": item["sample_id"], "role": role,
                    "raw_value": raw_text, "normalized_value": normalized,
                    "status": status,
                })
        item["label_status"] = "valid" if item["yield"] is not None and 0.0 <= item["yield"] <= 1.0 else "invalid_label"
        item["core_structure_status"] = "valid" if all(
            item[f"{role}_structure_status"] == "valid" for role in CORE_ROLES
        ) else "invalid_core_structure"
        item["condition_any_usable"] = any(
            item[f"{role}_structure_status"] == "valid" for role in CONDITION_ROLES
        )
        item["population_status"] = (
            "included" if item["label_status"] == "valid" and item["core_structure_status"] == "valid" and item["condition_any_usable"]
            else "excluded"
        )
        item["amine_group_id"] = "amine-" + _stable_key(item["amine_structure_key"])
        item["acid_group_id"] = "acid-" + _stable_key(item["acid_structure_key"])
        item["substrate_pair_group_id"] = "pair-" + _stable_key(
            item["amine_structure_key"], item["acid_structure_key"]
        )
        item["repeat_group_id"] = "full-input-" + _stable_key(*[
            item[f"{role}_structure_key"] for role in ROLE_COLUMNS
        ])
        records.append(item)

    standard = pd.DataFrame.from_records(records)
    included = standard.loc[standard["population_status"].eq("included")].copy()
    if included.empty:
        raise ValueError("data audit excluded every row")
    if included["sample_id"].duplicated().any():
        raise ValueError("prepared sample_id is not unique")

    duplicates = included.groupby("repeat_group_id", sort=True).agg(
        n_records=("sample_id", "size"),
        n_distinct_yields=("yield", "nunique"),
        yield_min=("yield", "min"),
        yield_max=("yield", "max"),
    ).reset_index()
    duplicate_groups = duplicates.loc[duplicates["n_records"] > 1].copy()
    exclusions = standard.loc[standard["population_status"].ne("included")].copy()
    pair_to_product = included.groupby("substrate_pair_group_id")["product_structure_key"].nunique()

    data_dir = output_root / "data"
    _atomic_csv(standard, data_dir / "standardized_all_rows.csv")
    _atomic_csv(included, data_dir / "standardized_population.csv")
    _atomic_csv(exclusions, data_dir / "exclusions.csv")
    _atomic_csv(pd.DataFrame.from_records(issues, columns=["sample_id", "role", "raw_value", "normalized_value", "status"]), data_dir / "invalid_smiles.csv")
    _atomic_csv(duplicate_groups, data_dir / "duplicate_input_groups.csv")
    _atomic_csv(
        included[["sample_id", "source_row_id", "amine_group_id", "acid_group_id", "substrate_pair_group_id", "repeat_group_id"]],
        data_dir / "population_manifest.csv",
    )

    standard_sha = _sha256(data_dir / "standardized_population.csv")
    report = {
        "task_kind": "ablation2 data preparation; no model fitting",
        "source": {"path": str(source.relative_to(ROOT)), "sha256": _sha256(source), "rows": int(len(raw))},
        "standardized_population": {
            "path": str((data_dir / "standardized_population.csv").relative_to(ROOT)),
            "sha256": standard_sha,
            "n_rows": int(len(included)),
            "population_id": "amide2-population-" + standard_sha[:16],
            "sample_id_semantics": "amide2:<immutable source row_id>",
        },
        "role_mapping": {
            "A": {"semantic_role": "amine", "source_column": "sub_1_smiles"},
            "B": {"semantic_role": "carboxylic acid", "source_column": "sub_2_smiles"},
            "P": {"semantic_role": "amide product", "source_column": "product_smiles"},
            "C": {"semantic_role": "reported activation/additive/base/solvent molecular inputs", "source_columns": [ROLE_COLUMNS[r] for r in CONDITION_ROLES]},
        },
        "role_heuristics": _role_heuristics(included),
        "label": {
            "column": "yield", "unit": "fraction", "range": [float(included["yield"].min()), float(included["yield"].max())],
            "invalid_or_out_of_range_rows": int(standard["label_status"].ne("valid").sum()),
        },
        "conditions": {
            "blank_semantics": "blank_source_value means the source has no field value; it is not asserted to mean a chemically confirmed no-addition control",
            "status_counts": {
                role: dict(Counter(standard[f"{role}_structure_status"])) for role in CONDITION_ROLES
            },
        },
        "quality": {
            "excluded_rows": int(len(exclusions)),
            "invalid_smiles_records": int(len(issues)),
            "n_amine_groups": int(included["amine_group_id"].nunique()),
            "n_acid_groups": int(included["acid_group_id"].nunique()),
            "n_substrate_pair_groups": int(included["substrate_pair_group_id"].nunique()),
            "pairs_with_more_than_one_product": int((pair_to_product > 1).sum()),
            "duplicate_full_input_groups": int(len(duplicate_groups)),
            "duplicate_full_input_rows": int(duplicate_groups["n_records"].sum()),
            "duplicate_groups_with_label_conflict": int((duplicate_groups["n_distinct_yields"] > 1).sum()),
        },
        "inclusion_rule": "finite yield in [0,1], valid A/B/P structures, and at least one usable reported C structure; this single population is shared by every ablation input set",
    }
    _atomic_json(report, output_root / "quality_report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare audited amide ablation2 population without model fitting.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--task-name", default="ablation2_amide_data_prep_v1")
    args = parser.parse_args()
    output = ROOT / "result" / str(args.task_name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing task output: {output}")
    report = prepare(args.source.resolve(), output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
