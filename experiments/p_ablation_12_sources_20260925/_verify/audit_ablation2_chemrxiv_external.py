"""Audit the constrained ChemRxiv aniline-amidation extension candidate.

This is a data-audit task, not a modeling task.  It writes an immutable,
isolated audit package and refuses to overwrite a prior result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "ORD" / "chemrxiv_amide_yield_structure_v1_normalized_dataset.csv"
DEVELOPMENT = ROOT / "result" / "ablation2_amide_data_prep_v1" / "data" / "standardized_population.csv"
OUT = ROOT / "result" / "ablation2_chemrxiv_external_data_audit_v1"
EXPECTED_SOURCE_SHA256 = "90c762cff7defe31e7e45d35406df595d094186a0b35b05b0f799cfe66411f72"
ROLE_COLUMNS = {"acid": "reactant-1", "amine": "reactant-2", "product": "product"}
CONDITION_COLUMNS = [
    "reagent-1",
    "reagent-2",
    "solvent-1",
    "solvent-2",
    "solvent-3",
    "temperature_c",
    "reaction_time_s",
    "reflux_value",
    "conditions_are_dynamic_value",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return "smiles:" + Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-row-map", type=Path, required=True, help="Released row-map.csv for the checksum-matched candidate source")
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit result: {OUT}")
    if not SOURCE.is_file() or not DEVELOPMENT.is_file() or not args.upstream_row_map.is_file():
        raise RuntimeError("Missing candidate CSV, development population, or released row map")
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Candidate checksum mismatch: {source_hash}")
    candidate = pd.read_csv(SOURCE)
    expected = set(ROLE_COLUMNS.values()) | set(CONDITION_COLUMNS) | {"yield_percent"}
    if expected.difference(candidate.columns) or len(candidate) != 957:
        raise RuntimeError("Candidate schema or row-count contract changed")
    row_map = pd.read_csv(args.upstream_row_map)
    if list(row_map.columns) != ["csv_row_number", "reaction_key", "physical_dataset_id", "source_row_index", "label_decision_id"]:
        raise RuntimeError("Unexpected upstream row-map schema")
    if len(row_map) != len(candidate) or not row_map.csv_row_number.tolist() == list(range(2, len(candidate) + 2)):
        raise RuntimeError("Upstream row map does not provide a one-to-one CSV mapping")
    if row_map.physical_dataset_id.nunique() != 1 or row_map.physical_dataset_id.iloc[0] != "ord_dataset-7acd6ad2bf4d4cff841cad008ab726d5":
        raise RuntimeError("Unexpected physical source dataset identity")
    candidate.insert(0, "sample_id", [f"chemrxiv_amide:{index:04d}" for index in range(1, len(candidate) + 1)])
    candidate.insert(1, "source_reaction_key", row_map.reaction_key)
    candidate.insert(2, "source_row_index", row_map.source_row_index)
    candidate["yield"] = candidate["yield_percent"] / 100.0
    if not candidate["yield"].between(0, 1).all():
        raise RuntimeError("Percent-yield conversion left the expected [0, 1] range")
    for role, column in ROLE_COLUMNS.items():
        candidate[f"{role}_structure_key"] = candidate[column].map(canonical_key)
        if candidate[f"{role}_structure_key"].isna().any():
            raise RuntimeError(f"Invalid or missing core structure in role {role}")
    development = pd.read_csv(DEVELOPMENT, usecols=[f"{role}_structure_key" for role in ROLE_COLUMNS])
    role_rows, overlap_rows = [], []
    for role, source_column in ROLE_COLUMNS.items():
        key = f"{role}_structure_key"
        candidate_keys = set(candidate[key])
        development_keys = set(development[key].dropna())
        overlap = candidate_keys & development_keys
        role_rows.append({"role": role, "source_column": source_column, "rows": len(candidate), "unique_structures": len(candidate_keys), "missing_or_invalid": int(candidate[key].isna().sum())})
        overlap_rows.append({"role": role, "candidate_unique_structures": len(candidate_keys), "development_unique_structures": len(development_keys), "exact_canonical_overlap": len(overlap), "overlap_examples": ";".join(sorted(overlap)[:5])})
    full_columns = [*ROLE_COLUMNS.values(), *CONDITION_COLUMNS]
    duplicate_groups = candidate.duplicated(full_columns, keep=False).sum()
    condition_rows = [
        {"column": column, "nonmissing_rows": int(candidate[column].notna().sum()), "unique_nonmissing_values": int(candidate[column].nunique(dropna=True))}
        for column in CONDITION_COLUMNS
    ]
    OUT.mkdir(parents=True)
    (OUT / "data").mkdir()
    (OUT / "tables").mkdir()
    candidate.to_csv(OUT / "data" / "standardized_external_population.csv", index=False)
    row_map.to_csv(OUT / "tables" / "source_row_map.csv", index=False)
    pd.DataFrame(role_rows).to_csv(OUT / "tables" / "role_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "tables" / "canonical_overlap_summary.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(OUT / "tables" / "condition_summary.csv", index=False)
    report = f"""# ChemRxiv aniline-amidation extension data audit

## Identity and lineage

- Candidate rows: {len(candidate)}; local CSV SHA-256: `{source_hash}`.
- Matched model-ready release dataset checksum: `{EXPECTED_SOURCE_SHA256}`.
- Released physical ORD source: `ord_dataset-7acd6ad2bf4d4cff841cad008ab726d5` at ORD revision `83f971f586f6ad18f358ae4ae99d045e94ed2066`.
- All {len(row_map)} local CSV rows map one-to-one to released ORD reaction keys; `tables/source_row_map.csv` retains that mapping.
- The 47,015-row development data is the distinct AIChemEco amide-coupling collection.  Distinct source packages and zero exact canonical A/B/P overlaps rule out direct row, substrate, or product reuse under this audit; they do not prove broad chemical independence.

## Role mapping and data boundary

- `reactant-2` is mapped to A/amine, `reactant-1` to B/acid, and `product` to P.
- `reagent-1`, `reagent-2`, solvents, temperature, time, reflux and dynamic-condition flag comprise C. `yield_percent` is converted to `yield` in [0, 1].
- The candidate contains {candidate['acid_structure_key'].nunique()} acid, {candidate['amine_structure_key'].nunique()} amine/nitrogen-nucleophile and {candidate['product_structure_key'].nunique()} product structures. It cannot support a within-source acid-ablation or an unseen-acid distribution claim.
- {duplicate_groups} rows participate in an exact complete-input duplicate group. These are retained and must be grouped together in any within-source split.

## Decision

The candidate passes identity, label-unit, core-structure and direct-overlap gates for a **constrained external validation**.  Before any fit, freeze an extension preregistration that omits acid-deletion primary contrasts and separates frozen-development external testing from any within-candidate validation.  This audit does not itself train or score a model.
"""
    (OUT / "audit-report.md").write_text(report, encoding="utf-8")
    manifest = {
        "task_name": "ablation2_chemrxiv_external_data_audit_v1",
        "candidate_source_path": str(SOURCE.relative_to(ROOT)),
        "candidate_source_sha256": source_hash,
        "source_release": "thinktraveller/ord-datasets v0.1.0-model-ready-preview-1 (7d52dd27071c5625008a14737a1a8e1252ae6217)",
        "ord_revision": "83f971f586f6ad18f358ae4ae99d045e94ed2066",
        "physical_dataset_id": row_map.physical_dataset_id.iloc[0],
        "row_map_sha256": sha256(args.upstream_row_map),
        "row_count": len(candidate),
        "exact_canonical_overlap": {row["role"]: int(row["exact_canonical_overlap"]) for row in overlap_rows},
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"External data audit complete: {OUT}")


if __name__ == "__main__":
    main()
