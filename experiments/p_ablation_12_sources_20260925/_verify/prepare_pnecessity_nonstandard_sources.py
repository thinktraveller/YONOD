"""Prepare lossless-role, 0–1-yield snapshots for SuFEx and full local USPTO.

Source rows are retained in order. Blank optional SuFEx molecular components stay
blank and receive a descriptor zero block; an absent component's equivalence is
declared as zero. USPTO's original split is preserved only as provenance because
the paired benchmark uses a new full-population grouped-CV protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "result" / "pnecessity_nonstandard_prepare_v1"
SUFEX = ROOT / "dataset" / "others" / "SuFEx" / "SuFExPredictor.csv"
USPTO = ROOT / "dataset" / "USPTO" / "merged.csv"
USPTO_EXPECTED_SHA256 = "e1da081d91652510337325386dfe5855d518f7ba18fb22d61b813b50eb11e49d"
USPTO_EXPECTED_ROWS = 526_468
SUFEX_EXPECTED_ROWS = 2_122

SUFEX_MAP = {
    "Base": "c_base_smiles",
    "Reactant1": "a_smiles",
    "Reactant2": "b_smiles",
    "Product": "p_smiles",
    "solvent": "c_solvent_smiles",
    "Additive1": "c_additive1_smiles",
    "Additive2": "c_additive2_smiles",
    "equiv(Base)": "c_equiv_base",
    "equiv(Reactant1)": "c_equiv_reactant1",
    "equiv(Reactant2)": "c_equiv_reactant2",
    "DielectricConstant": "c_dielectric_constant",
    "time": "c_time_h",
    "T": "c_temperature_c",
    "equiv(Additive1)": "c_equiv_additive1",
    "equiv(Additive2)": "c_equiv_additive2",
    "MS": "c_ms",
    "MWI": "c_mwi",
}
SUFEX_OPTIONAL_MOLECULES = ("Base", "Additive1", "Additive2")
SUFEX_OPTIONAL_EQUIV = ("equiv(Base)", "equiv(Additive1)", "equiv(Additive2)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yield_fraction(raw: str, source: str, row_number: int) -> str:
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError(f"{source} row {row_number}: invalid yield {raw!r}") from exc
    if not value.is_finite() or not Decimal(0) <= value <= Decimal(100):
        raise ValueError(f"{source} row {row_number}: yield outside 0–100: {raw!r}")
    return format(value / Decimal(100), "f")


def _atomic_csv(path: Path, fieldnames: list[str], records: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {path}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=".prepare_", suffix=".csv", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        try:
            for record in records:
                writer.writerow(record)
        except BaseException:
            handle.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare_sufex() -> dict[str, object]:
    target = DEST / "data" / "sufex.csv"
    counts: Counter[str] = Counter()
    yield_min, yield_max = Decimal(1), Decimal(0)
    fieldnames = ["sample_id", "source_row_index", "yield", "yield_percent", *SUFEX_MAP.values()]

    def records():
        nonlocal yield_min, yield_max
        with SUFEX.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(SUFEX_MAP).issubset(reader.fieldnames or []) or "yield" not in (reader.fieldnames or []):
                raise ValueError("SuFEx source columns differ from audited role map")
            for index, source in enumerate(reader, start=1):
                record = {"sample_id": f"sufex_{index:06d}", "source_row_index": str(index), "yield_percent": source["yield"].strip()}
                record["yield"] = yield_fraction(record["yield_percent"], "SuFEx", index)
                value = Decimal(record["yield"])
                yield_min, yield_max = min(yield_min, value), max(yield_max, value)
                for src, dst in SUFEX_MAP.items():
                    token = (source[src] or "").strip()
                    if src in SUFEX_OPTIONAL_EQUIV and not token:
                        token = "0"
                        counts[f"imputed_zero:{src}"] += 1
                    if src in SUFEX_OPTIONAL_MOLECULES and not token:
                        counts[f"blank_component:{src}"] += 1
                    if src in {"Reactant1", "Reactant2", "Product"} and not token:
                        raise ValueError(f"SuFEx row {index}: missing required {src}")
                    record[dst] = token
                counts["rows"] += 1
                yield record

    _atomic_csv(target, fieldnames, records())
    if counts["rows"] != SUFEX_EXPECTED_ROWS:
        raise ValueError(f"SuFEx row count drift: {counts['rows']}")
    return {"source": str(SUFEX.relative_to(ROOT)), "source_sha256": sha256(SUFEX), "prepared": str(target.relative_to(ROOT)), "prepared_sha256": sha256(target), "rows": counts["rows"], "yield_fraction_min": str(yield_min), "yield_fraction_max": str(yield_max), "audits": dict(counts)}


def prepare_uspto() -> dict[str, object]:
    current_sha256 = sha256(USPTO)
    if current_sha256 != USPTO_EXPECTED_SHA256:
        raise ValueError("USPTO merged.csv differs from the audited full local snapshot")
    target = DEST / "data" / "uspto_full.csv"
    counts: Counter[str] = Counter()
    fieldnames = ["sample_id", "source_row_index", "source_split", "yield", "yield_percent", "reactants_smiles", "reagents_smiles", "p_smiles"]

    def records():
        with USPTO.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "split", "reactants_smiles", "reagents_smiles", "products_smiles", "yield_percent"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("USPTO merged.csv lacks audited role columns")
            for index, source in enumerate(reader, start=1):
                split = source["split"].strip()
                if split not in {"train", "valid", "test"}:
                    raise ValueError(f"USPTO row {index}: unknown split {split!r}")
                if not source["sample_id"].startswith(f"uspto_{split}_"):
                    raise ValueError(f"USPTO row {index}: sample_id/split mismatch")
                for column in ("reactants_smiles", "products_smiles"):
                    if not source[column].strip():
                        raise ValueError(f"USPTO row {index}: blank required {column}")
                counts[split] += 1
                if not source["reagents_smiles"].strip():
                    counts["blank_reagents"] += 1
                yield {
                    "sample_id": source["sample_id"],
                    "source_row_index": str(index),
                    "source_split": split,
                    "yield": yield_fraction(source["yield_percent"], "USPTO", index),
                    "yield_percent": source["yield_percent"],
                    "reactants_smiles": source["reactants_smiles"],
                    "reagents_smiles": source["reagents_smiles"],
                    "p_smiles": source["products_smiles"],
                }

    _atomic_csv(target, fieldnames, records())
    if sum(counts[split] for split in ("train", "valid", "test")) != USPTO_EXPECTED_ROWS:
        raise ValueError("USPTO full population row count drift")
    if {split: counts[split] for split in ("train", "valid", "test")} != {"train": 473_963, "valid": 26_101, "test": 26_404}:
        raise ValueError("USPTO source split membership count drift")
    return {"source": str(USPTO.relative_to(ROOT)), "source_sha256": current_sha256, "prepared": str(target.relative_to(ROOT)), "prepared_sha256": sha256(target), "rows": USPTO_EXPECTED_ROWS, "source_split_counts": {split: counts[split] for split in ("train", "valid", "test")}, "blank_reagents": counts["blank_reagents"]}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite preparation manifest: {manifest_path}")
    manifest = {"schema_version": 1, "purpose": "within-source Full/minus-P utility for nonstandard SuFEx and complete local USPTO", "yield_unit": "fraction_0_to_1", "sufex": prepare_sufex(), "uspto_full": prepare_uspto()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
