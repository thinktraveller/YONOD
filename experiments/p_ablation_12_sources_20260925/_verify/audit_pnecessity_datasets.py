#!/usr/bin/env python3
"""Read-only eligibility audit for step 2-12a.

This program never imports YONOD descriptors or training code.  It inventories
every file below ``dataset/``; profiles readable tables; canonicalises only the
declared A/B/P structure columns; and writes an evidence bundle below
``result/pnecessity_dataset_audit_v1/``.  The static role/eligibility policy is
deliberately kept here rather than inferred from filenames at run time, so a
later preregistration can point to a concrete, hash-addressed audit.

The audit is intentionally conservative.  ``conditional`` means that a table
has enough locally observable structure to deserve an upstream-provenance or
design review; it is *not* approval to create a modelling YAML.  In particular,
no local ORD CSV can establish its upstream row mapping because the required
release sidecars are absent from this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover - environment precondition
    raise SystemExit("RDKit is required for the canonical overlap audit") from exc


AUDIT_VERSION = "pnecessity_dataset_audit_v1"
ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "result" / AUDIT_VERSION
STATUS_VALUES = {"eligible", "conditional", "excluded", "duplicate/derived"}
TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class RolePlan:
    """Declared interpretation for one local table family.

    Columns listed in ``a``, ``b`` or ``p`` are molecular structures.  ``c``
    is every recorded condition permitted in the future full model; it is kept
    as text here because its role is identity/duplicate auditing, not chemical
    featurisation.
    """

    source_family: str
    status: str
    reason: str
    label: str | None = None
    label_semantic: str = "unknown"
    a: tuple[str, ...] = ()
    b: tuple[str, ...] = ()
    p: tuple[str, ...] = ()
    c: tuple[str, ...] = ()
    locally_proven: bool = False


@dataclass
class TableSummary:
    entity: str
    path: Path
    relative_path: str
    role_plan: RolePlan
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    label_missing: int = 0
    label_numeric: int = 0
    label_min: float | None = None
    label_max: float | None = None
    missing_by_role: Counter = field(default_factory=Counter)
    invalid_smiles_by_role: Counter = field(default_factory=Counter)
    valid_core_rows: int = 0
    unique_a: set[str] = field(default_factory=set)
    unique_b: set[str] = field(default_factory=set)
    unique_p: set[str] = field(default_factory=set)
    ab: set[str] = field(default_factory=set)
    ab_unordered: set[str] = field(default_factory=set)
    abp: set[str] = field(default_factory=set)
    full: set[str] = field(default_factory=set)
    ab_to_p: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    full_to_labels: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    core_key_rows: int = 0

    def profile_record(self) -> dict[str, Any]:
        plan = self.role_plan
        mapping_counts = Counter(len(values) for values in self.ab_to_p.values())
        deterministic = mapping_counts.get(1, 0)
        duplicate_rows = max(0, self.core_key_rows - len(self.full))
        conflict_groups = sum(1 for values in self.full_to_labels.values() if len(values) > 1)
        return {
            "entity": self.entity,
            "relative_path": self.relative_path,
            "source_family": plan.source_family,
            "classification": plan.status,
            "decision_reason": plan.reason,
            "locally_proven_upstream_lineage": plan.locally_proven,
            "rows": self.rows,
            "columns_json": stable_json(self.columns),
            "label_column": plan.label or "",
            "label_semantic": plan.label_semantic,
            "label_numeric_rows": self.label_numeric,
            "label_missing_rows": self.label_missing,
            "label_min": self.label_min,
            "label_max": self.label_max,
            "inferred_label_scale": infer_scale(self.label_min, self.label_max, self.label_numeric),
            "a_columns_json": stable_json(plan.a),
            "b_columns_json": stable_json(plan.b),
            "p_columns_json": stable_json(plan.p),
            "c_columns_json": stable_json(plan.c),
            "missing_by_role_json": stable_json(dict(self.missing_by_role)),
            "invalid_smiles_by_role_json": stable_json(dict(self.invalid_smiles_by_role)),
            "valid_core_rows": self.valid_core_rows,
            "unique_a": len(self.unique_a),
            "unique_b": len(self.unique_b),
            "unique_p": len(self.unique_p),
            "unique_ab_ordered": len(self.ab),
            "unique_ab_unordered": len(self.ab_unordered),
            "unique_abp": len(self.abp),
            "unique_full_input": len(self.full),
            "duplicate_full_input_rows": duplicate_rows,
            "full_input_label_conflict_groups": conflict_groups,
            "ab_keys_with_one_product": deterministic,
            "ab_keys_total": len(self.ab_to_p),
            "ab_to_one_product_fraction": safe_ratio(deterministic, len(self.ab_to_p)),
            "usable_a_groups": len(self.unique_a),
            "usable_b_groups": len(self.unique_b),
            "usable_pair_groups": len(self.ab),
            "product_overlaps_reactants": len(self.unique_p.intersection(self.unique_a | self.unique_b)),
        }


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if not denominator else numerator / denominator


def infer_scale(minimum: float | None, maximum: float | None, numeric_rows: int) -> str:
    if not numeric_rows or minimum is None or maximum is None:
        return "unavailable"
    if minimum >= -1e-9 and maximum <= 1.000001:
        return "0-1"
    if minimum >= -1e-9 and maximum <= 100.000001:
        return "0-100"
    return "outside_0-1_or_0-100"


def clean_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"", "nan", "none", "null"} else rendered


def canonical_smiles(value: str, cache: dict[str, str | None]) -> str | None:
    if value in cache:
        return cache[value]
    molecule = Chem.MolFromSmiles(value)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else None
    cache[value] = canonical
    return canonical


def canonical_role(
    frame: pd.DataFrame,
    columns: Sequence[str],
    cache: dict[str, str | None],
    missing: Counter,
    invalid: Counter,
    role_name: str,
) -> list[str | None]:
    if not columns:
        return [""] * len(frame)
    output: list[str | None] = []
    values = frame.loc[:, list(columns)].fillna("").astype(str)
    for row in values.itertuples(index=False, name=None):
        canonical_parts: list[str] = []
        failed = False
        for raw in row:
            value = clean_cell(raw)
            if not value:
                missing[role_name] += 1
                failed = True
                break
            canonical = canonical_smiles(value, cache)
            if canonical is None:
                invalid[role_name] += 1
                failed = True
                break
            canonical_parts.append(canonical)
        output.append(None if failed else "||".join(canonical_parts))
    return output


def text_role(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    if not columns:
        return [""] * len(frame)
    values = frame.loc[:, list(columns)].fillna("").astype(str)
    return ["||".join(clean_cell(item) for item in row) for row in values.itertuples(index=False, name=None)]


def role_plan_for(relative_path: str, columns: Sequence[str]) -> RolePlan:
    """Return the predeclared local-only policy, with no score-dependent path."""

    lowered = relative_path.lower()
    names = set(columns)
    if Path(relative_path).suffix.lower() not in TABLE_SUFFIXES:
        return RolePlan("documentation/support", "duplicate/derived", "documentation or support metadata, not a table for the primary matrix")
    # Generic ORD mapping is intentionally explicit: all structured reaction
    # fields other than the label/product/reactants are recorded C candidates.
    if lowered.startswith("ord/"):
        label = next((name for name in columns if name.lower().endswith("yield_percent")), None)
        reactants = tuple(name for name in columns if name.startswith("reactant-"))
        products = ("product",) if "product" in names else ()
        condition_like = tuple(
            name for name in columns
            if name not in set(reactants) | set(products) | ({label} if label else set())
        )
        if "chemrxiv_amide_yield_structure" in lowered:
            return RolePlan(
                "ChemRxiv amide ORD", "excluded",
                "already-used ChemRxiv amide baseline; cannot be a new independent source",
                label, "yield", reactants, p=products, c=condition_like,
            )
        if "roche_borylation" in lowered:
            # The borylation partner is represented in reagent-1, rather than
            # in a second reactant-* column.  Treating it as B is a declared
            # audit adapter, while all remaining recorded fields remain C.
            c_without_b = tuple(name for name in condition_like if name != "reagent-1")
            return RolePlan(
                "ORD model-ready preview", "conditional",
                "ORD CSV has yield/A/B/P after the declared reagent-1-as-B adapter, but local row-map/audit/source-links are absent; upstream lineage unresolved",
                label, "yield", ("reactant-1",), ("reagent-1",), products, c_without_b,
            )
        if label and len(reactants) >= 2 and products:
            diversity_reason = {
                "nicolit": "ORD CSV has yield/A/B/P but local row-map/audit/source-links are absent; upstream lineage unresolved",
                "roche_borylation": "ORD CSV has yield/A/B/P but local row-map/audit/source-links are absent; upstream lineage unresolved",
                "chemrxiv_ch_arylation": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity must remain a boundary",
                "chan_lam": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity is limited",
                "nano": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity is low",
                "ahneman": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity is low",
                "c_n_yield": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity is very low",
                "shields": "ORD CSV has yield/A/B/P, but local lineage is unresolved and reaction-map diversity is very low",
                "suzuki_yield": "ORD CSV has yield/A/B/P, but local lineage is unresolved; it is also checked against the same-size SM table",
            }
            reason = next((value for key, value in diversity_reason.items() if key in lowered),
                          "ORD CSV has a yield/A/B/P surface but lacks local release lineage sidecars")
            return RolePlan("ORD model-ready preview", "conditional", reason, label, "yield", reactants[:1], reactants[1:2], products, condition_like)
        semantic = "yield_without_product_structure" if label else "non_yield_or_missing_label"
        return RolePlan("ORD model-ready preview", "excluded", "missing a complete explicit A/B/P yield representation for the primary matrix", label, semantic, reactants[:1], reactants[1:2], products, condition_like)

    if relative_path == "others/SuFEx/SuFExPredictor.csv":
        return RolePlan(
            "SuFEx literature/SI", "conditional",
            "locally visible yield/A/B/P/C; local copy has no independently verified upstream release hash or row map",
            "yield", "yield", ("Reactant1",), ("Reactant2",), ("Product",),
            ("Base", "equiv(Base)", "equiv(Reactant1)", "equiv(Reactant2)", "solvent", "DielectricConstant", "time", "T", "Additive1", "equiv(Additive1)", "Additive2", "equiv(Additive2)"),
        )
    if relative_path == "others/Buchwald-Hartwig 胺化（Denmark HTE）/denmark_buchwald_hartwig_amination_hte_yield.csv":
        return RolePlan("Denmark BH HTE", "excluded", "yield exists but no explicit product structure column", "Yield", "yield", ("Amine_SMILES",), ("Bromide_SMILES",), (), ("Ligand_SMILES", "Solvent_SMILES", "Base_SMILES"))
    if relative_path == "others/Buchwald-Hartwig 胺化（Doyle-Ahneman HTE）/doyle_ahneman_buchwald_hartwig_amination_hte_yield.csv":
        return RolePlan("Doyle-Ahneman BH HTE", "excluded", "yield exists but no explicit product structure column", "Yield", "yield", ("Aryl_halide_SMILES",), (), (), ("Additive_SMILES", "Base_SMILES", "Ligand_SMILES"))
    if relative_path == "others/Suzuki-Miyaura 偶联（SM）/SM.csv":
        return RolePlan("SM HTE", "excluded", "no product structure; suspected same-source/derived relationship with ORD Suzuki is audited separately", "Yield", "yield", ("reactant_1_smiles",), ("reactant_2_smiles",), (), ("ligand_smiles", "reagent_1_smiles", "solvent_1_smiles"))
    if relative_path == "others/SLAP 三组分反应（SL1）/SL1.csv":
        return RolePlan("SL1 HTE", "excluded", "label is documented LC-MS product ratio and no explicit product structure", "Yield", "lcms_product_ratio", ("Aldehyde_1",), ("Aldehyde_2",), (), ("bifunctional_reagent",))
    if "doyle_ahneman_buchwald_hartwig_c_n_coupling_ratio_subset" in lowered:
        return RolePlan("Doyle-Ahneman C-N adapted subset", "duplicate/derived", "adapted Ratio subset; non-yield endpoint and not independent of Doyle-Ahneman lineage", "Ratio", "ratio", ("Bromide",), ("Amine",), ("Product",), ())

    if relative_path in {"USPTO/train.csv", "USPTO/valid.csv", "USPTO/test.csv"}:
        return RolePlan(
            "USPTO fixed local snapshot", "conditional",
            "explicit yield/reactant/reagent/product fields and fixed split, but multi-component A/B adapter and independence from development remain unresolved",
            "yield_percent", "yield", ("reactants_smiles",), (), ("products_smiles",), ("reagents_smiles",), True,
        )
    if relative_path.startswith("USPTO/"):
        return RolePlan("USPTO provenance/support", "duplicate/derived", "raw/provenance/support file; not an independently model-ready candidate")

    if relative_path == "benchmark_smoke_fixture.csv":
        return RolePlan("project smoke fixture", "excluded", "synthetic engineering fixture; never external scientific evidence", "yield", "synthetic_fixture", ("reactant_1_smiles",), ("reactant_2_smiles",), (), ())

    # Liao group: only raw yield tables are potential boundary candidates.  All
    # split, RDKit, UMAP/t-SNE and SHAP representations are explicitly derived.
    if "amide-coupling" in lowered:
        return RolePlan("amide-coupling development", "excluded", "already-used 47,015-row development population or its additive-fixed representation", "yield", "yield", ("sub_1_smiles",), ("sub_2_smiles",), ("product_smiles",), ("activation", "additive", "base", "solvent"), True)
    if lowered.endswith("pd_catalyzed_ch_functionalization.csv"):
        return RolePlan("Pd C-H functionalisation HTE", "conditional", "yield/A/B/P visible, but no variable recorded C; only a fixed-setting A+B versus A+B+P boundary is possible", "yield", "yield", ("reactant1",), ("reactant2",), ("product",), ())
    if lowered.endswith("ru_catalyzed_insertion.csv"):
        return RolePlan("Ru O-H insertion HTE", "conditional", "yield/A/B/P visible, but no variable recorded C; only a fixed-setting boundary is possible", "yield (%)", "yield", ("sulfoxonium ylide",), ("P(O)OH",), ("product",), ())
    if lowered.endswith("ir_catalyzed_oh_insertion.csv"):
        return RolePlan("Ir O-H insertion HTE", "conditional", "yield/A/B/P visible, but no variable recorded C; existing 412/235 split must be retained", "yield (%)", "yield", ("acid",), ("ylide",), ("product",), ())
    if lowered.endswith("dhp_catalyzed_decarboxylative_selenation.csv"):
        return RolePlan("DHP decarboxylative selenation HTE", "conditional", "yield/A/B/P/C surface is visible, but upstream release identity and experimental split design are not locally proven", "Yield", "yield", ("Substrate_1",), ("Substrate_2",), ("Product",), ("Catalyst", "Solvent"))
    if lowered.endswith("all_hte_set.csv"):
        return RolePlan("Rh C-H amidation HTE", "conditional", "yield/A/B/P visible, but no variable recorded C; only a fixed-setting boundary is possible", "yield (%)", "yield", ("hydrazone",), ("dioxazolone",), ("product",), ())
    if re.search(r"/(model_set|external_set).*\.csv$", lowered) or "_rdkit.csv" in lowered or "/results_data/" in lowered or lowered.endswith(".xlsx"):
        return RolePlan("Liao split/feature derivative", "duplicate/derived", "published split, RDKit feature, visualisation, or case-study derivative; not a new raw source")
    if lowered.endswith("raw_dataset.csv"):
        return RolePlan("Ni enantioselective coupling", "excluded", "ee endpoint and no A/B reactant role columns; outside yield primary matrix", "ee (%)", "ee", (), (), ("Product_SMILES",), ())
    if lowered.endswith("acceptor_hte.csv") or lowered.endswith("donor_acceptor_hte.csv"):
        return RolePlan("Cu fluorocyclopropanation HTE", "excluded", "ee endpoint; cannot replace yield primary evidence", "ee", "ee", ("sub_1_smiles",), ("sub_2_smiles",), ("products_smile",), ("catalysts", "ligands", "intermediates", "solvents"))

    return RolePlan("unclassified local file", "excluded", "no predeclared primary-matrix role mapping")


def read_table_chunks(path: Path, relative_path: str) -> tuple[list[str], Iterable[pd.DataFrame]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype=str)
        return [str(item) for item in frame.columns], [frame]
    sep = "\t" if relative_path.startswith("USPTO/corpus/") else ","
    header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
    return [str(item) for item in header.columns], pd.read_csv(
        path, sep=sep, dtype=str, encoding="utf-8-sig", chunksize=50_000, keep_default_na=False,
    )


def xlsx_metadata(path: Path) -> tuple[list[str], int]:
    """Read sheet-1 header/row count without adding an optional XLSX package.

    The six local XLSX files are declared split/case-study derivatives, so this
    compact OpenXML reader is sufficient for the required all-file field
    inventory. It deliberately does not infer molecular roles from workbooks.
    """

    from xml.etree import ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{ns}si"):
                shared.append("".join(text.text or "" for text in item.iter(f"{ns}t")))
        sheet_paths = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        if not sheet_paths:
            raise ValueError("workbook has no worksheet XML")
        root = ET.fromstring(archive.read(sheet_paths[0]))
    rows = root.findall(f".//{ns}sheetData/{ns}row")
    if not rows:
        return [], 0
    header: list[str] = []
    for cell in rows[0].findall(f"{ns}c"):
        cell_type = cell.get("t")
        value_node = cell.find(f"{ns}v")
        if cell_type == "inlineStr":
            value = "".join(text.text or "" for text in cell.iter(f"{ns}t"))
        elif value_node is None:
            value = ""
        elif cell_type == "s":
            value = shared[int(value_node.text or "0")]
        else:
            value = value_node.text or ""
        header.append(value)
    return header, max(0, len(rows) - 1)


def profile_table(path: Path, relative_path: str, plan: RolePlan, cache: dict[str, str | None]) -> TableSummary:
    columns, chunks = read_table_chunks(path, relative_path)
    # A role mapping that references unavailable columns is an audit finding,
    # not an opportunity for best-effort inference.
    unavailable = [name for name in (*plan.a, *plan.b, *plan.p, *plan.c) if name not in columns]
    if plan.label and plan.label not in columns:
        unavailable.append(plan.label)
    if unavailable:
        plan = RolePlan(
            plan.source_family, "excluded",
            f"declared audit mapping columns absent: {', '.join(unavailable)}",
            plan.label, plan.label_semantic, plan.a, plan.b, plan.p, plan.c, plan.locally_proven,
        )
    summary = TableSummary(relative_path, path, relative_path, plan, columns=columns)
    for frame in chunks:
        summary.rows += len(frame)
        if plan.label and plan.label in frame.columns:
            labels = pd.to_numeric(frame[plan.label], errors="coerce")
            summary.label_missing += int(labels.isna().sum())
            finite = labels.dropna()
            summary.label_numeric += int(len(finite))
            if len(finite):
                local_min, local_max = float(finite.min()), float(finite.max())
                summary.label_min = local_min if summary.label_min is None else min(summary.label_min, local_min)
                summary.label_max = local_max if summary.label_max is None else max(summary.label_max, local_max)
        if not (plan.a and plan.p):
            continue
        a = canonical_role(frame, plan.a, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "A")
        b = canonical_role(frame, plan.b, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "B")
        p = canonical_role(frame, plan.p, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "P")
        c = text_role(frame, plan.c)
        if plan.label and plan.label in frame.columns:
            # Use numeric-normalised labels for duplicate/conflict keys. Thus
            # textual spellings such as ``1`` and ``1.0`` do not become a false
            # label conflict, and nonnumeric labels cannot enter a valid core.
            numeric_labels = pd.to_numeric(frame[plan.label], errors="coerce")
            labels_text = [
                "" if pd.isna(value) else format(float(value), ".17g")
                for value in numeric_labels.tolist()
            ]
        else:
            labels_text = [""] * len(frame)
        for a_key, b_key, p_key, c_key, raw_label in zip(a, b, p, c, labels_text):
            if a_key is None or p_key is None:
                continue
            # USPTO deliberately lacks a fixed B role until a reaction-role
            # adapter is reviewed. Retain its canonical A/P sets for overlap
            # screening, but do not promote a row to primary-matrix validity.
            summary.unique_a.add(a_key)
            summary.unique_p.add(p_key)
            if b_key is None:
                continue
            # A missing B is an actual requirement failure for primary A+B+C;
            # USPTO deliberately has no B role and therefore never reaches this
            # branch's core row count, while retaining its profile/field audit.
            if not b_key:
                continue
            label = clean_cell(raw_label)
            if not label:
                continue
            summary.valid_core_rows += 1
            summary.unique_b.add(b_key)
            ab_key = a_key + "\x1f" + b_key
            unordered = "\x1f".join(sorted((a_key, b_key)))
            abp_key = ab_key + "\x1f" + p_key
            full_key = abp_key + "\x1f" + c_key
            summary.ab.add(ab_key)
            summary.ab_unordered.add(unordered)
            summary.abp.add(abp_key)
            summary.full.add(full_key)
            summary.ab_to_p[ab_key].add(p_key)
            summary.full_to_labels[full_key].add(label)
            summary.core_key_rows += 1
    return summary


def baseline_summary(path: Path, entity: str, a_col: str, b_col: str, p_col: str) -> TableSummary:
    """Read an already-used baseline solely for overlap, not as a candidate."""

    plan = RolePlan("already-used baseline", "excluded", "reference only", a=(a_col,), b=(b_col,), p=(p_col,))
    summary = TableSummary(entity, path, entity, plan)
    cache: dict[str, str | None] = {}
    header = pd.read_csv(path, nrows=0)
    summary.columns = list(header.columns)
    for frame in pd.read_csv(path, dtype=str, chunksize=50_000, keep_default_na=False):
        summary.rows += len(frame)
        a = canonical_role(frame, plan.a, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "A")
        b = canonical_role(frame, plan.b, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "B")
        p = canonical_role(frame, plan.p, cache, summary.missing_by_role, summary.invalid_smiles_by_role, "P")
        for a_key, b_key, p_key in zip(a, b, p):
            if a_key is None or b_key is None or p_key is None:
                continue
            summary.unique_a.add(a_key); summary.unique_b.add(b_key); summary.unique_p.add(p_key)
            ab_key = a_key + "\x1f" + b_key
            summary.ab.add(ab_key); summary.ab_unordered.add("\x1f".join(sorted((a_key, b_key))))
            summary.abp.add(ab_key + "\x1f" + p_key)
    return summary


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def examples_for_overlap(values: set[str], category: str, left: str, right: str) -> list[dict[str, str]]:
    return [
        {"left": left, "right": right, "key_type": category, "key_sha256": sha256_text(value)}
        for value in sorted(values)[:10]
    ]


def overlap_records(left: TableSummary, right: TableSummary) -> tuple[dict[str, Any], list[dict[str, str]]]:
    sets = {
        "A": (left.unique_a, right.unique_a),
        "B": (left.unique_b, right.unique_b),
        "P": (left.unique_p, right.unique_p),
        "AB_ordered": (left.ab, right.ab),
        "AB_unordered": (left.ab_unordered, right.ab_unordered),
        "ABP": (left.abp, right.abp),
        "full_input": (left.full, right.full),
    }
    record: dict[str, Any] = {"left": left.entity, "right": right.entity}
    samples: list[dict[str, str]] = []
    for name, (a_values, b_values) in sets.items():
        overlap = a_values.intersection(b_values)
        record[f"shared_{name}"] = len(overlap)
        record[f"left_{name}"] = len(a_values)
        record[f"right_{name}"] = len(b_values)
        record[f"jaccard_{name}"] = safe_ratio(len(overlap), len(a_values.union(b_values)))
        samples.extend(examples_for_overlap(overlap, name, left.entity, right.entity))
    return record, samples


def status_reason_for_file(path: Path, relative_path: str, plan: RolePlan, duplicate_of: str | None) -> tuple[str, str]:
    if duplicate_of:
        return "duplicate/derived", f"byte-identical duplicate of {duplicate_of}"
    return plan.status, plan.reason


def markdown_report(
    inventory_count: int,
    tables: Sequence[TableSummary],
    profiles: Sequence[Mapping[str, Any]],
    overlap: Sequence[Mapping[str, Any]],
    duplicate_groups: Sequence[Mapping[str, Any]],
) -> str:
    status_counts = Counter(profile["classification"] for profile in profiles)
    candidate_profiles = [profile for profile in profiles if profile["classification"] == "conditional"]
    conditional_lines = []
    for profile in candidate_profiles:
        conditional_lines.append(
            f"| `{profile['relative_path']}` | {profile['rows']} | {profile['valid_core_rows']} | "
            f"{profile['unique_a']}/{profile['unique_b']}/{profile['unique_p']} | "
            f"{profile['ab_to_one_product_fraction']!s} | {profile['decision_reason']} |"
        )
    baseline_overlap = [row for row in overlap if row["right"].startswith("baseline:") or row["left"].startswith("baseline:")]
    nonzero_abp = [row for row in baseline_overlap if row["shared_ABP"]]
    return "\n".join([
        f"# Step 2-12a local dataset eligibility audit ({AUDIT_VERSION})",
        "",
        "## Scope and non-modelling boundary",
        "",
        f"This read-only audit inventoried **{inventory_count} files** below `dataset/` and profiled **{len(tables)} readable tables**. It did not create a YAML, descriptor, split, prediction, or fitted model.",
        "",
        "A `conditional` classification is not eligibility. It only says the local table exposed a potentially relevant A/B/P/label surface. No source is approved for step 2-12b in this run because locally available provenance does not establish all required independent-source and role-design gates.",
        "",
        "## Classification counts",
        "",
        "| classification | profiled tables |",
        "| --- | ---: |",
        *[f"| {status} | {status_counts.get(status, 0)} |" for status in ("eligible", "conditional", "excluded", "duplicate/derived")],
        "",
        "## Conditional, not approved, surfaces",
        "",
        "`A/B/P` are canonical unique structures after valid-core filtering. `A+B→one P` is descriptive representation determinism, not a scientific exclusion by itself.",
        "",
        "| local table | rows | valid A+B+P+label | A/B/P | A+B→one P | blocking reason |",
        "| --- | ---: | ---: | --- | --- | --- |",
        *(conditional_lines or ["| _none_ |  |  |  |  |  |"]),
        "",
        "## Provenance and overlap boundary",
        "",
        "`dataset/ORD/README.md` identifies a target upstream release but also states that this checkout lacks its row-map, audit, source-links, metadata and checksum sidecars. Accordingly, a local CSV hash is a snapshot identity only, not proof of independent ORD lineage. The USPTO files retain a local conversion/provenance chain, but their multi-component reactant roles and relationship to the existing amide development population are not resolved here. Existing amide development and ChemRxiv amide populations are overlap references only, never new candidates.",
        "",
        f"The pairwise overlap ledger contains {len(overlap)} comparisons. Exact canonical A+B+P overlap against the two already-used baseline populations occurred in **{len(nonzero_abp)}** comparison(s) (see `overlap_summary.csv`; structures in examples are SHA-256 keyed only). A zero exact overlap does not prove source independence.",
        "",
        f"Exact byte-hash duplicate groups: **{len(duplicate_groups)}**. They are enumerated in `exact_hash_duplicate_groups.csv`; aliases and derived feature/split tables are not counted as independent data sources.",
        "",
        "## Reproduction",
        "",
        "From the repository root, with the mandated environment active:",
        "",
        "```bash",
        "source /home/wangzh685/miniconda3/etc/profile.d/conda.sh",
        "conda activate yonod",
        "python _verify/audit_pnecessity_datasets.py",
        "```",
        "",
        "The script hashes every input it consumes and records its own hash in `run_manifest.json`. Re-running into an existing output directory requires `--overwrite`; it still only reads `dataset/` and baseline result CSVs.",
        "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="replace named audit outputs in an existing audit directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output already exists: {output}; use --overwrite after reviewing it")
    output.mkdir(parents=True, exist_ok=True)
    if not DATASET_ROOT.is_dir():
        raise SystemExit(f"dataset root does not exist: {DATASET_ROOT}")

    all_files = sorted(path for path in DATASET_ROOT.rglob("*") if path.is_file())
    inventory: list[dict[str, Any]] = []
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    plans: dict[str, RolePlan] = {}
    for path in all_files:
        relative = path.relative_to(DATASET_ROOT).as_posix()
        digest = sha256_file(path)
        plan = role_plan_for(relative, [])
        plans[relative] = plan
        hash_to_paths[digest].append(relative)
        inventory.append({
            "relative_path": relative,
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "classification": plan.status,
            "decision_reason": plan.reason,
            "source_family": plan.source_family,
        })

    duplicates: list[dict[str, Any]] = []
    duplicate_of: dict[str, str] = {}
    for digest, paths in sorted(hash_to_paths.items()):
        if len(paths) < 2:
            continue
        canonical = paths[0]
        duplicates.append({"sha256": digest, "canonical_path": canonical, "paths_json": stable_json(paths), "count": len(paths)})
        for path in paths[1:]:
            duplicate_of[path] = canonical
    inventory_by_path = {record["relative_path"]: record for record in inventory}
    for relative, record in inventory_by_path.items():
        status, reason = status_reason_for_file(DATASET_ROOT / relative, relative, plans[relative], duplicate_of.get(relative))
        record["classification"] = status
        record["decision_reason"] = reason

    summaries: list[TableSummary] = []
    profile_errors: list[dict[str, str]] = []
    canonical_cache: dict[str, str | None] = {}
    for path in all_files:
        if path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        relative = path.relative_to(DATASET_ROOT).as_posix()
        # The corpus TSV and provenance file are inventory/provenance evidence,
        # not direct candidates.  Their header and line count remain auditable
        # without needlessly parsing 800 MB of duplicate role material.
        if relative.startswith("USPTO/corpus/"):
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                    header = handle.readline().rstrip("\n\r").split("\t")
                    rows = sum(1 for _ in handle)
                plan = role_plan_for(relative, header)
                summaries.append(TableSummary(relative, path, relative, plan, rows=rows, columns=header))
            except OSError as exc:
                profile_errors.append({"relative_path": relative, "error": str(exc)})
            continue
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                columns, rows = xlsx_metadata(path)
                plan = role_plan_for(relative, columns)
                if relative in duplicate_of:
                    plan = RolePlan(plan.source_family, "duplicate/derived", f"byte-identical duplicate of {duplicate_of[relative]}", plan.label, plan.label_semantic, plan.a, plan.b, plan.p, plan.c, plan.locally_proven)
                summaries.append(TableSummary(relative, path, relative, plan, rows=rows, columns=columns))
            except Exception as exc:
                profile_errors.append({"relative_path": relative, "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            columns, _ = read_table_chunks(path, relative)
            plan = role_plan_for(relative, columns)
            if relative in duplicate_of:
                plan = RolePlan(plan.source_family, "duplicate/derived", f"byte-identical duplicate of {duplicate_of[relative]}", plan.label, plan.label_semantic, plan.a, plan.b, plan.p, plan.c, plan.locally_proven)
            summaries.append(profile_table(path, relative, plan, canonical_cache))
        except Exception as exc:  # record a malformed local table rather than hiding it
            profile_errors.append({"relative_path": relative, "error": f"{type(exc).__name__}: {exc}"})

    # Table headers are needed for role classification. Replace the provisional
    # path-only inventory statuses with the executed table audit outcomes.
    for summary in summaries:
        record = inventory_by_path[summary.relative_path]
        record["classification"] = summary.role_plan.status
        record["decision_reason"] = summary.role_plan.reason
        record["source_family"] = summary.role_plan.source_family

    # Existing development and ChemRxiv populations are only comparison anchors.
    baseline_specs = [
        (ROOT / "result/ablation2_amide_data_prep_v1/data/standardized_population.csv", "baseline:amide_development_47015", "acid_smiles", "amine_smiles", "product_smiles"),
        (ROOT / "result/ablation2_chemrxiv_external_data_audit_v1/data/standardized_external_population.csv", "baseline:chemrxiv_amide_957", "reactant-1", "reactant-2", "product"),
    ]
    baselines: list[TableSummary] = []
    baseline_inventory: list[dict[str, Any]] = []
    for path, entity, a_col, b_col, p_col in baseline_specs:
        if not path.is_file():
            profile_errors.append({"relative_path": entity, "error": f"missing baseline reference: {path}"})
            continue
        baselines.append(baseline_summary(path, entity, a_col, b_col, p_col))
        baseline_inventory.append({"entity": entity, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "rows": baselines[-1].rows})

    # Keep USPTO in the overlap ledger even though it has no frozen B adapter:
    # its canonical reactant-mixture and product sets can still reveal direct
    # structure reuse. Empty AB/ABP fields make the adapter limitation visible
    # instead of silently omitting this conditional source from comparisons.
    candidate_summaries = [summary for summary in summaries if summary.role_plan.status == "conditional"]
    overlap: list[dict[str, Any]] = []
    overlap_examples: list[dict[str, str]] = []
    all_overlap_entities = candidate_summaries + baselines
    for index, left in enumerate(all_overlap_entities):
        for right in all_overlap_entities[index + 1:]:
            record, samples = overlap_records(left, right)
            overlap.append(record); overlap_examples.extend(samples)

    profiles = [summary.profile_record() for summary in summaries]
    key_digests: list[dict[str, Any]] = []
    for summary in summaries + baselines:
        key_digests.append({
            "entity": summary.entity,
            "relative_path": summary.relative_path,
            "A_count": len(summary.unique_a), "A_sha256": digest_strings(summary.unique_a),
            "B_count": len(summary.unique_b), "B_sha256": digest_strings(summary.unique_b),
            "P_count": len(summary.unique_p), "P_sha256": digest_strings(summary.unique_p),
            "AB_count": len(summary.ab), "AB_sha256": digest_strings(summary.ab),
            "AB_unordered_count": len(summary.ab_unordered), "AB_unordered_sha256": digest_strings(summary.ab_unordered),
            "ABP_count": len(summary.abp), "ABP_sha256": digest_strings(summary.abp),
            "full_input_count": len(summary.full), "full_input_sha256": digest_strings(summary.full),
        })

    decisions = []
    for profile in profiles:
        decisions.append({
            "relative_path": profile["relative_path"], "source_family": profile["source_family"],
            "classification": profile["classification"], "locally_proven_upstream_lineage": profile["locally_proven_upstream_lineage"],
            "label_semantic": profile["label_semantic"], "inferred_label_scale": profile["inferred_label_scale"],
            "a_columns_json": profile["a_columns_json"], "b_columns_json": profile["b_columns_json"],
            "p_columns_json": profile["p_columns_json"], "c_columns_json": profile["c_columns_json"],
            "usable_a_groups": profile["usable_a_groups"], "usable_b_groups": profile["usable_b_groups"],
            "usable_pair_groups": profile["usable_pair_groups"], "decision_reason": profile["decision_reason"],
        })

    write_csv(output / "file_inventory.csv", inventory)
    write_csv(output / "table_profiles.csv", profiles)
    write_csv(output / "candidate_decisions.csv", decisions)
    write_csv(output / "canonical_key_digests.csv", key_digests)
    write_csv(output / "overlap_summary.csv", overlap)
    write_csv(output / "overlap_key_examples_hashed.csv", overlap_examples)
    write_csv(output / "exact_hash_duplicate_groups.csv", duplicates)
    write_csv(output / "profile_errors.csv", profile_errors)
    write_csv(output / "baseline_references.csv", baseline_inventory)
    (output / "audit_report.md").write_text(markdown_report(len(all_files), summaries, profiles, overlap, duplicates), encoding="utf-8")

    # Exclude self-referential manifests from the material-output digest; this
    # avoids carrying a stale hash when the audit is deliberately re-run with
    # ``--overwrite``.
    output_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"run_manifest.json", "verification.json"}
    )
    manifest = {
        "schema_version": "pnecessity_dataset_audit/1",
        "audit_id": AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "read_only_boundary": "No dataset file was modified; no YAML, feature, split, prediction, or model was created.",
        "script": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256_file(Path(__file__))},
        "inputs": {"dataset_file_count": len(all_files), "dataset_inventory_sha256": digest_strings([f"{item['relative_path']}\x1f{item['sha256']}" for item in inventory]), "baseline_references": baseline_inventory},
        "policy": {"statuses": sorted(STATUS_VALUES), "primary_label": "experimental yield with explicit A+B+P+C", "conditional_is_approval": False, "canonicalisation": "RDKit canonical isomeric SMILES; role order retained for ordered A+B keys"},
        "summary": {"profiled_tables": len(summaries), "profile_errors": len(profile_errors), "eligible_sources": sum(1 for profile in profiles if profile['classification'] == 'eligible'), "conditional_tables": sum(1 for profile in profiles if profile['classification'] == 'conditional'), "exact_duplicate_groups": len(duplicates), "overlap_comparisons": len(overlap)},
        "outputs_before_manifest": [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in output_files],
    }
    write_json(output / "run_manifest.json", manifest)
    verification = {
        "schema_version": "pnecessity_dataset_audit_verification/1",
        "passed": not profile_errors and len(inventory) == len(all_files) and all(record["classification"] in STATUS_VALUES for record in inventory),
        "checks": {
            "all_dataset_files_in_inventory": len(inventory) == len(all_files),
            "all_inventory_statuses_allowed": all(record["classification"] in STATUS_VALUES for record in inventory),
            "profile_errors_empty": not profile_errors,
            "baseline_references_loaded": len(baselines) == 2,
            "no_eligible_source_without_upstream_gate": not any(profile["classification"] == "eligible" for profile in profiles),
        },
    }
    write_json(output / "verification.json", verification)
    print(stable_json({"output": str(output), "verification_passed": verification["passed"], "profiled_tables": len(summaries), "conditional_tables": len(candidate_summaries)}))
    return 0 if verification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
