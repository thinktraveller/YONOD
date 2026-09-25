"""Create eight standalone schema-2 Full/minus-P YAMLs, in queue order."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "result" / "pnecessity_nonstandard_prepare_v1"
MANIFEST = PREP / "source_manifest.json"
QUEUE = PREP / "generated_config_manifest.json"

SOURCES = (
    {
        "id": "sufex",
        "manifest_key": "sufex",
        "molecular": ["a_smiles", "b_smiles", "c_base_smiles", "c_solvent_smiles", "c_additive1_smiles", "c_additive2_smiles"],
        "grouping": ["a_smiles", "b_smiles", "c_base_smiles", "c_solvent_smiles", "c_additive1_smiles", "c_additive2_smiles"],
        "numeric": {
            "c_equiv_base": "equivalent",
            "c_equiv_reactant1": "equivalent",
            "c_equiv_reactant2": "equivalent",
            "c_dielectric_constant": "relative_permittivity",
            "c_time_h": "h",
            "c_temperature_c": "degC",
            "c_equiv_additive1": "equivalent",
            "c_equiv_additive2": "equivalent",
            "c_ms": "indicator",
            "c_mwi": "indicator",
        },
        "notes": "Nonstandard literature/SI SuFEx; blank optional molecular C is a zero descriptor block and missing component equivalents were set to zero during preparation.",
    },
    {
        "id": "uspto_full",
        "manifest_key": "uspto_full",
        "molecular": ["reactants_smiles", "reagents_smiles"],
        "grouping": ["reactants_smiles"],
        "numeric": {},
        "notes": "Complete 526468-row local merged USPTO snapshot; original train/valid/test markers are provenance only. New reactant-grouped 5x3 CV, not the published fixed-split test protocol. Blank reagents become zero descriptor blocks.",
    },
)
LINES = (
    {
        "id": "morgan_rf",
        "descriptor": "morgan",
        "model": "rf",
        "params": {"rf": {"estimator": {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1, "random_state": 20260918, "n_jobs": 19}}},
    },
    {
        "id": "mfp_lightgbm",
        "descriptor": "mfp",
        "model": "lightgbm",
        "params": {"lightgbm": {"estimator": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 1, "random_state": 20260918, "n_jobs": 19}}},
    },
)
ARMS = ("full", "minus_p")


class ExplicitDumper(yaml.SafeDumper):
    """The strict loader rejects YAML aliases, even when aliases are valid YAML."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def explicit_yaml(payload: dict) -> str:
    return yaml.dump(payload, Dumper=ExplicitDumper, allow_unicode=True, sort_keys=False, width=96)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(source: dict, line: dict, arm: str, evidence: dict) -> dict:
    task = f"pnecessity_nonstandard_{source['id']}_{line['id']}_{arm}_v1"
    molecules = [*source["molecular"], *(["p_smiles"] if arm == "full" else [])]
    numeric = list(source["numeric"])
    numeric_contract = {
        "defaults": {"missing": {"strategy": "error"}, "scaling": "standard"},
        "columns": {
            name: {"name": name, "missing": {"strategy": "error"}, "scaling": "standard", "unit": {"source": unit, "target": unit, "scale": 1.0, "offset": 0.0}}
            for name, unit in source["numeric"].items()
        },
    }
    descriptor = {"id": f"{line['id']}_{arm}", "descriptor": line["descriptor"], "lifecycle": "static_descriptor", "mode": "concat", "columns": molecules}
    if line["descriptor"] == "mfp":
        descriptor["params"] = {"radius": 3, "fp_size": 1024, "profile": "standard"}
    return {
        "schema_version": "2.0",
        "project_name": task,
        "stage": "benchmark",
        "dataset": {
            "path": "./" + evidence["prepared"],
            "sample_id_col": "sample_id",
            "column_roles": {"label": "yield", "reactants": molecules, "products": ["p_smiles"] if arm == "full" else [], "others": [], "conditions": numeric, "categoricals": []},
            "numeric_conditions": numeric_contract,
        },
        "descriptors": [descriptor],
        "artifacts": {"output_dir": f"./result/{task}/feature"},
        "models": [line["model"]],
        "model_params": line["params"],
        "evaluation": {
            "protocol": "manifest_outer_cv",
            "n_splits": 5,
            "n_repeats": 3,
            "seed": 20260918,
            "grouping": {"strategy": "component_holdout", "component_cols": source["grouping"], "protocol_label": f"pnecessity_{source['id']}_non_p_component_full_population_v1", "max_group_fraction": 0.8},
        },
        "outputs": {"root": f"./result/{task}", "report_formats": ["html", "markdown"]},
        "benchmark": {"task_state": {"backend": "sqlite", "resumable": True}, "population_id": f"pnecessity-{source['id']}-{evidence['prepared_sha256'][:16]}", "dataset_id": source["id"]},
        "metadata": {
            "comparison": "within_source_paired_utility",
            "input_arm": arm,
            "source_type": "nonstandard_high_throughput_or_curated_patent_reactions",
            "source_path": "./" + evidence["source"],
            "source_sha256": evidence["source_sha256"],
            "prepared_sha256": evidence["prepared_sha256"],
            "preparation_manifest": "./" + str(MANIFEST.relative_to(ROOT)),
            "yield_unit": "fraction_0_to_1",
            "split_policy": "new within-source non-P component-holdout 5x3 CV on all prepared rows; the same membership is generated for both arms",
            "notes": source["notes"],
            "product_embedding_execution_mapping": "Full duplicates p_smiles in reactants for the strict descriptor input while retaining its declared products role" if arm == "full" else "minus-P excludes p_smiles from all model inputs",
            "limitation": "nonstandard local source; estimate only within-source P embedding utility, without pooled or causal claims",
        },
    }


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError("run prepare_pnecessity_nonstandard_sources.py first")
    if QUEUE.exists():
        raise FileExistsError(f"refusing to overwrite {QUEUE}")
    evidence = json.loads(MANIFEST.read_text(encoding="utf-8"))
    planned = []
    for source in SOURCES:
        entry = evidence[source["manifest_key"]]
        if sha256(ROOT / entry["prepared"]) != entry["prepared_sha256"]:
            raise ValueError(f"prepared data hash drift: {source['id']}")
        for line in LINES:
            for arm in ARMS:
                config = build_config(source, line, arm, entry)
                task = config["project_name"]
                path = ROOT / "config" / f"{task}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(explicit_yaml(config))
                planned.append({"source_id": source["id"], "line_id": line["id"], "arm": arm, "task_name": task, "config_path": str(path.relative_to(ROOT)), "config_sha256": sha256(path)})
    QUEUE.write_text(json.dumps({"schema_version": 1, "task_count": len(planned), "queue": planned, "prepared_manifest_sha256": sha256(MANIFEST)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("created", len(planned), "standalone schema-2 YAMLs")
    for item in planned:
        print(item["config_path"])


def repair_generated_aliases() -> None:
    """One-time rewrite of this generator's own just-created YAMLs only."""
    manifest = json.loads(QUEUE.read_text(encoding="utf-8"))
    if len(manifest.get("queue", [])) != 8:
        raise ValueError("expected exactly the eight newly generated YAMLs")
    for item in manifest["queue"]:
        path = ROOT / item["config_path"]
        if path.name != f"{item['task_name']}.yaml" or not path.name.startswith("pnecessity_nonstandard_"):
            raise ValueError(f"unexpected YAML repair target: {path}")
        if sha256(path) != item["config_sha256"]:
            raise ValueError(f"YAML changed since generation: {path}")
        if (ROOT / "result" / item["task_name"]).exists():
            raise ValueError(f"task has started; refusing to rewrite: {item['task_name']}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(explicit_yaml(payload), encoding="utf-8")
        item["config_sha256"] = sha256(path)
    QUEUE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("repaired eight generated YAMLs without aliases")


if __name__ == "__main__":
    if sys.argv[1:] == ["--repair-aliases"]:
        repair_generated_aliases()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("usage: generate_pnecessity_nonstandard_configs.py [--repair-aliases]")
