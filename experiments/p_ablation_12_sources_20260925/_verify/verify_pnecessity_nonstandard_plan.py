"""Read-only launch gate for the eight SuFEx / complete-USPTO YAML tasks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig
from verify_pnecessity_nonstandard_result import load_for_verification


PREP = ROOT / "result" / "pnecessity_nonstandard_prepare_v1"
EXPECTED = [(source, line, arm) for source in ("sufex", "uspto_full") for line in ("morgan_rf", "mfp_lightgbm") for arm in ("full", "minus_p")]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    source_manifest = PREP / "source_manifest.json"
    queue_manifest = PREP / "generated_config_manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    queue = json.loads(queue_manifest.read_text(encoding="utf-8"))
    assert queue["prepared_manifest_sha256"] == digest(source_manifest)
    items = queue["queue"]
    assert queue["task_count"] == len(items) == 8
    assert [(item["source_id"], item["line_id"], item["arm"]) for item in items] == EXPECTED
    configs = {}
    raw = {}
    for item in items:
        task = item["task_name"]
        path = ROOT / item["config_path"]
        assert path == ROOT / "config" / f"{task}.yaml"
        assert digest(path) == item["config_sha256"], f"config hash drift: {task}"
        config = load_for_verification(path) if (ROOT / "result" / task).exists() else BenchmarkConfig.from_file(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config.outputs_root == ROOT / "result" / task
        assert config.artifact_output_dir == config.outputs_root / "feature"
        assert data["outputs"]["report_formats"] == ["html", "markdown"]
        assert config.cv == {"n_splits": 5, "n_repeats": 3, "seed": 20260918}
        assert data["evaluation"]["grouping"]["strategy"] == "component_holdout"
        assert "p_smiles" not in data["evaluation"]["grouping"]["component_cols"]
        assert data["models"] == (["rf"] if item["line_id"] == "morgan_rf" else ["lightgbm"])
        assert data["model_params"][data["models"][0]]["estimator"]["n_jobs"] == 19
        assert list(data["model_params"]) == data["models"]
        assert data["dataset"]["path"] == "./" + source[item["source_id"]]["prepared"]
        assert data["metadata"]["prepared_sha256"] == source[item["source_id"]]["prepared_sha256"]
        configs[(item["source_id"], item["line_id"], item["arm"])] = config
        raw[(item["source_id"], item["line_id"], item["arm"])] = data
    for source_id, expected_rows in (("sufex", 2122), ("uspto_full", 526468)):
        record = source[source_id]
        assert digest(ROOT / record["prepared"]) == record["prepared_sha256"]
        assert digest(ROOT / record["source"]) == record["source_sha256"]
        frame = configs[(source_id, "morgan_rf", "full")].validate_dataset()
        assert len(frame) == expected_rows
        assert frame["yield"].between(0, 1).all()
        if source_id == "uspto_full":
            assert frame["source_split"].value_counts().to_dict() == {"train": 473963, "valid": 26101, "test": 26404}
        for line in ("morgan_rf", "mfp_lightgbm"):
            full = raw[(source_id, line, "full")]
            minus = raw[(source_id, line, "minus_p")]
            assert [col for col in full["dataset"]["column_roles"]["reactants"] if col != "p_smiles"] == minus["dataset"]["column_roles"]["reactants"]
            assert full["descriptors"][0]["columns"] == minus["descriptors"][0]["columns"] + ["p_smiles"]
            assert full["dataset"]["column_roles"]["products"] == ["p_smiles"]
            assert minus["dataset"]["column_roles"]["products"] == []
            assert full["dataset"]["numeric_conditions"] == minus["dataset"]["numeric_conditions"]
            assert full["evaluation"] == minus["evaluation"]
            assert full["model_params"] == minus["model_params"]
            assert full["benchmark"]["population_id"] == minus["benchmark"]["population_id"]
    print("passed: 8 standalone YAMLs; SuFEx 2122 rows; USPTO full 526468 rows; 4 paired comparisons; no modeling started")


if __name__ == "__main__":
    main()
