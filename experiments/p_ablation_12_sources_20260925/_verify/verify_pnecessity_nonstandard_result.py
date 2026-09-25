"""Read-only per-task completion gate for the nonstandard source queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.layout import resolve_benchmark_output_layout
from yonod.config.loader import load_yaml_mapping
from yonod.config.contracts import validate_run_config


def load_for_verification(path: Path) -> BenchmarkConfig:
    """Read pinned historical declarations without relaxing the launch schema."""
    raw = load_yaml_mapping(path)
    roles = raw["dataset"]["column_roles"]
    if set(roles["reactants"]) & set(roles["products"]):
        queue = json.loads((ROOT / "result/pnecessity_nonstandard_prepare_v1/generated_config_manifest.json").read_text())
        entry = next(item for item in queue["queue"] if ROOT / item["config_path"] == path.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["config_sha256"]:
            raise ValueError("historical YAML hash drift")
        roles["reactants"] = [col for col in roles["reactants"] if col not in roles["products"]]
        return BenchmarkConfig._from_schema2(path.resolve(), validate_run_config(raw))
    return BenchmarkConfig.from_file(path)


def verify(path: Path) -> dict:
    config = load_for_verification(path)
    contract = create_benchmark_contract(config)
    task = config.outputs_root.name
    if not task.startswith("pnecessity_nonstandard_"):
        raise ValueError(f"unexpected task: {task}")
    # Require a unique layout, then validate its complete normalized contract
    # and content hash. Older contracts predate the nullable split-hash field.
    layout = resolve_benchmark_output_layout(contract.run_dir)
    historical_layouts = [resolve_benchmark_output_layout(p.parent.parent.parent)
                          for p in config.outputs_root.glob("*/docs/manifests/run_manifest.json")]
    candidates = [candidate for candidate in [layout, *historical_layouts]
                  if (candidate.manifests / "run_manifest.json").is_file()]
    if len(candidates) > 1:
        raise ValueError(f"{task}: ambiguous current and historical run manifests")
    if candidates:
        layout = candidates[0]
    run_path = layout.manifests / "run_manifest.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {run_path}")
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    expected_config = config.normalized_for_hash(contract.dataset_sha256)
    stored_config = manifest["benchmark_config"]
    if "external_split_manifest_sha256" not in stored_config and expected_config.get("external_split_manifest_sha256") is None:
        expected_config.pop("external_split_manifest_sha256", None)
    if expected_config != stored_config:
        raise ValueError(f"{task}: normalized YAML/data differs from stored run")
    stored_hash = hashlib.sha256(json.dumps(stored_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    expected_run_id = f"{contract.run_id.rsplit('-', 1)[0]}-{stored_hash[:12]}"
    for key, expected in (("run_id", expected_run_id), ("config_hash", stored_hash), ("dataset_sha256", contract.dataset_sha256)):
        if manifest.get(key) != expected:
            raise ValueError(f"{task}: {key} differs from current YAML/data")
    state = layout.state / "tasks.sqlite"
    with sqlite3.connect(state.as_uri() + "?mode=ro", uri=True) as connection:
        statuses = dict(connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall())
    if statuses != {"succeeded": 15}:
        raise ValueError(f"{task}: incomplete task state {statuses}")
    predictions = sorted(layout.predictions.glob("*.parquet"))
    folds = sorted(layout.folds.glob("*.json"))
    if len(predictions) != 15 or len(folds) != 15:
        raise ValueError(f"{task}: incomplete prediction/fold shards")
    completeness = pd.read_parquet(layout.metrics / "completeness.parquet")
    if completeness.empty or not bool(completeness["is_complete"].all()):
        raise ValueError(f"{task}: incomplete metrics")
    for name in ("benchmark_report.html", "benchmark_report.md"):
        report = layout.report / name
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"{task}: missing {name}")
    return {"task_name": task, "run_id": manifest["run_id"], "status": "passed", "succeeded_folds": 15}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
