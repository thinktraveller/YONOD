"""Read-only integrity check for one completed frozen external-test task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.frozen_external import EXTERNAL_PROTOCOL, FrozenExternalTestConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON root must be an object: {0}".format(path))
    return value


def verify(config_path: Path) -> dict[str, Any]:
    config = FrozenExternalTestConfig.from_file(config_path)
    root = config.outputs_root
    manifests = root / "docs" / "manifests"
    run = read_json(manifests / "run_manifest.json")
    populations = read_json(manifests / "dual_population_manifest.json")
    if run.get("evaluation_protocol") != EXTERNAL_PROTOCOL or run.get("single_fit") is not True:
        raise RuntimeError("run manifest is not a frozen single-fit external test")
    if run.get("cv_folds") != 0 or run.get("strict_rank_eligible") is not False:
        raise RuntimeError("external test is incorrectly marked as CV/rank eligible")
    if populations.get("no_split_manifest") is not True or populations.get("sample_id_intersection_count") != 0:
        raise RuntimeError("dual-population manifest claims a split or an ID overlap")
    if (manifests / "split_manifest.parquet").exists():
        raise RuntimeError("frozen external test must not emit a split manifest")

    train = pd.read_csv(config.dataset_path, usecols=[config.sample_id_col])
    external = pd.read_csv(
        config.external_path,
        usecols=[config.external_sample_id_col, config.external_label_col, config.external_source_id_col, config.external_group_col],
    )
    train_ids = train[config.sample_id_col].astype(str).tolist()
    external_ids = external[config.external_sample_id_col].astype(str).tolist()
    if set(train_ids).intersection(external_ids):
        raise RuntimeError("train and external sample IDs overlap")
    expected_train = populations["train_population"]
    expected_external = populations["external_population"]
    if expected_train["sha256"] != sha256(config.dataset_path) or expected_train["sample_id_sha256"] != values_sha256(train_ids):
        raise RuntimeError("train population fingerprint mismatch")
    if expected_external["sha256"] != sha256(config.external_path) or expected_external["sample_id_sha256"] != values_sha256(external_ids):
        raise RuntimeError("external population fingerprint mismatch")
    source_ids = external[config.external_source_id_col].astype(str).tolist()
    if expected_external["source_id_sha256"] != values_sha256(source_ids):
        raise RuntimeError("external source-ID fingerprint mismatch")

    metrics = pd.read_csv(root / "docs" / "metrics" / "external_test_metrics.csv")
    expected_task_count = len(config.feature_sets) * len(config.models)
    if len(metrics) != expected_task_count:
        raise RuntimeError("metrics task count is incomplete")
    checked: list[dict[str, Any]] = []
    for feature in config.feature_sets:
        for model in config.models:
            suffix = "{0}__{1}".format(feature.name, model)
            prediction = pd.read_parquet(root / "docs" / "predictions" / (suffix + ".parquet"))
            metadata = read_json(root / "docs" / "folds" / (suffix + ".json"))
            if len(prediction) != len(external_ids) or prediction["sample_id"].astype(str).nunique() != len(external_ids):
                raise RuntimeError("external prediction rows are incomplete: {0}".format(suffix))
            if set(prediction["sample_id"].astype(str)) != set(external_ids):
                raise RuntimeError("external prediction IDs differ from external population: {0}".format(suffix))
            if prediction["external_group_id"].astype(str).nunique() != external[config.external_group_col].astype(str).nunique():
                raise RuntimeError("external bootstrap group count mismatch: {0}".format(suffix))
            y_true = prediction["y_true"].to_numpy(dtype=float)
            y_pred = prediction["y_pred"].to_numpy(dtype=float)
            expected_y = external.set_index(config.external_sample_id_col).loc[prediction["sample_id"].astype(str), config.external_label_col].to_numpy(dtype=float)
            if not (y_true == expected_y).all() or not all(math.isfinite(value) for value in y_pred):
                raise RuntimeError("external predictions/labels invalid: {0}".format(suffix))
            required_metadata = {
                "partition": "frozen_development_to_external_test",
                "n_train": len(train_ids),
                "n_external": len(external_ids),
                "external_label_phase": "loaded_after_predict",
                "external_labels_used_for_fit": False,
                "external_labels_used_for_scaling": False,
                "external_labels_used_for_model_selection": False,
            }
            for key, expected in required_metadata.items():
                if metadata.get(key) != expected:
                    raise RuntimeError("metadata {0!r} mismatch for {1}".format(key, suffix))
            metric = metrics.loc[(metrics["descriptor"] == feature.name) & (metrics["model"] == model)]
            if len(metric) != 1 or int(metric.iloc[0]["cv_folds"]) != 0 or bool(metric.iloc[0]["strict_rank_eligible"]):
                raise RuntimeError("metric row has invalid external-test protocol flags: {0}".format(suffix))
            observed = {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                "r2": r2_score(y_true, y_pred),
            }
            for key, value in observed.items():
                if abs(float(metric.iloc[0][key]) - value) > 1e-12:
                    raise RuntimeError("metric recomputation mismatch for {0}: {1}".format(suffix, key))
            checked.append({"descriptor": feature.name, "model": model, "metrics": observed})
    for report in (root / "report" / "external_test_report.html", root / "report" / "external_test_report.md"):
        if not report.is_file() or run["run_id"] not in report.read_text(encoding="utf-8"):
            raise RuntimeError("missing or mismatched report: {0}".format(report))
    return {
        "run_id": run["run_id"],
        "n_train": len(train_ids),
        "n_external": len(external_ids),
        "n_external_groups": int(external[config.external_group_col].astype(str).nunique()),
        "checked_tasks": checked,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
