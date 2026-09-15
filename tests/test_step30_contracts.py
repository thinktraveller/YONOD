"""Focused acceptance tests for the step-30.1 contracts only."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from yonod.artifacts.contracts import (
    ArtifactContractError,
    artifact_content_identity,
    resolve_manifest_path,
    validate_artifact_manifest,
)
from yonod.config.contracts import (
    ConfigContractError,
    resolve_config_path,
    validate_operation_config,
    validate_run_config,
)


_DIGEST = "a" * 64


def _dataset(*, label: str | None = "yield") -> dict:
    roles = {"reactants": ["reactant"]}
    if label is not None:
        roles["label"] = label
    return {
        "path": "data/reactions.csv",
        "sample_id_col": "sample_id",
        "column_roles": roles,
    }


def _descriptor() -> dict:
    return {
        "id": "morgan-r3",
        "descriptor": "morgan",
        "mode": "concat",
        "columns": ["reactant"],
        "params": {"radius": 3},
    }


def _manifest() -> dict:
    return {
        "schema_version": "2.0",
        "artifact_id": "morgan-r3-v1",
        "feature_id": "morgan-r3",
        "lifecycle": "static_descriptor",
        "status": "ready",
        "files": {
            "matrix": {"path": "data/matrix.npz", "sha256": _DIGEST, "bytes": 20},
            "sample_ids": {"path": "data/sample_ids.csv", "sha256": _DIGEST, "bytes": 10},
            "validity_mask": {"path": "data/validity.npy", "sha256": _DIGEST, "bytes": 3},
        },
        "matrix": {
            "file": "matrix",
            "dtype": "float32",
            "shape": [2, 128],
            "columns": {"kind": "generated_pattern", "pattern": "morgan_<index>"},
        },
        "samples": {
            "file": "sample_ids",
            "id_field": "sample_id",
            "row_count": 3,
            "matrix_row_mapping": "valid_rows_in_sample_order",
        },
        "validity": {
            "file": "validity_mask",
            "n_total": 3,
            "n_valid": 2,
            "semantics": "true iff the corresponding sample has a matrix row",
        },
        "sources": {
            "dataset_identity": "sha256:dataset",
            "feature_config_identity": "sha256:feature-config",
        },
    }


class RunConfigContractTests(unittest.TestCase):
    def test_three_stages_have_distinct_minimum_contracts(self) -> None:
        features = {
            "schema_version": "2.0",
            "project_name": "demo",
            "stage": "features",
            "dataset": _dataset(label=None),
            "descriptors": [_descriptor()],
            "artifacts": {"output_dir": "artifacts/demo/v1"},
        }
        train = {
            "schema_version": "2.0",
            "project_name": "demo",
            "stage": "train",
            "dataset": _dataset(),
            "artifacts": {"input_manifest": "artifacts/demo/v1/manifest.yaml"},
            "models": ["rf"],
            "model_params": {"rf": {"estimator": {"n_estimators": 17}}},
        }
        all_config = {
            **features,
            "stage": "all",
            "dataset": _dataset(),
            "models": ["rf"],
            "evaluation": {"protocol": "outer_kfold", "n_splits": 2, "n_repeats": 1},
        }

        self.assertEqual(validate_run_config(features)["stage"], "features")
        self.assertEqual(validate_run_config(train)["stage"], "train")
        self.assertEqual(validate_run_config(all_config)["stage"], "all")

    def test_stage_and_model_route_errors_fail_before_loading_data(self) -> None:
        missing_label = {
            "schema_version": "2.0", "project_name": "demo", "stage": "train",
            "dataset": _dataset(label=None),
            "artifacts": {"input_manifest": "artifact.yaml"}, "models": ["rf"],
        }
        with self.assertRaisesRegex(ConfigContractError, "label"):
            validate_run_config(missing_label)

        malformed = {
            "schema_version": "2.0", "project_name": "demo", "stage": "all",
            "dataset": _dataset(), "descriptors": [_descriptor()],
            "artifacts": {"output_dir": "out"}, "models": ["rf"],
            "model_params": {"rf": {"predictor": {"label": "yield"}}},
        }
        with self.assertRaisesRegex(ConfigContractError, "无路由"):
            validate_run_config(malformed)

    def test_operation_is_model_free_and_paths_belong_to_owning_yaml(self) -> None:
        operation = {
            "schema_version": "2.0",
            "operation": "derive_features",
            "input_manifest": "../source/manifest.yaml",
            "output_dir": "derived/v2",
            "operations": [{"kind": "select_samples", "sample_ids": "keep.csv"}],
        }
        self.assertEqual(validate_operation_config(operation)["operation"], "derive_features")
        invalid = {**operation, "models": ["rf"]}
        with self.assertRaisesRegex(ConfigContractError, "未知字段"):
            validate_operation_config(invalid)

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "configs" / "run.yaml"
            self.assertEqual(
                resolve_config_path(config_path, "../data/input.csv"),
                (config_path.parent / "../data/input.csv").resolve(),
            )


class ArtifactManifestContractTests(unittest.TestCase):
    def test_ready_manifest_is_portable_and_identity_ignores_location(self) -> None:
        manifest = _manifest()
        self.assertEqual(validate_artifact_manifest(manifest)["artifact_id"], "morgan-r3-v1")
        relocated = copy.deepcopy(manifest)
        for record in relocated["files"].values():
            record["path"] = "relocated/" + record["path"].split("/")[-1]
        self.assertEqual(artifact_content_identity(manifest), artifact_content_identity(relocated))

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "copied" / "manifest.yaml"
            self.assertEqual(
                resolve_manifest_path(manifest_path, "data/matrix.npz"),
                (manifest_path.parent / "data/matrix.npz").resolve(),
            )

    def test_manifest_rejects_broken_row_mapping_and_unsafe_references(self) -> None:
        bad_mapping = _manifest()
        bad_mapping["matrix"]["shape"][0] = 3
        with self.assertRaisesRegex(ArtifactContractError, "n_valid"):
            validate_artifact_manifest(bad_mapping)

        unsafe = _manifest()
        unsafe["files"]["matrix"]["path"] = "../outside.npz"
        with self.assertRaisesRegex(ArtifactContractError, "包内路径"):
            validate_artifact_manifest(unsafe)

    def test_derived_manifest_requires_parent_operation_and_mapping(self) -> None:
        derived = _manifest()
        derived.update({
            "artifact_id": "morgan-r3-selected-v1",
            "lifecycle": "derived",
            "derived_from": {
                "artifact_id": "morgan-r3-v1",
                "operation": "select_samples",
                "input_output_mapping": {"kept-s1": "row-0"},
            },
        })
        self.assertEqual(validate_artifact_manifest(derived)["lifecycle"], "derived")
        derived["derived_from"]["input_output_mapping"] = {}
        with self.assertRaisesRegex(ArtifactContractError, "不能为空"):
            validate_artifact_manifest(derived)


if __name__ == "__main__":
    unittest.main()
