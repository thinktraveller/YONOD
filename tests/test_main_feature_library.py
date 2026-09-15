"""Ordinary-entry integration tests for public MFP and OHE features."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import main
from yonod.descriptors.ohe import OHEFeature
from yonod.artifacts.reader import load_feature_artifact
from yonod.universal.descriptor_artifact import load_descriptor_artifact


class _StaticModel:
    def cross_validate(self, X, y, cv):
        return {
            "r2_mean": 0.0, "r2_std": 0.0, "rmse_mean": 1.0, "mae_mean": 1.0,
            "train_time_s": 0.0, "device": "cpu",
            "oof_pred": np.asarray(y, dtype=float), "oof_y_true": np.asarray(y, dtype=float),
        }


class MainFeatureLibraryTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "smiles": ["c1ccccc1", "CC", "CCC", "O", "N", "CO", "CN", "CCO", "CCN", "Cl"],
            "condition": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "unique-valid"],
            "yield": np.linspace(1.0, 10.0, 10),
        })

    def test_cli_mfp_creates_parameterized_count_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "input.csv"
            out_dir = root / "result"
            self._frame().to_csv(csv_path, index=False)
            with patch("main._make_model", return_value=_StaticModel()):
                result = main.main([
                    "--csv", str(csv_path), "--label-col", "yield", "--smiles-cols", "smiles",
                    "--descriptors", "mfp", "--mfp-radius", "2", "--mfp-fp-size", "64",
                    "--models", "rf", "--cv", "2", "--heartbeat", "0", "--output-dir", str(out_dir),
                    "--output-format", "md",
                ])
            self.assertEqual(result, 0)
            artifact = load_descriptor_artifact(out_dir / "descriptors" / "mfp.npz")
            self.assertEqual(artifact.X_smiles.shape, (10, 64))
            self.assertEqual(artifact.metadata["descriptor_config"]["feature_id"], "mfp")
            self.assertEqual(artifact.metadata["mfp_protocol"]["profile"], "standard")
            self.assertGreater(int(artifact.X_smiles.max()), 1)

    def test_yaml_ohe_uses_fold_state_not_global_encoded_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_name = "ordinary-ohe"
            dataset_path = root / f"{project_name}_normalized_dataset.csv"
            frame = self._frame()
            frame["sample_id"] = [f"s{i}" for i in range(len(frame))]
            frame["condition"] = [f"unique-{i}" for i in range(len(frame))]
            frame.to_csv(dataset_path, index=False)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "schema_version": "2.0", "stage": "all", "project_name": project_name,
                "dataset": {"path": dataset_path.name, "sample_id_col": "sample_id", "column_roles": {
                    "label": "yield", "reactants": ["smiles"], "products": [], "others": [],
                    "conditions": [], "categoricals": ["condition"],
                }},
                "descriptors": [{
                    "id": "condition_ohe", "descriptor": "ohe", "mode": "concat",
                    "columns": ["condition"],
                    "params": {"missing_policy": "as_category", "dtype": "float32", "handle_unknown": "ignore"},
                }],
                "artifacts": {"output_dir": "artifacts"},
                "models": ["rf"], "model_params": {"rf": {"estimator": {"n_jobs": 19}}},
                "evaluation": {"protocol": "outer_kfold", "n_splits": 5, "seed": 42, "shuffle": True},
                "outputs": {"root": "results", "report_formats": ["Markdown"]}, "metadata": {},
            }), encoding="utf-8")

            calls = []

            def runner(resolved, X_train, y_train, X_valid, *args):
                calls.append((X_train.copy(), X_valid.copy()))
                return np.full(len(X_valid), np.mean(y_train)), {}

            with patch("yonod.pipeline.training._default_fold_runner", side_effect=runner):
                result = main.main(["--config", str(config_path)])
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 5)
            for train, valid in calls:
                self.assertEqual(train.shape, (8, 8))
                self.assertEqual(valid.shape, (2, 8))
                np.testing.assert_array_equal(train.sum(axis=1), 1)
                np.testing.assert_array_equal(valid, 0)
            manifests = list((root / "results" / "runs").glob("*/run_manifest.yaml"))
            self.assertEqual(len(manifests), 1)
            run_dir = manifests[0].parent
            states = list((run_dir / "fold_transforms").glob("repeat-*/fold-*/transform"))
            self.assertEqual(len(states), 5)
            for state_dir in states:
                state = OHEFeature.load(state_dir)
                self.assertEqual(state.metadata()["fit_scope"], "train_only_per_fold")
                self.assertEqual(state.metadata()["category_counts"], [8])
            metrics = pd.read_csv(run_dir / "fold_metrics.csv")
            self.assertEqual(len(metrics), 5)
            for audit in metrics["transform_audit"].map(json.loads):
                self.assertEqual(audit["lifecycle"], "fold_transform")
            feature_manifests = list((root / "artifacts" / "features").rglob("manifest.yaml"))
            self.assertEqual(len(feature_manifests), 1)
            artifact = load_feature_artifact(feature_manifests[0])
            self.assertEqual(artifact.manifest["lifecycle"], "fold_transform")
            self.assertEqual(artifact.matrix.shape, (10, 1))
            np.testing.assert_array_equal(artifact.matrix[:, 0], frame["condition"])

    def test_legacy_json_is_rejected_with_migration_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps({"version": "1.1"}), encoding="utf-8")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                    patch("main._run_schema2_config") as run, \
                    patch("main._make_model") as make_model:
                self.assertEqual(main.main(["--json", str(config_path)]), 1)
            self.assertIn("migrate_config_to_yaml.py", stderr.getvalue())
            run.assert_not_called()
            make_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
