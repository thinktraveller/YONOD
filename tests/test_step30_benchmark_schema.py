"""Focused schema-2 strict benchmark adapter and protocol guards."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from yonod.benchmark.config import BenchmarkConfig, BenchmarkConfigError, create_benchmark_contract
from yonod.benchmark.metrics import tukey_hsd_comparisons
from yonod.benchmark.executor import FoldExecutionError, _fit_predict_model
from yonod.splits.manifest import create_split_manifest


class Step30BenchmarkSchemaTests(unittest.TestCase):
    def test_factory_preserves_snapshot_and_binds_paper_seed_without_mutating_config(self) -> None:
        config = {"estimator": {"n_estimators": 3, "max_features": 0.3, "n_jobs": 1}}
        original = copy.deepcopy(config)
        arguments = dict(
            model="rf", effective_model_kwargs={},
            X_train=np.arange(8, dtype=float).reshape(4, 2),
            y_train=np.asarray([0.1, 0.4, 0.7, 0.9]),
            X_valid=np.asarray([[2.0, 3.0]]), fold=1, protocol_seed=1000,
            config_seed=17, layout=None, suffix="seed-test",
            schema2_model_config=config,
        )
        prediction, _, _, snapshot, _ = _fit_predict_model(**arguments)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(snapshot["max_features"], 0.3)
        self.assertEqual(snapshot["random_state"], 1000)
        self.assertEqual(snapshot["effective_estimator_params"]["random_state"], 1000)
        self.assertEqual(config, original)
        config["estimator"]["random_state"] = 17
        with self.assertRaisesRegex(FoldExecutionError, "manifest seed"):
            _fit_predict_model(**arguments)
        # An ordinary non-paper run retains its explicitly configured seed.
        arguments["protocol_seed"] = None
        _, _, _, snapshot, _ = _fit_predict_model(**arguments)
        self.assertEqual(snapshot["random_state"], 17)

    @staticmethod
    def _config() -> dict:
        return {
            "schema_version": "2.0",
            "project_name": "strict-schema-fixture",
            "stage": "benchmark",
            "dataset": {
                "path": "fixture.csv",
                "sample_id_col": "sample_id",
                "column_roles": {"label": "yield", "reactants": ["reactant"]},
            },
            "descriptors": [{
                "id": "morgan-r2",
                "descriptor": "morgan",
                "lifecycle": "static_descriptor",
                "mode": "concat",
                "columns": ["reactant"],
                "params": {"radius": 2},
            }],
            "artifacts": {"output_dir": "out"},
            "models": ["rf"],
            "model_params": {"rf": {"estimator": {"n_estimators": 5, "n_jobs": 1, "random_state": 17}}},
            "evaluation": {
                "protocol": "manifest_outer_cv",
                "n_repeats": 1,
                "n_splits": 2,
                "seed": 17,
                "grouping": {"strategy": "repeated_kfold", "source_order": "raw_csv"},
            },
            "outputs": {"root": "out"},
            "benchmark": {"task_state": {"backend": "sqlite", "resumable": True}},
        }

    def _write(self, root: Path, config: dict) -> Path:
        pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "reactant": ["C", "CC", "CCC", "CCCC"],
            "yield": [0.1, 0.2, 0.3, 0.4],
        }).to_csv(root / "fixture.csv", index=False)
        path = root / "benchmark.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_schema2_adapter_preserves_feature_identity_grouping_and_resumable_external_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), self._config())
            config = BenchmarkConfig.from_file(path)
            contract = create_benchmark_contract(config)
            manifest = create_split_manifest(contract)

        self.assertEqual(config.descriptors, ("morgan-r2",))
        self.assertEqual(config.feature_sets[0]["algorithm"], "morgan")
        self.assertEqual(config.feature_sets[0]["component_cols"], ["reactant"])
        self.assertEqual(config.raw["evaluation_protocol"], "manifest_outer_cv")
        self.assertEqual(config.task_state, {"backend": "sqlite", "resumable": True})
        self.assertTrue(config.artifact_output_dir == config.outputs_root)
        self.assertEqual(len(manifest[["repeat", "fold"]].drop_duplicates()), 2)
        self.assertEqual(set(manifest["group_strategy"]), {"repeated_kfold"})
        self.assertEqual(set(manifest["role"]), {"train", "valid"})
        self.assertIn("schema2_model_params", contract.manifest["benchmark_config"])

    def test_legacy_root_outer_kfold_external_manifest_and_unsupported_tree_stopping_all_fail_closed(self) -> None:
        cases: list[tuple[str, dict, str]] = []
        legacy = {"benchmark": {"dataset_path": "fixture.csv"}}
        cases.append(("legacy", legacy, "schema_version"))
        ordinary = self._config()
        ordinary["evaluation"]["protocol"] = "outer_kfold"
        cases.append(("ordinary", ordinary, "manifest_outer_cv"))
        external = self._config()
        external["evaluation"]["split_manifest"] = "other.parquet"
        cases.append(("external", external, "逐行等价导入"))
        early = self._config()
        early["models"] = ["xgb"]
        early["model_params"] = {
            "xgb": {
                "estimator": {"n_estimators": 5, "n_jobs": 1},
                "runtime": {"early_stopping": {"rounds": 2, "validation_fraction": 0.25, "seed": 17}},
            }
        }
        cases.append(("early", early, "内层"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, config, message in cases:
                path = self._write(root, copy.deepcopy(config))
                path = path.with_name(name + ".yaml")
                path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                with self.assertRaisesRegex(BenchmarkConfigError, message):
                    BenchmarkConfig.from_file(path)

    def test_output_and_artifact_roots_cannot_diverge(self) -> None:
        config = self._config()
        config["outputs"] = {"root": "different-out"}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(BenchmarkConfigError, "同一任务根目录"):
                BenchmarkConfig.from_file(self._write(Path(temporary), config))

    def test_single_candidate_smoke_does_not_require_tukey_dependency(self) -> None:
        """No multiple comparison means no optional statsmodels import."""
        fold_metrics = pd.DataFrame([
            {
                "run_id": "run", "config_hash": "hash", "split_id": "split",
                "evaluation_protocol": "manifest_outer_cv", "descriptor": "morgan",
                "model": "rf", "repeat": 1, "fold": 1,
                "r2": 0.1, "rmse": 0.2, "mae": 0.1,
            },
            {
                "run_id": "run", "config_hash": "hash", "split_id": "split",
                "evaluation_protocol": "manifest_outer_cv", "descriptor": "morgan",
                "model": "rf", "repeat": 1, "fold": 2,
                "r2": 0.2, "rmse": 0.1, "mae": 0.08,
            },
        ])
        self.assertTrue(tukey_hsd_comparisons(fold_metrics, dimension="model").empty)


if __name__ == "__main__":
    unittest.main()
