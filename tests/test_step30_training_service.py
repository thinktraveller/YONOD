"""Acceptance coverage for the independent step-30.7 train service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.reader import publish_feature_artifact
from yonod.model_factory import ResolvedModelConfig
from yonod.pipeline.features import run_features
from yonod.pipeline.training import TrainingServiceError, _default_fold_runner, run_all, run_train


class Step30TrainingServiceTests(unittest.TestCase):
    def _write_train_config(self, root: Path, manifest: Path) -> Path:
        frame = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "reactant": ["CC", "CCC", "O", "N"],
            "yield": [1.0, 2.0, 3.0, 4.0],
        })
        frame.to_csv(root / "data.csv", index=False)
        config = {
            "schema_version": "2.0",
            "project_name": "train-demo",
            "stage": "train",
            "dataset": {
                "path": "data.csv", "sample_id_col": "sample_id",
                "column_roles": {"label": "yield", "reactants": ["reactant"]},
            },
            "artifacts": {"input_manifest": str(manifest)},
            "models": ["rf"],
            "model_params": {"rf": {"estimator": {"n_estimators": 7}}},
            "evaluation": {"protocol": "outer_kfold", "n_splits": 2, "n_repeats": 1, "seed": 9, "shuffle": True},
            "outputs": {"root": "results"},
        }
        path = root / "train.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _resolved_rf(*args, **kwargs) -> ResolvedModelConfig:
        return ResolvedModelConfig("rf", {"n_estimators": 7}, {}, {}, {}, {"estimator.n_estimators": "yaml"})

    @staticmethod
    def _fixed_splits(sample_ids, evaluation):
        return [
            (1, 1, np.asarray([0, 1]), np.asarray([2, 3])),
            (1, 2, np.asarray([2, 3]), np.asarray([0, 1])),
        ]

    @staticmethod
    def _fake_metrics(y_true, y_pred):
        return 0.0, float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))), 0.5

    @staticmethod
    def _fake_runner(resolved, X_train, y_train, X_valid, aligned, train_index, fold_dir, label_col, fold):
        # Deliberately use only training labels: validation labels never enter
        # a model fit callback.
        return np.full(len(X_valid), float(np.mean(y_train))), {"train_time_s": 0.0, "predict_time_s": 0.0}

    def test_new_process_train_consumes_manifest_by_id_and_reuses_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Establish an artifact identity with the exact same feature input
            # content but without calling a descriptor generator in training.
            source = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "reactant": ["CC", "CCC", "O", "N"]})
            source_path = root / "feature-source.csv"
            source.to_csv(source_path, index=False)
            features_yaml = root / "features.yaml"
            features_yaml.write_text(yaml.safe_dump({
                "schema_version": "2.0", "project_name": "feature-id", "stage": "features",
                "dataset": {"path": "feature-source.csv", "sample_id_col": "sample_id", "column_roles": {"reactants": ["reactant"]}},
                "descriptors": [{"id": "fake", "descriptor": "morgan", "columns": ["reactant"]}],
                "artifacts": {"output_dir": "features"},
            }, allow_unicode=True, sort_keys=False), encoding="utf-8")

            feature_result = run_features(
                features_yaml,
                feature_computer=lambda *args: (np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), np.ones(4, dtype=bool)),
            )
            manifest = feature_result.features[0].manifest_path
            train_config = self._write_train_config(root, manifest)

            with patch("yonod.pipeline.training.validate_training_model_parameters", return_value={}), \
                    patch("yonod.pipeline.training.resolve_model_config", side_effect=self._resolved_rf), \
                    patch("yonod.pipeline.training._split_rows", side_effect=self._fixed_splits), \
                    patch("yonod.pipeline.training._metrics", side_effect=self._fake_metrics):
                first = run_train(train_config, fold_runner=self._fake_runner)
                second = run_train(train_config, fold_runner=self._fake_runner)

            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].status, "complete")
            self.assertEqual(second[0].status, "reused")
            predictions = pd.read_csv(first[0].predictions_path)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(set(predictions["sample_id"]), {"s1", "s2", "s3", "s4"})
            manifest_text = first[0].manifest_path.read_text(encoding="utf-8")
            self.assertIn("feature_content_identity", manifest_text)
            self.assertTrue((first[0].run_dir / "source_config.yaml").is_file())
            self.assertTrue((first[0].run_dir / "effective_config.yaml").is_file())

    def test_all_never_starts_training_when_every_feature_candidate_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"sample_id": ["s1", "s2"], "reactant": ["CC", "O"], "yield": [1.0, 2.0]}).to_csv(root / "data.csv", index=False)
            config = {
                "schema_version": "2.0", "project_name": "all-fail", "stage": "all",
                "dataset": {"path": "data.csv", "sample_id_col": "sample_id", "column_roles": {"label": "yield", "reactants": ["reactant"]}},
                "descriptors": [{"id": "bad", "descriptor": "morgan", "columns": ["reactant"]}],
                "artifacts": {"output_dir": "artifacts"}, "models": ["rf"],
            }
            path = root / "all.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            with patch("yonod.pipeline.training.validate_training_model_parameters", return_value={}):
                with self.assertRaisesRegex(TrainingServiceError, "所有特征候选"):
                    run_all(path, feature_computer=lambda *args: (_ for _ in ()).throw(RuntimeError("broken")))
            self.assertFalse((root / "artifacts" / "training_results").exists())
            # `run_train` receives only the already-published manifest; the
            # injected runner proves no descriptor callback is available on
            # this code path.

    def test_real_model_fit_failure_is_published_and_sibling_model_completes(self) -> None:
        """A runtime fit failure must not discard a different model's result."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "sample_id": ["s1", "s2", "s3", "s4"],
                "reactant": ["CC", "CCC", "O", "N"],
                "yield": [1.0, 2.0, 3.0, 4.0],
            })
            frame.to_csv(root / "data.csv", index=False)
            config = {
                "schema_version": "2.0", "project_name": "failure-isolation", "stage": "all",
                "dataset": {
                    "path": "data.csv", "sample_id_col": "sample_id",
                    "column_roles": {"label": "yield", "reactants": ["reactant"]},
                },
                "descriptors": [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}],
                "artifacts": {"output_dir": "artifacts"},
                # The factory can construct this RF, but sklearn rejects it
                # during actual fit. SVM remains a real successful sibling.
                "models": ["rf", "svm"],
                "model_params": {
                    "rf": {"estimator": {"n_estimators": 0, "random_state": 7}},
                    "svm": {"estimator": {"C": 1.0, "epsilon": 0.1}},
                },
                "evaluation": {
                    "protocol": "outer_kfold", "n_splits": 2, "n_repeats": 1,
                    "seed": 7, "shuffle": True,
                },
                "outputs": {"root": "results"},
            }
            path = root / "failure-isolation.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            result = run_all(
                path,
                feature_computer=lambda *_: (
                    np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
                    np.ones(4, dtype=bool),
                ),
            )

            runs = {run.model: run for run in result.training_runs}
            self.assertEqual(set(runs), {"rf", "svm"})
            self.assertEqual(runs["rf"].status, "failed")
            self.assertEqual(runs["svm"].status, "complete")
            failed_manifest = yaml.safe_load(runs["rf"].manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(failed_manifest["status"], "failed")
            self.assertEqual(failed_manifest["model"], "rf")
            self.assertIn("n_estimators", failed_manifest["failure"]["reason"])
            self.assertTrue((runs["rf"].run_dir / "source_config.yaml").is_file())
            self.assertTrue((runs["rf"].run_dir / "effective_config.yaml").is_file())
            self.assertFalse(runs["rf"].predictions_path.exists())
            predictions = pd.read_csv(runs["svm"].predictions_path)
            metrics = pd.read_csv(runs["svm"].fold_metrics_path)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(metrics), 2)
            self.assertTrue(np.isfinite(predictions["y_pred"]).all())

            # A retry gets new immutable failed evidence while the successful
            # run is verified and reused rather than overwritten.
            second = run_all(
                path,
                feature_computer=lambda *_: (
                    np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
                    np.ones(4, dtype=bool),
                ),
            )
            second_runs = {run.model: run for run in second.training_runs}
            self.assertEqual(second_runs["rf"].status, "failed")
            self.assertNotEqual(second_runs["rf"].run_dir, runs["rf"].run_dir)
            self.assertTrue(runs["rf"].manifest_path.is_file())
            self.assertEqual(second_runs["svm"].status, "reused")
            self.assertEqual(second_runs["svm"].run_dir, runs["svm"].run_dir)

    def test_svm_subsample_keeps_preprocessing_on_full_training_fold(self) -> None:
        """The costly SVR fit may be sampled without sampling scaler/PCA fit."""
        resolved = ResolvedModelConfig(
            "svm", {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}, {},
            {
                "scaler": {"enabled": True, "params": {}},
                "pca": {"enabled": False, "feature_threshold": None, "params": {}},
                "subsample": {"n_samples": 3, "random_state": 19},
            }, {}, {},
        )
        X_train = np.arange(60, dtype=float).reshape(6, 10)
        prediction, audit = _default_fold_runner(
            resolved,
            X_train,
            np.linspace(0.1, 0.6, 6),
            X_train[:2] + 0.5,
            pd.DataFrame({"sample_id": [f"s{index}" for index in range(8)]}),
            np.asarray([2, 3, 4, 5, 6, 7]),
            Path("unused"),
            "yield",
            2,
        )
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(audit["svm_subsample"]["preprocessing_fit_rows"], 6)
        self.assertEqual(audit["svm_subsample"]["svr_fit_rows"], 3)

    def test_tree_early_stopping_uses_only_an_inner_outer_train_split(self) -> None:
        X_train = np.arange(80, dtype=float).reshape(8, 10)
        X_valid = X_train[:2] + 0.5
        aligned = pd.DataFrame({"sample_id": [f"s{index}" for index in range(10)]})
        for model, estimator in (
            ("xgb", {"n_estimators": 12, "max_depth": 2, "n_jobs": 1, "verbosity": 0}),
            ("lightgbm", {"n_estimators": 12, "max_depth": 2, "n_jobs": 1, "verbosity": -1, "min_child_samples": 1}),
        ):
            resolved = ResolvedModelConfig(
                model, estimator, {}, {},
                {"early_stopping": {"rounds": 2, "validation_fraction": 0.25, "seed": 3}}, {},
            )
            prediction, audit = _default_fold_runner(
                resolved, X_train, np.linspace(0.1, 0.8, 8), X_valid,
                aligned, np.arange(8, dtype=int), Path("unused"), "yield", 1,
            )
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertEqual(audit["early_stopping"]["outer_training_rows"], 8)
            self.assertEqual(audit["early_stopping"]["inner_training_rows"], 6)
            self.assertEqual(audit["early_stopping"]["inner_validation_rows"], 2)


if __name__ == "__main__":
    unittest.main()
