"""AutoGluon outer-fold adapter contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from yonod.models.autogluon_model import AutoGluonYieldModel


class _FakeTabularPredictor:
    instances = []

    def __init__(self, *, label, problem_type, path, verbosity):
        self.label = label
        self.problem_type = problem_type
        self.path = path
        self.verbosity = verbosity
        self.fit_kwargs = None
        self.train_data = None
        self.predict_data = None
        _FakeTabularPredictor.instances.append(self)

    def fit(self, *, train_data, time_limit, presets, num_cpus):
        self.train_data = train_data.copy()
        self.fit_kwargs = {
            "time_limit": time_limit,
            "presets": presets,
            "num_cpus": num_cpus,
        }
        return self

    def predict(self, valid_data):
        self.predict_data = valid_data.copy()
        value = float(self.train_data["yield"].mean())
        return pd.Series(np.full(len(valid_data), value, dtype=float))

    def leaderboard(self, silent=True):
        return pd.DataFrame({"model": ["fake"]})


class AutoGluonFoldAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTabularPredictor.instances.clear()

    def test_fit_predict_fold_uses_external_train_and_valid_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = AutoGluonYieldModel(
                time_limit=17,
                presets="medium_quality",
                num_cpus=3,
                random_state=123,
                save_path=root,
                cleanup=False,
            )
            X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
            y_train = np.array([10.0, 20.0])
            X_valid = np.array([[101.0, 102.0], [103.0, 104.0]])

            with patch("yonod.models.autogluon_model._load_tabular_predictor", return_value=_FakeTabularPredictor):
                pred, metadata = model.fit_predict_fold(
                    X_train,
                    y_train,
                    X_valid,
                    fold_index=3,
                    context={"evaluation_protocol": "outer_kfold", "outer_seed": 123},
                )

        self.assertEqual(pred.tolist(), [15.0, 15.0])
        self.assertEqual(len(_FakeTabularPredictor.instances), 1)
        predictor = _FakeTabularPredictor.instances[0]
        self.assertEqual(predictor.label, "yield")
        self.assertEqual(predictor.problem_type, "regression")
        self.assertEqual(Path(predictor.path).name, "fold-03")
        self.assertEqual(predictor.fit_kwargs, {
            "time_limit": 17,
            "presets": "medium_quality",
            "num_cpus": 3,
        })
        self.assertEqual(list(predictor.train_data.columns), ["f0", "f1", "yield"])
        self.assertEqual(list(predictor.predict_data.columns), ["f0", "f1"])
        self.assertFalse((predictor.train_data[["f0", "f1"]].to_numpy() >= 100).any())
        self.assertTrue((predictor.predict_data.to_numpy() >= 100).all())
        self.assertEqual(metadata["evaluation_protocol"], "outer_kfold")
        self.assertEqual(metadata["fold_index"], 3)
        self.assertEqual(metadata["outer_seed"], 123)
        self.assertFalse(metadata["model_artifact_cleanup"])
        self.assertEqual(metadata["leaderboard_rows"], 1)

    def test_cross_validate_runs_outer_kfold_and_completes_oof(self) -> None:
        X = np.arange(24, dtype=float).reshape(8, 3)
        y = np.arange(8, dtype=float)
        model = AutoGluonYieldModel(
            time_limit=5,
            presets="medium_quality",
            num_cpus=2,
            random_state=11,
            cleanup=False,
        )

        with patch("yonod.models.autogluon_model._load_tabular_predictor", return_value=_FakeTabularPredictor):
            metrics = model.cross_validate(X, y, cv=4)

        self.assertEqual(metrics["evaluation_protocol"], "outer_kfold")
        self.assertEqual(metrics["expected_folds"], 4)
        self.assertEqual(metrics["completed_folds"], 4)
        self.assertFalse(np.isnan(metrics["oof_pred"]).any())
        self.assertEqual(len(metrics["fold_metadata"]), 4)
        expected_splits = list(KFold(n_splits=4, shuffle=True, random_state=11).split(X))
        for predictor, (_, valid_idx) in zip(_FakeTabularPredictor.instances, expected_splits):
            np.testing.assert_array_equal(predictor.predict_data.to_numpy(), X[valid_idx])
        self.assertIn("rmse_std", metrics)
        self.assertIn("mae_std", metrics)

    def test_holdout_path_is_explicitly_marked_internal(self) -> None:
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10, dtype=float)
        model = AutoGluonYieldModel(random_state=5, cleanup=False)

        with patch("yonod.models.autogluon_model._load_tabular_predictor", return_value=_FakeTabularPredictor):
            metrics = model.fit_evaluate_holdout(X, y, holdout_frac=0.3)

        self.assertEqual(metrics["evaluation_protocol"], "autogluon_internal_holdout")
        self.assertEqual(metrics["expected_folds"], 1)
        self.assertEqual(metrics["completed_folds"], 1)
        self.assertEqual(metrics["holdout_frac"], 0.3)
        self.assertEqual(metrics["r2_std"], 0.0)
        self.assertEqual(metrics["rmse_std"], 0.0)
        self.assertEqual(metrics["mae_std"], 0.0)
        self.assertEqual(np.isfinite(metrics["oof_pred"]).sum(), 3)
        self.assertEqual(metrics["fold_metadata"][0]["evaluation_protocol"], "autogluon_internal_holdout")

    def test_validation_errors_are_clear(self) -> None:
        model = AutoGluonYieldModel()
        with self.assertRaisesRegex(ValueError, "行数不一致"):
            model.fit_predict_fold(np.zeros((2, 2)), np.zeros(3), np.zeros((1, 2)), fold_index=1)
        with self.assertRaisesRegex(ValueError, "特征维度不一致"):
            model.fit_predict_fold(np.zeros((2, 2)), np.zeros(2), np.zeros((1, 3)), fold_index=1)
        with self.assertRaisesRegex(ValueError, "NaN 或 inf"):
            model.fit_predict_fold(np.array([[np.nan]]), np.zeros(1), np.zeros((1, 1)), fold_index=1)


if __name__ == "__main__":
    unittest.main()
