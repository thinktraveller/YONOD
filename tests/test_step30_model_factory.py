"""Focused step-30.6 tests for parameter routing without project defaults."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from yonod.model_factory import (
    ModelConfigurationError,
    materialise_fit_parameters,
    resolve_model_config,
)


class Step30ModelFactoryTests(unittest.TestCase):
    def _resolve(self, model: str, payload: dict):
        # These cases exercise routing and source preservation.  Library
        # construction is separately exercised in environments that provide
        # the optional model dependencies.
        with patch(
                "yonod.model_factory._library_catalog",
                return_value=(
                    {"n_estimators", "max_depth", "epsilon", "reg_lambda", "device"},
                    {"sample_weight", "eval_set", "tuning_data", "train_data", "unlabeled_data"},
                ),
        ):
            return resolve_model_config(model, payload)

    def test_omitted_sections_stay_empty_instead_of_receiving_project_values(self) -> None:
        config = self._resolve("rf", {})
        self.assertEqual(config.estimator, {})
        self.assertEqual(config.fit, {})
        self.assertEqual(config.parameter_sources, {})

    def test_nondefault_tree_and_weight_options_are_kept_in_their_real_api_sections(self) -> None:
        config = self._resolve("rf", {
            "estimator": {"n_estimators": 17, "max_depth": None},
            "fit": {"sample_weight": {"column": "reaction_weight"}},
        })
        self.assertEqual(config.estimator["n_estimators"], 17)
        self.assertIsNone(config.estimator["max_depth"])
        self.assertEqual(config.fit["sample_weight"], {"column": "reaction_weight"})
        self.assertEqual(config.parameter_sources["estimator.n_estimators"], "yaml")
        frame = pd.DataFrame({"reaction_weight": [1.0, 2.0, 3.0]})
        materialised = materialise_fit_parameters(config, frame, np.asarray([2, 0], dtype=int))
        np.testing.assert_allclose(materialised["sample_weight"], [3.0, 1.0])

    def test_unknown_or_unsafe_external_data_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "不受支持"):
            self._resolve("rf", {"estimator": {"not_a_real_parameter": 1}})
        with self.assertRaisesRegex(ModelConfigurationError, "外部数据对象"):
            self._resolve("autogluon", {"fit": {"tuning_data": "validation.csv"}})

    def test_xgb_device_policy_cannot_silently_conflict_with_explicit_device(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "不能同时"):
            self._resolve("xgb", {
                "estimator": {"device": "cpu"},
                "runtime": {"device_policy": "auto"},
            })

    def test_external_validation_data_is_rejected_and_inner_early_stopping_is_structured(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "外部数据/回调对象"):
            self._resolve("xgb", {"fit": {"eval_set": [["not", "data"]]}})
        config = self._resolve("xgb", {
            "runtime": {
                "early_stopping": {"rounds": 3, "validation_fraction": 0.25, "seed": 7},
            },
        })
        self.assertEqual(config.runtime["early_stopping"], {
            "rounds": 3, "validation_fraction": 0.25, "seed": 7,
        })


if __name__ == "__main__":
    unittest.main()
