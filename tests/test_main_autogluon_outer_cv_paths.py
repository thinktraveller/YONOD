"""Ordinary main.py AutoGluon outer-CV entry path tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import main
from yonod.descriptors.registry import normalise_feature_specs


class _FoldAutoGluonProbe:
    def __init__(self, *, time_limit: int = 9, presets: str = "medium_quality", num_cpus: int = 2) -> None:
        self.time_limit = time_limit
        self.presets = presets
        self.num_cpus = num_cpus
        self.calls = []

    def fit_predict_fold(self, X_train, y_train, X_valid, *, fold_index, context=None):
        train = np.asarray(X_train, dtype=float)
        valid = np.asarray(X_valid, dtype=float)
        target = np.asarray(y_train, dtype=float)
        context = dict(context or {})
        self.calls.append({
            "fold_index": int(fold_index),
            "X_train": train.copy(),
            "X_valid": valid.copy(),
            "y_train": target.copy(),
            "context": context,
        })
        prediction = np.full(len(valid), float(np.mean(target)), dtype=float)
        return prediction, {
            "evaluation_protocol": context.get("evaluation_protocol", "outer_kfold"),
            "fold_index": int(fold_index),
            "outer_seed": int(context.get("outer_seed", 42)),
            "time_limit": self.time_limit,
            "presets": self.presets,
            "num_cpus": self.num_cpus,
            "random_state_policy": "fake outer split policy",
            "train_time_s": 0.01,
            "predict_time_s": 0.001,
        }


class _StaticAutoGluonProbe:
    def __init__(self, *, time_limit: int, presets: str, num_cpus: int) -> None:
        self.time_limit = time_limit
        self.presets = presets
        self.num_cpus = num_cpus
        self.calls = []

    def cross_validate(self, X, y, cv):
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        self.calls.append({"X": features.copy(), "y": target.copy(), "cv": int(cv)})
        return {
            "r2_mean": 1.0,
            "r2_std": 0.0,
            "rmse_mean": 0.0,
            "rmse_std": 0.0,
            "mae_mean": 0.0,
            "mae_std": 0.0,
            "train_time_s": 0.01,
            "predict_time_s": 0.001,
            "device": "cpu",
            "evaluation_protocol": "outer_kfold",
            "expected_folds": int(cv),
            "completed_folds": int(cv),
            "autogluon_time_limit": self.time_limit,
            "autogluon_presets": self.presets,
            "autogluon_num_cpus": self.num_cpus,
            "autogluon_seed_policy": "fake outer split policy",
            "oof_pred": target.copy(),
            "oof_y_true": target.copy(),
        }


class MainAutoGluonOuterCVPathTests(unittest.TestCase):
    def test_static_mfp_autogluon_uses_outer_cv_and_json_params(self) -> None:
        frame = pd.DataFrame({
            "smiles": ["C", "CC", "CCC", "CCCC", "CO", "CN", "CCO", "CCN", "O", "N"],
            "yield": np.linspace(1.0, 10.0, 10),
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_name = "main-static-ag"
            frame.to_csv(root / f"{project_name}_normalized_dataset.csv", index=False)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "version": "1.1",
                "project_name": project_name,
                "column_roles": {
                    "label": "yield", "reactants": ["smiles"], "products": [],
                    "others": [], "conditions": [], "categoricals": [],
                },
                "descriptors": [{
                    "id": "mfp_small", "descriptor": "mfp", "mode": "concat",
                    "columns": ["smiles"],
                    "params": {"radius": 2, "fp_size": 32, "profile": "standard"},
                }],
                "models": ["AutoGluon"],
                "model_params": {"AutoGluon": {"time_limit": 6, "presets": "medium_quality", "num_cpus": 2}},
                "report_formats": ["Markdown"],
                "metadata": {},
            }, ensure_ascii=False), encoding="utf-8")
            probe_box = {}

            def factory(model_name, args):
                self.assertEqual(model_name, "autogluon")
                self.assertEqual(args.autogluon_time_limit, 6)
                self.assertEqual(args.autogluon_presets, "medium_quality")
                self.assertEqual(args.autogluon_num_cpus, 2)
                probe = _StaticAutoGluonProbe(
                    time_limit=args.autogluon_time_limit,
                    presets=args.autogluon_presets,
                    num_cpus=args.autogluon_num_cpus,
                )
                probe_box["probe"] = probe
                return probe

            with patch("main._make_model", side_effect=factory):
                result = main.main(["--json", str(config_path)])

            self.assertEqual(result, 0)
            probe = probe_box["probe"]
            self.assertEqual(len(probe.calls), 1)
            self.assertEqual(probe.calls[0]["cv"], 5)
            self.assertEqual(probe.calls[0]["X"].shape, (10, 32))
            metrics = pd.read_csv(root / "docs" / "metrics_summary.csv")

        row = metrics.iloc[0]
        self.assertEqual(row["model"], "autogluon")
        self.assertEqual(row["evaluation_protocol"], "outer_kfold")
        self.assertEqual(int(row["expected_folds"]), 5)
        self.assertEqual(int(row["completed_folds"]), 5)
        self.assertEqual(int(row["autogluon_time_limit"]), 6)
        self.assertEqual(int(row["autogluon_num_cpus"]), 2)
        self.assertIn("rmse_std", metrics.columns)
        self.assertIn("mae_std", metrics.columns)

    def test_numeric_autogluon_scales_inside_each_outer_fold(self) -> None:
        X_smiles = np.arange(18, dtype=float).reshape(9, 2)
        X_numeric = np.array([[0.0], [1.0], [2.0], [50.0], [51.0], [52.0], [1000.0], [1001.0], [1002.0]])
        y = np.linspace(10.0, 90.0, 9)
        probe = _FoldAutoGluonProbe(time_limit=5, num_cpus=1)

        metrics = main._cv_with_numeric(
            "autogluon", probe, X_smiles, X_numeric, y, cv=3, svm_subsample=None
        )

        self.assertEqual(metrics["evaluation_protocol"], "outer_kfold")
        self.assertEqual(metrics["expected_folds"], 3)
        self.assertEqual(metrics["completed_folds"], 3)
        self.assertFalse(np.isnan(metrics["oof_pred"]).any())
        splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X_smiles))
        self.assertEqual(len(probe.calls), 3)
        for call, (train_idx, valid_idx) in zip(probe.calls, splits):
            scaler = StandardScaler()
            expected_train = np.hstack([X_smiles[train_idx], scaler.fit_transform(X_numeric[train_idx])])
            expected_valid = np.hstack([X_smiles[valid_idx], scaler.transform(X_numeric[valid_idx])])
            np.testing.assert_allclose(call["X_train"], expected_train)
            np.testing.assert_allclose(call["X_valid"], expected_valid)
            np.testing.assert_allclose(call["X_train"][:, -1].mean(), 0.0, atol=1e-12)
            self.assertEqual(call["context"]["evaluation_protocol"], "outer_kfold")

    def test_ohe_autogluon_does_not_learn_validation_only_categories(self) -> None:
        frame = pd.DataFrame({
            "condition": [f"valid_canary_{index}" for index in range(8)],
            "yield": np.linspace(1.0, 8.0, 8),
        })
        spec = normalise_feature_specs([{
            "id": "condition_ohe",
            "descriptor": "ohe",
            "mode": "concat",
            "columns": ["condition"],
            "params": {"missing_policy": "as_category", "dtype": "float32", "handle_unknown": "ignore"},
        }], label_col="yield")[0]
        args = SimpleNamespace(
            cv=2,
            autogluon_time_limit=9,
            autogluon_presets="medium_quality",
            autogluon_num_cpus=2,
            rf_n_jobs=1,
            rf_n_estimators=3,
            rf_max_depth=None,
        )
        probe = _FoldAutoGluonProbe(time_limit=9, presets="medium_quality", num_cpus=2)

        with tempfile.TemporaryDirectory() as temporary:
            state_root = Path(temporary) / "fold_transformers" / spec.id
            with patch("main._make_model", return_value=probe):
                metrics = main._cv_with_ohe_feature(
                    spec=spec,
                    frame=frame,
                    sample_ids=np.asarray([f"s{index}" for index in range(len(frame))], dtype=str),
                    numeric_cols=[],
                    y=frame["yield"].to_numpy(dtype=float),
                    model_name="autogluon",
                    args=args,
                    state_root=state_root,
                    svm_subsample=None,
                )
            summary = json.loads((state_root / "summary.json").read_text(encoding="utf-8"))
            fold_metadata = [
                json.loads((Path(item["state_directory"]) / "metadata.json").read_text(encoding="utf-8"))
                for item in summary["folds"]
            ]

        self.assertEqual(metrics["evaluation_protocol"], "outer_kfold")
        self.assertEqual(metrics["completed_folds"], 2)
        self.assertEqual(metrics["autogluon_time_limit"], 9)
        self.assertEqual(len(probe.calls), 2)
        for call in probe.calls:
            self.assertEqual(call["context"]["evaluation_protocol"], "outer_kfold")
            self.assertTrue(np.allclose(call["X_valid"], 0.0))
        self.assertEqual(summary["fit_scope"], "train_only_per_fold")
        self.assertTrue(all(item["valid_unseen_count"] == 4 for item in summary["folds"]))
        self.assertTrue(all(item["category_counts"] == [4] for item in fold_metadata))
        self.assertTrue(all(item["fit_scope"] == "train_only_per_fold" for item in fold_metadata))


if __name__ == "__main__":
    unittest.main()
