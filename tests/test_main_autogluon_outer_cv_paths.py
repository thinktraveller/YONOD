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
import yaml
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
    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict(self, frame):
        self.valid_frame = frame.copy()
        return np.full(len(frame), self.fit_kwargs["train_data"]["yield"].mean())

    def leaderboard(self, silent=True):
        return pd.DataFrame()


class MainAutoGluonOuterCVPathTests(unittest.TestCase):
    def test_static_mfp_autogluon_uses_outer_cv_and_yaml_params(self) -> None:
        frame = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(10)],
            "smiles": ["C", "CC", "CCC", "CCCC", "CO", "CN", "CCO", "CCN", "O", "N"],
            "yield": np.linspace(1.0, 10.0, 10),
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_name = "main-static-ag"
            frame.to_csv(root / f"{project_name}_normalized_dataset.csv", index=False)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "schema_version": "2.0", "stage": "all",
                "project_name": project_name,
                "dataset": {"path": f"{project_name}_normalized_dataset.csv", "sample_id_col": "sample_id", "column_roles": {
                    "label": "yield", "reactants": ["smiles"], "products": [],
                    "others": [], "conditions": [], "categoricals": [],
                }},
                "descriptors": [{
                    "id": "mfp_small", "descriptor": "mfp", "mode": "concat",
                    "columns": ["smiles"],
                    "params": {"radius": 2, "fp_size": 32, "profile": "standard"},
                }],
                "artifacts": {"output_dir": "artifacts"},
                "models": ["autogluon"],
                "model_params": {"autogluon": {"fit": {"time_limit": 6, "presets": "medium_quality", "num_cpus": 19}}},
                "evaluation": {"protocol": "outer_kfold", "n_splits": 5, "seed": 42, "shuffle": True},
                "outputs": {"root": "results", "report_formats": ["Markdown"]},
                "metadata": {},
            }), encoding="utf-8")
            probes = []
            model_paths = []

            def factory(resolved, *, label, path):
                self.assertEqual(resolved.model, "autogluon")
                self.assertEqual(label, "yield")
                probe = _StaticAutoGluonProbe()
                probes.append(probe)
                model_paths.append(path)
                return probe, {}

            with patch("yonod.pipeline.training.construct_autogluon_predictor", side_effect=factory):
                result = main.main(["--config", str(config_path)])

            self.assertEqual(result, 0)
            self.assertEqual(len(probes), 5)
            self.assertEqual(len(set(model_paths)), 5)
            for probe, (train_idx, _) in zip(probes, KFold(5, shuffle=True, random_state=42).split(frame)):
                kwargs = probe.fit_kwargs
                self.assertEqual(kwargs["time_limit"], 6)
                self.assertEqual(kwargs["presets"], "medium_quality")
                self.assertEqual(kwargs["num_cpus"], 19)
                self.assertNotIn("tuning_data", kwargs)
                self.assertEqual(kwargs["train_data"].shape, (8, 33))
                np.testing.assert_allclose(kwargs["train_data"]["yield"], frame.iloc[train_idx]["yield"])
                self.assertEqual(probe.valid_frame.shape, (2, 32))
                self.assertNotIn("yield", probe.valid_frame)
            manifests = list((root / "results" / "runs").glob("*/run_manifest.yaml"))
            self.assertEqual(len(manifests), 1)
            manifest = yaml.safe_load(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["evaluation"]["protocol"], "outer_kfold")
            self.assertEqual(manifest["n_folds"], 5)
            metrics = pd.read_csv(manifests[0].parent / "fold_metrics.csv")
            self.assertEqual(len(metrics), 5)
            self.assertTrue({"rmse", "mae", "r2"}.issubset(metrics.columns))
            for audit in metrics["model_audit"].map(json.loads):
                self.assertEqual(audit["effective_fit_parameters"]["num_cpus"], 19)
            predictions = pd.read_csv(manifests[0].parent / "predictions.csv")
            self.assertEqual(len(predictions), len(frame))
            self.assertEqual(set(predictions["sample_id"]), set(frame["sample_id"]))
            self.assertTrue(predictions["sample_id"].is_unique)
            self.assertTrue(np.isfinite(predictions["y_pred"]).all())

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
