"""Strict benchmark executor AutoGluon outer-fold tests."""

from __future__ import annotations

from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.executor import execute_fold
from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
from yonod.splits.manifest import create_split_manifest


_AUTOGUON_CALLS = []


@contextmanager
def _pickle_backed_parquet_io():
    """Use pickle as a local stand-in when optional parquet engines are absent."""
    original_to_parquet = pd.DataFrame.to_parquet
    original_read_parquet = pd.read_parquet

    def to_parquet_pickle(self, path, *args, index=False, **kwargs):
        frame = self if index else self.reset_index(drop=True)
        frame.to_pickle(path)

    def read_parquet_pickle(path, *args, **kwargs):
        return pd.read_pickle(path)

    with patch.object(pd.DataFrame, "to_parquet", to_parquet_pickle), \
            patch("pandas.read_parquet", side_effect=read_parquet_pickle):
        yield


def _fake_autogluon_fit_predict_fold(self, X_train, y_train, X_valid, *, fold_index, context=None):
    train = np.asarray(X_train, dtype=float)
    valid = np.asarray(X_valid, dtype=float)
    target = np.asarray(y_train, dtype=float)
    context = dict(context or {})
    _AUTOGUON_CALLS.append({
        "fold_index": int(fold_index),
        "X_train": train.copy(),
        "X_valid": valid.copy(),
        "y_train": target.copy(),
        "context": context,
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
    })
    return np.full(len(valid), float(np.mean(target)), dtype=float), {
        "evaluation_protocol": context.get("evaluation_protocol", "manifest_outer_cv"),
        "fold_index": int(fold_index),
        "outer_seed": int(context.get("outer_seed", -1)),
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
        "random_state": self.random_state,
        "random_state_policy": "fake manifest outer split policy",
        "autogluon_versions": {"autogluon.tabular": "fake-1.1.1"},
        "train_time_s": 0.2,
        "predict_time_s": 0.03,
        "model_artifact_path": context.get("model_artifact_path"),
        "model_artifact_cleanup": bool(context.get("cleanup")),
        "leaderboard_rows": 1,
    }


class BenchmarkAutoGluonExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        _AUTOGUON_CALLS.clear()

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sample_id": [f"s{index:02d}" for index in range(8)],
            "yield": np.linspace(5.0, 75.0, 8),
            "a": [f"A_unique_{index}" for index in range(8)],
            "b": ["B0", "B1", "B0", "B1", "B2", "B3", "B2", "B3"],
        })

    def _contract(self, directory: Path, feature_sets, models):
        frame = self._frame()
        frame.to_csv(directory / "fixture.csv", index=False)
        descriptors = [{
            "id": item["name"], "descriptor": item["name"],
            "lifecycle": "fold_transform" if item["kind"] == "fold_transform" else "static_descriptor",
            "mode": "concat", "columns": item["component_cols"], "params": item.get("params", {}),
        } for item in feature_sets]
        (directory / "config.yaml").write_text(yaml.safe_dump({
            "schema_version": "2.0", "project_name": "benchmark-ag", "stage": "benchmark",
            "dataset": {
                "path": "fixture.csv", "sample_id_col": "sample_id",
                "column_roles": {"label": "yield", "reactants": ["a", "b"]},
            },
            "descriptors": descriptors,
            "artifacts": {"output_dir": "results"},
            "models": models,
            "model_params": {
                **({"rf": {"estimator": {"n_estimators": 3, "n_jobs": 1}}} if "rf" in models else {}),
                **({"autogluon": {"fit": {"time_limit": 7, "presets": "medium_quality", "num_cpus": 2}}} if "autogluon" in models else {}),
            },
            "evaluation": {
                "protocol": "manifest_outer_cv", "n_repeats": 1, "n_splits": 2, "seed": 901,
                "grouping": {"strategy": "repeated_kfold"},
            },
            "outputs": {"root": "results"},
            "benchmark": {"task_state": {"backend": "sqlite", "resumable": True}},
        }), encoding="utf-8")
        config = BenchmarkConfig.from_file(directory / "config.yaml")
        contract = create_benchmark_contract(config)
        manifest = create_split_manifest(contract)
        return frame, contract, manifest

    def _fold_indices(self, manifest: pd.DataFrame, sample_ids: np.ndarray, *, repeat: int = 1, fold: int = 1):
        part = manifest[(manifest["repeat"] == repeat) & (manifest["fold"] == fold)]
        valid_ids = set(part.loc[part["role"] == "valid", "sample_id"].astype(str))
        train_ids = set(part.loc[part["role"] == "train", "sample_id"].astype(str))
        train_idx = np.flatnonzero(np.isin(sample_ids, list(train_ids)))
        valid_idx = np.flatnonzero(np.isin(sample_ids, list(valid_ids)))
        return train_idx, valid_idx

    def test_static_autogluon_uses_same_manifest_rows_and_fold_local_numeric_scaler_as_rf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame, contract, manifest = self._contract(
                directory,
                feature_sets=[{"name": "mfp", "kind": "precomputed_descriptor", "component_cols": ["a", "b"], "params": {}}],
                models=["rf", "autogluon"],
            )
            sample_ids = frame["sample_id"].astype(str).to_numpy()
            X_smiles = np.arange(len(frame) * 3, dtype=float).reshape(len(frame), 3)
            X_numeric = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0], [1000.0], [1001.0]])
            y = frame["yield"].to_numpy(dtype=float)
            with _pickle_backed_parquet_io():
                rf_result = execute_fold(
                    contract, manifest, sample_ids, X_smiles, y,
                    "mfp", "rf", 1, 1, X_numeric=X_numeric,
                    model_kwargs={"n_estimators": 3, "n_jobs": 1},
                )
                with patch(
                    "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                    new=_fake_autogluon_fit_predict_fold,
                ):
                    ag_result = execute_fold(
                        contract, manifest, sample_ids, X_smiles, y,
                        "mfp", "autogluon", 1, 1, X_numeric=X_numeric,
                        model_kwargs={"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
                    )
                    resumed = execute_fold(
                        contract, manifest, sample_ids, X_smiles, y,
                        "mfp", "autogluon", 1, 1, X_numeric=X_numeric,
                        model_kwargs={"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
                    )
                rf_prediction = pd.read_parquet(rf_result.prediction_path)
                ag_prediction = pd.read_parquet(ag_result.prediction_path)
            metadata = json.loads(ag_result.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(rf_prediction["sample_id"].tolist(), ag_prediction["sample_id"].tolist())
        self.assertNotIn("evaluation_protocol", ag_prediction.columns)
        self.assertEqual(resumed.prediction_path, ag_result.prediction_path)
        self.assertEqual(len(_AUTOGUON_CALLS), 1)
        call = _AUTOGUON_CALLS[0]
        train_idx, valid_idx = self._fold_indices(manifest, sample_ids)
        scaler = StandardScaler()
        expected_train = np.hstack([X_smiles[train_idx], scaler.fit_transform(X_numeric[train_idx])])
        expected_valid = np.hstack([X_smiles[valid_idx], scaler.transform(X_numeric[valid_idx])])
        np.testing.assert_allclose(call["X_train"], expected_train)
        np.testing.assert_allclose(call["X_valid"], expected_valid)
        self.assertEqual(call["context"]["evaluation_protocol"], "manifest_outer_cv")
        self.assertEqual(call["context"]["outer_seed"], 901)
        self.assertTrue(call["context"]["model_artifact_path"].endswith("mfp__autogluon__r01__f01"))
        self.assertEqual(metadata["evaluation_protocol"], "manifest_outer_cv")
        self.assertEqual(metadata["autogluon_time_limit"], 7)
        self.assertEqual(metadata["autogluon_num_cpus"], 2)
        self.assertEqual(metadata["autogluon_version"], "fake-1.1.1")
        self.assertTrue(metadata["model_artifact_cleanup"])
        self.assertEqual(metadata["model_adapter_metadata"]["evaluation_protocol"], "manifest_outer_cv")

    def test_ohe_autogluon_records_fold_transform_metadata_and_blocks_validation_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame, contract, manifest = self._contract(
                directory,
                feature_sets=[{"name": "ohe", "kind": "fold_transform", "component_cols": ["a", "b"], "params": {}}],
                models=["autogluon"],
            )
            sample_ids = frame["sample_id"].astype(str).to_numpy()
            y = frame["yield"].to_numpy(dtype=float)
            with _pickle_backed_parquet_io():
                with patch(
                    "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                    new=_fake_autogluon_fit_predict_fold,
                ):
                    result = execute_fold(
                        contract, manifest, sample_ids, None, y,
                        "ohe", "autogluon", 1, 1,
                        model_kwargs={"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
                        component_frame=frame[["a", "b"]],
                        fold_transformer=ReactionComponentOHE(["a", "b"]),
                    )
                metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
                prediction = pd.read_parquet(result.prediction_path)

        transformer_meta = metadata["feature_transformer"]
        self.assertEqual(metadata["evaluation_protocol"], "manifest_outer_cv")
        self.assertEqual(metadata["model_adapter_metadata"]["evaluation_protocol"], "manifest_outer_cv")
        self.assertEqual(transformer_meta["fit_scope"], "train_only_per_fold")
        self.assertEqual(transformer_meta["component_cols"], ["a", "b"])
        self.assertGreater(transformer_meta["valid_unseen_component_count"], 0)
        self.assertNotIn("categories", transformer_meta)
        self.assertEqual(len(_AUTOGUON_CALLS), 1)
        first_block_width = int(transformer_meta["category_counts"][0])
        self.assertTrue(np.allclose(_AUTOGUON_CALLS[0]["X_valid"][:, :first_block_width], 0.0))
        self.assertEqual(len(prediction), int(metadata["n_valid"]))
        self.assertNotIn("evaluation_protocol", prediction.columns)


if __name__ == "__main__":
    unittest.main()
