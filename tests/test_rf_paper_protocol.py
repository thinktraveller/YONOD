"""RF hyperparameter and per-repeat seed evidence for the paper protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.executor import FoldExecutionError, execute_fold
from yonod.splits.manifest import create_split_manifest


class PaperRFProtocolTests(unittest.TestCase):
    def _setup(self, directory: Path):
        frame = pd.DataFrame({
            "sample_id": ["s{0:02d}".format(index) for index in range(15)],
            "yield": np.linspace(5.0, 95.0, 15),
            "a": ["C", "CC", "CCC"] * 5,
        })
        frame.to_csv(directory / "fixture.csv", index=False)
        params = {"n_estimators": 500, "max_features": 0.3, "n_jobs": -1}
        (directory / "config.yaml").write_text(yaml.safe_dump({"benchmark": {
            "dataset_path": "fixture.csv", "sample_id_col": "sample_id", "label_col": "yield",
            "smiles_cols": ["a"], "feature_sets": [
                {"name": "mfp", "kind": "precomputed_descriptor", "component_cols": ["a"], "params": {}}
            ],
            "models": ["rf"], "model_params": {"rf": params},
            "grouping": {"strategy": "repeated_kfold"},
            "cv": {"n_repeats": 5, "n_splits": 5, "seed": 1000},
            "outputs": {"root": "results"},
            "reproduction_protocol": {"name": "vjethbkm_rf_5x5"},
        }}), encoding="utf-8")
        config = BenchmarkConfig.from_file(directory / "config.yaml")
        return frame, config, create_benchmark_contract(config), params

    def test_manifest_seed_overrides_each_paper_rf_repeat_and_snapshots_params(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, _config, contract, params = self._setup(Path(temporary))
            manifest = create_split_manifest(contract)
            X = np.arange(len(frame) * 4, dtype=float).reshape(len(frame), 4)
            metadata = []
            for repeat in range(1, 6):
                result = execute_fold(
                    contract, manifest, frame["sample_id"].tolist(), X, frame["yield"].to_numpy(),
                    "mfp", "rf", repeat, 1, model_kwargs=params,
                )
                metadata.append(json.loads(result.metadata_path.read_text(encoding="utf-8")))

        self.assertEqual([item["manifest_seed"] for item in metadata], list(range(1000, 1005)))
        self.assertEqual([item["model_random_seed"] for item in metadata], list(range(1000, 1005)))
        defaults = RandomForestRegressor().get_params()
        for item in metadata:
            snapshot = item["estimator_params_snapshot"]
            self.assertEqual(snapshot["n_estimators"], 500)
            self.assertEqual(snapshot["max_features"], 0.3)
            self.assertEqual(snapshot["n_jobs"], -1)
            self.assertEqual(snapshot["min_samples_leaf"], defaults["min_samples_leaf"])
            self.assertEqual(item["requested_model_kwargs"], params)
            self.assertEqual(item["effective_model_kwargs"]["random_state"], item["manifest_seed"])

    def test_conflicting_configured_random_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, _config, contract, params = self._setup(Path(temporary))
            manifest = create_split_manifest(contract)
            with self.assertRaises(FoldExecutionError):
                execute_fold(
                    contract, manifest, frame["sample_id"].tolist(), np.ones((len(frame), 2)),
                    frame["yield"].to_numpy(), "mfp", "rf", 1, 1,
                    model_kwargs={**params, "random_state": 42},
                )


if __name__ == "__main__":
    unittest.main()
