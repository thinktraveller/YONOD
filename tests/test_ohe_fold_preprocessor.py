"""Leakage guards for the paper-aligned fold-local OHE baseline."""

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

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.executor import execute_fold
from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
from yonod.splits.manifest import create_split_manifest


@contextmanager
def _pickle_backed_parquet_io():
    """Keep executor semantics testable when the optional runtime engine is absent."""
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


class FoldLocalOHETests(unittest.TestCase):
    def test_unseen_and_missing_components_are_zeroed_without_row_loss(self):
        train = pd.DataFrame({"a": ["C", "CC", None], "b": ["O", "N", "O"]})
        valid = pd.DataFrame({"a": ["CCC", ""], "b": ["N", "O"]})
        transformer = ReactionComponentOHE(["a", "b"]).fit(train, sample_ids=["t1", "t2", "t3"])
        X_train = transformer.transform(train, partition="train")
        X_valid = transformer.transform(valid, partition="valid")
        metadata = transformer.metadata()

        self.assertEqual(X_train.shape[0], len(train))
        self.assertEqual(X_valid.shape[0], len(valid))
        first_start, first_end = transformer._splits[0]
        self.assertTrue(np.all(X_train[2, first_start:first_end] == 0.0))
        self.assertTrue(np.all(X_valid[0, first_start:first_end] == 0.0))
        self.assertTrue(np.all(X_valid[1, first_start:first_end] == 0.0))
        self.assertNotIn("CCC", transformer._categories[0])
        self.assertEqual(metadata["fit_scope"], "train_only_per_fold")
        self.assertEqual(metadata["valid_missing_block_count"], 1)
        self.assertEqual(metadata["valid_unseen_component_count"], 1)
        self.assertIsNotNone(metadata["train_sample_ids_hash"])

    def test_each_fold_has_its_own_category_hash(self):
        first = ReactionComponentOHE(["a"]).fit(pd.DataFrame({"a": ["C", "CC"]}), sample_ids=["a", "b"])
        second = ReactionComponentOHE(["a"]).fit(pd.DataFrame({"a": ["C", "CCC"]}), sample_ids=["a", "c"])
        self.assertNotEqual(first.metadata()["categories_hash"], second.metadata()["categories_hash"])

    def test_executor_records_fold_transformer_metadata_without_global_matrix(self):
        frame = pd.DataFrame({
            "sample_id": ["s{0}".format(index) for index in range(10)],
            "yield": np.linspace(5.0, 95.0, 10),
            "a": ["C", "CC", "CCC", "CCCC", "CCCCC"] * 2,
            "b": ["O", "N", "O", "N", "O"] * 2,
        })
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame.to_csv(directory / "fixture.csv", index=False)
            (directory / "config.yaml").write_text(yaml.safe_dump({
                "schema_version": "2.0", "project_name": "ohe-fold", "stage": "benchmark",
                "dataset": {
                    "path": "fixture.csv", "sample_id_col": "sample_id",
                    "column_roles": {"label": "yield", "reactants": ["a", "b"]},
                },
                "descriptors": [{
                    "id": "ohe", "descriptor": "ohe", "lifecycle": "fold_transform", "mode": "concat",
                    "columns": ["a", "b"], "params": {},
                }],
                "artifacts": {"output_dir": "results"}, "models": ["rf"],
                "model_params": {"rf": {"estimator": {"n_estimators": 3, "n_jobs": 1}}},
                "evaluation": {
                    "protocol": "manifest_outer_cv", "n_repeats": 1, "n_splits": 2, "seed": 1000,
                    "grouping": {"strategy": "repeated_kfold"},
                },
                "outputs": {"root": "results"},
                "benchmark": {"task_state": {"backend": "sqlite", "resumable": True}},
            }), encoding="utf-8")
            contract = create_benchmark_contract(BenchmarkConfig.from_file(directory / "config.yaml"))
            manifest = create_split_manifest(contract)
            with _pickle_backed_parquet_io():
                result = execute_fold(
                    contract, manifest, frame["sample_id"].tolist(), None, frame["yield"].to_numpy(),
                    "ohe", "rf", 1, 1, model_kwargs={"n_estimators": 3, "n_jobs": 1},
                    component_frame=frame[["a", "b"]], fold_transformer=ReactionComponentOHE(["a", "b"]),
                )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["feature_transformer"]["fit_scope"], "train_only_per_fold")
        self.assertEqual(metadata["feature_transformer"]["component_cols"], ["a", "b"])
        self.assertGreater(metadata["feature_transformer"]["output_dim"], 0)
        self.assertNotIn("categories", metadata["feature_transformer"])


if __name__ == "__main__":
    unittest.main()
