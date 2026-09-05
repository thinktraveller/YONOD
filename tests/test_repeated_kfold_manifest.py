"""Regression coverage for the paper-aligned row-wise repeated KFold split."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.splits.manifest import create_split_manifest


class RepeatedKFoldManifestTests(unittest.TestCase):
    def _contract(self, directory: Path):
        frame = pd.DataFrame({
            "sample_id": ["s{0:02d}".format(index) for index in range(12)],
            "yield": np.linspace(10.0, 90.0, 12),
            "component_a": ["CCO", "CCC"] * 6,
            "component_b": ["O", "N"] * 6,
        })
        csv_path = directory / "fixture.csv"
        frame.to_csv(csv_path, index=False)
        config_path = directory / "paper.yaml"
        config_path.write_text(yaml.safe_dump({"benchmark": {
            "dataset_path": "fixture.csv",
            "sample_id_col": "sample_id",
            "label_col": "yield",
            "smiles_cols": ["component_a", "component_b"],
            "feature_sets": [
                {"name": "mfp", "kind": "precomputed_descriptor",
                 "component_cols": ["component_a", "component_b"],
                 "params": {"algorithm": "morgan_count", "radius": 3, "fp_size": 1024}},
                {"name": "ohe", "kind": "fold_transform",
                 "component_cols": ["component_a", "component_b"],
                 "params": {"encoder": "OneHotEncoder", "handle_unknown": "ignore"}},
            ],
            "models": ["rf"],
            "grouping": {"strategy": "repeated_kfold", "source_order": "raw_csv"},
            "cv": {"n_repeats": 5, "n_splits": 5, "seed": 1000},
            "outputs": {"root": "results"},
            "reproduction_protocol": {"name": "vjethbkm_rf_5x5"},
        }}, allow_unicode=True), encoding="utf-8")
        return create_benchmark_contract(BenchmarkConfig.from_file(config_path)), frame

    def test_matches_sklearn_row_by_row_and_has_complete_audit_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, frame = self._contract(Path(temporary))
            manifest = create_split_manifest(contract)

        self.assertEqual(len(manifest), len(frame) * 5 * 5)
        self.assertEqual(set(manifest["group_strategy"]), {"repeated_kfold"})
        self.assertTrue((manifest["group_id"] == manifest["sample_id"]).all())
        self.assertEqual(set(manifest["source_row_index"]), set(range(len(frame))))
        self.assertEqual(set(manifest["seed"]), set(range(1000, 1005)))
        self.assertEqual(len(manifest[["repeat", "fold"]].drop_duplicates()), 25)

        for repeat in range(1, 6):
            seed = 999 + repeat
            expected = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(frame))
            repeat_part = manifest[manifest["repeat"] == repeat]
            valid_per_row = repeat_part.loc[repeat_part["role"] == "valid", "source_row_index"].value_counts()
            self.assertTrue(valid_per_row.reindex(range(len(frame)), fill_value=0).eq(1).all())
            for fold, (_, expected_valid) in enumerate(expected, start=1):
                actual_valid = repeat_part.loc[
                    (repeat_part["fold"] == fold) & (repeat_part["role"] == "valid"),
                    "source_row_index",
                ].to_numpy()
                np.testing.assert_array_equal(np.sort(actual_valid), np.sort(expected_valid))

    def test_legacy_descriptors_expand_to_precomputed_feature_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = pd.DataFrame({
                "sample_id": ["a", "b", "c", "d"], "yield": [1, 2, 3, 4],
                "smiles": ["C", "CC", "CCC", "CCCC"],
            })
            frame.to_csv(directory / "fixture.csv", index=False)
            (directory / "legacy.yaml").write_text(yaml.safe_dump({"benchmark": {
                "dataset_path": "fixture.csv", "sample_id_col": "sample_id", "label_col": "yield",
                "smiles_cols": ["smiles"], "descriptors": ["morgan"], "models": ["rf"],
                "grouping": {"strategy": "component_holdout", "component_cols": ["smiles"]},
                "cv": {"n_repeats": 1, "n_splits": 2, "seed": 7},
            }}), encoding="utf-8")
            config = BenchmarkConfig.from_file(directory / "legacy.yaml")

        self.assertEqual(config.descriptors, ("morgan",))
        self.assertEqual(config.feature_sets[0]["kind"], "precomputed_descriptor")
        self.assertEqual(config.feature_sets[0]["component_cols"], ["smiles"])


if __name__ == "__main__":
    unittest.main()
