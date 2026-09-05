"""Public feature-library contracts for MFP and fold-local OHE."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from yonod.descriptors import MFPDescriptor, OHEFeature, get_feature_provider, normalise_feature_specs
from yonod.descriptors.ohe import OHEFeatureError
from yonod.descriptors.registry import FeatureRegistryError


class PublicFeatureRegistryTests(unittest.TestCase):
    def test_mfp_and_ohe_have_distinct_lifecycles(self) -> None:
        self.assertEqual(get_feature_provider("mfp").lifecycle, "static_descriptor")
        self.assertEqual(get_feature_provider("ohe").lifecycle, "fold_transform")
        self.assertEqual(MFPDescriptor.__name__, "MFPDescriptor")

        specs = normalise_feature_specs([
            {"id": "mfp_r2", "descriptor": "mfp", "params": {"radius": 2, "fp_size": 64}},
            {"id": "ohe_conditions", "descriptor": "ohe", "columns": ["base", "solvent"]},
        ], label_col="yield")
        self.assertEqual([spec.id for spec in specs], ["mfp_r2", "ohe_conditions"])
        self.assertEqual(specs[0].params["profile"], "standard")
        self.assertEqual(specs[1].params["missing_policy"], "as_category")
        self.assertNotEqual(specs[0].schema_hash, specs[1].schema_hash)

    def test_invalid_public_declarations_fail_closed(self) -> None:
        with self.assertRaises(FeatureRegistryError):
            normalise_feature_specs([{"descriptor": "ohe"}], label_col="yield")
        with self.assertRaises(FeatureRegistryError):
            normalise_feature_specs([
                {"id": "same", "descriptor": "mfp"},
                {"id": "same", "descriptor": "mfp", "params": {"radius": 2}},
            ], label_col="yield")
        with self.assertRaises(FeatureRegistryError):
            normalise_feature_specs([
                {"descriptor": "ohe", "columns": ["yield"]},
            ], label_col="yield")


class OHEFeatureStateTests(unittest.TestCase):
    def test_train_only_unknown_missing_and_state_restore(self) -> None:
        train = pd.DataFrame({"base": ["K2CO3", "Cs2CO3", None], "solvent": ["THF", "THF", "DMF"]})
        valid = pd.DataFrame({"base": ["NaOtBu", None], "solvent": ["MeCN", "DMF"]})
        feature = OHEFeature(["base", "solvent"], missing_policy="zero_block")
        with self.assertRaises(OHEFeatureError):
            feature.transform(valid, partition="valid")

        feature.fit(train, sample_ids=["t1", "t2", "t3"])
        matrix = feature.transform(valid, partition="valid")
        metadata = feature.metadata()
        self.assertEqual(matrix.dtype, np.float32)
        self.assertGreaterEqual(metadata["valid_unseen_count"], 2)
        base_width = metadata["category_counts"][0]
        self.assertTrue(np.all(matrix[1, :base_width] == 0.0))

        context = {"feature_id": "ohe_conditions", "fold": 1, "train_sample_ids_hash": metadata["train_sample_ids_hash"]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            feature.save(directory, context=context)
            restored = OHEFeature.load(directory, expected_context=context)
            np.testing.assert_array_equal(
                restored.transform(valid, partition="predict"),
                feature.transform(valid, partition="predict"),
            )
            with self.assertRaises(OHEFeatureError):
                OHEFeature.load(directory, expected_context={**context, "fold": 2})
            metadata_path = directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sklearn_version"] = "incompatible-test-version"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(OHEFeatureError):
                OHEFeature.load(directory, expected_context=context)


if __name__ == "__main__":
    unittest.main()
