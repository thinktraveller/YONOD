"""Paper-protocol MFP count semantics and artifact invariants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from yonod.descriptors.mfp import MFPDescriptor
from yonod.universal.descriptor_artifact import (
    load_descriptor_artifact,
    prepare_descriptor_artifact,
)
from yonod.universal.feature_builder import build_universal_features


class MFPDescriptorTests(unittest.TestCase):
    @staticmethod
    def _params(radius: int = 3):
        return {
            "algorithm": "morgan_count",
            "radius": radius,
            "fp_size": 1024,
            "input_normalization": "raw_csv_value",
            "blank_or_invalid_component": "zero_block_keep_row",
        }

    def test_single_component_matches_rdkit_count_reference_not_binary_bits(self):
        descriptor = MFPDescriptor(radius=3, fp_size=1024)
        actual, mask = descriptor.featurize(["CC(C)C"])
        reference = GetMorganGenerator(radius=3, fpSize=1024).GetCountFingerprintAsNumPy(
            Chem.MolFromSmiles("CC(C)C")
        )
        np.testing.assert_array_equal(actual[0], reference)
        self.assertTrue(mask[0])
        self.assertGreater(int(actual.max()), 1)

    def test_component_concat_uses_raw_order_and_retains_blank_invalid_rows(self):
        frame = pd.DataFrame({
            "first": ["CC(C)C", "", "not valid"],
            "second": ["c1ccccc1", "O", "N"],
        })
        X, _numeric, valid_mask = build_universal_features(
            smiles_cols=["first", "second"], numeric_cols=[], df=frame,
            desc_name="mfp", descriptor_config=self._params(),
        )
        self.assertEqual(X.shape, (3, 2048))
        self.assertTrue(valid_mask.all())
        self.assertTrue(np.all(X[1, :1024] == 0))
        self.assertTrue(np.all(X[2, :1024] == 0))
        self.assertGreater(int(X[0, :1024].max()), 1)
        # The second component block is independent from failures in the first.
        self.assertGreater(int(X[2, 1024:].sum()), 0)

    def test_artifact_hashes_params_and_records_zero_block_audit(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c"],
            "first": ["CC(C)C", None, "not valid"],
            "second": ["O", "N", "Cl"],
        })
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = prepare_descriptor_artifact(
                directory, descriptor="mfp", smiles_cols=["first", "second"], df=frame,
                sample_ids=frame["sample_id"].tolist(), sample_id_name="sample_id",
                descriptor_config=self._params(),
            )
            again = prepare_descriptor_artifact(
                directory, descriptor="mfp", smiles_cols=["first", "second"], df=frame,
                sample_ids=frame["sample_id"].tolist(), sample_id_name="sample_id",
                descriptor_config=self._params(),
            )
            changed = prepare_descriptor_artifact(
                directory, descriptor="mfp", smiles_cols=["first", "second"], df=frame,
                sample_ids=frame["sample_id"].tolist(), sample_id_name="sample_id",
                descriptor_config=self._params(radius=2),
            )
            artifact = load_descriptor_artifact(changed.path)

        self.assertEqual(first.status, "computed")
        self.assertEqual(again.status, "reused")
        self.assertEqual(changed.status, "recomputed")
        self.assertTrue(artifact.valid_mask.all())
        self.assertEqual(artifact.metadata["mfp_protocol"]["component_order"], ["first", "second"])
        self.assertEqual(artifact.metadata["mfp_protocol"]["block_dim"], 1024)
        self.assertEqual(artifact.metadata["mfp_protocol"]["blank_component_count"], 1)
        self.assertEqual(artifact.metadata["mfp_protocol"]["invalid_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
