"""Step-31.4 focused tests for the verified Chemical VAE descriptor adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from yonod.descriptors.chemical_vae import ChemicalVaeDescriptor
from yonod.descriptors.chemical_vae_backend import ChemicalVaeAssetError
from yonod.descriptors.registry import FeatureRegistryError, normalise_feature_specs
from yonod.pipeline.features import run_features
from yonod.universal.feature_builder import build_universal_features


class ChemicalVaeDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.manifest = (
            cls.repository_root / "WEIGHTS" / "chemical_vae" / "zinc-37e96cd3bc8f9680" / "v5" / "conversion_manifest.json"
        )
        cls.properties_manifest = (
            cls.repository_root / "WEIGHTS" / "chemical_vae" / "zinc_properties-9f923d03c5ce558d" / "v5" / "conversion_manifest.json"
        )

    def test_verified_runtime_uses_persisted_state_and_preserves_identity_failures(self) -> None:
        # Descriptor runtime must not fall back to opening the original HDF5.
        with patch("yonod.descriptors.chemical_vae_backend._import_h5py", side_effect=AssertionError("HDF5 opened")):
            descriptor = ChemicalVaeDescriptor(model_manifest=self.manifest, batch_size=2)
            features, mask = descriptor.featurize(["CCO", "CCO", "", "C" * 121, "CCO.CC", "c1ccccc1"])
        self.assertEqual(features.shape, (6, 196))
        self.assertEqual(mask.tolist(), [True, True, False, False, False, True])
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[0], features[1])
        self.assertTrue(np.all(features[2:5] == 0.0))
        one_at_a_time, one_at_a_time_mask = ChemicalVaeDescriptor(
            model_manifest=self.manifest, batch_size=1
        ).featurize(["CCO", "c1ccccc1"])
        together, together_mask = ChemicalVaeDescriptor(
            model_manifest=self.manifest, batch_size=8
        ).featurize(["CCO", "c1ccccc1"])
        np.testing.assert_array_equal(one_at_a_time_mask, together_mask)
        np.testing.assert_allclose(one_at_a_time, together, atol=2e-5, rtol=2e-5)

    def test_concat_keeps_rows_with_one_successful_role_and_does_not_rewrite_text(self) -> None:
        frame = pd.DataFrame({
            "left": ["CCO", "C" * 121, "CCO.CC"],
            "right": ["CCO.CC", "c1ccccc1", "C" * 121],
        })
        matrix, numeric, mask = build_universal_features(
            smiles_cols=["left", "right"], numeric_cols=[], df=frame,
            desc_name="chemical_vae", mode="concat",
            descriptor_config={"params": {"model_manifest": str(self.manifest), "batch_size": 2}},
        )
        self.assertIsNone(numeric)
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertEqual(matrix.shape, (2, 392))
        self.assertTrue(np.all(matrix[0, 196:] == 0.0))
        self.assertTrue(np.all(matrix[1, :196] == 0.0))

    def test_separately_verified_zinc_properties_encoder_uses_persisted_state_only(self) -> None:
        # The separately validated properties *encoder* is selectable, while
        # its source HDF5/property head remains outside normal inference.
        with patch("yonod.descriptors.chemical_vae_backend._import_h5py", side_effect=AssertionError("HDF5 opened")):
            descriptor = ChemicalVaeDescriptor(model_manifest=self.properties_manifest, batch_size=2)
            features, mask = descriptor.featurize(["CCO", "c1ccccc1", "CCO.CC"])
        self.assertEqual(descriptor._runtime.variant, "zinc_properties")
        self.assertEqual(features.shape, (3, 196))
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[2] == 0.0))

    def test_registry_rejects_unverified_or_unknown_parameters(self) -> None:
        with self.assertRaisesRegex(FeatureRegistryError, "model_manifest"):
            normalise_feature_specs([{"descriptor": "chemical_vae", "params": {}}])
        with self.assertRaisesRegex(FeatureRegistryError, "未知字段"):
            normalise_feature_specs([{
                "descriptor": "chemical_vae",
                "params": {"model_manifest": "v4.json", "sampling": True},
            }])
        v3_manifest = self.manifest.parent.parent / "v3" / "conversion_manifest.json"
        with self.assertRaisesRegex(ChemicalVaeAssetError, "v5"):
            ChemicalVaeDescriptor(model_manifest=v3_manifest)

    def test_features_service_resolves_manifest_relative_to_owning_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"sample_id": ["a", "b"], "reactant": ["CCO", "c1ccccc1"]}).to_csv(root / "input.csv", index=False)
            config = {
                "schema_version": "2.0", "project_name": "chemical-vae-relative", "stage": "features",
                "dataset": {"path": "input.csv", "sample_id_col": "sample_id", "column_roles": {"reactants": ["reactant"]}},
                "descriptors": [{"id": "vae", "descriptor": "chemical_vae", "columns": ["reactant"], "params": {
                    "model_manifest": os.path.relpath(self.manifest, root), "batch_size": 1,
                }}],
                "artifacts": {"output_dir": "artifacts"},
            }
            config_path = root / "features.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            result = run_features(config_path)
            self.assertEqual(result.status, "ready")
            self.assertEqual(result.features[0].status, "ready")
            self.assertIsNotNone(result.features[0].manifest_path)

    def test_broken_vae_asset_isolated_from_another_static_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"sample_id": ["a", "b"], "reactant": ["CCO", "c1ccccc1"]}).to_csv(root / "input.csv", index=False)
            config = {
                "schema_version": "2.0", "project_name": "chemical-vae-isolation", "stage": "features",
                "dataset": {"path": "input.csv", "sample_id_col": "sample_id", "column_roles": {"reactants": ["reactant"]}},
                "descriptors": [
                    {"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]},
                    {"id": "vae", "descriptor": "chemical_vae", "columns": ["reactant"], "params": {"model_manifest": "missing.json"}},
                ],
                "artifacts": {"output_dir": "artifacts"},
            }
            config_path = root / "features.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            result = run_features(config_path)
            statuses = {item.feature_id: item for item in result.features}
            self.assertEqual(result.status, "partial")
            self.assertEqual(statuses["morgan"].status, "ready")
            self.assertEqual(statuses["vae"].status, "failed")
            self.assertIn("model_manifest", statuses["vae"].reason)


if __name__ == "__main__":
    unittest.main()
