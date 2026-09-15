"""Focused tests for the step-31.3 encoder conversion primitives."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from yonod.descriptors.chemical_vae_backend import (
    ChemicalVaeAssetError,
    converted_payload,
    converted_torch_encoder_forward,
    load_chemical_vae_encoder_asset,
    numpy_encoder_forward,
    smiles_to_right_padded_one_hot,
)


class ChemicalVaeEncoderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.asset = load_chemical_vae_encoder_asset(
            cls.repository_root / "reference-proejct" / "chemical_vae" / "models" / "zinc" / "zinc_encoder.h5",
            exp_path=cls.repository_root / "reference-proejct" / "chemical_vae" / "models" / "zinc" / "exp.json",
            charset_path=cls.repository_root / "reference-proejct" / "chemical_vae" / "models" / "zinc" / "zinc.json",
        )

    def test_original_right_padding_uses_space_token(self) -> None:
        one_hot = smiles_to_right_padded_one_hot(["CCO"], self.asset)
        space_index = self.asset.charset.index(" ")
        self.assertEqual(one_hot.shape, (1, 120, 35))
        self.assertEqual(float(one_hot[0, 3, space_index]), 1.0)
        self.assertEqual(float(one_hot[0, 119, space_index]), 1.0)
        self.assertTrue(np.allclose(one_hot.sum(axis=2), 1.0))

    def test_unknown_character_and_overlong_input_fail_before_inference(self) -> None:
        with self.assertRaisesRegex(ChemicalVaeAssetError, "unsupported character"):
            smiles_to_right_padded_one_hot(["CCO.CC"], self.asset)
        with self.assertRaisesRegex(ChemicalVaeAssetError, "exceeding MAX_LEN"):
            smiles_to_right_padded_one_hot(["C" * 121], self.asset)

    def test_persisted_torch_state_matches_independent_numpy_forward(self) -> None:
        one_hot = smiles_to_right_padded_one_hot(
            ["CCO", "c1ccccc1", "O=C=O", "C1CC1", "C" * 120], self.asset
        )
        reference, reference_layers = numpy_encoder_forward(one_hot, self.asset, capture_layers=True)
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "encoder_state_dict.pt"
            torch.save(converted_payload(self.asset, converter_version="test/v1"), state_path)
            candidate, candidate_layers = converted_torch_encoder_forward(
                one_hot, state_path, self.asset, capture_layers=True
            )
        self.assertTrue(np.allclose(reference, candidate, atol=2e-5, rtol=2e-5))
        self.assertEqual(set(reference_layers), set(candidate_layers))
        for name, reference_layer in reference_layers.items():
            with self.subTest(layer=name):
                self.assertTrue(
                    np.allclose(reference_layer, candidate_layers[name], atol=2e-5, rtol=2e-5)
                )

    def test_zinc_properties_persisted_state_matches_its_own_numpy_reference(self) -> None:
        """The extra variant has a distinct source hash and must not reuse ZINC."""

        properties_dir = self.repository_root / "reference-proejct" / "chemical_vae" / "models" / "zinc_properties"
        asset = load_chemical_vae_encoder_asset(
            properties_dir / "zinc_encoder.h5",
            exp_path=properties_dir / "exp.json",
            charset_path=properties_dir / "zinc.json",
            variant="zinc_properties",
        )
        self.assertNotEqual(asset.source_sha256, self.asset.source_sha256)
        one_hot = smiles_to_right_padded_one_hot(["CCO", "c1ccccc1", "C" * 120], asset)
        reference, reference_layers = numpy_encoder_forward(one_hot, asset, capture_layers=True)
        state_path = (
            self.repository_root / "WEIGHTS" / "chemical_vae" / "zinc_properties-9f923d03c5ce558d"
            / "v5" / "encoder_state_dict.pt"
        )
        candidate, candidate_layers = converted_torch_encoder_forward(
            one_hot, state_path, asset, capture_layers=True
        )
        self.assertTrue(np.allclose(reference, candidate, atol=2e-5, rtol=2e-5))
        for name, reference_layer in reference_layers.items():
            with self.subTest(layer=name):
                self.assertTrue(
                    np.allclose(reference_layer, candidate_layers[name], atol=2e-5, rtol=2e-5)
                )

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") is not None,
        "optional direct-HDF5 parity reference requires tensorflow-cpu",
    )
    def test_original_hdf5_tensorflow_execution_matches_independent_numpy_forward(self) -> None:
        """The source graph itself agrees with the non-Torch layer reference.

        This regression deliberately uses the original ``.h5`` model rather
        than the converted state dict, so a shared conversion defect cannot
        make both sides agree.  The optional dependency is intentionally not
        needed for ordinary project tests or eventual descriptor inference.
        """

        converter_path = self.repository_root / "scripts" / "convert_chemical_vae_encoder.py"
        spec = importlib.util.spec_from_file_location("chemical_vae_converter_test", converter_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        one_hot = smiles_to_right_padded_one_hot(
            ["CCO", "c1ccccc1", "O=C=O", "C" * 120], self.asset
        )
        tensorflow_output, tensorflow_layers, environment = converter._tensorflow_reference_forward(
            one_hot,
            self.asset.source_path,
            required_layers=converter.PARITY_PROTOCOL["required_layers"],
            capture_layers=True,
        )
        numpy_output, numpy_layers = numpy_encoder_forward(one_hot, self.asset, capture_layers=True)
        self.assertEqual(environment["tensorflow_version"], "2.15.1")
        self.assertTrue(np.allclose(numpy_output, tensorflow_output, atol=2e-5, rtol=2e-5))
        self.assertEqual(set(numpy_layers), set(tensorflow_layers))
        for name, numpy_layer in numpy_layers.items():
            with self.subTest(layer=name):
                self.assertTrue(
                    np.allclose(numpy_layer, tensorflow_layers[name], atol=2e-5, rtol=2e-5)
                )


if __name__ == "__main__":
    unittest.main()
