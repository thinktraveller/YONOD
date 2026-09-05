"""Wizard configuration contracts for public MFP and OHE features."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "yonod_wizard_feature_library", Path(__file__).resolve().parents[1] / "yonod.py"
)
assert _SPEC is not None and _SPEC.loader is not None
wizard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wizard)


class WizardFeatureLibraryTests(unittest.TestCase):
    def test_default_mfp_has_public_feature_id_and_parameters(self) -> None:
        config = wizard.generate_default_descriptor_config("mfp", [
            {"name": "reactant", "role": "reactant"},
            {"name": "catalyst", "role": "others"},
        ])
        self.assertEqual(config["id"], "mfp")
        self.assertEqual(config["columns"], ["reactant", "catalyst"])
        self.assertEqual(
            config["params"], {"radius": 3, "fp_size": 1024, "profile": "standard"}
        )

    def test_ohe_requires_an_explicit_ordered_column_choice(self) -> None:
        columns = [
            {"name": "reactant", "role": "reactant"},
            {"name": "solvent", "role": "condition"},
            {"name": "yield", "role": "label"},
        ]
        with patch("builtins.input", side_effect=["1", "2"]):
            config = wizard.step4_configure_descriptor("ohe", columns)
        self.assertEqual(config["id"], "ohe")
        self.assertEqual(config["columns"], ["solvent"])
        self.assertEqual(config["params"]["missing_policy"], "zero_block")


if __name__ == "__main__":
    unittest.main()
