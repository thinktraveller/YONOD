"""Regression tests for the interactive wizard's model choices."""

from __future__ import annotations

import io
import importlib.util
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main

_WIZARD_SPEC = importlib.util.spec_from_file_location(
    "yonod_wizard", Path(__file__).resolve().parents[1] / "yonod.py"
)
assert _WIZARD_SPEC is not None and _WIZARD_SPEC.loader is not None
yonod_wizard = importlib.util.module_from_spec(_WIZARD_SPEC)
_WIZARD_SPEC.loader.exec_module(yonod_wizard)


class WizardModelSelectionTests(unittest.TestCase):
    def test_lightgbm_is_listed_and_maps_to_a_supported_cli_model(self) -> None:
        output = io.StringIO()
        with patch("builtins.input", return_value="5"), redirect_stdout(output):
            selected = yonod_wizard.step5_select_models()

        self.assertEqual(selected, ["LightGBM"])
        self.assertIn("[5] LightGBM", output.getvalue())
        self.assertIn("可选 lightgbm 依赖", output.getvalue())
        self.assertEqual(main._map_model_name("LightGBM"), "lightgbm")
        self.assertIn("lightgbm", main._MODEL_NAMES)

    def test_wizard_lightgbm_config_is_retained_by_main(self) -> None:
        config = {
            "project_name": "wizard-smoke",
            "column_roles": {"label": "yield", "reactants": ["reactant"]},
            "descriptors": [{"descriptor": "morgan"}],
            "models": ["LightGBM"],
            "_dataset_path": Path("wizard-smoke_normalized_dataset.csv"),
        }

        args = main.config_to_args(config, Path("wizard-smoke_yonod_config.json"))

        self.assertEqual(args.models, ["lightgbm"])


if __name__ == "__main__":
    unittest.main()
