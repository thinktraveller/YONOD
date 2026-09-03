"""Regression tests for the interactive wizard's model choices."""

from __future__ import annotations

import io
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
import pandas as pd

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

    def test_new_output_contract_separates_wizard_and_main_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs_dir, pictures_dir, report_dir = main._create_output_layout(root)
            self.assertTrue(docs_dir.is_dir())
            self.assertTrue(pictures_dir.is_dir())
            self.assertTrue(report_dir.is_dir())
            self.assertEqual(
                main._log_path_in_docs(Path("legacy/subdir/run.log"), docs_dir),
                docs_dir / "run.log",
            )

            frame = pd.DataFrame({"smiles": ["CC"], "yield": [0.5]})
            invalid_report = yonod_wizard.step3_1_generate_invalid_report(
                frame,
                [{"invalid_rows": [0], "origin_name": "smiles"}],
                str(root),
                "layout",
            )
            fixed_dataset = yonod_wizard.generate_fixed_dataset(
                frame, {0}, str(root), "layout",
            )
            config_path = yonod_wizard.save_config_file(
                str(root), "layout", "source.csv", None, [],
                [{"descriptor": "morgan"}], ["RandomForest"], {}, ["Markdown"],
            )

            self.assertEqual(Path(invalid_report), root / "report" / "layout_invalid_report.md")
            self.assertEqual(Path(fixed_dataset), root / "docs" / "layout_修复后数据集.csv")
            self.assertEqual(Path(config_path), root / "docs" / "layout_yonod_config.json")
            self.assertEqual(json.loads(Path(config_path).read_text(encoding="utf-8"))["project_name"], "layout")


if __name__ == "__main__":
    unittest.main()
