"""Regression tests for CSV loading in the interactive wizard."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


_WIZARD_SPEC = importlib.util.spec_from_file_location(
    "yonod_wizard_csv_loading", Path(__file__).resolve().parents[1] / "yonod.py"
)
assert _WIZARD_SPEC is not None and _WIZARD_SPEC.loader is not None
yonod_wizard = importlib.util.module_from_spec(_WIZARD_SPEC)
_WIZARD_SPEC.loader.exec_module(yonod_wizard)


class WizardCsvLoadingTests(unittest.TestCase):
    def test_delimiter_only_empty_rows_are_ignored_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "input.csv"
            csv_path.write_text("reactant,yield\nCC,0.5\n,,\n,,\n", encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                frame = yonod_wizard.load_and_preview_dataset(str(csv_path))

        self.assertEqual(frame.shape, (1, 2))
        self.assertIn("已忽略 2 行仅包含分隔符的空记录", output.getvalue())
        self.assertNotIn("CSV 格式错误", output.getvalue())

    def test_precheck_keeps_nonempty_field_count_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "input.csv"
            csv_path.write_text("reactant,yield\nCC,0.5,unexpected\n", encoding="utf-8")

            bad_lines = yonod_wizard._precheck_csv_format(str(csv_path))

        self.assertEqual(len(bad_lines), 1)
        self.assertEqual(bad_lines[0]["line_num"], 2)
        self.assertEqual(bad_lines[0]["actual"], 3)


if __name__ == "__main__":
    unittest.main()
