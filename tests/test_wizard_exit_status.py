"""The public YAML launcher must expose failures to headless callers."""

import importlib.util
import io
import runpy
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_benchmark  # Import dependencies before mocking subprocess.run.


WIZARD_PATH = Path(__file__).resolve().parents[1] / "yonod.py"
SPEC = importlib.util.spec_from_file_location("wizard_exit_status", WIZARD_PATH)
wizard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wizard)


class WizardExitStatusTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(wizard, "step1_collect_basic_info", return_value={
            "is_config_file": True, "dataset_path": "task.yaml",
        }))
        self.validate = stack.enter_context(patch.object(wizard, "validate_config_file", return_value=True))
        self.load = stack.enter_context(patch("yonod.config.loader.load_run_config"))
        self.launch = stack.enter_context(patch.object(wizard.subprocess, "run"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.load.return_value = SimpleNamespace(effective={"stage": "all"})

    def test_invalid_yaml_returns_failure(self):
        self.validate.return_value = False
        self.assertEqual(wizard.main(), 1)
        self.load.assert_not_called()
        self.launch.assert_not_called()

    def test_regular_yaml_propagates_runtime_status(self):
        for returncode in (0, 1, 7):
            with self.subTest(returncode=returncode):
                self.launch.return_value = SimpleNamespace(returncode=returncode)
                self.assertEqual(wizard.main(), returncode)
                self.assertEqual(self.launch.call_args.args[0][-2:], ["--config", "task.yaml"])

    def test_regular_yaml_launch_exception_returns_failure(self):
        self.launch.side_effect = OSError("cannot start runtime")
        self.assertEqual(wizard.main(), 1)

    def test_benchmark_propagates_runtime_status(self):
        self.load.return_value.effective["stage"] = "benchmark"
        for returncode in (0, 1, 7):
            with self.subTest(returncode=returncode):
                with patch("scripts.run_benchmark.run_benchmark", return_value=returncode) as benchmark:
                    self.assertEqual(wizard.main(), returncode)
                    benchmark.assert_called_once_with(Path("task.yaml"))
        self.launch.assert_not_called()

    def test_benchmark_exception_returns_failure(self):
        self.load.return_value.effective["stage"] = "benchmark"
        with patch("scripts.run_benchmark.run_benchmark", side_effect=RuntimeError("strict failure")):
            self.assertEqual(wizard.main(), 1)
        self.launch.assert_not_called()

    def test_stage_load_exception_returns_failure(self):
        self.load.side_effect = ValueError("configuration changed after validation")
        self.assertEqual(wizard.main(), 1)
        self.launch.assert_not_called()

    def test_script_entrypoint_exits_nonzero_for_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.yaml"
            config.write_text("schema: invalid\n", encoding="utf-8")
            self.load.side_effect = ValueError("invalid schema")
            with patch("builtins.input", return_value=str(config)):
                with self.assertRaises(SystemExit) as exc:
                    runpy.run_path(str(WIZARD_PATH), run_name="__main__")
            self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
