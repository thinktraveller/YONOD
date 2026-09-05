"""Regression tests for the interactive wizard's model choices."""

from __future__ import annotations

import io
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import numpy as np
import pandas as pd
from scripts import run_benchmark
from yonod.universal.descriptor_artifact import load_descriptor_artifact, prepare_descriptor_artifact
from yonod.universal.report import generate_markdown_report, generate_report

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
            self.assertEqual(Path(fixed_dataset), root / "docs" / "layout_fixed_dataset.csv")
            self.assertEqual(Path(config_path), root / "docs" / "layout_yonod_config.json")
            self.assertEqual(json.loads(Path(config_path).read_text(encoding="utf-8"))["project_name"], "layout")


class DescriptorArtifactTests(unittest.TestCase):
    def test_reuses_matching_artifact_and_recomputes_changed_input(self) -> None:
        frame = pd.DataFrame({"smiles": ["CC", "CCC", "O"], "yield": [0.1, 0.2, 0.3]})
        calls = []

        def compute(**kwargs):
            calls.append(kwargs["df"]["smiles"].tolist())
            count = len(kwargs["df"])
            return np.arange(count * 2, dtype=np.float32).reshape(count, 2), None, np.ones(count, dtype=bool)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = prepare_descriptor_artifact(
                directory, descriptor="morgan", smiles_cols=["smiles"], df=frame,
                sample_ids=["a", "b", "c"], compute=compute,
            )
            self.assertEqual(first.status, "computed")
            self.assertEqual(len(calls), 1)
            second = prepare_descriptor_artifact(
                directory, descriptor="morgan", smiles_cols=["smiles"], df=frame,
                sample_ids=["a", "b", "c"],
                compute=lambda **_: self.fail("matching artifact must not recompute"),
            )
            self.assertEqual(second.status, "reused")
            artifact = load_descriptor_artifact(second.path)
            self.assertEqual(artifact.sample_ids.tolist(), ["a", "b", "c"])
            self.assertEqual(artifact.metadata["feature_structure"]["kind"], "column_blocks")

            changed = frame.copy()
            changed.loc[1, "smiles"] = "N"
            third = prepare_descriptor_artifact(
                directory, descriptor="morgan", smiles_cols=["smiles"], df=changed,
                sample_ids=["a", "b", "c"], compute=compute,
            )
            self.assertEqual(third.status, "recomputed")
            self.assertEqual(len(calls), 2)

    def test_failure_status_is_rendered_in_both_reports(self) -> None:
        metrics = pd.DataFrame(columns=main._CSV_COLUMNS)
        task_info = {
            "task_name": "failure-report",
            "n_combinations": 0,
            "descriptor_statuses": [{
                "descriptor": "fisd", "status": "failed", "stage": "precompute",
                "artifact_path": "descriptors/fisd.npz", "n_total": 3,
                "n_valid": None, "feature_dim": None, "skipped_model_count": 2,
                "reason": "RuntimeError: missing weights",
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            html = generate_report(metrics, task_info, root).read_text(encoding="utf-8")
            markdown = generate_markdown_report(metrics, task_info, root).read_text(encoding="utf-8")
            self.assertIn("描述符预计算状态", html)
            self.assertIn("missing weights", html)
            self.assertIn("描述符预计算状态", markdown)
            self.assertIn("missing weights", markdown)

    def test_main_reuses_files_and_isolates_descriptor_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "input.csv"
            out_dir = root / "result"
            pd.DataFrame({
                "smiles": ["CC", "CCC", "O", "N"],
                "yield": [0.1, 0.2, 0.3, 0.4],
            }).to_csv(csv_path, index=False)
            calls = []

            def compute(**kwargs):
                descriptor = kwargs["desc_name"]
                calls.append(descriptor)
                if descriptor == "maccs":
                    raise RuntimeError("synthetic descriptor failure")
                count = len(kwargs["df"])
                return np.ones((count, 3), dtype=np.float32), None, np.ones(count, dtype=bool)

            class FakeModel:
                def cross_validate(self, X, y, cv):
                    return {
                        "r2_mean": 0.5, "r2_std": 0.1, "rmse_mean": 0.2,
                        "mae_mean": 0.1, "train_time_s": 0.01, "device": "cpu",
                    }

            common = [
                "--csv", str(csv_path), "--label-col", "yield", "--smiles-cols", "smiles",
                "--task-name", "two-stage", "--output-dir", str(out_dir),
                "--cv", "2", "--heartbeat", "0", "--output-format", "md",
            ]
            original_stdout = sys.stdout
            try:
                with patch("main.build_universal_features", side_effect=compute), patch("main._make_model", return_value=FakeModel()):
                    self.assertEqual(main.main(common + ["--descriptors", "morgan", "maccs", "--models", "rf"]), 0)
            finally:
                sys.stdout = original_stdout

            self.assertEqual(calls, ["morgan", "maccs"])
            self.assertTrue((out_dir / "descriptors" / "morgan.npz").is_file())
            statuses = pd.read_csv(out_dir / "docs" / "descriptor_status.csv")
            failed = statuses.loc[statuses["descriptor"] == "maccs"].iloc[0]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(int(failed["skipped_model_count"]), 1)
            self.assertIn("synthetic descriptor failure", (out_dir / "report" / "report.md").read_text(encoding="utf-8"))

            original_stdout = sys.stdout
            try:
                with patch("main.build_universal_features", side_effect=AssertionError("must reuse")), patch("main._make_model", return_value=FakeModel()):
                    self.assertEqual(main.main(common + ["--descriptors", "morgan", "--models", "xgb", "--cv", "3"]), 0)
            finally:
                sys.stdout = original_stdout
            self.assertEqual(pd.read_csv(out_dir / "docs" / "descriptor_status.csv").iloc[0]["status"], "reused")

    def test_strict_benchmark_prepares_files_before_fold_execution(self) -> None:
        frame = pd.DataFrame({
            "reaction_id": ["r1", "r2"],
            "smiles": ["CC", "O"],
            "yield": [0.1, 0.2],
        })
        config = SimpleNamespace(
            sample_id_col="reaction_id",
            smiles_cols=("smiles",),
            descriptors=("morgan",),
            models=("rf", "xgb"),
        )
        calls = []

        def compute(**kwargs):
            calls.append(kwargs["desc_name"])
            return np.ones((2, 4), dtype=np.float32), None, np.ones(2, dtype=bool)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with patch("scripts.run_benchmark.build_universal_features", side_effect=compute):
                features, failures = run_benchmark._prepare_feature_artifacts(config, frame, run_dir)
            self.assertEqual(failures, {})
            self.assertEqual(features["morgan"][0].shape, (2, 4))
            self.assertEqual(calls, ["morgan"])
            self.assertTrue((run_dir / "descriptors" / "morgan.npz").is_file())

            with patch("scripts.run_benchmark.build_universal_features", side_effect=AssertionError("must reuse")):
                features, failures = run_benchmark._prepare_feature_artifacts(config, frame, run_dir)
            self.assertEqual(failures, {})
            self.assertEqual(features["morgan"][0].shape, (2, 4))
            self.assertEqual(pd.read_csv(run_dir / "docs" / "descriptor_status.csv").iloc[0]["status"], "reused")


if __name__ == "__main__":
    unittest.main()
