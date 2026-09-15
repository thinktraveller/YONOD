"""Step-30.8 tests for explicit, fail-closed legacy configuration migration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.migrate_config_to_yaml import migrate
from yonod.config.loader import load_run_config


class Step30ConfigMigrationTests(unittest.TestCase):
    def test_legacy_json_with_explicit_sample_id_becomes_schema2_yaml_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo_normalized_dataset.csv").write_text(
                "sample_id,reactant,yield\ns1,CC,1.0\ns2,O,2.0\n", encoding="utf-8"
            )
            source = root / "demo.json"
            source.write_text(json.dumps({
                "version": "1.1", "project_name": "demo",
                "column_roles": {"label": "yield", "reactants": ["reactant"]},
                "descriptors": [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}],
                "models": ["Random Forest"],
                "model_params": {"Random Forest": {"n_estimators": 17, "max_depth": None}},
            }), encoding="utf-8")
            output = root / "demo.yaml"
            report = migrate(source, output, sample_id_col="sample_id")

            self.assertEqual(report["status"], "converted")
            self.assertTrue(output.is_file())
            self.assertTrue((root / "demo.yaml.migration.json").is_file())
            loaded = load_run_config(output)
            self.assertEqual(loaded.effective["models"], ["rf"])
            # The old wizard's main.py JSON route ignored this RF field and
            # always instantiated its fixed 300-tree adapter profile.  The
            # converter records the difference instead of enabling a value
            # that never affected the historical run.
            self.assertEqual(loaded.effective["model_params"]["rf"]["estimator"]["n_estimators"], 300)
            self.assertTrue(report["ignored_legacy_fields"])

    def test_legacy_autogluon_values_are_the_only_old_model_params_that_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo_normalized_dataset.csv").write_text(
                "sample_id,reactant,yield\ns1,CC,1.0\ns2,O,2.0\n", encoding="utf-8"
            )
            source = root / "demo.json"
            source.write_text(json.dumps({
                "version": "1.1", "project_name": "demo",
                "column_roles": {"label": "yield", "reactants": ["reactant"]},
                "descriptors": [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}],
                "models": ["AutoGluon", "SVM"],
                "model_params": {
                    "AutoGluon": {"time_limit": 17, "presets": "medium_quality", "num_cpus": 2, "save_path": "ignored"},
                    "SVM": {"C": 99},
                },
            }), encoding="utf-8")
            output = root / "demo.yaml"
            report = migrate(source, output, sample_id_col="sample_id")

            self.assertEqual(report["status"], "converted")
            loaded = load_run_config(output)
            self.assertEqual(loaded.effective["model_params"]["autogluon"]["fit"]["time_limit"], 17)
            self.assertEqual(loaded.effective["model_params"]["svm"]["estimator"]["C"], 1.0)
            self.assertEqual(report["effective_parameter_profile"], "legacy_main_json_entry_v1")

    def test_legacy_json_without_stable_id_and_unsupported_grouping_stop_without_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "old.json"
            source.write_text(json.dumps({"project_name": "old", "column_roles": {}, "descriptors": [], "models": []}), encoding="utf-8")
            output = root / "old.yaml"
            report = migrate(source, output)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(output.exists())
            self.assertIn("sample_id", report["error"])

            benchmark = root / "benchmark.yaml"
            benchmark.write_text(
                "benchmark:\n"
                "  dataset_path: data.csv\n"
                "  sample_id_col: sample_id\n"
                "  label_col: yield\n"
                "  descriptors: [morgan]\n"
                "  models: [rf]\n"
                "  grouping: {strategy: component_holdout}\n"
                "  cv: {n_repeats: 1, n_splits: 2, seed: 1}\n",
                encoding="utf-8",
            )
            rejected = migrate(benchmark, root / "benchmark-v2.yaml")
            self.assertEqual(rejected["status"], "failed")
            self.assertIn("grouping", rejected["error"])


if __name__ == "__main__":
    unittest.main()
