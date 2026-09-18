"""Verify the public ``yonod.py`` strict route without fitting a model."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StrictLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_dir = ROOT / "config" / ("_verify_strict_launcher_" + uuid.uuid4().hex)
        self.config_dir.mkdir(parents=True)
        self.config_path = self.config_dir / "strict_launcher.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "schema_version": "2.0",
            "project_name": "ablation2_strict_launcher_probe",
            "stage": "benchmark",
            "dataset": {
                "path": "./dataset/benchmark_smoke_fixture.csv",
                "sample_id_col": "reaction_id",
                "column_roles": {
                    "label": "yield", "reactants": ["reactant_1_smiles", "reactant_2_smiles"],
                    "products": [], "others": [], "conditions": [], "categoricals": [],
                },
            },
            "descriptors": [{
                "id": "morgan", "descriptor": "morgan", "lifecycle": "static_descriptor",
                "mode": "concat", "columns": ["reactant_1_smiles", "reactant_2_smiles"],
            }],
            "artifacts": {"output_dir": "./result/ablation2_strict_launcher_probe/feature"},
            "models": ["rf"],
            "model_params": {"rf": {"estimator": {"n_estimators": 1, "n_jobs": 19, "random_state": 42}}},
            "evaluation": {
                "protocol": "manifest_outer_cv", "n_splits": 2, "n_repeats": 1, "seed": 42,
                "grouping": {"strategy": "precomputed_column", "group_column": "reaction_id"},
            },
            "outputs": {"root": "./result/ablation2_strict_launcher_probe", "report_formats": ["html", "markdown"]},
            "benchmark": {"task_state": {"backend": "sqlite", "resumable": True}},
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.config_dir, ignore_errors=True)

    def test_config_paths_are_repository_relative_and_yonod_dispatches(self) -> None:
        from yonod.config.loader import load_run_config

        loaded = load_run_config(self.config_path)
        self.assertEqual(
            Path(loaded.effective["dataset"]["path"]),
            Path("./dataset/benchmark_smoke_fixture.csv"),
        )
        # The adapter resolves the reference during benchmark validation.
        from yonod.benchmark.config import BenchmarkConfig
        strict = BenchmarkConfig.from_file(self.config_path)
        self.assertEqual(strict.dataset_path, ROOT / "dataset" / "benchmark_smoke_fixture.csv")
        self.assertEqual(strict.outputs_root, ROOT / "result" / "ablation2_strict_launcher_probe")
        self.assertEqual(strict.artifact_output_dir, strict.outputs_root / "feature")

        spec = importlib.util.spec_from_file_location("yonod_public_launcher", ROOT / "yonod.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with patch("builtins.input", side_effect=[str(self.config_path)]), patch(
            "_verify.run_benchmark.run_benchmark", return_value=0
        ) as run_benchmark:
            self.assertEqual(module.main(), 0)
        run_benchmark.assert_called_once_with(self.config_path)


if __name__ == "__main__":
    unittest.main()
