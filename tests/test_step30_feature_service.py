"""Focused acceptance tests for the independent step-30.4 features service."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.reader import load_feature_artifact
from yonod.pipeline.features import run_features


class Step30FeatureServiceTests(unittest.TestCase):
    def _run_with_clean_imports(self) -> bool:
        """Test dependency isolation before any unrelated test imports models."""
        if getattr(self, "_isolated_process", False):
            return False
        probe = (
            "import sys, unittest; sys.path.insert(0, 'tests'); "
            "import test_step30_feature_service as module; "
            "module.Step30FeatureServiceTests._isolated_process = True; "
            "suite = unittest.TestSuite([module.Step30FeatureServiceTests(sys.argv[1])]); "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(not result.wasSuccessful())"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, self._testMethodName],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def _write_config(self, root: Path, descriptors: list[dict]) -> Path:
        data = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "reactant": ["CC", "CCC", "O"],
            "condition": ["A", None, "B"],
        })
        data.to_csv(root / "input.csv", index=False)
        config = {
            "schema_version": "2.0",
            "project_name": "feature-only",
            "stage": "features",
            "dataset": {
                "path": "input.csv",
                "sample_id_col": "sample_id",
                "column_roles": {
                    "reactants": ["reactant"],
                    "categoricals": ["condition"],
                },
            },
            "descriptors": descriptors,
            "artifacts": {"output_dir": "artifacts"},
        }
        path = root / "features.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_feature_only_without_label_or_models_keeps_success_when_another_descriptor_fails(self) -> None:
        if self._run_with_clean_imports():
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root, [
                {"id": "good", "descriptor": "morgan", "columns": ["reactant"]},
                {"id": "broken", "descriptor": "morgan", "columns": ["reactant"]},
            ])
            calls: list[str] = []

            def computer(spec, frame, columns, roles):
                calls.append(spec.id)
                if spec.id == "broken":
                    raise RuntimeError("intentional test descriptor failure")
                return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.asarray([True, False, True])

            result = run_features(config, feature_computer=computer)

            self.assertEqual(result.status, "partial")
            self.assertEqual(calls, ["good", "broken"])
            statuses = {item.feature_id: item for item in result.features}
            self.assertEqual(statuses["good"].status, "ready")
            self.assertEqual(statuses["broken"].status, "failed")
            self.assertIsNotNone(statuses["good"].manifest_path)
            artifact = load_feature_artifact(statuses["good"].manifest_path)
            np.testing.assert_array_equal(artifact.valid_mask, np.asarray([True, False, True]))
            self.assertTrue(result.status_manifest_path.is_file())
            self.assertFalse((root / "artifacts" / "docs").exists())
            self.assertFalse((root / "artifacts" / "report").exists())
            self.assertFalse(any(name == "yonod.models" or name.startswith("yonod.models.") for name in sys.modules))
            self.assertFalse(any(isinstance(value, pd.DataFrame) for value in result.__dict__.values()))

    def test_same_identity_reuses_verified_package_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root, [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}])
            calls: list[str] = []

            def computer(spec, frame, columns, roles):
                calls.append(spec.id)
                return np.asarray([[1.0], [2.0]], dtype=np.float32), np.asarray([True, False, True])

            first = run_features(config, feature_computer=computer)
            second = run_features(config, feature_computer=computer)

            self.assertEqual(first.status, "ready")
            self.assertEqual(second.status, "ready")
            self.assertEqual(calls, ["morgan"])
            self.assertEqual(second.features[0].status, "reused")
            self.assertEqual(first.features[0].manifest_path, second.features[0].manifest_path)

    def test_fold_transform_persists_raw_categories_without_ohe_fit_or_static_computer(self) -> None:
        if self._run_with_clean_imports():
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root, [{
                "id": "condition-ohe",
                "descriptor": "ohe",
                "columns": ["condition"],
                "params": {"missing_policy": "as_category", "dtype": "float32", "handle_unknown": "ignore"},
            }])

            def should_not_run(*args, **kwargs):
                raise AssertionError("OHE raw-input preparation must not invoke the static feature computer")

            result = run_features(config, feature_computer=should_not_run)
            artifact = load_feature_artifact(result.features[0].manifest_path)

            self.assertEqual(result.status, "ready")
            self.assertEqual(artifact.manifest["lifecycle"], "fold_transform")
            self.assertEqual(artifact.manifest["metadata"]["fold_transform"]["fit_scope"], "training_fold_only")
            self.assertEqual(artifact.matrix.shape, (3, 1))
            self.assertEqual(artifact.valid_mask.tolist(), [True, True, True])
            self.assertNotIn("yonod.descriptors.ohe", sys.modules)

    def test_default_service_path_delegates_to_existing_production_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._write_config(root, [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}])
            with patch(
                "yonod.universal.feature_builder.build_universal_features",
                return_value=(np.asarray([[1.0], [2.0]], dtype=np.float32), None, np.asarray([True, False, True])),
            ) as production_builder:
                result = run_features(config)

            self.assertEqual(result.status, "ready")
            production_builder.assert_called_once()
            self.assertEqual(production_builder.call_args.kwargs["desc_name"], "morgan")
            self.assertEqual(production_builder.call_args.kwargs["numeric_cols"], [])


if __name__ == "__main__":
    unittest.main()
