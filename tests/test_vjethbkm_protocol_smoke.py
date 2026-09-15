"""End-to-end smoke gate for the MFP/OHE/RF paper-aligned protocol."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VjethbkmProtocolSmokeTests(unittest.TestCase):
    def test_two_feature_lifecycles_complete_25_folds_and_rebuild_report(self):
        frame = pd.DataFrame({
            "sample_id": ["s{0:02d}".format(index) for index in range(10)],
            "yield": np.linspace(0.1, 0.9, 10),
            "a": ["C", "CC", "CCC", "CCCC", "CCO"] * 2,
            "b": ["O", "N", "Cl", "Br", "F"] * 2,
        })
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame.to_csv(directory / "fixture.csv", index=False)
            config_path = directory / "protocol.yaml"
            config_path.write_text(yaml.safe_dump({
                "schema_version": "2.0", "project_name": "vjethbkm-protocol-test", "stage": "benchmark",
                "dataset": {
                    "path": "fixture.csv", "sample_id_col": "sample_id",
                    "column_roles": {"label": "yield", "reactants": ["a", "b"]},
                },
                "descriptors": [
                    {"id": "mfp", "descriptor": "mfp", "lifecycle": "static_descriptor", "mode": "concat", "columns": ["a", "b"], "params": {
                        "algorithm": "morgan_count", "radius": 3, "fp_size": 1024,
                        "input_normalization": "raw_csv_value", "blank_or_invalid_component": "zero_block_keep_row",
                    }},
                    {"id": "ohe", "descriptor": "ohe", "lifecycle": "fold_transform", "mode": "concat", "columns": ["a", "b"], "params": {
                        "encoder": "OneHotEncoder", "handle_unknown": "ignore",
                    }},
                ],
                "models": ["rf"],
                # P4 separately asserts the paper's n_jobs=-1 parameter.
                # Keep this lifecycle smoke single-threaded so it can run
                # reliably inside CI without proliferating joblib workers.
                "artifacts": {"output_dir": "results"},
                "model_params": {"rf": {"estimator": {"n_estimators": 500, "max_features": 0.3, "n_jobs": 1}}},
                "evaluation": {
                    "protocol": "manifest_outer_cv", "n_repeats": 5, "n_splits": 5, "seed": 1000,
                    "grouping": {"strategy": "repeated_kfold", "source_order": "raw_csv"},
                },
                "outputs": {"root": "results"},
                "benchmark": {
                    "task_state": {"backend": "sqlite", "resumable": True},
                    "reproduction_protocol": {"name": "vjethbkm_rf_5x5", "literature_doi": "10.1021/jacs.6c02213"},
                },
            }, allow_unicode=True), encoding="utf-8")
            contract = create_benchmark_contract(BenchmarkConfig.from_file(config_path))
            command = [sys.executable, "-u", "yonod.py"]
            subprocess.run(
                command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
                input=str(config_path) + "\n", timeout=120,
            )

            run_dir = contract.run_dir
            manifest = pd.read_parquet(run_dir / "docs" / "manifests" / "split_manifest.parquet")
            fold_dir = run_dir / "docs" / "folds"
            mfp_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(fold_dir.glob("mfp__rf__*.json"))]
            ohe_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(fold_dir.glob("ohe__rf__*.json"))]

            self.assertEqual(len(manifest[["repeat", "fold"]].drop_duplicates()), 25)
            self.assertEqual(len(mfp_metadata), 25)
            self.assertEqual(len(ohe_metadata), 25)
            self.assertTrue((run_dir / "descriptors" / "mfp.npz").is_file())
            self.assertFalse((run_dir / "descriptors" / "ohe.npz").exists())
            self.assertEqual(sorted({item["manifest_seed"] for item in mfp_metadata}), list(range(1000, 1005)))
            self.assertTrue(all(item["feature_transformer"] is None for item in mfp_metadata))
            self.assertTrue(all(item["feature_transformer"]["fit_scope"] == "train_only_per_fold" for item in ohe_metadata))
            self.assertTrue(all(item["estimator_params_snapshot"]["max_features"] == 0.3 for item in mfp_metadata + ohe_metadata))
            for item in mfp_metadata + ohe_metadata:
                snapshot = item["estimator_params_snapshot"]
                self.assertEqual(snapshot["construction"], "schema2_model_factory")
                self.assertEqual(snapshot["effective_estimator_params"]["max_features"], 0.3)
                self.assertEqual(snapshot["random_state"], item["manifest_seed"])
                self.assertEqual(item["model_random_seed"], item["manifest_seed"])

            subprocess.run(
                [sys.executable, "scripts/rebuild_benchmark_report.py", "--run-dir", str(run_dir)],
                cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=120,
            )
            markdown = (run_dir / "report" / "benchmark_report.md").read_text(encoding="utf-8")

        self.assertIn("论文协议对齐状态", markdown)
        self.assertIn("流程对齐不等同于论文数值复现", markdown)


if __name__ == "__main__":
    unittest.main()
