"""Ordinary-entry integration tests for public MFP and OHE features."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import main
from yonod.descriptors.ohe import OHEFeature
from yonod.universal.descriptor_artifact import load_descriptor_artifact


class _StaticModel:
    def cross_validate(self, X, y, cv):
        return {
            "r2_mean": 0.0, "r2_std": 0.0, "rmse_mean": 1.0, "mae_mean": 1.0,
            "train_time_s": 0.0, "device": "cpu",
            "oof_pred": np.asarray(y, dtype=float), "oof_y_true": np.asarray(y, dtype=float),
        }


class _FoldModel:
    def _build(self):
        return DummyRegressor(strategy="mean")


class MainFeatureLibraryTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "smiles": ["c1ccccc1", "CC", "CCC", "O", "N", "CO", "CN", "CCO", "CCN", "Cl"],
            "condition": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "unique-valid"],
            "yield": np.linspace(1.0, 10.0, 10),
        })

    def test_cli_mfp_creates_parameterized_count_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "input.csv"
            out_dir = root / "result"
            self._frame().to_csv(csv_path, index=False)
            with patch("main._make_model", return_value=_StaticModel()):
                result = main.main([
                    "--csv", str(csv_path), "--label-col", "yield", "--smiles-cols", "smiles",
                    "--descriptors", "mfp", "--mfp-radius", "2", "--mfp-fp-size", "64",
                    "--models", "rf", "--cv", "2", "--heartbeat", "0", "--output-dir", str(out_dir),
                    "--output-format", "md",
                ])
            self.assertEqual(result, 0)
            artifact = load_descriptor_artifact(out_dir / "descriptors" / "mfp.npz")
            self.assertEqual(artifact.X_smiles.shape, (10, 64))
            self.assertEqual(artifact.metadata["descriptor_config"]["feature_id"], "mfp")
            self.assertEqual(artifact.metadata["mfp_protocol"]["profile"], "standard")
            self.assertGreater(int(artifact.X_smiles.max()), 1)

    def test_json_ohe_uses_fold_state_not_global_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_name = "ordinary-ohe"
            dataset_path = root / f"{project_name}_normalized_dataset.csv"
            self._frame().to_csv(dataset_path, index=False)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "version": "1.1", "project_name": project_name,
                "column_roles": {
                    "label": "yield", "reactants": ["smiles"], "products": [], "others": [],
                    "conditions": [], "categoricals": ["condition"],
                },
                "descriptors": [{
                    "id": "condition_ohe", "descriptor": "ohe", "mode": "concat",
                    "columns": ["condition"],
                    "params": {"missing_policy": "as_category", "dtype": "float32", "handle_unknown": "ignore"},
                }],
                "models": ["Random Forest"], "report_formats": ["Markdown"], "metadata": {},
            }, ensure_ascii=False), encoding="utf-8")

            with patch("main._make_model", return_value=_FoldModel()):
                result = main.main(["--json", str(config_path)])
            self.assertEqual(result, 0)
            state_root = root / "fold_transformers" / "condition_ohe"
            self.assertTrue((state_root / "summary.json").is_file())
            self.assertFalse((root / "descriptors" / "condition_ohe.npz").exists())
            summary = json.loads((state_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_folds"], 5)
            for fold in summary["folds"]:
                state = OHEFeature.load(
                    fold["state_directory"],
                    expected_context={"feature_id": "condition_ohe", "fold": fold["fold"]},
                )
                self.assertEqual(state.metadata()["fit_scope"], "train_only_per_fold")
            status = pd.read_csv(root / "docs" / "descriptor_status.csv")
            row = status.loc[status["feature_id"] == "condition_ohe"].iloc[0]
            self.assertEqual(row["lifecycle"], "fold_transform")
            self.assertEqual(int(row["completed_folds"]), 5)


if __name__ == "__main__":
    unittest.main()
