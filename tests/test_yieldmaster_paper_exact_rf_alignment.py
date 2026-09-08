"""Guards for RF-vs-native alignment decisions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from yonod.benchmark.paper_exact import PaperExactError
from yonod.benchmark.paper_exact_pipeline import compare_rf_to_native_reference


class YieldMasterPaperExactRFAlignmentTests(unittest.TestCase):
    def _reference_root(self, directory: Path) -> Path:
        native = directory / "outputs" / "runs" / "native_mfp_rf_matrix_20260907"
        native.mkdir(parents=True)
        fold_rows = []
        for repeat in range(1, 6):
            for fold in range(1, 6):
                fold_rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "repeat_seed": 999 + repeat,
                    "train_rows": 8,
                    "test_rows": 2,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "r2": 0.9,
                    "kendall": 0.8,
                })
        pd.DataFrame(fold_rows).to_csv(native / "BH1_MFP_RF_fold_metrics.csv", index=False)
        pd.DataFrame({
            "dataset_id": ["BH1", "BH1", "BH1", "BH1"],
            "descriptor": ["MFP", "MFP", "MFP", "MFP"],
            "model": ["RF", "RF", "RF", "RF"],
            "metric": ["mae", "rmse", "r2", "kendall_tau"],
            "independent_native_mean": [0.1, 0.2, 0.9, 0.8],
            "independent_native_std": [0.0, 0.0, 0.0, 0.0],
        }).to_csv(native / "MFP_RF_matrix_comparison.csv", index=False)
        return directory

    def _fold_metrics(self, mae: float = 0.1) -> pd.DataFrame:
        rows = []
        for repeat in range(1, 6):
            for fold in range(1, 6):
                rows.append({
                    "population_id": "bh1_paper_exact",
                    "dataset_id": "BH1",
                    "feature_id": "mfp",
                    "model": "rf",
                    "repeat": repeat,
                    "fold": fold,
                    "seed": 999 + repeat,
                    "train_rows": 8,
                    "test_rows": 2,
                    "mae": mae,
                    "rmse": 0.2,
                    "r2": 0.9,
                    "kendall_tau": 0.8,
                })
        return pd.DataFrame(rows)

    def _summary(self, mae: float = 0.1) -> pd.DataFrame:
        return pd.DataFrame({
            "population_id": ["bh1_paper_exact", "bh1_paper_exact", "bh1_paper_exact", "bh1_paper_exact"],
            "dataset_id": ["BH1", "BH1", "BH1", "BH1"],
            "feature_id": ["mfp", "mfp", "mfp", "mfp"],
            "model": ["rf", "rf", "rf", "rf"],
            "metric": ["mae", "rmse", "r2", "kendall_tau"],
            "mean": [mae, 0.2, 0.9, 0.8],
            "std": [0.0, 0.0, 0.0, 0.0],
        })

    def test_alignment_passes_when_metrics_match_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = self._reference_root(Path(temporary))
            result = compare_rf_to_native_reference(
                self._fold_metrics(),
                self._summary(),
                reference,
                tolerance=1e-12,
            )
        self.assertEqual(len(result), 4)
        self.assertTrue(result["passes_tolerance"].all())

    def test_alignment_fails_closed_on_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = self._reference_root(Path(temporary))
            result = compare_rf_to_native_reference(
                self._fold_metrics(mae=0.11),
                self._summary(mae=0.11),
                reference,
                tolerance=1e-12,
            )
        mae = result[result["metric"] == "mae"].iloc[0]
        self.assertFalse(bool(mae["passes_tolerance"]))
        self.assertGreater(mae["max_abs_fold_metric_diff"], 0)

    def test_alignment_refuses_incomplete_fold_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = self._reference_root(Path(temporary))
            with self.assertRaises(PaperExactError):
                compare_rf_to_native_reference(
                    self._fold_metrics().iloc[0:0],
                    self._summary(),
                    reference,
                )


if __name__ == "__main__":
    unittest.main()
