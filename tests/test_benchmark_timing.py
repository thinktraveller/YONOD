"""Regression coverage for descriptor × model timing reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from yonod.benchmark.metrics import (
    MetricRebuildResult,
    summarize_combination_times,
    write_metric_tables,
)
from yonod.benchmark.report import generate_benchmark_report
from yonod.universal.report import generate_markdown_report, generate_report


def _fold_metrics() -> pd.DataFrame:
    rows = []
    # Two complete combinations with intentionally different costs, followed
    # by one incomplete and one invalid-time combination.
    for descriptor, model, train, predict in [
        ("morgan", "rf", [2.0, 4.0], [0.5, 0.5]),
        ("rdkit2d", "xgb", [10.0, 20.0], [1.0, 2.0]),
        ("morgan", "svm", [1.0], [0.2]),
        ("rdkit2d", "lightgbm", [np.nan, 3.0], [0.3, 0.4]),
    ]:
        for fold, (train_time, predict_time) in enumerate(zip(train, predict)):
            rows.append({
                "run_id": "timing-fixture", "config_hash": "config", "split_id": "split",
                "descriptor": descriptor, "model": model, "repeat": 0, "fold": fold,
                "n_valid": 3, "r2": 0.5, "rmse": 0.2, "mae": 0.1,
                "train_time_s": train_time, "predict_time_s": predict_time,
                "prediction_path": "", "metadata_path": "",
            })
    return pd.DataFrame.from_records(rows)


def _completeness() -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {
            "run_id": "timing-fixture", "config_hash": "config", "split_id": "split",
            "descriptor": "morgan", "model": "rf", "expected_folds": 2,
            "available_folds": 2, "valid_metric_folds": 2, "is_complete": True,
            "missing_or_excluded_reason": "",
        },
        {
            "run_id": "timing-fixture", "config_hash": "config", "split_id": "split",
            "descriptor": "rdkit2d", "model": "xgb", "expected_folds": 2,
            "available_folds": 2, "valid_metric_folds": 2, "is_complete": True,
            "missing_or_excluded_reason": "",
        },
        {
            "run_id": "timing-fixture", "config_hash": "config", "split_id": "split",
            "descriptor": "morgan", "model": "svm", "expected_folds": 2,
            "available_folds": 1, "valid_metric_folds": 1, "is_complete": False,
            "missing_or_excluded_reason": "missing_or_excluded_folds=[(0, 1)]",
        },
        {
            "run_id": "timing-fixture", "config_hash": "config", "split_id": "split",
            "descriptor": "rdkit2d", "model": "lightgbm", "expected_folds": 2,
            "available_folds": 2, "valid_metric_folds": 2, "is_complete": True,
            "missing_or_excluded_reason": "",
        },
    ])


class BenchmarkTimingTests(unittest.TestCase):
    def test_time_summary_preserves_completeness_and_rejects_invalid_values(self) -> None:
        summary = summarize_combination_times(_fold_metrics(), _completeness())
        rf = summary[(summary["descriptor"] == "morgan") & (summary["model"] == "rf")].iloc[0]
        xgb = summary[(summary["descriptor"] == "rdkit2d") & (summary["model"] == "xgb")].iloc[0]
        incomplete = summary[(summary["descriptor"] == "morgan") & (summary["model"] == "svm")].iloc[0]
        invalid = summary[(summary["descriptor"] == "rdkit2d") & (summary["model"] == "lightgbm")].iloc[0]

        self.assertEqual(rf["total_train_time_s"], 6.0)
        self.assertEqual(rf["mean_train_time_s"], 3.0)
        self.assertEqual(rf["median_train_time_s"], 3.0)
        self.assertEqual(rf["max_train_time_s"], 4.0)
        self.assertEqual(rf["total_predict_time_s"], 1.0)
        self.assertEqual(rf["total_model_time_s"], 7.0)
        self.assertTrue(rf["is_time_comparable"])
        self.assertTrue(xgb["is_time_comparable"])
        self.assertGreater(xgb["total_model_time_s"], rf["total_model_time_s"])
        self.assertFalse(incomplete["is_time_comparable"])
        self.assertIn("incomplete_folds", incomplete["time_status"])
        self.assertFalse(invalid["is_time_comparable"])
        self.assertIn("invalid_or_missing_fold_time", invalid["time_status"])
        self.assertTrue(np.isnan(invalid["total_model_time_s"]))

        negative_times = _fold_metrics()
        negative_times.loc[
            (negative_times["descriptor"] == "morgan") & (negative_times["model"] == "rf") & (negative_times["fold"] == 0),
            "train_time_s",
        ] = -1.0
        negative = summarize_combination_times(negative_times, _completeness())
        negative_rf = negative[(negative["descriptor"] == "morgan") & (negative["model"] == "rf")].iloc[0]
        self.assertFalse(negative_rf["is_time_comparable"])
        self.assertIn("invalid_or_missing_fold_time", negative_rf["time_status"])

    def test_strict_report_rebuilds_timing_table_without_training(self) -> None:
        fold_metrics = _fold_metrics()
        completeness = _completeness()
        rebuilt = MetricRebuildResult(
            fold_metrics=fold_metrics,
            exclusions=pd.DataFrame(),
            completeness=completeness,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir(parents=True)
            split = pd.DataFrame.from_records([
                {"run_id": "timing-fixture", "dataset_sha256": "dataset", "split_id": "split", "sample_id": "a", "role": "train", "repeat": 0, "fold": 0, "group_id": "g1"},
                {"run_id": "timing-fixture", "dataset_sha256": "dataset", "split_id": "split", "sample_id": "b", "role": "valid", "repeat": 0, "fold": 0, "group_id": "g2"},
                {"run_id": "timing-fixture", "dataset_sha256": "dataset", "split_id": "split", "sample_id": "c", "role": "train", "repeat": 0, "fold": 1, "group_id": "g2"},
                {"run_id": "timing-fixture", "dataset_sha256": "dataset", "split_id": "split", "sample_id": "d", "role": "valid", "repeat": 0, "fold": 1, "group_id": "g1"},
            ])
            split.to_parquet(root / "manifests" / "split_manifest.parquet", index=False)
            (root / "manifests" / "run_manifest.json").write_text(json.dumps({
                "run_id": "timing-fixture", "config_hash": "config", "dataset_sha256": "dataset",
                "benchmark_config": {"descriptors": ["morgan", "rdkit2d"], "models": ["rf", "xgb"]},
            }), encoding="utf-8")
            empty = pd.DataFrame()
            write_metric_tables(root, rebuilt, empty, empty, empty, empty, empty, empty)
            (root / "metrics" / "combination_time_summary.parquet").unlink()

            report = generate_benchmark_report(root)

            self.assertTrue((root / "metrics" / "combination_time_summary.parquet").is_file())
            self.assertTrue((root / "figures" / "combination_modeling_time.png").is_file())
            self.assertIn("建模耗时与成本对比", report.html_path.read_text(encoding="utf-8"))
            markdown = report.markdown_path.read_text(encoding="utf-8")
            self.assertIn("组合级建模耗时柱状图", markdown)
            self.assertIn("../figures/combination_modeling_time.png", markdown)

    def test_universal_reports_handle_present_and_absent_train_time(self) -> None:
        base = pd.DataFrame.from_records([
            {"descriptor": "morgan", "model": "rf", "r2": 0.6, "rmse": 0.2, "mae": 0.1},
            {"descriptor": "rdkit2d", "model": "xgb", "r2": 0.7, "rmse": 0.15, "mae": 0.08},
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_info = {"task_name": "timing-fixture", "n_combinations": 2}
            html_without = generate_report(base, task_info, root, "without.html")
            markdown_without = generate_markdown_report(base, task_info, root, "without.md")
            self.assertIn("未提供组合级", html_without.read_text(encoding="utf-8"))
            self.assertIn("未提供组合级", markdown_without.read_text(encoding="utf-8"))
            self.assertFalse((root / "combination_training_time.png").exists())

            timed = base.assign(train_time_s=[2.0, 12.0])
            html_with = generate_report(timed, task_info, root, "with.html")
            markdown_with = generate_markdown_report(timed, task_info, root, "with.md")
            self.assertTrue((root / "combination_training_time.png").is_file())
            self.assertIn("建模耗时与成本对比", html_with.read_text(encoding="utf-8"))
            self.assertIn("组合训练时间柱状图", markdown_with.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
