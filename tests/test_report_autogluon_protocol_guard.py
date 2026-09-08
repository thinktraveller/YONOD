import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import _CSV_COLUMNS
from yonod.benchmark.metrics import paired_comparisons, summarize_combinations, tukey_hsd_comparisons
from yonod.universal.report import generate_markdown_report, generate_report, rank_combinations, ranking_eligibility_table


def _universal_metrics_frame():
    return pd.DataFrame([
        {
            "descriptor": "mfp", "model": "rf", "cv": 5,
            "evaluation_protocol": "outer_kfold", "expected_folds": 5, "completed_folds": 5,
            "oof_complete": True, "oof_n_observed": 100, "oof_n_total": 100,
            "r2_mean": 0.60, "r2_std": 0.03, "rmse_mean": 0.20, "rmse_std": 0.02,
            "mae_mean": 0.15, "mae_std": 0.01, "train_time_s": 10.0,
        },
        {
            "descriptor": "mfp", "model": "autogluon", "cv": 5,
            "evaluation_protocol": "autogluon_internal_holdout", "expected_folds": 1, "completed_folds": 1,
            "oof_complete": False, "oof_n_observed": 20, "oof_n_total": 100,
            "r2_mean": 0.99, "r2_std": 0.00, "rmse_mean": 0.01, "rmse_std": 0.00,
            "mae_mean": 0.01, "mae_std": 0.00, "train_time_s": 3.0,
            "autogluon_time_limit": 300, "autogluon_presets": "medium_quality",
            "autogluon_num_cpus": 19, "autogluon_seed_policy": "outer_seed_plus_fold",
        },
        {
            "descriptor": "ohe", "model": "autogluon", "cv": 5,
            "evaluation_protocol": "outer_kfold", "expected_folds": 5, "completed_folds": 4,
            "oof_complete": False, "oof_n_observed": 80, "oof_n_total": 100,
            "r2_mean": 0.95, "r2_std": 0.02, "rmse_mean": 0.03, "rmse_std": 0.01,
            "mae_mean": 0.02, "mae_std": 0.01, "train_time_s": 12.0,
            "autogluon_time_limit": 300, "autogluon_presets": "medium_quality",
            "autogluon_num_cpus": 19, "autogluon_seed_policy": "outer_seed_plus_fold",
        },
        {
            "descriptor": "ohe", "model": "rf", "cv": 5,
            "evaluation_protocol": "outer_kfold", "expected_folds": 5, "completed_folds": 5,
            "oof_complete": True, "oof_n_observed": 100, "oof_n_total": 100,
            "r2_mean": 0.58, "r2_std": 0.02, "rmse_mean": 0.21, "rmse_std": 0.02,
            "mae_mean": 0.16, "mae_std": 0.01, "train_time_s": 11.0,
        },
    ])


def _fold_metrics_frame():
    rows = []
    model_specs = [
        ("rf", "manifest_outer_cv", 0.72, 0.19, 0.14),
        ("xgb", "manifest_outer_cv", 0.75, 0.17, 0.13),
        ("autogluon", "autogluon_internal_holdout", 0.99, 0.01, 0.01),
        ("svm", "autogluon_internal_holdout", 0.98, 0.02, 0.02),
    ]
    for model, protocol, r2, rmse, mae in model_specs:
        for fold in range(3):
            rows.append({
                "run_id": "run", "config_hash": "hash", "split_id": "split",
                "evaluation_protocol": protocol, "descriptor": "mfp", "model": model,
                "repeat": 0, "fold": fold, "n_valid": 10,
                "r2": r2 + fold * 0.01, "rmse": rmse + fold * 0.001, "mae": mae + fold * 0.001,
                "train_time_s": 1.0, "predict_time_s": 0.1,
                "prediction_path": "", "metadata_path": "",
            })
    return pd.DataFrame(rows)


def _completeness_frame():
    return pd.DataFrame([
        {
            "run_id": "run", "config_hash": "hash", "split_id": "split",
            "evaluation_protocol": protocol, "descriptor": "mfp", "model": model,
            "expected_folds": 3, "available_folds": 3, "valid_metric_folds": 3,
            "is_complete": True, "missing_or_excluded_reason": "",
        }
        for model, protocol in [
            ("rf", "manifest_outer_cv"),
            ("xgb", "manifest_outer_cv"),
            ("autogluon", "autogluon_internal_holdout"),
            ("svm", "autogluon_internal_holdout"),
        ]
    ])


class UniversalReportProtocolGuardTest(unittest.TestCase):
    def test_csv_schema_includes_oof_guard_fields(self):
        self.assertIn("oof_complete", _CSV_COLUMNS)
        self.assertIn("oof_n_observed", _CSV_COLUMNS)
        self.assertIn("oof_n_total", _CSV_COLUMNS)

    def test_rank_excludes_internal_holdout_and_incomplete_oof(self):
        metrics = _universal_metrics_frame()
        guard = ranking_eligibility_table(metrics)
        internal = guard.loc[guard["evaluation_protocol"] == "autogluon_internal_holdout"].iloc[0]
        self.assertFalse(bool(internal["ranking_eligible"]))
        self.assertIn("协议不是 outer_kfold", internal["ranking_exclusion_reason"])
        incomplete = guard.loc[(guard["desc_name"] == "ohe") & (guard["model_name"] == "autogluon")].iloc[0]
        self.assertFalse(bool(incomplete["ranking_eligible"]))
        self.assertIn("OOF 不完整", incomplete["ranking_exclusion_reason"])

        ranked = rank_combinations(metrics)
        self.assertEqual(set(ranked["evaluation_protocol"]), {"outer_kfold"})
        self.assertNotIn("autogluon", set(ranked["model_name"]))
        self.assertEqual(set(ranked["model_name"]), {"rf"})

    def test_markdown_and_html_show_protocol_guard_details(self):
        metrics = _universal_metrics_frame()
        task_info = {
            "task_name": "protocol-guard", "csv_path": "input.csv", "project_folder": "out",
            "n_samples": 100, "smiles_cols": ["rxn"], "numeric_cols": "（无）",
            "label_col": "yield", "n_combinations": len(metrics),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html_path = generate_report(metrics, task_info, root)
            md_path = generate_markdown_report(metrics, task_info, root)
            html = html_path.read_text(encoding="utf-8")
            markdown = md_path.read_text(encoding="utf-8")
        for rendered in (html, markdown):
            self.assertIn("evaluation_protocol", rendered)
            self.assertIn("autogluon_internal_holdout", rendered)
            self.assertIn("RMSE", rendered)
            self.assertIn("MAE", rendered)
            self.assertIn("autogluon_time_limit", rendered)
            self.assertIn("seed_policy", rendered)
            self.assertIn("只展示", rendered)
            self.assertIn("不参与排名", rendered)


class BenchmarkProtocolGuardTest(unittest.TestCase):
    def test_benchmark_comparisons_are_protocol_scoped_and_exclude_holdout(self):
        fold_metrics = _fold_metrics_frame()
        completeness = _completeness_frame()

        summary = summarize_combinations(fold_metrics, n_bootstrap=100, seed=7)
        self.assertIn("evaluation_protocol", summary.columns)
        self.assertIn("autogluon_internal_holdout", set(summary["evaluation_protocol"]))

        comparisons, exclusions = paired_comparisons(
            fold_metrics, completeness, dimension="model", n_bootstrap=100, seed=7
        )
        self.assertFalse(comparisons.empty)
        self.assertEqual(set(comparisons["evaluation_protocol"]), {"manifest_outer_cv"})
        compared_models = set(comparisons["left"]).union(set(comparisons["right"]))
        self.assertEqual(compared_models, {"rf", "xgb"})
        self.assertIsInstance(exclusions, pd.DataFrame)

    def test_tukey_excludes_internal_holdout_protocols(self):
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            self.skipTest("statsmodels not installed")
        result = tukey_hsd_comparisons(_fold_metrics_frame(), dimension="model")
        self.assertFalse(result.empty)
        self.assertEqual(set(result["evaluation_protocol"]), {"manifest_outer_cv"})
        compared_models = set(result["left"]).union(set(result["right"]))
        self.assertEqual(compared_models, {"rf", "xgb"})


if __name__ == "__main__":
    unittest.main()
