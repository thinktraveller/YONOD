
"""Tests for the YieldMaster paper_exact AutoGluon runner."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from yonod.benchmark.paper_exact import PAPER_EXACT_EVALUATION_PROTOCOL, PaperExactError
from yonod.benchmark.paper_exact_autogluon import (
    PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR,
    build_rf_autogluon_paired_input,
    load_or_prepare_paper_exact_material,
    migrate_trusted_mfp_npz_sample_ids_to_unicode,
    run_autogluon_for_material,
    run_paper_exact_autogluon_matrix,
    validate_complete_paper_exact_model_folds,
    validate_mfp_npz_safe_string_dtype,
    validate_paper_exact_autogluon_prerequisites,
    write_paper_exact_autogluon_launch_materials,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference-proejct" / "vjethbkm"
_FAKE_CALLS: list[dict[str, object]] = []


def _rewrite_mfp_string_members_as_object(material) -> str:
    path = material.feature_paths["mfp"]
    with np.load(path, allow_pickle=False) as feature:
        arrays = {name: feature[name] for name in feature.files}
    feature_hash = str(arrays["feature_hash"])
    arrays["sample_id"] = np.asarray([str(value) for value in arrays["sample_id"].tolist()], dtype=object)
    arrays["smiles_columns"] = np.asarray([str(value) for value in arrays["smiles_columns"].tolist()], dtype=object)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return feature_hash


def _fake_fit_predict_fold(self, X_train, y_train, X_valid, *, fold_index, context=None):
    context = dict(context or {})
    train = np.asarray(X_train, dtype=float)
    valid = np.asarray(X_valid, dtype=float)
    target = np.asarray(y_train, dtype=float)
    _FAKE_CALLS.append({
        "fold_index": int(fold_index),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "feature_dim": int(train.shape[1]),
        "context": context,
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
        "random_state": self.random_state,
    })
    scale = float(target.mean()) if len(target) else 0.0
    prediction = valid[:, 0].astype(float) * 0.001 + scale + np.arange(len(valid), dtype=float) * 0.0001
    return prediction, {
        "evaluation_protocol": context.get("evaluation_protocol"),
        "protocol_family": context.get("protocol_family"),
        "fold_index": int(fold_index),
        "outer_seed": int(context.get("outer_seed", -1)),
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
        "random_state": self.random_state,
        "random_state_policy": "fake AG; outer split fixed",
        "autogluon_versions": {"autogluon.tabular": "fake-1.1.1"},
        "train_time_s": 0.01,
        "predict_time_s": 0.002,
        "model_artifact_path": context.get("model_artifact_path"),
        "model_artifact_cleanup": bool(context.get("cleanup")),
        "leaderboard_rows": 1,
    }


@unittest.skipUnless(REFERENCE_ROOT.is_dir(), "vjethbkm reference project is required")
class PaperExactAutoGluonRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        _FAKE_CALLS.clear()

    def test_prepare_launch_materials_writes_eight_tasks_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            prepared = write_paper_exact_autogluon_launch_materials(
                REFERENCE_ROOT,
                output_root,
                time_limit=300,
                presets="medium_quality",
                num_cpus=19,
            )
            tasks = pd.read_csv(prepared["tasks_path"])
            manifest = prepared["manifest_path"].read_text(encoding="utf-8")

        self.assertEqual(prepared["descriptor_task_count"], 8)
        self.assertEqual(prepared["expected_autogluon_fold_fits"], 200)
        self.assertEqual(set(tasks["model"]), {"autogluon"})
        self.assertEqual(set(tasks["expected_folds"]), {PAPER_EXACT_EXPECTED_FOLDS_PER_DESCRIPTOR})
        self.assertEqual(set(tasks["evaluation_protocol"]), {PAPER_EXACT_EVALUATION_PROTOCOL})
        self.assertIn("rf_gate_required", manifest)
        self.assertIn("run_autogluon_paper_exact_nohup", manifest)

    def test_mock_autogluon_matrix_uses_same_manifest_and_ohe_train_fold_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                result = run_paper_exact_autogluon_matrix(
                    REFERENCE_ROOT,
                    output_root,
                    population_ids=["sl1_paper_exact"],
                    feature_ids=["ohe"],
                    time_limit=3,
                    presets="medium_quality",
                    num_cpus=2,
                    require_rf_alignment=False,
                )
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            metadata_path = material.population_dir / "autogluon" / "folds" / "ohe__autogluon__r01__f01.json"
            metadata = metadata_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.expected_fold_count, 25)
        self.assertEqual(result.completed_fold_count, 25)
        self.assertEqual(len(result.fold_metrics), 25)
        self.assertEqual(len(_FAKE_CALLS), 25)
        self.assertEqual(set(result.fold_metrics["evaluation_protocol"]), {PAPER_EXACT_EVALUATION_PROTOCOL})
        self.assertEqual(set(result.fold_metrics["model"]), {"autogluon"})
        self.assertTrue(result.fold_metrics["fold_feature_hash"].astype(str).str.len().gt(20).all())
        self.assertEqual(set(result.summary["completed_folds"]), {25})
        self.assertIn('"fit_scope": "train_only_per_fold"', metadata)
        self.assertIn('"manifest_seed": 1000', metadata)
        self.assertEqual(_FAKE_CALLS[0]["context"]["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)
        self.assertEqual(_FAKE_CALLS[0]["context"]["protocol_family"], "manifest_outer_cv")
        self.assertEqual(_FAKE_CALLS[0]["context"]["outer_seed"], 1000)
        self.assertNotIn("tuning_data", _FAKE_CALLS[0]["context"])

    def test_mfp_npz_uses_fixed_unicode_and_object_array_is_fail_fast_migrated_without_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            validation = validate_mfp_npz_safe_string_dtype(material)
            self.assertEqual(validation["present"], True)
            self.assertTrue(str(validation["sample_id_dtype"]).startswith("<U"))
            before_hash = _rewrite_mfp_string_members_as_object(material)

            with self.assertRaisesRegex(PaperExactError, "object dtype"):
                validate_mfp_npz_safe_string_dtype(material)
            with self.assertRaisesRegex(PaperExactError, "object dtype"):
                validate_paper_exact_autogluon_prerequisites([material])
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                with self.assertRaisesRegex(PaperExactError, "object dtype"):
                    run_paper_exact_autogluon_matrix(
                        REFERENCE_ROOT,
                        output_root,
                        population_ids=["sl1_paper_exact"],
                        feature_ids=["mfp"],
                        time_limit=3,
                        num_cpus=2,
                        require_rf_alignment=False,
                        allow_partial=True,
                        max_folds_per_task=1,
                    )

            migrated = migrate_trusted_mfp_npz_sample_ids_to_unicode(material, overwrite=True)
            self.assertEqual(migrated["changed"], True)
            self.assertEqual(migrated["feature_hash_before"], before_hash)
            self.assertEqual(migrated["feature_hash_after"], before_hash)
            self.assertTrue(str(migrated["sample_id_dtype"]).startswith("<U"))
            validate_paper_exact_autogluon_prerequisites([material])

    def test_mock_mfp_single_fold_reads_prepared_npz_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                folds, predictions = run_autogluon_for_material(
                    material,
                    feature_ids=["mfp"],
                    time_limit=3,
                    num_cpus=2,
                    max_folds_per_task=1,
                )

            self.assertEqual(len(folds), 1)
            self.assertEqual(set(folds["feature_id"]), {"mfp"})
            self.assertEqual(set(predictions["feature_id"]), {"mfp"})
            self.assertEqual(_FAKE_CALLS[0]["context"]["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)
            self.assertGreater(_FAKE_CALLS[0]["feature_dim"], 1000)

    def test_quarantine_incomplete_artifact_moves_aside_before_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            artifact = material.population_dir / "autogluon" / "models" / "mfp__autogluon__r01__f01"
            artifact.mkdir(parents=True)
            marker = artifact / "partial.txt"
            marker.write_text("partial artifact from interrupted launch", encoding="utf-8")
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                folds, _predictions = run_autogluon_for_material(
                    material,
                    feature_ids=["mfp"],
                    time_limit=3,
                    num_cpus=2,
                    max_folds_per_task=1,
                    quarantine_incomplete_artifacts=True,
                )
            quarantines = list((material.population_dir / "autogluon" / "incomplete_artifacts").glob("mfp__autogluon__r01__f01__quarantine_*"))

            self.assertEqual(len(folds), 1)
            self.assertEqual(len(quarantines), 1)
            self.assertTrue((quarantines[0] / "partial.txt").is_file())
            self.assertTrue((material.population_dir / "autogluon" / "predictions" / "mfp__autogluon__r01__f01.csv").is_file())

    def test_incomplete_autogluon_folds_are_blocked_from_strict_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                folds, _predictions = run_autogluon_for_material(
                    material,
                    feature_ids=["ohe"],
                    time_limit=3,
                    num_cpus=2,
                    max_folds_per_task=1,
                )

            self.assertEqual(len(folds), 1)
            with self.assertRaisesRegex(PaperExactError, "不完整"):
                validate_complete_paper_exact_model_folds(folds, [material], model="autogluon", feature_ids=["ohe"])

    def test_paired_input_requires_complete_rf_and_autogluon_fold_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            with patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_fit_predict_fold,
            ):
                ag_folds, _predictions = run_autogluon_for_material(
                    material,
                    feature_ids=["ohe"],
                    time_limit=3,
                    num_cpus=2,
                )
            rf_folds = ag_folds.copy()
            rf_folds["model"] = "rf"
            rf_folds["rmse"] = rf_folds["rmse"] + 1.0
            rf_folds["mae"] = rf_folds["mae"] + 0.5
            rf_folds["r2"] = rf_folds["r2"] - 0.1
            rf_folds["kendall_tau"] = rf_folds["kendall_tau"] - 0.01
            paired = build_rf_autogluon_paired_input(rf_folds, ag_folds, [material], feature_ids=["ohe"])

            self.assertEqual(len(paired), 25)
            self.assertTrue((paired["delta_rmse_autogluon_minus_rf"] < 0).all())
            self.assertTrue((paired["paired_comparison_ready"] == True).all())
            with self.assertRaisesRegex(PaperExactError, "不完整"):
                build_rf_autogluon_paired_input(rf_folds.iloc[:-1], ag_folds, [material], feature_ids=["ohe"])

    @unittest.skipUnless(os.environ.get("YONOD_RUN_REAL_AG_SMOKE") == "1", "set YONOD_RUN_REAL_AG_SMOKE=1 to run real AG smoke")
    def test_real_autogluon_one_ohe_fold_smoke(self) -> None:
        try:
            import autogluon.tabular  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest("AutoGluon unavailable: {0}".format(exc))
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            material = load_or_prepare_paper_exact_material(
                REFERENCE_ROOT,
                output_root,
                "sl1_paper_exact",
            )
            folds, predictions = run_autogluon_for_material(
                material,
                feature_ids=["ohe"],
                time_limit=5,
                presets="medium_quality",
                num_cpus=2,
                cleanup=True,
                max_folds_per_task=1,
            )

        self.assertEqual(len(folds), 1)
        self.assertEqual(set(folds["evaluation_protocol"]), {PAPER_EXACT_EVALUATION_PROTOCOL})
        self.assertEqual(set(folds["model"]), {"autogluon"})
        self.assertEqual(set(predictions["model"]), {"autogluon"})
        self.assertTrue(np.isfinite(folds[["r2", "rmse", "mae", "kendall_tau"]].to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
