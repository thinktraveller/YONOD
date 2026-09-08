"""Paper-exact 5x5 AutoGluon/RF manifest and mock-runner contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
from yonod.benchmark.paper_exact import (
    PAPER_EXACT_EVALUATION_PROTOCOL,
    PAPER_EXACT_POPULATIONS,
    PaperExactPopulationSpec,
    build_population_manifest,
    compute_fold_metrics,
    enumerate_paper_exact_tasks,
    execute_paper_exact_fold,
    validate_paper_exact_split_manifest,
)
from yonod.splits.manifest import create_split_manifest


_AUTOGUON_CALLS = []


@contextmanager
def _pickle_backed_parquet_io():
    original_to_parquet = pd.DataFrame.to_parquet
    original_read_parquet = pd.read_parquet

    def to_parquet_pickle(self, path, *args, index=False, **kwargs):
        frame = self if index else self.reset_index(drop=True)
        frame.to_pickle(path)

    def read_parquet_pickle(path, *args, **kwargs):
        return pd.read_pickle(path)

    with patch.object(pd.DataFrame, "to_parquet", to_parquet_pickle), \
            patch("pandas.read_parquet", side_effect=read_parquet_pickle):
        yield


def _fake_autogluon_fit_predict_fold(self, X_train, y_train, X_valid, *, fold_index, context=None):
    train = np.asarray(X_train, dtype=float)
    valid = np.asarray(X_valid, dtype=float)
    target = np.asarray(y_train, dtype=float)
    context = dict(context or {})
    _AUTOGUON_CALLS.append({
        "fold_index": int(fold_index),
        "X_train": train.copy(),
        "X_valid": valid.copy(),
        "y_train": target.copy(),
        "context": context,
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
    })
    # Tie-free deterministic fake predictions keep Kendall tau finite without
    # spending time in a real AutoGluon training loop.
    prediction = valid[:, 0].astype(float) + np.arange(len(valid), dtype=float) * 0.001
    return prediction, {
        "evaluation_protocol": context.get("evaluation_protocol"),
        "protocol_family": context.get("protocol_family"),
        "fold_index": int(fold_index),
        "outer_seed": int(context.get("outer_seed", -1)),
        "time_limit": self.time_limit,
        "presets": self.presets,
        "num_cpus": self.num_cpus,
        "random_state": self.random_state,
        "random_state_policy": "fake paper exact split fixed; AutoGluon internals not bitwise-seeded",
        "autogluon_versions": {"autogluon.tabular": "fake-1.1.1"},
        "train_time_s": 0.11,
        "predict_time_s": 0.02,
        "model_artifact_path": context.get("model_artifact_path"),
        "model_artifact_cleanup": bool(context.get("cleanup")),
        "leaderboard_rows": 1,
    }


class PaperExactRepeatedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        _AUTOGUON_CALLS.clear()

    def _frame(self, n: int = 12) -> pd.DataFrame:
        return pd.DataFrame({
            "sample_id": [f"s{index:02d}" for index in range(n)],
            "Yield": np.linspace(1.0, 99.0, n),
            "a": [f"A_unique_{index}" for index in range(n)],
            "b": ["B0", "B1", "B2", "B3"] * (n // 4),
        })

    def _contract(self, directory: Path, frame: pd.DataFrame, *, population_id: str = "fixture_paper_exact"):
        frame.to_csv(directory / "fixture.csv", index=False)
        config_path = directory / "paper_exact.yaml"
        config_path.write_text(yaml.safe_dump({"benchmark": {
            "dataset_path": "fixture.csv",
            "population_id": population_id,
            "dataset_id": "BH1",
            "sample_id_col": "sample_id",
            "label_col": "Yield",
            "smiles_cols": ["a", "b"],
            "feature_sets": [
                {"name": "mfp", "kind": "precomputed_descriptor", "component_cols": ["a", "b"], "params": {
                    "algorithm": "morgan_count", "radius": 3, "fp_size": 1024,
                }},
                {"name": "ohe", "kind": "fold_transform", "component_cols": ["a", "b"], "params": {
                    "encoder": "OneHotEncoder", "handle_unknown": "ignore",
                }},
            ],
            "models": ["rf", "autogluon"],
            "model_params": {
                "rf": {"n_estimators": 3, "max_features": 0.3, "n_jobs": 1},
                "autogluon": {"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
            },
            "grouping": {"strategy": "repeated_kfold", "source_order": "raw_csv"},
            "cv": {"n_repeats": 5, "n_splits": 5, "seed": 1000},
            "reproduction_protocol": {
                "name": "vjethbkm_paper_exact_5x5",
                "evaluation_protocol": PAPER_EXACT_EVALUATION_PROTOCOL,
                "dataset_id": "BH1",
            },
            "outputs": {"root": "paper_exact_results"},
        }}, allow_unicode=True), encoding="utf-8")
        config = BenchmarkConfig.from_file(config_path)
        return create_benchmark_contract(config)

    def test_population_specs_keep_sm_ohe_and_sm_mfp_boundaries_separate(self) -> None:
        self.assertEqual(PAPER_EXACT_POPULATIONS["sm_ohe_paper_exact_5760"].expected_rows, 5760)
        self.assertEqual(PAPER_EXACT_POPULATIONS["sm_ohe_paper_exact_5760"].feature_ids, ("ohe",))
        self.assertEqual(PAPER_EXACT_POPULATIONS["sm_mfp_paper_exact_4620"].expected_rows, 4620)
        self.assertEqual(PAPER_EXACT_POPULATIONS["sm_mfp_paper_exact_4620"].feature_ids, ("mfp",))

        frame = self._frame(4)
        spec = replace(PAPER_EXACT_POPULATIONS["sm_mfp_paper_exact_4620"], expected_rows=4)
        population = build_population_manifest(
            frame.rename(columns={"a": "reactant_1_smiles", "b": "reactant_2_smiles"}).assign(
                ligand_smiles="L", reagent_1_smiles="R", solvent_1_smiles="S"
            ),
            spec,
            dataset_sha256="dataset-hash",
            source_row_indices=[0, 2, 4, 7],
        )
        self.assertEqual(len(population), 4)
        self.assertEqual(population["source_row_index"].tolist(), [0, 2, 4, 7])
        self.assertEqual(set(population["population_id"]), {"sm_mfp_paper_exact_4620"})
        self.assertIn("mfp", population["feature_ids_json"].iloc[0])

        with self.assertRaisesRegex(Exception, "行数"):
            build_population_manifest(frame, PaperExactPopulationSpec(
                population_id="bad", dataset_id="BAD", feature_ids=("mfp",), expected_rows=5,
                label_col="Yield", component_cols=("a", "b"), source="fixture",
            ), dataset_sha256="dataset-hash")

    def test_repeated_kfold_manifest_matches_literal_sklearn_and_enumerates_shared_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = self._frame(12)
            contract = self._contract(directory, frame)
            population = build_population_manifest(
                frame,
                PaperExactPopulationSpec(
                    population_id="fixture_paper_exact", dataset_id="BH1", feature_ids=("mfp", "ohe"),
                    expected_rows=len(frame), label_col="Yield", component_cols=("a", "b"), source="fixture.csv",
                ),
                dataset_sha256=contract.dataset_sha256,
                sample_id_col="sample_id",
            )
            manifest = create_split_manifest(contract)
            validate_paper_exact_split_manifest(manifest, population)
            tasks = enumerate_paper_exact_tasks(contract, manifest)

        self.assertEqual(len(manifest), len(frame) * 25)
        self.assertEqual(set(manifest["population_id"]), {"fixture_paper_exact"})
        self.assertEqual(manifest["split_hash"].nunique(), 1)
        self.assertTrue(str(manifest["split_id"].iloc[0]).endswith(manifest["split_hash"].iloc[0][:12]))
        self.assertEqual(set(manifest["seed"]), set(range(1000, 1005)))
        self.assertEqual(len(tasks), 2 * 2 * 25)
        self.assertEqual(tasks[0].population_id, "fixture_paper_exact")
        self.assertEqual(tasks[0].feature_id, "mfp")
        self.assertEqual(tasks[0].model, "rf")
        self.assertEqual(tasks[0].seed, 1000)
        self.assertEqual(tasks[-1].task_key, "fixture_paper_exact:ohe:autogluon:r05:f05")

        for repeat in range(1, 6):
            seed = 999 + repeat
            expected = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(frame))
            repeat_part = manifest[manifest["repeat"] == repeat]
            for fold, (_, expected_valid) in enumerate(expected, start=1):
                actual = repeat_part.loc[
                    (repeat_part["fold"] == fold) & (repeat_part["role"] == "valid"),
                    "source_row_index",
                ].to_numpy()
                np.testing.assert_array_equal(np.sort(actual), np.sort(expected_valid))

    def test_paper_exact_autogluon_fold_uses_manifest_seed_and_writes_auditable_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = self._frame(12)
            contract = self._contract(directory, frame)
            manifest = create_split_manifest(contract)
            sample_ids = frame["sample_id"].astype(str).to_numpy()
            X = np.arange(len(frame) * 4, dtype=float).reshape(len(frame), 4)
            y = frame["Yield"].to_numpy(dtype=float)
            with _pickle_backed_parquet_io(), patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_autogluon_fit_predict_fold,
            ):
                result = execute_paper_exact_fold(
                    contract, manifest, sample_ids, X, y,
                    "mfp", "autogluon", 2, 3,
                    model_kwargs={"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
                )
                metadata = json.loads(result.execution.metadata_path.read_text(encoding="utf-8"))

        part = manifest[(manifest["repeat"] == 2) & (manifest["fold"] == 3)]
        valid_ids = set(part.loc[part["role"] == "valid", "sample_id"].astype(str))
        train_ids = set(part.loc[part["role"] == "train", "sample_id"].astype(str))
        train_idx = np.flatnonzero(np.isin(sample_ids, list(train_ids)))
        valid_idx = np.flatnonzero(np.isin(sample_ids, list(valid_ids)))
        call = _AUTOGUON_CALLS[0]

        np.testing.assert_array_equal(call["X_train"], X[train_idx])
        np.testing.assert_array_equal(call["X_valid"], X[valid_idx])
        self.assertEqual(call["context"]["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)
        self.assertEqual(call["context"]["protocol_family"], "manifest_outer_cv")
        self.assertEqual(call["context"]["outer_seed"], 1001)
        self.assertTrue(call["context"]["model_artifact_path"].endswith("mfp__autogluon__r02__f03"))

        prediction = result.prediction_frame
        self.assertEqual(set(prediction["evaluation_protocol"]), {PAPER_EXACT_EVALUATION_PROTOCOL})
        self.assertEqual(set(prediction["population_id"]), {"fixture_paper_exact"})
        self.assertEqual(set(prediction["dataset_id"]), {"BH1"})
        self.assertEqual(set(prediction["feature_id"]), {"mfp"})
        self.assertEqual(set(prediction["seed"]), {1001})
        self.assertEqual(prediction["sample_id"].tolist(), sample_ids[valid_idx].tolist())
        self.assertEqual(prediction["source_row_index"].tolist(), valid_idx.tolist())
        self.assertIn("kendall_tau", result.fold_metrics)
        self.assertTrue(np.isfinite(result.fold_metrics["rmse"]))

        self.assertEqual(metadata["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)
        self.assertEqual(metadata["protocol_family"], "manifest_outer_cv")
        self.assertEqual(metadata["population_id"], "fixture_paper_exact")
        self.assertEqual(metadata["dataset_id"], "BH1")
        self.assertEqual(metadata["manifest_seed"], 1001)
        self.assertEqual(metadata["outer_seed"], 1001)
        self.assertEqual(metadata["autogluon_time_limit"], 7)
        self.assertEqual(metadata["autogluon_version"], "fake-1.1.1")
        self.assertIsNotNone(metadata["split_hash"])
        self.assertEqual(metadata["feature_schema_hash"], metadata["feature_schema_hash"])
        self.assertEqual(metadata["model_adapter_metadata"]["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)

    def test_paper_exact_ohe_autogluon_still_fits_categories_inside_each_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = self._frame(12)
            contract = self._contract(directory, frame)
            manifest = create_split_manifest(contract)
            sample_ids = frame["sample_id"].astype(str).to_numpy()
            y = frame["Yield"].to_numpy(dtype=float)
            with _pickle_backed_parquet_io(), patch(
                "yonod.models.autogluon_model.AutoGluonYieldModel.fit_predict_fold",
                new=_fake_autogluon_fit_predict_fold,
            ):
                result = execute_paper_exact_fold(
                    contract, manifest, sample_ids, None, y,
                    "ohe", "autogluon", 1, 1,
                    model_kwargs={"time_limit": 7, "presets": "medium_quality", "num_cpus": 2},
                    component_frame=frame[["a", "b"]],
                    fold_transformer=ReactionComponentOHE(["a", "b"]),
                )
                metadata = json.loads(result.execution.metadata_path.read_text(encoding="utf-8"))

        transformer_meta = metadata["feature_transformer"]
        self.assertEqual(metadata["evaluation_protocol"], PAPER_EXACT_EVALUATION_PROTOCOL)
        self.assertEqual(transformer_meta["fit_scope"], "train_only_per_fold")
        self.assertIsNotNone(transformer_meta["train_sample_ids_hash"])
        self.assertGreater(transformer_meta["valid_unseen_component_count"], 0)
        first_block_width = int(transformer_meta["category_counts"][0])
        self.assertTrue(np.allclose(_AUTOGUON_CALLS[0]["X_valid"][:, :first_block_width], 0.0))
        self.assertEqual(set(result.prediction_frame["feature_id"]), {"ohe"})
        self.assertEqual(set(result.prediction_frame["evaluation_protocol"]), {PAPER_EXACT_EVALUATION_PROTOCOL})

    def test_fold_metrics_include_kendall_tau_b(self) -> None:
        metrics = compute_fold_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 2.0, 4.0])
        self.assertEqual(set(metrics), {"r2", "rmse", "mae", "kendall_tau"})
        self.assertAlmostEqual(metrics["kendall_tau"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
