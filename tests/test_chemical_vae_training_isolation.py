"""Regression proof that a schema-2 train stage consumes only an artifact."""

from __future__ import annotations

import builtins
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from yonod.artifacts.reader import publish_feature_artifact
from yonod.pipeline.training import run_train
from yonod.provenance import feature_dataset_identity


class ChemicalVaeTrainingIsolationTests(unittest.TestCase):
    """The second process must not import the Chemical VAE implementation."""

    def test_train_reads_a_verified_artifact_without_importing_the_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "sample_id": ["s1", "s2", "s3", "s4"],
                "yield": [0.1, 0.2, 0.3, 0.4],
                "reactant": ["C", "CC", "CCC", "CCCC"],
            })
            dataset_path = root / "input.csv"
            frame.to_csv(dataset_path, index=False)
            roles = {"label": "yield", "reactants": ["reactant"]}
            artifact_path = publish_feature_artifact(
                root / "features" / "chemical-vae-static-artifact",
                artifact_id="chemical-vae-static-artifact",
                feature_id="chemical-vae-zinc-v5",
                matrix=np.arange(8, dtype=np.float32).reshape(4, 2),
                sample_ids=frame["sample_id"].to_numpy(dtype=str),
                valid_mask=np.ones(4, dtype=bool),
                dataset_identity=feature_dataset_identity(
                    frame, sample_id_col="sample_id", column_roles=roles,
                ),
                feature_config_identity="sha256:training-isolation-fixture",
                column_structure={"kind": "named", "names": ["vae_0", "vae_1"]},
            )
            config_path = root / "train.yaml"
            config_path.write_text(yaml.safe_dump({
                "schema_version": "2.0", "project_name": "vae-train-isolation", "stage": "train",
                "dataset": {"path": "input.csv", "sample_id_col": "sample_id", "column_roles": roles},
                "artifacts": {"input_manifest": str(artifact_path), "output_dir": "train_artifacts"},
                "models": ["rf"],
                "model_params": {"rf": {"estimator": {"n_estimators": 1, "n_jobs": 1, "random_state": 42}}},
                "evaluation": {"protocol": "outer_kfold", "n_splits": 2, "n_repeats": 1, "shuffle": True, "seed": 42},
                "outputs": {"root": "train_outputs"},
            }, allow_unicode=True, sort_keys=False), encoding="utf-8")

            def fake_fold_runner(_resolved, _x_train, y_train, x_valid, _aligned, _indices, _fold_dir, _label, _fold):
                return np.full(len(x_valid), float(np.mean(y_train))), {"runner": "test_fake"}

            original_import = builtins.__import__

            def reject_encoder_import(name, *args, **kwargs):
                if name == "yonod.descriptors.chemical_vae" or name.startswith("yonod.descriptors.chemical_vae."):
                    raise AssertionError("train stage attempted to import Chemical VAE encoder")
                return original_import(name, *args, **kwargs)

            prior_module = sys.modules.pop("yonod.descriptors.chemical_vae", None)
            try:
                with patch("builtins.__import__", side_effect=reject_encoder_import):
                    runs = run_train(config_path, fold_runner=fake_fold_runner)
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].status, "complete")
                self.assertNotIn("yonod.descriptors.chemical_vae", sys.modules)
            finally:
                if prior_module is not None:
                    sys.modules["yonod.descriptors.chemical_vae"] = prior_module


if __name__ == "__main__":
    unittest.main()
