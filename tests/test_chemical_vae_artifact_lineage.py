"""Step-31.5 contracts for Chemical VAE asset identity and diagnostics lineage."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from yonod.artifacts.operations import derive_features
from yonod.artifacts.reader import ArtifactReadError, load_feature_artifact
from yonod.pipeline.features import run_features


class ChemicalVaeArtifactLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.source_asset = (
            cls.repository_root / "WEIGHTS" / "chemical_vae" / "zinc-37e96cd3bc8f9680" / "v5"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_features_config(self, root: Path, manifest: Path) -> Path:
        pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "left": ["CCO", None, "C" * 121],
            "right": ["CCO.CC", "c1ccccc1", None],
        }).to_csv(root / "input.csv", index=False)
        config = {
            "schema_version": "2.0", "project_name": "chemical-vae-lineage", "stage": "features",
            "dataset": {
                "path": "input.csv", "sample_id_col": "sample_id",
                "column_roles": {"reactants": ["left"], "others": ["right"]},
            },
            "descriptors": [{
                "id": "vae", "descriptor": "chemical_vae", "columns": ["left", "right"],
                "params": {"model_manifest": str(manifest.relative_to(root)), "batch_size": 2},
            }],
            "artifacts": {"output_dir": "artifacts"},
        }
        path = root / "features.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_sidecar_is_full_population_hash_verified_and_select_samples_stays_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset"
            shutil.copytree(self.source_asset, asset)
            features_config = self._write_features_config(root, asset / "conversion_manifest.json")
            result = run_features(features_config)
            source = load_feature_artifact(result.features[0].manifest_path)
            self.assertEqual(source.valid_mask.tolist(), [True, True, False])
            self.assertIsNotNone(source.diagnostics)
            self.assertEqual([row["sample_id"] for row in source.diagnostics["rows"]], ["s1", "s2", "s3"])
            self.assertEqual(source.diagnostics["rows"][0]["roles"]["right"]["reason"], "unsupported_character")
            self.assertEqual(source.diagnostics["rows"][1]["roles"]["left"]["reason"], "missing_input")

            operation = {
                "schema_version": "2.0", "operation": "derive_features",
                "input_manifest": str(result.features[0].manifest_path), "output_dir": "derived",
                "operations": [
                    {"kind": "select_samples", "sample_ids": ["s2", "s1"]},
                    {"kind": "select_features", "columns": ["vae::chemical_vae_0"]},
                ],
            }
            operation_path = root / "derive.yaml"
            operation_path.write_text(yaml.safe_dump(operation, allow_unicode=True, sort_keys=False), encoding="utf-8")
            derived = load_feature_artifact(derive_features(operation_path).manifest_path)
            self.assertEqual(derived.sample_ids.tolist(), ["s2", "s1"])
            self.assertEqual(derived.valid_mask.tolist(), [True, True])
            self.assertEqual(derived.matrix.shape, (2, 1))
            self.assertEqual([row["sample_id"] for row in derived.diagnostics["rows"]], ["s2", "s1"])
            self.assertEqual(derived.diagnostics["rows"][0]["roles"]["left"]["reason"], "missing_input")

            diagnostic_path = result.features[0].manifest_path.parent / "data" / "diagnostics.json"
            original = diagnostic_path.read_text(encoding="utf-8")
            self.assertIn('"encoded"', original)
            diagnostic_path.write_text(original.replace('"encoded"', '"xncoded"', 1), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactReadError, "内容哈希不匹配"):
                load_feature_artifact(result.features[0].manifest_path)

    def test_same_path_state_replacement_creates_new_feature_identity_before_cache_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset"
            shutil.copytree(self.source_asset, asset)
            manifest_path = asset / "conversion_manifest.json"
            config = self._write_features_config(root, manifest_path)
            first = run_features(config)
            self.assertEqual(first.features[0].status, "ready")

            state_path = asset / "encoder_state_dict.pt"
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            payload["step31_5_rewrite_marker"] = "same-path content replacement"
            torch.save(payload, state_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["converted_state_sha256"] = self._sha256(state_path)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            second = run_features(config)
            self.assertEqual(second.features[0].status, "ready")
            self.assertNotEqual(first.features[0].artifact_id, second.features[0].artifact_id)
            self.assertNotEqual(first.features[0].feature_identity, second.features[0].feature_identity)
            self.assertNotEqual(first.features[0].manifest_path, second.features[0].manifest_path)


if __name__ == "__main__":
    unittest.main()
