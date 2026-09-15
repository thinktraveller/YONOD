"""Focused acceptance tests for the dependency-light step-30.3 reader."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from yonod.artifacts.reader import (
    ArtifactReadError,
    import_legacy_npz,
    load_feature_artifact,
    publish_feature_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Step30ArtifactReaderTests(unittest.TestCase):
    def _publish(self, root: Path, name: str = "feature-package") -> Path:
        return publish_feature_artifact(
            root / name,
            artifact_id="morgan-v1",
            feature_id="morgan",
            matrix=np.arange(6, dtype=np.float32).reshape(2, 3),
            sample_ids=np.asarray(["sample-1", "sample-2", "sample-3"], dtype=np.str_),
            valid_mask=np.asarray([True, False, True], dtype=bool),
            dataset_identity="sha256:dataset-input",
            feature_config_identity="sha256:morgan-config",
            column_structure={"kind": "named", "names": ["m0", "m1", "m2"]},
        )

    @staticmethod
    def _rewrite_file_record(manifest_path: Path, relative: str) -> None:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        for record in raw["files"].values():
            if record["path"] == relative:
                file_path = manifest_path.parent / relative
                record["sha256"] = _sha256(file_path)
                record["bytes"] = file_path.stat().st_size
                break
        else:  # pragma: no cover - a fixture/programming error
            raise AssertionError(f"manifest does not reference {relative}")
        manifest_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def test_full_package_is_readable_after_copy_in_fresh_dependency_light_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._publish(root)
            artifact = load_feature_artifact(manifest, expected_artifact_id="morgan-v1")
            np.testing.assert_array_equal(artifact.valid_row_indices, np.asarray([0, 2]))
            self.assertFalse(artifact.matrix.flags.writeable)
            self.assertEqual(artifact.sample_ids.tolist(), ["sample-1", "sample-2", "sample-3"])

            moved = root / "moved-package"
            shutil.copytree(manifest.parent, moved)
            moved_manifest = moved / "manifest.yaml"
            moved_artifact = load_feature_artifact(moved_manifest, expected_feature_id="morgan")
            np.testing.assert_array_equal(moved_artifact.matrix, artifact.matrix)

            code = (
                "import sys; "
                "from yonod.artifacts.reader import load_feature_artifact; "
                f"a = load_feature_artifact({str(moved_manifest)!r}); "
                "blocked = {'rdkit', 'torch', 'yonod.descriptors', 'yonod.universal.feature_builder'}; "
                "assert not blocked.intersection(sys.modules), sorted(blocked.intersection(sys.modules)); "
                "print(a.matrix.shape)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("(2, 3)", completed.stdout)

    def test_tampered_or_missing_material_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tampered_manifest = self._publish(root, "tampered")
            np.save(tampered_manifest.parent / "data" / "matrix.npy", np.zeros((2, 3), dtype=np.float32))
            with self.assertRaisesRegex(ArtifactReadError, "哈希"):
                load_feature_artifact(tampered_manifest)

            missing_manifest = self._publish(root, "missing")
            (missing_manifest.parent / "data" / "sample_ids.npy").unlink()
            with self.assertRaisesRegex(ArtifactReadError, "不存在"):
                load_feature_artifact(missing_manifest)

    def test_duplicate_ids_and_invalid_mask_are_rejected_even_when_hashes_are_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_manifest = self._publish(root, "duplicates")
            duplicate_path = duplicate_manifest.parent / "data" / "sample_ids.npy"
            np.save(duplicate_path, np.asarray(["sample-1", "sample-1", "sample-3"], dtype=np.str_))
            self._rewrite_file_record(duplicate_manifest, "data/sample_ids.npy")
            with self.assertRaisesRegex(ArtifactReadError, "唯一"):
                load_feature_artifact(duplicate_manifest)

            mask_manifest = self._publish(root, "mask")
            mask_path = mask_manifest.parent / "data" / "validity_mask.npy"
            np.save(mask_path, np.asarray([True, True, True], dtype=bool))
            self._rewrite_file_record(mask_manifest, "data/validity_mask.npy")
            with self.assertRaisesRegex(ArtifactReadError, "有效行数"):
                load_feature_artifact(mask_manifest)

    def test_publisher_fails_without_replacing_any_existing_or_partial_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._publish(root, "published")
            original_hash = _sha256(manifest)
            with self.assertRaisesRegex(ArtifactReadError, "拒绝覆盖"):
                self._publish(root, "published")
            self.assertEqual(_sha256(manifest), original_hash)

            invalid_target = root / "invalid"
            with self.assertRaisesRegex(ArtifactReadError, "matrix 行数"):
                publish_feature_artifact(
                    invalid_target,
                    artifact_id="bad-v1",
                    feature_id="bad",
                    matrix=np.zeros((1, 2), dtype=np.float32),
                    sample_ids=np.asarray(["s1", "s2"], dtype=np.str_),
                    valid_mask=np.asarray([True, True], dtype=bool),
                    dataset_identity="sha256:d",
                    feature_config_identity="sha256:f",
                    column_structure={"kind": "generated_pattern"},
                )
            self.assertFalse(invalid_target.exists())

    def test_explicit_legacy_npz_import_never_changes_source_or_guesses_label_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy-v1.npz"
            np.savez_compressed(
                legacy,
                X_smiles=np.arange(6, dtype=np.int32).reshape(2, 3),
                sample_ids=np.asarray(["s1", "s2", "s3"], dtype=np.str_),
                valid_mask=np.asarray([True, False, True], dtype=bool),
                metadata_json=np.asarray(json.dumps({
                    "artifact_schema_version": 1,
                    "feature_dim": 3,
                    "n_total": 3,
                    "n_valid": 2,
                }), dtype=np.str_),
            )
            original_hash = _sha256(legacy)
            manifest = import_legacy_npz(
                legacy,
                root / "imported",
                artifact_id="legacy-import-v1",
                feature_id="mfp",
                dataset_identity="sha256:caller-supplied-dataset",
                feature_config_identity="sha256:caller-supplied-feature-config",
                column_structure={"kind": "generated_pattern", "pattern": "mfp_<index>"},
            )
            artifact = load_feature_artifact(manifest)
            self.assertEqual(_sha256(legacy), original_hash)
            self.assertEqual(artifact.manifest["metadata"]["legacy_import"]["source_filename"], "legacy-v1.npz")
            self.assertEqual(artifact.manifest["sources"]["dataset_identity"], "sha256:caller-supplied-dataset")
            self.assertNotIn("label", artifact.manifest)
            np.testing.assert_array_equal(artifact.valid_row_indices, np.asarray([0, 2]))


if __name__ == "__main__":
    unittest.main()
