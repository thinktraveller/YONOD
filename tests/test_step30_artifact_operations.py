"""Focused acceptance tests for step-30.5 immutable feature operations."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from yonod.artifacts.operations import (
    ArtifactOperationError,
    derive_features,
    inspect_feature_artifact,
)
from yonod.artifacts.reader import load_feature_artifact, publish_feature_artifact


def _package_hashes(package: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        result[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class Step30ArtifactOperationsTests(unittest.TestCase):
    def _source(self, root: Path, *, columns: list[str] | None = None) -> Path:
        return publish_feature_artifact(
            root / "source",
            artifact_id="source-static-v1",
            feature_id="demo",
            matrix=np.asarray([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32),
            sample_ids=np.asarray(["s1", "s2", "s3"], dtype=np.str_),
            valid_mask=np.asarray([True, False, True], dtype=bool),
            dataset_identity="sha256:source-dataset",
            feature_config_identity="sha256:source-feature",
            column_structure={"kind": "named", "names": columns or ["a", "b"]},
        )

    @staticmethod
    def _operation(root: Path, *, operations: list[dict[str, object]], **extra: object) -> Path:
        path = root / "derive.yaml"
        payload: dict[str, object] = {
            "schema_version": "2.0",
            "operation": "derive_features",
            "input_manifest": "source/manifest.yaml",
            "output_dir": "derived",
            "operations": operations,
        }
        payload.update(extra)
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_inspect_is_verified_read_only_and_select_operations_keep_declared_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            before = _package_hashes(source.parent)
            inspection = inspect_feature_artifact(source)
            self.assertEqual(inspection.artifact_id, "source-static-v1")
            self.assertEqual(inspection.matrix_shape, (2, 2))
            self.assertEqual(inspection.sample_count, 3)
            self.assertEqual(inspection.valid_sample_count, 2)
            self.assertEqual(inspection.column_names, ("a", "b"))
            self.assertEqual(_package_hashes(source.parent), before)

            config = self._operation(root, operations=[
                {"kind": "select_samples", "sample_ids": ["s3", "s1"]},
                {"kind": "select_features", "columns": ["b"]},
            ])
            result = derive_features(config)
            self.assertEqual(len(result.operations), 2)
            first, second = result.operations
            self.assertNotEqual(first.manifest_path.parent, source.parent)
            self.assertNotEqual(second.manifest_path.parent, first.manifest_path.parent)
            final = load_feature_artifact(result.manifest_path)
            self.assertEqual(final.sample_ids.tolist(), ["s3", "s1"])
            self.assertEqual(final.valid_mask.tolist(), [True, True])
            self.assertEqual(final.manifest["matrix"]["columns"]["names"], ["b"])
            np.testing.assert_array_equal(final.matrix, np.asarray([[30.0], [10.0]], dtype=np.float32))
            self.assertEqual(final.manifest["derived_from"]["artifact_id"], first.artifact_id)
            self.assertEqual(final.manifest["derived_from"]["operation"], "select_features")
            self.assertEqual(_package_hashes(source.parent), before)

    def test_join_reorders_external_rows_by_id_without_expanding_parent_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            (root / "external.csv").write_text(
                "sample_id,side,unselected\n"
                "s3,300,ignore\n"
                "s1,100,ignore\n"
                "s2,200,ignore\n"
                "not-in-parent,999,ignore\n",
                encoding="utf-8",
            )
            config = self._operation(root, operations=[{
                "kind": "join_features",
                "table_path": "external.csv",
                "sample_id_col": "sample_id",
                "columns": ["side"],
                "prefix": "external_",
            }])
            result = derive_features(config)
            derived = load_feature_artifact(result.manifest_path)
            self.assertEqual(derived.sample_ids.tolist(), ["s1", "s2", "s3"])
            self.assertEqual(derived.valid_mask.tolist(), [True, False, True])
            self.assertEqual(derived.manifest["matrix"]["columns"]["names"], ["a", "b", "external_side"])
            np.testing.assert_array_equal(
                derived.matrix,
                np.asarray([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]]),
            )
            mapping = derived.manifest["derived_from"]["input_output_mapping"]
            self.assertEqual(mapping["sample_order"], "unchanged_from_parent")
            self.assertEqual(mapping["external_rows_ignored_outside_parent_population"], 1)
            self.assertEqual(derived.manifest["derived_from"]["artifact_id"], "source-static-v1")
            self.assertEqual(derived.manifest["metadata"]["derived_operation"]["kind"], "join_features")
            self.assertEqual(derived.manifest["metadata"]["derived_operation"]["declaration"]["external_table_filename"], "external.csv")
            self.assertNotIn(str(root), yaml.safe_dump(derived.manifest, allow_unicode=True))
            self.assertTrue(source.is_file())

    def test_invalid_selection_join_and_leakage_declarations_are_rejected_before_publication(self) -> None:
        cases = [
            (
                "unknown-sample",
                [{"kind": "select_samples", "sample_ids": ["s1", "missing"]}],
                None,
                "不在父产物",
            ),
            (
                "duplicate-requested-sample",
                [{"kind": "select_samples", "sample_ids": ["s1", "s1"]}],
                None,
                "不能包含重复值",
            ),
            (
                "unknown-column",
                [{"kind": "select_features", "columns": ["does_not_exist"]}],
                None,
                "列不存在",
            ),
            (
                "label-leak",
                [{
                    "kind": "join_features", "table_path": "external.csv", "sample_id_col": "sample_id",
                    "columns": ["yield"], "prefix": "external_",
                }],
                "sample_id,yield\ns1,1\ns2,2\ns3,3\n",
                "标签泄漏",
            ),
            (
                "duplicate-external-id",
                [{
                    "kind": "join_features", "table_path": "external.csv", "sample_id_col": "sample_id",
                    "columns": ["side"], "prefix": "external_",
                }],
                "sample_id,side\ns1,1\ns1,2\ns2,3\ns3,4\n",
                "重复",
            ),
            (
                "missing-external-id",
                [{
                    "kind": "join_features", "table_path": "external.csv", "sample_id_col": "sample_id",
                    "columns": ["side"], "prefix": "external_",
                }],
                "sample_id,side\ns1,1\ns3,3\n",
                "缺少父产物",
            ),
        ]
        for name, operations, external, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._source(root)
                before = _package_hashes(source.parent)
                if external is not None:
                    (root / "external.csv").write_text(external, encoding="utf-8")
                config = self._operation(root, operations=operations, protected_columns=["yield"])
                with self.assertRaisesRegex(ArtifactOperationError, message):
                    derive_features(config)
                self.assertFalse((root / "derived").exists())
                self.assertEqual(_package_hashes(source.parent), before)

    def test_conflicting_join_columns_existing_destination_and_parent_package_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, columns=["a", "external_a"])
            (root / "external.csv").write_text("sample_id,a\ns1,1\ns2,2\ns3,3\n", encoding="utf-8")
            conflict = self._operation(root, operations=[{
                "kind": "join_features", "table_path": "external.csv", "sample_id_col": "sample_id",
                "columns": ["a"], "prefix": "external_",
            }])
            with self.assertRaisesRegex(ArtifactOperationError, "列冲突"):
                derive_features(conflict)
            self.assertFalse((root / "derived").exists())

            valid = self._operation(root, operations=[{"kind": "select_features", "columns": ["a"]}])
            first = derive_features(valid)
            source_before = _package_hashes(source.parent)
            with self.assertRaisesRegex(ArtifactOperationError, "拒绝覆盖"):
                derive_features(valid)
            self.assertEqual(_package_hashes(source.parent), source_before)
            self.assertTrue(first.manifest_path.is_file())

            inside_parent = self._operation(
                root,
                operations=[{"kind": "select_features", "columns": ["a"]}],
                output_dir="source/attempted-child",
            )
            with self.assertRaisesRegex(ArtifactOperationError, "父产物包内"):
                derive_features(inside_parent)


if __name__ == "__main__":
    unittest.main()
