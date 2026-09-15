"""Focused acceptance tests for the step-30.2 safe YAML loader."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from yonod.config.contracts import MISSING
from yonod.config.loader import (
    ConfigLoadError,
    build_explicit_cli_overrides,
    load_operation_config,
    load_run_config,
    load_yaml_mapping,
    merge_config_layers,
)


def _run_yaml(*, schema_version: str = "2.0", label: str = "yield", models: str = "[xgboost]") -> str:
    label_line = f"    label: {label}\n" if label else ""
    return textwrap.dedent(
        f"""\
        schema_version: \"{schema_version}\"
        project_name: loader-demo
        stage: all
        dataset:
          path: data/reactions.csv
          sample_id_col: sample_id
          column_roles:
        {label_line.rstrip()}
            reactants: [reactant]
        descriptors:
          - id: morgan-r3
            descriptor: morgan
            columns: [reactant]
        artifacts:
          output_dir: artifacts/demo
        models: {models}
        model_params:
          xgboost:
            estimator:
              n_estimators: 25
              max_depth: null
        evaluation:
          n_splits: 2
          n_repeats: 1
          shuffle: true
        """
    )


class Step30YamlLoaderTests(unittest.TestCase):
    def _write(self, root: Path, name: str, content: str) -> Path:
        path = root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_yaml_before_defaults_then_explicit_cli_preserves_null_false_and_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._write(root, "run.yaml", _run_yaml())
            cli = build_explicit_cli_overrides(
                {"trees": 0, "shuffle": False, "depth": None},
                {
                    "model_params.xgb.estimator.n_estimators": "trees",
                    "model_params.xgb.estimator.max_depth": "depth",
                    "evaluation.shuffle": "shuffle",
                },
            )
            loaded = load_run_config(
                path,
                library_defaults={
                    "model_params": {"xgb": {"estimator": {"n_estimators": 100, "n_jobs": 1}}},
                    "evaluation": {"shuffle": False, "seed": 5},
                },
                explicit_cli=cli,
            )

        self.assertEqual(loaded.explicit_yaml["models"], ["xgb"])
        estimator = loaded.effective["model_params"]["xgb"]["estimator"]
        self.assertEqual(estimator["n_estimators"], 0)
        self.assertIsNone(estimator["max_depth"])
        self.assertEqual(estimator["n_jobs"], 1)
        self.assertFalse(loaded.effective["evaluation"]["shuffle"])
        self.assertEqual(loaded.effective["evaluation"]["seed"], 5)

    def test_missing_cli_overlay_cannot_override_yaml_and_direct_merge_keeps_explicit_values(self) -> None:
        merged = merge_config_layers(
            {"number": 4, "nested": {"keep": "library", "toggle": True}},
            {"number": None, "zero": 0, "nested": {"toggle": False}},
            MISSING,
        )
        self.assertIsNone(merged["number"])
        self.assertEqual(merged["zero"], 0)
        self.assertEqual(merged["nested"], {"keep": "library", "toggle": False})
        self.assertEqual(build_explicit_cli_overrides({"ignored": 9}, []), {})

    def test_duplicate_keys_multidoc_non_yaml_and_unsupported_version_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = self._write(root, "duplicate.yaml", "dataset:\n  path: first\n  path: second\n")
            with self.assertRaisesRegex(ConfigLoadError, "dataset.path"):
                load_yaml_mapping(duplicate)

            multi = self._write(root, "multi.yml", "---\na: 1\n---\na: 2\n")
            with self.assertRaisesRegex(ConfigLoadError, "恰好包含一个"):
                load_yaml_mapping(multi)

            json_named = self._write(root, "legacy.json", _run_yaml())
            with self.assertRaisesRegex(ConfigLoadError, "仅接受 .yaml 或 .yml"):
                load_run_config(json_named)

            old = self._write(root, "old.yaml", _run_yaml(schema_version="1.0"))
            with self.assertRaisesRegex(ConfigLoadError, "schema_version"):
                load_run_config(old)

    def test_alias_conflict_has_complete_field_path_and_static_contract_precedes_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conflict = self._write(root, "conflict.yaml", _run_yaml(models="[xgb, XGBoost]"))
            with self.assertRaisesRegex(ConfigLoadError, r"models\[1\].*models\[0\]"):
                load_run_config(conflict)

            no_label = self._write(root, "no-label.yaml", _run_yaml(label=""))
            with self.assertRaisesRegex(ConfigLoadError, "label"):
                load_run_config(
                    no_label,
                    library_defaults={"dataset": {"column_roles": {"label": "must-not-fill"}}},
                )

    def test_operation_config_uses_same_safe_yaml_gate(self) -> None:
        operation = textwrap.dedent(
            """\
            schema_version: "2.0"
            operation: derive_features
            input_manifest: features/source/manifest.yaml
            output_dir: features/derived/v2
            operations:
              - kind: select_features
                columns: [morgan_0]
            """
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), "derive.yml", operation)
            config = load_operation_config(path)
        self.assertEqual(config["operation"], "derive_features")


if __name__ == "__main__":
    unittest.main()
