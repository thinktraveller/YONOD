"""Public main.py YAML dispatch acceptance tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

import main


class Step30MainYamlCliTests(unittest.TestCase):
    def test_json_is_rejected_with_migration_guidance(self) -> None:
        self.assertEqual(main.main(["--json", "legacy.json"]), 1)

    def test_features_yaml_is_a_supported_public_entry_without_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"sample_id": ["s1", "s2"], "reactant": ["CC", "O"]}).to_csv(root / "data.csv", index=False)
            config = {
                "schema_version": "2.0", "project_name": "main-features", "stage": "features",
                "dataset": {"path": "data.csv", "sample_id_col": "sample_id", "column_roles": {"reactants": ["reactant"]}},
                "descriptors": [{"id": "morgan", "descriptor": "morgan", "columns": ["reactant"]}],
                "artifacts": {"output_dir": "artifacts"},
            }
            path = root / "features.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            self.assertEqual(main.main(["--config", str(path)]), 0)
            self.assertTrue(any((root / "artifacts" / "feature_runs").glob("*.yaml")))


if __name__ == "__main__":
    unittest.main()
