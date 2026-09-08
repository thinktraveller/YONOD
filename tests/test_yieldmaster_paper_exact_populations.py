"""Integration guards for real YieldMaster paper-exact materials."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from yonod.benchmark.paper_exact_pipeline import (
    PAPER_EXACT_SEEDS,
    prepare_all_paper_exact_materials,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference-proejct" / "vjethbkm"


class YieldMasterPaperExactPopulationTests(unittest.TestCase):
    def test_real_populations_have_expected_rows_features_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            materials = prepare_all_paper_exact_materials(
                REFERENCE_ROOT,
                Path(temporary),
                stamp="test",
            )

            self.assertEqual(len(materials), 5)
            row_counts = {item.population_id: item.row_count for item in materials}
            self.assertEqual(row_counts["bh1_paper_exact"], 3955)
            self.assertEqual(row_counts["bh2_paper_exact"], 3359)
            self.assertEqual(row_counts["sl1_paper_exact"], 1150)
            self.assertEqual(row_counts["sm_mfp_paper_exact_4620"], 4620)
            self.assertEqual(row_counts["sm_ohe_paper_exact_5760"], 5760)

            descriptor_tasks = sum(len(item.feature_paths) for item in materials)
            self.assertEqual(descriptor_tasks, 8)

            for item in materials:
                split = pd.read_csv(item.split_manifest_path)
                self.assertEqual(len(split[["repeat", "fold"]].drop_duplicates()), 25)
                self.assertEqual(sorted(split["seed"].unique().tolist()), list(PAPER_EXACT_SEEDS))
                self.assertEqual(set(split["population_id"]), {item.population_id})
                self.assertEqual(split["split_hash"].nunique(), 1)
                for repeat, part in split.groupby("repeat"):
                    valid_counts = part.loc[part["role"] == "valid", "sample_id"].value_counts()
                    sample_ids = pd.read_csv(item.population_csv_path)["sample_id"].astype(str)
                    self.assertTrue(valid_counts.reindex(sample_ids, fill_value=0).eq(1).all(), repeat)

                if "mfp" in item.feature_paths:
                    self.assertIsNotNone(item.mfp_alignment)
                    self.assertTrue(item.mfp_alignment["x_exact"])
                    self.assertTrue(item.mfp_alignment["y_exact"])
                    self.assertTrue(item.mfp_alignment["smiles_columns_exact"])
                    with np.load(item.feature_paths["mfp"]) as feature:
                        self.assertEqual(feature["X"].shape[0], item.row_count)
                        self.assertEqual(str(feature["feature_hash"]), item.feature_hashes["mfp"])
                if "ohe" in item.feature_paths:
                    ohe_contract = Path(item.feature_paths["ohe"]).read_text(encoding="utf-8")
                    self.assertIn("train_only_per_fold", ohe_contract)
                    self.assertIn("__MISSING__", ohe_contract)


if __name__ == "__main__":
    unittest.main()
