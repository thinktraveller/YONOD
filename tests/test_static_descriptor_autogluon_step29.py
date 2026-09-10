"""Step29 static-descriptor AutoGluon manifest gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_static_descriptor_autogluon_matrix.py"


def _load_runner():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("step29_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless((ROOT / "derived/descriptor_model_effect/step28_3_matrix_plan/task_plan.json").is_file(), "step28 plan required")
class Step29StaticDescriptorAutoGluonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_prepare_payload_has_exact_275_fold_scope_and_rf_references(self):
        payload = self.runner._prepare_payload(ROOT / self.runner.DEFAULT_PLAN_DIR)
        tasks = payload["tasks"]
        combos = {(row["population_id"], row["descriptor"]) for row in tasks}
        model_config = next(iter(payload["model_configs"].values()))

        self.assertEqual(len(tasks), 275)
        self.assertEqual(len(combos), 11)
        self.assertEqual(combos, self.runner.EXPECTED_MATRIX_SET)
        self.assertEqual({row["model"] for row in tasks}, {"autogluon"})
        self.assertNotIn(("BH1-static-3955", "DFT"), combos)
        self.assertFalse(any(row["descriptor"] in {"MFP", "OHE"} for row in tasks))
        self.assertTrue(all(row["rf_predictions_sha256"] for row in tasks))
        self.assertEqual({int(row["n_valid"]) + int(row["n_train"]) for row in tasks if row["population_id"] == "SL1-static-1150"}, {1150})
        self.assertEqual(model_config["params"]["time_limit"], 300)
        self.assertEqual(model_config["params"]["num_cpus"], 19)


if __name__ == "__main__":
    unittest.main()
