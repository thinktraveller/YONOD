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

    def test_smoke_clone_task_id_is_stable_without_changing_formal_task_id(self):
        payload = self.runner._load_plan(ROOT / self.runner.DEFAULT_PLAN_DIR)
        formal = next(
            row for row in payload["tasks"]
            if row["population_id"] == "SL1-static-1150"
            and row["descriptor"] == "PhysChem"
            and int(row["repeat_zero_based"]) == 0
            and int(row["fold_zero_based"]) == 0
        )
        formal_contract_hash = self.runner.smoke._hash_payload(self.runner._task_contract(formal))
        formal_task_id = formal["task_id"]

        self.assertEqual(formal["task_contract_hash"], formal_contract_hash)
        self.assertEqual(self.runner._task_id(formal, formal_contract_hash), formal_task_id)
        self.assertTrue(formal_task_id.startswith("s29-sl1-physchem-autogluon-r00-f00-"))

        smoke_task = self.runner._clone_smoke_task(
            formal,
            ROOT / self.runner.DEFAULT_PLAN_DIR,
            time_limit=30,
            num_cpus=2,
        )
        smoke_contract_hash = self.runner.smoke._hash_payload(self.runner._task_contract(smoke_task))

        self.assertEqual(smoke_task["task_contract_hash"], smoke_contract_hash)
        self.assertEqual(self.runner._task_id(smoke_task, smoke_contract_hash), smoke_task["task_id"])
        self.assertTrue(smoke_task["task_id"].startswith("s29-smoke-sl1-physchem-autogluon-r00-f00-"))
        self.assertNotEqual(smoke_task["task_id"], formal_task_id)
        self.assertEqual(formal["task_id"], formal_task_id)
        self.runner._verify_task_inputs(smoke_task, verify_feature_array=False)


if __name__ == "__main__":
    unittest.main()
