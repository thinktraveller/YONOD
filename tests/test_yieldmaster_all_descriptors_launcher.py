"""Opt-in launcher integration: real Conda/config loader, simulated model fit.

YONOD_RUN_LAUNCHER_INTEGRATION=1 python3 -m unittest \
    tests.test_yieldmaster_all_descriptors_launcher -v
All workers and outputs are confined to temporary fixture repositories.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "run_yieldmaster_all_descriptors_19cpu_nohup.sh"
DATASETS = ("bh1", "bh2", "sl1", "sm")
CONDA = os.environ.get("YONOD_CONDA_BIN", "/home/wangzh685/miniconda3/bin/conda")

SHIM_MAIN = '''import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["YONOD_TEST_SOURCE"])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("launcher_real_main", root / "main.py")
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)
_model_params_for = real._model_params_for
config_to_args = real.config_to_args
load_config_from_json = real.load_config_from_json

if __name__ == "__main__":
    config_path = Path(sys.argv[sys.argv.index("--json") + 1])
    csv_path = Path(sys.argv[sys.argv.index("--csv") + 1])
    config = load_config_from_json(config_path, csv_path)
    args = config_to_args(config, config_path)
    key = csv_path.stem.lower()
    assert len(os.sched_getaffinity(0)) == 19
    assert args.autogluon_num_cpus == 19
    assert len(args._feature_specs) == 8
    assert len(args.models) == 5
    assert args.output_dir == config_path.parent.parent
    with open("model_calls.txt", "a", encoding="utf-8") as stream:
        stream.write(key + "\\n")
    print("SIMULATED_STDOUT " + key, flush=True)
    print("SIMULATED_STDERR " + key, file=sys.stderr, flush=True)
    sys.exit(7 if key == os.environ.get("YONOD_TEST_FAIL_DATASET") else 0)
'''


@unittest.skipUnless(
    os.environ.get("YONOD_RUN_LAUNCHER_INTEGRATION") == "1",
    "set YONOD_RUN_LAUNCHER_INTEGRATION=1 to exercise real Conda",
)
class LauncherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="yonod-launcher-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / SCRIPT
        shutil.copy2(ROOT / "scripts" / SCRIPT, self.script)
        (self.root / "main.py").write_text(SHIM_MAIN, encoding="utf-8")
        self.sources = {}
        for key in DATASETS:
            name = f"yieldmaster_{key}_all_descriptors_all_models"
            relative = Path("configs") / name / f"{name}.json"
            target = self.root / relative
            target.parent.mkdir(parents=True)
            shutil.copy2(ROOT / relative, target)
            self.sources[key] = target
            csv_relative = Path("dataset/yieldmaster") / f"{key.upper()}.csv"
            csv_target = self.root / csv_relative
            csv_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / csv_relative, csv_target)
        self.env = dict(os.environ)
        cpus = sorted(os.sched_getaffinity(0))[:19]
        self.assertEqual(len(cpus), 19)
        self.env.update(
            YONOD_CONDA_BIN=CONDA,
            YONOD_CPU_SET=",".join(map(str, cpus)),
            YONOD_TEST_SOURCE=str(ROOT),
            PYTHONDONTWRITEBYTECODE="1",
        )

    def worker(self):
        return subprocess.run(
            ["bash", str(self.script), "--worker", "regression"],
            cwd=self.root, env=self.env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )

    def calls(self):
        path = self.root / "model_calls.txt"
        return path.read_text().splitlines() if path.exists() else []

    def test_nohup_serial_order_runtime_resources_and_logs(self):
        # Internal model key previously could shadow the display-key override.
        source = self.sources["bh1"]
        config = json.loads(source.read_text())
        config["model_params"] = {"autogluon": {"num_cpus": 1, "time_limit": 42}}
        source.write_text(json.dumps(config), encoding="utf-8")
        source_before = {k: p.read_bytes() for k, p in self.sources.items()}
        launch = subprocess.run(
            ["bash", str(self.script)], cwd=self.root, env=self.env,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(launch.returncode, 0, launch.stdout + launch.stderr)
        log_line = next(line for line in launch.stdout.splitlines() if line.startswith("完整日志: "))
        log_path = Path(log_line.split(": ", 1)[1])
        deadline = time.monotonic() + 150
        content = ""
        while time.monotonic() < deadline:
            content = log_path.read_text(encoding="utf-8")
            if "BATCH_FINISH" in content:
                break
            time.sleep(0.2)
        self.assertIn("BATCH_FINISH succeeded=4 failed=0 total=4", content)
        self.assertEqual(self.calls(), list(DATASETS))
        for key, source in self.sources.items():
            self.assertEqual(source.read_bytes(), source_before[key])
            runtime = next((source.parent / "docs").glob("runtime_*.json"))
            config = json.loads(runtime.read_text())
            self.assertEqual(config["runtime"]["cpu_affinity"], self.env["YONOD_CPU_SET"])
            self.assertEqual(config["model_params"]["autogluon"]["num_cpus"], 19)
            if key == "bh1":
                self.assertEqual(config["model_params"]["autogluon"]["time_limit"], 42)
            dataset_log = next((source.parent / "docs").glob("batch_*.full.log")).read_text()
            for marker in ("[config:validated]", "SIMULATED_STDOUT " + key, "SIMULATED_STDERR " + key):
                self.assertIn(marker, dataset_log)
                self.assertIn(marker, content)

    def test_invalid_config_blocks_first_fit_and_continues(self):
        config = json.loads(self.sources["bh1"].read_text())
        config["descriptors"][0]["descriptor"] = "invalid_descriptor"
        self.sources["bh1"].write_text(json.dumps(config), encoding="utf-8")
        result = self.worker()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), ["bh2", "sl1", "sm"])
        self.assertIn("dataset=bh1 stage=config", result.stdout)
        self.assertIn("succeeded=3 failed=1", result.stdout)

    def test_silent_config_producer_does_not_start_models(self):
        # Reproduce the original return-code=0, missing-output symptom.
        wrapper = self.root / "conda-silent-producer"
        wrapper.write_text(
            '#!/usr/bin/env bash\ncase " $* " in\n'
            '  *" python - "*) exit 0 ;;\nesac\n'
            'exec "$YONOD_TEST_REAL_CONDA" "$@"\n', encoding="utf-8",
        )
        wrapper.chmod(0o755)
        self.env.update(YONOD_CONDA_BIN=str(wrapper), YONOD_TEST_REAL_CONDA=CONDA)
        result = self.worker()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertIn("succeeded=0 failed=4", result.stdout)
        self.assertIn("运行时配置未生成或为空", result.stdout)

    def test_model_failure_is_not_hidden_by_tee(self):
        self.env["YONOD_TEST_FAIL_DATASET"] = "bh1"
        result = self.worker()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), list(DATASETS))
        self.assertIn("FINISH dataset=bh1 exit_code=7", result.stdout)
        self.assertIn("succeeded=3 failed=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
