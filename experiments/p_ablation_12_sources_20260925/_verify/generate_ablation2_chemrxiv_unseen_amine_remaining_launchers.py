"""Generate frozen launchers for the eleven remaining grouped unseen-A tasks.

The baseline Morgan/RF full-input task must already have completed and pass
the read-only result verifier.  This generator does not start a model.  It
publishes one foreground launcher per remaining task and a master Bash runner
that launches precisely one headless ``yonod.py`` task at a time, waits for
it, verifies its evidence, then advances.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import stat
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.benchmark.config import BenchmarkConfig


PLAN_ID = "ablation2_chemrxiv_unseen_amine_remaining_v1"
BASELINE = "ablation2_chemrxiv_unseen_amine_morgan_rf_full_v1"
REMAINING = (
    "ablation2_chemrxiv_unseen_amine_morgan_rf_minus_a_v1",
    "ablation2_chemrxiv_unseen_amine_morgan_rf_minus_p_v1",
    "ablation2_chemrxiv_unseen_amine_morgan_rf_minus_a_p_v1",
    "ablation2_chemrxiv_unseen_amine_morgan_rf_product_conditions_v1",
    "ablation2_chemrxiv_unseen_amine_morgan_rf_conditions_only_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_full_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_minus_a_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_minus_p_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_minus_a_p_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_product_conditions_v1",
    "ablation2_chemrxiv_unseen_amine_mfp_lightgbm_conditions_only_v1",
)
GENERATED_ROOT = ROOT / "_verify" / "generated" / PLAN_ID
TASK_DIR = GENERATED_ROOT / "tasks"
MASTER = GENERATED_ROOT / "run_remaining_linearly.sh"
RESULT_VERIFIER = ROOT / "_verify" / "verify_ablation2_chemrxiv_unseen_amine_result.py"


def _config_path(task_name: str) -> Path:
    return ROOT / "config" / f"{task_name}.yaml"


def _result_root(task_name: str) -> Path:
    return ROOT / "result" / task_name


def _task_state_counts(path: Path) -> dict[str, int]:
    database = path / "docs" / "state" / "tasks.sqlite"
    if not database.is_file():
        raise RuntimeError(f"baseline task-state database is missing: {database}")
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
    return {str(status): int(count) for status, count in rows}


def _assert_baseline_completed() -> None:
    root = _result_root(BASELINE)
    required = (
        root / "report" / "benchmark_report.html",
        root / "report" / "benchmark_report.md",
        RESULT_VERIFIER,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("baseline completion evidence is missing: {0}".format(", ".join(missing)))
    statuses = _task_state_counts(root)
    if sum(statuses.values()) != 15 or statuses.get("succeeded") != 15 or statuses.get("failed", 0) != 0:
        raise RuntimeError(f"baseline task-state is not 15 succeeded / 0 failed: {statuses}")
    if any(name != "succeeded" and count for name, count in statuses.items()):
        raise RuntimeError(f"baseline task-state contains non-success rows: {statuses}")


def _assert_formal_config(task_name: str) -> None:
    config_path = _config_path(task_name)
    if not config_path.is_file():
        raise RuntimeError(f"remaining task configuration is missing: {config_path}")
    config = BenchmarkConfig.from_file(config_path)
    if config.outputs_root != _result_root(task_name):
        raise RuntimeError(f"{task_name}: output root is not the isolated task root")
    if config.artifact_output_dir != config.outputs_root / "feature":
        raise RuntimeError(f"{task_name}: feature path is not isolated below task root")
    if config.cv != {"n_splits": 5, "n_repeats": 3, "seed": 20260918}:
        raise RuntimeError(f"{task_name}: expected 5-fold x 3-repeat configuration")
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report_formats = tuple(dict(source.get("outputs", {})).get("report_formats", ()))
    if report_formats != ("html", "markdown"):
        raise RuntimeError(f"{task_name}: HTML and Markdown reports are not both declared")
    model_name = config.models[0]
    if model_name not in {"rf", "lightgbm"}:
        raise RuntimeError(f"{task_name}: unexpected model {model_name}")
    if config.model_configs[model_name]["estimator"].get("n_jobs") != 19:
        raise RuntimeError(f"{task_name}: selected model is not assigned 19 CPU threads")


def _assert_absent_evidence(task_name: str) -> None:
    evidence = (
        _result_root(task_name),
        ROOT / "logs" / f"{task_name}.log",
        ROOT / "logs" / f"{task_name}.pid",
        ROOT / "logs" / f"{task_name}.verification.json",
    )
    existing = [str(path.relative_to(ROOT)) for path in evidence if path.exists()]
    if existing:
        raise RuntimeError(f"{task_name}: refusing to overwrite existing evidence: {', '.join(existing)}")


def _render_or_validate(path: Path, content: str, *, executable: bool, write: bool) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite divergent generated file: {path}")
        if executable and not os.access(path, os.X_OK):
            if not write:
                raise RuntimeError(f"generated launcher is not executable: {path}")
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return
    if not write:
        raise RuntimeError(f"generated file is missing; rerun with --write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    if executable:
        temporary.chmod(temporary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    temporary.replace(path)


def _task_launcher(task_name: str) -> str:
    return """#!/usr/bin/env bash
# Generated by {generator}; Bash/Linux only.  This foreground launcher does
# not make its own log or PID: the serial master owns task evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." && pwd)"
cd "$repo_root"
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod

exec taskset --cpu-list 0-18 python -u yonod.py <<'YONOD_CONFIG'
config/{task_name}.yaml
YONOD_CONFIG
""".format(generator=Path(__file__).name, task_name=task_name)


def _master_runner(task_paths: list[Path]) -> str:
    launchers = "\n".join('  "${{script_dir}}/tasks/{0}"'.format(path.name) for path in task_paths)
    return """#!/usr/bin/env bash
# Generated by {generator}; Bash/Linux only.
#
# This runner is strictly serial: one task is headlessly launched with the
# exact CPU list 0-18, awaited as its direct child, then checked for 15/15
# SQLite successes and both reports.  Any startup, run, or integrity failure
# stops the sequence before another YONOD task can begin.
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
repo_root="$(cd "${{script_dir}}/../../.." && pwd)"
cd "$repo_root"
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
mkdir -p logs

verify_task() {{
  python _verify/verify_ablation2_chemrxiv_unseen_amine_result.py --config "$1"
}}

# Verify all frozen plans and the completed full-input baseline before any
# remaining result/log/PID is created.
python _verify/verify_ablation2_chemrxiv_unseen_amine_plan.py >/dev/null
verify_task "config/{baseline}.yaml" >/dev/null

task_scripts=(
{launchers}
)

for launcher in "${{task_scripts[@]}}"; do
  task_name="$(basename "$launcher" .sh)"
  config_path="config/${{task_name}}.yaml"
  result_root="result/${{task_name}}"
  log_path="logs/${{task_name}}.log"
  pid_path="logs/${{task_name}}.pid"
  verification_path="logs/${{task_name}}.verification.json"

  if [[ ! -x "$launcher" || ! -f "$config_path" ]]; then
    echo "[batch:blocked] missing executable launcher or configuration for $task_name" >&2
    exit 2
  fi
  if [[ -e "$result_root" || -e "$log_path" || -e "$pid_path" || -e "$verification_path" ]]; then
    echo "[batch:blocked] refusing to overwrite existing evidence for $task_name" >&2
    exit 2
  fi

  echo "[batch:start] $task_name"
  nohup setsid bash "$launcher" </dev/null > "$log_path" 2>&1 &
  task_pid=$!
  printf '%s\\n' "$task_pid" > "$pid_path"

  # This waits on the direct child; it never polls and ensures the next task
  # cannot start while this one remains live.
  if wait "$task_pid"; then
    :
  else
    exit_status=$?
    echo "[batch:failed] $task_name exit=$exit_status; inspect $log_path" >&2
    exit "$exit_status"
  fi

  if ! verify_task "$config_path" > "$verification_path"; then
    echo "[batch:failed] $task_name completed with invalid/incomplete evidence" >&2
    exit 3
  fi
  echo "[batch:complete] $task_name"
done

echo "[batch:complete] {plan_id}: all {task_count} remaining tasks passed verification"
""".format(
        generator=Path(__file__).name,
        baseline=BASELINE,
        launchers=launchers,
        plan_id=PLAN_ID,
        task_count=len(REMAINING),
    )


def preflight() -> None:
    if len(REMAINING) != 11 or len(set(REMAINING)) != len(REMAINING):
        raise RuntimeError("the remaining plan must contain exactly eleven unique tasks")
    if BASELINE in REMAINING:
        raise RuntimeError("baseline must not appear in the remaining task plan")
    _assert_formal_config(BASELINE)
    _assert_baseline_completed()
    for task_name in REMAINING:
        _assert_formal_config(task_name)
        _assert_absent_evidence(task_name)


def generate(*, write: bool) -> list[Path]:
    preflight()
    task_paths = [TASK_DIR / f"{task_name}.sh" for task_name in REMAINING]
    for task_name, path in zip(REMAINING, task_paths):
        _render_or_validate(path, _task_launcher(task_name), executable=True, write=write)
    _render_or_validate(MASTER, _master_runner(task_paths), executable=True, write=write)
    return [*task_paths, MASTER]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the frozen launchers after read-only preflight")
    args = parser.parse_args()
    paths = generate(write=args.write)
    mode = "written" if args.write else "validated"
    print(
        "[unseen-amine-launchers] plan={0} remaining={1} mode={2} master={3}".format(
            PLAN_ID, len(REMAINING), mode, MASTER.relative_to(ROOT)
        )
    )
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
