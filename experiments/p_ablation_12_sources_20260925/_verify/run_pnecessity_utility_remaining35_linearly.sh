#!/usr/bin/env bash
# Bash/Linux only.  This launcher does not run when it is generated.
#
# When invoked, it validates the frozen 36-item plan and the completed random
# pilot, then starts exactly one of the remaining 35 `yonod.py` tasks at a
# time.  It waits for that isolated headless task, runs a read-only integrity
# check, and only then moves on.  A launch, task, hash, split, prediction, or
# report failure stops the queue before any later task starts.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pilot_task="pnecessity_utility_ord_ahneman_morgan_rf_minus_p_v1"
pilot_config="config/${pilot_task}.yaml"
cpu_list="0-18"
dry_run=false

if [[ $# -gt 1 ]] || [[ $# -eq 1 && $1 != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  dry_run=true
fi

source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
mkdir -p logs

# The configured range contains exactly 19 logical CPUs.  This also fails
# early if a later shell has a different CPU-affinity allowance.
taskset --cpu-list "$cpu_list" true >/dev/null

python _verify/verify_pnecessity_utility_plan.py >/dev/null
python _verify/verify_pnecessity_utility_result.py --config "$pilot_config" >/dev/null

mapfile -t queue < <(
  python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
pilot = "pnecessity_utility_ord_ahneman_morgan_rf_minus_p_v1"
payload = json.loads((root / "result/pnecessity_utility_prepare_v1/generated_config_manifest.json").read_text(encoding="utf-8"))
entries = [entry for entry in payload["generated"] if entry["task_name"] != pilot]
if len(payload["generated"]) != 36 or len(entries) != 35:
    raise SystemExit("expected exactly 36 generated tasks and 35 remaining tasks")
if len({entry["task_name"] for entry in entries}) != 35:
    raise SystemExit("remaining task names are not unique")
for entry in entries:
    path = root / entry["config_path"]
    if not path.is_file():
        raise SystemExit(f"missing queued configuration: {path}")
    print(f'{entry["config_path"]}\t{entry["task_name"]}')
PY
)

if [[ ${#queue[@]} -ne 35 ]]; then
  echo "expected 35 remaining tasks, found ${#queue[@]}" >&2
  exit 1
fi

for index in "${!queue[@]}"; do
  IFS=$'\t' read -r config_path task_name <<< "${queue[$index]}"
  result_root="result/${task_name}"
  log_path="logs/${task_name}.log"
  pid_path="logs/${task_name}.pid"

  if [[ -e "$result_root" ]]; then
    # A rerun may only skip a task with complete, identity-matched evidence.
    # A partial or corrupted root fails the verifier and stops the queue.
    python _verify/verify_pnecessity_utility_result.py --config "$config_path" >/dev/null
    printf '[%d/35] verified complete, skipping %s\n' "$((index + 1))" "$task_name"
    continue
  fi
  if [[ -e "$log_path" || -e "$pid_path" ]]; then
    echo "[$((index + 1))/35] stale log or PID without a completed result: $task_name" >&2
    exit 1
  fi
  if [[ "$dry_run" == true ]]; then
    printf '[%d/35] preflight passed: %s\n' "$((index + 1))" "$task_name"
    continue
  fi

  printf '[%d/35] launching %s\n' "$((index + 1))" "$task_name"
  # --wait makes the background PID represent the task lifetime even if
  # setsid must fork because the invoking shell has job control enabled.
  nohup setsid --wait bash -lc '
    source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
    conda activate yonod
    printf "%s\n" "$1" | taskset --cpu-list "$2" python -u yonod.py
  ' bash "$config_path" "$cpu_list" </dev/null > "$log_path" 2>&1 &
  task_pid=$!
  printf '%s\n' "$task_pid" > "$pid_path"

  if ! wait "$task_pid"; then
    echo "[$((index + 1))/35] yonod.py exited unsuccessfully: $task_name" >&2
    exit 1
  fi
  python _verify/verify_pnecessity_utility_result.py --config "$config_path" >/dev/null
  printf '[%d/35] verified complete: %s\n' "$((index + 1))" "$task_name"
done

if [[ "$dry_run" == true ]]; then
  echo "dry run passed: 35 remaining tasks are launchable; no model was started"
else
  echo "all 35 remaining product-utility tasks verified complete"
fi
