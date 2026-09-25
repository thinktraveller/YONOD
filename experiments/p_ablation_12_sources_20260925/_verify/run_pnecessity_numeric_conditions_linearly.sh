#!/usr/bin/env bash
# Bash/Linux only.  A single queue process starts exactly one YONOD task at a
# time; it never launches concurrent descriptors, models, or folds.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

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
taskset --cpu-list "$cpu_list" true >/dev/null

python _verify/verify_pnecessity_numeric_conditions_plan.py >/dev/null

mapfile -t queue < <(
  python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
payload = json.loads((root / "result/pnecessity_numeric_conditions_prepare_v1/generated_config_manifest.json").read_text(encoding="utf-8"))
order = ("ord_roche_borylation", "ord_nicolit", "ord_shields", "ord_c_n")
lines = ("morgan_rf", "mfp_lightgbm")
arms = ("full", "minus_p")
by_key = {(item["source_id"], item["line_id"], item["arm"]): item for item in payload["generated"]}
if len(by_key) != 16:
    raise SystemExit("expected exactly 16 unique generated tasks")
for source in order:
    for line in lines:
        for arm in arms:
            item = by_key.get((source, line, arm))
            if item is None:
                raise SystemExit(f"queue entry missing: {source}/{line}/{arm}")
            path = root / item["config_path"]
            if not path.is_file():
                raise SystemExit(f"queued YAML is missing: {path}")
            print(f'{item["config_path"]}\t{item["task_name"]}')
PY
)

if [[ ${#queue[@]} -ne 16 ]]; then
  echo "expected 16 queued tasks, found ${#queue[@]}" >&2
  exit 1
fi

for index in "${!queue[@]}"; do
  IFS=$'\t' read -r config_path task_name <<< "${queue[$index]}"
  result_root="result/${task_name}"
  log_path="logs/${task_name}.log"
  pid_path="logs/${task_name}.pid"
  position="$((index + 1))/16"

  if [[ -e "$result_root" ]]; then
    python _verify/verify_pnecessity_numeric_conditions_result.py --config "$config_path" >/dev/null
    printf '[%s] verified complete, skipping %s\n' "$position" "$task_name"
    continue
  fi
  if [[ -e "$log_path" || -e "$pid_path" ]]; then
    echo "[$position] stale log or PID without a verified complete result: $task_name" >&2
    exit 1
  fi
  if [[ "$dry_run" == true ]]; then
    printf '[%s] preflight passed: %s\n' "$position" "$task_name"
    continue
  fi

  printf '[%s] launching %s\n' "$position" "$task_name"
  nohup setsid --wait bash -lc '
    source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
    conda activate yonod
    printf "%s\n" "$1" | taskset --cpu-list "$2" python -u yonod.py
  ' bash "$config_path" "$cpu_list" </dev/null > "$log_path" 2>&1 &
  task_pid=$!
  printf '%s\n' "$task_pid" > "$pid_path"
  if ! wait "$task_pid"; then
    echo "[$position] yonod.py exited unsuccessfully: $task_name" >&2
    exit 1
  fi
  python _verify/verify_pnecessity_numeric_conditions_result.py --config "$config_path" >/dev/null
  printf '[%s] verified complete: %s\n' "$position" "$task_name"
done

if [[ "$dry_run" == true ]]; then
  echo "dry run passed: all 16 numeric-condition tasks are launchable; no model started"
else
  echo "all 16 numeric-condition product-utility tasks verified complete"
fi
