#!/usr/bin/env bash
# Linux/Bash queue: eight standalone schema-2 tasks, one YONOD run at a time.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cpu_list="0-18"
dry_run=false
selected_index=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --task)
      if [[ $# -lt 2 || ! $2 =~ ^[1-8]$ || $selected_index -ne 0 ]]; then
        echo "--task requires one index from 1 to 8" >&2
        exit 2
      fi
      selected_index="$2"
      shift 2
      ;;
    *)
      echo "usage: bash _verify/run_pnecessity_nonstandard_linearly.sh [--dry-run] [--task 1-8]" >&2
      exit 2
      ;;
  esac
done

source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
mkdir -p logs
taskset --cpu-list "$cpu_list" true >/dev/null
python _verify/verify_pnecessity_nonstandard_plan.py

mapfile -t queue < <(
  python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
payload = json.loads((root / "result/pnecessity_nonstandard_prepare_v1/generated_config_manifest.json").read_text(encoding="utf-8"))
for item in payload["queue"]:
    print(f'{item["config_path"]}\t{item["task_name"]}')
PY
)
if [[ ${#queue[@]} -ne 8 ]]; then
  echo "expected exactly eight queued tasks" >&2
  exit 1
fi

for index in "${!queue[@]}"; do
  IFS=$'\t' read -r config_path task_name <<< "${queue[$index]}"
  result_root="result/${task_name}"
  log_path="logs/${task_name}.log"
  pid_path="logs/${task_name}.pid"
  position="$((index + 1))/8"

  if [[ $selected_index -gt 0 && $((index + 1)) -gt $selected_index ]]; then
    break
  fi
  if [[ $selected_index -gt 0 && $((index + 1)) -lt $selected_index ]]; then
    python _verify/verify_pnecessity_nonstandard_result.py --config "$config_path" >/dev/null
    continue
  fi

  if [[ -e "$result_root" ]]; then
    python _verify/verify_pnecessity_nonstandard_result.py --config "$config_path" >/dev/null
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
  python _verify/verify_pnecessity_nonstandard_result.py --config "$config_path" >/dev/null
  printf '[%s] verified complete: %s\n' "$position" "$task_name"
done

if [[ "$dry_run" == true ]]; then
  echo "dry run passed: selected task(s) are launchable; no model started"
elif [[ $selected_index -gt 0 ]]; then
  echo "selected task $selected_index verified complete"
else
  echo "all eight nonstandard-source P-utility tasks verified complete"
fi
