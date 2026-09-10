#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_TASK_COUNT="275"
EXPECTED_COMBINATION_COUNT="11"
CPUSET="${YONOD_STEP29_CPUSET:-0-18}"
CPU_COUNT="19"
TASKS_PER_COMBO="25"
CONDA_BIN="${YONOD_CONDA_BIN:-/home/wangzh685/miniconda3/bin/conda}"
CONDA_ENV_NAME="${YONOD_CONDA_ENV:-yonod}"
DRY_RUN_PYTHON="${YONOD_DRY_RUN_PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scripts/run_static_descriptor_autogluon_matrix.py"
PLAN_DIR="${PROJECT_ROOT}/derived/descriptor_model_effect/step29_autogluon_static_rf_compare"
PLAN_JSON="${PLAN_DIR}/task_manifest.json"
PLAN_CSV="${PLAN_DIR}/task_manifest.csv"
LOCK_FILE="${PLAN_DIR}/step29_autogluon_19cpu.lock"

MODE="dry-run"
RESUME="0"
TOTAL_PLANNED_TASKS="0"
TOTAL_COMBO_COMMANDS="0"
FAILED_COMMANDS="0"

RUN_ENV=(
  "LOKY_MAX_CPU_COUNT=19"
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_static_descriptor_autogluon_19cpu_nohup.sh [--dry-run|--run] [--resume]

Modes:
  --dry-run  Verify the frozen 275-fold manifest and print exact formal commands. Default.
  --run      Execute the 275 AutoGluon folds serially through conda env "yonod".
  --resume   Keep using per-fold same-hash completion checks when --run is selected.

Formal scope:
BH1-static-3955: SOAP, PhysChem
BH2-static-3359: DFT, SOAP, PhysChem
SL1-static-1150: DFT, SOAP, PhysChem
SM-static-4620:  DFT, SOAP, PhysChem
USAGE
}

timestamp() {
  date "+%Y-%m-%d %H:%M:%S%z"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

quote_command() {
  local arg
  printf 'COMMAND:'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "required executable not found or not executable: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found on PATH: $1"
}

run_python_json_gate() {
  "${DRY_RUN_PYTHON}" - "$PLAN_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tasks = payload.get("tasks", [])
combos = {(row["population_id"], row["descriptor"]) for row in tasks}
bad = [
    row for row in tasks
    if row["model"] != "autogluon"
    or row["descriptor"] in {"MFP", "OHE"}
    or (row["dataset_id"] == "BH1" and row["descriptor"] == "DFT")
]
if len(tasks) != 275:
    raise SystemExit(f"task_count gate failed: {len(tasks)}")
if len(combos) != 11:
    raise SystemExit(f"combination_count gate failed: {len(combos)}")
if bad:
    raise SystemExit(f"forbidden task gate failed: {bad[0]}")
print(json.dumps({
    "status": "json_gate_passed",
    "task_count": len(tasks),
    "combination_count": len(combos),
    "plan_hash": payload["plan_hash"],
}, ensure_ascii=False, sort_keys=True))
PY
}

run_runner_dry_modes() {
  "${DRY_RUN_PYTHON}" "$RUNNER" --plan-dir "$PLAN_DIR" --verify-plan
  "${DRY_RUN_PYTHON}" "$RUNNER" --plan-dir "$PLAN_DIR" --dry-run \
    --conda-bin "$CONDA_BIN" --conda-env "$CONDA_ENV_NAME" --cpu-set "$CPUSET"
}

run_combo() {
  local population="$1"
  local descriptor="$2"
  local -a command=(
    taskset -c "$CPUSET"
    env "${RUN_ENV[@]}"
    "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python "$RUNNER"
    --plan-dir "$PLAN_DIR"
    --population-id "$population"
    --descriptor "$descriptor"
    --model autogluon
    --max-tasks "$TASKS_PER_COMBO"
  )
  TOTAL_COMBO_COMMANDS=$((TOTAL_COMBO_COMMANDS + 1))
  TOTAL_PLANNED_TASKS=$((TOTAL_PLANNED_TASKS + TASKS_PER_COMBO))
  quote_command "${command[@]}"
  if [[ "$MODE" == "run" ]]; then
    log "START population=${population} descriptor=${descriptor} expected_folds=${TASKS_PER_COMBO}"
    if "${command[@]}"; then
      log "FINISH population=${population} descriptor=${descriptor} status=ok"
    else
      FAILED_COMMANDS=$((FAILED_COMMANDS + 1))
      log "FINISH population=${population} descriptor=${descriptor} status=failed"
      return 1
    fi
  fi
}

run_all_combos() {
  run_combo "BH1-static-3955" "SOAP"
  run_combo "BH1-static-3955" "PhysChem"
  run_combo "BH2-static-3359" "DFT"
  run_combo "BH2-static-3359" "SOAP"
  run_combo "BH2-static-3359" "PhysChem"
  run_combo "SL1-static-1150" "DFT"
  run_combo "SL1-static-1150" "SOAP"
  run_combo "SL1-static-1150" "PhysChem"
  run_combo "SM-static-4620" "DFT"
  run_combo "SM-static-4620" "SOAP"
  run_combo "SM-static-4620" "PhysChem"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --run)
      MODE="run"
      ;;
    --resume)
      RESUME="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
  shift
done

cd "$PROJECT_ROOT"
require_file "$RUNNER"
require_file "$PLAN_JSON"
require_file "$PLAN_CSV"
require_command taskset
require_command env
if [[ "$MODE" == "run" ]]; then
  require_executable "$CONDA_BIN"
fi

log "STEP29_AUTOGUON_LAUNCHER mode=${MODE} resume=${RESUME} cpu_set=${CPUSET} cpu_count=${CPU_COUNT}"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "PLAN_DIR=${PLAN_DIR}"
run_python_json_gate
run_runner_dry_modes

if [[ "$MODE" == "run" ]]; then
  mkdir -p "$PLAN_DIR"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    die "another step29 AutoGluon run is active: $LOCK_FILE"
  fi
fi

run_all_combos

if [[ "$TOTAL_PLANNED_TASKS" != "$EXPECTED_TASK_COUNT" ]]; then
  die "planned task count ${TOTAL_PLANNED_TASKS} != ${EXPECTED_TASK_COUNT}"
fi
if [[ "$TOTAL_COMBO_COMMANDS" != "$EXPECTED_COMBINATION_COUNT" ]]; then
  die "combo command count ${TOTAL_COMBO_COMMANDS} != ${EXPECTED_COMBINATION_COUNT}"
fi
if [[ "$FAILED_COMMANDS" != "0" ]]; then
  die "failed combo commands: ${FAILED_COMMANDS}"
fi

if [[ "$MODE" == "run" ]]; then
  "${CONDA_BIN}" run -n "$CONDA_ENV_NAME" python "$RUNNER" \
    --plan-dir "$PLAN_DIR" --completion-summary --require-complete
fi

log "SCRIPT_SUCCESS mode=${MODE} planned_tasks=${TOTAL_PLANNED_TASKS} combo_commands=${TOTAL_COMBO_COMMANDS} cpu_set=${CPUSET} cpu_count=${CPU_COUNT}"
