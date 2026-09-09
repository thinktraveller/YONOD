#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_PLAN_HASH="71d297cb894f13851e61a8e9aa6341148d70166cc51a40ebc7e6086eae16b599"
EXPECTED_TASK_COUNT="1500"
EXPECTED_COMBINATION_COUNT="60"
EXPECTED_DESCRIPTOR_MATRIX_COUNT="15"
EXPECTED_POPULATION_COUNT="4"
CPUSET="0-18"
CPU_COUNT="19"
TASKS_PER_COMBO="25"
CONDA_BIN="/home/wangzh685/miniconda3/bin/conda"
CONDA_ENV_NAME="yonod"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scripts/run_static_descriptor_model_matrix.py"
PLAN_DIR="${PROJECT_ROOT}/derived/descriptor_model_effect/step28_3_matrix_plan"
PLAN_JSON="${PLAN_DIR}/task_plan.json"
PLAN_CSV="${PLAN_DIR}/task_plan.csv"
PREFLIGHT_JSON="${PLAN_DIR}/preflight_summary.json"
VERIFICATION_JSON="${PLAN_DIR}/verification_summary.json"
LOCK_FILE="${PLAN_DIR}/full_matrix_19cpu.lock"

MODE="dry-run"
CONTINUE_ON_FAILURE="0"
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

MODELS=(rf xgboost svm lightgbm)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_static_descriptor_model_matrix_19cpu_nohup.sh [--dry-run|--run] [--continue-on-failure]

Modes:
  --dry-run  Print the exact verification and population/model commands. This is the default.
  --run      Execute the frozen 1500-task matrix serially through conda env "yonod".

The script expects the frozen step 28.3 plan hash
71d297cb894f13851e61a8e9aa6341148d70166cc51a40ebc7e6086eae16b599
and runs only the main static descriptor matrix:
BH1-static-3955: SOAP, PhysChem, MFP
BH2-static-3359: DFT, SOAP, PhysChem, MFP
SL1-static-1150: DFT, SOAP, PhysChem, MFP
SM-static-4620:  DFT, SOAP, PhysChem, MFP
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

json_has_string() {
  local path="$1"
  local key="$2"
  local value="$3"
  grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*\"${value}\"" "$path"
}

json_has_number() {
  local path="$1"
  local key="$2"
  local value="$3"
  grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*${value}" "$path"
}

json_has_false() {
  local path="$1"
  local key="$2"
  grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*false" "$path"
}

validate_summary_file() {
  local path="$1"
  local label="$2"
  require_file "$path"
  json_has_string "$path" "plan_hash" "$EXPECTED_PLAN_HASH" || die "${label} plan_hash is not the frozen hash"
  json_has_number "$path" "task_count" "$EXPECTED_TASK_COUNT" || die "${label} task_count is not ${EXPECTED_TASK_COUNT}"
  json_has_number "$path" "combination_count" "$EXPECTED_COMBINATION_COUNT" || die "${label} combination_count is not ${EXPECTED_COMBINATION_COUNT}"
  json_has_number "$path" "included_descriptor_matrix_count" "$EXPECTED_DESCRIPTOR_MATRIX_COUNT" || die "${label} descriptor matrix count is not ${EXPECTED_DESCRIPTOR_MATRIX_COUNT}"
  json_has_number "$path" "population_count" "$EXPECTED_POPULATION_COUNT" || die "${label} population_count is not ${EXPECTED_POPULATION_COUNT}"
  json_has_false "$path" "full_training_started" || die "${label} full_training_started is not false"
}

validate_plan_gate() {
  require_file "$RUNNER"
  require_file "$PLAN_JSON"
  require_file "$PLAN_CSV"
  validate_summary_file "$PREFLIGHT_JSON" "preflight_summary"
  validate_summary_file "$VERIFICATION_JSON" "verification_summary"
  json_has_string "$PLAN_JSON" "plan_hash" "$EXPECTED_PLAN_HASH" || die "task_plan.json plan_hash is not the frozen hash"
  json_has_number "$PLAN_JSON" "task_count" "$EXPECTED_TASK_COUNT" || die "task_plan.json task_count is not ${EXPECTED_TASK_COUNT}"
}

run_command() {
  local label="$1"
  shift
  log "COMMAND_START label=${label}"
  quote_command "$@"
  if [[ "$MODE" == "dry-run" ]]; then
    log "COMMAND_DRY_RUN label=${label}"
    return 0
  fi

  set +e
  "$@"
  local rc=$?
  set -e
  log "COMMAND_END label=${label} exit_code=${rc}"
  if [[ "$rc" != "0" ]]; then
    FAILED_COMMANDS=$((FAILED_COMMANDS + 1))
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      die "command failed: ${label}"
    fi
  fi
  return 0
}

runner_command() {
  taskset -c "$CPUSET" env "${RUN_ENV[@]}" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python "$RUNNER" "$@"
}

run_verification() {
  run_command "verify-plan" \
    taskset -c "$CPUSET" env "${RUN_ENV[@]}" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python "$RUNNER" --plan-dir "$PLAN_DIR" --verify-plan
  if [[ "$MODE" == "run" ]]; then
    validate_summary_file "$VERIFICATION_JSON" "verification_summary_after_verify_plan"
  fi
}

run_environment_snapshot() {
  run_command "conda-list-${CONDA_ENV_NAME}" "$CONDA_BIN" list -n "$CONDA_ENV_NAME"
}

run_combo() {
  local population="$1"
  local descriptor="$2"
  local model="$3"
  local label="population=${population} descriptor=${descriptor} model=${model}"
  TOTAL_COMBO_COMMANDS=$((TOTAL_COMBO_COMMANDS + 1))
  TOTAL_PLANNED_TASKS=$((TOTAL_PLANNED_TASKS + TASKS_PER_COMBO))
  run_command "$label" \
    taskset -c "$CPUSET" env "${RUN_ENV[@]}" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python "$RUNNER" --plan-dir "$PLAN_DIR" \
    --population-id "$population" --descriptor "$descriptor" --model "$model" \
    --max-tasks "$TASKS_PER_COMBO"
}

completion_check_code='import json, pathlib, sys
plan_path = pathlib.Path(sys.argv[1])
population_id = sys.argv[2]
expected = int(sys.argv[3])
payload = json.loads(plan_path.read_text(encoding="utf-8"))
tasks = [task for task in payload["tasks"] if task["population_id"] == population_id]
complete = 0
missing = []
for task in tasks:
    completion_path = pathlib.Path(task["output_path"]) / "completion.json"
    if not completion_path.is_file():
        missing.append(task["task_id"])
        continue
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    ok = (
        completion.get("status") == "complete"
        and completion.get("task_id") == task["task_id"]
        and completion.get("task_contract_hash") == task["task_contract_hash"]
        and completion.get("plan_hash") == payload["plan_hash"]
    )
    if ok:
        complete += 1
    else:
        missing.append(task["task_id"])
summary = {
    "status": "passed" if len(tasks) == expected and complete == expected else "failed",
    "population_id": population_id,
    "expected_tasks": expected,
    "planned_tasks": len(tasks),
    "complete_same_hash_tasks": complete,
    "missing_or_mismatched_tasks": len(missing),
    "first_missing_or_mismatched": missing[:5],
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if summary["status"] == "passed" else 1)'

run_population_completion_check() {
  local population="$1"
  local expected="$2"
  if [[ "$MODE" == "dry-run" ]]; then
    log "COMPLETION_CHECK_DRY_RUN population=${population} expected_tasks=${expected}"
    return 0
  fi
  run_command "completion-check-${population}" \
    taskset -c "$CPUSET" env "${RUN_ENV[@]}" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python -c "$completion_check_code" "$PLAN_JSON" "$population" "$expected"
}

run_population() {
  local population="$1"
  local expected_tasks="$2"
  shift 2
  local descriptors=("$@")
  local descriptor
  local model
  local started_epoch
  local ended_epoch
  started_epoch="$(date +%s)"
  log "POPULATION_START population=${population} expected_tasks=${expected_tasks} descriptors=${descriptors[*]}"
  for descriptor in "${descriptors[@]}"; do
    for model in "${MODELS[@]}"; do
      run_combo "$population" "$descriptor" "$model"
    done
  done
  run_population_completion_check "$population" "$expected_tasks"
  ended_epoch="$(date +%s)"
  log "POPULATION_END population=${population} exit_failures=${FAILED_COMMANDS} elapsed_seconds=$((ended_epoch - started_epoch))"
}

acquire_lock() {
  require_command flock
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    die "another step28.3 full matrix launcher already holds ${LOCK_FILE}"
  fi
  log "LOCK_ACQUIRED path=${LOCK_FILE}"
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      MODE="dry-run"
      ;;
    --run)
      MODE="run"
      ;;
    --continue-on-failure)
      CONTINUE_ON_FAILURE="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $arg"
      ;;
  esac
done

trap 'rc=$?; log "SCRIPT_END mode=${MODE} exit_code=${rc} planned_tasks=${TOTAL_PLANNED_TASKS} combo_commands=${TOTAL_COMBO_COMMANDS} failed_commands=${FAILED_COMMANDS}"' EXIT

cd "$PROJECT_ROOT"
require_command grep
require_command date
require_command taskset
require_executable "$CONDA_BIN"
validate_plan_gate

log "SCRIPT_START mode=${MODE} project_root=${PROJECT_ROOT}"
log "PLAN_GATE_OK plan_hash=${EXPECTED_PLAN_HASH} task_count=${EXPECTED_TASK_COUNT} populations=${EXPECTED_POPULATION_COUNT} cpu_set=${CPUSET} cpu_count=${CPU_COUNT}"
log "FAILURE_POLICY continue_on_failure=${CONTINUE_ON_FAILURE}"
if git_head="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null)"; then
  log "GIT_COMMIT ${git_head}"
else
  log "GIT_COMMIT unavailable"
fi

if [[ "$MODE" == "run" ]]; then
  acquire_lock
fi

run_verification
run_environment_snapshot
run_population "BH1-static-3955" "300" SOAP PhysChem MFP
run_population "BH2-static-3359" "400" DFT SOAP PhysChem MFP
run_population "SL1-static-1150" "400" DFT SOAP PhysChem MFP
run_population "SM-static-4620" "400" DFT SOAP PhysChem MFP
run_verification

if [[ "$TOTAL_PLANNED_TASKS" != "$EXPECTED_TASK_COUNT" ]]; then
  die "internal launcher task count mismatch: ${TOTAL_PLANNED_TASKS}"
fi
if [[ "$TOTAL_COMBO_COMMANDS" != "$EXPECTED_COMBINATION_COUNT" ]]; then
  die "internal launcher combo command count mismatch: ${TOTAL_COMBO_COMMANDS}"
fi
if [[ "$FAILED_COMMANDS" != "0" ]]; then
  die "one or more commands failed; see the nohup log above"
fi

log "SCRIPT_SUCCESS mode=${MODE} planned_tasks=${TOTAL_PLANNED_TASKS} combo_commands=${TOTAL_COMBO_COMMANDS} cpu_set=${CPUSET} cpu_count=${CPU_COUNT}"
