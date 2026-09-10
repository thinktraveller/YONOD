#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_PLAN_HASH="71d297cb894f13851e61a8e9aa6341148d70166cc51a40ebc7e6086eae16b599"
EXPECTED_TASK_COUNT="1500"
EXPECTED_COMBINATION_COUNT="60"
EXPECTED_DESCRIPTOR_MATRIX_COUNT="15"
EXPECTED_POPULATION_COUNT="4"
EXPECTED_INITIAL_PENDING_NON_SVM_TASKS="250"
EXPECTED_RESUME_TASKS="250"
EXPECTED_RESUME_COMBINATION_COUNT="10"
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
  bash scripts/run_static_descriptor_model_matrix_19cpu_resume_no_svm.sh [--dry-run|--run]

Modes:
  --dry-run  Validate the frozen plan and print the exact non-SVM resume commands. This is the default.
  --run      Resume only the unfinished RF/XGBoost/LightGBM tasks through conda env "yonod".

This launcher is intentionally narrower than the full matrix launcher:
- It never selects model=svm.
- It only targets the expected 250 unfinished non-SVM folds in SM-static-4620.
- It relies on the existing runner completion/hash checks to skip already complete folds.
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

validate_summary_file() {
  local path="$1"
  local label="$2"
  require_file "$path"
  json_has_string "$path" "plan_hash" "$EXPECTED_PLAN_HASH" || die "${label} plan_hash is not the frozen hash"
  json_has_number "$path" "task_count" "$EXPECTED_TASK_COUNT" || die "${label} task_count is not ${EXPECTED_TASK_COUNT}"
  json_has_number "$path" "combination_count" "$EXPECTED_COMBINATION_COUNT" || die "${label} combination_count is not ${EXPECTED_COMBINATION_COUNT}"
  json_has_number "$path" "included_descriptor_matrix_count" "$EXPECTED_DESCRIPTOR_MATRIX_COUNT" || die "${label} descriptor matrix count is not ${EXPECTED_DESCRIPTOR_MATRIX_COUNT}"
  json_has_number "$path" "population_count" "$EXPECTED_POPULATION_COUNT" || die "${label} population_count is not ${EXPECTED_POPULATION_COUNT}"
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

status_check_code='import collections, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_plan_hash = sys.argv[2]
mode = sys.argv[3]
expected_initial_pending = int(sys.argv[4])
plan_path = root / "derived/descriptor_model_effect/step28_3_matrix_plan/task_plan.json"
payload = json.loads(plan_path.read_text(encoding="utf-8"))
tasks = payload.get("tasks", [])
expected_combos = {
    ("SM-static-4620", "DFT", "lightgbm"),
    ("SM-static-4620", "SOAP", "rf"),
    ("SM-static-4620", "SOAP", "xgboost"),
    ("SM-static-4620", "SOAP", "lightgbm"),
    ("SM-static-4620", "PhysChem", "rf"),
    ("SM-static-4620", "PhysChem", "xgboost"),
    ("SM-static-4620", "PhysChem", "lightgbm"),
    ("SM-static-4620", "MFP", "rf"),
    ("SM-static-4620", "MFP", "xgboost"),
    ("SM-static-4620", "MFP", "lightgbm"),
}
errors = []
if payload.get("plan_hash") != expected_plan_hash:
    errors.append("plan_hash drift")
if len(tasks) != 1500 or payload.get("task_count") != 1500:
    errors.append("task_count drift")
model_counts = collections.Counter(task.get("model") for task in tasks)
expected_model_counts = collections.Counter({"rf": 375, "xgboost": 375, "svm": 375, "lightgbm": 375})
if model_counts != expected_model_counts:
    errors.append("model count drift")
if len({task.get("task_id") for task in tasks}) != 1500:
    errors.append("task_id uniqueness drift")
complete = []
incomplete = []
bad_completion = []
for task in tasks:
    completion_path = root / task["output_path"] / "completion.json"
    ok = False
    if completion_path.is_file():
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except Exception as exc:
            bad_completion.append({"task_id": task["task_id"], "error": type(exc).__name__})
        else:
            ok = (
                completion.get("status") == "complete"
                and completion.get("task_id") == task["task_id"]
                and completion.get("task_contract_hash") == task["task_contract_hash"]
                and completion.get("plan_hash") == payload.get("plan_hash")
            )
    if ok:
        complete.append(task)
    else:
        incomplete.append(task)
pending_non_svm = [task for task in incomplete if task.get("model") != "svm"]
pending_svm = [task for task in incomplete if task.get("model") == "svm"]
expected_resume_tasks = [
    task for task in tasks
    if (task["population_id"], task["descriptor"], task["model"]) in expected_combos
]
unexpected_non_svm = [
    task for task in pending_non_svm
    if (task["population_id"], task["descriptor"], task["model"]) not in expected_combos
]
if len(expected_resume_tasks) != 250:
    errors.append("expected resume task set is not 250")
if bad_completion:
    errors.append("bad completion json present")
if unexpected_non_svm:
    errors.append("unexpected pending non-SVM tasks present")
if len(pending_non_svm) > expected_initial_pending:
    errors.append("pending non-SVM task count exceeds expected initial pending count")
if mode == "post" and pending_non_svm:
    errors.append("post-run pending non-SVM tasks remain")
combo_counter = collections.Counter((task["population_id"], task["descriptor"], task["model"]) for task in pending_non_svm)
summary = {
    "mode": mode,
    "status": "passed" if not errors else "failed",
    "plan_hash": payload.get("plan_hash"),
    "task_count": len(tasks),
    "complete_same_hash_tasks": len(complete),
    "incomplete_or_missing_tasks": len(incomplete),
    "complete_by_model": dict(sorted(collections.Counter(task["model"] for task in complete).items())),
    "incomplete_by_model": dict(sorted(collections.Counter(task["model"] for task in incomplete).items())),
    "expected_initial_pending_non_svm_tasks": expected_initial_pending,
    "current_pending_non_svm_tasks": len(pending_non_svm),
    "current_pending_svm_tasks": len(pending_svm),
    "expected_resume_task_set_size": len(expected_resume_tasks),
    "pending_non_svm_by_combo": {"/".join(key): value for key, value in sorted(combo_counter.items())},
    "unexpected_pending_non_svm_count": len(unexpected_non_svm),
    "bad_completion_json_count": len(bad_completion),
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
if errors:
    print(json.dumps({"errors": errors, "first_unexpected_non_svm": [task["task_id"] for task in unexpected_non_svm[:5]], "first_bad_completion": bad_completion[:5]}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
'

run_status_check() {
  local mode="$1"
  log "STATUS_CHECK_START mode=${mode}"
  python3 -c "$status_check_code" "$PROJECT_ROOT" "$EXPECTED_PLAN_HASH" "$mode" "$EXPECTED_INITIAL_PENDING_NON_SVM_TASKS"
  log "STATUS_CHECK_END mode=${mode}"
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
    die "command failed: ${label}"
  fi
  return 0
}

run_environment_snapshot() {
  run_command "conda-list-${CONDA_ENV_NAME}" "$CONDA_BIN" list -n "$CONDA_ENV_NAME"
}

run_combo() {
  local population="$1"
  local descriptor="$2"
  local model="$3"
  local label="population=${population} descriptor=${descriptor} model=${model}"
  if [[ "$model" == "svm" ]]; then
    die "internal error: this resume launcher must never select SVM"
  fi
  TOTAL_COMBO_COMMANDS=$((TOTAL_COMBO_COMMANDS + 1))
  TOTAL_PLANNED_TASKS=$((TOTAL_PLANNED_TASKS + TASKS_PER_COMBO))
  run_command "$label" \
    taskset -c "$CPUSET" env "${RUN_ENV[@]}" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" \
    python "$RUNNER" --plan-dir "$PLAN_DIR" \
    --population-id "$population" --descriptor "$descriptor" --model "$model" \
    --max-tasks "$TASKS_PER_COMBO"
}

run_resume_matrix() {
  local model
  local descriptor
  run_combo "SM-static-4620" "DFT" "lightgbm"
  for descriptor in SOAP PhysChem MFP; do
    for model in rf xgboost lightgbm; do
      run_combo "SM-static-4620" "$descriptor" "$model"
    done
  done
}

acquire_lock() {
  require_command flock
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    die "another step28.3 matrix launcher already holds ${LOCK_FILE}"
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
require_command python3
require_command taskset
require_executable "$CONDA_BIN"
validate_plan_gate

log "SCRIPT_START mode=${MODE} project_root=${PROJECT_ROOT}"
log "PLAN_GATE_OK plan_hash=${EXPECTED_PLAN_HASH} task_count=${EXPECTED_TASK_COUNT} resume_tasks=${EXPECTED_RESUME_TASKS} resume_combinations=${EXPECTED_RESUME_COMBINATION_COUNT} cpu_set=${CPUSET} cpu_count=${CPU_COUNT}"
log "SVM_POLICY excluded=true retry_svm=false"
if git_head="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null)"; then
  log "GIT_COMMIT ${git_head}"
else
  log "GIT_COMMIT unavailable"
fi

run_status_check "pre"

if [[ "$MODE" == "run" ]]; then
  acquire_lock
fi

run_environment_snapshot
run_resume_matrix

if [[ "$TOTAL_PLANNED_TASKS" != "$EXPECTED_RESUME_TASKS" ]]; then
  die "internal launcher task count mismatch: ${TOTAL_PLANNED_TASKS}"
fi
if [[ "$TOTAL_COMBO_COMMANDS" != "$EXPECTED_RESUME_COMBINATION_COUNT" ]]; then
  die "internal launcher combo command count mismatch: ${TOTAL_COMBO_COMMANDS}"
fi
if [[ "$FAILED_COMMANDS" != "0" ]]; then
  die "one or more commands failed; see the nohup log above"
fi

if [[ "$MODE" == "run" ]]; then
  run_status_check "post"
fi

log "SCRIPT_SUCCESS mode=${MODE} planned_tasks=${TOTAL_PLANNED_TASKS} combo_commands=${TOTAL_COMBO_COMMANDS} cpu_set=${CPUSET} cpu_count=${CPU_COUNT} svm_excluded=true"
