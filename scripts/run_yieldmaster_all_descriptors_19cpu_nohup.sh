#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

YONOD_CONDA_BIN="${YONOD_CONDA_BIN:-/home/wangzh685/miniconda3/bin/conda}"
YONOD_CONDA_ENV="${YONOD_CONDA_ENV:-yonod}"
YONOD_CPU_SET="${YONOD_CPU_SET:-0-18}"
YONOD_CPU_COUNT=19
RUN_NAME="yieldmaster_all_descriptors_all_models_19cpu"
LOG_DIR="${PROJECT_ROOT}/logs"
LATEST_PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

DATASET_IDS=(bh1 bh2 sl1 sm)
CONFIG_PATHS=(
  configs/yieldmaster_bh1_all_descriptors_all_models/yieldmaster_bh1_all_descriptors_all_models.json
  configs/yieldmaster_bh2_all_descriptors_all_models/yieldmaster_bh2_all_descriptors_all_models.json
  configs/yieldmaster_sl1_all_descriptors_all_models/yieldmaster_sl1_all_descriptors_all_models.json
  configs/yieldmaster_sm_all_descriptors_all_models/yieldmaster_sm_all_descriptors_all_models.json
)
CSV_PATHS=(
  dataset/yieldmaster/BH1.csv
  dataset/yieldmaster/BH2.csv
  dataset/yieldmaster/SL1.csv
  dataset/yieldmaster/SM.csv
)

export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT="${YONOD_CPU_COUNT}"
export OMP_NUM_THREADS="${YONOD_CPU_COUNT}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

quote_command() {
  local rendered
  printf -v rendered '%q ' "$@"
  printf '%s' "${rendered% }"
}

preflight() {
  [[ -x "${YONOD_CONDA_BIN}" ]] || {
    printf 'ERROR: conda 不可执行：%s\n' "${YONOD_CONDA_BIN}" >&2
    return 2
  }
  command -v taskset >/dev/null 2>&1 || {
    printf 'ERROR: 未找到 taskset，无法限制到 19 个 CPU。\n' >&2
    return 2
  }
  taskset -c "${YONOD_CPU_SET}" true >/dev/null 2>&1 || {
    printf 'ERROR: 当前系统不能使用 CPU 集合 %s。\n' "${YONOD_CPU_SET}" >&2
    return 2
  }
  local required_tool
  for required_tool in nohup setsid tee; do
    command -v "${required_tool}" >/dev/null 2>&1 || return 2
  done

  local index
  for index in "${!DATASET_IDS[@]}"; do
    [[ -f "${PROJECT_ROOT}/${CONFIG_PATHS[$index]}" ]] || {
      printf 'ERROR: 配置不存在：%s\n' "${CONFIG_PATHS[$index]}" >&2
      return 2
    }
    [[ -f "${PROJECT_ROOT}/${CSV_PATHS[$index]}" ]] || {
      printf 'ERROR: 数据集不存在：%s\n' "${CSV_PATHS[$index]}" >&2
      return 2
    }
  done

  taskset -c "${YONOD_CPU_SET}" "${YONOD_CONDA_BIN}" run --no-capture-output \
    -n "${YONOD_CONDA_ENV}" python -c \
    'import os, sys; import autogluon, lightgbm, rdkit, sklearn, torch, xgboost; cpus=sorted(os.sched_getaffinity(0)); assert len(cpus)==int(sys.argv[1]), ("需要恰好19个可用逻辑CPU", cpus); print("依赖预检通过，CPU:", cpus)' \
    "${YONOD_CPU_COUNT}"
}

make_runtime_config() {
  local source_config="$1"
  local runtime_config="$2"
  local csv_path="$3"

  # 默认 conda 捕获输入时 python - 读不到 heredoc，却可能返回成功。
  # run_one 位于 if 条件中，必须显式传播失败，不能依赖 set -e。
  if "${YONOD_CONDA_BIN}" run --no-capture-output -n "${YONOD_CONDA_ENV}" python - \
    "${source_config}" "${runtime_config}" "${YONOD_CPU_COUNT}" \
    "${YONOD_CPU_SET}" "${csv_path}" <<'PY'
import json
import sys
from pathlib import Path
from main import _model_params_for, config_to_args, load_config_from_json

source = Path(sys.argv[1])
target = Path(sys.argv[2])
cpu_count = int(sys.argv[3])
config = json.loads(source.read_text(encoding="utf-8"))
model_params = config.setdefault("model_params", {})
autogluon_params = _model_params_for(config, "autogluon")
autogluon_params.setdefault("time_limit", 300)
autogluon_params.setdefault("presets", "medium_quality")
autogluon_params["num_cpus"] = cpu_count
model_params["autogluon"] = autogluon_params
runtime = config.setdefault("runtime", {})
runtime["source_config"] = str(source)
runtime["cpu_affinity"] = sys.argv[4]
runtime["cpu_count"] = cpu_count
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
args = config_to_args(load_config_from_json(target, Path(sys.argv[5])), target)
if args.autogluon_num_cpus != cpu_count:
    raise ValueError("AutoGluon CPU 参数未生效")
if args.output_dir != target.parent.parent:
    raise ValueError("运行时配置必须位于输出根目录的 docs/ 下")
print(f"[config:validated] {target}; features={len(args._feature_specs)}; models={len(args.models)}; AutoGluon.num_cpus={args.autogluon_num_cpus}", flush=True)
PY
  then
    [[ -s "${runtime_config}" ]] || {
      log "ERROR 运行时配置未生成或为空：${runtime_config}" >&2
      return 2
    }
  else
    local status=$?
    log "ERROR 运行时配置生成/加载失败：${runtime_config} exit_code=${status}" >&2
    return "${status}"
  fi
}

run_one() {
  local dataset_id="$1"
  local config_path="$2"
  local csv_path="$3"
  local run_stamp="$4"
  local output_root="${PROJECT_ROOT}/$(dirname "${config_path}")"
  local docs_dir="${output_root}/docs"
  local runtime_config="${docs_dir}/runtime_${dataset_id}_${run_stamp}_19cpu.json"
  local dataset_log="${docs_dir}/batch_${dataset_id}_${run_stamp}.full.log"
  local command

  mkdir -p "${docs_dir}" || return $?
  log "PREPARE dataset=${dataset_id}"
  local -a preparation_status
  if make_runtime_config \
    "${PROJECT_ROOT}/${config_path}" "${runtime_config}" \
    "${PROJECT_ROOT}/${csv_path}" 2>&1 | tee -a "${dataset_log}"; then
    preparation_status=("${PIPESTATUS[@]}")
  else
    preparation_status=("${PIPESTATUS[@]}")
  fi
  if (( preparation_status[0] != 0 || preparation_status[1] != 0 )); then
    log "FINISH dataset=${dataset_id} stage=config exit_code=${preparation_status[0]} log_exit_code=${preparation_status[1]}"
    return 1
  fi

  command=(
    taskset -c "${YONOD_CPU_SET}"
    "${YONOD_CONDA_BIN}" run --no-capture-output -n "${YONOD_CONDA_ENV}"
    python -u main.py
    --json "${runtime_config}"
    --csv "${PROJECT_ROOT}/${csv_path}"
  )

  log "START dataset=${dataset_id}"
  log "source_config=${config_path}"
  log "runtime_config=${runtime_config}"
  log "csv=${csv_path}"
  log "output_root=${output_root}"
  log "dataset_log=${dataset_log}"
  log "config_sha256=$(sha256sum "${PROJECT_ROOT}/${config_path}" | awk '{print $1}')"
  log "csv_sha256=$(sha256sum "${PROJECT_ROOT}/${csv_path}" | awk '{print $1}')"
  log "command=$(quote_command "${command[@]}")"

  local -a command_status
  if "${command[@]}" 2>&1 | tee -a "${dataset_log}"; then
    command_status=("${PIPESTATUS[@]}")
  else
    command_status=("${PIPESTATUS[@]}")
  fi
  local status=${command_status[0]}
  if (( command_status[1] != 0 )); then
    log "ERROR 日志写入失败：${dataset_log} exit_code=${command_status[1]}"
    [[ "${status}" -ne 0 ]] || status=${command_status[1]}
  fi

  log "FINISH dataset=${dataset_id} exit_code=${status}"
  return "${status}"
}

worker_main() {
  local run_stamp="$1"
  local succeeded=0
  local failed=0
  local index

  cd "${PROJECT_ROOT}"
  log "RUN_NAME=${RUN_NAME}"
  log "RUN_STAMP=${run_stamp}"
  log "PROJECT_ROOT=${PROJECT_ROOT}"
  log "CONDA_ENV=${YONOD_CONDA_ENV}"
  log "CPU_POLICY=taskset:${YONOD_CPU_SET}, cpu_count:${YONOD_CPU_COUNT}, sequential_jobs:1"
  log "THREAD_ENV=LOKY:${LOKY_MAX_CPU_COUNT}, OMP:${OMP_NUM_THREADS}, OPENBLAS:${OPENBLAS_NUM_THREADS}, MKL:${MKL_NUM_THREADS}, NUMEXPR:${NUMEXPR_NUM_THREADS}"
  log "NOTE=不同模型可能因自身实现少用核心，但整个进程最多只能在这 19 个逻辑 CPU 上运行。"
  preflight

  for index in "${!DATASET_IDS[@]}"; do
    if run_one \
      "${DATASET_IDS[$index]}" \
      "${CONFIG_PATHS[$index]}" \
      "${CSV_PATHS[$index]}" \
      "${run_stamp}"; then
      ((succeeded += 1))
    else
      ((failed += 1))
      log "ERROR dataset=${DATASET_IDS[$index]}；继续执行下一数据集。"
    fi
  done

  log "BATCH_FINISH succeeded=${succeeded} failed=${failed} total=${#DATASET_IDS[@]}"
  [[ "${failed}" -eq 0 ]]
}

dry_run() {
  cd "${PROJECT_ROOT}"
  preflight
  local check_root
  check_root="$(mktemp -d "${TMPDIR:-/tmp}/yonod-launcher-check.XXXXXXXX")"
  log "DRY_RUN：执行真实配置生成和加载校验；临时材料：${check_root}"
  local index
  for index in "${!DATASET_IDS[@]}"; do
    printf '%d. %s | %s\n' \
      "$((index + 1))" "${CONFIG_PATHS[$index]}" "${CSV_PATHS[$index]}"
    make_runtime_config "${PROJECT_ROOT}/${CONFIG_PATHS[$index]}" \
      "${check_root}/${DATASET_IDS[$index]}/docs/runtime.json" \
      "${PROJECT_ROOT}/${CSV_PATHS[$index]}" || return $?
  done
  log "DRY_RUN_PASS：四份运行时配置生成、YONOD 加载和 AutoGluon 19 核参数均通过，未启动建模。"
}

launch() {
  mkdir -p "${LOG_DIR}"
  cd "${PROJECT_ROOT}"
  preflight

  if [[ -s "${LATEST_PID_PATH}" ]]; then
    local previous_pid
    read -r previous_pid < "${LATEST_PID_PATH}"
    if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
      printf 'ERROR: 检测到仍在运行的批任务 PID=%s；未重复启动。\n' "${previous_pid}" >&2
      return 3
    fi
  fi

  local run_stamp
  run_stamp="$(date '+%Y%m%d_%H%M%S')"
  local batch_log="${LOG_DIR}/${RUN_NAME}_${run_stamp}.full.log"
  local stamped_pid="${LOG_DIR}/${RUN_NAME}_${run_stamp}.pid"

  nohup setsid bash "${SCRIPT_PATH}" --worker "${run_stamp}" \
    </dev/null >"${batch_log}" 2>&1 &
  local worker_pid=$!
  printf '%s\n' "${worker_pid}" > "${stamped_pid}"
  printf '%s\n' "${worker_pid}" > "${LATEST_PID_PATH}"

  printf '批任务已启动。\n'
  printf 'PID: %s\n' "${worker_pid}"
  printf 'PID 文件: %s\n' "${stamped_pid}"
  printf '完整日志: %s\n' "${batch_log}"
  printf '实时查看: tail -f %q\n' "${batch_log}"
}

case "${1:-}" in
  --worker)
    [[ $# -eq 2 ]] || {
      printf 'ERROR: --worker 需要运行时间戳。\n' >&2
      exit 2
    }
    worker_main "$2"
    ;;
  --dry-run)
    dry_run
    ;;
  "")
    launch
    ;;
  *)
    printf '用法：%s [--dry-run]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
