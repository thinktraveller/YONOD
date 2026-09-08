#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="/home/wangzh685/miniconda3/bin/conda"
CONDA_ENV="yonod"
RUN_BATCH="yieldmaster_mfp_ohe_outercv_20260908"
CPU_LIMIT="19"
AUTOGLOON_TIME_LIMIT="300"
AUTOGLOON_PRESETS="medium_quality"
AUTOGLOON_NUM_CPUS="19"

export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT="$CPU_LIMIT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

DATASET_IDS=(bh1 bh2 sl1 sm)
CSV_PATHS=(
  dataset/yieldmaster/BH1.csv
  dataset/yieldmaster/BH2.csv
  dataset/yieldmaster/SL1.csv
  dataset/yieldmaster/SM.csv
)
CONFIG_PATHS=(
  configs/yieldmaster_bh1_mfp_ohe_outercv_20260908/yieldmaster_bh1_mfp_ohe_outercv_20260908.json
  configs/yieldmaster_bh2_mfp_ohe_outercv_20260908/yieldmaster_bh2_mfp_ohe_outercv_20260908.json
  configs/yieldmaster_sl1_mfp_ohe_outercv_20260908/yieldmaster_sl1_mfp_ohe_outercv_20260908.json
  configs/yieldmaster_sm_mfp_ohe_outercv_20260908/yieldmaster_sm_mfp_ohe_outercv_20260908.json
)

make_manifest() {
  local manifest="$1"
  local status="$2"
  local dataset_id="$3"
  local config_path="$4"
  local csv_path="$5"
  local output_root="$6"
  local per_dataset_log="$7"
  local command_line="$8"
  local timestamp="$9"
  local exit_code="${10:-}"
  local git_commit
  git_commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  python3 - "$manifest" "$status" "$dataset_id" "$config_path" "$csv_path" "$output_root" "$per_dataset_log" "$command_line" "$timestamp" "$git_commit" "$exit_code" <<'PY'
import json
import sys
from pathlib import Path
manifest, status, dataset_id, config_path, csv_path, output_root, per_dataset_log, command_line, timestamp, git_commit, exit_code = sys.argv[1:12]
payload = {
    "run_batch": "yieldmaster_mfp_ohe_outercv_20260908",
    "status": status,
    "dataset_id": dataset_id,
    "timestamp": timestamp,
    "git_commit": git_commit,
    "conda_env": "yonod",
    "config_path": config_path,
    "csv_path": csv_path,
    "output_root": output_root,
    "per_dataset_log": per_dataset_log,
    "command_line": command_line,
    "evaluation_protocol": "outer_kfold",
    "expected_folds": 5,
    "feature_scope": ["mfp_vjethbkm", "ohe_train_fold"],
    "model_scope": ["XGBoost", "Random Forest", "SVM", "AutoGluon", "LightGBM"],
    "autogluon": {
        "time_limit_per_outer_fold_seconds": 300,
        "presets": "medium_quality",
        "num_cpus_per_fold": 19,
        "seed_strategy": "外层折独立 TabularPredictor；随机种子由 YONOD 外层折适配器固定/派生。",
    },
    "cpu_policy": {
        "dataset_concurrency": 1,
        "cpu_limit": 19,
        "LOKY_MAX_CPU_COUNT": "19",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "parquet_backend": {
        "pyarrow": "not_installed",
        "fastparquet": "not_installed",
        "note": "普通 main.py 路径不依赖 parquet；strict benchmark parquet 路径需安装 parquet 后端后再用。",
    },
    "expected_outputs": [
        f"{output_root}/docs/metrics_summary.csv",
        f"{output_root}/docs/descriptor_status.csv",
        f"{output_root}/docs/run_*.log",
        f"{output_root}/report/report.md",
        f"{output_root}/report/report.html",
        f"{output_root}/pictures/",
    ],
}
if exit_code:
    payload["exit_code"] = int(exit_code)
Path(manifest).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_one() {
  local dataset_id="$1"
  local config_path="$2"
  local csv_path="$3"
  local output_root
  output_root="$(dirname "$config_path")"
  mkdir -p "$output_root/docs"

  local stamp
  stamp="$(date '+%Y%m%d_%H%M%S')"
  local per_dataset_log="$output_root/docs/outercv_${dataset_id}_${stamp}.log"
  local manifest="$output_root/docs/outercv_run_manifest_${dataset_id}_${stamp}.json"
  local command=("$CONDA_BIN" run -n "$CONDA_ENV" python main.py --json "$config_path" --csv "$csv_path")
  local command_line
  printf -v command_line '%q ' "${command[@]}"

  make_manifest "$manifest" "running" "$dataset_id" "$config_path" "$csv_path" "$output_root" "$per_dataset_log" "$command_line" "$(date '+%Y-%m-%d %H:%M:%S %z')"

  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] START dataset=${dataset_id}"
  echo "config=${config_path}"
  echo "csv=${csv_path}"
  echo "output_root=${output_root}"
  echo "per_dataset_log=${per_dataset_log}"
  echo "manifest=${manifest}"
  echo "AutoGluon per outer fold: time_limit=${AUTOGLOON_TIME_LIMIT}, presets=${AUTOGLOON_PRESETS}, num_cpus=${AUTOGLOON_NUM_CPUS}"
  echo "CPU policy: dataset_concurrency=1, LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT}, OMP/BLAS/MKL/NUMEXPR=1"
  echo "command=${command_line}"

  set +e
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] COMMAND ${command_line}"
    "${command[@]}"
  } 2>&1 | tee -a "$per_dataset_log"
  local status=${PIPESTATUS[0]}
  set -e

  make_manifest "$manifest" "finished" "$dataset_id" "$config_path" "$csv_path" "$output_root" "$per_dataset_log" "$command_line" "$(date '+%Y-%m-%d %H:%M:%S %z')" "$status"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] FINISH dataset=${dataset_id} exit_code=${status}"
  if [[ "$status" -ne 0 ]]; then
    echo "dataset=${dataset_id} failed; see ${per_dataset_log}" >&2
    exit "$status"
  fi
}

main() {
  echo "RUN_BATCH=${RUN_BATCH}"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "CONDA_ENV=${CONDA_ENV}"
  echo "Start time=$(date '+%Y-%m-%d %H:%M:%S %z')"
  for idx in "${!DATASET_IDS[@]}"; do
    run_one "${DATASET_IDS[$idx]}" "${CONFIG_PATHS[$idx]}" "${CSV_PATHS[$idx]}"
  done
  echo "End time=$(date '+%Y-%m-%d %H:%M:%S %z')"
}

main "$@"
