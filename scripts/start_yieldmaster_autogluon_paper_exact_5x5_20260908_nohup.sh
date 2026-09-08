#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/yieldmaster_autogluon_paper_exact_5x5_${RUN_STAMP}.full.log"
PID_PATH="${LOG_DIR}/yieldmaster_autogluon_paper_exact_5x5_${RUN_STAMP}.pid"
OUTPUT_ROOT="${PROJECT_ROOT}/configs/yieldmaster_paper_exact_5x5_20260908"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

FORMAL_OUTPUTS=(
  "${OUTPUT_ROOT}/autogluon_fold_metrics.csv"
  "${OUTPUT_ROOT}/autogluon_predictions.csv"
  "${OUTPUT_ROOT}/autogluon_summary.csv"
  "${OUTPUT_ROOT}/paper_exact_autogluon_run_manifest.json"
  "${OUTPUT_ROOT}/rf_vs_autogluon_paired_fold_input.csv"
)
for output_path in "${FORMAL_OUTPUTS[@]}"; do
  if [[ -e "${output_path}" ]]; then
    echo "REFUSE_TO_START_EXISTING_FORMAL_OUTPUT: ${output_path}" >&2
    exit 2
  fi
done

/home/wangzh685/miniconda3/bin/conda run -n yonod \
  python scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.py \
    --validate-materials-only \
    --reference-root reference-proejct/vjethbkm \
    --output-root configs \
    --stamp 20260908

echo "NOHUP_COMMAND: nohup setsid bash scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.sh < /dev/null > ${LOG_PATH} 2>&1 &"
nohup setsid bash scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.sh < /dev/null > "${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_PATH}"
echo "PID: ${PID}"
echo "LOG: ${LOG_PATH}"
echo "PID_FILE: ${PID_PATH}"
echo "EXPECTED_OUTPUT_ROOT: ${OUTPUT_ROOT}"
