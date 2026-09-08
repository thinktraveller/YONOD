#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/yieldmaster_autogluon_paper_exact_5x5_${RUN_STAMP}.full.log"
PID_PATH="${LOG_DIR}/yieldmaster_autogluon_paper_exact_5x5_${RUN_STAMP}.pid"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "NOHUP_COMMAND: nohup bash scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.sh > ${LOG_PATH} 2>&1 &"
nohup bash scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.sh > "${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_PATH}"
echo "PID: ${PID}"
echo "LOG: ${LOG_PATH}"
echo "PID_FILE: ${PID_PATH}"
echo "EXPECTED_OUTPUT_ROOT: ${PROJECT_ROOT}/configs/yieldmaster_paper_exact_5x5_20260908"
