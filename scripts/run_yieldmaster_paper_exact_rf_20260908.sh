#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-19}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

exec /home/wangzh685/miniconda3/bin/conda run -n yonod \
  python scripts/run_yieldmaster_paper_exact_rf_20260908.py \
    --reference-root reference-proejct/vjethbkm \
    --output-root configs \
    --stamp 20260908 \
    --tolerance 1e-9 \
    --n-estimators 500 \
    --max-features 0.30 \
    --n-jobs -1
