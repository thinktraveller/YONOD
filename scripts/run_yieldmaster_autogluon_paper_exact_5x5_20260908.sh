#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AG_TIME_LIMIT="${YONOD_PAPER_EXACT_AG_TIME_LIMIT:-300}"
AG_PRESETS="${YONOD_PAPER_EXACT_AG_PRESETS:-medium_quality}"
AG_NUM_CPUS="${YONOD_PAPER_EXACT_AG_NUM_CPUS:-19}"
AG_RANDOM_STATE="${YONOD_PAPER_EXACT_AG_RANDOM_STATE:-42}"

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-${AG_NUM_CPUS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"

echo "COMMAND: conda run -n yonod python scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.py --reference-root reference-proejct/vjethbkm --output-root configs --stamp 20260908 --time-limit ${AG_TIME_LIMIT} --presets ${AG_PRESETS} --num-cpus ${AG_NUM_CPUS} --random-state ${AG_RANDOM_STATE} --quarantine-incomplete-artifacts"
echo "CPU_POLICY: LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT} OMP_NUM_THREADS=${OMP_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"

exec /home/wangzh685/miniconda3/bin/conda run -n yonod \
  python scripts/run_yieldmaster_autogluon_paper_exact_5x5_20260908.py \
    --reference-root reference-proejct/vjethbkm \
    --output-root configs \
    --stamp 20260908 \
    --time-limit "${AG_TIME_LIMIT}" \
    --presets "${AG_PRESETS}" \
    --num-cpus "${AG_NUM_CPUS}" \
    --random-state "${AG_RANDOM_STATE}" \
    --quarantine-incomplete-artifacts
