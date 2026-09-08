#!/usr/bin/env bash
set -euo pipefail

ENTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ENTRY_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

exec bash scripts/start_yieldmaster_autogluon_paper_exact_5x5_20260908_nohup.sh
