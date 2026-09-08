#!/usr/bin/env bash
set -euo pipefail

# Stage 27 RF-only gate.  This regenerates the five immutable paper_exact
# populations and reruns only the RF baseline; it must pass before any
# AutoGluon 25-fold job is launched.
bash scripts/run_yieldmaster_paper_exact_rf_20260908.sh
