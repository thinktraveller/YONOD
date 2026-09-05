#!/usr/bin/env python3
"""Compatibility wrapper for the amide-coupling ablation batch runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "ablation_experiments" / "amide-coupling" / "run_different_order_tasks.py"

if not TARGET.exists():
    raise SystemExit(f"Batch runner not found: {TARGET}")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
