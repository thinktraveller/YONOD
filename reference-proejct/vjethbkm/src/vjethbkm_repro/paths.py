"""Path helpers that keep reproduction files inside reference-proejct."""

from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[2]
YONOD_ROOT = REPRO_ROOT.parents[1]


def ensure_yonod_root() -> Path:
    """Return the YONOD repository root and add it to sys.path once."""

    if not (YONOD_ROOT / "yonod").exists():
        raise RuntimeError(f"Cannot find YONOD package at {YONOD_ROOT}")
    yonod_root_str = str(YONOD_ROOT)
    if yonod_root_str not in sys.path:
        sys.path.insert(0, yonod_root_str)
    return YONOD_ROOT

