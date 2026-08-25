from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_bh2_external import analyze


def test_bh2_external_per_product_files_are_complete() -> None:
    result = analyze(REPRO_ROOT)
    assert result["per_product_files"]["input_count"] == 13
    assert result["per_product_files"]["prediction_count"] == 13
    assert result["corrected_187"]["rows"] == 187
