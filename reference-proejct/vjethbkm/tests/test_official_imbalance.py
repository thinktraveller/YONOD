from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_official_imbalance import analyze


def test_official_imbalance_outputs_expected_row_counts() -> None:
    result = analyze(REPRO_ROOT)
    assert result["dataset_bucket_rows"] == 24
    assert result["high_yield_rows"] == 12
    assert result["bucket_error_rows"] >= 60
