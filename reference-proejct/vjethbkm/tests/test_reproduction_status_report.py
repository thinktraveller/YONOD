from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_reproduction_status_report import build_report


def test_reproduction_status_report_has_full_table_s3_coverage() -> None:
    status = build_report(REPRO_ROOT)
    assert status["table_s3_rows"] == 48
    assert status["table_s3_matched_rows"] == 48
    assert status["descriptor_max_abs_diff"]["MFP"] == 0.0
    assert status["unresolved_rows"] == 4
