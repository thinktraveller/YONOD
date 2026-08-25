from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extract_official_metrics import extract_records


def test_extracts_expected_official_rf_metric_records() -> None:
    all_records, table_s3_records = extract_records(REPRO_ROOT)
    assert len(all_records) == 200
    assert len(table_s3_records) == 48
    assert {row["dataset"] for row in table_s3_records} == {"BH", "BH2", "SLAP", "SM"}
    assert {row["descriptor"] for row in table_s3_records} == {"MFP", "OHE", "PhysChem"}


def test_known_suzuki_mfp_rf_mae_target_is_stable() -> None:
    _, table_s3_records = extract_records(REPRO_ROOT)
    matches = [
        row
        for row in table_s3_records
        if row["dataset"] == "SM"
        and row["descriptor"] == "MFP"
        and row["metric"] == "mae"
    ]
    assert len(matches) == 1
    assert abs(float(matches[0]["paper_value"]) - 7.398945621312399) < 1e-12
