from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_final_metrics_report import build_final_metrics


def test_builds_32_final_mfp_ohe_rf_rows(tmp_path: Path) -> None:
    rows = build_final_metrics(REPRO_ROOT, tmp_path)
    assert len(rows) == 32
    assert len({(row["dataset_id"], row["descriptor"], row["metric"]) for row in rows}) == 32
    assert {row["valid_fold_count"] for row in rows} == {25}
    assert all(row["table_s3_display_match"] for row in rows)
    assert (tmp_path / "outputs" / "tables" / "final_mfp_ohe_rf_metrics.csv").exists()
    assert (tmp_path / "outputs" / "tables" / "final_mfp_ohe_rf_metrics.json").exists()
    assert (tmp_path / "outputs" / "reports" / "final_mfp_ohe_rf_metrics.md").exists()
    assert (tmp_path / "outputs" / "reports" / "final_mfp_ohe_rf_metrics.html").exists()


def test_preserves_units_full_precision_and_sm_audit_boundary(tmp_path: Path) -> None:
    rows = build_final_metrics(REPRO_ROOT, tmp_path)
    bh1 = next(row for row in rows if (row["dataset_id"], row["descriptor"], row["metric"]) == ("BH1", "MFP", "mae"))
    assert bh1["mean"] == 4.531535492347868
    assert bh1["std"] == 0.21444793548210375
    assert bh1["unit"] == "yield_percent"
    sl1 = next(row for row in rows if (row["dataset_id"], row["descriptor"], row["metric"]) == ("SL1", "OHE", "rmse"))
    assert sl1["unit"] == "lc_ms_product_ratio"
    sm_mfp = next(row for row in rows if (row["dataset_id"], row["descriptor"]) == ("SM", "MFP"))
    sm_ohe = next(row for row in rows if (row["dataset_id"], row["descriptor"]) == ("SM", "OHE"))
    assert sm_mfp["population_rows"] == 4620
    assert sm_ohe["population_rows"] == 5760
    assert sm_ohe["population_status"] == "artifact_5760_conflicts_with_si_cleaning_text"
