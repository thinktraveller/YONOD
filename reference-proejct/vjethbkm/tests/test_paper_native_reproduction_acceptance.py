from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_paper_native_reproduction_acceptance import status_for_row


def test_mfp_status_requires_exact_features_and_tight_metrics() -> None:
    item = {"mfp_alignment": {"x_exact": True, "y_exact": True, "smiles_columns_exact": True}}
    assert status_for_row("MFP", item, 1e-13)[0] == "pass"
    assert status_for_row("MFP", item, 1e-10)[0] == "fail"
    item["mfp_alignment"]["x_exact"] = False
    assert status_for_row("MFP", item, 0.0)[0] == "fail"


def test_ohe_status_requires_identity_and_prediction_alignment() -> None:
    item = {"oof_alignment": {"y_true_exact": True, "fold_id_exact": True, "y_pred_max_abs_diff": 1e-13}}
    assert status_for_row("OHE", item, 0.0)[0] == "pass"
    item["oof_alignment"]["fold_id_exact"] = False
    assert status_for_row("OHE", item, 0.0)[0] == "fail"
