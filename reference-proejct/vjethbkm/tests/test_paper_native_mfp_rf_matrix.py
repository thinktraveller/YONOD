from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_paper_native_mfp_rf_matrix import MFP_SPECS


def test_mfp_matrix_keeps_all_four_paper_populations_explicit() -> None:
    assert set(MFP_SPECS) == {"BH1", "BH2", "SL1", "SM"}
    assert MFP_SPECS["SM"]["skip_rows_with_missing_values"] is True
    assert all("official_npz" in spec and "official_summary" in spec for spec in MFP_SPECS.values())
