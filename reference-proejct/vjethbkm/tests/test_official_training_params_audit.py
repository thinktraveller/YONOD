from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_official_training_params import audit


def test_official_rf_defaults_are_detected() -> None:
    result = audit(REPRO_ROOT)
    assert "n_estimators=500" in result["rf_defaults_literal"]
    assert "max_features=0.3" in result["rf_defaults_literal"]
    assert result["config_summaries"]["BH1_OHE.json"]["outer_splits"] == 5
    assert result["config_summaries"]["BH1_OHE.json"]["seed"] == 1000
