from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_official_component_splits import audit


def test_official_component_split_schema_covers_four_datasets() -> None:
    summary = audit(REPRO_ROOT)
    assert set(summary["dataset"]) == {"BH", "BH2", "SM", "SLAP"}
    assert {"0d", "1d", "2d"}.issubset(set(summary["holdout_dim"]))
    assert summary["n_rows"].min() > 1000
