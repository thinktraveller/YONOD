from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from merge_official_differences import _local_records


def test_local_records_map_descriptor_and_model_names() -> None:
    summary = pd.DataFrame(
        [
            {
                "stage": "core_rf_5x5",
                "dataset": "SLAP",
                "descriptor": "ohe",
                "model": "rf",
                "mae_mean": 1.0,
                "rmse_mean": 2.0,
                "r2_mean": 0.3,
                "kendall_tau_mean": 0.4,
                "feature_dim": 115,
                "folds": 25,
            }
        ]
    )
    records = _local_records(summary)
    assert len(records) == 4
    assert set(records["descriptor"]) == {"OHE"}
    assert set(records["model"]) == {"RF"}
    assert set(records["metric"]) == {"mae", "rmse", "r2", "kendall_tau"}
