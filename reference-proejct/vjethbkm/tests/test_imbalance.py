from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vjethbkm_repro.imbalance import bucket_sample_weights, high_yield_summary, yield_bucket_summary


def test_yield_bucket_summary_counts_all_rows() -> None:
    df = pd.DataFrame({"yield": [0.0, 0.1, 0.25, 0.45, 0.81]})
    summary = yield_bucket_summary(df, "yield")
    assert int(summary["count"].sum()) == len(df)


def test_bucket_sample_weights_upweight_rare_bins() -> None:
    weights = bucket_sample_weights([0.1, 0.15, 0.18, 0.85])
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[0]


def test_high_yield_summary_reports_precision_and_recall() -> None:
    predictions = pd.DataFrame(
        {
            "descriptor": ["ohe", "ohe", "ohe"],
            "model": ["rf", "rf", "rf"],
            "y_true": [0.9, 0.7, 0.85],
            "y_pred": [0.82, 0.81, 0.2],
        }
    )
    summary = high_yield_summary(predictions, threshold=0.8)
    assert float(summary.loc[0, "precision"]) == 0.5
    assert float(summary.loc[0, "recall"]) == 0.5

