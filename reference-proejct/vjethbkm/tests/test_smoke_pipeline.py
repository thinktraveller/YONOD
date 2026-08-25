from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vjethbkm_repro.features import build_ohe
from vjethbkm_repro.metrics import regression_metrics
from vjethbkm_repro.splits import repeated_kfold_manifest


def test_repeated_kfold_manifest_has_train_and_test_rows() -> None:
    df = pd.DataFrame({"x": range(10)})
    manifest = repeated_kfold_manifest(df, n_splits=2, repeats=1, random_state=42)
    assert set(manifest["is_train"]) == {True, False}
    assert manifest.groupby(["repeat", "fold", "is_train"]).size().min() > 0


def test_ohe_ignores_unseen_categories() -> None:
    train = pd.DataFrame({"a": ["A", "B"], "b": ["X", "Y"]})
    test = pd.DataFrame({"a": ["C"], "b": ["Y"]})
    result = build_ohe(train, test, ["a", "b"])
    assert result.train.shape == (2, result.dim)
    assert result.test.shape == (1, result.dim)


def test_regression_metrics_are_finite_for_regular_case() -> None:
    metrics = regression_metrics(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8]))
    assert metrics["mae"] > 0
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["r2"])

