"""AutoGluon TabularPredictor adapter for YONOD yield prediction.

AutoGluon does its own internal bagging/stacking/CV, so unlike XGB/RF/SVM
this adapter exposes a single ``fit_evaluate(X, y, holdout_frac=0.2)``
method (random hold-out, not K-fold). YONOD §五 T2.4 explicitly chooses
this design to avoid the cost of K outer folds × N internal models.

Windows-specific knobs:
  * ``num_cpus=1`` -- AutoGluon's multiprocessing has known issues on
    Windows fork-less Python; single-process is the documented fallback.
  * ``path`` set to a temp directory so the trained model bundle does not
    pollute the project root (AutoGluon default is ``./AutogluonModels``).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _make_df(X: np.ndarray, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Wrap an ndarray into the DataFrame format AutoGluon expects."""
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    if y is not None:
        df["yield"] = y
    return df


class AutoGluonYieldModel:
    """AutoGluon TabularPredictor wrapper with random hold-out evaluation."""

    name = "autogluon"

    def __init__(
        self,
        time_limit: int = 300,
        presets: str = "medium_quality",
        num_cpus: int = 1,
        random_state: int = 42,
        save_path: Optional[Path] = None,
        cleanup: bool = True,
    ) -> None:
        self.time_limit = time_limit
        self.presets = presets
        self.num_cpus = num_cpus
        self.random_state = random_state
        self.save_path = save_path
        self.cleanup = cleanup

    def fit_evaluate(
        self, X: np.ndarray, y: np.ndarray, holdout_frac: float = 0.2
    ) -> Dict[str, float]:
        """Train on ``1-holdout_frac`` of (X, y) and score on the rest."""
        from autogluon.tabular import TabularPredictor  # imported lazily

        rng = np.random.default_rng(self.random_state)
        n = len(y)
        idx = rng.permutation(n)
        n_test = max(1, int(round(n * holdout_frac)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        train_df = _make_df(X[train_idx], y[train_idx])
        test_df = _make_df(X[test_idx])
        y_true = y[test_idx]

        path = self.save_path or Path(tempfile.mkdtemp(prefix="yonod_ag_"))
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            predictor = TabularPredictor(
                label="yield",
                problem_type="regression",
                path=str(path),
                verbosity=1,
            ).fit(
                train_data=train_df,
                time_limit=self.time_limit,
                presets=self.presets,
                num_cpus=self.num_cpus,
            )
            pred = predictor.predict(test_df).values
            elapsed = time.time() - t0
            # Scatter the test-only predictions into a full-length oof array
            # (train indices stay NaN so plotting can drop them).
            oof = np.full(n, np.nan, dtype=np.float64)
            oof[test_idx] = pred
            metrics = {
                "r2_mean": float(r2_score(y_true, pred)),
                "r2_std": 0.0,  # single hold-out, no fold variance
                "rmse_mean": float(np.sqrt(mean_squared_error(y_true, pred))),
                "mae_mean": float(mean_absolute_error(y_true, pred)),
                "train_time_s": float(elapsed),
                "device": "cpu",
                "holdout_frac": holdout_frac,
                "oof_pred": oof,
                "oof_y_true": y,
            }
        finally:
            if self.cleanup and self.save_path is None:
                shutil.rmtree(path, ignore_errors=True)

        return metrics

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        """Provided for API parity; internally just calls ``fit_evaluate``.

        AutoGluon already bag-stacks across many models -- doing K external
        folds on top would be redundant and 5x more expensive. ``cv`` is
        ignored; ``r2_std`` is reported as 0.
        """
        return self.fit_evaluate(X, y, holdout_frac=1.0 / cv)
