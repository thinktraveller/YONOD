"""Random Forest regressor adapter for YONOD yield prediction.

CPU-only by design (sklearn RF is single-threaded per tree but parallel across
trees via ``n_jobs=-1``). Same ``cross_validate(X, y, cv=5)`` contract as
``XGBYieldModel`` so the pipeline can swap them transparently.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


class RFYieldModel:
    """5-fold CV Random Forest regressor."""

    name = "rf"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: float | int | str | None = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _build(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.params)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2s, rmses, maes = [], [], []
        oof = np.full(len(y), np.nan, dtype=np.float64)
        t0 = time.time()
        for tr, te in kf.split(X):
            model = self._build()
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            oof[te] = pred
            r2s.append(r2_score(y[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
            maes.append(mean_absolute_error(y[te], pred))
        elapsed = time.time() - t0
        return {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "mae_mean": float(np.mean(maes)),
            "train_time_s": float(elapsed),
            "device": "cpu",
            "oof_pred": oof,
            "oof_y_true": y,
        }
