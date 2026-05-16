"""XGBoost regressor adapter for YONOD yield prediction.

Wraps ``xgboost.XGBRegressor`` with a uniform ``cross_validate(X, y, cv=5)``
interface so the pipeline can swap between XGB / RF / SVM / AutoGluon
without changing call-sites.

Hyperparameters follow YONOD §五 T2.1: 300 trees, lr=0.05, hist algorithm,
GPU when available. Returns a metrics dict with the canonical keys used by
``evaluate.py``.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


def _auto_device() -> str:
    """Return 'cuda' if a usable GPU is present, else 'cpu'."""
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class XGBYieldModel:
    """5-fold CV XGBoost regressor returning ``{r2_mean, r2_std, rmse_mean, mae_mean, train_time_s}``."""

    name = "xgb"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        tree_method: str = "hist",
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        if device == "auto":
            device = _auto_device()
        self.device = device
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            tree_method=tree_method,
            device=device,
            random_state=random_state,
            verbosity=0,
        )

    def _build(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**self.params)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2s, rmses, maes = [], [], []
        oof = np.full(len(y), np.nan, dtype=np.float64)
        t0 = time.time()
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
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
            "device": self.device,
            "oof_pred": oof,
            "oof_y_true": y,
        }
