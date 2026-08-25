"""Metrics used by the VJETHBKM reproduction benchmark."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    tau = kendalltau(truth, pred, nan_policy="omit").statistic
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "r2": float(r2_score(truth, pred)) if len(truth) > 1 else float("nan"),
        "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
    }

