"""LightGBM baseline adapter for comparable reaction-level regression.

The class retains the legacy ``cross_validate`` convenience method, but the
strict benchmark uses :meth:`_build` through ``benchmark.execute_fold`` so
that every model receives the same externally generated train/valid fold.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - depends on the optional environment package
    lgb = None


class LightGBMYieldModel:
    """CPU-only fixed-parameter LightGBM regression baseline.

    Missing/NaN feature values are deliberately rejected by the strict fold
    executor before this adapter is built.  Constant and sparse feature
    columns are accepted by LightGBM; their treatment is recorded by the
    executor's feature schema and fold metadata instead of silently changing
    the training population.
    """

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 1,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators 必须 >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须 > 0")
        if num_leaves < 2:
            raise ValueError("num_leaves 必须 >= 2")
        if min_child_samples < 1:
            raise ValueError("min_child_samples 必须 >= 1")
        if n_jobs == 0:
            raise ValueError("n_jobs 不能为 0")
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "device_type": "cpu",
            "verbosity": -1,
        }

    @staticmethod
    def _require_lightgbm() -> None:
        if lgb is None:
            raise ImportError(
                "LightGBM 未安装。请在 conda yonod 环境安装并确认 `import lightgbm` 可用后再运行严格 benchmark。"
            )

    def _build(self):
        self._require_lightgbm()
        return lgb.LGBMRegressor(**self.params)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float | np.ndarray | str]:
        """Compatibility path; strict comparisons must use the external executor."""
        X_array = np.asarray(X)
        y_array = np.asarray(y, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] != len(y_array):
            raise ValueError("X 必须为与 y 行数一致的二维特征矩阵")
        if len(y_array) < cv:
            raise ValueError("样本数必须不少于 cv 折数")
        if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
            raise ValueError("LightGBM 基线不接受 NaN 或 inf；请先在数据契约阶段修复")
        kf = KFold(n_splits=cv, shuffle=True, random_state=int(self.params["random_state"]))
        r2s, rmses, maes = [], [], []
        oof = np.full(len(y_array), np.nan, dtype=float)
        started = time.time()
        for train_idx, valid_idx in kf.split(X_array):
            estimator = self._build()
            estimator.fit(X_array[train_idx], y_array[train_idx])
            prediction = np.asarray(estimator.predict(X_array[valid_idx]), dtype=float)
            oof[valid_idx] = prediction
            r2s.append(r2_score(y_array[valid_idx], prediction))
            rmses.append(float(np.sqrt(mean_squared_error(y_array[valid_idx], prediction))))
            maes.append(mean_absolute_error(y_array[valid_idx], prediction))
        return {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "mae_mean": float(np.mean(maes)),
            "train_time_s": float(time.time() - started),
            "device": "cpu",
            "oof_pred": oof,
            "oof_y_true": y_array,
        }
