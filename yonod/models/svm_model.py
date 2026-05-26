"""Support Vector Regression adapter for YONOD yield prediction.

SVR is sensitive to feature scale and scales as O(n_samples^2 * n_features)
per fold, so this adapter:

  * always ``StandardScaler``-normalizes;
  * when input dim > ``pca_threshold`` (default 512), inserts a PCA step
    targeting ``pca_n_components`` (default 256) to keep RBF kernel
    computation tractable. This matches YONOD §2.3 SVM PCA policy: Morgan
    (6144) / MolMetaLM (4608) / ATMOMACCS (~1002) trigger PCA; FISD (300)
    does not.

Scaler and PCA are fit inside each CV fold (via sklearn Pipeline) to avoid
test-set leakage.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVMYieldModel:
    """5-fold CV SVR with optional auto-PCA on high-dim inputs."""

    name = "svm"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        auto_pca: bool = True,
        pca_threshold: int = 512,
        pca_n_components: int = 256,
        random_state: int = 42,
        subsample_n: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.auto_pca = auto_pca
        self.pca_threshold = pca_threshold
        self.pca_n_components = pca_n_components
        self.random_state = random_state
        # RBF SVR is O(n^2 d) -- intractable above ~10k rows. If subsample_n
        # is set, each fold's TRAIN set is randomly subsampled to this size
        # before fitting. Test set is untouched.
        self.subsample_n = subsample_n

    def _build(self, n_features: int, n_train: int) -> Pipeline:
        steps = [("scaler", StandardScaler(with_mean=False))]
        if self.auto_pca and n_features > self.pca_threshold:
            # PCA n_components must not exceed min(n_train, n_features).
            n_comp = min(self.pca_n_components, n_train, n_features)
            steps.append(
                ("pca", PCA(n_components=n_comp, random_state=self.random_state))
            )
        steps.append(
            ("svr", SVR(kernel=self.kernel, C=self.C, gamma=self.gamma))
        )
        return Pipeline(steps)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2s, rmses, maes = [], [], []
        oof = np.full(len(y), np.nan, dtype=np.float64)
        t0 = time.time()
        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        for tr, te in kf.split(X):
            tr_use = tr
            if self.subsample_n is not None and len(tr) > self.subsample_n:
                tr_use = rng.choice(tr, size=self.subsample_n, replace=False)
            pipe = self._build(n_features=n_features, n_train=len(tr_use))
            pipe.fit(X[tr_use], y[tr_use])
            pred = pipe.predict(X[te])
            oof[te] = pred
            r2s.append(r2_score(y[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
            maes.append(mean_absolute_error(y[te], pred))
        elapsed = time.time() - t0
        pca_used = self.auto_pca and n_features > self.pca_threshold
        return {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "mae_mean": float(np.mean(maes)),
            "train_time_s": float(elapsed),
            "device": "cpu",
            "pca": self.pca_n_components if pca_used else None,
            "subsample_n": self.subsample_n,
            "oof_pred": oof,
            "oof_y_true": y,
        }
