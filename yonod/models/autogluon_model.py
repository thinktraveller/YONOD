"""AutoGluon TabularPredictor adapter for YONOD yield prediction.

The legacy ``fit_evaluate_holdout`` path is kept for older exploratory runs
and is explicitly marked as ``autogluon_internal_holdout``. Strict model
comparisons use ``fit_predict_fold`` or ``cross_validate`` so AutoGluon sees
the same externally defined validation folds as the other estimators.

Windows-specific knobs:
  * ``num_cpus=1`` -- AutoGluon's multiprocessing has known issues on
    Windows fork-less Python; single-process is the documented fallback.
  * ``path`` set to a temp directory so the trained model bundle does not
    pollute the project root (AutoGluon default is ``./AutogluonModels``).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


def _make_df(X: np.ndarray, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Wrap an ndarray into the DataFrame format AutoGluon expects."""
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    if y is not None:
        df["yield"] = y
    return df


def _load_tabular_predictor() -> Any:
    from autogluon.tabular import TabularPredictor  # imported lazily

    return TabularPredictor


def _autogluon_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("autogluon.common", "autogluon.core", "autogluon.tabular"):
        try:
            module = __import__(package_name, fromlist=["__version__"])
            version = getattr(module, "__version__", None)
            versions[package_name] = str(version) if version is not None else "unknown"
        except Exception:
            versions[package_name] = "unavailable"
    return versions


def _validate_matrix_pair(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train)
    valid = np.asarray(X_valid)
    target = np.asarray(y_train, dtype=float)
    if train.ndim != 2 or valid.ndim != 2:
        raise ValueError("AutoGluon 输入必须是二维矩阵")
    if len(train) != len(target):
        raise ValueError("X_train 与 y_train 行数不一致")
    if train.shape[1] != valid.shape[1]:
        raise ValueError("训练折和验证折特征维度不一致")
    if not np.isfinite(train).all() or not np.isfinite(valid).all() or not np.isfinite(target).all():
        raise ValueError("AutoGluon 输入包含 NaN 或 inf")
    return train, target, valid


class AutoGluonYieldModel:
    """AutoGluon TabularPredictor wrapper with auditable outer folds."""

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

    def _resolve_fold_path(
        self,
        fold_index: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Path, bool]:
        context = context or {}
        explicit = context.get("model_artifact_path")
        root = context.get("model_root") or context.get("save_path") or self.save_path
        if explicit:
            path = Path(explicit)
            remove_after = bool(context.get("cleanup", self.cleanup))
        elif root is not None:
            path = Path(root) / f"fold-{int(fold_index):02d}"
            remove_after = bool(context.get("cleanup", self.cleanup))
        else:
            path = Path(tempfile.mkdtemp(prefix=f"yonod_ag_fold{int(fold_index):02d}_"))
            remove_after = bool(context.get("cleanup", self.cleanup))
        path.mkdir(parents=True, exist_ok=True)
        return path, remove_after

    @staticmethod
    def _cleanup_path(path: Path) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def fit_predict_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        *,
        fold_index: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Fit AutoGluon on one external training fold and predict validation."""
        train, target, valid = _validate_matrix_pair(X_train, y_train, X_valid)
        TabularPredictor = _load_tabular_predictor()
        path, remove_after = self._resolve_fold_path(fold_index, context)
        train_df = _make_df(train, target)
        valid_df = _make_df(valid)

        started_train = time.time()
        predictor = None
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
            train_time = time.time() - started_train
            started_predict = time.time()
            prediction_raw = predictor.predict(valid_df)
            predict_time = time.time() - started_predict
            prediction = np.asarray(prediction_raw, dtype=float)
            if prediction.ndim != 1:
                prediction = prediction.reshape(-1)
            if len(prediction) != len(valid) or not np.isfinite(prediction).all():
                raise ValueError("AutoGluon 预测结果行数或数值非法")

            metadata: Dict[str, Any] = {
                "evaluation_protocol": str((context or {}).get("evaluation_protocol", "outer_kfold")),
                "fold_index": int(fold_index),
                "outer_seed": int((context or {}).get("outer_seed", self.random_state)),
                "time_limit": self.time_limit,
                "presets": self.presets,
                "num_cpus": self.num_cpus,
                "random_state": self.random_state,
                "random_state_policy": (
                    "outer split fixed; AutoGluon 1.1.1 has no single top-level seed covering all submodels"
                ),
                "autogluon_versions": _autogluon_versions(),
                "train_time_s": float(train_time),
                "predict_time_s": float(predict_time),
                "model_artifact_path": str(path),
                "model_artifact_cleanup": bool(remove_after),
            }
            if predictor is not None and hasattr(predictor, "leaderboard"):
                try:
                    leaderboard = predictor.leaderboard(silent=True)
                    metadata["leaderboard_rows"] = int(len(leaderboard))
                except Exception as exc:
                    metadata["leaderboard_error"] = f"{type(exc).__name__}: {exc}"
            return prediction.astype(float, copy=False), metadata
        finally:
            if remove_after:
                self._cleanup_path(path)

    def fit_evaluate_holdout(
        self, X: np.ndarray, y: np.ndarray, holdout_frac: float = 0.2
    ) -> Dict[str, Any]:
        """Train on ``1-holdout_frac`` of (X, y) and score on the rest.

        This is an exploratory compatibility path and must not be mixed into
        strict K-fold rankings.
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y, dtype=float)
        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("AutoGluon holdout 输入必须为与 y 行数一致的二维矩阵")
        if len(y_array) < 2:
            raise ValueError("AutoGluon holdout 至少需要 2 个样本")

        rng = np.random.default_rng(self.random_state)
        n = len(y_array)
        idx = rng.permutation(n)
        n_test = max(1, int(round(n * holdout_frac)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        if len(train_idx) == 0:
            raise ValueError("AutoGluon holdout 训练集为空，请降低 holdout_frac")

        prediction, metadata = self.fit_predict_fold(
            X_array[train_idx],
            y_array[train_idx],
            X_array[test_idx],
            fold_index=1,
            context={"evaluation_protocol": "autogluon_internal_holdout", "outer_seed": self.random_state},
        )
        y_true = y_array[test_idx]
        oof = np.full(n, np.nan, dtype=np.float64)
        oof[test_idx] = prediction

        return {
            "r2_mean": float(r2_score(y_true, prediction)),
            "r2_std": 0.0,
            "rmse_mean": float(np.sqrt(mean_squared_error(y_true, prediction))),
            "rmse_std": 0.0,
            "mae_mean": float(mean_absolute_error(y_true, prediction)),
            "mae_std": 0.0,
            "train_time_s": float(metadata["train_time_s"]),
            "predict_time_s": float(metadata["predict_time_s"]),
            "device": "cpu",
            "evaluation_protocol": "autogluon_internal_holdout",
            "expected_folds": 1,
            "completed_folds": 1,
            "holdout_frac": float(holdout_frac),
            "autogluon_time_limit": self.time_limit,
            "autogluon_presets": self.presets,
            "autogluon_num_cpus": self.num_cpus,
            "autogluon_seed_policy": metadata.get("random_state_policy"),
            "oof_pred": oof,
            "oof_y_true": y_array,
            "fold_metadata": [metadata],
        }

    def fit_evaluate(
        self, X: np.ndarray, y: np.ndarray, holdout_frac: float = 0.2
    ) -> Dict[str, Any]:
        """Backward-compatible name for the exploratory holdout path."""
        return self.fit_evaluate_holdout(X, y, holdout_frac=holdout_frac)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Any]:
        """Run strict externally defined K-fold CV for comparable rankings."""
        X_array = np.asarray(X)
        y_array = np.asarray(y, dtype=float)
        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("AutoGluon CV 输入必须为与 y 行数一致的二维矩阵")
        if len(y_array) < cv:
            raise ValueError("样本数必须不少于 cv 折数")
        if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
            raise ValueError("AutoGluon CV 输入包含 NaN 或 inf")

        splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        r2s, rmses, maes = [], [], []
        oof = np.full(len(y_array), np.nan, dtype=np.float64)
        metadata_rows = []
        started = time.time()
        predict_time = 0.0

        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_array), start=1):
            prediction, metadata = self.fit_predict_fold(
                X_array[train_idx],
                y_array[train_idx],
                X_array[valid_idx],
                fold_index=fold_index,
                context={
                    "evaluation_protocol": "outer_kfold",
                    "outer_seed": self.random_state,
                    "model_root": self.save_path,
                },
            )
            y_valid = y_array[valid_idx]
            oof[valid_idx] = prediction
            r2s.append(float(r2_score(y_valid, prediction)))
            rmses.append(float(np.sqrt(mean_squared_error(y_valid, prediction))))
            maes.append(float(mean_absolute_error(y_valid, prediction)))
            metadata_rows.append(metadata)
            predict_time += float(metadata.get("predict_time_s", 0.0))

        return {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "train_time_s": float(time.time() - started),
            "predict_time_s": float(predict_time),
            "device": "cpu",
            "evaluation_protocol": "outer_kfold",
            "expected_folds": int(cv),
            "completed_folds": int(len(metadata_rows)),
            "autogluon_time_limit": self.time_limit,
            "autogluon_presets": self.presets,
            "autogluon_num_cpus": self.num_cpus,
            "autogluon_versions": metadata_rows[0].get("autogluon_versions", {}) if metadata_rows else {},
            "autogluon_seed_policy": (
                "outer split fixed; AutoGluon 1.1.1 has no single top-level seed covering all submodels"
            ),
            "fold_metadata": metadata_rows,
            "oof_pred": oof,
            "oof_y_true": y_array,
        }
