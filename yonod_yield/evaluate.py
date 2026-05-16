"""Unified evaluation entry point.

Every concrete model class in ``yonod_yield.models`` exposes
``cross_validate(X, y, cv)`` returning a metrics dict with the canonical
keys ``{r2_mean, r2_std, rmse_mean, mae_mean, train_time_s, device}``.

This module provides:

  * ``MODEL_REGISTRY`` / ``DESCRIPTOR_REGISTRY``  -- name -> class
  * ``build_model(name, **kw)`` / ``build_descriptor(name, **kw)``
  * ``evaluate_one(descriptor, model, df, cv)`` -- runs one (desc, model)
    combination end-to-end and returns a row dict suitable for assembly
    into ``results/metrics_summary.csv``.

The main script (``run_yield_prediction.py``) loops over the cartesian
product of selected descriptors and models, calling ``evaluate_one``.
"""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import pandas as pd

from .descriptors.atmomaccs import ATMOMACCSDescriptor
from .descriptors.base import BaseDescriptor
from .descriptors.fisd import FISDDescriptor
from .descriptors.molmetalm import MolMetaLMDescriptor
from .descriptors.morgan import MorganDescriptor
from .features.reaction_featurizer import ReactionFeaturizer
from .models.autogluon_model import AutoGluonYieldModel
from .models.rf_model import RFYieldModel
from .models.svm_model import SVMYieldModel
from .models.xgb_model import XGBYieldModel


DESCRIPTOR_REGISTRY: Dict[str, Type[BaseDescriptor]] = {
    "morgan": MorganDescriptor,
    "atmomaccs": ATMOMACCSDescriptor,
    "fisd": FISDDescriptor,
    "molmetalm": MolMetaLMDescriptor,
}

MODEL_REGISTRY: Dict[str, Type[Any]] = {
    "xgb": XGBYieldModel,
    "rf": RFYieldModel,
    "svm": SVMYieldModel,
    "autogluon": AutoGluonYieldModel,
}


def build_descriptor(name: str, **kwargs: Any) -> BaseDescriptor:
    if name not in DESCRIPTOR_REGISTRY:
        raise KeyError(
            f"Unknown descriptor {name!r}. Available: {sorted(DESCRIPTOR_REGISTRY)}"
        )
    return DESCRIPTOR_REGISTRY[name](**kwargs)


def build_model(name: str, **kwargs: Any) -> Any:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def evaluate_one(
    descriptor_name: str,
    model_name: str,
    df: pd.DataFrame,
    cv: int = 5,
    descriptor_kwargs: Dict[str, Any] | None = None,
    model_kwargs: Dict[str, Any] | None = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Featurize ``df`` with one descriptor, evaluate one model under K-fold CV.

    Returns a flat dict suitable for appending as a row to
    ``results/metrics_summary.csv``. By default the large ``oof_pred`` /
    ``oof_y_true`` ndarrays are stripped; pass ``return_predictions=True`` to
    keep them (used by ``run_yield_prediction.py`` for scatter plots).
    """
    descriptor = build_descriptor(descriptor_name, **(descriptor_kwargs or {}))
    model = build_model(model_name, **(model_kwargs or {}))

    featurizer = ReactionFeaturizer(descriptor)
    X, mask = featurizer.transform(df)
    y = df["yield"].to_numpy()[mask]

    metrics = model.cross_validate(X, y, cv=cv)
    if not return_predictions:
        metrics.pop("oof_pred", None)
        metrics.pop("oof_y_true", None)

    row: Dict[str, Any] = {
        "descriptor": descriptor_name,
        "model": model_name,
        "n_samples": int(mask.sum()),
        "n_total": int(len(mask)),
        "coverage": float(mask.sum() / max(len(mask), 1)),
        "feature_dim": int(X.shape[1]),
        "cv": cv,
    }
    row.update(metrics)
    return row
