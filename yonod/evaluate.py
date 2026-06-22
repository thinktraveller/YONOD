"""Unified evaluation entry point.

Every concrete model class in ``yonod_yield.models`` exposes
``cross_validate(X, y, cv)`` returning a metrics dict with the canonical
keys ``{r2_mean, r2_std, rmse_mean, mae_mean, train_time_s, device}``.

This module provides:

  * ``MODEL_REGISTRY`` / ``DESCRIPTOR_REGISTRY``  -- name -> class
  * ``build_model(name, **kw)`` / ``build_descriptor(name, **kw)``
  * ``evaluate_one(descriptor, model, smiles, y, cv)`` -- runs one
    (desc, model) combination end-to-end on a SMILES list + label vector
    and returns a row dict suitable for assembly into the metrics CSV.

The main script (``run_yield_prediction.py``) loops over the cartesian
product of selected descriptors and models, calling ``evaluate_one``.

v1.1+ contract: a dataset is a SMILES list + a float label vector.
Use ``features.dataset.load_dataset(csv_path)`` to obtain both from any
2-column CSV. The legacy reaction-level featurizer
(``features.reaction_featurizer.ReactionFeaturizer``) remains available
for amide-style 6-molecule features but is no longer the default.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from .descriptors.atmomaccs import ATMOMACCSDescriptor
from .descriptors.base import BaseDescriptor
from .descriptors.fisd import FISDDescriptor
from .descriptors.maf import MAFDescriptor
from .descriptors.molmetalm import MolMetaLMDescriptor
from .descriptors.morgan import MorganDescriptor
from .features.dataset import MoleculeFeaturizer
from .models.autogluon_model import AutoGluonYieldModel
from .models.rf_model import RFYieldModel
from .models.svm_model import SVMYieldModel
from .models.xgb_model import XGBYieldModel


DESCRIPTOR_REGISTRY: Dict[str, Type[BaseDescriptor]] = {
    "morgan": MorganDescriptor,
    "atmomaccs": ATMOMACCSDescriptor,
    "fisd": FISDDescriptor,
    "molmetalm": MolMetaLMDescriptor,
    "maf": MAFDescriptor,
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
    smiles: Sequence[str],
    y: np.ndarray,
    cv: int = 5,
    descriptor_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Featurize ``smiles`` with one descriptor, train one model with K-fold CV.

    Args:
        descriptor_name: key in ``DESCRIPTOR_REGISTRY``.
        model_name:      key in ``MODEL_REGISTRY``.
        smiles:          list/sequence of SMILES strings.
        y:               float ndarray of labels, len(smiles) == len(y).
        cv:              K-fold splits.
        descriptor_kwargs / model_kwargs: forwarded to constructors.
        return_predictions: if True keep ``oof_pred`` / ``oof_y_true`` in the
            returned dict (used by the main script for scatter plots).

    Returns:
        Flat dict suitable for appending as a row to ``metrics_summary.csv``.
    """
    smiles_list: List[str] = list(smiles)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(smiles_list) != len(y_arr):
        raise ValueError(
            f"smiles ({len(smiles_list)}) and y ({len(y_arr)}) length mismatch"
        )

    descriptor = build_descriptor(descriptor_name, **(descriptor_kwargs or {}))
    model = build_model(model_name, **(model_kwargs or {}))

    featurizer = MoleculeFeaturizer(descriptor)
    X, mask = featurizer.transform(smiles_list)
    y_used = y_arr[mask]

    metrics = model.cross_validate(X, y_used, cv=cv)
    if not return_predictions:
        metrics.pop("oof_pred", None)
        metrics.pop("oof_y_true", None)

    row: Dict[str, Any] = {
        "descriptor": descriptor_name,
        "model": model_name,
        "n_samples": int(mask.sum()),
        "n_total": int(len(mask)),
        "coverage": float(mask.sum() / max(len(mask), 1)),
        "feature_dim": int(X.shape[1]) if X.size else 0,
        "cv": cv,
    }
    row.update(metrics)
    return row
