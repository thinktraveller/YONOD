"""Nested split and fold-local feature construction for HPO.

The search executor receives only the current outer-training population.
Callers keep outer validation rows and labels outside this module until the
winner has been selected.  All learned transforms are fitted on each actual
inner training subset, never on a sliced outer-training transform.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from yonod.features.numeric_conditions import NumericConditionsTransformer


class NestedFoldError(ValueError):
    """The inner split or its feature population violates the outer boundary."""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InnerFold:
    number: int
    train_index: np.ndarray
    valid_index: np.ndarray
    train_ids_sha256: str
    valid_ids_sha256: str

    def audit(self) -> dict[str, Any]:
        return {
            "fold": self.number,
            "n_train": int(len(self.train_index)),
            "n_valid": int(len(self.valid_index)),
            "train_ids_sha256": self.train_ids_sha256,
            "valid_ids_sha256": self.valid_ids_sha256,
        }


def make_inner_folds(
    sample_ids: Sequence[str], *, n_splits: int, seed: int, shuffle: bool,
    groups: Sequence[str] | None = None,
) -> tuple[InnerFold, ...]:
    """Return repeatable splits of *only* the supplied outer-train IDs.

    With groups, each group is exclusive to one validation fold.  For strict
    grouped protocols the caller must provide the audited outer group keys;
    this function never guesses groups from whole-dataset features.
    """
    ids = [str(item) for item in sample_ids]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise NestedFoldError("内层人口 sample_id 必须唯一且非空")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2 or len(ids) < n_splits:
        raise NestedFoldError("内层折数必须 >=2 且不超过外层训练样本数")
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(shuffle, bool):
        raise NestedFoldError("内层 seed/shuffle 无效")
    if groups is not None:
        keys = [str(item) for item in groups]
        if len(keys) != len(ids) or any(not item for item in keys):
            raise NestedFoldError("内层分组键必须与训练人口逐行对应且非空")
        if len(set(keys)) < n_splits:
            raise NestedFoldError("内层独立组数少于 n_splits")
        from sklearn.model_selection import GroupKFold
        # GroupKFold in sklearn 1.4 has no shuffle option.  The user-declared
        # seed remains audited but cannot silently turn grouped CV into KFold.
        if shuffle:
            raise NestedFoldError("当前 grouped inner CV 不支持 shuffle=true；请显式设为 false")
        splitter = GroupKFold(n_splits=n_splits)
        pairs = splitter.split(np.arange(len(ids)), groups=keys)
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        pairs = splitter.split(np.arange(len(ids)))
    folds: list[InnerFold] = []
    seen_valid: set[int] = set()
    for number, (train, valid) in enumerate(pairs, start=1):
        train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
        if not len(train) or not len(valid) or set(train).intersection(valid):
            raise NestedFoldError("内层存在空折或训练/验证交叉")
        seen_valid.update(valid.tolist())
        folds.append(InnerFold(number, train, valid, _digest([ids[i] for i in train]), _digest([ids[i] for i in valid])))
    if seen_valid != set(range(len(ids))):
        raise NestedFoldError("内层验证折没有恰好覆盖外层训练人口")
    return tuple(folds)


@dataclass
class FoldMatrices:
    X_train: np.ndarray
    X_valid: np.ndarray
    state: dict[str, Any]
    ohe_transformer: Any | None = None
    numeric_transformer: NumericConditionsTransformer | None = None


def make_fold_matrices(
    *, sample_ids: Sequence[str], train_index: Sequence[int], valid_index: Sequence[int],
    static_matrix: np.ndarray | None = None,
    categorical_frame: pd.DataFrame | None = None,
    ohe_factory: Callable[[], Any] | None = None,
    numeric_frame: pd.DataFrame | None = None,
    numeric_contract: Mapping[str, Any] | None = None,
    phase: str = "inner",
) -> FoldMatrices:
    """Fit OHE/numeric state on train IDs, then transform valid IDs once."""
    ids = np.asarray([str(item) for item in sample_ids], dtype=str)
    train, valid = np.asarray(train_index, dtype=int), np.asarray(valid_index, dtype=int)
    if not len(train) or not len(valid) or len(set(train.tolist())) != len(train) or len(set(valid.tolist())) != len(valid):
        raise NestedFoldError("折索引必须非空且各自无重复")
    if min(np.r_[train, valid]) < 0 or max(np.r_[train, valid]) >= len(ids) or set(train).intersection(valid):
        raise NestedFoldError("折索引越界或训练/验证交叉")
    if len(set(ids.tolist())) != len(ids):
        raise NestedFoldError("特征人口 sample_id 必须唯一")
    if (static_matrix is None) == (categorical_frame is None):
        raise NestedFoldError("必须且只能提供静态矩阵或原始类别 frame")
    ohe = None
    if static_matrix is not None:
        matrix = np.asarray(static_matrix)
        if matrix.ndim != 2 or len(matrix) != len(ids) or not matrix.shape[1]:
            raise NestedFoldError("静态特征必须与人口等长且为非空二维矩阵")
        X_train, X_valid = np.asarray(matrix[train], dtype=float), np.asarray(matrix[valid], dtype=float)
        state: dict[str, Any] = {"lifecycle": "static_descriptor"}
    else:
        if ohe_factory is None or not isinstance(categorical_frame, pd.DataFrame) or len(categorical_frame) != len(ids):
            raise NestedFoldError("OHE 需要等长原始类别 frame 与未拟合 transformer 工厂")
        ohe = ohe_factory()
        ohe.fit(categorical_frame.iloc[train], sample_ids=ids[train])
        X_train = np.asarray(ohe.transform(categorical_frame.iloc[train], partition="train"), dtype=float)
        X_valid = np.asarray(ohe.transform(categorical_frame.iloc[valid], partition="valid"), dtype=float)
        state = {"lifecycle": "fold_transform", "ohe": ohe.metadata()}
    numeric = None
    if numeric_frame is not None:
        if numeric_contract is None or len(numeric_frame) != len(ids):
            raise NestedFoldError("声明式数值块需要等长 frame 与契约")
        numeric = NumericConditionsTransformer(numeric_contract)
        values_train = numeric.fit_transform(numeric_frame.iloc[train], ids[train], phase=f"{phase}/train")
        values_valid = numeric.transform(numeric_frame.iloc[valid], ids[valid], phase=f"{phase}/valid")
        X_train = np.hstack([X_train, values_train])
        X_valid = np.hstack([X_valid, values_valid])
        state["numeric_conditions"] = numeric.state_dict()
    elif numeric_contract is not None and numeric_contract.get("columns"):
        raise NestedFoldError("有数值条件契约却缺少原始 numeric_frame")
    if X_train.ndim != 2 or X_valid.ndim != 2 or not X_train.shape[1] or X_train.shape[1] != X_valid.shape[1]:
        raise NestedFoldError("折特征维度不一致")
    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise NestedFoldError("折特征含 NaN/inf")
    state["fit_sample_ids_sha256"] = _digest(ids[train].tolist())
    state["feature_dim"] = int(X_train.shape[1])
    return FoldMatrices(X_train, X_valid, state, ohe, numeric)
