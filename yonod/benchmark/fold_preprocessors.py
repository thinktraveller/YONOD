"""Feature transformers whose state must be learned inside each CV fold."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FoldPreprocessorError(ValueError):
    """Raised when a fold-local transformer cannot preserve its contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_strings(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(values)).encode("utf-8")).hexdigest()


def _make_encoder() -> OneHotEncoder:
    """Support both current and older sklearn sparse-output keyword names."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # pragma: no cover - depends on the installed sklearn
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


class ReactionComponentOHE:
    """Fold-local component identity encoder used by the literature baseline.

    Categories are learnt only in :meth:`fit`.  A raw empty/missing component
    is fed through a private sentinel to retain a stable transform shape, then
    its complete component block is zeroed so missingness is not a chemical
    identity feature.  Unseen validation values rely on sklearn's
    ``handle_unknown='ignore'`` and are audited rather than fitted globally.
    """

    def __init__(
        self,
        component_cols: Sequence[str],
        *,
        missing_token: str = "__YONOD_MISSING_COMPONENT__",
    ) -> None:
        columns = tuple(str(column) for column in component_cols)
        if not columns or any(not column for column in columns):
            raise FoldPreprocessorError("OHE component_cols 必须是非空列名列表")
        if len(set(columns)) != len(columns):
            raise FoldPreprocessorError("OHE component_cols 不能重复")
        self.component_cols = columns
        self.missing_token = str(missing_token)
        self._encoder: Optional[OneHotEncoder] = None
        self._splits: list[tuple[int, int]] = []
        self._categories: list[set[str]] = []
        self._train_sample_ids_hash: Optional[str] = None
        self._stats: Dict[str, int] = {
            "train_missing_block_count": 0,
            "valid_missing_block_count": 0,
            "valid_unseen_component_count": 0,
        }

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.component_cols if column not in frame.columns]
        if missing:
            raise FoldPreprocessorError("OHE 输入缺少组件列：{0}".format(", ".join(missing)))
        return frame.loc[:, list(self.component_cols)]

    def _missing_mask(self, frame: pd.DataFrame) -> np.ndarray:
        masks = []
        for column in self.component_cols:
            series = frame[column]
            masks.append(series.isna().to_numpy() | series.astype(str).str.strip().eq("").to_numpy())
        return np.column_stack(masks)

    def _prepared_frame(self, frame: pd.DataFrame, missing_mask: np.ndarray) -> pd.DataFrame:
        prepared = self._validate_frame(frame).copy()
        for index, column in enumerate(self.component_cols):
            values = prepared[column].astype(object)
            values.loc[missing_mask[:, index]] = self.missing_token
            prepared[column] = values.astype(str)
        return prepared

    def fit(
        self,
        train_component_frame: pd.DataFrame,
        *,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> "ReactionComponentOHE":
        frame = self._validate_frame(train_component_frame)
        if frame.empty:
            raise FoldPreprocessorError("OHE 训练折不能为空")
        if sample_ids is not None and len(sample_ids) != len(frame):
            raise FoldPreprocessorError("OHE 训练 sample_ids 与行数不一致")
        missing_mask = self._missing_mask(frame)
        encoder = _make_encoder()
        encoder.fit(self._prepared_frame(frame, missing_mask))
        splits = []
        start = 0
        categories: list[set[str]] = []
        for values in encoder.categories_:
            end = start + len(values)
            splits.append((start, end))
            categories.append(set(str(value) for value in values))
            start = end
        self._encoder = encoder
        self._splits = splits
        self._categories = categories
        self._stats = {
            "train_missing_block_count": 0,
            "valid_missing_block_count": 0,
            "valid_unseen_component_count": 0,
        }
        self._train_sample_ids_hash = _hash_strings([str(value) for value in sample_ids]) if sample_ids is not None else None
        return self

    def transform(self, component_frame: pd.DataFrame, *, partition: str) -> np.ndarray:
        if self._encoder is None:
            raise FoldPreprocessorError("OHE 必须先在当前训练折 fit 后才能 transform")
        if partition not in {"train", "valid"}:
            raise FoldPreprocessorError("OHE partition 必须为 'train' 或 'valid'")
        frame = self._validate_frame(component_frame)
        missing_mask = self._missing_mask(frame)
        prepared = self._prepared_frame(frame, missing_mask)
        matrix = np.asarray(self._encoder.transform(prepared), dtype=np.float64)
        missing_count = int(missing_mask.sum())
        if partition == "train":
            self._stats["train_missing_block_count"] += missing_count
        else:
            self._stats["valid_missing_block_count"] += missing_count
            unseen_count = 0
            for index, column in enumerate(self.component_cols):
                raw_values = prepared[column].to_numpy(dtype=str)
                unseen_count += sum(
                    value != self.missing_token and value not in self._categories[index]
                    for value in raw_values
                )
            self._stats["valid_unseen_component_count"] += int(unseen_count)
        for index, (start, end) in enumerate(self._splits):
            if missing_mask[:, index].any():
                matrix[missing_mask[:, index], start:end] = 0.0
        return matrix

    def metadata(self) -> Dict[str, Any]:
        if self._encoder is None:
            raise FoldPreprocessorError("未拟合的 OHE 没有可审计 metadata")
        categories_payload = [sorted(values) for values in self._categories]
        return {
            "transformer": "ReactionComponentOHE",
            "fit_scope": "train_only_per_fold",
            "component_cols": list(self.component_cols),
            "handle_unknown": "ignore",
            "missing_component_policy": "encode_then_zero_component_block",
            "categories_hash": hashlib.sha256(
                _canonical_json(categories_payload).encode("utf-8")
            ).hexdigest(),
            "category_counts": [len(values) for values in self._categories],
            "output_dim": int(sum(len(values) for values in self._categories)),
            "train_sample_ids_hash": self._train_sample_ids_hash,
            **dict(self._stats),
        }
