"""Compatibility adapters for strict benchmark fold-local transformers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ..descriptors.ohe import OHEFeature, OHEFeatureError


class FoldPreprocessorError(OHEFeatureError):
    """Legacy strict-benchmark exception name."""


class ReactionComponentOHE(OHEFeature):
    """Paper-compatible alias of :class:`OHEFeature`.

    The strict paper protocol retains its ``zero_block`` missing-component
    semantics and historic metadata keys, while sharing the public encoding
    implementation used by ordinary tasks.
    """

    def __init__(self, component_cols: Sequence[str], *, missing_token: str = "__YONOD_MISSING_COMPONENT__") -> None:
        try:
            super().__init__(
                component_cols,
                missing_policy="zero_block",
                dtype="float64",
                handle_unknown="ignore",
                missing_token=missing_token,
            )
        except OHEFeatureError as exc:
            raise FoldPreprocessorError(str(exc)) from exc
        self.component_cols = self.columns

    def fit(self, train_component_frame: pd.DataFrame, *, sample_ids: Optional[Sequence[str]] = None) -> "ReactionComponentOHE":
        try:
            super().fit(train_component_frame, sample_ids=sample_ids)
        except OHEFeatureError as exc:
            raise FoldPreprocessorError(str(exc)) from exc
        return self

    def transform(self, component_frame: pd.DataFrame, *, partition: str):
        try:
            return super().transform(component_frame, partition=partition)
        except OHEFeatureError as exc:
            raise FoldPreprocessorError(str(exc)) from exc

    def metadata(self) -> Dict[str, Any]:
        try:
            result = super().metadata()
        except OHEFeatureError as exc:
            raise FoldPreprocessorError(str(exc)) from exc
        result.update({
            "transformer": "ReactionComponentOHE",
            "missing_component_policy": "encode_then_zero_component_block",
            "train_missing_block_count": result.pop("train_missing_count"),
            "valid_missing_block_count": result.pop("valid_missing_count"),
            "valid_unseen_component_count": result.pop("valid_unseen_count"),
        })
        result.pop("predict_missing_count", None)
        result.pop("predict_unseen_count", None)
        return result
