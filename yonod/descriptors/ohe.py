"""Fold-local public one-hot feature transformer.

Unlike molecular descriptors, category vocabularies are learned data. This
object therefore intentionally does *not* inherit ``BaseDescriptor`` and
cannot be sent through a global ``descriptors/*.npz`` artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder


class OHEFeatureError(ValueError):
    """Raised when an OHE fold transformer violates its lifecycle contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_strings(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(values)).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_encoder(dtype: np.dtype) -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=dtype)
    except TypeError:  # pragma: no cover - depends on the installed sklearn
        return OneHotEncoder(sparse=False, handle_unknown="ignore", dtype=dtype)


class OHEFeature:
    """Ordered categorical OHE fitted only from the current training fold."""

    STATE_VERSION = 1

    def __init__(
        self,
        columns: Sequence[str],
        *,
        missing_policy: str = "as_category",
        dtype: str | np.dtype = "float32",
        handle_unknown: str = "ignore",
        missing_token: str = "__YONOD_MISSING_COMPONENT__",
    ) -> None:
        self.columns = tuple(str(column).strip() for column in columns)
        if not self.columns or any(not column for column in self.columns):
            raise OHEFeatureError("OHE columns 必须是非空列名列表")
        if len(set(self.columns)) != len(self.columns):
            raise OHEFeatureError("OHE columns 不能重复")
        if missing_policy not in {"as_category", "zero_block", "error"}:
            raise OHEFeatureError("missing_policy 仅支持 as_category、zero_block 或 error")
        if handle_unknown != "ignore":
            raise OHEFeatureError("OHE handle_unknown 当前固定为 ignore")
        self.missing_policy = missing_policy
        self.dtype = np.dtype(dtype)
        if self.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise OHEFeatureError("OHE dtype 仅支持 float32 或 float64")
        self.handle_unknown = handle_unknown
        self.missing_token = str(missing_token)
        self._encoder: Optional[OneHotEncoder] = None
        self._splits: list[tuple[int, int]] = []
        self._categories: list[set[str]] = []
        self._train_sample_ids_hash: Optional[str] = None
        self._reset_transform_audit()

    def _reset_transform_audit(self) -> None:
        self._stats: dict[str, int] = {
            "train_missing_count": 0,
            "valid_missing_count": 0,
            "predict_missing_count": 0,
            "valid_unseen_count": 0,
            "predict_unseen_count": 0,
        }

    def reset_transform_audit(self) -> None:
        """Clear per-transform counters after restoring a persisted fold state."""
        if self._encoder is None:
            raise OHEFeatureError("未拟合的 OHE 无法重置审计计数")
        self._reset_transform_audit()

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.columns if column not in frame.columns]
        if missing:
            raise OHEFeatureError("OHE 输入缺少类别列：{0}".format(", ".join(missing)))
        return frame.loc[:, list(self.columns)]

    def _missing_mask(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame[column].isna().to_numpy() | frame[column].astype(str).str.strip().eq("").to_numpy()
            for column in self.columns
        ])

    def _prepared_frame(self, frame: pd.DataFrame, missing_mask: np.ndarray) -> pd.DataFrame:
        if self.missing_policy == "error" and bool(missing_mask.any()):
            raise OHEFeatureError("OHE missing_policy=error 不允许缺失类别值")
        prepared = self._validate_frame(frame).copy()
        for index, column in enumerate(self.columns):
            values = prepared[column].astype(object)
            values.loc[missing_mask[:, index]] = self.missing_token
            prepared[column] = values.astype(str)
        return prepared

    def fit(self, train_frame: pd.DataFrame, *, sample_ids: Optional[Sequence[str]] = None) -> "OHEFeature":
        frame = self._validate_frame(train_frame)
        if frame.empty:
            raise OHEFeatureError("OHE 训练折不能为空")
        if sample_ids is not None and len(sample_ids) != len(frame):
            raise OHEFeatureError("OHE 训练 sample_ids 与行数不一致")
        missing_mask = self._missing_mask(frame)
        encoder = _make_encoder(self.dtype)
        encoder.fit(self._prepared_frame(frame, missing_mask))
        start = 0
        splits: list[tuple[int, int]] = []
        categories: list[set[str]] = []
        for values in encoder.categories_:
            end = start + len(values)
            splits.append((start, end))
            categories.append(set(str(value) for value in values))
            start = end
        self._encoder = encoder
        self._splits = splits
        self._categories = categories
        self._train_sample_ids_hash = _hash_strings([str(value) for value in sample_ids]) if sample_ids is not None else None
        self._reset_transform_audit()
        return self

    def transform(self, frame: pd.DataFrame, *, partition: str) -> np.ndarray:
        if self._encoder is None:
            raise OHEFeatureError("OHE 必须先在当前训练折 fit 后才能 transform")
        if partition not in {"train", "valid", "predict"}:
            raise OHEFeatureError("OHE partition 必须为 train、valid 或 predict")
        selected = self._validate_frame(frame)
        missing_mask = self._missing_mask(selected)
        prepared = self._prepared_frame(selected, missing_mask)
        matrix = np.asarray(self._encoder.transform(prepared), dtype=self.dtype)
        missing_count = int(missing_mask.sum())
        self._stats[f"{partition}_missing_count"] += missing_count
        if partition in {"valid", "predict"}:
            unseen_count = 0
            for index, column in enumerate(self.columns):
                values = prepared[column].to_numpy(dtype=str)
                unseen_count += sum(
                    value != self.missing_token and value not in self._categories[index]
                    for value in values
                )
            self._stats[f"{partition}_unseen_count"] += int(unseen_count)
        if self.missing_policy == "zero_block":
            for index, (start, end) in enumerate(self._splits):
                if missing_mask[:, index].any():
                    matrix[missing_mask[:, index], start:end] = 0.0
        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        if self._encoder is None:
            raise OHEFeatureError("OHE 必须 fit 后才能生成 feature names")
        names: list[str] = []
        for column, (start, end) in zip(self.columns, self._splits):
            names.extend(f"{column}::ohe_{index - start}" for index in range(start, end))
        return np.asarray(names, dtype=str)

    def metadata(self) -> dict[str, Any]:
        if self._encoder is None:
            raise OHEFeatureError("未拟合的 OHE 没有可审计 metadata")
        categories_payload = [sorted(values) for values in self._categories]
        return {
            "transformer": "OHEFeature",
            "state_version": self.STATE_VERSION,
            "fit_scope": "train_only_per_fold",
            "columns": list(self.columns),
            "component_cols": list(self.columns),
            "handle_unknown": self.handle_unknown,
            "missing_policy": self.missing_policy,
            "dtype": self.dtype.name,
            "sklearn_version": str(getattr(sklearn, "__version__", "unknown")),
            "categories_hash": hashlib.sha256(_canonical_json(categories_payload).encode("utf-8")).hexdigest(),
            "category_counts": [len(values) for values in self._categories],
            "output_dim": int(sum(len(values) for values in self._categories)),
            "train_sample_ids_hash": self._train_sample_ids_hash,
            **dict(self._stats),
        }

    def save(self, directory: Path | str, *, context: Optional[Mapping[str, Any]] = None) -> Path:
        """Atomically save a local fold state and readable audit metadata."""
        if self._encoder is None:
            raise OHEFeatureError("未拟合的 OHE 不能保存")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        state_path = target / "state.joblib"
        metadata_path = target / "metadata.json"
        fd, temporary_name = tempfile.mkstemp(prefix=".state-", suffix=".joblib", dir=target)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            joblib.dump(self, temporary)
            checksum = _sha256_file(temporary)
            os.replace(temporary, state_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        payload = {
            **self.metadata(),
            "context": dict(context or {}),
            "state_file": state_path.name,
            "state_sha256": checksum,
        }
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_metadata, metadata_path)
        return state_path

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        expected_context: Optional[Mapping[str, Any]] = None,
    ) -> "OHEFeature":
        """Restore state only when its identity exactly matches the requested fold."""
        target = Path(directory)
        metadata_path = target / "metadata.json"
        if not metadata_path.is_file():
            raise OHEFeatureError(f"OHE 状态元数据不存在：{metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            state_path = target / str(metadata["state_file"])
        except Exception as exc:
            raise OHEFeatureError(f"无法读取 OHE 状态元数据：{exc}") from exc
        if int(metadata.get("state_version", -1)) != cls.STATE_VERSION:
            raise OHEFeatureError("OHE 状态版本不兼容")
        if metadata.get("sklearn_version") != str(getattr(sklearn, "__version__", "unknown")):
            raise OHEFeatureError("OHE 状态的 sklearn 版本不兼容，请在当前环境重新拟合该折")
        if not state_path.is_file() or _sha256_file(state_path) != metadata.get("state_sha256"):
            raise OHEFeatureError("OHE 状态校验和不匹配")
        stored_context = metadata.get("context")
        if not isinstance(stored_context, Mapping):
            raise OHEFeatureError("OHE 状态缺少 context")
        for key, expected in dict(expected_context or {}).items():
            if stored_context.get(key) != expected:
                raise OHEFeatureError(f"OHE 状态 context 不匹配：{key}")
        try:
            restored = joblib.load(state_path)
        except Exception as exc:
            raise OHEFeatureError(f"无法恢复 OHE 状态：{exc}") from exc
        if not isinstance(restored, cls) or restored._encoder is None:
            raise OHEFeatureError("OHE 状态文件不是有效的 OHEFeature")
        if restored.metadata().get("train_sample_ids_hash") != metadata.get("train_sample_ids_hash"):
            raise OHEFeatureError("OHE 状态与元数据的训练样本哈希不一致")
        return restored
