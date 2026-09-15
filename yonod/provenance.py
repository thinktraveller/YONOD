"""Small, shared content identities for schema-2 feature/training boundaries."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import pandas as pd


def _normalise(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    # CSV-oriented feature inputs are scalar.  Preserve a stable string view
    # for values such as RDKit SMILES and categorical condition identifiers.
    return str(value)


def feature_dataset_identity(
    frame: pd.DataFrame,
    *,
    sample_id_col: str,
    column_roles: Mapping[str, Any],
) -> str:
    """Hash only sample IDs and feature-source columns, never the label.

    This separation is what permits a corrected target column to create a new
    training result while safely reusing static descriptors that did not depend
    on the target.  Conditions are training-time auxiliary inputs, so they are
    likewise excluded from static-feature identity.
    """
    columns: list[str] = []
    for role in ("reactants", "products", "others", "categoricals"):
        columns.extend(str(item) for item in (column_roles.get(role, []) or []))
    columns = list(dict.fromkeys(columns))
    required = [sample_id_col, *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("计算特征数据身份时缺少列：" + ", ".join(missing))
    records = []
    for _, row in frame.loc[:, required].iterrows():
        records.append({column: _normalise(row[column]) for column in required})
    payload = {"sample_id_col": sample_id_col, "feature_source_columns": columns, "records": records}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
