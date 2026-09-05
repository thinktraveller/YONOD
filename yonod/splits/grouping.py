"""反应级 benchmark 分组策略。

三个策略都返回与输入行一一对应的稳定 group_id。任何无法解析的 SMILES
都会被显式编码为 fallback 组，而不是被静默删除或退回普通 KFold。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


class GroupingError(ValueError):
    """Raised when grouping inputs cannot define an auditable partition."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_smiles(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace(",", ".")
    return ".".join(part for part in text.split(".") if part)


def _canonical_smiles(value: Any) -> str | None:
    normalized = _normalize_smiles(value)
    if not normalized:
        return None
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def _fallback_id(kind: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return "fallback:{0}:{1}".format(kind, digest)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], field: str) -> None:
    if not columns:
        raise GroupingError("{0} 不能为空".format(field))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise GroupingError("数据集缺少 {0}：{1}".format(field, ", ".join(missing)))


def _component_groups(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    _require_columns(frame, columns, "grouping.component_cols")
    groups: List[str] = []
    for _, row in frame.loc[:, list(columns)].iterrows():
        values: List[str] = []
        for column in columns:
            raw = _normalize_smiles(row[column])
            canonical = _canonical_smiles(raw)
            values.append(canonical if canonical is not None else _fallback_id("invalid_component", raw))
        groups.append("component:" + "||".join(values))
    return pd.Series(groups, index=frame.index, dtype="string")


def _scaffold_groups(frame: pd.DataFrame, substrate_col: str) -> pd.Series:
    _require_columns(frame, [substrate_col], "grouping.substrate_col")
    groups: List[str] = []
    for raw_value in frame[substrate_col]:
        raw = _normalize_smiles(raw_value)
        molecule = Chem.MolFromSmiles(raw) if raw else None
        if molecule is None:
            groups.append(_fallback_id("invalid_scaffold", raw))
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        if scaffold:
            groups.append("scaffold:" + scaffold)
        else:
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            groups.append("fallback:acyclic:" + canonical)
    return pd.Series(groups, index=frame.index, dtype="string")


def _reaction_cluster_groups(
    frame: pd.DataFrame,
    sample_id_col: str,
    columns: Sequence[str],
    params: Mapping[str, Any],
) -> pd.Series:
    _require_columns(frame, columns, "grouping.reaction_cols")
    if sample_id_col not in frame.columns:
        raise GroupingError("数据集缺少 sample_id 列：{0}".format(sample_id_col))
    threshold = float(params.get("similarity_threshold", 0.7))
    if not 0.0 < threshold <= 1.0:
        raise GroupingError("grouping.similarity_threshold 必须处于 (0, 1]")
    radius = int(params.get("radius", 2))
    n_bits = int(params.get("n_bits", 2048))
    if radius < 0 or n_bits < 8:
        raise GroupingError("grouping.radius 必须 >= 0 且 grouping.n_bits 必须 >= 8")

    ordered = frame[[sample_id_col, *columns]].copy()
    ordered[sample_id_col] = ordered[sample_id_col].astype(str)
    ordered = ordered.sort_values(sample_id_col, kind="mergesort")
    fingerprints: List[Any] = []
    valid_sample_ids: List[str] = []
    fallback: Dict[str, str] = {}
    for _, row in ordered.iterrows():
        parts = [_normalize_smiles(row[column]) for column in columns]
        reaction_smiles = ".".join(part for part in parts if part)
        molecule = Chem.MolFromSmiles(reaction_smiles) if reaction_smiles else None
        sample_id = str(row[sample_id_col])
        if molecule is None:
            fallback[sample_id] = _fallback_id("invalid_reaction", reaction_smiles)
            continue
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits))
        valid_sample_ids.append(sample_id)

    assignments: Dict[str, str] = dict(fallback)
    if fingerprints:
        distances: List[float] = []
        for current in range(1, len(fingerprints)):
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[current], fingerprints[:current])
            distances.extend(1.0 - similarity for similarity in similarities)
        clusters = Butina.ClusterData(
            distances,
            len(fingerprints),
            1.0 - threshold,
            isDistData=True,
            reordering=False,
        )
        canonical_clusters = sorted(
            (tuple(sorted(valid_sample_ids[index] for index in cluster)) for cluster in clusters),
            key=lambda members: members[0],
        )
        for number, members in enumerate(canonical_clusters, start=1):
            group_id = "reaction-cluster:{0:04d}".format(number)
            assignments.update({sample_id: group_id for sample_id in members})

    return frame[sample_id_col].astype(str).map(assignments).astype("string")


def build_group_ids(
    frame: pd.DataFrame,
    sample_id_col: str,
    grouping: Mapping[str, Any],
    default_smiles_cols: Sequence[str],
) -> pd.Series:
    """Build one stable group_id per row for a declared grouping strategy."""
    strategy = str(grouping.get("strategy", "")).strip()
    if strategy == "component_holdout":
        columns = tuple(grouping.get("component_cols", ()))
        result = _component_groups(frame, columns)
    elif strategy == "substrate_scaffold":
        substrate_col = str(grouping.get("substrate_col", "")).strip()
        result = _scaffold_groups(frame, substrate_col)
    elif strategy == "reaction_fingerprint_cluster":
        columns = tuple(grouping.get("reaction_cols", default_smiles_cols))
        result = _reaction_cluster_groups(frame, sample_id_col, columns, grouping)
    else:
        raise GroupingError(
            "未知 grouping.strategy {0!r}；可用值：reaction_fingerprint_cluster、"
            "substrate_scaffold、component_holdout".format(strategy)
        )
    if result.isna().any():
        raise GroupingError("group_id 生成失败：存在空 group_id")
    return result.astype(str)
