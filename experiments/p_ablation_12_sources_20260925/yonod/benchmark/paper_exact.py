"""Paper-exact VJETHBKM 5x5 repeated-KFold benchmark helpers.

This module intentionally stays small: it defines auditable population
manifests, validates the paper split contract, enumerates fold tasks, and wraps
one manifest-defined fold execution.  Long-running RF/AutoGluon jobs can build
on these helpers without reimplementing split semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import BenchmarkContract
from .executor import (
    FoldExecutionError,
    FoldExecutionResult,
    PAPER_EXACT_EVALUATION_PROTOCOL,
    _contract_evaluation_protocol,
    execute_fold,
)


class PaperExactError(ValueError):
    """Raised when paper-exact data, split or execution contracts diverge."""


@dataclass(frozen=True)
class PaperExactPopulationSpec:
    population_id: str
    dataset_id: str
    feature_ids: tuple[str, ...]
    expected_rows: int
    label_col: str
    component_cols: tuple[str, ...]
    source: str
    notes: str = ""


@dataclass(frozen=True)
class PaperExactFoldTask:
    run_id: str
    population_id: str
    feature_id: str
    model: str
    repeat: int
    fold: int
    seed: int
    split_id: str
    split_hash: str

    @property
    def task_key(self) -> str:
        return "{0}:{1}:{2}:r{3:02d}:f{4:02d}".format(
            self.population_id, self.feature_id, self.model, self.repeat, self.fold
        )


@dataclass(frozen=True)
class PaperExactFoldRunResult:
    execution: FoldExecutionResult
    fold_metrics: Mapping[str, float]
    prediction_frame: pd.DataFrame


PAPER_EXACT_POPULATIONS: dict[str, PaperExactPopulationSpec] = {
    "bh1_paper_exact": PaperExactPopulationSpec(
        population_id="bh1_paper_exact",
        dataset_id="BH1",
        feature_ids=("mfp", "ohe"),
        expected_rows=3955,
        label_col="Yield",
        component_cols=("Aryl_halide_SMILES", "Additive_SMILES", "Base_SMILES", "Ligand_SMILES"),
        source="Data/HTE_datasets/BH1/BH1.csv",
        notes="论文官方 CSV 原始行顺序；MFP 与 OHE 共用总体。",
    ),
    "bh2_paper_exact": PaperExactPopulationSpec(
        population_id="bh2_paper_exact",
        dataset_id="BH2",
        feature_ids=("mfp", "ohe"),
        expected_rows=3359,
        label_col="Yield",
        component_cols=("Amine_SMILES", "Bromide_SMILES", "Ligand_SMILES", "Solvent_SMILES", "Base_SMILES"),
        source="Data/HTE_datasets/BH2/BH2.csv",
        notes="论文官方 CSV 原始行顺序；MFP 与 OHE 共用总体。",
    ),
    "sl1_paper_exact": PaperExactPopulationSpec(
        population_id="sl1_paper_exact",
        dataset_id="SL1",
        feature_ids=("mfp", "ohe"),
        expected_rows=1150,
        label_col="Yield",
        component_cols=("Aldehyde_1", "bifunctional_reagent", "Aldehyde_2"),
        source="Data/HTE_datasets/SL1/SL1.csv",
        notes="论文官方 CSV 原始行顺序；MFP 与 OHE 共用总体。",
    ),
    "sm_ohe_paper_exact_5760": PaperExactPopulationSpec(
        population_id="sm_ohe_paper_exact_5760",
        dataset_id="SM",
        feature_ids=("ohe",),
        expected_rows=5760,
        label_col="Yield",
        component_cols=("reactant_1_smiles", "reactant_2_smiles", "ligand_smiles", "reagent_1_smiles", "solvent_1_smiles"),
        source="Data/HTE_datasets/SM/SM.csv",
        notes="OHE 使用原始 SM CSV 5760 行总体。",
    ),
    "sm_mfp_paper_exact_4620": PaperExactPopulationSpec(
        population_id="sm_mfp_paper_exact_4620",
        dataset_id="SM",
        feature_ids=("mfp",),
        expected_rows=4620,
        label_col="Yield",
        component_cols=("reactant_1_smiles", "reactant_2_smiles", "ligand_smiles", "reagent_1_smiles", "solvent_1_smiles"),
        source="Gen_MFP.py --skip_rows_with_missing_values over Data/HTE_datasets/SM/SM.csv",
        notes="MFP 严格使用完整组件 4620 行总体，不得复用普通 5760 行 MFP。",
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def get_population_spec(population_id: str) -> PaperExactPopulationSpec:
    try:
        return PAPER_EXACT_POPULATIONS[population_id]
    except KeyError as exc:
        raise PaperExactError("未知 paper_exact population_id：{0}".format(population_id)) from exc


def build_population_manifest(
    frame: pd.DataFrame,
    spec: PaperExactPopulationSpec,
    *,
    dataset_sha256: str,
    sample_id_col: Optional[str] = None,
    source_row_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Return an auditable row manifest for one paper-exact population."""
    if len(frame) != int(spec.expected_rows):
        raise PaperExactError(
            "{0} 行数为 {1}，预期 {2}".format(spec.population_id, len(frame), spec.expected_rows)
        )
    required = [spec.label_col, *spec.component_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PaperExactError("{0} 缺少列：{1}".format(spec.population_id, ", ".join(missing)))
    if sample_id_col is not None:
        if sample_id_col not in frame.columns:
            raise PaperExactError("sample_id_col 不存在：{0}".format(sample_id_col))
        sample_ids = frame[sample_id_col].astype(str).tolist()
    else:
        sample_ids = ["{0}:row-{1:06d}".format(spec.population_id, index) for index in range(len(frame))]
    if len(set(sample_ids)) != len(sample_ids):
        raise PaperExactError("{0} 的 sample_id 不唯一".format(spec.population_id))
    if source_row_indices is None:
        source_rows = list(range(len(frame)))
    else:
        source_rows = [int(value) for value in source_row_indices]
        if len(source_rows) != len(frame):
            raise PaperExactError("source_row_indices 与总体行数不一致")
    row_payload = [
        {"sample_id": sample_id, "source_row_index": source_row}
        for sample_id, source_row in zip(sample_ids, source_rows)
    ]
    population_hash = _hash_payload({
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "dataset_sha256": dataset_sha256,
        "feature_ids": spec.feature_ids,
        "expected_rows": spec.expected_rows,
        "row_payload": row_payload,
    })
    result = pd.DataFrame({
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "sample_id": sample_ids,
        "source_row_index": source_rows,
        "dataset_sha256": dataset_sha256,
        "population_hash": population_hash,
        "label_col": spec.label_col,
        "component_cols_json": _canonical_json(list(spec.component_cols)),
        "feature_ids_json": _canonical_json(list(spec.feature_ids)),
        "y": pd.to_numeric(frame[spec.label_col], errors="raise").astype(float).to_numpy(),
    })
    return result


def validate_paper_exact_split_manifest(
    split_manifest: pd.DataFrame,
    population_manifest: pd.DataFrame,
    *,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1000,
) -> None:
    """Validate the literal paper split: five independent KFold seeds."""
    required = {
        "run_id", "split_id", "sample_id", "repeat", "fold", "role", "seed",
        "source_row_index", "dataset_sha256", "population_id", "split_hash",
    }
    missing = required.difference(split_manifest.columns)
    if missing:
        raise PaperExactError("paper_exact split manifest 缺少列：{0}".format(", ".join(sorted(missing))))
    if len(split_manifest[["repeat", "fold"]].drop_duplicates()) != n_repeats * n_splits:
        raise PaperExactError("paper_exact split manifest 必须包含 {0} 个 repeat/fold".format(n_repeats * n_splits))
    expected_seeds = set(range(seed, seed + n_repeats))
    actual_seeds = set(pd.to_numeric(split_manifest["seed"], errors="raise").astype(int).tolist())
    if actual_seeds != expected_seeds:
        raise PaperExactError("paper_exact seeds 不一致：{0}".format(sorted(actual_seeds)))
    expected_ids = population_manifest["sample_id"].astype(str).tolist()
    row_by_id = population_manifest.set_index("sample_id")["source_row_index"].astype(int).to_dict()
    populations = split_manifest["population_id"].dropna().astype(str).unique().tolist()
    expected_population = population_manifest["population_id"].dropna().astype(str).unique().tolist()
    if len(populations) != 1 or populations != expected_population:
        raise PaperExactError("split manifest 与 population manifest 的 population_id 不一致")
    if split_manifest["split_hash"].dropna().astype(str).nunique() != 1:
        raise PaperExactError("一个 paper_exact split manifest 必须只有一个 split_hash")
    for repeat, part in split_manifest.groupby("repeat", sort=True):
        valid_counts = part.loc[part["role"] == "valid", "sample_id"].astype(str).value_counts()
        if not valid_counts.reindex(expected_ids, fill_value=0).eq(1).all():
            raise PaperExactError("repeat={0} 中每个样本必须恰好一次进入验证集".format(repeat))
    unique_rows = split_manifest.drop_duplicates("sample_id").set_index("sample_id")["source_row_index"].astype(int).to_dict()
    if unique_rows != row_by_id:
        raise PaperExactError("split manifest 的 source_row_index 与 population manifest 不一致")


def enumerate_paper_exact_tasks(
    contract: BenchmarkContract,
    split_manifest: pd.DataFrame,
    *,
    feature_ids: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
) -> list[PaperExactFoldTask]:
    """Enumerate auditable RF/AutoGluon fold tasks from one immutable manifest."""
    protocol = _contract_evaluation_protocol(contract)
    if protocol != PAPER_EXACT_EVALUATION_PROTOCOL:
        raise PaperExactError("任务枚举需要 evaluation_protocol=paper_exact_5x5，当前为 {0}".format(protocol))
    features = tuple(feature_ids or contract.config.descriptors)
    model_names = tuple(models or contract.config.models)
    pairs = split_manifest[["split_id", "split_hash", "population_id", "repeat", "fold", "seed"]].drop_duplicates()
    pairs = pairs.sort_values(["repeat", "fold"], kind="mergesort")
    tasks: list[PaperExactFoldTask] = []
    for feature_id in features:
        for model in model_names:
            for row in pairs.itertuples(index=False):
                tasks.append(PaperExactFoldTask(
                    run_id=contract.run_id,
                    population_id=str(row.population_id),
                    feature_id=str(feature_id),
                    model=str(model),
                    repeat=int(row.repeat),
                    fold=int(row.fold),
                    seed=int(row.seed),
                    split_id=str(row.split_id),
                    split_hash=str(row.split_hash),
                ))
    return tasks


def kendall_tau_b(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Kendall's tau-b without requiring scipy at runtime."""
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    if len(x) != len(y):
        raise PaperExactError("Kendall tau 输入长度不一致")
    if len(x) < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x) - 1):
        dx = np.sign(x[i] - x[i + 1:])
        dy = np.sign(y[i] - y[i + 1:])
        concordant += int(np.sum((dx != 0) & (dy != 0) & (dx == dy)))
        discordant += int(np.sum((dx != 0) & (dy != 0) & (dx != dy)))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dx != 0) & (dy == 0)))
    denominator = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denominator == 0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def compute_fold_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(truth) != len(pred) or len(truth) == 0:
        raise PaperExactError("折级指标输入必须为等长非空向量")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise PaperExactError("折级指标输入包含 NaN 或 inf")
    return {
        "r2": float(r2_score(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "mae": float(mean_absolute_error(truth, pred)),
        "kendall_tau": kendall_tau_b(truth, pred),
    }


def execute_paper_exact_fold(
    contract: BenchmarkContract,
    split_manifest: pd.DataFrame,
    sample_ids: Sequence[str],
    X_smiles: np.ndarray,
    y: Sequence[float],
    feature_id: str,
    model: str,
    repeat: int,
    fold: int,
    *,
    X_numeric: Optional[np.ndarray] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    component_frame: Optional[pd.DataFrame] = None,
    fold_transformer: Optional[Any] = None,
) -> PaperExactFoldRunResult:
    """Execute one paper-exact fold and return prediction plus fold metrics."""
    protocol = _contract_evaluation_protocol(contract)
    if protocol != PAPER_EXACT_EVALUATION_PROTOCOL:
        raise PaperExactError("paper_exact fold runner 拒绝协议 {0}".format(protocol))
    result = execute_fold(
        contract,
        split_manifest,
        sample_ids,
        X_smiles,
        y,
        feature_id,
        model,
        repeat,
        fold,
        X_numeric=X_numeric,
        model_kwargs=model_kwargs,
        component_frame=component_frame,
        fold_transformer=fold_transformer,
    )
    prediction = pd.read_parquet(result.prediction_path)
    if "evaluation_protocol" not in prediction.columns or set(prediction["evaluation_protocol"]) != {PAPER_EXACT_EVALUATION_PROTOCOL}:
        raise PaperExactError("paper_exact 预测分片缺少协议字段或协议不一致")
    metrics = compute_fold_metrics(prediction["y_true"].to_numpy(), prediction["y_pred"].to_numpy())
    return PaperExactFoldRunResult(result, metrics, prediction)


__all__ = [
    "PAPER_EXACT_EVALUATION_PROTOCOL",
    "PAPER_EXACT_POPULATIONS",
    "PaperExactError",
    "PaperExactFoldRunResult",
    "PaperExactFoldTask",
    "PaperExactPopulationSpec",
    "build_population_manifest",
    "compute_fold_metrics",
    "enumerate_paper_exact_tasks",
    "execute_paper_exact_fold",
    "get_population_spec",
    "kendall_tau_b",
    "sha256_file",
    "validate_paper_exact_split_manifest",
]
