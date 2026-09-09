#!/usr/bin/env python3
"""只读盘点 VJETHBKM Compare_Complexity 的官方静态描述符。

本脚本对应 ``project-docs/project-plan.md`` 的步骤 28.1。它不截断、
不重排、更不改写作者提供的 NPZ；只把可复核的输入身份和对齐证据写入
一个隔离的 ``derived/`` 目录。特别地，BH1/DFT 的 3960 与 3955 行差异
没有作者提供的行级标识时必须保持 ``blocked_alignment``。

示例：
    python scripts/audit_official_static_descriptors.py
    python scripts/audit_official_static_descriptors.py --verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_SCHEMA_VERSION = "1.0"
DESCRIPTORS = ("DFT", "SOAP", "PhysChem", "MFP")
IDENTIFIER_KEY_CANDIDATES = (
    "sample_id",
    "sample_ids",
    "source_row_index",
    "source_row_indices",
    "row_index",
    "row_indices",
    "index",
    "indices",
)


@dataclass(frozen=True)
class DatasetSpec:
    """不含任何推断性截断规则的官方输入规格。"""

    dataset_id: str
    source_folder: str
    prefix: str
    target_unit: str
    source_csv: str
    source_target_column: str
    source_required_columns: tuple[str, ...]
    source_selection_rule: str
    source_generator_evidence: tuple[str, ...]


DATASETS = (
    DatasetSpec(
        dataset_id="BH1",
        source_folder="Doyle_2018",
        prefix="Doyle",
        target_unit="yield_percent",
        source_csv="reference-proejct/vjethbkm/yieldsmarter/Data/HTE_datasets/BH1/BH1.csv",
        source_target_column="Yield",
        source_required_columns=(),
        source_selection_rule="保留 BH1.csv 的全部行，原始 CSV 行号从 0 开始。",
        source_generator_evidence=(
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH1_ConfGen.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH1_SOAP.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH1_PhysChem.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/Gen_MFP.py",
        ),
    ),
    DatasetSpec(
        dataset_id="BH2",
        source_folder="Denmark_2023",
        prefix="Denmark",
        target_unit="yield_percent",
        source_csv="reference-proejct/vjethbkm/yieldsmarter/Data/HTE_datasets/BH2/BH2.csv",
        source_target_column="Yield",
        source_required_columns=(),
        source_selection_rule="保留 BH2.csv 的全部行，原始 CSV 行号从 0 开始。",
        source_generator_evidence=(
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH2_DFT.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH2_ConfGen.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/BH2_PhysChem.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/Gen_MFP.py",
        ),
    ),
    DatasetSpec(
        dataset_id="SL1",
        source_folder="Bode_2023",
        prefix="Bode",
        target_unit="lc_ms_product_ratio",
        source_csv="reference-proejct/vjethbkm/yieldsmarter/Data/HTE_datasets/SL1/SL1.csv",
        source_target_column="Yield",
        source_required_columns=(),
        source_selection_rule="保留 SL1.csv 的全部行，原始 CSV 行号从 0 开始。",
        source_generator_evidence=(
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/SL1_ConfGen.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/SL1_PhysChem.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/Gen_MFP.py",
        ),
    ),
    DatasetSpec(
        dataset_id="SM",
        source_folder="Suzuki_2018",
        prefix="Suzuki",
        target_unit="yield_percent",
        source_csv="reference-proejct/vjethbkm/yieldsmarter/Data/HTE_datasets/SM/SM.csv",
        source_target_column="Yield",
        source_required_columns=(
            "reactant_1_smiles",
            "reactant_2_smiles",
            "ligand_smiles",
            "reagent_1_smiles",
            "solvent_1_smiles",
        ),
        source_selection_rule=(
            "仅保留 SM.csv 中五个静态特征组分及 Yield 都非空的行；"
            "保留其原始 CSV 行号，不把 5760 行 OHE 总体混入。"
        ),
        source_generator_evidence=(
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/SM_ConfGen.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/SM_PhysChem.json",
            "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/Gen_MFP.py",
        ),
    ),
)

EXPECTED_X_SHAPES = {
    ("BH1", "DFT"): (3960, 120),
    ("BH1", "SOAP"): (3955, 420),
    ("BH1", "PhysChem"): (3955, 15),
    ("BH1", "MFP"): (3955, 4096),
    ("BH2", "DFT"): (3359, 130),
    ("BH2", "SOAP"): (3359, 525),
    ("BH2", "PhysChem"): (3359, 36),
    ("BH2", "MFP"): (3359, 5120),
    ("SL1", "DFT"): (1150, 99),
    ("SL1", "SOAP"): (1150, 315),
    ("SL1", "PhysChem"): (1150, 39),
    ("SL1", "MFP"): (1150, 3072),
    ("SM", "DFT"): (4620, 145),
    ("SM", "SOAP"): (4620, 525),
    ("SM", "PhysChem"): (4620, 20),
    ("SM", "MFP"): (4620, 5120),
}


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """哈希同时覆盖 dtype、shape 与 C-order 数值，避免仅比字节的歧义。"""

    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_int_sequence(values: Iterable[int]) -> str:
    return sha256_array(np.asarray(tuple(values), dtype=np.int64))


def _nonempty(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def read_source_candidate(repo_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    """按作者随附脚本中可审计的缺失值规则构造 *候选* 行号，不改 NPZ。"""

    csv_path = repo_root / spec.source_csv
    selected_indices: list[int] = []
    selected_y: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {csv_path}")
        required = (spec.source_target_column, *spec.source_required_columns)
        absent = [column for column in required if column not in reader.fieldnames]
        if absent:
            raise ValueError(f"CSV 缺少必要列 {absent}: {csv_path}")
        source_row_count = 0
        for source_row_index, row in enumerate(reader):
            source_row_count += 1
            if all(_nonempty(row[column]) for column in required):
                selected_indices.append(source_row_index)
                selected_y.append(float(row[spec.source_target_column]))

    y = np.asarray(selected_y, dtype=np.float64)
    return {
        "source_csv": spec.source_csv,
        "source_csv_sha256": sha256_file(csv_path),
        "source_csv_row_count": source_row_count,
        "source_target_column": spec.source_target_column,
        "source_selection_rule": spec.source_selection_rule,
        "candidate_row_count": int(y.shape[0]),
        "candidate_source_row_index_sha256": sha256_int_sequence(selected_indices),
        "candidate_y_sha256": sha256_array(y),
        # 只记录摘要；完整行映射将由步骤 28.2 在通过 gate 后生成。
        "candidate_source_row_index_preview": selected_indices[:10],
    }, y


def _identifier_keys(keys: Iterable[str]) -> list[str]:
    lower_to_original = {key.lower(): key for key in keys}
    return [
        lower_to_original[candidate]
        for candidate in IDENTIFIER_KEY_CANDIDATES
        if candidate in lower_to_original
    ]


def generator_evidence_for(spec: DatasetSpec, descriptor: str) -> dict[str, Any]:
    """区分“能说明候选来源”与“NPZ 内确有行级 ID”两种证据。"""

    root = "reference-proejct/vjethbkm/yieldsmarter/Src/Featurize/"
    config_by_dataset = {
        "BH1": {"SOAP": "BH1_SOAP.json", "PhysChem": "BH1_PhysChem.json"},
        "BH2": {
            "DFT": "BH2_DFT.json",
            "SOAP": "BH2_ConfGen.json",
            "PhysChem": "BH2_PhysChem.json",
        },
        "SL1": {"SOAP": "SL1_ConfGen.json", "PhysChem": "SL1_PhysChem.json"},
        "SM": {"SOAP": "SM_ConfGen.json", "PhysChem": "SM_PhysChem.json"},
    }
    if descriptor == "MFP":
        return {
            "descriptor_generator_code": [root + "Gen_MFP.py"],
            "descriptor_generator_config": None,
            "evidence_scope": "通用生成器说明其按输入表顺序处理；包内没有每个MFP产物的运行配置。",
        }
    config_name = config_by_dataset.get(spec.dataset_id, {}).get(descriptor)
    if config_name is None:
        return {
            "descriptor_generator_code": [],
            "descriptor_generator_config": None,
            "evidence_scope": (
                "当前随附包没有该数据集/描述符的专属生成配置；"
                "不得把其他描述符的配置外推为行级映射证据。"
            ),
        }
    code_name = {
        "DFT": "Gen_DFT.py",
        "SOAP": "Gen_SOAP.py",
        "PhysChem": "Gen_PhysChem.py",
    }[descriptor]
    return {
        "descriptor_generator_code": [root + code_name],
        "descriptor_generator_config": root + config_name,
        "evidence_scope": "配置可说明候选输入 CSV/处理规则；NPZ 是否可配对仍以行级标识与步骤28.2校验为准。",
    }


def npz_record(
    *,
    repo_root: Path,
    compare_root: Path,
    spec: DatasetSpec,
    descriptor: str,
    source_candidate: dict[str, Any],
    source_y: np.ndarray,
    read_at_utc: str,
) -> tuple[dict[str, Any], np.ndarray]:
    npz_path = compare_root / spec.source_folder / descriptor / f"{spec.prefix}_{descriptor}.npz"
    result_source = (
        compare_root
        / spec.source_folder
        / descriptor
        / f"{spec.prefix}_{descriptor}_metrics_summaries_RF.json"
    )
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not result_source.is_file():
        raise FileNotFoundError(result_source)

    with np.load(npz_path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if "X" not in archive or "y" not in archive:
            raise ValueError(f"{npz_path} 必须包含 X/y，实际键为 {keys}")
        x = archive["X"]
        y = archive["y"]
        embedded_identifier_keys = _identifier_keys(keys)

    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"非法 X/y 形状 {npz_path}: X={x.shape}, y={y.shape}; 预期 X=(n,d), y=(n,)"
        )
    source_y_equal = bool(y.shape == source_y.shape and np.array_equal(y, source_y))
    mapping_status = (
        "candidate_source_order_only"
        if source_y_equal and not embedded_identifier_keys
        else "embedded_identifier_present"
        if embedded_identifier_keys
        else "unresolved_source_mapping"
    )
    return (
        {
            "dataset_id": spec.dataset_id,
            "descriptor": descriptor,
            "source_npz": _relative(repo_root, npz_path),
            "source_npz_sha256": sha256_file(npz_path),
            "npz_keys": keys,
            "x_key": "X",
            "x_shape": list(x.shape),
            "x_dtype": str(x.dtype),
            "y_key": "y",
            "y_shape": list(y.shape),
            "y_dtype": str(y.dtype),
            "y_sha256": sha256_array(y),
            "target_unit": spec.target_unit,
            "official_result_source": _relative(repo_root, result_source),
            "official_result_source_sha256": sha256_file(result_source),
            "read_at_utc": read_at_utc,
            "sample_mapping": {
                "npz_embedded_identifier_keys": embedded_identifier_keys,
                "mapping_available_in_npz": bool(embedded_identifier_keys),
                "descriptor_generator_evidence": generator_evidence_for(spec, descriptor),
                **source_candidate,
                "source_candidate_y_exact_match": source_y_equal,
                "alignment_basis": (
                    "NPZ 不含 sample_id/source_row_index；仅有作者生成脚本所指 CSV "
                    "与目标向量的逐元素相等候选证据，不能单独证明反应身份。"
                    if not embedded_identifier_keys
                    else "NPZ 含候选标识键；仍需步骤28.2验证其唯一性与取值。"
                ),
                "mapping_status": mapping_status,
            },
            "row_selection_performed": False,
            "primary_gate_status": "pending_dataset_audit",
        },
        y,
    )


def dataset_audit(entries: list[dict[str, Any]], spec: DatasetSpec) -> dict[str, Any]:
    by_descriptor = {entry["descriptor"]: entry for entry in entries}
    reference = by_descriptor["MFP"]
    reference_y_hash = reference["y_sha256"]
    reference_rows = reference["x_shape"][0]
    comparisons = []
    for descriptor in DESCRIPTORS:
        entry = by_descriptor[descriptor]
        comparisons.append(
            {
                "descriptor": descriptor,
                "row_count": entry["x_shape"][0],
                "same_row_count_as_mfp": entry["x_shape"][0] == reference_rows,
                "same_y_sha256_as_mfp": entry["y_sha256"] == reference_y_hash,
                "y_exact_match_uses": "NPZ y dtype/shape/value canonical hash",
            }
        )

    if spec.dataset_id == "BH1":
        dft = by_descriptor["DFT"]
        for entry in entries:
            if entry["descriptor"] == "DFT":
                entry["primary_gate_status"] = "blocked_alignment"
            else:
                entry["primary_gate_status"] = "candidate_population_3955_pending_step28_2"
        return {
            "dataset_id": "BH1",
            "candidate_population_id": "BH1-static-3955",
            "candidate_population_row_count": reference_rows,
            "cross_descriptor_y_audit": comparisons,
            "bh1_dft_gate_status": "blocked_alignment",
            "main_matrix_inclusion": {
                "DFT": False,
                "SOAP": True,
                "PhysChem": True,
                "MFP": True,
            },
            "reason": (
                f"BH1/DFT 为 {dft['x_shape'][0]} 行，MFP/SOAP/PhysChem 为 {reference_rows} 行；"
                "DFT NPZ 只有 X/y，包内没有 BH1 DFT 专属生成配置或 sample_id/source_row_index。"
                "禁止用前3955行、按 y 值猜测或任何未证明的删行规则补齐。"
            ),
        }

    if spec.dataset_id == "SM":
        for entry in entries:
            entry["primary_gate_status"] = "candidate_static_4620_pending_step28_2"
        return {
            "dataset_id": "SM",
            "candidate_population_id": "SM-static-4620",
            "candidate_population_row_count": reference_rows,
            "cross_descriptor_y_audit": comparisons,
            "main_matrix_inclusion": {descriptor: True for descriptor in DESCRIPTORS},
            "separate_from_population_id": "SM-OHE-5760",
            "reason": (
                "SM 静态矩阵均为 4620 行；不能与使用原始 SM.csv 的 5760 行 OHE 总体"
                "合并统计、排名或平均。"
            ),
        }

    for entry in entries:
        entry["primary_gate_status"] = "candidate_population_pending_step28_2"
    return {
        "dataset_id": spec.dataset_id,
        "candidate_population_id": f"{spec.dataset_id}-static-{reference_rows}",
        "candidate_population_row_count": reference_rows,
        "cross_descriptor_y_audit": comparisons,
        "main_matrix_inclusion": {descriptor: True for descriptor in DESCRIPTORS},
        "reason": (
            "四个静态矩阵的行数与 y 哈希摘要已盘点；仍须在步骤28.2生成"
            "共享 sample_id/source_row_index population manifest 后才可开始建模。"
        ),
    }


def bh1_dft_alignment_audit(inventory: dict[str, Any]) -> dict[str, Any]:
    entries = {
        entry["descriptor"]: entry
        for entry in inventory["entries"]
        if entry["dataset_id"] == "BH1"
    }
    dft = entries["DFT"]
    reference = entries["MFP"]
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "dataset_id": "BH1",
        "descriptor": "DFT",
        "gate_status": "blocked_alignment",
        "main_experiment_allowed": False,
        "dft_row_count": dft["x_shape"][0],
        "reference_descriptor": "MFP",
        "reference_row_count": reference["x_shape"][0],
        "unresolved_extra_row_count": dft["x_shape"][0] - reference["x_shape"][0],
        "extra_dft_source_row_indices": None,
        "extra_row_identity_status": "not_identifiable_without_author_row_level_mapping",
        "npz_embedded_identifier_keys": dft["sample_mapping"]["npz_embedded_identifier_keys"],
        "source_candidate_y_exact_match": dft["sample_mapping"]["source_candidate_y_exact_match"],
        "permitted_operations": ["只读盘点", "保留完整3960行NPZ", "等待作者行级映射证据"],
        "prohibited_operations": [
            "截取前3955行",
            "按 y 值或行数猜测五个待删行",
            "从3960行重排、删除或构造主比较特征矩阵",
        ],
        "conclusion": (
            "当前包内证据无法唯一标注五条多出 DFT 行的反应身份，也无法证明删除后"
            "与3955行共同总体的样本顺序完全一致；因此维持 blocked_alignment。"
        ),
    }


def population_boundary_audit(repo_root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    sm_entries = [entry for entry in inventory["entries"] if entry["dataset_id"] == "SM"]
    ohe_summary = Path(
        "reference-proejct/vjethbkm/yieldsmarter/Results/Compare_Complexity/"
        "Suzuki_2018/OHE/Suzuki_OHE_metrics_summaries_RF.json"
    )
    raw_sm = Path("reference-proejct/vjethbkm/yieldsmarter/Data/HTE_datasets/SM/SM.csv")
    ohe_payload = json.loads((repo_root / ohe_summary).read_text(encoding="utf-8"))
    if ohe_payload.get("reaction_csv") != "../../Data/HTE_datasets/SM/SM.csv":
        raise ValueError("SM OHE 官方汇总的 reaction_csv 与预期不一致")
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "populations": [
            {
                "population_id": "SM-static-4620",
                "kind": "main_static_descriptor_candidate",
                "row_count": 4620,
                "members": [entry["descriptor"] for entry in sm_entries],
                "member_row_counts": {
                    entry["descriptor"]: entry["x_shape"][0] for entry in sm_entries
                },
                "status": "candidate_pending_step28_2",
            },
            {
                "population_id": "SM-OHE-5760",
                "kind": "supplementary_fold_local_ohe",
                "row_count": 5760,
                "official_summary": ohe_summary.as_posix(),
                "official_summary_sha256": sha256_file(repo_root / ohe_summary),
                "raw_source_csv": raw_sm.as_posix(),
                "raw_source_csv_sha256": sha256_file(repo_root / raw_sm),
                "fold_fit_scope": ohe_payload.get("ohe", {}).get("fit_scope"),
                "status": "separate_supplementary_population",
            },
        ],
        "separation_rule": (
            "SM-static-4620 与 SM-OHE-5760 不得进入同一主统计表、排名、平均值、"
            "shared split manifest 或模型/描述符主效应估计。"
        ),
        "mixed_population_allowed": False,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset_id",
        "descriptor",
        "source_npz",
        "source_npz_sha256",
        "npz_keys",
        "x_key",
        "x_shape",
        "x_dtype",
        "y_key",
        "y_shape",
        "y_dtype",
        "y_sha256",
        "target_unit",
        "official_result_source",
        "official_result_source_sha256",
        "source_csv",
        "source_csv_sha256",
        "source_candidate_row_count",
        "source_candidate_y_sha256",
        "source_candidate_y_exact_match",
        "npz_embedded_identifier_keys",
        "mapping_status",
        "alignment_basis",
        "primary_gate_status",
        "row_selection_performed",
        "read_at_utc",
    )
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for entry in rows:
                mapping = entry["sample_mapping"]
                writer.writerow(
                    {
                        "dataset_id": entry["dataset_id"],
                        "descriptor": entry["descriptor"],
                        "source_npz": entry["source_npz"],
                        "source_npz_sha256": entry["source_npz_sha256"],
                        "npz_keys": json.dumps(entry["npz_keys"], ensure_ascii=False),
                        "x_key": entry["x_key"],
                        "x_shape": json.dumps(entry["x_shape"]),
                        "x_dtype": entry["x_dtype"],
                        "y_key": entry["y_key"],
                        "y_shape": json.dumps(entry["y_shape"]),
                        "y_dtype": entry["y_dtype"],
                        "y_sha256": entry["y_sha256"],
                        "target_unit": entry["target_unit"],
                        "official_result_source": entry["official_result_source"],
                        "official_result_source_sha256": entry["official_result_source_sha256"],
                        "source_csv": mapping["source_csv"],
                        "source_csv_sha256": mapping["source_csv_sha256"],
                        "source_candidate_row_count": mapping["candidate_row_count"],
                        "source_candidate_y_sha256": mapping["candidate_y_sha256"],
                        "source_candidate_y_exact_match": mapping["source_candidate_y_exact_match"],
                        "npz_embedded_identifier_keys": json.dumps(
                            mapping["npz_embedded_identifier_keys"], ensure_ascii=False
                        ),
                        "mapping_status": mapping["mapping_status"],
                        "alignment_basis": mapping["alignment_basis"],
                        "primary_gate_status": entry["primary_gate_status"],
                        "row_selection_performed": entry["row_selection_performed"],
                        "read_at_utc": entry["read_at_utc"],
                    }
                )
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def build_inventory(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    compare_root = repo_root / "reference-proejct/vjethbkm/yieldsmarter/Results/Compare_Complexity"
    if not compare_root.is_dir():
        raise FileNotFoundError(compare_root)
    read_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for spec in DATASETS:
        source_candidate, source_y = read_source_candidate(repo_root, spec)
        dataset_entries: list[dict[str, Any]] = []
        for descriptor in DESCRIPTORS:
            entry, _ = npz_record(
                repo_root=repo_root,
                compare_root=compare_root,
                spec=spec,
                descriptor=descriptor,
                source_candidate=source_candidate,
                source_y=source_y,
                read_at_utc=read_at_utc,
            )
            dataset_entries.append(entry)
        entries.extend(dataset_entries)
        audits.append(dataset_audit(dataset_entries, spec))

    inventory = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "purpose": "步骤28.1官方静态描述符只读inventory；不执行截断、重排或建模。",
        "read_at_utc": read_at_utc,
        "source_root": _relative(repo_root, compare_root),
        "expected_matrix_count": len(DATASETS) * len(DESCRIPTORS),
        "entries": entries,
        "dataset_audits": audits,
    }
    paths = {
        "inventory_json": output_dir / "official_static_descriptor_inventory.json",
        "inventory_csv": output_dir / "official_static_descriptor_inventory.csv",
        "bh1_dft_alignment_json": output_dir / "bh1_dft_alignment_audit.json",
        "population_boundary_json": output_dir / "population_boundary_audit.json",
    }
    _atomic_json(paths["inventory_json"], inventory)
    _atomic_csv(paths["inventory_csv"], entries)
    _atomic_json(paths["bh1_dft_alignment_json"], bh1_dft_alignment_audit(inventory))
    _atomic_json(paths["population_boundary_json"], population_boundary_audit(repo_root, inventory))
    return paths


def verify_inventory(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = {
        "inventory_json": output_dir / "official_static_descriptor_inventory.json",
        "inventory_csv": output_dir / "official_static_descriptor_inventory.csv",
        "bh1_dft_alignment_json": output_dir / "bh1_dft_alignment_audit.json",
        "population_boundary_json": output_dir / "population_boundary_audit.json",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"缺少审计产物: {absent}")
    inventory = json.loads(paths["inventory_json"].read_text(encoding="utf-8"))
    entries = inventory.get("entries", [])
    if len(entries) != 16:
        raise AssertionError(f"inventory 应有16个矩阵，实际为 {len(entries)}")
    observed = {(entry["dataset_id"], entry["descriptor"]) for entry in entries}
    expected = {(spec.dataset_id, descriptor) for spec in DATASETS for descriptor in DESCRIPTORS}
    if observed != expected:
        raise AssertionError("inventory 的 dataset × descriptor 组合不完整或重复")
    for entry in entries:
        npz_path = repo_root / entry["source_npz"]
        if sha256_file(npz_path) != entry["source_npz_sha256"]:
            raise AssertionError(f"NPZ SHA-256 漂移: {entry['source_npz']}")
        expected_shape = EXPECTED_X_SHAPES[(entry["dataset_id"], entry["descriptor"])]
        if tuple(entry["x_shape"]) != expected_shape:
            raise AssertionError(
                f"X shape 漂移: {entry['source_npz']}; "
                f"实际={entry['x_shape']}，预期={list(expected_shape)}"
            )
        if entry["row_selection_performed"]:
            raise AssertionError(f"审计不应选择行: {entry['source_npz']}")

    bh1 = json.loads(paths["bh1_dft_alignment_json"].read_text(encoding="utf-8"))
    if bh1.get("gate_status") != "blocked_alignment" or bh1.get("main_experiment_allowed"):
        raise AssertionError("BH1/DFT 必须保持 blocked_alignment 且不可进入主实验")
    if bh1.get("unresolved_extra_row_count") != 5:
        raise AssertionError("BH1/DFT 行数差必须为5")

    boundary = json.loads(paths["population_boundary_json"].read_text(encoding="utf-8"))
    populations = {item["population_id"]: item for item in boundary.get("populations", [])}
    if set(populations) != {"SM-static-4620", "SM-OHE-5760"}:
        raise AssertionError("SM 必须恰有静态4620和OHE5760两个隔离总体")
    if populations["SM-static-4620"]["row_count"] != 4620:
        raise AssertionError("SM 静态总体必须是4620行")
    if populations["SM-OHE-5760"]["row_count"] != 5760 or boundary.get("mixed_population_allowed"):
        raise AssertionError("SM OHE 5760必须独立且禁止混合")

    with paths["inventory_csv"].open("r", encoding="utf-8", newline="") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != 16:
            raise AssertionError("CSV inventory 必须恰有16行")
    return {
        "status": "passed",
        "matrix_count": len(entries),
        "bh1_dft_gate_status": bh1["gate_status"],
        "sm_population_boundary": "SM-static-4620 != SM-OHE-5760",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="derived/descriptor_model_effect/step28_1_input_audit",
        help="相对于仓库根目录的隔离审计产物目录。",
    )
    parser.add_argument("--verify", action="store_true", help="只校验既有审计产物，不重建。")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    try:
        output_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("--output-dir 必须在仓库目录内") from exc

    if args.verify:
        result = verify_inventory(repo_root, output_dir)
    else:
        paths = build_inventory(repo_root, output_dir)
        result = {
            "status": "built",
            "outputs": {name: _relative(repo_root, path) for name, path in paths.items()},
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
