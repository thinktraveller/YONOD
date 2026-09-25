"""Generate and audit the four paired ablation2 split manifests.

No descriptors or estimators are constructed here.  Each manifest is made
once from the prepared population and is later imported verbatim by every
input-ablation task in that protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.splits.manifest import create_split_manifest, validate_split_manifest


DATASET = ROOT / "result" / "ablation2_amide_data_prep_v1" / "data" / "standardized_population.csv"
TASK_NAME = "ablation2_amide_splits_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_ids(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        checked = pd.read_parquet(temporary)
        if len(checked) != len(frame) or list(checked.columns) != list(frame.columns):
            raise RuntimeError("parquet staging validation failed")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class _SplitConfig:
    frame: pd.DataFrame
    sample_id_col: str
    grouping: dict[str, Any]
    cv: dict[str, int]
    raw: dict[str, str]
    smiles_cols: tuple[str, ...]

    def validate_dataset(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass(frozen=True)
class _SplitContract:
    config: _SplitConfig
    run_id: str
    dataset_sha256: str


PROTOCOLS = {
    "random_repeat_group": {
        "strategy": "precomputed_column", "group_column": "repeat_group_id",
        "interpretation": "random 5×3 CV constrained so identical full inputs cannot cross train/test",
    },
    "unseen_amine": {
        "strategy": "precomputed_column", "group_column": "amine_group_id",
        "interpretation": "A group holdout; an amine structure key is never in both train and test within a fold",
    },
    "unseen_acid": {
        "strategy": "precomputed_column", "group_column": "acid_group_id",
        "interpretation": "B group holdout; a carboxylic-acid structure key is never in both train and test within a fold",
    },
    "unseen_substrate_pair": {
        "strategy": "precomputed_column", "group_column": "substrate_pair_group_id",
        "interpretation": "A+B pair holdout only; individual A and B structures may still have training occurrences",
    },
}


def _audit_manifest(frame: pd.DataFrame, manifest: pd.DataFrame, protocol: str, group_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    index = frame.set_index("sample_id")
    for (repeat, fold), part in manifest.groupby(["repeat", "fold"], sort=True):
        train_ids = sorted(part.loc[part["role"].eq("train"), "sample_id"].astype(str).tolist())
        valid_ids = sorted(part.loc[part["role"].eq("valid"), "sample_id"].astype(str).tolist())
        train = index.loc[train_ids]
        valid = index.loc[valid_ids]
        record = {
            "protocol": protocol, "repeat": int(repeat), "fold": int(fold),
            "n_train": len(train_ids), "n_valid": len(valid_ids),
            "train_sample_ids_sha256": _hash_ids(train_ids),
            "valid_sample_ids_sha256": _hash_ids(valid_ids),
            "group_column": group_column,
            "n_train_target_groups": int(train[group_column].nunique()),
            "n_valid_target_groups": int(valid[group_column].nunique()),
            "target_group_intersection": int(len(set(train[group_column]).intersection(valid[group_column]))),
            "amine_intersection": int(len(set(train["amine_group_id"]).intersection(valid["amine_group_id"]))),
            "acid_intersection": int(len(set(train["acid_group_id"]).intersection(valid["acid_group_id"]))),
            "pair_intersection": int(len(set(train["substrate_pair_group_id"]).intersection(valid["substrate_pair_group_id"]))),
            "repeat_group_intersection": int(len(set(train["repeat_group_id"]).intersection(valid["repeat_group_id"]))),
        }
        records.append(record)
    audit = pd.DataFrame.from_records(records)
    if not audit["target_group_intersection"].eq(0).all():
        raise RuntimeError(f"{protocol}: target grouping leaks across a fold")
    if protocol == "random_repeat_group" and not audit["repeat_group_intersection"].eq(0).all():
        raise RuntimeError("random_repeat_group: duplicate full input leaks across a fold")
    if protocol == "unseen_amine" and not audit["amine_intersection"].eq(0).all():
        raise RuntimeError("unseen_amine: amine group leaks across a fold")
    if protocol == "unseen_acid" and not audit["acid_intersection"].eq(0).all():
        raise RuntimeError("unseen_acid: acid group leaks across a fold")
    if protocol == "unseen_substrate_pair" and not audit["pair_intersection"].eq(0).all():
        raise RuntimeError("unseen_substrate_pair: substrate pair leaks across a fold")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paired, external ablation2 split manifests without fitting models.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = ROOT / "result" / str(args.task_name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing task output: {output}")
    frame = pd.read_csv(dataset, keep_default_na=False, encoding="utf-8")
    if frame["sample_id"].duplicated().any() or not frame["population_status"].eq("included").all():
        raise ValueError("split source must be the fully included, unique standardized population")
    dataset_sha = _sha256(dataset)
    if args.n_splits < 2 or args.n_repeats < 1:
        raise ValueError("n-splits must be >= 2 and n-repeats must be >= 1")
    cv = {"n_splits": int(args.n_splits), "n_repeats": int(args.n_repeats), "seed": int(args.seed)}
    component_cols = (
        "amine_smiles", "acid_smiles", "product_smiles",
        "activation_smiles", "additive_smiles", "base_smiles", "solvent_smiles",
    )
    audits: list[pd.DataFrame] = []
    manifests: dict[str, Any] = {}
    for protocol, definition in PROTOCOLS.items():
        config = _SplitConfig(
            frame=frame, sample_id_col="sample_id", grouping={
                "strategy": definition["strategy"], "group_column": definition["group_column"],
                "protocol_label": protocol,
            }, cv=cv, raw={"population_id": "amide2-population-" + dataset_sha[:16]}, smiles_cols=component_cols,
        )
        contract = _SplitContract(config=config, run_id=f"ablation2-{protocol}-split-v1", dataset_sha256=dataset_sha)
        manifest = create_split_manifest(contract)
        validate_split_manifest(manifest, n_splits=cv["n_splits"], n_repeats=cv["n_repeats"])
        path = output / "manifests" / f"{protocol}_split_manifest.parquet"
        _atomic_parquet(manifest, path)
        audit = _audit_manifest(frame, manifest, protocol, definition["group_column"])
        audits.append(audit)
        manifests[protocol] = {
            "path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
            "split_id": str(manifest["split_id"].iloc[0]), "n_rows": int(len(manifest)),
            "n_folds": int(manifest[["repeat", "fold"]].drop_duplicates().shape[0]),
            "group_column": definition["group_column"], "interpretation": definition["interpretation"],
        }
    membership = pd.concat(audits, ignore_index=True)
    _atomic_csv(membership, output / "audits" / "fold_membership_audit.csv")
    report = {
        "task_kind": "ablation2 paired split creation; no model fitting",
        "dataset": {"path": str(dataset.relative_to(ROOT)), "sha256": dataset_sha, "n_samples": int(len(frame))},
        "cv": cv,
        "manifests": manifests,
        "negative_case_contracts": [
            "unknown sample_id or changed dataset_sha256 is rejected by external manifest import",
            "a source manifest with any group shared by train/test is rejected by validate_split_manifest",
            "a task whose output root differs from its feature child is rejected by strict BenchmarkConfig",
        ],
    }
    _atomic_json(report, output / "split_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
