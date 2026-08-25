"""Component-holdout split helpers for generalization checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .splits import repeated_kfold_manifest


@dataclass(frozen=True)
class LeakageCheck:
    split_name: str
    ok: bool
    checked_folds: int
    violations: int
    message: str


def random_0d_split(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    manifest = repeated_kfold_manifest(df, n_splits=n_splits, repeats=1, random_state=random_state)
    manifest["holdout_dim"] = "0d"
    manifest["heldout_component"] = ""
    manifest["heldout_value"] = ""
    return manifest


def leave_component_out_1d(df: pd.DataFrame, component_col: str) -> pd.DataFrame:
    records: list[dict] = []
    values = sorted(str(value) for value in df[component_col].dropna().unique())
    for fold, value in enumerate(values, start=1):
        is_test = df[component_col].astype(str) == value
        for sample_id, test_flag in is_test.items():
            records.append(
                {
                    "sample_id": int(sample_id),
                    "repeat": 1,
                    "fold": fold,
                    "split_name": f"holdout_1d_{component_col}",
                    "is_train": not bool(test_flag),
                    "holdout_dim": "1d",
                    "heldout_component": component_col,
                    "heldout_value": value,
                }
            )
    return pd.DataFrame.from_records(records)


def leave_component_pair_out_2d(df: pd.DataFrame, component_cols: list[str]) -> pd.DataFrame:
    if len(component_cols) != 2:
        raise ValueError("2D component split requires exactly two component columns.")
    records: list[dict] = []
    pair_frame = df[component_cols].astype(str)
    pairs = sorted(tuple(row) for row in pair_frame.drop_duplicates().itertuples(index=False, name=None))
    for fold, pair in enumerate(pairs, start=1):
        is_test = (pair_frame[component_cols[0]] == pair[0]) & (pair_frame[component_cols[1]] == pair[1])
        for sample_id, test_flag in is_test.items():
            records.append(
                {
                    "sample_id": int(sample_id),
                    "repeat": 1,
                    "fold": fold,
                    "split_name": f"holdout_2d_{component_cols[0]}__{component_cols[1]}",
                    "is_train": not bool(test_flag),
                    "holdout_dim": "2d",
                    "heldout_component": "+".join(component_cols),
                    "heldout_value": "||".join(pair),
                }
            )
    return pd.DataFrame.from_records(records)


def check_no_component_leakage(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    component_cols: list[str],
) -> LeakageCheck:
    split_name = str(manifest["split_name"].iloc[0]) if not manifest.empty else "empty"
    violations = 0
    checked_folds = 0
    for (_, fold), fold_manifest in manifest.groupby(["repeat", "fold"]):
        train_ids = fold_manifest.loc[fold_manifest["is_train"], "sample_id"].tolist()
        test_ids = fold_manifest.loc[~fold_manifest["is_train"], "sample_id"].tolist()
        if not train_ids or not test_ids:
            violations += 1
            checked_folds += 1
            continue
        holdout_dim = str(fold_manifest["holdout_dim"].iloc[0])
        train = df.iloc[train_ids]
        test = df.iloc[test_ids]
        if holdout_dim == "0d":
            checked_folds += 1
            continue
        if holdout_dim == "1d":
            column = str(fold_manifest["heldout_component"].iloc[0])
            overlap = set(train[column].astype(str)) & set(test[column].astype(str))
            violations += int(bool(overlap))
        elif holdout_dim == "2d":
            train_pairs = set(train[component_cols].astype(str).itertuples(index=False, name=None))
            test_pairs = set(test[component_cols].astype(str).itertuples(index=False, name=None))
            violations += int(bool(train_pairs & test_pairs))
        else:
            violations += 1
        checked_folds += 1

    ok = violations == 0
    return LeakageCheck(
        split_name=split_name,
        ok=ok,
        checked_folds=checked_folds,
        violations=violations,
        message="no leakage detected" if ok else "component leakage or empty fold detected",
    )


def split_summary(df: pd.DataFrame, manifest: pd.DataFrame, component_cols: list[str]) -> dict:
    check = check_no_component_leakage(df, manifest, component_cols)
    test_rows = manifest.loc[~manifest["is_train"]]
    return {
        "split_name": check.split_name,
        "holdout_dim": str(manifest["holdout_dim"].iloc[0]),
        "folds": check.checked_folds,
        "n_rows": int(len(df)),
        "test_assignments": int(len(test_rows)),
        "min_test_size": int(test_rows.groupby(["repeat", "fold"]).size().min()),
        "max_test_size": int(test_rows.groupby(["repeat", "fold"]).size().max()),
        "leakage_ok": check.ok,
        "violations": check.violations,
        "message": check.message,
    }

