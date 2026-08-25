"""Split generation helpers for reproducible smoke benchmarks."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold


def repeated_kfold_manifest(
    df: pd.DataFrame,
    n_splits: int,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    records: list[dict] = []
    sample_ids = list(df.index)
    for repeat in range(repeats):
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            for idx in train_idx:
                records.append(
                    {
                        "sample_id": sample_ids[idx],
                        "repeat": repeat + 1,
                        "fold": fold,
                        "split_name": "random_5x5_smoke",
                        "is_train": True,
                    }
                )
            for idx in test_idx:
                records.append(
                    {
                        "sample_id": sample_ids[idx],
                        "repeat": repeat + 1,
                        "fold": fold,
                        "split_name": "random_5x5_smoke",
                        "is_train": False,
                    }
                )
    return pd.DataFrame.from_records(records)

