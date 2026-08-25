from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vjethbkm_repro.component_splits import (
    check_no_component_leakage,
    leave_component_out_1d,
    leave_component_pair_out_2d,
    random_0d_split,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sub_1_smiles": ["A", "A", "B", "B", "C", "C"],
            "sub_2_smiles": ["X", "Y", "X", "Y", "X", "Z"],
            "yield": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_0d_random_split_has_no_component_constraint() -> None:
    df = _toy_df()
    manifest = random_0d_split(df, n_splits=3, random_state=42)
    check = check_no_component_leakage(df, manifest, ["sub_1_smiles", "sub_2_smiles"])
    assert check.ok
    assert check.checked_folds == 3


def test_1d_component_holdout_has_no_component_overlap() -> None:
    df = _toy_df()
    manifest = leave_component_out_1d(df, "sub_1_smiles")
    check = check_no_component_leakage(df, manifest, ["sub_1_smiles", "sub_2_smiles"])
    assert check.ok
    assert check.checked_folds == 3


def test_2d_pair_holdout_has_no_pair_overlap() -> None:
    df = _toy_df()
    manifest = leave_component_pair_out_2d(df, ["sub_1_smiles", "sub_2_smiles"])
    check = check_no_component_leakage(df, manifest, ["sub_1_smiles", "sub_2_smiles"])
    assert check.ok
    assert check.checked_folds == 6

