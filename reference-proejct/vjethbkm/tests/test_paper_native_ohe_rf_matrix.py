from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_paper_native_ohe_rf_matrix import OHE_SPECS, ohe_fit_transform_fold, oof_alignment


def test_ohe_matrix_keeps_all_four_paper_populations_explicit() -> None:
    assert set(OHE_SPECS) == {"BH1", "BH2", "SL1", "SM"}
    assert OHE_SPECS["SM"]["csv"].endswith("SM/SM.csv")


def test_train_only_ohe_zeroes_missing_blocks_and_ignores_unseen_values() -> None:
    train = pd.DataFrame({"first": ["a", "b", None], "second": ["x", "x", "y"]})
    test = pd.DataFrame({"first": ["new", None], "second": ["z", "x"]})
    train_features, test_features, audit = ohe_fit_transform_fold(train, test, ["first", "second"])
    assert train_features.shape == (3, 5)
    assert test_features.shape == (2, 5)
    assert np.array_equal(train_features[2, :3], np.zeros(3))
    assert np.array_equal(test_features[1, :3], np.zeros(3))
    assert np.array_equal(test_features[0], np.zeros(5))
    assert audit == {"ohe_output_dim": 5, "train_missing_value_count": 1, "test_missing_value_count": 1, "unseen_test_value_count": 2}


def test_oof_alignment_reports_exact_and_difference(tmp_path: Path) -> None:
    official_path = tmp_path / "official.npz"
    np.savez(official_path, y_true=np.array([1.0, 2.0]), y_pred=np.array([0.5, 2.5]), fold_id=np.array([1, 2]))
    exact = {"y_true": np.array([1.0, 2.0]), "y_pred": np.array([0.5, 2.5]), "fold_id": np.array([1, 2])}
    exact_alignment = oof_alignment(exact, official_path)
    assert exact_alignment["y_pred_exact"] is True
    assert exact_alignment["y_pred_within_1e_12"] is True
    changed = dict(exact, y_pred=np.array([0.5, 2.75]))
    assert oof_alignment(changed, official_path)["y_pred_max_abs_diff"] == 0.25
