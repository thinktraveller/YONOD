from __future__ import annotations

from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]


def test_sm_ohe_inputs_exist_for_forensics() -> None:
    assert (REPRO_ROOT / "yieldsmarter/Data/HTE_datasets/SM/SM.csv").exists()
    assert (REPRO_ROOT / "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/2SM.csv").exists()
    assert (
        REPRO_ROOT
        / "yieldsmarter/Results/Compare_Complexity/Suzuki_2018/OHE/Suzuki_OHE_oof_predictions_RF.npz"
    ).exists()
