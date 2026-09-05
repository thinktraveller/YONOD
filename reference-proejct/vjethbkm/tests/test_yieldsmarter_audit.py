from __future__ import annotations

from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPRO_ROOT / "yieldsmarter"


def test_yieldsmarter_package_readme_and_license_exist() -> None:
    assert (PACKAGE_ROOT / "README.txt").exists()
    assert (PACKAGE_ROOT / "LICENSE.txt").exists()


def test_yieldsmarter_core_datasets_exist() -> None:
    for relative_path in [
        "Data/HTE_datasets/BH1/BH1.csv",
        "Data/HTE_datasets/BH2/BH2.csv",
        "Data/HTE_datasets/SM/SM.csv",
        "Data/HTE_datasets/SL1/SL1.csv",
    ]:
        assert (PACKAGE_ROOT / relative_path).exists()

