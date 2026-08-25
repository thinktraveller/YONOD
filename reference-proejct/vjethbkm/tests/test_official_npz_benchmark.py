from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_official_npz_benchmark import DATASET_MAP


def test_official_npz_dataset_map_points_to_known_folders() -> None:
    assert DATASET_MAP["SLAP"]["source_folder"] == "Bode_2023"
    assert DATASET_MAP["SM"]["source_folder"] == "Suzuki_2018"
