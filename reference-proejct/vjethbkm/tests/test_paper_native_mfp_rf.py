from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPRO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPRO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_paper_native_mfp_rf import SPEC, fold_metrics, generate_mfp, mfp_alignment, metric_rows


def test_unmodified_gen_mfp_regenerates_official_sl1_features(tmp_path: Path) -> None:
    generated = tmp_path / "generated.npz"
    generate_mfp(REPRO_ROOT, REPRO_ROOT / SPEC["csv"], generated)
    assert mfp_alignment(generated, REPRO_ROOT / SPEC["official_npz"]) == {
        "generated_shape": [1150, 3072],
        "official_shape": [1150, 3072],
        "x_exact": True,
        "x_max_abs_diff": 0,
        "y_exact": True,
        "smiles_columns_exact": True,
    }


def test_unmodified_generator_accepts_relative_output_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = Path("nested") / "generated.npz"
    generate_mfp(REPRO_ROOT, REPRO_ROOT / SPEC["csv"], output)
    assert output.exists()
    assert mfp_alignment(output, REPRO_ROOT / SPEC["official_npz"])["x_exact"] is True


def test_fold_metrics_and_comparison_keep_paper_facing_contract() -> None:
    metrics = fold_metrics(np.array([1.0, 3.0]), np.array([1.0, 2.0]))
    assert metrics == {"mae": 0.5, "rmse": 2 ** -0.5, "r2": 0.5, "kendall": 1.0}
    official = {"test": {metric: {"mean": 0.0, "std": 0.0} for metric in ("mae", "rmse", "r2", "kendall")}}
    local = {metric: {"mean": 1.0, "std": 0.1} for metric in official["test"]}
    rows = metric_rows(local, official)
    assert len(rows) == 4
    assert {row["metric"] for row in rows} == {"mae", "rmse", "r2", "kendall_tau"}
    assert {row["valid_fold_count"] for row in rows} == {25}
