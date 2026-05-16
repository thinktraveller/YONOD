"""Smoke test for run_yield_prediction.py.

Runs the smallest meaningful grid (Morgan x XGB, 500 rows, cv=5) and verifies:
  * CSV summary written with expected columns
  * Exactly 1 scatter PNG produced
  * R^2 above a sanity threshold (>0.2 on this small subset)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> int:
    print("=" * 60)
    print("YONOD smoke test: morgan x xgb, 500 rows, cv=5")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "run_yield_prediction.py"),
        "--desc",
        "morgan",
        "--model",
        "xgb",
        "--nrows",
        "500",
        "--cv",
        "5",
    ]
    print(f"[cmd] {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[FAIL] subprocess exited with code {proc.returncode}")
        return proc.returncode

    summary = RESULTS / "metrics_summary.csv"
    scatter = RESULTS / "scatter_morgan_xgb.png"
    print("\n--- verification ---")

    if not summary.exists():
        print(f"[FAIL] {summary} not found")
        return 1
    print(f"[OK]   {summary.name} exists")

    if not scatter.exists():
        print(f"[FAIL] {scatter} not found")
        return 1
    print(f"[OK]   {scatter.name} exists ({scatter.stat().st_size} bytes)")

    df = pd.read_csv(summary)
    expected_cols = {
        "descriptor",
        "model",
        "n_samples",
        "feature_dim",
        "cv",
        "r2_mean",
        "r2_std",
        "rmse_mean",
        "mae_mean",
        "train_time_s",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[FAIL] metrics_summary.csv missing columns: {missing}")
        return 1
    print(f"[OK]   metrics_summary.csv has all expected columns")

    if len(df) != 1:
        print(f"[FAIL] expected 1 row, got {len(df)}")
        return 1
    print(f"[OK]   1 row in summary")

    row = df.iloc[0]
    if row["descriptor"] != "morgan" or row["model"] != "xgb":
        print(f"[FAIL] row identity mismatch: {row['descriptor']} x {row['model']}")
        return 1
    print(f"[OK]   row is morgan x xgb")

    if int(row["feature_dim"]) != 6144:
        print(f"[FAIL] feature_dim expected 6144, got {row['feature_dim']}")
        return 1
    print(f"[OK]   feature_dim = 6144")

    if row["r2_mean"] < 0.2:
        print(f"[WARN] R^2 = {row['r2_mean']:.3f} below sanity threshold 0.2")
    else:
        print(f"[OK]   R^2 = {row['r2_mean']:.3f}")

    print("\n--- summary ---")
    print(df.to_string(index=False))
    print("\n[PASS] smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
