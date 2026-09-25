"""Lightweight numerical/provenance QA for the cost/multimetric audit."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analyze_pnecessity_cost_metrics import cluster_intervals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/pnecessity_cost_metrics_audit_20260924"


def main():
    s = pd.read_csv(OUT / "paired_summary.csv")
    t = pd.read_csv(OUT / "fold_timings.csv")
    r = pd.read_csv(OUT / "repeat_metrics.csv")
    assert len(s) == s.pair_id.nunique() == 40
    assert len(set(s.full) | set(s.no_p)) == 80
    assert set(s.n_timed_fits_per_arm) == {1, 15}
    assert np.allclose(s.train_increase_pct, 100 * (s.full_train_s / s.no_p_train_s - 1))
    assert np.allclose(s.no_p_train_saving_pct, 100 * (1 - s.no_p_train_s / s.full_train_s))
    assert np.allclose(s.delta_r2, s.full_r2-s.no_p_r2)
    assert np.allclose(s.delta_rmse, s.no_p_rmse-s.full_rmse)
    assert np.allclose(s.delta_mae, s.no_p_mae-s.full_mae)
    assert np.allclose(s.rmse_reduction_pct, 100*s.delta_rmse/s.no_p_rmse)
    assert s.feature_time_increase_pct.isna().all()
    assert s.end_to_end_time_increase_pct.isna().all()
    assert (s.r2_bootstrap_finite_draws == 4000).all()
    for row in s.itertuples():
        part = t[t.pair_id == row.pair_id]
        assert len(part) == row.n_timed_fits_per_arm
        for arm in ("full", "no_p"):
            for metric in ("train", "predict", "model"):
                assert np.isclose(part[f"{metric}_s_{arm}"].sum(), getattr(row, f"{arm}_{metric}_s"))
            rp = r[r.pair_id == row.pair_id]
            assert np.isclose(rp[f"{arm}_r2"].mean(), getattr(row, f"{arm}_r2"))
            assert np.isclose(np.sqrt((rp[f"{arm}_rmse"]**2).mean()), getattr(row, f"{arm}_rmse"))
        if row.batch == "amide_frozen_external":
            for arm, task in (("full", row.full), ("no_p", row.no_p)):
                m = pd.read_csv(ROOT / "result" / task / "docs/metrics/external_test_metrics.csv")
                for metric in ("r2", "mae", "rmse"):
                    assert np.isclose(m[metric].iloc[0], getattr(row, f"{arm}_{metric}"))
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_fits_started"] == 0
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    assert hashlib.sha256((ROOT / "_verify/analyze_pnecessity_cost_metrics.py").read_bytes()).hexdigest() == manifest["script_sha256"]
    # Synthetic groups test: average losses over repetitions, not predictions.
    test = pd.DataFrame({"sample_id": list("abcd")*2, "group_id_f": ["g1", "g1", "g2", "g2"]*2,
                         "y_true_f": [0., 1., 0., 1.]*2, "y_true_n": [0., 1., 0., 1.]*2,
                         "y_pred_f": [.1, .9, .1, .9, -.1, 1.1, -.1, 1.1], "y_pred_n": [.2, .8, .2, .8]*2})
    v = cluster_intervals(test, 1000, 42)
    assert np.isclose(v["full_rmse"], .1)
    assert np.isclose(v["delta_rmse"], .1)
    assert np.isclose(v["delta_r2"], .12)
    docs = [ROOT / "project-docs" / "docs" / "开题报告_产物表示的增量预测价值_20260924.md"]
    docs.extend((ROOT / "project-docs" / "docs" / "product_necessity_cost_metrics_20260924").glob("*.md"))
    count = 0
    for p in docs:
        text = p.read_text(encoding="utf-8")
        for link in re.findall(r"\]\((/[^)]+)\)", text):
            assert Path(link).exists(), (p, link)
            count += 1
    print(f"PASS: 40 pairs / 80 tasks; {len(t)} paired fit records; numerical identities, synthetic estimand test, {len(manifest['files'])} source hashes, {count} local links")


if __name__ == "__main__":
    main()
