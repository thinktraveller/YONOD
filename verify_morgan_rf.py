"""Standalone diagnostic for Morgan x RandomForest hangs on the full dataset.

Symptom:
    `python run_yield_prediction.py` stalls silently at the morgan x rf step.

What this script does (in order, with timestamps after every step):
    1. Loads the full CSV (or --nrows subset).
    2. Featurizes Morgan -> shape (~47015, 6144); reports timing + memory.
    3. Trains RandomForestRegressor with verbose=2 so you see per-tree progress.
       Defaults to a SINGLE 80/20 holdout (not 5-fold), to find out the
       wall-clock cost of one fit before committing to 5x more.
    4. Optional --cv 5 will do the full 5-fold.

Common knobs to try if RF still feels slow:
    --n-jobs 1            # rule out joblib pickle stalls (Windows specific)
    --n-estimators 100    # 1/3 the trees, 1/3 the time
    --max-depth 20        # cap depth -- helps when feature dim is huge
    --nrows 5000          # quick sanity on a subset

Example:
    python verify_morgan_rf.py --nrows 5000 --n-estimators 100 --n-jobs 1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer

DEFAULT_CSV = ROOT / "数据集" / "酰胺缩合反应数据集.csv"


def ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose Morgan + RF hangs on the full dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--nrows", type=int, default=None,
                   help="None = full dataset (47015 rows)")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None,
                   help="None = unlimited (sklearn default)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="-1 = all cores, 1 = serial (test joblib stall)")
    p.add_argument("--cv", type=int, default=0,
                   help="0 = single 80/20 holdout; >0 = K-fold")
    p.add_argument("--verbose", type=int, default=2,
                   help="sklearn verbosity (2 = print every tree)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log(f"loading CSV: {args.csv}  nrows={args.nrows or 'all'}")
    df = pd.read_csv(args.csv, nrows=args.nrows)
    log(f"shape={df.shape}")

    log("featurizing with Morgan ECFP4 (6 molecules x 1024 bits)")
    t0 = time.time()
    feat = ReactionFeaturizer(MorganDescriptor())
    X, mask = feat.transform(df)
    y = df["yield"].to_numpy()[mask]
    log(f"X.shape={X.shape}  y.shape={y.shape}  "
        f"dtype={X.dtype}  mem={X.nbytes / 1e6:.1f} MB  "
        f"({time.time() - t0:.1f}s)")

    # Sanity-print first row to verify it's not all zeros
    nonzero = int((X[0] != 0).sum())
    log(f"row[0] nonzero count = {nonzero}  (Morgan x 6 mols typical: 50-300)")

    log("importing sklearn RandomForestRegressor")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, train_test_split

    if args.cv > 0:
        kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
        splits = list(kf.split(X))
        log(f"running {args.cv}-fold CV")
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        # Mimic KFold's interface so the loop below can be uniform
        tr_idx = np.arange(len(Xtr))
        te_idx = np.arange(len(Xtr), len(Xtr) + len(Xte))
        Xall = np.concatenate([Xtr, Xte], axis=0)
        yall = np.concatenate([ytr, yte], axis=0)
        X, y = Xall, yall
        splits = [(tr_idx, te_idx)]
        log("running single 80/20 holdout (one fit only)")

    log(f"RF params: n_estimators={args.n_estimators}  "
        f"max_depth={args.max_depth}  n_jobs={args.n_jobs}  "
        f"verbose={args.verbose}")

    r2s, rmses, maes, fold_times = [], [], [], []
    for i, (tr, te) in enumerate(splits, start=1):
        log(f"--- fold {i}/{len(splits)} : train={len(tr)}  test={len(te)}")
        log(f"fold {i}: building RF and starting fit ...")
        t_fit0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
            random_state=42,
            verbose=args.verbose,
        )
        rf.fit(X[tr], y[tr])
        t_fit = time.time() - t_fit0
        log(f"fold {i}: fit done in {t_fit:.1f}s")

        log(f"fold {i}: predicting on test ({len(te)} samples)")
        t_pred0 = time.time()
        pred = rf.predict(X[te])
        t_pred = time.time() - t_pred0
        log(f"fold {i}: predict done in {t_pred:.1f}s")

        r2 = r2_score(y[te], pred)
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        mae = float(mean_absolute_error(y[te], pred))
        log(f"fold {i}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        r2s.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        fold_times.append(t_fit + t_pred)

    log("=" * 60)
    log(f"FINAL  r2_mean={np.mean(r2s):.4f}  r2_std={np.std(r2s):.4f}  "
        f"rmse={np.mean(rmses):.4f}  mae={np.mean(maes):.4f}")
    log(f"timings: per-fold mean={np.mean(fold_times):.1f}s  "
        f"total={sum(fold_times):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
