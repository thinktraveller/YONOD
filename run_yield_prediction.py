"""YONOD main entry point: descriptor x model grid evaluation.

Usage examples
--------------

  # Smoke (Morgan x XGB, 500 rows):
  python run_yield_prediction.py --desc morgan --model xgb --nrows 500

  # Full 4 x 4 grid on full dataset:
  python run_yield_prediction.py

  # Custom CSV path, 3-fold CV, two descriptors:
  python run_yield_prediction.py \
      --csv "my\reactions.csv" \
      --desc morgan fisd \
      --cv 3

Outputs
-------
  results/metrics_summary.csv        # one row per (desc, model)
  results/scatter_{desc}_{model}.png # one PNG per combination
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, TextIO

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yonod_yield.evaluate import (
    DESCRIPTOR_REGISTRY,
    MODEL_REGISTRY,
    evaluate_one,
)
from yonod_yield.plot import plot_scatter


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = DEFAULT_ROOT / "数据集" / "酰胺缩合反应数据集.csv"
DEFAULT_RESULTS = DEFAULT_ROOT / "results"

# Columns to keep in the CSV summary (predictions are stripped before write).
CSV_COLUMNS = [
    "descriptor",
    "model",
    "n_samples",
    "n_total",
    "coverage",
    "feature_dim",
    "cv",
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "mae_mean",
    "train_time_s",
    "device",
    "pca",
    "holdout_frac",
]


class _Tee:
    """Mirror writes to stdout AND a log file, with auto-prefixed timestamps."""

    def __init__(self, file_handle: TextIO) -> None:
        self.file = file_handle
        self.stdout = sys.__stdout__
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        out_chunks = []
        for ch in data:
            if self._at_line_start and ch != "\n":
                ts = _dt.datetime.now().strftime("%H:%M:%S")
                out_chunks.append(f"[{ts}] ")
                self._at_line_start = False
            out_chunks.append(ch)
            if ch == "\n":
                self._at_line_start = True
        out = "".join(out_chunks)
        self.stdout.write(out)
        self.file.write(out)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


class _Heartbeat:
    """Background thread that prints a 'still running' line every N seconds."""

    def __init__(self, interval: float = 30.0, label: str = "") -> None:
        self.interval = interval
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._t0
            print(f"  [hb] {self.label} still running, elapsed {elapsed:.0f}s", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YONOD: amide-condensation yield prediction grid runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the reaction dataset CSV",
    )
    p.add_argument(
        "--desc",
        nargs="+",
        default=list(DESCRIPTOR_REGISTRY.keys()),
        choices=list(DESCRIPTOR_REGISTRY.keys()),
        help="Descriptors to evaluate",
    )
    p.add_argument(
        "--model",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to evaluate",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit CSV to first N rows (None = full dataset)",
    )
    p.add_argument(
        "--cv",
        type=int,
        default=5,
        help="K for K-fold CV (ignored by autogluon, which uses holdout)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Where to write metrics_summary.csv and scatter PNGs",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip scatter plot generation",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to mirror all output (default: results/run_<timestamp>.log)",
    )
    p.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="Seconds between 'still running' prints during long fits (0 disables)",
    )
    p.add_argument(
        "--rf-verbose",
        type=int,
        default=0,
        help="Verbosity for RandomForestRegressor (1 prints per-tree progress)",
    )
    p.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
        help="n_jobs for RandomForestRegressor (-1 uses all cores; try 1 if RF stalls)",
    )
    p.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Number of trees in RF (reduce to 100 for 3x speedup with slight R^2 loss)",
    )
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Max tree depth in RF (cap to 30 to halve time on high-dim features)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing metrics_summary.csv instead of overwriting "
             "(use to merge multiple split runs).",
    )
    p.add_argument(
        "--svm-subsample",
        type=int,
        default=8000,
        help="Random subsample size for SVM train set per fold. RBF SVR is "
             "O(n^2 d); set to 5000-8000 for tractable full-dataset runs, "
             "or 0 to disable (uses the whole train fold).",
    )
    p.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generating the self-contained HTML report at the end.",
    )
    return p.parse_args()


def _print_table(rows: List[dict]) -> None:
    """Pretty-print the metrics summary as an aligned table."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    display_cols = [
        c
        for c in [
            "descriptor",
            "model",
            "feature_dim",
            "n_samples",
            "r2_mean",
            "r2_std",
            "rmse_mean",
            "mae_mean",
            "train_time_s",
        ]
        if c in df.columns
    ]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print()
    print(df[display_cols].to_string(index=False))


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        print(f"[error] CSV not found: {args.csv}", file=sys.stderr)
        return 1

    log_path = args.log_file or (
        args.results_dir
        / f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log_fh)

    print(f"[log]  mirroring stdout to {log_path}")
    print(f"[load] {args.csv}  (nrows={args.nrows or 'all'})")
    df = pd.read_csv(args.csv, nrows=args.nrows)
    print(f"[load] shape={df.shape}")
    print(f"[grid] descriptors={args.desc}  models={args.model}  cv={args.cv}")

    # Per-model overrides applied at build time inside evaluate_one.
    svm_subsample = args.svm_subsample if args.svm_subsample > 0 else None
    model_kwargs_by_name = {
        "rf": {
            "n_jobs": args.rf_n_jobs,
            "verbose": args.rf_verbose,
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
        },
        "svm": {
            "subsample_n": svm_subsample,
        },
    }

    rows: List[dict] = []
    total = len(args.desc) * len(args.model)
    counter = 0
    grid_t0 = time.time()

    for desc in args.desc:
        for model in args.model:
            counter += 1
            label = f"{desc} x {model}"
            print(
                f"\n[{counter}/{total}] {label}  ...",
                flush=True,
            )
            hb: Optional[_Heartbeat] = None
            if args.heartbeat > 0:
                hb = _Heartbeat(interval=args.heartbeat, label=label)
                hb.start()
            try:
                row = evaluate_one(
                    desc,
                    model,
                    df,
                    cv=args.cv,
                    return_predictions=True,
                    model_kwargs=model_kwargs_by_name.get(model),
                )
            except Exception as e:
                if hb is not None:
                    hb.stop()
                print(f"  [fail] {type(e).__name__}: {e}", file=sys.stderr)
                rows.append(
                    {
                        "descriptor": desc,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            finally:
                if hb is not None:
                    hb.stop()

            oof_pred = row.pop("oof_pred", None)
            oof_y_true = row.pop("oof_y_true", None)

            print(
                f"  R2={row.get('r2_mean'):.4f}  "
                f"RMSE={row.get('rmse_mean'):.4f}  "
                f"t={row.get('train_time_s'):.1f}s",
                flush=True,
            )

            # Plot using OOF predictions (drop NaN rows -- AutoGluon only fills
            # the holdout subset).
            if (
                not args.skip_plots
                and oof_pred is not None
                and oof_y_true is not None
            ):
                valid = ~np.isnan(oof_pred)
                if valid.sum() >= 2:
                    out = plot_scatter(
                        oof_y_true[valid],
                        oof_pred[valid],
                        desc=desc,
                        model=model,
                        save_dir=args.results_dir,
                    )
                    print(f"  [plot] {out.name}")

            rows.append(row)

    # Write CSV summary (append-aware: merges with previous runs if --append).
    df_out = pd.DataFrame(rows)
    keep = [c for c in CSV_COLUMNS if c in df_out.columns]
    extra = [c for c in df_out.columns if c not in keep]
    df_out = df_out[keep + extra]
    summary_path = args.results_dir / "metrics_summary.csv"
    if args.append and summary_path.exists():
        prev = pd.read_csv(summary_path)
        # Drop any prior rows with the same (descriptor, model) to avoid duplicates.
        if {"descriptor", "model"}.issubset(prev.columns):
            keys = set(zip(df_out["descriptor"], df_out["model"]))
            prev = prev[
                ~prev.apply(lambda r: (r["descriptor"], r["model"]) in keys, axis=1)
            ]
        df_out = pd.concat([prev, df_out], ignore_index=True, sort=False)
        print(f"[save] appending to existing summary ({len(prev)} prior rows)")
    df_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {summary_path}  ({len(df_out)} total rows)")

    _print_table(rows)
    print(f"\n[done] {counter} combinations in {time.time() - grid_t0:.1f}s")

    # ---- Final step: build the self-contained HTML report -----------------
    if not args.skip_report:
        try:
            from generate_report import main as _gen_report_main

            print("\n[report] generating HTML report ...")
            # generate_report uses its own argparse; we invoke via sys.argv override.
            saved_argv = sys.argv
            sys.argv = [
                "generate_report.py",
                "--results-dir", str(args.results_dir),
            ]
            try:
                _gen_report_main()
            finally:
                sys.argv = saved_argv
        except Exception as e:
            print(f"[report] generation failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
