"""ECC-Track entry point: predict enantioselectivity (ΔΔG) from SMILES.

Input:  Raw_Dataset.xlsx  (Ligand_SMILES + Product_SMILES + Temperature + ΔΔG)
Output: results/ecc_results/metrics_summary.csv + scatter plots + HTML report

Pipeline (mirrors run_yield_prediction.py):
  1. Load ECC dataset -> ligand_smiles, product_smiles, temperatures, ddG
  2. Featurize each SMILES list independently with the chosen descriptor
  3. Concatenate [ligand_feat | product_feat | temperature] as reaction vector
  4. Run K-fold CV with chosen model(s)
  5. Report R², RMSE, MAE on ΔΔG + ee MAE (chemically interpretable)

Supported descriptors: morgan, atmomaccs, rdkit_desc
  (fisd and molmetalm require pretrained model files -- skipped by default)
Supported models: xgb, rf, svm, autogluon

Usage examples
--------------
  # Quick smoke test (Morgan x XGB, first 500 rows):
  python run_ecc_prediction.py --desc morgan --model xgb --nrows 500

  # Full 3x4 grid:
  python run_ecc_prediction.py

  # Only XGBoost, all descriptors:
  python run_ecc_prediction.py --model xgb
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yonod_yield.features.ecc_dataset import load_ecc_dataset
from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.descriptors.atmomaccs import ATMOMACCSDescriptor
from yonod_yield.descriptors.base import BaseDescriptor
from yonod_yield.features.dataset import MoleculeFeaturizer

DESCRIPTOR_REGISTRY = {
    "morgan": MorganDescriptor,
    "atmomaccs": ATMOMACCSDescriptor,
}

# Model names available; actual classes loaded lazily to avoid import errors
# for packages that may not be installed in the current environment.
_ALL_MODELS = ["xgb", "rf", "svm", "autogluon"]


def build_descriptor(name: str, **kwargs) -> BaseDescriptor:
    if name not in DESCRIPTOR_REGISTRY:
        raise KeyError(f"Unknown descriptor {name!r}. Available: {sorted(DESCRIPTOR_REGISTRY)}")
    return DESCRIPTOR_REGISTRY[name](**kwargs)


def build_model(name: str, **kwargs):
    if name == "xgb":
        from yonod_yield.models.xgb_model import XGBYieldModel
        return XGBYieldModel(**kwargs)
    elif name == "rf":
        from yonod_yield.models.rf_model import RFYieldModel
        return RFYieldModel(**kwargs)
    elif name == "svm":
        from yonod_yield.models.svm_model import SVMYieldModel
        return SVMYieldModel(**kwargs)
    elif name == "autogluon":
        from yonod_yield.models.autogluon_model import AutoGluonYieldModel
        return AutoGluonYieldModel(**kwargs)
    else:
        raise KeyError(f"Unknown model {name!r}. Available: {_ALL_MODELS}")
from yonod_yield.metrics.ee_metrics import ee_mae as compute_ee_mae
from yonod_yield.plot import plot_scatter

DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_XLSX = (
    DEFAULT_ROOT
    / "数据集"
    / "Enantioselective-Cross-Coupling-Prediction"
    / "Data"
    / "csv"
    / "Raw_Dataset.csv"
)
DEFAULT_RESULTS_DIR = DEFAULT_ROOT / "results" / "ecc_results"

# Descriptors available without pretrained model files
ECC_DESCRIPTORS = ["morgan", "atmomaccs"]

CSV_COLUMNS = [
    "descriptor", "model", "n_samples", "feature_dim", "cv",
    "r2_mean", "r2_std", "rmse_mean", "mae_mean", "ee_mae_mean", "train_time_s",
]


# ---------------------------------------------------------------------------
# Logging helpers (copied from run_yield_prediction.py)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, file_handle: TextIO) -> None:
        self.file = file_handle
        self.stdout = sys.__stdout__
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        out_chunks: List[str] = []
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


# ---------------------------------------------------------------------------
# Core featurization + CV
# ---------------------------------------------------------------------------

def _build_reaction_features(
    ligand_smiles: List[str],
    product_smiles: List[str],
    temperatures: np.ndarray,
    descriptor_name: str,
    descriptor_kwargs: Optional[Dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Featurize two SMILES lists, concatenate with temperature scalar.

    Returns:
        X:    (n_valid, feat_dim_lig + feat_dim_prod + 1)
        mask: boolean (len(ligand_smiles),) -- rows that survived featurization
    """
    desc = build_descriptor(descriptor_name, **(descriptor_kwargs or {}))
    feat_lig = MoleculeFeaturizer(desc)

    # Re-build a fresh descriptor instance for products (same type/params)
    desc2 = build_descriptor(descriptor_name, **(descriptor_kwargs or {}))
    feat_prod = MoleculeFeaturizer(desc2)

    X_lig, mask_lig = feat_lig.transform(ligand_smiles)
    X_prod, mask_prod = feat_prod.transform(product_smiles)

    # Keep only rows valid in BOTH
    mask = mask_lig & mask_prod
    X_lig_valid = X_lig[mask[mask_lig]]   # rows of X_lig that also pass mask_prod
    X_prod_valid = X_prod[mask[mask_prod]]

    # Align by joint mask
    lig_idx = np.where(mask_lig)[0]
    prod_idx = np.where(mask_prod)[0]
    joint_lig = np.isin(lig_idx, np.where(mask)[0])
    joint_prod = np.isin(prod_idx, np.where(mask)[0])
    X_lig_final = X_lig[joint_lig]
    X_prod_final = X_prod[joint_prod]

    T_valid = temperatures[mask].reshape(-1, 1)

    X = np.concatenate([X_lig_final, X_prod_final, T_valid], axis=1)
    return X, mask


def _cross_validate_ecc(
    X: np.ndarray,
    y_ddG: np.ndarray,
    temperatures_K: np.ndarray,
    model_name: str,
    cv: int = 5,
    model_kwargs: Optional[Dict] = None,
) -> Dict:
    """CV evaluation returning ΔΔG metrics + ee MAE.

    For sklearn-style models (xgb/rf/svm): true K-fold CV via _simple_fold_predict.
    For AutoGluon: single random hold-out (AutoGluon bags/stacks internally,
    so an outer K-fold is redundant). Mirrors run_yield_prediction.py behavior.
    """
    # --- AutoGluon: delegate to its own hold-out evaluator ---
    if model_name == "autogluon":
        t0 = time.time()
        model = build_model(model_name, **(model_kwargs or {}))
        ag_metrics = model.fit_evaluate(X, y_ddG, holdout_frac=1.0 / cv)
        elapsed = time.time() - t0

        oof_pred = ag_metrics["oof_pred"]        # NaN for train rows
        oof_y = ag_metrics["oof_y_true"]
        valid = ~np.isnan(oof_pred)
        y_pred_val = oof_pred[valid]
        y_true_val = oof_y[valid]
        T_val = temperatures_K[valid]

        return {
            "r2_mean": float(r2_score(y_true_val, y_pred_val)),
            "r2_std": 0.0,
            "rmse_mean": float(np.sqrt(mean_squared_error(y_true_val, y_pred_val))),
            "mae_mean": float(mean_absolute_error(y_true_val, y_pred_val)),
            "ee_mae_mean": float(compute_ee_mae(y_true_val, y_pred_val, T_val)),
            "train_time_s": elapsed,
            "oof_pred": oof_pred,
            "oof_y_true": oof_y,
        }

    # --- sklearn-style models: K-fold CV ---
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores, ee_mae_scores = [], [], [], []
    oof_pred = np.full(len(y_ddG), np.nan)
    t0 = time.time()

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_ddG[train_idx], y_ddG[val_idx]
        T_val = temperatures_K[val_idx]

        y_pred = _simple_fold_predict(model_name, model_kwargs, X_tr, y_tr, X_val)

        oof_pred[val_idx] = y_pred
        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        mae_scores.append(float(mean_absolute_error(y_val, y_pred)))
        ee_mae_scores.append(compute_ee_mae(y_val, y_pred, T_val))

    elapsed = time.time() - t0
    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "ee_mae_mean": float(np.mean(ee_mae_scores)),
        "train_time_s": elapsed,
        "oof_pred": oof_pred,
        "oof_y_true": y_ddG.copy(),
    }


def _simple_fold_predict(
    model_name: str,
    model_kwargs: Optional[Dict],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """Train a fresh sklearn-style model on one fold and predict."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler

    kw = model_kwargs or {}
    if model_name == "rf":
        m = RandomForestRegressor(
            n_estimators=kw.get("n_estimators", 300),
            n_jobs=kw.get("n_jobs", -1),
            random_state=42,
        )
        m.fit(X_tr, y_tr)
        return m.predict(X_val)
    elif model_name == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict(X_val)
    elif model_name == "svm":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        subsample = kw.get("subsample_n", 4000)
        if subsample and len(X_tr_s) > subsample:
            idx = np.random.choice(len(X_tr_s), subsample, replace=False)
            X_tr_s, y_tr = X_tr_s[idx], y_tr[idx]
        m = SVR(kernel="rbf", C=10.0, epsilon=0.1)
        m.fit(X_tr_s, y_tr)
        return m.predict(X_val_s)
    else:
        return np.full(len(X_val), np.nan)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ECC-Track: ΔΔG prediction from SMILES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX,
                   help="Path to Raw_Dataset.xlsx")
    p.add_argument("--desc", nargs="+", default=ECC_DESCRIPTORS,
                   choices=list(DESCRIPTOR_REGISTRY.keys()),
                   help="Descriptors to evaluate")
    p.add_argument("--model", nargs="+", default=_ALL_MODELS,
                   choices=_ALL_MODELS,
                   help="Models to evaluate")
    p.add_argument("--nrows", type=int, default=None,
                   help="Limit to first N rows (None = all)")
    p.add_argument("--cv", type=int, default=5,
                   help="K for K-fold CV")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--heartbeat", type=float, default=30.0)
    p.add_argument("--svm-subsample", type=int, default=4000)
    p.add_argument("--append", action="store_true",
                   help="Append to existing metrics_summary.csv instead of overwriting")
    return p.parse_args()


def _print_table(rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [c for c in [
        "descriptor", "model", "feature_dim", "n_samples",
        "r2_mean", "r2_std", "rmse_mean", "mae_mean", "ee_mae_mean", "train_time_s",
    ] if c in df.columns]
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print()
    print(df[cols].to_string(index=False))


def main() -> int:
    args = parse_args()

    if not args.xlsx.exists():
        print(f"[error] XLSX not found: {args.xlsx}", file=sys.stderr)
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log_fh)

    print(f"[log]  {log_path}")
    print(f"[load] {args.xlsx}  (nrows={args.nrows or 'all'})")

    lig_smi, prod_smi, temps_K, ddG = load_ecc_dataset(args.xlsx)

    if args.nrows is not None:
        lig_smi = lig_smi[: args.nrows]
        prod_smi = prod_smi[: args.nrows]
        temps_K = temps_K[: args.nrows]
        ddG = ddG[: args.nrows]

    print(
        f"[load] n={len(lig_smi)}  "
        f"ΔΔG range=[{ddG.min():.3f}, {ddG.max():.3f}] kcal/mol  "
        f"T range=[{temps_K.min():.1f}, {temps_K.max():.1f}] K"
    )
    print(f"[out]  {args.results_dir}")
    print(f"[grid] descriptors={args.desc}  models={args.model}  cv={args.cv}")

    model_kwargs_by_name = {
        "svm": {"subsample_n": args.svm_subsample},
    }

    rows: List[dict] = []
    total = len(args.desc) * len(args.model)
    counter = 0
    grid_t0 = time.time()

    for desc in args.desc:
        # Build reaction features once per descriptor (shared across models)
        print(f"\n[feat] building features with '{desc}' ...")
        try:
            X, mask = _build_reaction_features(lig_smi, prod_smi, temps_K, desc)
        except Exception as e:
            print(f"  [fail] featurization failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            for model in args.model:
                counter += 1
                rows.append({"descriptor": desc, "model": model,
                             "error": f"feat failed: {e}"})
            continue

        y_used = ddG[mask]
        T_used = temps_K[mask]
        print(f"  n_valid={int(mask.sum())}  feature_dim={X.shape[1]}")

        for model_name in args.model:
            counter += 1
            label = f"{desc} x {model_name}"
            print(f"\n[{counter}/{total}] {label}  ...", flush=True)

            hb: Optional[_Heartbeat] = None
            if args.heartbeat > 0:
                hb = _Heartbeat(interval=args.heartbeat, label=label)
                hb.start()
            try:
                metrics = _cross_validate_ecc(
                    X, y_used, T_used,
                    model_name=model_name,
                    cv=args.cv,
                    model_kwargs=model_kwargs_by_name.get(model_name),
                )
            except Exception as e:
                if hb is not None:
                    hb.stop()
                print(f"  [fail] {type(e).__name__}: {e}", file=sys.stderr)
                rows.append({"descriptor": desc, "model": model_name,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            finally:
                if hb is not None:
                    hb.stop()

            oof_pred = metrics.pop("oof_pred", None)
            oof_y_true = metrics.pop("oof_y_true", None)

            print(
                f"  R2={metrics.get('r2_mean'):.4f}  "
                f"RMSE={metrics.get('rmse_mean'):.4f}  "
                f"ee_MAE={metrics.get('ee_mae_mean'):.2f}%  "
                f"t={metrics.get('train_time_s'):.1f}s",
                flush=True,
            )

            if not args.skip_plots and oof_pred is not None:
                valid = ~np.isnan(oof_pred)
                if valid.sum() >= 2:
                    out = plot_scatter(
                        oof_y_true[valid], oof_pred[valid],
                        desc=desc, model=model_name,
                        save_dir=args.results_dir,
                        x_label="True ΔΔG (kcal/mol)",
                        y_label="Predicted ΔΔG (kcal/mol)",
                    )
                    print(f"  [plot] {out.name}")

            row: dict = {
                "descriptor": desc,
                "model": model_name,
                "n_samples": int(mask.sum()),
                "n_total": int(len(mask)),
                "feature_dim": int(X.shape[1]),
                "cv": args.cv,
            }
            row.update(metrics)
            rows.append(row)

    df_out = pd.DataFrame(rows)
    keep = [c for c in CSV_COLUMNS if c in df_out.columns]
    extra = [c for c in df_out.columns if c not in keep]
    df_out = df_out[keep + extra]
    summary_path = args.results_dir / "metrics_summary.csv"
    if args.append and summary_path.exists():
        prev = pd.read_csv(summary_path)
        if {"descriptor", "model"}.issubset(prev.columns):
            keys = set(zip(df_out["descriptor"], df_out["model"]))
            prev = prev[
                ~prev.apply(lambda r: (r["descriptor"], r["model"]) in keys, axis=1)
            ]
        df_out = pd.concat([prev, df_out], ignore_index=True, sort=False)
        print(f"[save] appending ({len(prev)} prior rows kept)")
    df_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {summary_path}")

    _print_table(rows)
    print(f"\n[done] {counter} combinations in {time.time() - grid_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
