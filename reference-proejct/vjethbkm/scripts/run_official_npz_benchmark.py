from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


DATASET_MAP = {
    "BH": {"source_folder": "Doyle_2018", "prefix": "Doyle"},
    "BH2": {"source_folder": "Denmark_2023", "prefix": "Denmark"},
    "SM": {"source_folder": "Suzuki_2018", "prefix": "Suzuki"},
    "SLAP": {"source_folder": "Bode_2023", "prefix": "Bode"},
}


def _git_commit(repro_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repro_root.parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "kendall_tau": float(kendalltau(y_true, y_pred)[0]),
    }


def run_npz_benchmark(
    repro_root: Path,
    *,
    dataset: str,
    descriptors: list[str],
    folds: int,
    repeats: int,
    seed: int,
    n_estimators: int,
    max_features: float,
) -> dict:
    dataset_info = DATASET_MAP[dataset]
    compare_root = repro_root / "yieldsmarter" / "Results" / "Compare_Complexity"
    run_id = f"official_npz_rf_5x5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = repro_root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for descriptor in descriptors:
        npz_path = (
            compare_root
            / dataset_info["source_folder"]
            / descriptor
            / f"{dataset_info['prefix']}_{descriptor}.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        X = data["X"]
        y = data["y"]
        if np.issubdtype(X.dtype, np.floating):
            X = np.nan_to_num(X, copy=True, nan=0.0)

        metric_rows = []
        for repeat in range(repeats):
            inner_seed = seed + repeat
            splitter = KFold(n_splits=folds, shuffle=True, random_state=inner_seed)
            for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    n_jobs=-1,
                    random_state=inner_seed,
                )
                start = perf_counter()
                model.fit(X.take(train_idx, axis=0), y.take(train_idx, axis=0))
                elapsed = round(perf_counter() - start, 6)
                pred = model.predict(X.take(test_idx, axis=0))
                metric_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "train_elapsed_s": elapsed,
                        **_metrics(y.take(test_idx, axis=0), pred),
                    }
                )
        metrics = pd.DataFrame(metric_rows)
        rows.append(
            {
                "stage": "official_npz_rf_5x5",
                "dataset": dataset,
                "descriptor": descriptor,
                "model": "rf",
                "mae_mean": metrics["mae"].mean(),
                "rmse_mean": metrics["rmse"].mean(),
                "r2_mean": metrics["r2"].mean(),
                "kendall_tau_mean": metrics["kendall_tau"].mean(),
                "feature_dim": int(X.shape[1]),
                "folds": int(len(metrics)),
                "train_elapsed_s_mean": metrics["train_elapsed_s"].mean(),
                "feature_elapsed_s_mean": 0.0,
                "source_npz": str(npz_path.relative_to(repro_root)).replace("\\", "/"),
            }
        )
        metrics.to_csv(run_dir / f"{dataset}_{descriptor}_fold_metrics.csv", index=False)

    table_dir = repro_root / "outputs" / "tables"
    report_dir = repro_root / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / f"official_npz_rf_5x5_{dataset}_metrics_summary.csv"
    new_summary = pd.DataFrame(rows)
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous.loc[~previous["descriptor"].isin(new_summary["descriptor"])]
        summary = pd.concat([previous, new_summary], ignore_index=True)
    else:
        summary = new_summary
    summary = summary.sort_values(["dataset", "descriptor"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "run_id": run_id,
        "stage": "official_npz_rf_5x5",
        "dataset": dataset,
        "descriptors": descriptors,
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "git_commit": _git_commit(repro_root),
        "summary_path": str(summary_path),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = report_dir / f"official_npz_rf_5x5_{dataset}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Official NPZ RF 5x5 benchmark: {dataset}",
                "",
                f"Run id: `{run_id}`",
                "",
                "This run uses official precomputed `.npz` descriptors from the local yieldsmarter package.",
                "Raw `.npz` files remain ignored and are not copied into git.",
                "",
                "```csv",
                summary.round(6).to_csv(index=False).strip(),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_MAP))
    parser.add_argument("--descriptors", default="MFP,PhysChem")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", type=float, default=0.3)
    args = parser.parse_args()
    repro_root = Path(__file__).resolve().parents[1]
    descriptors = [item.strip() for item in args.descriptors.split(",") if item.strip()]
    result = run_npz_benchmark(
        repro_root,
        dataset=args.dataset,
        descriptors=descriptors,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
