"""Stage A smoke benchmark for the VJETHBKM reproduction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
import json
import platform
import subprocess

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import build_descriptor
from .manifest import dataset_config_from_sources, read_dataset
from .metrics import regression_metrics
from .paths import REPRO_ROOT
from .splits import repeated_kfold_manifest


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPRO_ROOT.parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _run_id(stage: str) -> str:
    return f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [columns, ["---"] * len(columns)]
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in columns])
    return "\n".join("| " + " | ".join(values) + " |" for values in rows)


def run_smoke() -> dict:
    dataset_cfg = dataset_config_from_sources("smoke_local_yonod")
    df = read_dataset(dataset_cfg).reset_index(drop=True)
    y = pd.to_numeric(df[dataset_cfg.label_col], errors="raise")

    descriptors = ["ohe", "morgan", "physchem"]
    repeats = 1
    folds = 2
    random_state = 42
    run_id = _run_id("smoke")
    run_dir = REPRO_ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = repeated_kfold_manifest(df, folds, repeats, random_state)
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)

    metrics_records: list[dict] = []
    prediction_records: list[dict] = []
    feature_records: list[dict] = []

    for descriptor in descriptors:
        for repeat in range(1, repeats + 1):
            for fold in range(1, folds + 1):
                test_ids = split_manifest.loc[
                    (split_manifest["repeat"] == repeat)
                    & (split_manifest["fold"] == fold)
                    & (~split_manifest["is_train"]),
                    "sample_id",
                ].tolist()
                train_ids = split_manifest.loc[
                    (split_manifest["repeat"] == repeat)
                    & (split_manifest["fold"] == fold)
                    & (split_manifest["is_train"]),
                    "sample_id",
                ].tolist()
                train_df = df.iloc[train_ids]
                test_df = df.iloc[test_ids]

                feature = build_descriptor(
                    descriptor,
                    train_df,
                    test_df,
                    dataset_cfg.component_cols,
                )
                feature_records.append(
                    {
                        "run_id": run_id,
                        "descriptor": descriptor,
                        "repeat": repeat,
                        "fold": fold,
                        "feature_dim": feature.dim,
                        "feature_elapsed_s": feature.elapsed_s,
                        "invalid_smiles": feature.invalid_smiles,
                    }
                )

                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                    min_samples_leaf=1,
                )
                train_start = perf_counter()
                model.fit(feature.train, y.iloc[train_ids])
                train_elapsed_s = round(perf_counter() - train_start, 6)

                predict_start = perf_counter()
                pred = model.predict(feature.test)
                predict_elapsed_s = round(perf_counter() - predict_start, 6)
                fold_metrics = regression_metrics(y.iloc[test_ids], pred)
                metrics_records.append(
                    {
                        "run_id": run_id,
                        "stage": "smoke",
                        "dataset": dataset_cfg.dataset_id,
                        "descriptor": descriptor,
                        "model": "rf",
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(train_ids),
                        "n_test": len(test_ids),
                        "feature_dim": feature.dim,
                        "invalid_smiles": feature.invalid_smiles,
                        "feature_elapsed_s": feature.elapsed_s,
                        "train_elapsed_s": train_elapsed_s,
                        "predict_elapsed_s": predict_elapsed_s,
                        **fold_metrics,
                    }
                )
                for sample_id, truth, value in zip(test_ids, y.iloc[test_ids], pred):
                    prediction_records.append(
                        {
                            "run_id": run_id,
                            "sample_id": int(sample_id),
                            "descriptor": descriptor,
                            "model": "rf",
                            "repeat": repeat,
                            "fold": fold,
                            "y_true": float(truth),
                            "y_pred": float(value),
                        }
                    )

    metrics = pd.DataFrame(metrics_records)
    predictions = pd.DataFrame(prediction_records)
    feature_summary = pd.DataFrame(feature_records)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    feature_summary.to_csv(run_dir / "feature_summary.csv", index=False)

    table_dir = REPRO_ROOT / "outputs" / "tables"
    report_dir = REPRO_ROOT / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        metrics.groupby(["stage", "dataset", "descriptor", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            kendall_tau_mean=("kendall_tau", "mean"),
            feature_dim=("feature_dim", "first"),
            folds=("fold", "count"),
            train_elapsed_s_mean=("train_elapsed_s", "mean"),
            feature_elapsed_s_mean=("feature_elapsed_s", "mean"),
        )
        .sort_values(["mae_mean", "rmse_mean"])
    )
    summary_path = table_dir / "smoke_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "run_id": run_id,
        "stage": "smoke",
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset_cfg.dataset_id,
        "dataset_path": str(dataset_cfg.path),
        "n_rows": int(len(df)),
        "descriptors": descriptors,
        "model": "rf",
        "folds": folds,
        "repeats": repeats,
        "random_state": random_state,
        "outputs": {
            "run_dir": str(run_dir),
            "summary": str(summary_path),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = report_dir / "smoke_reproduction_report.md"
    report_lines = [
        "# VJETHBKM smoke reproduction report",
        "",
        f"Run id: `{run_id}`",
        "",
        "This smoke run validates the Stage A pipeline shape on a small local YONOD dataset.",
        "It is not a claim of numerical reproduction of the JACS paper.",
        "",
        "## Metric summary",
        "",
        _markdown_table(summary.round(6)),
        "",
        "## Evidence status",
        "",
        "- Paper/SI numerical targets are still marked `pending_direct_si_audit`.",
        "- Official ETH data/code package inspection is pending.",
        "- Outputs were kept under `reference-proejct/vjethbkm/outputs/`.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary.to_dict(orient="records"),
    }
