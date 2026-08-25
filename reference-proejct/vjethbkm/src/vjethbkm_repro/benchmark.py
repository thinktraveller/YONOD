"""Repeated-CV benchmarks for the VJETHBKM reproduction."""

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


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
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


def table_s3_style_summary(summary: pd.DataFrame, status: str, explanation: str) -> pd.DataFrame:
    records: list[dict] = []
    metric_map = {
        "mae": "mae_mean",
        "rmse": "rmse_mean",
        "r2": "r2_mean",
        "kendall_tau": "kendall_tau_mean",
    }
    for _, row in summary.iterrows():
        for metric, column in metric_map.items():
            records.append(
                {
                    "paper_target_id": "si_table_s3_rf_cv",
                    "dataset": row["dataset"],
                    "descriptor": row["descriptor"],
                    "model": row["model"],
                    "split": "5x5_repeated_cv" if row["stage"] == "core_rf_5x5" else row["stage"],
                    "metric": metric,
                    "paper_value": "",
                    "reproduced_value": row[column],
                    "abs_diff": "",
                    "rel_diff": "",
                    "status": status,
                    "explanation": explanation,
                }
            )
    return pd.DataFrame.from_records(records)


def _run_rf_repeated_cv(
    *,
    stage: str,
    dataset_id: str,
    repeats: int,
    folds: int,
    descriptors: list[str],
    n_estimators: int,
    rf_max_features: float | int | str | None,
    random_state: int,
    summary_name: str,
    report_name: str,
    table_s3_name: str | None = None,
) -> dict:
    dataset_cfg = dataset_config_from_sources(dataset_id)
    df = read_dataset(dataset_cfg).reset_index(drop=True)
    y = pd.to_numeric(df[dataset_cfg.label_col], errors="raise")

    run_id = _run_id(stage)
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
                model_random_state = random_state + repeat - 1
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
                    n_estimators=n_estimators,
                    random_state=model_random_state,
                    n_jobs=-1,
                    max_features=rf_max_features,
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
                        "stage": stage,
                        "dataset": dataset_cfg.dataset_id,
                        "descriptor": descriptor,
                        "model": "rf",
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(train_ids),
                        "n_test": len(test_ids),
                        "feature_dim": feature.dim,
                        "invalid_smiles": feature.invalid_smiles,
                        "model_random_state": model_random_state,
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
    summary = _summarize_metrics(metrics)
    summary_path = table_dir / summary_name
    summary.to_csv(summary_path, index=False)

    table_s3_path = None
    if table_s3_name:
        status = "smoke_validation_only" if dataset_id == "smoke_local_yonod" else "pending_paper_value_materialization"
        explanation = (
            "Pipeline shape validated on local smoke data; official VJETHBKM data and Table S3 values are not yet materialized."
            if dataset_id == "smoke_local_yonod"
            else "Official data run completed, but paper values still need to be filled from tracked Table S3 targets."
        )
        table_s3 = table_s3_style_summary(summary, status=status, explanation=explanation)
        table_s3_path = table_dir / table_s3_name
        table_s3.to_csv(table_s3_path, index=False)

    manifest = {
        "run_id": run_id,
        "stage": stage,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset_cfg.dataset_id,
        "dataset_path": str(dataset_cfg.path),
        "n_rows": int(len(df)),
        "descriptors": descriptors,
        "model": "rf",
        "n_estimators": n_estimators,
        "rf_max_features": rf_max_features,
        "folds": folds,
        "repeats": repeats,
        "random_state": random_state,
        "outputs": {
            "run_dir": str(run_dir),
            "summary": str(summary_path),
            "table_s3_style_summary": str(table_s3_path) if table_s3_path else None,
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = report_dir / report_name
    report_lines = [
        f"# VJETHBKM {stage} reproduction report",
        "",
        f"Run id: `{run_id}`",
        "",
        f"This run validates `{stage}` on dataset `{dataset_id}`.",
        "It is not a claim of numerical reproduction of the JACS paper.",
        "",
        "## Metric summary",
        "",
        _markdown_table(summary.round(6)),
        "",
        "## Evidence status",
        "",
        "- Official yieldsmarter package targets can be materialized from `outputs/tables/table_s3_official_rf_targets.csv` when available.",
        "- This runner uses the local compatibility implementation under `reference-proejct/vjethbkm/src/`.",
        "- DFT/SOAP are not generated by this runner unless an explicit adapter is added.",
        "- Outputs were kept under `reference-proejct/vjethbkm/outputs/`.",
    ]
    if table_s3_path:
        report_lines.extend(["", f"Table S3 style summary: `{table_s3_path}`"])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "table_s3_style_summary": str(table_s3_path) if table_s3_path else None,
        "summary": summary.to_dict(orient="records"),
    }


def run_smoke() -> dict:
    return _run_rf_repeated_cv(
        stage="smoke",
        dataset_id="smoke_local_yonod",
        repeats=1,
        folds=2,
        descriptors=["ohe", "morgan", "physchem"],
        n_estimators=200,
        rf_max_features=1.0,
        random_state=42,
        summary_name="smoke_metrics_summary.csv",
        report_name="smoke_reproduction_report.md",
    )


def run_core_rf_5x5(
    dataset_id: str,
    allow_smoke_dataset: bool = False,
    descriptors: list[str] | None = None,
    repeats: int = 5,
    folds: int = 5,
    n_estimators: int = 500,
    rf_max_features: float | int | str | None = 0.3,
    random_state: int = 1000,
) -> dict:
    if dataset_id == "smoke_local_yonod" and not allow_smoke_dataset:
        raise ValueError(
            "core_rf_5x5 on smoke_local_yonod requires --allow-smoke-dataset "
            "so the output is not confused with official VJETHBKM reproduction."
        )
    selected_descriptors = descriptors or ["ohe", "morgan", "physchem"]
    return _run_rf_repeated_cv(
        stage="core_rf_5x5",
        dataset_id=dataset_id,
        repeats=repeats,
        folds=folds,
        descriptors=selected_descriptors,
        n_estimators=n_estimators,
        rf_max_features=rf_max_features,
        random_state=random_state,
        summary_name=f"core_rf_5x5_{dataset_id}_metrics_summary.csv",
        report_name=f"core_rf_5x5_{dataset_id}_reproduction_report.md",
        table_s3_name=f"table_s3_style_{dataset_id}_summary.csv",
    )
