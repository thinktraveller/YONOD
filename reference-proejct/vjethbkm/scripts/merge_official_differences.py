from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


DESCRIPTOR_MAP = {
    "ohe": "OHE",
    "morgan": "MFP",
    "physchem": "PhysChem",
}
MODEL_MAP = {"rf": "RF"}
METRIC_COLUMNS = {
    "mae": "mae_mean",
    "rmse": "rmse_mean",
    "r2": "r2_mean",
    "kendall_tau": "kendall_tau_mean",
}


def _local_records(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for _, row in summary.iterrows():
        descriptor = DESCRIPTOR_MAP.get(str(row["descriptor"]), str(row["descriptor"]))
        model = MODEL_MAP.get(str(row["model"]), str(row["model"]))
        for metric, column in METRIC_COLUMNS.items():
            records.append(
                {
                    "dataset": row["dataset"],
                    "descriptor": descriptor,
                    "model": model,
                    "metric": metric,
                    "reproduced_value": float(row[column]),
                    "local_descriptor_name": row["descriptor"],
                    "local_model_name": row["model"],
                    "local_folds": int(row["folds"]),
                    "local_feature_dim": int(row["feature_dim"]),
                }
            )
    return pd.DataFrame.from_records(records)


def merge_targets(repro_root: Path) -> pd.DataFrame:
    table_dir = repro_root / "outputs" / "tables"
    targets_path = table_dir / "table_s3_official_rf_targets.csv"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing official target table: {targets_path}")
    targets = pd.read_csv(targets_path)

    local_frames = []
    for summary_path in sorted(table_dir.glob("core_rf_5x5_*_metrics_summary.csv")):
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        local_frames.append(_local_records(summary))
    if local_frames:
        local = pd.concat(local_frames, ignore_index=True)
    else:
        local = pd.DataFrame(
            columns=[
                "dataset",
                "descriptor",
                "model",
                "metric",
                "reproduced_value",
                "local_descriptor_name",
                "local_model_name",
                "local_folds",
                "local_feature_dim",
            ]
        )

    merged = targets.drop(columns=["reproduced_value", "abs_diff", "rel_diff", "status"], errors="ignore").merge(
        local,
        how="left",
        on=["dataset", "descriptor", "model", "metric"],
    )
    merged["abs_diff"] = (merged["reproduced_value"] - merged["paper_value"]).abs()
    merged["rel_diff"] = merged["abs_diff"] / merged["paper_value"].abs()
    merged["status"] = "official_target_extracted_pending_local_rerun"
    has_local = merged["reproduced_value"].notna()
    merged.loc[has_local, "status"] = "local_compatible_rerun_compared"
    merged.loc[
        has_local, "explanation"
    ] = (
        "Local compatible benchmark merged against official target. "
        "This is not guaranteed bit-identical to the official code path unless descriptor/model settings are audited separately."
    )
    return merged.sort_values(["dataset", "descriptor", "metric"]).reset_index(drop=True)


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    merged = merge_targets(repro_root)
    output_path = repro_root / "outputs" / "tables" / "table_s3_local_vs_official_differences.csv"
    merged.to_csv(output_path, index=False)
    summary = {
        "rows": int(len(merged)),
        "matched_local_rows": int(merged["reproduced_value"].notna().sum()),
        "datasets_with_local_rows": sorted(merged.loc[merged["reproduced_value"].notna(), "dataset"].unique().tolist()),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
