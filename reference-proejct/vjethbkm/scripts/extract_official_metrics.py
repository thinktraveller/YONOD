from __future__ import annotations

import csv
import json
from pathlib import Path


DATASET_MAP = {
    "Doyle_2018": {"dataset_id": "BH", "source_dataset": "BH1", "short_name": "Doyle"},
    "Denmark_2023": {"dataset_id": "BH2", "source_dataset": "BH2", "short_name": "Denmark"},
    "Suzuki_2018": {"dataset_id": "SM", "source_dataset": "SM", "short_name": "Suzuki"},
    "Bode_2023": {"dataset_id": "SLAP", "source_dataset": "SL1", "short_name": "Bode"},
}

TABLE_S3_DESCRIPTORS = {"OHE", "MFP", "PhysChem"}
TABLE_S3_METRICS = {"mae", "rmse", "r2", "kendall"}
METRIC_CANONICAL = {"kendall": "kendall_tau"}


def _write_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _iter_summary_files(compare_root: Path) -> list[Path]:
    return sorted(compare_root.glob("*/*/*_metrics_summaries_RF.json"))


def extract_records(repro_root: Path) -> tuple[list[dict], list[dict]]:
    compare_root = repro_root / "yieldsmarter" / "Results" / "Compare_Complexity"
    if not compare_root.exists():
        raise FileNotFoundError(f"Missing official Compare_Complexity root: {compare_root}")

    all_records: list[dict] = []
    table_s3_records: list[dict] = []
    for path in _iter_summary_files(compare_root):
        source_folder = path.relative_to(compare_root).parts[0]
        descriptor = path.relative_to(compare_root).parts[1]
        dataset_info = DATASET_MAP.get(source_folder)
        if not dataset_info:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", "RF")
        for split in ("train", "test"):
            for metric, payload in data.get(split, {}).items():
                metric_name = METRIC_CANONICAL.get(metric, metric)
                record = {
                    "official_source": "yieldsmarter/Results/Compare_Complexity",
                    "source_folder": source_folder,
                    "dataset_id": dataset_info["dataset_id"],
                    "source_dataset": dataset_info["source_dataset"],
                    "descriptor": descriptor,
                    "model": model,
                    "split": split,
                    "metric": metric_name,
                    "mean": payload.get("mean"),
                    "std": payload.get("std"),
                    "splits": data.get("splits"),
                    "outer_splits": data.get("outer_splits", ""),
                    "seed": data.get("seed"),
                    "descriptors_npz": data.get("descriptors_npz", ""),
                    "relative_json_path": str(path.relative_to(repro_root)).replace("\\", "/"),
                }
                all_records.append(record)

                if (
                    split == "test"
                    and descriptor in TABLE_S3_DESCRIPTORS
                    and metric in TABLE_S3_METRICS
                ):
                    table_s3_records.append(
                        {
                            "paper_target_id": "official_yieldsmarter_compare_complexity_rf",
                            "dataset": dataset_info["dataset_id"],
                            "source_dataset": dataset_info["source_dataset"],
                            "descriptor": descriptor,
                            "model": model,
                            "split": "5x5_repeated_cv",
                            "metric": metric_name,
                            "paper_value": payload.get("mean"),
                            "paper_std": payload.get("std"),
                            "reproduced_value": "",
                            "abs_diff": "",
                            "rel_diff": "",
                            "status": "official_target_extracted_pending_local_rerun",
                            "explanation": (
                                "Target extracted from local official yieldsmarter RF summary JSON; "
                                "local benchmark rerun has not yet been merged into this row."
                            ),
                            "relative_json_path": str(path.relative_to(repro_root)).replace("\\", "/"),
                        }
                    )
    return all_records, sorted(
        table_s3_records,
        key=lambda row: (row["dataset"], row["descriptor"], row["metric"]),
    )


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    all_records, table_s3_records = extract_records(repro_root)
    if len(all_records) != 200:
        raise RuntimeError(f"Expected 200 official RF metric records, got {len(all_records)}")
    if len(table_s3_records) != 48:
        raise RuntimeError(f"Expected 48 Table S3 target records, got {len(table_s3_records)}")

    table_dir = repro_root / "outputs" / "tables"
    _write_csv(
        table_dir / "official_compare_complexity_rf_metrics.csv",
        all_records,
        [
            "official_source",
            "source_folder",
            "dataset_id",
            "source_dataset",
            "descriptor",
            "model",
            "split",
            "metric",
            "mean",
            "std",
            "splits",
            "outer_splits",
            "seed",
            "descriptors_npz",
            "relative_json_path",
        ],
    )
    _write_csv(
        table_dir / "table_s3_official_rf_targets.csv",
        table_s3_records,
        [
            "paper_target_id",
            "dataset",
            "source_dataset",
            "descriptor",
            "model",
            "split",
            "metric",
            "paper_value",
            "paper_std",
            "reproduced_value",
            "abs_diff",
            "rel_diff",
            "status",
            "explanation",
            "relative_json_path",
        ],
    )
    print(
        json.dumps(
            {
                "official_rf_metric_records": len(all_records),
                "table_s3_target_records": len(table_s3_records),
                "datasets": sorted({row["dataset"] for row in table_s3_records}),
                "descriptors": sorted({row["descriptor"] for row in table_s3_records}),
                "metrics": sorted({row["metric"] for row in table_s3_records}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
