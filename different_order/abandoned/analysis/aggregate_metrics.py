"""Aggregate different_order metrics and check result completeness."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"
EXPECTED_DESCRIPTORS = {"morgan", "maccs", "rdkit2d", "fisd", "molmetalm", "maf"}
EXPECTED_MODELS = {"xgb", "rf", "svm", "autogluon"}
EXPECTED_ROWS_PER_TASK = len(EXPECTED_DESCRIPTORS) * len(EXPECTED_MODELS)


def task_dirs() -> list[Path]:
    return sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    completeness: list[dict[str, str]] = []

    for task_dir in task_dirs():
        task = task_dir.name
        metrics_path = task_dir / "metrics_summary.csv"
        task_rows = read_metrics(metrics_path) if metrics_path.exists() else []
        rows.extend(task_rows)

        descriptors = {row.get("descriptor", "") for row in task_rows}
        models = {row.get("model", "") for row in task_rows}
        pairs = {
            (row.get("descriptor", ""), row.get("model", ""))
            for row in task_rows
        }
        expected_pairs = {
            (descriptor, model)
            for descriptor in EXPECTED_DESCRIPTORS
            for model in EXPECTED_MODELS
        }

        completeness.append(
            {
                "task_name": task,
                "metrics_file_exists": str(metrics_path.exists()),
                "row_count": str(len(task_rows)),
                "expected_row_count": str(EXPECTED_ROWS_PER_TASK),
                "descriptor_count": str(len(descriptors)),
                "model_count": str(len(models)),
                "missing_pairs": ";".join(
                    f"{descriptor}:{model}"
                    for descriptor, model in sorted(expected_pairs - pairs)
                ),
                "extra_pairs": ";".join(
                    f"{descriptor}:{model}"
                    for descriptor, model in sorted(pairs - expected_pairs)
                ),
                "complete": str(
                    metrics_path.exists()
                    and len(task_rows) == EXPECTED_ROWS_PER_TASK
                    and pairs == expected_pairs
                ),
            }
        )

    all_metrics_path = OUT_DIR / "all_metrics.csv"
    if rows:
        with all_metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    completeness_path = OUT_DIR / "completeness_summary.csv"
    with completeness_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completeness[0].keys()))
        writer.writeheader()
        writer.writerows(completeness)

    complete_count = sum(row["complete"] == "True" for row in completeness)
    print(f"tasks={len(completeness)}")
    print(f"metrics_rows={len(rows)}")
    print(f"complete_tasks={complete_count}")
    print(f"all_metrics={all_metrics_path}")
    print(f"completeness_summary={completeness_path}")
    return 0 if complete_count == len(completeness) else 1


if __name__ == "__main__":
    raise SystemExit(main())
