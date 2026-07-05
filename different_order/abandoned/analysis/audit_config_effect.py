"""Audit whether task config columns appear to affect recorded metrics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"
COLUMN_MAP = {
    "1": "reactant-amide",
    "2": "reactant-acid",
    "3": "product",
    "4": "activation",
    "5": "additive",
    "6": "base",
    "7": "solvent",
}
METRIC_FIELDS = ["r2_mean", "r2_std", "rmse_mean", "mae_mean", "feature_dim", "n_smiles_cols"]


def task_dirs() -> list[Path]:
    return sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_columns(task: str) -> list[str]:
    return [COLUMN_MAP[digit] for digit in task]


def float_key(value: str) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return value


def metric_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(float_key(row.get(field, "")) for field in METRIC_FIELDS)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, str]] = []
    metric_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for task_dir in task_dirs():
        task = task_dir.name
        config_path = task_dir / f"{task}_yonod_config.json"
        metrics_path = task_dir / "metrics_summary.csv"

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        expected = expected_columns(task)
        descriptors = config.get("descriptors", [])
        descriptor_columns_ok = all(
            descriptor.get("columns") == expected for descriptor in descriptors
        )

        rows = read_metrics(metrics_path)
        for row in rows:
            descriptor = row.get("descriptor", "")
            model = row.get("model", "")
            metric_groups[(descriptor, model)].append(row)

            descriptor_config = next(
                (
                    item
                    for item in descriptors
                    if item.get("descriptor") == descriptor
                ),
                {},
            )
            configured_columns = descriptor_config.get("columns", [])
            expected_feature_dim = ""
            if row.get("feature_dim") and row.get("n_smiles_cols"):
                try:
                    actual_dim = int(float(row["feature_dim"]))
                    actual_cols = int(float(row["n_smiles_cols"]))
                    per_column_dim = actual_dim // actual_cols if actual_cols else 0
                    expected_feature_dim = str(per_column_dim * len(configured_columns))
                except ValueError:
                    expected_feature_dim = ""

            audit_rows.append(
                {
                    "task_name": task,
                    "descriptor": descriptor,
                    "model": model,
                    "configured_columns": "|".join(configured_columns),
                    "configured_column_count": str(len(configured_columns)),
                    "config_matches_task_name": str(configured_columns == expected),
                    "all_descriptor_configs_match_task_name": str(descriptor_columns_ok),
                    "reported_n_smiles_cols": row.get("n_smiles_cols", ""),
                    "reported_feature_dim": row.get("feature_dim", ""),
                    "expected_feature_dim_if_config_used": expected_feature_dim,
                    "reported_dim_matches_configured_columns": str(
                        expected_feature_dim == row.get("feature_dim", "")
                    ),
                    "r2_mean": row.get("r2_mean", ""),
                    "rmse_mean": row.get("rmse_mean", ""),
                    "mae_mean": row.get("mae_mean", ""),
                }
            )

    audit_path = OUT_DIR / "config_effect_audit.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    variation_rows: list[dict[str, str]] = []
    for (descriptor, model), rows in sorted(metric_groups.items()):
        signatures = {metric_signature(row) for row in rows}
        r2_values = [float(row["r2_mean"]) for row in rows]
        feature_dims = {row.get("feature_dim", "") for row in rows}
        n_smiles_cols = {row.get("n_smiles_cols", "") for row in rows}
        variation_rows.append(
            {
                "descriptor": descriptor,
                "model": model,
                "task_count": str(len(rows)),
                "unique_metric_signatures": str(len(signatures)),
                "r2_min": f"{min(r2_values):.12g}",
                "r2_max": f"{max(r2_values):.12g}",
                "r2_range": f"{max(r2_values) - min(r2_values):.12g}",
                "unique_feature_dims": "|".join(sorted(feature_dims)),
                "unique_n_smiles_cols": "|".join(sorted(n_smiles_cols)),
                "metrics_identical_across_tasks": str(len(signatures) == 1),
            }
        )

    variation_path = OUT_DIR / "metric_variation_by_descriptor_model.csv"
    with variation_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(variation_rows)

    mismatched_dim_count = sum(
        row["reported_dim_matches_configured_columns"] == "False"
        for row in audit_rows
    )
    identical_groups = sum(
        row["metrics_identical_across_tasks"] == "True"
        for row in variation_rows
    )
    print(f"audit_rows={len(audit_rows)}")
    print(f"reported_dim_mismatches={mismatched_dim_count}")
    print(f"descriptor_model_groups={len(variation_rows)}")
    print(f"identical_metric_groups={identical_groups}")
    print(f"config_effect_audit={audit_path}")
    print(f"metric_variation={variation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
