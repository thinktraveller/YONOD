from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import KFold


DATASETS = ["BH", "BH2", "SM", "SLAP"]


def _group_summary(df: pd.DataFrame, columns: list[str]) -> dict:
    group_sizes = df.groupby(columns, dropna=False).size()
    return {
        "folds": int(len(group_sizes)),
        "min_test_size": int(group_sizes.min()),
        "max_test_size": int(group_sizes.max()),
        "mean_test_size": float(group_sizes.mean()),
    }


def audit(repro_root: Path) -> pd.DataFrame:
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from vjethbkm_repro.manifest import dataset_config_from_sources, read_dataset

    rows = []
    for dataset_id in DATASETS:
        cfg = dataset_config_from_sources(dataset_id)
        df = read_dataset(cfg).reset_index(drop=True)
        splitter = KFold(n_splits=5, shuffle=True, random_state=1000)
        fold_sizes = [len(test_idx) for _, test_idx in splitter.split(df)]
        rows.append(
            {
                "dataset": dataset_id,
                "split_name": "holdout_0d_random_kfold_seed1000",
                "holdout_dim": "0d",
                "heldout_component": "",
                "n_rows": int(len(df)),
                "component_cols": "|".join(cfg.component_cols),
                "folds": 5,
                "min_test_size": int(min(fold_sizes)),
                "max_test_size": int(max(fold_sizes)),
                "mean_test_size": float(sum(fold_sizes) / len(fold_sizes)),
                "leakage_check_scope": "random split; component leakage not constrained",
                "status": "schema_summary_only",
            }
        )
        for component in cfg.component_cols:
            summary = _group_summary(df, [component])
            rows.append(
                {
                    "dataset": dataset_id,
                    "split_name": f"holdout_1d_{component}",
                    "holdout_dim": "1d",
                    "heldout_component": component,
                    "n_rows": int(len(df)),
                    "component_cols": "|".join(cfg.component_cols),
                    **summary,
                    "leakage_check_scope": "leave-one-component-value-out; no same component value appears in train by construction",
                    "status": "schema_summary_only",
                }
            )
        if len(cfg.component_cols) >= 2:
            pair_cols = cfg.component_cols[:2]
            summary = _group_summary(df, pair_cols)
            rows.append(
                {
                    "dataset": dataset_id,
                    "split_name": f"holdout_2d_{pair_cols[0]}__{pair_cols[1]}",
                    "holdout_dim": "2d",
                    "heldout_component": "+".join(pair_cols),
                    "n_rows": int(len(df)),
                    "component_cols": "|".join(cfg.component_cols),
                    **summary,
                    "leakage_check_scope": "leave-one-component-pair-out for first two configured component columns; full manifest intentionally not materialized",
                    "status": "schema_summary_only",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    summary = audit(repro_root)
    table_dir = repro_root / "outputs" / "tables"
    report_dir = repro_root / "outputs" / "reports"
    manifest_dir = repro_root / "data" / "manifest"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "official_component_split_schema_summary.csv"
    summary.to_csv(table_path, index=False)
    audit_json = {
        "datasets": DATASETS,
        "rows": int(len(summary)),
        "status": "schema_summary_only_not_paper_exact_training_result",
        "boundary": (
            "This audit avoids materializing large 2D manifests. It verifies formal dataset schema and holdout group sizes; "
            "paper-exact 0D/1D/2D training still requires the exact component mapping from the SI or official plotting scripts."
        ),
        "output": str(table_path.relative_to(repro_root)).replace("\\", "/"),
    }
    (manifest_dir / "official_component_split_audit.json").write_text(
        json.dumps(audit_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = report_dir / "official_component_split_audit_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Official dataset component split schema audit",
                "",
                audit_json["boundary"],
                "",
                "```csv",
                summary.to_csv(index=False).strip(),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
