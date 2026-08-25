from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def build_report(repro_root: Path) -> dict:
    table_dir = repro_root / "outputs" / "tables"
    report_dir = repro_root / "outputs" / "reports"
    manifest_dir = repro_root / "data" / "manifest"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    diff = pd.read_csv(table_dir / "table_s3_local_vs_official_differences.csv")
    bh2_external = pd.read_csv(table_dir / "bh2_external_187_summary.csv")
    high_yield = pd.read_csv(table_dir / "official_oof_high_yield_summary.csv")
    component = pd.read_csv(table_dir / "official_component_split_schema_summary.csv")

    descriptor_diff = (
        diff.groupby("descriptor", as_index=False)
        .agg(rows=("metric", "count"), matched=("reproduced_value", lambda s: int(s.notna().sum())), max_abs_diff=("abs_diff", "max"))
        .sort_values("descriptor")
    )
    unresolved = diff.loc[diff["status"].astype(str).str.contains("forensic_unresolved", na=False)]
    status = {
        "table_s3_rows": int(len(diff)),
        "table_s3_matched_rows": int(diff["reproduced_value"].notna().sum()),
        "descriptor_max_abs_diff": {
            row["descriptor"]: float(row["max_abs_diff"]) for _, row in descriptor_diff.iterrows()
        },
        "unresolved_rows": int(len(unresolved)),
        "unresolved_scope": sorted(
            {
                f"{row.dataset}/{row.descriptor}/{row.metric}"
                for row in unresolved.itertuples(index=False)
            }
        ),
        "bh2_external_187": bh2_external.iloc[0].to_dict(),
        "high_yield_rows": int(len(high_yield)),
        "component_split_schema_rows": int(len(component)),
        "boundary": (
            "MFP and PhysChem RF Table S3 targets are reproduced through official precomputed NPZ features. "
            "OHE is a local compatibility rerun; SM/OHE remains a documented unresolved official artifact difference."
        ),
    }
    manifest_path = manifest_dir / "reproduction_status_summary.json"
    manifest_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = report_dir / "reproduction_status_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# VJETHBKM reproduction status report",
                "",
                status["boundary"],
                "",
                "## Table S3 coverage",
                "",
                f"- Rows: `{status['table_s3_rows']}`",
                f"- Matched local rows: `{status['table_s3_matched_rows']}`",
                f"- Unresolved rows: `{status['unresolved_rows']}`",
                "",
                "```csv",
                descriptor_diff.to_csv(index=False).strip(),
                "```",
                "",
                "## Unresolved scope",
                "",
                "\n".join(f"- `{item}`" for item in status["unresolved_scope"]) if status["unresolved_scope"] else "- None",
                "",
                "## BH2 external 187 summary",
                "",
                "```csv",
                bh2_external.to_csv(index=False).strip(),
                "```",
                "",
                "## Available downstream analyses",
                "",
                f"- High-yield summary rows: `{status['high_yield_rows']}`",
                f"- Component split schema rows: `{status['component_split_schema_rows']}`",
                "- Detailed tables are stored under `outputs/tables/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return status


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    status = build_report(repro_root)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
