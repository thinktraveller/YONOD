"""Build a standalone MFP/OHE RF final-metrics package from official JSON."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from extract_official_metrics import extract_records


FINAL_DESCRIPTORS = {"MFP", "OHE"}
FINAL_METRICS = {"mae", "rmse", "r2", "kendall_tau"}

# SI Table S3 at its printed precision. Full-precision artifact values remain
# in the output so print rounding is not mistaken for a reproduction tolerance.
_PRINTED = """
BH1 MFP mae 4.5 .2
BH1 MFP rmse 6.9 .5
BH1 MFP r2 .94 .01
BH1 MFP kendall_tau .85 .01
BH1 OHE mae 6.4 .3
BH1 OHE rmse 9.3 .5
BH1 OHE r2 .88 .01
BH1 OHE kendall_tau .80 .01
BH2 MFP mae 12.1 .5
BH2 MFP rmse 17.9 .8
BH2 MFP r2 .73 .02
BH2 MFP kendall_tau .68 .01
BH2 OHE mae 15.3 .6
BH2 OHE rmse 21.9 .9
BH2 OHE r2 .59 .03
BH2 OHE kendall_tau .61 .02
SM MFP mae 7.4 .2
SM MFP rmse 11.0 .3
SM MFP r2 .85 .01
SM MFP kendall_tau .76 .01
SM OHE mae 9.4 .2
SM OHE rmse 13.1 .3
SM OHE r2 .78 .01
SM OHE kendall_tau .71 .01
SL1 MFP mae 18.0 1.8
SL1 MFP rmse 35.1 5.1
SL1 MFP r2 .77 .05
SL1 MFP kendall_tau .51 .04
SL1 OHE mae 22.0 2.8
SL1 OHE rmse 45.9 7.6
SL1 OHE r2 .61 .10
SL1 OHE kendall_tau .48 .04
"""
TABLE_S3 = {
    (dataset, descriptor, metric): (float(mean), float(std))
    for dataset, descriptor, metric, mean, std in (
        line.split() for line in _PRINTED.splitlines() if line.strip()
    )
}


def _digits(metric: str) -> int:
    return 1 if metric in {"mae", "rmse"} else 2


def _metadata(dataset: str, descriptor: str) -> dict[str, Any]:
    known = {
        "BH1": ("BH", "yield_percent", "Yield (%)", 3955, "official CSV: 3955 reactions", "paper_and_artifact_aligned"),
        "BH2": ("BH2", "yield_percent", "Yield (%)", 3359, "official CSV: 3359 reactions", "paper_and_artifact_aligned"),
        "SL1": ("SLAP", "lc_ms_product_ratio", "LC-MS product ratio", 1150, "official CSV: 1150 reactions", "paper_and_artifact_aligned"),
    }
    if dataset in known:
        names = ("paper_dataset_id", "unit", "unit_label", "population_rows", "population", "population_status")
        return dict(zip(names, known[dataset], strict=True))
    if descriptor == "MFP":
        return {"paper_dataset_id": "SM", "unit": "yield_percent", "unit_label": "Yield (%)", "population_rows": 4620, "population": "SI all-components-specified subset: 4620 reactions", "population_status": "si_cleaned_4620"}
    return {"paper_dataset_id": "SM", "unit": "yield_percent", "unit_label": "Yield (%)", "population_rows": 5760, "population": "official OHE artifact: raw SM.csv, 5760 reactions", "population_status": "artifact_5760_conflicts_with_si_cleaning_text"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict[str, Any]], html_output: bool) -> str:
    lines = []
    for row in rows:
        value = f"{row['mean']:.{_digits(row['metric'])}f} ± {row['std']:.{_digits(row['metric'])}f}"
        values = (row["dataset_id"], row["descriptor"], row["metric"], value, row["unit_label"], row["valid_fold_count"], row["population"], row["table_s3_display_match"])
        if html_output:
            lines.append("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in values) + "</tr>")
        else:
            lines.append("| " + " | ".join(str(item) for item in values) + " |")
    return "\n".join(lines)


def _write_reports(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    note = "Each mean and standard deviation summarizes 25 fold-level values from 5 repeats × 5 folds. This is an official-artifact summary, not an independent retraining claim. SM/MFP uses 4620 rows, while bundled SM/OHE artifacts use 5760 rows; this remains an audit boundary."
    markdown = "\n".join((
        "# Final MFP/OHE + RF metrics from official YieldSmarter artifacts", "", note, "",
        "| Dataset | Descriptor | Metric | Mean ± std | Unit | Valid folds | Population | Table S3 display |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |", _table(rows, False), "",
    ))
    document = "\n".join((
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Final MFP/OHE RF metrics</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.4rem;text-align:left}th{background:#f1f5f9}</style></head><body>",
        "<h1>Final MFP/OHE + RF metrics from official YieldSmarter artifacts</h1>", f"<p>{html.escape(note)}</p>",
        "<table><thead><tr><th>Dataset</th><th>Descriptor</th><th>Metric</th><th>Mean ± std</th><th>Unit</th><th>Valid folds</th><th>Population</th><th>Table S3 display</th></tr></thead><tbody>",
        _table(rows, True), "</tbody></table></body></html>",
    ))
    (report_dir / "final_mfp_ohe_rf_metrics.md").write_text(markdown, encoding="utf-8")
    (report_dir / "final_mfp_ohe_rf_metrics.html").write_text(document, encoding="utf-8")


def build_final_metrics(repro_root: Path, output_root: Path | None = None) -> list[dict[str, Any]]:
    """Write 32 final metric records, four metrics for eight data/descriptor pairs."""
    output_root = output_root or repro_root
    _, targets = extract_records(repro_root)
    rows: list[dict[str, Any]] = []
    for target in targets:
        dataset, descriptor, metric = str(target["source_dataset"]), str(target["descriptor"]), str(target["metric"])
        if descriptor not in FINAL_DESCRIPTORS or metric not in FINAL_METRICS:
            continue
        expected_mean, expected_std = TABLE_S3[(dataset, descriptor, metric)]
        metadata = _metadata(dataset, descriptor)
        digits = _digits(metric)
        mean, std = float(target["paper_value"]), float(target["paper_std"])
        rows.append({
            "dataset_id": dataset, "paper_dataset_id": metadata["paper_dataset_id"], "descriptor": descriptor, "model": target["model"], "metric": metric,
            "mean": mean, "std": std, "unit": metadata["unit"], "unit_label": metadata["unit_label"], "valid_fold_count": 25,
            "population": metadata["population"], "population_rows": metadata["population_rows"], "population_status": metadata["population_status"],
            "source_table": "SI Table S3", "relative_json_path": target["relative_json_path"],
            "table_s3_printed_mean": expected_mean, "table_s3_printed_std": expected_std,
            "table_s3_display_match": round(mean, digits) == expected_mean and round(std, digits) == expected_std,
        })
    rows.sort(key=lambda row: (row["dataset_id"], row["descriptor"], row["metric"]))
    if len(rows) != 32 or not all(row["table_s3_display_match"] for row in rows):
        raise RuntimeError("Expected 32 SI Table S3-aligned MFP/OHE/RF metric rows")
    table_dir = output_root / "outputs" / "tables"
    _write_csv(table_dir / "final_mfp_ohe_rf_metrics.csv", rows)
    (table_dir / "final_mfp_ohe_rf_metrics.json").write_text(json.dumps({"schema_version": 1, "source": "local official YieldSmarter artifacts", "records": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_reports(output_root / "outputs" / "reports", rows)
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({"rows": len(build_final_metrics(root)), "valid_fold_count": 25}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
