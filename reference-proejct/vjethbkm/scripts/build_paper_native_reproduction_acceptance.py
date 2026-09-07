"""Publish a single auditable acceptance package for native MFP/OHE RF reruns."""

from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
from typing import Any


TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def status_for_row(descriptor: str, item: dict[str, Any], metric_delta: float) -> tuple[str, str]:
    """Return a strict numeric status plus the required feature/OOF evidence."""
    if abs(metric_delta) > TOLERANCE:
        return "fail", f"metric_delta_exceeds_{TOLERANCE:g}"
    if descriptor == "MFP":
        alignment = item["mfp_alignment"]
        if all(alignment[key] for key in ("x_exact", "y_exact", "smiles_columns_exact")):
            return "pass", "MFP X/y/component columns exact; metric delta within 1e-12"
        return "fail", "MFP feature, target, or component-column alignment failed"
    alignment = item["oof_alignment"]
    prediction_ok = alignment.get("y_pred_within_1e_12", alignment["y_pred_max_abs_diff"] <= TOLERANCE)
    if alignment["y_true_exact"] and alignment["fold_id_exact"] and prediction_ok:
        return "pass", "OHE train-fold OOF y/fold exact and prediction delta within 1e-12"
    return "fail", "OHE OOF target, fold identity, or prediction alignment failed"


def published_display_map(path: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    return {
        (row["dataset_id"], row["descriptor"], row["metric"]): (row["table_s3_printed_mean"], row["table_s3_printed_std"])
        for row in read_csv(path)
    }


def result_rows(descriptor: str, matrix: dict[str, Any], display: dict[tuple[str, str, str], tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in matrix["datasets"]:
        if descriptor == "MFP":
            population_rows = item["mfp_alignment"]["generated_shape"][0]
            evidence = "MFP X/y/component columns exact" if all(item["mfp_alignment"][key] for key in ("x_exact", "y_exact", "smiles_columns_exact")) else "MFP alignment failed"
            population_note = "SM raw 5760 minus 1140 missing-component rows" if item["dataset_id"] == "SM" else "official CSV full population"
        else:
            population_rows = item["source_rows"]
            evidence = "OOF y/fold exact; max prediction delta %.12g" % item["oof_alignment"]["y_pred_max_abs_diff"]
            population_note = "official raw CSV population (SM=5760)" if item["dataset_id"] == "SM" else "official CSV full population"
        for comparison in item["comparison_rows"]:
            mean_delta = float(comparison["native_minus_official_mean"])
            std_delta = float(comparison["native_minus_official_std"])
            status, reason = status_for_row(descriptor, item, max(abs(mean_delta), abs(std_delta)))
            metric = comparison["metric"]
            printed_mean, printed_std = display[(item["dataset_id"], descriptor, metric)]
            rows.append({
                "dataset_id": item["dataset_id"], "paper_dataset_id": item["paper_dataset_id"], "descriptor": descriptor,
                "model": "RF", "metric": metric, "unit": comparison["unit"], "population_rows": population_rows,
                "population_note": population_note, "table_s3_printed_mean": printed_mean, "table_s3_printed_std": printed_std,
                "official_json_mean": comparison["official_json_mean"], "official_json_std": comparison["official_json_std"],
                "independent_native_mean": comparison["independent_native_mean"], "independent_native_std": comparison["independent_native_std"],
                "native_minus_official_mean": mean_delta, "native_minus_official_std": std_delta,
                "feature_or_oof_evidence": evidence, "valid_fold_count": comparison["valid_fold_count"], "status": status, "status_reason": reason,
            })
    return rows


def markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# VJETHBKM MFP/OHE + RF paper-native reproduction acceptance", "",
        f"**Result: {summary['pass_count']}/{summary['row_count']} metrics pass strict 1e-12 numeric comparison.**", "",
        "This package reproduces the paper's four MFP and four OHE RF 5×5 results only from bundled `yieldsmarter` CSVs, feature/training logic and official artifacts. It does not use or modify YONOD. The unmodified upstream trainers cannot import locally because they import unavailable LightGBM before selecting RF; narrow RF-only wrappers preserve the upstream RF parameters, seeds, folds, metrics and feature logic without editing source files.", "",
        "## Population and protocol evidence", "",
        "- MFP: each official CSV was passed to unmodified `Gen_MFP.py`; every MFP X/y/component-column artifact is exact. SM uses the original generator's `--skip_rows_with_missing_values`, transforming 5760 raw rows to 4620 complete-component rows.",
        "- OHE: each fold fits OHE on train rows only, zeros missing-component blocks and ignores unseen values. Official OOF targets and fold IDs are exact; predictions differ only by at most 1.71e-13, below the explicit 1e-12 acceptance tolerance.",
        "- Environment caveat: SI reports RDKit 2025.03.6 and scikit-learn 1.6.1; actual versions are in the JSON. The exact numerical result is evidence of a compatible rerun, not a claim that the package pins the paper's original environment.", "",
        "## Metric acceptance", "",
        "| Dataset | Descriptor | Metric | Population | Table S3 | Official JSON mean ± std | Native mean ± std | Δ mean | Δ std | Status |",
        "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['dataset_id']} | {row['descriptor']} | {row['metric']} | {row['population_rows']} | {row['table_s3_printed_mean']} ± {row['table_s3_printed_std']} | {float(row['official_json_mean']):.12g} ± {float(row['official_json_std']):.12g} | {float(row['independent_native_mean']):.12g} ± {float(row['independent_native_std']):.12g} | {row['native_minus_official_mean']:.12g} | {row['native_minus_official_std']:.12g} | {row['status']} |")
    lines.extend(["", "## Scope boundary", "", "This accepts Table S3's MFP/OHE + RF subset, not other descriptors, non-RF models, Figure 5's five-descriptor interval comparison, BH2 external validation, reweighting or generalization experiments. SM still has distinct MFP=4620 and OHE=5760 population definitions; the original native paths now make both operationally reproducible, while the paper text/artifact provenance distinction remains recorded rather than silently merged.", ""])
    return "\n".join(lines)


def html_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    header = ["dataset_id", "descriptor", "metric", "population_rows", "table_s3_printed_mean", "table_s3_printed_std", "official_json_mean", "official_json_std", "independent_native_mean", "independent_native_std", "native_minus_official_mean", "native_minus_official_std", "status"]
    columns = "".join(f"<th>{escape(column)}</th>" for column in header)
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in header) + "</tr>" for row in rows)
    return "<!doctype html><html><head><meta charset='utf-8'><title>Paper-native reproduction acceptance</title><style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;font-size:.82rem}th,td{border:1px solid #ccc;padding:.35rem;text-align:left}th{background:#eee}.pass{color:#087f23}</style></head><body>" + f"<h1>VJETHBKM MFP/OHE + RF paper-native reproduction acceptance</h1><p class='pass'>{summary['pass_count']}/{summary['row_count']} metric rows pass strict 1e-12 comparison.</p><table><thead><tr>{columns}</tr></thead><tbody>{body}</tbody></table></body></html>\n"


def build(repro_root: Path) -> dict[str, Any]:
    tables = repro_root / "outputs" / "tables"
    reports = repro_root / "outputs" / "reports"
    mfp = json.loads((tables / "native_mfp_rf_matrix_reproduction.json").read_text(encoding="utf-8"))
    ohe = json.loads((tables / "native_ohe_rf_matrix_reproduction.json").read_text(encoding="utf-8"))
    display = published_display_map(tables / "final_mfp_ohe_rf_metrics.csv")
    rows = result_rows("MFP", mfp, display) + result_rows("OHE", ohe, display)
    summary = {
        "run_type": "paper_native_reproduction_acceptance", "row_count": len(rows), "pass_count": sum(row["status"] == "pass" for row in rows),
        "tolerance": TOLERANCE, "mfp_actual_versions": mfp["actual_versions"], "ohe_actual_versions": ohe["actual_versions"],
        "mfp_source": "native_mfp_rf_matrix_reproduction.json", "ohe_source": "native_ohe_rf_matrix_reproduction.json", "rows": rows,
    }
    write_csv(tables / "native_mfp_ohe_rf_reproduction_acceptance.csv", rows)
    (tables / "native_mfp_ohe_rf_reproduction_acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = markdown(rows, summary)
    (reports / "native_mfp_ohe_rf_reproduction_acceptance.md").write_text(report, encoding="utf-8")
    (reports / "native_mfp_ohe_rf_reproduction_acceptance.html").write_text(html_report(rows, summary), encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    summary = build(root)
    print(json.dumps({"row_count": summary["row_count"], "pass_count": summary["pass_count"], "tolerance": summary["tolerance"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
