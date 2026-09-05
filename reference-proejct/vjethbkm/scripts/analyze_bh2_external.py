from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.metrics import mean_absolute_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(repro_root: Path) -> dict:
    package_root = repro_root / "yieldsmarter"
    per_product_root = package_root / "Data" / "HTE_datasets" / "BH2" / "Per_Product"
    official_pred_root = package_root / "Results" / "BH2" / "Holdout" / "MFP"
    corrected_187 = (
        package_root
        / "Data"
        / "HTE_datasets"
        / "BH2"
        / "csv"
        / "187_with_corrected_catalyst_smiles.csv"
    )
    model_file = official_pred_root / "model_MFP_RF.pkl"
    script_file = package_root / "Src" / "Train" / "BH2_holdout_predict.py"

    per_product_rows = []
    for input_path in sorted(per_product_root.glob("Product_?.csv")):
        product = input_path.stem.replace("Product_", "")
        pred_path = official_pred_root / f"Product_{product}_preds.csv"
        input_df = pd.read_csv(input_path)
        pred_df = pd.read_csv(pred_path)
        per_product_rows.append(
            {
                "product": product,
                "input_rows": int(len(input_df)),
                "prediction_rows": int(len(pred_df)),
                "true_yield_mean": float(pred_df["True_Yield"].mean()),
                "official_mfp_mae": float(mean_absolute_error(pred_df["True_Yield"], pred_df["Predicted_Yield"])),
                "dft_mae": float(mean_absolute_error(pred_df["True_Yield"], pred_df["DFT_Yield"])),
                "official_prediction_file": str(pred_path.relative_to(repro_root)).replace("\\", "/"),
                "input_file": str(input_path.relative_to(repro_root)).replace("\\", "/"),
            }
        )
    per_product = pd.DataFrame(per_product_rows).sort_values("product")

    corrected_df = pd.read_csv(corrected_187)
    external_187_summary = {
        "rows": int(len(corrected_df)),
        "predict_mae_from_columns": float(mean_absolute_error(corrected_df["Yield"], corrected_df["predict"])),
        "ours_mae_from_columns": float(mean_absolute_error(corrected_df["Yield"], corrected_df["ours"])),
        "ae_predict_mean": float(corrected_df["AE_predict"].mean()),
        "ae_ours_mean": float(corrected_df["AE_ours"].mean()),
    }

    table_dir = repro_root / "outputs" / "tables"
    report_dir = repro_root / "outputs" / "reports"
    manifest_dir = repro_root / "data" / "manifest"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    per_product_path = table_dir / "bh2_external_per_product_mae.csv"
    external_187_path = table_dir / "bh2_external_187_summary.csv"
    per_product.to_csv(per_product_path, index=False)
    pd.DataFrame([external_187_summary]).to_csv(external_187_path, index=False)

    audit = {
        "official_script": str(script_file.relative_to(repro_root)).replace("\\", "/"),
        "official_script_requires_model": str(model_file.relative_to(repro_root)).replace("\\", "/"),
        "model_file_present": model_file.exists(),
        "per_product_files": {
            "input_count": len(list(per_product_root.glob("Product_?.csv"))),
            "prediction_count": len(list(official_pred_root.glob("Product_?_preds.csv"))),
        },
        "corrected_187": {
            "path": str(corrected_187.relative_to(repro_root)).replace("\\", "/"),
            "sha256": _sha256(corrected_187),
            **external_187_summary,
        },
        "outputs": {
            "per_product_mae": str(per_product_path.relative_to(repro_root)).replace("\\", "/"),
            "external_187_summary": str(external_187_path.relative_to(repro_root)).replace("\\", "/"),
        },
        "boundary": (
            "The bundled package contains Product_*_preds.csv but not model_MFP_RF.pkl, "
            "so this step audits official holdout outputs instead of claiming a local reprediction."
        ),
    }
    (manifest_dir / "bh2_external_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = report_dir / "bh2_external_validation_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# BH2 external validation audit",
                "",
                audit["boundary"],
                "",
                "## 187 corrected catalyst smiles summary",
                "",
                "```csv",
                pd.DataFrame([external_187_summary]).to_csv(index=False).strip(),
                "```",
                "",
                "## Per-product MAE",
                "",
                "```csv",
                per_product.to_csv(index=False).strip(),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    result = analyze(repro_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
