from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATASET_MAP = {
    "BH": {"source_folder": "Doyle_2018", "prefix": "Doyle"},
    "BH2": {"source_folder": "Denmark_2023", "prefix": "Denmark"},
    "SM": {"source_folder": "Suzuki_2018", "prefix": "Suzuki"},
    "SLAP": {"source_folder": "Bode_2023", "prefix": "Bode"},
}
DESCRIPTORS = ["OHE", "MFP", "PhysChem"]
BINS = [-np.inf, 20, 40, 60, 80, 100, np.inf]
LABELS = ["<20", "20-40", "40-60", "60-80", "80-100", ">100"]


def _bucket(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=BINS, labels=LABELS, right=False)


def analyze(repro_root: Path) -> dict:
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from vjethbkm_repro.manifest import dataset_config_from_sources, read_dataset

    table_dir = repro_root / "outputs" / "tables"
    report_dir = repro_root / "outputs" / "reports"
    manifest_dir = repro_root / "data" / "manifest"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for dataset_id in DATASET_MAP:
        cfg = dataset_config_from_sources(dataset_id)
        df = read_dataset(cfg)
        y = pd.to_numeric(df[cfg.label_col], errors="coerce")
        counts = _bucket(y).value_counts(sort=False)
        for bucket, count in counts.items():
            dataset_rows.append(
                {
                    "dataset": dataset_id,
                    "bucket": str(bucket),
                    "rows": int(count),
                    "fraction": float(count / len(y)),
                    "target_min": float(y.min()),
                    "target_max": float(y.max()),
                    "target_mean": float(y.mean()),
                }
            )

    compare_root = repro_root / "yieldsmarter" / "Results" / "Compare_Complexity"
    bucket_error_rows = []
    high_yield_rows = []
    for dataset_id, info in DATASET_MAP.items():
        for descriptor in DESCRIPTORS:
            pred_path = (
                compare_root
                / info["source_folder"]
                / descriptor
                / f"{info['prefix']}_{descriptor}_oof_predictions_RF.npz"
            )
            if not pred_path.exists():
                continue
            data = np.load(pred_path)
            frame = pd.DataFrame({"y_true": data["y_true"], "y_pred": data["y_pred"]})
            frame["bucket"] = _bucket(frame["y_true"])
            for bucket, part in frame.groupby("bucket", observed=False):
                if part.empty:
                    continue
                bucket_error_rows.append(
                    {
                        "dataset": dataset_id,
                        "descriptor": descriptor,
                        "bucket": str(bucket),
                        "rows": int(len(part)),
                        "mae": float(mean_absolute_error(part["y_true"], part["y_pred"])),
                        "rmse": float(np.sqrt(mean_squared_error(part["y_true"], part["y_pred"]))),
                        "prediction_file": str(pred_path.relative_to(repro_root)).replace("\\", "/"),
                        "status": "official_oof_predictions",
                    }
                )
            y_true_high = frame["y_true"] >= 80
            y_pred_high = frame["y_pred"] >= 80
            tp = int((y_true_high & y_pred_high).sum())
            fp = int((~y_true_high & y_pred_high).sum())
            fn = int((y_true_high & ~y_pred_high).sum())
            high_yield_rows.append(
                {
                    "dataset": dataset_id,
                    "descriptor": descriptor,
                    "threshold": 80,
                    "true_high": int(y_true_high.sum()),
                    "pred_high": int(y_pred_high.sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                    "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                    "status": "official_oof_predictions",
                }
            )

    dataset_summary = pd.DataFrame(dataset_rows)
    bucket_error = pd.DataFrame(bucket_error_rows)
    high_yield = pd.DataFrame(high_yield_rows)
    dataset_path = table_dir / "official_yield_bucket_dataset_summary.csv"
    bucket_error_path = table_dir / "official_oof_bucket_error_summary.csv"
    high_yield_path = table_dir / "official_oof_high_yield_summary.csv"
    dataset_summary.to_csv(dataset_path, index=False)
    bucket_error.to_csv(bucket_error_path, index=False)
    high_yield.to_csv(high_yield_path, index=False)

    audit = {
        "datasets": list(DATASET_MAP),
        "descriptors": DESCRIPTORS,
        "yield_buckets": LABELS,
        "high_yield_threshold": 80,
        "dataset_bucket_rows": int(len(dataset_summary)),
        "bucket_error_rows": int(len(bucket_error)),
        "high_yield_rows": int(len(high_yield)),
        "boundary": "Uses bundled official OOF predictions for RF descriptors; SM/OHE remains marked elsewhere as a forensic unresolved artifact.",
        "outputs": {
            "dataset_summary": str(dataset_path.relative_to(repro_root)).replace("\\", "/"),
            "bucket_error": str(bucket_error_path.relative_to(repro_root)).replace("\\", "/"),
            "high_yield": str(high_yield_path.relative_to(repro_root)).replace("\\", "/"),
        },
    }
    (manifest_dir / "official_imbalance_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = report_dir / "official_imbalance_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Official OOF imbalance and high-yield audit",
                "",
                audit["boundary"],
                "",
                "## Dataset yield buckets",
                "",
                "```csv",
                dataset_summary.to_csv(index=False).strip(),
                "```",
                "",
                "## High-yield summary",
                "",
                "```csv",
                high_yield.to_csv(index=False).strip(),
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
