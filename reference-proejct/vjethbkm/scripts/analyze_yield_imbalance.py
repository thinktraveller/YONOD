from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


def _latest_predictions(repro_root: Path) -> Path | None:
    run_root = repro_root / "outputs" / "runs"
    candidates = sorted(run_root.glob("core_rf_5x5_*/predictions.csv"))
    return candidates[-1] if candidates else None


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.imbalance import (
        high_yield_summary,
        prediction_bucket_errors,
        yield_bucket_summary,
    )
    from vjethbkm_repro.manifest import dataset_config_from_sources, read_dataset

    dataset_cfg = dataset_config_from_sources("smoke_local_yonod")
    df = read_dataset(dataset_cfg).reset_index(drop=True)
    table_dir = repro_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    bucket_path = table_dir / "yield_bucket_smoke_summary.csv"
    yield_bucket_summary(df, dataset_cfg.label_col).to_csv(bucket_path, index=False)

    prediction_path = _latest_predictions(repro_root)
    result = {"yield_bucket_summary": str(bucket_path), "prediction_source": str(prediction_path) if prediction_path else None}
    if prediction_path:
        predictions = pd.read_csv(prediction_path)
        error_path = table_dir / "yield_bucket_error_smoke_summary.csv"
        high_yield_path = table_dir / "high_yield_smoke_summary.csv"
        prediction_bucket_errors(predictions).to_csv(error_path, index=False)
        high_yield_summary(predictions).to_csv(high_yield_path, index=False)
        result["yield_bucket_error_summary"] = str(error_path)
        result["high_yield_summary"] = str(high_yield_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

