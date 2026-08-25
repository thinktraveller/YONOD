from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys

import pandas as pd


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.component_splits import (
        leave_component_out_1d,
        leave_component_pair_out_2d,
        random_0d_split,
        split_summary,
    )
    from vjethbkm_repro.manifest import dataset_config_from_sources, read_dataset

    dataset_cfg = dataset_config_from_sources("smoke_local_yonod")
    df = read_dataset(dataset_cfg).reset_index(drop=True)
    run_id = "component_splits_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repro_root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_manifests = {
        "holdout_0d_random": random_0d_split(df, n_splits=5, random_state=42),
        "holdout_1d_sub_1_smiles": leave_component_out_1d(df, "sub_1_smiles"),
        "holdout_2d_sub_1_smiles__sub_2_smiles": leave_component_pair_out_2d(
            df,
            ["sub_1_smiles", "sub_2_smiles"],
        ),
    }
    summaries = []
    for name, manifest in split_manifests.items():
        manifest.to_csv(run_dir / f"{name}.csv", index=False)
        summaries.append(split_summary(df, manifest, dataset_cfg.component_cols))

    summary = pd.DataFrame(summaries)
    summary_path = repro_root / "outputs" / "tables" / "component_split_smoke_summary.csv"
    summary.to_csv(summary_path, index=False)
    result = {"run_id": run_id, "run_dir": str(run_dir), "summary_path": str(summary_path)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_csv(index=False))
    return 0 if bool(summary["leakage_ok"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

