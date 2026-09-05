from __future__ import annotations

from pathlib import Path
import json
import re


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def audit(repro_root: Path) -> dict:
    train_root = repro_root / "yieldsmarter" / "Src" / "Train"
    train_ohe = _read(train_root / "Train_OHE.py")
    train_descriptors = _read(train_root / "Train_descriptors.py")
    description = _read(train_root / "description.txt")

    configs = {}
    for cfg_path in sorted(train_root.glob("*.json")):
        configs[cfg_path.name] = json.loads(cfg_path.read_text(encoding="utf-8"))

    rf_defaults = re.search(r"DEFAULT_RF_PARAMS\s*=\s*dict\(([^)]*)\)", train_ohe)
    lgbm_defaults = re.search(r"DEFAULT_LGBM_PARAMS\s*=\s*dict\(([^)]*)\)", train_ohe)
    evidence = {
        "train_description": description.strip(),
        "ohe_fit_scope": "OneHotEncoder is fit on train_df[cols] inside each fold, then used to transform train/test.",
        "cv_strategy": "For each outer_fold in range(outer_splits), KFold uses random_state=seed+outer_fold.",
        "rf_defaults_literal": rf_defaults.group(1) if rf_defaults else "",
        "lgbm_defaults_literal": lgbm_defaults.group(1) if lgbm_defaults else "",
        "metrics": ["mae", "rmse", "r2", "medae", "kendall"],
        "config_summaries": {
            name: {
                "splits": cfg.get("splits"),
                "outer_splits": cfg.get("outer_splits"),
                "seed": cfg.get("seed"),
                "reaction_csv": cfg.get("reaction_csv", ""),
                "descriptors_npz": cfg.get("descriptors_npz", ""),
                "target_col": cfg.get("target_col", ""),
                "missing_token": cfg.get("missing_token", ""),
                "zero_out_missing": cfg.get("zero_out_missing", ""),
            }
            for name, cfg in configs.items()
        },
        "local_runner_alignment": {
            "implemented_after_audit": [
                "core_rf_5x5 default n_estimators=500",
                "core_rf_5x5 default rf_max_features=0.3",
                "core_rf_5x5 default random_state=1000",
                "RF model random_state varies by repeat as seed+repeat_index, matching official outer_fold behavior",
            ],
            "remaining_differences": [
                "Official OHE writes medae in addition to MAE/RMSE/R2/Kendall; local Table S3 merge currently tracks four paper-facing metrics.",
                "Official descriptor runner consumes precomputed .npz for MFP/PhysChem/DFT/SOAP; local Morgan/PhysChem descriptors are compatibility implementations until .npz adapter is added.",
            ],
        },
    }
    return evidence


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    output_path = repro_root / "data" / "manifest" / "official_training_params_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = audit(repro_root)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["local_runner_alignment"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
