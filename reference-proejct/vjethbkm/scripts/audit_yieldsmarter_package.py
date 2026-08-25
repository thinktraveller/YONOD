from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.manifest import sha256_file

    package_root = repro_root / "yieldsmarter"
    if not package_root.exists():
        raise FileNotFoundError(f"Missing package root: {package_root}")

    files = [path for path in package_root.rglob("*") if path.is_file()]
    extension_counts = {}
    for path in files:
        extension_counts[path.suffix.lower() or "<none>"] = extension_counts.get(path.suffix.lower() or "<none>", 0) + 1

    key_files = [
        "README.txt",
        "LICENSE.txt",
        "yieldsmarter.yml",
        "Data/HTE_datasets/BH1/BH1.csv",
        "Data/HTE_datasets/BH2/BH2.csv",
        "Data/HTE_datasets/BH2/csv/187_with_corrected_catalyst_smiles.csv",
        "Data/HTE_datasets/SM/SM.csv",
        "Data/HTE_datasets/SL1/SL1.csv",
        "Results/Compare_Complexity/Bode_2023/OHE/Bode_OHE_metrics_summaries_RF.json",
        "Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP_metrics_summaries_RF.json",
        "Results/Compare_Complexity/Denmark_2023/PhysChem/Denmark_PhysChem_metrics_summaries_RF.json",
        "Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP_metrics_summaries_RF.json",
    ]
    checksums = []
    for rel in key_files:
        path = package_root / rel
        checksums.append(
            {
                "relative_path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )

    dataset_files = {
        "BH1": package_root / "Data/HTE_datasets/BH1/BH1.csv",
        "BH2": package_root / "Data/HTE_datasets/BH2/BH2.csv",
        "BH2_187_external": package_root / "Data/HTE_datasets/BH2/csv/187_with_corrected_catalyst_smiles.csv",
        "SM": package_root / "Data/HTE_datasets/SM/SM.csv",
        "SL1": package_root / "Data/HTE_datasets/SL1/SL1.csv",
    }
    schema_records = []
    for dataset_id, path in dataset_files.items():
        df = pd.read_csv(path)
        target_col = "Yield" if "Yield" in df.columns else "yield"
        target = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(dtype=float)
        schema_records.append(
            {
                "dataset_id": dataset_id,
                "relative_path": str(path.relative_to(package_root)).replace("\\", "/"),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": "|".join(df.columns),
                "target_col": target_col if target_col in df.columns else "",
                "target_missing": int(target.isna().sum()) if len(target) else None,
                "target_min": float(target.min()) if len(target) else None,
                "target_max": float(target.max()) if len(target) else None,
                "target_mean": float(target.mean()) if len(target) else None,
            }
        )

    manifest_dir = repro_root / "data" / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "package_root": str(package_root),
        "readme": "README.txt",
        "license": "MIT License, Copyright (c) 2025 ETH Zurich",
        "official_evidence": "README title matches the paper and package contains Data/Src/Results/yieldsmarter.yml described by README.",
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "git_policy": {
            "raw_package_ignored": True,
            "commit_raw_package": False,
            "commit_generated_manifests": True,
        },
    }
    (manifest_dir / "yieldsmarter_package_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(checksums).to_csv(manifest_dir / "yieldsmarter_key_file_checksums.csv", index=False)
    pd.DataFrame(schema_records).to_csv(manifest_dir / "yieldsmarter_dataset_schema.csv", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(pd.DataFrame(schema_records).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

