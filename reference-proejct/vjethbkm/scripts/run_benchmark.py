from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VJETHBKM reproduction benchmarks.")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "core_rf_5x5"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--descriptors",
        default=None,
        help="Comma-separated descriptors for core_rf_5x5, e.g. ohe,morgan,physchem.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--rf-max-features",
        default="0.3",
        help="RandomForest max_features; use none for sklearn default.",
    )
    parser.add_argument("--random-state", type=int, default=1000)
    parser.add_argument(
        "--allow-smoke-dataset",
        action="store_true",
        help="Allow formal 5x5 workflow validation on the local smoke dataset.",
    )
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.benchmark import run_core_rf_5x5, run_smoke

    if args.stage == "smoke":
        result = run_smoke()
    elif args.stage == "core_rf_5x5":
        descriptors = None
        if args.descriptors:
            descriptors = [item.strip() for item in args.descriptors.split(",") if item.strip()]
        rf_max_features = None
        if args.rf_max_features.lower() != "none":
            try:
                rf_max_features = float(args.rf_max_features)
            except ValueError:
                rf_max_features = args.rf_max_features
        result = run_core_rf_5x5(
            dataset_id=args.dataset or "smoke_local_yonod",
            allow_smoke_dataset=args.allow_smoke_dataset,
            descriptors=descriptors,
            repeats=args.repeats,
            folds=args.folds,
            n_estimators=args.n_estimators,
            rf_max_features=rf_max_features,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
